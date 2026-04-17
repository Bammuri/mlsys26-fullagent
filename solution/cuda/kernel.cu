/*
 * GDN decode + prefill CUDA kernels with TVM FFI binding.
 *
 * Delta rule update per (batch/seq, v_head, row):
 *   g         = exp(-exp(A_log) * softplus(a + dt_bias))   // scalar per head
 *   beta      = sigmoid(b)                                 // scalar per head
 *   old_v     = k · (g * state)                            // per-row, 128-elem dot
 *   delta     = beta * (v - old_v)                         // scalar per row
 *   new_state = g * state + k * delta                      // 128-elem rank-1 update
 *   output    = scale * (qs + delta * qk)
 *                where qs = q · (g * state), qk = q · k    // avoids a second 128-dot
 *
 * State: [B/N, 8, 128, 128] fp32, k-last.
 * Parallelization: 4-way V-dim split → 1 warp (32 threads) per block; each thread owns
 * one V-row of the 128×128 state, held in registers as `float4 sr[32]`.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/error.h>

namespace {

constexpr int kNumQHeads = 4;
constexpr int kNumKHeads = 4;
constexpr int kNumVHeads = 8;
constexpr int kHeadSize  = 128;
constexpr int kVecsPerRow = kHeadSize / 4;          // 32 float4 per state row

constexpr int kSplits       = 4;                    // V-dim splits per (batch, v_head)
constexpr int kRowsPerBlock = kHeadSize / kSplits;  // 32 rows per block
constexpr int kWarpThreads  = kRowsPerBlock;        // 1 warp = 32 threads

__device__ __forceinline__ float softplusf_stable(float x) {
    if (x >  20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

__device__ __forceinline__ float sigmoidf_stable(float x) {
    if (x >= 0.0f) {
        const float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    const float z = expf(x);
    return z / (1.0f + z);
}

__device__ __forceinline__ void compute_gate(
    const __nv_bfloat16* __restrict__ a_in,
    const __nv_bfloat16* __restrict__ b_in,
    int64_t g_off, float A_log_val, float dt_bias_val,
    float& g, float& beta)
{
    const float x = __bfloat162float(a_in[g_off]) + dt_bias_val;
    g    = expf(-expf(A_log_val) * softplusf_stable(x));
    beta = sigmoidf_stable(__bfloat162float(b_in[g_off]));
}

__device__ __forceinline__ void cp_async_16b(void* smem, const void* gmem) {
    unsigned sa = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    // .cg = L2-only (skip L1): K/Q is stream-read once per token.
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit()   { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_group 0;\n"); }

__device__ __forceinline__ float warp_reduce_sum(float x) {
    x += __shfl_xor_sync(0xffffffff, x, 16);
    x += __shfl_xor_sync(0xffffffff, x,  8);
    x += __shfl_xor_sync(0xffffffff, x,  4);
    x += __shfl_xor_sync(0xffffffff, x,  2);
    x += __shfl_xor_sync(0xffffffff, x,  1);
    return x;
}

// ---------------------------------------------------------------------------
// Decode kernel — single token per batch entry.
// ---------------------------------------------------------------------------
__global__
void gdn_decode_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float*         __restrict__ state,
    const float*         __restrict__ A_log,
    const __nv_bfloat16* __restrict__ a_in,
    const float*         __restrict__ dt_bias,
    const __nv_bfloat16* __restrict__ b_in,
    float scale,
    __nv_bfloat16*       __restrict__ output,
    float*               __restrict__ new_state,
    int64_t batch_size)
{
    const int bid       = blockIdx.x;
    const int split     = bid % kSplits;
    const int v_head    = (bid / kSplits) % kNumVHeads;
    const int batch_idx = bid / (kSplits * kNumVHeads);
    const int tid       = threadIdx.x;
    const int row       = split * kRowsPerBlock + tid;

    if (batch_idx >= batch_size) return;

    const int     qk_head = v_head / (kNumVHeads / kNumQHeads);
    const int64_t q_off   = (batch_idx * kNumQHeads + qk_head) * kHeadSize;
    const int64_t k_off   = (batch_idx * kNumKHeads + qk_head) * kHeadSize;
    const int64_t v_off   = (batch_idx * kNumVHeads + v_head)  * kHeadSize;
    const int64_t g_off   = batch_idx * kNumVHeads + v_head;
    const int64_t st_off  = ((int64_t)(batch_idx * kNumVHeads + v_head) * kHeadSize + row) * kHeadSize;

    __shared__ __align__(16) float s_q[kHeadSize];
    __shared__ __align__(16) float s_k[kHeadSize];

    #pragma unroll
    for (int j = 0; j < kHeadSize / kWarpThreads; ++j) {
        s_q[tid + j * kWarpThreads] = __bfloat162float(q[q_off + tid + j * kWarpThreads]);
        s_k[tid + j * kWarpThreads] = __bfloat162float(k[k_off + tid + j * kWarpThreads]);
    }

    float g, beta;
    compute_gate(a_in, b_in, g_off, A_log[v_head], dt_bias[v_head], g, beta);

    __syncwarp();

    // qk = q · k is a block-scalar; split across lanes and warp-reduce.
    float qk = 0.0f;
    #pragma unroll
    for (int i = tid; i < kHeadSize; i += kWarpThreads) qk += s_q[i] * s_k[i];
    qk = warp_reduce_sum(qk);

    // Fused first pass: sr ← g * state, accumulate ov = k·sr and qs = q·sr.
    const auto* st_vec = reinterpret_cast<const float4*>(state + st_off);
    float4 sr[kVecsPerRow];
    float ov0 = 0.f, ov1 = 0.f, ov2 = 0.f, ov3 = 0.f;
    float qs0 = 0.f, qs1 = 0.f, qs2 = 0.f, qs3 = 0.f;

    #pragma unroll
    for (int i = 0; i < kVecsPerRow; ++i) {
        float4 tmp = __ldg(&st_vec[i]);
        const int b4 = i * 4;
        tmp.x *= g; tmp.y *= g; tmp.z *= g; tmp.w *= g;
        sr[i] = tmp;
        ov0 += s_k[b4+0] * tmp.x; ov1 += s_k[b4+1] * tmp.y;
        ov2 += s_k[b4+2] * tmp.z; ov3 += s_k[b4+3] * tmp.w;
        qs0 += s_q[b4+0] * tmp.x; qs1 += s_q[b4+1] * tmp.y;
        qs2 += s_q[b4+2] * tmp.z; qs3 += s_q[b4+3] * tmp.w;
    }
    const float old_v = ov0 + ov1 + ov2 + ov3;
    const float qs    = qs0 + qs1 + qs2 + qs3;

    const float delta   = beta * (__bfloat162float(v[v_off + row]) - old_v);
    const float out_acc = qs + delta * qk;

    // Second pass: new_state = sr + k * delta (sr already holds g * state).
    auto* ns_vec = reinterpret_cast<float4*>(new_state + st_off);
    #pragma unroll
    for (int i = 0; i < kVecsPerRow; ++i) {
        const int b4 = i * 4;
        sr[i].x += s_k[b4+0] * delta; sr[i].y += s_k[b4+1] * delta;
        sr[i].z += s_k[b4+2] * delta; sr[i].w += s_k[b4+3] * delta;
        ns_vec[i] = sr[i];
    }

    output[(batch_idx * kNumVHeads + v_head) * kHeadSize + row] =
        __float2bfloat16(scale * out_acc);
}

// ---------------------------------------------------------------------------
// Prefill kernel — token-sequential delta-rule recurrence.
// K + Q are double-buffered via cp.async.cg (16B per thread). Per token we do
//   1) pre-scale: sr ← g * state, accumulate ov = k·sr, qs = q·sr;
//   2) warp-split reduction: qk = q · k;
//   3) scalar close-up: delta = β·(v - ov); output = scale·(qs + delta·qk);
//   4) state update: sr ← sr + k · delta.
// This mirrors the decode kernel's algebra — 128 FMAs/token fewer than the
// v6 prefill, which did a full q · new_state dot in a second pass.
// ---------------------------------------------------------------------------
__global__ void gdn_prefill_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float*         __restrict__ state,
    const float*         __restrict__ A_log,
    const __nv_bfloat16* __restrict__ a_in,
    const float*         __restrict__ dt_bias,
    const __nv_bfloat16* __restrict__ b_in,
    const int64_t*       __restrict__ cu_seqlens,
    float scale,
    __nv_bfloat16*       __restrict__ output,
    float*               __restrict__ new_state,
    int64_t num_seqs)
{
    const int bid      = blockIdx.x;
    const int split    = bid % kSplits;
    const int v_head   = (bid / kSplits) % kNumVHeads;
    const int seq_idx  = bid / (kSplits * kNumVHeads);
    const int tid      = threadIdx.x;
    const int row      = split * kRowsPerBlock + tid;

    if (seq_idx >= num_seqs) return;

    const int64_t seq_start = cu_seqlens[seq_idx];
    const int64_t seq_end   = cu_seqlens[seq_idx + 1];
    if (seq_end <= seq_start) return;

    const int     qk_head = v_head / (kNumVHeads / kNumQHeads);
    const int64_t st_off  = ((int64_t)(seq_idx * kNumVHeads + v_head) * kHeadSize + row) * kHeadSize;

    const float A_log_val   = A_log[v_head];
    const float dt_bias_val = dt_bias[v_head];

    float4 sr[kVecsPerRow];
    {
        const auto* sv = reinterpret_cast<const float4*>(state + st_off);
        #pragma unroll
        for (int i = 0; i < kVecsPerRow; ++i) sr[i] = __ldg(&sv[i]);
    }

    __shared__ __align__(16) __nv_bfloat16 s_k_bf16[2][kHeadSize];
    __shared__ __align__(16) __nv_bfloat16 s_q_bf16[2][kHeadSize];

    // 32 lanes × 16B = 512B = 128 bf16 K + 128 bf16 Q; first 16 lanes load K, next 16 load Q.
    int buf = 0;
    {
        const int64_t k0 = (seq_start * kNumKHeads + qk_head) * kHeadSize;
        const int64_t q0 = (seq_start * kNumQHeads + qk_head) * kHeadSize;
        if (tid < 16) {
            cp_async_16b(&s_k_bf16[0][tid * 8], &k[k0 + tid * 8]);
        } else {
            const int qi = tid - 16;
            cp_async_16b(&s_q_bf16[0][qi * 8], &q[q0 + qi * 8]);
        }
        cp_async_commit();
        cp_async_wait_all();
    }
    __syncwarp();

    for (int64_t t = seq_start; t < seq_end; ++t) {
        if (t + 1 < seq_end) {
            const int     nb = 1 - buf;
            const int64_t nk = ((t + 1) * kNumKHeads + qk_head) * kHeadSize;
            const int64_t nq = ((t + 1) * kNumQHeads + qk_head) * kHeadSize;
            if (tid < 16) {
                cp_async_16b(&s_k_bf16[nb][tid * 8], &k[nk + tid * 8]);
            } else {
                const int qi = tid - 16;
                cp_async_16b(&s_q_bf16[nb][qi * 8], &q[nq + qi * 8]);
            }
            cp_async_commit();
        }

        const int64_t g_off = t * kNumVHeads + v_head;
        float g, beta;
        compute_gate(a_in, b_in, g_off, A_log_val, dt_bias_val, g, beta);

        const __nv_bfloat16* ck = s_k_bf16[buf];
        const __nv_bfloat16* cq = s_q_bf16[buf];

        // Pass 1: sr ← g * sr; accumulate ov = k · sr and qs = q · sr.
        float ov0 = 0.f, ov1 = 0.f, ov2 = 0.f, ov3 = 0.f;
        float qs0 = 0.f, qs1 = 0.f, qs2 = 0.f, qs3 = 0.f;
        #pragma unroll
        for (int i = 0; i < kVecsPerRow; ++i) {
            const int b4 = i * 4;
            sr[i].x *= g; sr[i].y *= g; sr[i].z *= g; sr[i].w *= g;
            const float k0f = __bfloat162float(ck[b4+0]);
            const float k1f = __bfloat162float(ck[b4+1]);
            const float k2f = __bfloat162float(ck[b4+2]);
            const float k3f = __bfloat162float(ck[b4+3]);
            ov0 += k0f * sr[i].x; ov1 += k1f * sr[i].y;
            ov2 += k2f * sr[i].z; ov3 += k3f * sr[i].w;
            qs0 += __bfloat162float(cq[b4+0]) * sr[i].x;
            qs1 += __bfloat162float(cq[b4+1]) * sr[i].y;
            qs2 += __bfloat162float(cq[b4+2]) * sr[i].z;
            qs3 += __bfloat162float(cq[b4+3]) * sr[i].w;
        }
        const float old_v = ov0 + ov1 + ov2 + ov3;
        const float qs    = qs0 + qs1 + qs2 + qs3;

        // qk = q · k — block-scalar, 4 terms per lane, warp-reduced.
        float qk = 0.f;
        #pragma unroll
        for (int j = 0; j < kHeadSize / kWarpThreads; ++j) {
            const int c = tid + j * kWarpThreads;
            qk += __bfloat162float(ck[c]) * __bfloat162float(cq[c]);
        }
        qk = warp_reduce_sum(qk);

        const float v_val   = __bfloat162float(v[(t * kNumVHeads + v_head) * kHeadSize + row]);
        const float delta   = beta * (v_val - old_v);
        const float out_acc = qs + delta * qk;

        output[(t * kNumVHeads + v_head) * kHeadSize + row] =
            __float2bfloat16(scale * out_acc);

        // Pass 2: sr ← sr + k * delta (now equals new_state).
        #pragma unroll
        for (int i = 0; i < kVecsPerRow; ++i) {
            const int b4 = i * 4;
            sr[i].x += __bfloat162float(ck[b4+0]) * delta;
            sr[i].y += __bfloat162float(ck[b4+1]) * delta;
            sr[i].z += __bfloat162float(ck[b4+2]) * delta;
            sr[i].w += __bfloat162float(ck[b4+3]) * delta;
        }

        cp_async_wait_all();
        __syncwarp();
        buf = 1 - buf;
    }

    auto* ns_vec = reinterpret_cast<float4*>(new_state + st_off);
    #pragma unroll
    for (int i = 0; i < kVecsPerRow; ++i) ns_vec[i] = sr[i];
}

// ---------------------------------------------------------------------------
// TVM FFI host functions
// ---------------------------------------------------------------------------

void GDNDecode(
    tvm::ffi::TensorView q, tvm::ffi::TensorView k, tvm::ffi::TensorView v,
    tvm::ffi::TensorView state, tvm::ffi::TensorView A_log, tvm::ffi::TensorView a,
    tvm::ffi::TensorView dt_bias, tvm::ffi::TensorView b,
    double scale, tvm::ffi::TensorView output, tvm::ffi::TensorView new_state)
{
    const int64_t bs = q.size(0);
    float sf = (scale == 0.0) ? (1.0f / sqrtf(128.0f)) : static_cast<float>(scale);
    DLDevice dev = q.device();
    cudaStream_t st = static_cast<cudaStream_t>(TVMFFIEnvGetStream(dev.device_type, dev.device_id));
    gdn_decode_kernel<<<bs * kNumVHeads * kSplits, kWarpThreads, 0, st>>>(
        static_cast<const __nv_bfloat16*>(q.data_ptr()),
        static_cast<const __nv_bfloat16*>(k.data_ptr()),
        static_cast<const __nv_bfloat16*>(v.data_ptr()),
        static_cast<const float*>(state.data_ptr()),
        static_cast<const float*>(A_log.data_ptr()),
        static_cast<const __nv_bfloat16*>(a.data_ptr()),
        static_cast<const float*>(dt_bias.data_ptr()),
        static_cast<const __nv_bfloat16*>(b.data_ptr()),
        sf, static_cast<__nv_bfloat16*>(output.data_ptr()),
        static_cast<float*>(new_state.data_ptr()), bs);
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(kernel, GDNDecode);

void GDNPrefill(
    tvm::ffi::TensorView q, tvm::ffi::TensorView k, tvm::ffi::TensorView v,
    tvm::ffi::TensorView state, tvm::ffi::TensorView A_log, tvm::ffi::TensorView a,
    tvm::ffi::TensorView dt_bias, tvm::ffi::TensorView b,
    tvm::ffi::TensorView cu_seqlens, double scale,
    tvm::ffi::TensorView output, tvm::ffi::TensorView new_state)
{
    const int64_t ns = cu_seqlens.size(0) - 1;
    float sf = (scale == 0.0) ? (1.0f / sqrtf(128.0f)) : static_cast<float>(scale);
    DLDevice dev = q.device();
    cudaStream_t st = static_cast<cudaStream_t>(TVMFFIEnvGetStream(dev.device_type, dev.device_id));
    gdn_prefill_kernel<<<ns * kNumVHeads * kSplits, kWarpThreads, 0, st>>>(
        static_cast<const __nv_bfloat16*>(q.data_ptr()),
        static_cast<const __nv_bfloat16*>(k.data_ptr()),
        static_cast<const __nv_bfloat16*>(v.data_ptr()),
        static_cast<const float*>(state.data_ptr()),
        static_cast<const float*>(A_log.data_ptr()),
        static_cast<const __nv_bfloat16*>(a.data_ptr()),
        static_cast<const float*>(dt_bias.data_ptr()),
        static_cast<const __nv_bfloat16*>(b.data_ptr()),
        static_cast<const int64_t*>(cu_seqlens.data_ptr()),
        sf, static_cast<__nv_bfloat16*>(output.data_ptr()),
        static_cast<float*>(new_state.data_ptr()), ns);
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(kernel_prefill, GDNPrefill);

}  // namespace
