#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace {

constexpr int kHeadSize = 128;
constexpr int kNumQHeads = 4;
constexpr int kNumKHeads = 4;
constexpr int kNumVHeads = 8;
constexpr int kWarpSize = 32;
constexpr int kVecSize = 4;
constexpr int kWarpsPerBlock = 2;
constexpr int kRowsPerBlock = kWarpsPerBlock;
constexpr int kThreads = kWarpsPerBlock * kWarpSize;
constexpr int kRowTilesPerHead = kHeadSize / kRowsPerBlock;
constexpr int kChunkSize = 256;
// Head-pair fusion: each block handles kVHeadsPerBlock v_heads sharing the
// same q_head + k_head (V_PER_Q == V_PER_K == 2 for this workload).
constexpr int kVHeadsPerBlock = kNumVHeads / kNumQHeads;          // = 2
constexpr int kHeadPairs       = kNumVHeads / kVHeadsPerBlock;    // = 4

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_BF16(x) TORCH_CHECK((x).scalar_type() == torch::kBFloat16, #x " must be bfloat16")
#define CHECK_F32(x) TORCH_CHECK((x).scalar_type() == torch::kFloat32, #x " must be float32")
#define CHECK_I64(x) TORCH_CHECK((x).scalar_type() == torch::kInt64, #x " must be int64")

__device__ __forceinline__ float bf16_to_float(const c10::BFloat16* ptr) {
  const __nv_bfloat16* raw = reinterpret_cast<const __nv_bfloat16*>(ptr);
  return __bfloat162float(*raw);
}

__device__ __forceinline__ void float_to_bf16(float x, c10::BFloat16* ptr) {
  __nv_bfloat16* raw = reinterpret_cast<__nv_bfloat16*>(ptr);
  *raw = __float2bfloat16(x);
}

__device__ __forceinline__ float softplusf_stable(float x) {
  if (x > 20.0f) return x;
  if (x < -20.0f) return expf(x);
  return log1pf(expf(x));
}

__device__ __forceinline__ float4 load_bf16x4(const c10::BFloat16* ptr) {
  const __nv_bfloat162* raw = reinterpret_cast<const __nv_bfloat162*>(ptr);
  const float2 xy = __bfloat1622float2(raw[0]);
  const float2 zw = __bfloat1622float2(raw[1]);
  return make_float4(xy.x, xy.y, zw.x, zw.y);
}

__device__ __forceinline__ float dot_float4(const float4& a, const float4& b) {
  float acc = 0.0f;
  acc = fmaf(a.x, b.x, acc);
  acc = fmaf(a.y, b.y, acc);
  acc = fmaf(a.z, b.z, acc);
  acc = fmaf(a.w, b.w, acc);
  return acc;
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

// All-lane butterfly reduction — result is identical in every lane,
// so callers don't need a follow-up broadcast shuffle.
__device__ __forceinline__ float warp_sum_all(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ __forceinline__ float warp_broadcast_0(float value) {
  return __shfl_sync(0xffffffffu, value, 0);
}

__global__ __launch_bounds__(kThreads, 4) void gdn_prefill_kernel(
    const c10::BFloat16* __restrict__ q,
    const c10::BFloat16* __restrict__ k,
    const c10::BFloat16* __restrict__ v,
    const float* __restrict__ state_in,
    float* __restrict__ state_out,
    const float* __restrict__ A_log,
    const c10::BFloat16* __restrict__ a,
    const float* __restrict__ dt_bias,
    const c10::BFloat16* __restrict__ b,
    const int64_t* __restrict__ cu_seqlens,
    c10::BFloat16* __restrict__ output,
    int64_t num_seqs,
    double scale,
    bool has_state) {
  __builtin_assume(blockDim.x == kThreads);
  __builtin_assume(num_seqs > 0);
  const int seq_idx = blockIdx.y;
  const int head_pair_idx = blockIdx.x / kRowTilesPerHead;
  const int row_tile_idx = blockIdx.x % kRowTilesPerHead;
  const int warp_idx = threadIdx.x / kWarpSize;
  const int lane_idx = threadIdx.x % kWarpSize;
  const int row_idx = row_tile_idx * kRowsPerBlock + warp_idx;

  if (seq_idx >= num_seqs || head_pair_idx >= kHeadPairs || row_idx >= kHeadSize) {
    return;
  }
  __builtin_assume(seq_idx >= 0 && seq_idx < num_seqs);
  __builtin_assume(head_pair_idx >= 0 && head_pair_idx < kHeadPairs);
  __builtin_assume(row_idx >= 0 && row_idx < kHeadSize);
  __builtin_assume(lane_idx >= 0 && lane_idx < kWarpSize);
  __builtin_assume(warp_idx >= 0 && warp_idx < kWarpsPerBlock);

  const int v_head_a = head_pair_idx * kVHeadsPerBlock;
  const int v_head_b = v_head_a + 1;
  // V_PER_Q = V_PER_K = kVHeadsPerBlock => head_pair_idx is exactly q_head/k_head.
  const int q_head_idx = head_pair_idx;
  const int k_head_idx = head_pair_idx;

  const int col_base = lane_idx * kVecSize;
  const float scale_f = static_cast<float>(scale);
  const int64_t seq_start = cu_seqlens[seq_idx];
  const int64_t seq_end = cu_seqlens[seq_idx + 1];
  __builtin_assume(seq_end >= seq_start);

  const int64_t state_offset_a =
      (((static_cast<int64_t>(seq_idx) * kNumVHeads + v_head_a) * kHeadSize + row_idx) *
           kHeadSize +
       col_base);
  const int64_t state_offset_b =
      (((static_cast<int64_t>(seq_idx) * kNumVHeads + v_head_b) * kHeadSize + row_idx) *
           kHeadSize +
       col_base);

  float4 state_vec_a;
  float4 state_vec_b;
  if (has_state) {
    state_vec_a = reinterpret_cast<const float4*>(state_in + state_offset_a)[0];
    state_vec_b = reinterpret_cast<const float4*>(state_in + state_offset_b)[0];
  } else {
    state_vec_a = make_float4(0.f, 0.f, 0.f, 0.f);
    state_vec_b = make_float4(0.f, 0.f, 0.f, 0.f);
  }

  // G1 fusion: per-head constants for both v_heads in the pair, preloaded once per block.
  const float a_log_exp_a = expf(A_log[v_head_a]);
  const float dt_bias_a   = dt_bias[v_head_a];
  const float a_log_exp_b = expf(A_log[v_head_b]);
  const float dt_bias_b   = dt_bias[v_head_b];

  __shared__ float2 sh_gate_beta_a[kChunkSize];
  __shared__ float2 sh_gate_beta_b[kChunkSize];

  for (int64_t chunk_start = seq_start; chunk_start < seq_end; chunk_start += kChunkSize) {
    const int chunk_rem = static_cast<int>(seq_end - chunk_start);
    const int C_actual = chunk_rem < kChunkSize ? chunk_rem : kChunkSize;

    // Cooperative gate/β load for both v_heads. Total slots = 2*C_actual; threads = 64.
    // Pack as: slots [0, C_actual)  -> v_head_a, slots [C_actual, 2*C_actual) -> v_head_b.
    const int total_slots = 2 * C_actual;
    #pragma unroll
    for (int slot_base = 0; slot_base < 2 * kChunkSize; slot_base += kThreads) {
      const int slot = slot_base + threadIdx.x;
      if (slot < total_slots) {
        const bool is_b = slot >= C_actual;
        const int local_i = is_b ? (slot - C_actual) : slot;
        const int v_head = is_b ? v_head_b : v_head_a;
        const float a_log_exp_h = is_b ? a_log_exp_b : a_log_exp_a;
        const float dt_bias_h   = is_b ? dt_bias_b   : dt_bias_a;
        const int64_t t_tok = chunk_start + local_i;
        const int64_t ab_offset = t_tok * kNumVHeads + v_head;
        const float a_val = bf16_to_float(a + ab_offset);
        const float b_val = bf16_to_float(b + ab_offset);
        const float gate = expf(-a_log_exp_h * softplusf_stable(a_val + dt_bias_h));
        const float beta = 1.0f / (1.0f + expf(-b_val));
        const float2 packed = make_float2(gate, beta);
        if (is_b) {
          sh_gate_beta_b[local_i] = packed;
        } else {
          sh_gate_beta_a[local_i] = packed;
        }
      }
    }
    __syncthreads();

    #pragma unroll 4
    for (int i = 0; i < C_actual; ++i) {
      const int64_t t = chunk_start + i;
      const int64_t q_offset = ((t * kNumQHeads + q_head_idx) * kHeadSize) + col_base;
      const int64_t k_offset = ((t * kNumKHeads + k_head_idx) * kHeadSize) + col_base;
      const int64_t v_offset_a = ((t * kNumVHeads + v_head_a) * kHeadSize) + row_idx;
      const int64_t v_offset_b = v_offset_a + kHeadSize;  // contiguous v_head_b right after v_head_a

      const float4 q_vec = load_bf16x4(q + q_offset);
      const float4 k_vec = load_bf16x4(k + k_offset);

      const float2 gb_a = sh_gate_beta_a[i];
      const float2 gb_b = sh_gate_beta_b[i];
      const float gate_a = gb_a.x;
      const float beta_a = gb_a.y;
      const float gate_b = gb_b.x;
      const float beta_b = gb_b.y;
      const float v_val_a = bf16_to_float(v + v_offset_a);
      const float v_val_b = bf16_to_float(v + v_offset_b);

      const float p_kS_a = dot_float4(k_vec, state_vec_a);
      const float p_qS_a = dot_float4(q_vec, state_vec_a);
      const float p_kS_b = dot_float4(k_vec, state_vec_b);
      const float p_qS_b = dot_float4(q_vec, state_vec_b);
      const float p_qk = dot_float4(q_vec, k_vec);

      const float kS_a = warp_sum_all(p_kS_a);
      const float qS_a = warp_sum_all(p_qS_a);
      const float kS_b = warp_sum_all(p_kS_b);
      const float qS_b = warp_sum_all(p_qS_b);
      const float qk = warp_sum_all(p_qk);

      const float diff_a = beta_a * (v_val_a - gate_a * kS_a);
      const float diff_b = beta_b * (v_val_b - gate_b * kS_b);
      const float out_a = gate_a * qS_a + qk * diff_a;
      const float out_b = gate_b * qS_b + qk * diff_b;

      state_vec_a.x = fmaf(k_vec.x, diff_a, gate_a * state_vec_a.x);
      state_vec_a.y = fmaf(k_vec.y, diff_a, gate_a * state_vec_a.y);
      state_vec_a.z = fmaf(k_vec.z, diff_a, gate_a * state_vec_a.z);
      state_vec_a.w = fmaf(k_vec.w, diff_a, gate_a * state_vec_a.w);

      state_vec_b.x = fmaf(k_vec.x, diff_b, gate_b * state_vec_b.x);
      state_vec_b.y = fmaf(k_vec.y, diff_b, gate_b * state_vec_b.y);
      state_vec_b.z = fmaf(k_vec.z, diff_b, gate_b * state_vec_b.z);
      state_vec_b.w = fmaf(k_vec.w, diff_b, gate_b * state_vec_b.w);

      if (lane_idx == 0) {
        float_to_bf16(scale_f * out_a, output + v_offset_a);
        float_to_bf16(scale_f * out_b, output + v_offset_b);
      }
    }
    __syncthreads();
  }

  reinterpret_cast<float4*>(state_out + state_offset_a)[0] = state_vec_a;
  reinterpret_cast<float4*>(state_out + state_offset_b)[0] = state_vec_b;
}

}  // namespace

void gdn_prefill_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    c10::optional<torch::Tensor> state,
    torch::Tensor A_log,
    torch::Tensor a,
    torch::Tensor dt_bias,
    torch::Tensor b,
    torch::Tensor cu_seqlens,
    double scale,
    torch::Tensor output,
    torch::Tensor new_state) {
  CHECK_CUDA(q);
  CHECK_CUDA(k);
  CHECK_CUDA(v);
  CHECK_CUDA(A_log);
  CHECK_CUDA(a);
  CHECK_CUDA(dt_bias);
  CHECK_CUDA(b);
  CHECK_CUDA(cu_seqlens);
  CHECK_CUDA(output);
  CHECK_CUDA(new_state);

  CHECK_CONTIGUOUS(q);
  CHECK_CONTIGUOUS(k);
  CHECK_CONTIGUOUS(v);
  CHECK_CONTIGUOUS(A_log);
  CHECK_CONTIGUOUS(a);
  CHECK_CONTIGUOUS(dt_bias);
  CHECK_CONTIGUOUS(b);
  CHECK_CONTIGUOUS(cu_seqlens);
  CHECK_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(new_state);

  CHECK_BF16(q);
  CHECK_BF16(k);
  CHECK_BF16(v);
  CHECK_BF16(a);
  CHECK_BF16(b);
  CHECK_BF16(output);
  CHECK_F32(A_log);
  CHECK_F32(dt_bias);
  CHECK_F32(new_state);
  CHECK_I64(cu_seqlens);

  TORCH_CHECK(q.dim() == 3, "q must have shape [total_seq_len, 4, 128]");
  TORCH_CHECK(k.dim() == 3, "k must have shape [total_seq_len, 4, 128]");
  TORCH_CHECK(v.dim() == 3, "v must have shape [total_seq_len, 8, 128]");
  TORCH_CHECK(q.size(1) == kNumQHeads && k.size(1) == kNumKHeads && v.size(1) == kNumVHeads,
              "unexpected head counts");
  TORCH_CHECK(q.size(2) == kHeadSize && k.size(2) == kHeadSize && v.size(2) == kHeadSize,
              "head size must be 128");
  TORCH_CHECK(A_log.numel() == kNumVHeads, "A_log must have 8 elements");
  TORCH_CHECK(dt_bias.numel() == kNumVHeads, "dt_bias must have 8 elements");
  TORCH_CHECK(a.size(0) == q.size(0) && a.size(1) == kNumVHeads, "a must have shape [T, 8]");
  TORCH_CHECK(b.size(0) == q.size(0) && b.size(1) == kNumVHeads, "b must have shape [T, 8]");
  TORCH_CHECK(cu_seqlens.dim() == 1 && cu_seqlens.numel() >= 2, "cu_seqlens must be [N+1]");

  c10::cuda::CUDAGuard device_guard(q.device());

  if (scale == 0.0) {
    scale = 1.0 / std::sqrt(static_cast<double>(kHeadSize));
  }

  const bool has_state = state.has_value() && state.value().defined();
  torch::Tensor state_in;
  if (has_state) {
    state_in = state.value();
    CHECK_CUDA(state_in);
    CHECK_CONTIGUOUS(state_in);
    CHECK_F32(state_in);
  }

  const int64_t num_seqs = cu_seqlens.numel() - 1;

  if (has_state) {
    TORCH_CHECK(
        state_in.sizes() == new_state.sizes(),
        "state must have shape [num_seqs, 8, 128, 128]");
  }
  TORCH_CHECK(
      output.dim() == 3 && output.size(0) == q.size(0) && output.size(1) == kNumVHeads &&
          output.size(2) == kHeadSize,
      "output must have shape [total_seq_len, 8, 128]");
  TORCH_CHECK(
      new_state.dim() == 4 && new_state.size(0) == num_seqs && new_state.size(1) == kNumVHeads &&
          new_state.size(2) == kHeadSize && new_state.size(3) == kHeadSize,
      "new_state must have shape [num_seqs, 8, 128, 128]");
  TORCH_CHECK(output.device() == q.device(), "output must be on the same device as q");
  TORCH_CHECK(new_state.device() == q.device(), "new_state must be on the same device as q");

  const dim3 grid(kHeadPairs * kRowTilesPerHead, static_cast<unsigned int>(num_seqs), 1);
  const dim3 block(kThreads, 1, 1);

  auto stream = c10::cuda::getDefaultCUDAStream();

  gdn_prefill_kernel<<<grid, block, 0, stream.stream()>>>(
      q.data_ptr<c10::BFloat16>(),
      k.data_ptr<c10::BFloat16>(),
      v.data_ptr<c10::BFloat16>(),
      has_state ? state_in.data_ptr<float>() : nullptr,
      new_state.data_ptr<float>(),
      A_log.data_ptr<float>(),
      a.data_ptr<c10::BFloat16>(),
      dt_bias.data_ptr<float>(),
      b.data_ptr<c10::BFloat16>(),
      cu_seqlens.data_ptr<int64_t>(),
      output.data_ptr<c10::BFloat16>(),
      num_seqs,
      scale,
      has_state);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
