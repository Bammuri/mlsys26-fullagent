"""GDN decode — CUDA C hybrid kernel (v1 + v4_tma).

Two variants compiled:
  hybrid  - v1 for B<=4, v4_tma for B>4 (default)
  v1      - column-parallel __ldg (fallback)
  v4_tma  - warp-per-row + bulk TMA load+store (diagnostic)

Selected via MSINFER_KERNEL env var. Requires sm_100a (Blackwell).

Perf (Modal B200, 본판 50iter×3trial, 54 workloads):
  B<=8 : 0.012–0.015 ms  (v1 or v4_tma path, up to 1.7× over v1-only)
  B=32 : 0.018 ms         (v4_tma, 1.7× over v1)
  B=64 : 0.020 ms         (v4_tma, 2.2× over v1)
"""
from __future__ import annotations

import os
import sys
import threading

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tma_desc import get_tensor_map_pair  # noqa: E402

HQ, HV, D = 4, 8, 128

# ---------------------------------------------------------------------------
# CUDA C kernel — two variants in one file
# ---------------------------------------------------------------------------
_CUDA_SRC = r"""
#include <cuda.h>             // CUtensorMap
#include <cuda_runtime.h>
#include <cuda_bf16.h>

static __device__ __forceinline__ float softplus_stable(float x) {
    if (x >  20.0f) return x;
    if (x < -20.0f) return __expf(x);
    return __logf(1.0f + __expf(x));
}

static __device__ __forceinline__ float sigmoid_stable(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + __expf(-x));
    float e = __expf(x);
    return e / (1.0f + e);
}

// ── v1: plain __ldg loads ──────────────────────────────────────────────────
__global__ void __launch_bounds__(128, 8)
gdn_decode_col_v1(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float*         __restrict__ state_in,
    const float*         __restrict__ A_log,
    const __nv_bfloat16* __restrict__ a_in,
    const float*         __restrict__ dt_bias,
    const __nv_bfloat16* __restrict__ b_in,
    __nv_bfloat16*       __restrict__ out,
    float*               __restrict__ state_out,
    float scale, int kSplits
) {
    const int D_=128, HV_=8, HQ_=4, HK_=4;
    const int bid     = blockIdx.x;
    const int split   = bid % kSplits;
    const int v_head  = (bid / kSplits) % HV_;
    const int batch   = bid / (kSplits * HV_);
    const int qk_head = v_head / (HV_/HQ_);
    const int col     = threadIdx.x;
    const int warp_id = col >> 5;
    const int lane_id = col & 31;
    const int rows    = D_ / kSplits;
    const int row0    = split * rows;

    __shared__ float sQ[128], sK[128], sV[128];
    __shared__ float w_ov[4], w_qs[4];
    __shared__ float s_delta, s_qs_final;

    sQ[col] = __bfloat162float(q[batch*HQ_*D_ + qk_head*D_ + col]);
    sK[col] = __bfloat162float(k[batch*HK_*D_ + qk_head*D_ + col]);
    sV[col] = __bfloat162float(v[batch*HV_*D_ + v_head *D_ + col]);

    const float a_val = __bfloat162float(a_in[batch*HV_+v_head]) + dt_bias[v_head];
    const float g     = __expf(-__expf(A_log[v_head]) * softplus_stable(a_val));
    const float beta  = sigmoid_stable(__bfloat162float(b_in[batch*HV_+v_head]));

    __syncthreads();

    const float k_reg = sK[col], q_reg = sQ[col];

    // qk = dot(q,k)
    float qk = q_reg * k_reg;
    #pragma unroll
    for (int off=16; off>=1; off>>=1) qk += __shfl_xor_sync(0xffffffff, qk, off);
    if (lane_id==0) w_ov[warp_id] = qk;
    __syncthreads();
    if (col==0) s_delta = w_ov[0]+w_ov[1]+w_ov[2]+w_ov[3];
    __syncthreads();
    qk = s_delta;

    const float* si = state_in  + (batch*HV_+v_head)*D_*D_;
    float*       so = state_out + (batch*HV_+v_head)*D_*D_;
    __nv_bfloat16* op = out + (batch*HV_+v_head)*D_;

    #pragma unroll 2
    for (int r=row0; r<row0+rows; ++r) {
        const float s_new = g * __ldg(&si[r*D_+col]);
        float ov_p = k_reg * s_new, qs_p = q_reg * s_new;
        #pragma unroll
        for (int off=16; off>=1; off>>=1) {
            ov_p += __shfl_xor_sync(0xffffffff, ov_p, off);
            qs_p += __shfl_xor_sync(0xffffffff, qs_p, off);
        }
        if (lane_id==0) { w_ov[warp_id]=ov_p; w_qs[warp_id]=qs_p; }
        __syncthreads();
        if (col==0) {
            const float delta = beta*(sV[r]-w_ov[0]-w_ov[1]-w_ov[2]-w_ov[3]);
            s_delta     = delta;
            s_qs_final  = w_qs[0]+w_qs[1]+w_qs[2]+w_qs[3];
            op[r] = __float2bfloat16(scale*(s_qs_final + delta*qk));
        }
        __syncthreads();
        so[r*D_+col] = s_new + k_reg * s_delta;
    }
}

// ── C-linkage launchers ───────────────────────────────────────────────────
extern "C" void launch_gdn_v1(
    const void* q, const void* k, const void* v,
    const float* si, const float* A_log,
    const void* a_in, const float* dt_bias, const void* b_in,
    void* out, float* so,
    float scale, int kSplits, int B, cudaStream_t stream)
{
    gdn_decode_col_v1<<<dim3(B*8*kSplits), dim3(128), 0, stream>>>(
        (const __nv_bfloat16*)q, (const __nv_bfloat16*)k, (const __nv_bfloat16*)v,
        si, A_log, (const __nv_bfloat16*)a_in, dt_bias, (const __nv_bfloat16*)b_in,
        (__nv_bfloat16*)out, so, scale, kSplits);
}


// ── v4 FINAL: warp-per-row, bulk TMA load + bulk TMA store, in-place smem ─
template<int kRows>
__global__ void __launch_bounds__(128, 8)
gdn_decode_col_v4_tma(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const __grid_constant__ CUtensorMap tensor_map_in,
    const __grid_constant__ CUtensorMap tensor_map_out,
    const float*         __restrict__ A_log,
    const __nv_bfloat16* __restrict__ a_in,
    const float*         __restrict__ dt_bias,
    const __nv_bfloat16* __restrict__ b_in,
    __nv_bfloat16*       __restrict__ out,
    float scale
) {
    constexpr int D_=128, HV_=8, HQ_=4, HK_=4;
    constexpr int kSplits = D_ / kRows;
    const int bid     = blockIdx.x;
    const int split   = bid % kSplits;
    const int v_head  = (bid / kSplits) % HV_;
    const int batch   = bid / (kSplits * HV_);
    const int qk_head = v_head / (HV_/HQ_);
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int row0    = split * kRows;
    const int col0    = lane_id * 4;
    const int row_base_elem = batch * HV_ * D_ + v_head * D_ + row0;

    __shared__ float sQ[128], sK[128], sV[128];
    __shared__ float s_qk;
    __shared__ alignas(128) float state_tile[kRows][128];   // 128-byte aligned for TMA
    __shared__ alignas(8) uint64_t mbar_load;

    sQ[tid] = __bfloat162float(q[batch*HQ_*D_ + qk_head*D_ + tid]);
    sK[tid] = __bfloat162float(k[batch*HK_*D_ + qk_head*D_ + tid]);
    sV[tid] = __bfloat162float(v[batch*HV_*D_ + v_head *D_ + tid]);

    const float a_val = __bfloat162float(a_in[batch*HV_+v_head]) + dt_bias[v_head];
    const float g     = __expf(-__expf(A_log[v_head]) * softplus_stable(a_val));
    const float beta  = sigmoid_stable(__bfloat162float(b_in[batch*HV_+v_head]));

    if (tid == 0) {
        uint32_t mbar_ptr = __cvta_generic_to_shared(&mbar_load);
        uint32_t tile_ptr = __cvta_generic_to_shared(&state_tile[0][0]);
        asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(mbar_ptr));
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        asm volatile(
            "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
            :: "r"(mbar_ptr), "r"((unsigned)(kRows * 128 * 4))
        );
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%2, %3}], [%4];"
            :: "r"(tile_ptr),
               "l"(&tensor_map_in),
               "r"(0),
               "r"(row_base_elem),
               "r"(mbar_ptr)
        );
    }
    __syncthreads();

    if (warp_id == 0) {
        float p = sQ[col0+0]*sK[col0+0] + sQ[col0+1]*sK[col0+1]
                + sQ[col0+2]*sK[col0+2] + sQ[col0+3]*sK[col0+3];
        #pragma unroll
        for (int off=16; off>=1; off>>=1) p += __shfl_xor_sync(0xffffffff, p, off);
        if (lane_id == 0) s_qk = p;
    }

    if (tid == 0) {
        uint32_t mbar_ptr = __cvta_generic_to_shared(&mbar_load);
        asm volatile(
            "{.reg .pred P;\n"
            " WAIT_%=: mbarrier.try_wait.parity.shared.b64 P, [%0], 0;\n"
            " @P bra DONE_%=;\n"
            " bra WAIT_%=;\n"
            " DONE_%=: }"
            :: "r"(mbar_ptr)
        );
    }
    __syncthreads();
    const float qk = s_qk;

    float k_reg[4], q_reg[4];
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        k_reg[c] = sK[col0 + c];
        q_reg[c] = sQ[col0 + c];
    }

    __nv_bfloat16* op = out + (batch*HV_+v_head)*D_;

    constexpr int iters = kRows / 4;
    #pragma unroll 1
    for (int iter = 0; iter < iters; ++iter) {
        const int local_row  = iter*4 + warp_id;
        const int global_row = row0 + local_row;

        float st[4];
        #pragma unroll
        for (int c = 0; c < 4; ++c) st[c] = state_tile[local_row][col0 + c] * g;

        float ov_p = k_reg[0]*st[0] + k_reg[1]*st[1] + k_reg[2]*st[2] + k_reg[3]*st[3];
        float qs_p = q_reg[0]*st[0] + q_reg[1]*st[1] + q_reg[2]*st[2] + q_reg[3]*st[3];
        #pragma unroll
        for (int off=16; off>=1; off>>=1) {
            ov_p += __shfl_xor_sync(0xffffffff, ov_p, off);
            qs_p += __shfl_xor_sync(0xffffffff, qs_p, off);
        }

        float delta;
        if (lane_id == 0) {
            delta = beta * (sV[global_row] - ov_p);
            op[global_row] = __float2bfloat16(scale * (qs_p + delta * qk));
        }
        delta = __shfl_sync(0xffffffff, delta, 0);

        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            state_tile[local_row][col0 + c] = st[c] + k_reg[c] * delta;
        }
    }

    __syncthreads();
    // Fence needed before TMA store so smem writes in the compute loop
    // are visible to the async TMA engine.
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

    if (tid == 0) {
        uint32_t tile_ptr = __cvta_generic_to_shared(&state_tile[0][0]);
        asm volatile(
            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
            " [%0, {%1, %2}], [%3];"
            :: "l"(&tensor_map_out),
               "r"(0),
               "r"(row_base_elem),
               "r"(tile_ptr)
        );
        asm volatile("cp.async.bulk.commit_group;");
        asm volatile("cp.async.bulk.wait_group 0;");
    }
    __syncthreads();
}

extern "C" void launch_gdn_v4_tma(
    const void* q, const void* k, const void* v,
    const CUtensorMap* desc_in,
    const CUtensorMap* desc_out,
    const float* A_log,
    const void* a_in, const float* dt_bias, const void* b_in,
    void* out,
    float scale, int kSplits, int B, cudaStream_t stream)
{
    dim3 grid(B * 8 * kSplits);
    dim3 block(128);
    if (kSplits == 4) {
        gdn_decode_col_v4_tma<32><<<grid, block, 0, stream>>>(
            (const __nv_bfloat16*)q, (const __nv_bfloat16*)k, (const __nv_bfloat16*)v,
            *desc_in, *desc_out,
            A_log, (const __nv_bfloat16*)a_in, dt_bias, (const __nv_bfloat16*)b_in,
            (__nv_bfloat16*)out, scale);
    } else if (kSplits == 8) {
        gdn_decode_col_v4_tma<16><<<grid, block, 0, stream>>>(
            (const __nv_bfloat16*)q, (const __nv_bfloat16*)k, (const __nv_bfloat16*)v,
            *desc_in, *desc_out,
            A_log, (const __nv_bfloat16*)a_in, dt_bias, (const __nv_bfloat16*)b_in,
            (__nv_bfloat16*)out, scale);
    } else {
        abort();
    }
}
"""

# ---------------------------------------------------------------------------
# C++ wrapper
# ---------------------------------------------------------------------------
_CPP_SRC = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cstdlib>
#include <c10/cuda/CUDAStream.h>

extern "C" void launch_gdn_v1(
    const void*, const void*, const void*,
    const float*, const float*,
    const void*, const float*, const void*,
    void*, float*, float, int, int, cudaStream_t);

extern "C" void launch_gdn_v4_tma(
    const void*, const void*, const void*,
    const CUtensorMap*, const CUtensorMap*,
    const float*,
    const void*, const float*, const void*,
    void*, float, int, int, cudaStream_t);

void gdn_decode_v1(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor si, torch::Tensor A_log,
    torch::Tensor a_in, torch::Tensor dt_bias, torch::Tensor b_in,
    torch::Tensor out, torch::Tensor so, float scale, int kSplits)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    int B = q.size(0);
    launch_gdn_v1(
        q.data_ptr(), k.data_ptr(), v.data_ptr(),
        si.data_ptr<float>(), A_log.data_ptr<float>(),
        a_in.data_ptr(), dt_bias.data_ptr<float>(), b_in.data_ptr(),
        out.data_ptr(), so.data_ptr<float>(),
        scale, kSplits, B, stream);
}

void gdn_decode_v4_tma(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor A_log,
    torch::Tensor a_in, torch::Tensor dt_bias, torch::Tensor b_in,
    torch::Tensor out,
    py::bytes desc_in_bytes, py::bytes desc_out_bytes,
    float scale, int kSplits)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    int B = q.size(0);
    std::string din = desc_in_bytes;
    std::string dout = desc_out_bytes;
    TORCH_CHECK(din.size()  == 128, "desc_in must be 128 bytes");
    TORCH_CHECK(dout.size() == 128, "desc_out must be 128 bytes");
    launch_gdn_v4_tma(
        q.data_ptr(), k.data_ptr(), v.data_ptr(),
        reinterpret_cast<const CUtensorMap*>(din.data()),
        reinterpret_cast<const CUtensorMap*>(dout.data()),
        A_log.data_ptr<float>(),
        a_in.data_ptr(), dt_bias.data_ptr<float>(), b_in.data_ptr(),
        out.data_ptr(),
        scale, kSplits, B, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gdn_decode_v1",     &gdn_decode_v1,     "col-parallel v1 (plain ldg, fallback)");
    m.def("gdn_decode_v4_tma", &gdn_decode_v4_tma, "v4 warp-per-row + bulk TMA (final)");
}
"""

# ---------------------------------------------------------------------------
# Compile and cache
# ---------------------------------------------------------------------------
_LOCK = threading.Lock()
_EXT  = None

def _get_ext():
    global _EXT
    if _EXT is None:
        with _LOCK:
            if _EXT is None:
                from torch.utils.cpp_extension import load_inline
                _EXT = load_inline(
                    name="gdn_cuda_col_v4",
                    cpp_sources=_CPP_SRC,
                    cuda_sources=_CUDA_SRC,
                    extra_cuda_cflags=[
                        "-O3",
                        "--use_fast_math",
                        "-arch=sm_100a",   # Blackwell native ISA
                    ],
                    verbose=False,
                )
    return _EXT


# ---------------------------------------------------------------------------
# Kernel selection
# ---------------------------------------------------------------------------
#
# MSINFER_KERNEL env var:
#   "hybrid"  - v1 for B<=4, v4_tma for B>4 (default)
#   "v1"      - plain __ldg column-parallel (fallback)
#   "v4_tma"  - warp-per-row + bulk TMA (diagnostic)
_VALID_KERNELS = ("v1", "v4_tma", "hybrid")


def _select_kernel() -> str:
    name = os.environ.get("MSINFER_KERNEL", "hybrid")
    if name not in _VALID_KERNELS:
        raise ValueError(
            f"MSINFER_KERNEL={name!r} invalid; must be one of {_VALID_KERNELS}"
        )
    return name


def _ksplits(B: int) -> int:
    """Adaptive kSplits: more blocks at small batch for better SM coverage."""
    if B <= 4:
        return 8   # B=1: 64 blocks (vs 32 for kSplits=4)
    return 4       # B>4: 32+ blocks, coalescing benefit dominates


def _dispatch_v4_tma(ext, q, k, v, state, A_log, a, dt_bias, b, output, new_state, scale, B, ks):
    desc_in, desc_out = get_tensor_map_pair(
        state.data_ptr(), new_state.data_ptr(), B, HV, D // ks,
    )
    ext.gdn_decode_v4_tma(q, k, v, A_log, a, dt_bias, b, output, desc_in, desc_out, scale, ks)


def run(q, k, v, state, A_log, a, dt_bias, b, scale, output, new_state):
    """DPS entrypoint."""
    ext    = _get_ext()
    B      = q.size(0)
    ks     = _ksplits(B)
    kernel = _select_kernel()
    if kernel == "v1":
        ext.gdn_decode_v1(q, k, v, state, A_log, a, dt_bias, b, output, new_state, scale, ks)
    elif kernel == "v4_tma":
        _dispatch_v4_tma(ext, q, k, v, state, A_log, a, dt_bias, b, output, new_state, scale, B, ks)
    elif kernel == "hybrid":
        if B <= 4:
            ext.gdn_decode_v1(q, k, v, state, A_log, a, dt_bias, b, output, new_state, scale, ks)
        else:
            _dispatch_v4_tma(ext, q, k, v, state, A_log, a, dt_bias, b, output, new_state, scale, B, ks)
    else:  # pragma: no cover
        raise AssertionError(f"unreachable: {kernel!r}")
