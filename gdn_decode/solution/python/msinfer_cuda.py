"""GDN decode — CUDA C column-parallel kernel.

Algorithm change vs CuTe DSL v9 (row-parallel):

  v9 (row-parallel):
    Grid  = (B * HV * 4, 1, 1),  Block = (32, 1, 1)
    Thread[tid] owns row (split*32+tid), iterates over all 128 cols.
    State load: thread[tid] loads state[row_tid, i*4 : i*4+4]
      → 32 threads × stride-128 → 32 separate cache lines per tile  (non-coalesced)

  This version (column-parallel):
    Grid  = (B * HV * kSplits, 1, 1),  Block = (128, 1, 1)
    Thread[col] owns column col, iterates over rowsPerBlock rows.
    State load: 128 threads load state[r, 0..127] simultaneously
      → 128 consecutive floats = 4 cache lines  (perfectly coalesced)
    State store: same pattern, perfectly coalesced.

    Plus: q, k, v loaded into smem once via coalesced load → reused per row.

Reduction: per row, partial ov/qs are warp-reduced then cross-warp via smem (4 warps).
Two __syncthreads() per row; total ~67 syncs for kSplits=4 (rowsPerBlock=32).
"""
from __future__ import annotations

import math
import os
import threading

import torch

HQ, HK, HV, D = 4, 4, 8, 128
DEFAULT_SCALE = 1.0 / math.sqrt(float(D))
_KSPLITS = int(os.environ.get("MSINFER_KSPLITS", "4"))

# ---------------------------------------------------------------------------
# CUDA C kernel source
# ---------------------------------------------------------------------------
_CUDA_SRC = r"""
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

// Column-parallel GDN decode
//   Grid  = (B * HV * kSplits, 1, 1)
//   Block = (128, 1, 1)  →  4 warps, 1 thread per state column
//
// State access coalescing:
//   Load:  128 threads read state[r, 0..127]  →  4 × 128-byte cache lines
//   Store: same pattern  →  perfectly coalesced
//
// v9 comparison (row-parallel, 32 threads):
//   Load:  thread[tid] reads state[row+tid, col:col+4]  →  stride=512B
//          →  32 separate cache-line transactions per tile (8× worse)
__global__ void __launch_bounds__(128, 8)
gdn_decode_col_v1(
    const __nv_bfloat16* __restrict__ q,        // (B, 1, HQ, D) bf16
    const __nv_bfloat16* __restrict__ k,        // (B, 1, HK, D) bf16
    const __nv_bfloat16* __restrict__ v,        // (B, 1, HV, D) bf16
    const float*         __restrict__ state_in, // (B, HV, D, D) f32
    const float*         __restrict__ A_log,    // (HV,) f32
    const __nv_bfloat16* __restrict__ a_in,     // (B, 1, HV) bf16
    const float*         __restrict__ dt_bias,  // (HV,) f32
    const __nv_bfloat16* __restrict__ b_in,     // (B, 1, HV) bf16
    __nv_bfloat16*       __restrict__ out,      // (B, 1, HV, D) bf16
    float*               __restrict__ state_out,// (B, HV, D, D) f32
    float scale,
    int   kSplits
) {
    const int D_  = 128, HV_ = 8, HQ_ = 4, HK_ = 4;

    const int bid     = blockIdx.x;
    const int split   = bid % kSplits;
    const int v_head  = (bid / kSplits) % HV_;
    const int batch   = bid / (kSplits * HV_);
    const int qk_head = v_head / (HV_ / HQ_);
    const int col     = threadIdx.x;           // 0..127
    const int warp_id = col >> 5;
    const int lane_id = col & 31;
    const int rows    = D_ / kSplits;         // rows handled by this block
    const int row0    = split * rows;

    // Shared memory layout:
    //   sQ[128], sK[128], sV[128]  — coalesced-loaded input vectors
    //   w_ov[4], w_qs[4]           — warp-level partial sums
    //   s_delta, s_qs              — broadcast scalars
    __shared__ float sQ[128];
    __shared__ float sK[128];
    __shared__ float sV[128];
    __shared__ float w_ov[4];
    __shared__ float w_qs[4];
    __shared__ float s_delta;
    __shared__ float s_qs_final;

    // ---- Coalesced load of Q, K, V into smem (128 threads × 1 float = 4 CL each) ----
    sQ[col] = __bfloat162float(q[batch * HQ_ * D_ + qk_head * D_ + col]);
    sK[col] = __bfloat162float(k[batch * HK_ * D_ + qk_head * D_ + col]);
    sV[col] = __bfloat162float(v[batch * HV_ * D_ + v_head  * D_ + col]);

    // ---- Gate scalars (all threads compute identically — pure MUFU, no mem) ----
    const float a_val = __bfloat162float(a_in[batch * HV_ + v_head]) + dt_bias[v_head];
    const float g     = __expf(-__expf(A_log[v_head]) * softplus_stable(a_val));
    const float beta  = sigmoid_stable(__bfloat162float(b_in[batch * HV_ + v_head]));

    __syncthreads();  // sQ, sK, sV ready

    const float k_reg = sK[col];
    const float q_reg = sQ[col];

    // ---- qk = dot(q, k): 128-thread reduction ----
    float qk = q_reg * k_reg;
    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1)
        qk += __shfl_xor_sync(0xffffffff, qk, off);
    if (lane_id == 0) w_ov[warp_id] = qk;      // reuse w_ov temporarily
    __syncthreads();
    if (col == 0) s_delta = w_ov[0]+w_ov[1]+w_ov[2]+w_ov[3];
    __syncthreads();
    qk = s_delta;

    // ---- State / output base pointers ----
    const float*         si = state_in  + ((batch * HV_ + v_head) * D_ * D_);
    float*               so = state_out + ((batch * HV_ + v_head) * D_ * D_);
    __nv_bfloat16*       op = out       + ((batch * HV_ + v_head) * D_);

    // ---- Main loop: rowsPerBlock rows, perfectly coalesced state access ----
    #pragma unroll 2
    for (int r = row0; r < row0 + rows; ++r) {

        // Coalesced state load: 128 threads × 1 float = 4 × 128-byte cache lines
        // __ldg: read-only cache (state_in is read-only in this kernel)
        const float s_reg = __ldg(&si[r * D_ + col]);
        const float s_new = g * s_reg;

        // Partial ov = k[col]*s_new, qs = q[col]*s_new
        float ov_p = k_reg * s_new;
        float qs_p = q_reg * s_new;

        // Warp reduce (5 levels of butterfly)
        #pragma unroll
        for (int off = 16; off >= 1; off >>= 1) {
            ov_p += __shfl_xor_sync(0xffffffff, ov_p, off);
            qs_p += __shfl_xor_sync(0xffffffff, qs_p, off);
        }

        // Cross-warp reduce via smem (4 warp lane-0s write, thread-0 sums)
        if (lane_id == 0) { w_ov[warp_id] = ov_p; w_qs[warp_id] = qs_p; }
        __syncthreads();

        if (col == 0) {
            const float ov    = w_ov[0]+w_ov[1]+w_ov[2]+w_ov[3];
            const float qs    = w_qs[0]+w_qs[1]+w_qs[2]+w_qs[3];
            // sV[col] = v[batch, 0, v_head, col], so sV[r] = v[batch, 0, v_head, r]
            const float delta = beta * (sV[r] - ov);
            s_delta     = delta;
            s_qs_final  = qs;
            op[r] = __float2bfloat16(scale * (qs + delta * qk));
        }
        __syncthreads();

        const float delta = s_delta;

        // Coalesced state store: 128 consecutive floats = 4 × 128-byte cache lines
        so[r * D_ + col] = s_new + k_reg * delta;
    }
}

// C-linkage launcher so the C++ wrapper needs no CUDA headers
extern "C" void launch_gdn_decode_col_v1(
    const void* q, const void* k, const void* v,
    const float* state_in, const float* A_log,
    const void* a_in, const float* dt_bias, const void* b_in,
    void* out, float* state_out,
    float scale, int kSplits, int B, cudaStream_t stream
) {
    dim3 grid(B * 8 * kSplits);   // HV = 8
    dim3 block(128);
    gdn_decode_col_v1<<<grid, block, 0, stream>>>(
        (const __nv_bfloat16*)q,
        (const __nv_bfloat16*)k,
        (const __nv_bfloat16*)v,
        state_in, A_log,
        (const __nv_bfloat16*)a_in, dt_bias,
        (const __nv_bfloat16*)b_in,
        (__nv_bfloat16*)out, state_out,
        scale, kSplits
    );
}
"""

# ---------------------------------------------------------------------------
# C++ host-side wrapper
# ---------------------------------------------------------------------------
_CPP_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

// Declared in the CUDA translation unit with C linkage — no CUDA headers needed here
extern "C" void launch_gdn_decode_col_v1(
    const void* q, const void* k, const void* v,
    const float* state_in, const float* A_log,
    const void* a_in, const float* dt_bias, const void* b_in,
    void* out, float* state_out,
    float scale, int kSplits, int B, cudaStream_t stream);

void gdn_decode_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor state_in,
    torch::Tensor A_log,
    torch::Tensor a_in,
    torch::Tensor dt_bias,
    torch::Tensor b_in,
    torch::Tensor out,
    torch::Tensor state_out,
    float scale,
    int kSplits
) {
    launch_gdn_decode_col_v1(
        q.data_ptr(), k.data_ptr(), v.data_ptr(),
        state_in.data_ptr<float>(), A_log.data_ptr<float>(),
        a_in.data_ptr(), dt_bias.data_ptr<float>(), b_in.data_ptr(),
        out.data_ptr(), state_out.data_ptr<float>(),
        scale, kSplits, (int)q.size(0),
        at::cuda::getCurrentCUDAStream()
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gdn_decode_cuda", &gdn_decode_cuda, "GDN decode column-parallel CUDA C");
}
"""

# ---------------------------------------------------------------------------
# Compile and cache
# ---------------------------------------------------------------------------
_LOCK = threading.Lock()
_EXT = None


def _get_ext():
    global _EXT
    if _EXT is None:
        with _LOCK:
            if _EXT is None:
                from torch.utils.cpp_extension import load_inline
                _EXT = load_inline(
                    name="gdn_cuda_col_v1",
                    cpp_sources=_CPP_SRC,
                    cuda_sources=_CUDA_SRC,
                    extra_cuda_cflags=[
                        "-O3",
                        "--use_fast_math",
                        "-arch=sm_100",
                    ],
                    verbose=False,
                )
    return _EXT


def run(q, k, v, state, A_log, a, dt_bias, b, scale, output, new_state):
    """DPS entrypoint — drop-in replacement for msinfer_entry.py::run."""
    ext = _get_ext()
    ext.gdn_decode_cuda(
        q, k, v, state, A_log, a, dt_bias, b,
        output, new_state,
        DEFAULT_SCALE, _KSPLITS,
    )
