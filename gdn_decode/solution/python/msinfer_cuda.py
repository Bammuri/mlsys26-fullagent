"""GDN decode — CUDA C column-parallel kernel v2.

Optimizations over v1:
  1. Adaptive kSplits: B<=4 -> kSplits=8 (2x more blocks -> better SM coverage)
  2. sm_100a compile flag: Blackwell native ISA
  3. TMA (cp.async double-buffering): overlap state load with computation

Column-parallel design (Block=128 threads, 1 per column):
  State access: 128 threads × 1 float per row = 4 cache lines (perfectly coalesced)
  vs v9 row-parallel: 32 threads × stride-128 = 32 cache lines (8× worse)
"""
from __future__ import annotations

import math
import os
import threading

import torch

HQ, HK, HV, D = 4, 4, 8, 128
DEFAULT_SCALE = 1.0 / math.sqrt(float(D))

# ---------------------------------------------------------------------------
# CUDA C kernel — two variants in one file
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

// ── v2: cp.async double-buffering (TMA-lite) ──────────────────────────────
// Prefetch state[r+1] into smem while computing state[r].
// cp.async hides global memory latency behind FMA+shuffle computation.
__global__ void __launch_bounds__(128, 6)
gdn_decode_col_v2_tma(
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

    // Double buffer: buf[0] and buf[1] alternate for consecutive rows
    __shared__ float sQ[128], sK[128], sV[128];
    __shared__ float state_buf[2][128];   // ping-pong state buffers
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

    // Prefetch first row into buf[0]
    {
        uint32_t smem_ptr = __cvta_generic_to_shared(&state_buf[0][col]);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;"
                     :: "r"(smem_ptr), "l"((unsigned long long)(si + row0*D_ + col)));
        asm volatile("cp.async.commit_group;");
    }

    #pragma unroll 2
    for (int r=row0; r<row0+rows; ++r) {
        const int cur  = (r-row0) & 1;
        const int nxt  = 1 - cur;

        // Prefetch next row while current is being waited on
        if (r+1 < row0+rows) {
            uint32_t smem_ptr = __cvta_generic_to_shared(&state_buf[nxt][col]);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4;"
                         :: "r"(smem_ptr), "l"((unsigned long long)(si + (r+1)*D_ + col)));
            asm volatile("cp.async.commit_group;");
            // Wait for all but the in-flight prefetch (keep 1 outstanding)
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_all;");
        }
        __syncthreads();   // ensure all threads see the completed cp.async data

        const float s_new = g * state_buf[cur][col];
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
            s_delta    = delta;
            s_qs_final = w_qs[0]+w_qs[1]+w_qs[2]+w_qs[3];
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

extern "C" void launch_gdn_v2_tma(
    const void* q, const void* k, const void* v,
    const float* si, const float* A_log,
    const void* a_in, const float* dt_bias, const void* b_in,
    void* out, float* so,
    float scale, int kSplits, int B, cudaStream_t stream)
{
    gdn_decode_col_v2_tma<<<dim3(B*8*kSplits), dim3(128), 0, stream>>>(
        (const __nv_bfloat16*)q, (const __nv_bfloat16*)k, (const __nv_bfloat16*)v,
        si, A_log, (const __nv_bfloat16*)a_in, dt_bias, (const __nv_bfloat16*)b_in,
        (__nv_bfloat16*)out, so, scale, kSplits);
}
"""

# ---------------------------------------------------------------------------
# C++ wrapper
# ---------------------------------------------------------------------------
_CPP_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

extern "C" void launch_gdn_v1(
    const void*, const void*, const void*,
    const float*, const float*,
    const void*, const float*, const void*,
    void*, float*, float, int, int, cudaStream_t);

extern "C" void launch_gdn_v2_tma(
    const void*, const void*, const void*,
    const float*, const float*,
    const void*, const float*, const void*,
    void*, float*, float, int, int, cudaStream_t);

static void _dispatch(
    bool use_tma,
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor si, torch::Tensor A_log,
    torch::Tensor a_in, torch::Tensor dt_bias, torch::Tensor b_in,
    torch::Tensor out, torch::Tensor so,
    float scale, int kSplits)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    int B = q.size(0);
    if (use_tma) {
        launch_gdn_v2_tma(
            q.data_ptr(), k.data_ptr(), v.data_ptr(),
            si.data_ptr<float>(), A_log.data_ptr<float>(),
            a_in.data_ptr(), dt_bias.data_ptr<float>(), b_in.data_ptr(),
            out.data_ptr(), so.data_ptr<float>(),
            scale, kSplits, B, stream);
    } else {
        launch_gdn_v1(
            q.data_ptr(), k.data_ptr(), v.data_ptr(),
            si.data_ptr<float>(), A_log.data_ptr<float>(),
            a_in.data_ptr(), dt_bias.data_ptr<float>(), b_in.data_ptr(),
            out.data_ptr(), so.data_ptr<float>(),
            scale, kSplits, B, stream);
    }
}

void gdn_decode_v1(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor si, torch::Tensor A_log,
    torch::Tensor a_in, torch::Tensor dt_bias, torch::Tensor b_in,
    torch::Tensor out, torch::Tensor so, float scale, int kSplits)
{ _dispatch(false, q,k,v,si,A_log,a_in,dt_bias,b_in,out,so,scale,kSplits); }

void gdn_decode_v2_tma(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor si, torch::Tensor A_log,
    torch::Tensor a_in, torch::Tensor dt_bias, torch::Tensor b_in,
    torch::Tensor out, torch::Tensor so, float scale, int kSplits)
{ _dispatch(true, q,k,v,si,A_log,a_in,dt_bias,b_in,out,so,scale,kSplits); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gdn_decode_v1",     &gdn_decode_v1,     "col-parallel v1 (plain ldg)");
    m.def("gdn_decode_v2_tma", &gdn_decode_v2_tma, "col-parallel v2 (cp.async double-buf)");
}
"""

# ---------------------------------------------------------------------------
# Compile and cache
# ---------------------------------------------------------------------------
_LOCK = threading.Lock()
_EXT  = None

# MSINFER_TMA=1 → use cp.async double-buffering kernel
_USE_TMA = os.environ.get("MSINFER_TMA", "0") == "1"


def _get_ext():
    global _EXT
    if _EXT is None:
        with _LOCK:
            if _EXT is None:
                from torch.utils.cpp_extension import load_inline
                _EXT = load_inline(
                    name="gdn_cuda_col_v3",
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


def _ksplits(B: int) -> int:
    """Adaptive kSplits: more blocks at small batch for better SM coverage."""
    if B <= 4:
        return 8   # B=1: 64 blocks (vs 32 for kSplits=4)
    return 4       # B>=8: 32+ blocks, coalescing benefit dominates


def run(q, k, v, state, A_log, a, dt_bias, b, scale, output, new_state):
    """DPS entrypoint."""
    ext = _get_ext()
    ks  = _ksplits(q.size(0))
    fn  = ext.gdn_decode_v2_tma if _USE_TMA else ext.gdn_decode_v1
    fn(q, k, v, state, A_log, a, dt_bias, b, output, new_state, DEFAULT_SCALE, ks)
