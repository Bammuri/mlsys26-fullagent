"""GDN decode via CuTe DSL runtime compile.

Surface contract (matches the MSInfer reference build):
    language     = "python"
    entry_point  = "msinfer_entry.py::run"            # for gdn_decode
    dependencies = ["nvidia-cutlass-dsl", "cuda-python>=12.8"]

On first call per input shape we compile a CuTe kernel with
``cute.compile(..., options="--enable-tvm-ffi --gpu-arch=sm_100a ...")``.
The resulting TVM-FFI callable is cached and reused for subsequent calls
whose (dtype, shape, stride) signature matches.

The explicit ``--gpu-arch=sm_100a`` is the load-bearing bit: without the
``a`` suffix nvcc will not emit Blackwell-native tensor-core / ``tcgen05`` /
CTA-pair instructions, and scoring-environment performance diverges from
Modal bench numbers. Keep the rest of the options pinned here so the SASS
is deterministic across environments.
"""

from __future__ import annotations

import math
import os
import threading
from typing import Any, Callable, Dict, Tuple

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


# ----------------------------------------------------------------------------
# Constants — pinned to the contest definition.
# ----------------------------------------------------------------------------
HQ = 4
HK = 4
HV = 8
D = 128
DEFAULT_SCALE = 1.0 / math.sqrt(float(D))
SOFTPLUS_BETA = 1.0
SOFTPLUS_THRESHOLD = 20.0

# High-B variant (default): 4-way split, 32 threads per block.
#   grid = (B * HV * 4, 1, 1); block = (32, 1, 1)
kSplits_H       = 4
kRowsPerBlock_H = D // kSplits_H   # 32
kWarpThreads_H  = kRowsPerBlock_H  # 32

# Low-B variant (B ≤ LOW_B_MAX): 8-way split, 16 threads per block.
#   grid = (B * HV * 8, 1, 1); block = (16, 1, 1)
#   Doubles grid size at low batch → more SMs active, ~2× throughput.
kSplits_L       = 8
kRowsPerBlock_L = D // kSplits_L   # 16
kWarpThreads_L  = kRowsPerBlock_L  # 16

# Dispatch threshold: use low variant for batch_size ≤ this value.
_LOW_B_MAX = int(os.environ.get("MSINFER_LOW_B_MAX", "8"))

# Keep aliases for backward compatibility in _gdn_decode_jit.
kSplits       = kSplits_H
kRowsPerBlock = kRowsPerBlock_H
kWarpThreads  = kWarpThreads_H

# ----------------------------------------------------------------------------
# Compile options — sm_100a + -O3 + fast math are the ones that move SASS.
# MSINFER_DUMP_SASS=1 additionally emits PTX/cubin + ptxas -v stats so the
# runner log captures register/spill/barrier/smem counts for the unit under
# test. Default path is unchanged (no extra work on the compile critical path).
# ----------------------------------------------------------------------------
_DUMP_SASS     = os.environ.get("MSINFER_DUMP_SASS") == "1"
_SASS_DUMP_DIR = os.environ.get("MSINFER_SASS_DIR", "/tmp/cute-asm")
if _DUMP_SASS:
    os.makedirs(_SASS_DUMP_DIR, exist_ok=True)

_REGCAP = os.environ.get("MSINFER_REGCAP", "")  # e.g. "128", "96", "80", "64"; "" = off

_SASS_OPTS = (
    f" --keep-ptx --keep-cubin --dump-dir={_SASS_DUMP_DIR} --ptxas-options=-v"
    if _DUMP_SASS
    else ""
)

_COMPILE_OPTS_BASE = (
    "--enable-tvm-ffi "
    "--gpu-arch=sm_100a "
    "--opt-level=3"
    + _SASS_OPTS
)

# High-B variant: optional register cap via --maxrregcount (top-level, passed to ptxas).
_COMPILE_OPTS_HIGH = (
    _COMPILE_OPTS_BASE
    + (f" --maxrregcount={_REGCAP}" if _REGCAP else "")
)

# Low-B variant: no reg cap.
_COMPILE_OPTS = _COMPILE_OPTS_BASE

# Per-shape compiled-callable cache (thread-safe).
_LOCK = threading.Lock()
_CACHE: Dict[Tuple[Any, ...], Callable[..., None]] = {}


# ----------------------------------------------------------------------------
# Device side — gate helpers (numerically-stable softplus + sigmoid).
# ----------------------------------------------------------------------------
@cute.jit
def _softplus_stable(x: cutlass.Float32) -> cutlass.Float32:
    # Single-exit form (DSL preprocessor rejects early-return inside @cute.jit).
    result = cutlass.Float32(0.0)
    if x > cutlass.Float32(SOFTPLUS_THRESHOLD):
        result = x
    elif x < cutlass.Float32(-SOFTPLUS_THRESHOLD):
        result = cute.exp(x, fastmath=True)
    else:
        result = cute.log(cutlass.Float32(1.0) + cute.exp(x, fastmath=True), fastmath=True)
    return result


@cute.jit
def _sigmoid_stable(x: cutlass.Float32) -> cutlass.Float32:
    result = cutlass.Float32(0.0)
    if x >= cutlass.Float32(0.0):
        result = cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.exp(-x, fastmath=True))
    else:
        e = cute.exp(x, fastmath=True)
        result = e / (cutlass.Float32(1.0) + e)
    return result


# ----------------------------------------------------------------------------
# Decode kernel — v2: 4-way V-split, 1 warp per block (mirrors static CUDA v17).
#   Grid:  (B * HV * kSplits, 1, 1)
#   Block: (kWarpThreads=32, 1, 1)
#   Each lane owns one V-row `row = split*32 + tid` and holds the full 128-fp32
#   state as a register array `sr[D]`. qk is reduced warp-only via butterfly
#   shuffle — no cross-warp smem, no block barrier in the critical path.
# ----------------------------------------------------------------------------
@cute.kernel
def _gdn_decode_dev(
    q: cute.Tensor,          # (B, S=1, HQ, D) bf16
    k: cute.Tensor,          # (B, S=1, HK, D) bf16
    v: cute.Tensor,          # (B, S=1, HV, D) bf16
    state_in: cute.Tensor,   # (B, HV, D, D) f32, k-last
    A_log: cute.Tensor,      # (HV,) f32
    a_in: cute.Tensor,       # (B, S=1, HV) bf16
    dt_bias: cute.Tensor,    # (HV,) f32
    b_in: cute.Tensor,       # (B, S=1, HV) bf16
    out: cute.Tensor,        # (B, S=1, HV, D) bf16
    state_out: cute.Tensor,  # (B, HV, D, D) f32
    scale: cutlass.Constexpr[float],
):
    bid_x, _, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()

    split   = bid_x % kSplits
    v_head  = (bid_x // kSplits) % HV
    batch   = bid_x // (kSplits * HV)
    qk_head = v_head // (HV // HQ)
    row     = split * kRowsPerBlock + tid   # 0..127

    # Cooperative smem load — 32 lanes × 4 elements = 128.
    smem = cutlass.utils.SmemAllocator()
    sQ = smem.allocate_tensor(cutlass.Float32, cute.make_layout((D,)), 16)
    sK = smem.allocate_tensor(cutlass.Float32, cute.make_layout((D,)), 16)
    for j in cutlass.range_constexpr(D // kWarpThreads):  # 4
        idx = tid + j * kWarpThreads
        sQ[idx] = cutlass.Float32(q[batch, 0, qk_head, idx])
        sK[idx] = cutlass.Float32(k[batch, 0, qk_head, idx])

    # Gate scalars — every lane computes redundantly (MUFU is per-lane SIMD).
    a_val = cutlass.Float32(a_in[batch, 0, v_head]) + cutlass.Float32(dt_bias[v_head])
    sp = _softplus_stable(a_val)
    g = cute.exp(-cute.exp(cutlass.Float32(A_log[v_head]), fastmath=True) * sp, fastmath=True)
    beta = _sigmoid_stable(cutlass.Float32(b_in[batch, 0, v_head]))

    cute.arch.sync_warp()  # bar.warp.sync — 1-warp block doesn't need bar.sync 0.

    # qk = q · k (block scalar). Warp-only butterfly reduce — every lane ends
    # up with full qk. Each lane contributes 4 of the 128 elements.
    qk = cutlass.Float32(0.0)
    for j in cutlass.range_constexpr(D // kWarpThreads):  # 4
        idx = tid + j * kWarpThreads
        qk += sQ[idx] * sK[idx]
    for offset in [16, 8, 4, 2, 1]:
        qk += cute.arch.shuffle_sync_bfly(qk, offset=offset, mask=-1, mask_and_clamp=31)

    # Fused first pass — vectorized state load (ldg.128 × 32 tiles).
    #   sr is held as a 2D (32, 4) register tile — outer axis = the 32 vec-tiles
    #   along D, inner axis = 4 fp32 inside each tile. Matches v17's
    #   `float4 sr[kVecsPerRow=32]` shape and lets the DSL emit vector register
    #   moves + vectorized gmem/smem reads on the inner axis.
    sr = cute.make_rmem_tensor(cute.make_layout((D // 4, 4), stride=(4, 1)), cutlass.Float32)
    tmp = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), cutlass.Float32)
    # State load: EVICT_FIRST — each line read once, no reuse in this call.
    # State store: EVICT_LAST — this call's output is the next call's input;
    # keeping it hot in L2 avoids an HBM round-trip on the next decode step.
    ev_first = cute.nvgpu.CacheEvictionPriority.EVICT_FIRST
    ev_last  = cute.nvgpu.CacheEvictionPriority.EVICT_LAST
    ov = cutlass.Float32(0.0)
    qs = cutlass.Float32(0.0)
    for i in cutlass.range_constexpr(D // 4):  # 32 tiles × 4 elems
        state_tile = cute.local_tile(
            state_in, (1, 1, 1, 4), (batch, v_head, row, i),
        )
        cute.autovec_copy(state_tile, tmp, l1c_evict_priority=ev_first)
        for c in cutlass.range_constexpr(4):
            s = tmp[c] * g
            sr[i, c] = s
            ov += sK[i * 4 + c] * s
            qs += sQ[i * 4 + c] * s

    v_val = cutlass.Float32(v[batch, 0, v_head, row])
    delta = beta * (v_val - ov)
    out_acc = qs + delta * qk

    # Second pass — vectorized state store (stg.128 × 32 tiles).
    for i in cutlass.range_constexpr(D // 4):
        for c in cutlass.range_constexpr(4):
            tmp[c] = sr[i, c] + sK[i * 4 + c] * delta
        state_tile = cute.local_tile(
            state_out, (1, 1, 1, 4), (batch, v_head, row, i),
        )
        cute.autovec_copy(tmp, state_tile, l1c_evict_priority=ev_last)

    out[batch, 0, v_head, row] = cutlass.BFloat16(cutlass.Float32(scale) * out_acc)


@cute.kernel
def _gdn_decode_dev_low(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    state_in: cute.Tensor,
    A_log: cute.Tensor,
    a_in: cute.Tensor,
    dt_bias: cute.Tensor,
    b_in: cute.Tensor,
    out: cute.Tensor,
    state_out: cute.Tensor,
    scale: cutlass.Constexpr[float],
):
    """Low-B variant: kSplits=8, 16 threads/block → 2× grid at low batch."""
    bid_x, _, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()

    split   = bid_x % kSplits_L
    v_head  = (bid_x // kSplits_L) % HV
    batch   = bid_x // (kSplits_L * HV)
    qk_head = v_head // (HV // HQ)
    row     = split * kRowsPerBlock_L + tid   # 0..127

    smem = cutlass.utils.SmemAllocator()
    sQ = smem.allocate_tensor(cutlass.Float32, cute.make_layout((D,)), 16)
    sK = smem.allocate_tensor(cutlass.Float32, cute.make_layout((D,)), 16)
    for j in cutlass.range_constexpr(D // kWarpThreads_L):  # 8
        idx = tid + j * kWarpThreads_L
        sQ[idx] = cutlass.Float32(q[batch, 0, qk_head, idx])
        sK[idx] = cutlass.Float32(k[batch, 0, qk_head, idx])

    a_val = cutlass.Float32(a_in[batch, 0, v_head]) + cutlass.Float32(dt_bias[v_head])
    sp = _softplus_stable(a_val)
    g = cute.exp(-cute.exp(cutlass.Float32(A_log[v_head]), fastmath=True) * sp, fastmath=True)
    beta = _sigmoid_stable(cutlass.Float32(b_in[batch, 0, v_head]))

    cute.arch.sync_warp()

    # qk butterfly: only 4 levels (log2(16)=4) with mask_and_clamp=15.
    qk = cutlass.Float32(0.0)
    for j in cutlass.range_constexpr(D // kWarpThreads_L):  # 8
        idx = tid + j * kWarpThreads_L
        qk += sQ[idx] * sK[idx]
    for offset in [8, 4, 2, 1]:
        qk += cute.arch.shuffle_sync_bfly(qk, offset=offset, mask=-1, mask_and_clamp=15)

    sr = cute.make_rmem_tensor(cute.make_layout((D // 4, 4), stride=(4, 1)), cutlass.Float32)
    tmp = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), cutlass.Float32)
    ev_first = cute.nvgpu.CacheEvictionPriority.EVICT_FIRST
    ev_last  = cute.nvgpu.CacheEvictionPriority.EVICT_LAST
    ov = cutlass.Float32(0.0)
    qs = cutlass.Float32(0.0)
    for i in cutlass.range_constexpr(D // 4):
        state_tile = cute.local_tile(state_in, (1, 1, 1, 4), (batch, v_head, row, i))
        cute.autovec_copy(state_tile, tmp, l1c_evict_priority=ev_first)
        for c in cutlass.range_constexpr(4):
            s = tmp[c] * g
            sr[i, c] = s
            ov += sK[i * 4 + c] * s
            qs += sQ[i * 4 + c] * s

    v_val = cutlass.Float32(v[batch, 0, v_head, row])
    delta = beta * (v_val - ov)
    out_acc = qs + delta * qk

    for i in cutlass.range_constexpr(D // 4):
        for c in cutlass.range_constexpr(4):
            tmp[c] = sr[i, c] + sK[i * 4 + c] * delta
        state_tile = cute.local_tile(state_out, (1, 1, 1, 4), (batch, v_head, row, i))
        cute.autovec_copy(tmp, state_tile, l1c_evict_priority=ev_last)

    out[batch, 0, v_head, row] = cutlass.BFloat16(cutlass.Float32(scale) * out_acc)


@cute.jit
def _gdn_decode_jit_low(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    state_in: cute.Tensor,
    A_log: cute.Tensor,
    a_in: cute.Tensor,
    dt_bias: cute.Tensor,
    b_in: cute.Tensor,
    out: cute.Tensor,
    state_out: cute.Tensor,
):
    B = q.layout.shape[0]
    _gdn_decode_dev_low(
        q, k, v, state_in, A_log, a_in, dt_bias, b_in, out, state_out,
        DEFAULT_SCALE,
    ).launch(
        grid=(B * HV * kSplits_L, 1, 1),
        block=(kWarpThreads_L, 1, 1),
    )


@cute.jit
def _gdn_decode_jit(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    state_in: cute.Tensor,
    A_log: cute.Tensor,
    a_in: cute.Tensor,
    dt_bias: cute.Tensor,
    b_in: cute.Tensor,
    out: cute.Tensor,
    state_out: cute.Tensor,
):
    B = q.layout.shape[0]
    _gdn_decode_dev(
        q, k, v, state_in, A_log, a_in, dt_bias, b_in, out, state_out,
        DEFAULT_SCALE,
    ).launch(
        grid=(B * HV * kSplits, 1, 1),
        block=(kWarpThreads, 1, 1),
    )


# ----------------------------------------------------------------------------
# Python dispatch surface.
# ----------------------------------------------------------------------------
def _cache_key(tensors) -> Tuple[Any, ...]:
    return tuple(
        (str(t.dtype), tuple(t.shape), tuple(t.stride())) for t in tensors
    )


def _wrap_cute(tensors):
    return [
        from_dlpack(t, enable_tvm_ffi=True).mark_layout_dynamic() for t in tensors
    ]


def _get_compiled(name: str, jit_fn, tensors, opts: str = _COMPILE_OPTS):
    key = (name, opts) + _cache_key(tensors)
    with _LOCK:
        fn = _CACHE.get(key)
        if fn is None:
            cute_tensors = _wrap_cute(tensors)
            fn = cute.compile(jit_fn, *cute_tensors, options=opts)
            _CACHE[key] = fn
    return fn


def _dispatch(name, jit_fn, tensors, call_args, opts: str = _COMPILE_OPTS):
    """Shared dispatch — catches and surfaces exceptions that the
    flashinfer-bench evaluator would otherwise swallow into RUNTIME_ERROR."""
    import sys
    import traceback

    try:
        fn = _get_compiled(name, jit_fn, tensors, opts)
        fn(*call_args)
    except Exception as e:
        print(f"[msinfer_entry:{name}] exception: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        # Also write to stdout so Modal relays it back to the local driver.
        sys.stdout.flush()
        raise


def run(q, k, v, state, A_log, a, dt_bias, b, scale, output, new_state):
    """DPS entrypoint for gdn_decode_qk4_v8_d128_k_last."""
    tensors = [q, k, v, state, A_log, a, dt_bias, b, output, new_state]
    call_args = [q, k, v, state, A_log, a, dt_bias, b, output, new_state]
    B = q.shape[0]
    if B <= _LOW_B_MAX:
        _dispatch("decode_low", _gdn_decode_jit_low, tensors, call_args, _COMPILE_OPTS)
    else:
        _dispatch("decode_high", _gdn_decode_jit, tensors, call_args, _COMPILE_OPTS_HIGH)

