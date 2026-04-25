"""CUtensorMap builder for GDN decode v4 bulk-TMA kernel.

Wraps `cuTensorMapEncodeTiled` from libcuda.so (CUDA driver API) via ctypes,
and caches the resulting 128-byte descriptors per-shape so we don't rebuild
them on every decode call.

Flat 2D view of state: shape (B*HV*D, D) with tile (kRows, D=128). This
choice keeps the per-block origin computation trivial:
    row_base = batch * HV * D + v_head * D + split * kRows
    col_base = 0
"""
from __future__ import annotations

import ctypes
import threading
from typing import Tuple


# ---------------------------------------------------------------------------
# libcuda.so binding (resolved lazily — avoids import failure on CPU-only hosts)
# ---------------------------------------------------------------------------
_LIBCUDA = None
_FN_ENCODE = None
_LIBCUDA_LOCK = threading.Lock()


def _libcuda():
    global _LIBCUDA, _FN_ENCODE
    if _FN_ENCODE is not None:
        return _FN_ENCODE
    with _LIBCUDA_LOCK:
        if _FN_ENCODE is None:
            _LIBCUDA = ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
            fn = _LIBCUDA.cuTensorMapEncodeTiled
            fn.restype = ctypes.c_int
            fn.argtypes = [
                ctypes.c_void_p,                       # CUtensorMap* (out, 128 B buffer)
                ctypes.c_uint,                         # CUtensorMapDataType
                ctypes.c_uint,                         # tensorRank
                ctypes.c_void_p,                       # globalAddress
                ctypes.POINTER(ctypes.c_uint64),       # globalDim[rank]
                ctypes.POINTER(ctypes.c_uint64),       # globalStrides[rank-1] (bytes)
                ctypes.POINTER(ctypes.c_uint32),       # boxDim[rank]
                ctypes.POINTER(ctypes.c_uint32),       # elementStrides[rank]
                ctypes.c_uint,                         # interleave
                ctypes.c_uint,                         # swizzle
                ctypes.c_uint,                         # l2Promotion
                ctypes.c_uint,                         # oobFill
            ]
            _FN_ENCODE = fn
    return _FN_ENCODE


# Enum values from cuda.h (CUDA 13.x).
CU_TENSOR_MAP_DATA_TYPE_FLOAT32     = 2
CU_TENSOR_MAP_INTERLEAVE_NONE       = 0
CU_TENSOR_MAP_SWIZZLE_NONE          = 0
CU_TENSOR_MAP_L2_PROMOTION_NONE     = 0
CU_TENSOR_MAP_L2_PROMOTION_L2_128B  = 1
CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE   = 0


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------
def build_state_tensor_map(base_ptr: int, B: int, HV: int, kRows: int) -> bytes:
    """Build a 128-byte CUtensorMap for fp32 state viewed as (B*HV*D, D), tile (kRows, D).

    Args:
        base_ptr: device pointer to the first fp32 element (must be 16-byte aligned —
                  PyTorch tensors satisfy this).
        B, HV:    outer dims.
        kRows:    tile row count (32 for kSplits=4, 16 for kSplits=8).

    Returns:
        128 bytes suitable for passing to the kernel via __grid_constant__ CUtensorMap.
    """
    D = 128

    buf = (ctypes.c_byte * 128)()
    global_dim      = (ctypes.c_uint64 * 2)(D, B * HV * D)
    global_strides  = (ctypes.c_uint64 * 1)(D * 4)         # bytes — inner-dim stride is implicit
    box_dim         = (ctypes.c_uint32 * 2)(D, kRows)
    elem_strides    = (ctypes.c_uint32 * 2)(1, 1)

    rc = _libcuda()(
        ctypes.byref(buf),
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        2,
        ctypes.c_void_p(base_ptr),
        global_dim,
        global_strides,
        box_dim,
        elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    if rc != 0:
        raise RuntimeError(
            f"cuTensorMapEncodeTiled failed (CUresult={rc}): "
            f"base={base_ptr:#x} B={B} HV={HV} kRows={kRows}"
        )
    return bytes(buf)


# ---------------------------------------------------------------------------
# Per-shape cache
# ---------------------------------------------------------------------------
_CACHE_LOCK = threading.Lock()
_CACHE: dict = {}


def get_tensor_map_pair(
    state_in_ptr: int, state_out_ptr: int, B: int, HV: int, kRows: int
) -> Tuple[bytes, bytes]:
    """Return (desc_in, desc_out), cached per shape+exact-pointer.

    The TMA descriptor stores the exact base pointer, so the cache key must
    be the exact pointer — not a masked/aligned region. Across distinct
    workloads PyTorch may reuse the same 2 MB region at different offsets;
    a coarser key would return a stale descriptor pointing into the wrong
    memory and produce silent numerical corruption.

    Within the bench loop, repeated trials of the same workload reuse the
    same allocator slot, so the hit rate remains near 100% for steady-state
    benchmarking.
    """
    key = (B, HV, kRows, state_in_ptr, state_out_ptr)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        if hit is not None:
            return hit
        desc_in  = build_state_tensor_map(state_in_ptr,  B, HV, kRows)
        desc_out = build_state_tensor_map(state_out_ptr, B, HV, kRows)
        _CACHE[key] = (desc_in, desc_out)
        return desc_in, desc_out


def clear_cache() -> None:
    """Test helper — forces a rebuild on next call."""
    with _CACHE_LOCK:
        _CACHE.clear()
