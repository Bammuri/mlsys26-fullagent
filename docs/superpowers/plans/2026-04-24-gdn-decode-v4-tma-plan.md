# GDN Decode v4 — Bulk-TMA Column-Parallel Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the v3 column-parallel GDN decode kernel with a warp-per-row kernel that uses Blackwell bulk TMA (`cp.async.bulk.tensor.2d`) for state I/O, targeting ≥ 2× latency reduction at B=32/64 on Modal B200.

**Architecture:** New kernel `gdn_decode_col_v4_tma` lives alongside existing v1 / v2_cpasync in `msinfer_cuda.py`'s `_CUDA_SRC` string. Thread layout: 4 warps × 4 rows/iter (each lane owns 4 cols); zero block-level syncs in the compute loop. State I/O: one `cp.async.bulk.tensor.2d` per block for load + one for store, both via a `CUtensorMap` built in Python via `ctypes → cuTensorMapEncodeTiled` and cached per-shape. `MSINFER_KERNEL` env var (`v1 | v2_cpasync | v4_tma`) selects the kernel.

**Tech Stack:** CUDA C + inline PTX (`cp.async.bulk.tensor.2d`, mbarrier), Torch `load_inline` cpp_extension, Python `ctypes` for CUDA driver API, Modal CLI for remote B200 bench via `scripts/run_modal.py`, `flashinfer-bench` for correctness + latency measurement.

**Spec:** [`docs/superpowers/specs/2026-04-24-gdn-decode-v4-tma-design.md`](../specs/2026-04-24-gdn-decode-v4-tma-design.md)

---

## File Structure

- **Modify:** `gdn_decode/solution/python/msinfer_cuda.py`
  - `_CUDA_SRC` — add v4 kernel + launcher alongside v1, v2_tma
  - `_CPP_SRC` — add v4 binding + `<cuda.h>` include for `CUtensorMap`
  - Python dispatcher — add `MSINFER_KERNEL` env var, retire `MSINFER_TMA`
  - `run()` — build tensor maps per-call via cached builder, dispatch by env var

- **Create:** `gdn_decode/solution/python/tma_desc.py`
  - `ctypes` wrapper for `cuTensorMapEncodeTiled` from `libcuda.so`
  - Per-shape cache keyed on `(B, HV, kRows, in_ptr_region, out_ptr_region)`
  - Public API: `get_tensor_map_pair(state_in, state_out, B, HV, kRows) -> (bytes, bytes)` returning two 128-byte blobs

- **No other files change.** `config.toml` entry point stays `msinfer_cuda.py::run`. The extra `tma_desc.py` is auto-picked by `scripts/run_modal._minimal_pack` (iterates all `.py` files in `solution/python/`).

---

## Bench / test commands (used in every task)

```bash
# Dev iteration (기본): warmup=3 iter=20 trial=3
python3 -m modal run scripts/run_modal.py --kernel-dir gdn_decode --env "MSINFER_KERNEL=<variant>"

# Submission (본판): warmup=3 iter=50 trial=3 — used only for Gate A in Task 7
# To switch, edit scripts/run_modal.py line ~80 BenchmarkConfig(iterations=50)
python3 -m modal run scripts/run_modal.py --kernel-dir gdn_decode --env "MSINFER_KERNEL=<variant>"

# SASS dump (for Task 7 ptxas verification):
python3 -m modal run scripts/run_modal.py --kernel-dir gdn_decode --env "MSINFER_KERNEL=v4_tma" --dump-sass
```

**Note on bench iterations:** `scripts/run_modal.py` currently hardcodes `BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)` (line 80). For dev cycles, edit to `iterations=20, num_trials=3` per project benchmark policy. Revert to `iterations=50, num_trials=3` for the Task 7 gate run.

**v1 baseline** (captured 2026-04-24, 100 iters × 5 trials; see `out/opt-baseline-v3.log`):

| B | latency (ms) |
|---|--------------|
| 1 | 0.017 |
| 2 | 0.017 |
| 4 | 0.022 |
| 8 | 0.023 |
| 16 | 0.025 |
| 32 | 0.031 |
| 64 | 0.045 |

---

## Task 1: Capture v1 baseline under 기본 (20 iter × 3 trial)

**Why:** The stored baseline is 100 iters × 5 trials. For dev cycles we compare under 기본 (20 iter × 3 trial). Freeze a matching-config baseline log so per-task bench results are apples-to-apples.

**Files:**
- Modify: `scripts/run_modal.py:80` — bench config iterations
- Test: bench output saved to `out/v4-baseline-v1-geoban.log`

- [ ] **Step 1: Edit bench config to 기본 (20, 3)**

Change line 80 of `scripts/run_modal.py`:

```python
config = BenchmarkConfig(warmup_runs=3, iterations=20, num_trials=3)
```

- [ ] **Step 2: Run v1 baseline, save log**

```bash
python3 -m modal run scripts/run_modal.py --kernel-dir gdn_decode 2>&1 | tee out/v4-baseline-v1-geoban.log
```

Expected output: 54 workloads, all `PASSED`, latency figures roughly matching the 100-iter baseline within ±10% noise.

- [ ] **Step 3: Sanity-check the log**

```bash
grep -c PASSED out/v4-baseline-v1-geoban.log
```

Expected: `54`.

```bash
grep "0\." out/v4-baseline-v1-geoban.log | awk '{print $5}' | tail -8
```

Expected: latencies for B=64 bucket in the 0.04-0.05 ms range.

- [ ] **Step 4: Commit the bench config change + baseline log**

```bash
git add scripts/run_modal.py out/v4-baseline-v1-geoban.log
git commit -m "bench(gdn_decode): switch to 기본 (20 iter × 3 trial) + baseline log

Freezes v1 baseline under the policy-default iteration count so
v4 dev cycles can compare apples-to-apples. Submission runs switch
back to 본판 (50 iter × 3 trial) in Task 7."
```

---

## Task 2: Retire `MSINFER_TMA`, introduce `MSINFER_KERNEL` dispatcher

**Why:** v4 adds a third kernel variant — a boolean env var is no longer enough. Also: isolating the dispatcher refactor from the v4 kernel itself keeps regressions easy to bisect.

**Files:**
- Modify: `gdn_decode/solution/python/msinfer_cuda.py` — lines ~325-365 (dispatcher + `run()`)

- [ ] **Step 1: Write a "failing" smoke test — invoke the kernel with `MSINFER_KERNEL=bogus` and expect the dispatcher to raise**

Run the bench once with a sentinel invalid value, expect a clear error (not a crash):

```bash
python3 -m modal run scripts/run_modal.py --kernel-dir gdn_decode --env "MSINFER_KERNEL=bogus" 2>&1 | grep -E "ValueError|RuntimeError|Unknown kernel"
```

Expected: **no match** (before the change — current code has no such validation). This fails the TDD step.

- [ ] **Step 2: Implement `MSINFER_KERNEL` dispatcher**

Replace lines 327-365 of `msinfer_cuda.py` (everything from `_USE_TMA = ...` through the end of `run()`) with:

```python
# ---------------------------------------------------------------------------
# Kernel selection
# ---------------------------------------------------------------------------
#
# MSINFER_KERNEL env var selects the kernel variant:
#   "v1"         - plain __ldg, per-row scalar state I/O (current default)
#   "v2_cpasync" - cp.async double-buffered state load (kept during v4 ramp)
#   "v4_tma"     - warp-per-row + bulk cp.async.bulk.tensor.2d (new; gated)
#
# Defaults to "v1" until Gate A passes (Task 7). Invalid values raise.
_VALID_KERNELS = ("v1", "v2_cpasync", "v4_tma")


def _select_kernel() -> str:
    name = os.environ.get("MSINFER_KERNEL", "v1")
    if name not in _VALID_KERNELS:
        raise ValueError(
            f"MSINFER_KERNEL={name!r} invalid; must be one of {_VALID_KERNELS}"
        )
    return name


def _ksplits(B: int) -> int:
    """Adaptive kSplits: more blocks at small batch for better SM coverage."""
    if B <= 4:
        return 8   # B=1: 64 blocks (vs 32 for kSplits=4)
    return 4       # B>=8: 32+ blocks, coalescing benefit dominates


def run(q, k, v, state, A_log, a, dt_bias, b, scale, output, new_state):
    """DPS entrypoint."""
    ext = _get_ext()
    ks  = _ksplits(q.size(0))
    kernel = _select_kernel()
    if kernel == "v1":
        ext.gdn_decode_v1(q, k, v, state, A_log, a, dt_bias, b, output, new_state, DEFAULT_SCALE, ks)
    elif kernel == "v2_cpasync":
        ext.gdn_decode_v2_tma(q, k, v, state, A_log, a, dt_bias, b, output, new_state, DEFAULT_SCALE, ks)
    elif kernel == "v4_tma":
        raise NotImplementedError("v4_tma landed in Task 6 of the implementation plan")
    else:  # pragma: no cover — already validated in _select_kernel
        raise AssertionError(f"unreachable: {kernel!r}")
```

The `NotImplementedError` for `v4_tma` is load-bearing: it lets the dispatcher branch land independently of the kernel.

- [ ] **Step 3: Re-run the smoke test — expect the error now**

```bash
python3 -m modal run scripts/run_modal.py --kernel-dir gdn_decode --env "MSINFER_KERNEL=bogus" 2>&1 | grep "MSINFER_KERNEL='bogus' invalid"
```

Expected: line match. (The error surfaces through `flashinfer-bench`'s log capture.)

- [ ] **Step 4: Run full bench with `MSINFER_KERNEL=v1` — must match baseline exactly**

```bash
python3 -m modal run scripts/run_modal.py --kernel-dir gdn_decode --env "MSINFER_KERNEL=v1" 2>&1 | tee out/v4-task2-v1-smoke.log
```

Expected: 54 `PASSED`, latencies within ±5% of `out/v4-baseline-v1-geoban.log`.

- [ ] **Step 5: Run full bench with `MSINFER_KERNEL=v2_cpasync` — must PASS on all 54**

```bash
python3 -m modal run scripts/run_modal.py --kernel-dir gdn_decode --env "MSINFER_KERNEL=v2_cpasync" 2>&1 | tee out/v4-task2-v2-smoke.log
```

Expected: 54 `PASSED`. Latencies should match the earlier v2 run (B=1 ~0.013, B=64 ~0.042).

- [ ] **Step 6: Commit**

```bash
git add gdn_decode/solution/python/msinfer_cuda.py out/v4-task2-*.log
git commit -m "refactor(gdn_decode): MSINFER_KERNEL env dispatcher

Retires the MSINFER_TMA boolean in favor of MSINFER_KERNEL, which
will carry v1 / v2_cpasync / v4_tma once the v4 kernel lands. Default
stays v1 — no behavioral change in this commit."
```

---

## Task 3: Create `tma_desc.py` — ctypes wrapper for `cuTensorMapEncodeTiled`

**Why:** The v4 kernel needs two `CUtensorMap` descriptors (state_in, state_out) passed via `__grid_constant__` kernel args. Building them requires the CUDA driver API, not runtime API — ctypes is the cleanest path.

**Files:**
- Create: `gdn_decode/solution/python/tma_desc.py`

- [ ] **Step 1: Write the minimal module with constants + builder**

Create `gdn_decode/solution/python/tma_desc.py`:

```python
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


def _libcuda():
    global _LIBCUDA, _FN_ENCODE
    if _LIBCUDA is None:
        _LIBCUDA = ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
        _FN_ENCODE = _LIBCUDA.cuTensorMapEncodeTiled
        _FN_ENCODE.restype = ctypes.c_int
        _FN_ENCODE.argtypes = [
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

# Two 2 MB-aligned-region keys — if PyTorch re-allocates to a different region
# we invalidate. Within the bench loop the allocator reuses the same buffer, so
# the hit rate is ≈ 100%.
_ALIGN = 2 * 1024 * 1024


def get_tensor_map_pair(
    state_in_ptr: int, state_out_ptr: int, B: int, HV: int, kRows: int
) -> Tuple[bytes, bytes]:
    """Return (desc_in, desc_out), cached per shape+region."""
    in_region  = state_in_ptr  & ~(_ALIGN - 1)
    out_region = state_out_ptr & ~(_ALIGN - 1)
    key = (B, HV, kRows, in_region, out_region)
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
```

- [ ] **Step 2: Write a Modal-hosted sanity check** (since libcuda is not on macOS)

Add a small function to `scripts/run_modal.py` below `run_benchmark` — this is a dev-only probe, keep it short:

```python
@app.function(image=image, gpu="B200:1", timeout=600)
def probe_tma_desc() -> dict:
    """Smoke test for tma_desc.build_state_tensor_map — runs on B200."""
    import sys
    import torch
    sys.path.insert(0, "/workspace/gdn_decode/solution/python")
    from tma_desc import build_state_tensor_map, get_tensor_map_pair, clear_cache

    st_in  = torch.empty((4, 8, 128, 128), dtype=torch.float32, device="cuda")
    st_out = torch.empty_like(st_in)
    desc_in, desc_out = get_tensor_map_pair(st_in.data_ptr(), st_out.data_ptr(), B=4, HV=8, kRows=32)
    return {
        "len_in":  len(desc_in),
        "len_out": len(desc_out),
        "nonzero_in":  sum(desc_in) > 0,
        "nonzero_out": sum(desc_out) > 0,
        "same_contents": desc_in == desc_out,  # different base pointers → must be False
    }
```

And a local entrypoint:

```python
@app.local_entrypoint()
def probe_tma():
    """Run just the TMA descriptor probe."""
    from scripts.run_modal import probe_tma_desc
    import json
    result = probe_tma_desc.remote()
    print(json.dumps(result, indent=2))
```

Note: this won't work as-is because Modal can't import from the local repo by the plain "/workspace" path. Use an alternative: pack the two Python files (`msinfer_cuda.py`, `tma_desc.py`) into the Modal image at build time, OR invoke directly via `modal.Mount.from_local_dir`.

Simplest alternative — skip the standalone probe and validate via the full kernel-integration path in Task 4 (where `tma_desc` gets invoked by `run()` anyway).

**Decision:** **Skip the probe**. Ship `tma_desc.py` unused by anything in this task, and validate its output in Task 4 when the v4 kernel first calls it.

- [ ] **Step 3: Confirm `tma_desc.py` imports cleanly on the packing side**

```bash
python3 -c "import ast; ast.parse(open('gdn_decode/solution/python/tma_desc.py').read()); print('OK')"
```

Expected: `OK` (syntax only — ctypes dlopen won't fire on macOS).

- [ ] **Step 4: Commit**

```bash
git add gdn_decode/solution/python/tma_desc.py
git commit -m "feat(gdn_decode): tma_desc.py — CUtensorMap builder via ctypes

Wraps cuTensorMapEncodeTiled with a per-shape cache. 2D flat view of
state (B*HV*D, D) with tile (kRows, 128). Used by the upcoming v4
kernel; not wired into any dispatch path yet."
```

---

## Task 4: v4 kernel skeleton — warp-per-row layout, **no TMA** (uses __ldg)

**Why:** Isolate the thread-layout change (warp-per-row, 4 cols/lane, warp-internal reductions) from the TMA change. This task ships the new layout with the *same* transport as v1 (`__ldg`) — so any correctness issue is attributable to the layout, not to PTX intrinsics.

**Files:**
- Modify: `gdn_decode/solution/python/msinfer_cuda.py`
  - `_CUDA_SRC` — append new kernel `gdn_decode_col_v4_noTMA` + launcher
  - `_CPP_SRC` — add binding + decl
  - Python `run()` — route `MSINFER_KERNEL=v4_noTMA` to the new binding

This variant is **temporary** (removed in Task 6). It exists to isolate layout bugs.

- [ ] **Step 1: Add the CUDA kernel to `_CUDA_SRC`**

Append to the end of `_CUDA_SRC` (after the v2 kernel, before the launcher section):

```cpp
// ── v4 skeleton: warp-per-row, 4 cols/lane, __ldg transport (no TMA) ─────
// Intermediate step for Task 4 of the v4 plan. Removed in Task 6.
template<int kRows>
__global__ void __launch_bounds__(128, 8)
gdn_decode_col_v4_noTMA(
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
    const int col0    = lane_id * 4;                // each lane owns cols col0..col0+3

    __shared__ float sQ[128], sK[128], sV[128];
    __shared__ float s_qk;

    // Init smem Q/K/V (one lane per col).
    sQ[tid] = __bfloat162float(q[batch*HQ_*D_ + qk_head*D_ + tid]);
    sK[tid] = __bfloat162float(k[batch*HK_*D_ + qk_head*D_ + tid]);
    sV[tid] = __bfloat162float(v[batch*HV_*D_ + v_head *D_ + tid]);

    const float a_val = __bfloat162float(a_in[batch*HV_+v_head]) + dt_bias[v_head];
    const float g     = __expf(-__expf(A_log[v_head]) * softplus_stable(a_val));
    const float beta  = sigmoid_stable(__bfloat162float(b_in[batch*HV_+v_head]));

    __syncthreads();

    // qk = dot(sQ, sK) — warp0 only, internal shfl reduce.
    if (warp_id == 0) {
        float p = sQ[col0+0]*sK[col0+0] + sQ[col0+1]*sK[col0+1]
                + sQ[col0+2]*sK[col0+2] + sQ[col0+3]*sK[col0+3];
        #pragma unroll
        for (int off=16; off>=1; off>>=1) p += __shfl_xor_sync(0xffffffff, p, off);
        if (lane_id == 0) s_qk = p;
    }
    __syncthreads();
    const float qk = s_qk;

    // Preload this lane's 4 cols of Q, K into registers.
    float k_reg[4], q_reg[4];
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        k_reg[c] = sK[col0 + c];
        q_reg[c] = sQ[col0 + c];
    }

    const float* si = state_in  + (batch*HV_+v_head)*D_*D_;
    float*       so = state_out + (batch*HV_+v_head)*D_*D_;
    __nv_bfloat16* op = out + (batch*HV_+v_head)*D_;

    constexpr int iters = kRows / 4;
    #pragma unroll 1
    for (int iter = 0; iter < iters; ++iter) {
        const int local_row  = iter*4 + warp_id;
        const int global_row = row0 + local_row;

        // Scalar __ldg load of 4 cols.
        float st[4];
        #pragma unroll
        for (int c = 0; c < 4; ++c) st[c] = __ldg(&si[global_row*D_ + col0 + c]) * g;

        // Per-warp inner products.
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
        delta = __shfl_sync(0xffffffff, delta, 0);   // broadcast within warp

        // Scalar store of the updated row.
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            so[global_row*D_ + col0 + c] = st[c] + k_reg[c] * delta;
        }
    }
}
```

- [ ] **Step 2: Add the launcher to `_CUDA_SRC`**

Append after the existing `launch_gdn_v2_tma` launcher:

```cpp
extern "C" void launch_gdn_v4_noTMA(
    const void* q, const void* k, const void* v,
    const float* si, const float* A_log,
    const void* a_in, const float* dt_bias, const void* b_in,
    void* out, float* so,
    float scale, int kSplits, int B, cudaStream_t stream)
{
    dim3 grid(B * 8 * kSplits);
    dim3 block(128);
    if (kSplits == 4) {
        gdn_decode_col_v4_noTMA<32><<<grid, block, 0, stream>>>(
            (const __nv_bfloat16*)q, (const __nv_bfloat16*)k, (const __nv_bfloat16*)v,
            si, A_log, (const __nv_bfloat16*)a_in, dt_bias, (const __nv_bfloat16*)b_in,
            (__nv_bfloat16*)out, so, scale);
    } else if (kSplits == 8) {
        gdn_decode_col_v4_noTMA<16><<<grid, block, 0, stream>>>(
            (const __nv_bfloat16*)q, (const __nv_bfloat16*)k, (const __nv_bfloat16*)v,
            si, A_log, (const __nv_bfloat16*)a_in, dt_bias, (const __nv_bfloat16*)b_in,
            (__nv_bfloat16*)out, so, scale);
    } else {
        // Assertion via trap — only 4 and 8 are supported; launcher wouldn't be called otherwise.
        asm("trap;");
    }
}
```

- [ ] **Step 3: Add the C++ binding to `_CPP_SRC`**

Add the `extern "C"` decl and PYBIND11 export. Insert the decl after the existing launcher declarations, and add to the module init:

```cpp
extern "C" void launch_gdn_v4_noTMA(
    const void*, const void*, const void*,
    const float*, const float*,
    const void*, const float*, const void*,
    void*, float*, float, int, int, cudaStream_t);
```

Add the binding function above `PYBIND11_MODULE`:

```cpp
void gdn_decode_v4_noTMA(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor si, torch::Tensor A_log,
    torch::Tensor a_in, torch::Tensor dt_bias, torch::Tensor b_in,
    torch::Tensor out, torch::Tensor so, float scale, int kSplits)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    int B = q.size(0);
    launch_gdn_v4_noTMA(
        q.data_ptr(), k.data_ptr(), v.data_ptr(),
        si.data_ptr<float>(), A_log.data_ptr<float>(),
        a_in.data_ptr(), dt_bias.data_ptr<float>(), b_in.data_ptr(),
        out.data_ptr(), so.data_ptr<float>(),
        scale, kSplits, B, stream);
}
```

And add to `PYBIND11_MODULE`:

```cpp
m.def("gdn_decode_v4_noTMA", &gdn_decode_v4_noTMA, "v4 warp-per-row (no TMA)");
```

- [ ] **Step 4: Wire `MSINFER_KERNEL=v4_noTMA` into the dispatcher**

In `msinfer_cuda.py`, update `_VALID_KERNELS` and `run()`:

```python
_VALID_KERNELS = ("v1", "v2_cpasync", "v4_noTMA", "v4_tma")
```

Add the dispatch branch inside `run()`:

```python
    elif kernel == "v4_noTMA":
        ext.gdn_decode_v4_noTMA(q, k, v, state, A_log, a, dt_bias, b, output, new_state, DEFAULT_SCALE, ks)
```

- [ ] **Step 5: Change the Python extension name** so Modal doesn't hit a stale cache

In `msinfer_cuda.py`, change the `load_inline(name=...)` from `"gdn_cuda_col_v3"` to `"gdn_cuda_col_v4"`:

```python
_EXT = load_inline(
    name="gdn_cuda_col_v4",
    cpp_sources=_CPP_SRC,
    cuda_sources=_CUDA_SRC,
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "-arch=sm_100a",
    ],
    verbose=False,
)
```

- [ ] **Step 6: Run full bench with `MSINFER_KERNEL=v4_noTMA` — correctness gate**

```bash
python3 -m modal run scripts/run_modal.py --kernel-dir gdn_decode --env "MSINFER_KERNEL=v4_noTMA" 2>&1 | tee out/v4-task4-noTMA.log
```

**Expected:** 54 PASSED. Per-workload `abs_err` within 2× of v1 baseline, `rel_err` within 2× of v1 baseline.

**Diagnostic if a workload FAILS:** the layout change is the only variable — most likely suspect is the `sV[global_row]` broadcast or the `__shfl_sync(0, delta)` pattern. Add a `printf` at `(batch=0, v_head=0, lane=0)` for 1 row and compare to a tiny Python reference, OR run v1 on the same workload and diff the output tensor.

**Diagnostic if latency is much worse than v1:** likely register pressure. Check ptxas with `--dump-sass`. Expected ≤ 48 registers per thread at `__launch_bounds__(128, 8)`.

- [ ] **Step 7: Commit**

```bash
git add gdn_decode/solution/python/msinfer_cuda.py out/v4-task4-noTMA.log
git commit -m "feat(gdn_decode): v4 skeleton — warp-per-row layout, __ldg transport

Intermediate kernel without TMA. Uses the new layout (4 warps x 4 rows,
each lane owns 4 cols, warp-internal reductions) with __ldg state I/O
so layout bugs can be caught without PTX TMA complications.

Gated behind MSINFER_KERNEL=v4_noTMA. Removed in Task 6 once v4_tma
supersedes it."
```

---

## Task 5: Add TMA **load** (state_in → smem), keep scalar store

**Why:** Break TMA rollout in half — load first, store second. If the TMA load wire-up has a bug (descriptor format, mbarrier timing, PTX constraints), it surfaces with a clean `__ldg`-store path unchanged.

**Files:**
- Modify: `gdn_decode/solution/python/msinfer_cuda.py`
  - `_CUDA_SRC` — add new kernel `gdn_decode_col_v4_tma_loadOnly` (copy of v4_noTMA with state load replaced by TMA)
  - `_CPP_SRC` — add binding that accepts a CUtensorMap blob for state_in
  - Python `run()` — route `MSINFER_KERNEL=v4_tma_loadOnly` with built desc_in

Like v4_noTMA, this variant is **temporary** (removed in Task 6).

- [ ] **Step 1: Include `<cuda.h>` in `_CUDA_SRC`** (required for `CUtensorMap`)

Edit the top of `_CUDA_SRC`:

```cpp
_CUDA_SRC = r"""
#include <cuda.h>             // CUtensorMap
#include <cuda_runtime.h>
#include <cuda_bf16.h>
```

Also in `_CPP_SRC` — add `<cuda.h>` under the `torch/extension.h` include.

- [ ] **Step 2: Add the TMA-load kernel to `_CUDA_SRC`**

Append after `gdn_decode_col_v4_noTMA`:

```cpp
// ── v4 TMA-load-only: bulk TMA for state_in, scalar store for state_out ──
// Task 5 intermediate. Removed in Task 6.
template<int kRows>
__global__ void __launch_bounds__(128, 8)
gdn_decode_col_v4_tma_loadOnly(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const __grid_constant__ CUtensorMap tensor_map_in,
    const float*         __restrict__ A_log,
    const __nv_bfloat16* __restrict__ a_in,
    const float*         __restrict__ dt_bias,
    const __nv_bfloat16* __restrict__ b_in,
    __nv_bfloat16*       __restrict__ out,
    float*               __restrict__ state_out,
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
    __shared__ alignas(16) float state_tile[kRows][128];
    __shared__ alignas(8) uint64_t mbar_load;

    sQ[tid] = __bfloat162float(q[batch*HQ_*D_ + qk_head*D_ + tid]);
    sK[tid] = __bfloat162float(k[batch*HK_*D_ + qk_head*D_ + tid]);
    sV[tid] = __bfloat162float(v[batch*HV_*D_ + v_head *D_ + tid]);

    const float a_val = __bfloat162float(a_in[batch*HV_+v_head]) + dt_bias[v_head];
    const float g     = __expf(-__expf(A_log[v_head]) * softplus_stable(a_val));
    const float beta  = sigmoid_stable(__bfloat162float(b_in[batch*HV_+v_head]));

    // Init mbarrier + issue TMA load (thread 0 only).
    if (tid == 0) {
        uint32_t mbar_ptr  = __cvta_generic_to_shared(&mbar_load);
        uint32_t tile_ptr  = __cvta_generic_to_shared(&state_tile[0][0]);
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

    // qk = dot(sQ, sK) — warp0 only — runs while TMA load is in flight.
    if (warp_id == 0) {
        float p = sQ[col0+0]*sK[col0+0] + sQ[col0+1]*sK[col0+1]
                + sQ[col0+2]*sK[col0+2] + sQ[col0+3]*sK[col0+3];
        #pragma unroll
        for (int off=16; off>=1; off>>=1) p += __shfl_xor_sync(0xffffffff, p, off);
        if (lane_id == 0) s_qk = p;
    }

    // Wait for TMA load.
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

    float*         so = state_out + (batch*HV_+v_head)*D_*D_;
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
            so[global_row*D_ + col0 + c] = st[c] + k_reg[c] * delta;
        }
    }
}
```

- [ ] **Step 3: Add the launcher for TMA-load-only**

Append to `_CUDA_SRC`:

```cpp
extern "C" void launch_gdn_v4_tma_loadOnly(
    const void* q, const void* k, const void* v,
    const CUtensorMap* desc_in,
    const float* A_log,
    const void* a_in, const float* dt_bias, const void* b_in,
    void* out, float* so,
    float scale, int kSplits, int B, cudaStream_t stream)
{
    dim3 grid(B * 8 * kSplits);
    dim3 block(128);
    if (kSplits == 4) {
        gdn_decode_col_v4_tma_loadOnly<32><<<grid, block, 0, stream>>>(
            (const __nv_bfloat16*)q, (const __nv_bfloat16*)k, (const __nv_bfloat16*)v,
            *desc_in,
            A_log, (const __nv_bfloat16*)a_in, dt_bias, (const __nv_bfloat16*)b_in,
            (__nv_bfloat16*)out, so, scale);
    } else if (kSplits == 8) {
        gdn_decode_col_v4_tma_loadOnly<16><<<grid, block, 0, stream>>>(
            (const __nv_bfloat16*)q, (const __nv_bfloat16*)k, (const __nv_bfloat16*)v,
            *desc_in,
            A_log, (const __nv_bfloat16*)a_in, dt_bias, (const __nv_bfloat16*)b_in,
            (__nv_bfloat16*)out, so, scale);
    } else {
        asm("trap;");
    }
}
```

- [ ] **Step 4: Add the C++ binding**

In `_CPP_SRC`, add the decl:

```cpp
extern "C" void launch_gdn_v4_tma_loadOnly(
    const void*, const void*, const void*,
    const CUtensorMap*,
    const float*,
    const void*, const float*, const void*,
    void*, float*, float, int, int, cudaStream_t);
```

And the binding function (accepts `py::bytes` for the 128-B descriptor):

```cpp
void gdn_decode_v4_tma_loadOnly(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor A_log,
    torch::Tensor a_in, torch::Tensor dt_bias, torch::Tensor b_in,
    torch::Tensor out, torch::Tensor so,
    py::bytes desc_in_bytes,
    float scale, int kSplits)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    int B = q.size(0);
    std::string din = desc_in_bytes;
    TORCH_CHECK(din.size() == 128, "desc_in must be 128 bytes");
    launch_gdn_v4_tma_loadOnly(
        q.data_ptr(), k.data_ptr(), v.data_ptr(),
        reinterpret_cast<const CUtensorMap*>(din.data()),
        A_log.data_ptr<float>(),
        a_in.data_ptr(), dt_bias.data_ptr<float>(), b_in.data_ptr(),
        out.data_ptr(), so.data_ptr<float>(),
        scale, kSplits, B, stream);
}
```

And register:

```cpp
m.def("gdn_decode_v4_tma_loadOnly", &gdn_decode_v4_tma_loadOnly, "v4 warp-per-row (TMA load, scalar store)");
```

- [ ] **Step 5: Wire `MSINFER_KERNEL=v4_tma_loadOnly` through `run()`**

At the top of `msinfer_cuda.py`, import:

```python
from .tma_desc import get_tensor_map_pair
```

Wait — relative imports won't work because `msinfer_cuda.py` is loaded as the solution entry, not as a package. Fix: use absolute import within the same dir:

```python
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from tma_desc import get_tensor_map_pair
```

Update `_VALID_KERNELS` and `run()`:

```python
_VALID_KERNELS = ("v1", "v2_cpasync", "v4_noTMA", "v4_tma_loadOnly", "v4_tma")
```

Add the dispatch branch inside `run()`:

```python
    elif kernel == "v4_tma_loadOnly":
        desc_in, _desc_out = get_tensor_map_pair(
            state.data_ptr(), new_state.data_ptr(),
            q.size(0), HV, D // ks,
        )
        ext.gdn_decode_v4_tma_loadOnly(
            q, k, v, A_log, a, dt_bias, b, output, new_state, desc_in, DEFAULT_SCALE, ks,
        )
```

Note: `desc_out` is built but ignored at this step — cache hit covers Task 6.

- [ ] **Step 6: Run full bench — correctness gate**

```bash
python3 -m modal run scripts/run_modal.py --kernel-dir gdn_decode --env "MSINFER_KERNEL=v4_tma_loadOnly" 2>&1 | tee out/v4-task5-tma-loadOnly.log
```

**Expected:** 54 PASSED. Latency should be similar to v4_noTMA (TMA load by itself doesn't change much — the win comes from freeing LSU, which will manifest once store is also TMA). Absolute value: **don't gate on latency here**, only correctness.

**Diagnostic if load fails to complete (typically: kernel hangs or `mbarrier.try_wait` loops forever):**
- Check the `fence.proxy.async.shared::cta` is emitted before the `cp.async.bulk.tensor` issue — required for init → use ordering on Blackwell.
- Check the mbarrier `expect_tx` byte count matches the actual bytes delivered: for `float` kRows×128 tiles, `kRows * 128 * 4`.
- Verify `__cvta_generic_to_shared` is used for all shared memory pointers passed to PTX.

**Diagnostic if CUresult error at descriptor build (Python side):**
- `libcuda.so.1` resolution may fail — confirm Modal image has driver libs on `LD_LIBRARY_PATH` (`/usr/local/cuda/lib64` typical).
- Pointer alignment — `globalAddress` must be 16-B aligned. PyTorch fp32 tensors meet this.

- [ ] **Step 7: Commit**

```bash
git add gdn_decode/solution/python/msinfer_cuda.py out/v4-task5-tma-loadOnly.log
git commit -m "feat(gdn_decode): v4 TMA-load-only variant

Bulk cp.async.bulk.tensor.2d for state_in; scalar store for
state_out. Isolated TMA-load plumbing so descriptor/mbarrier bugs
surface cleanly. Gated behind MSINFER_KERNEL=v4_tma_loadOnly.
Removed in Task 6 once v4_tma supersedes."
```

---

## Task 6: Final `v4_tma` kernel — TMA load **and** TMA store, drop intermediates

**Why:** Fold store into TMA and land the final kernel. Then clean up the two intermediates (`v4_noTMA`, `v4_tma_loadOnly`) so the compile unit stays focused.

**Files:**
- Modify: `gdn_decode/solution/python/msinfer_cuda.py`
  - Add `gdn_decode_col_v4_tma` kernel (TMA load + TMA store in-place via `state_tile`)
  - Add launcher + binding accepting both descriptors
  - **Remove** `gdn_decode_col_v4_noTMA`, `gdn_decode_col_v4_tma_loadOnly` and their launchers / bindings
  - Update `_VALID_KERNELS`, `run()` dispatcher

- [ ] **Step 1: Add the final v4_tma kernel**

Append to `_CUDA_SRC` (replacing the two intermediates, which will be deleted in Step 4):

```cpp
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
    __shared__ alignas(16) float state_tile[kRows][128];
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
    // fence needed before TMA store to ensure smem writes are visible
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
```

- [ ] **Step 2: Add the launcher**

```cpp
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
        asm("trap;");
    }
}
```

- [ ] **Step 3: Add the C++ binding**

In `_CPP_SRC`:

```cpp
extern "C" void launch_gdn_v4_tma(
    const void*, const void*, const void*,
    const CUtensorMap*, const CUtensorMap*,
    const float*,
    const void*, const float*, const void*,
    void*, float, int, int, cudaStream_t);
```

Binding:

```cpp
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
```

Register:

```cpp
m.def("gdn_decode_v4_tma", &gdn_decode_v4_tma, "v4 warp-per-row + bulk TMA (final)");
```

- [ ] **Step 4: Delete the two intermediates**

In `_CUDA_SRC`: delete `gdn_decode_col_v4_noTMA`, `gdn_decode_col_v4_tma_loadOnly`, `launch_gdn_v4_noTMA`, `launch_gdn_v4_tma_loadOnly`.

In `_CPP_SRC`: delete the two `extern "C"` decls, the two binding functions, and the two `m.def(...)` lines.

In the Python dispatcher: remove `"v4_noTMA"` and `"v4_tma_loadOnly"` from `_VALID_KERNELS` and their `elif` branches from `run()`.

- [ ] **Step 5: Update the `run()` dispatcher to wire v4_tma**

`run()`'s v4_tma branch:

```python
    elif kernel == "v4_tma":
        desc_in, desc_out = get_tensor_map_pair(
            state.data_ptr(), new_state.data_ptr(),
            q.size(0), HV, D // ks,
        )
        ext.gdn_decode_v4_tma(
            q, k, v, A_log, a, dt_bias, b, output,
            desc_in, desc_out,
            DEFAULT_SCALE, ks,
        )
```

Final `_VALID_KERNELS`:

```python
_VALID_KERNELS = ("v1", "v2_cpasync", "v4_tma")
```

- [ ] **Step 6: Run full bench — correctness gate**

```bash
python3 -m modal run scripts/run_modal.py --kernel-dir gdn_decode --env "MSINFER_KERNEL=v4_tma" 2>&1 | tee out/v4-task6-tma.log
```

**Expected:** 54 PASSED. Per-workload abs_err within 2× of v1 baseline.

**Diagnostic if correctness fails (typically: all workloads show same-pattern error):**
- The TMA store descriptor order might be wrong — `{col, row}` in the coord tuple maps to `{inner, outer}` of the global tensor. Check that `col_base=0, row_base=row_base_elem` maps to the correct 2D offset.
- The fence before the TMA store ensures smem writes are visible to the TMA engine. Omitting it can yield "store 0s" behavior on some runs.

**Diagnostic if the kernel hangs:**
- `cp.async.bulk.wait_group 0` waits for ALL outstanding bulk ops. If an earlier bulk op (from a different kernel?) was pending, it stalls — shouldn't happen here since we're the only kernel issuing bulk.

- [ ] **Step 7: Commit**

```bash
git add gdn_decode/solution/python/msinfer_cuda.py out/v4-task6-tma.log
git commit -m "feat(gdn_decode): v4_tma final — TMA load + TMA store

Drops the v4_noTMA and v4_tma_loadOnly intermediates. The final
kernel loads the state tile via one cp.async.bulk.tensor.2d issue,
computes in-place in smem with warp-per-row layout (no block-level
sync in the compute loop), and stores via a second bulk TMA issue.

Gated behind MSINFER_KERNEL=v4_tma. Default remains v1 until Gate A."
```

---

## Task 7: Gate A — Performance verification (본판 50 iter × 3 trial)

**Why:** The spec's acceptance criteria require ≥ 2× at B=32/B=64 and no B bucket regressing > 10%. This task runs the full suite under the submission config and checks against the v1 baseline.

**Files:**
- Modify: `scripts/run_modal.py:80` — bench config → 본판 (50 iter × 3 trial)
- Save: `out/v4-task7-gateA.log`

- [ ] **Step 1: Flip bench config to 본판**

Edit line 80 of `scripts/run_modal.py`:

```python
config = BenchmarkConfig(warmup_runs=3, iterations=50, num_trials=3)
```

- [ ] **Step 2: Run v1 reference under 본판**

```bash
python3 -m modal run scripts/run_modal.py --kernel-dir gdn_decode --env "MSINFER_KERNEL=v1" 2>&1 | tee out/v4-task7-v1-bonpan.log
```

Capture the B=32 and B=64 buckets — these are the gate anchors.

- [ ] **Step 3: Run v4_tma under 본판 with SASS dump**

```bash
python3 -m modal run scripts/run_modal.py --kernel-dir gdn_decode --env "MSINFER_KERNEL=v4_tma" --dump-sass 2>&1 | tee out/v4-task7-gateA.log
```

- [ ] **Step 4: Verify TMA ops emitted in SASS**

```bash
tar -xzf out/sass-dump.tar.gz -C out/
grep -l "cp.async.bulk.tensor" out/cute-asm/*.ptx out/cute-asm/*.sass 2>/dev/null | head -5
```

Expected: at least one file matches. If empty, v4_tma kernel was not emitted under the expected name — investigate `ptxas-verbose` for errors.

- [ ] **Step 5: Check register pressure**

```bash
grep -E "registers|spill" out/cute-asm/*.log 2>/dev/null
```

Expected: `gdn_decode_col_v4_tma` registers ≤ 48, spills = 0. If spills > 0 or registers > 48, occupancy drops below 8 blocks/SM — performance suffers. Iterate on kernel (try `#pragma unroll 1` around the iter loop, or reduce register holding of `k_reg`, `q_reg` to `half2` packed) before declaring Gate A.

- [ ] **Step 6: Compute per-bucket latency and check gates**

```bash
# Extract B=32 and B=64 latency bands
grep "PASSED" out/v4-task7-gateA.log | awk '{print $5}' | sort -n | uniq -c
```

Manual check against gates:

| B bucket | v1 latency (from Task 7 Step 2) | v4_tma latency | Gate |
|----------|--------------------------------|----------------|------|
| 1        | *from log*                     | *from log*     | v4 ≤ 1.10 × v1 |
| 2        | *from log*                     | *from log*     | v4 ≤ 1.10 × v1 |
| 4        | *from log*                     | *from log*     | v4 ≤ 1.10 × v1 |
| 8        | *from log*                     | *from log*     | v4 ≤ 1.10 × v1 |
| 16       | *from log*                     | *from log*     | v4 ≤ 1.10 × v1 |
| **32**   | *from log* (~0.031)            | *from log*     | **v4 ≤ 0.50 × v1** |
| **64**   | *from log* (~0.045)            | *from log*     | **v4 ≤ 0.50 × v1** |

**If the B=32 or B=64 gate misses:**
- Run with `--dump-sass` → inspect ptxas for register count; if > 48, occupancy is the problem.
- Check whether `cp.async.bulk.tensor` is in the SASS (Step 4). If not, compile fell back — address that first.
- Run `ncu` (via Modal `run_ncu` if available, or just read the CUDA API profiler output) to see TMA completion vs compute overlap. If TMA load finishes before compute starts, we're not gaining — consider merging stages.

**If a low-B bucket regresses by > 10%:**
- Likely TMA setup overhead (descriptor build). Check descriptor cache hit rate — if rebuilding every call, the cache key is wrong.
- Alternative: set `MSINFER_KERNEL=v1` for B=1/2 via a Python-side B-conditional dispatch. Only do this if the regression is real and persistent; the spec allows it.

- [ ] **Step 7: Commit results**

```bash
git add scripts/run_modal.py out/v4-task7-*.log out/v4-task7-gateA.log out/sass-dump.tar.gz
git commit -m "test(gdn_decode): Gate A — v4_tma latency verification (본판 50/3)

Captures v1 vs v4_tma under submission config. SASS dump confirms
cp.async.bulk.tensor.2d emitted and register pressure within
__launch_bounds__(128, 8) budget."
```

---

## Task 8: Flip default `MSINFER_KERNEL` to `v4_tma`

**Why:** Once Gate A is green, make v4_tma the default so the unmodified `run()` entry point uses it.

**Files:**
- Modify: `gdn_decode/solution/python/msinfer_cuda.py` — `_select_kernel()` default

- [ ] **Step 1: Change the default**

In `_select_kernel()`:

```python
def _select_kernel() -> str:
    name = os.environ.get("MSINFER_KERNEL", "v4_tma")   # was "v1"
    if name not in _VALID_KERNELS:
        raise ValueError(
            f"MSINFER_KERNEL={name!r} invalid; must be one of {_VALID_KERNELS}"
        )
    return name
```

- [ ] **Step 2: Run full bench without env override (default path)**

```bash
python3 -m modal run scripts/run_modal.py --kernel-dir gdn_decode 2>&1 | tee out/v4-task8-default.log
```

**Expected:** 54 PASSED, latencies match Task 7's v4_tma log (within ±5% noise).

- [ ] **Step 3: Smoke-test the fallback still works**

```bash
python3 -m modal run scripts/run_modal.py --kernel-dir gdn_decode --env "MSINFER_KERNEL=v1" 2>&1 | grep "PASSED" | wc -l
```

Expected: `54`.

- [ ] **Step 4: Commit**

```bash
git add gdn_decode/solution/python/msinfer_cuda.py out/v4-task8-default.log
git commit -m "feat(gdn_decode): default to v4_tma kernel

Flips MSINFER_KERNEL default from v1 to v4_tma after Gate A green.
MSINFER_KERNEL=v1 remains available as emergency fallback."
```

---

## Task 9: Confirmation run + remove `v2_cpasync`

**Why:** The spec requires 2 consecutive green full-suite runs before removing v2_cpasync. This task does the second run and then prunes.

**Files:**
- Modify: `gdn_decode/solution/python/msinfer_cuda.py`
  - Remove `gdn_decode_col_v2_tma` kernel, `launch_gdn_v2_tma`, `gdn_decode_v2_tma` binding, and `"v2_cpasync"` from `_VALID_KERNELS` + dispatcher

- [ ] **Step 1: Second confirmation run**

```bash
python3 -m modal run scripts/run_modal.py --kernel-dir gdn_decode 2>&1 | tee out/v4-task9-confirm.log
```

Expected: 54 PASSED, latencies match Task 8's log within ±5%.

- [ ] **Step 2: Remove v2_cpasync kernel**

In `_CUDA_SRC`: delete the `gdn_decode_col_v2_tma` kernel body (the 100-ish lines starting with `// ── v2: cp.async double-buffering (TMA-lite) ──`).

In `_CUDA_SRC`: delete `extern "C" void launch_gdn_v2_tma(...)`.

In `_CPP_SRC`: delete the extern decl for `launch_gdn_v2_tma`, the `gdn_decode_v2_tma` binding function, and its `m.def` line.

In Python dispatcher: remove `"v2_cpasync"` from `_VALID_KERNELS`; remove its `elif` branch from `run()`.

Final `_VALID_KERNELS`:

```python
_VALID_KERNELS = ("v1", "v4_tma")
```

- [ ] **Step 3: Verify the default and fallback both still work**

```bash
# Default (v4_tma)
python3 -m modal run scripts/run_modal.py --kernel-dir gdn_decode 2>&1 | grep -c PASSED
# Fallback (v1)
python3 -m modal run scripts/run_modal.py --kernel-dir gdn_decode --env "MSINFER_KERNEL=v1" 2>&1 | grep -c PASSED
# Invalid (should surface a ValueError)
python3 -m modal run scripts/run_modal.py --kernel-dir gdn_decode --env "MSINFER_KERNEL=v2_cpasync" 2>&1 | grep "invalid"
```

Expected: `54`, `54`, match.

- [ ] **Step 4: Update the module docstring**

Edit the top docstring of `msinfer_cuda.py`:

```python
"""GDN decode — CUDA C warp-per-row bulk-TMA kernel (v4).

Two variants compiled:
  v4_tma  - warp-per-row + cp.async.bulk.tensor.2d state I/O (default)
  v1      - legacy __ldg column-parallel (emergency fallback)

Selected via MSINFER_KERNEL env var. Requires sm_100a (Blackwell).

Design rationale: the v4 kernel eliminates per-row block sync
(warps process independent rows) and replaces scalar LSU state
I/O with bulk TMA (dedicated engine), targeting high HBM
utilization at B ≥ 16.
"""
```

- [ ] **Step 5: Final bench + commit**

```bash
python3 -m modal run scripts/run_modal.py --kernel-dir gdn_decode 2>&1 | tee out/v4-task9-final.log
git add gdn_decode/solution/python/msinfer_cuda.py out/v4-task9-*.log
git commit -m "chore(gdn_decode): remove v2_cpasync after v4_tma stabilized

Two consecutive green full-suite runs with v4_tma as default.
v1 retained as emergency fallback; v2_cpasync's cp.async-based
state load is fully superseded by v4's bulk TMA path."
```

---

## Self-Review — Spec coverage checklist

Mapping spec sections → plan tasks:

| Spec section | Task(s) |
|--------------|---------|
| §3.1 New kernel `gdn_decode_col_v4_tma` | Task 6 |
| §3.2 Warp-per-row thread layout | Task 4 (layout introduced without TMA) + Task 6 (final) |
| §3.3 Shared memory layout | Task 5 (state_tile introduced) + Task 6 |
| §3.4 CUtensorMap descriptors | Task 3 (builder) + Tasks 5/6 (wired into kernel) |
| §3.5 Pipeline (stages 0-6) | Task 5 (stages 0-4 with TMA load) + Task 6 (full pipeline) |
| §3.6 `MSINFER_KERNEL` env dispatch | Task 2 (dispatcher) + Tasks 4-6 (new values) + Task 8 (default flip) + Task 9 (prune v2) |
| §4 Correctness (arithmetic identity, tolerance) | Correctness gate in Tasks 4, 5, 6 (every task runs 54 workloads) |
| §5 Implementation order + gates | Tasks 1-9 in sequence |
| §5 Gate A (latency at B=32/64) | Task 7 |
| §5 Gate B (correctness all 54) | Every task's correctness step |
| §6 Out of scope (TF32 TC, cluster DSM, persistent block) | Not in plan — correctly parked per spec |
| §7 Acceptance checklist | Covered by Tasks 4-9 |

## Self-Review — Placeholder scan

Grepped the plan for `TBD`, `TODO`, `Similar to`, `implement later`: none.

Each step that changes code includes the actual code. Each bench step includes the exact command.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-24-gdn-decode-v4-tma-plan.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. Best when steps are large (this plan has Modal-bench round trips that take 1-2 min each).

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints. Best when you want to watch every step live.

**Which approach?**
