# GDN Decode v4 — Bulk-TMA Column-Parallel Kernel

**Date:** 2026-04-24
**Target:** `gdn_decode/solution/python/msinfer_cuda.py`
**Branch baseline:** `decode-col-v3` (commit `4085640`)
**Scoring env:** Modal B200 (kjms2026), sm_100a, bench `(warmup=3, iter=20, trial=3)` for dev / `(warmup=3, iter=50, trial=3)` for submission

---

## 1. Problem statement

The v3 column-parallel CUDA kernel is **deeply memory-bound at high batch sizes** but achieves only a small fraction of HBM bandwidth. The optimization target chosen by the user is the **B=32 / B=64 bucket** — where the kernel-launch floor is amortized away and the remaining cost is state read/write traffic.

### Baseline measurement (v3, both variants, Modal B200, 100 iters × 5 trials)

State traffic per call = `2 × B × HV × D² × sizeof(fp32) = B × 1 MB` (read + write).

| B | v1 `__ldg` (ms) | v2 cp.async (ms) | state R+W | BW (v1) | **HBM utilization (v1, 8 TB/s peak)** |
|---|-----------------|-------------------|-----------|---------|---------------------------------------|
| 1 | 0.017 | 0.013 | 1 MB | 59 GB/s | 0.7% (launch-latency floor) |
| 2 | 0.017 | 0.013 | 2 MB | 118 GB/s | 1.5% |
| 4 | 0.022 | 0.018 | 4 MB | 182 GB/s | 2% |
| 8 | 0.023 | 0.019 | 8 MB | 348 GB/s | 4% |
| 16 | 0.025 | 0.024 | 16 MB | 640 GB/s | 8% |
| **32** | **0.031** | 0.033 | 32 MB | **1.03 TB/s** | **13%** |
| **64** | **0.045** | 0.042 | 64 MB | **1.42 TB/s** | **18%** |

v2's cp.async wins ~20% at low-B (latency hiding) but is flat-to-negative at high-B — confirming that cp.async is still LSU-bound (128 scalar 4 B issues/row). True bulk TMA is a separate engine and does not compete with LSU.

### Ceiling

At a realistic 70% HBM utilization (~5.5 TB/s), the physics-limited latency is:

| B | Current (v1) | HBM-limit @ 5.5 TB/s | Ceiling |
|---|--------------|-----------------------|---------|
| 32 | 0.031 ms | 0.006 ms | ~5× |
| 64 | 0.045 ms | 0.012 ms | ~4× |

The v4 design target is **≥ 2× speedup at B=32 and B=64** (half the ceiling), with no regression > 10% at any B bucket.

---

## 2. Root causes in v3 (for the design to address)

1. **Scalar 4 B state loads/stores.** `__ldg(&si[r*D + col])` issues 128 LSU operations per row per block, saturating the LSU instruction queue before HBM is saturated.
2. **Serial per-row memory dependency.** Each row's compute blocks on its state load; latency is fully exposed.
3. **2 `__syncthreads` per row.** At kSplits=4, 32 rows × 2 = 64 block-level syncs/block, times 2048 blocks at B=64 = 128 K barrier operations per call.
4. **cp.async is a half-measure.** Still scalar per-thread issues — relieves latency at low-B but keeps the LSU bottleneck at high-B.

---

## 3. Design

### 3.1 Kernel

New kernel `gdn_decode_col_v4_tma` — replaces v1 as the default path once validated. v1 kept as emergency fallback. v2 cp.async removed after v4 validates green.

### 3.2 Thread layout — *warp-per-row*, 4-cols-per-lane

- **Grid:** `(B × HV × kSplits)` — unchanged from v3.
- **Block:** `128` threads = 4 warps. Two template instantiations at compile time:
  - `kSplits=4` (selected when B ≥ 8) → 32 rows/block, 8 compute iterations.
  - `kSplits=8` (selected when B ≤ 4) → 16 rows/block, 4 compute iterations.
- **Per warp:** each of 4 warps owns a distinct row per iteration — **warps are independent, no block-level sync in the compute loop**.
- **Per lane inside a warp:** lane `l` owns columns `{4l+0, 4l+1, 4l+2, 4l+3}`. 32 lanes × 4 cols = D=128.

Implication: reductions are **warp-internal** (5 shuffles), eliminating the v3 smem-merge + `__syncthreads` × 2 per row.

### 3.3 Shared memory layout

```
__shared__ float    sQ[128], sK[128], sV[128];       //    1.5 KB — loaded at block start
__shared__ float    state_tile[kRows][128];          //  16 KB (kSplits=4) / 8 KB (kSplits=8)
__shared__ uint64_t mbar[2];                         //   16 B  — mbar[0]=load, mbar[1]=store
```

`state_tile` is **updated in place**: TMA loads into it, compute mutates it, TMA stores it out.

### 3.4 Tensor map (`CUtensorMap`) descriptors

Two descriptors per shape `(B, HV)`:
- `tensor_map_in` → state_in
- `tensor_map_out` → state_out

**View state as flat 2D:** `(B · HV · D, D)` with tile shape `(kRows, D=128)`. (4D descriptors also work but add unnecessary complexity for a per-block flat tile origin.)

**Parameters:**
- `tensorDataType`: `CU_TENSOR_MAP_DATA_TYPE_FLOAT32`
- `tensorRank`: 2
- `globalDim`: `{D=128, B·HV·D}` (inner, outer)
- `globalStrides`: `{D*4}` bytes (outer stride only; inner stride = element size)
- `boxDim`: `{128, kRows}`
- `swizzle`: `CU_TENSOR_MAP_SWIZZLE_NONE`
- `l2Promotion`: `CU_TENSOR_MAP_L2_PROMOTION_L2_128B`
- `oobFill`: `CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE`

Per-block tile origin for `(batch, v_head, split)`:
```
row_base = batch * HV * D + v_head * D + split * kRows
col_base = 0
```

**Lifecycle:**
- Host (Python): `ctypes`-call `cuTensorMapEncodeTiled` from `libcuda.so` when shape `(B, HV)` first seen. Cache the two 128-B descriptors keyed on `(B, HV, kSplits, state_in_dptr_region, state_out_dptr_region)`. Region = 2 MB-aligned base; per-call base change invalidates the cache (will be rare in bench; pointers are typically reused across iterations).
- Passed as `__grid_constant__ const CUtensorMap` kernel args (128 B each, fits in kernel-arg buffer).

### 3.5 Pipeline (per block)

```
Stage 0  │ warp0: init mbar[0..1]; all threads: load sQ/sK/sV into smem; compute g, β
Stage 1  │ thread 0: cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes
         │            (state_tile ← state_in, origin=(0, row_base), bytes=kRows*512)
         │           mbarrier.arrive.expect_tx mbar[0], kRows*512
Stage 2  │ warp0 only: qk = dot(sQ, sK)  —  runs in parallel with TMA load
Stage 3  │ __syncthreads()
         │ mbarrier.try_wait.parity mbar[0], 0
Stage 4  │ compute loop  —  for iter in 0 .. (kRows/4 − 1):
         │   (each warp w owns row r = row0 + iter*4 + w; warps independent)
         │     state_reg[0..3] = state_tile[r][4l..4l+3]
         │     s_new[0..3]     = g * state_reg[0..3]
         │     ov_partial      = Σ k[4l+c] * s_new[c]
         │     qs_partial      = Σ q[4l+c] * s_new[c]
         │     (pack (ov, qs) as float2; 5 shfl_xor_sync reductions)
         │     lane 0: δ = β * (sV[r] − ov);  out[r] = bf16(scale * (qs + δ*qk))
         │     δ ← __shfl_sync(0xFFFFFFFF, δ, 0)              # broadcast inside warp
         │     state_tile[r][4l..4l+3] = s_new[0..3] + k[4l..4l+3] * δ
Stage 5  │ __syncthreads()
Stage 6  │ thread 0: cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group
         │            (state_out ← state_tile)
         │           cp.async.bulk.commit_group; cp.async.bulk.wait_group 0
         │ kernel exit
```

**Block barriers:** 2 (stages 3, 5). Down from 64 at kSplits=4.

### 3.6 Dispatch & env var

```
MSINFER_KERNEL=v4_tma   (default once validated)
MSINFER_KERNEL=v1       (emergency fallback to current default)
MSINFER_KERNEL=v2_cpasync  (kept during ramp; removed after 2 green full-suite runs)
```

Python `_get_ext()` exposes all three bindings. `run()` picks via the env var, defaulting to `v4_tma`.

---

## 4. Correctness

- **Arithmetic is identical** to v1: same FP32 multiplies / adds, same softplus/sigmoid/exp helpers. Only transport (`__ldg` → bulk TMA) and reduction *topology* change.
- **Reduction order differs** from v1: warp-internal only, no smem-staged cross-warp sum. Worst-case drift vs v1: ≤ 1 ULP × 128 summands ≈ 1e-5 absolute.
- **Tolerance:** workloads currently pass with rel_err up to 5.76e-1 and abs_err up to 3.05e-5 (v1 baseline measured 2026-04-24). v4 must stay within **2× of v1's per-workload abs_err and rel_err**; any single-workload regression > 2× halts rollout.

---

## 5. Testing & rollout

### Implementation order
1. Land `gdn_decode_col_v4_tma` next to v1 in the same `.cu` string. v1 remains default.
2. Add Python-side tensor-map builder (`ctypes` → `cuTensorMapEncodeTiled`) with per-shape cache. Wire `__grid_constant__` kernel args through the existing C++ launcher.
3. Expose `MSINFER_KERNEL` env var; run full 54-workload bench with `v4_tma`.
4. **Gate A — latency:** `latency(v4_tma) / latency(v1) ≤ 0.5` at B=32 and B=64; `≤ 1.1` at every B bucket.
5. **Gate B — correctness:** all 54 workloads PASSED; per-workload errors within 2× of v1.
6. Both gates green → flip default to `v4_tma`, re-run full suite to confirm.
7. After 2 consecutive green full-suite runs, remove v2_cpasync.

### Bench commands
```bash
# Iteration cycles (기본)
python3 -m modal run scripts/run_modal.py --kernel-dir gdn_decode --env "MSINFER_KERNEL=v4_tma"

# Submission candidate (본판) — iterations=50 in BenchmarkConfig
python3 -m modal run scripts/run_modal.py --kernel-dir gdn_decode --env "MSINFER_KERNEL=v4_tma"
```

### Debug artifacts
- `--dump_sass` once per design iteration to confirm ptxas emits `cp.async.bulk.tensor.2d` ops and register pressure stays within `__launch_bounds__(128, 8)`.
- ncu trace if Gate A misses — checking for bank conflicts in `state_tile` smem, TMA completion-time vs compute-time overlap.

---

## 6. Out of scope (parked as follow-ups)

- **Approach C — TF32 tensor cores for K·State / Q·State GEMV.** Only worth pursuing if post-v4 ncu shows LSU-bound, not HBM-bound. Separate spec.
- **Cluster-level distributed shared memory** for Q/K sharing across v_heads. State (the dominant traffic) is per-head, so DSM is irrelevant.
- **Persistent-block multi-head pipelining** (one block drains multiple heads with double-buffered TMA). Expected ~1.2×; complexity high. Reopen only if v4 lands short of target.

---

## 7. Acceptance checklist

- [ ] `gdn_decode_col_v4_tma` compiled with `-arch=sm_100a` and emits `cp.async.bulk.tensor.2d` in SASS
- [ ] Tensor-map builder works for both kSplits=4 and kSplits=8 instantiations
- [ ] All 54 workloads PASSED under `MSINFER_KERNEL=v4_tma`
- [ ] B=32 latency ≤ 0.016 ms (≥ 2× over v1's 0.031 ms)
- [ ] B=64 latency ≤ 0.023 ms (≥ 2× over v1's 0.045 ms)
- [ ] No B bucket regresses by > 10% vs v1
- [ ] Per-workload abs_err ≤ 2× v1 baseline; rel_err ≤ 2× v1 baseline
- [ ] `MSINFER_KERNEL=v1` reverts to previous v3 default behavior exactly (smoke test)
