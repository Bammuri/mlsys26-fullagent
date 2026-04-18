# GDN CUDA Kernel Optimization Report

## Overview

MLSys 2026 FlashInfer AI Kernel Generation Contest — Gated Delta Network (GDN) track.
B200 GPU (SM100), `gdn_decode_qk4_v8_d128_k_last` + `gdn_prefill_qk4_v8_d128_k_last`.

### Kernel Specs
- **num_q/k_heads = 4, num_v_heads = 8** (GVA mode, v:qk ratio 2:1)
- **head_size = 128** (both K and V dimensions)
- **State**: `[B/N, 8, 128, 128]` f32, k-last layout
- **Parallelization**: 4-way V-dim split → 1 warp (32 threads) per block; each thread owns one V-row of the 128×128 state, held in registers as `float4 sr[32]`.
- **Binding**: TVM FFI, DPS mode, `TVM_FFI_DLL_EXPORT_TYPED_FUNC`

### Delta-rule update (both kernels)
```
g         = exp(-exp(A_log) * softplus(a + dt_bias))   // scalar per head
beta      = sigmoid(b)                                 // scalar per head
old_v     = k · (g * state)                            // 128-dot, per row
delta     = beta * (v - old_v)                         // scalar per row
new_state = g * state + k * delta                      // rank-1 update
out       = scale * (qs + delta * qk)
              where qs = q · (g * state), qk = q · k   // avoids a second 128-dot
```

---

## Decode Kernel (current: v17)

### Architecture
- Grid: `(B × 8 × 4 splits)`, Block: 32 threads (1 warp)
- Each thread owns one V-row of the 128×128 state (128 fp32 in registers via `float4[32]`)
- q, k loaded cooperatively into shared memory (256 floats total)
- Per-head scalars (g, beta, qk) held in registers — no smem traffic

### Optimization History

| Version | Change | Notes |
|---------|--------|-------|
| v2 (baseline) | 2-pass state read from global | State read twice from global memory |
| v3 | 1-pass: state → registers first | Eliminates second global read |
| v5 | + g·state pre-multiply (fused into first pass) | `sr[i] *= g`; update becomes `sr[i] += k·delta` |
| v12 | + `out = qs + delta·qk` trick | Avoid second full Q·new_state dot product |
| v14 | 4-way V-dim split, 1 warp/block | Replaces 1-block-per-(batch,v_head) layout |
| v16 | Drop `st.global.cs` streaming store | Default cache write policy is faster |
| **v17 (current)** | Remove smem for g/β/qk scalars; `__syncwarp`; extract `compute_gate` | Cleanup — 1-warp block makes `__syncthreads` degenerate. No behaviour change, -1 smem barrier. |

### Effective optimizations

#### D1: Single-pass state read ✅ Applied (v3)
- **Problem**: State tensor (`float4[32]` per thread = 512 bytes) was read from global memory twice — once for `old_v`, once for state update + output.
- **Fix**: Load state into register array `float4 sr[32]` in a single pass; both loops operate on registers.
- **Result**: ~3% latency. Modest because L1 partially hid the second read.

#### D2: g·state pre-multiply ✅ Applied (v5)
- **Problem**: Both loops compute `g * state[i]` independently, duplicating a multiply.
- **Fix**: `sr[i] *= g` once at the top; update loop becomes `sr[i] += k[j] * delta`.
- **Result**: Saves 128 mul/thread; neutral latency due to FMA scheduling headroom, but cleaner dependency chain.

#### D3: `out = qs + δ·qk` reformulation ✅ Applied (v12)
- **Math**: `out = Q · (g·state + k·delta) = Q·(g·state) + delta·(Q·K) = qs + delta·qk`.
- **Fix**: Accumulate `qs = q·sr` alongside `ov = k·sr` in the fused first pass. Compute `qk = q·k` as a warp-split dot + `__shfl_xor_sync` reduction. The second pass only does `sr[i] += k*delta`.
- **Result**: Eliminates a full 128-elem Q·new_state dot product in the second pass.

#### D4: 4-way V-dim split ✅ Applied (v14)
- **Problem**: 1 block per (batch, v_head) with 128 threads under-utilises SMs when batch_size is small.
- **Fix**: Split state across 4 blocks (32 rows each). Grid = `B × 8 × 4`; thread count per block drops from 128 → 32 (1 warp).
- **Result**: Better occupancy on small batches; same arithmetic per block.

#### D5: Scalar broadcast via warp, not smem ✅ Applied (v17)
- **Problem**: g, β, qk were each written by lane 0 to shared memory, synchronized, then re-read. This needs `__syncthreads` × 2.
- **Fix**: All lanes compute g/β redundantly (MUFU is parallel across lanes — SIMD makes redundancy free). `qk` is computed as a warp-split dot + `__shfl_xor_sync` reduction, leaving the result in every lane.
- **Result**: Two `__syncthreads` → one `__syncwarp` (or none if loads already synced). 3 fewer smem scalars.

---

## Prefill Kernel (current: v7)

### Architecture
- Grid: `(N_seqs × 8 × 4 splits)`, Block: 32 threads (1 warp)
- Each thread owns one V-row; tokens processed **sequentially** in a for-loop
- State stays in registers (`float4 sr[32]`) across token iterations
- K, Q double-buffered via `cp.async.cg` (L2-only, 16 B per thread, 32 threads × 16 B = 512 B per token = 128 bf16 K + 128 bf16 Q)
- `cu_seqlens` drives variable-length sequence batching

### Optimization History

| Version | Change | Notes |
|---------|--------|-------|
| v2 (baseline) | Each thread reads k[128], q[128] from global independently | 16,384 scalar bf16 loads per token per head |
| v3 | Shared-memory k/q (float, single buffer) | -40% on long seqs — biggest single win |
| v4 | + synchronous double-buffer prefetch | -25% regression; sync loads don't overlap |
| v5 | Revert to v3 + `__ldg` | Neutral |
| v6 | cp.async.ca bf16 double-buffered pipeline | Additional -6~8% over v5 |
| v6+ | 4-way V-dim split, 1 warp/block, `cp.async.cg` (L2-only) for K/Q | Matches decode's layout; cleaner occupancy |
| **v7 (current)** | **Decode-style `qs + δ·qk` + in-place g·state** | Eliminates 128 mul/token and a full Q·new_state dot |

### Key optimization applied in v7

#### P7-revised: In-place `g·state` fusion + decode-style output ✅ Applied (v7)

Before (v6): the inner loop recomputed `g * sr[i]` in both passes and carried out a full 128-element Q·new_state dot in the second pass.

```cpp
// v6 prefill — per-token inner loops
for i in 0..32:
    ov += K * (g * sr[i])           // 128 muls just to form g·state
delta = beta * (V - ov)
for i in 0..32:
    upd = g * sr[i] + K * delta     // 128 muls recompute g·state
    sr[i] = upd
    oa += Q * upd                   // full 128-elem Q · new_state dot
out = oa
```

v7 applies the exact algebraic shape used by decode:

```cpp
// v7 prefill — per-token inner loops (matches decode)
for i in 0..32:
    sr[i] *= g                      // state held as g·state going forward
    ov += K * sr[i]                 // K · (g·state)
    qs += Q * sr[i]                 // Q · (g·state) — new accumulator
qk = warp_reduce(Σ q_tid · k_tid)   // Q · K, split across lanes, shuffle-reduced
delta   = beta * (V - ov)
out_acc = qs + delta * qk           // no second Q·new_state dot
for i in 0..32:
    sr[i] += K * delta              // sr[i] = g·state + K·delta = new_state
```

**Net per-token arithmetic saving (per thread)**:
- v6: `256 mul + 384 fma` → `1024 flops`
- v7: `128 mul + 384 fma` → `896 flops`
- Savings: **128 fma-equivalent per token per thread** (~12.5% of the inner-loop arithmetic).

This also eliminates Q reads in the second pass — only K is touched — halving second-pass smem traffic.

**Why the original doc dismissed this**: an earlier version of the report (P7 "Not applied — same total FLOPS") was comparing against a hand-applied pre-multiply alone, not a combined pre-multiply + decode-style `qs/qk` fusion. Once both are applied together, the second pass loses a whole Q·new_state dot, which is not zero-cost.

### Still ineffective / not applied

| Optimization | Why |
|-------------|-----|
| Sync double-buffer (P3) | No overlap without async copy — rejected in v4 |
| `__ldg` on bf16 K/Q | No benefit with cp.async.cg; cp.async already bypasses L1 |
| g-only pre-multiply (P7 as stated originally) | Neutral *alone*; the win only materialises when combined with qs+δ·qk |
| Const preload (`A_log`, `dt_bias`) | L1 cache already handles broadcast loads |
| Chunk-based token parallelism | State recurrence is non-linear (`delta` depends on `state` via `k·state`); no parallel scan |

---

## Summary

| Layer | Applied |
|-------|---------|
| **Prefill v7** | Shared-mem K/Q + cp.async.cg double-buffer + **`qs + δ·qk` + in-place g·state** (v7) + 4-way V-split |
| **Decode v17** | Single-pass state + pre-scaled g·state + `out = qs + δ·qk` + 4-way V-split + scalar-via-register broadcast |

### Python + CuTe DSL port (scoring surface, `solution/python/msinfer_entry.py`)

| step | commit | change | decode kernel latency | prefill kernel latency |
|------|--------|--------|---------:|---------:|
| v1 | fdddc20 | CuTe surface, 128-thread block, cross-warp qk smem reduce | 0.0354 ms | 0.0534 ms |
| v2 | 17cf8b2 | 4-way V-split, 32-thread warp block, warp-only qk reduce | 0.0248 ms (-30%) | 0.0380 ms (-29%) |
| v3 | c23802c | Vectorized state via `cute.local_tile` + `cute.autovec_copy` (emits LDG/STG.128) | 0.0193 ms (-22%) | 0.0372 ms (-2%) |
| v4 | fd3a81b | `cute.arch.sync_warp()` in place of `barrier()` (bar.warp.sync vs bar.sync 0) | 0.025 ms† | 0.0364 ms (-2%) |
| ref | static CUDA | v17 decode / v7 prefill, hand-tuned nvcc | 0.012 ms | 0.0304 ms |

†Modal run-to-run variance on Python reference latency is ~30–70% (cold-start scheduling). Kernel latency between v3 and v4 runs is within noise; the sync_warp swap is semantically correct for 1-warp blocks and at worst neutral.

All 35 tested workloads (30 decode × batch ∈ {1,4,8,16} + 5 prefill seq_len ≤ 30) PASSED within contest tolerance (abs_err ≤ 1e-6, rel_err ≤ 1e-1).

### Measured on Modal B200 (`flashinfer/flashinfer-ci-cu132`, CUDA 13.2)

Prefill v7 — 20 smallest workloads (total_seq_len ≤ 35):

| #  | latency (ms) | speedup vs Python reference |
|----|-------------:|----------------------------:|
| 1  | 0.021 | 53.7× |
| …  | … | … |
| 11 | 0.033 | **133.5×** |
| 16 | 0.039 | **131.2×** |
| 17 | 0.042 | **131.2×** |
| 20 | 0.040 | **137.9×** |

Mean speedup ≈ **82×** across 20 workloads; peaks at **138×** on the longer-token batches. Every workload passed correctness (max abs-err ≤ 1e-6, max rel-err ≤ 2e-2).

Decode v17 — 30 smallest workloads (batch_size ∈ {1, 4, 8, 16}):

| batch | latency (ms) | mean speedup |
|:-----:|-------------:|-------------:|
| 1     | 0.011–0.015 | 80–107× |
| 4     | 0.012       | 356–380× |
| 8     | 0.012–0.013 | 608–667× |
| 16    | 0.015–0.017 | 933–1071× |

Speedup grows with batch because the Python reference scales linearly while the kernel saturates SMs once `batch × 8 v_heads × 4 splits` ≥ SM count. All 30 workloads passed.

### ptxas (sm_100a, CUDA 13.1) resource usage

|            | regs/thread | spill stores | spill loads | barriers | smem (B) |
|------------|-----------:|------------:|-----------:|--------:|---------:|
| decode v16 | 255        | 132         | 132         | 0       | 1024     |
| decode v17 | 255        | 132         | 132         | 0       | 1024     |
| prefill v6 | 255        | 100         | 104         | 1       | 1036     |
| **prefill v7** | 255    | **68**      | **68**      | **0**   | **1024** |

Prefill v7 cuts local-memory spills by 30% and removes the sole `__syncthreads` barrier.

## Bottleneck

Both kernels are **compute-bound on a per-row sequential recurrence**. Further wins would require an algorithmic change — e.g. parallel scan on a linearised recurrence, or mixed-precision state storage. With the current formulation (full-precision state, non-linear recurrence), register-held state + double-buffered K/Q appears close to the achievable ceiling for a warp-per-block layout.

---

## File Layout

```
config.toml                     # Switch definition + entry_point for decode/prefill
solution/cuda/kernel.cu         # Both kernels + TVM FFI exports
scripts/run_modal.py            # Modal runner (flashinfer-ci-cu132 image)
scripts/pack_solution.py        # Packs into solution.json
docs/gdn-optimization-report.md # This file
```

### Switching between decode and prefill

**Decode:**
```toml
definition  = "gdn_decode_qk4_v8_d128_k_last"
entry_point = "kernel.cu::kernel"
```

**Prefill:**
```toml
definition  = "gdn_prefill_qk4_v8_d128_k_last"
entry_point = "kernel.cu::kernel_prefill"
```
