---
title: "perf: GDN CuTe kernel closing v17/v7 gap via 4-way V-split port"
type: refactor
status: active
date: 2026-04-18
---

# perf: GDN CuTe kernel closing v17/v7 gap via 4-way V-split port

## Overview

The GDN decode + prefill kernels in `solution/python/msinfer_entry.py` run correctly on
the Python + CuTe + TVM-FFI surface the team verified (Modal bench ↔ scoring-runner
numbers agree), but are **~2–3× slower** than the hand-tuned static-CUDA references
(`solution/cuda/kernel.cu` v17 decode, v7 prefill) on the `submission-gdn-v7-v17`
branch of `fullagent`.

Measured baseline (Modal B200, `flashinfer/flashinfer-ci-cu132` @
`sha256:87e3b10d...`):

| kernel | version | mean speedup vs Python ref | latency |
|--------|---------|---------------------------:|--------:|
| decode (30 wl) | static CUDA v17 | 457× | 0.012 ms |
| decode (30 wl) | CuTe v1 (current) | 172× | 0.035 ms |
| prefill (5 smallest wl) | static CUDA v7 | ~75× (extrapolated) | 0.022 ms |
| prefill (5 smallest wl) | CuTe v1 (current) | 47× | 0.053 ms |

The fix is a direct algorithmic port — every optimization that lives in the static-
CUDA reference has a one-to-one CuTe-DSL idiom already proven in flashinfer's own
`gdn_decode_pretranspose.py`. The dominant single change is grid re-layout: 4-way
V-split with 32 threads/block (1 warp), each lane owning 32 V-rows held in registers
as a `(32, 4)` `float4`-shaped tile — matching v17/v7's `float4 sr[kVecsPerRow]`. The
remaining five levers (warp-reduce qk, `cp.async.cg` double-buffered K/Q, in-place
`sr *= g`, `out = qs + δ·qk`, compile-time specialization) compose on top of that
refactor.

Success = **≥ 90% of static-CUDA perf** on both tracks while remaining on the
Python+CuTe surface (non-negotiable — build-surface drift is exactly what caused
Modal ↔ scoring divergence in the first place).

## Problem Frame

We traded **2–3× of throughput for a surface that Modal and the scoring runner agree
on**. That tradeoff was correct (scoring matters more than Modal numbers) but the
per-call perf gap is recoverable without touching the surface: the CuTe DSL exposes
every primitive we used in the static-CUDA kernel (`cp.async.cg`, warp shuffles,
register tiles, constexpr specialization). Our v1 just hasn't wired them up yet — it
targets the simplest correct layout (`block=(D=128,1,1)`, 1 thread/row) rather than
the v17/v7 layout (`block=(32,1,1)`, 32 rows/thread in a vec-tile).

The root-cause chain:

1. **1 thread per V-row** → only 128 threads per block → 1 block per (batch, v_head) →
   under-utilized SMs on small batches, plus the inner per-row compute is a single
   128-entry flat register array which forces large unroll and spilling.
2. **Cross-warp smem reductions** (4-warp block needs a block barrier to finalize
   `qk` scalar) → two `cute.arch.barrier()` per iteration that disappear in v17.
3. **Synchronous K/Q load** per token in prefill → no memory/compute overlap → v7's
   `cp.async.cg` pipeline simply isn't present.
4. **Flat `sr[D]` rmem layout** → compiler can't fuse the inner 32×4 pattern into
   `float4` moves, so register pressure stays high and the `ldg.128` / `ld.shared.128`
   opportunities on state load/writeback are missed.

Every one of these is addressable without changing surface.

## Requirements Trace

- **R1.** Stay on `language = "python"` + `entry_point = "msinfer_entry.py::run[_prefill]"` + `cute.compile(..., options="--enable-tvm-ffi --gpu-arch=sm_100a --opt-level=3")`. No reversion to static `.cu`. (origin: `docs/runtime-pin.md`)
- **R2.** All 30 decode workloads (batch ∈ {1,4,8,16}) PASS on Modal B200, within the contest's correctness tolerance (max abs-err ≤ 1e-6, max rel-err ≤ 1e-1).
- **R3.** All 5 smallest prefill workloads (seq_len ≤ 30) PASS. Stretch: all 20 smallest (seq_len ≤ 35) PASS.
- **R4.** Decode mean speedup ≥ 405× (90% of v17's 457× on the same 30-workload set).
- **R5.** Prefill mean speedup on the 5-workload set ≥ 68× (90% of v7 extrapolated).
- **R6.** `ptxas --verbose` after port: **≤ 72 spill stores, ≤ 72 spill loads, ≤ 1 barrier, ≤ 1280 B smem** per kernel (relaxed 5% over v7's 68/68/0/1024 budget to allow for DSL lowering overhead).
- **R7.** Runtime pins preserved: `nvidia-cutlass-dsl==4.4.2`, `cuda-python==13.2.0`, `apache-tvm-ffi==0.1.10`, image digest unchanged.

## Scope Boundaries

- **In scope:** Rewriting the kernel bodies of `_gdn_decode_dev` and `_gdn_prefill_dev` in `solution/python/msinfer_entry.py`, plus the surrounding host-side launch code, cache key, and DPS dispatch.
- **Not in scope — explicit non-goals:**
  - Blackwell-native `tcgen05.*` / 5th-gen MMA / TMA paths (flashinfer's `blackwell_prefill/gdn.py` pattern). GDN's recurrence is non-linear — tensor cores don't apply. State is fp32 per contest, so no bf16-state compression path either.
  - Parallel-scan restructuring. Blocked by the `delta` ↔ `state` dependency; already ruled out in the v7 optimization report.
  - Changing the contest state layout `[B, HV, D, D]` k-last.
  - Modifying `config.toml`, `scripts/pack_solution.py`, or `scripts/run_modal.py` (already pinned in prior commits).
  - `run_qk_l2norm` variants — contest definition doesn't use the norm axis.

### Deferred to Separate Tasks

- **Docs update for `docs/gdn-optimization-report.md`**: append CuTe v2 section once numbers land. Done as the last commit of this plan, but tracked separately in case we land the kernel in stages.
- **`docs/solutions/` learnings entry** capturing the Python+CuTe ↔ static-CUDA idiom mapping. Worth having for the next time someone hits this mismatch, but not blocking the perf work.

## Context & Research

### Relevant Code and Patterns

- `solution/python/msinfer_entry.py` — current CuTe v1 (decode `_gdn_decode_dev` lines ~84–170, prefill `_gdn_prefill_dev` lines ~180–300). This is the file being rewritten.
- `solution/cuda/kernel.cu` (on `submission-gdn-v7-v17` branch of `fullagent`, locally accessible via `git show fullagent/submission-gdn-v7-v17:solution/cuda/kernel.cu`) — **the oracle**. Decode v17 (lines ~84–170) and prefill v7 (lines ~183–320). Every optimization in this plan is already present here in CUDA form — the plan is strictly a CuTe-DSL lowering of these bodies.
- `flash/lib/python3.12/site-packages/flashinfer/gdn_kernels/gdn_decode_pretranspose.py` — **the DSL idiom source**. Key lines:
  - `tiled_copy_load = cute.make_tiled_copy_tv(copy_atom, thread_layout, val_layout)` — the `cp.async.cg` staging pattern.
  - `copy_atom = cute.make_copy_atom(cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), ..., num_bits_per_copy=128)` — L2-only 16-B cp.async, matches v7.
  - `cute.local_tile(sData, (1, vec_size, 1), (row+row_offset, lane_id, stage))` + `cute.autovec_copy(sData_tile, r_h)` — the vectorized inner-tile load.
  - `cute.arch.shuffle_sync_bfly(x, offset=offset, mask=-1, mask_and_clamp=31)` — butterfly warp reduce.
  - `cute.arch.cp_async_commit_group()` / `cp_async_wait_group(0)` + `cute.arch.barrier()` staging.
- `flash/lib/python3.12/site-packages/flashinfer/data/cutlass/examples/python/CuTeDSL/cute/tvm_ffi/ampere_gemm_with_fake_tensor.py` — reference for `cute.compile` with `--enable-tvm-ffi` + dynamic shapes via `make_fake_compact_tensor` / `sym_int32(divisibility=...)`.

### Institutional Learnings

- `docs/gdn-optimization-report.md` — full v2→v17/v7 optimization history, ptxas budgets, and the specific traps (sync double-buffer regresses; `__ldg` neutral; g-only pre-multiply neutral *alone* but wins when combined with `qs + δ·qk`).
- `docs/runtime-pin.md` — `--gpu-arch=sm_100a` is load-bearing. Missing the `a` suffix is exactly the class of regression to watch for; flag immediately if ptxas output changes arch.
- `docs/superpowers/reports/2026-04-17-{decode,prefill}-kernel-optimization.md` — iteration histories from the original static-CUDA optimization work. Useful as a diagnostic reference if the CuTe port reproduces a symptom earlier iterations already diagnosed.
- No `docs/solutions/` directory exists yet. Out of scope to create here, but noted in "Deferred to Separate Tasks" above.

### External References

Not pursued — local grounding is unusually strong (flashinfer ships canonical CuTe-DSL GDN idioms and the static-CUDA oracle is in our own repo). External research would add little.

## Key Technical Decisions

- **Port the v17/v7 layout verbatim, not a fresh design.** Thread/block shape, register tile geometry, mainloop sequence, warp-reduce-vs-smem choice — all copy v17/v7. Any deviation is a liability because the algebraic shape is already proven. The DSL port is a *syntactic translation*, not a design exercise.
- **Decode and prefill share a state-tile layout.** Both want `sr` as a `(32, 4)` register tile where outer axis = V-row tile owned by this lane and inner = vec_size-4 inside D. This maps cleanly to `cute.make_rmem_tensor(cute.make_layout((32, 4), stride=(4, 1)), Float32)` and lets `cute.autovec_copy` emit `ldg.128` on state load / `stg.128` on writeback.
- **Prefill's K/Q pipeline uses `cpasync.CopyG2SOp(cache_mode=LoadCacheMode.GLOBAL)` with `num_bits_per_copy=128`, `NUM_STAGES=2`.** Directly mirrors v7's `cp.async.cg.shared.global [%0], [%1], 16;` + double-buffer + `cp_async_wait_group(0)`. Contest K/Q are bf16 so 16 B = 8 bf16 per lane per copy, exactly v7.
- **Warp-reduce scalar `qk` via `shuffle_sync_bfly`, no smem.** With 32 threads/block (1 warp), the cross-warp smem ping-pong from v1 goes away entirely. Single `__syncwarp` equivalent (`cute.arch.barrier()` with 1 warp degenerates) instead of two block barriers.
- **Redundant per-lane `g` / `β` compute.** MUFU throughput is per-lane-SIMD so redundant expf/softplus is wall-clock-free. Avoids the lane-0 + `shuffle_sync(0)` broadcast. Already correct in v1; preserve post-refactor.
- **Keep `scale` as `Constexpr[float] = 1.0 / sqrt(128.0)`.** Contest always passes `scale=0.0` → resolved to the default. Compile-time constant eliminates a runtime multiply on the output write.
- **Cache key stays `(dtype, shape, stride) ×` tensors — `mark_layout_dynamic()` folds the batch axis.** One compile per unique (decode vs prefill) × GPU arch; not per batch size. Already correct in v1 and worth confirming post-refactor with a log print of `len(_CACHE)` after a multi-batch run.
- **Baseline must be recaptured on the exact same workload slice before/after each unit.** Perf claims in this plan are relative to v17/v7; the comparison is meaningful only if the workload slice is identical (`--max-workloads 30` for decode, `--max-workloads 5` for prefill, both sorted by `total_seq_len`).

## Open Questions

### Resolved During Planning

- **Q: Is the CuTe DSL capable of the same SASS v17 emits?** A: Yes — every primitive v17 uses (`cp.async.cg`, warp `__shfl_xor_sync`, register tile, block barrier, constexpr unroll) has a direct CuTe-DSL surface (`cpasync.CopyG2SOp`, `cute.arch.shuffle_sync_bfly`, `cute.make_rmem_tensor`, `cute.arch.barrier`, `cutlass.range_constexpr`).
- **Q: What's the target perf bar?** A: 90% of v17/v7 on identical workload slices. See R4–R6.
- **Q: Do we need to re-validate `sm_100a` after the port?** A: Yes — every commit in this plan captures ptxas output as verification; if the arch string drifts, we catch it there.

### Deferred to Implementation

- **Exact `make_rmem_tensor` layout shape/stride that avoids spill.** The pretranspose reference uses `cute.make_layout((vec_size,), stride=(1,))` per-row-group (vec_size=4) and tiles the 32 V-rows via `cute.local_tile(sData, (1, vec_size, 1), (row+row_offset, lane_id, stage))`. The equivalent rmem shape for our 32-rows-per-thread case is `(32, 4)` but the stride choice (`(4,1)` vs `(1,32)`) affects whether the compiler emits `float4` or scalar register moves. Measure both with `--keep-ptx`; the one with zero spill wins.
- **Whether `cutlass.Constexpr[int]` specialization of `HV=8`, `HQ=4`, `D=128` measurably changes SASS.** Our constants are already module-level Python ints, but threading them through the kernel signature as `Constexpr` may unlock additional unrolling. Defer — add only if ptxas shows a win.
- **Whether a third cp.async stage helps for very long sequences.** `NUM_STAGES=2` matches v7; `NUM_STAGES=3` costs 512 B extra smem and may hide more latency on seq_len > 1000. Defer until we have a workload-set that actually exercises that regime.
- **Whether the GVA-ratio indexing `qk_head = v_head // (HV//HQ)` triggers non-uniform warp divergence.** The indices are uniform within a warp (all 32 lanes share the same `v_head`), so this should be fine, but verify via `cuobjdump --dump-sass` on the first unit.

## High-Level Technical Design

> *This illustrates the intended approach and is directional guidance for review, not implementation specification. The implementing agent should treat it as context, not code to reproduce.*

The CuTe v2 decode kernel maps 1:1 onto v17's structure. Per-block work:

```text
grid  = (B · HV · kSplits=4, 1, 1)
block = (kWarpThreads=32, 1, 1)    # one warp
smem  = float s_q[128], s_k[128]   # 1024 B total, cooperative load
regs  = float4 sr[kVecsPerRow=32]  # each lane owns 32 V-rows; (32,4) tile

per block:
  load s_q, s_k cooperatively             # 32 threads × 4 elems each
  g, beta ← compute_gate(...)             # redundant per lane, MUFU parallel
  __syncwarp

  qk ← Σ_{i=tid..127 step 32} s_q[i]·s_k[i]
  qk ← warp_reduce_bfly(qk)               # every lane now has full qk

  # Fused pass 1: sr ← g·state, accumulate ov + qs
  for i in 0..31 (constexpr):
      tmp ← ldg(state_row[lane·32+i, i·4 : i·4+4])
      tmp ← g · tmp
      sr[i] ← tmp
      ov += s_k[b4+0..3] · tmp.{x,y,z,w}
      qs += s_q[b4+0..3] · tmp.{x,y,z,w}

  delta ← beta · (v_val − ov)
  out_acc ← qs + delta · qk
  out[lane·32+...] ← scale · out_acc

  # Pass 2: sr ← sr + k·delta (now = new_state)
  for i in 0..31 (constexpr):
      sr[i] += s_k[b4+0..3] · delta
      new_state_row[...] ← sr[i]
```

Prefill is the same shape with an outer token loop and `cp.async.cg`-double-buffered K/Q:

```text
grid  = (N_seqs · HV · kSplits, 1, 1)
block = (kWarpThreads=32, 1, 1)
smem  = bf16 s_k[2][128], bf16 s_q[2][128]   # double-buffer, 1024 B total

  initial cp.async issue + wait
  for t in seq_start..seq_end (Int32 cast):
      if t+1 < end:
          issue cp.async for t+1 into [1-buf]
          commit_group
      g, beta ← compute_gate(t)
      ov, qs ← Pass 1 over sr with current buffer
      qk ← warp_reduce over current buffer
      delta, out_acc ← scalars
      write output[t]
      sr ← sr + k·delta
      wait_group(0); __syncwarp
      buf ← 1 - buf
  write new_state
```

The key directional points for a reviewer:

1. `block=(32,1,1)` — **one warp only**. This is the single change that cascades.
2. `sr` is a `(32, 4)` register tile, not a `(128,)` flat array. `cute.autovec_copy` can then vectorize state load/store.
3. `qk` stays in registers the whole time — warp shuffle only, no smem.
4. Prefill's cp.async pipeline uses `cpasync.CopyG2SOp(cache_mode=LoadCacheMode.GLOBAL)` with `num_bits_per_copy=128` — the DSL-native equivalent of `cp.async.cg.shared.global`.
5. All hot loops are `cutlass.range_constexpr(32)` not `range`, so they fully unroll.

## Implementation Units

- [ ] **Unit 1: Baseline capture and ptxas snapshot**

**Goal:** Record the exact starting point so every subsequent unit can claim measured improvement (not a vibes-based "feels faster").

**Requirements:** R1, R7.

**Dependencies:** None.

**Files:**
- Modify: `docs/gdn-optimization-report.md` (append "CuTe v1 baseline" section)
- Run-only: `scripts/run_modal.py` — no code change, just recorded outputs

**Approach:**
- Run `modal run scripts/run_modal.py --max-workloads 30` with current `msinfer_entry.py` + `entry_point = "msinfer_entry.py::run"` (decode). Record all 30 speedups + latencies.
- Flip `entry_point = "::run_prefill"` and `definition = gdn_prefill_...`. Run `--max-workloads 5` (short seqs).
- Extract ptxas via `nvcc --ptxas-options=-v` on the shader the DSL emits (workflow: `keep_ptx=True` in compile options, then read `/tmp/...ptx`).
- Write the numbers verbatim into the report so subsequent units have a baseline table to diff against.

**Execution note:** Measurement-first. Do not touch kernel code in this unit.

**Test scenarios:**
- Happy path: 30-workload decode + 5-workload prefill both complete without `RUNTIME_ERROR`; numbers match the plan overview table ±5%.
- Edge case: confirm `_COMPILE_OPTS` still contains `--gpu-arch=sm_100a` via a Python `assert` added temporarily to the top of `msinfer_entry.py` (removed before commit) — catches silent arch drift.

**Verification:**
- Appended baseline table in `docs/gdn-optimization-report.md` showing per-batch decode speedups and per-workload prefill speedups.
- ptxas line showing `Used 255 registers, … spill stores, … spill loads, … barriers, … smem`.

---

- [ ] **Unit 2: Decode kernel — 4-way V-split + register-tile port**

**Goal:** Move decode from `block=(128,1,1)` / 1-row-per-thread to `block=(32,1,1)` / 32-rows-per-thread with `sr` as a `(32, 4)` register tile. Absorb the warp-reduce-only qk path. This is the dominant win — expected to close >70% of the gap on its own.

**Requirements:** R1, R2, R4, R6.

**Dependencies:** Unit 1 (baseline).

**Files:**
- Modify: `solution/python/msinfer_entry.py` — `_gdn_decode_dev`, `_gdn_decode_jit`, `run`.
- No test file (runtime verification via Modal — there is no unit-test harness for CuTe DSL kernels in this repo, by design).

**Approach:**
- Change grid: `bid_x // kSplits // HV` = batch, `(bid_x / kSplits) % HV` = v_head, `bid_x % kSplits` = split, `row_base = split * 32`.
- Change block: `block=(32, 1, 1)`. `tid` now lane_id 0..31.
- Reshape `sr`:
  ```text
  sr = cute.make_rmem_tensor(cute.make_layout((32, 4), stride=(4, 1)), cutlass.Float32)
  ```
  (outer = 32 rows this lane owns, inner = vec_size-4 within D.)
- Cooperative smem load of `s_q`, `s_k`: each of 32 lanes loads 4 bf16 → float. Inner loop `for j in range_constexpr(4): s_q[lane + j*32] = Float32(q[batch, 0, qk_head, lane + j*32])`.
- `qk` via warp-only reduce (butterfly `shuffle_sync_bfly`). Result lives in every lane. Remove the 4-warp smem ping-pong path entirely — delete `sPartials` and `sScalar`.
- Two-pass mainloop iterates `for i in range_constexpr(32): for j in range_constexpr(4):` over `sr[i]`-shaped `float4`s. Mirror v17 lines 141–166 (pre-scale in pass 1, `sr += k·delta` in pass 2).
- `out = scale · (qs + delta · qk)` scalar close-up unchanged.
- Grid launch in `_gdn_decode_jit`: `grid=(B * HV * kSplits, 1, 1)`, `block=(kWarpThreads, 1, 1)`.

**Execution note:** Capture `ptxas` + full 30-workload run after the port, before moving to prefill. If spill > 132 B or barriers > 0, stop and diagnose before layering more changes.

**Technical design:**

```text
bid = block_idx.x
split     = bid %  4
v_head    = (bid / 4) % 8
batch_idx = bid / (4 * 8)
row_base  = split * 32   # this lane's first V-row

# cooperative smem load — 32 lanes × 4 elements = 128
for j in range_constexpr(4):
    idx = tid + j*32
    s_q[idx] = Float32(q[batch_idx, 0, qk_head, idx])
    s_k[idx] = Float32(k[batch_idx, 0, qk_head, idx])
__syncwarp

# warp-only qk reduce — every lane ends up with full qk
qk = 0.0
for j in range_constexpr(4):
    idx = tid + j*32
    qk += s_q[idx] * s_k[idx]
for off in [16, 8, 4, 2, 1]:
    qk += shuffle_sync_bfly(qk, offset=off, mask=-1, mask_and_clamp=31)

# mainloop — 32 (rows) × 4 (inner) = 128 state entries per lane
for i in range_constexpr(32):
    row = row_base + i  # which V-row inside the 128×128 state
    tmp = ldg_float4(state[batch, v_head, row, i*4 : i*4+4])  # as float4
    for c in range_constexpr(4):
        tmp[c] = tmp[c] * g
    sr[i] = tmp
    for c in range_constexpr(4):
        ov += s_k[i*4 + c] * tmp[c]
        qs += s_q[i*4 + c] * tmp[c]

delta = beta * (Float32(v[batch, 0, v_head, row_base + tid]) - ov)
out_acc = qs + delta * qk

for i in range_constexpr(32):
    for c in range_constexpr(4):
        sr[i][c] += s_k[i*4 + c] * delta
    new_state[batch, v_head, row_base + i, i*4 : i*4+4] = sr[i]

out[batch, 0, v_head, row_base + tid] = BFloat16(scale * out_acc)
```

Note: the `row` per iteration is **not** `row_base + tid` — each lane owns 32 rows, so iteration `i` handles `row_base + i`. `tid` selects which V-row subset we belong to within the split. This is the biggest readability hazard of the refactor; name the indices explicitly.

**Patterns to follow:**
- `solution/cuda/kernel.cu` v17 decode lines 84–170 on `fullagent/submission-gdn-v7-v17` — literal algorithmic oracle.
- `flash/lib/python3.12/site-packages/flashinfer/gdn_kernels/gdn_decode_pretranspose.py` lines 76–140, 240–340 — DSL idioms for `make_rmem_tensor`, `autovec_copy`, `shuffle_sync_bfly`.

**Test scenarios:**
- Happy path — batch=1 × 10 workloads: PASSED, speedup ≥ 80× per workload (v17 delivers ~103× here).
- Happy path — batch=16 × 5 workloads: PASSED, speedup ≥ 900× per workload (v17 delivers ~1040× here).
- Edge case — batch=64 workload (if present in the 30 sampled): PASSED without layout-dynamic recompile (confirm `len(_CACHE["decode"]) == 1` at end of run).
- Error path — correctness drift: `abs_err ≤ 1e-6`, `rel_err ≤ 1e-1` on every workload. If either exceeds tolerance, the `row_base + i` indexing is the prime suspect.
- Integration — end-to-end via `flashinfer-bench`'s evaluator (not just a Python-level unit test), because mocks can't prove TVM-FFI marshalling and DPS allocation parity.

**Verification:**
- 30/30 decode workloads PASSED on Modal B200.
- Mean decode speedup ≥ 405× (R4).
- ptxas: ≤ 72 spill stores, ≤ 72 spill loads, 0 barriers, ≤ 1280 B smem (R6).

---

- [ ] **Unit 3: Prefill kernel — 4-way V-split port (synchronous K/Q)**

**Goal:** Same grid/block refactor as Unit 2, applied to `_gdn_prefill_dev`. No `cp.async` yet — synchronous smem load per token, matching v3-equivalent synchronous-prefill. This isolates the layout-refactor win from the async-pipeline win and keeps bisection tight if something regresses.

**Requirements:** R1, R3, R5, R6.

**Dependencies:** Unit 2 (shares the register-tile layout decision, and we want the decode path green first as a smoke signal that `make_rmem_tensor((32, 4), (4, 1))` works).

**Files:**
- Modify: `solution/python/msinfer_entry.py` — `_gdn_prefill_dev`, `_gdn_prefill_jit`, `run_prefill`.

**Approach:**
- Same split/v_head/batch/row_base derivation as Unit 2 but with `seq_idx = bid_x / (kSplits * HV)` instead of batch.
- Keep the Int32 cast on `seq_start`/`seq_end` (already present).
- Token loop: `for t in range(seq_start, seq_end)` — no cp.async yet, just straight scalar smem load of K[t], Q[t] cooperatively. Each of 32 lanes loads 4 bf16 → float into `s_q`/`s_k` (single-buffered).
- `cute.arch.barrier()` / `__syncwarp` between the smem load and the mainloop body.
- Pass 1 / pass 2 inner loops identical shape to Unit 2 (with `row = row_base + i` naming).
- Redundant per-lane `g`, `β` compute inside the token loop.
- After the outer loop, persist `sr` into `new_state[seq_idx, v_head, row_base + i, i*4+c]` with `cute.autovec_copy` if available (else `range_constexpr` scalar store).

**Execution note:** Keep synchronous K/Q load deliberately. Unit 4 layers cp.async on top; bisection gets easier if the layout port alone is already green.

**Patterns to follow:**
- `solution/cuda/kernel.cu` v7 prefill on `fullagent/submission-gdn-v7-v17` — algorithmic oracle, ignoring the cp.async lines for now.
- `flash/lib/python3.12/site-packages/flashinfer/gdn_kernels/gdn_decode_pretranspose.py` lines 150–250 for state-tile layout.

**Test scenarios:**
- Happy path — 5 smallest prefill workloads (seq_len ≤ 30): all PASSED, mean speedup ≥ 50× (R5 allowing slack for the still-missing cp.async).
- Edge case — `seq_len = 1` degenerate case (decode-shaped prefill, if present in workloads): PASSED, no infinite loop on the outer token loop bound.
- Edge case — `cu_seqlens` with zero-length sequences: the existing `if seq_end <= seq_start: return` guard is preserved.
- Error path — correctness across all 5 workloads: `abs_err ≤ 1e-6`, `rel_err ≤ 1e-1`.
- Integration — `len(_CACHE["prefill"]) == 1` after the 5-workload run (confirm dynamic-layout compile works across seq lengths).

**Verification:**
- 5/5 prefill workloads PASSED.
- Mean prefill speedup ≥ 50× on this workload slice (weaker than R5 because cp.async comes in Unit 4).
- ptxas: 0 barriers (the smem fence becomes `__syncwarp` / `cute.arch.barrier()` with one warp).

---

- [ ] **Unit 4: Prefill kernel — cp.async.cg double-buffered K/Q pipeline**

**Goal:** Layer `cpasync.CopyG2SOp` + double-buffered `s_k[2]` / `s_q[2]` onto Unit 3. Match v7's async pipeline exactly. Expected additional win: ~15–25% on short seqs, larger on long.

**Requirements:** R1, R3, R5, R6.

**Dependencies:** Unit 3 (needs the 4-way V-split port to be green before adding staging).

**Files:**
- Modify: `solution/python/msinfer_entry.py` — `_gdn_prefill_dev`, `_gdn_prefill_jit`.

**Approach:**
- Define a `cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL)` copy atom with `cutlass.BFloat16` and `num_bits_per_copy=128` (matches v7's 16-B cp.async.cg, 8 bf16/lane).
- Thread layout: `cute.make_layout((32,), stride=(1,))` — one warp, one lane per contiguous tile slot. Val layout: `(1, 8)` — 8 bf16 per copy. Alternatively: `thread_layout=(2, 16)` splitting K (16 lanes) + Q (16 lanes), matching v7's "first 16 lanes load K, next 16 load Q" split. Decide by ptxas: whichever avoids a bank conflict on smem writes.
- `s_k_bf16 = smem.allocate_tensor(BFloat16, cute.make_layout((2, 128), stride=(128, 1)), 16)` (128 bf16 × 2 stages × 2 B = 512 B; same for Q → 1024 B total smem = v7 exact).
- Pre-loop: `issue cp.async for seq_start; commit_group; wait_group(0); __syncwarp`.
- Inside token loop:
  ```
  if t + 1 < seq_end:
      issue cp.async for t+1 into buf=1-buf
      cute.arch.cp_async_commit_group()
  # compute using current buf ...
  cute.arch.cp_async_wait_group(0)
  __syncwarp
  buf = 1 - buf
  ```
- The inner mainloop reads from `s_k_bf16[buf][...]` / `s_q_bf16[buf][...]` and converts bf16 → float32 inline via `Float32(...)`. Keep `vec_size=4` shape so `cute.autovec_copy` can emit `ld.shared.128`.

**Execution note:** Validate that `cpasync.CopyG2SOp` actually lowers to `cp.async.cg.shared.global` PTX — grep the `--keep-ptx` output for `cp.async.cg`. If it emits `cp.async.ca` (L1-cached) we've regressed to a worse cache mode; specify `cache_mode=LoadCacheMode.GLOBAL` explicitly (this is the whole point of v7's `.cg`).

**Patterns to follow:**
- `solution/cuda/kernel.cu` v7 prefill — async pipeline structure (double-buffer, commit/wait group, `__syncwarp` sequencing).
- `flash/lib/python3.12/site-packages/flashinfer/gdn_kernels/gdn_decode_pretranspose.py` lines 680–750 — the `cpasync.CopyG2SOp(cache_mode=...GLOBAL)` + `make_tiled_copy_tv` + `cp_async_commit_group` / `wait_group` sequence.

**Test scenarios:**
- Happy path — all 5 prefill workloads: PASSED, mean speedup improved over Unit 3's number.
- Happy path — 20-workload prefill run (seq_len ≤ 35, stretch R3): all PASSED, mean speedup ≥ 68× (R5).
- Edge case — `seq_len = 1` (no `t+1` to prefetch): confirm the `if t+1 < seq_end` guard still holds and no spurious cp.async issues.
- Error path — `cp.async` wait-group ordering drift: if wait_group comes before commit_group due to a refactor slip, kernel will either hang (timeout RUNTIME_ERROR) or read garbage. Correctness check catches this.
- Integration — ptxas must show `cp.async.cg` (not `cp.async.ca`) in the emitted SASS / PTX.

**Verification:**
- 5/5 short + 20/20 stretch prefill workloads PASSED.
- Mean prefill speedup ≥ 68× (R5).
- PTX contains `cp.async.cg.shared.global.L2::128B` or equivalent; no `cp.async.ca`.
- ptxas: ≤ 72 spill stores / loads, ≤ 1 barrier, ≤ 1280 B smem (R6).

---

- [ ] **Unit 5: Compile-time specialization + cleanup**

**Goal:** Thread `HV`, `HQ`, `HK`, `D`, `kSplits`, `kVecsPerRow` as `cutlass.Constexpr[int]` into the `@cute.kernel` signatures and replace any remaining `range` with `range_constexpr`. Close the last few percent of gap by letting the DSL constant-fold all loop bounds, divmods, and indexing arithmetic.

**Requirements:** R1, R4, R5, R6.

**Dependencies:** Unit 4.

**Files:**
- Modify: `solution/python/msinfer_entry.py` — kernel signatures, `_gdn_decode_jit`, `_gdn_prefill_jit`.

**Approach:**
- Move `HQ`, `HK`, `HV`, `D`, `kSplits` from module-level Python ints to kernel-level `Constexpr` parameters, passed from the `@cute.jit` launcher.
- Convert any remaining `range(...)` in the kernel body to `cutlass.range_constexpr(...)` — grep for it; only the outer token loop in prefill (`for t in range(seq_start, seq_end)`) stays dynamic (bounds are runtime).
- Collapse any scalar reductions that were left over from the v1 4-warp layout (`sPartials`, `sScalar` should already be gone after Unit 2 / Unit 3; sanity-check).
- Delete any dead code paths created by the layout port.
- Collapse redundant casts: `cutlass.Float32(...)` on an already-f32 value is free at runtime but clutters the source.

**Execution note:** Diff ptxas before and after this unit. If spill / barrier / smem don't move, the constexpr specialization didn't help and the refactor is cosmetic-only — still land it for readability but note the neutral result.

**Test scenarios:**
- Happy path — full decode 30-wl + prefill 20-wl re-run: all PASSED, mean speedups within ±2% of the Unit 4 snapshot (specialization should not regress correctness).
- Edge case — cold-cache first call (recompile): confirm `cute.compile` still succeeds with the new `Constexpr` parameters and `mark_layout_dynamic()` still folds the batch axis.
- Error path — if `Constexpr[int]` accidentally captures a Python value that varies between calls, the compiled kernel will be silently stale. Verify by asserting `_CACHE` size stays at 2 (one decode + one prefill) after running all 50 workloads.

**Verification:**
- Final ptxas snapshot within R6 budget.
- Mean decode speedup ≥ 405× (R4); mean prefill speedup ≥ 68× (R5).
- `len(_CACHE) == 2` after running all 50 workloads end-to-end.

---

- [ ] **Unit 6: Docs update, commit, push to `fullagent/opt_v2`**

**Goal:** Make the perf work discoverable and reviewable. Land it on the same branch as the rest of the CuTe surface work.

**Requirements:** R1–R7 (full roll-up).

**Dependencies:** Units 1–5.

**Files:**
- Modify: `docs/gdn-optimization-report.md` — append CuTe v2 section mirroring the existing v7/v17 section structure (algebra, ptxas diff, measured speedups table, decisions, deferred).
- Modify: `docs/runtime-pin.md` — bump the date stamp if any pinned version changed during implementation. (Expected: no change.)
- No change: `config.toml`, `scripts/pack_solution.py`, `scripts/run_modal.py`.

**Approach:**
- Mirror the v7/v17 "Effective optimizations" + "Still ineffective / not applied" tables, adapted for the CuTe v2 port.
- Include the ptxas table showing v1 → v2 diff alongside the existing v16/v17/v7 table.
- Include the measured-speedup table (decode per-batch, prefill per-workload).
- Reference the `fullagent` PR (existing `submission-gdn-v7-v17` is CUDA; this lands on `opt_v2` and extends the existing Python+CuTe path).
- Commit message (conventional-commit, multi-line, Co-Authored-By Claude Opus 4.7 per `CLAUDE.md`): `perf(cute): 4-way V-split port closes static-CUDA gap on Blackwell`.
- `git push fullagent opt_v2`.

**Test scenarios:**
- Happy path — `git log --oneline opt_v2` shows the new commit with the expected message shape; `git push` returns successful.
- Edge case — docs render cleanly (no broken links, no truncated tables).

**Verification:**
- Commit visible on `fullagent/opt_v2`.
- `docs/gdn-optimization-report.md` shows CuTe v2 section with final numbers.
- All units' checkboxes marked complete.

## System-Wide Impact

- **Interaction graph:** The Python+CuTe kernel is called by the `flashinfer_bench.compile.builders.python_builder.PythonBuilder`-loaded `Runnable` on the Modal evaluator's `PersistentRunner` worker. The worker redirects stdio to a log file surfaced via `Evaluation.log` (already wired in `scripts/run_modal.py`). Kernel-internal exceptions propagate via our `_dispatch` try/except to stdout → log → `Evaluation.log` → `run_modal.print_results`. No change to this chain.
- **Error propagation:** A CuTe compile error or runtime assertion raises during `run`/`run_prefill` and is captured as a `RUNTIME_ERROR` status with the traceback inlined into `Evaluation.log`. The existing log-tail surfacing is sufficient. If we introduce new failure modes (e.g., cp.async wait-group mismatch on seq_len=1), they'll show up through the same path.
- **State lifecycle risks:** None — the kernel is stateless between calls. `_CACHE` is module-level and only grows. Verify its size doesn't balloon (Unit 5).
- **API surface parity:** `run` (decode) and `run_prefill` (prefill) signatures are frozen by the contest's DPS definition. The refactor must preserve them byte-for-byte. No caller-facing change.
- **Integration coverage:** End-to-end Modal runs are the only credible verification. Python-level unit tests can't prove TVM-FFI marshalling or DPS output aliasing. Every unit includes Modal bench as its verification.
- **Unchanged invariants:** `config.toml` (`language=python`, `entry_point`, `dependencies`), `_COMPILE_OPTS` string (specifically `--gpu-arch=sm_100a`), contest tensor shapes/dtypes, `destination_passing_style=true`, Modal image digest. Any accidental drift is a scoring-parity regression — catch via the assertion in Unit 1.

## Risks & Dependencies

| Risk | Likelihood | Impact | Mitigation |
|------|-----------:|-------:|-----------|
| `cute.make_rmem_tensor((32, 4), stride=(4, 1))` doesn't vectorize, compiler emits scalar moves → no spill reduction | Med | High — kills Unit 2's primary win | Unit 2 captures ptxas before and after; if spill doesn't drop, test `(32, 4), stride=(1, 32)` and `(4, 32), stride=(32, 1)`. Ultimately, worst case fall back to `float4 sr[32]` struct-of-arrays-ish layout — still closes most of the gap. |
| `cpasync.CopyG2SOp` silently lowers to `cp.async.ca` instead of `cp.async.cg` | Low | High — re-introduces the L1-cached path v7 explicitly rejected | Unit 4 grep-verifies `cp.async.cg` in `--keep-ptx` output. If wrong, raise `LoadCacheMode.GLOBAL` explicitly or fall back to inline-PTX `asm volatile` via a `cute.arch` hook. |
| DSL's dynamic-range loop Int32-requirement bites again in a subtle place (see prior `seq_start` bug) | Med | Low — fast to diagnose via `Evaluation.log` | Every hot loop in the refactor uses `range_constexpr`; the only dynamic range is the prefill token loop whose bounds are already cast to Int32 in the current file. Unit 3 test scenarios explicitly include `seq_len=1` to catch degenerate-bound issues. |
| `--gpu-arch=sm_100a` gets dropped or clobbered during refactoring of `_COMPILE_OPTS` | Low | Critical — silently reverts to the exact regression the pin was put in to prevent | Unit 1 adds a temporary `assert "sm_100a" in _COMPILE_OPTS` at module top; Unit 6 keeps the pin intact; CI-equivalent is the ptxas log showing Blackwell-native instructions. |
| Register pressure explodes with `Constexpr` specialization (Unit 5), spilling more than the v1 baseline | Low | Medium — negates the specialization | Unit 5 execution note: diff ptxas and roll back specialization if spill increases. |
| `_CACHE` key miscomputes post-`Constexpr` specialization → silent stale-compile | Low | Medium | Unit 5 test scenario: `len(_CACHE) == 2` assertion after all-workloads run. |
| Long-seq prefill (seq_len > 1000) overflows the warp-reg state cycle time vs cp.async latency | Low | Low | Explicitly out of scope — we only bench up to seq_len=35 currently. If the contest's long-seq workloads matter, revisit with `NUM_STAGES=3`. |
| PR noise — the v17/v7 static-CUDA PR (`submission-gdn-v7-v17`) still open on `fullagent/main` | Low | Low — diverges if reviewer merges the wrong one | Unit 6 docs update explicitly clarifies that `fullagent/opt_v2` is the live scoring branch; the `submission-gdn-v7-v17` PR is reference-only now. |

## Documentation / Operational Notes

- `docs/gdn-optimization-report.md` gets a CuTe v2 section (Unit 6).
- `docs/runtime-pin.md` date-stamp refresh if any pinned version changed (expected: no change).
- `CLAUDE.md` unchanged — the existing guidance ("no emojis", DPS true, prefer editing existing files, don't add explanatory comments) remains correct.
- No operational rollout to worry about — this is a single-file solution that gets repacked on submission.

## Sources & References

- **Origin document:** None. Planning from the user's direct request and the existing repo state (fullagent/opt_v2 @ `7195a44`).
- **Oracle:** `solution/cuda/kernel.cu` on `fullagent/submission-gdn-v7-v17` (v17 decode + v7 prefill).
- **DSL idiom source:** `flash/lib/python3.12/site-packages/flashinfer/gdn_kernels/gdn_decode_pretranspose.py`.
- **Current implementation:** `solution/python/msinfer_entry.py` on `fullagent/opt_v2` @ `7195a44`.
- **Optimization history:** `docs/gdn-optimization-report.md`.
- **Runtime pin:** `docs/runtime-pin.md`.
- **Measurement tooling:** `scripts/run_modal.py`, `scripts/pack_solution.py`.
- **Prior iteration reports:** `docs/superpowers/reports/2026-04-17-{decode,prefill}-kernel-optimization.md`.
