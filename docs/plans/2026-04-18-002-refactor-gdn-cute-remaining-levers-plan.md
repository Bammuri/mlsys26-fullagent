---
title: "perf: GDN CuTe remaining optimization levers (post-v4)"
type: refactor
status: active
date: 2026-04-18
---

# perf: GDN CuTe remaining optimization levers (post-v4)

## Overview

Current CuTe v4 (`solution/python/msinfer_entry.py` on `fullagent/opt_v2` @ `a6048ce`) closed a significant portion of the gap to static-CUDA v17/v7, but is still:

- **Decode: ~48%** of static-CUDA v17 on kernel latency (0.025 vs 0.012 ms)
- **Prefill: ~84%** of static-CUDA v7 on kernel latency for short seqs (0.036 vs 0.030 ms)

Prior plan `docs/plans/2026-04-18-001-refactor-gdn-cute-v17-gap-closing-plan.md` executed Units 1-3 (baseline, decode V-split, prefill V-split) and partially Unit 5 (state vectorization). Units 4 (prefill `cp.async.cg`) and 6 (constexpr) never landed — Unit 4 was deferred as marginal on the short-seq bench we had, Unit 6 as "probably neutral".

This plan enumerates the **remaining levers**, prioritized by expected impact-to-risk ratio, and sequences them with measurement instrumentation up front so each unit's effect is visible rather than lost in Modal's run-to-run variance (currently 30–70% on the Python reference).

The "what else can we try" breaks into three phases:

- **Phase A — instrumentation & known-high-confidence refactors**: get ptxas/SASS visibility, reshape `sr` for `float4` register moves, switch K/Q smem to bf16 with vectorized loads and inline conversion.
- **Phase B — prefill-specific (long-seq)**: extend the workload bench past seq_len=30 and land the `cp.async.cg` double-buffered pipeline. cp.async's actual payoff only shows on seqs ≥~200; we currently have no data there.
- **Phase C — cleanup**: `Constexpr` specialization sweep.

Success = **≥ 70% of static-CUDA on both tracks** while keeping the Python+CuTe+TVM-FFI surface pinned (non-negotiable — this is the team-verified scoring path).

## Problem Frame

Three concrete mechanical reasons the CuTe v4 trails v17/v7, all recoverable without leaving the DSL:

1. **`sr` as flat 128-fp32 register array.** Inner loops are `for c in range_constexpr(4): sr[i*4 + c]`. The DSL may or may not be promoting these to `float4` register moves. A 2D `(32, 4)` rmem layout with `stride=(4,1)` makes the vec-tile shape explicit and matches `cute.autovec_copy`'s expected shape.
2. **K/Q smem held as fp32, built via scalar cooperative loads.** Each lane does `sQ[tid+j*32] = Float32(q[batch,0,qk_head,tid+j*32])` in 4 unrolled iterations — 4 gmem bf16 reads + 4 casts + 4 smem fp32 writes per lane per token. `v7` CUDA uses bf16 smem + inline conversion in the inner loop, halving smem footprint and exposing the load path for `cp.async.cg` / vectorized `autovec_copy`.
3. **Prefill has no memory/compute overlap.** Every token's K/Q load fully completes before compute starts. Static v7 uses double-buffered `cp.async.cg` + `cp_async_commit_group` / `wait_group(0)` to overlap load of `t+1` with compute of `t`. On short sequences (seq ≤ 30) this is ~5% marginal; on seq ≥ 200 it's the dominant lever.

Additional concerns that shape the plan but aren't levers themselves:

4. **We have no PTX/SASS visibility.** `--keep-ptx` is supported by `cute.compile` but not wired in. Every decision below — "did reshape actually emit `float4` moves?", "did `cp.async.cg` actually land as `.cg` not `.ca`?" — should be validated by looking at emitted SASS, not guessed from latency deltas.
5. **Modal reference-latency variance is 30–70% run-to-run.** Speedup numbers are unreliable; kernel latency is the honest metric. The bench harness should expose `reference_latency_ms` alongside kernel latency so we can detect when Python-ref cold-starts move and not over-interpret a single run.
6. **Our workload slice never exceeds seq_len=30.** Anything targeting long-seq prefill behavior is measuring against an empty set. Extending the bench to seq ≥ 200, 1000 is a measurement prerequisite, not a perf change.

## Requirements Trace

- **R1.** Stay on `language = "python"` + `entry_point = "msinfer_entry.py::run[_prefill]"` + `cute.compile(..., options="--enable-tvm-ffi --gpu-arch=sm_100a ...")`. No reversion to static `.cu`.
- **R2.** All 30 decode workloads + 5 baseline prefill workloads remain PASSED at contest tolerance (max abs-err ≤ 1e-6, max rel-err ≤ 1e-1) through every unit.
- **R3.** Additionally, 5+ long-seq prefill workloads (seq_len ≥ 200) PASS after Unit 5 extends the bench slice.
- **R4.** Decode mean kernel latency ≤ 0.017 ms across 30 workloads (≈ 70% of static-CUDA 0.012 ms).
- **R5.** Prefill mean kernel latency ≤ 0.043 ms on seq_len ≥ 200 (≈ 70% of static-CUDA v7 on equivalent workloads, once we measure them).
- **R6.** Per-unit `--keep-ptx` output captured in `docs/gdn-optimization-report.md` — every perf claim backed by a SASS diff, not just a latency number.
- **R7.** Runtime pins preserved: image digest `sha256:87e3b10d…`, `nvidia-cutlass-dsl==4.4.2`, `cuda-python==13.2.0`, `apache-tvm-ffi==0.1.10`.

## Scope Boundaries

- **In scope:** Kernel body, smem layout, and compile options inside `solution/python/msinfer_entry.py`; bench harness extensions in `scripts/run_modal.py`; per-unit PTX/SASS snapshots in `docs/gdn-optimization-report.md`.
- **Not in scope — explicit non-goals:**
  - Blackwell tensor core (`tcgen05.*`) for the K·state dot. GDN's non-linear recurrence doesn't fit tensor-core GEMM; even if it did, Blackwell fp32 MMA requires restructuring the delta-rule to be expressible as a GEMM sub-problem. Defer as its own exploration.
  - Multi-warp blocks (e.g. 2 warps × 64 threads). Changes the V-split layout and every per-row calculation downstream; out of scope for a "remaining levers" plan, would need its own plan.
  - Parallel-scan restructuring for prefill. Blocked by the `delta` ↔ `state` nonlinear dependency; already ruled out in both v7 and the 001 plan.
  - Fresh port of `flashinfer.gdn_kernels.gdn_decode_pretranspose` as the kernel body. Adapter surface (cu_seqlens, pool indexing, h0_source) is massive and diverges from our contest signature.

### Deferred to Separate Tasks

- **Tensor-core feasibility spike**: Isolated investigation in a sibling plan. Write up a one-page design, not a perf iteration.
- **Multi-warp-block design**: If Phase A + B land within perf budget, no need. If gap persists > 30%, re-evaluate with a dedicated plan.
- **`docs/solutions/` learnings entry**: Worth writing once Phase A + B numbers stabilize; not blocking.

## Context & Research

### Relevant Code and Patterns

- `solution/python/msinfer_entry.py` — current CuTe v4 kernel (on `fullagent/opt_v2` @ `a6048ce`). Specifically: `_gdn_decode_dev` lines ~84–170, `_gdn_prefill_dev` lines ~200–300, `_COMPILE_OPTS` at module top.
- `solution/cuda/kernel.cu` — static-CUDA v17/v7 oracle (same repo, on `submission-gdn-v7-v17` branch of `fullagent`). The algebraic shape and mainloop sequence we want to match in SASS.
- `flash/lib/python3.12/site-packages/flashinfer/gdn_kernels/gdn_decode_pretranspose.py` — canonical DSL idioms:
  - Lines 690-703: `cpasync.CopyG2SOp(cache_mode=LoadCacheMode.GLOBAL)` + `make_copy_atom(..., num_bits_per_copy=128)` + `make_tiled_copy_tv(copy_atom, thread_layout, val_layout)`.
  - Lines 172-176: `cute.copy(tiled_copy_load, thr_copy.partition_S(...), thr_copy.partition_D(...))` + `cute.arch.cp_async_commit_group()`.
  - Lines 287-291: `cute.local_tile(sData, (1, vec_size, 1), (row+row_offset, lane_id, stage))` + `cute.autovec_copy(sData_tile, r_h)` — vectorized gmem/smem → rmem copy with explicit per-lane tile shape.
  - Lines 240-244: `cute.arch.shuffle_sync_bfly(x, offset=offset, mask=-1, mask_and_clamp=31)` for warp reductions.
- `flash/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/base_dsl/compiler.py` — compiler option parser. `--keep-ptx`, `--keep-cubin`, `--dump-dir`, `--ptxas-options` are all valid. We're not passing any of them yet.

### Institutional Learnings

- `docs/gdn-optimization-report.md` — v17/v7 ptxas budget (255 regs / 68 B spill / 0 barriers / 1024 B smem for prefill) as the target envelope. Current CuTe v4 hasn't been measured against this yet.
- `docs/runtime-pin.md` — `--gpu-arch=sm_100a` is load-bearing for Blackwell SASS. Every unit in this plan must preserve it.
- `docs/plans/2026-04-18-001-refactor-gdn-cute-v17-gap-closing-plan.md` — prior plan; Units 4 and 6 were never executed. This plan subsumes them as Units 5 and 6 respectively, with clearer priority.
- Modal run-to-run variance empirically 30–70% on Python reference. Established this session — every perf claim in prior ce-work rounds got confused once until we switched to kernel latency as the comparator.

### External References

Not pursued — local grounding is strong. Flashinfer's own DSL GDN kernels + the DSL compiler source cover every primitive we need. External best-practices research would add little here.

## Key Technical Decisions

- **PTX/SASS visibility precedes all perf work.** Unit 1 wires `--keep-ptx --dump-dir=/tmp/cute-asm` into `_COMPILE_OPTS`, pulls the dump out of the Modal container, and captures a baseline SASS snapshot. Every subsequent unit is judged by *"did SASS change the way we expected?"* before *"did latency move?"* — because latency will drift 30-70% from ref variance alone.
- **Kernel latency, not speedup, is the perf comparator.** This plan commits to the honest metric. `scripts/run_modal.py` will be augmented to surface `reference_latency_ms` alongside kernel latency (Unit 5 harness work) so we can see when ref cold-starts have moved and not mistake it for a kernel regression.
- **`sr` as `(32, 4)` with stride `(4, 1)`** is the canonical reshape matching v7's `float4 sr[kVecsPerRow=32]`. Outer axis = V-row-group owned by this lane, inner = vec_size-4 inside D. This lines up with `cute.autovec_copy` on `(1,1,1,4)` state slices and should let the DSL emit `LDG.E.128` + vector register moves.
- **bf16 smem for K/Q** (not fp32). Half the smem footprint, unblocks `cpasync.CopyG2SOp` with `num_bits_per_copy=128` on a vectorized `(1, 8)` tile — matches static v7's `cp.async.cg` 16-byte transfer. Inner loop does `Float32(sK[c])` on the hot path; bf16→fp32 is a single PTX `cvt.f32.bf16` per use.
- **`cp.async.cg` not `cp.async.ca`.** `cache_mode=LoadCacheMode.GLOBAL` in `cpasync.CopyG2SOp` emits the L2-only `.cg` variant. This is load-bearing — v7 explicitly rejected `.ca` (L1-cached) after measuring a regression.
- **Long-seq bench slice**: seq_len ∈ {200, 500, 1000, 2000} — enough range to show where cp.async overtakes synchronous loading. Capped at 2000 because the 8192-token workloads hit the Python reference's quadratic wall (per `run_modal.py`'s `--max-seq-len` guard rationale).
- **Measurement discipline**: run each unit's bench **twice** back-to-back. If v(N) / v(N-1) kernel-latency delta is within ±15% between the two runs, the result is signal; outside that window, it's noise and we re-measure rather than commit.
- **`Constexpr` specialization is cleanup, not optimization.** Plan the unit but don't expect latency to move. The value is readability + potential constant folding in the DSL's MLIR pass.

## Open Questions

### Resolved During Planning

- **Q: Does `cute.compile` support `--keep-ptx`?** A: Yes — `compiler.py::parse_options` registers `--keep-ptx`, `--keep-cubin`, `--dump-dir`, `--ptxas-options`. Unit 1 just adds them to `_COMPILE_OPTS`.
- **Q: What's the shape of K/Q cp.async tiled_copy?** A: Mirror v7 exactly — 16 lanes load K (each 16 B = 8 bf16), 16 lanes load Q (each 16 B). In DSL terms: `thread_layout = make_layout((32,), stride=(1,))`, `val_layout = make_layout((1, 8))`, `num_bits_per_copy=128`. Use two tiled_copies (one for K, one for Q) or partition a unified tile — defer the exact choice to implementation.
- **Q: Which long-seq workloads does the trace set have?** A: From prior exploration of `mlsys26-contest/workloads/gdn/gdn_prefill_qk4_v8_d128_k_last.jsonl`, seq_lens go 6 → 8192 with ~30% of entries in the 200–4000 range. Unit 5 picks those via `--max-seq-len 4000` + `--min-seq-len 200` (the latter being a new flag this unit adds).

### Deferred to Implementation

- **Exact `make_rmem_tensor((32, 4), stride=...)` stride choice** — `(4, 1)` vs `(1, 32)`. First try `(4, 1)` (matches `[i, c]` access pattern with inner dim contiguous). If ptxas shows spill or scalar moves, try `(1, 32)`.
- **Single vs two `tiled_copy` for K and Q.** Cleaner as two (separate atoms, cleaner partition), but one unified atom + a split tile is fewer smem layout objects. Measure both via `--keep-ptx` if time permits.
- **Whether `Constexpr[int]` on `HV=8`, `HQ=4`, `D=128` measurably changes SASS.** Threading them through the kernel signature may unlock additional constant folding in the DSL's MLIR→PTX lowering. Defer; only land if ptxas shows a win.
- **NUM_STAGES = 2 vs 3 for cp.async prefill.** v7 uses 2. For seq ≥ 1000 a third stage could hide more latency at ~512 B extra smem. Keep 2 in Unit 6; if Unit 5 bench shows saturated memory pipeline at NUM_STAGES=2, revisit.
- **`--ptxas-options=-v` output parsing** — probably pipe to a small Python parser that extracts `registers`, `spill stores`, `spill loads`, `barriers`, `smem` into a table. Script is simple; implementation time lives in the actual unit.

## High-Level Technical Design

> *This illustrates the intended approach and is directional guidance for review, not implementation specification. The implementing agent should treat it as context, not code to reproduce.*

The plan is an instrumentation-first funnel:

```text
Unit 1 (diagnostics)
    │
    └── enables SASS-backed verdicts on every subsequent unit
        │
        ├── Unit 2 (sr reshape) ──┬── SASS: float4 moves? yes/no
        ├── Unit 3 (bf16 K/Q smem + autovec coop load) ─┬── SASS: ldg.128 on K/Q? yes/no
        │
        Unit 4 (long-seq bench + ref-latency surfacing in run_modal)
            │
            └── Unit 5 (prefill cp.async.cg) ─┬── SASS: cp.async.cg? yes/no
                                               └── bench: seq ≥ 200 shows expected win?
        │
        └── Unit 6 (Constexpr cleanup) — cosmetic / speculative
```

After Unit 1, the code-change units (2, 3, 5, 6) each land as a single commit on `fullagent/opt_v2` with the SASS diff inlined into the commit body. Bench-harness units (4) land earlier — they're measurement prerequisites.

The prefill cp.async pattern mirrors `gdn_decode_pretranspose.py` 170–280:

```text
# @cute.jit host side
copy_atom = make_copy_atom(
    CopyG2SOp(cache_mode=LoadCacheMode.GLOBAL), BFloat16, num_bits_per_copy=128)
tiled_copy_load = make_tiled_copy_tv(copy_atom, thread_layout, val_layout)

# @cute.kernel
thr_copy = tiled_copy_load.get_slice(tid)
# pre-loop: issue cp.async for seq_start
issue_load(buf=0, t=seq_start)
cp_async_commit_group(); cp_async_wait_group(0); sync_warp
for t in seq_start..seq_end:
    if t + 1 < seq_end:
        issue_load(1 - buf, t + 1)    # async, fire-and-forget
        cp_async_commit_group()
    compute from s_k_bf16[buf], s_q_bf16[buf]
    cp_async_wait_group(0); sync_warp
    buf = 1 - buf
```

## Implementation Units

- [ ] **Unit 1: PTX/SASS instrumentation (`--keep-ptx`)**

**Goal:** Make the DSL-emitted SASS visible so every subsequent unit's perf claim is grounded in an instruction-level diff rather than a noisy latency delta.

**Requirements:** R6.

**Dependencies:** None.

**Files:**
- Modify: `solution/python/msinfer_entry.py` — `_COMPILE_OPTS` string; add a debug-build branch.
- Modify: `scripts/run_modal.py` — expose a `dump_ptx: bool` entrypoint flag; have the modal function tar up `/tmp/cute-asm/` and return the tarball alongside the results dict.
- Modify: `docs/gdn-optimization-report.md` — append a "SASS baseline (CuTe v4)" section with ptxas stats table (regs / spills / barriers / smem) for decode and prefill.

**Approach:**
- Extend `_COMPILE_OPTS` with `--keep-ptx --dump-dir=/tmp/cute-asm --ptxas-options=-v` when an env var (e.g. `MSINFER_DUMP_SASS=1`) is set. In prod runs, do not set it (keeps steady-state compile times unchanged).
- `run_modal.py::run_benchmark` honors the env var, invokes the kernel once, tars `/tmp/cute-asm/`, inlines the ptxas verbose output into the returned dict.
- Capture PTX + cubin + ptxas-verbose for both `_gdn_decode_jit` and `_gdn_prefill_jit` on v4.
- Write the baseline numbers into the doc so Units 2/3/5 have a table row to diff against.

**Execution note:** Measurement-first. No kernel code change.

**Patterns to follow:**
- `cutlass.base_dsl.compiler.py::parse_options` for exact flag syntax.
- `scripts/run_modal.py` existing `log_tail` pattern — re-use the "return artifact from remote" mechanism.

**Test scenarios:**
- Happy path — `MSINFER_DUMP_SASS=1 modal run scripts/run_modal.py --max-workloads 1`: run completes PASSED, return dict includes `ptxas_stats` field with `{registers, spill_stores, spill_loads, barriers, smem_bytes}` for each compiled kernel, and the extracted ptx/cubin files written to a local `out/` dir.
- Edge case — env var unset: run completes normally with no SASS extraction (compile time unchanged; `_COMPILE_OPTS` stays identical).
- Error path — `--dump-dir` points at a non-existent path: log warning, continue (don't fail the run on instrumentation failure).

**Verification:**
- `docs/gdn-optimization-report.md` "SASS baseline (CuTe v4)" table populated with register count, spill bytes, barrier count, smem bytes for decode and prefill.
- `.ptx` and `.cubin` files for both kernels extractable from `/tmp/cute-asm/` via the harness (no manual docker cp needed).
- Non-instrumented runs (default) produce identical compile output to prior sessions (no regression from the new flags being present but unset).

---

- [ ] **Unit 2: Register tile reshape — `sr` as `(32, 4)` rmem layout**

**Goal:** Let the DSL emit `float4` register moves for the 128-fp32 per-thread state tile. Currently `sr` is a flat `(D,)` register array, which the MLIR lowering may or may not promote to vector moves; explicit 2D layout matches what `cute.autovec_copy` already expects and what v7's `float4 sr[32]` struct array is.

**Requirements:** R1, R2, R4, R6.

**Dependencies:** Unit 1 (to diff SASS).

**Files:**
- Modify: `solution/python/msinfer_entry.py` — `_gdn_decode_dev`, `_gdn_prefill_dev` inner loops.

**Approach:**
- Change
  ```
  sr = make_rmem_tensor(make_layout((D,), stride=(1,)), Float32)
  ...
  sr[i * 4 + c] = ...    # in range_constexpr(32) × range_constexpr(4)
  ```
  to
  ```
  sr = make_rmem_tensor(make_layout((32, 4), stride=(4, 1)), Float32)
  ...
  sr[i, c] = ...
  ```
- Same change in both decode and prefill.
- Also update `tmp` (the per-iter float4 scratch) to `make_layout((4,), stride=(1,))` if not already.
- Keep all outer `for i in range_constexpr(D // 4):` / `for c in range_constexpr(4):` loops identical in shape — only the indexing expression changes.

**Execution note:** Capture PTX before and after. The commit body should contain a SASS diff showing `LDS.128`/`STS.128` or `FMUL.F32.F32.F32.F32.V4` patterns appearing where previously there were 4× scalar equivalents. If SASS is unchanged, the reshape is cosmetic — still land it for readability but note the neutral result in the commit.

**Patterns to follow:**
- `gdn_decode_pretranspose.py` lines 102-110: `r_k = make_rmem_tensor(make_layout((vec_size,), stride=(1,)), Float32)` where `vec_size=4`. Our `sr` outer axis then carries the 32 V-row tiles.

**Test scenarios:**
- Happy path — full decode 30-workload sweep: all PASSED, abs/rel err unchanged within 1 ULP of v4.
- Happy path — prefill 5-workload sweep: all PASSED.
- Edge case — index arithmetic drift: confirm `sr[i, c]` access maps to the same memory as `sr[i*4 + c]` by running Unit 1's byte-compare on the output tensor.
- Integration — end-to-end kernel latency within Modal's ±15% measurement window (per "run twice" discipline above).

**Verification:**
- SASS diff in commit body shows vector-register patterns where v4 had scalar patterns, OR commit explicitly notes neutral result.
- Decode + prefill both PASS 30+5 workloads at tolerance.
- ptxas spill-bytes column in the optimization report updated; decode spill ≤ 90 B (currently unmeasured, v17 baseline is 132 B).

---

- [ ] **Unit 3: bf16 K/Q smem + vectorized cooperative load + inline conversion**

**Goal:** Match v7's K/Q smem layout — bf16 smem (half the footprint), loaded via `cute.autovec_copy` on a `(1, 8)` tile per lane, converted to fp32 inline inside the inner loop where the value is actually consumed. This unblocks the Unit 5 cp.async pipeline (which requires bf16 smem since `cp.async` is byte-copy, not format-converting).

**Requirements:** R1, R2, R4, R6.

**Dependencies:** Unit 1 (SASS baseline), independent of Unit 2 but cleaner to land after Unit 2 to isolate deltas.

**Files:**
- Modify: `solution/python/msinfer_entry.py` — `_gdn_decode_dev`, `_gdn_prefill_dev` smem alloc + cooperative load + inner-loop consumption sites.

**Approach:**
- Change smem allocation:
  ```
  sQ = smem.allocate_tensor(Float32, make_layout((D,)), 16)   # 512 B
  sK = smem.allocate_tensor(Float32, make_layout((D,)), 16)   # 512 B
  ```
  to
  ```
  sQ = smem.allocate_tensor(BFloat16, make_layout((D,)), 16)  # 256 B
  sK = smem.allocate_tensor(BFloat16, make_layout((D,)), 16)  # 256 B
  ```
  Total smem drops 1024 → 512 B. Creates headroom for Unit 5's 2× double buffer (1024 B total, same as v7 budget).
- Replace the cooperative load from the current `for j in range_constexpr(4): sQ[idx] = Float32(q[...])` pattern with `cute.autovec_copy` on a `(1, 8)` tile slice per lane (matches v7's 8-bf16-per-lane, 16 B per cp.async.cg).
  - Use `cute.local_tile(q, (1, 1, 1, 8), (batch, 0, qk_head, tid))` for decode; equivalent for prefill (adjust rank).
  - Or compute the tile index and use `autovec_copy` directly from gmem to a 4-bf16 rmem staging tensor, then write to smem.
- Inline conversion in the inner loops:
  ```
  ov += Float32(sK[idx]) * sr[...]     # was: ov += sK[idx] * sr[...]  (sK was fp32)
  qs += Float32(sQ[idx]) * sr[...]
  ```
  bf16→fp32 is `cvt.f32.bf16`, single PTX instruction, should emit a paired `cvt.f32.bf16.v2` when compiler sees two adjacent conversions.

**Execution note:** SASS diff in commit should show 4× fewer smem writes in the coop-load phase (one vectorized `STS.128` instead of 4 scalar `STS.32`s), plus `cvt.f32.bf16` in the inner loop where previously there were plain `LDS.32`s.

**Patterns to follow:**
- `gdn_decode_pretranspose.py` line 181-182: `cute.autovec_copy(q_tile, r_q_bf16)` then `r_q[i] = Float32(r_q_bf16[i])` in line 206-208.
- v7 static-CUDA `solution/cuda/kernel.cu` prefill smem: `__shared__ __align__(16) __nv_bfloat16 s_k_bf16[2][kHeadSize]`.

**Test scenarios:**
- Happy path — decode 30-workload: all PASSED, abs_err within 1 ULP of v4 (bf16 → fp32 is lossless for the fp32 accumulators we write back, so no additional error expected).
- Happy path — prefill 5-workload: all PASSED.
- Edge case — alignment: smem bf16 must be 16-B-aligned for `STS.128` — confirm via the `allocate_tensor(..., 16)` alignment arg.
- Error path — bf16→fp32 conversion precedence: test a workload with K values near fp16 normal/subnormal boundary (seek out a workload with A_log near 0 so gate is near 1 and state values are near raw K). Must not change abs/rel err beyond tolerance.
- Integration — smem budget sanity: `--ptxas-options=-v` reports smem ≤ 512 B (vs 1024 B in v4), confirming the allocation change lowered, not merely renamed, the footprint.

**Verification:**
- SASS diff shows vectorized `STS.128` in coop load and `cvt.f32.bf16` in inner loop.
- Both kernels PASS full workload sets.
- Smem footprint halves per ptxas output. (This is the headroom for Unit 5.)

---

- [ ] **Unit 4: Long-seq workload bench + reference-latency surfacing in `run_modal`**

**Goal:** Expose seq_len ≥ 200 prefill workloads in the bench harness (they exist in the trace set, we just never query them) and surface `reference_latency_ms` alongside `latency_ms` + `speedup_factor` in `run_modal.py`'s output dict. Prerequisite for Unit 5 — we have no data to claim cp.async is a win without long-seq measurements.

**Requirements:** R3, R5, R6.

**Dependencies:** None (pure harness work).

**Files:**
- Modify: `scripts/run_modal.py` — add `--min-seq-len` flag; always include `reference_latency_ms` in the per-trace entry; `print_results` formats it alongside kernel latency.
- Modify: `docs/gdn-optimization-report.md` — add a "long-seq prefill baseline" row to the progression table.

**Approach:**
- Add a `min_seq_len: int = 0` argument to `run_benchmark` and the `main` entrypoint. If set, drop workloads with `total_seq_len < min_seq_len`.
- Between the existing max_seq_len filter and max_workloads cap:
  ```
  if min_seq_len > 0:
      workloads = [w for w in workloads if _len_key(w) >= min_seq_len]
  ```
- Always populate `entry["reference_latency_ms"]` — it's already available from `trace.evaluation.performance.reference_latency_ms`, just not always surfaced. Update `print_results` to show it: `f"{result['latency_ms']:.3f} ms (ref {result['reference_latency_ms']:.3f} ms)"`.
- Run v4 with `--min-seq-len 200 --max-seq-len 4000 --max-workloads 6` to establish a long-seq baseline. Record in the doc.

**Execution note:** No kernel change. Harness only. Lands before Unit 5 so Unit 5 has a workload slice to validate against.

**Patterns to follow:**
- Existing `max_seq_len` / `max_workloads` filtering in `run_modal.py::run_benchmark`.

**Test scenarios:**
- Happy path — `modal run scripts/run_modal.py --min-seq-len 200 --max-seq-len 4000 --max-workloads 6`: filters to 6 workloads in the [200, 4000] seq range, all PASSED on current CuTe v4.
- Edge case — no workloads match the filter: raises a clear `ValueError` rather than silently running empty.
- Edge case — `min_seq_len > max_seq_len`: caller error, surfaces as empty filter + clear error.
- Integration — output dict contains both `latency_ms` and `reference_latency_ms` on every PASSED entry; `print_results` shows both.

**Verification:**
- A long-seq baseline table added to the optimization report: v4 decode stays at seq=1 workloads (baseline only, no long-seq concept for decode); prefill v4 on 6 workloads in [200, 4000] with kernel latency + ref latency recorded.
- `reference_latency_ms` visible on all non-errored entries in the printed output.
- `run_modal.py` still works without `--min-seq-len` (default 0 = no filter).

---

- [ ] **Unit 5: Prefill `cp.async.cg` double-buffered K/Q pipeline**

**Goal:** Land the `cp.async.cg` pattern for K/Q in the prefill token loop. On seq ≥ 200 this should show the overlap win that short-seq bench (Units 2-3) couldn't surface. Matches v7's async pipeline semantics.

**Requirements:** R1, R2, R3, R5, R6.

**Dependencies:** Unit 3 (bf16 smem is a hard prerequisite — cp.async doesn't do format conversion), Unit 4 (bench harness must be able to measure long-seq).

**Files:**
- Modify: `solution/python/msinfer_entry.py` — `_gdn_prefill_dev` (token loop structure + smem double-buffer), `_gdn_prefill_jit` (construct `tiled_copy_load`, pass to kernel).

**Approach:**
- Host-side in `_gdn_prefill_jit`:
  - `copy_atom = cute.make_copy_atom(cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), BFloat16, num_bits_per_copy=128)`
  - Two tiled_copies: one for K (`thread_layout=(16,)`, `val_layout=(1, 8)`), one for Q (same shape). Or one unified tiled_copy if partitioning is cleaner — defer to implementation.
  - Pass both `tiled_copy_k`, `tiled_copy_q` as kernel args.
- Kernel side:
  - Smem layout changes from `BFloat16 s_k_bf16[D]` (single buffer from Unit 3) to `BFloat16 s_k_bf16[2][D]` (double-buffer). Same for Q.
  - Pre-loop: issue cp.async for `seq_start`, `commit_group`, `wait_group(0)`, `sync_warp`.
  - Inside token loop:
    - If `t + 1 < seq_end`: issue cp.async for `t + 1` into buf `1 - buf`, `commit_group`.
    - Compute from `s_k_bf16[buf]`, `s_q_bf16[buf]`.
    - After compute: `cp_async_wait_group(0)`, `sync_warp`, flip `buf`.
- `sync_warp` after `wait_group(0)` — matches v7; needed so all lanes see the new buffer's writes before they read it.

**Execution note:** SASS diff must show `cp.async.cg.shared.global` in the emitted PTX. If it lowers to `cp.async.ca.shared.global` instead, the `cache_mode` arg isn't reaching the copy atom — regression vs v7 (which explicitly rejected the L1-cached variant). Validate via grep on the `.ptx` file from Unit 1's harness before benching.

**Technical design:** *(token loop structure — directional, not implementation)*

```text
buf = 0
issue_async_load(buf, seq_start); commit; wait(0); sync_warp
for t in seq_start..seq_end:
    if t + 1 < seq_end:
        issue_async_load(1 - buf, t + 1); commit
    ck = s_k_bf16[buf]; cq = s_q_bf16[buf]
    # compute unchanged from v4 structure:
    g, beta = compute_gate(t)
    sync_warp
    qk = warp_reduce( Σ cq[idx] · ck[idx] over tid+j*32 )
    pass_1: sr[i,c] *= g;  ov += ck[b4+c]·sr[i,c];  qs += cq[b4+c]·sr[i,c]
    delta = beta · (v[t,v_head,row] - ov)
    out[t,v_head,row] = scale · (qs + delta · qk)
    pass_2: sr[i,c] += ck[b4+c] · delta
    cp_async_wait_group(0); sync_warp
    buf = 1 - buf
persist sr to state_out[seq_idx, v_head, row, :]
```

**Patterns to follow:**
- `gdn_decode_pretranspose.py` lines 170-280 (token-loop structure, commit/wait ordering).
- `solution/cuda/kernel.cu` v7 prefill (on `submission-gdn-v7-v17` branch) — same token-loop shape, authoritative on sync placement.

**Test scenarios:**
- Happy path — prefill 5 short-seq workloads (seq_len ≤ 30): all PASSED, latency within ±15% of Unit 3 numbers. cp.async overhead at this scale should wash out.
- Happy path — 6 long-seq workloads (seq_len ∈ [200, 4000], from Unit 4's slice): all PASSED, mean kernel latency ≤ 0.6× of Unit 3 on the same slice (the real win).
- Edge case — `seq_len=1`: no `t+1` to prefetch; the guarded `if t + 1 < seq_end` branch must hold. Include a seq_len=1 workload if the trace set has one, else synthetic.
- Edge case — `seq_end == seq_start` (zero-length seq): existing guard `if seq_end <= seq_start: return` must still trigger before the first cp.async issue.
- Error path — wait/commit order inversion: a flipped `wait` before `commit` either hangs (bench timeout → RUNTIME_ERROR) or reads stale data (correctness fail). Detected by the correctness check on every workload.
- Integration — SASS: confirm `cp.async.cg.shared.global.L2::128B` appears in the emitted ptx (grep the `--keep-ptx` output from Unit 1's harness).

**Verification:**
- PTX contains `cp.async.cg` (not `cp.async.ca`).
- Short-seq and long-seq workloads all PASS correctness.
- Long-seq kernel latency meets R5 (≤ 70% of static-CUDA v7 on equivalent workloads) — or, if the reference has no v7 numbers on long seqs yet, record them for the first time.

---

- [ ] **Unit 6: `Constexpr` specialization sweep**

**Goal:** Thread the fixed structural constants (`HV`, `HQ`, `D`, `kSplits`, `kRowsPerBlock`) through the `@cute.kernel` signature as `cutlass.Constexpr[int]` rather than reading them as Python module globals. Hypothesis: allows the DSL's MLIR pass to constant-fold divmods and unroll more aggressively.

**Requirements:** R1, R2, R6.

**Dependencies:** None (can land anytime, but most informative after Unit 5 SASS is captured — any improvement then is specialization's actual contribution, not a hidden side-effect of an earlier change).

**Files:**
- Modify: `solution/python/msinfer_entry.py` — both kernel signatures, both `@cute.jit` launchers.

**Approach:**
- Promote `HQ`, `HV`, `D`, `kSplits`, `kRowsPerBlock` from Python module-level ints to kernel parameters with `cutlass.Constexpr[int]` type.
- Pass them from the `@cute.jit` launcher (still static Python ints at the call site, so they remain compile-time).
- Convert any remaining `range(...)` in the kernel body to `cutlass.range_constexpr(...)`. Only dynamic loop is the prefill outer token loop.
- Delete any dead Python globals that are now kernel-local.

**Execution note:** Plan expectation is this is neutral. If SASS diff shows anything substantive (fewer instructions, different unroll factor) that's a bonus. If `len(_CACHE)` stays at 2 after a multi-batch run, the Constexpr args aren't wrongly re-specializing. If `_CACHE` grows, one of the "Constexpr"-promoted values is actually dynamic — revert that one.

**Patterns to follow:**
- `gdn_decode_pretranspose.py` kernel signatures heavily use `cutlass.Constexpr[int]` on `HV`, `B`, `T`, `H`, `K`, `V` — canonical shape.

**Test scenarios:**
- Happy path — decode 30-wl + prefill 5-wl + prefill 6 long-seq: all PASSED, latency within ±5% of the prior unit's numbers (Unit 5 for prefill, Unit 3 for decode).
- Edge case — cache size: `len(_CACHE) == 2` (one decode compile + one prefill compile) after running all ~40 workloads end-to-end. If > 2, Constexpr captured something workload-varying — diagnose via the cache key contents.
- Integration — SASS diff vs Unit 5: any changes in register count, unroll factor, or instruction count recorded in the commit body.

**Verification:**
- All workloads PASS.
- Compile cache stays at size 2.
- Ptxas diff inlined in commit (even if no latency-detectable change).

## System-Wide Impact

- **Interaction graph:** Unit 4 changes `scripts/run_modal.py`'s result dict schema (adds `reference_latency_ms`). Any downstream consumer reading `print_results` output keeps working — the new field is additive. The modal function signature changes (`min_seq_len` param); callers pass via CLI flag, no existing call sites break.
- **Error propagation:** Unit 1's SASS extraction failure must not crash benches — catch and log, never raise. Unit 5's cp.async timing bugs show up as either hang (timeout → RUNTIME_ERROR) or correctness fail, both caught by the existing evaluator.
- **State lifecycle risks:** Unit 5's double-buffered smem is the only unit that introduces state. Ensure `buf = 0` before loop entry, flip only after `wait_group(0)`. Wrong flip ordering = correctness regression.
- **API surface parity:** `run` and `run_prefill` signatures in `msinfer_entry.py` are frozen by the contest definition. Every unit preserves them byte-for-byte. Internal kernel args (e.g. Unit 6's `Constexpr` promotions) are free to change.
- **Integration coverage:** End-to-end Modal benches are the only credible verification (no local unit tests for CuTe DSL kernels). Every code-change unit re-runs the full short-seq sweep at minimum; Unit 5 extends to long-seq.
- **Unchanged invariants:** `config.toml` (`language=python`, `entry_point`, `dependencies`), `_COMPILE_OPTS`'s `--enable-tvm-ffi --gpu-arch=sm_100a` (Unit 1 only ADDS flags, never removes), the contest DPS tensor shapes/dtypes, Modal image digest. Drift = scoring-parity regression.

## Alternative Approaches Considered

- **Port flashinfer's `gdn_decode_pretranspose.py` wholesale.** Rejected: adapter surface (cu_seqlens, pool indexing, h0_source, qk_l2norm flags) is massive relative to the perf upside, and their kernel's state layout `[B*HV, V, K]` differs from our contest signature `[B, HV, D, D]`. Would be larger work than fresh Phase A + B units.
- **Fall back to static CUDA v17/v7.** Rejected: violates the surface pin (R1). Team verified empirically that the static path is what diverges between Modal and the scoring runner; switching back guarantees scoring regression even if Modal numbers look better.
- **Multi-warp blocks (2 warps × 64 threads).** Deferred: doubles per-block parallelism at low batch but requires re-deriving the V-split indexing and adding a second smem reduction path. Out of scope for a "remaining levers" plan — belongs in its own plan if Phase A + B don't close enough gap.
- **Blackwell `tcgen05.*` tensor-core MMA for K·state dot.** Deferred: GDN's delta-rule recurrence doesn't naturally decompose into GEMM sub-problems; even if it did, Blackwell fp32 MMA (`tcgen05.mma` f32) requires TMA + shared tensor memory staging that the rest of our kernel isn't set up for. Separate spike plan, not this one.
- **Accept v4 as final (no further optimization).** Considered: decode at 48% of v17, prefill at 84%. Per scoring-parity argument this still beats a hypothetical static-CUDA submission that scores at 0.8×. The 70% R4/R5 threshold chosen here is conservative — if Phase A lands and plateaus at 60%, "ship it" is a reasonable exit.

## Risks & Dependencies

| Risk | Likelihood | Impact | Mitigation |
|------|-----------:|-------:|-----------|
| Unit 2's `(32, 4)` rmem reshape doesn't change SASS — DSL already emits vector moves under `(D,)` | Med | Low | Unit 1's SASS diff will show this immediately. If neutral, land the reshape anyway for readability and proceed to Unit 3 without re-planning. |
| Unit 3's `bf16` smem + inline conversion adds bf16→fp32 ops inside hot loop that weren't there in v4's fp32-smem path | Med | Low-Med | bf16→fp32 is 1 cycle (`cvt.f32.bf16`). Hot-loop does 256 of them per token vs 256 smem reads — same order of magnitude. ptxas `cvt.v2` pairing halves instruction count if compiler detects it. Worst case: unit is neutral on latency, still wins on smem (enables Unit 5). |
| Unit 5's `cpasync.CopyG2SOp` lowers to `cp.async.ca` (L1-cached) instead of `cp.async.cg` | Low | High — re-introduces the exact regression v7 rejected | Grep `--keep-ptx` output in Unit 1's harness before committing Unit 5. If wrong cache mode, pass `cache_mode=LoadCacheMode.GLOBAL` explicitly (already in the plan; this is just a grep-verification checkpoint). Fall back to inline PTX `asm volatile` if DSL refuses to honor the hint. |
| Unit 5's double-buffer wait/commit order regression (flipped before compute starts) | Low | High — correctness fail | Match v7's token-loop sequence exactly. Every workload's correctness check catches it; `--max-workloads 1` catches it in under 30s. |
| `--gpu-arch=sm_100a` gets dropped during `_COMPILE_OPTS` string edits in Units 1/6 | Low | Critical | Grep check in CI-equivalent script + visual review of commit. Units 1/6 also capture ptxas output which will show sm_100a vs sm_100/sm_90 in the emitted arch string — any drift is immediately visible. |
| Unit 6's `Constexpr` promotion accidentally captures a workload-dynamic value | Low | Medium | `len(_CACHE) == 2` invariant in Unit 6 test scenarios. Fails loudly if specialization over-specializes. |
| Long-seq workloads exceed Modal function timeout (7200 s currently) | Low | Low | `run_modal.py` already has `timeout=7200`. At seq=4000 even a slow kernel completes in seconds; reference Python at seq=4000 may take longer but fits within the timeout. Monitor via bench logs. |
| Modal run-to-run variance masks a real regression as noise | Med | Med | "Run twice, within ±15%" discipline in Key Technical Decisions. Every commit's perf claim cites two consecutive runs' kernel latencies. |

## Documentation / Operational Notes

- `docs/gdn-optimization-report.md` gains: SASS baseline table (Unit 1), per-unit SASS diff rows (Units 2, 3, 5, 6), long-seq prefill numbers (Unit 4 harness + subsequent units).
- `docs/runtime-pin.md` unchanged — no pinned version changes anticipated.
- `CLAUDE.md` unchanged.
- No operational/deployment impact — single-file kernel submission.

## Sources & References

- **Prior plan**: `docs/plans/2026-04-18-001-refactor-gdn-cute-v17-gap-closing-plan.md` (Units 4, 6 deferred from that plan are Units 5, 6 here).
- **Current implementation**: `solution/python/msinfer_entry.py` @ `fullagent/opt_v2` commit `a6048ce` (post-v4).
- **Oracle**: `solution/cuda/kernel.cu` @ `fullagent/submission-gdn-v7-v17` (v17 decode + v7 prefill).
- **DSL idiom source**: `flash/lib/python3.12/site-packages/flashinfer/gdn_kernels/gdn_decode_pretranspose.py`.
- **DSL compiler options**: `flash/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/base_dsl/compiler.py::parse_options`.
- **Optimization history**: `docs/gdn-optimization-report.md`.
- **Runtime pin**: `docs/runtime-pin.md`.
- **Bench tooling**: `scripts/run_modal.py`, `scripts/pack_solution.py`.
- **PR**: [Bammuri/mlsys26-fullagent#1](https://github.com/Bammuri/mlsys26-fullagent/pull/1) — two-commit stack (static CUDA reference + CuTe v1→v4). This plan's output lands as subsequent commits on `fullagent/opt_v2`, then gets cherry-picked into the PR branch similar to how v4 landed.
