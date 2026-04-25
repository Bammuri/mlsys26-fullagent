# GDN Decode v4 TMA — Session Handoff (2026-04-25)

## Where we are

- **Branch:** `decode-col-v3`
- **HEAD:** `1f7234d` — *"feat(gdn_decode): v4_tma final — TMA load + TMA store"*
- **Working tree:** clean (the modal logs and ncu reports under `out/` are untracked, gitignored)
- **Spec:** `docs/superpowers/specs/2026-04-24-gdn-decode-v4-tma-design.md`
- **Plan:** `docs/superpowers/plans/2026-04-24-gdn-decode-v4-tma-plan.md` (9 tasks; **Tasks 1-6 completed**, **7-9 pending**)

## What's landed (Tasks 1-6)

| Plan task | Commit | Summary |
|-----------|--------|---------|
| Task 1 | `fcee414` | bench config → 기본 (20 iter × 3 trial), v1 baseline log |
| Task 2 | `0233937` | `MSINFER_KERNEL` env var dispatcher (retires `MSINFER_TMA` bool) |
| Task 3 | `8fd0e5f` | `tma_desc.py` — ctypes wrapper for `cuTensorMapEncodeTiled` |
| Task 4 | `2889ecb` | v4 skeleton: warp-per-row layout, `__ldg` transport (no TMA) |
| Task 5 | `2adeb3f` | v4 + TMA load only (`cp.async.bulk.tensor.2d`); cache-key fix in `tma_desc.py` |
| Task 6 | `1f7234d` | v4_tma final: TMA load + TMA store, in-place smem update; intermediates removed |

Earlier preparatory commits in this session: `3678a0e` (spec), `083f92e` (run_modal env forwarding + py3.9 compat), `4ef8020` (plan).

## Current kernel surface

`gdn_decode/solution/python/msinfer_cuda.py` exposes 3 kernels via `MSINFER_KERNEL`:

- `v1` — current default; original `__ldg` column-parallel
- `v2_cpasync` — kept during ramp; removed in Task 9
- `v4_tma` — new warp-per-row + bulk TMA (this work)

Plus `gdn_decode/solution/python/tma_desc.py` (ctypes wrapper). `scripts/run_modal.py` forwards `--env KEY=VALUE,...` to the Modal worker.

## Plan defects we hit (already fixed in code, **not in plan doc**)

If Tasks 7-9 are re-dispatched verbatim from the plan, two patches must be applied at top:

1. **state_tile alignment:** plan says `__shared__ alignas(16) float state_tile[...]`; correct value is `alignas(128)`. `cp.async.bulk.tensor.2d` requires 128-byte aligned smem destination, otherwise runtime "misaligned address" XID 13.
2. **Launcher else branch:** plan says `asm("trap;");` (PTX, host-side compile error). Use `abort();` from `<cstdlib>`.
3. **Cache key in `tma_desc.py`:** plan keyed cache on `(B, HV, kRows, in_ptr & ~2MB-1, out_ptr & ~2MB-1)`. Correct key uses **exact** pointers — PyTorch reuses the 2 MB region at different offsets across workloads, causing silent stale-descriptor corruption otherwise. Already in code.

These three are committed; the plan doc still has the buggy snippets. If the plan is re-read, mentally substitute.

## Bench results (기본 = 20 iter × 3 trial, Modal B200)

| B | v1 (commit `fcee414`) | v4_tma (commit `1f7234d`) | Δ |
|---|-----------------------|---------------------------|---|
| 1 | 0.013 ms | 0.019 ms | **+46% slower** |
| 2 | 0.013 ms | 0.019 ms | +46% slower |
| 4 | 0.020 ms | 0.019 ms | -5% |
| 8 | 0.020 ms | 0.019 ms | -5% |
| 16 | 0.020 ms | 0.021 ms | +5% |
| 32 | 0.020 ms | 0.023 ms | **+15% slower** |
| **64** | **0.040 ms** | **0.027 ms** | **-33% faster** |

Correctness on all 7 buckets / 54 workloads: **54 PASSED**, max abs_err ~3e-5 (within v1 envelope).

## Why low-B regresses

`cp.async.bulk.tensor.2d` is issued by **a single thread** while the other 127 wait on the mbarrier. The fixed TMA issue latency dominates when the per-block work is small (B=1: only 64 blocks total, kSplits=8 → small tiles, low parallelism). v1's 128-thread parallel `__ldg` saturates LSU faster on these tiny workloads.

## Spec gate status

The spec's Acceptance Checklist requires:
- ≥ 2× at B=32 → not met (we have +15% **regression**)
- ≥ 2× at B=64 → not met (we have 1.5×, not 2×)
- No bucket regresses > 10% → not met (B=1/2/32 all regress)

So **Gate A as written is failed**. Tasks 7-9 (perf gate, flip default, prune v2) cannot proceed without an upstream decision.

## Decision needed before resuming

Four options, ranked by my recommendation:

### (a) Hybrid dispatch — recommended

Route by batch size in `run()`:
```python
if q.size(0) <= 4:
    ext.gdn_decode_v1(...)
else:
    ext.gdn_decode_v4_tma(...)
```
- Pros: ships the high-B win immediately (B=64: 1.5×); zero low-B regression; minimal code change; reversible
- Cons: doesn't address why low-B is slow (defers to follow-up); 2× target at B=32 still missed
- Effort: ~30 min — implement + bench + commit

### (b) Optimize v4_tma further

Hot ideas, ordered by likely impact:
- Eliminate `s_qk` smem broadcast — pass qk via `__shfl_sync` cross-warp through smem with one fewer barrier
- Issue TMA load earlier — overlap with Q/K/V smem load instead of sequencing them
- 4-thread cooperative TMA issue (multiple `cp.async.bulk.tensor` in flight via `cp.async.bulk.commit_group`) — cuts mbarrier-wait latency
- Investigate ptxas register count + spill (need `--dump-sass`); if >48, occupancy drops below 8 blocks/SM
- Effort: 1-3 hours; uncertain gain at low-B (TMA fixed cost may not shrink below ~5 µs)

### (c) Re-baseline Gate A

Spec's 2× target was an estimate from arithmetic intensity. The TMA fixed-issue floor we observed (~5 µs at low-B regardless of work size) suggests the realistic ceiling is closer to 1.5-1.7× at B=64. Relax gate to:
- High-B (B≥16): v4_tma must be at least as fast as v1
- Low-B (B≤4): v1 path used (via hybrid)
- Then proceed Task 7-9 unchanged
- Effort: 5 min plan/spec edit + Task 7 bench

### (d) ncu profile first

Get hard data on where 0.022 ms at B=32 is actually spent — TMA wait? Compute? Smem bank conflicts?
- Pros: data-driven choice between (a)/(b)/(c)
- Cons: 30-60 min setup; ncu on Modal needs the `run_ncu` function; may just confirm what we suspect
- Effort: 1 hour

## Untracked artifacts

`out/v4-*.log` files contain per-task bench logs (some force-added to git, some still untracked). `out/sass-dump.tar.gz` is from an earlier session.

## File map (recent changes only)

```
docs/superpowers/specs/2026-04-24-gdn-decode-v4-tma-design.md   spec
docs/superpowers/plans/2026-04-24-gdn-decode-v4-tma-plan.md     plan (Tasks 1-9)
docs/superpowers/2026-04-25-gdn-decode-v4-tma-handoff.md        ← this file
gdn_decode/solution/python/msinfer_cuda.py                      v1 + v2_cpasync + v4_tma
gdn_decode/solution/python/tma_desc.py                          ctypes builder
scripts/run_modal.py                                            --env forwarding
out/v4-baseline-v1-geoban.log                                   기본 v1 baseline
out/v4-task[2-6]-*.log                                          per-task bench logs
```

## Bench commands (cheat sheet)

```bash
PY=/Users/hjun20.kim/Workspace/mlsys26/flash/bin/python3

# v1 (current default)
$PY -m modal run scripts/run_modal.py --kernel-dir gdn_decode

# v4_tma
$PY -m modal run scripts/run_modal.py --kernel-dir gdn_decode --env "MSINFER_KERNEL=v4_tma"

# v2_cpasync
$PY -m modal run scripts/run_modal.py --kernel-dir gdn_decode --env "MSINFER_KERNEL=v2_cpasync"

# SASS dump (for register-pressure / TMA-emit verification)
$PY -m modal run scripts/run_modal.py --kernel-dir gdn_decode --env "MSINFER_KERNEL=v4_tma" --dump-sass
```

`scripts/run_modal.py` line 87 hard-codes `BenchmarkConfig(warmup_runs=3, iterations=20, num_trials=3)` (기본). Switch to `iterations=50` for 본판 (Task 7 / submission).

---

# Next-session prompt

Paste this at the start of the next session to resume:

```
Resuming GDN decode v4_tma optimization on branch decode-col-v3.

Read these in order:
  1. docs/superpowers/2026-04-25-gdn-decode-v4-tma-handoff.md   (current state, decisions pending)
  2. docs/superpowers/specs/2026-04-24-gdn-decode-v4-tma-design.md   (spec)
  3. docs/superpowers/plans/2026-04-24-gdn-decode-v4-tma-plan.md   (plan; Tasks 1-6 done, 7-9 pending)

Plan defects already fixed in code (NOT in the plan doc itself):
  - state_tile must use alignas(128), not alignas(16)
  - launcher else branch uses abort() not asm("trap;")
  - tma_desc.py cache key uses exact pointers, not 2 MB-aligned region

Status: v4_tma works correctly (54/54 PASSED) but does not meet Gate A —
B=64 is 1.5x v1 (target was 2x), B=1/2/32 all regress. The handoff doc
spells out the four options I left for you to choose.

Open decision: pick one of (a) hybrid v1/v4_tma dispatch by batch size,
(b) optimize v4_tma further, (c) re-baseline Gate A, (d) ncu profile first.
My recommendation in the handoff is (a) — ships the high-B win,
zero low-B regression. Read the handoff and confirm or override.

Modal Python: /Users/hjun20.kim/Workspace/mlsys26/flash/bin/python3
(modal CLI installed there; system python3 is too old).

Bench (기본 = 20 iter × 3 trial, what's currently in run_modal.py):
  $PY -m modal run scripts/run_modal.py --kernel-dir gdn_decode --env "MSINFER_KERNEL=v4_tma"

Bench (본판 = 50 iter × 3 trial, used at Task 7 submission gate):
  edit scripts/run_modal.py:87 to iterations=50, then run as above

Don't push, don't amend, don't --no-verify. Single-commit per task per
the original plan style.
```
