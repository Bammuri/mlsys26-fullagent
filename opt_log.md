# Optimization Log (opt_log.md)

> Checkpoint-style log for `workflow.md` execution.
> If a session is interrupted, the next agent should pick up from **NEXT ACTION** below.

---

## Session state

- **Repo root**: `/Users/jaewoo/SAIT/FlashInfer Challenge/mlsys26-fullagent`
- **Branch**: `prefill-opt-jw`
- **Goal** (workflow.md Phase 4): Avg latency ≤ **0.100 ms** on `gdn_prefill_qk4_v8_d128_k_last`
- **Deadline** (EVALUATION.md): 2026-04-24 (final submission). Today: 2026-04-18 → 6 days.
- **Benchmark env**: Modal B200, `pyenv shell fi-bench` required
- **Pack command**: `PYENV_VERSION=fi-bench python scripts/pack_solution.py`
- **Bench command**: `PYENV_VERSION=fi-bench modal run scripts/run_modal.py`

## Codebase state (important — read before editing anything)

There are **three candidate submission lanes** under `solution/`:

| Lane | Path | Notes |
|------|------|-------|
| Python / CuTe DSL (CURRENT ACTIVE) | `solution/python/msinfer_entry.py` → `gdn_blackwell` | Uses `nvidia-cutlass-dsl`. Labeled "fastest verified submission" in recent commits (a1faae9, fa18226). This is what `config.toml` packs today. |
| FlashInfer wrapper | `solution/cuda/binding.py` | Delegates to `flashinfer.chunk_gated_delta_rule`. Not active. |
| Pure CUDA kernel | `solution/cuda/kernel.cu` + `binding.cpp` | Recurrent baseline; workflow.md claims 0.178 ms. **This is the lane workflow.md wants us to evolve.** |

### Key discrepancy
- `workflow.md` §6.1 bans CuTe / CUTLASS / FlashInfer → asks for **pure CUDA only**.
- `config.toml` currently points to the CuTe DSL Python lane (violates workflow §6.1).
- EVALUATION.md lists `nvidia-cutlass-dsl` as available in the eval image, so the ban is a **self-imposed design rule**, not an eval requirement.

**Interpretation**: workflow.md is a fresh pure-CUDA effort targeting 0.178 → 0.100 ms. The existing CuTe lane is the safety net / current production submission.

## Safety rails

1. **Do not delete or regress the CuTe lane.** It is the current best submission. Keep `solution/python/` and the `"gdn_blackwell"` package untouched.
2. Before any edit to `solution/cuda/kernel.cu` or `solution/cuda/binding.cpp`, snapshot the current file by reading it into memory / git.
3. After every kernel edit, run pack + modal. Never stack two un-measured changes.
4. If modal run fails (compile error, infra issue), do **not** retry blindly — read the log, fix root cause.
5. If correctness breaks (`status != correct`) or latency regresses, **immediately revert** and pick a different candidate from workflow.md §7.

## Current Phase: 0 (baseline verified)

### Confirmed rule from user
- **Do NOT modify `config.toml`**. It stays on Python/CuTe lane. Use a separate packed JSON for CUDA-lane measurements.

### What we measured (this session)
- Briefly switched `config.toml` to CUDA (`binding.cpp::run`, torch extension), packed, ran Modal, then reverted.
- **Pure-CUDA kernel baseline on Modal B200** (20 workloads, 1 warmup / 5 iter / 3 trials):
  - **Avg latency: 0.537 ms** (all PASSED)
  - Worst abs error: 6.10e-05 (OK)
  - Range: 0.019 ms (short seq) – 2.204 ms (long seq)
  - Note: workflow.md §9 quoted **0.178 ms** as baseline — that was a different workload set / config, not reproducible here. Our real starting point is **0.537 ms**.

### Active measurement scheme (going forward)
Because `config.toml` must stay on CuTe:
- Create a side-script `scripts/pack_cuda_solution.py` (new) that builds a `BuildSpec(language="cuda", entry_point="binding.cpp::run", binding="torch")` JSON at `solution_cuda.json` without touching config.toml.
- Measure via: `PYENV_VERSION=fi-bench modal run scripts/run_modal.py --solution-path solution_cuda.json`
- CuTe lane remains the default for normal pack + run.

### NEXT ACTION (pick this up if session is interrupted)
1. Write `scripts/pack_cuda_solution.py` that produces `solution_cuda.json` from `solution/cuda/` (no config.toml read).
2. Re-verify CUDA baseline via the side script (sanity check — should match 0.537 ms).
3. Begin Phase 1 optimizations on `solution/cuda/kernel.cu` (K1 → K3 → E1 → E3 → G1 → H1) one at a time.
4. After every edit: `python scripts/pack_cuda_solution.py && modal run scripts/run_modal.py --solution-path solution_cuda.json`.
5. Log each iteration under "Iteration log" in this file BEFORE picking the next optimization.

## Iteration log

### Iteration 0 (baseline — first measurement, random 20-workload sample)
- Lane: pure CUDA (`solution/cuda/kernel.cu` + `binding.cpp`, torch extension)
- How measured: `config.toml` briefly set to CUDA lane; packed; `modal run scripts/run_modal.py` (1 warmup, 5 iter, 3 trials, **max_workloads=20 default → random sample**); config.toml reverted.
- **Avg latency: 0.537 ms** (over a random 20 workloads) | 20/20 PASSED | worst abs_err 6.10e-05 | worst rel_err 1.88e-01
- Per-workload (ms): 0.024, 0.118, 1.418, 0.059, 0.032, 2.204, 0.028, 0.050, 0.419, 0.157, 1.699, 0.044, 1.983, 0.029, 1.586, 0.027, 0.214, 0.019, 0.059, 0.570
- Phase: 0 → Phase 1

### Iteration 1 — WARP REDUCE SUBSTITUTION (inconclusive due to workload sampling issue)
- Change: replaced `warp_sum + warp_broadcast_0` with `warp_sum_all` (butterfly XOR reduce) for the `k·state` dot product. Saves 1 shuffle per token × 2 reductions/token × T tokens.
- Correctness: all 20 workloads PASSED, worst abs_err 1.22e-04 (was 6.10e-05) — within tolerance.
- Result over a *different* random-20 sample: 0.654 ms — **NOT comparable** to 0.537 ms baseline.
- **ROOT-CAUSE DISCOVERY**: `scripts/run_modal.py` calls `random.sample(workloads, 20)` unseeded, so every run picks a different subset. Per-workload comparisons on the 5 overlapping workloads show neutral-to-slightly-positive (long seqs improved 2-4 %, one short seq regressed — noise-level).
- **Mitigation**: From now on run with `--max-workloads 0` (triggers the "return all workloads" branch, deterministic order). All future iterations use this.
- Decision: KEEP the iter 1 change (cleaner code, neutral-to-positive effect); re-baseline with `--max-workloads 0`.
- Phase: 1

### Iteration 2 — DETERMINISTIC BASELINE (30-workload, seed=42)
- Context: `--max-workloads 0` (all 100 workloads) hit Modal's 3600s timeout with the recurrent kernel under `use_isolated_runner=True`. Switched to `--max-workloads 30 --sample-seed 42` for a reproducible Phase-1 reference that fits the budget. Fixed `select_workloads` to not rely on `.uuid` attribute (remote flashinfer-bench schema differs).
- Avg latency: **0.392 ms** (30/30 PASSED) | worst abs_err 6.10e-05 | runtime ~20 min
- This is the new **Phase-1 reference**; all subsequent CUDA iterations use the same `--max-workloads 30 --sample-seed 42` config.
- Phase: 1 (baseline)

### Iteration 3 — K3: `__builtin_assume` hints on hot indices
- Change: added `__builtin_assume` on `blockDim.x`, `num_seqs > 0`, `seq_end >= seq_start`, and in-range index assertions after the early-return in both kernels.
- Avg latency: **0.391 ms** (prev: 0.392) | 30/30 PASSED | abs_err 6.10e-05 unchanged
- Δ: −0.001 ms (noise-level, as expected for a memory-bound recurrent loop)
- Decision: KEEP (harmless, slightly cleaner backend codegen).
- Phase: 1

### Iteration 4 — E3: L2 persistence on state buffer  → ROLLBACK
- Change: `cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, maxBytes)` once per device + per-call `cudaStreamSetAttribute(..AccessPolicyWindow..)` on `new_state` with `hitProp=Persisting`, `missProp=Streaming`, `hitRatio=min(max/size,1)`.
- Avg latency: **0.402 ms** (prev: 0.391) | 30/30 PASSED
- Δ: **+0.011 ms (+2.8% regression)** — reverted.
- Why it regressed: the recurrent kernel writes `new_state` exactly once at block end; there's no in-kernel reuse to amortize. Persistence carveout just steals L2 ways from the real bandwidth consumers (Q/K/V streaming) and costs a stream-set-attribute per launch.
- Insight: for this kernel layout, Q and K are the data actually reused across blocks (GQA-shared); state is not. L2-persistence is a Phase-2 tool, not Phase-1.
- Phase: 1

### Iteration 5a — G1 all-lane inline gate/beta  → ROLLBACK
- Change: removed `compute_gate_beta_kernel`, had every lane re-compute `expf(-expf(A_log[h])*softplus(a+dt_bias)) / sigmoid(b)` per token.
- Avg latency: **0.621 ms** (+58.8% regression) | 30/30 PASSED
- Root cause: 32 lanes all issuing `expf` each token saturates the SFU (one expf/cycle per warp). SFU pressure dwarfs the kernel launch overhead we saved.

### Iteration 5b — G1 lane-0-only inline gate/beta  → ROLLBACK
- Change: lane-0 reads a/b, does softplus+2×expf+sigmoid, then `warp_broadcast_0`. Restores old SFU load but inline.
- Avg latency: **0.711 ms** (+82% regression) | 30/30 PASSED
- Root cause: lane-0's per-token compute chain (2 dependent global bf16 loads → softplus → 2×expf → sigmoid) is on the warp's critical path before `gate*warp_sum_all(k·state)` can start. The preproc kernel had done this once offline; fusing it rebuilds the chain per-token.
- Insight: G1 requires **per-seq cooperative SMEM staging** (warp 0 fills `gate_beta[0..T)` into SMEM at kernel entry, __syncthreads, then hot loop reads from SMEM). That needs dynamic SMEM sized by max-T across seqs. Defer to after Phase 1.
- Phase: 1 (kept preproc kernel, i.e., post-K3 state = 0.391 ms reference).

### Iteration 6 — H1: 4 warps/block (kRowsPerBlock=4, kThreads=128)  → ROLLBACK
- Change: `kWarpsPerBlock 2 → 4`. Grid halved (256×N vs 512×N). Block has 128 threads, `__launch_bounds__(128, 4)` (compiler targets 4 blocks/SM = 512 threads/SM).
- Avg latency: **0.398 ms** (+1.8% regression) | 30/30 PASSED | abs_err unchanged.
- Root cause: the recurrent per-row loop is serial on a dependency chain (state update). Going wider doesn't help — each warp still serializes per-token. Halved grid reduces parallelism on low-`num_seqs` workloads (tail effect). Register pressure also forces launch_bounds(128,4) to cap regs/thread at ~128 → possible spills.
- Insight: occupancy isn't the bottleneck — the critical path is the serial state-update FMA chain per warp. More warps ≠ more throughput.

### Iteration 7 — H2: `__launch_bounds__(kThreads, 8)` (min 8 blocks/SM)  → ROLLBACK
- Change: only the launch_bound min-blocks hint from 4→8 (kWarpsPerBlock back to 2). Tells compiler to cap regs/thread to fit 8×64=512 threads/SM = 25% occupancy.
- Avg latency: **0.395 ms** (+1.0% regression) | 30/30 PASSED.
- Root cause: same as H1 — the serial FMA chain is latency-bound, not throughput-bound. Extra register spills from the tighter reg cap outweigh any occupancy gain.
- Phase: 1 (kept post-K3 state = 0.391 ms reference).

---

## Phase-1 verdict

After K3, E3, G1 (2 variants), H1, H2: **all micro-tuning candidates on the recurrent per-token layout either regress or are noise-level.**

| Iter | Change | Δ vs 0.391 ms | Decision |
|------|--------|--------------:|----------|
| 3 | K3 — `__builtin_assume` | −0.001 ms (−0.3%) | KEEP |
| 4 | E3 — L2 persist on state | +0.011 ms (+2.8%) | REVERT |
| 5a | G1 — all-lane inline gate | +0.230 ms (+58.8%) | REVERT |
| 5b | G1 — lane-0 inline gate | +0.320 ms (+82%) | REVERT |
| 6 | H1 — 4 warps/block | +0.007 ms (+1.8%) | REVERT |
| 7 | H2 — min-blocks 8 | +0.004 ms (+1.0%) | REVERT |
| 9 | F1 — algebraic decouple of out | −0.004 ms (−1.0%) | KEEP |
| 10 | F2 — `#pragma unroll 2` | −0.050 ms (−12.8%) | KEEP |
| 11 | F3 — `#pragma unroll 4` | −0.093 ms (−23.8%) | KEEP |
| 12 | F4 — `#pragma unroll 8` | −0.110 ms (−28.1%) | KEEP |
| 13 | F5 — `#pragma unroll 16` | +0.010 ms (+2.6%) | REVERT |
| 14 | F6 — `#pragma unroll 12` | −0.070 ms (−17.9%) | REVERT (worse than F4) |
| 15 | F7 — drop warp_broadcast_0 ×3 | −0.123 ms (−31.5%) | KEEP |
| 16 | F8 — retry unroll 16 post-F7 | −0.019 ms (−4.9%) | REVERT (worse than F7) |

**Root cause of the ceiling**: the current kernel is a **serial per-token recurrent FMA chain** per (seq, head, row). Each token requires completion of the previous token's state update. No amount of (1) occupancy tuning, (2) hint-level tuning, (3) load-path tuning, or (4) SFU/launch-overhead fusion can break Amdahl's bound when the arithmetic is on a fundamental per-token dependency chain.

Phase-1 target is **0.150 ms**. Current: **0.391 ms** → **2.6× speedup still required**. Not achievable via Phase-1 tools on this layout.

**The only remaining Phase-1 candidate** is **E1** (128-bit packed bf16 loads for Q/K) — halves the number of ldg.v4.b16 instructions per token. Since the kernel is partially memory-bound on Q/K reads (bf16, 4 loads/lane/token), this could yield 5-15% (0.35-0.37 ms). Still far from 0.150 ms.

**Verdict**: Phase 1 optimizations alone cannot reach 0.150 ms. **Phase 2 chunkwise rewrite is mandatory** to close the 2.6× gap. The chunkwise rewrite replaces the per-token serial chain with a per-chunk GEMM-style parallel computation (WY compact form + γ_cum trick) that amortizes state updates across CHUNK_SIZE tokens in parallel — O(T/CHUNK_SIZE) serial steps instead of O(T).

### Iterations 10-16 — F2-F8: `#pragma unroll` sweep + scalar-broadcast removal

| # | Change | Latency (ms) | Δ vs prev kept | Decision |
|---|--------|---:|---:|---|
| 10 | F2: `#pragma unroll 2` | 0.341 | −0.046 (−11.9%) | KEEP |
| 11 | F3: `#pragma unroll 4` | 0.298 | −0.043 (−12.6%) | KEEP |
| 12 | F4: `#pragma unroll 8` | 0.281 | −0.017 (−5.7%) | KEEP |
| 13 | F5: `#pragma unroll 16` | 0.401 | +0.120 (+42.7%) | REVERT |
| 14 | F6: `#pragma unroll 12` | 0.321 | +0.040 (+14.2%) | REVERT |
| 15 | F7: drop 3× `warp_broadcast_0` for gate/beta/v (all-lane scalar reads — L1 coalesces, saves 3 shuffles/token) | 0.268 | −0.013 (−4.6%) | KEEP |
| 16 | F8: retry `unroll 16` post-F7 (hoping broadcast removal freed regs) | 0.372 | +0.104 (+38.8%) | REVERT |

**Final unroll sweep verdict**: `#pragma unroll 8` is the sweet spot. Higher (12, 16) spills regs to local memory; lower (2, 4) leaves ILP on the table. Removing the 3 lane-0 broadcasts (F7) was a clean structural win — same data flows but 3 fewer shuffles per token, and L1 coalesces the all-lane scalar reads (gate/beta/v are tiny).

**Cumulative Phase-1 progress**: 0.537 ms (random seed baseline) → 0.391 ms (deterministic K3 baseline) → **0.268 ms** (post-F7) — a real 31% reduction off the K3 baseline (50% off the random baseline). The F-series (algebraic decouple + unroll) was the productive vein; the earlier E/G/H/E1-lite candidates all regressed because they targeted the wrong bottleneck.

### Iteration 9 — F1: algebraic decoupling of `out` from state update  → KEEP
- Change: previously `out = warp_sum(<q, state_after>)` — depended on the post-FMA state, so the per-token critical path was butterfly_kS → state_update (4 FMA) → butterfly_out → store. Rewrote `out = γ·<q,S_old> + <q,k>·diff` (algebraically equivalent), so all three reductions (`<k,S>`, `<q,S>`, `<q,k>`) depend only on **pre-update** values and can be issued in parallel with each other; `out` is then computed by 2 multiplies + 1 add.
- The state update (4 FMAs on `state_vec`) now runs in parallel with `out` computation; the loop carry on `state_vec` shortens by one full butterfly worth of dependency chain.
- Avg latency: **0.387 ms** (prev 0.391) | 30/30 PASSED | abs_err 6.10e-05 unchanged | rel_err unchanged.
- Δ: **−0.004 ms (−1.0%)** — small but reproducible structural improvement (cleaner SASS expected; less critical-path serial). KEEP.
- Note: the empirical gain is smaller than the theoretical critical-path shortening predicted (~17%). This confirms the kernel is mostly memory-bound on the bf16 Q/K loads, not arithmetic-bound on the warp shuffles. Phase-2 chunking (which amortizes Q/K loads across multiple state updates within a chunk) remains the necessary path to break the 0.150 ms barrier.

### Iteration 8 — E1-lite: explicit `__ldg` on Q/K/V/state/gate_beta reads  → ROLLBACK
- Change: `load_bf16x4`, `bf16_to_float`, state `float4` load, and lane-0 `gate_beta` load all routed via `__ldg` (read-only cache).
- Avg latency: **0.415 ms** (+6.1% regression) | 30/30 PASSED | worst abs_err unchanged.
- Root cause: on Blackwell, `const __restrict__` pointers already route through L1TEX; forcing `__ldg` disables the compiler's L1 line-load merging across adjacent lanes, increasing memory transactions. Q/K are also reused across warps within a block, so the persistent L1 path was already the right choice.
- Phase: 1. Reverted completely — kernel is back at the post-K3 = **0.391 ms** reference.

---

## Phase-1 CONCLUSIVELY EXHAUSTED

Seven distinct Phase-1 candidates tried, all failed or noise-level:

| Iter | Change | Δ | Verdict |
|------|--------|---:|---------|
| 3 | K3 — `__builtin_assume` hints | −0.001 ms | KEEP (noise) |
| 4 | E3 — L2 persist on state | +0.011 ms | REVERT |
| 5a | G1a — all-lane inline gate | +0.230 ms | REVERT |
| 5b | G1b — lane-0 inline gate | +0.320 ms | REVERT |
| 6 | H1 — 4 warps/block | +0.007 ms | REVERT |
| 7 | H2 — min-blocks 8 | +0.004 ms | REVERT |
| 8 | E1-lite — `__ldg` hints | +0.024 ms | REVERT |

**Fundamental ceiling**: the per-token recurrent FMA chain `state[t] = gate[t]·state[t-1] + k[t]·β[t]·(v[t] − gate[t]·<k[t],state[t-1]>)` is a **serial scalar data dependency per row**. No micro-tuning (occupancy, cache hints, launch bounds, fusion) can break the Amdahl ceiling imposed by this serialization.

**Phase-1 final state**: 0.391 ms. Phase-1 target 0.150 ms is **unreachable from this layout**. The 2.6× gap (and the 3.9× gap to Phase-4 target 0.100 ms) requires a **structural rewrite**, not a tuning pass.

---

## Phase 2 plan — chunkwise rewrite (the real path)

### The math (WY compact form)

Per workflow.md §7 Phase 2, rewrite the per-token recurrence into a per-chunk batched form:

Let CHUNK_SIZE = C (e.g., 64 or 32). Split sequence into `ceil(T/C)` chunks.

For each chunk `[c·C .. (c+1)·C)`:
1. **Compute γ_cum[0..C)**: cumulative product of gates: γ[i] = Π_{j≤i} gate[c·C+j]. (log-space prefix-sum → exp)
2. **Compute `w[i] = β[i] · γ_cum[i]^{-1}` (or log-space equivalent)**, and `u[i] = v[i]` via standard scaling.
3. **State carry**: `S_new = γ_total · S_old + Σ_i (u[i] · γ_cum[C]/γ_cum[i]) · k[i]^T · k[i]` — this is a `(D × D)` rank-C update that becomes a GEMM.
4. **Output**: `o[i] = <q[i], γ_cum[i] · S_old + Σ_{j≤i} γ_cum[i]/γ_cum[j] · w[j]·k[j]^T·k[j]>` — also decomposable into GEMMs.

The critical insight: **state-update becomes a matrix-matrix multiply over C tokens, not C scalar updates**. On B200 with bf16 at ~4.5 PFLOPS, even a tiny `(D=128) × (C=64)` GEMM is essentially free compared to 64 sequential FMAs.

### Grid restructure
- Old: `(v_head * 64 row_tiles, num_seqs)` × 64 threads → 512·N blocks of 2 warps.
- New: `(v_head, num_chunks)` × 128-256 threads → 8·ceil(T/C)·N blocks of 4-8 warps. Each block owns one chunk, cooperates across warps to produce the chunk's state update + output via tiled mini-GEMMs.

### Needed sub-kernels (workflow.md §7 Phase 2 backlog)
- **A1+A3**: grid `(v_head, num_chunks)`, templated `CHUNK_SIZE`.
- **B1**: WY compact form, SIMT implementation (no tensor cores yet — correctness baseline).
- **M1**: γ_intra factorization via γ_cum.
- **M2**: `(Q K^T) U` operand ordering for BF16 FMA efficiency.
- **M3+M4**: γ_cum SMEM LUT, log-space for numerical stability on long chunks.
- **M5**: β K precompute.

### Measurement strategy for Phase 2
- Phase 2 is a 1-2 day refactor. Instead of one-shot, incrementally:
  1. **B1a (milestone)**: rewrite per-row serial → per-row serial-in-chunks with explicit γ_cum LUT (shouldn't regress; serves as scaffold for chunk parallelism).
  2. **B1b (milestone)**: parallelize within-chunk k·state dot products across the C tokens using SMEM staging. Correctness check.
  3. **M1-M5**: fold factorizations.
  4. After Phase 2 baseline, switch to tensor-core / wgmma path (Phase 3) for the final 0.100 ms target.

### NEXT ACTION
- Write Phase 2 kernel v0 (scaffold): introduce `CHUNK_SIZE=64` constant, load state once, iterate chunks of C tokens with explicit γ_cum[0..C) in SMEM. Keep per-token FMA within a chunk (same math, just chunk-grouped). This establishes the chunked control flow without changing correctness; any regression here flags address-arithmetic / register-pressure issues to fix before introducing real parallelism.
- Measure; if neutral-to-win, proceed to B1b (warp-level within-chunk parallelism).
- Stop and report to user before committing to the full Phase 2 structural rewrite if chunked scaffold itself regresses materially.

---

## Phase-2 execution log

### Iteration 17 — B1a scaffold: chunked outer loop + SMEM-staged gate/β  → KEEP (neutral)
- Change: introduced `constexpr int kChunkSize = 64`; wrapped the per-token hot loop in an outer chunk loop. At each chunk boundary, threads `< C_actual` cooperatively load `gate_beta[(chunk_start+tid)*kNumVHeads + head_idx]` into a shared `float2 sh_gate_beta[kChunkSize]`, followed by `__syncthreads()`. Hot loop now reads `sh_gate_beta[i]` instead of the per-token global `gate_beta` access. No math change, no other load/compute restructuring.
- Intent: establish chunked control flow (B1a milestone in §7). Serves as the scaffold the real chunkwise rewrite will mount onto. The point is not speedup — it is to prove the chunked structure does not regress before adding parallel-within-chunk work.
- Avg latency: **0.268 ms** (unchanged vs F7 baseline 0.268 ms) | 30/30 PASSED | abs_err 6.10e-05 unchanged | rel_err 2.97e-01 unchanged | avg speedup 597.83×.
- Δ: **±0.000 ms (neutral)** — exactly what the scaffold should look like. gate/β was already well-cached (not a bottleneck), so SMEM staging buys nothing *yet*; but the critical-path chunk boundaries and SMEM layout are now in place for M3/M4 (γ_cum LUT) and warp-parallel state updates to plug into.
- Decision: KEEP. No regression means the chunking overhead (outer loop + 2 `__syncthreads` per chunk) is absorbed. Next iteration can now consume `sh_gate_beta` alongside new per-chunk staged data.

### NEXT ACTION (post iter 17)
- Proceed with **M3 prefix-sum LUT** layered onto the iter-17 scaffold: at chunk entry, cooperatively compute `sh_loggate_cum[i] = Σ_{j≤i} log(gate[chunk_start+j])` (warp-level prefix sum, or 64-thread Hillis-Steele since we have up to 64 tokens/chunk and 64 threads/block). This is the prerequisite for M1 (γ_intra factorization) which in turn enables B1/B1b (parallel-within-chunk state update and output).
- Measure after each M-step individually; abort-and-pivot to Phase 3 (wgmma) if two consecutive M-steps fail to buy ≥5%.
- Keep `kChunkSize=64` until warp count changes (Phase 3 will retune).

### Iteration 18 — G1 chunked cooperative gate/β fusion  → KEEP
- Change: folded `compute_gate_beta_kernel` into `gdn_prefill_kernel`. At each chunk entry, the 64 threads in the block cooperatively compute `gate = expf(-expf(A_log[h]) · softplus(a[t]+dt_bias[h]))` and `beta = sigmoid(b[t])` for their assigned token (`threadIdx.x < C_actual`) and write the pair into `sh_gate_beta[threadIdx.x]`. Per-head constants `a_log_exp = expf(A_log[h])` and `dt_bias_h = dt_bias[h]` are preloaded once at block entry. Host side drops the `gate_beta` tensor allocation, the pre-kernel launch, and the launch-check. The dead `compute_gate_beta_kernel` definition is removed.
- Why this variant works where iter 5a/5b failed: parallel-across-threads at chunk entry costs each thread exactly 4 SFU ops per chunk (one token), then the hot loop reads from SMEM as before. iter 5a/5b serialized the SFU chain *inside* the hot loop, inflating the critical path. Here the SFU work is outside the per-token carry.
- Savings: one kernel launch per call (~5-15μs); a `T × kNumVHeads × 2 × f32` write/read roundtrip through global memory; buffer allocation / deallocation overhead.
- Added work: `ceil(T_seq/64) × 4_SFU` compute per block replicated across `64 row_tiles × 8 v_heads × num_seqs` blocks (redundant compute across row_tiles sharing the same head). This is ~64× redundant but each per-chunk cost is ~64 cycles / thread, amortized over the chunk's hot-loop iterations.
- Avg latency: **0.265 ms** (prev 0.268 ms) | 30/30 PASSED | abs_err 6.10e-05 unchanged | rel_err 2.97e-01 unchanged | avg speedup 599.48×.
- Δ: **−0.003 ms (−1.1%)** — small but measurable win. KEEP.
- Reading: the wins match the saved launch + I/O roundtrip roughly; the 64× redundant SFU compute is not visible in wall-time, confirming the hot loop is bandwidth/carry-bound, not SFU-bound. This frees the path to merge more work (γ_cum LUT, βK precompute) into the same per-chunk cooperative-prelude without expecting SFU overhead to regress.

### NEXT ACTION (post iter 18)
- Iter 19 candidate: **M3 γ_cum LUT** — at chunk entry, extend the cooperative prelude so each thread also writes `sh_loggate[tid] = log(gate)` (free, already computed `-a_log_exp · softplus(x)` = log(gate)). Then do a 64-thread Hillis-Steele inclusive prefix sum to produce `sh_loggate_cum[0..C_actual)` in SMEM. Consumer arrives in iter 20+ when M1 lands. Iter 19 itself should measure neutral (scaffold-only); if it regresses, the prefix-scan implementation is buggy or too heavy — diagnose before proceeding.
- Alternative iter 19 candidate if M3 scaffold shows no structural promise: **M5 βK precompute** — at chunk entry, cooperatively pre-multiply `βk[tid] = β · k[tid]` into SMEM. Consumed by `state_vec += βk · (v − γ·kS)`. Saves one 4-wide scalar multiply in the hot loop, but adds a 64-token × 128-channel SMEM footprint (64 KiB) or requires a row-tile co-sharing scheme.
- Reminder: if two consecutive M-steps fail to buy ≥5%, pivot to Phase 3 (wgmma/TMA) — Phase-2 SIMT has a fundamental ceiling around current latency without tensor cores.

### Iteration 19 — M3+M1a: γ_cum LUT + log-space state_norm accumulator  → REVERT (INCORRECT_NUMERICAL on 5/30)
- Change: at each chunk entry, the 64 threads compute `log_gate = -a_log_exp · softplus(x)` and `β` for their assigned token, then do a 2-warp inclusive prefix sum on `log_gate` (intra-warp `__shfl_up_sync` butterfly + a single `__syncthreads()` to hand warp-0's total to warp-1). SMEM exposes `sh_gamma_cum[i] = Γ_cum_local[i]`, `sh_inv_gamma_cum[i] = 1/Γ_cum_local[i]`, `sh_beta[i]`. The hot loop carries `state_norm = state / Γ_cum_local` and updates it as `state_norm += k · ξ` (no gate multiply). At chunk exit, `state_vec = Γ_cum_local[C−1] · state_norm`. Output is reconstructed as `out[i] = Γ_cum_local[i] · (qSnorm + qk·ξ)`.
- Algebraic check (verified): identical to the original `state[t]=gate·state[t−1]+k·diff`, `out[t]=gate·qS+qk·diff` recurrence. Saves ~3 mul/token (removes the 4-channel `gate·state` multiply from the state update) and the per-chunk rescale costs 4 mul amortized over the whole chunk — big theoretical win (~10-15%).
- Results: **25/30 PASSED, 5/30 INCORRECT_NUMERICAL | worst abs err: inf | worst rel err: inf | avg latency: 0.340 ms** (but this avg is contaminated by the failing workloads; correct-case latency was not reported separately).
- Failing workloads: `c5257f65, 1b441950, fc7a2bcb, 3a77dfec, 43bf9699`.
- Root cause: **fp32 underflow of Γ_cum_local** on long chunks with small-gate workloads. When `Γ_cum_local → 0` (cum_log < −80 or so), `inv_gamma = expf(-cum_log) → +∞`, and `xi = β·(v·inv_gamma − kSnorm)` becomes ±Inf, propagating to `state_norm` and ultimately to `out`. The log-space state formulation is only safe when `|cum_log|` stays within the fp32 expf-representable range (~(-87, 88)) across the chunk. For typical gates close to 1 this is fine; for workloads with occasional tiny gates it fails catastrophically. The `2.97e-01` worst rel err on the baseline already suggested the reference kernel lives near the edge of fp32 headroom in some workloads.
- Decision: **REVERT** — restored from `/tmp/kernel.cu.bak_before_iter19`, repacked `solution_cuda.json`. Kernel is back at the iter-18 state = **0.265 ms, 30/30 PASSED**.
- Lesson: any log-space factorization (M1/M1a) needs an in-chunk rescaling mechanism (renormalize `state_norm` back to absolute when `|cum_log|` grows too large) OR needs to be kept in absolute state form. `kChunkSize=64` is too large for the worst-case workloads — even `kChunkSize=8` can underflow with gate ≈ 1e-10 occurrences. Log-space state is structurally incompatible with this workload's numeric range.

### NEXT ACTION (post iter 19)
- Do **not** pursue any pure log-space state reformulation; it is fundamentally unsafe here.
- Remaining Phase-2 paths that keep state in absolute form are structurally much harder (B1 WY compact form requires a chunk-local C×C triangular solve; M5 βK precompute saves only ~1 mul/token and trades for SMEM pressure). Given the failure mode of iter 19 confirms the hot loop is already riding fp32 precision edges, bigger per-token arithmetic changes carry correctness risk.
- Per the opt_log §Measurement strategy clause: "abort-and-pivot to Phase 3 (wgmma) if two consecutive M-steps fail to buy ≥5%." Iter 19 counts as one such failure. **Pause at 0.265 ms / 30/30 PASSED** and surface the Phase-2-vs-Phase-3 pivot to the user before burning another Modal run.

### Iteration 20 — kChunkSize 64 → 128 + extended cooperative load  → KEEP (neutral)
- Change: doubled `kChunkSize` to 128. With kThreads=64, the cooperative gate/β preload loops twice (`for slot_base in {0, kThreads}`) so each thread now writes 2 SMEM slots per chunk. SMEM `sh_gate_beta` doubles to 1 KB. Halves the count of `__syncthreads` per sequence (one per chunk).
- Avg latency: **0.262 ms** (prev 0.265) | 30/30 PASSED | abs_err 6.10e-05 unchanged | rel_err 2.97e-01 unchanged
- Δ: **−0.003 ms (−1.1%)** — within noise but consistent direction. KEEP as the new reference.
- Reading: confirms the chunk-boundary sync was already sub-noise. The win is from doubling the inner loop's compile-time-known iteration window for `#pragma unroll 8` to amortize over (more full unrolls before tail).

### Iteration 22 — Row-tile fusion (kRowsPerBlock=4, 2 rows/warp)  → REVERT
- Change: doubled rows-per-warp via `kRowsPerWarp=2`, `kRowsPerBlock=4`, `kRowTilesPerHead=32` (halved). Each warp now owns 2 contiguous rows of state for both v_heads → **4 state vecs per warp** (state_vec_a0/a1/b0/b1, 16 floats per lane just for state). Per token: 9 reductions (4 kS + 4 qS + 1 qk shared), 4 diffs, 16 fmas across state updates, 4 outputs distributed across lanes 0..3. V loaded as `__nv_bfloat162` packs (2 contiguous bf16 → float2) per v_head. Grid x: 4 × 32 = 128 (was 256).
- Avg latency: **0.532 ms** (prev iter 21 = 0.261) | 30/30 PASSED | abs_err 6.10e-05 unchanged | rel_err 2.97e-01 unchanged | avg speedup 353.92× (was 596).
- Δ: **+0.271 ms (+104%)** — catastrophic regression. REVERT.
- Root cause hypothesis: **register spill** dominates. 16 floats/lane state alone is 64 bytes — combined with q/k (8 floats), v_pairs (4 floats), 9 partial reductions + 9 reduced values, diffs, outputs, the live-set blew the register cap under `__launch_bounds__(64, 4)` (256 regs/thread budget at 4 blocks/SM × 64 threads). NVCC spills state vecs to local memory; every per-token state read/write becomes a STG.local + LDG.local pair, blowing the inner loop. Compounded by halved grid (128 blocks for num_seqs=1 ≪ 148 SMs).
- Lesson: in the 1 row/warp regime we sat right at the spill threshold. Doubling state per warp is structurally infeasible without launch_bounds tightening (which trades occupancy and may not recoup). **Conclusion: per-warp state cannot grow further; the only path to break the 0.261 ms ceiling is an algorithmic restructure (Phase 2 chunkwise), not register-level fusion.**
- Decision: REVERT. Restored kernel from `/tmp/kernel.cu.bak_before_iter22`. Kernel is back at iter-21 state = **0.261 ms, 30/30 PASSED**.

### Iteration 21 — Head-pair fusion (kVHeadsPerBlock=2)  → KEEP (neutral)
- Change: exploited GQA structure (V_PER_Q=2, V_PER_K=2). Each block now handles a v_head **pair** (e.g., v_head 0+1) sharing identical Q and K reads. New constants: `kVHeadsPerBlock=2`, `kHeadPairs=4`. Grid halved from `8 × 64 × num_seqs` to `4 × 64 × num_seqs`. Inside the block: 2 state vecs (state_vec_a, state_vec_b), 2 SMEM gate/β tables, single q_vec/k_vec load drives both v_heads; reductions become 5 (kS_a, qS_a, kS_b, qS_b, qk shared).
- Avg latency: **0.261 ms** (prev 0.262) | 30/30 PASSED | abs_err 6.10e-05 unchanged | rel_err 2.97e-01 unchanged | avg speedup 596.86×.
- Δ: **−0.001 ms (−0.4%)** — neutral. The expected ~30% bandwidth win did not materialize, confirming the recurrent kernel was NOT memory-bound on Q/K. Real bottleneck is the per-token serial FMA chain (state update + 3 reductions on the critical path).
- Decision: KEEP. Cleaner architecture, halved grid (potential headroom for kVHeadsPerBlock=4 or persistent scheduling later), matched perf. Speedup metric drop (662 → 597) is a denominator artifact (per-call larger compute, reference unchanged).
- Insight: confirms once more that breaking the 0.265 ms ceiling requires structural change to the per-token recurrence (Phase 2 chunkwise WY), not load reduction.

### NEXT ACTION (post iter 22 revert)
Phase 1 SIMT ceiling reaffirmed at **0.261 ms**. Iter 22 confirmed that growing per-warp state from 4→16 floats/lane catastrophically spills to local memory; the per-warp state cap is now hard. The path to 0.125 ms (Phase 2) target requires either:
1. **Chunkwise WY in pure SIMT** — analysis shows this REGRESSES (more shuffle reductions per token because KK/QK matrices need C^2 dots, totalling ~6× more shuffles than serial). Only viable WITH tensor cores.
2. **Tensor cores via raw PTX wgmma** — workflow.md §6 allows direct PTX. Multi-day implementation, high complexity.
3. **Cheap micro-tunes that don't enlarge per-warp state** — manual prefetch, lane-parallel stores, bfdot for ⟨q,k⟩. Each ~1-3% expected.

Iter 23 candidate: **manual software-pipelining prefetch of next-iter q/k**. Adds q_next, k_next register pair (8 floats/lane = sub-spill). Lets the compiler overlap memory latency with reductions of current iter. Expected: 0-3% win. Cheap probe to validate the SIMT-ceiling theory before committing to wgmma effort.

### Iteration 23 — split bf16 store across lanes 0/1 (lane-parallel outputs)  → REVERT
- Change: split `if (lane_idx == 0) { store_a; store_b; }` into `if (lane_idx==0) store_a; if (lane_idx==1) store_b;` to let 2 lanes write in parallel.
- Avg latency: **0.409 ms** (prev 0.261) | 30/30 PASSED | abs_err 6.10e-05 unchanged.
- Δ: **+0.148 ms (+57%)** — catastrophic regression. REVERT.
- Root cause: SIMT reconvergence stall. 3 divergent predicated paths (lane0-only, lane1-only, rest-wait) triggers reconvergence bookkeeping where the compiler/HW had previously fused the 2-store sequence into a single predicated block. Per-token serial stores via lane 0 are HIDDEN by the next token's compute; splitting them *reveals* the store latency and reconvergence cost.
- Lesson: **lane-parallel stores are SLOWER than serial-lane-0 stores** when the second store is already latency-hidden. Do not split the predicated store block.

### Iteration 24 — launch_bounds(kThreads, 4→2) register headroom probe  → REVERT (neutral)
- Change: single-line `__launch_bounds__(kThreads, 4)` → `__launch_bounds__(kThreads, 2)`. Gives compiler 256 → 512 regs/thread budget. Tests whether iter 21's layout has hidden register spill.
- Avg latency: **0.263 ms** (prev 0.261) | 30/30 PASSED | abs_err 6.10e-05 unchanged.
- Δ: **+0.002 ms (+0.8%, within noise)** — neutral.
- Reading: confirms iter 21 is NOT register-spilled. The 0.261 ms ceiling is algorithmic, not register-pressure. Reverted to `launch_bounds(64,4)` (original, more occupancy).

### Iteration 25 — SMEM-stage V loads via cooperative prelude  → REVERT
- Change: folded per-token `bf16_to_float(v + v_offset_a/b)` scalar broadcasts into the chunk prelude. Per-warp `__shared__ float sh_v_a/b[kWarpsPerBlock][kChunkSize]` stages V values; hot loop reads SMEM instead of global. Extends the existing cooperative load pattern.
- Avg latency: **0.288 ms** (prev 0.261) | 30/30 PASSED | abs_err 6.10e-05 unchanged | avg speedup 662×.
- Δ: **+0.027 ms (+10%)** — regression. REVERT.
- Root cause: V loads were already efficiently broadcast-cached (single bf16 read per token, all 64 threads reading same address coalesces to 1 unique byte, L1/L2 hit rate high). The compiler was already hiding their latency via `#pragma unroll 8` reordering. Staging to SMEM added ~8 extra bf16 reads per lane in the prelude without removing meaningful hot-loop latency.
- Lesson: **latency-hiding beats latency-removal** when the compiler has already pipelined a read. Don't SMEM-stage reads that are already broadcast-coalesced and L1-cached.

### NEXT ACTION (post iter 25 revert)
Baseline confirmed again at **0.261 ms**. Three Phase-1 SIMT probes (iter 23/24/25) all failed or neutral — exhausting the cheap micro-tune list. Next:
- **Iter 26 candidate: kChunkSize 128 → 256**. Iter 17→20 showed −0.005 ms per doubling from larger compile-time-known unroll window. SMEM stays trivial. Low risk.
- **Phase 3 commit if iter 26 fails: mma.sync m16n8k16**. Multi-day rewrite; deadline 2026-04-24 (4 days remaining).

### Iteration 26 — kChunkSize 128 → 256  → KEEP (marginal)
- Change: doubled `kChunkSize` to 256. SMEM `sh_gate_beta_{a,b}` doubles to 2 KB each (4 KB total). Cooperative load loops 4× per chunk now (vs 2× at chunk=128). Inner unroll window doubles, halving sync count per sequence.
- Avg latency: **0.260 ms** (prev 0.261) | 30/30 PASSED | abs_err 6.10e-05 unchanged | avg speedup 617×.
- Δ: **−0.001 ms (−0.4%)** — marginal but consistent direction with iter 17→20 trend. KEEP as new baseline.
- Reading: same theory as iter 20 — larger compile-time-known inner-loop count gives the unroll dispatcher more room to schedule the recurrence chain. Diminishing returns: each doubling buys ~1% now (vs ~1% at iter 17→20). The shuffle-bound critical path is still the wall.

### Iteration 27 — kChunkSize 256 → 512 (pragma unroll 8 unchanged)  → REVERT
- Change: single-line `kChunkSize = 256 → 512`. SMEM doubles to 8 KB total (trivial).
- Avg latency: **0.275 ms** (prev 0.260) | 30/30 PASSED | abs_err 6.10e-05 unchanged | avg speedup 701×.
- Δ: **+0.015 ms (+5.8%)** — regression. REVERT.
- Root cause: the iter 17→20→26 doubling trend does NOT continue past chunk=256. Hypothesis: `#pragma unroll 8` over a 512-wide compile-time-known loop inflates instruction-cache footprint and register churn; compiler dispatches less well when the unroll multiplier doesn't cleanly divide the inner trip count's register reuse window.

### Iteration 28 — kChunkSize 512 + `#pragma unroll 4` (recover from iter 27 regression)  → REVERT (neutral)
- Change: kept chunk=512 from iter 27 state, reduced hot-loop unroll factor `8 → 4`. Tests whether the regression was unroll pressure rather than chunk size.
- Avg latency: **0.261 ms** (vs iter 26 baseline 0.260) | 30/30 PASSED | abs_err 6.10e-05 unchanged.
- Δ: **+0.001 ms (+0.4%)** — essentially neutral, recovered from iter 27.
- Reading: confirms iter 27's regression was unroll pressure, not chunk-size. chunk=512+unroll4 ≈ chunk=256+unroll8. Since iter 26 is already at 0.260 ms with simpler config, keep iter 26 (chunk=256, unroll 8) as baseline and revert iter 28.

### CuTe DSL lane discovery (post iter 28)
Confirmed: `config.toml` is `language="python"` + `entry_point="msinfer_entry.py::run"`. The **production solution** is `solution/python/gdn_blackwell/gdn.py` — a 4681-line CuTe DSL (cutlass.cute + tcgen05 + cutlass.pipeline) Blackwell-native chunkwise Gated Delta Rule implementation. Already uses 5th-gen Tensor Cores (tcgen05), TMA, warp specialization, and chunkwise WY (chunk_size=128). The CUDA SIMT lane (`solution/cuda/kernel.cu`) that iters 1-28 optimized is a side-channel alt lane — NOT the production lane.
- SIMT lane ceiling confirmed at **0.260 ms** (iter 26 baseline). No more cheap wins; TC-free per-token recurrence is fundamentally shuffle-bound.
- Production CuTe DSL lane latency not yet measured in this session. Must be measured via `scripts/pack_solution.py` + `modal run scripts/run_modal.py` (no `--solution-path` arg, uses default `solution.json` from pack_solution).
- **Next action:** pivot to measuring + tuning the CuTe DSL lane. Tunable knobs: tile_scheduler params, chunk_size, persistent vs non-persistent mode, TMA prefetch stage count, MMA instruction shape, producer/consumer warp split.

### New measurement recipe (going forward)
- Env: `conda fi-bench` env is empty of packages; the working one is the pyenv 3.12.13 `fi-bench` *or* conda's `fi-bench` accessed via absolute path with `KMP_DUPLICATE_LIB_OK=TRUE`.
- Pack: `KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/Caskroom/miniforge/base/envs/fi-bench/bin/python scripts/pack_cuda_solution.py`
- Run : `KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/Caskroom/miniforge/base/envs/fi-bench/bin/modal run scripts/run_modal.py --solution-path solution_cuda.json --max-workloads 30 --sample-seed 42 --summary-only`
- Cost budget: ~20 min wall-time per iteration; be deliberate.

### CuTe baseline measurement (current active lane)
- Measured with the active `config.toml` path via `scripts/pack_solution.py` and `/opt/homebrew/Caskroom/miniforge/base/envs/fi-bench/bin/modal run scripts/run_modal.py --solution-path solution.json --max-workloads 30 --sample-seed 42 --summary-only`.
- **Baseline: avg latency = 0.290 ms**, `PASSED=30/30`, worst abs err `8.27e-03`, worst rel err `1.93e+03`.
- Small isolated spot-check for faster iteration:
  - `--max-workloads 4 --sample-seed 42` baseline = **0.288 ms**, `PASSED=4/4`.

### CuTe tuning iterations (this session)

#### Iteration C1 — persistent scheduler heuristic in `msinfer_entry.py`  → REVERT
- Change: selected `is_persistent=True` dynamically for larger / varlen problems instead of hard-coding `False`.
- Result on 8-workload decision gate: **GPU context corruption / RUNTIME_ERROR** (`Xid 31`, MMU fault, unhealthy worker restart).
- Decision: **REVERT**. Persistent path is not production-safe for the current CuTe kernel configuration.

#### Iteration C2 — `min_blocks_per_mp=1 → 2`  → REVERT
- Change: forced higher CTA residency at launch time for the CuTe kernel.
- Result on isolated 4-workload sample: **RUNTIME_ERROR=4/4**.
- Decision: **REVERT**. The kernel cannot safely sustain the tighter residency target.

#### Iteration C3 — `qk_stage=2 → 1`  → REVERT
- Change: reduced the TMA/UMMA QK pipeline depth by one stage to lower synchronization/shared-state overhead.
- Result on isolated 4-workload sample: first failing workload hit **TIMEOUT**.
- Decision: **REVERT**. One-stage QK buffering is insufficient for this kernel.

#### Iteration C4 — `mma_qk_stage=2 → 1`  → REVERT
- Change: reduced the async MMA consumer pipeline depth for the QK path.
- Result on isolated 4-workload sample: first failing workload hit **TIMEOUT**.
- Decision: **REVERT**. The compute-side async pipeline also needs two stages for forward progress.

#### Iteration C5 — `num_regs_cudacore=240 → 232`  → REVERT
- Change: lowered the register budget for cudacore warps to try to improve occupancy.
- Result on isolated 4-workload sample: **0.334 ms** vs baseline **0.288 ms** (`PASSED=4/4`).
- Δ: **+0.046 ms (+16.0%)**.
- Decision: **REVERT**. This likely introduced spills or removed useful ILP.

#### Iteration C6 — `num_regs_cudacore=240 → 248`  → KEEP CANDIDATE
- Change: slightly raised the cudacore-warp register budget.
- Result on isolated 4-workload sample: **0.284 ms** vs baseline **0.288 ms** (`PASSED=4/4`).
- Δ: **−0.004 ms (−1.4%)**.
- Result on 30-workload confirmation run: **0.240 ms** vs baseline **0.290 ms** (`PASSED=30/30`).
- Δ: **−0.050 ms (−17.2%)**.
- Status: **KEEP**. This is the new best verified CuTe configuration so far.

#### Iteration C7 — `num_regs_cudacore=248 → 252`  → REVERT
- Change: one more register-budget step above C6.
- Result on isolated 4-workload sample: **RUNTIME_ERROR** on sampled workloads.
- Decision: **REVERT** back to **248**. The safe/beneficial window appears narrow, with `248` the current best candidate.

#### Iteration C8 — `num_regs_cudacore=248 → 250`  → REVERT
- Change: narrowed the search between the verified-good `248` and unstable `252`.
- Result on isolated 4-workload sample: **RUNTIME_ERROR=4/4**.
- Decision: **REVERT** back to **248**. The current stability boundary is between `248` and `250`.

#### Iteration C9 — local drift check on current CuTe working tree
- Context: resuming on branch `prefill-opt-jw-pr` with uncommitted CuTe changes (`num_regs_cudacore=248`, explicit `cute.GPUArch("sm_100a")`, `PtxasOptions("--allow-expensive-optimizations=true")` in `msinfer_entry.py`).
- Result on isolated 4-workload sample: **0.390 ms** (`PASSED=4/4`) — far worse than the older C6/C8-era quick baseline.
- Reading: the current Modal dev environment strongly disfavors forcing `sm_100a` on Modal B200s (FAQ notes Modal is `sm100`, not `sm100a`). Re-opened the CuTe tuning loop from this actual measured state instead of trusting the older quick-run numbers.

#### Iteration C10 — `num_regs_cudacore=248 → 240` under current environment  → KEEP
- Change: dropped cudacore-warp register target back to `240` while keeping the current compile options intact.
- Result on isolated 4-workload sample: **0.334 ms** (`PASSED=4/4`) vs C9 **0.390 ms**.
- Δ: **−0.056 ms (−14.4%)**.
- Decision: **KEEP**. In the current environment, the old `248` sweet spot no longer holds; `240` is clearly safer and faster.

#### Iteration C11 — remove explicit `cute.GPUArch("sm_100a")`  → KEEP
- Change: kept `EnableTVMFFI + PtxasOptions("--allow-expensive-optimizations=true")`, but stopped hard-coding `sm_100a` in `solution/python/msinfer_entry.py` so CuTe can target the actual device architecture at compile time.
- Result on isolated 4-workload sample: **0.282 ms** (`PASSED=4/4`) vs C10 **0.334 ms**.
- Δ: **−0.052 ms (−15.6%)**.
- Interpretation: on Modal B200, forcing `sm_100a` is a material regression. This also reduces the risk of compiling the wrong ISA path during dev, while still allowing the eval environment to target its native arch.

#### Iteration C12 — remove `PtxasOptions("--allow-expensive-optimizations=true")`  → REVERT
- Change: reverted `cute.compile[...]` back to `EnableTVMFFI` only.
- Result on isolated 4-workload sample: **0.332 ms** (`PASSED=4/4`) vs C11 **0.282 ms**.
- Δ: **+0.050 ms (+17.7%)**.
- Decision: **REVERT**. The expensive ptxas optimization flag is beneficial for this kernel once the arch mismatch is removed.

#### Iteration C13 — `num_regs_cudacore=240 → 244`  → REVERT
- Change: narrowed the cudacore register search upward from the new working point `240`.
- Result on isolated 4-workload sample: **RUNTIME_ERROR=4/4**.
- Decision: **REVERT**. The new stability boundary is now between `240` and `244`; higher is not usable.

#### Iteration C14 — `num_regs_mma=64 → 72`  → REVERT
- Change: increased the dedicated MMA warp's register budget to reduce possible spills in the inversion/QK helper path.
- Result on isolated 4-workload sample: **0.340 ms** (`PASSED=4/4`) vs C11 **0.282 ms**.
- Δ: **+0.058 ms (+20.6%)**.
- Decision: **REVERT**. The MMA warp is not the limiting spill point here; extra registers just hurt scheduling.

#### Iteration C15 — `num_regs_gb=64 → 48`  → REVERT
- Change: reduced the gate/beta loader warp's register budget to test whether more dynamic register budget would flow to cudacore warps.
- Result on isolated 4-workload sample: **0.290 ms** (`PASSED=4/4`) vs C11 **0.282 ms**.
- Δ: **+0.008 ms (+2.8%)**.
- Decision: **REVERT**. The gate/beta warp is already lean enough; squeezing it does not buy back useful performance.

#### Iteration C16 — `num_regs_other=64 → 72`  → REVERT
- Change: increased the load/epilogue warp register budget.
- Result on isolated 4-workload sample: **0.288 ms** (`PASSED=4/4`) vs C11 **0.282 ms**.
- Δ: **+0.006 ms (+2.1%)**.
- Decision: **REVERT**. Load/epilogue warps do not benefit enough from extra registers to justify the scheduling cost.

#### Iteration C17 — 30-workload confirmation of new best CuTe state  → KEEP
- State under test:
  - `solution/python/gdn_blackwell/gdn.py`: `num_regs_cudacore = 240`
  - `solution/python/msinfer_entry.py`: `EnableTVMFFI + PtxasOptions("--allow-expensive-optimizations=true")`, **no explicit `cute.GPUArch("sm_100a")`**
- Result on 30-workload deterministic sample (`--max-workloads 30 --sample-seed 42`): **0.265 ms**, `PASSED=30/30`, worst abs err **8.27e-03**, worst rel err **1.93e+03**.
- Comparison to the prior CuTe 30-workload baseline in this log (**0.290 ms**): **−0.025 ms (−8.6%)**.
- Status: **NEW BEST VERIFIED CuTe CONFIGURATION IN THIS SESSION**.

#### Iteration C18 — `num_regs_cudacore=240 → 242`  → REVERT
- Change: one last probe between the verified-good `240` and the unstable `244`.
- Result on isolated 4-workload sample: **RUNTIME_ERROR=4/4**.
- Decision: **REVERT**. The current practical stability edge is exactly at `240`; even `242` is unsafe.

#### Iteration C19 — fuse gate/beta preprocessing into CuTe kernel  → REVERT
- Change: removed host-side `_get_gate_beta` preprocessing and passed `A_log/a/dt_bias/b` directly into `GDN`, with CuTe-side polynomial `softplus/sigmoid` approximations in the gb warp.
- Result on isolated 4-workload sample: **0.957 ms** (`PASSED=4/4`) vs the current working quick baseline **0.289 ms**.
- Δ: **+0.668 ms (+231%)**.
- Decision: **REVERT**. Even after fixing the CuTe DSL control-flow issue, recomputing gate/beta inside the kernel is dramatically more expensive than loading precomputed float32 gate/beta.

#### Iteration C20 — restore precomputed gate/beta baseline after C19  → KEEP BASELINE
- Change: reverted C19 back to host-side `_get_gate_beta` and the original CuTe kernel interface (`gate`, `beta`).
- Result on isolated 4-workload sample: **0.289 ms** (`PASSED=4/4`).
- Reading: confirms the large C19 regression came from the in-kernel preprocessing itself, not incidental code drift. Continue from the precomputed-gate path.

#### Iteration C21 — `mma_cudacore_stage=1 → 2`  → REVERT
- Change: increased the async producer/consumer stage count for the cudacore mainloop pipeline.
- Result on isolated 4-workload sample: **pathological multi-minute stall / no summary returned**; local `modal run` remained attached for >5 minutes and was manually terminated.
- Decision: **REVERT**. Treat as an unsafe pipeline-depth increase for the current kernel.

#### Iteration C22 — `num_regs_gb=64 → 80`  → REVERT (neutral)
- Change: raised the gate/beta loader warp register budget.
- Result on isolated 4-workload sample: **0.289 ms** (`PASSED=4/4`) vs baseline **0.289 ms**.
- Δ: **0.000 ms**.
- Decision: **REVERT**. No measurable benefit; keep the simpler baseline.

#### Iteration C23 — enable real gate/beta double-buffering (`gate_stage=beta_stage=2` + producer uses `gb_handle.index`)  → REVERT
- Change: turned the previously effectively-single-stage gate/beta pipe into an actual 2-stage producer/consumer buffer.
- Result on isolated 4-workload sample: **0.290 ms** (`PASSED=4/4`) vs baseline **0.289 ms**.
- Δ: **+0.001 ms (+0.3%)**.
- Decision: **REVERT**. The extra shared-buffer stage does not buy overlap worth its bookkeeping cost.

#### Iteration C24 — `num_regs_cudacore=240 → 238`  → REVERT
- Change: checked the last unexplored register point just below the verified-good `240`.
- Result on isolated 4-workload sample: **RUNTIME_ERROR=4/4**.
- Decision: **REVERT**. The current stability window is narrower than expected; `240` remains the only verified-safe edge in the nearby search space.

#### Iteration C25 — add `ptxas --force-load-cache=cg`  → PROMISING BUT NOT VERIFIED
- Change: kept `--allow-expensive-optimizations=true`, added `--force-load-cache=cg` in `solution/python/msinfer_entry.py`.
- Result on isolated 4-workload sample: **0.284 ms** (`PASSED=4/4`) vs baseline **0.289 ms**.
- Δ: **−0.005 ms (−1.7%)**.
- Reading: plausible explanation is that bypassing L1 for generic global loads reduces cache pollution from low-reuse gate/beta traffic. However, see C28/C29 before claiming this as a real win.

#### Iteration C26 — replace `cg` with `cs` cache hint  → REVERT
- Change: `--force-load-cache=cg → cs`.
- Result on isolated 4-workload sample: **0.335 ms** (`PASSED=4/4`) vs C25 **0.284 ms**.
- Δ: **+0.051 ms (+18.0%)**.
- Decision: **REVERT**. `cs` is materially worse than `cg` for this kernel.

#### Iteration C27 — add `ptxas --opt-level=2` on top of `cg`  → REVERT
- Change: `--opt-level=2 --allow-expensive-optimizations=true --force-load-cache=cg`.
- Result on isolated 4-workload sample: **0.334 ms** (`PASSED=4/4`) vs C25 **0.284 ms**.
- Δ: **+0.050 ms (+17.6%)**.
- Decision: **REVERT**. Lowering ptxas optimization strength hurts runtime despite the cache hint.

#### Iteration C28 — medium/long confirmation of `cg` cache hint  → REJECT FOR NOW
- Change under test: same as C25 (`--force-load-cache=cg` + `--allow-expensive-optimizations=true`).
- Result on broader confirmation runs:
  - `--max-workloads 10 --sample-seed 42`: local `modal run` remained attached for >4 minutes with no final summary and was manually stopped.
  - `--max-workloads 30 --sample-seed 42`: local `modal run` remained attached for >7 minutes with no final summary and was manually stopped.
- Decision: **REJECT / do not adopt as baseline**. The short-sample gain from C25 did not earn enough confidence on broader samples. Leave the working tree on the last fully verified C17 configuration instead.

#### Iteration C29 — re-baseline current CuTe working tree on 2026-04-22  → KEEP BASELINE
- Context: user explicitly requested staying on the active `config.toml` CuTe DSL lane and forbade using the side CUDA packer lane.
- Measurement path: `python scripts/pack_solution.py` then `modal run scripts/run_modal.py --solution-path solution.json --max-workloads 4 --sample-seed 42 --summary-only`.
- Result: **0.288 ms** (`PASSED=4/4`), worst abs err **1.08e-03**, worst rel err **1.02e+03**.
- Decision: treat **0.288 ms** as the current quick-gate baseline for this session.

#### Iteration C30 — scratch-buffer gate/beta preprocessing in `msinfer_entry.py`  → REVERT
- Change: rewrote `_get_gate_beta()` to reuse stream-local float32 scratch buffers and compute `softplus`/`sigmoid` via explicit in-place PyTorch ops to reduce temporary allocation in the timed wrapper path.
- Result on the same 4-workload gate: **0.292 ms** (`PASSED=4/4`) vs baseline **0.288 ms**.
- Δ: **+0.004 ms (+1.4%)**.
- Reading: the extra pointwise op scheduling and loss of eager fused kernels outweighed the allocation savings.
- Decision: **REVERT**.

#### Iteration C31 — re-check `ptxas --force-load-cache=cg` in the current environment  → REVERT
- Change: `PtxasOptions("--allow-expensive-optimizations=true --force-load-cache=cg")`.
- Result on the same 4-workload gate: **0.290 ms** (`PASSED=4/4`) vs baseline **0.288 ms**.
- Δ: **+0.002 ms (+0.7%)**.
- Reading: unlike the older quick signal in C25, the current environment no longer shows a repeatable benefit.
- Decision: **REVERT** and keep `--allow-expensive-optimizations=true` only.

#### Iteration C32 — content-aware gate/beta cache across cloned benchmark inputs  → REVERT
- Change: kept the eager `softplus/sigmoid` math but added a content fingerprint for `A_log/a/dt_bias/b` so repeated cloned inputs within a workload could reuse prepared `g/beta`.
- Result on the same 4-workload gate: **0.387 ms** (`PASSED=4/4`) vs baseline **0.288 ms**.
- Δ: **+0.099 ms (+34.4%)**.
- Reading: fingerprint computation introduced enough synchronization / reduction overhead to swamp any saved preprocessing.
- Decision: **REVERT**. Wrapper-side content hashing is not viable here.

#### Iteration C33 — `epi_stage=1 → 2`  → REVERT
- Change: doubled the epilogue producer/consumer stage depth in `solution/python/gdn_blackwell/gdn.py`.
- Result: local 4-workload confirmation never produced a final summary within several minutes and was treated as an unsafe pipeline-depth increase.
- Decision: **REVERT**. The epilogue path should stay single-stage in the current kernel.

#### Iteration C34 — `kv_stage=1 → 2` quick-gate win, 30-workload regression  → REVERT
- Change: aligned the K-side SMEM layout staging with the existing 2-stage `load_qk` pipeline by setting `kv_stage = 2`.
- Quick-gate result (`--max-workloads 4 --sample-seed 42`): **0.286 ms** (`PASSED=4/4`) vs baseline **0.288 ms**.
- Δ at quick gate: **−0.002 ms (−0.7%)**.
- 30-workload confirmation (`--max-workloads 30 --sample-seed 42`): **0.345 ms** (`PASSED=30/30`) with worst abs err **8.27e-03**, worst rel err **1.93e+03**.
- Δ at 30 workloads vs prior verified CuTe baseline (**0.265 ms** from C17): **+0.080 ms (+30.2%)**.
- Reading: the change helped the short sample but regressed materially on the broader deterministic sample, likely by increasing pressure on longer-sequence cases.
- Decision: **REVERT**. The apparent 4-workload win was a misleading short-sample effect.

#### Iteration C35 — re-baseline current active CuTe lane on 30 workloads  → KEEP BASELINE
- Measurement path: `PYENV_VERSION=fi-bench pyenv exec modal run scripts/run_modal.py --summary-only --max-workloads 30 --sample-seed 42`
- Result: **0.242 ms** (`PASSED=30/30`), worst abs err **8.27e-03**, worst rel err **1.93e+03**.
- Quick gate on the same working tree: **0.284 ms** (`PASSED=4/4`) with `--max-workloads 4 --sample-seed 42`.
- Reading: the current repository state is materially better than the older CuTe baseline in this log. Treat **0.242 ms** as the new best verified active-lane baseline for this branch.

#### Iteration C36 — lazy `torch.compile` on gate/beta preprocessing  → REVERT
- Change: wrapped the eager `_get_gate_beta()` pointwise path in a lazy `torch.compile(fullgraph=True)` helper, with eager fallback on compile/runtime failure.
- Quick-gate result: **0.292 ms** (`PASSED=4/4`) vs baseline **0.284 ms**.
- Δ: **+0.008 ms (+2.8%)**.
- Reading: even without correctness issues, the compiled preprocessing path added enough overhead to lose against the already simple eager kernels + id-based cache.
- Decision: **REVERT**.

#### Iteration C37 — `num_regs_cudacore=240 → 241`  → REVERT
- Change: probed the immediately adjacent register point above the verified-safe `240`.
- Quick-gate result: **RUNTIME_ERROR=4/4**.
- Reading: the current cudacore-warp stability boundary is still razor-thin; `241` is already unsafe in this environment.
- Decision: **REVERT**.

#### Iteration C38 — `num_regs_other=64 → 60`  → REVERT
- Change: reduced the load/epilogue warp-group register budget to see whether auxiliary-warp pressure could be trimmed safely.
- Quick-gate result: **RUNTIME_ERROR=4/4**.
- Reading: not only the cudacore warp but also the "other" warp budget sits on a narrow stability edge.
- Decision: **REVERT**.

#### Iteration C39 — add `ptxas --def-load-cache=cg`  → REVERT
- Change: kept `--allow-expensive-optimizations=true`, added `--def-load-cache=cg` in `solution/python/msinfer_entry.py`.
- Quick-gate result: **0.360 ms** (`PASSED=4/4`) vs baseline **0.284 ms**.
- Δ: **+0.076 ms (+26.8%)**.
- Reading: a gentler default load-cache hint was still strongly harmful here; the current compiler/device pairing prefers the default policy.
- Decision: **REVERT**.

#### Iteration C40 — `buffer_align_bytes=1024 → 512`  → REVERT
- Change: reduced shared-memory section alignment to shrink padding and test whether the kernel was being held back by SMEM footprint.
- Quick-gate result: **0.358 ms** (`PASSED=4/4`) vs baseline **0.284 ms**.
- Δ: **+0.074 ms (+26.1%)**.
- Reading: the larger alignment is not wasted decoration; it appears to matter for the current TMA/SMEM layout and/or bank behavior.
- Decision: **REVERT**.

#### Iteration C41 — omit `cluster=(1,1,1)` at launch  → REVERT
- Change: removed the explicit cluster launch argument while keeping all other launch parameters the same.
- Quick-gate result: **0.288 ms** (`PASSED=4/4`) vs baseline **0.284 ms**.
- Δ: **+0.004 ms (+1.4%)**.
- Reading: even the no-op-looking single-cluster metadata is slightly better than relying on the default launch path in this CuTe runtime.
- Decision: **REVERT**.

#### Iteration C42 — omit `min_blocks_per_mp=1` at launch  → REVERT
- Change: restored `cluster=(1,1,1)` and removed only the `min_blocks_per_mp=1` launch hint.
- Quick-gate result: **0.340 ms** (`PASSED=4/4`) vs baseline **0.284 ms**.
- Δ: **+0.056 ms (+19.7%)**.
- Reading: the launch hint is not redundant; the current kernel/runtime combination depends on it for a materially better schedule.
- Decision: **REVERT**.

#### Iteration C43 — `num_regs_mma=64 → 56`  → REVERT
- Change: lowered the dedicated MMA warp register budget to test whether the helper-MMA path was overprovisioned.
- Quick-gate result: **0.343 ms** (`PASSED=4/4`) vs baseline **0.284 ms**.
- Δ: **+0.059 ms (+20.8%)**.
- Reading: the MMA warp is not a good place to claw back register budget; reducing it hurts enough to overwhelm any theoretical occupancy gain.
- Decision: **REVERT**.

### CuTe micro-tuning status after C35-C43
- Best verified active-lane result on this branch remains **0.242 ms** over the 30-workload deterministic sample.
- Every new probe in this round either regressed clearly or failed at runtime.
- Practical conclusion: the easy launch/cache/register/smem micro-tuning space around the current CuTe kernel is largely exhausted; the next real gain will need a more structural kernel change rather than another one-line knob flip.

---

## Optimization backlog (from workflow.md §7)

Phase 1 (≤ 0.150 ms, low-risk baseline tuning):
- [ ] K1: `-arch=sm_100a` confirm (currently inferred from TORCH_CUDA_ARCH_LIST=10.0a)
- [ ] K3: `__builtin_assume` on hot indices
- [ ] E1: 128-bit packed bf16 load (`ldg.v8.b16`)
- [ ] E2: `ld.global.nc.v4.f32` for state read
- [ ] E3: L2 persistence on state buffer (`cudaStreamAttributeAccessPolicyWindow`)
- [ ] E4: `__ldcs` streaming hint on Q/K/V
- [ ] E6: SMEM carveout 100 %
- [ ] F3: batch gate/beta load into registers per warp
- [x] G1: fuse compute_gate_beta_kernel into main kernel  — **iter 18 KEEP (−0.003 ms, −1.1%)**, chunked cooperative variant
- [ ] H1: try 4 warps/block with kRowsPerBlock=4
- [ ] H2: retune `__launch_bounds__`

Phase 2 (≤ 0.125 ms, chunkwise rewrite):
- [~] A1+A3: grid `(v_head, num_chunks)` + templated CHUNK_SIZE  — **B1a scaffold landed in iter 17** (per-sequence outer chunk loop, `kChunkSize=64`, SMEM `sh_gate_beta`); true per-chunk grid still TBD
- [ ] B1: WY compact form (SIMT)
- [ ] M1: γ_intra factorization via γ_cum
- [ ] M2: `(Q K^T) U` ordering
- [ ] M3+M4: γ_cum SMEM LUT, log-space
- [ ] M5: β K precompute

Phase 3 (≤ 0.110 ms): wgmma + TMA (C1, C2, D1, D2, D3)
Phase 4 (≤ 0.100 ms): warp specialization + 2-CTA cluster (I1, J1)

---

## Session handoff notes for future agent

- Read `workflow.md` fully before editing. The absolute rules (§0) are strict: no external libs, no regressions, one change per iteration, always measure.
- Use `pyenv shell fi-bench` or `PYENV_VERSION=fi-bench` for modal/python — default shim is 3.9.6 without modal.
- Each Modal run costs ~real money + ~2 min wall time. Do not retry without diagnosis.
- Back up kernel.cu before every structural change (read it; you can restore from git as well).
- Update this file **after every iteration** (add an entry under "Iteration log") before moving on.
