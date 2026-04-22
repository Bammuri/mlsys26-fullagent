# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a starter kit for the **FlashInfer AI Kernel Generation Contest @ MLSys 2026** — a competition to create high-performance GPU kernels for LLM operations on NVIDIA Blackwell B200 GPUs. Three tracks: fused_moe, sparse_attention, gated_delta_net.

## Common Commands

```bash
# Pack solution source files into solution.json for submission
python scripts/pack_solution.py

# Run benchmarks on local GPU (requires FIB_DATASET_PATH env var)
python scripts/run_local.py

# Run benchmarks on Modal B200 cloud instances
modal run scripts/run_modal.py

# Run FlashInfer-Bench evaluation directly (example: GDN decode)
flashinfer-bench run \
  --local /path/to/mlsys26-contest \
  --definitions gdn_decode_qk4_v8_d128_k_last \
  --use-isolated-runner --timeout 300

# Run sanitizers on a solution (from Python)
# flashinfer_bench.agents.flashinfer_bench_run_sanitizer(solution, workload, sanitizer_types=["memcheck", "racecheck", "synccheck", "initcheck"])

# NCU profiling (from Python)
# flashinfer_bench.agents.flashinfer_bench_run_ncu(solution, workload, set="detailed", page="details")
```

## Environment Setup

```bash
conda create -n fi-bench python=3.12
conda activate fi-bench
pip install flashinfer-bench modal
export FIB_DATASET_PATH=/path/to/mlsys26-contest
```

## Architecture

### Workflow

1. Edit `config.toml` — set track (`definition`), language (`triton`/`cuda`), and `entry_point`
2. Implement kernel in `solution/triton/kernel.py` or `solution/cuda/kernel.cu` + `binding.py`
3. Run `python scripts/pack_solution.py` — reads config.toml, packs source files into `solution.json`
4. Benchmark with `run_local.py` or `run_modal.py` — these load `solution.json` and run via flashinfer-bench
5. Submit: tag the commit (e.g., `git tag submission-v1`) and push

### Key Concepts

- **Destination Passing Style (DPS)**: Default mode where both inputs and outputs are function parameters. Set `destination_passing_style = false` in config.toml if your kernel returns output tensors instead. Mismatch causes "expected xx parameters" errors.
- **CUDA bindings**: Use TVM FFI (default) or PyTorch (`binding = "torch"` in solution spec). See `solution/cuda/binding.py`.
- **Scoring**: Arithmetic mean of speedups across all workloads per definition, measured against a simple Python reference (not the FlashInfer baseline).
- **Track B/C**: Sparse attention and GDN tracks require submitting both operators; ranking uses the average.

### Evaluation Environment

- Docker: `flashinfer/flashinfer-ci-cu132:latest`
- Hardware: Bare-metal NVIDIA B200, GPU clocks locked at 3996/1965
- Correctness tolerances vary by track (MoE: atol=1, rtol=0.3, required-matched-ratio=0.9)
