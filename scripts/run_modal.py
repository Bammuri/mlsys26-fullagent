"""
FlashInfer-Bench Modal Cloud Benchmark Runner.

Runs benchmarks on Modal using either a packed solution JSON or source files.

By default this script packs the solution from source files and runs benchmarks
on NVIDIA B200 GPUs via Modal.

Setup (one-time):
    modal setup
    modal volume create mlsys26-contest
    modal volume put mlsys26-contest /path/to/mlsys26-contest /data/
"""

import random
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal
from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet

SUCCESS_STATUSES = {"OK", "PASSED"}

app = modal.App("flashinfer-bench")

trace_volume = modal.Volume.from_name("mlsys26-contest", create_if_missing=True)
VOLUME_MOUNT_PATH = "/data"
TRACE_SET_PATH = "/data/data/mlsys26-contest"

image = (
    modal.Image.from_registry("nvidia/cuda:13.0.2-devel-ubuntu24.04", add_python="3.12")
    .env({"CUDA_HOME": "/usr/local/cuda", "TORCH_CUDA_ARCH_LIST": "10.0a"})
    .pip_install("flashinfer-bench", "flashinfer-python", "torch", "triton", "numpy")
)


def log_event(message: str):
    """Print a timestamped progress message."""
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def format_elapsed(seconds: float) -> str:
    """Format elapsed seconds for logs."""
    return f"{seconds:.2f}s"


def select_workloads(workloads: list, max_workloads: int, seed: int | None = None) -> list:
    """Select up to max_workloads workloads. When seed is not None, sampling is
    deterministic across runs — essential for apples-to-apples optimization
    comparisons. When seed is None, falls back to the original unseeded behavior.
    """
    if max_workloads <= 0 or len(workloads) <= max_workloads:
        return workloads

    if seed is not None:
        def _sort_key(w):
            u = getattr(w, "uuid", None)
            if u is None:
                u = getattr(getattr(w, "workload", None), "uuid", None)
            return u if u is not None else repr(w)

        rng = random.Random(seed)
        ordered = sorted(workloads, key=_sort_key)
        return rng.sample(ordered, max_workloads)
    return random.sample(workloads, max_workloads)


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={VOLUME_MOUNT_PATH: trace_volume})
def run_benchmark(
    solution: Solution,
    config: BenchmarkConfig = None,
    max_workloads: int = 0,
    sample_seed: int | None = None,
) -> dict:
    """Run benchmark on Modal B200 and return results."""
    if config is None:
        config = BenchmarkConfig(
            warmup_runs=1,
            iterations=5,
            num_trials=3,
            use_isolated_runner=True,
            timeout_seconds=300,
        )

    started_at = time.perf_counter()
    log_event(f"Remote benchmark start: solution={solution.name}, definition={solution.definition}")
    log_event(
        "BenchmarkConfig("
        f"warmup_runs={config.warmup_runs}, iterations={config.iterations}, num_trials={config.num_trials}"
        ")"
    )

    trace_load_started = time.perf_counter()
    log_event(f"Loading trace set from {TRACE_SET_PATH}")
    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    log_event(f"Loaded trace set in {format_elapsed(time.perf_counter() - trace_load_started)}")

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])
    workloads = select_workloads(workloads, max_workloads, seed=sample_seed)
    log_event(
        f"Selected {len(workloads)} workloads (max_workloads={max_workloads}, sample_seed={sample_seed})"
    )

    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    benchmark_started = time.perf_counter()
    log_event(f"Running benchmark across {len(workloads)} workloads")
    benchmark = Benchmark(bench_trace_set, config)
    result_trace_set = benchmark.run_all(dump_traces=True)
    log_event(f"Benchmark completed in {format_elapsed(time.perf_counter() - benchmark_started)}")

    traces = result_trace_set.traces.get(definition.name, [])
    results = {definition.name: {}}

    for trace in traces:
        if trace.evaluation:
            entry = {
                "status": trace.evaluation.status.value,
                "solution": trace.solution,
            }
            if trace.evaluation.log:
                entry["log"] = trace.evaluation.log
            if trace.evaluation.performance:
                entry["latency_ms"] = trace.evaluation.performance.latency_ms
                entry["reference_latency_ms"] = trace.evaluation.performance.reference_latency_ms
                entry["speedup_factor"] = trace.evaluation.performance.speedup_factor
            if trace.evaluation.correctness:
                entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
                entry["max_rel_error"] = trace.evaluation.correctness.max_relative_error
            results[definition.name][trace.workload.uuid] = entry

    log_event(f"Remote benchmark finished in {format_elapsed(time.perf_counter() - started_at)}")
    return results


def print_results(results: dict, summary_only: bool = False):
    """Print benchmark results in a formatted way."""
    for def_name, traces in results.items():
        total = len(traces)
        statuses = {}
        latency_values = []
        speedup_values = []
        abs_errors = []
        rel_errors = []

        for result in traces.values():
            status = result.get("status", "UNKNOWN")
            statuses[status] = statuses.get(status, 0) + 1
            if result.get("latency_ms") is not None:
                latency_values.append(result["latency_ms"])
            if result.get("speedup_factor") is not None:
                speedup_values.append(result["speedup_factor"])
            if result.get("max_abs_error") is not None:
                abs_errors.append(result["max_abs_error"])
            if result.get("max_rel_error") is not None:
                rel_errors.append(result["max_rel_error"])

        print(f"\n{def_name}:")
        print(f"  workloads: {total}")
        print("  status counts:", ", ".join(f"{k}={v}" for k, v in sorted(statuses.items())))
        if latency_values:
            print(f"  avg latency: {sum(latency_values) / len(latency_values):.3f} ms")
        if speedup_values:
            print(f"  avg speedup: {sum(speedup_values) / len(speedup_values):.2f}x")
        if abs_errors:
            print(f"  worst abs error: {max(abs_errors):.2e}")
        if rel_errors:
            print(f"  worst rel error: {max(rel_errors):.2e}")

        if summary_only:
            failed = [
                uuid[:8]
                for uuid, result in traces.items()
                if result.get("status") not in SUCCESS_STATUSES
            ]
            if failed:
                print(f"  failed workloads: {', '.join(failed)}")
            continue

        for workload_uuid, result in traces.items():
            status = result.get("status")
            print(f"  Workload {workload_uuid[:8]}...: {status}", end="")

            if result.get("latency_ms") is not None:
                print(f" | {result['latency_ms']:.3f} ms", end="")

            if result.get("speedup_factor") is not None:
                print(f" | {result['speedup_factor']:.2f}x speedup", end="")

            if result.get("max_abs_error") is not None:
                abs_err = result["max_abs_error"]
                rel_err = result.get("max_rel_error", 0)
                print(f" | abs_err={abs_err:.2e}, rel_err={rel_err:.2e}", end="")

            print()
            if status not in SUCCESS_STATUSES and result.get("log"):
                print("    --- log start ---")
                print(result["log"].rstrip())
                print("    --- log end ---")


def load_solution(solution_path: Path | None = None) -> Solution:
    """Load a solution from JSON or pack it from source files."""
    started_at = time.perf_counter()
    if solution_path is None:
        from scripts.pack_solution import pack_solution

        log_event("Packing solution from source files...")
        solution_path = pack_solution()
    else:
        log_event(f"Loading solution from JSON: {solution_path}")

    log_event("Validating solution JSON...")
    solution = Solution.model_validate_json(Path(solution_path).read_text())
    log_event(
        f"Loaded solution {solution.name} ({solution.definition}) in "
        f"{format_elapsed(time.perf_counter() - started_at)}"
    )
    return solution


@app.local_entrypoint()
def main(
    solution_path: str = "",
    summary_only: bool = False,
    max_workloads: int = 20,
    quick: bool = False,
    decision_gate: bool = False,
    sample_seed: int = -1,
):
    """Load the solution and run benchmark on Modal."""
    started_at = time.perf_counter()
    solution = load_solution(Path(solution_path) if solution_path else None)

    config = None
    if decision_gate:
        config = BenchmarkConfig(
            warmup_runs=1,
            iterations=5,
            num_trials=2,
            use_isolated_runner=False,
            timeout_seconds=300,
        )
        log_event(
            "Decision-gate mode enabled: warmup_runs=1, iterations=5, num_trials=2, "
            "use_isolated_runner=False"
        )
    elif quick:
        config = BenchmarkConfig(
            warmup_runs=1,
            iterations=1,
            num_trials=1,
            use_isolated_runner=False,
            timeout_seconds=300,
        )
        log_event(
            "Quick mode enabled: warmup_runs=1, iterations=1, num_trials=1, "
            "use_isolated_runner=False"
        )

    seed_arg: int | None = sample_seed if sample_seed >= 0 else None
    if seed_arg is not None:
        log_event(f"Deterministic workload sampling enabled (sample_seed={seed_arg})")

    log_event("Dispatching benchmark to Modal B200...")
    remote_started = time.perf_counter()
    results = run_benchmark.remote(solution, config, max_workloads, seed_arg)
    log_event(f"Received benchmark results in {format_elapsed(time.perf_counter() - remote_started)}")

    if not results:
        log_event("No results returned!")
        return

    print_results(results, summary_only=summary_only)
    log_event(f"Local entrypoint finished in {format_elapsed(time.perf_counter() - started_at)}")
