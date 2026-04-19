"""
FlashInfer-Bench Local Benchmark Runner.

Runs benchmarks locally using either a packed solution JSON or source files.
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet
from scripts.pack_solution import pack_solution

SUCCESS_STATUSES = {"OK", "PASSED"}


def get_trace_set_path() -> str:
    """Get trace set path from environment variable."""
    path = os.environ.get("FIB_DATASET_PATH")
    if not path:
        raise EnvironmentError(
            "FIB_DATASET_PATH environment variable not set. "
            "Please set it to the path of your mlsys26-contest dataset."
        )
    return path


def run_benchmark(solution: Solution, config: BenchmarkConfig = None) -> dict:
    """Run benchmark locally and return results."""
    if config is None:
        config = BenchmarkConfig(warmup_runs=3, iterations=10, num_trials=1)

    trace_set_path = get_trace_set_path()
    trace_set = TraceSet.from_path(trace_set_path)

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])

    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    benchmark = Benchmark(bench_trace_set, config)
    result_trace_set = benchmark.run_all(dump_traces=True)

    traces = result_trace_set.traces.get(definition.name, [])
    results = {definition.name: {}}

    for trace in traces:
        if trace.evaluation:
            entry = {
                "status": trace.evaluation.status.value,
                "solution": trace.solution,
            }
            if trace.evaluation.performance:
                entry["latency_ms"] = trace.evaluation.performance.latency_ms
                entry["reference_latency_ms"] = trace.evaluation.performance.reference_latency_ms
                entry["speedup_factor"] = trace.evaluation.performance.speedup_factor
            if trace.evaluation.correctness:
                entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
                entry["max_rel_error"] = trace.evaluation.correctness.max_relative_error
            results[definition.name][trace.workload.uuid] = entry

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


def load_solution(solution_path: Path | None = None) -> Solution:
    """Load a solution from JSON or pack it from source files."""
    if solution_path is None:
        print("Packing solution from source files...")
        solution_path = pack_solution()
    else:
        print(f"Loading solution from JSON: {solution_path}")

    print("\nLoading solution...")
    solution = Solution.model_validate_json(Path(solution_path).read_text())
    print(f"Loaded: {solution.name} ({solution.definition})")
    return solution


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run local FlashInfer benchmarks")
    parser.add_argument(
        "--solution-path",
        type=Path,
        default=None,
        help="Path to an existing solution JSON. If omitted, pack from local source files.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print only aggregated benchmark summary instead of per-workload details.",
    )
    return parser.parse_args()


def main():
    """Load the solution and run benchmark."""
    args = parse_args()
    solution = load_solution(args.solution_path)

    print("\nRunning benchmark...")
    results = run_benchmark(solution)

    if not results:
        print("No results returned!")
        return

    print_results(results, summary_only=args.summary_only)


if __name__ == "__main__":
    main()
