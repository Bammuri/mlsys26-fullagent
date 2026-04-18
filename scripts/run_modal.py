"""
FlashInfer-Bench Modal Cloud Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks
on NVIDIA B200 GPUs via Modal.

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/flashinfer-trace/
"""

import os
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal

app = modal.App("flashinfer-bench-v2")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

# Base image selection:
# - MODAL_IMAGE_ID = "im-..."       → reuse a pre-built Modal image (no pip step)
# - MODAL_IMAGE_SHA = "<hex64>"     → pin flashinfer/flashinfer-ci-cu132 by digest,
#                                      then layer flashinfer-bench on top
# - default                          → :latest + pip install
_PREBUILT_IMAGE_ID  = os.environ.get("MODAL_IMAGE_ID")
_BASE_IMAGE_SHA     = os.environ.get("MODAL_IMAGE_SHA")

if _PREBUILT_IMAGE_ID:
    image = modal.Image.from_id(_PREBUILT_IMAGE_ID)
else:
    base_tag = (
        f"flashinfer/flashinfer-ci-cu132@sha256:{_BASE_IMAGE_SHA}"
        if _BASE_IMAGE_SHA
        else "flashinfer/flashinfer-ci-cu132:latest"
    )
    image = (
        modal.Image.from_registry(base_tag, add_python="3.12")
        .pip_install("flashinfer-bench", "torch", "triton", "numpy")
    )


@app.function(image=image, gpu="B200:1", timeout=7200, volumes={TRACE_SET_PATH: trace_volume})
def run_benchmark(solution_json: str, max_workloads: int = 0, max_seq_len: int = 0) -> dict:
    """Run benchmark on Modal B200 and return results.

    max_workloads: if >0, cap the number of workloads (smallest seqs first).
    max_seq_len:   if >0, drop workloads whose total_seq_len exceeds this bound —
                   useful for excluding the 8192-token reference-Python baselines.
    """
    from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet

    solution = Solution.model_validate_json(solution_json)
    config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)

    trace_set = TraceSet.from_path(TRACE_SET_PATH)

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])

    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

    def _len_key(w):
        try:
            return int(w.workload.axes.get("total_seq_len", 0))
        except Exception:
            return 0

    if max_seq_len > 0:
        before = len(workloads)
        workloads = [w for w in workloads if _len_key(w) <= max_seq_len]
        print(f"Filtered to {len(workloads)} workloads with total_seq_len ≤ {max_seq_len} (from {before})")

    if max_workloads > 0 and len(workloads) > max_workloads:
        # Sort by total_seq_len asc, pick smallest N.
        workloads = sorted(workloads, key=_len_key)[:max_workloads]
        max_len = _len_key(workloads[-1]) if workloads else 0
        print(f"Subsampled to {len(workloads)} smallest workloads (max_seq_len={max_len})")

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

    import pathlib as _pl
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
            # On non-PASSED status, surface the captured stdio log — the
            # PersistentRunner redirects stdio to a per-worker file that
            # the evaluator inlines into Evaluation.log.
            log_text = getattr(trace.evaluation, "log", "") or ""
            if log_text and trace.evaluation.status.value != "PASSED":
                entry["log_tail"] = log_text[-6000:] if len(log_text) > 6000 else log_text
            results[definition.name][trace.workload.uuid] = entry

    return results


def print_results(results: dict):
    """Print benchmark results in a formatted way."""
    for def_name, traces in results.items():
        print(f"\n{def_name}:")
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

            tail = result.get("log_tail")
            if tail:
                print("    --- worker log tail (last 6 KB) ---")
                for line in tail.splitlines():
                    print(f"    {line}")
                print("    --- end log ---")


@app.local_entrypoint()
def main(max_workloads: int = 0, max_seq_len: int = 0):
    """Pack solution and run benchmark on Modal."""
    # Attempt flashinfer-bench-based packing first; fall back to a minimal
    # local JSON packer when the library isn't installed locally (e.g. macOS).
    try:
        from scripts.pack_solution import pack_solution
        print("Packing solution from source files (flashinfer-bench)...")
        solution_path = pack_solution()
    except ImportError:
        print("flashinfer-bench not available locally; using minimal JSON packer...")
        solution_path = _minimal_pack()

    solution_json = solution_path.read_text()
    import json as _json
    meta = _json.loads(solution_json)
    print(f"Loaded: {meta['name']} ({meta['definition']})")

    print(f"\nRunning benchmark on Modal B200 (max_workloads={max_workloads}, max_seq_len={max_seq_len})...")
    results = run_benchmark.remote(solution_json, max_workloads=max_workloads, max_seq_len=max_seq_len)

    if not results:
        print("No results returned!")
        return

    print_results(results)


def _minimal_pack() -> Path:
    """Minimal solution.json writer with no flashinfer-bench dependency."""
    import json

    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore

    with open(PROJECT_ROOT / "config.toml", "rb") as f:
        cfg = tomllib.load(f)
    s = cfg["solution"]
    b = cfg["build"]
    lang = b["language"]
    src_dir = PROJECT_ROOT / "solution" / ("cuda" if lang == "cuda" else "triton")

    sources = []
    for p in sorted(src_dir.iterdir()):
        if p.name.startswith("__") or p.suffix == ".pyc" or not p.is_file():
            continue
        sources.append({"path": p.name, "content": p.read_text()})

    out = {
        "name": s["name"],
        "definition": s["definition"],
        "author": s["author"],
        "spec": {
            "language": lang,
            "target_hardware": ["cuda"],
            "entry_point": b["entry_point"],
            "dependencies": [],
            "destination_passing_style": b.get("destination_passing_style", True),
            "binding": b.get("binding", None),
        },
        "sources": sources,
        "description": "",
    }
    out_path = PROJECT_ROOT / "solution.json"
    out_path.write_text(json.dumps(out, indent=2))
    return out_path
