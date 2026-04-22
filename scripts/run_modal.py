"""
FlashInfer-Bench Modal Cloud Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks
on NVIDIA B200 GPUs via Modal.

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/flashinfer-trace/
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal

app = modal.App(os.environ.get("MODAL_APP_NAME", "flashinfer-bench-v3"))

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
def run_benchmark(
    solution_json: str,
    max_workloads: int = 0,
    max_seq_len: int = 0,
    min_seq_len: int = 0,
    dump_sass: bool = False,
    workload_offset: int = 0,
) -> dict:
    """Run benchmark on Modal B200 and return results.

    max_workloads: if >0, cap the number of workloads (smallest seqs first).
    max_seq_len:   if >0, drop workloads whose total_seq_len exceeds this bound —
                   useful for excluding the 8192-token reference-Python baselines.
    min_seq_len:   if >0, drop workloads whose total_seq_len is below this bound —
                   used to bench long-seq prefill behavior.
    dump_sass:     if True, sets MSINFER_DUMP_SASS=1 so msinfer_entry.py emits
                   PTX/cubin/ptxas-verbose into /tmp/cute-asm, then tars + base64
                   encodes the directory into the returned dict's `sass_dump`.
    """
    import base64
    import io
    import os
    import tarfile

    if dump_sass:
        os.environ["MSINFER_DUMP_SASS"] = "1"

    from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet

    solution = Solution.model_validate_json(solution_json)
    _iter = int(os.environ.get("FIB_ITERATIONS", "20"))
    config = BenchmarkConfig(warmup_runs=3, iterations=_iter, num_trials=3)

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

    if min_seq_len > 0:
        before = len(workloads)
        workloads = [w for w in workloads if _len_key(w) >= min_seq_len]
        print(f"Filtered to {len(workloads)} workloads with total_seq_len ≥ {min_seq_len} (from {before})")

    if not workloads:
        raise ValueError(
            f"No workloads remain after filters max_seq_len={max_seq_len}, min_seq_len={min_seq_len}"
        )

    workloads = sorted(workloads, key=_len_key)
    if workload_offset > 0:
        workloads = workloads[workload_offset:]
        print(f"Skipped first {workload_offset} workloads (offset)")
    if max_workloads > 0 and len(workloads) > max_workloads:
        workloads = workloads[:max_workloads]
        max_len = _len_key(workloads[-1]) if workloads else 0
        print(f"Subsampled to {max_workloads} workloads (offset={workload_offset}, max_seq_len={max_len})")

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
            # When SASS dumping is enabled, surface the ptxas verbose lines on
            # the PASSED path too so perf claims are grounded in SASS stats.
            if dump_sass and log_text:
                ptxas_lines = [
                    ln for ln in log_text.splitlines()
                    if "ptxas" in ln.lower() or "registers" in ln or "spill" in ln or "smem" in ln
                ]
                if ptxas_lines:
                    entry["ptxas_tail"] = "\n".join(ptxas_lines[-40:])
            results[definition.name][trace.workload.uuid] = entry

    if dump_sass and os.path.isdir("/tmp/cute-asm"):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            tf.add("/tmp/cute-asm", arcname="cute-asm")
        results["_sass_tarball_b64"] = base64.b64encode(buf.getvalue()).decode("ascii")
        print(f"SASS dump size: {len(buf.getvalue())} B (base64 in results._sass_tarball_b64)")

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

            ptxas_tail = result.get("ptxas_tail")
            if ptxas_tail:
                print("    --- ptxas-verbose ---")
                for line in ptxas_tail.splitlines():
                    print(f"    {line}")
                print("    --- end ptxas ---")


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={TRACE_SET_PATH: trace_volume})
def run_ncu(
    solution_json: str,
    batch_size: int = 1,
) -> dict:
    """Run NCU profiling on one workload for a given batch_size.

    Returns a dict with keys:
      "batch_size", "workload_uuid", "ncu_output" (raw NCU report text),
      "ptxas_tail" (ptxas register/spill lines if available).
    """
    import io
    import os
    import sys

    os.environ["MSINFER_DUMP_SASS"] = "1"

    from flashinfer_bench import Solution, TraceSet
    import flashinfer_bench.agents as agents

    solution = Solution.model_validate_json(solution_json)
    trace_set = TraceSet.from_path(TRACE_SET_PATH)

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    workloads = trace_set.workloads.get(solution.definition, [])
    matching = [
        w for w in workloads
        if w.workload.axes.get("batch_size") == batch_size
    ]
    if not matching:
        raise ValueError(f"No workload found with batch_size={batch_size}")

    workload = matching[0]
    print(f"[run_ncu] batch_size={batch_size} workload={workload.workload.uuid[:8]}")

    agents_api = [a for a in dir(agents) if not a.startswith("_")]
    print(f"[run_ncu] agents API: {agents_api}", flush=True)

    # --- Try flashinfer_bench_list_ncu_options to understand the API ---
    list_opts_fn = getattr(agents, "flashinfer_bench_list_ncu_options", None)
    if list_opts_fn is not None:
        try:
            ncu_opts = list_opts_fn()
            print(f"[run_ncu] ncu_options: {ncu_opts}", flush=True)
        except Exception as e:
            print(f"[run_ncu] list_ncu_options error: {e}", flush=True)

    # --- Try ncu directly (simpler function) ---
    ncu_direct = getattr(agents, "ncu", None)
    ncu_direct_help = None
    if ncu_direct is not None:
        try:
            import inspect
            ncu_direct_help = inspect.signature(ncu_direct)
            print(f"[run_ncu] agents.ncu signature: {ncu_direct_help}", flush=True)
        except Exception as e:
            print(f"[run_ncu] ncu sig error: {e}", flush=True)

    # --- Try solution_handler approach ---
    sh = getattr(agents, "solution_handler", None)
    sh_help = None
    if sh is not None:
        try:
            import inspect
            sh_help = str(inspect.signature(sh))
            print(f"[run_ncu] solution_handler sig: {sh_help}", flush=True)
        except Exception as e:
            print(f"[run_ncu] sh sig error: {e}", flush=True)

    # --- Try ncu subprocess as fallback ---
    import subprocess as sp
    ncu_path = sp.run(["which", "ncu"], capture_output=True, text=True).stdout.strip()
    print(f"[run_ncu] ncu binary: {ncu_path!r}", flush=True)

    result = {
        "batch_size": batch_size,
        "workload_uuid": workload.workload.uuid,
        "agents_api": agents_api,
        "ncu_path": ncu_path,
        "ncu_direct_sig": str(ncu_direct_help) if ncu_direct_help else None,
        "sh_sig": sh_help,
    }

    # Grab ptxas stats from the SASS dump dir if populated.
    import subprocess
    ptxas_lines = []
    sass_dir = "/tmp/cute-asm"
    if os.path.isdir(sass_dir):
        for fn in os.listdir(sass_dir):
            fp = os.path.join(sass_dir, fn)
            try:
                with open(fp) as fh:
                    for ln in fh:
                        if any(kw in ln for kw in ("registers", "spill", "smem", "ptxas")):
                            ptxas_lines.append(ln.rstrip())
            except Exception:
                pass
    result["ptxas_tail"] = "\n".join(ptxas_lines[-40:])

    return result


@app.local_entrypoint()
def main(
    kernel_dir: str = "",
    max_workloads: int = 0,
    max_seq_len: int = 0,
    min_seq_len: int = 0,
    dump_sass: bool = False,
    sass_out: str = "out/sass-dump.tar.gz",
    workload_offset: int = 0,
):
    """Pack solution and run benchmark on Modal.

    kernel_dir: which per-kernel subdir to pack (e.g. "gdn_decode", "gdn_prefill").
                If empty, inferred when only one subdir under the repo has a config.toml.
    """
    kdir = kernel_dir or None
    # Attempt flashinfer-bench-based packing first; fall back to a minimal
    # local JSON packer when the library isn't installed locally (e.g. macOS).
    try:
        from scripts.pack_solution import pack_solution
        print("Packing solution from source files (flashinfer-bench)...")
        solution_path = pack_solution(kernel_dir=kdir)
    except ImportError:
        print("flashinfer-bench not available locally; using minimal JSON packer...")
        solution_path = _minimal_pack(kdir)

    solution_json = solution_path.read_text()
    import json as _json
    meta = _json.loads(solution_json)
    print(f"Loaded: {meta['name']} ({meta['definition']})")

    print(
        f"\nRunning benchmark on Modal B200 "
        f"(max_workloads={max_workloads}, max_seq_len={max_seq_len}, "
        f"min_seq_len={min_seq_len}, dump_sass={dump_sass})..."
    )
    results = run_benchmark.remote(
        solution_json,
        max_workloads=max_workloads,
        max_seq_len=max_seq_len,
        min_seq_len=min_seq_len,
        dump_sass=dump_sass,
        workload_offset=workload_offset,
    )

    # If SASS dumping was requested, peel off the tarball before print_results.
    if isinstance(results, dict) and "_sass_tarball_b64" in results:
        import base64 as _b64
        blob = _b64.b64decode(results.pop("_sass_tarball_b64"))
        out_path = Path(sass_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(blob)
        print(f"SASS dump written to {out_path} ({len(blob)} B)")

    if not results:
        print("No results returned!")
        return

    print_results(results)


@app.local_entrypoint()
def ncu_profile(
    kernel_dir: str = "gdn_decode",
    batch_sizes: str = "1,8,64",
    out_file: str = "out/ncu-baseline.md",
):
    """Run NCU profiling for batch_sizes (comma-separated) and write a markdown report.

    Usage:
        python3 -m modal run scripts/run_modal.py::ncu_profile --kernel-dir gdn_decode
    """
    import json as _json

    kdir = kernel_dir or None
    try:
        from scripts.pack_solution import pack_solution
        solution_path = pack_solution(kernel_dir=kdir)
    except ImportError:
        solution_path = _minimal_pack(kdir)

    solution_json = solution_path.read_text()
    meta = _json.loads(solution_json)
    print(f"Loaded: {meta['name']} ({meta['definition']})")

    target_batches = [int(b.strip()) for b in batch_sizes.split(",")]
    all_results = []

    for bs in target_batches:
        print(f"\nRunning NCU profile for batch_size={bs}...")
        result = run_ncu.remote(solution_json, batch_size=bs)
        all_results.append(result)
        print(f"  workload: {result['workload_uuid'][:8]}")
        if result.get("ncu_output"):
            print("  --- NCU output ---")
            for ln in result["ncu_output"].splitlines()[-80:]:
                print(f"  {ln}")
        if result.get("ncu_result"):
            print(f"  --- ncu_result ---\n  {result['ncu_result'][:2000]}")
        if result.get("ptxas_tail"):
            print("  --- ptxas stats ---")
            for ln in result["ptxas_tail"].splitlines():
                print(f"  {ln}")

    # Write markdown report.
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        f.write(f"# NCU Baseline — {meta['name']}\n\n")
        for r in all_results:
            f.write(f"## batch_size={r['batch_size']} (workload {r['workload_uuid'][:8]})\n\n")
            if r.get("ncu_output"):
                f.write("### NCU Output\n```\n")
                f.write(r["ncu_output"][-4000:])
                f.write("\n```\n\n")
            if r.get("ptxas_tail"):
                f.write("### ptxas stats\n```\n")
                f.write(r["ptxas_tail"])
                f.write("\n```\n\n")
    print(f"\nNCU report written to {out_file}")


def _minimal_pack(kernel_dir: str | None = None) -> Path:
    """Minimal solution.json writer with no flashinfer-bench dependency.

    Reads per-kernel `config.toml` from a subdirectory of the repo root.
    """
    import json

    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore

    if kernel_dir:
        kdir = (PROJECT_ROOT / kernel_dir).resolve() if not Path(kernel_dir).is_absolute() else Path(kernel_dir)
    else:
        candidates = [
            p for p in PROJECT_ROOT.iterdir()
            if p.is_dir() and not p.name.startswith(".") and (p / "config.toml").exists()
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"Cannot infer kernel dir; candidates={[p.name for p in candidates]}"
            )
        kdir = candidates[0]

    with open(kdir / "config.toml", "rb") as f:
        cfg = tomllib.load(f)
    s = cfg["solution"]
    b = cfg["build"]
    lang = b["language"]
    src_dir = kdir / "solution" / lang

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
            "dependencies": b.get("dependencies", []),
            "destination_passing_style": b.get("destination_passing_style", True),
            "binding": b.get("binding", None),
        },
        "sources": sources,
        "description": "",
    }
    out_path = kdir / "solution.json"
    out_path.write_text(json.dumps(out, indent=2))
    return out_path
