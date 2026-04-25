"""
Pack solution source files into solution.json.

Reads configuration from a per-kernel subdirectory's config.toml and packs the
appropriate source files into a Solution JSON file for submission.

Each kernel lives in its own subdirectory (e.g. ``gdn_decode/``, ``gdn_prefill/``)
containing its own ``config.toml`` and ``solution/<language>/...``. The evaluation
pipeline auto-packs each subdir independently.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from flashinfer_bench import BuildSpec
from flashinfer_bench.agents import pack_solution_from_files


def _resolve_kernel_dir(kernel_dir: Path | str | None) -> Path:
    """Resolve the kernel directory. Precedence:

    1. Explicit ``kernel_dir`` argument (CLI flag or caller kwarg).
    2. Current working directory if it contains a ``config.toml``.
    3. The single immediate subdirectory of ``PROJECT_ROOT`` that contains a
       ``config.toml`` (errors if multiple candidates).
    """
    if kernel_dir is not None:
        p = Path(kernel_dir)
        if not p.is_absolute():
            p = (PROJECT_ROOT / p).resolve()
        if not (p / "config.toml").exists():
            raise FileNotFoundError(f"No config.toml at {p}")
        return p

    cwd = Path.cwd()
    if (cwd / "config.toml").exists() and cwd != PROJECT_ROOT:
        return cwd

    candidates = [
        p for p in PROJECT_ROOT.iterdir()
        if p.is_dir() and not p.name.startswith(".") and (p / "config.toml").exists()
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"No kernel subdirectory with config.toml under {PROJECT_ROOT}. "
            f"Pass --kernel-dir <name> or create one of gdn_decode/, gdn_prefill/."
        )
    names = ", ".join(p.name for p in candidates)
    raise ValueError(
        f"Multiple kernel subdirs found ({names}). Pass --kernel-dir to disambiguate."
    )


def load_config(kernel_dir: Path) -> dict:
    config_path = kernel_dir / "config.toml"
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def pack_solution(
    output_path: Path | None = None,
    kernel_dir: Path | str | None = None,
) -> Path:
    """Pack solution files into a Solution JSON for the given kernel subdir."""
    kdir = _resolve_kernel_dir(kernel_dir)
    config = load_config(kdir)

    solution_config = config["solution"]
    build_config = config["build"]

    language = build_config["language"]
    entry_point = build_config["entry_point"]

    source_dir = kdir / "solution" / language
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    dps = build_config.get("destination_passing_style", True)
    binding = build_config.get("binding", None)
    dependencies = build_config.get("dependencies", None)
    spec_kwargs = dict(
        language=language,
        target_hardware=["cuda"],
        entry_point=entry_point,
        destination_passing_style=dps,
    )
    if binding is not None:
        spec_kwargs["binding"] = binding
    if dependencies is not None:
        spec_kwargs["dependencies"] = dependencies
    spec = BuildSpec(**spec_kwargs)

    solution = pack_solution_from_files(
        path=str(source_dir),
        spec=spec,
        name=solution_config["name"],
        definition=solution_config["definition"],
        author=solution_config["author"],
    )

    if output_path is None:
        output_path = kdir / "solution.json"

    output_path.write_text(solution.model_dump_json(indent=2))
    print(f"Solution packed: {output_path}")
    print(f"  Kernel dir: {kdir.relative_to(PROJECT_ROOT) if kdir.is_relative_to(PROJECT_ROOT) else kdir}")
    print(f"  Name: {solution.name}")
    print(f"  Definition: {solution.definition}")
    print(f"  Author: {solution.author}")
    print(f"  Language: {language}")

    return output_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pack solution files into solution.json")
    parser.add_argument(
        "-k", "--kernel-dir",
        type=str,
        default=None,
        help="Kernel subdirectory (e.g. gdn_decode, gdn_prefill). "
             "If omitted, inferred from cwd or the single subdir with a config.toml.",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output path for solution.json (default: <kernel-dir>/solution.json)",
    )
    args = parser.parse_args()

    try:
        pack_solution(output_path=args.output, kernel_dir=args.kernel_dir)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
