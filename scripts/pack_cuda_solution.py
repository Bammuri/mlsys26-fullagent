"""
Side-channel packer for the pure CUDA lane.

Produces solution_cuda.json from solution/cuda/ WITHOUT reading or modifying
config.toml. This lets us iterate on kernel.cu while the main config.toml
continues to point at the production Python/CuTe lane.

Run:
    PYENV_VERSION=fi-bench python scripts/pack_cuda_solution.py
    PYENV_VERSION=fi-bench modal run scripts/run_modal.py --solution-path solution_cuda.json
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flashinfer_bench import BuildSpec, Solution, SourceFile

VALID_SOURCE_EXTENSIONS = {".py", ".cu", ".cuh", ".cpp", ".c", ".h", ".hpp"}


def pack_cuda(output_path: Path | None = None) -> Path:
    source_dir = PROJECT_ROOT / "solution" / "cuda"
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)

    sources: list[SourceFile] = []
    for file_path in sorted(source_dir.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in VALID_SOURCE_EXTENSIONS:
            continue
        rel_path = file_path.relative_to(source_dir).as_posix()
        sources.append(SourceFile(path=rel_path, content=file_path.read_text(encoding="utf-8")))
    if not sources:
        raise ValueError(f"No CUDA sources found in {source_dir}")

    spec = BuildSpec(
        language="cuda",
        target_hardware=["cuda"],
        entry_point="binding.cpp::run",
        dependencies=[],
        destination_passing_style=False,
        binding="torch",
    )

    solution = Solution(
        name="gdn-prefill-v1-cuda",
        definition="gdn_prefill_qk4_v8_d128_k_last",
        author="MSInfer",
        description="Pure CUDA lane packed directly (no config.toml read)",
        spec=spec,
        sources=sources,
    )

    if output_path is None:
        output_path = PROJECT_ROOT / "solution_cuda.json"
    output_path.write_text(solution.model_dump_json(indent=2))
    print(f"Packed CUDA solution: {output_path}")
    print(f"  Sources: {len(sources)} files from {source_dir}")
    return output_path


if __name__ == "__main__":
    pack_cuda()
