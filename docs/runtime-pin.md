# Runtime Pin — Modal ↔ Bare Scoring Parity

The contest saw measurable divergence between Modal benchmarks and the
scoring runner when the build surface drifted (static `.cu` w/ implicit
nvcc defaults vs Python+CuTe with explicit flags). This document pins the
exact runtime we depend on so the **same solution runs identically in
both environments**.

## Captured from the Modal container `flashinfer/flashinfer-ci-cu132:latest`

Snapshot taken 2026-04-18 from a running container (ID `97e9ec536ab1`).

### Base OS + toolchain

| Component | Version |
|-----------|---------|
| Image tag             | `flashinfer/flashinfer-ci-cu132:latest` |
| Image digest          | `sha256:87e3b10d657b4c3f31fd028b01748d82fe09f41a30d820b1afb39ef77fbfa453` |
| OS                    | Ubuntu 24.04.3 LTS |
| Architecture          | aarch64 (SBSA) |
| CUDA toolkit          | 13.1.115 (Build cuda_13.1.r13.1/compiler.37061995_0) |
| `nvcc`                | `/usr/local/cuda/bin/nvcc` |
| CUDA runtime linkage  | `/usr/local/cuda → /etc/alternatives/cuda`, libs at `/usr/local/cuda/targets/sbsa-linux/lib` |
| Python                | 3.12.3 (`/usr/bin/python3`) |
| `PATH`                | `/usr/local/nvidia/bin:/usr/local/cuda/bin:...` |

### Critical Python packages

| Package             | Pinned version |
|---------------------|----------------|
| `nvidia-cutlass-dsl`  | 4.4.2  |
| `cuda-python`         | 13.2.0 |
| `flashinfer-bench`    | 0.1.2  |
| `flashinfer-python`   | 0.6.8  |
| `torch`               | 2.11.0 |
| `triton`              | 3.6.0  |
| `numpy`               | 2.4.4  |
| `apache-tvm-ffi`      | 0.1.10 |

### GPU runtime (determined on scoring host, not in image)

| Component | Expected |
|-----------|---------|
| GPU       | NVIDIA B200 (Blackwell, sm_100) |
| Driver    | must provide `libcuda.so.1` with CUDA 13 support |
| Clocks    | locked 3996/1965 (per `EVALUATION.md`) |

## Where each piece is pinned

| Pin                           | Where                                   |
|-------------------------------|-----------------------------------------|
| Base container image          | `scripts/run_modal.py` (`from_registry("flashinfer/flashinfer-ci-cu132:latest")`). Can digest-pin via `MODAL_IMAGE_SHA`. |
| Python package versions       | `config.toml` → `build.dependencies` (flashinfer-bench forwards these to pip) |
| nvcc compile flags            | `solution/python/msinfer_entry.py` → `_COMPILE_OPTS`: `--enable-tvm-ffi --gpu-arch=sm_100a --opt-level=3` |
| Kernel entry surface          | `config.toml` → `build.entry_point = msinfer_entry.py::run[_prefill]` |
| Output allocation convention  | `config.toml` → `destination_passing_style = true` (no alloc churn between calls) |

## Reproducing the bare environment

```bash
# 1. Use the same base image — digest-pinned for byte-for-byte parity
docker run --rm --gpus all -it \
  -v $PWD:/workspace -w /workspace \
  flashinfer/flashinfer-ci-cu132@sha256:87e3b10d657b4c3f31fd028b01748d82fe09f41a30d820b1afb39ef77fbfa453 \
  bash

# 2. Install pinned dependencies into the container (matches what
#    flashinfer-bench installs from config.toml's `build.dependencies`)
python3 -m pip install \
    nvidia-cutlass-dsl==4.4.2 \
    cuda-python==13.2.0 \
    apache-tvm-ffi==0.1.10 \
    flashinfer-bench==0.1.2

# 3. Run benchmark
python3 scripts/pack_solution.py
flashinfer-bench run \
  --local /workspace \
  --definitions gdn_decode_qk4_v8_d128_k_last \
  --use-isolated-runner --timeout 300
```

The `--gpu-arch=sm_100a` pin in `_COMPILE_OPTS` is what actually makes the
Blackwell-native instructions (`tcgen05.*`, 5th-gen MMA, CTA-pair) light
up. Without it, nvcc falls back to generic SM 10.0 or even earlier SASS
and the PTX-JIT path can pick a non-optimal scheduler — that was the root
of the previously observed Modal ↔ scoring divergence.
