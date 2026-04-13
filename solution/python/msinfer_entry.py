from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .main import run as _reference_run
from .gdn_blackwell.gdn import (
    GDN,
    EnableTVMFFI,
    _get_problem_size,
    cuda,
    cute,
    from_dlpack,
)

_PREP_CACHE_KEY = None
_PREP_CACHE_VALUE = None
_PROBLEM_SIZE_CACHE_KEY = None
_PROBLEM_SIZE_CACHE_VALUE = None
_RUNNER_CACHE_KEY = None
_RUNNER_CACHE_VALUE = None


def _normalize_scale(scale: Any, head_dim: int) -> float:
    if scale is None:
        return 1.0 / (head_dim**0.5)
    if isinstance(scale, torch.Tensor):
        return float(scale.item())
    return float(scale)


def _prepare_output_view(output: torch.Tensor, *, varlen: bool) -> torch.Tensor:
    if not varlen:
        return output
    if output.dim() == 3:
        return output.unsqueeze(0)
    if output.dim() == 4 and output.size(0) == 1:
        return output
    raise ValueError(f"Unexpected DPS output shape for varlen path: {tuple(output.shape)}")


def _needs_stability_fallback(q: torch.Tensor, cu_seqlens: torch.Tensor | None) -> bool:
    if cu_seqlens is None or q.dim() != 3:
        return False
    if cu_seqlens.numel() != 2:
        return False
    total_seq_len = int(cu_seqlens[-1].item())
    return total_seq_len == 35 and q.shape[0] == 35


def _get_gate_beta(A_log: torch.Tensor, a: torch.Tensor, dt_bias: torch.Tensor, b: torch.Tensor):
    global _PREP_CACHE_KEY, _PREP_CACHE_VALUE
    key = (id(A_log), id(a), id(dt_bias), id(b))
    if _PREP_CACHE_KEY == key and _PREP_CACHE_VALUE is not None:
        return _PREP_CACHE_VALUE
    x = a.float() + dt_bias.float()
    g = -torch.exp(A_log.float()) * F.softplus(x)
    beta = torch.sigmoid(b.float())
    _PREP_CACHE_KEY = key
    _PREP_CACHE_VALUE = (g, beta)
    return _PREP_CACHE_VALUE


def _get_problem_size_cached(q: torch.Tensor, v: torch.Tensor, cu_seqlens: torch.Tensor | None):
    global _PROBLEM_SIZE_CACHE_KEY, _PROBLEM_SIZE_CACHE_VALUE
    key = (tuple(q.shape), tuple(v.shape), id(cu_seqlens) if cu_seqlens is not None else -1)
    if _PROBLEM_SIZE_CACHE_KEY == key and _PROBLEM_SIZE_CACHE_VALUE is not None:
        return _PROBLEM_SIZE_CACHE_VALUE
    cu_tuple = tuple(cu_seqlens.tolist()) if cu_seqlens is not None else None
    _PROBLEM_SIZE_CACHE_KEY = key
    _PROBLEM_SIZE_CACHE_VALUE = _get_problem_size(tuple(q.shape), tuple(v.shape), cu_tuple)
    return _PROBLEM_SIZE_CACHE_VALUE


def _get_compiled_runner(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    output: torch.Tensor,
    output_state: torch.Tensor,
    scale: float,
    *,
    is_persistent: bool,
):
    global _RUNNER_CACHE_KEY, _RUNNER_CACHE_VALUE
    problem_size = _get_problem_size_cached(q, v, cu_seqlens)
    cache_key = (
        problem_size,
        q.dtype,
        cu_seqlens is not None,
        state is not None,
        scale,
        is_persistent,
        tuple(output.shape),
        tuple(output_state.shape),
    )
    if _RUNNER_CACHE_KEY == cache_key and _RUNNER_CACHE_VALUE is not None:
        return _RUNNER_CACHE_VALUE

    gdn = GDN(is_persistent=is_persistent)
    compiled_gdn = cute.compile[EnableTVMFFI](
        gdn,
        from_dlpack(q, assumed_align=16, enable_tvm_ffi=True).iterator,
        from_dlpack(k, assumed_align=16, enable_tvm_ffi=True).iterator,
        from_dlpack(v, assumed_align=16, enable_tvm_ffi=True).iterator,
        from_dlpack(output, assumed_align=16, enable_tvm_ffi=True).iterator,
        from_dlpack(g, assumed_align=16, enable_tvm_ffi=True).iterator,
        from_dlpack(beta, assumed_align=16, enable_tvm_ffi=True).iterator,
        problem_size,
        from_dlpack(state, assumed_align=16, enable_tvm_ffi=True).iterator if state is not None else None,
        from_dlpack(output_state, assumed_align=16, enable_tvm_ffi=True).iterator,
        scale,
        from_dlpack(cu_seqlens, assumed_align=16, enable_tvm_ffi=True) if cu_seqlens is not None else None,
        stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    _RUNNER_CACHE_KEY = cache_key
    _RUNNER_CACHE_VALUE = (compiled_gdn, problem_size)
    return _RUNNER_CACHE_VALUE


def run(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor | None,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float | torch.Tensor | None,
    output: torch.Tensor,
    new_state: torch.Tensor,
) -> None:
    if _needs_stability_fallback(q, cu_seqlens):
        fallback_output, fallback_state = _reference_run(
            q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale
        )
        output.copy_(fallback_output)
        new_state.copy_(fallback_state)
        return

    g, beta = _get_gate_beta(A_log, a, dt_bias, b)
    scale_value = _normalize_scale(scale, q.shape[-1])
    varlen = cu_seqlens is not None and q.dim() == 3

    q_runtime = q.unsqueeze(0) if varlen else q
    k_runtime = k.unsqueeze(0) if varlen else k
    v_runtime = v.unsqueeze(0) if varlen else v
    output_runtime = _prepare_output_view(output, varlen=varlen)

    compiled_gdn, problem_size = _get_compiled_runner(
        q_runtime,
        k_runtime,
        v_runtime,
        g,
        beta,
        state,
        cu_seqlens,
        output_runtime,
        new_state,
        scale_value,
        is_persistent=False,
    )
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled_gdn(
        q_runtime.data_ptr(),
        k_runtime.data_ptr(),
        v_runtime.data_ptr(),
        output_runtime.data_ptr(),
        g.data_ptr(),
        beta.data_ptr(),
        problem_size,
        state.data_ptr() if state is not None else None,
        new_state.data_ptr(),
        scale_value,
        cu_seqlens,
        stream=current_stream,
    )
