"""
FlashInfer-backed GDN prefill wrapper living under solution/cuda/.

We keep the repo layout CUDA-centric, but the current contest harness executes
this file as a Python entrypoint. The fast path delegates to FlashInfer's CUDA
implementation; the fallback path is a correctness-first PyTorch reference.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    from flashinfer import chunk_gated_delta_rule
except Exception:  # pragma: no cover - runtime fallback for environments without FlashInfer
    chunk_gated_delta_rule = None


def _normalize_scale(scale: float | torch.Tensor | None, head_size: int) -> float:
    if scale is None:
        return 1.0 / (head_size**0.5)
    if isinstance(scale, torch.Tensor):
        return float(scale.item())
    return float(scale)


def _reference_gdn_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor | None,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale_val: float,
):
    num_q_heads = q.size(1)
    num_k_heads = k.size(1)
    num_v_heads = v.size(1)
    head_size = q.size(2)
    num_seqs = cu_seqlens.numel() - 1
    device = q.device

    x = a.float() + dt_bias.float()
    g = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))
    beta = torch.sigmoid(b.float())

    q_exp = q.repeat_interleave(num_v_heads // num_q_heads, dim=1).float()
    k_exp = k.repeat_interleave(num_v_heads // num_k_heads, dim=1).float()
    v_f32 = v.float()

    output = torch.empty(
        (q.size(0), num_v_heads, head_size),
        dtype=q.dtype,
        device=device,
    )
    new_state = torch.empty(
        (num_seqs, num_v_heads, head_size, head_size),
        dtype=torch.float32,
        device=device,
    )

    for seq_idx in range(num_seqs):
        seq_start = int(cu_seqlens[seq_idx].item())
        seq_end = int(cu_seqlens[seq_idx + 1].item())
        seq_len = seq_end - seq_start

        if state is not None:
            state_hkv = state[seq_idx].float().transpose(-1, -2).contiguous()
        else:
            state_hkv = torch.zeros(
                (num_v_heads, head_size, head_size),
                dtype=torch.float32,
                device=device,
            )

        for offset in range(seq_len):
            t = seq_start + offset
            q_h1k = q_exp[t].unsqueeze(1)
            k_h1k = k_exp[t].unsqueeze(1)
            v_h1v = v_f32[t].unsqueeze(1)
            g_h11 = g[t].unsqueeze(1).unsqueeze(2)
            beta_h11 = beta[t].unsqueeze(1).unsqueeze(2)

            old_state = g_h11 * state_hkv
            old_v = torch.matmul(k_h1k, old_state)
            new_v = beta_h11 * v_h1v + (1.0 - beta_h11) * old_v
            state_remove = torch.einsum("hkl,hlv->hkv", k_h1k.transpose(-1, -2), old_v)
            state_update = torch.einsum("hkl,hlv->hkv", k_h1k.transpose(-1, -2), new_v)
            state_hkv = old_state - state_remove + state_update

            out = scale_val * torch.matmul(q_h1k, state_hkv)
            output[t] = out.squeeze(1).to(dtype=q.dtype)

        new_state[seq_idx] = state_hkv.transpose(-1, -2)

    return output, new_state


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
):
    """Contest entrypoint for GDN prefill."""
    scale_val = _normalize_scale(scale, q.shape[-1])
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    state = None if state is None else state.contiguous()
    cu_seqlens = cu_seqlens.to(torch.int64).contiguous()
    if not output.is_contiguous():
        raise ValueError("output must be contiguous")
    if not new_state.is_contiguous():
        raise ValueError("new_state must be contiguous")

    if chunk_gated_delta_rule is not None:
        x = a.float() + dt_bias.float()
        g = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))
        beta = torch.sigmoid(b.float())
        try:
            fast_output, fast_new_state = chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g.contiguous(),
                beta=beta.contiguous(),
                scale=scale_val,
                initial_state=state,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=False,
                output=output,
                output_state=new_state,
            )
            if fast_output.data_ptr() != output.data_ptr():
                output.copy_(fast_output)
            if fast_new_state.data_ptr() != new_state.data_ptr():
                new_state.copy_(fast_new_state)
            return
        except Exception:
            pass

    ref_output, ref_new_state = _reference_gdn_prefill(
        q=q,
        k=k,
        v=v,
        state=state,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        cu_seqlens=cu_seqlens,
        scale_val=scale_val,
    )
    output.copy_(ref_output)
    new_state.copy_(ref_new_state)
