"""Pure-PyTorch fallback implementations for Triton kernels.

These are used when Triton kernels are unavailable (import fails) or when
tensors are on CPU (Triton requires CUDA).

All functions match the semantics of the original Chronos2 operations exactly.
"""

from __future__ import annotations

import torch


def rms_layernorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """T5-style RMS LayerNorm: no mean subtraction, FP32 variance."""
    variance = x.float().pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    if weight.dtype in (torch.float16, torch.bfloat16):
        x = x.to(weight.dtype)
    return weight * x


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dimension into the first half (negated)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to Q and K.  cos/sin are (B, S, D); unsqueeze for heads."""
    cos = cos.unsqueeze(1)  # (B, 1, S, D)
    sin = sin.unsqueeze(1)
    q_out = q * cos + rotate_half(q) * sin
    k_out = k * cos + rotate_half(k) * sin
    return q_out, k_out


def compute_cos_sin(
    inv_freq: torch.Tensor,
    position_ids: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce the original RoPE.forward cos/sin computation.

    Args:
        inv_freq: (dim/2,) buffer from RoPE
        position_ids: (B, S) long tensor
        dtype: target dtype for cos/sin

    Returns:
        cos, sin each of shape (B, S, dim)
    """
    # inv_freq_expanded: (B, dim/2, 1)
    inv_freq_expanded = (
        inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    )
    # position_ids_expanded: (B, 1, S)
    position_ids_expanded = position_ids[:, None, :].float()
    # freqs: (B, dim/2, S) -> (B, S, dim/2)
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
    # emb: (B, S, dim)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype)
    sin = emb.sin().to(dtype)
    return cos, sin


def unscaled_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Unscaled dot-product attention (NO 1/sqrt(d_k) scaling)."""
    scores = torch.matmul(q, k.transpose(-2, -1))
    if mask is not None:
        scores = scores + mask
    weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.matmul(weights, v)
