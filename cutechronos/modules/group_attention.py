"""Fused GroupSelfAttention module for Chronos2.

Identical to TimeSelfAttention EXCEPT:
1. No RoPE -- position embeddings are not applied
2. Transposed axes -- attention operates along the batch dimension instead of time
3. Uses group_time_mask instead of time attention mask

Reference: chronos-forecasting/src/chronos/chronos2/layers.py (class GroupSelfAttention)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Import Triton kernels with pure-PyTorch fallbacks (same pattern as
# time_attention.py — avoids duplicating fallback implementations).
# ---------------------------------------------------------------------------

def _rms_layernorm_fallback(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """T5-style RMS LayerNorm: no mean subtraction, FP32 variance."""
    variance = x.float().pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    if weight.dtype in (torch.float16, torch.bfloat16):
        x = x.to(weight.dtype)
    return weight * x


def _unscaled_attention_fallback(
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


try:
    from cutechronos.triton_kernels.rms_layernorm import rms_layernorm
    _rms_layernorm = rms_layernorm
except (ImportError, ModuleNotFoundError):
    _rms_layernorm = _rms_layernorm_fallback

try:
    from cutechronos.triton_kernels.attention import unscaled_attention
    _unscaled_attention = unscaled_attention
except (ImportError, ModuleNotFoundError):
    _unscaled_attention = _unscaled_attention_fallback


class FusedGroupSelfAttention(nn.Module):
    """Fused GroupSelfAttention for inference.

    Combines RMS LayerNorm + Q/K/V projections + unscaled attention + O projection
    with batch/time axis transpose for group-wise attention.

    No RoPE is used because there's no natural ordering along the batch axis.
    Dropout is skipped at inference time.
    """

    def __init__(self, d_model: int, n_heads: int, d_kv: int, eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_kv = d_kv
        self.inner_dim = n_heads * d_kv
        self.eps = eps

        # LayerNorm weight (RMS norm, no bias)
        self.ln_weight = nn.Parameter(torch.ones(d_model))

        # Q, K, V, O projections (no bias, matching original MHA)
        self.q_weight = nn.Parameter(torch.empty(self.inner_dim, d_model))
        self.k_weight = nn.Parameter(torch.empty(self.inner_dim, d_model))
        self.v_weight = nn.Parameter(torch.empty(self.inner_dim, d_model))
        self.o_weight = nn.Parameter(torch.empty(d_model, self.inner_dim))

    def load_from_original(self, original_layer: nn.Module) -> None:
        """Copy weights from an original GroupSelfAttention layer.

        Args:
            original_layer: A GroupSelfAttention instance from chronos2 layers.
        """
        with torch.no_grad():
            self.ln_weight.copy_(original_layer.layer_norm.weight)
            self.q_weight.copy_(original_layer.self_attention.q.weight)
            self.k_weight.copy_(original_layer.self_attention.k.weight)
            self.v_weight.copy_(original_layer.self_attention.v.weight)
            self.o_weight.copy_(original_layer.self_attention.o.weight)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass matching original GroupSelfAttention.

        Args:
            hidden_states: (batch, time, d_model)
            attention_mask: (time, n_heads, batch, batch) additive attention mask.
                After transposing batch and time, the attention operates along the
                original batch dimension.

        Returns:
            (batch, time, d_model) - output with residual connection applied.
        """
        # Transpose batch and time axes via stride manipulation (no data copy)
        # (batch, time, d_model) -> (time, batch, d_model)
        x = hidden_states.transpose(0, 1)

        # RMS LayerNorm
        normed = _rms_layernorm(x, self.ln_weight, self.eps)

        # Q, K, V projections using cuBLAS via F.linear
        q = F.linear(normed, self.q_weight)  # (time, batch, inner_dim)
        k = F.linear(normed, self.k_weight)
        v = F.linear(normed, self.v_weight)

        # Reshape to multi-head: (time, batch, inner_dim) -> (time, n_heads, batch, d_kv)
        time_len, batch_size = q.shape[0], q.shape[1]
        q = q.view(time_len, batch_size, self.n_heads, self.d_kv).permute(0, 2, 1, 3)
        k = k.view(time_len, batch_size, self.n_heads, self.d_kv).permute(0, 2, 1, 3)
        v = v.view(time_len, batch_size, self.n_heads, self.d_kv).permute(0, 2, 1, 3)

        # Unscaled attention (no RoPE, no scaling by 1/sqrt(d))
        attn_output = _unscaled_attention(q, k, v, mask=attention_mask)

        # Reshape back: (time, n_heads, batch, d_kv) -> (time, batch, inner_dim)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(time_len, batch_size, self.inner_dim)

        # Output projection
        attn_output = F.linear(attn_output, self.o_weight)

        # Residual connection (still in transposed form)
        output = x + attn_output

        # Transpose back: (time, batch, d_model) -> (batch, time, d_model)
        return output.transpose(0, 1)
