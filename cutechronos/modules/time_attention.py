"""Fused TimeSelfAttention module for Chronos2 inference.

Replaces the original TimeSelfAttention (RMSNorm -> QKV proj -> RoPE ->
unscaled attention -> O proj -> residual) with a streamlined version that
uses the fastest available attention backend:
  1. SDPA with scale=1.0 (preferred on CUDA -- auto-selects FlashAttention2/cuDNN)
  2. Triton custom kernel (if available)
  3. Pure-PyTorch eager fallback

cuBLAS-backed F.linear is used for the QKV / O projections (already fast
for 768x768).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from cutechronos.modules._fallbacks import (
    rms_layernorm as _rms_layernorm_fallback,
    apply_rope as _apply_rope_fallback,
    compute_cos_sin as _compute_cos_sin_fallback,
    unscaled_attention as _unscaled_attention_fallback,
)
from cutechronos.modules.flex_attention import sdpa_unscaled_attention as _sdpa_attn

# ---------------------------------------------------------------------------
# Try importing Triton kernels; fall back to PyTorch implementations.
# ---------------------------------------------------------------------------

try:
    from cutechronos.triton_kernels.rms_layernorm import rms_layernorm as _rms_layernorm_triton
    _has_triton_rms = True
except (ImportError, ModuleNotFoundError):
    _has_triton_rms = False

try:
    from cutechronos.triton_kernels.rope import apply_rope as _triton_apply_rope
    _has_triton_rope = True
except (ImportError, ModuleNotFoundError):
    _has_triton_rope = False

try:
    from cutechronos.triton_kernels.attention import unscaled_attention as _unscaled_attention_triton
    _has_triton_attn = True
except (ImportError, ModuleNotFoundError):
    _has_triton_attn = False


class FusedTimeSelfAttention(nn.Module):
    """Drop-in replacement for Chronos2 ``TimeSelfAttention``.

    Fuses RMSNorm + QKV projection + RoPE + unscaled attention + O
    projection + residual into a single module optimised for inference.

    Weight shapes are identical to the original so that
    ``load_from_original`` can copy parameters directly.
    """

    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 12,
        d_kv: int = 64,
        layer_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_kv = d_kv
        self.inner_dim = num_heads * d_kv
        self.layer_norm_eps = layer_norm_eps

        # Layer norm weight (T5-style RMS, no bias)
        self.layer_norm_weight = nn.Parameter(torch.ones(d_model))

        # QKV + O projections (no bias, same as original)
        self.q = nn.Linear(d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, d_model, bias=False)

        # RoPE inverse frequencies
        inv_freq = 1.0 / (
            rope_theta
            ** (torch.arange(0, d_kv, 2, dtype=torch.int64).float() / d_kv)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def load_from_original(self, original_layer: nn.Module) -> None:
        """Copy weights from a Chronos2 ``TimeSelfAttention`` layer.

        ``original_layer`` is expected to have:
        - ``layer_norm.weight``
        - ``self_attention.{q,k,v,o}.weight``
        - ``self_attention.rope_embed.inv_freq``
        """
        with torch.no_grad():
            self.layer_norm_weight.copy_(original_layer.layer_norm.weight)
            self.q.weight.copy_(original_layer.self_attention.q.weight)
            self.k.weight.copy_(original_layer.self_attention.k.weight)
            self.v.weight.copy_(original_layer.self_attention.v.weight)
            self.o.weight.copy_(original_layer.self_attention.o.weight)
            self.inv_freq.copy_(original_layer.self_attention.rope_embed.inv_freq)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, None]:
        """Forward pass matching the original ``TimeSelfAttention.forward`` signature.

        Args:
            hidden_states: (B, S, d_model)
            attention_mask: (B, num_heads, S, S) additive mask
            position_ids: (B, S) long tensor
            output_attentions: ignored (always returns None for weights)

        Returns:
            (output_hidden_states, None)
        """
        B, S, _ = hidden_states.shape
        on_cuda = hidden_states.is_cuda

        # 1. RMS LayerNorm
        if on_cuda and _has_triton_rms:
            normed = _rms_layernorm_triton(hidden_states, self.layer_norm_weight, self.layer_norm_eps)
        else:
            normed = _rms_layernorm_fallback(hidden_states, self.layer_norm_weight, self.layer_norm_eps)

        # 2-3. QKV projections -> reshape to (B, H, S, d_kv)
        query_states = F.linear(normed, self.q.weight).view(B, S, self.num_heads, self.d_kv).transpose(1, 2)
        key_states = F.linear(normed, self.k.weight).view(B, S, self.num_heads, self.d_kv).transpose(1, 2)
        value_states = F.linear(normed, self.v.weight).view(B, S, self.num_heads, self.d_kv).transpose(1, 2)

        # 4. Apply RoPE
        if on_cuda and _has_triton_rope:
            query_states, key_states = _triton_apply_rope(
                query_states, key_states, self.inv_freq, position_ids
            )
        else:
            cos, sin = _compute_cos_sin_fallback(
                self.inv_freq, position_ids, query_states.dtype
            )
            query_states, key_states = _apply_rope_fallback(
                query_states, key_states, cos, sin
            )

        # 5. Unscaled attention (SDPA preferred: auto-selects FlashAttn2/cuDNN)
        if on_cuda:
            attn_output = _sdpa_attn(query_states, key_states, value_states, attention_mask)
        elif _has_triton_attn:
            attn_output = _unscaled_attention_triton(query_states, key_states, value_states, attention_mask)
        else:
            attn_output = _unscaled_attention_fallback(query_states, key_states, value_states, attention_mask)

        # 6. Reshape (B, H, S, d_kv) -> (B, S, inner_dim) and O projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, self.inner_dim)
        attn_output = F.linear(attn_output, self.o.weight)

        # 7. Residual connection (dropout is no-op at inference)
        output = hidden_states + attn_output

        return output, None
