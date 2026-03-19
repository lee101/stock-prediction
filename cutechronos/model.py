"""CuteChronos2Model: full model assembly with weight loading.

Assembles all kernel modules into a complete Chronos2 model that can load
HuggingFace checkpoint weights and produce identical outputs to the original.

Uses pure PyTorch fallback implementations for all submodules, with optional
Triton kernel swapins when available.
"""

from __future__ import annotations

import json
import time as _time_module
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as safetensors_load_file


# ---------------------------------------------------------------------------
# Config dataclass (mirrors Chronos2ForecastingConfig)
# ---------------------------------------------------------------------------

@dataclass
class CuteChronos2Config:
    d_model: int = 768
    d_kv: int = 64
    d_ff: int = 3072
    num_layers: int = 12
    num_heads: int = 12
    dropout_rate: float = 0.1
    layer_norm_epsilon: float = 1e-6
    dense_act_fn: str = "relu"
    rope_theta: float = 10000.0
    vocab_size: int = 2
    reg_token_id: int = 1
    # Chronos2-specific forecasting config
    context_length: int = 8192
    input_patch_size: int = 16
    input_patch_stride: int = 16
    output_patch_size: int = 16
    num_quantiles: int = 21
    quantiles: list[float] | None = None
    use_reg_token: bool = True
    use_arcsinh: bool = True
    time_encoding_scale: int | None = None

    def __post_init__(self):
        if self.quantiles is None:
            self.quantiles = [
                0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99,
            ]
        self.num_quantiles = len(self.quantiles)
        if self.time_encoding_scale is None:
            self.time_encoding_scale = self.context_length


# ---------------------------------------------------------------------------
# Pure PyTorch fallback building blocks
# ---------------------------------------------------------------------------

def rms_layernorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """T5-style RMS LayerNorm: no mean subtraction, FP32 variance."""
    variance = x.float().pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    if weight.dtype in (torch.float16, torch.bfloat16):
        x = x.to(weight.dtype)
    return weight * x


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def compute_cos_sin_fallback(
    inv_freq: torch.Tensor,
    position_ids: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute RoPE cos/sin from inv_freq and position_ids.

    Returns cos, sin each of shape (B, S, dim) where dim = 2 * len(inv_freq).
    """
    inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype)
    sin = emb.sin().to(dtype)
    return cos, sin


def apply_rope_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to Q and K. cos/sin: (B, S, D), unsqueeze for heads."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_out = q * cos + _rotate_half(q) * sin
    k_out = k * cos + _rotate_half(k) * sin
    return q_out, k_out


def unscaled_attention_fallback(
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


# ---------------------------------------------------------------------------
# Activation lookup
# ---------------------------------------------------------------------------

_ACT_FNS = {
    "relu": F.relu,
    "gelu": F.gelu,
    "silu": F.silu,
    "tanh": torch.tanh,
}


def _get_act_fn(name: str):
    if name in _ACT_FNS:
        return _ACT_FNS[name]
    raise ValueError(f"Unknown activation function: {name}")


# ---------------------------------------------------------------------------
# Sub-modules (pure PyTorch)
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Residual block: hidden + output + skip path."""

    def __init__(self, in_dim: int, h_dim: int, out_dim: int, act_fn_name: str = "relu"):
        super().__init__()
        self.hidden_layer = nn.Linear(in_dim, h_dim)
        self.output_layer = nn.Linear(h_dim, out_dim)
        self.residual_layer = nn.Linear(in_dim, out_dim)
        self.act = _get_act_fn(act_fn_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hid = self.act(self.hidden_layer(x))
        out = self.output_layer(hid)
        res = self.residual_layer(x)
        return out + res


class TimeSelfAttentionFallback(nn.Module):
    """Pure PyTorch TimeSelfAttention (LN + QKV + RoPE + Attn + O + residual)."""

    def __init__(self, config: CuteChronos2Config):
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.d_kv = config.d_kv
        self.inner_dim = config.num_heads * config.d_kv
        self.eps = config.layer_norm_epsilon

        self.layer_norm_weight = nn.Parameter(torch.ones(config.d_model))
        self.q = nn.Linear(config.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(config.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(config.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, config.d_model, bias=False)

        inv_freq = 1.0 / (
            config.rope_theta ** (torch.arange(0, config.d_kv, 2, dtype=torch.int64).float() / config.d_kv)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        B, S, _ = hidden_states.shape
        normed = rms_layernorm(hidden_states, self.layer_norm_weight, self.eps)

        q = F.linear(normed, self.q.weight).view(B, S, self.num_heads, self.d_kv).transpose(1, 2)
        k = F.linear(normed, self.k.weight).view(B, S, self.num_heads, self.d_kv).transpose(1, 2)
        v = F.linear(normed, self.v.weight).view(B, S, self.num_heads, self.d_kv).transpose(1, 2)

        cos, sin = compute_cos_sin_fallback(self.inv_freq, position_ids, q.dtype)
        q, k = apply_rope_fallback(q, k, cos, sin)

        attn_output = unscaled_attention_fallback(q, k, v, attention_mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, self.inner_dim)
        attn_output = F.linear(attn_output, self.o.weight)

        return hidden_states + attn_output


class GroupSelfAttentionFallback(nn.Module):
    """Pure PyTorch GroupSelfAttention (no RoPE, transposed axes)."""

    def __init__(self, config: CuteChronos2Config):
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.d_kv = config.d_kv
        self.inner_dim = config.num_heads * config.d_kv
        self.eps = config.layer_norm_epsilon

        self.layer_norm_weight = nn.Parameter(torch.ones(config.d_model))
        self.q = nn.Linear(config.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(config.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(config.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, config.d_model, bias=False)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # (batch, time, d_model) -> (time, batch, d_model)
        x = hidden_states.transpose(0, 1)
        normed = rms_layernorm(x, self.layer_norm_weight, self.eps)

        time_len, batch_size = normed.shape[0], normed.shape[1]
        q = F.linear(normed, self.q.weight).view(time_len, batch_size, self.num_heads, self.d_kv).permute(0, 2, 1, 3)
        k = F.linear(normed, self.k.weight).view(time_len, batch_size, self.num_heads, self.d_kv).permute(0, 2, 1, 3)
        v = F.linear(normed, self.v.weight).view(time_len, batch_size, self.num_heads, self.d_kv).permute(0, 2, 1, 3)

        attn_output = unscaled_attention_fallback(q, k, v, attention_mask)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(time_len, batch_size, self.inner_dim)
        attn_output = F.linear(attn_output, self.o.weight)

        output = x + attn_output
        return output.transpose(0, 1)


class FeedForwardFallback(nn.Module):
    """Pure PyTorch FeedForward (LN + MLP + residual)."""

    def __init__(self, config: CuteChronos2Config):
        super().__init__()
        self.layer_norm_weight = nn.Parameter(torch.ones(config.d_model))
        self.eps = config.layer_norm_epsilon
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.act = _get_act_fn(config.dense_act_fn)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed = rms_layernorm(hidden_states, self.layer_norm_weight, self.eps)
        ff_out = self.act(self.wi(normed))
        ff_out = self.wo(ff_out)
        return hidden_states + ff_out


class EncoderBlock(nn.Module):
    """Single encoder block: TimeSelfAttention + GroupSelfAttention + FeedForward."""

    def __init__(self, config: CuteChronos2Config):
        super().__init__()
        self.time_attn = TimeSelfAttentionFallback(config)
        self.group_attn = GroupSelfAttentionFallback(config)
        self.feed_forward = FeedForwardFallback(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        group_time_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.time_attn(hidden_states, attention_mask, position_ids)
        hidden_states = self.group_attn(hidden_states, group_time_mask)
        hidden_states = self.feed_forward(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# Instance Normalization (matches chronos_bolt.InstanceNorm)
# ---------------------------------------------------------------------------

class InstanceNorm(nn.Module):
    """Standardize along last dim, optionally apply arcsinh."""

    def __init__(self, eps: float = 1e-5, use_arcsinh: bool = False):
        super().__init__()
        self.eps = eps
        self.use_arcsinh = use_arcsinh

    def forward(
        self,
        x: torch.Tensor,
        loc_scale: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if loc_scale is None:
            loc = torch.nan_to_num(torch.nanmean(x, dim=-1, keepdim=True), nan=0.0)
            scale = torch.nan_to_num((x - loc).square().nanmean(dim=-1, keepdim=True).sqrt(), nan=1.0)
            scale = torch.where(scale == 0, self.eps, scale)
        else:
            loc, scale = loc_scale
        scaled_x = (x - loc) / scale
        if self.use_arcsinh:
            scaled_x = torch.arcsinh(scaled_x)
        return scaled_x.to(orig_dtype), (loc, scale)

    def inverse(self, x: torch.Tensor, loc_scale: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        loc, scale = loc_scale
        if self.use_arcsinh:
            x = torch.sinh(x)
        x = x * scale + loc
        return x.to(orig_dtype)


# ---------------------------------------------------------------------------
# Patching (matches chronos_bolt.Patch)
# ---------------------------------------------------------------------------

class Patch(nn.Module):
    def __init__(self, patch_size: int, patch_stride: int):
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]
        if length % self.patch_size != 0:
            padding_size = (*x.shape[:-1], self.patch_size - (length % self.patch_size))
            padding = torch.full(size=padding_size, fill_value=torch.nan, dtype=x.dtype, device=x.device)
            x = torch.cat((padding, x), dim=-1)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)
        return x


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class CuteChronos2Model(nn.Module):
    """Complete Chronos2 model with pure PyTorch fallbacks.

    Can load weights from a HuggingFace Chronos2 checkpoint and produce
    identical outputs to the original Chronos2Model.
    """

    def __init__(self, config: CuteChronos2Config):
        super().__init__()
        self.config = config

        # Embedding for special tokens (PAD=0, REG=1)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # Input patch embedding: in_dim = patch_size * 3 (time_enc, patch, mask)
        in_dim = config.input_patch_size * 3
        self.input_patch_embedding = ResidualBlock(
            in_dim=in_dim,
            h_dim=config.d_ff,
            out_dim=config.d_model,
            act_fn_name=config.dense_act_fn,
        )

        # Patching and instance norm
        self.patch = Patch(config.input_patch_size, config.input_patch_stride)
        self.instance_norm = InstanceNorm(use_arcsinh=config.use_arcsinh)

        # Encoder blocks
        self.blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config.num_layers)])

        # Final layer norm
        self.final_layer_norm_weight = nn.Parameter(torch.ones(config.d_model))

        # Output patch embedding
        out_dim = config.num_quantiles * config.output_patch_size
        self.output_patch_embedding = ResidualBlock(
            in_dim=config.d_model,
            h_dim=config.d_ff,
            out_dim=out_dim,
            act_fn_name=config.dense_act_fn,
        )

        # Quantiles buffer
        quantiles = torch.tensor(config.quantiles, dtype=torch.float32)
        self.register_buffer("quantiles", quantiles, persistent=False)

    def _prepare_patched_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Normalize, patch, and prepare context with time encodings."""
        dtype = self._param_dtype()

        if context_mask is not None:
            context_mask = context_mask.to(context.dtype)
        else:
            context_mask = torch.isnan(context).logical_not().to(context.dtype)

        batch_size, context_length = context.shape
        if context_length > self.config.context_length:
            context = context[..., -self.config.context_length:]
            context_mask = context_mask[..., -self.config.context_length:]

        # Instance normalization (in float32)
        context, loc_scale = self.instance_norm(context)
        context = context.to(dtype)
        context_mask = context_mask.to(dtype)

        # Patching
        patched_context = self.patch(context)
        patched_mask = torch.nan_to_num(self.patch(context_mask), nan=0.0)
        patched_context = torch.where(patched_mask > 0.0, patched_context, 0.0)

        # Attention mask: 1 if at least one item in patch is observed
        attention_mask = patched_mask.sum(dim=-1) > 0  # (B, num_patches)
        num_context_patches = attention_mask.shape[-1]

        # Context time encoding
        final_context_length = num_context_patches * self.config.input_patch_size
        context_time_enc = torch.arange(
            start=-final_context_length, end=0, device=context.device, dtype=torch.float32
        )
        context_time_enc = (
            context_time_enc
            .view(1, num_context_patches, self.config.input_patch_size)
            .expand(batch_size, -1, -1)
            .div(cast(int, self.config.time_encoding_scale))
            .to(dtype)
        )

        patched_context = torch.cat([context_time_enc, patched_context, patched_mask], dim=-1)
        return patched_context, attention_mask, loc_scale

    def _prepare_patched_future(
        self,
        num_output_patches: int,
        batch_size: int,
    ) -> torch.Tensor:
        """Prepare future patches (no covariates, just zeros + time encoding)."""
        dtype = self._param_dtype()
        output_patch_size = self.config.output_patch_size

        patched_future_covariates = torch.zeros(
            batch_size, num_output_patches, output_patch_size,
            device=self._param_device(), dtype=dtype,
        )
        patched_future_covariates_mask = torch.zeros_like(patched_future_covariates)

        final_future_length = num_output_patches * output_patch_size
        future_time_enc = torch.arange(
            start=0, end=final_future_length,
            device=self._param_device(), dtype=torch.float32,
        )
        future_time_enc = (
            future_time_enc
            .view(1, num_output_patches, output_patch_size)
            .expand(batch_size, -1, -1)
            .div(cast(int, self.config.time_encoding_scale))
            .to(dtype)
        )

        return torch.cat([future_time_enc, patched_future_covariates, patched_future_covariates_mask], dim=-1)

    @staticmethod
    def _expand_and_invert_time_attention_mask(
        attention_mask: torch.Tensor, floating_type: torch.dtype
    ) -> torch.Tensor:
        attention_mask = attention_mask[:, None, None, :]
        attention_mask = attention_mask.to(dtype=floating_type)
        attention_mask = (1.0 - attention_mask) * torch.finfo(floating_type).min
        return attention_mask

    @staticmethod
    def _construct_and_invert_group_time_mask(
        group_ids: torch.Tensor, attention_mask: torch.Tensor, floating_type: torch.dtype
    ) -> torch.Tensor:
        group_mask = group_ids[:, None] == group_ids[None, :]
        group_time_mask = torch.einsum("qb, bt -> qbt", group_mask.to(floating_type), attention_mask.to(floating_type))
        # reshape: (q, b, t) -> (t, 1, q, b) to match attention scores shape
        group_time_mask = group_time_mask.permute(2, 0, 1).unsqueeze(1)  # (t, 1, q, b)
        group_time_mask = (1.0 - group_time_mask) * torch.finfo(floating_type).min
        return group_time_mask

    def _param_dtype(self) -> torch.dtype:
        """Get the dtype of model parameters."""
        return self.shared.weight.dtype

    def _param_device(self) -> torch.device:
        """Get the device of model parameters."""
        return self.shared.weight.device

    def forward(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        num_output_patches: int = 1,
    ) -> torch.Tensor:
        """Forward pass producing quantile predictions.

        Args:
            context: (B, L) raw time series values (may contain NaN)
            context_mask: (B, L) binary mask (1=valid, 0=missing), optional
            num_output_patches: number of output patches to predict

        Returns:
            quantile_preds: (B, Q, H) where Q=num_quantiles, H=num_output_patches*patch_size
        """
        dtype = self._param_dtype()
        batch_size = context.shape[0]

        # Prepare context
        patched_context, attention_mask, loc_scale = self._prepare_patched_context(context, context_mask)
        num_context_patches = attention_mask.shape[-1]

        # Input embedding
        input_embeds = self.input_patch_embedding(patched_context)

        # Append REG token
        if self.config.use_reg_token:
            reg_input_ids = torch.full(
                (batch_size, 1), self.config.reg_token_id,
                device=input_embeds.device, dtype=torch.long,
            )
            reg_embeds = self.shared(reg_input_ids)
            input_embeds = torch.cat([input_embeds, reg_embeds], dim=-2)
            attention_mask = torch.cat(
                [attention_mask.to(dtype), torch.ones(batch_size, 1, device=input_embeds.device, dtype=dtype)],
                dim=-1,
            )

        # Prepare future patches
        patched_future = self._prepare_patched_future(num_output_patches, batch_size)
        future_embeds = self.input_patch_embedding(patched_future)
        future_attention_mask = torch.ones(
            batch_size, num_output_patches, dtype=dtype, device=input_embeds.device,
        )

        # Concatenate context + future
        input_embeds = torch.cat([input_embeds, future_embeds], dim=-2)
        attention_mask = torch.cat([attention_mask.to(dtype), future_attention_mask], dim=-1)

        # Group IDs: each series independent
        group_ids = torch.arange(batch_size, dtype=torch.long, device=input_embeds.device)

        # Build masks
        seq_length = input_embeds.shape[1]
        extended_attention_mask = self._expand_and_invert_time_attention_mask(attention_mask, dtype)
        group_time_mask = self._construct_and_invert_group_time_mask(group_ids, attention_mask, dtype)

        # Position IDs
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_embeds.device).unsqueeze(0)

        # Encoder forward
        hidden_states = input_embeds  # Note: original has dropout here, skip for inference
        for block in self.blocks:
            hidden_states = block(hidden_states, position_ids, extended_attention_mask, group_time_mask)

        # Final layer norm
        hidden_states = rms_layernorm(hidden_states, self.final_layer_norm_weight, self.config.layer_norm_epsilon)

        # Extract last num_output_patches hidden states
        forecast_embeds = hidden_states[:, -num_output_patches:]

        # Output patch embedding
        quantile_preds = self.output_patch_embedding(forecast_embeds)

        # Rearrange: (B, N, Q*P) -> (B, Q, N*P)
        quantile_preds = quantile_preds.view(
            batch_size, num_output_patches, self.config.num_quantiles, self.config.output_patch_size,
        )
        quantile_preds = quantile_preds.permute(0, 2, 1, 3).contiguous()
        quantile_preds = quantile_preds.view(
            batch_size, self.config.num_quantiles, num_output_patches * self.config.output_patch_size,
        )

        # Unscale predictions (inverse instance norm)
        horizon = num_output_patches * self.config.output_patch_size
        quantile_preds = quantile_preds.view(batch_size, self.config.num_quantiles * horizon)
        quantile_preds = self.instance_norm.inverse(quantile_preds, loc_scale)
        quantile_preds = quantile_preds.view(batch_size, self.config.num_quantiles, horizon)

        return quantile_preds

    def load_chronos2_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load weights from a Chronos2 HuggingFace checkpoint state dict.

        Maps the original weight names to our module structure.
        """
        with torch.no_grad():
            # Shared embedding
            self.shared.weight.copy_(state_dict["shared.weight"])

            # Input and output patch embeddings
            for name in ("input_patch_embedding", "output_patch_embedding"):
                block = getattr(self, name)
                for layer in ("hidden_layer", "output_layer", "residual_layer"):
                    for param in ("weight", "bias"):
                        key = f"{name}.{layer}.{param}"
                        getattr(getattr(block, layer), param).copy_(state_dict[key])

            # Final layer norm
            self.final_layer_norm_weight.copy_(state_dict["encoder.final_layer_norm.weight"])

            # Encoder blocks
            for i, block in enumerate(self.blocks):
                prefix = f"encoder.block.{i}"

                # TimeSelfAttention
                block.time_attn.layer_norm_weight.copy_(
                    state_dict[f"{prefix}.layer.0.layer_norm.weight"]
                )
                block.time_attn.q.weight.copy_(
                    state_dict[f"{prefix}.layer.0.self_attention.q.weight"]
                )
                block.time_attn.k.weight.copy_(
                    state_dict[f"{prefix}.layer.0.self_attention.k.weight"]
                )
                block.time_attn.v.weight.copy_(
                    state_dict[f"{prefix}.layer.0.self_attention.v.weight"]
                )
                block.time_attn.o.weight.copy_(
                    state_dict[f"{prefix}.layer.0.self_attention.o.weight"]
                )

                # GroupSelfAttention
                block.group_attn.layer_norm_weight.copy_(
                    state_dict[f"{prefix}.layer.1.layer_norm.weight"]
                )
                block.group_attn.q.weight.copy_(
                    state_dict[f"{prefix}.layer.1.self_attention.q.weight"]
                )
                block.group_attn.k.weight.copy_(
                    state_dict[f"{prefix}.layer.1.self_attention.k.weight"]
                )
                block.group_attn.v.weight.copy_(
                    state_dict[f"{prefix}.layer.1.self_attention.v.weight"]
                )
                block.group_attn.o.weight.copy_(
                    state_dict[f"{prefix}.layer.1.self_attention.o.weight"]
                )

                # FeedForward
                block.feed_forward.layer_norm_weight.copy_(
                    state_dict[f"{prefix}.layer.2.layer_norm.weight"]
                )
                block.feed_forward.wi.weight.copy_(
                    state_dict[f"{prefix}.layer.2.mlp.wi.weight"]
                )
                block.feed_forward.wo.weight.copy_(
                    state_dict[f"{prefix}.layer.2.mlp.wo.weight"]
                )

    @classmethod
    def from_pretrained(cls, model_path: str | Path) -> "CuteChronos2Model":
        """Load a CuteChronos2Model from a HuggingFace Chronos2 checkpoint directory.

        Args:
            model_path: path to directory containing config.json and model weights
                        (safetensors or pytorch bin format)

        Returns:
            Initialized model with loaded weights, in eval mode.
        """
        model_path = Path(model_path)

        # Load config
        config_path = model_path / "config.json"
        with open(config_path) as f:
            raw_config = json.load(f)

        chronos_cfg = raw_config.get("chronos_config", {})
        config = CuteChronos2Config(
            d_model=raw_config.get("d_model", 768),
            d_kv=raw_config.get("d_kv", 64),
            d_ff=raw_config.get("d_ff", 3072),
            num_layers=raw_config.get("num_layers", 12),
            num_heads=raw_config.get("num_heads", 12),
            dropout_rate=raw_config.get("dropout_rate", 0.1),
            layer_norm_epsilon=raw_config.get("layer_norm_epsilon", 1e-6),
            dense_act_fn=raw_config.get("dense_act_fn", "relu"),
            rope_theta=raw_config.get("rope_theta", 10000.0),
            vocab_size=raw_config.get("vocab_size", 2),
            reg_token_id=raw_config.get("reg_token_id", 1),
            context_length=chronos_cfg.get("context_length", 8192),
            input_patch_size=chronos_cfg.get("input_patch_size", 16),
            input_patch_stride=chronos_cfg.get("input_patch_stride", 16),
            output_patch_size=chronos_cfg.get("output_patch_size", 16),
            quantiles=chronos_cfg.get("quantiles"),
            use_reg_token=chronos_cfg.get("use_reg_token", True),
            use_arcsinh=chronos_cfg.get("use_arcsinh", True),
            time_encoding_scale=chronos_cfg.get("time_encoding_scale"),
        )

        model = cls(config)

        # Load weights
        safetensors_path = model_path / "model.safetensors"
        bin_path = model_path / "pytorch_model.bin"
        if safetensors_path.exists():
            state_dict = safetensors_load_file(str(safetensors_path))
        elif bin_path.exists():
            state_dict = torch.load(str(bin_path), map_location="cpu", weights_only=True)
        else:
            raise FileNotFoundError(
                f"No model weights found in {model_path}. "
                f"Expected model.safetensors or pytorch_model.bin"
            )

        model.load_chronos2_weights(state_dict)
        model.eval()
        return model

    @classmethod
    def from_original(cls, original_model) -> "CuteChronos2Model":
        """Create a CuteChronos2Model from an already-loaded original Chronos2Model.

        Args:
            original_model: a chronos.chronos2.model.Chronos2Model instance

        Returns:
            Initialized CuteChronos2Model with copied weights, in eval mode.
        """
        orig_config = original_model.config
        chronos_config = original_model.chronos_config

        config = CuteChronos2Config(
            d_model=orig_config.d_model,
            d_kv=orig_config.d_kv,
            d_ff=orig_config.d_ff,
            num_layers=orig_config.num_layers,
            num_heads=orig_config.num_heads,
            dropout_rate=orig_config.dropout_rate,
            layer_norm_epsilon=orig_config.layer_norm_epsilon,
            dense_act_fn=orig_config.dense_act_fn,
            rope_theta=orig_config.rope_theta,
            vocab_size=orig_config.vocab_size,
            reg_token_id=getattr(orig_config, "reg_token_id", 1),
            context_length=chronos_config.context_length,
            input_patch_size=chronos_config.input_patch_size,
            input_patch_stride=chronos_config.input_patch_stride,
            output_patch_size=chronos_config.output_patch_size,
            quantiles=chronos_config.quantiles,
            use_reg_token=chronos_config.use_reg_token,
            use_arcsinh=chronos_config.use_arcsinh,
            time_encoding_scale=chronos_config.time_encoding_scale,
        )

        model = cls(config)
        state_dict = original_model.state_dict()
        model.load_chronos2_weights(state_dict)
        model.eval()
        return model

    @classmethod
    def from_pretrained_compiled(
        cls,
        model_path: str | Path,
        compile_mode: str = "reduce-overhead",
    ) -> "CuteChronos2Model":
        """Load a CuteChronos2Model and apply torch.compile for faster inference.

        The forward method is compiled while preprocessing (NaN handling,
        instance normalization) runs in eager mode to avoid graph breaks.

        Args:
            model_path: path to directory containing config.json and model weights
            compile_mode: torch.compile mode (e.g., "reduce-overhead", "max-autotune",
                          "default"). Defaults to "reduce-overhead" for best latency.

        Returns:
            Initialized model with compiled forward, in eval mode.
        """
        model = cls.from_pretrained(model_path)
        model = _apply_torch_compile(model, compile_mode=compile_mode)
        return model

    @torch.inference_mode()
    def predict(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        num_output_patches: int = 1,
    ) -> torch.Tensor:
        """Run inference with torch.inference_mode for optimal performance.

        This is the recommended entry point for inference. It wraps forward()
        with inference_mode which disables autograd tracking and version
        counting for maximum throughput.

        Args:
            context: (B, L) raw time series values (may contain NaN)
            context_mask: (B, L) binary mask (1=valid, 0=missing), optional
            num_output_patches: number of output patches to predict

        Returns:
            quantile_preds: (B, Q, H) where Q=num_quantiles, H=num_output_patches*patch_size
        """
        return self.forward(context, context_mask=context_mask, num_output_patches=num_output_patches)


def _apply_torch_compile(
    model: CuteChronos2Model,
    compile_mode: str = "reduce-overhead",
) -> CuteChronos2Model:
    """Apply torch.compile to a CuteChronos2Model's forward method.

    Uses fullgraph=False so torch.compile can handle graph breaks from
    NaN handling and dynamic control flow in preprocessing gracefully.

    Args:
        model: the model to compile (must be in eval mode)
        compile_mode: torch.compile mode

    Returns:
        The same model instance with compiled forward.
    """
    if not hasattr(torch, "compile"):
        print("[cutechronos] torch.compile not available, using eager mode.")
        return model

    try:
        model.forward = torch.compile(  # type: ignore[assignment]
            model.forward,
            mode=compile_mode,
            fullgraph=False,
        )
        print(f"[cutechronos] torch.compile enabled (mode={compile_mode}).")
    except Exception as exc:
        print(f"[cutechronos] torch.compile failed ({exc}); using eager mode.")
    return model


def benchmark_eager_vs_compiled(
    model_path: str | Path,
    *,
    context_length: int = 512,
    batch_size: int = 4,
    warmup_iters: int = 3,
    bench_iters: int = 10,
    device: str = "cpu",
    compile_mode: str = "reduce-overhead",
) -> dict[str, float]:
    """Benchmark eager vs compiled CuteChronos2Model inference.

    Args:
        model_path: path to Chronos2 checkpoint directory
        context_length: length of input context
        batch_size: batch size for benchmark
        warmup_iters: number of warmup iterations (not timed)
        bench_iters: number of benchmark iterations
        device: device to run on ("cpu" or "cuda")
        compile_mode: torch.compile mode

    Returns:
        Dict with eager_ms, compiled_ms, speedup keys.
    """
    torch.manual_seed(42)
    context = torch.randn(batch_size, context_length, device=device) * 0.1 + 100.0

    # Eager model
    eager_model = CuteChronos2Model.from_pretrained(model_path)
    eager_model = eager_model.to(device).eval()

    # Warmup eager
    for _ in range(warmup_iters):
        eager_model.predict(context)
    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark eager
    t0 = _time_module.perf_counter()
    for _ in range(bench_iters):
        eager_model.predict(context)
    if device == "cuda":
        torch.cuda.synchronize()
    eager_total = _time_module.perf_counter() - t0
    eager_ms = (eager_total / bench_iters) * 1000.0

    # Compiled model
    compiled_model = CuteChronos2Model.from_pretrained_compiled(model_path, compile_mode=compile_mode)
    compiled_model = compiled_model.to(device).eval()

    # Warmup compiled (includes compilation)
    for _ in range(warmup_iters):
        compiled_model.predict(context)
    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark compiled
    t0 = _time_module.perf_counter()
    for _ in range(bench_iters):
        compiled_model.predict(context)
    if device == "cuda":
        torch.cuda.synchronize()
    compiled_total = _time_module.perf_counter() - t0
    compiled_ms = (compiled_total / bench_iters) * 1000.0

    speedup = eager_ms / compiled_ms if compiled_ms > 0 else 0.0

    print(f"[cutechronos benchmark] eager={eager_ms:.1f}ms  compiled={compiled_ms:.1f}ms  speedup={speedup:.2f}x")
    return {"eager_ms": eager_ms, "compiled_ms": compiled_ms, "speedup": speedup}
