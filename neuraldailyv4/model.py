"""V4 Transformer model with Chronos-2 inspired architecture.

Key features:
- Patching: 5-day patches for faster processing
- Multi-window: Direct prediction of multiple future windows
- Quantile outputs: Price distribution instead of point estimates
- Learned position sizing: Dynamic sizing based on uncertainty
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuraldailyv4.config import PolicyConfigV4


def rms_norm(x: torch.Tensor) -> torch.Tensor:
    """RMSNorm without learnable parameters."""
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary position embeddings."""
    x_r = x[..., ::2]
    x_i = x[..., 1::2]
    out_r = x_r * cos - x_i * sin
    out_i = x_r * sin + x_i * cos
    out = torch.stack([out_r, out_i], dim=-1)
    return out.flatten(-2)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE)."""

    def __init__(self, dim: int, max_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        cos = freqs.cos()[None, :, None, :]
        sin = freqs.sin()[None, :, None, :]
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)
        self.cached_len = seq_len

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cached_len:
            self._build_cache(seq_len)
        return self.cos_cached[:, :seq_len], self.sin_cached[:, :seq_len]


class PatchEmbedding(nn.Module):
    """
    Embed time series patches (Chronos-2 style).

    Groups sequential timesteps into patches and embeds them.
    """

    def __init__(self, config: PolicyConfigV4):
        super().__init__()
        self.patch_size = config.patch_size
        self.input_dim = config.input_dim

        # Linear projection from flattened patch to hidden dim
        patch_dim = config.patch_size * config.input_dim
        self.embed = nn.Linear(patch_dim, config.hidden_dim, bias=False)

        # Optional residual MLP for richer patch representations
        if config.patch_residual:
            self.residual = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim * 2, bias=False),
                nn.GELU(),
                nn.Linear(config.hidden_dim * 2, config.hidden_dim, bias=False),
            )
        else:
            self.residual = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            (batch, num_patches, hidden_dim)
        """
        B, T, D = x.shape
        num_patches = T // self.patch_size

        # Truncate to full patches
        x = x[:, :num_patches * self.patch_size]

        # Reshape to patches: (B, num_patches, patch_size * input_dim)
        x = x.view(B, num_patches, self.patch_size * D)

        # Embed
        h = self.embed(x)

        # Residual
        if self.residual is not None:
            h = h + self.residual(h)

        return h


class MultiQueryAttention(nn.Module):
    """Multi-Query Attention with QK normalization."""

    def __init__(self, config: PolicyConfigV4, layer_idx: int):
        super().__init__()
        self.n_head = config.num_heads
        self.n_kv_head = config.num_kv_heads
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(config.hidden_dim, config.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, config.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, config.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = x.size()
        cos, sin = cos_sin

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim)

        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.n_kv_head != self.n_head:
            n_rep = self.n_head // self.n_kv_head
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            is_causal=True,
            dropout_p=self.dropout.p if self.training else 0.0,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.out_proj(y)


class CrossSymbolAttention(nn.Module):
    """Cross-symbol attention (group attention in Chronos-2 terms)."""

    def __init__(self, config: PolicyConfigV4):
        super().__init__()
        self.n_head = config.num_heads
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(config.hidden_dim, config.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, config.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, config.num_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        group_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, D = x.shape
        x_t = x.transpose(0, 1)

        q = self.q_proj(x_t).view(T, B, self.n_head, self.head_dim)
        k = self.k_proj(x_t).view(T, B, self.n_head, self.head_dim)
        v = self.v_proj(x_t).view(T, B, self.n_head, self.head_dim)

        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.einsum("thbd,thcd->thbc", q, k) * scale

        if group_mask is not None:
            mask = group_mask | torch.eye(B, device=group_mask.device, dtype=torch.bool)
            mask = mask[None, None, :, :]
            attn = attn.masked_fill(~mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        y = torch.einsum("thbc,thcd->thbd", attn, v)
        y = y.permute(2, 0, 1, 3).contiguous()
        y = y.view(B, T, -1)
        return self.out_proj(y)


class MLP(nn.Module):
    """MLP with ReLU^2 activation."""

    def __init__(self, config: PolicyConfigV4):
        super().__init__()
        self.fc = nn.Linear(config.hidden_dim, 4 * config.hidden_dim, bias=False)
        self.proj = nn.Linear(4 * config.hidden_dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = F.relu(x).square()
        x = self.dropout(x)
        return self.proj(x)


class TransformerBlock(nn.Module):
    """Transformer block with time attention, cross-symbol attention, and MLP."""

    def __init__(self, config: PolicyConfigV4, layer_idx: int):
        super().__init__()
        self.time_attn = MultiQueryAttention(config, layer_idx)
        self.mlp = MLP(config)
        self.cross_attn = CrossSymbolAttention(config) if config.use_cross_attention else None

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: Tuple[torch.Tensor, torch.Tensor],
        group_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.time_attn(rms_norm(x), cos_sin)

        if self.cross_attn is not None:
            x = x + self.cross_attn(rms_norm(x), group_mask)

        x = x + self.mlp(rms_norm(x))
        return x


class MultiWindowHead(nn.Module):
    """
    Multi-window output head with quantile predictions.

    Outputs predictions for multiple future windows simultaneously,
    with quantile estimates for prices.
    """

    def __init__(self, config: PolicyConfigV4):
        super().__init__()
        self.config = config
        self.num_windows = config.num_windows
        self.num_quantiles = config.num_quantiles

        # Per window outputs:
        # - buy_quantiles: num_quantiles (e.g., 3 for q25, q50, q75)
        # - sell_quantiles: num_quantiles
        # - position_size: 1
        # - confidence: 1
        # - exit_day_in_window: 1
        outputs_per_window = config.num_quantiles * 2 + 3
        self.outputs_per_window = outputs_per_window

        self.head = nn.Linear(config.hidden_dim, config.num_windows * outputs_per_window, bias=False)

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            h: (batch, hidden_dim) - pooled representation

        Returns:
            Dict with multi-window predictions
        """
        # Project to all window outputs
        logits = self.head(h)  # (batch, num_windows * outputs_per_window)
        logits = logits.view(-1, self.num_windows, self.outputs_per_window)

        nq = self.num_quantiles

        return {
            "buy_quantile_logits": logits[..., :nq],           # (batch, num_windows, num_quantiles)
            "sell_quantile_logits": logits[..., nq:2*nq],      # (batch, num_windows, num_quantiles)
            "position_size_logits": logits[..., 2*nq:2*nq+1],  # (batch, num_windows, 1)
            "confidence_logits": logits[..., 2*nq+1:2*nq+2],   # (batch, num_windows, 1)
            "exit_day_logits": logits[..., 2*nq+2:2*nq+3],     # (batch, num_windows, 1)
        }


class MultiSymbolPolicyV4(nn.Module):
    """
    V4 Policy network with Chronos-2 inspired architecture.

    Key features:
    - Patching: Processes 5-day patches instead of individual days
    - Multi-window: Predicts multiple future windows directly
    - Quantile outputs: Predicts price distributions
    - Learned position sizing: Dynamic sizing based on uncertainty
    """

    def __init__(self, config: PolicyConfigV4):
        super().__init__()
        self.config = config

        # V4: Patch embedding instead of per-timestep
        if config.use_patch_embedding:
            self.patch_embed = PatchEmbedding(config)
        else:
            self.patch_embed = None
            self.embed = nn.Linear(config.input_dim, config.hidden_dim, bias=False)

        # Rotary embeddings (for patch positions)
        self.rope = RotaryEmbedding(config.head_dim, max_len=config.max_len)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config, i) for i in range(config.num_layers)
        ])

        # V4: Multi-window output head
        self.multiwindow_head = MultiWindowHead(config)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                fan_out = module.weight.size(0)
                fan_in = module.weight.size(1)
                std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
                nn.init.normal_(module.weight, mean=0.0, std=std)

        # Small init for output head
        nn.init.normal_(self.multiwindow_head.head.weight, mean=0.0, std=0.1)

        for block in self.blocks:
            nn.init.zeros_(block.time_attn.out_proj.weight)
            nn.init.zeros_(block.mlp.proj.weight)
            if block.cross_attn is not None:
                nn.init.zeros_(block.cross_attn.out_proj.weight)

    def forward(
        self,
        features: torch.Tensor,
        group_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: (batch, seq_len, input_dim) - normalized features
            group_mask: (batch, batch) - cross-symbol attention mask

        Returns:
            Dict with multi-window logits
        """
        # Patch embedding
        if self.patch_embed is not None:
            h = self.patch_embed(features)  # (batch, num_patches, hidden_dim)
        else:
            h = self.embed(features)

        h = rms_norm(h)

        # Get sequence length after patching
        T = h.size(1)
        cos_sin = self.rope(T)

        # Transformer blocks
        for block in self.blocks:
            h = block(h, cos_sin, group_mask)

        h = rms_norm(h)

        # Pool: use last patch (most recent information)
        h_pooled = h[:, -1, :]  # (batch, hidden_dim)

        # Multi-window output
        outputs = self.multiwindow_head(h_pooled)

        # Logits softcap
        softcap = self.config.logits_softcap
        for key in outputs:
            outputs[key] = softcap * torch.tanh(outputs[key] / softcap)

        return outputs

    def decode_actions(
        self,
        outputs: Dict[str, torch.Tensor],
        *,
        reference_close: torch.Tensor,
        chronos_high: torch.Tensor,
        chronos_low: torch.Tensor,
        asset_class: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Decode model outputs to trading actions.

        Args:
            outputs: Dict from forward() with logits
            reference_close: (batch,) - current close price
            chronos_high: (batch,) - forecast high
            chronos_low: (batch,) - forecast low
            asset_class: (batch,) - 0 for stocks, 1 for crypto

        Returns:
            Dict with decoded actions for each window
        """
        config = self.config
        device = reference_close.device
        batch_size = reference_close.size(0)

        # Expand reference prices for windows: (batch,) -> (batch, num_windows)
        ref_close = reference_close.unsqueeze(-1).expand(-1, config.num_windows)
        ch_high = chronos_high.unsqueeze(-1).expand(-1, config.num_windows)
        ch_low = chronos_low.unsqueeze(-1).expand(-1, config.num_windows)

        # Decode buy price quantiles: (batch, num_windows, num_quantiles)
        # V4.2: Remove Chronos hard-clamp - allows learning when forecasts are trivial
        buy_offset = config.price_offset_pct * torch.sigmoid(outputs["buy_quantile_logits"])
        buy_quantiles = ref_close.unsqueeze(-1) * (1.0 - buy_offset)
        # Soft clamp: only ensure buy <= reference (can't buy above current price)
        buy_quantiles = torch.minimum(buy_quantiles, ref_close.unsqueeze(-1))

        # Decode sell price quantiles
        # V4.2: Use Chronos as soft target when available, else use offset from reference
        sell_offset = config.price_offset_pct * torch.sigmoid(outputs["sell_quantile_logits"])
        # Blend between reference+offset and Chronos high based on forecast quality
        # ch_high has shape (batch, num_windows), chronos_range will too
        chronos_range = ch_high - ref_close  # How much upside Chronos predicts
        # has_forecast: (batch, num_windows) -> need (batch, num_windows, 1) for broadcasting
        has_forecast = (chronos_range > ref_close * 0.001).float().unsqueeze(-1)

        # When Chronos has good forecast, use it; otherwise use reference+offset
        sell_from_ref = ref_close.unsqueeze(-1) * (1.0 + sell_offset)
        sell_from_chronos = ch_high.unsqueeze(-1) * (0.9 + 0.2 * torch.sigmoid(outputs["sell_quantile_logits"]))
        sell_quantiles = has_forecast * sell_from_chronos + (1 - has_forecast) * sell_from_ref
        # Ensure sell > reference
        sell_quantiles = torch.maximum(sell_quantiles, ref_close.unsqueeze(-1) * 1.001)

        # Ensure minimum gap (use median quantiles for gap check)
        median_idx = config.num_quantiles // 2
        buy_median = buy_quantiles[..., median_idx]
        sell_median = sell_quantiles[..., median_idx]
        min_gap = ref_close * config.min_price_gap_pct
        sell_quantiles = torch.maximum(
            sell_quantiles,
            buy_quantiles + min_gap.unsqueeze(-1)
        )

        # Confidence: (batch, num_windows)
        confidence = torch.sigmoid(outputs["confidence_logits"]).squeeze(-1)

        # Position size: learned from confidence and uncertainty
        base_position = torch.sigmoid(outputs["position_size_logits"]).squeeze(-1)

        # Scale position by confidence if enabled
        if config.position_from_confidence:
            position_scale = 0.5 + 0.5 * confidence  # 0.5 to 1.0 based on confidence
            base_position = base_position * position_scale

        # Scale position by inverse uncertainty if enabled
        if config.position_from_uncertainty:
            # Uncertainty = spread between quantiles (higher spread = less certain)
            buy_spread = buy_quantiles[..., -1] - buy_quantiles[..., 0]  # q75 - q25
            sell_spread = sell_quantiles[..., -1] - sell_quantiles[..., 0]
            avg_spread = (buy_spread + sell_spread) / 2
            # Use absolute value to handle inverted quantiles (training artifact)
            rel_spread = torch.abs(avg_spread) / ref_close  # Relative uncertainty

            # Inverse uncertainty scaling (more uncertain = smaller position)
            uncertainty_scale = 1.0 / (1.0 + 10 * rel_spread)
            base_position = base_position * uncertainty_scale

        # Apply leverage limits based on asset class
        if asset_class is not None:
            leverage_limit = torch.where(
                asset_class.unsqueeze(-1) > 0.5,
                torch.tensor(config.crypto_max_leverage, device=device),
                torch.tensor(config.equity_max_leverage, device=device),
            )
            leverage_limit = leverage_limit.expand(-1, config.num_windows)
        else:
            leverage_limit = torch.full(
                (batch_size, config.num_windows),
                config.equity_max_leverage,
                device=device
            )

        # Final position size with min/max constraints
        position_size = base_position * leverage_limit
        position_size = torch.clamp(position_size, min=config.min_position, max=config.max_position)

        # Exit day within each window: 1 to window_size
        exit_day_raw = torch.sigmoid(outputs["exit_day_logits"]).squeeze(-1)
        exit_day_in_window = 1.0 + exit_day_raw * (config.window_size - 1)

        # Absolute exit day considering window offset
        window_offsets = torch.arange(config.num_windows, device=device).float() * config.window_size
        exit_days = exit_day_in_window + window_offsets.unsqueeze(0)

        return {
            "buy_quantiles": buy_quantiles,         # (batch, num_windows, num_quantiles)
            "sell_quantiles": sell_quantiles,       # (batch, num_windows, num_quantiles)
            "position_size": position_size,         # (batch, num_windows)
            "confidence": confidence,               # (batch, num_windows)
            "exit_days": exit_days,                 # (batch, num_windows)
            "exit_day_in_window": exit_day_in_window,  # (batch, num_windows)
        }


def create_group_mask(group_ids: torch.Tensor) -> torch.Tensor:
    """Create cross-symbol attention mask from group IDs."""
    return group_ids.unsqueeze(0) == group_ids.unsqueeze(1)
