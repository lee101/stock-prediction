from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict
import math

import torch
from torch import nn
from einops import rearrange

from hourlycryptotraining.model import HourlyCryptoPolicy, PolicyHeadConfig


# =============================================================================
# Cross-Symbol Attention Layers (inspired by Chronos2)
# =============================================================================

class GroupSelfAttention(nn.Module):
    """Self-attention applied along the batch axis to learn cross-symbol patterns.

    This allows correlated symbols (e.g., tech stocks, crypto) to share information.
    Adapted from Chronos2's GroupSelfAttention.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        group_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            group_mask: (batch, batch) - which items can attend to each other
                        If None, all items attend to all others

        Returns:
            (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Swap batch and time axes: (batch, time, d) -> (time, batch, d)
        hidden_states = rearrange(hidden_states, "b t d -> t b d")

        # Apply layer norm
        normed = self.layer_norm(hidden_states)

        # Project Q, K, V
        q = self.q_proj(normed)
        k = self.k_proj(normed)
        v = self.v_proj(normed)

        # Reshape for multi-head attention: (time, batch, d) -> (time, batch, heads, head_dim)
        q = q.view(seq_len, batch_size, self.num_heads, self.head_dim)
        k = k.view(seq_len, batch_size, self.num_heads, self.head_dim)
        v = v.view(seq_len, batch_size, self.num_heads, self.head_dim)

        # Transpose for attention: (time, batch, heads, head_dim) -> (time, heads, batch, head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Compute attention scores: (time, heads, batch, batch)
        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        # Apply group mask if provided
        if group_mask is not None:
            # Expand mask: (batch, batch) -> (1, 1, batch, batch)
            base = group_mask | torch.eye(group_mask.size(0), device=group_mask.device, dtype=group_mask.dtype)
            mask = base.unsqueeze(0).unsqueeze(0)
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Apply attention to values
        attn_output = torch.matmul(attn_probs, v)  # (time, heads, batch, head_dim)

        # Reshape back: (time, heads, batch, head_dim) -> (time, batch, d)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(seq_len, batch_size, self.hidden_dim)

        # Output projection
        attn_output = self.out_proj(attn_output)

        # Residual connection
        hidden_states = hidden_states + self.dropout(attn_output)

        # Swap back: (time, batch, d) -> (batch, time, d)
        hidden_states = rearrange(hidden_states, "t b d -> b t d")

        return hidden_states


class MultiSymbolEncoderBlock(nn.Module):
    """Encoder block with both time self-attention and group cross-attention."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_cross_attention: bool = True,
    ):
        super().__init__()
        self.use_cross_attention = use_cross_attention

        # Time self-attention (within each symbol's sequence)
        self.time_attn = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        # Group self-attention (across symbols in batch)
        if use_cross_attention:
            self.group_attn = GroupSelfAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        group_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Time self-attention
        hidden_states = self.time_attn(hidden_states)

        # Group self-attention (cross-symbol patterns)
        if self.use_cross_attention:
            hidden_states = self.group_attn(hidden_states, group_mask)

        return hidden_states


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, dim: int, dropout: float = 0.0, max_len: int = 4096):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:seq_len].unsqueeze(0)
        return self.dropout(x)


@dataclass
class DailyPolicyConfig(PolicyHeadConfig):
    """Policy head configuration with daily-friendly defaults."""

    price_offset_pct: float = 0.025
    min_price_gap_pct: float = 0.0005
    max_len: int = 4096
    equity_max_leverage: float = 2.0
    crypto_max_leverage: float = 1.0
    use_cross_attention: bool = True  # Enable cross-symbol learning
    symbol_dropout_rate: float = 0.1  # Dropout rate for symbol masking


class DailyMultiAssetPolicy(HourlyCryptoPolicy):
    """Thin wrapper around HourlyCryptoPolicy for clarity."""

    def __init__(self, config: DailyPolicyConfig) -> None:
        super().__init__(config)
        self.equity_max_leverage = float(config.equity_max_leverage)
        self.crypto_max_leverage = float(config.crypto_max_leverage)

    def decode_actions(
        self,
        outputs,
        *,
        reference_close,
        chronos_high,
        chronos_low,
        asset_class: Optional[torch.Tensor] = None,
    ):
        decoded = super().decode_actions(
            outputs,
            reference_close=reference_close,
            chronos_high=chronos_high,
            chronos_low=chronos_low,
        )
        if asset_class is not None:
            mask = asset_class.to(decoded["trade_amount"].device, dtype=decoded["trade_amount"].dtype)
            mask = mask.view(-1)
            limits = torch.where(
                mask > 0.5,
                torch.as_tensor(self.crypto_max_leverage, device=decoded["trade_amount"].device, dtype=decoded["trade_amount"].dtype),
                torch.as_tensor(self.equity_max_leverage, device=decoded["trade_amount"].device, dtype=decoded["trade_amount"].dtype),
            )
            # Scale sigmoid output (0-1) by asset-specific max leverage
            # Stocks: 0-2x, Crypto: 0-1x
            decoded["trade_amount"] = decoded["trade_amount"] * limits.unsqueeze(-1)
        return decoded


class MultiSymbolDailyPolicy(nn.Module):
    """Enhanced policy with cross-symbol attention for multivariate learning.

    This model can learn patterns like "when tech rallies, buy NVDA"
    by attending across symbols in the same batch.
    """

    def __init__(self, config: DailyPolicyConfig) -> None:
        super().__init__()
        self.config = config
        self.price_offset_pct = config.price_offset_pct
        self.max_trade_qty = config.max_trade_qty
        self.min_gap_pct = config.min_price_gap_pct
        self.equity_max_leverage = float(config.equity_max_leverage)
        self.crypto_max_leverage = float(config.crypto_max_leverage)

        # Input embedding
        self.embed = nn.Linear(config.input_dim, config.hidden_dim)
        self.pos_encoding = PositionalEncoding(
            config.hidden_dim,
            dropout=config.dropout,
            max_len=config.max_len
        )

        # Encoder blocks with optional cross-attention
        self.blocks = nn.ModuleList([
            MultiSymbolEncoderBlock(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
                use_cross_attention=config.use_cross_attention,
            )
            for _ in range(config.num_layers)
        ])

        self.norm = nn.LayerNorm(config.hidden_dim)
        self.head = nn.Linear(config.hidden_dim, 3)

    def forward(
        self,
        features: torch.Tensor,
        group_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: (batch, seq_len, input_dim)
            group_mask: (batch, batch) - which symbols can attend to each other

        Returns:
            Dict with buy_price_logits, sell_price_logits, trade_amount_logits
        """
        h = self.embed(features)
        h = self.pos_encoding(h)

        # Apply encoder blocks
        for block in self.blocks:
            h = block(h, group_mask)

        h = self.norm(h)
        logits = self.head(h)

        return {
            "buy_price_logits": logits[..., 0:1],
            "sell_price_logits": logits[..., 1:2],
            "trade_amount_logits": logits[..., 2:3],
        }

    def decode_actions(
        self,
        outputs: Dict[str, torch.Tensor],
        *,
        reference_close: torch.Tensor,
        chronos_high: torch.Tensor,
        chronos_low: torch.Tensor,
        asset_class: Optional[torch.Tensor] = None,
        dynamic_offset_pct: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Decode model outputs into trading actions."""
        if dynamic_offset_pct is None:
            offset_pct = torch.full_like(reference_close, self.price_offset_pct)
        else:
            offset_pct = dynamic_offset_pct.to(reference_close.device, reference_close.dtype)
        offset_pct = torch.clamp(offset_pct, min=1e-6)

        price_scale = reference_close * offset_pct
        buy_raw = torch.tanh(outputs["buy_price_logits"]).squeeze(-1)
        sell_raw = torch.tanh(outputs["sell_price_logits"]).squeeze(-1)

        min_buy = reference_close * (1.0 - 2 * offset_pct)
        max_sell = reference_close * (1.0 + 2 * offset_pct)

        buy_price = torch.maximum(min_buy, torch.minimum(chronos_low + price_scale * buy_raw, reference_close))
        sell_price = torch.minimum(max_sell, torch.maximum(chronos_high + price_scale * sell_raw, reference_close))

        gap = reference_close * self.min_gap_pct
        sell_price = torch.maximum(sell_price, buy_price + gap)

        # Base trade amount from sigmoid (0-1)
        trade_amount = torch.sigmoid(outputs["trade_amount_logits"]).squeeze(-1)

        # Scale by asset-specific max leverage
        if asset_class is not None:
            mask = asset_class.to(trade_amount.device, dtype=trade_amount.dtype)
            mask = mask.view(-1)
            limits = torch.where(
                mask > 0.5,
                torch.as_tensor(self.crypto_max_leverage, device=trade_amount.device, dtype=trade_amount.dtype),
                torch.as_tensor(self.equity_max_leverage, device=trade_amount.device, dtype=trade_amount.dtype),
            )
            # Scale sigmoid output (0-1) by asset-specific max leverage
            # Stocks: 0-2x, Crypto: 0-1x
            trade_amount = trade_amount * limits.unsqueeze(-1)

        return {
            "buy_price": buy_price,
            "sell_price": sell_price,
            "trade_amount": trade_amount,
        }


def create_group_mask(group_ids: torch.Tensor) -> torch.Tensor:
    """Create attention mask from group IDs.

    Args:
        group_ids: (batch,) - group ID for each item in batch

    Returns:
        (batch, batch) boolean mask - True where items can attend to each other
    """
    # Items with same group_id can attend to each other
    return group_ids.unsqueeze(0) == group_ids.unsqueeze(1)


__all__ = [
    "DailyPolicyConfig",
    "DailyMultiAssetPolicy",
    "MultiSymbolDailyPolicy",
    "GroupSelfAttention",
    "MultiSymbolEncoderBlock",
    "create_group_mask",
]
