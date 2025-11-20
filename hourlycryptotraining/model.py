from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import math
import torch
from torch import nn


@dataclass
class PolicyHeadConfig:
    input_dim: int
    hidden_dim: int = 256
    depth: int = 3
    dropout: float = 0.1
    price_offset_pct: float = 0.0003
    max_trade_qty: float = 3.0
    min_price_gap_pct: float = 0.0003
    num_heads: int = 8
    num_layers: int = 4
    max_len: int = 2048  # Positional encoding max sequence length


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0, max_len: int = 2048) -> None:
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


class HourlyCryptoPolicy(nn.Module):
    """Transformer encoder that outputs limit prices and trade intensity."""

    def __init__(self, config: PolicyHeadConfig) -> None:
        super().__init__()
        self.price_offset_pct = config.price_offset_pct
        self.max_trade_qty = config.max_trade_qty
        self.min_gap_pct = config.min_price_gap_pct
        self.embed = nn.Linear(config.input_dim, config.hidden_dim)
        self.pos_encoding = PositionalEncoding(config.hidden_dim, dropout=config.dropout, max_len=config.max_len)
        # Use sdpa_kernel to enable flash attention for long sequences
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_heads,
                dim_feedforward=config.hidden_dim * 4,
                dropout=config.dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.norm = nn.LayerNorm(config.hidden_dim)
        # Output heads: buy price, sell price, buy intensity, sell intensity
        self.head = nn.Linear(config.hidden_dim, 4)

    @staticmethod
    def upgrade_legacy_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Expand legacy 3-channel heads (buy/sell/amount) to 4-channel buy/sell/buy_amt/sell_amt."""
        weight_key = "head.weight"
        bias_key = "head.bias"
        if weight_key in state_dict and state_dict[weight_key].shape[0] == 3:
            old_w = state_dict[weight_key]
            new_w = torch.zeros((4, old_w.shape[1]), dtype=old_w.dtype)
            new_w[:3] = old_w
            # Use old trade_amount weights for both buy/sell intensity to preserve prior behaviour
            new_w[3] = old_w[2]
            state_dict[weight_key] = new_w
        if bias_key in state_dict and state_dict[bias_key].shape[0] == 3:
            old_b = state_dict[bias_key]
            new_b = torch.zeros((4,), dtype=old_b.dtype)
            new_b[:3] = old_b
            new_b[3] = old_b[2]
            state_dict[bias_key] = new_b
        return state_dict

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.embed(features)
        h = self.pos_encoding(h)
        h = self.encoder(h)
        h = self.norm(h)
        logits = self.head(h)
        return {
            "buy_price_logits": logits[..., 0:1],
            "sell_price_logits": logits[..., 1:2],
            "buy_amount_logits": logits[..., 2:3],
            "sell_amount_logits": logits[..., 3:4],
        }

    def decode_actions(
        self,
        outputs: Dict[str, torch.Tensor],
        *,
        reference_close: torch.Tensor,
        chronos_high: torch.Tensor,
        chronos_low: torch.Tensor,
        dynamic_offset_pct: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
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
        buy_amount = torch.sigmoid(outputs["buy_amount_logits"]).squeeze(-1)
        sell_amount = torch.sigmoid(outputs["sell_amount_logits"]).squeeze(-1)
        # Backward compatibility: trade_amount retains the larger side for simulators
        trade_amount = torch.maximum(buy_amount, sell_amount)
        return {
            "buy_price": buy_price,
            "sell_price": sell_price,
            "trade_amount": trade_amount,
            "buy_amount": buy_amount,
            "sell_amount": sell_amount,
        }
