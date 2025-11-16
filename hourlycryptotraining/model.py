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
    price_offset_pct: float = 0.03
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
        self.head = nn.Linear(config.hidden_dim, 3)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.embed(features)
        h = self.pos_encoding(h)
        h = self.encoder(h)
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
    ) -> Dict[str, torch.Tensor]:
        price_scale = reference_close * self.price_offset_pct
        buy_raw = torch.tanh(outputs["buy_price_logits"]).squeeze(-1)
        sell_raw = torch.tanh(outputs["sell_price_logits"]).squeeze(-1)
        buy_price = torch.clamp(
            chronos_low + price_scale * buy_raw,
            min=reference_close * (1.0 - 2 * self.price_offset_pct),
            max=reference_close,
        )
        sell_price = torch.clamp(
            chronos_high + price_scale * sell_raw,
            min=reference_close,
            max=reference_close * (1.0 + 2 * self.price_offset_pct),
        )
        gap = reference_close * self.min_gap_pct
        sell_price = torch.maximum(sell_price, buy_price + gap)
        trade_amount = torch.sigmoid(outputs["trade_amount_logits"]).squeeze(-1)
        return {
            "buy_price": buy_price,
            "sell_price": sell_price,
            "trade_amount": trade_amount,
        }
