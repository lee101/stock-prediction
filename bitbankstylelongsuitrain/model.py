"""Bitbank-style HourlyCryptoPolicy model."""
from __future__ import annotations

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class OutputHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, activation: str = "tanh"):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim // 2)
        self.fc2 = nn.Linear(input_dim // 2, output_dim)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.gelu(self.fc1(x))
        x = self.fc2(x)
        if self.activation == "tanh":
            x = torch.tanh(x)
        elif self.activation == "sigmoid":
            x = torch.sigmoid(x)
        return x


class BitbankHourlyCryptoPolicy(nn.Module):
    """Btcmarketsbot-style transformer policy."""

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        seq_len: int = 72,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=seq_len * 2, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(hidden_dim)

        self.buy_head = OutputHead(hidden_dim, 1, activation="tanh")
        self.sell_head = OutputHead(hidden_dim, 1, activation="tanh")
        self.amount_head = OutputHead(hidden_dim, 1, activation="sigmoid")
        self.confidence_head = OutputHead(hidden_dim, 1, activation="sigmoid")

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.ln(x)

        last = x[:, -1, :]

        buy_logits = self.buy_head(last)
        sell_logits = self.sell_head(last)
        amount = self.amount_head(last)
        confidence = self.confidence_head(last)

        return {
            "buy_logits": buy_logits.squeeze(-1),
            "sell_logits": sell_logits.squeeze(-1),
            "buy_amount": amount.squeeze(-1),
            "sell_amount": amount.squeeze(-1),
            "confidence": confidence.squeeze(-1),
        }

    def decode_actions(
        self,
        outputs: dict,
        reference_close: torch.Tensor,
        chronos_low: torch.Tensor = None,
        chronos_high: torch.Tensor = None,
        aggressiveness: float = 0.8,
        max_spread_pct: float = 0.03,
    ) -> dict:
        buy_logits = outputs["buy_logits"]
        sell_logits = outputs["sell_logits"]

        # Ensure 1D tensors (batch,) - take last timestep if 2D
        ref = reference_close[:, -1] if reference_close.dim() > 1 else reference_close
        c_low_in = chronos_low[:, -1] if chronos_low is not None and chronos_low.dim() > 1 else chronos_low
        c_high_in = chronos_high[:, -1] if chronos_high is not None and chronos_high.dim() > 1 else chronos_high

        min_price = ref * 0.01

        if c_low_in is not None and c_high_in is not None:
            c_low = torch.clamp(c_low_in, min=min_price)
            c_high = torch.clamp(c_high_in, min=c_low * 1.001)

            buy_range = ref - c_low
            buy_blend = (1 + buy_logits) * 0.5
            buy_blend = buy_blend * (1 - aggressiveness) + aggressiveness
            buy_price = c_low + buy_range * buy_blend

            sell_range = c_high - ref
            sell_blend = (1 + sell_logits) * 0.5
            sell_blend = sell_blend * (1 - aggressiveness) + aggressiveness
            sell_price = c_high - sell_range * sell_blend
        else:
            buy_mult = 1.0 + buy_logits * max_spread_pct
            buy_price = ref * buy_mult
            sell_mult = 1.0 + sell_logits * max_spread_pct
            sell_price = ref * sell_mult

        buy_price = torch.clamp(buy_price, min=min_price)
        sell_price = torch.clamp(sell_price, min=min_price)

        min_spread = ref * 0.002
        sell_price = torch.max(sell_price, buy_price + min_spread)

        return {
            "buy_price": buy_price,
            "sell_price": sell_price,
            "buy_amount": outputs["buy_amount"],
            "sell_amount": outputs["sell_amount"],
            "trade_amount": outputs["buy_amount"],
            "confidence": outputs["confidence"],
        }
