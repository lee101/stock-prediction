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


class HourlyCryptoPolicy(nn.Module):
    """Lightweight per-hour MLP that outputs limit prices & position sizes."""

    def __init__(self, config: PolicyHeadConfig) -> None:
        super().__init__()
        if config.depth < 1:
            raise ValueError("depth must be >= 1")
        self.price_offset_pct = config.price_offset_pct
        self.max_trade_qty = config.max_trade_qty
        layers = []
        in_dim = config.input_dim
        for _ in range(config.depth):
            linear = nn.Linear(in_dim, config.hidden_dim)
            nn.init.kaiming_uniform_(linear.weight, a=math.sqrt(5))
            layers.append(linear)
            in_dim = config.hidden_dim
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None
        self.head = nn.Linear(in_dim, 4)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = features
        for layer in self.layers:
            h = torch.relu(layer(h))
            if self.dropout is not None:
                h = self.dropout(h)
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
    ) -> Dict[str, torch.Tensor]:
        price_scale = reference_close * self.price_offset_pct
        buy_raw = torch.tanh(outputs["buy_price_logits"])
        sell_raw = torch.tanh(outputs["sell_price_logits"])
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
        buy_amount = self.max_trade_qty * torch.sigmoid(outputs["buy_amount_logits"])
        sell_amount = self.max_trade_qty * torch.sigmoid(outputs["sell_amount_logits"])
        return {
            "buy_price": buy_price.squeeze(-1),
            "sell_price": sell_price.squeeze(-1),
            "buy_amount": buy_amount.squeeze(-1),
            "sell_amount": sell_amount.squeeze(-1),
        }
