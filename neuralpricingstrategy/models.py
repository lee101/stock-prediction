from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class PricingModelConfig:
    input_dim: int
    hidden_dim: int = 192
    depth: int = 3
    dropout: float = 0.1
    max_delta_pct: float = 0.08


class PricingAdjustmentModel(nn.Module):
    """Small MLP that predicts price deltas plus expected PnL gain."""

    def __init__(self, config: PricingModelConfig) -> None:
        super().__init__()
        if config.depth < 1:
            raise ValueError("depth must be >= 1")
        layers = []
        in_dim = config.input_dim
        for _ in range(config.depth):
            layers.append(nn.Linear(in_dim, config.hidden_dim))
            in_dim = config.hidden_dim
        self.hidden_layers = nn.ModuleList(layers)
        self.out = nn.Linear(in_dim, 3)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None
        self.max_delta_pct = float(config.max_delta_pct)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = features
        for layer in self.hidden_layers:
            hidden = layer(hidden)
            hidden = F.relu(hidden)
            if self.dropout is not None:
                hidden = self.dropout(hidden)
        raw = self.out(hidden)
        low_delta = torch.tanh(raw[:, 0]) * self.max_delta_pct
        high_delta = torch.tanh(raw[:, 1]) * self.max_delta_pct
        pnl_gain = raw[:, 2]
        return torch.stack([low_delta, high_delta, pnl_gain], dim=-1)


__all__ = ["PricingAdjustmentModel", "PricingModelConfig"]
