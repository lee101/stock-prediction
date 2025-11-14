from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class PolicyConfig:
    input_dim: int
    hidden_dim: int = 128
    depth: int = 2
    dropout: float = 0.0
    allow_short: bool = False
    max_weight: float = 1.0


class PortfolioPolicy(nn.Module):
    """Simple MLP that outputs position weights per strategy-day row."""

    def __init__(self, config: PolicyConfig) -> None:
        super().__init__()
        if config.depth < 1:
            raise ValueError("depth must be >= 1")
        layers = []
        in_dim = config.input_dim
        for _ in range(config.depth):
            layers.append(nn.Linear(in_dim, config.hidden_dim))
            in_dim = config.hidden_dim
        self.hidden_layers = nn.ModuleList(layers)
        self.out = nn.Linear(in_dim, 1)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None
        self.allow_short = config.allow_short
        self.max_weight = config.max_weight

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        h = features
        for layer in self.hidden_layers:
            h = layer(h)
            h = F.relu(h)
            if self.dropout is not None:
                h = self.dropout(h)
        logits = self.out(h).squeeze(-1)
        if self.allow_short:
            weights = torch.tanh(logits) * self.max_weight
        else:
            weights = torch.sigmoid(logits) * self.max_weight
        return weights
