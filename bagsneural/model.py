"""Neural model for Bags.fm trade signals and sizing."""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
from torch import nn


class BagsNeuralModel(nn.Module):
    """Simple MLP with dual heads for signal and size."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        dims = [input_dim, *hidden_dims]
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        self.backbone = nn.Sequential(*layers)
        last_dim = dims[-1]
        self.signal_head = nn.Linear(last_dim, 1)
        self.size_head = nn.Linear(last_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        signal_logit = self.signal_head(features).squeeze(-1)
        size_logit = self.size_head(features).squeeze(-1)
        return signal_logit, size_logit
