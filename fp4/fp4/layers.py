"""Composite NVFP4 layers."""
from __future__ import annotations

import torch
from torch import nn

from .linear import NVFP4Linear


class NVFP4MLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int, bias: bool = True):
        super().__init__()
        self.fc1 = NVFP4Linear(d_in, d_hidden, bias=bias, seed=1)
        self.fc2 = NVFP4Linear(d_hidden, d_out, bias=bias, seed=2)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))
