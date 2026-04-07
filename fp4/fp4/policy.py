"""Tiny actor-critic MLP for the fp4 PPO trainer.

Per the NeMo NVFP4 recipe:
- Hidden layers use NVFP4Linear (4-bit emulated, RHT + block quant).
- Input projection (embedding equivalent) and the policy / value heads stay in
  BF16 (or FP32 on CPU) — these are the small, sensitive layers.

Continuous action space: a Gaussian with state-independent log-std (the same
parametrization the pufferlib BF16 baseline uses). Returns dist + value.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .linear import NVFP4Linear


def _high_precision_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.bfloat16
    return torch.float32


class ActorCritic(nn.Module):
    """Two-hidden-layer MLP. Hidden layers NVFP4, head + input proj BF16/FP32."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64, seed: int = 0):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.hidden = int(hidden)
        # Input projection — keep high precision (small, sensitive).
        self.in_proj = nn.Linear(self.obs_dim, self.hidden)
        # Hidden layer — NVFP4 (the whole point of fp4).
        self.h1 = NVFP4Linear(self.hidden, self.hidden, seed=seed)
        # Heads — high precision.
        self.pi_head = nn.Linear(self.hidden, self.act_dim)
        self.v_head = nn.Linear(self.hidden, 1)
        # State-independent log-std, init at -0.5 → std≈0.6.
        self.log_std = nn.Parameter(torch.full((self.act_dim,), -0.5))
        self._init_heads()

    def _init_heads(self) -> None:
        nn.init.orthogonal_(self.in_proj.weight, gain=math.sqrt(2.0))
        nn.init.zeros_(self.in_proj.bias)
        nn.init.orthogonal_(self.pi_head.weight, gain=0.01)
        nn.init.zeros_(self.pi_head.bias)
        nn.init.orthogonal_(self.v_head.weight, gain=1.0)
        nn.init.zeros_(self.v_head.bias)

    def features(self, obs: torch.Tensor) -> torch.Tensor:
        x = F.tanh(self.in_proj(obs))
        x = F.tanh(self.h1(x))
        return x

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (mean, std, value)."""
        z = self.features(obs)
        mean = self.pi_head(z)
        value = self.v_head(z).squeeze(-1)
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std, value

    @staticmethod
    def gaussian_logprob(mean: torch.Tensor, std: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        var = std * std
        logp = -0.5 * (((action - mean) ** 2) / (var + 1e-8) + 2 * torch.log(std + 1e-8) + math.log(2 * math.pi))
        return logp.sum(dim=-1)

    @staticmethod
    def gaussian_entropy(std: torch.Tensor) -> torch.Tensor:
        # H = 0.5 * log(2*pi*e) * D + sum(log std)
        return (0.5 * math.log(2 * math.pi * math.e) + torch.log(std + 1e-8)).sum(dim=-1)

    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std, value = self.forward(obs)
        eps = torch.randn_like(mean)
        action = mean + std * eps
        logp = self.gaussian_logprob(mean, std, action)
        return action, logp, value, mean
