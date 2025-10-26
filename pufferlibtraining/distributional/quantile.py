from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantileValueHead(nn.Module):
    """
    Simple linear head that predicts a set of return quantiles instead of a single value.
    """

    def __init__(self, in_dim: int, n_quantiles: int = 32):
        super().__init__()
        self.n_quantiles = n_quantiles
        self.proj = nn.Linear(in_dim, n_quantiles, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def quantile_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    taus: torch.Tensor,
    kappa: float = 1.0,
) -> torch.Tensor:
    """
    Quantile regression loss with Huber smoothing (as in QR-DQN / Implicit Quantile Networks).

    Args:
        pred:   [N, Q] predicted quantiles.
        target: [N, 1] target returns broadcastable against ``pred``.
        taus:   [Q] quantile locations in (0, 1).
        kappa:  Huber threshold.
    """
    if pred.dim() != 2:
        raise ValueError("pred must have shape [N, Q]")
    if target.dim() != 2 or target.size(1) != 1:
        raise ValueError("target must have shape [N, 1]")
    if taus.dim() != 1 or taus.size(0) != pred.size(1):
        raise ValueError("taus must have length equal to number of quantiles")

    delta = target - pred
    abs_delta = delta.abs()
    huber = torch.where(abs_delta <= kappa, 0.5 * delta.pow(2), kappa * (abs_delta - 0.5 * kappa))
    tau = taus.view(1, -1)
    weight = torch.abs((delta.detach() < 0).float() - tau)
    loss = (weight * huber).mean()
    return loss
