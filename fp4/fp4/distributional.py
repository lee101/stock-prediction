"""Distributional value head utilities for QR-PPO.

Provides a ``QuantileValueHead`` module plus a quantile Huber loss and a
CVaR-from-quantiles helper.  All ops are differentiable and stay on device
(no host syncs in the hot path).

References: Dabney et al., "Distributional Reinforcement Learning with
Quantile Regression" (QR-DQN).  Here we apply the same idea to the PPO
value function: the critic outputs ``K`` quantile estimates of the return,
and the loss is the quantile Huber loss against scalar sample returns.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


class QuantileValueHead(nn.Module):
    """Linear head producing ``[B, num_quantiles]`` quantile predictions.

    Kept deliberately small and in high precision — per the NeMo recipe,
    value heads stay BF16/FP32 even when hidden layers are NVFP4.
    """

    def __init__(self, in_dim: int, num_quantiles: int = 51):
        super().__init__()
        if num_quantiles < 1:
            raise ValueError(f"num_quantiles must be >= 1, got {num_quantiles}")
        self.in_dim = int(in_dim)
        self.num_quantiles = int(num_quantiles)
        self.fc = nn.Linear(self.in_dim, self.num_quantiles)
        nn.init.orthogonal_(self.fc.weight, gain=1.0)
        nn.init.zeros_(self.fc.bias)
        # Midpoint quantile fractions tau_k = (k + 0.5) / K in (0, 1).
        taus = (torch.arange(self.num_quantiles, dtype=torch.float32) + 0.5) / float(self.num_quantiles)
        self.register_buffer("taus", taus, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    def mean_value(self, quantiles: torch.Tensor) -> torch.Tensor:
        """Scalar value estimate = mean over quantiles (shape [B])."""
        return quantiles.mean(dim=-1)


def quantile_huber_loss(
    pred_quantiles: torch.Tensor,
    target_returns: torch.Tensor,
    kappa: float = 1.0,
) -> torch.Tensor:
    """Standard quantile Huber loss for a distributional value head.

    Args:
        pred_quantiles: ``[B, K]`` predicted quantiles.
        target_returns: ``[B]`` scalar sample returns (e.g. GAE returns).
        kappa: Huber threshold.

    Returns:
        Scalar loss (mean over batch and quantiles).
    """
    if pred_quantiles.dim() != 2:
        raise ValueError(f"pred_quantiles must be [B, K], got {tuple(pred_quantiles.shape)}")
    if target_returns.dim() != 1:
        raise ValueError(f"target_returns must be [B], got {tuple(target_returns.shape)}")
    B, K = pred_quantiles.shape
    if target_returns.shape[0] != B:
        raise ValueError(f"batch mismatch: preds {B} vs targets {target_returns.shape[0]}")

    # tau_k = (k + 0.5) / K, shape [1, K].
    device = pred_quantiles.device
    dtype = pred_quantiles.dtype
    taus = (torch.arange(K, device=device, dtype=dtype) + 0.5) / float(K)
    taus = taus.unsqueeze(0)  # [1, K]

    target = target_returns.detach().to(dtype).unsqueeze(-1)  # [B, 1]
    diff = target - pred_quantiles  # [B, K], u = target - pred
    abs_diff = diff.abs()
    huber = torch.where(
        abs_diff <= kappa,
        0.5 * diff * diff,
        kappa * (abs_diff - 0.5 * kappa),
    )
    # Quantile weighting: |tau - I(u < 0)| * huber / kappa.
    indicator = (diff < 0).to(dtype)
    weight = (taus - indicator).abs()
    loss = weight * huber / kappa
    # Mean over quantiles, then over batch (equivalent to .mean()).
    return loss.mean()


def cvar_from_quantiles(quantiles: torch.Tensor, alpha: float = 0.05) -> torch.Tensor:
    """Mean of the lowest ``ceil(K*alpha)`` quantiles.

    Args:
        quantiles: ``[B, K]`` or ``[K]`` quantile predictions, assumed to be
            stored at midpoint fractions tau_k = (k+0.5)/K.  We sort along
            the last dim to be robust to any reordering.
        alpha: CVaR level in (0, 1].

    Returns:
        ``[B]`` (or scalar for 1-D input) CVaR estimate.  Fully differentiable.
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    squeeze = False
    if quantiles.dim() == 1:
        quantiles = quantiles.unsqueeze(0)
        squeeze = True
    K = quantiles.shape[-1]
    n = max(1, int(math.ceil(K * float(alpha))))
    # Sort ascending along quantile dim, take the lowest n.
    sorted_q, _ = torch.sort(quantiles, dim=-1)
    tail = sorted_q[..., :n]
    out = tail.mean(dim=-1)
    if squeeze:
        out = out.squeeze(0)
    return out
