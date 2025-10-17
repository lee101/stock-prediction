"""Robust loss helpers tuned for financial forecasting."""

from __future__ import annotations

import torch


def huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    delta: float = 0.01,
    reduction: str = "mean",
) -> torch.Tensor:
    """Smooth L1 (Huber) loss with configurable transition point."""
    if delta <= 0:
        raise ValueError("delta must be positive.")

    err = pred - target
    abs_err = err.abs()
    delta_tensor = abs_err.new_tensor(delta)
    quadratic = torch.minimum(abs_err, delta_tensor)
    linear = abs_err - quadratic
    loss = 0.5 * quadratic.square() + delta_tensor * linear
    return _reduce(loss, reduction)


def heteroscedastic_gaussian_nll(
    mean: torch.Tensor,
    log_sigma: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
    min_sigma: float = 1e-5,
) -> torch.Tensor:
    """Negative log-likelihood for Gaussian with learned variance."""
    sigma = torch.exp(log_sigma).clamp_min(min_sigma)
    safe_log_sigma = torch.log(sigma)
    nll = 0.5 * ((target - mean) ** 2 / (sigma**2) + 2 * safe_log_sigma)
    return _reduce(nll, reduction)


def pinball_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    quantile: float,
    reduction: str = "mean",
) -> torch.Tensor:
    """Quantile (pinball) loss."""
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be in (0, 1)")
    diff = target - pred
    loss = torch.maximum(quantile * diff, (quantile - 1) * diff)
    return _reduce(loss, reduction)


def _reduce(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"Unsupported reduction '{reduction}'.")


__all__ = ["huber_loss", "heteroscedastic_gaussian_nll", "pinball_loss"]
