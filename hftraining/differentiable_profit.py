#!/usr/bin/env python3
"""
Differentiable portfolio PnL utilities.
"""

from __future__ import annotations

import torch


def compute_portfolio_pnl(
    allocations: torch.Tensor,
    future_returns: torch.Tensor,
    transaction_cost: float = 0.0001,
    leverage_limit: float = 2.0,
    borrowing_cost: float = 0.0675,
    trading_days: int = 252,
) -> torch.Tensor:
    """
    Compute per-sample profit and loss using differentiable operations.

    Args:
        allocations: Target portfolio weights in [-1, 1]. Shape [batch] or [batch, assets].
        future_returns: Realised returns for the matching assets. Same shape as allocations.
        transaction_cost: Linear transaction cost applied to absolute allocation size.
        leverage_limit: Maximum absolute exposure allowed across assets.
        borrowing_cost: Annualised borrowing rate applied to leverage above 1×.
        trading_days: Number of trading days per year used to annualise borrowing cost.

    Returns:
        Tensor with per-sample pnl values.
    """
    if allocations.dim() == 1:
        allocations = allocations.unsqueeze(0)
    if future_returns.dim() == 1:
        future_returns = future_returns.unsqueeze(0)

    allocations = allocations.to(future_returns.dtype)

    gross = allocations.abs().sum(dim=-1, keepdim=True)
    scale = torch.clamp(gross / leverage_limit, min=1.0)
    constrained_allocations = allocations / scale

    pnl = (constrained_allocations * future_returns).sum(dim=-1)
    trading_penalty = torch.abs(constrained_allocations) * transaction_cost
    pnl = pnl - trading_penalty.sum(dim=-1)

    realised_gross = constrained_allocations.abs().sum(dim=-1)
    excess_leverage = torch.clamp(realised_gross - 1.0, min=0.0)
    financing_penalty = excess_leverage * (borrowing_cost / trading_days)
    pnl = pnl - financing_penalty

    return pnl


def sharpe_like_ratio(pnl: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute a differentiable Sharpe-like ratio (mean/std) for stability penalties.
    """
    mean = pnl.mean()
    std = pnl.std(unbiased=False)
    return mean / (std + eps)
