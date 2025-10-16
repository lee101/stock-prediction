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
    per_asset_costs: torch.Tensor | None = None,
    return_per_asset: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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

    per_asset_returns = constrained_allocations * future_returns

    if per_asset_costs is not None:
        per_asset_costs = per_asset_costs.to(
            dtype=future_returns.dtype,
            device=future_returns.device,
        )
        if per_asset_costs.dim() == 1:
            per_asset_costs = per_asset_costs.unsqueeze(0)
        elif per_asset_costs.dim() == 0:
            per_asset_costs = per_asset_costs.view(1, 1)
        per_asset_costs = per_asset_costs.expand_as(constrained_allocations)
    else:
        per_asset_costs = torch.full_like(constrained_allocations, float(transaction_cost))

    trading_penalty = torch.abs(constrained_allocations) * per_asset_costs
    per_asset_net = per_asset_returns - trading_penalty

    realised_gross = constrained_allocations.abs().sum(dim=-1)
    excess_leverage = torch.clamp(realised_gross - 1.0, min=0.0)
    financing_penalty = excess_leverage * (borrowing_cost / trading_days)
    if return_per_asset:
        exposure_share = constrained_allocations.abs()
        exposure_denominator = exposure_share.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        exposure_share = exposure_share / exposure_denominator
        per_asset_net = per_asset_net - exposure_share * financing_penalty.unsqueeze(-1)
        pnl = per_asset_net.sum(dim=-1)
        return pnl, per_asset_net

    pnl = per_asset_net.sum(dim=-1) - financing_penalty

    return pnl


def sharpe_like_ratio(pnl: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute a differentiable Sharpe-like ratio (mean/std) for stability penalties.
    """
    mean = pnl.mean()
    std = pnl.std(unbiased=False)
    return mean / (std + eps)
