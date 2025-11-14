from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


def aggregate_daily_pnl(
    trade_weights: torch.Tensor,
    trade_returns: torch.Tensor,
    day_index: torch.Tensor,
) -> torch.Tensor:
    """
    Sum weighted trade returns into a daily series.
    """

    if not (trade_weights.shape == trade_returns.shape == day_index.shape):
        raise ValueError("weights, returns, and day_index must share the same shape")

    num_days = int(torch.max(day_index).item()) + 1
    daily = torch.zeros(num_days, device=trade_weights.device, dtype=trade_weights.dtype)
    daily.index_add_(0, day_index, trade_weights * trade_returns)
    return daily


def annualised_sortino(daily_pnl: torch.Tensor, trading_days: int = 252) -> torch.Tensor:
    mean_ret = daily_pnl.mean()
    downside = torch.clamp(-daily_pnl, min=0.0)
    downside_std = torch.sqrt(torch.mean(downside**2) + 1e-12)
    base = mean_ret / (downside_std + 1e-12)
    return base * math.sqrt(float(trading_days))


def annualised_return(daily_pnl: torch.Tensor, trading_days: int = 252) -> torch.Tensor:
    return daily_pnl.mean() * trading_days


@dataclass
class ObjectiveResult:
    score: torch.Tensor
    sortino: torch.Tensor
    annual_return: torch.Tensor
    downside_std: torch.Tensor


def combine_sortino_and_return(
    daily_pnl: torch.Tensor,
    *,
    trading_days: int = 252,
    return_weight: float = 0.05,
) -> ObjectiveResult:
    sortino = annualised_sortino(daily_pnl, trading_days)
    downside = torch.sqrt(torch.mean(torch.clamp(-daily_pnl, min=0.0) ** 2) + 1e-12)
    ann_return = annualised_return(daily_pnl, trading_days)
    score = sortino + return_weight * ann_return
    return ObjectiveResult(score=score, sortino=sortino, annual_return=ann_return, downside_std=downside)


def l2_penalty(weights: torch.Tensor, strength: float) -> torch.Tensor:
    if strength <= 0:
        return torch.zeros((), device=weights.device, dtype=weights.dtype)
    return strength * torch.mean(weights**2)
