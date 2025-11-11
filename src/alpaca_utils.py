"""
Shared Alpaca-related utilities.

This module centralises leverage and financing rate helpers so that
all trading components apply consistent borrowing costs and leverage
clamps.  The defaults align with the production brokerage setup:

* 6.75% annual borrowing cost.
* 252 trading days per year.
* Baseline 1× gross exposure (unlevered).
* End-of-day leverage target capped at 2× with an intraday ceiling of 4×.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

ANNUAL_MARGIN_RATE: float = 0.065
TRADING_DAYS_PER_YEAR: int = 252
BASE_GROSS_EXPOSURE: float = 1.0
MAX_GROSS_EXPOSURE: float = 2.0
INTRADAY_GROSS_EXPOSURE: float = 4.0


def annual_to_daily_rate(annual_rate: float, *, trading_days: int = TRADING_DAYS_PER_YEAR) -> float:
    """Convert an annualised rate to an equivalent per-trading-day rate."""
    trading_days = max(1, int(trading_days))
    return float(annual_rate) / float(trading_days)


def leverage_penalty(
    gross_exposure: float,
    *,
    base_exposure: float = BASE_GROSS_EXPOSURE,
    daily_rate: float | None = None,
    annual_rate: float = ANNUAL_MARGIN_RATE,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Compute the daily financing penalty for excess leverage.

    Args:
        gross_exposure: The absolute gross exposure applied during the period.
        base_exposure: Exposure that does not accrue borrowing costs (typically 1×).
        daily_rate: Optional explicit daily borrowing rate. When None the value
            is derived from ``annual_rate`` and ``trading_days``.
        annual_rate: Annualised borrowing cost applied when ``daily_rate`` is None.
        trading_days: Trading days per year used when deriving the daily rate.

    Returns:
        The financing cost to subtract from returns for this period.
    """
    if daily_rate is None:
        daily_rate = annual_to_daily_rate(annual_rate, trading_days=trading_days)
    excess = max(0.0, float(gross_exposure) - float(base_exposure))
    return excess * float(daily_rate)


def clamp_end_of_day_weights(
    weights: np.ndarray,
    *,
    max_gross: float = MAX_GROSS_EXPOSURE,
) -> Tuple[np.ndarray, float]:
    """
    Clamp portfolio weights so that end-of-day gross exposure does not exceed ``max_gross``.

    Args:
        weights: Executed weights for the current step (1-D array).
        max_gross: Maximum gross exposure permitted after the close.

    Returns:
        Tuple of (clamped_weights, reduction_turnover) where ``reduction_turnover`` is
        the additional turnover implied by scaling the weights down.
    """
    max_gross = max(float(max_gross), 1.0)
    gross = float(np.sum(np.abs(weights)))
    if gross <= max_gross + 1e-9:
        return weights.astype(np.float32, copy=True), 0.0

    scale = max_gross / max(gross, 1e-8)
    clamped = weights * scale
    turnover = float(np.sum(np.abs(weights - clamped)))
    return clamped.astype(np.float32, copy=False), turnover


__all__ = [
    "ANNUAL_MARGIN_RATE",
    "TRADING_DAYS_PER_YEAR",
    "BASE_GROSS_EXPOSURE",
    "MAX_GROSS_EXPOSURE",
    "INTRADAY_GROSS_EXPOSURE",
    "annual_to_daily_rate",
    "leverage_penalty",
    "clamp_end_of_day_weights",
]
