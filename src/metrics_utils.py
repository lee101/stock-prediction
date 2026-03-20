"""Metrics utilities for trading performance analysis.

Sortino Ratio Definition Used Here:
====================================
  SR = (mean_return * periods_per_year) / (downside_dev * sqrt(periods_per_year))
     = (mean_return / downside_dev) * sqrt(periods_per_year)

  where downside_dev = sqrt(sum(returns[returns < 0] ** 2) / total_n)
                     = sqrt(sum_neg_sq / ret_count)
                     = "partial RMS": sum of negative squares divided by TOTAL n,
                       not divided by the number of negative returns only.

This formula MATCHES the C trading environment in
pufferlib_market/src/trading_env.c:
  // sum_neg_sq accumulated only for ret < 0; ret_count incremented every step
  downside_dev = sqrtf(sum_neg_sq / ret_count)   // divide by ALL steps, not neg-only
  sortino = mean_ret / downside_dev * sqrtf(ppy)  // annualised

Annualisation factors for crypto (trades 24/7, 365 days/year):
  - Hourly data : periods_per_year = 8760  (24 * 365)
  - Daily data  : periods_per_year = 365   (NOT 252 — crypto never closes)

DO NOT use ddof=1 here; that would give different numbers from the C env
and break comparisons between Python evaluation and C-based training.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def compute_step_returns(equity_curve: Iterable[float]) -> np.ndarray:
    """Convert an equity curve into step returns (percentage changes).
    
    Args:
        equity_curve: Time series of portfolio values
        
    Returns:
        Array of returns where return[i] = (equity[i+1] - equity[i]) / equity[i]
    """
    series = np.asarray(list(equity_curve), dtype=np.float64)
    if series.size < 2:
        return np.array([], dtype=np.float64)
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        returns = np.diff(series) / series[:-1]
    
    # Replace inf/nan with 0
    returns = np.where(np.isfinite(returns), returns, 0.0)
    return returns


def annualized_sharpe(returns: Iterable[float], periods_per_year: float = 365.0, risk_free_rate: float = 0.0) -> float:
    """Compute annualized Sharpe ratio.

    Args:
        returns: Array of periodic returns
        periods_per_year: Number of periods in a year.
            Use 365 for crypto daily data, 8760 for crypto hourly data.
            Do NOT use 252 for crypto — that is for equity markets.
        risk_free_rate: Annual risk-free rate (default 0.0)

    Returns:
        Annualized Sharpe ratio, or 0.0 on degenerate data.
    """
    returns_arr = np.asarray(list(returns), dtype=np.float64)
    if returns_arr.size == 0:
        return 0.0
    
    # Filter out non-finite values
    returns_arr = returns_arr[np.isfinite(returns_arr)]
    if returns_arr.size == 0:
        return 0.0
    
    mean_return = np.mean(returns_arr)
    std_return = np.std(returns_arr, ddof=1) if returns_arr.size > 1 else 0.0
    
    if std_return == 0.0:
        return 0.0
    
    # Annualize
    annualized_mean = mean_return * periods_per_year
    annualized_std = std_return * np.sqrt(periods_per_year)
    
    sharpe = (annualized_mean - risk_free_rate) / annualized_std
    return float(sharpe) if np.isfinite(sharpe) else 0.0


def annualized_sortino(returns: Iterable[float], periods_per_year: float = 365.0, risk_free_rate: float = 0.0) -> float:
    """Compute annualized Sortino ratio using RMS of negative returns as downside deviation.

    Formula matches the C trading environment (pufferlib_market/src/trading_env.c):
        downside_dev = sqrt(mean(neg_returns ** 2))   # population RMS, no ddof
        sortino = (mean_return / downside_dev) * sqrt(periods_per_year)

    Args:
        returns: Array of periodic returns
        periods_per_year: Number of periods in a year.
            Use 365 for crypto daily data (trades 24/7),
            use 8760 (24*365) for crypto hourly data.
            Do NOT use 252 for crypto — that is for equity markets.
        risk_free_rate: Annual risk-free rate (default 0.0)

    Returns:
        Annualized Sortino ratio, or 0.0 on insufficient/degenerate data.
    """
    returns_arr = np.asarray(list(returns), dtype=np.float64)
    if returns_arr.size == 0:
        return 0.0

    # Filter out non-finite values
    returns_arr = returns_arr[np.isfinite(returns_arr)]
    if returns_arr.size < 2:
        return 0.0

    mean_return = np.mean(returns_arr)

    # Downside deviation: matches C env exactly.
    # C code: sum_neg_sq accumulated only for negative returns,
    #         ret_count incremented for ALL steps.
    # => downside_dev = sqrt(sum_neg_sq / ret_count)
    #                 = sqrt(sum(neg**2) / total_n)
    # This is NOT the same as sqrt(mean(neg**2)) which divides by neg_count only.
    downside_returns = returns_arr[returns_arr < 0.0]
    if downside_returns.size == 0:
        # No downside at all — fall back to Sharpe denominator so ratio is
        # still meaningful (all returns positive → large positive Sortino).
        return annualized_sharpe(returns_arr, periods_per_year=periods_per_year, risk_free_rate=risk_free_rate)

    # sum_neg_sq / total_n  — matches C: sqrtf(sum_neg_sq / ret_count)
    downside_dev = float(np.sqrt(np.sum(downside_returns ** 2) / returns_arr.size))

    if downside_dev <= 0.0:
        return 0.0

    ppy = float(periods_per_year) if periods_per_year > 0 else 365.0
    sortino = ((mean_return - risk_free_rate / ppy) / downside_dev) * np.sqrt(ppy)
    return float(sortino) if np.isfinite(sortino) else 0.0
