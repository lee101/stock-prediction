"""Metrics utilities for trading performance analysis."""

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


def annualized_sharpe(returns: Iterable[float], periods_per_year: float = 252.0, risk_free_rate: float = 0.0) -> float:
    """Compute annualized Sharpe ratio.
    
    Args:
        returns: Array of periodic returns
        periods_per_year: Number of periods in a year (252 for daily, 24*365 for hourly)
        risk_free_rate: Annual risk-free rate (default 0.0)
        
    Returns:
        Annualized Sharpe ratio
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


def annualized_sortino(returns: Iterable[float], periods_per_year: float = 252.0, risk_free_rate: float = 0.0) -> float:
    """Compute annualized Sortino ratio (uses downside deviation).
    
    Args:
        returns: Array of periodic returns
        periods_per_year: Number of periods in a year (252 for daily, 24*365 for hourly)
        risk_free_rate: Annual risk-free rate (default 0.0)
        
    Returns:
        Annualized Sortino ratio
    """
    returns_arr = np.asarray(list(returns), dtype=np.float64)
    if returns_arr.size == 0:
        return 0.0
    
    # Filter out non-finite values
    returns_arr = returns_arr[np.isfinite(returns_arr)]
    if returns_arr.size == 0:
        return 0.0
    
    mean_return = np.mean(returns_arr)
    
    # Downside deviation: only consider negative returns
    downside_returns = returns_arr[returns_arr < 0.0]
    if downside_returns.size == 0:
        # No downside = infinite Sortino, return a large value
        return float(mean_return * periods_per_year * 100.0) if mean_return > 0 else 0.0
    
    downside_std = np.std(downside_returns, ddof=1) if downside_returns.size > 1 else np.abs(downside_returns[0])
    
    if downside_std == 0.0:
        return 0.0
    
    # Annualize
    annualized_mean = mean_return * periods_per_year
    annualized_downside = downside_std * np.sqrt(periods_per_year)
    
    sortino = (annualized_mean - risk_free_rate) / annualized_downside
    return float(sortino) if np.isfinite(sortino) else 0.0
