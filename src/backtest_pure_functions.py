"""
Pure utility functions extracted from backtest logic for unit testing.
These functions have no side effects and can be tested in isolation.
"""

from typing import Tuple
import torch
import numpy as np
from numpy.typing import NDArray


def validate_forecast_order(high_pred: torch.Tensor, low_pred: torch.Tensor) -> torch.Tensor:
    """
    Validate that forecasted price movements maintain logical order.

    Returns a mask of valid forecasts (True where low_pred < high_pred).
    """
    return low_pred < high_pred


def compute_return_profile(
    daily_returns: NDArray[np.float64],
    trading_days_per_year: int
) -> Tuple[float, float]:
    """
    Compute average daily and annualized returns from a series of returns.

    Args:
        daily_returns: Array of daily returns
        trading_days_per_year: Number of trading days in a year (252 for stocks, 365 for crypto)

    Returns:
        Tuple of (avg_daily_return, annualized_return)
    """
    if trading_days_per_year <= 0:
        return 0.0, 0.0
    if daily_returns.size == 0:
        return 0.0, 0.0

    finite_mask = np.isfinite(daily_returns)
    if not np.any(finite_mask):
        return 0.0, 0.0

    cleaned = daily_returns[finite_mask]
    if cleaned.size == 0:
        return 0.0, 0.0

    avg_daily = float(np.mean(cleaned))
    annualized = float(avg_daily * trading_days_per_year)
    return avg_daily, annualized


def calibrate_signal(
    predictions: NDArray[np.float64],
    actual_returns: NDArray[np.float64]
) -> Tuple[float, float]:
    """
    Calibrate predictions to actual returns using linear regression.

    Returns (slope, intercept) for the line: actual = slope * predicted + intercept
    """
    matched = min(len(predictions), len(actual_returns))
    if matched > 1:
        slope, intercept = np.polyfit(predictions[:matched], actual_returns[:matched], 1)
        return float(slope), float(intercept)
    return 1.0, 0.0


def simple_buy_sell_strategy(predictions: torch.Tensor, is_crypto: bool = False) -> torch.Tensor:
    """
    Generate positions based on predictions.

    Args:
        predictions: Predicted returns
        is_crypto: If True, only allow long positions (no shorts)

    Returns:
        Tensor of positions (1 for long, -1 for short, 0 for neutral)
    """
    predictions = torch.as_tensor(predictions)
    if is_crypto:
        return (predictions > 0).float()
    return (predictions > 0).float() * 2 - 1


def all_signals_strategy(
    close_pred: torch.Tensor,
    high_pred: torch.Tensor,
    low_pred: torch.Tensor,
    is_crypto: bool = False
) -> torch.Tensor:
    """
    Buy if all signals are positive; sell if all are negative; else hold.

    Args:
        close_pred: Predicted close returns
        high_pred: Predicted high returns
        low_pred: Predicted low returns
        is_crypto: If True, no short trades

    Returns:
        Tensor of positions (1 for long, -1 for short, 0 for hold)
    """
    close_pred, high_pred, low_pred = map(torch.as_tensor, (close_pred, high_pred, low_pred))

    buy_signal = (close_pred > 0) & (high_pred > 0) & (low_pred > 0)
    if is_crypto:
        return buy_signal.float()

    sell_signal = (close_pred < 0) & (high_pred < 0) & (low_pred < 0)
    return buy_signal.float() - sell_signal.float()


def buy_hold_strategy(predictions: torch.Tensor) -> torch.Tensor:
    """
    Buy when prediction is positive, hold otherwise.

    Returns:
        Tensor of positions (1 for buy, 0 for hold)
    """
    predictions = torch.as_tensor(predictions)
    return (predictions > 0).float()


def calculate_position_notional_value(market_value: float, qty: float, current_price: float) -> float:
    """
    Calculate absolute dollar notional for a position.

    Args:
        market_value: Market value of position (if available)
        qty: Quantity of shares/units
        current_price: Current price per unit

    Returns:
        Absolute notional value
    """
    if market_value and np.isfinite(market_value):
        return abs(float(market_value))

    if current_price > 0 and np.isfinite(current_price):
        return abs(float(qty * current_price))

    return abs(float(qty))
