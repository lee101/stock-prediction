"""Price calculation utilities for trading strategies.

This module extracts common price movement calculations that were duplicated
across backtest_test3_inline.py in multiple strategy implementations.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray


def compute_close_to_extreme_movements(
    close_vals: NDArray[Any],
    high_vals: NDArray[Any],
    low_vals: NDArray[Any],
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Calculate percentage movements from close to high/low prices.

    This function computes how far the high and low prices moved from the
    close price, as a percentage. It handles division by zero safely and
    replaces NaN/inf values with 0.0.

    Args:
        close_vals: Array of closing prices
        high_vals: Array of high prices
        low_vals: Array of low prices

    Returns:
        Tuple of (close_to_high_pct, close_to_low_pct) where each is a
        numpy array of the same shape as the input arrays.

    Examples:
        >>> close = np.array([100.0, 200.0, 50.0])
        >>> high = np.array([110.0, 210.0, 55.0])
        >>> low = np.array([95.0, 190.0, 48.0])
        >>> high_pct, low_pct = compute_close_to_extreme_movements(close, high, low)
        >>> np.allclose(high_pct, [0.1, 0.05, 0.1])
        True
        >>> np.allclose(low_pct, [0.05, 0.05, 0.04])
        True

    Note:
        This was previously duplicated in backtest_test3_inline.py at lines:
        - 414-421 (maxdiff strategy)
        - 645-670 (maxdiff_always_on strategy)
        - 859-908 (pctdiff strategy)
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        # Calculate |1 - high/close| to get percentage movement
        close_to_high_np = np.abs(
            1.0 - np.divide(
                high_vals,
                close_vals,
                out=np.zeros_like(high_vals),
                where=close_vals != 0.0,
            )
        )

        # Calculate |1 - low/close| to get percentage movement
        close_to_low_np = np.abs(
            1.0 - np.divide(
                low_vals,
                close_vals,
                out=np.zeros_like(low_vals),
                where=close_vals != 0.0,
            )
        )

    # Replace NaN/inf with 0.0 for safety
    close_to_high_np = np.nan_to_num(close_to_high_np, nan=0.0, posinf=0.0, neginf=0.0)
    close_to_low_np = np.nan_to_num(close_to_low_np, nan=0.0, posinf=0.0, neginf=0.0)

    return close_to_high_np, close_to_low_np


def compute_price_range_pct(
    high_vals: NDArray[Any],
    low_vals: NDArray[Any],
    reference_vals: NDArray[Any],
) -> NDArray[Any]:
    """Calculate the percentage range between high and low prices.

    Args:
        high_vals: Array of high prices
        low_vals: Array of low prices
        reference_vals: Array of reference prices (typically close) for percentage

    Returns:
        Array of (high - low) / reference as percentages

    Examples:
        >>> high = np.array([110.0, 220.0])
        >>> low = np.array([90.0, 180.0])
        >>> close = np.array([100.0, 200.0])
        >>> ranges = compute_price_range_pct(high, low, close)
        >>> np.allclose(ranges, [0.2, 0.2])
        True
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        range_pct = np.divide(
            high_vals - low_vals,
            reference_vals,
            out=np.zeros_like(high_vals),
            where=reference_vals != 0.0,
        )

    return np.nan_to_num(range_pct, nan=0.0, posinf=0.0, neginf=0.0)


def safe_price_ratio(
    numerator: NDArray[Any],
    denominator: NDArray[Any],
    default: float = 1.0,
) -> NDArray[Any]:
    """Safely compute price ratios with division by zero handling.

    Args:
        numerator: Array of numerator values
        denominator: Array of denominator values
        default: Default value to use when denominator is zero

    Returns:
        Array of ratios with safe handling of division by zero

    Examples:
        >>> nums = np.array([100.0, 200.0, 50.0])
        >>> denoms = np.array([50.0, 0.0, 25.0])
        >>> ratios = safe_price_ratio(nums, denoms, default=1.0)
        >>> np.allclose(ratios, [2.0, 1.0, 2.0])
        True
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, default),
            where=denominator != 0.0,
        )

    return np.nan_to_num(ratio, nan=default, posinf=default, neginf=default)
