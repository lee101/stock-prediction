"""Utility functions for Bags.fm trading bot."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def lamports_to_sol(lamports: int) -> float:
    """Convert lamports to SOL.

    Args:
        lamports: Amount in lamports

    Returns:
        Amount in SOL
    """
    return lamports / 1_000_000_000


def sol_to_lamports(sol: float) -> int:
    """Convert SOL to lamports.

    Args:
        sol: Amount in SOL

    Returns:
        Amount in lamports
    """
    return int(sol * 1_000_000_000)


def token_to_smallest_units(amount: float, decimals: int) -> int:
    """Convert token amount to smallest units.

    Args:
        amount: Amount in token units
        decimals: Token decimals

    Returns:
        Amount in smallest units
    """
    return int(amount * (10 ** decimals))


def smallest_units_to_token(amount: int, decimals: int) -> float:
    """Convert smallest units to token amount.

    Args:
        amount: Amount in smallest units
        decimals: Token decimals

    Returns:
        Amount in token units
    """
    return amount / (10 ** decimals)


def calculate_price_impact(
    amount: int,
    out_amount: int,
    fair_rate: float,
) -> float:
    """Calculate price impact from a swap quote.

    Args:
        amount: Input amount
        out_amount: Output amount from quote
        fair_rate: Fair exchange rate

    Returns:
        Price impact as decimal (0.01 = 1%)
    """
    if amount <= 0 or fair_rate <= 0:
        return 0.0

    actual_rate = out_amount / amount
    impact = (fair_rate - actual_rate) / fair_rate
    return max(0, impact)


def format_sol(amount: float, decimals: int = 4) -> str:
    """Format SOL amount for display.

    Args:
        amount: Amount in SOL
        decimals: Decimal places

    Returns:
        Formatted string
    """
    return f"{amount:.{decimals}f} SOL"


def format_usd(amount: float, decimals: int = 2) -> str:
    """Format USD amount for display.

    Args:
        amount: Amount in USD
        decimals: Decimal places

    Returns:
        Formatted string
    """
    return f"${amount:,.{decimals}f}"


def format_pct(value: float, decimals: int = 2) -> str:
    """Format percentage for display.

    Args:
        value: Value as decimal (0.01 = 1%)
        decimals: Decimal places

    Returns:
        Formatted string
    """
    return f"{value * 100:.{decimals}f}%"


def calculate_sharpe_ratio(
    returns: List[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 365 * 24 * 6,  # 10-min bars
) -> float:
    """Calculate Sharpe ratio from returns.

    Args:
        returns: List of period returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year

    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0

    returns = np.array(returns)
    excess_returns = returns - risk_free_rate / periods_per_year

    mean = np.mean(excess_returns)
    std = np.std(excess_returns)

    if std == 0:
        return 0.0

    return mean / std * np.sqrt(periods_per_year)


def calculate_sortino_ratio(
    returns: List[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 365 * 24 * 6,
) -> float:
    """Calculate Sortino ratio from returns.

    Args:
        returns: List of period returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year

    Returns:
        Annualized Sortino ratio
    """
    if len(returns) < 2:
        return 0.0

    returns = np.array(returns)
    excess_returns = returns - risk_free_rate / periods_per_year

    mean = np.mean(excess_returns)
    downside = returns[returns < 0]

    if len(downside) == 0:
        return float("inf") if mean > 0 else 0.0

    downside_std = np.std(downside)

    if downside_std == 0:
        return float("inf") if mean > 0 else 0.0

    return mean / downside_std * np.sqrt(periods_per_year)


def calculate_max_drawdown(equity_curve: List[float]) -> Tuple[float, int, int]:
    """Calculate maximum drawdown from equity curve.

    Args:
        equity_curve: List of portfolio values

    Returns:
        Tuple of (max_drawdown, peak_idx, trough_idx)
    """
    if len(equity_curve) < 2:
        return 0.0, 0, 0

    equity = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max

    max_dd_idx = np.argmin(drawdown)
    max_dd = abs(drawdown[max_dd_idx])

    # Find peak before trough
    peak_idx = np.argmax(equity[:max_dd_idx + 1])

    return max_dd, peak_idx, max_dd_idx


def calculate_returns(prices: List[float]) -> List[float]:
    """Calculate period returns from prices.

    Args:
        prices: List of prices

    Returns:
        List of returns
    """
    if len(prices) < 2:
        return []

    prices = np.array(prices)
    returns = np.diff(prices) / prices[:-1]
    return returns.tolist()


def exponential_moving_average(
    values: List[float],
    span: int,
) -> List[float]:
    """Calculate exponential moving average.

    Args:
        values: Input values
        span: EMA span

    Returns:
        EMA values
    """
    if len(values) == 0:
        return []

    alpha = 2 / (span + 1)
    ema = [values[0]]

    for i in range(1, len(values)):
        ema.append(alpha * values[i] + (1 - alpha) * ema[-1])

    return ema


def simple_moving_average(
    values: List[float],
    window: int,
) -> List[float]:
    """Calculate simple moving average.

    Args:
        values: Input values
        window: Window size

    Returns:
        SMA values (length = len(values) - window + 1)
    """
    if len(values) < window:
        return []

    values = np.array(values)
    weights = np.ones(window) / window
    return np.convolve(values, weights, mode="valid").tolist()


def bollinger_bands(
    values: List[float],
    window: int = 20,
    num_std: float = 2.0,
) -> Tuple[List[float], List[float], List[float]]:
    """Calculate Bollinger Bands.

    Args:
        values: Input values
        window: Window size
        num_std: Number of standard deviations

    Returns:
        Tuple of (middle, upper, lower) bands
    """
    if len(values) < window:
        return [], [], []

    values = np.array(values)
    middle = simple_moving_average(values.tolist(), window)

    # Calculate rolling std
    std = []
    for i in range(len(values) - window + 1):
        std.append(np.std(values[i : i + window]))

    upper = [m + num_std * s for m, s in zip(middle, std)]
    lower = [m - num_std * s for m, s in zip(middle, std)]

    return middle, upper, lower


def is_market_hours() -> bool:
    """Check if it's within typical crypto trading hours.

    Crypto trades 24/7, but this can be used to identify
    higher-volume periods.

    Returns:
        True if during high-volume hours (roughly US/Asia overlap)
    """
    now = datetime.utcnow()
    hour = now.hour

    # High volume: 12:00-22:00 UTC (US morning to Asia evening)
    return 12 <= hour < 22


def truncate_pubkey(pubkey: str, chars: int = 4) -> str:
    """Truncate a public key for display.

    Args:
        pubkey: Full public key
        chars: Number of chars to show at each end

    Returns:
        Truncated string (e.g., "So11...1112")
    """
    if len(pubkey) <= chars * 2 + 3:
        return pubkey
    return f"{pubkey[:chars]}...{pubkey[-chars:]}"


def validate_solana_address(address: str) -> bool:
    """Validate a Solana address format.

    Args:
        address: Address to validate

    Returns:
        True if valid format
    """
    import base58

    try:
        decoded = base58.b58decode(address)
        return len(decoded) == 32
    except Exception:
        return False


def get_explorer_url(signature: str, network: str = "mainnet") -> str:
    """Get Solana Explorer URL for a transaction.

    Args:
        signature: Transaction signature
        network: Network (mainnet, devnet, testnet)

    Returns:
        Explorer URL
    """
    base = "https://solscan.io/tx"
    if network != "mainnet":
        return f"{base}/{signature}?cluster={network}"
    return f"{base}/{signature}"


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
):
    """Decorator for retry with exponential backoff.

    Args:
        max_retries: Maximum retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
    """
    import asyncio
    import functools

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed: {e}. "
                        f"Retrying in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)

            raise last_exception

        return wrapper

    return decorator


class RateLimiter:
    """Simple rate limiter for API calls."""

    def __init__(
        self,
        calls_per_minute: int = 60,
    ) -> None:
        self.calls_per_minute = calls_per_minute
        self._call_times: List[datetime] = []

    async def acquire(self) -> None:
        """Wait until a call can be made."""
        import asyncio

        now = datetime.utcnow()
        cutoff = now - timedelta(minutes=1)

        # Remove old calls
        self._call_times = [t for t in self._call_times if t > cutoff]

        if len(self._call_times) >= self.calls_per_minute:
            # Wait until oldest call expires
            wait_time = (self._call_times[0] - cutoff).total_seconds()
            if wait_time > 0:
                await asyncio.sleep(wait_time)

        self._call_times.append(datetime.utcnow())

    @property
    def remaining(self) -> int:
        """Remaining calls in current window."""
        now = datetime.utcnow()
        cutoff = now - timedelta(minutes=1)
        recent = len([t for t in self._call_times if t > cutoff])
        return max(0, self.calls_per_minute - recent)
