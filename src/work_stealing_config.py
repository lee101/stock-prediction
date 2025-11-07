"""Work stealing configuration for maxdiff order management."""

import os
from datetime import datetime
from typing import Optional

import pytz

# Crypto symbols (imported from fixtures for consistency)
from src.fixtures import active_crypto_symbols

# Market hours timezone
EST = pytz.timezone("US/Eastern")

# Crypto out-of-hours settings
CRYPTO_OUT_OF_HOURS_FORCE_IMMEDIATE_COUNT = int(
    os.getenv("CRYPTO_OUT_OF_HOURS_FORCE_COUNT", "2")  # Top 2 cryptos (only have 3 total)
)
CRYPTO_OUT_OF_HOURS_TOLERANCE_PCT = float(
    os.getenv("CRYPTO_OUT_OF_HOURS_TOLERANCE", "0.020")  # 2.0% - wider spreads out of hours
)
CRYPTO_NORMAL_TOLERANCE_PCT = float(
    os.getenv("CRYPTO_NORMAL_TOLERANCE", "0.0066")  # 0.66% - proven default
)

# Work stealing settings
WORK_STEALING_ENABLED = os.getenv("WORK_STEALING_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}
WORK_STEALING_ENTRY_TOLERANCE_PCT = float(
    os.getenv("WORK_STEALING_TOLERANCE", "0.0066")  # 0.66% - match normal tolerance so watchers can steal
)
WORK_STEALING_PROTECTION_PCT = float(
    os.getenv("WORK_STEALING_PROTECTION", "0.004")  # 0.4% - tighter than normal, protects near-execution
)
WORK_STEALING_COOLDOWN_SECONDS = int(
    os.getenv("WORK_STEALING_COOLDOWN", "120")  # 2 minutes - allow price movement recovery
)
WORK_STEALING_FIGHT_THRESHOLD = int(
    os.getenv("WORK_STEALING_FIGHT_THRESHOLD", "5")  # 5 steals = fighting (relaxed since PnL resolves)
)
WORK_STEALING_FIGHT_WINDOW_SECONDS = int(
    os.getenv("WORK_STEALING_FIGHT_WINDOW", "1800")  # 30 minutes - good detection window
)
WORK_STEALING_FIGHT_COOLDOWN_SECONDS = int(
    os.getenv("WORK_STEALING_FIGHT_COOLDOWN", "600")  # 10 minutes - less punitive than 30min
)
WORK_STEALING_DRY_RUN = os.getenv("WORK_STEALING_DRY_RUN", "0").strip().lower() in {"1", "true", "yes", "on"}

# Crypto symbols for out-of-hours logic
CRYPTO_SYMBOLS = frozenset(active_crypto_symbols)


def is_nyse_open(dt: Optional[datetime] = None) -> bool:
    """Check if NYSE is currently open.

    Args:
        dt: Datetime to check (defaults to now)

    Returns:
        True if NYSE is open for trading
    """
    if dt is None:
        dt = datetime.now(pytz.UTC)

    # Convert to EST
    dt_est = dt.astimezone(EST)

    # Weekend check
    if dt_est.weekday() >= 5:  # Saturday=5, Sunday=6
        return False

    # Market hours: 9:30 AM - 4:00 PM EST
    market_open = dt_est.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = dt_est.replace(hour=16, minute=0, second=0, microsecond=0)

    return market_open <= dt_est < market_close


def is_crypto_out_of_hours(dt: Optional[datetime] = None) -> bool:
    """Check if we're in crypto-only trading period (NYSE closed).

    Args:
        dt: Datetime to check (defaults to now)

    Returns:
        True if NYSE is closed (crypto can trade more aggressively)
    """
    return not is_nyse_open(dt)


def get_entry_tolerance_for_symbol(
    symbol: str,
    is_top_crypto: bool = False,
    dt: Optional[datetime] = None,
) -> float:
    """Get appropriate entry tolerance for a symbol.

    Args:
        symbol: Trading symbol
        is_top_crypto: Whether this is top-ranked crypto
        dt: Current datetime (defaults to now)

    Returns:
        Tolerance percentage as decimal (e.g., 0.0066 for 0.66%)
    """
    is_crypto = symbol in CRYPTO_SYMBOLS

    if not is_crypto:
        return CRYPTO_NORMAL_TOLERANCE_PCT

    # Crypto during stock hours uses normal tolerance
    if not is_crypto_out_of_hours(dt):
        return CRYPTO_NORMAL_TOLERANCE_PCT

    # Out of hours: top crypto is most aggressive
    if is_top_crypto:
        return CRYPTO_OUT_OF_HOURS_TOLERANCE_PCT

    # Other cryptos during out-of-hours
    return CRYPTO_OUT_OF_HOURS_TOLERANCE_PCT


def should_force_immediate_crypto(rank: int, dt: Optional[datetime] = None) -> bool:
    """Check if crypto should use force_immediate based on rank.

    Args:
        rank: Crypto rank (1-indexed, 1 = best)
        dt: Current datetime (defaults to now)

    Returns:
        True if this crypto should ignore tolerance and enter immediately
    """
    if not is_crypto_out_of_hours(dt):
        return False

    return rank <= CRYPTO_OUT_OF_HOURS_FORCE_IMMEDIATE_COUNT
