from datetime import datetime
from functools import lru_cache
from typing import Optional
from zoneinfo import ZoneInfo

from loguru import logger

UTC = ZoneInfo("UTC")
NEW_YORK = ZoneInfo("America/New_York")

# Lazy-loaded exchange calendar for NYSE
_nyse_calendar = None


def _get_nyse_calendar():
    """Lazily load the NYSE calendar to avoid import overhead."""
    global _nyse_calendar
    if _nyse_calendar is None:
        try:
            import exchange_calendars as xcals
            _nyse_calendar = xcals.get_calendar("XNYS")
        except ImportError:
            logger.warning("exchange_calendars not installed, falling back to weekday check")
            return None
    return _nyse_calendar


@lru_cache(maxsize=1024)
def _is_nyse_session_cached(date_str: str) -> bool:
    """Cache-friendly check if a date is an NYSE trading session.

    Args:
        date_str: Date in YYYY-MM-DD format

    Returns:
        True if NYSE is open on this date
    """
    cal = _get_nyse_calendar()
    if cal is None:
        # Fallback: weekday check only
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.weekday() < 5

    import pandas as pd
    return cal.is_session(pd.Timestamp(date_str))


def is_nyse_open_on_date(dt: datetime) -> bool:
    """Check if NYSE is open on a given date (handles holidays).

    This uses exchange_calendars to properly handle:
    - Weekends
    - US public holidays (MLK Day, Presidents Day, Memorial Day, etc.)
    - Early closes (day before Thanksgiving, Christmas Eve, etc.)

    Args:
        dt: The datetime to check

    Returns:
        True if NYSE has a trading session on this date
    """
    date_str = dt.strftime("%Y-%m-%d")
    return _is_nyse_session_cached(date_str)


def _timestamp_in_new_york(timestamp: Optional[datetime] = None) -> datetime:
    """Convert timestamp to America/New_York, defaulting to current time."""
    base = timestamp or datetime.now(tz=UTC)
    # Ensure timezone aware before conversion
    aware = base if base.tzinfo else base.replace(tzinfo=UTC)
    return aware.astimezone(NEW_YORK)


def is_nyse_trading_day_ending(timestamp: Optional[datetime] = None) -> bool:
    """Return True when the NYSE trading day is ending (2-5pm ET)."""
    now_nyse = _timestamp_in_new_york(timestamp)
    return now_nyse.hour in {14, 15, 16, 17}


def is_nyse_trading_day_now(timestamp: Optional[datetime] = None) -> bool:
    """Return True during NYSE trading hours for the provided or current time.

    Uses exchange_calendars for accurate holiday detection when available.
    """
    now_nyse = _timestamp_in_new_york(timestamp)

    # Use calendar-based check for holidays (falls back to weekday if unavailable)
    if not is_nyse_open_on_date(now_nyse):
        return False

    market_open = now_nyse.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_nyse.replace(hour=16, minute=0, second=0, microsecond=0)

    return market_open <= now_nyse <= market_close
