from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

UTC = ZoneInfo("UTC")
NEW_YORK = ZoneInfo("America/New_York")


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
    """Return True during NYSE trading hours for the provided or current time."""
    now_nyse = _timestamp_in_new_york(timestamp)

    if now_nyse.weekday() >= 5:
        return False

    market_open = now_nyse.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_nyse.replace(hour=16, minute=0, second=0, microsecond=0)

    return market_open <= now_nyse <= market_close
