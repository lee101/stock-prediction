from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional, Tuple


def floor_hour(dt: datetime) -> datetime:
    """Return the UTC hour bucket for a timestamp."""
    return floor_bucket(dt, bucket_minutes=60)


def floor_bucket(dt: datetime, *, bucket_minutes: int = 60) -> datetime:
    """Return the UTC time bucket for the provided minute interval."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    bucket_minutes = int(bucket_minutes) if isinstance(bucket_minutes, int | float) else 60
    if bucket_minutes <= 0:
        bucket_minutes = 60

    total_minutes = dt.hour * 60 + dt.minute
    bucket_total = (total_minutes // bucket_minutes) * bucket_minutes
    bucket_hour = (bucket_total // 60) % 24
    bucket_minute = bucket_total % 60

    return dt.replace(hour=bucket_hour, minute=bucket_minute, second=0, microsecond=0)


def update_intrahour_baseline(
    *,
    current_price: float,
    now: datetime,
    baseline_price: Optional[float],
    baseline_hour: Optional[datetime],
    min_price: float,
    bucket_minutes: int = 60,
) -> Tuple[Optional[float], Optional[datetime], str]:
    """Update the intrahour baseline price used for percent move checks."""
    try:
        current_price = float(current_price)
    except (TypeError, ValueError):
        return baseline_price, baseline_hour, "invalid_price"

    if not math.isfinite(current_price) or current_price <= 0:
        return baseline_price, baseline_hour, "invalid_price"

    if not math.isfinite(min_price) or min_price <= 0:
        min_price = 0.0

    if current_price < min_price:
        return None, None, "below_min"

    hour_bucket = floor_bucket(now, bucket_minutes=bucket_minutes)
    if baseline_price is None or baseline_hour is None or hour_bucket > baseline_hour:
        return current_price, hour_bucket, "reset"

    return baseline_price, baseline_hour, "keep"


def intrahour_triggered(
    *,
    current_price: float,
    baseline_price: Optional[float],
    pct_above: float,
) -> bool:
    """Return True when current_price rises pct_above above the baseline."""
    try:
        current_price = float(current_price)
        pct_above = float(pct_above)
    except (TypeError, ValueError):
        return False

    if not math.isfinite(current_price) or current_price <= 0:
        return False

    if baseline_price is None:
        return False
    try:
        baseline_price = float(baseline_price)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(baseline_price) or baseline_price <= 0:
        return False

    if not math.isfinite(pct_above) or pct_above < 0:
        return False

    return current_price >= baseline_price * (1.0 + pct_above)


__all__ = [
    "floor_hour",
    "floor_bucket",
    "update_intrahour_baseline",
    "intrahour_triggered",
]
