"""UTC clock helpers for Bags.fm."""

from __future__ import annotations

from datetime import UTC, date, datetime


def ensure_utc(value: datetime) -> datetime:
    """Normalize a datetime to timezone-aware UTC.

    Naive datetimes are treated as already being in UTC because Bags.fm has
    historically persisted timestamps without an offset while still using UTC
    semantics.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""
    return datetime.now(UTC)


def utc_today() -> date:
    """Return today's date in UTC."""
    return utc_now().date()
