"""Helpers for reproducible backtest window selection."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def parse_utc_timestamp(value: object) -> pd.Timestamp:
    """Parse a timestamp-like value and normalize it to UTC."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def resolve_window_end(
    forecast_cutoffs: Iterable[object],
    requested_end: object | None = None,
) -> pd.Timestamp:
    """Resolve a valid UTC backtest end timestamp from available forecast cutoffs."""
    available = [
        parse_utc_timestamp(value)
        for value in forecast_cutoffs
        if value is not None and not pd.isna(value)
    ]
    if not available:
        raise ValueError("No forecast cutoffs available")

    latest_common_end = min(available)
    if requested_end is None:
        return latest_common_end

    requested = parse_utc_timestamp(requested_end)
    if requested > latest_common_end:
        raise ValueError(
            f"Requested end timestamp {requested.isoformat()} exceeds earliest forecast cutoff "
            f"{latest_common_end.isoformat()}"
        )
    return requested
