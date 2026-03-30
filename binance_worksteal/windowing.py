"""Shared rolling-window helpers for binance_worksteal tools."""
from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


def compute_rolling_windows(
    all_bars: Mapping[str, pd.DataFrame],
    *,
    window_days: int = 60,
    n_windows: int = 3,
) -> list[tuple[str, str]]:
    earliest: pd.Timestamp | None = None
    latest: pd.Timestamp | None = None
    for df in all_bars.values():
        timestamps = df["timestamp"]
        if getattr(timestamps.dtype, "tz", None) is None or str(getattr(timestamps.dtype, "tz", "")) != "UTC":
            timestamps = pd.to_datetime(timestamps, utc=True, errors="coerce", format="mixed")
        series_min = timestamps.min()
        series_max = timestamps.max()
        if pd.isna(series_min) or pd.isna(series_max):
            continue
        earliest = series_min if earliest is None or series_min < earliest else earliest
        latest = series_max if latest is None or series_max > latest else latest
    if earliest is None or latest is None:
        return []

    windows: list[tuple[str, str]] = []
    for i in range(n_windows):
        end = latest - pd.Timedelta(days=i * window_days)
        start = end - pd.Timedelta(days=window_days)
        if start < earliest:
            break
        windows.append((str(start.date()), str(end.date())))
    return windows
