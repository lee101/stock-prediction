from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.binance_symbol_utils import forecast_cache_symbol_candidates


def load_latest_forecast_from_cache(
    symbol: str,
    horizon: int,
    cache_root: Path,
    *,
    as_of: pd.Timestamp | str | None = None,
) -> Optional[dict[str, float]]:
    """Load the newest numeric forecast row for a symbol/horizon from cache.

    Symbol lookup is alias-aware so ``BNBUSD`` can resolve ``BNBUSDT`` or
    ``BNBFDUSD`` forecast files when the exact cache key is absent.
    """
    root = Path(cache_root)
    horizon_dir = root / f"h{int(horizon)}"
    as_of_ts: pd.Timestamp | None = None
    if as_of is not None:
        as_of_ts = pd.Timestamp(as_of)
        if as_of_ts.tzinfo is None:
            as_of_ts = as_of_ts.tz_localize("UTC")
        else:
            as_of_ts = as_of_ts.tz_convert("UTC")

    for candidate in forecast_cache_symbol_candidates(symbol):
        path = horizon_dir / f"{candidate}.parquet"
        if not path.exists():
            continue
        try:
            frame = pd.read_parquet(path)
        except Exception:
            continue
        if frame.empty:
            continue

        frame = frame.copy()
        time_col = "issued_at" if "issued_at" in frame.columns else "timestamp"
        if time_col in frame.columns:
            frame[time_col] = pd.to_datetime(frame[time_col], utc=True, errors="coerce")
            frame = frame.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
        if frame.empty:
            continue

        if as_of_ts is not None and time_col in frame.columns:
            eligible = frame[frame[time_col] <= as_of_ts]
            if eligible.empty:
                continue
            row = eligible.iloc[-1]
        else:
            row = frame.iloc[-1]

        result: dict[str, float] = {}
        for column in frame.columns:
            if column == "timestamp":
                continue
            value = row.get(column)
            if pd.isna(value):
                continue
            try:
                result[column] = float(value)
            except (TypeError, ValueError):
                continue
        if result:
            return result
    return None


__all__ = ["load_latest_forecast_from_cache"]
