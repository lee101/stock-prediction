"""Load Chronos2 daily forecasts and compute forecast multipliers for candidate scoring."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.binance_symbol_utils import forecast_cache_symbol_candidates
from src.forecast_cache_lookup import _normalize_forecast_frame

DEFAULT_CACHE_DIR = "binanceneural/forecast_cache/h24/"


def load_daily_forecasts(
    symbols: list[str],
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> Dict[str, pd.DataFrame]:
    root = Path(cache_dir)
    result: Dict[str, pd.DataFrame] = {}
    if not root.exists():
        return result
    for sym in symbols:
        for candidate in forecast_cache_symbol_candidates(sym):
            path = root / f"{candidate}.parquet"
            if not path.exists():
                continue
            try:
                df = _normalize_forecast_frame(pd.read_parquet(path))
            except Exception:
                continue
            if df.empty:
                continue
            result[sym] = df
            break
    return result


def get_forecast_multiplier(
    symbol: str,
    date: pd.Timestamp,
    forecasts: Dict[str, pd.DataFrame],
    current_close: float,
) -> float:
    if symbol not in forecasts or current_close <= 0:
        return 0.0
    df = forecasts[symbol]
    if df.empty or "predicted_close_p50" not in df.columns:
        return 0.0
    time_col = "issued_at" if "issued_at" in df.columns else "timestamp"
    if time_col not in df.columns:
        return 0.0
    ts = pd.Timestamp(date)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    eligible = df[df[time_col] <= ts]
    if eligible.empty:
        return 0.0
    row = eligible.iloc[-1]
    predicted = float(row["predicted_close_p50"])
    if predicted <= 0:
        return 0.0
    return (predicted - current_close) / current_close
