from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd


def compute_predicted_close_return(predicted_close: Optional[float], prev_close: Optional[float]) -> Optional[float]:
    """Compute fractional return (e.g. 0.01 == +1%) from prev_close to predicted_close.

    Returns None if inputs are missing/invalid/non-finite, or if prev_close <= 0.
    """

    try:
        pred = float(predicted_close)  # type: ignore[arg-type]
        prev = float(prev_close)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if prev <= 0.0:
        return None
    value = (pred - prev) / prev
    if not math.isfinite(value):
        return None
    return float(value)


def should_trade_predicted_close_return(
    predicted_close_return: Optional[float],
    *,
    min_return_pct: Optional[float],
) -> bool:
    """Return True if the direction filter allows trading.

    min_return_pct is a fractional threshold (e.g. 0.002 == +0.2%). If None, always True.
    """

    if min_return_pct is None:
        return True
    try:
        thr = float(min_return_pct)
    except (TypeError, ValueError):
        raise ValueError("min_return_pct must be a float or None") from None
    if not math.isfinite(thr):
        raise ValueError("min_return_pct must be finite") from None

    if predicted_close_return is None:
        return False
    try:
        value = float(predicted_close_return)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(value):
        return False
    return value >= thr


def filter_forecasts_by_predicted_close_return(
    forecasts: pd.DataFrame,
    daily: pd.DataFrame,
    *,
    min_return_pct: float,
    predicted_close_col: str = "predicted_close_p50",
) -> pd.DataFrame:
    """Keep only forecast days where predicted close exceeds previous day's realized close.

    This is primarily for backtests: if the filter removes a day, the simulator will not place
    that day's buy/sell watchers.
    """

    if forecasts is None or forecasts.empty:
        return pd.DataFrame()
    if daily is None or daily.empty:
        return forecasts

    if predicted_close_col not in forecasts.columns:
        raise KeyError(f"Forecast frame missing {predicted_close_col!r}")

    daily_ts = daily[["timestamp", "close"]].copy()
    daily_ts["timestamp"] = pd.to_datetime(daily_ts["timestamp"], utc=True, errors="coerce")
    daily_ts = daily_ts.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    daily_ts["prev_close"] = pd.to_numeric(daily_ts["close"], errors="coerce").shift(1)

    merged = forecasts.merge(daily_ts[["timestamp", "prev_close"]], on="timestamp", how="left", sort=False)
    prev = pd.to_numeric(merged["prev_close"], errors="coerce")
    pred = pd.to_numeric(merged[predicted_close_col], errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = (pred - prev) / prev
    keep = ret >= float(min_return_pct)
    keep = keep.fillna(False)

    # Preserve the original columns and order.
    out = forecasts.loc[keep.to_numpy()].reset_index(drop=True)
    return out

