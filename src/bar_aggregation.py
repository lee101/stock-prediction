from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DailyAggregationStats:
    """Lightweight diagnostics returned by `hourly_to_daily_ohlcv`."""

    dropped_incomplete_last_day: bool
    dropped_incomplete_days: int


def hourly_to_daily_ohlcv(
    hourly: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
    expected_bars_per_day: int = 24,
    drop_incomplete_last_day: bool = True,
    drop_incomplete_days: bool = False,
    output_symbol: Optional[str] = None,
) -> tuple[pd.DataFrame, DailyAggregationStats]:
    """Aggregate an hourly OHLCV dataframe into daily bars (UTC).

    This is intended for crypto spot data where days are defined in UTC and should
    normally contain 24 hourly bars. We default to dropping the trailing day if it
    is incomplete (common when hourly data includes the current in-progress UTC day).

    Args:
        hourly: Hourly bars dataframe. Must contain at least `timestamp`, `open`,
            `high`, `low`, `close`. Additional columns (volume/trade_count/vwap/symbol)
            are optional and will be aggregated when present.
        timestamp_col: Timestamp column name (UTC).
        symbol_col: Symbol column name (optional).
        expected_bars_per_day: Expected number of bars per day (24 for hourly).
        drop_incomplete_last_day: Drop only the last day when it has fewer than
            `expected_bars_per_day` bars.
        drop_incomplete_days: Drop *all* days with fewer than `expected_bars_per_day` bars.
            If True, takes precedence over `drop_incomplete_last_day`.
        output_symbol: Override symbol value in the output dataframe (when symbol column is present).

    Returns:
        (daily_bars, stats)
    """

    if not isinstance(hourly, pd.DataFrame):
        raise TypeError(f"hourly must be a pandas DataFrame, got {type(hourly).__name__}")
    if timestamp_col not in hourly.columns:
        raise KeyError(f"Missing required column '{timestamp_col}'")

    required = {"open", "high", "low", "close"}
    missing = sorted(required - set(hourly.columns))
    if missing:
        raise KeyError(f"Missing required OHLC column(s): {missing}")

    df = hourly.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")
    df = df.dropna(subset=[timestamp_col]).sort_values(timestamp_col)
    df = df.drop_duplicates(subset=[timestamp_col], keep="last").reset_index(drop=True)
    if df.empty:
        stats = DailyAggregationStats(dropped_incomplete_last_day=False, dropped_incomplete_days=0)
        return pd.DataFrame(), stats

    # Normalize dtypes for aggregation (do not coerce timestamp/symbol).
    numeric_cols = [col for col in ("open", "high", "low", "close", "volume", "trade_count", "vwap") if col in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # A day is defined by UTC midnight boundaries.
    day = df[timestamp_col].dt.floor("D")
    df = df.assign(_day=day)
    group = df.groupby("_day", sort=True, dropna=True)

    # Base OHLC aggregation.
    daily = group.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    )

    if "volume" in df.columns:
        daily["volume"] = group["volume"].sum(min_count=1)
    if "trade_count" in df.columns:
        # Binance `number_of_trades` is int-like but may be missing for some data sources.
        daily["trade_count"] = group["trade_count"].sum(min_count=1)

    if "vwap" in df.columns and "volume" in df.columns:
        # Weighted-average VWAP across the day.
        # If a day has 0 total volume, fall back to the daily close.
        weighted = (df["volume"] * df["vwap"]).astype(float)
        vwap_num = weighted.groupby(df["_day"], sort=True, dropna=True).sum(min_count=1)
        vwap_den = df["volume"].groupby(df["_day"], sort=True, dropna=True).sum(min_count=1)
        denom = vwap_den.to_numpy(dtype=float)
        num = vwap_num.to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            vwap = np.where(denom > 0.0, num / denom, daily["close"].to_numpy(dtype=float))
        daily["vwap"] = vwap
    elif "vwap" in df.columns:
        daily["vwap"] = group["vwap"].mean()

    # Attach symbol if available (expect a single symbol; reject mixed frames defensively).
    if symbol_col in df.columns:
        symbols = df[symbol_col].dropna().astype(str).str.upper().unique().tolist()
        if output_symbol is not None:
            symbols = [str(output_symbol).upper()]
        if len(symbols) == 1:
            daily[symbol_col] = symbols[0]
        elif len(symbols) > 1:
            raise ValueError(f"Expected a single symbol in frame, got {sorted(set(symbols))[:5]}...")

    daily["bar_count"] = group.size().astype(int)
    daily = daily.reset_index().rename(columns={"_day": timestamp_col})

    # Drop days with incomplete bar counts (optional).
    dropped_days = 0
    dropped_last = False
    if expected_bars_per_day and expected_bars_per_day > 0 and not daily.empty:
        expected = int(expected_bars_per_day)
        if drop_incomplete_days:
            before = len(daily)
            daily = daily[daily["bar_count"] >= expected].reset_index(drop=True)
            dropped_days = before - len(daily)
        elif drop_incomplete_last_day:
            last_count = int(daily["bar_count"].iloc[-1])
            if last_count < expected:
                daily = daily.iloc[:-1].reset_index(drop=True)
                dropped_last = True

    # Drop any remaining NaNs in OHLC (e.g., fully-missing days after coercion).
    daily = daily.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)

    stats = DailyAggregationStats(
        dropped_incomplete_last_day=bool(dropped_last),
        dropped_incomplete_days=int(dropped_days),
    )
    return daily, stats


__all__ = ["DailyAggregationStats", "hourly_to_daily_ohlcv"]
