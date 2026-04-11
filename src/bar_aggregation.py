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


@dataclass(frozen=True)
class ShiftedSessionStats:
    """Lightweight diagnostics returned by `hourly_to_shifted_session_daily_ohlcv`."""

    total_virtual_days: int
    dropped_boundary: int


def hourly_to_shifted_session_daily_ohlcv(
    hourly: pd.DataFrame,
    *,
    offset_bars: int = 0,
    timestamp_col: str = "timestamp",
    require_full_shift: bool = True,
) -> tuple[pd.DataFrame, ShiftedSessionStats]:
    """Create shifted-session daily OHLCV bars from hourly data.

    Each "virtual day" for offset_bars=N uses the last (M-N) bars from calendar
    day i and the first N bars from calendar day i+1 (where M is the number of
    bars in day i).  This produces alternative daily bars without lookahead bias
    because the shifted bar only depends on already-elapsed hourly bars.

    For offset_bars=0 the result is equivalent to a standard daily OHLCV
    aggregation (same as ``hourly_to_daily_ohlcv``).

    Args:
        hourly: Hourly bars DataFrame.  Must contain ``timestamp_col``, ``open``,
            ``high``, ``low``, ``close``.  ``volume`` is aggregated when present.
        offset_bars: Number of hours to shift the session window.
        timestamp_col: Name of the UTC timestamp column.
        require_full_shift: When True (default), virtual days that do not have
            *offset_bars* bars available from the following calendar day are
            dropped.  This ensures every virtual day covers a complete shifted
            window.

    Returns:
        ``(daily_df, stats)`` where *daily_df* has columns
        ``[timestamp_col, open, high, low, close, volume?]`` and *stats*
        carries :class:`ShiftedSessionStats` diagnostics.
    """
    if not isinstance(hourly, pd.DataFrame):
        raise TypeError(f"hourly must be a pandas DataFrame, got {type(hourly).__name__}")
    if timestamp_col not in hourly.columns:
        raise KeyError(f"Missing required column '{timestamp_col}'")

    required = {"open", "high", "low", "close"}
    missing_cols = sorted(required - set(hourly.columns))
    if missing_cols:
        raise KeyError(f"Missing required OHLC column(s): {missing_cols}")

    df = hourly.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")
    df = df.dropna(subset=[timestamp_col]).sort_values(timestamp_col).reset_index(drop=True)

    if df.empty:
        return pd.DataFrame(), ShiftedSessionStats(0, 0)

    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Group into calendar days.
    df["_day"] = df[timestamp_col].dt.floor("D")
    days = sorted(df["_day"].unique())

    if offset_bars == 0:
        # Standard daily aggregation — combine all bars for each calendar day.
        records = []
        for day in days:
            day_bars = df[df["_day"] == day].sort_values(timestamp_col)
            if day_bars.empty:
                continue
            row: dict = {
                timestamp_col: day,
                "open": float(day_bars["open"].iloc[0]),
                "high": float(day_bars["high"].max()),
                "low": float(day_bars["low"].min()),
                "close": float(day_bars["close"].iloc[-1]),
            }
            if "volume" in day_bars.columns:
                vol = day_bars["volume"].sum(min_count=1)
                row["volume"] = float(vol) if not pd.isna(vol) else 0.0
            records.append(row)
        if not records:
            return pd.DataFrame(), ShiftedSessionStats(0, 0)
        result = pd.DataFrame(records)
        result[timestamp_col] = pd.to_datetime(result[timestamp_col], utc=True)
        return result, ShiftedSessionStats(len(result), 0)

    # Shifted-session aggregation.
    virtual_days = []
    dropped = 0

    for i in range(len(days) - 1):
        day_i_bars = df[df["_day"] == days[i]].sort_values(timestamp_col)
        day_j_bars = df[df["_day"] == days[i + 1]].sort_values(timestamp_col)

        if len(day_i_bars) <= offset_bars:
            # Not enough bars on day i to take a suffix after the offset.
            dropped += 1
            continue

        n_from_j = min(offset_bars, len(day_j_bars))
        if require_full_shift and n_from_j < offset_bars:
            # Day i+1 doesn't have enough bars to complete the shifted window.
            dropped += 1
            continue

        bars_from_i = day_i_bars.iloc[offset_bars:]
        bars_from_j = day_j_bars.iloc[:n_from_j]

        combined = pd.concat([bars_from_i, bars_from_j], ignore_index=True)
        if combined.empty:
            dropped += 1
            continue

        row = {
            timestamp_col: days[i],
            "open": float(combined["open"].iloc[0]),
            "high": float(combined["high"].max()),
            "low": float(combined["low"].min()),
            "close": float(combined["close"].iloc[-1]),
        }
        if "volume" in combined.columns:
            vol = combined["volume"].sum(min_count=1)
            row["volume"] = float(vol) if not pd.isna(vol) else 0.0
        virtual_days.append(row)

    if not virtual_days:
        return pd.DataFrame(), ShiftedSessionStats(0, dropped)

    result = pd.DataFrame(virtual_days)
    result[timestamp_col] = pd.to_datetime(result[timestamp_col], utc=True)
    return result, ShiftedSessionStats(len(result), dropped)


__all__ = [
    "DailyAggregationStats",
    "ShiftedSessionStats",
    "hourly_to_daily_ohlcv",
    "hourly_to_shifted_session_daily_ohlcv",
]
