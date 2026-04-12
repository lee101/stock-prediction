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


@dataclass(frozen=True)
class ShiftedDailyAggregationStats:
    """Diagnostics returned by `hourly_to_shifted_session_daily_ohlcv`."""

    offset_bars: int
    source_sessions: int
    emitted_days: int
    dropped_incomplete_days: int


def _prepare_hourly_ohlcv_frame(
    hourly: pd.DataFrame,
    *,
    timestamp_col: str,
) -> pd.DataFrame:
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
        return df

    numeric_cols = [
        col for col in ("open", "high", "low", "close", "volume", "trade_count", "vwap")
        if col in df.columns
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _aggregate_ohlcv_window(
    frame: pd.DataFrame,
    *,
    timestamp_value: pd.Timestamp,
    timestamp_col: str,
    symbol_col: str,
    output_symbol: Optional[str],
) -> dict[str, object]:
    timestamp = pd.Timestamp(timestamp_value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    row: dict[str, object] = {
        timestamp_col: timestamp,
        "open": float(frame["open"].iloc[0]),
        "high": float(frame["high"].max()),
        "low": float(frame["low"].min()),
        "close": float(frame["close"].iloc[-1]),
        "bar_count": int(len(frame)),
    }

    if "volume" in frame.columns:
        row["volume"] = float(frame["volume"].sum(min_count=1))
    if "trade_count" in frame.columns:
        row["trade_count"] = float(frame["trade_count"].sum(min_count=1))

    if "vwap" in frame.columns and "volume" in frame.columns:
        vol = frame["volume"].astype(float).to_numpy(copy=False)
        vwap = frame["vwap"].astype(float).to_numpy(copy=False)
        denom = float(np.nansum(vol))
        row["vwap"] = float(np.nansum(vol * vwap) / denom) if denom > 0.0 else float(row["close"])
    elif "vwap" in frame.columns:
        row["vwap"] = float(pd.to_numeric(frame["vwap"], errors="coerce").mean())

    if symbol_col in frame.columns:
        symbols = frame[symbol_col].dropna().astype(str).str.upper().unique().tolist()
        resolved_symbol = str(output_symbol).upper() if output_symbol is not None else None
        if resolved_symbol is not None:
            row[symbol_col] = resolved_symbol
        elif len(symbols) == 1:
            row[symbol_col] = symbols[0]
        elif len(symbols) > 1:
            raise ValueError(f"Expected a single symbol in frame, got {sorted(set(symbols))[:5]}...")

    return row


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
    df = _prepare_hourly_ohlcv_frame(hourly, timestamp_col=timestamp_col)
    if df.empty:
        stats = DailyAggregationStats(dropped_incomplete_last_day=False, dropped_incomplete_days=0)
        return pd.DataFrame(), stats

    # A day is defined by UTC midnight boundaries.
    day = df[timestamp_col].dt.floor("D")
    df = df.assign(_day=day)
    rows = [
        _aggregate_ohlcv_window(
            day_frame.reset_index(drop=True),
            timestamp_value=day_value,
            timestamp_col=timestamp_col,
            symbol_col=symbol_col,
            output_symbol=output_symbol,
        )
        for day_value, day_frame in df.groupby("_day", sort=True, dropna=True)
    ]
    daily = pd.DataFrame(rows)

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


def hourly_to_shifted_session_daily_ohlcv(
    hourly: pd.DataFrame,
    *,
    offset_bars: int,
    timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
    require_full_shift: bool = True,
    output_symbol: Optional[str] = None,
) -> tuple[pd.DataFrame, ShiftedDailyAggregationStats]:
    """Aggregate hourly stock bars into synthetic daily bars with shifted session boundaries.

    Each real UTC trading session is treated as a source day. For `offset_bars > 0`,
    the synthetic day for session `i` is built from:
      - the tail of session `i` after skipping the first `offset_bars`, and
      - the first `offset_bars` bars from session `i + 1`.

    This yields a daily-like sample whose boundary is shifted forward by market bars,
    which is useful for daily-policy data augmentation from hourly stock bars.
    """

    if int(offset_bars) < 0:
        raise ValueError("offset_bars must be non-negative")

    df = _prepare_hourly_ohlcv_frame(hourly, timestamp_col=timestamp_col)
    if df.empty:
        return (
            pd.DataFrame(),
            ShiftedDailyAggregationStats(
                offset_bars=int(offset_bars),
                source_sessions=0,
                emitted_days=0,
                dropped_incomplete_days=0,
            ),
        )

    df = df.assign(_day=df[timestamp_col].dt.floor("D"))
    sessions = [
        (
            pd.Timestamp(day_value).tz_localize("UTC")
            if pd.Timestamp(day_value).tzinfo is None
            else pd.Timestamp(day_value).tz_convert("UTC"),
            day_frame.reset_index(drop=True),
        )
        for day_value, day_frame in df.groupby("_day", sort=True, dropna=True)
    ]

    rows: list[dict[str, object]] = []
    dropped = 0
    offset = int(offset_bars)

    for idx, (session_day, session_frame) in enumerate(sessions):
        if offset == 0:
            combined = session_frame
        else:
            next_frame = sessions[idx + 1][1] if idx + 1 < len(sessions) else None
            if next_frame is None:
                if require_full_shift:
                    dropped += 1
                    continue
                head = session_frame.iloc[0:0]
            else:
                head = next_frame.iloc[: min(offset, len(next_frame))]

            tail_start = min(offset, len(session_frame))
            tail = session_frame.iloc[tail_start:]
            if require_full_shift and (len(session_frame) <= offset or len(head) < offset):
                dropped += 1
                continue
            combined = pd.concat([tail, head], ignore_index=True)

        if combined.empty:
            dropped += 1
            continue

        rows.append(
            _aggregate_ohlcv_window(
                combined,
                timestamp_value=session_day,
                timestamp_col=timestamp_col,
                symbol_col=symbol_col,
                output_symbol=output_symbol,
            )
        )

    daily = pd.DataFrame(rows)
    if not daily.empty:
        daily = daily.sort_values(timestamp_col).reset_index(drop=True)

    return (
        daily,
        ShiftedDailyAggregationStats(
            offset_bars=offset,
            source_sessions=len(sessions),
            emitted_days=int(len(daily)),
            dropped_incomplete_days=int(dropped),
        ),
    )


__all__ = [
    "DailyAggregationStats",
    "ShiftedDailyAggregationStats",
    "hourly_to_daily_ohlcv",
    "hourly_to_shifted_session_daily_ohlcv",
]
