from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


OHLC_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")


@dataclass(frozen=True)
class DailyOHLC:
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float

    def as_dict(self) -> dict[str, float | pd.Timestamp]:
        return {
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
        }


def load_ohlc_csv(path: Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing OHLC CSV: {csv_path}")

    frame = pd.read_csv(csv_path)
    if "timestamp" not in frame.columns:
        raise KeyError(f"{csv_path} missing 'timestamp' column")

    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    for column in OHLC_COLUMNS:
        if column not in frame.columns:
            raise KeyError(f"{csv_path} missing required OHLC column {column!r}")
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=list(OHLC_COLUMNS)).reset_index(drop=True)
    return frame


def normalize_symbol_token(symbol: str) -> str:
    return str(symbol or "").strip().upper()


def aggregate_intraday_ohlc(frame: pd.DataFrame) -> DailyOHLC:
    if frame.empty:
        raise ValueError("Cannot aggregate an empty intraday frame.")

    working = frame.copy()
    if "timestamp" not in working.columns:
        raise KeyError("Intraday frame missing 'timestamp' column.")
    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True, errors="coerce")
    working = working.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if working.empty:
        raise ValueError("Intraday frame contains no valid timestamps.")

    for column in OHLC_COLUMNS:
        if column not in working.columns:
            raise KeyError(f"Intraday frame missing {column!r}.")
        working[column] = pd.to_numeric(working[column], errors="coerce")
    working = working.dropna(subset=list(OHLC_COLUMNS)).reset_index(drop=True)
    if working.empty:
        raise ValueError("Intraday frame contains no usable OHLC rows.")

    first_ts = pd.Timestamp(working["timestamp"].iloc[0]).floor("D")
    return DailyOHLC(
        timestamp=first_ts,
        open=float(working["open"].iloc[0]),
        high=float(working["high"].max()),
        low=float(working["low"].min()),
        close=float(working["close"].iloc[-1]),
    )


def aggregate_hourly_prediction_frame_to_daily(frame: pd.DataFrame) -> DailyOHLC:
    return aggregate_intraday_ohlc(frame.reset_index().rename(columns={"index": "timestamp"}))


def ohlc_mape_percent(
    predicted: DailyOHLC | Mapping[str, object],
    actual: DailyOHLC | Mapping[str, object],
    *,
    columns: Sequence[str] = OHLC_COLUMNS,
) -> float:
    predicted_arr = _ohlc_array(predicted, columns=columns)
    actual_arr = _ohlc_array(actual, columns=columns)
    denom = np.clip(np.abs(actual_arr), 1e-8, None)
    return float(np.mean(np.abs(predicted_arr - actual_arr) / denom) * 100.0)


def ohlc_error_percent_by_column(
    predicted: DailyOHLC | Mapping[str, object],
    actual: DailyOHLC | Mapping[str, object],
    *,
    columns: Sequence[str] = OHLC_COLUMNS,
) -> dict[str, float]:
    predicted_arr = _ohlc_array(predicted, columns=columns)
    actual_arr = _ohlc_array(actual, columns=columns)
    denom = np.clip(np.abs(actual_arr), 1e-8, None)
    errors = np.abs(predicted_arr - actual_arr) / denom * 100.0
    return {str(column): float(error) for column, error in zip(columns, errors)}


def complete_utc_days_from_hourly(frame: pd.DataFrame, *, expected_bars: int = 24) -> list[pd.Timestamp]:
    if frame.empty:
        return []
    working = frame.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True, errors="coerce")
    working = working.dropna(subset=["timestamp"])
    if working.empty:
        return []
    working["day"] = working["timestamp"].dt.floor("D")
    counts = working.groupby("day", sort=True)["timestamp"].nunique()
    valid = counts[counts >= int(expected_bars)].index
    return [pd.Timestamp(day).floor("D") for day in valid]


def available_daily_days(frame: pd.DataFrame) -> list[pd.Timestamp]:
    if frame.empty:
        return []
    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    ts = ts.dropna()
    return [pd.Timestamp(day).floor("D") for day in sorted(set(ts.dt.floor("D")))]


def intersect_symbol_days(
    *,
    daily_frames: Mapping[str, pd.DataFrame],
    hourly_frames: Mapping[str, pd.DataFrame],
    symbols: Iterable[str],
    expected_hourly_bars: int = 24,
) -> list[pd.Timestamp]:
    common_days: set[pd.Timestamp] | None = None
    for symbol in symbols:
        daily = daily_frames.get(symbol)
        hourly = hourly_frames.get(symbol)
        if daily is None or hourly is None or daily.empty or hourly.empty:
            continue
        daily_days = set(available_daily_days(daily))
        hourly_days = set(complete_utc_days_from_hourly(hourly, expected_bars=expected_hourly_bars))
        symbol_days = daily_days & hourly_days
        if common_days is None:
            common_days = set(symbol_days)
        else:
            common_days &= symbol_days
    if not common_days:
        return []
    return sorted(common_days)


def _ohlc_array(
    value: DailyOHLC | Mapping[str, object],
    *,
    columns: Sequence[str],
) -> np.ndarray:
    if isinstance(value, DailyOHLC):
        source: Mapping[str, object] = value.as_dict()
    else:
        source = value
    numeric: list[float] = []
    for column in columns:
        raw = source.get(str(column))
        try:
            parsed = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Missing or invalid {column!r} value: {raw!r}") from exc
        if not np.isfinite(parsed):
            raise ValueError(f"Non-finite {column!r} value: {parsed!r}")
        numeric.append(parsed)
    return np.asarray(numeric, dtype=np.float64)


__all__ = [
    "DailyOHLC",
    "OHLC_COLUMNS",
    "aggregate_hourly_prediction_frame_to_daily",
    "aggregate_intraday_ohlc",
    "available_daily_days",
    "complete_utc_days_from_hourly",
    "intersect_symbol_days",
    "load_ohlc_csv",
    "normalize_symbol_token",
    "ohlc_error_percent_by_column",
    "ohlc_mape_percent",
]
