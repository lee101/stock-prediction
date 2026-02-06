from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from kronostraining.metrics_utils import compute_mae_percent


@dataclass(frozen=True)
class ForecastMAE:
    symbol: str
    horizon_hours: int
    count: int
    mae: float
    mae_percent: float
    start_timestamp: pd.Timestamp
    end_timestamp: pd.Timestamp


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, context: str) -> None:
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise KeyError(f"{context} missing required column(s): {missing}")


def load_hourly_history_close(csv_path: Path) -> pd.DataFrame:
    """Load an hourly OHLC CSV and return timestamps with a canonical `close` column.

    The repo has multiple hourly CSV formats (Binance/Alpaca legacy). This helper
    normalizes the minimum required schema for forecast error evaluation.
    """

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Hourly history not found: {path}")

    frame = pd.read_csv(path)
    _require_columns(frame, ["timestamp"], context=str(path))

    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.assign(timestamp=ts).dropna(subset=["timestamp"]).sort_values("timestamp")
    frame = frame.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

    close_col: Optional[str] = None
    for candidate in ("close", "Close", "c", "C"):
        if candidate in frame.columns:
            close_col = candidate
            break
    if close_col is None:
        raise KeyError(f"{path} missing a close column (expected one of close/Close)")

    close = pd.to_numeric(frame[close_col], errors="coerce")
    frame = frame.assign(close=close).dropna(subset=["close"]).reset_index(drop=True)
    return frame[["timestamp", "close"]]


def load_forecast_cache(parquet_path: Path) -> pd.DataFrame:
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Forecast cache not found: {path}")
    frame = pd.read_parquet(path)
    if frame.empty:
        return frame
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if "issued_at" in frame.columns:
        frame["issued_at"] = pd.to_datetime(frame["issued_at"], utc=True, errors="coerce")
    if "target_timestamp" in frame.columns:
        frame["target_timestamp"] = pd.to_datetime(frame["target_timestamp"], utc=True, errors="coerce")
    return frame


def compute_forecast_mae(
    *,
    symbol: str,
    horizon_hours: int,
    history_close: pd.DataFrame,
    forecasts: pd.DataFrame,
    predicted_close_col: str = "predicted_close_p50",
) -> ForecastMAE:
    """Compute MAE (and MAE%) for a forecast cache against realized closes.

    The forecast cache is expected to contain a `timestamp` (the forecast row key)
    plus `predicted_close_p50`. Newer caches also include `target_timestamp` and
    `horizon_hours` (added with the multi-horizon alignment fix); when absent,
    this function derives `target_timestamp = timestamp + (horizon_hours - 1)h`.
    """

    if horizon_hours <= 0:
        raise ValueError("horizon_hours must be positive")

    if history_close.empty:
        raise ValueError("history_close must be non-empty")
    if forecasts.empty:
        raise ValueError("forecasts must be non-empty")

    history = history_close.copy()
    _require_columns(history, ["timestamp", "close"], context="history_close")
    history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True, errors="coerce")
    history["close"] = pd.to_numeric(history["close"], errors="coerce")
    history = history.dropna(subset=["timestamp", "close"]).drop_duplicates(subset=["timestamp"], keep="last")

    forecast = forecasts.copy()
    _require_columns(forecast, ["timestamp", predicted_close_col], context="forecasts")
    forecast["timestamp"] = pd.to_datetime(forecast["timestamp"], utc=True, errors="coerce")

    if "target_timestamp" in forecast.columns:
        forecast["target_timestamp"] = pd.to_datetime(forecast["target_timestamp"], utc=True, errors="coerce")
    else:
        forecast["target_timestamp"] = forecast["timestamp"] + pd.Timedelta(hours=int(horizon_hours) - 1)

    forecast["predicted_close"] = pd.to_numeric(forecast[predicted_close_col], errors="coerce")
    forecast = forecast.dropna(subset=["target_timestamp", "predicted_close"])

    actual = history.rename(columns={"timestamp": "target_timestamp", "close": "actual_close"})
    merged = forecast.merge(actual, on="target_timestamp", how="inner")
    merged = merged.dropna(subset=["actual_close"]).reset_index(drop=True)

    if merged.empty:
        raise ValueError(
            f"No overlapping timestamps between forecasts and history for {symbol} (horizon={horizon_hours}h)."
        )

    pred = merged["predicted_close"].to_numpy(dtype=np.float64, copy=False)
    act = merged["actual_close"].to_numpy(dtype=np.float64, copy=False)
    errors = np.abs(pred - act)
    mae = float(np.mean(errors))
    mae_pct = float(compute_mae_percent(mae, act))
    start_ts = pd.to_datetime(merged["target_timestamp"].min(), utc=True)
    end_ts = pd.to_datetime(merged["target_timestamp"].max(), utc=True)

    return ForecastMAE(
        symbol=str(symbol).upper(),
        horizon_hours=int(horizon_hours),
        count=int(len(merged)),
        mae=mae,
        mae_percent=mae_pct,
        start_timestamp=start_ts,
        end_timestamp=end_ts,
    )


def compute_forecast_cache_mae_for_paths(
    *,
    symbol: str,
    horizon_hours: int,
    history_csv: Path,
    forecast_parquet: Path,
    predicted_close_col: str = "predicted_close_p50",
) -> ForecastMAE:
    history = load_hourly_history_close(history_csv)
    forecasts = load_forecast_cache(forecast_parquet)
    return compute_forecast_mae(
        symbol=symbol,
        horizon_hours=horizon_hours,
        history_close=history,
        forecasts=forecasts,
        predicted_close_col=predicted_close_col,
    )


__all__ = [
    "ForecastMAE",
    "load_hourly_history_close",
    "load_forecast_cache",
    "compute_forecast_mae",
    "compute_forecast_cache_mae_for_paths",
]

