"""Causal historical MAE error bands for hourly forecast prompts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_LOOKBACK_DAYS = 30.0
DEFAULT_MIN_SAMPLES = 24


@dataclass(frozen=True)
class HistoricalMAEBand:
    horizon_hours: int
    mae_pct: float
    samples: int

    def as_prompt_context(self) -> dict[str, float | int]:
        return {
            "horizon_hours": int(self.horizon_hours),
            "mae_pct": float(self.mae_pct),
            "samples": int(self.samples),
        }


class HistoricalForecastErrorEstimator:
    """Estimate forecast error bands using only targets resolved by the query time."""

    def __init__(self, horizon_hours: int, target_ns: np.ndarray, error_pct: np.ndarray) -> None:
        self.horizon_hours = int(horizon_hours)
        self._target_ns = np.asarray(target_ns, dtype=np.int64)
        self._error_pct = np.asarray(error_pct, dtype=np.float64)
        self._error_cumsum = (
            np.cumsum(self._error_pct, dtype=np.float64)
            if self._error_pct.size
            else np.asarray([], dtype=np.float64)
        )

    @classmethod
    def from_frames(
        cls,
        *,
        bars: pd.DataFrame,
        forecasts: pd.DataFrame,
        horizon_hours: int,
        predicted_close_col: str = "predicted_close_p50",
    ) -> "HistoricalForecastErrorEstimator":
        if bars.empty or forecasts.empty or predicted_close_col not in forecasts.columns:
            return cls(horizon_hours, np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64))

        history = bars.loc[:, ["timestamp", "close"]].copy()
        history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True, errors="coerce")
        history["close"] = pd.to_numeric(history["close"], errors="coerce")
        history = (
            history.dropna(subset=["timestamp", "close"])
            .sort_values("timestamp")
            .drop_duplicates(subset=["timestamp"], keep="last")
            .rename(columns={"timestamp": "target_timestamp", "close": "actual_close"})
        )
        if history.empty:
            return cls(horizon_hours, np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64))

        forecast = forecasts.copy()
        forecast["timestamp"] = pd.to_datetime(forecast["timestamp"], utc=True, errors="coerce")
        if "target_timestamp" in forecast.columns:
            forecast["target_timestamp"] = pd.to_datetime(
                forecast["target_timestamp"], utc=True, errors="coerce"
            )
        else:
            forecast["target_timestamp"] = forecast["timestamp"] + pd.Timedelta(
                hours=max(0, int(horizon_hours) - 1)
            )
        forecast["predicted_close"] = pd.to_numeric(forecast[predicted_close_col], errors="coerce")
        forecast = forecast.dropna(subset=["timestamp", "target_timestamp", "predicted_close"])
        if forecast.empty:
            return cls(horizon_hours, np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64))

        merged = (
            forecast.merge(history, on="target_timestamp", how="inner")
            .dropna(subset=["actual_close"])
            .sort_values(["target_timestamp", "timestamp"])
            .drop_duplicates(subset=["target_timestamp"], keep="last")
            .reset_index(drop=True)
        )
        if merged.empty:
            return cls(horizon_hours, np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64))

        actual = merged["actual_close"].to_numpy(dtype=np.float64, copy=False)
        predicted = merged["predicted_close"].to_numpy(dtype=np.float64, copy=False)
        with np.errstate(divide="ignore", invalid="ignore"):
            error_pct = np.abs(predicted - actual) / np.abs(actual) * 100.0
        valid = np.isfinite(error_pct)
        if not bool(valid.any()):
            return cls(horizon_hours, np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64))

        target_ns = pd.to_datetime(
            merged.loc[valid, "target_timestamp"], utc=True, errors="coerce"
        ).astype("int64", copy=False).to_numpy()
        return cls(horizon_hours, target_ns, error_pct[valid])

    def band_at(
        self,
        asof: pd.Timestamp,
        *,
        lookback_days: float = DEFAULT_LOOKBACK_DAYS,
        min_samples: int = DEFAULT_MIN_SAMPLES,
    ) -> HistoricalMAEBand | None:
        if self._target_ns.size == 0:
            return None

        asof_ts = pd.Timestamp(asof)
        if asof_ts.tzinfo is None:
            asof_ts = asof_ts.tz_localize("UTC")
        else:
            asof_ts = asof_ts.tz_convert("UTC")

        end = int(np.searchsorted(self._target_ns, asof_ts.value, side="right"))
        if end <= 0:
            return None

        start = 0
        if lookback_days > 0:
            lookback_start_ns = (asof_ts - pd.Timedelta(days=float(lookback_days))).value
            start = int(np.searchsorted(self._target_ns, lookback_start_ns, side="left"))

        count = end - start
        if count < int(min_samples):
            start = 0
            count = end
        if count <= 0:
            return None

        total_error = self._error_cumsum[end - 1]
        if start > 0:
            total_error -= self._error_cumsum[start - 1]
        mae_pct = float(total_error / count)
        return HistoricalMAEBand(
            horizon_hours=self.horizon_hours,
            mae_pct=mae_pct,
            samples=int(count),
        )

