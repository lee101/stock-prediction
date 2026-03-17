"""Shared Chronos2 post-processing helpers (scaling, sampling, aggregation)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.models.toto_aggregation import aggregate_with_spec

TARGET_COLUMNS: Tuple[str, ...] = ("open", "high", "low", "close")
GAUSSIAN_Q_FACTOR = 2.0 * 1.2815515655446004


@dataclass
class Chronos2AggregationSpec:
    aggregation: str = "median"
    sample_count: int = 0
    scaler: str = "none"

    def requires_samples(self) -> bool:
        return self.sample_count and self.sample_count > 1


@dataclass(frozen=True)
class Chronos2ForecastRepair:
    close_p10: float
    close_p50: float
    close_p90: float
    high_p50: float
    low_p50: float
    forecast_move_pct: float
    forecast_volatility_pct: float


class ColumnScaler:
    """Column-wise scaler that can be inverted (currently supports mean/std)."""

    def __init__(self, method: str, frame: pd.DataFrame, columns: Sequence[str]) -> None:
        self.method = (method or "none").lower()
        self.columns = tuple(columns)
        self.params: Dict[str, Dict[str, float]] = {}

        if self.method == "none":
            return

        if self.method == "meanstd":
            for column in self.columns:
                if column not in frame.columns:
                    continue
                series = frame[column].astype("float64")
                mean = float(series.mean())
                std = float(series.std(ddof=0))
                if not math.isfinite(std) or std < 1e-6:
                    std = max(abs(mean) * 1e-3, 1.0)
                self.params[column] = {"mean": mean, "std": std}
            return

        raise ValueError(f"Unsupported scaler '{self.method}'")

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.method == "none":
            return frame
        result = frame.copy()
        for column, stats in self.params.items():
            if column not in result.columns:
                continue
            result[column] = (result[column] - stats["mean"]) / stats["std"]
        return result

    def inverse(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.method == "none":
            return frame
        result = frame.copy()
        for column, stats in self.params.items():
            if column not in result.columns:
                continue
            result[column] = result[column] * stats["std"] + stats["mean"]
        return result


def resolve_quantile_levels(base_levels: Sequence[float], sample_count: int) -> Tuple[float, ...]:
    levels = set(base_levels)
    levels.add(0.5)
    if sample_count and sample_count > 1:
        levels.update((0.1, 0.9))
    return tuple(sorted(levels))


def _gaussian_sample_matrix(
    median: np.ndarray,
    q10: np.ndarray,
    q90: np.ndarray,
    sample_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if sample_count <= 0:
        return median.reshape(1, -1)
    spread = (q90 - q10) / GAUSSIAN_Q_FACTOR
    eps = np.maximum(1e-6, np.abs(median) * 1e-4)
    std = np.clip(spread, eps, None)
    samples = rng.normal(loc=median, scale=std, size=(sample_count, median.size))
    return samples.astype(np.float64)


def aggregate_quantile_forecasts(
    quantile_frames: Mapping[float, pd.DataFrame],
    *,
    columns: Sequence[str],
    spec: Chronos2AggregationSpec,
    rng: np.random.Generator,
) -> Dict[str, float]:
    if 0.5 not in quantile_frames:
        raise ValueError("Chronos2 output missing 0.5 quantile needed for aggregation")

    results: Dict[str, float] = {}
    median_frame = quantile_frames[0.5]
    fallback_low = min(quantile_frames)
    fallback_high = max(quantile_frames)

    for column in columns:
        if column not in median_frame.columns:
            continue
        median_series = median_frame[column].to_numpy(dtype=np.float64)
        if not spec.requires_samples():
            results[column] = float(np.atleast_1d(median_series)[0])
            continue

        low_frame = quantile_frames.get(0.1, quantile_frames[fallback_low])
        high_frame = quantile_frames.get(0.9, quantile_frames[fallback_high])
        q10_series = low_frame[column].to_numpy(dtype=np.float64)
        q90_series = high_frame[column].to_numpy(dtype=np.float64)

        sample_matrix = _gaussian_sample_matrix(
            median_series,
            q10_series,
            q90_series,
            spec.sample_count,
            rng,
        )

        try:
            aggregated = aggregate_with_spec(sample_matrix, spec.aggregation)
        except ValueError as exc:
            raise RuntimeError(
                f"Aggregation '{spec.aggregation}' failed for samples shape={sample_matrix.shape}"
            ) from exc

        results[column] = float(np.atleast_1d(aggregated)[0])

    return results


def chronos_rng(seed_value: int) -> np.random.Generator:
    seed = seed_value % (2**32)
    return np.random.default_rng(seed)


def _coerce_positive_or_fallback(value: float | None, fallback: float) -> float:
    try:
        numeric = float(value) if value is not None else fallback
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(numeric) or numeric <= 0.0:
        return fallback
    return numeric


def repair_forecast_ohlc(
    *,
    last_close: float,
    close_p50: float | None,
    close_p10: float | None = None,
    close_p90: float | None = None,
    high_p50: float | None = None,
    low_p50: float | None = None,
) -> Chronos2ForecastRepair:
    baseline = _coerce_positive_or_fallback(last_close, 0.0)
    close50 = _coerce_positive_or_fallback(close_p50, baseline or 1.0)
    close10 = _coerce_positive_or_fallback(close_p10, close50)
    close90 = _coerce_positive_or_fallback(close_p90, close50)

    ordered_close10 = min(close10, close50, close90)
    ordered_close90 = max(close10, close50, close90)
    ordered_close50 = min(max(close50, ordered_close10), ordered_close90)

    repaired_high = _coerce_positive_or_fallback(
        high_p50,
        max(ordered_close50, ordered_close90),
    )
    repaired_low = _coerce_positive_or_fallback(
        low_p50,
        min(ordered_close50, ordered_close10),
    )
    repaired_high = max(repaired_high, repaired_low, ordered_close50)
    repaired_low = min(repaired_low, repaired_high, ordered_close50)

    move_pct = 0.0
    if baseline:
        move_pct = (ordered_close50 - baseline) / baseline

    volatility_pct = 0.0
    if baseline:
        volatility_pct = max(ordered_close90 - ordered_close10, 0.0) / abs(baseline)

    return Chronos2ForecastRepair(
        close_p10=ordered_close10,
        close_p50=ordered_close50,
        close_p90=ordered_close90,
        high_p50=repaired_high,
        low_p50=repaired_low,
        forecast_move_pct=move_pct,
        forecast_volatility_pct=volatility_pct,
    )
