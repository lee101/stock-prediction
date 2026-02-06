from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.forecast_cache_metrics import compute_forecast_cache_mae_for_paths, compute_forecast_mae


def _write_history_csv(path: Path, timestamps: list[pd.Timestamp], closes: list[float]) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [ts.isoformat() for ts in timestamps],
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": 0.0,
            "trade_count": 0,
            "vwap": closes,
            "symbol": "TEST",
        }
    )
    frame.to_csv(path, index=False)


def _write_forecast_parquet(
    path: Path,
    *,
    timestamps: list[pd.Timestamp],
    predicted: list[float],
    horizon_hours: int,
    include_target_timestamp: bool,
) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [ts.isoformat() for ts in timestamps],
            "symbol": "TEST",
            "horizon_hours": horizon_hours,
            "predicted_close_p50": predicted,
        }
    )
    if include_target_timestamp:
        frame["target_timestamp"] = [
            (ts + pd.Timedelta(hours=horizon_hours - 1)).isoformat() for ts in timestamps
        ]
    frame.to_parquet(path, index=False)


@pytest.mark.parametrize("include_target_timestamp", [False, True])
def test_compute_forecast_cache_mae_for_paths(include_target_timestamp: bool, tmp_path: Path) -> None:
    # History: 5 hours, close=100..104
    t0 = pd.Timestamp("2026-02-01T00:00:00Z")
    timestamps = [t0 + pd.Timedelta(hours=i) for i in range(5)]
    closes = [100.0 + float(i) for i in range(5)]

    history_csv = tmp_path / "TEST.csv"
    _write_history_csv(history_csv, timestamps, closes)

    # Forecast horizon=3: forecast rows keyed by `timestamp`, evaluating against close at +2h.
    forecast_ts = [timestamps[0], timestamps[1]]  # target closes at t2=102, t3=103
    actual = np.array([102.0, 103.0], dtype=np.float64)
    predicted = [103.0, 101.0]  # abs errors 1 and 2 => MAE 1.5
    mae = 1.5
    expected_mae_pct = float((mae / float(np.mean(np.abs(actual)))) * 100.0)

    forecast_parquet = tmp_path / "TEST.parquet"
    _write_forecast_parquet(
        forecast_parquet,
        timestamps=forecast_ts,
        predicted=predicted,
        horizon_hours=3,
        include_target_timestamp=include_target_timestamp,
    )

    result = compute_forecast_cache_mae_for_paths(
        symbol="TEST",
        horizon_hours=3,
        history_csv=history_csv,
        forecast_parquet=forecast_parquet,
    )

    assert result.symbol == "TEST"
    assert result.horizon_hours == 3
    assert result.count == 2
    assert result.mae == pytest.approx(mae, rel=1e-12, abs=1e-12)
    assert result.mae_percent == pytest.approx(expected_mae_pct, rel=1e-9, abs=1e-9)


def test_compute_forecast_mae_requires_overlap() -> None:
    history = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-02-01T00:00:00Z")],
            "close": [100.0],
        }
    )
    forecasts = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-02-02T00:00:00Z")],
            "predicted_close_p50": [101.0],
        }
    )
    with pytest.raises(ValueError, match="No overlapping timestamps"):
        compute_forecast_mae(symbol="TEST", horizon_hours=1, history_close=history, forecasts=forecasts)


def test_compute_forecast_cache_mae_fills_missing_target_timestamp(tmp_path: Path) -> None:
    # History: 5 hours, close=100..104
    t0 = pd.Timestamp("2026-02-01T00:00:00Z")
    timestamps = [t0 + pd.Timedelta(hours=i) for i in range(5)]
    closes = [100.0 + float(i) for i in range(5)]

    history_csv = tmp_path / "TEST.csv"
    _write_history_csv(history_csv, timestamps, closes)

    # Forecast horizon=3: forecast rows keyed by `timestamp`, evaluating against close at +2h.
    forecast_ts = [timestamps[0], timestamps[1]]  # target closes at t2=102, t3=103
    predicted = [103.0, 101.0]  # abs errors 1 and 2 => MAE 1.5

    forecast_parquet = tmp_path / "TEST.parquet"
    frame = pd.DataFrame(
        {
            "timestamp": [ts.isoformat() for ts in forecast_ts],
            "symbol": "TEST",
            "horizon_hours": 3,
            "predicted_close_p50": predicted,
            # First row has explicit target timestamp, second row is missing (older caches
            # or partial rebuilds can look like this).
            "target_timestamp": [
                (forecast_ts[0] + pd.Timedelta(hours=2)).isoformat(),
                None,
            ],
        }
    )
    frame.to_parquet(forecast_parquet, index=False)

    result = compute_forecast_cache_mae_for_paths(
        symbol="TEST",
        horizon_hours=3,
        history_csv=history_csv,
        forecast_parquet=forecast_parquet,
    )

    assert result.count == 2
    assert result.mae == pytest.approx(1.5, rel=1e-12, abs=1e-12)
