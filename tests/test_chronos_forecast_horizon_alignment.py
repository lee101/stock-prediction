from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from binanceneural.config import ForecastConfig
from binanceneural.forecasts import ChronosForecastManager


@dataclass
class _DummyBatch:
    quantile_frames: dict[float, pd.DataFrame]


class _DummyWrapper:
    def predict_ohlc_batch(
        self,
        contexts: list[pd.DataFrame],
        *,
        symbols: list[str] | None = None,
        prediction_length: int,
        context_length: int | None = None,
        batch_size: int | None = None,
        predict_kwargs: dict | None = None,
        **_: object,
    ) -> list[_DummyBatch]:
        batches: list[_DummyBatch] = []
        for ctx in contexts:
            last_ts = pd.to_datetime(ctx["timestamp"].iloc[-1], utc=True)
            future = [last_ts + pd.Timedelta(hours=i) for i in range(1, int(prediction_length) + 1)]
            steps = list(range(1, int(prediction_length) + 1))
            frame = pd.DataFrame(
                {
                    "close": [float(s) for s in steps],
                    "high": [float(s) + 0.5 for s in steps],
                    "low": [float(s) - 0.5 for s in steps],
                },
                index=pd.DatetimeIndex(future, name="timestamp"),
            )
            batches.append(_DummyBatch(quantile_frames={0.1: frame, 0.5: frame, 0.9: frame}))
        return batches


def _write_hourly_csv(path: Path, symbol: str, rows: int) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")
    close = pd.Series(range(rows), dtype="float32") + 100.0
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1.0,
            "symbol": symbol,
        }
    )
    frame.to_csv(path, index=False)
    return frame


def test_forecast_horizon_extracts_future_step(tmp_path: Path) -> None:
    symbol = "TESTUSD"
    data_root = tmp_path
    history = _write_hourly_csv(data_root / f"{symbol}.csv", symbol, rows=60)

    # With context_hours=16, the first forecast row will be at history[16].timestamp.
    target_ts = pd.to_datetime(history["timestamp"].iloc[16], utc=True)
    end_ts = target_ts

    cfg = ForecastConfig(
        symbol=symbol,
        data_root=data_root,
        context_hours=16,
        prediction_horizon_hours=24,
        quantile_levels=(0.1, 0.5, 0.9),
        batch_size=4,
        cache_dir=tmp_path / "cache",
    )
    manager = ChronosForecastManager(cfg, wrapper_factory=lambda: _DummyWrapper())
    out = manager.ensure_latest(end=end_ts, cache_only=False, force_rebuild=True)

    row = out.loc[out["timestamp"] == target_ts].iloc[0]
    assert int(row["horizon_hours"]) == 24
    assert pd.to_datetime(row["target_timestamp"], utc=True) == target_ts + pd.Timedelta(hours=23)
    # Dummy wrapper sets close=step index, so horizon=24 should select the 24th step.
    assert float(row["predicted_close_p50"]) == 24.0
    assert float(row["predicted_high_p50"]) == 24.5
    assert float(row["predicted_low_p50"]) == 23.5

