from __future__ import annotations

from pathlib import Path

import pandas as pd

from binanceneural.config import ForecastConfig
from binanceneural.forecasts import ChronosForecastManager


class _DummyBatch:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.quantile_frames = {0.1: frame, 0.5: frame, 0.9: frame}


class _Wrapper:
    def predict_ohlc_batch(
        self,
        contexts: list[pd.DataFrame],
        *,
        symbols: list[str] | None = None,
        prediction_length: int,
        context_length: int | None = None,
        batch_size: int | None = None,
        predict_kwargs: dict | None = None,
    ) -> list[_DummyBatch]:
        del symbols, context_length, batch_size, predict_kwargs
        batches: list[_DummyBatch] = []
        for context in contexts:
            last_ts = pd.to_datetime(context["timestamp"].iloc[-1], utc=True)
            future = [last_ts + pd.Timedelta(hours=i) for i in range(1, int(prediction_length) + 1)]
            frame = pd.DataFrame(
                {
                    "close": [105.0] * int(prediction_length),
                    "high": [106.0] * int(prediction_length),
                    "low": [104.0] * int(prediction_length),
                },
                index=pd.DatetimeIndex(future, name="timestamp"),
            )
            batches.append(_DummyBatch(frame))
        return batches


def _write_history(path: Path, symbol: str = "DOGEUSD", rows: int = 120) -> None:
    timestamps = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC")
    close = pd.Series([100.0 + (idx * 0.15) for idx in range(rows)], dtype="float64")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": symbol,
            "open": close - 0.1,
            "high": close + 0.3,
            "low": close - 0.3,
            "close": close,
            "volume": 1000.0 + pd.Series(range(rows), dtype="float64"),
        }
    )
    frame.to_csv(path, index=False)


def test_forecast_manager_writes_narrative_adjusted_cache(tmp_path: Path) -> None:
    history_path = tmp_path / "DOGEUSD.csv"
    _write_history(history_path)
    cfg = ForecastConfig(
        symbol="DOGEUSD",
        data_root=tmp_path,
        context_hours=24,
        prediction_horizon_hours=1,
        quantile_levels=(0.1, 0.5, 0.9),
        batch_size=8,
        cache_dir=tmp_path / "forecast_cache" / "h1",
        narrative_backend="heuristic",
        narrative_summary_cache_dir=tmp_path / "summary_cache" / "h1",
        narrative_context_hours=24 * 3,
    )
    manager = ChronosForecastManager(cfg, wrapper_factory=lambda: _Wrapper())

    out = manager.ensure_latest(cache_only=False)

    assert not out.empty
    assert "narrative_summary" in out.columns
    assert "base_predicted_close_p50" in out.columns
    assert (tmp_path / "forecast_cache" / "h1" / "DOGEUSD.parquet").exists()
    assert (tmp_path / "summary_cache" / "h1" / "DOGEUSD.parquet").exists()
