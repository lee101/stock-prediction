from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from binanceexp1.config import DatasetConfig
from binanceexp1.data import BinanceExp1DataModule


def _write_history(path: Path, rows: int = 80) -> None:
    timestamps = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC")
    close = np.linspace(100.0, 179.0, rows)
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close - 1.0,
            "high": close + 1.0,
            "low": close - 2.0,
            "close": close,
            "volume": np.linspace(10.0, 20.0, rows),
            "symbol": "ETHUSD",
        }
    )
    frame.to_csv(path, index=False)


def test_data_module_passes_chronos_context_and_batch_env_overrides(tmp_path: Path, monkeypatch) -> None:
    _write_history(tmp_path / "ETHUSD.csv")
    captured: dict[str, object] = {}

    def _identity_build_feature_frame(frame: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
        work = frame.copy()
        work["reference_close"] = work["close"]
        return work

    def _fake_build_forecast_bundle(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        history = pd.read_csv(tmp_path / "ETHUSD.csv")
        history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True)
        rows = []
        for timestamp, close in zip(history["timestamp"].iloc[24:], history["close"].iloc[24:]):
            rows.append(
                {
                    "timestamp": timestamp,
                    "symbol": "ETHUSD",
                    "predicted_close_p10_h1": float(close * 0.99),
                    "predicted_close_p50_h1": float(close * 1.01),
                    "predicted_close_p90_h1": float(close * 1.02),
                    "predicted_high_p50_h1": float(close * 1.03),
                    "predicted_low_p50_h1": float(close * 0.98),
                }
            )
        return pd.DataFrame(rows)

    monkeypatch.setattr("binanceexp1.data.build_forecast_bundle", _fake_build_forecast_bundle)
    monkeypatch.setattr("binanceexp1.data.build_feature_frame", _identity_build_feature_frame)
    monkeypatch.setenv("CHRONOS2_CONTEXT_HOURS", "1024")
    monkeypatch.setenv("CHRONOS2_BATCH_SIZE", "16")

    BinanceExp1DataModule(
        DatasetConfig(
            symbol="ETHUSD",
            data_root=tmp_path,
            forecast_cache_root=tmp_path / "forecast_cache",
            forecast_horizons=(1,),
            sequence_length=16,
            min_history_hours=24,
            validation_days=0,
            feature_columns=(
                "close",
                "predicted_close_p50_h1",
                "predicted_high_p50_h1",
                "predicted_low_p50_h1",
            ),
        )
    )

    assert captured["context_hours"] == 1024
    assert captured["batch_size"] == 16
