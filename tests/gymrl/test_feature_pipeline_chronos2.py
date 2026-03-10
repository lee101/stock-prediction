from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gymrl.config import FeatureBuilderConfig
from gymrl.feature_pipeline import FeatureBuilder


def _write_eth_history(path: Path, rows: int = 40) -> None:
    ts = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC")
    close = np.linspace(100.0, 139.0, rows)
    frame = pd.DataFrame(
        {
            "timestamp": ts,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.linspace(10.0, 20.0, rows),
        }
    )
    frame.to_csv(path, index=False)


def test_feature_builder_chronos2_backend_adds_horizon_snapshot_features(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    _write_eth_history(data_dir / "ETHUSD.csv")

    def _fake_build_forecast_bundle(**kwargs):  # type: ignore[no-untyped-def]
        assert kwargs["symbol"] == "ETHUSD"
        assert list(kwargs["horizons"]) == [1, 6]
        assert kwargs["force_multivariate"] is True
        assert kwargs["force_cross_learning"] is True
        history = pd.read_csv(data_dir / "ETHUSD.csv")
        history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True)
        rows = []
        for timestamp, close in zip(history["timestamp"].iloc[8:], history["close"].iloc[8:]):
            rows.append(
                {
                    "timestamp": timestamp,
                    "symbol": "ETHUSD",
                    "predicted_close_p10_h1": float(close * 0.99),
                    "predicted_close_p50_h1": float(close * 1.01),
                    "predicted_close_p90_h1": float(close * 1.03),
                    "predicted_high_p50_h1": float(close * 1.04),
                    "predicted_low_p50_h1": float(close * 0.98),
                    "predicted_close_p10_h6": float(close * 0.97),
                    "predicted_close_p50_h6": float(close * 1.05),
                    "predicted_close_p90_h6": float(close * 1.10),
                    "predicted_high_p50_h6": float(close * 1.12),
                    "predicted_low_p50_h6": float(close * 0.95),
                }
            )
        return pd.DataFrame(rows)

    monkeypatch.setattr("binanceneural.forecasts.build_forecast_bundle", _fake_build_forecast_bundle)

    builder = FeatureBuilder(
        config=FeatureBuilderConfig(
            forecast_backend="chronos2",
            context_window=8,
            min_history=8,
            prediction_length=1,
            realized_horizon=1,
            num_samples=32,
            enforce_common_index=True,
        ),
        backend_kwargs={
            "chronos2_horizons": "1,6",
            "chronos2_cache_only": True,
            "chronos2_context_hours": 32,
            "chronos2_force_multivariate": True,
            "chronos2_force_cross_learning": True,
        },
    )
    cube = builder.build_from_directory(data_dir, symbols=["ETHUSD"])

    assert builder.backend_name == "chronos2"
    assert "chronos2_return_h1" in cube.feature_names
    assert "chronos2_return_h6" in cube.feature_names
    assert "chronos2_spread_h6" in cube.feature_names
    assert cube.features.shape[1] == 1
    assert cube.features.shape[0] > 0
    assert np.isfinite(cube.features).all()
