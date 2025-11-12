"""Regression tests for Chronos2 hyperparam selection/integration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import update_best_configs_by_return_mae as updater
from hyperparamstore.store import HyperparamRecord, HyperparamStore
from src.models.chronos2_wrapper import Chronos2PreparedPanel, Chronos2PredictionBatch
from stockagentcombined.forecaster import CombinedForecastGenerator


class DummyChronos2Wrapper:
    """Lightweight stub that mimics Chronos2OHLCWrapper output."""

    def __init__(self, forecast_value: float = 42.0):
        self.forecast_value = forecast_value

    def predict_ohlc(self, context_df: pd.DataFrame, **kwargs) -> Chronos2PredictionBatch:
        symbol = kwargs.get("symbol", "TEST")
        prediction_length = int(kwargs.get("prediction_length", 1))
        timestamp_column = "timestamp"
        last_timestamp = pd.to_datetime(context_df[timestamp_column].iloc[-1])
        horizon_index = pd.date_range(last_timestamp, periods=prediction_length + 1, freq="D", tz="UTC")[1:]

        median = pd.DataFrame(
            {"close": [self.forecast_value] * prediction_length},
            index=pd.DatetimeIndex(horizon_index, name=timestamp_column),
        )

        context_payload = context_df[["symbol", timestamp_column, "close"]].copy()
        actual_df = pd.DataFrame(columns=["close"])
        actual_df.index = pd.DatetimeIndex([], name=timestamp_column)

        panel = Chronos2PreparedPanel(
            symbol=symbol,
            context_df=context_payload,
            future_df=None,
            actual_df=actual_df,
            context_length=len(context_payload),
            prediction_length=prediction_length,
            id_column="symbol",
            timestamp_column=timestamp_column,
            target_columns=("close",),
        )
        raw = median.reset_index().assign(target_name="close")
        return Chronos2PredictionBatch(panel=panel, raw_dataframe=raw, quantile_frames={0.5: median})


def _write_record(
    store: HyperparamStore,
    model: str,
    symbol: str,
    *,
    price_mae: float,
    pct_return_mae: float,
    latency: float,
) -> None:
    record = HyperparamRecord(
        config={"name": f"{model}-config"},
        validation={"price_mae": price_mae, "pct_return_mae": pct_return_mae, "latency_s": latency},
        test={"price_mae": price_mae, "pct_return_mae": pct_return_mae, "latency_s": latency},
    )
    store.save(model, symbol, record, windows={"val_window": 1, "test_window": 1, "forecast_horizon": 1})


def test_update_model_selection_prefers_chronos2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hyper_root = tmp_path / "hyperparams"
    store = HyperparamStore(hyper_root)
    _write_record(store, "toto", "TEST", price_mae=10.0, pct_return_mae=0.04, latency=1.0)
    _write_record(store, "chronos2", "TEST", price_mae=5.0, pct_return_mae=0.02, latency=0.2)

    monkeypatch.setenv("HYPERPARAM_ROOT", str(hyper_root))
    updater.HYPERPARAM_ROOT = hyper_root
    updater.MODEL_DIRS = {
        "kronos": hyper_root / "kronos",
        "toto": hyper_root / "toto",
        "chronos2": hyper_root / "chronos2",
    }
    selection = updater.update_model_selection("TEST")
    assert selection is not None
    assert selection["model"] == "chronos2"
    metadata = selection["metadata"]
    assert pytest.approx(metadata["chronos2_pct_return_mae"], rel=1e-6) == 0.02
    assert metadata["candidate_pct_return_mae"]["chronos2"] < metadata["candidate_pct_return_mae"]["toto"]


def test_combined_forecast_generator_uses_chronos2_when_best(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hyper_root = tmp_path / "hyperparams"
    data_root = tmp_path / "trainingdata"
    data_root.mkdir(parents=True, exist_ok=True)
    store = HyperparamStore(hyper_root)

    _write_record(store, "chronos2", "TEST", price_mae=4.0, pct_return_mae=0.01, latency=0.1)

    selection_payload = {
        "symbol": "TEST",
        "model": "chronos2",
        "config": {"name": "chronos2-config", "context_length": 8, "prediction_length": 1},
        "validation": {"price_mae": 4.0, "pct_return_mae": 0.01, "latency_s": 0.1},
        "test": {"price_mae": 6.0, "pct_return_mae": 0.02, "latency_s": 0.1},
        "windows": {"val_window": 1, "test_window": 1, "forecast_horizon": 1},
        "config_path": "hyperparams/chronos2/TEST.json",
    }
    store.save_selection("TEST", selection_payload)

    # Minimal training data
    dates = pd.date_range("2024-01-01", periods=32, freq="D", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": 100 + np.arange(32, dtype=float),
            "high": 101 + np.arange(32, dtype=float),
            "low": 99 + np.arange(32, dtype=float),
            "close": 100 + np.arange(32, dtype=float),
            "symbol": ["TEST"] * 32,
        }
    )
    df.to_csv(data_root / "TEST.csv", index=False)

    monkeypatch.setenv("HYPERPARAM_ROOT", str(hyper_root))
    updater.HYPERPARAM_ROOT = hyper_root
    updater.MODEL_DIRS = {
        "kronos": hyper_root / "kronos",
        "toto": hyper_root / "toto",
        "chronos2": hyper_root / "chronos2",
    }

    generator = CombinedForecastGenerator(
        data_root=data_root,
        hyperparam_root=hyper_root,
        prediction_columns=("close",),
        chronos2_factory=lambda config: DummyChronos2Wrapper(forecast_value=1337.0),
    )

    forecast = generator.generate_for_symbol("TEST", prediction_length=1)
    assert forecast.best_model == "chronos2"
    assert pytest.approx(forecast.combined["close"], rel=1e-6) == 1337.0
    assert "chronos2" in forecast.model_forecasts
