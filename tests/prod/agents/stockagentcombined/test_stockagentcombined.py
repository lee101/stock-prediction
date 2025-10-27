import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from hyperparamstore.store import HyperparamStore
from stockagentcombined.forecaster import CombinedForecastGenerator


class FakeTotoPipeline:
    def __init__(self, step: float = 1.0):
        self.step = step
        self.calls = 0

    def predict(
        self,
        *,
        context,
        prediction_length,
        num_samples,
        samples_per_batch,
    ):
        self.calls += 1
        value = float(context[-1] + self.step)
        samples = np.full((num_samples, prediction_length), value, dtype=np.float32)
        return [SimpleNamespace(samples=samples)]


class FakeKronosWrapper:
    max_context = 128
    temperature = 0.1
    top_p = 0.9
    top_k = 0
    sample_count = 32

    def __init__(self, increment: float = 4.0):
        self.increment = increment
        self.calls = 0

    def predict_series(
        self,
        *,
        data,
        timestamp_col,
        columns,
        pred_len,
        **_: object,
    ):
        self.calls += 1
        results = {}
        for column in columns:
            series = pd.Series(data[column]).dropna()
            value = float(series.iloc[-1] + self.increment)
            results[column] = SimpleNamespace(absolute=np.array([value], dtype=float))
        return results


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, default=str)


def test_combined_forecast_with_stub_models(tmp_path):
    data_root = tmp_path / "trainingdata"
    hyper_root = tmp_path / "hyperparams"
    data_root.mkdir()

    timestamps = pd.date_range("2024-01-01", periods=6, freq="1D")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": np.linspace(10, 15, 6),
            "high": np.linspace(20, 25, 6),
            "low": np.linspace(5, 10, 6),
            "close": np.linspace(15, 20, 6),
            "volume": np.linspace(1000, 2000, 6),
        }
    )
    frame.to_csv(data_root / "AAPL.csv", index=False)

    toto_payload = {
        "symbol": "AAPL",
        "model": "toto",
        "config": {
            "name": "toto_mean_stub",
            "aggregate": "mean",
            "num_samples": 4,
            "samples_per_batch": 2,
        },
        "validation": {"price_mae": 1.0, "pct_return_mae": 0.1, "latency_s": 9.0},
        "test": {"price_mae": 2.0, "pct_return_mae": 0.2, "latency_s": 9.5},
        "windows": {"forecast_horizon": 1, "val_window": 5, "test_window": 5},
    }
    kronos_payload = {
        "symbol": "AAPL",
        "model": "kronos",
        "config": {
            "name": "kronos_stub",
            "temperature": 0.2,
            "top_p": 0.8,
            "top_k": 16,
            "sample_count": 64,
            "max_context": 256,
            "clip": 1.5,
        },
        "validation": {"price_mae": 2.0, "pct_return_mae": 0.3, "latency_s": 1.5},
        "test": {"price_mae": 3.0, "pct_return_mae": 0.4, "latency_s": 1.7},
        "windows": {"forecast_horizon": 1, "val_window": 5, "test_window": 5},
    }
    best_payload = {
        "symbol": "AAPL",
        "model": "toto",
        "config": toto_payload["config"],
        "validation": toto_payload["validation"],
        "test": toto_payload["test"],
        "windows": toto_payload["windows"],
    }

    _write_json(hyper_root / "toto" / "AAPL.json", toto_payload)
    _write_json(hyper_root / "kronos" / "AAPL.json", kronos_payload)
    _write_json(hyper_root / "best" / "AAPL.json", best_payload)

    fake_toto = FakeTotoPipeline(step=1.0)
    fake_kronos = FakeKronosWrapper(increment=4.0)

    generator = CombinedForecastGenerator(
        data_root=data_root,
        hyperparam_root=hyper_root,
        hyperparam_store=HyperparamStore(hyper_root),
        toto_factory=lambda _: fake_toto,
        kronos_factory=lambda config: fake_kronos,
    )

    result = generator.generate_for_symbol("AAPL")

    # Toto average MAE = 1.5, Kronos average MAE = 2.5 => weights 0.625 / 0.375
    assert pytest.approx(result.weights["toto"], rel=1e-4) == 0.625
    assert pytest.approx(result.weights["kronos"], rel=1e-4) == 0.375

    expected_totals = {
        "open": 0.625 * 16.0 + 0.375 * 19.0,
        "high": 0.625 * 26.0 + 0.375 * 29.0,
        "low": 0.625 * 11.0 + 0.375 * 14.0,
        "close": 0.625 * 21.0 + 0.375 * 24.0,
    }
    for column, expected in expected_totals.items():
        assert pytest.approx(result.combined[column], rel=1e-4) == expected

    assert result.best_model == "toto"
    assert result.selection_source == "hyperparams/best"

    toto_forecast = result.model_forecasts["toto"]
    kronos_forecast = result.model_forecasts["kronos"]
    assert pytest.approx(toto_forecast.average_price_mae, rel=1e-6) == 1.5
    assert pytest.approx(kronos_forecast.average_price_mae, rel=1e-6) == 2.5

    assert fake_toto.calls == len(generator.columns)
    assert fake_kronos.calls == 1


def test_generate_for_symbol_missing_configs(tmp_path):
    data_root = tmp_path / "trainingdata"
    hyper_root = tmp_path / "hyperparams"
    data_root.mkdir()

    timestamps = pd.date_range("2024-01-01", periods=3, freq="1D")
    pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [1.0, 2.0, 3.0],
            "high": [1.5, 2.5, 3.5],
            "low": [0.5, 1.5, 2.5],
            "close": [1.2, 2.2, 3.2],
        }
    ).to_csv(data_root / "MSFT.csv", index=False)

    generator = CombinedForecastGenerator(
        data_root=data_root,
        hyperparam_root=hyper_root,
        hyperparam_store=HyperparamStore(hyper_root),
        toto_factory=lambda _: FakeTotoPipeline(),
        kronos_factory=lambda _: FakeKronosWrapper(),
    )

    with pytest.raises(FileNotFoundError):
        generator.generate_for_symbol("MSFT")


def test_generate_with_historical_override(tmp_path):
    data_root = tmp_path / "trainingdata"
    hyper_root = tmp_path / "hyperparams"
    data_root.mkdir()

    # Write minimal baseline files to satisfy loader (not used because we pass override)
    pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=3), "open": [1, 2, 3], "high": [1, 2, 3], "low": [1, 2, 3], "close": [1, 2, 3]}).to_csv(
        data_root / "AAPL.csv", index=False
    )

    payload = {
        "symbol": "AAPL",
        "model": "toto",
        "config": {
            "name": "toto_stub",
            "aggregate": "mean",
            "num_samples": 4,
            "samples_per_batch": 2,
        },
        "validation": {"price_mae": 1.0, "pct_return_mae": 0.1, "latency_s": 10.0},
        "test": {"price_mae": 2.0, "pct_return_mae": 0.2, "latency_s": 11.0},
        "windows": {"forecast_horizon": 1},
    }
    kronos_payload = {
        "symbol": "AAPL",
        "model": "kronos",
        "config": {"name": "kronos_stub"},
        "validation": {"price_mae": 3.0, "pct_return_mae": 0.3, "latency_s": 1.0},
        "test": {"price_mae": 4.0, "pct_return_mae": 0.4, "latency_s": 1.2},
        "windows": {"forecast_horizon": 1},
    }
    best_payload = {
        "symbol": "AAPL",
        "model": "toto",
        "config": payload["config"],
        "validation": payload["validation"],
        "test": payload["test"],
        "windows": payload["windows"],
    }
    _write_json(hyper_root / "toto" / "AAPL.json", payload)
    _write_json(hyper_root / "kronos" / "AAPL.json", kronos_payload)
    _write_json(hyper_root / "best" / "AAPL.json", best_payload)

    history = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-03-01", periods=5, freq="1D"),
            "open": np.linspace(50, 54, 5),
            "high": np.linspace(55, 59, 5),
            "low": np.linspace(45, 49, 5),
            "close": np.linspace(52, 56, 5),
        }
    )

    fake_toto = FakeTotoPipeline(step=2.0)
    fake_kronos = FakeKronosWrapper(increment=5.0)

    generator = CombinedForecastGenerator(
        data_root=data_root,
        hyperparam_root=hyper_root,
        toto_factory=lambda _: fake_toto,
        kronos_factory=lambda _: fake_kronos,
    )

    result = generator.generate_for_symbol("AAPL", historical_frame=history)

    expected_toto_close = history["close"].iloc[-1] + 2.0
    expected_kronos_close = history["close"].iloc[-1] + 5.0
    toto_forecast = result.model_forecasts["toto"].forecasts["close"]
    kronos_forecast = result.model_forecasts["kronos"].forecasts["close"]

    assert pytest.approx(toto_forecast, rel=1e-6) == expected_toto_close
    assert pytest.approx(kronos_forecast, rel=1e-6) == expected_kronos_close
    assert fake_toto.calls == len(generator.columns)
    assert fake_kronos.calls == 1
