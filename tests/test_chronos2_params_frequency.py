from typing import Any, List, Tuple

import pytest

from hyperparamstore.store import HyperparamRecord
import src.chronos2_params as chronos_params


def _make_record(context_length: int, name: str = "ctx") -> HyperparamRecord:
    return HyperparamRecord(
        config={
            "name": name,
            "model_id": "amazon/chronos-2",
            "device_map": "cuda",
            "context_length": context_length,
            "batch_size": 32,
            "quantile_levels": [0.1, 0.5, 0.9],
            "aggregation": "median",
            "sample_count": 0,
            "scaler": "none",
            "predict_kwargs": {},
        },
        validation={"price_mae": 1.0, "pct_return_mae": 0.1, "latency_s": 1.0},
        test={"price_mae": 1.2, "pct_return_mae": 0.12, "latency_s": 1.1},
        metadata={"source": "unit-test"},
    )


@pytest.fixture()
def params_module():
    chronos_params._chronos2_params_cache.clear()  # type: ignore[attr-defined]
    yield chronos_params
    chronos_params._chronos2_params_cache.clear()  # type: ignore[attr-defined]


def test_resolve_chronos2_params_prefers_hourly_variant(monkeypatch, params_module):
    calls: List[Tuple[str, str]] = []
    hourly_record = _make_record(8192, name="hourly")
    daily_record = _make_record(1024, name="daily")

    def fake_load_best_config(model: str, symbol: str, store: Any = None):
        calls.append((model, symbol))
        if model == "hourly":
            return hourly_record
        if model == "chronos2":
            return daily_record
        return None

    class DummyStore:
        def __init__(self, root):
            self.root = root

    monkeypatch.setattr(params_module, "load_best_config", fake_load_best_config)
    monkeypatch.setattr(params_module, "HyperparamStore", DummyStore)

    params = params_module.resolve_chronos2_params("AAPL", frequency="hourly")

    assert params["context_length"] == 8192
    assert params["_config_name"] == "hourly"
    assert ("hourly", "AAPL") in calls
    assert ("chronos2", "AAPL") in calls  # daily lookup still happens first


def test_resolve_chronos2_params_falls_back_when_variant_missing(monkeypatch, params_module):
    daily_record = _make_record(1536, name="fallback")

    def fake_load_best_config(model: str, symbol: str, store: Any = None):
        if model == "chronos2":
            return daily_record
        return None

    class DummyStore:
        def __init__(self, root):
            self.root = root

    monkeypatch.setattr(params_module, "load_best_config", fake_load_best_config)
    monkeypatch.setattr(params_module, "HyperparamStore", DummyStore)

    params = params_module.resolve_chronos2_params("MSFT", frequency="hourly")
    assert params["context_length"] == 1536
    assert params["_config_name"] == "fallback"


def test_frequency_cache_isolated(monkeypatch, params_module):
    hourly_record = _make_record(7000, name="hourly")
    daily_record = _make_record(1200, name="daily")

    def fake_load_best_config(model: str, symbol: str, store: Any = None):
        if model == "hourly":
            return hourly_record
        return daily_record

    class DummyStore:
        def __init__(self, root):
            self.root = root

    monkeypatch.setattr(params_module, "load_best_config", fake_load_best_config)
    monkeypatch.setattr(params_module, "HyperparamStore", DummyStore)

    hourly_params = params_module.resolve_chronos2_params("ETHUSD", frequency="hourly")
    daily_params = params_module.resolve_chronos2_params("ETHUSD")

    assert hourly_params["context_length"] == 7000
    assert daily_params["context_length"] == 1200
