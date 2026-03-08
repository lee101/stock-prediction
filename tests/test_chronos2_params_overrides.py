from __future__ import annotations

from typing import Any

import pytest

import src.chronos2_params as chronos_params


@pytest.fixture()
def params_module():
    chronos_params._chronos2_params_cache.clear()  # type: ignore[attr-defined]
    yield chronos_params
    chronos_params._chronos2_params_cache.clear()  # type: ignore[attr-defined]


def _fake_load_best_config(model: str, symbol: str, store: Any = None):  # type: ignore[no-untyped-def]
    del model, symbol, store
    return None


def test_env_overrides_can_force_context_batch_and_multiscale(monkeypatch, params_module) -> None:
    monkeypatch.setattr(params_module, "load_best_config", _fake_load_best_config)
    monkeypatch.setenv("CHRONOS2_CONTEXT_LENGTH", "1024")
    monkeypatch.setenv("CHRONOS2_BATCH_SIZE", "16")
    monkeypatch.setenv("CHRONOS2_SKIP_RATES", "1,2,4")
    monkeypatch.setenv("CHRONOS2_AGGREGATION_METHOD", "weighted")
    monkeypatch.setenv("CHRONOS2_FORCE_MULTISCALE", "1")

    params = params_module.resolve_chronos2_params("ETHUSD", frequency="hourly")

    assert params["context_length"] == 1024
    assert params["batch_size"] == 16
    assert params["skip_rates"] == (1, 2, 4)
    assert params["aggregation_method"] == "weighted"
    assert params["multiscale_method"] == "weighted"
    assert params["use_multiscale"] is True


def test_env_overrides_can_force_multivariate_and_disable_cross_learning(monkeypatch, params_module) -> None:
    monkeypatch.setattr(params_module, "load_best_config", _fake_load_best_config)
    monkeypatch.setenv("CHRONOS2_FORCE_MULTIVARIATE", "1")
    monkeypatch.setenv("CHRONOS2_FORCE_CROSS_LEARNING", "0")

    params = params_module.resolve_chronos2_params("BTCUSD", frequency="hourly")

    assert params["use_multivariate"] is True
    assert params["use_cross_learning"] is False
    assert "predict_batches_jointly" not in params["predict_kwargs"]
