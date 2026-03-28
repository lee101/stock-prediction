import importlib

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def backtest_module():
    return importlib.import_module("backtest_test3_inline")


def test_cpu_fallback_enabled_respects_env(monkeypatch, backtest_module):
    monkeypatch.delenv(backtest_module._GPU_FALLBACK_ENV, raising=False)
    assert backtest_module._cpu_fallback_enabled() is False

    monkeypatch.setenv(backtest_module._GPU_FALLBACK_ENV, "1")
    assert backtest_module._cpu_fallback_enabled() is True

    monkeypatch.setenv(backtest_module._GPU_FALLBACK_ENV, " false ")
    assert backtest_module._cpu_fallback_enabled() is False


def test_require_cuda_raises_without_fallback(monkeypatch, backtest_module):
    monkeypatch.setattr(backtest_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(backtest_module, "_cpu_fallback_log_state", set())
    monkeypatch.delenv(backtest_module._GPU_FALLBACK_ENV, raising=False)

    with pytest.raises(RuntimeError) as excinfo:
        backtest_module._require_cuda("feature", allow_cpu_fallback=False)

    assert "feature" in str(excinfo.value)


def test_require_cuda_logs_once_with_fallback(monkeypatch, backtest_module):
    monkeypatch.setattr(backtest_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(backtest_module, "_cpu_fallback_log_state", set())
    monkeypatch.setenv(backtest_module._GPU_FALLBACK_ENV, "1")

    backtest_module._require_cuda("analytics", symbol="XYZ")
    assert backtest_module._cpu_fallback_log_state == {("analytics", "XYZ")}

    backtest_module._require_cuda("analytics", symbol="XYZ")
    assert backtest_module._cpu_fallback_log_state == {("analytics", "XYZ")}


def test_load_chronos2_wrapper_uses_cpu_fallback(monkeypatch, backtest_module):
    calls = []

    def _fake_loader(cls, **kwargs):
        calls.append(dict(kwargs))
        return object()

    monkeypatch.setattr(backtest_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv(backtest_module._GPU_FALLBACK_ENV, "1")
    monkeypatch.setattr(backtest_module, "_cpu_fallback_log_state", set())
    monkeypatch.setattr(backtest_module, "_chronos2_wrapper_cache", {})
    monkeypatch.setattr(backtest_module.Chronos2OHLCWrapper, "from_pretrained", classmethod(_fake_loader))

    params = {
        "model_id": "stub/chronos2",
        "device_map": "cuda",
        "context_length": 32,
        "batch_size": 8,
        "quantile_levels": (0.1, 0.5, 0.9),
        "symbol": "BTCUSD",
    }

    wrapper = backtest_module.load_chronos2_wrapper(params)

    assert wrapper is not None
    assert len(calls) == 1
    assert calls[0]["device_map"] == "cpu"
    assert ("Chronos2 forecasting", "BTCUSD") in backtest_module._cpu_fallback_log_state


def test_load_chronos2_wrapper_raises_without_cpu_fallback(monkeypatch, backtest_module):
    monkeypatch.setattr(backtest_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.delenv(backtest_module._GPU_FALLBACK_ENV, raising=False)
    monkeypatch.setattr(backtest_module, "_chronos2_wrapper_cache", {})

    params = {
        "model_id": "stub/chronos2",
        "device_map": "cuda",
        "context_length": 32,
        "batch_size": 8,
        "quantile_levels": (0.1, 0.5, 0.9),
        "symbol": "BTCUSD",
    }

    with pytest.raises(RuntimeError, match="CUDA-capable GPU"):
        backtest_module.load_chronos2_wrapper(params)


def test_compute_walk_forward_stats(monkeypatch, backtest_module):
    df = pd.DataFrame(
        {
            "simple_strategy_sharpe": [1.0, 2.0],
            "simple_strategy_return": [0.1, -0.2],
            "highlow_sharpe": [0.5, 0.7],
        }
    )

    stats = backtest_module.compute_walk_forward_stats(df)

    assert stats["walk_forward_oos_sharpe"] == pytest.approx(1.5)
    assert stats["walk_forward_turnover"] == pytest.approx(0.15)
    assert stats["walk_forward_highlow_sharpe"] == pytest.approx(0.6)
    assert "walk_forward_takeprofit_sharpe" not in stats

    empty = backtest_module.compute_walk_forward_stats(pd.DataFrame())
    assert empty == {}


def test_compute_walk_forward_stats_includes_takeprofit(backtest_module):
    df = pd.DataFrame(
        {
            "simple_strategy_sharpe": [0.5, 1.5],
            "simple_strategy_return": [0.2, 0.4],
            "entry_takeprofit_sharpe": [0.3, 0.9],
        }
    )

    stats = backtest_module.compute_walk_forward_stats(df)
    assert stats["walk_forward_takeprofit_sharpe"] == pytest.approx(0.6)


def test_calibrate_signal_defaults_with_short_inputs(backtest_module):
    slope, intercept = backtest_module.calibrate_signal(np.array([1.0]), np.array([2.0]))
    assert slope == pytest.approx(1.0)
    assert intercept == pytest.approx(0.0)


def test_calibrate_signal_fits_linear_relationship(backtest_module):
    preds = np.array([0.0, 1.0, 2.0, 3.0])
    actual = np.array([1.0, 3.0, 5.0, 7.0])

    slope, intercept = backtest_module.calibrate_signal(preds, actual)
    assert slope == pytest.approx(2.0)
    assert intercept == pytest.approx(1.0)
