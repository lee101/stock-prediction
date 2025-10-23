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
