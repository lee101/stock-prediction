import types

import pandas as pd
import pytest

from marketsimulator import backtest_test3_inline as ms_backtest


def test_validate_pctdiff_passes_under_limit(monkeypatch):
    dummy = types.SimpleNamespace(
        backtest_forecasts=lambda symbol, num_simulations=None: pd.DataFrame({
            "pctdiff_return": [0.03, -0.05]
        })
    )
    monkeypatch.setattr(ms_backtest, "_REAL_BACKTEST_MODULE", dummy)
    max_abs = ms_backtest.validate_pctdiff("BTCUSD", max_return=0.1)
    assert pytest.approx(max_abs) == 0.05


def test_validate_pctdiff_raises_when_limit_exceeded(monkeypatch):
    dummy = types.SimpleNamespace(
        backtest_forecasts=lambda symbol, num_simulations=None: pd.DataFrame({
            "pctdiff_return": [0.2]
        })
    )
    monkeypatch.setattr(ms_backtest, "_REAL_BACKTEST_MODULE", dummy)
    with pytest.raises(ValueError):
        ms_backtest.validate_pctdiff("BTCUSD", max_return=0.1)
