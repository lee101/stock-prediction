"""Tests for pctdiff price cache lazy numpy conversion."""

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

tradeapi_mod = sys.modules.setdefault("alpaca_trade_api", types.ModuleType("alpaca_trade_api"))
if not hasattr(tradeapi_mod, "REST"):
    class _TestDummyREST:
        def __init__(self, *args, **kwargs):
            self._orders = []

        def get_all_positions(self):
            return []

        def get_account(self):
            return types.SimpleNamespace(
                equity=1.0,
                cash=1.0,
                multiplier=1,
                buying_power=1.0,
            )

        def get_clock(self):
            return types.SimpleNamespace(is_open=True)

        def cancel_orders(self):
            self._orders.clear()
            return []

        def submit_order(self, *args, **kwargs):
            self._orders.append((args, kwargs))
            return types.SimpleNamespace(id=len(self._orders))

    tradeapi_mod.REST = _TestDummyREST

backtest_module = importlib.import_module("backtest_test3_inline")

if not hasattr(backtest_module, "_prepare_price_window_cache"):
    module_path = Path(__file__).resolve().parents[1] / "backtest_test3_inline.py"

    spec = importlib.util.spec_from_file_location("backtest_test3_inline_actual", module_path)
    if spec is None or spec.loader is None:
        pytest.skip("Unable to load backtest_test3_inline module for pctdiff price cache tests", allow_module_level=True)

    backtest_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = backtest_module
    spec.loader.exec_module(backtest_module)

    if not hasattr(backtest_module, "_prepare_price_window_cache"):
        pytest.skip(
            "backtest_test3_inline module lacks pctdiff helpers; cannot run cache tests",
            allow_module_level=True,
        )

_prepare_price_window_cache = backtest_module._prepare_price_window_cache
_require_price_cache_numpy = backtest_module._require_price_cache_numpy
evaluate_pctdiff_strategy = backtest_module.evaluate_pctdiff_strategy


def test_require_price_cache_numpy_lazy_conversion_cpu():
    validation_len = 4
    rows = validation_len + 2
    data = {
        "High": np.linspace(100, 110, rows),
        "Low": np.linspace(90, 100, rows),
        "Close": np.linspace(95, 105, rows),
    }
    simulation_data = pd.DataFrame(data)

    last_preds = {
        "high_actual_movement_values": torch.zeros(rows),
        "low_actual_movement_values": torch.zeros(rows),
        "high_predictions": torch.zeros(rows),
        "low_predictions": torch.zeros(rows),
    }

    cache = _prepare_price_window_cache(last_preds, simulation_data, validation_len, torch.device("cpu"))
    assert cache is not None
    assert "high_pred_base_np" not in cache

    result = _require_price_cache_numpy(cache, "high_pred_base")
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == validation_len
    assert cache["high_pred_base_np"] is result


def test_pctdiff_zero_metadata_uses_entry_prices():
    last_preds = {
        "close_actual_movement_values": torch.zeros(0, dtype=torch.float32),
        "low_predicted_price_value": 95_000.0,
        "high_predicted_price_value": 105_000.0,
    }
    data = {
        "Close": np.linspace(100, 101, 5),
        "High": np.linspace(101, 102, 5),
        "Low": np.linspace(99, 100, 5),
    }
    simulation_data = pd.DataFrame(data)

    evaluation, returns, metadata = evaluate_pctdiff_strategy(
        last_preds,
        simulation_data,
        trading_fee=0.0,
        trading_days_per_year=365,
        is_crypto=True,
    )

    assert evaluation.total_return == 0.0
    assert returns.size == 0
    assert metadata["pctdiff_entry_low_price"] == pytest.approx(95_000.0)
    assert metadata["pctdiff_entry_high_price"] == pytest.approx(105_000.0)
    assert metadata["pctdiff_takeprofit_high_price"] == pytest.approx(metadata["pctdiff_entry_low_price"])
    assert metadata["pctdiff_takeprofit_low_price"] == pytest.approx(metadata["pctdiff_entry_high_price"])
