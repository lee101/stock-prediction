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
            return types.SimpleNamespace(equity=1.0, cash=1.0, multiplier=1, buying_power=1.0)

        def get_clock(self):
            return types.SimpleNamespace(is_open=True)

        def cancel_orders(self):
            self._orders.clear()
            return []

        def submit_order(self, *args, **kwargs):
            self._orders.append((args, kwargs))
            return types.SimpleNamespace(id=len(self._orders))

    tradeapi_mod.REST = _TestDummyREST

bt3 = importlib.import_module("backtest_test3_inline")

if not hasattr(bt3, "_align_close_actual_window"):
    module_path = Path(__file__).resolve().parents[1] / "backtest_test3_inline.py"
    spec = importlib.util.spec_from_file_location("backtest_test3_inline_actual", module_path)
    if spec is None or spec.loader is None:
        pytest.skip("Unable to load backtest_test3_inline module for maxdiff cache tests", allow_module_level=True)
    bt3 = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = bt3
    spec.loader.exec_module(bt3)

    if not hasattr(bt3, "_align_close_actual_window"):
        pytest.skip("backtest_test3_inline module lacks maxdiff helpers", allow_module_level=True)


def test_align_close_actual_window_trims_to_shortest_sequence():
    close_actual = torch.arange(6, dtype=torch.float32)
    last_preds = {
        "close_actual_movement_values": close_actual.clone(),
        "high_actual_movement_values": torch.arange(4, dtype=torch.float32),
        "low_actual_movement_values": torch.arange(4, dtype=torch.float32) * -1,
        "high_predictions": torch.linspace(0.01, 0.04, 4),
        "low_predictions": torch.linspace(-0.04, -0.01, 4),
    }

    adjusted, window_len = bt3._align_close_actual_window(  # type: ignore[attr-defined]
        last_preds,
        close_actual,
        logger_prefix="Test",
    )

    assert window_len == 4
    assert adjusted.shape[0] == 4
    assert torch.allclose(adjusted, close_actual[-4:])
    assert isinstance(last_preds["close_actual_movement_values"], torch.Tensor)
    assert last_preds["close_actual_movement_values"].shape[0] == 4


def test_evaluate_maxdiff_strategy_handles_short_prediction_horizon(monkeypatch):
    # Chronos2-like scenario: close_actual has 6 rows but prediction tensors provide only 4.
    last_preds = {
        "close_actual_movement_values": torch.tensor([0.01, 0.02, -0.01, 0.015, 0.005, -0.02]),
        "high_actual_movement_values": torch.tensor([0.01, 0.015, 0.02, 0.025]),
        "low_actual_movement_values": torch.tensor([-0.02, -0.015, -0.01, -0.005]),
        "high_predictions": torch.tensor([0.012, 0.018, 0.02, 0.03]),
        "low_predictions": torch.tensor([-0.03, -0.025, -0.02, -0.015]),
        "high_predicted_price_value": 110.0,
        "low_predicted_price_value": 90.0,
    }

    data = {
        "Close": np.linspace(100, 140, 32),
        "High": np.linspace(101, 141, 32),
        "Low": np.linspace(99, 139, 32),
        "Open": np.linspace(100, 140, 32),
    }
    simulation_data = pd.DataFrame(data)

    class DummyOptResult:
        def __init__(self, length: int) -> None:
            self.base_profit = torch.zeros(length, dtype=torch.float32)
            self.final_profit = torch.zeros(length, dtype=torch.float32)
            self.best_high_multiplier = 0.0
            self.best_low_multiplier = 0.0
            self.best_close_at_eod = False
            self.timings = {}

    def fake_optimize(close_actual, *_args, **_kwargs):
        return DummyOptResult(close_actual.shape[0])

    monkeypatch.setattr(bt3, "optimize_maxdiff_entry_exit", fake_optimize)

    evaluation, returns, metadata = bt3.evaluate_maxdiff_strategy(
        last_preds.copy(),
        simulation_data,
        trading_fee=0.001,
        trading_days_per_year=365,
        is_crypto=True,
    )

    assert evaluation.total_return == 0.0
    assert returns.size == 4
    assert metadata["maxdiff_valid_forecasts"] == 4
