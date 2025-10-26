import importlib
import os
import sys
import types

import numpy as np

os.environ.setdefault('TESTING', 'True')

tradeapi_mod = sys.modules.setdefault("alpaca_trade_api", types.ModuleType("alpaca_trade_api"))
tradeapi_rest = sys.modules.setdefault(
    "alpaca_trade_api.rest", types.ModuleType("alpaca_trade_api.rest")
)

if not hasattr(tradeapi_rest, "APIError"):
    class _APIError(Exception):
        pass

    tradeapi_rest.APIError = _APIError  # type: ignore[attr-defined]


if not hasattr(tradeapi_mod, "REST"):
    class _DummyREST:
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

    tradeapi_mod.REST = _DummyREST  # type: ignore[attr-defined]


import backtest_test3_inline as backtest_module

if not hasattr(backtest_module, "calibrate_signal"):
    backtest_module = importlib.reload(backtest_module)

calibrate_signal = backtest_module.calibrate_signal


def test_eth_calibration_small_delta_stability():
    """Regression test: tiny normalized ETH deltas should not explode after calibration."""
    predictions = np.array([-0.010, 0.000, 0.003, 0.006, 0.0025], dtype=float)
    actual_returns = np.array([-0.008, 0.001, 0.002, 0.005, 0.0018], dtype=float)
    slope, intercept = calibrate_signal(predictions, actual_returns)
    raw_delta = 0.005098  # ~0.51% normalized signal from ETH incident
    calibrated_delta = slope * raw_delta + intercept
    assert abs(calibrated_delta) < 0.02, (
        "Calibrated ETH move deviated more than 2%, indicating scaler instability"
    )
