from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from unified_hourly_experiment.marketsimulator.portfolio_sim_native import load_portfolio_native_extension
from unified_hourly_experiment.marketsimulator.portfolio_simulator import (
    PortfolioConfig,
    run_portfolio_simulation,
)


@pytest.mark.slow
def test_native_backend_matches_python_backend_on_directional_portfolio_case():
    if load_portfolio_native_extension(verbose=False) is None:
        pytest.skip("native portfolio simulator extension unavailable")

    t0 = pd.Timestamp("2026-03-03T15:00:00Z")
    t1 = pd.Timestamp("2026-03-03T16:00:00Z")
    bars = pd.DataFrame(
        [
            {"timestamp": t0, "symbol": "NVDA", "open": 100.0, "high": 105.0, "low": 99.0, "close": 101.0},
            {"timestamp": t0, "symbol": "MTCH", "open": 30.0, "high": 31.0, "low": 29.0, "close": 30.1},
            {"timestamp": t1, "symbol": "NVDA", "open": 101.5, "high": 104.6, "low": 100.1, "close": 103.9},
            {"timestamp": t1, "symbol": "MTCH", "open": 30.2, "high": 30.4, "low": 28.7, "close": 29.0},
        ]
    )
    actions = pd.DataFrame(
        [
            {
                "timestamp": t0,
                "symbol": "NVDA",
                "buy_price": 100.0,
                "sell_price": 104.0,
                "buy_amount": 50.0,
                "sell_amount": 0.0,
                "trade_amount": 50.0,
                "predicted_high_p50_h1": 104.5,
                "predicted_low_p50_h1": 99.0,
                "predicted_close_p50_h1": 103.0,
            },
            {
                "timestamp": t0,
                "symbol": "MTCH",
                "buy_price": 29.0,
                "sell_price": 30.0,
                "buy_amount": 0.0,
                "sell_amount": 60.0,
                "trade_amount": 60.0,
                "predicted_high_p50_h1": 30.3,
                "predicted_low_p50_h1": 29.1,
                "predicted_close_p50_h1": 29.4,
            },
            {
                "timestamp": t1,
                "symbol": "NVDA",
                "buy_price": 100.0,
                "sell_price": 104.0,
                "buy_amount": 50.0,
                "sell_amount": 0.0,
                "trade_amount": 50.0,
                "predicted_high_p50_h1": 104.5,
                "predicted_low_p50_h1": 99.0,
                "predicted_close_p50_h1": 103.0,
            },
            {
                "timestamp": t1,
                "symbol": "MTCH",
                "buy_price": 29.0,
                "sell_price": 30.0,
                "buy_amount": 0.0,
                "sell_amount": 60.0,
                "trade_amount": 60.0,
                "predicted_high_p50_h1": 30.3,
                "predicted_low_p50_h1": 29.1,
                "predicted_close_p50_h1": 29.4,
            },
        ]
    )

    base_cfg = PortfolioConfig(
        initial_cash=10_000.0,
        max_positions=2,
        min_edge=0.0,
        max_hold_hours=100,
        enforce_market_hours=False,
        close_at_eod=False,
        symbols=["NVDA", "MTCH"],
        trade_amount_scale=100.0,
        decision_lag_bars=0,
        market_order_entry=False,
        bar_margin=0.0,
        int_qty=True,
        fee_by_symbol={"NVDA": 0.001, "MTCH": 0.001},
        sim_backend="python",
    )

    py_result = run_portfolio_simulation(bars, actions, base_cfg, horizon=1)
    native_result = run_portfolio_simulation(
        bars,
        actions,
        PortfolioConfig(**{**base_cfg.__dict__, "sim_backend": "native"}),
        horizon=1,
    )

    np.testing.assert_allclose(
        py_result.equity_curve.to_numpy(dtype=float),
        native_result.equity_curve.to_numpy(dtype=float),
        rtol=1e-10,
        atol=1e-10,
    )
    assert py_result.equity_curve.index.equals(native_result.equity_curve.index)

    assert len(py_result.trades) == len(native_result.trades)
    for lhs, rhs in zip(py_result.trades, native_result.trades):
        assert lhs.timestamp == rhs.timestamp
        assert lhs.symbol == rhs.symbol
        assert lhs.side == rhs.side
        assert lhs.reason == rhs.reason
        assert math.isclose(lhs.price, rhs.price, rel_tol=1e-12, abs_tol=1e-12)
        assert math.isclose(lhs.quantity, rhs.quantity, rel_tol=1e-12, abs_tol=1e-12)
        assert math.isclose(lhs.cash_after, rhs.cash_after, rel_tol=1e-12, abs_tol=1e-12)
        assert math.isclose(lhs.inventory_after, rhs.inventory_after, rel_tol=1e-12, abs_tol=1e-12)

    for key in py_result.metrics:
        assert key in native_result.metrics
        assert math.isclose(
            float(py_result.metrics[key]),
            float(native_result.metrics[key]),
            rel_tol=1e-10,
            abs_tol=1e-10,
        )
