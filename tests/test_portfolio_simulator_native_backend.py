from __future__ import annotations

import pandas as pd
import pytest

from unified_hourly_experiment.marketsimulator.portfolio_sim_native import load_portfolio_native_extension
from unified_hourly_experiment.marketsimulator.portfolio_simulator import PortfolioConfig, run_portfolio_simulation


def _require_native() -> None:
    if load_portfolio_native_extension(verbose=False) is None:
        pytest.skip("native portfolio simulator backend unavailable")


def test_native_backend_reflects_realized_exit_in_equity_curve() -> None:
    _require_native()

    t0 = pd.Timestamp("2026-03-03T15:00:00Z")
    t1 = pd.Timestamp("2026-03-03T16:00:00Z")
    bars = pd.DataFrame(
        [
            {"timestamp": t0, "symbol": "NVDA", "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0},
            {"timestamp": t1, "symbol": "NVDA", "open": 100.0, "high": 110.0, "low": 100.0, "close": 100.0},
        ]
    )
    actions = pd.DataFrame(
        [
            {"timestamp": t0, "symbol": "NVDA", "buy_price": 100.0, "sell_price": 110.0, "buy_amount": 100.0, "sell_amount": 0.0, "trade_amount": 100.0},
            {"timestamp": t1, "symbol": "NVDA", "buy_price": 0.0, "sell_price": 0.0, "buy_amount": 0.0, "sell_amount": 0.0, "trade_amount": 0.0},
        ]
    )
    cfg = PortfolioConfig(
        initial_cash=1_000.0,
        max_positions=1,
        max_leverage=2.0,
        trade_amount_scale=100.0,
        fee_by_symbol={"NVDA": 0.0},
        decision_lag_bars=0,
        enforce_market_hours=False,
        close_at_eod=False,
        max_hold_hours=0,
        bar_margin=0.0,
        int_qty=True,
        sim_backend="native",
    )
    result = run_portfolio_simulation(bars, actions, cfg, horizon=1)

    assert result.metrics["final_equity"] == 1_200.0
    assert result.equity_curve.iloc[-1] == 1_200.0


def test_native_backend_matches_python_for_sparse_lagged_signal_alignment() -> None:
    _require_native()

    t0 = pd.Timestamp("2026-03-03T15:00:00Z")
    t1 = pd.Timestamp("2026-03-03T16:00:00Z")
    t2 = pd.Timestamp("2026-03-03T17:00:00Z")
    t3 = pd.Timestamp("2026-03-03T18:00:00Z")
    bars = pd.DataFrame(
        [
            {"timestamp": t0, "symbol": "NVDA", "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0},
            {"timestamp": t1, "symbol": "NVDA", "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0},
            {"timestamp": t2, "symbol": "NVDA", "open": 100.0, "high": 101.0, "low": 100.0, "close": 100.0},
            {"timestamp": t3, "symbol": "NVDA", "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0},
        ]
    )
    actions = pd.DataFrame(
        [
            {"timestamp": t0, "symbol": "NVDA", "buy_price": 100.0, "sell_price": 101.0, "buy_amount": 100.0, "sell_amount": 0.0, "trade_amount": 100.0},
            {"timestamp": t2, "symbol": "NVDA", "buy_price": 100.0, "sell_price": 101.0, "buy_amount": 100.0, "sell_amount": 0.0, "trade_amount": 100.0},
        ]
    )
    base_cfg = dict(
        initial_cash=1_000.0,
        max_positions=1,
        max_leverage=2.0,
        trade_amount_scale=100.0,
        fee_by_symbol={"NVDA": 0.0},
        decision_lag_bars=1,
        enforce_market_hours=False,
        close_at_eod=False,
        max_hold_hours=0,
        bar_margin=0.0,
        int_qty=True,
    )
    python_result = run_portfolio_simulation(
        bars,
        actions,
        PortfolioConfig(**base_cfg, sim_backend="python"),
        horizon=1,
    )
    native_result = run_portfolio_simulation(
        bars,
        actions,
        PortfolioConfig(**base_cfg, sim_backend="native"),
        horizon=1,
    )

    python_entries = [trade.timestamp for trade in python_result.trades if trade.side == "buy"]
    native_entries = [trade.timestamp for trade in native_result.trades if trade.side == "buy"]
    assert python_entries == [t1, t3]
    assert native_entries == python_entries
