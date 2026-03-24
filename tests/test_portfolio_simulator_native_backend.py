from __future__ import annotations

import pandas as pd
import pytest

import unified_hourly_experiment.marketsimulator.portfolio_simulator as portfolio_simulator
from unified_hourly_experiment.marketsimulator.portfolio_sim_native import load_portfolio_native_extension
from unified_hourly_experiment.marketsimulator.portfolio_simulator import (
    PortfolioConfig,
    PortfolioResult,
    run_portfolio_simulation,
)


def _require_native() -> None:
    if load_portfolio_native_extension(verbose=False) is None:
        pytest.skip("native portfolio simulator backend unavailable")


def test_auto_backend_prefers_native_when_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    t0 = pd.Timestamp("2026-03-03T15:00:00Z")
    bars = pd.DataFrame(
        [
            {"timestamp": t0, "symbol": "NVDA", "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0},
        ]
    )
    actions = pd.DataFrame(
        [
            {"timestamp": t0, "symbol": "NVDA", "buy_price": 100.0, "sell_price": 101.0, "buy_amount": 100.0, "sell_amount": 0.0, "trade_amount": 100.0},
        ]
    )
    sentinel = PortfolioResult(equity_curve=pd.Series(dtype=float), trades=[], metrics={"fast_path": 1.0})

    def _fake_native(**kwargs):
        del kwargs
        return sentinel

    monkeypatch.setattr(portfolio_simulator, "_run_portfolio_simulation_native", _fake_native)

    result = run_portfolio_simulation(bars, actions, PortfolioConfig(), horizon=1)

    assert result is sentinel


def test_auto_backend_falls_back_to_python_for_pending_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    t0 = pd.Timestamp("2026-03-03T15:00:00Z")
    t1 = pd.Timestamp("2026-03-03T16:00:00Z")
    bars = pd.DataFrame(
        [
            {"timestamp": t0, "symbol": "NVDA", "open": 100.0, "high": 100.2, "low": 99.95, "close": 100.0},
            {"timestamp": t1, "symbol": "NVDA", "open": 100.0, "high": 100.2, "low": 99.75, "close": 100.0},
        ]
    )
    actions = pd.DataFrame(
        [
            {"timestamp": t0, "symbol": "NVDA", "buy_price": 99.9, "sell_price": 101.0, "buy_amount": 100.0, "sell_amount": 0.0, "trade_amount": 100.0},
            {"timestamp": t1, "symbol": "NVDA", "buy_price": 99.9, "sell_price": 101.0, "buy_amount": 0.0, "sell_amount": 0.0, "trade_amount": 0.0},
        ]
    )

    def _should_not_run_native(**kwargs):
        del kwargs
        raise AssertionError("native backend should be skipped when pending-entry TTL is enabled")

    monkeypatch.setattr(portfolio_simulator, "_run_portfolio_simulation_native", _should_not_run_native)

    result = run_portfolio_simulation(
        bars,
        actions,
        PortfolioConfig(
            initial_cash=1_000.0,
            max_positions=1,
            max_leverage=1.0,
            trade_amount_scale=100.0,
            fee_by_symbol={"NVDA": 0.0},
            decision_lag_bars=0,
            enforce_market_hours=False,
            close_at_eod=False,
            max_hold_hours=0,
            bar_margin=0.001,
            int_qty=True,
            entry_order_ttl_hours=1,
            sim_backend="auto",
        ),
        horizon=1,
    )

    buys = [trade for trade in result.trades if trade.side == "buy"]
    assert [trade.timestamp for trade in buys] == [t1]


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


def test_native_backend_matches_python_for_first_trigger_entry_selection() -> None:
    _require_native()

    t0 = pd.Timestamp("2026-03-03T15:00:00Z")
    bars = pd.DataFrame(
        [
            {"timestamp": t0, "symbol": "NVDA", "open": 100.0, "high": 101.5, "low": 94.0, "close": 100.0},
            {"timestamp": t0, "symbol": "GOOG", "open": 100.0, "high": 101.0, "low": 98.8, "close": 100.0},
        ]
    )
    actions = pd.DataFrame(
        [
            {
                "timestamp": t0,
                "symbol": "NVDA",
                "buy_price": 95.0,
                "sell_price": 110.0,
                "buy_amount": 100.0,
                "sell_amount": 0.0,
                "trade_amount": 100.0,
                "predicted_high_p50_h1": 110.0,
                "predicted_low_p50_h1": 90.0,
                "predicted_close_p50_h1": 108.0,
            },
            {
                "timestamp": t0,
                "symbol": "GOOG",
                "buy_price": 99.0,
                "sell_price": 103.0,
                "buy_amount": 100.0,
                "sell_amount": 0.0,
                "trade_amount": 100.0,
                "predicted_high_p50_h1": 103.0,
                "predicted_low_p50_h1": 97.0,
                "predicted_close_p50_h1": 102.0,
            },
        ]
    )
    cfg = dict(
        initial_cash=1_000.0,
        max_positions=1,
        max_leverage=1.0,
        trade_amount_scale=100.0,
        fee_by_symbol={"NVDA": 0.0, "GOOG": 0.0},
        decision_lag_bars=0,
        enforce_market_hours=False,
        close_at_eod=False,
        max_hold_hours=0,
        bar_margin=0.0,
        int_qty=True,
        entry_selection_mode="first_trigger",
    )

    python_result = run_portfolio_simulation(
        bars,
        actions,
        PortfolioConfig(**cfg, sim_backend="python"),
        horizon=1,
    )
    native_result = run_portfolio_simulation(
        bars,
        actions,
        PortfolioConfig(**cfg, sim_backend="native"),
        horizon=1,
    )

    python_entries = [(trade.symbol, trade.timestamp) for trade in python_result.trades if trade.side == "buy"]
    native_entries = [(trade.symbol, trade.timestamp) for trade in native_result.trades if trade.side == "buy"]
    assert python_entries == [("GOOG", t0)]
    assert native_entries == python_entries


def test_native_backend_matches_python_for_short_timeout_exit() -> None:
    _require_native()

    t0 = pd.Timestamp("2026-03-03T15:00:00Z")
    t1 = pd.Timestamp("2026-03-03T16:00:00Z")
    t2 = pd.Timestamp("2026-03-03T17:00:00Z")
    bars = pd.DataFrame(
        [
            {"timestamp": t0, "symbol": "MTCH", "open": 30.0, "high": 30.0, "low": 29.8, "close": 30.0},
            {"timestamp": t1, "symbol": "MTCH", "open": 30.0, "high": 30.0, "low": 29.6, "close": 30.0},
            {"timestamp": t2, "symbol": "MTCH", "open": 30.0, "high": 30.0, "low": 29.6, "close": 30.0},
        ]
    )
    actions = pd.DataFrame(
        [
            {"timestamp": t0, "symbol": "MTCH", "buy_price": 29.0, "sell_price": 30.0, "buy_amount": 0.0, "sell_amount": 100.0, "trade_amount": 100.0},
            {"timestamp": t1, "symbol": "MTCH", "buy_price": 29.0, "sell_price": 30.0, "buy_amount": 0.0, "sell_amount": 0.0, "trade_amount": 0.0},
            {"timestamp": t2, "symbol": "MTCH", "buy_price": 29.0, "sell_price": 30.0, "buy_amount": 0.0, "sell_amount": 0.0, "trade_amount": 0.0},
        ]
    )
    cfg = dict(
        initial_cash=1_000.0,
        max_positions=1,
        max_leverage=2.0,
        trade_amount_scale=100.0,
        fee_by_symbol={"MTCH": 0.0},
        decision_lag_bars=0,
        enforce_market_hours=False,
        close_at_eod=False,
        max_hold_hours=1,
        bar_margin=0.0,
        int_qty=True,
    )

    python_result = run_portfolio_simulation(
        bars,
        actions,
        PortfolioConfig(**cfg, sim_backend="python"),
        horizon=1,
    )
    native_result = run_portfolio_simulation(
        bars,
        actions,
        PortfolioConfig(**cfg, sim_backend="native"),
        horizon=1,
    )

    python_trace = [(trade.side, trade.reason, trade.timestamp) for trade in python_result.trades]
    native_trace = [(trade.side, trade.reason, trade.timestamp) for trade in native_result.trades]

    assert python_trace == [
        ("short_sell", "entry", t0),
        ("buy_cover", "timeout", t1),
    ]
    assert native_trace == python_trace
    assert native_result.metrics["final_equity"] == pytest.approx(python_result.metrics["final_equity"])


def test_native_backend_matches_python_for_drawdown_profit_early_exit() -> None:
    _require_native()

    index = pd.date_range("2026-03-03T15:00:00Z", periods=30, freq="h")
    prices = [100.0 + i * 2.0 for i in range(10)] + [118.0 - i * 3.0 for i in range(20)]
    bars = pd.DataFrame(
        [
            {
                "timestamp": ts,
                "symbol": "NVDA",
                "open": price,
                "high": price * 1.001,
                "low": price * 0.999,
                "close": price,
            }
            for ts, price in zip(index, prices)
        ]
    )
    actions = pd.DataFrame(
        [
            {
                "timestamp": ts,
                "symbol": "NVDA",
                "buy_price": price,
                "sell_price": price * 10.0,
                "buy_amount": 100.0,
                "sell_amount": 0.0,
                "trade_amount": 100.0,
            }
            for ts, price in zip(index, prices)
        ]
    )
    cfg = dict(
        initial_cash=1_000.0,
        max_positions=1,
        max_leverage=1.0,
        trade_amount_scale=100.0,
        fee_by_symbol={"NVDA": 0.0},
        decision_lag_bars=0,
        enforce_market_hours=False,
        close_at_eod=False,
        max_hold_hours=0,
        bar_margin=0.0,
        int_qty=True,
    )

    python_result = run_portfolio_simulation(
        bars,
        actions,
        PortfolioConfig(**cfg, sim_backend="python"),
        horizon=1,
    )
    native_result = run_portfolio_simulation(
        bars,
        actions,
        PortfolioConfig(**cfg, sim_backend="native"),
        horizon=1,
    )

    assert len(python_result.equity_curve) < len(index)
    assert len(native_result.equity_curve) == len(python_result.equity_curve)
    assert native_result.metrics["final_equity"] == pytest.approx(python_result.metrics["final_equity"])
    assert native_result.metrics["max_drawdown"] == pytest.approx(python_result.metrics["max_drawdown"])


def test_native_backend_matches_python_for_concentrated_entry_allocator() -> None:
    _require_native()

    t0 = pd.Timestamp("2026-03-03T15:00:00Z")
    bars = pd.DataFrame(
        [
            {"timestamp": t0, "symbol": "AAA", "open": 100.0, "high": 110.0, "low": 99.0, "close": 100.0},
            {"timestamp": t0, "symbol": "BBB", "open": 100.0, "high": 108.0, "low": 99.0, "close": 100.0},
        ]
    )
    actions = pd.DataFrame(
        [
            {"timestamp": t0, "symbol": "AAA", "buy_price": 100.0, "sell_price": 110.0, "buy_amount": 100.0, "sell_amount": 0.0, "trade_amount": 100.0},
            {"timestamp": t0, "symbol": "BBB", "buy_price": 100.0, "sell_price": 106.0, "buy_amount": 100.0, "sell_amount": 0.0, "trade_amount": 100.0},
        ]
    )
    cfg = dict(
        initial_cash=10_000.0,
        max_positions=5,
        max_leverage=2.0,
        trade_amount_scale=100.0,
        fee_by_symbol={"AAA": 0.0, "BBB": 0.0},
        decision_lag_bars=0,
        enforce_market_hours=False,
        close_at_eod=False,
        max_hold_hours=1000,
        bar_margin=0.0,
        int_qty=True,
        entry_allocator_mode="concentrated",
        entry_allocator_edge_power=2.0,
        entry_allocator_max_single_position_fraction=0.6,
        entry_allocator_reserve_fraction=0.1,
    )

    python_result = run_portfolio_simulation(
        bars,
        actions,
        PortfolioConfig(**cfg, sim_backend="python"),
        horizon=1,
    )
    native_result = run_portfolio_simulation(
        bars,
        actions,
        PortfolioConfig(**cfg, sim_backend="native"),
        horizon=1,
    )

    python_entries = {(trade.symbol, trade.side): trade.quantity for trade in python_result.trades}
    native_entries = {(trade.symbol, trade.side): trade.quantity for trade in native_result.trades}

    assert python_entries == {("AAA", "buy"): 47.0, ("BBB", "buy"): 42.0}
    assert native_entries == python_entries
    assert native_result.metrics["final_equity"] == pytest.approx(python_result.metrics["final_equity"])
