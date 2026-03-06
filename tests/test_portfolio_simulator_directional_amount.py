from __future__ import annotations

import pandas as pd

from unified_hourly_experiment.marketsimulator.portfolio_simulator import (
    PortfolioConfig,
    run_portfolio_simulation,
)


def test_short_entry_uses_sell_amount_for_sizing():
    ts = pd.Timestamp("2026-03-03T15:00:00Z")
    bars = pd.DataFrame(
        [
            {
                "timestamp": ts,
                "symbol": "MTCH",
                "open": 30.0,
                "high": 31.0,
                "low": 29.0,
                "close": 30.0,
            }
        ]
    )
    actions = pd.DataFrame(
        [
            {
                "timestamp": ts,
                "symbol": "MTCH",
                "buy_price": 29.0,
                "sell_price": 30.0,
                "buy_amount": 0.1,
                "sell_amount": 80.0,
                "trade_amount": 80.0,
                "predicted_high_p50_h1": 30.2,
                "predicted_low_p50_h1": 29.4,
                "predicted_close_p50_h1": 29.7,
            }
        ]
    )
    cfg = PortfolioConfig(
        initial_cash=10_000.0,
        max_positions=1,
        max_leverage=2.0,
        trade_amount_scale=100.0,
        fee_by_symbol={"MTCH": 0.0},
        decision_lag_bars=0,
        enforce_market_hours=False,
        close_at_eod=False,
        max_hold_hours=1000,
        bar_margin=0.0,
        int_qty=True,
    )
    result = run_portfolio_simulation(bars, actions, cfg, horizon=1)

    entries = [t for t in result.trades if t.side == "short_sell"]
    assert len(entries) == 1
    assert entries[0].quantity == 533.0


def test_short_entry_intensity_power_boosts_size():
    ts = pd.Timestamp("2026-03-03T15:00:00Z")
    bars = pd.DataFrame(
        [
            {
                "timestamp": ts,
                "symbol": "MTCH",
                "open": 30.0,
                "high": 31.0,
                "low": 29.0,
                "close": 30.0,
            }
        ]
    )
    actions = pd.DataFrame(
        [
            {
                "timestamp": ts,
                "symbol": "MTCH",
                "buy_price": 29.0,
                "sell_price": 30.0,
                "buy_amount": 0.1,
                "sell_amount": 25.0,
                "trade_amount": 25.0,
                "predicted_high_p50_h1": 30.2,
                "predicted_low_p50_h1": 29.4,
                "predicted_close_p50_h1": 29.7,
            }
        ]
    )
    cfg = PortfolioConfig(
        initial_cash=10_000.0,
        max_positions=1,
        max_leverage=2.0,
        trade_amount_scale=100.0,
        entry_intensity_power=0.5,
        fee_by_symbol={"MTCH": 0.0},
        decision_lag_bars=0,
        enforce_market_hours=False,
        close_at_eod=False,
        max_hold_hours=1000,
        bar_margin=0.0,
        int_qty=True,
    )
    result = run_portfolio_simulation(bars, actions, cfg, horizon=1)
    entries = [t for t in result.trades if t.side == "short_sell"]
    assert len(entries) == 1
    assert entries[0].quantity == 333.0


def test_long_entry_works_without_predicted_columns():
    ts = pd.Timestamp("2026-03-03T15:00:00Z")
    bars = pd.DataFrame(
        [
            {
                "timestamp": ts,
                "symbol": "NVDA",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
            }
        ]
    )
    actions = pd.DataFrame(
        [
            {
                "timestamp": ts,
                "symbol": "NVDA",
                "buy_price": 100.0,
                "sell_price": 101.0,
                "buy_amount": 50.0,
                "sell_amount": 0.0,
                "trade_amount": 50.0,
            }
        ]
    )
    cfg = PortfolioConfig(
        initial_cash=10_000.0,
        max_positions=1,
        max_leverage=2.0,
        trade_amount_scale=100.0,
        fee_by_symbol={"NVDA": 0.0},
        decision_lag_bars=0,
        enforce_market_hours=False,
        close_at_eod=False,
        max_hold_hours=1000,
        bar_margin=0.0,
        int_qty=True,
    )
    result = run_portfolio_simulation(bars, actions, cfg, horizon=1)

    entries = [t for t in result.trades if t.side == "buy"]
    assert len(entries) == 1
    assert entries[0].quantity == 100.0


def test_short_entry_works_without_predicted_columns():
    ts = pd.Timestamp("2026-03-03T15:00:00Z")
    bars = pd.DataFrame(
        [
            {
                "timestamp": ts,
                "symbol": "MTCH",
                "open": 30.0,
                "high": 31.0,
                "low": 29.0,
                "close": 30.0,
            }
        ]
    )
    actions = pd.DataFrame(
        [
            {
                "timestamp": ts,
                "symbol": "MTCH",
                "buy_price": 29.5,
                "sell_price": 30.5,
                "buy_amount": 0.1,
                "sell_amount": 80.0,
                "trade_amount": 80.0,
            }
        ]
    )
    cfg = PortfolioConfig(
        initial_cash=10_000.0,
        max_positions=1,
        max_leverage=2.0,
        trade_amount_scale=100.0,
        fee_by_symbol={"MTCH": 0.0},
        decision_lag_bars=0,
        enforce_market_hours=False,
        close_at_eod=False,
        max_hold_hours=1000,
        bar_margin=0.0,
        int_qty=True,
    )
    result = run_portfolio_simulation(bars, actions, cfg, horizon=1)

    entries = [t for t in result.trades if t.side == "short_sell"]
    assert len(entries) == 1
    assert entries[0].quantity == 524.0


def test_decision_lag_handles_more_actions_than_bars():
    bars = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-03-03T15:00:00Z"),
                "symbol": "NVDA",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
            },
            {
                "timestamp": pd.Timestamp("2026-03-03T16:00:00Z"),
                "symbol": "NVDA",
                "open": 100.5,
                "high": 101.5,
                "low": 99.5,
                "close": 100.7,
            },
            {
                "timestamp": pd.Timestamp("2026-03-03T17:00:00Z"),
                "symbol": "NVDA",
                "open": 101.0,
                "high": 102.0,
                "low": 100.0,
                "close": 101.2,
            },
        ]
    )
    # Five action rows for three bars mirrors dense live logs (closed-hour signals).
    actions = pd.DataFrame(
        [
            {"timestamp": pd.Timestamp("2026-03-03T12:00:00Z"), "symbol": "NVDA", "buy_price": 100.0, "sell_price": 101.0, "buy_amount": 50.0, "sell_amount": 0.0, "trade_amount": 50.0},
            {"timestamp": pd.Timestamp("2026-03-03T13:00:00Z"), "symbol": "NVDA", "buy_price": 100.0, "sell_price": 101.0, "buy_amount": 50.0, "sell_amount": 0.0, "trade_amount": 50.0},
            {"timestamp": pd.Timestamp("2026-03-03T14:00:00Z"), "symbol": "NVDA", "buy_price": 100.0, "sell_price": 101.0, "buy_amount": 50.0, "sell_amount": 0.0, "trade_amount": 50.0},
            {"timestamp": pd.Timestamp("2026-03-03T15:00:00Z"), "symbol": "NVDA", "buy_price": 100.0, "sell_price": 101.0, "buy_amount": 50.0, "sell_amount": 0.0, "trade_amount": 50.0},
            {"timestamp": pd.Timestamp("2026-03-03T16:00:00Z"), "symbol": "NVDA", "buy_price": 100.0, "sell_price": 101.0, "buy_amount": 50.0, "sell_amount": 0.0, "trade_amount": 50.0},
        ]
    )
    cfg = PortfolioConfig(
        initial_cash=10_000.0,
        max_positions=1,
        max_leverage=2.0,
        trade_amount_scale=100.0,
        fee_by_symbol={"NVDA": 0.0},
        decision_lag_bars=1,
        enforce_market_hours=False,
        close_at_eod=False,
        max_hold_hours=1000,
        bar_margin=0.0,
        int_qty=True,
    )
    result = run_portfolio_simulation(bars, actions, cfg, horizon=1)

    assert isinstance(result.metrics, dict)


def test_entry_selection_mode_edge_rank_prefers_higher_edge():
    ts = pd.Timestamp("2026-03-03T15:00:00Z")
    bars = pd.DataFrame(
        [
            {"timestamp": ts, "symbol": "AAA", "open": 100.0, "high": 106.0, "low": 90.0, "close": 100.0},
            {"timestamp": ts, "symbol": "BBB", "open": 100.0, "high": 102.0, "low": 98.0, "close": 100.0},
        ]
    )
    actions = pd.DataFrame(
        [
            {"timestamp": ts, "symbol": "AAA", "buy_price": 92.0, "sell_price": 105.0, "buy_amount": 100.0, "sell_amount": 0.0, "trade_amount": 100.0},
            {"timestamp": ts, "symbol": "BBB", "buy_price": 99.0, "sell_price": 101.0, "buy_amount": 100.0, "sell_amount": 0.0, "trade_amount": 100.0},
        ]
    )
    cfg = PortfolioConfig(
        initial_cash=10_000.0,
        max_positions=1,
        max_leverage=2.0,
        trade_amount_scale=100.0,
        fee_by_symbol={"AAA": 0.0, "BBB": 0.0},
        decision_lag_bars=0,
        enforce_market_hours=False,
        close_at_eod=False,
        max_hold_hours=1000,
        bar_margin=0.0,
        int_qty=True,
        entry_selection_mode="edge_rank",
    )
    result = run_portfolio_simulation(bars, actions, cfg, horizon=1)
    entries = [t for t in result.trades if t.side == "buy"]
    assert len(entries) == 1
    assert entries[0].symbol == "AAA"


def test_entry_selection_mode_first_trigger_prefers_smallest_move():
    ts = pd.Timestamp("2026-03-03T15:00:00Z")
    bars = pd.DataFrame(
        [
            {"timestamp": ts, "symbol": "AAA", "open": 100.0, "high": 106.0, "low": 90.0, "close": 100.0},
            {"timestamp": ts, "symbol": "BBB", "open": 100.0, "high": 102.0, "low": 98.0, "close": 100.0},
        ]
    )
    actions = pd.DataFrame(
        [
            {"timestamp": ts, "symbol": "AAA", "buy_price": 92.0, "sell_price": 105.0, "buy_amount": 100.0, "sell_amount": 0.0, "trade_amount": 100.0},
            {"timestamp": ts, "symbol": "BBB", "buy_price": 99.0, "sell_price": 101.0, "buy_amount": 100.0, "sell_amount": 0.0, "trade_amount": 100.0},
        ]
    )
    cfg = PortfolioConfig(
        initial_cash=10_000.0,
        max_positions=1,
        max_leverage=2.0,
        trade_amount_scale=100.0,
        fee_by_symbol={"AAA": 0.0, "BBB": 0.0},
        decision_lag_bars=0,
        enforce_market_hours=False,
        close_at_eod=False,
        max_hold_hours=1000,
        bar_margin=0.0,
        int_qty=True,
        entry_selection_mode="first_trigger",
    )
    result = run_portfolio_simulation(bars, actions, cfg, horizon=1)
    entries = [t for t in result.trades if t.side == "buy"]
    assert len(entries) == 1
    assert entries[0].symbol == "BBB"


def test_pending_entry_ttl_disabled_does_not_fill_on_later_bar_without_signal():
    t0 = pd.Timestamp("2026-03-03T15:00:00Z")
    t1 = pd.Timestamp("2026-03-03T16:00:00Z")
    bars = pd.DataFrame(
        [
            {"timestamp": t0, "symbol": "NVDA", "open": 100.0, "high": 100.8, "low": 99.5, "close": 100.0},
            {"timestamp": t1, "symbol": "NVDA", "open": 100.0, "high": 101.0, "low": 98.5, "close": 99.2},
        ]
    )
    actions = pd.DataFrame(
        [
            {"timestamp": t0, "symbol": "NVDA", "buy_price": 99.0, "sell_price": 110.0, "buy_amount": 100.0, "sell_amount": 0.0, "trade_amount": 100.0},
            {"timestamp": t1, "symbol": "NVDA", "buy_price": 99.0, "sell_price": 110.0, "buy_amount": 0.0, "sell_amount": 0.0, "trade_amount": 0.0},
        ]
    )
    cfg = PortfolioConfig(
        initial_cash=10_000.0,
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
        entry_order_ttl_hours=0,
    )
    result = run_portfolio_simulation(bars, actions, cfg, horizon=1)
    entries = [t for t in result.trades if t.side == "buy"]
    assert entries == []


def test_pending_entry_ttl_allows_fill_on_later_bar_without_new_signal():
    t0 = pd.Timestamp("2026-03-03T15:00:00Z")
    t1 = pd.Timestamp("2026-03-03T16:00:00Z")
    bars = pd.DataFrame(
        [
            {"timestamp": t0, "symbol": "NVDA", "open": 100.0, "high": 100.8, "low": 99.5, "close": 100.0},
            {"timestamp": t1, "symbol": "NVDA", "open": 100.0, "high": 101.0, "low": 98.5, "close": 99.2},
        ]
    )
    actions = pd.DataFrame(
        [
            {"timestamp": t0, "symbol": "NVDA", "buy_price": 99.0, "sell_price": 110.0, "buy_amount": 100.0, "sell_amount": 0.0, "trade_amount": 100.0},
            {"timestamp": t1, "symbol": "NVDA", "buy_price": 99.0, "sell_price": 110.0, "buy_amount": 0.0, "sell_amount": 0.0, "trade_amount": 0.0},
        ]
    )
    cfg = PortfolioConfig(
        initial_cash=10_000.0,
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
        entry_order_ttl_hours=2,
    )
    result = run_portfolio_simulation(bars, actions, cfg, horizon=1)
    entries = [t for t in result.trades if t.side == "buy"]
    assert len(entries) == 1
    assert entries[0].timestamp == t1


def test_pending_entry_ttl_expires_before_late_touch():
    t0 = pd.Timestamp("2026-03-03T15:00:00Z")
    t1 = pd.Timestamp("2026-03-03T16:00:00Z")
    t2 = pd.Timestamp("2026-03-03T17:00:00Z")
    bars = pd.DataFrame(
        [
            {"timestamp": t0, "symbol": "NVDA", "open": 100.0, "high": 100.8, "low": 99.5, "close": 100.0},
            {"timestamp": t1, "symbol": "NVDA", "open": 100.0, "high": 100.7, "low": 99.4, "close": 99.8},
            {"timestamp": t2, "symbol": "NVDA", "open": 100.0, "high": 101.0, "low": 98.5, "close": 99.0},
        ]
    )
    actions = pd.DataFrame(
        [
            {"timestamp": t0, "symbol": "NVDA", "buy_price": 99.0, "sell_price": 110.0, "buy_amount": 100.0, "sell_amount": 0.0, "trade_amount": 100.0},
            {"timestamp": t1, "symbol": "NVDA", "buy_price": 99.0, "sell_price": 110.0, "buy_amount": 0.0, "sell_amount": 0.0, "trade_amount": 0.0},
            {"timestamp": t2, "symbol": "NVDA", "buy_price": 99.0, "sell_price": 110.0, "buy_amount": 0.0, "sell_amount": 0.0, "trade_amount": 0.0},
        ]
    )
    cfg = PortfolioConfig(
        initial_cash=10_000.0,
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
        entry_order_ttl_hours=1,
    )
    result = run_portfolio_simulation(bars, actions, cfg, horizon=1)
    entries = [t for t in result.trades if t.side == "buy"]
    assert entries == []


def test_pending_entry_can_fill_on_bar_without_action_row():
    t0 = pd.Timestamp("2026-03-03T15:00:00Z")
    t1 = pd.Timestamp("2026-03-03T16:00:00Z")
    bars = pd.DataFrame(
        [
            {"timestamp": t0, "symbol": "NVDA", "open": 100.0, "high": 100.8, "low": 99.5, "close": 100.0},
            {"timestamp": t1, "symbol": "NVDA", "open": 100.0, "high": 101.0, "low": 98.5, "close": 99.2},
        ]
    )
    actions = pd.DataFrame(
        [
            {"timestamp": t0, "symbol": "NVDA", "buy_price": 99.0, "sell_price": 110.0, "buy_amount": 100.0, "sell_amount": 0.0, "trade_amount": 100.0},
        ]
    )
    cfg = PortfolioConfig(
        initial_cash=10_000.0,
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
        entry_order_ttl_hours=2,
    )
    result = run_portfolio_simulation(bars, actions, cfg, horizon=1)
    entries = [t for t in result.trades if t.side == "buy"]
    assert len(entries) == 1
    assert entries[0].timestamp == t1


def test_equity_curve_keeps_all_bar_timestamps_with_sparse_actions():
    t0 = pd.Timestamp("2026-03-03T15:00:00Z")
    t1 = pd.Timestamp("2026-03-03T16:00:00Z")
    t2 = pd.Timestamp("2026-03-03T17:00:00Z")
    bars = pd.DataFrame(
        [
            {"timestamp": t0, "symbol": "NVDA", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0},
            {"timestamp": t1, "symbol": "NVDA", "open": 101.0, "high": 102.0, "low": 100.0, "close": 101.0},
            {"timestamp": t2, "symbol": "NVDA", "open": 102.0, "high": 103.0, "low": 101.0, "close": 102.0},
        ]
    )
    actions = pd.DataFrame(
        [
            {"timestamp": t0, "symbol": "NVDA", "buy_price": 100.0, "sell_price": 110.0, "buy_amount": 0.0, "sell_amount": 0.0, "trade_amount": 0.0},
        ]
    )
    cfg = PortfolioConfig(
        initial_cash=10_000.0,
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
    result = run_portfolio_simulation(bars, actions, cfg, horizon=1)
    assert len(result.equity_curve) == 3
    assert result.equity_curve.index.tolist() == [t0, t1, t2]


def test_equity_curve_reflects_realized_target_exit_on_exit_bar():
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
    )
    result = run_portfolio_simulation(bars, actions, cfg, horizon=1)

    assert result.metrics["final_equity"] == 1_200.0
    assert result.equity_curve.iloc[-1] == 1_200.0


def test_sparse_decision_lag_preserves_later_signal_bar_alignment():
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
    cfg = PortfolioConfig(
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
    result = run_portfolio_simulation(bars, actions, cfg, horizon=1)

    entry_timestamps = [trade.timestamp for trade in result.trades if trade.side == "buy"]
    exit_timestamps = [trade.timestamp for trade in result.trades if trade.side == "sell"]
    assert entry_timestamps == [t1, t3]
    assert exit_timestamps == [t2]
