from __future__ import annotations

from datetime import datetime, timezone

import pytest

from marketsimulator.runner import DailySnapshot, SimulationReport, SymbolPerformance, TradeExecution
from marketsimulator.telemetry import (
    build_symbol_performance_table,
    build_portfolio_stack_series,
    build_price_history_table,
    build_trade_events_table,
    compute_breakdowns,
    compute_equity_timeseries,
    compute_fee_breakdown,
    compute_risk_timeseries,
    summarize_daily_analysis,
)


def _make_snapshot(
    day: int,
    phase: str,
    equity: float,
    cash: float,
    positions_detail: dict | None = None,
) -> DailySnapshot:
    return DailySnapshot(
        day_index=day,
        phase=phase,
        timestamp=datetime(
            2025,
            1,
            1 + day,
            9 if phase == "open" else 16,
            30 if phase == "open" else 0,
            tzinfo=timezone.utc,
        ),
        equity=equity,
        cash=cash,
        positions={},
        positions_detail=positions_detail or {},
    )


def _make_report(**overrides):
    base = dict(
        initial_cash=100_000.0,
        final_cash=99_500.0,
        final_equity=100_500.0,
        total_return=500.0,
        total_return_pct=0.005,
        fees_paid=50.0,
        trading_fees_paid=35.0,
        financing_cost_paid=15.0,
        trades_executed=3,
        max_drawdown=1_500.0,
        max_drawdown_pct=0.015,
        daily_snapshots=[],
        symbol_performance=[],
        generated_files=[],
        trade_executions=[],
        symbol_metadata={},
        price_history={},
        daily_analysis=[],
    )
    base.update(overrides)
    return SimulationReport(**base)


def test_compute_equity_timeseries_returns_daily_closes():
    snapshots = [
        _make_snapshot(0, "open", 100_000.0, 100_000.0),
        _make_snapshot(0, "close", 101_000.0, 100_500.0),
        _make_snapshot(1, "open", 101_100.0, 100_600.0),
        _make_snapshot(1, "close", 99_000.0, 99_100.0),
    ]
    report = _make_report(daily_snapshots=snapshots, final_equity=99_000.0, final_cash=99_100.0, total_return=-1_000.0, total_return_pct=-0.01)
    curve = compute_equity_timeseries(report)
    assert [entry["day_index"] for entry in curve] == [0, 1]
    assert pytest.approx(curve[0]["daily_return"], rel=1e-6) == 0.01
    assert pytest.approx(curve[1]["daily_return"], rel=1e-6) == (99_000.0 - 101_000.0) / 101_000.0
    assert pytest.approx(curve[1]["cumulative_return"], rel=1e-6) == (99_000.0 - 100_000.0) / 100_000.0


def test_compute_breakdowns_aggregates_by_asset_mode_and_strategy():
    performances = [
        SymbolPerformance(
            symbol="AAPL",
            cash_flow=500.0,
            market_value=0.0,
            position_qty=0.0,
            unrealized_pl=0.0,
            total_value=500.0,
            trades=2,
            realised_pl=500.0,
        ),
        SymbolPerformance(
            symbol="BTCUSD",
            cash_flow=-200.0,
            market_value=50.0,
            position_qty=1.0,
            unrealized_pl=50.0,
            total_value=-150.0,
            trades=4,
            realised_pl=-150.0,
        ),
    ]
    metadata = {
        "AAPL": {"asset_class": "equity", "trade_mode": "normal", "strategy": "simple"},
        "BTCUSD": {"asset_class": "crypto", "trade_mode": "probe", "strategy": "maxdiff"},
    }
    report = _make_report(symbol_performance=performances, symbol_metadata=metadata)
    breakdowns = compute_breakdowns(report)
    assert pytest.approx(breakdowns["asset"]["equity"]["realised_pnl"], rel=1e-6) == 500.0
    assert pytest.approx(breakdowns["asset"]["crypto"]["realised_pnl"], rel=1e-6) == -150.0
    assert pytest.approx(breakdowns["trade_mode"]["normal"]["trades"], rel=1e-6) == 2.0
    assert pytest.approx(breakdowns["trade_mode"]["probe"]["trades"], rel=1e-6) == 4.0
    assert pytest.approx(breakdowns["strategy"]["simple"]["realised_pnl"], rel=1e-6) == 500.0
    assert pytest.approx(breakdowns["strategy"]["maxdiff"]["realised_pnl"], rel=1e-6) == -150.0


def test_build_symbol_performance_table_includes_metadata():
    performances = [
        SymbolPerformance(
            symbol="AAPL",
            cash_flow=500.0,
            market_value=10.0,
            position_qty=1.0,
            unrealized_pl=5.0,
            total_value=515.0,
            trades=3,
            realised_pl=505.0,
        ),
    ]
    metadata = {"AAPL": {"asset_class": "equity", "trade_mode": "normal", "strategy": "simple"}}
    report = _make_report(symbol_performance=performances, symbol_metadata=metadata)
    columns, rows = build_symbol_performance_table(report)
    assert columns[:3] == ["symbol", "trades", "cash_flow"]
    assert rows[0][0] == "AAPL"
    assert rows[0][-1] == "equity"
    assert rows[0][-2] == "normal"
    assert rows[0][-3] == "simple"


def test_compute_risk_timeseries_uses_market_value():
    snapshots = [
        _make_snapshot(
            0,
            "open",
            100_000.0,
            95_000.0,
            positions_detail={"AAPL": {"market_value": 5_000.0}},
        ),
        _make_snapshot(
            0,
            "close",
            102_000.0,
            97_000.0,
            positions_detail={"AAPL": {"market_value": 7_000.0}},
        ),
    ]
    report = _make_report(daily_snapshots=snapshots)
    risk_series = compute_risk_timeseries(report)
    assert pytest.approx(risk_series[0]["gross_exposure"], rel=1e-6) == 5_000.0
    assert pytest.approx(risk_series[1]["gross_exposure"], rel=1e-6) == 7_000.0
    assert pytest.approx(risk_series[1]["leverage"], rel=1e-6) == 7_000.0 / 102_000.0


def test_compute_fee_breakdown_splits_trading_and_financing():
    report = _make_report()
    fees = compute_fee_breakdown(report)
    assert fees["fees/total"] == 50.0
    assert fees["fees/trading"] == 35.0
    assert fees["fees/financing"] == 15.0


def test_build_portfolio_stack_series_emits_rows():
    snapshots = [
        _make_snapshot(0, "close", 101_000.0, 99_000.0, positions_detail={"MSFT": {"market_value": 4_000.0}}),
        _make_snapshot(1, "close", 103_000.0, 98_500.0, positions_detail={"MSFT": {"market_value": 3_000.0}, "AAPL": {"market_value": 2_500.0}}),
    ]
    report = _make_report(daily_snapshots=snapshots)
    columns, rows = build_portfolio_stack_series(report)
    assert columns[0] == "timestamp"
    assert any(row[3] == "MSFT" for row in rows)
    assert any(row[3] == "AAPL" for row in rows)


def test_build_trade_events_table_returns_trades():
    trade = TradeExecution(
        timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
        symbol="AAPL",
        side="buy",
        price=150.0,
        qty=2.0,
        notional=300.0,
        fee=0.3,
        cash_delta=-300.3,
        slip_bps=5.0,
    )
    report = _make_report(trade_executions=[trade])
    columns, rows = build_trade_events_table(report)
    assert columns[0] == "timestamp"
    assert rows[0][1] == "AAPL"
    assert pytest.approx(rows[0][3], rel=1e-6) == 2.0


def test_build_price_history_table_flattens_entries():
    history = {
        "AAPL": [
            {"timestamp": "2025-01-01T16:00:00+00:00", "close": 150.0, "open": 149.0, "high": 151.0, "low": 148.5, "volume": 1_000},
            {"timestamp": "2025-01-02T16:00:00+00:00", "close": 152.0, "open": 150.5, "high": 153.0, "low": 150.0, "volume": 1_200},
        ]
    }
    report = _make_report(price_history=history)
    columns, rows = build_price_history_table(report)
    assert columns == ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
    assert rows[0][0] == "AAPL"
    assert rows[1][4] == 150.0


def test_summarize_daily_analysis_rolls_up_counts():
    daily_analysis = [
        {"symbols_analyzed": 5, "portfolio_size": 2, "forecasts_generated": 3, "probe_candidates": 1, "blocked_candidates": 0, "strategy_counts": {"simple": 2}, "trade_mode_counts": {"normal": 2}},
        {"symbols_analyzed": 7, "portfolio_size": 3, "forecasts_generated": 4, "probe_candidates": 2, "blocked_candidates": 1, "strategy_counts": {"maxdiff": 1}, "trade_mode_counts": {"probe": 3}},
    ]
    report = _make_report(daily_analysis=daily_analysis)
    summary = summarize_daily_analysis(report)
    assert summary["days_recorded"] == 2
    assert pytest.approx(summary["avg_symbols_analyzed"], rel=1e-6) == 6.0
    assert summary["strategy_counts"]["simple"] == 2.0
    assert summary["trade_mode_counts"]["probe"] == 3.0
