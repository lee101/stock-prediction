from __future__ import annotations

from datetime import date

import pytest
from scripts.xgb_symbol_attribution import aggregate_symbol_attribution
from xgbnew.backtest import BacktestConfig, DayResult, DayTrade


def _trade(symbol: str, net_pct: float, score: float = 0.9) -> DayTrade:
    return DayTrade(
        symbol=symbol,
        score=score,
        actual_open=100.0,
        actual_close=100.0 + net_pct,
        entry_fill_price=100.0,
        exit_fill_price=100.0 + net_pct,
        spread_bps=1.0,
        commission_bps=0.0,
        fee_rate=0.0,
        fill_buffer_bps=0.0,
        leverage=1.0,
        gross_return_pct=net_pct,
        net_return_pct=net_pct,
        intraday_worst_dd_pct=max(0.0, -net_pct),
        side=1,
    )


def test_aggregate_symbol_attribution_reconciles_day_return() -> None:
    cfg = BacktestConfig(allocation_mode="equal")
    day = DayResult(
        day=date(2025, 1, 2),
        equity_start=10_000.0,
        equity_end=10_100.0,
        daily_return_pct=1.0,
        trades=[_trade("AAA", 2.0), _trade("BBB", 0.0)],
    )

    rows = aggregate_symbol_attribution([day], cfg)

    assert sum(row["pnl_dollars"] for row in rows) == pytest.approx(100.0)
    by_symbol = {row["symbol"]: row for row in rows}
    assert by_symbol["AAA"]["pnl_dollars"] == pytest.approx(100.0)
    assert by_symbol["BBB"]["pnl_dollars"] == pytest.approx(0.0)
    assert by_symbol["AAA"]["trade_count"] == 1
    assert by_symbol["AAA"]["win_count"] == 1


def test_aggregate_symbol_attribution_applies_short_scale_weights() -> None:
    cfg = BacktestConfig(
        allocation_mode="equal",
        short_allocation_scale=0.5,
    )
    short = _trade("SHORT", -3.0, score=0.1)
    short.side = -1
    day = DayResult(
        day=date(2025, 1, 2),
        equity_start=10_000.0,
        equity_end=10_100.0,
        daily_return_pct=1.0,
        trades=[_trade("LONG", 2.0), short],
    )

    rows = aggregate_symbol_attribution([day], cfg)

    assert sum(row["pnl_dollars"] for row in rows) == pytest.approx(100.0)
    by_symbol = {row["symbol"]: row for row in rows}
    assert by_symbol["LONG"]["pnl_dollars"] > 0
    assert by_symbol["SHORT"]["pnl_dollars"] < 0
    assert by_symbol["SHORT"]["short_count"] == 1
