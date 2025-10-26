from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd

from stockagent.agentsimulator.data_models import (
    ExecutionSession,
    PlanActionType,
    TradingInstruction,
    TradingPlan,
)
from stockagent.agentsimulator.market_data import MarketDataBundle
from stockagentcombined_entrytakeprofit import EntryTakeProfitSimulator


def _bundle() -> MarketDataBundle:
    index = pd.date_range("2025-01-01", periods=2, freq="D", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": [100.0, 200.0],
            "high": [110.0, 205.0],
            "low": [90.0, 190.0],
            "close": [105.0, 198.0],
        },
        index=index,
    )
    return MarketDataBundle(
        bars={"AAPL": frame},
        lookback_days=2,
        as_of=index[-1].to_pydatetime(),
    )


def test_entry_take_profit_hits_target() -> None:
    simulator = EntryTakeProfitSimulator(market_data=_bundle())
    plans = [
        TradingPlan(
            target_date=date(2025, 1, 1),
            instructions=[
                TradingInstruction(
                    symbol="AAPL",
                    action=PlanActionType.BUY,
                    quantity=10.0,
                    execution_session=ExecutionSession.MARKET_OPEN,
                    entry_price=100.0,
                ),
                TradingInstruction(
                    symbol="AAPL",
                    action=PlanActionType.EXIT,
                    quantity=0.0,
                    execution_session=ExecutionSession.MARKET_CLOSE,
                    exit_price=108.0,
                ),
            ],
        )
    ]
    result = simulator.run(plans)
    assert result.realized_pnl == (108.0 - 100.0) * 10.0


def test_entry_take_profit_falls_back_to_close_when_target_missed() -> None:
    simulator = EntryTakeProfitSimulator(market_data=_bundle())
    plans = [
        TradingPlan(
            target_date=date(2025, 1, 2),
            instructions=[
                TradingInstruction(
                    symbol="AAPL",
                    action=PlanActionType.SELL,
                    quantity=5.0,
                    execution_session=ExecutionSession.MARKET_OPEN,
                    entry_price=200.0,
                ),
                TradingInstruction(
                    symbol="AAPL",
                    action=PlanActionType.EXIT,
                    quantity=0.0,
                    execution_session=ExecutionSession.MARKET_CLOSE,
                    exit_price=188.0,  # below day's low; won't be hit
                ),
            ],
        )
    ]
    result = simulator.run(plans)
    # Entry at 200 (short), exit fallback at close 198 -> profit of 2 per share.
    assert abs(result.realized_pnl - (200.0 - 198.0) * 5.0) < 1e-9


def test_entry_take_profit_metrics() -> None:
    simulator = EntryTakeProfitSimulator(market_data=_bundle())
    plans = [
        TradingPlan(
            target_date=date(2025, 1, 1),
            instructions=[
                TradingInstruction(
                    symbol="AAPL",
                    action=PlanActionType.BUY,
                    quantity=10.0,
                    execution_session=ExecutionSession.MARKET_OPEN,
                    entry_price=100.0,
                ),
                TradingInstruction(
                    symbol="AAPL",
                    action=PlanActionType.EXIT,
                    quantity=0.0,
                    execution_session=ExecutionSession.MARKET_CLOSE,
                    exit_price=105.0,
                ),
            ],
        )
    ]
    result = simulator.run(plans)
    metrics = result.return_metrics(starting_nav=10_000.0, periods=1)
    assert metrics.daily_pct > 0
    summary = result.summary(starting_nav=10_000.0, periods=1)
    assert "monthly_return_pct" in summary
    assert summary["net_pnl"] == result.net_pnl
