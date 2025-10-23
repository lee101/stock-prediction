from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd

from stockagent.agentsimulator import AgentSimulator, AccountSnapshot
from stockagent.agentsimulator.data_models import (
    ExecutionSession,
    PlanActionType,
    TradingInstruction,
    TradingPlan,
)
from stockagent.agentsimulator.market_data import MarketDataBundle
from stockagentcombinedprofitshutdown import SymbolDirectionLossGuard


def _bundle() -> MarketDataBundle:
    index = pd.date_range("2025-01-01", periods=2, freq="D", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": [100.0, 90.0],
            "close": [90.0, 95.0],
        },
        index=index,
    )
    return MarketDataBundle(
        bars={"AAPL": frame},
        lookback_days=2,
        as_of=index[-1].to_pydatetime(),
    )


def test_loss_guard_skips_followup_after_loss() -> None:
    bundle = _bundle()
    snapshot = AccountSnapshot(
        equity=10_000.0,
        cash=10_000.0,
        buying_power=None,
        timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
        positions=[],
    )

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
                    exit_price=90.0,
                ),
            ],
        ),
        TradingPlan(
            target_date=date(2025, 1, 2),
            instructions=[
                TradingInstruction(
                    symbol="AAPL",
                    action=PlanActionType.BUY,
                    quantity=5.0,
                    execution_session=ExecutionSession.MARKET_OPEN,
                    entry_price=90.0,
                ),
                TradingInstruction(
                    symbol="AAPL",
                    action=PlanActionType.EXIT,
                    quantity=0.0,
                    execution_session=ExecutionSession.MARKET_CLOSE,
                    exit_price=95.0,
                ),
            ],
        ),
    ]

    simulator = AgentSimulator(
        market_data=bundle,
        account_snapshot=snapshot,
        starting_cash=10_000.0,
    )
    result = simulator.simulate(plans, strategies=[SymbolDirectionLossGuard()])

    symbols_executed = [trade["symbol"] for trade in result.trades]
    assert symbols_executed == ["AAPL", "AAPL"]  # only the day-one buy and exit executed
