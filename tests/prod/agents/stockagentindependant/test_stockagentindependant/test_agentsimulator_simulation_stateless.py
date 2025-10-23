from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from stockagentindependant.agentsimulator.data_models import (
    ExecutionSession,
    PlanActionType,
    TradingInstruction,
    TradingPlan,
)
from stockagentindependant.agentsimulator.market_data import MarketDataBundle
from stockagentindependant.agentsimulator.risk_strategies import (
    ProbeTradeStrategy,
    ProfitShutdownStrategy,
)
from stockagentindependant.agentsimulator.simulator import AgentSimulator
from stockagentindependant.agentsimulator.interfaces import DaySummary


def _bundle() -> MarketDataBundle:
    index = pd.date_range("2025-01-01", periods=3, freq="D", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": [50.0, 55.0, 60.0],
            "close": [55.0, 53.0, 62.0],
        },
        index=index,
    )
    return MarketDataBundle(
        bars={"MSFT": frame},
        lookback_days=3,
        as_of=index[-1].to_pydatetime(),
    )


def test_stateless_simulator_runs_plans_and_summarizes_trades() -> None:
    plans = [
        TradingPlan(
            target_date=date(2025, 1, 3),  # intentionally out-of-order to test sorting
            instructions=[
                TradingInstruction(
                    symbol="MSFT",
                    action=PlanActionType.EXIT,
                    quantity=0.0,
                    execution_session=ExecutionSession.MARKET_OPEN,
                ),
                TradingInstruction(
                    symbol="FAKE",
                    action=PlanActionType.BUY,
                    quantity=1.0,
                    execution_session=ExecutionSession.MARKET_OPEN,
                ),
            ],
        ),
        TradingPlan(
            target_date=date(2025, 1, 1),
            instructions=[
                TradingInstruction(
                    symbol="MSFT",
                    action=PlanActionType.BUY,
                    quantity=5.0,
                    execution_session=ExecutionSession.MARKET_OPEN,
                )
            ],
        ),
        TradingPlan(
            target_date=date(2025, 1, 2),
            instructions=[
                TradingInstruction(
                    symbol="MSFT",
                    action=PlanActionType.SELL,
                    quantity=3.0,
                    execution_session=ExecutionSession.MARKET_CLOSE,
                )
            ],
        ),
    ]

    simulator = AgentSimulator(market_data=_bundle())
    result = simulator.simulate(plans)

    assert result.trades[0]["symbol"] == "MSFT"
    assert result.trades[0]["direction"] == "long"
    assert result.trades[1]["action"] == "sell"
    # Exit creates a bookkeeping trade with zero quantity in current implementation
    assert result.trades[-1]["quantity"] == 0.0
    assert result.total_fees == pytest.approx(0.2045, rel=1e-4)
    assert result.realized_pnl == pytest.approx(28.7955, rel=1e-4)


def test_stateless_probe_trade_strategy_appends_notes() -> None:
    strategy = ProbeTradeStrategy(probe_multiplier=0.3, min_quantity=0.2)
    instruction = TradingInstruction(
        symbol="MSFT",
        action=PlanActionType.BUY,
        quantity=10.0,
        notes=None,
    )

    strategy.on_simulation_start()
    baseline = strategy.before_day(
        day_index=0,
        date=date(2025, 1, 1),
        instructions=[instruction],
        simulator=None,
    )
    assert baseline[0].quantity == 10.0
    assert baseline[0].notes is None

    strategy.after_day(
        DaySummary(
            date=date(2025, 1, 1),
            realized_pnl=-2.0,
            total_equity=1000.0,
            trades=[],
            per_symbol_direction={("MSFT", "long"): -5.0},
        )
    )
    reduced = strategy.before_day(
        day_index=1,
        date=date(2025, 1, 2),
        instructions=[instruction],
        simulator=None,
    )
    assert reduced[0].quantity == pytest.approx(3.0)
    assert reduced[0].notes == "|probe_trade"


def test_stateless_profit_shutdown_strategy_marks_probe_mode() -> None:
    strategy = ProfitShutdownStrategy(probe_multiplier=0.2, min_quantity=0.1)
    instruction = TradingInstruction(
        symbol="MSFT",
        action=PlanActionType.SELL,
        quantity=4.0,
        notes="seed",
    )

    strategy.on_simulation_start()
    baseline = strategy.before_day(
        day_index=0,
        date=date(2025, 1, 1),
        instructions=[instruction],
        simulator=None,
    )
    assert baseline[0].quantity == 4.0

    strategy.after_day(
        DaySummary(
            date=date(2025, 1, 1),
            realized_pnl=-1.0,
            total_equity=900.0,
            trades=[],
            per_symbol_direction={("MSFT", "short"): -1.0},
        )
    )
    probed = strategy.before_day(
        day_index=1,
        date=date(2025, 1, 2),
        instructions=[instruction],
        simulator=None,
    )
    assert probed[0].quantity == pytest.approx(0.8)
    assert probed[0].notes.endswith("|profit_shutdown_probe")

    strategy.after_day(
        DaySummary(
            date=date(2025, 1, 2),
            realized_pnl=5.0,
            total_equity=950.0,
            trades=[],
            per_symbol_direction={("MSFT", "short"): 3.0},
        )
    )
    restored = strategy.before_day(
        day_index=2,
        date=date(2025, 1, 3),
        instructions=[instruction],
        simulator=None,
    )
    assert restored[0].quantity == 4.0
