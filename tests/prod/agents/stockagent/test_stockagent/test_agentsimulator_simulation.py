from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd
import pytest

from stockagent.agentsimulator.data_models import (
    AccountPosition,
    AccountSnapshot,
    ExecutionSession,
    PlanActionType,
    TradingInstruction,
    TradingPlan,
)
from stockagent.agentsimulator.interfaces import BaseRiskStrategy, DaySummary
from stockagent.agentsimulator.market_data import MarketDataBundle
from stockagent.agentsimulator.risk_strategies import ProbeTradeStrategy, ProfitShutdownStrategy
from stockagent.agentsimulator.simulator import AgentSimulator


def _build_bundle() -> MarketDataBundle:
    index = pd.date_range("2025-01-01", periods=3, freq="D", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": [100.0, 112.0, 109.0],
            "close": [110.0, 111.0, 115.0],
        },
        index=index,
    )
    return MarketDataBundle(
        bars={"AAPL": frame},
        lookback_days=3,
        as_of=index[-1].to_pydatetime(),
    )


def test_agent_simulator_executes_plans_and_tracks_results() -> None:
    bundle = _build_bundle()
    snapshot = AccountSnapshot(
        equity=6000.0,
        cash=4000.0,
        buying_power=10000.0,
        timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
        positions=[
            AccountPosition(
                symbol="AAPL",
                quantity=2.0,
                side="long",
                market_value=200.0,
                avg_entry_price=90.0,
                unrealized_pl=20.0,
                unrealized_plpc=0.1,
            )
        ],
    )

    class RecorderStrategy(BaseRiskStrategy):
        def __init__(self) -> None:
            self.before_calls: list[int] = []
            self.after_realized: list[float] = []
            self.started = 0
            self.ended = 0

        def on_simulation_start(self) -> None:
            self.started += 1

        def before_day(self, *, day_index, date, instructions, simulator):
            self.before_calls.append(day_index)
            return instructions

        def after_day(self, summary: DaySummary) -> None:
            self.after_realized.append(summary.realized_pnl)

        def on_simulation_end(self) -> None:
            self.ended += 1

    plans = [
        TradingPlan(
            target_date=date(2025, 1, 1),
            instructions=[
                TradingInstruction(
                    symbol="AAPL",
                    action=PlanActionType.BUY,
                    quantity=5.0,
                    execution_session=ExecutionSession.MARKET_OPEN,
                    entry_price=100.0,
                ),
                TradingInstruction(
                    symbol="AAPL",
                    action=PlanActionType.HOLD,
                    quantity=0.0,
                    execution_session=ExecutionSession.MARKET_CLOSE,
                ),
            ],
        ),
        TradingPlan(
            target_date=date(2025, 1, 2),
            instructions=[
                TradingInstruction(
                    symbol="AAPL",
                    action=PlanActionType.SELL,
                    quantity=4.0,
                    execution_session=ExecutionSession.MARKET_CLOSE,
                    exit_price=111.0,
                )
            ],
        ),
        TradingPlan(
            target_date=date(2025, 1, 3),
            instructions=[
                TradingInstruction(
                    symbol="AAPL",
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
    ]

    recorder = RecorderStrategy()
    simulator = AgentSimulator(
        market_data=bundle,
        account_snapshot=snapshot,
        starting_cash=5000.0,
    )
    result = simulator.simulate(plans, strategies=[recorder])

    assert recorder.started == recorder.ended == 1
    assert recorder.before_calls == [0, 1, 2]
    assert len(recorder.after_realized) == 3
    assert result.starting_cash == pytest.approx(5000.0)
    assert result.ending_cash == pytest.approx(5270.3645, rel=1e-6)
    assert result.ending_equity == pytest.approx(result.ending_cash, rel=1e-6)
    assert result.realized_pnl == pytest.approx(90.6142, rel=1e-4)
    assert result.total_fees == pytest.approx(0.6355, rel=1e-4)
    assert result.final_positions == {}
    assert [trade["symbol"] for trade in result.trades] == ["AAPL", "AAPL", "AAPL"]


def test_agent_simulator_requires_plans() -> None:
    simulator = AgentSimulator(market_data=_build_bundle())
    with pytest.raises(ValueError):
        simulator.simulate([])


def test_price_lookup_includes_open_and_close_prices() -> None:
    simulator = AgentSimulator(market_data=_build_bundle())
    open_price = simulator._price_for("AAPL", date(2025, 1, 1), ExecutionSession.MARKET_OPEN)
    close_price = simulator._price_for("AAPL", date(2025, 1, 1), ExecutionSession.MARKET_CLOSE)
    assert open_price == 100.0
    assert close_price == 110.0
    with pytest.raises(KeyError):
        simulator._get_symbol_frame("MSFT")
    with pytest.raises(KeyError):
        simulator._price_for("AAPL", date(2025, 1, 5), ExecutionSession.MARKET_OPEN)


def test_probe_trade_strategy_toggles_quantities() -> None:
    strategy = ProbeTradeStrategy(probe_multiplier=0.2, min_quantity=0.5)
    instruction = TradingInstruction(symbol="AAPL", action=PlanActionType.BUY, quantity=10.0)

    strategy.on_simulation_start()
    first = strategy.before_day(
        day_index=0,
        date=date(2025, 1, 1),
        instructions=[instruction],
        simulator=None,
    )
    assert first[0].quantity == 10.0
    assert first[0] is not instruction  # ensure we returned a copy

    strategy.after_day(
        DaySummary(
            date=date(2025, 1, 1),
            realized_pnl=-5.0,
            total_equity=5000.0,
            trades=[],
            per_symbol_direction={("AAPL", "long"): -5.0},
        )
    )
    second = strategy.before_day(
        day_index=1,
        date=date(2025, 1, 2),
        instructions=[instruction],
        simulator=None,
    )
    assert second[0].quantity == pytest.approx(2.0)  # 10 * 0.2

    strategy.after_day(
        DaySummary(
            date=date(2025, 1, 2),
            realized_pnl=10.0,
            total_equity=5200.0,
            trades=[],
            per_symbol_direction={("AAPL", "long"): 1.0},
        )
    )
    third = strategy.before_day(
        day_index=2,
        date=date(2025, 1, 3),
        instructions=[instruction],
        simulator=None,
    )
    assert third[0].quantity == 10.0


def test_profit_shutdown_strategy_reduces_after_losses() -> None:
    strategy = ProfitShutdownStrategy(probe_multiplier=0.1, min_quantity=0.25)
    instruction = TradingInstruction(symbol="AAPL", action=PlanActionType.SELL, quantity=8.0)

    strategy.on_simulation_start()
    baseline = strategy.before_day(
        day_index=0,
        date=date(2025, 1, 1),
        instructions=[instruction],
        simulator=None,
    )
    assert baseline[0].quantity == 8.0

    strategy.after_day(
        DaySummary(
            date=date(2025, 1, 1),
            realized_pnl=-1.0,
            total_equity=4800.0,
            trades=[],
            per_symbol_direction={("AAPL", "short"): -1.0},
        )
    )
    reduced = strategy.before_day(
        day_index=1,
        date=date(2025, 1, 2),
        instructions=[instruction],
        simulator=None,
    )
    assert reduced[0].quantity == pytest.approx(0.8)

    strategy.after_day(
        DaySummary(
            date=date(2025, 1, 2),
            realized_pnl=5.0,
            total_equity=5000.0,
            trades=[],
            per_symbol_direction={("AAPL", "short"): 5.0},
        )
    )
    recovered = strategy.before_day(
        day_index=2,
        date=date(2025, 1, 3),
        instructions=[instruction],
        simulator=None,
    )
    assert recovered[0].quantity == 8.0
