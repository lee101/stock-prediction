import json
import sys
import types
from datetime import date, datetime, timezone

import pandas as pd
import pytest

# Provide a minimal stub so stockagent.agent can import gpt5_queries without the real package.
if "openai" not in sys.modules:
    openai_stub = types.ModuleType("openai")

    class _DummyClient:
        def __init__(self, *_, **__):
            pass

    openai_stub.AsyncOpenAI = _DummyClient
    openai_stub.OpenAI = _DummyClient
    sys.modules["openai"] = openai_stub

from stockagent.agentsimulator import prompt_builder as stateful_prompt_builder
from stockagent.agentsimulator.data_models import AccountPosition, AccountSnapshot
from stockagent.agentsimulator.market_data import MarketDataBundle
from stockagent.agent import (
    generate_stockagent_plan,
    simulate_stockagent_plan,
    simulate_stockagent_replanning,
)


@pytest.fixture(autouse=True)
def _patch_state_loader(monkeypatch):
    monkeypatch.setattr(stateful_prompt_builder, "load_all_state", lambda *_, **__: {})
    dummy_snapshot = AccountSnapshot(
        equity=75_000.0,
        cash=50_000.0,
        buying_power=75_000.0,
        timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
        positions=[],
    )
    monkeypatch.setattr(
        "stockagent.agentsimulator.prompt_builder.get_account_snapshot",
        lambda: dummy_snapshot,
    )
    yield


def _sample_market_bundle() -> MarketDataBundle:
    index = pd.date_range("2025-01-01", periods=3, freq="D", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": [110.0, 112.0, 111.0],
            "close": [112.0, 113.5, 114.0],
            "high": [112.0, 114.0, 115.0],
            "low": [109.0, 110.5, 110.0],
        },
        index=index,
    )
    return MarketDataBundle(
        bars={"AAPL": frame},
        lookback_days=3,
        as_of=index[-1].to_pydatetime(),
    )


def test_generate_stockagent_plan_parses_payload(monkeypatch):
    plan_payload = {
        "target_date": "2025-01-02",
        "instructions": [
            {
                "symbol": "AAPL",
                "action": "buy",
                "quantity": 5,
                "execution_session": "market_open",
                "entry_price": 110.0,
                "exit_price": 114.0,
                "exit_reason": "initial position",
                "notes": "increase exposure",
            },
            {
                "symbol": "AAPL",
                "action": "sell",
                "quantity": 5,
                "execution_session": "market_close",
                "entry_price": 110.0,
                "exit_price": 114.0,
                "exit_reason": "close for profit",
                "notes": "close position",
            },
        ],
        "risk_notes": "Focus on momentum while keeping exposure bounded.",
        "focus_symbols": ["AAPL"],
        "stop_trading_symbols": [],
        "execution_window": "market_open",
        "metadata": {"capital_allocation_plan": "Allocate 100% to AAPL for the session."},
    }
    monkeypatch.setattr(
        "stockagent.agent.query_gpt5_structured",
        lambda **_: json.dumps(plan_payload),
    )

    snapshot = AccountSnapshot(
        equity=25_000.0,
        cash=20_000.0,
        buying_power=25_000.0,
        timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
        positions=[
            AccountPosition(
                symbol="AAPL",
                quantity=0.0,
                side="flat",
                market_value=0.0,
                avg_entry_price=0.0,
                unrealized_pl=0.0,
                unrealized_plpc=0.0,
            )
        ],
    )

    envelope, raw_text = generate_stockagent_plan(
        market_data=_sample_market_bundle(),
        account_snapshot=snapshot,
        target_date=date(2025, 1, 2),
    )

    assert raw_text.strip().startswith("{")
    assert len(envelope.plan.instructions) == 2
    assert envelope.plan.instructions[0].action.value == "buy"
    assert envelope.plan.instructions[1].action.value == "sell"


def test_simulate_stockagent_plan_matches_expected(monkeypatch):
    plan_payload = {
        "target_date": "2025-01-02",
        "instructions": [
            {
                "symbol": "AAPL",
                "action": "buy",
                "quantity": 5,
                "execution_session": "market_open",
                "entry_price": 110.0,
                "exit_price": 114.0,
                "exit_reason": "initial position",
                "notes": "increase exposure",
            },
            {
                "symbol": "AAPL",
                "action": "sell",
                "quantity": 5,
                "execution_session": "market_close",
                "entry_price": 110.0,
                "exit_price": 114.0,
                "exit_reason": "close for profit",
                "notes": "close position",
            },
        ],
        "metadata": {"capital_allocation_plan": "Allocate 100% to AAPL for the session."},
    }
    monkeypatch.setattr(
        "stockagent.agent.query_gpt5_structured",
        lambda **_: json.dumps(plan_payload),
    )

    snapshot = AccountSnapshot(
        equity=20_000.0,
        cash=16_000.0,
        buying_power=24_000.0,
        timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
        positions=[],
    )

    result = simulate_stockagent_plan(
        market_data=_sample_market_bundle(),
        account_snapshot=snapshot,
        target_date=date(2025, 1, 2),
    )

    simulation = result.simulation
    assert simulation.realized_pnl == pytest.approx(7.21625, rel=1e-4)
    assert simulation.total_fees == pytest.approx(0.56375, rel=1e-4)
    assert simulation.ending_cash == pytest.approx(16006.93625, rel=1e-4)


def test_stockagent_replanning_infers_trading_days(monkeypatch):
    bundle = _sample_market_bundle()
    day_one = {
        "target_date": "2025-01-02",
        "instructions": [
            {
                "symbol": "AAPL",
                "action": "buy",
                "quantity": 5,
                "execution_session": "market_open",
                "entry_price": 110.0,
                "exit_price": 114.0,
                "exit_reason": "initial position",
                "notes": "increase exposure",
            },
            {
                "symbol": "AAPL",
                "action": "sell",
                "quantity": 5,
                "execution_session": "market_close",
                "entry_price": 110.0,
                "exit_price": 114.0,
                "exit_reason": "close for profit",
                "notes": "close position",
            },
        ],
        "metadata": {"capital_allocation_plan": "Allocate 100% to AAPL"},
    }
    day_two = {
        "target_date": "2025-01-03",
        "instructions": [
            {
                "symbol": "AAPL",
                "action": "buy",
                "quantity": 4,
                "execution_session": "market_open",
                "entry_price": 111.0,
                "exit_price": 115.0,
                "exit_reason": "probe continuation",
                "notes": "momentum follow through",
            },
            {
                "symbol": "AAPL",
                "action": "sell",
                "quantity": 4,
                "execution_session": "market_close",
                "entry_price": 111.0,
                "exit_price": 115.0,
                "exit_reason": "lock profits",
                "notes": "lock in gains",
            },
        ],
        "metadata": {"capital_allocation_plan": "Focus on AAPL with reduced sizing"},
    }
    responses = iter([json.dumps(day_one), json.dumps(day_two)])

    monkeypatch.setattr(
        "stockagent.agent.query_gpt5_structured",
        lambda **_: next(responses),
    )

    snapshot = AccountSnapshot(
        equity=30_000.0,
        cash=24_000.0,
        buying_power=36_000.0,
        timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
        positions=[],
    )

    result = simulate_stockagent_replanning(
        market_data_by_date={
            date(2025, 1, 2): bundle,
            date(2025, 1, 3): bundle,
        },
        account_snapshot=snapshot,
        target_dates=[date(2025, 1, 2), date(2025, 1, 3)],
    )

    assert len(result.steps) == 2
    assert result.annualization_days == 252
    expected_total = (result.ending_equity - result.starting_equity) / result.starting_equity
    assert result.total_return_pct == pytest.approx(expected_total, rel=1e-6)
    expected_annual = (result.ending_equity / result.starting_equity) ** (252 / len(result.steps)) - 1
    assert result.annualized_return_pct == pytest.approx(expected_annual, rel=1e-6)
