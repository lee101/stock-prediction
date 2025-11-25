"""Tests for the Claude Opus 4.5 Stock Trading Agent."""

import json
from datetime import datetime, timezone, date

import pandas as pd
import pytest

from stockagent.agentsimulator.data_models import AccountPosition, AccountSnapshot
from stockagent.agentsimulator.market_data import MarketDataBundle


@pytest.fixture()
def sample_bundle() -> MarketDataBundle:
    """Create sample market data for testing."""
    index = pd.date_range("2025-01-01", periods=5, freq="D", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": [100.0, 102.0, 101.5, 103.0, 104.0],
            "high": [103.0, 104.0, 103.5, 105.0, 106.0],
            "low": [99.0, 100.5, 100.0, 101.5, 102.0],
            "close": [102.0, 101.5, 103.0, 104.5, 105.5],
        },
        index=index,
    )
    return MarketDataBundle(
        bars={"AAPL": frame, "MSFT": frame * 1.5},
        lookback_days=5,
        as_of=index[-1].to_pydatetime(),
    )


@pytest.fixture()
def sample_snapshot() -> AccountSnapshot:
    """Create sample account snapshot for testing."""
    return AccountSnapshot(
        equity=100_000.0,
        cash=80_000.0,
        buying_power=100_000.0,
        timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
        positions=[],
    )


@pytest.fixture()
def sample_plan_response() -> str:
    """Sample JSON response that the LLM would return."""
    return json.dumps({
        "target_date": "2025-01-05",
        "instructions": [
            {
                "symbol": "AAPL",
                "action": "buy",
                "quantity": 50,
                "execution_session": "market_open",
                "entry_price": 103.0,
                "exit_price": 105.5,
                "exit_reason": "momentum entry",
                "notes": "enter on dip, target resistance",
            },
            {
                "symbol": "AAPL",
                "action": "exit",
                "quantity": 50,
                "execution_session": "market_close",
                "entry_price": None,
                "exit_price": 105.5,
                "exit_reason": "take profit",
                "notes": "close position at target",
            },
        ],
        "risk_notes": "Conservative position sizing given market volatility.",
        "focus_symbols": ["AAPL"],
        "stop_trading_symbols": [],
        "metadata": {"strategy": "momentum_breakout"},
    })


def test_opus_wrapper_imports():
    """Test that opus wrapper can be imported."""
    from stockagentopus.opus_wrapper import (
        call_opus_chat,
        reset_client,
        DEFAULT_MODEL,
        MAX_OUTPUT_TOKENS,
    )
    assert DEFAULT_MODEL == "claude-opus-4-5-20251101"
    assert MAX_OUTPUT_TOKENS > 0


def test_prompt_builder_creates_messages(sample_bundle, sample_snapshot):
    """Test that the prompt builder creates proper messages."""
    from stockagentopus.prompt_builder import build_opus_messages, SYSTEM_PROMPT

    messages = build_opus_messages(
        market_data=sample_bundle,
        target_date=date(2025, 1, 5),
        account_snapshot=sample_snapshot,
        symbols=["AAPL"],
    )

    assert len(messages) >= 2
    assert messages[0]["role"] == "user"
    assert "upcoming trading session" in messages[0]["content"]
    assert "AAPL" in messages[-1]["content"]


def test_simulate_opus_plan_with_mock(monkeypatch, sample_bundle, sample_snapshot, sample_plan_response):
    """Test simulation with mocked LLM response."""
    from stockagentopus.agent import simulate_opus_plan

    monkeypatch.setattr(
        "stockagentopus.agent.call_opus_chat",
        lambda *_, **__: sample_plan_response,
    )

    result = simulate_opus_plan(
        market_data=sample_bundle,
        account_snapshot=sample_snapshot,
        target_date=date(2025, 1, 5),
        symbols=["AAPL"],
        calibration_window=0,
    )

    assert result.plan.instructions[0].symbol == "AAPL"
    assert result.plan.instructions[0].action.value == "buy"
    assert result.simulation is not None


def test_replanning_accumulates_returns(monkeypatch, sample_snapshot):
    """Test multi-day replanning."""
    from stockagentopus.agent import simulate_opus_replanning

    # Use weekday dates (Jan 6 2025 = Monday, Jan 7 = Tuesday)
    index = pd.date_range("2025-01-06", periods=5, freq="B", tz="UTC")  # Business days
    frame = pd.DataFrame(
        {
            "open": [100.0, 102.0, 101.5, 103.0, 104.0],
            "high": [103.0, 104.0, 103.5, 105.0, 106.0],
            "low": [99.0, 100.5, 100.0, 101.5, 102.0],
            "close": [102.0, 101.5, 103.0, 104.5, 105.5],
        },
        index=index,
    )
    weekday_bundle = MarketDataBundle(
        bars={"AAPL": frame},
        lookback_days=5,
        as_of=index[-1].to_pydatetime(),
    )

    day_plans = [
        {
            "target_date": "2025-01-06",  # Monday
            "instructions": [
                {
                    "symbol": "AAPL",
                    "action": "buy",
                    "quantity": 20,
                    "execution_session": "market_open",
                    "entry_price": 102.0,
                    "exit_price": 104.0,
                },
                {
                    "symbol": "AAPL",
                    "action": "exit",
                    "quantity": 20,
                    "execution_session": "market_close",
                    "exit_price": 104.0,
                },
            ],
        },
        {
            "target_date": "2025-01-07",  # Tuesday
            "instructions": [
                {
                    "symbol": "AAPL",
                    "action": "buy",
                    "quantity": 25,
                    "execution_session": "market_open",
                    "entry_price": 103.5,
                    "exit_price": 105.0,
                },
                {
                    "symbol": "AAPL",
                    "action": "exit",
                    "quantity": 25,
                    "execution_session": "market_close",
                    "exit_price": 105.0,
                },
            ],
        },
    ]
    responses = iter(json.dumps(p) for p in day_plans)

    call_count = {"value": 0}

    def _fake_chat(*_, **__):
        call_count["value"] += 1
        return next(responses)

    monkeypatch.setattr("stockagentopus.agent.call_opus_chat", _fake_chat)

    result = simulate_opus_replanning(
        market_data_by_date={
            date(2025, 1, 6): weekday_bundle,
            date(2025, 1, 7): weekday_bundle,
        },
        account_snapshot=sample_snapshot,
        target_dates=[date(2025, 1, 6), date(2025, 1, 7)],
        symbols=["AAPL"],
        calibration_window=0,
    )

    assert call_count["value"] == 2
    assert len(result.steps) == 2
    assert result.annualization_days == 252


def test_opus_plan_result_dataclass():
    """Test OpusPlanResult dataclass."""
    from stockagentopus.agent import OpusPlanResult
    from stockagent.agentsimulator.data_models import TradingPlan
    from stockagentdeepseek_maxdiff.simulator import MaxDiffResult

    plan = TradingPlan(target_date=date(2025, 1, 5))
    simulation = MaxDiffResult(
        realized_pnl=100.0,
        total_fees=5.0,
        ending_cash=10100.0,
        ending_equity=10100.0,
    )

    result = OpusPlanResult(
        plan=plan,
        raw_response="{}",
        simulation=simulation,
    )

    assert result.simulation.net_pnl == 95.0


def test_replan_result_summary():
    """Test OpusReplanResult summary generation."""
    from stockagentopus.agent import OpusReplanResult, OpusPlanStep
    from stockagent.agentsimulator.data_models import TradingPlan
    from stockagentdeepseek_maxdiff.simulator import MaxDiffResult

    steps = [
        OpusPlanStep(
            date=date(2025, 1, 4),
            plan=TradingPlan(target_date=date(2025, 1, 4)),
            raw_response="{}",
            simulation=MaxDiffResult(50.0, 2.0, 10050.0, 10050.0),
            starting_equity=10000.0,
            ending_equity=10048.0,
            daily_return_pct=0.0048,
        ),
    ]

    result = OpusReplanResult(
        steps=steps,
        starting_equity=10000.0,
        ending_equity=10048.0,
        total_return_pct=0.0048,
        annualized_return_pct=0.0048 * 252,
        annualization_days=252,
    )

    summary = result.summary()
    assert "Claude Opus replanning results" in summary
    assert "Days simulated: 1" in summary
    assert "Annualized return" in summary
