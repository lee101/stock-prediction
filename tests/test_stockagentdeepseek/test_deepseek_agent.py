import json
from datetime import datetime, timezone, date

import pandas as pd
import pytest

from stockagent.agentsimulator import prompt_builder as stateful_prompt_builder
from stockagent.agentsimulator.data_models import AccountPosition, AccountSnapshot
from stockagent.agentsimulator.market_data import MarketDataBundle
from stockagentdeepseek.agent import simulate_deepseek_plan, simulate_deepseek_replanning
from stockagentdeepseek_entrytakeprofit.agent import simulate_deepseek_entry_takeprofit_plan
from stockagentdeepseek_maxdiff.agent import simulate_deepseek_maxdiff_plan
from stockagentdeepseek_neural.agent import simulate_deepseek_neural_plan
from stockagentdeepseek_neural.forecaster import ModelForecastSummary, NeuralForecast
from stockagentdeepseek.prompt_builder import build_deepseek_messages


@pytest.fixture(autouse=True)
def _patch_state_loader(monkeypatch):
    monkeypatch.setattr(stateful_prompt_builder, "load_all_state", lambda *_, **__: {})
    dummy_snapshot = AccountSnapshot(
        equity=50_000.0,
        cash=25_000.0,
        buying_power=25_000.0,
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


def test_simulate_deepseek_plan_produces_expected_pnl(monkeypatch):
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
    plan_json = json.dumps(plan_payload)

    monkeypatch.setattr(
        "stockagentdeepseek.agent.call_deepseek_chat",
        lambda *_, **__: plan_json,
    )

    snapshot = AccountSnapshot(
        equity=10_000.0,
        cash=8_000.0,
        buying_power=12_000.0,
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

    result = simulate_deepseek_plan(
        market_data=_sample_market_bundle(),
        account_snapshot=snapshot,
        target_date=date(2025, 1, 2),
    )

    assert result.plan.instructions[0].action.value == "buy"
    assert result.plan.instructions[1].action.value == "sell"

    simulation = result.simulation
    assert simulation.realized_pnl == pytest.approx(7.21625, rel=1e-4)
    assert simulation.total_fees == pytest.approx(0.56375, rel=1e-4)
    assert simulation.ending_cash == pytest.approx(8006.93625, rel=1e-4)


def test_simulate_deepseek_replanning_reuses_updated_snapshot(monkeypatch):
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

    call_count = {"value": 0}

    def _fake_chat(*_args, **_kwargs):
        call_count["value"] += 1
        return next(responses)

    monkeypatch.setattr("stockagentdeepseek.agent.call_deepseek_chat", _fake_chat)

    initial_snapshot = AccountSnapshot(
        equity=10_000.0,
        cash=8_000.0,
        buying_power=12_000.0,
        timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
        positions=[],
    )

    result = simulate_deepseek_replanning(
        market_data_by_date={
            date(2025, 1, 2): bundle,
            date(2025, 1, 3): bundle,
        },
        account_snapshot=initial_snapshot,
        target_dates=[date(2025, 1, 2), date(2025, 1, 3)],
    )

    assert call_count["value"] == 2
    assert len(result.steps) == 2
    assert result.steps[0].simulation.realized_pnl > 0
    assert result.steps[1].simulation.realized_pnl > 0
    assert result.steps[1].simulation.starting_cash == pytest.approx(result.steps[0].simulation.ending_cash, rel=1e-6)
    assert result.steps[0].daily_return_pct == pytest.approx(0.00086703125, rel=1e-6)
    assert result.steps[1].daily_return_pct == pytest.approx(0.001442499308, rel=1e-6)
    expected_total = (result.ending_equity - result.starting_equity) / result.starting_equity
    assert result.total_return_pct == pytest.approx(expected_total, rel=1e-6)
    expected_annual = (result.ending_equity / result.starting_equity) ** (252 / len(result.steps)) - 1
    assert result.annualized_return_pct == pytest.approx(expected_annual, rel=1e-6)
    assert result.annualization_days == 252

    summary_text = result.summary()
    assert "Annualized return (252d/yr)" in summary_text
    assert "daily return" in summary_text


def test_build_deepseek_messages_mentions_leverage_guidance():
    bundle = _sample_market_bundle()
    snapshot = AccountSnapshot(
        equity=50_000.0,
        cash=40_000.0,
        buying_power=60_000.0,
        timestamp=datetime(2025, 1, 2, tzinfo=timezone.utc),
        positions=[],
    )
    messages = build_deepseek_messages(
        market_data=bundle,
        target_date=date(2025, 1, 3),
        account_snapshot=snapshot,
    )
    combined = " ".join(message["content"] for message in messages if message["role"] == "user")
    assert "gross exposure can reach 4×" in combined
    assert "2× or lower" in combined
    assert "6.75%" in combined
    assert "Day-" in combined

    payload_data = json.loads(messages[-1]["content"])
    for bars in payload_data["market_data"].values():
        assert "timestamp" not in bars[0]
        assert "day_label" in bars[0]
        assert "sequence_index" in bars[0]


def test_entry_takeprofit_strategy(monkeypatch):
    bundle = _sample_market_bundle()
    plan_payload = {
        "target_date": "2025-01-02",
        "instructions": [
            {
                "symbol": "AAPL",
                "action": "buy",
                "quantity": 5,
                "execution_session": "market_open",
                "entry_price": 112.0,
                "exit_price": 113.5,
                "exit_reason": "take profit",
                "notes": "limit entry",
            },
            {
                "symbol": "AAPL",
                "action": "exit",
                "quantity": 5,
                "execution_session": "market_close",
                "entry_price": None,
                "exit_price": 113.5,
                "exit_reason": "target hit",
                "notes": "flatten",
            },
        ],
        "metadata": {"capital_allocation_plan": "Focus on AAPL"},
    }
    monkeypatch.setattr(
        "stockagentdeepseek.agent.call_deepseek_chat",
        lambda *_, **__: json.dumps(plan_payload),
    )
    snapshot = AccountSnapshot(
        equity=15_000.0,
        cash=10_000.0,
        buying_power=15_000.0,
        timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
        positions=[],
    )
    result = simulate_deepseek_entry_takeprofit_plan(
        market_data=bundle,
        account_snapshot=snapshot,
        target_date=date(2025, 1, 2),
    )
    assert result.simulation.realized_pnl > 0


def test_maxdiff_strategy(monkeypatch):
    bundle = _sample_market_bundle()
    plan_payload = {
        "target_date": "2025-01-02",
        "instructions": [
            {
                "symbol": "AAPL",
                "action": "buy",
                "quantity": 4,
                "execution_session": "market_open",
                "entry_price": 111.0,
                "exit_price": 113.5,
                "exit_reason": "limit hit",
                "notes": "enter if dip fills",
            },
            {
                "symbol": "AAPL",
                "action": "exit",
                "quantity": 4,
                "execution_session": "market_close",
                "entry_price": None,
                "exit_price": 113.5,
                "exit_reason": "target",
                "notes": "close when hit",
            },
        ],
        "metadata": {"capital_allocation_plan": "Dip buying"},
    }
    monkeypatch.setattr(
        "stockagentdeepseek.agent.call_deepseek_chat",
        lambda *_, **__: json.dumps(plan_payload),
    )
    snapshot = AccountSnapshot(
        equity=20_000.0,
        cash=12_000.0,
        buying_power=20_000.0,
        timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
        positions=[],
    )
    result = simulate_deepseek_maxdiff_plan(
        market_data=bundle,
        account_snapshot=snapshot,
        target_date=date(2025, 1, 2),
    )
    assert result.simulation.realized_pnl >= 0


def test_replanning_uses_365_when_weekend_data(monkeypatch):
    index = pd.date_range("2025-01-03", periods=3, freq="D", tz="UTC")  # Fri, Sat, Sun
    frame = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "close": [101.0, 102.0, 103.0],
            "high": [102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0],
        },
        index=index,
    )
    bundle = MarketDataBundle(bars={"BTCUSD": frame}, lookback_days=3, as_of=index[-1].to_pydatetime())

    plans = [
        {
            "target_date": "2025-01-04",
            "instructions": [
                {
                    "symbol": "BTCUSD",
                    "action": "buy",
                    "quantity": 1,
                    "execution_session": "market_open",
                    "entry_price": 101.0,
                    "exit_price": 103.0,
                    "exit_reason": "weekend trade",
                    "notes": "enter if dip",
                },
                {
                    "symbol": "BTCUSD",
                    "action": "exit",
                    "quantity": 1,
                    "execution_session": "market_close",
                    "entry_price": None,
                    "exit_price": 103.0,
                    "exit_reason": "target",
                    "notes": "flatten",
                },
            ],
            "metadata": {"capital_allocation_plan": "Crypto focus"},
        },
        {
            "target_date": "2025-01-05",
            "instructions": [
                {
                    "symbol": "BTCUSD",
                    "action": "buy",
                    "quantity": 1,
                    "execution_session": "market_open",
                    "entry_price": 102.0,
                    "exit_price": 104.0,
                    "exit_reason": "carry",
                    "notes": "weekend continuation",
                },
                {
                    "symbol": "BTCUSD",
                    "action": "exit",
                    "quantity": 1,
                    "execution_session": "market_close",
                    "entry_price": None,
                    "exit_price": 104.0,
                    "exit_reason": "target",
                    "notes": "close",
                },
            ],
            "metadata": {"capital_allocation_plan": "Crypto focus"},
        },
    ]
    responses = iter(json.dumps(plan) for plan in plans)
    monkeypatch.setattr(
        "stockagentdeepseek.agent.call_deepseek_chat",
        lambda *_, **__: next(responses),
    )

    snapshot = AccountSnapshot(
        equity=5_000.0,
        cash=5_000.0,
        buying_power=5_000.0,
        timestamp=datetime(2025, 1, 3, tzinfo=timezone.utc),
        positions=[],
    )

    result = simulate_deepseek_replanning(
        market_data_by_date={
            date(2025, 1, 4): bundle,
            date(2025, 1, 5): bundle,
        },
        account_snapshot=snapshot,
        target_dates=[date(2025, 1, 4), date(2025, 1, 5)],
    )
    assert result.annualization_days == 365


def test_neural_plan_appends_forecast_context(monkeypatch):
    bundle = _sample_market_bundle()
    plan_payload = {
        "target_date": "2025-01-02",
        "instructions": [
            {
                "symbol": "AAPL",
                "action": "buy",
                "quantity": 3,
                "execution_session": "market_open",
                "entry_price": 112.0,
                "exit_price": 113.5,
                "exit_reason": "neural entry",
                "notes": "forecast assisted",
            },
            {
                "symbol": "AAPL",
                "action": "exit",
                "quantity": 3,
                "execution_session": "market_close",
                "entry_price": None,
                "exit_price": 113.5,
                "exit_reason": "limit fill",
                "notes": "close",
            },
        ],
        "metadata": {"capital_allocation_plan": "AAPL neural strategy"},
    }
    captured: dict[str, list[dict[str, str]]] = {}

    def _fake_chat(messages, **_kwargs):
        captured["messages"] = messages
        return json.dumps(plan_payload)

    monkeypatch.setattr("stockagentdeepseek_neural.agent.call_deepseek_chat", _fake_chat)

    neural_forecasts = {
        "AAPL": NeuralForecast(
            symbol="AAPL",
            combined={"open": 113.2, "high": 114.6, "low": 111.8, "close": 113.9},
            best_model="toto",
            selection_source="hyperparams/best",
            model_summaries={
                "toto": ModelForecastSummary(
                    model="toto",
                    config_name="toto_best",
                    average_price_mae=0.74,
                    forecasts={"open": 113.5, "high": 114.8, "low": 112.0, "close": 114.1},
                ),
                "kronos": ModelForecastSummary(
                    model="kronos",
                    config_name="kronos_best",
                    average_price_mae=0.92,
                    forecasts={"open": 113.0, "high": 114.4, "low": 111.5, "close": 113.6},
                ),
            },
        )
    }

    monkeypatch.setattr(
        "stockagentdeepseek_neural.agent.build_neural_forecasts",
        lambda **_kwargs: neural_forecasts,
    )

    snapshot = AccountSnapshot(
        equity=12_000.0,
        cash=9_000.0,
        buying_power=12_000.0,
        timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
        positions=[],
    )

    result = simulate_deepseek_neural_plan(
        market_data=bundle,
        account_snapshot=snapshot,
        target_date=date(2025, 1, 2),
    )

    assert captured["messages"][1]["content"].count("Neural forecasts") == 1
    assert "AAPL: combined forecast" in captured["messages"][1]["content"]
    payload = json.loads(captured["messages"][-1]["content"])
    assert "neural_forecasts" in payload
    assert "AAPL" in payload["neural_forecasts"]
    assert result.simulation.realized_pnl >= 0
