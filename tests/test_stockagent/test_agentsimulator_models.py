import json
from datetime import date

import pytest

from stockagent.agentsimulator.data_models import (
    ExecutionSession,
    PlanActionType,
    TradingInstruction,
    TradingPlan,
    TradingPlanEnvelope,
)


def test_execution_session_and_plan_action_type_parsing() -> None:
    assert ExecutionSession.from_value("market_open") is ExecutionSession.MARKET_OPEN
    assert ExecutionSession.from_value(" MARKET_CLOSE ") is ExecutionSession.MARKET_CLOSE
    assert ExecutionSession.from_value("") is ExecutionSession.MARKET_OPEN

    assert PlanActionType.from_value("buy") is PlanActionType.BUY
    assert PlanActionType.from_value(" SELL ") is PlanActionType.SELL
    assert PlanActionType.from_value(None) is PlanActionType.HOLD

    with pytest.raises(ValueError):
        ExecutionSession.from_value("overnight")
    with pytest.raises(ValueError):
        PlanActionType.from_value("scale-in")


def test_trading_instruction_round_trip_serialization() -> None:
    instruction = TradingInstruction.from_dict(
        {
            "symbol": "aapl",
            "action": "BUY",
            "quantity": "5",
            "execution_session": "market_close",
            "entry_price": "101.5",
            "exit_price": "bad-input",
            "exit_reason": "test",
            "notes": "note",
        }
    )

    assert instruction.symbol == "AAPL"
    assert instruction.action is PlanActionType.BUY
    assert instruction.execution_session is ExecutionSession.MARKET_CLOSE
    assert instruction.entry_price == pytest.approx(101.5)
    assert instruction.exit_price is None  # bad input should be sanitized
    assert instruction.exit_reason == "test"
    assert instruction.notes == "note"

    serialized = instruction.to_dict()
    assert serialized["symbol"] == "AAPL"
    assert serialized["action"] == "buy"
    assert serialized["execution_session"] == "market_close"

    with pytest.raises(ValueError):
        TradingInstruction.from_dict({"action": "buy", "quantity": 1})


def test_trading_plan_parsing_and_envelope_round_trip() -> None:
    raw_plan = {
        "target_date": "2025-02-05",
        "instructions": [
            {"symbol": "msft", "action": "sell", "quantity": 2, "execution_session": "market_open"},
        ],
        "risk_notes": "Stay nimble",
        "focus_symbols": ["msft", "aapl"],
        "stop_trading_symbols": ["btcusd"],
        "metadata": {"source": "unit"},
        "execution_window": "market_close",
    }
    plan = TradingPlan.from_dict(raw_plan)
    assert plan.target_date == date(2025, 2, 5)
    assert plan.execution_window is ExecutionSession.MARKET_CLOSE
    assert plan.focus_symbols == ["MSFT", "AAPL"]
    assert plan.stop_trading_symbols == ["BTCUSD"]
    assert len(plan.instructions) == 1
    assert plan.instructions[0].action is PlanActionType.SELL

    serialized_plan = plan.to_dict()
    assert serialized_plan["target_date"] == "2025-02-05"
    assert serialized_plan["instructions"][0]["symbol"] == "MSFT"

    envelope = TradingPlanEnvelope(plan=plan)
    payload = json.loads(envelope.to_json())
    assert payload["instructions"][0]["symbol"] == "MSFT"

    round_trip = TradingPlanEnvelope.from_json(json.dumps(payload))
    assert round_trip.plan.to_dict() == serialized_plan

    legacy_payload = {"plan": raw_plan, "commentary": "legacy comment"}
    legacy_round_trip = TradingPlanEnvelope.from_json(json.dumps(legacy_payload))
    assert legacy_round_trip.plan.to_dict() == serialized_plan

    with pytest.raises(ValueError):
        TradingPlan.from_dict({"target_date": "bad-date", "instructions": []})
    with pytest.raises(ValueError):
        TradingPlan.from_dict({"target_date": "2025-01-01", "instructions": 42})
    with pytest.raises(ValueError):
        TradingPlanEnvelope.from_json(json.dumps({"commentary": "missing plan"}))
