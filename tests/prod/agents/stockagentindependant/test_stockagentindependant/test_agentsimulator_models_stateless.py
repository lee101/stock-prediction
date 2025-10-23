import json
from datetime import date

import pytest

from stockagentindependant.agentsimulator.data_models import (
    ExecutionSession,
    PlanActionType,
    TradingInstruction,
    TradingPlan,
    TradingPlanEnvelope,
)


def test_execution_session_and_plan_action_type_lowercase_defaults() -> None:
    assert ExecutionSession.from_value("MARKET_OPEN") is ExecutionSession.MARKET_OPEN
    assert ExecutionSession.from_value("market_close ") is ExecutionSession.MARKET_CLOSE
    assert ExecutionSession.from_value(None) is ExecutionSession.MARKET_OPEN

    assert PlanActionType.from_value("hold") is PlanActionType.HOLD
    assert PlanActionType.from_value(" exit ") is PlanActionType.EXIT

    with pytest.raises(ValueError):
        ExecutionSession.from_value("after_hours")
    with pytest.raises(ValueError):
        PlanActionType.from_value("reduce")


def test_trading_instruction_serde_handles_missing_prices() -> None:
    instruction = TradingInstruction.from_dict(
        {
            "symbol": "msft",
            "action": "sell",
            "quantity": "3",
            "execution_session": "market_open",
            "entry_price": "",
            "exit_price": "invalid",
        }
    )

    assert instruction.symbol == "MSFT"
    assert instruction.action is PlanActionType.SELL
    assert instruction.execution_session is ExecutionSession.MARKET_OPEN
    assert instruction.entry_price is None
    assert instruction.exit_price is None

    payload = instruction.to_dict()
    assert payload["symbol"] == "MSFT"
    assert payload["action"] == "sell"


def test_trading_plan_and_envelope_round_trip() -> None:
    raw = {
        "target_date": "2025-03-15",
        "instructions": [{"symbol": "aapl", "action": "buy", "quantity": 1}],
        "risk_notes": None,
        "focus_symbols": ["aapl", "ethusd"],
        "stop_trading_symbols": ["btcusd"],
        "metadata": {"source": "unit"},
        "execution_window": "market_close",
    }
    plan = TradingPlan.from_dict(raw)
    assert plan.target_date == date(2025, 3, 15)
    assert plan.focus_symbols == ["AAPL", "ETHUSD"]
    assert plan.stop_trading_symbols == ["BTCUSD"]
    assert plan.execution_window is ExecutionSession.MARKET_CLOSE

    serialized = plan.to_dict()
    assert serialized["metadata"] == {"source": "unit"}

    envelope = TradingPlanEnvelope(plan=plan)
    payload = json.loads(envelope.to_json())
    assert payload["execution_window"] == "market_close"

    round_trip = TradingPlanEnvelope.from_json(json.dumps(payload))
    assert round_trip.plan.to_dict() == serialized

    legacy_payload = {"plan": raw, "commentary": "legacy"}
    legacy_round_trip = TradingPlanEnvelope.from_json(json.dumps(legacy_payload))
    assert legacy_round_trip.plan.to_dict() == serialized

    with pytest.raises(ValueError):
        TradingPlan.from_dict({"target_date": "", "instructions": []})
    with pytest.raises(ValueError):
        TradingPlan.from_dict({"target_date": "2025-01-01", "instructions": "not-iterable"})
    with pytest.raises(ValueError):
        TradingPlanEnvelope.from_json(json.dumps({"commentary": "oops"}))
