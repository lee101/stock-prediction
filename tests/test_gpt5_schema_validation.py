from __future__ import annotations

import json

from gpt5_queries import (
    _build_schema_retry_message,
    collect_structured_payload_issues,
    validate_structured_payload,
)
from stockagent.agentsimulator.prompt_builder import plan_response_schema


def _base_payload() -> dict:
    payload = {
        "target_date": "2025-10-17",
        "instructions": [
            {
                "symbol": "AAPL",
                "action": "buy",
                "quantity": 10,
                "execution_session": "market_open",
                "entry_price": 100.0,
                "exit_price": None,
                "exit_reason": None,
                "notes": None,
            }
        ],
        "risk_notes": None,
        "focus_symbols": [],
        "stop_trading_symbols": [],
        "execution_window": "market_open",
        "metadata": {},
    }
    return payload


def test_validate_structured_payload_accepts_valid_payload() -> None:
    schema = plan_response_schema()
    payload = _base_payload()
    assert validate_structured_payload(payload, schema) is None


def test_validate_structured_payload_detects_missing_quantity() -> None:
    schema = plan_response_schema()
    payload = _base_payload()
    del payload["instructions"][0]["quantity"]

    error = validate_structured_payload(payload, schema)
    assert error is not None
    assert "instructions[0]" in error
    assert "quantity" in error


def test_validate_structured_payload_enforces_positive_quantity_for_trades() -> None:
    schema = plan_response_schema()
    payload = _base_payload()
    payload["instructions"][0]["quantity"] = 0

    error = validate_structured_payload(payload, schema)
    assert error is not None
    assert "instructions[0].quantity" in error
    assert "greater than zero" in error


def test_collect_structured_payload_issues_reports_missing_quantity() -> None:
    schema = plan_response_schema()
    payload = _base_payload()
    del payload["instructions"][0]["quantity"]

    issues = collect_structured_payload_issues(payload, schema)

    assert issues
    assert issues[0].path_display == "instructions[0].quantity"
    assert "missing quantity" in issues[0].message
    assert "quantity" in issues[0].fix_hint


def test_collect_structured_payload_issues_detects_null_disallowed() -> None:
    schema = plan_response_schema()
    payload = _base_payload()
    payload["target_date"] = None

    issues = collect_structured_payload_issues(payload, schema)

    assert any(issue.path_display == "target_date" for issue in issues)
    target_issue = next(issue for issue in issues if issue.path_display == "target_date")
    assert target_issue.issue_type == "null_disallowed"
    assert "Replace null" in target_issue.fix_hint


def test_build_schema_retry_message_is_contextual() -> None:
    schema = plan_response_schema()
    payload = _base_payload()
    payload["instructions"][0]["quantity"] = 0
    payload["target_date"] = None

    issues = collect_structured_payload_issues(payload, schema)
    raw_text = json.dumps(payload)
    message = _build_schema_retry_message(issues, raw_text=raw_text)

    assert "Issues detected" in message
    assert "instructions[0].quantity" in message
    assert "Replace null" in message
    assert "Previous response" in message
