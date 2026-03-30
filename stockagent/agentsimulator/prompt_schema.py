"""Shared JSON schema for stockagent plan responses."""

from __future__ import annotations

from typing import Any


def plan_response_schema() -> dict[str, Any]:
    instruction_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "symbol": {"type": "string"},
            "action": {"type": "string", "enum": ["buy", "sell", "exit", "hold"]},
            "quantity": {"type": "number", "minimum": 0},
            "execution_session": {"type": "string", "enum": ["market_open", "market_close"]},
            "entry_price": {"type": ["number", "null"]},
            "exit_price": {"type": ["number", "null"]},
            "exit_reason": {"type": ["string", "null"]},
            "notes": {"type": ["string", "null"]},
        },
        "required": [
            "symbol",
            "action",
            "quantity",
            "execution_session",
            "entry_price",
            "exit_price",
            "exit_reason",
            "notes",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "target_date": {"type": "string", "format": "date"},
            "instructions": {"type": "array", "items": instruction_schema},
            "risk_notes": {"type": ["string", "null"]},
            "focus_symbols": {"type": "array", "items": {"type": "string"}},
            "stop_trading_symbols": {"type": "array", "items": {"type": "string"}},
            "execution_window": {"type": "string", "enum": ["market_open", "market_close"]},
            "metadata": {"type": "object"},
        },
        "required": ["target_date", "instructions"],
        "additionalProperties": False,
    }
