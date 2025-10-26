"""Prompt construction helpers for the stateless agent."""

from __future__ import annotations

import json
from datetime import date
from collections.abc import Sequence

from .market_data import MarketDataBundle
from ..constants import DEFAULT_SYMBOLS, SIMULATION_DAYS, TRADING_FEE, CRYPTO_TRADING_FEE


SYSTEM_PROMPT = "You are GPT-5, a benchmark trading planner. Always respond with the enforced JSON schema."


def plan_response_schema() -> dict[str, object]:
    instruction_schema: dict[str, object] = {
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


def build_daily_plan_prompt(
    market_data: MarketDataBundle,
    target_date: date,
    symbols: Sequence[str] | None = None,
    include_market_history: bool = True,
) -> tuple[str, dict[str, object]]:
    symbols = list(symbols) if symbols is not None else list(DEFAULT_SYMBOLS)
    market_payload = market_data.to_payload() if include_market_history else {"symbols": list(symbols)}

    prompt = f"""
You are devising a one-day allocation for a paper-trading benchmark.

Context:
- Usable symbols: {", ".join(symbols)}.
- Historical payload contains the last {market_data.lookback_days} trading days of OHLC percent changes per symbol sourced from trainingdata/.
- No prior portfolio exists; work entirely in a sandbox and perform capital allocation across the available cash before issuing trades.
- Execution windows: `market_open` (09:30 ET) or `market_close` (16:00 ET). Choose one per instruction.
- Assume round-trip trading fees of {TRADING_FEE:.4%} for equities and {CRYPTO_TRADING_FEE:.4%} for crypto, and keep the plan profitable after fees.
- Plans will be benchmarked over {SIMULATION_DAYS} simulated days.

Structured output requirements:
- Follow the schema exactly.
- Return a single JSON object containing the plan fields at the top level—do not wrap the payload under `plan` or include `commentary`.
- Record a `capital_allocation_plan` string inside `metadata` describing how funds are distributed (percentages or dollar targets per symbol).
- Provide realistic `entry_price` / `exit_price` targets, even if you expect not to trade (use `null`).
- Supply `exit_reason` when recommending exits; use `null` otherwise.
- Return ONLY the JSON object—no markdown, narrative, or extra fields.
""".strip()

    user_payload: dict[str, object] = {
        "market_data": market_payload,
        "target_date": target_date.isoformat(),
    }

    return prompt, user_payload


def dump_prompt_package(
    market_data: MarketDataBundle,
    target_date: date,
    include_market_history: bool = True,
) -> dict[str, str]:
    prompt, user_payload = build_daily_plan_prompt(
        market_data=market_data,
        target_date=target_date,
        include_market_history=include_market_history,
    )
    return {
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": prompt,
        "user_payload_json": json.dumps(user_payload, ensure_ascii=False, indent=2),
    }
