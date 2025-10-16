"""Prompt construction helpers for the stateful agent."""

from __future__ import annotations

import json
from datetime import date
from typing import Dict, Tuple

from .account_state import get_account_snapshot
from .market_data import MarketDataBundle
from ..constants import DEFAULT_SYMBOLS, SIMULATION_DAYS, TRADING_FEE, CRYPTO_TRADING_FEE


SYSTEM_PROMPT = (
    "You are GPT-5, a cautious equities and crypto execution planner that always replies using the enforced JSON schema."
)


def plan_response_schema() -> Dict:
    return {
        "type": "object",
        "properties": {
            "plan": {
                "type": "object",
                "properties": {
                    "target_date": {"type": "string", "format": "date"},
                    "instructions": {
                        "type": "array",
                        "items": {
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
                        },
                    },
                    "risk_notes": {"type": ["string", "null"]},
                    "focus_symbols": {"type": "array", "items": {"type": "string"}},
                    "stop_trading_symbols": {"type": "array", "items": {"type": "string"}},
                    "execution_window": {"type": "string", "enum": ["market_open", "market_close"]},
                    "metadata": {"type": "object"},
                },
                "required": ["target_date", "instructions"],
                "additionalProperties": False,
            },
            "commentary": {"type": ["string", "null"]},
        },
        "required": ["plan"],
        "additionalProperties": False,
    }


def build_daily_plan_prompt(
    market_data: MarketDataBundle,
    account_payload: Dict,
    target_date: date,
    symbols=None,
    include_market_history: bool = True,
) -> Tuple[str, Dict]:
    symbols = symbols or DEFAULT_SYMBOLS
    market_payload = market_data.to_payload() if include_market_history else {"symbols": symbols}

    prompt = f"""
You are a disciplined multi-asset execution planner. Build a one-day trading plan for {target_date.isoformat()}.

Context:
- You may trade the following symbols only: {', '.join(symbols)}.
- Account details include current positions and PnL metrics.
- Historical context: the payload includes the last {market_data.lookback_days} trading days of OHLC percent changes per symbol.
- Plans must respect position sizing, preserve capital and explicitly call out assets to stop trading.
- Valid execution windows are `market_open` (09:30 ET) and `market_close` (16:00 ET). Choose one per instruction.
- Simulation harness will run your plan across {SIMULATION_DAYS} days to evaluate performance.
- Assume round-trip trading fees of {TRADING_FEE:.4%} for equities and {CRYPTO_TRADING_FEE:.4%} for crypto; ensure the plan remains profitable after fees.

Structured output requirements:
- Produce JSON matching the provided schema exactly.
- The top-level object must contain only the keys ``plan`` and ``commentary``.
- Use `exit` to close positions you no longer want, specifying the quantity to exit (0 = all) and an `exit_reason`.
- Provide realistic limit prices using `entry_price` / `exit_price` fields reflecting desired fills for the session.
- Include `risk_notes` summarizing risk considerations in under 3 sentences.
- Return ONLY the JSON object; do not include markdown or extra fields.
- Every instruction must include values for `entry_price`, `exit_price`, `exit_reason`, and `notes` (use `null` when not applicable).
- Populate `execution_window` to indicate whether trades are intended for market_open or market_close.
""".strip()

    user_payload = {
        "account": account_payload,
        "market_data": market_payload,
        "target_date": target_date.isoformat(),
    }

    return prompt, user_payload


def dump_prompt_package(
    market_data: MarketDataBundle,
    target_date: date,
    include_market_history: bool = True,
) -> Dict[str, str]:
    snapshot = get_account_snapshot()
    prompt, user_payload = build_daily_plan_prompt(
        market_data=market_data,
        account_payload=snapshot.to_payload(),
        target_date=target_date,
        include_market_history=include_market_history,
    )
    return {
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": prompt,
        "user_payload_json": json.dumps(user_payload, ensure_ascii=False, indent=2),
    }
