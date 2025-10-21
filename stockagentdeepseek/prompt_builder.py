"""Prompt construction utilities for the DeepSeek trading agent."""

from __future__ import annotations

import json
from datetime import date
from typing import Any, Mapping, Sequence

from stockagent.agentsimulator.data_models import AccountSnapshot
from stockagent.agentsimulator.market_data import MarketDataBundle
from stockagent.agentsimulator.prompt_builder import (
    build_daily_plan_prompt as _build_stateful_prompt,
    plan_response_schema as _stateful_schema,
)

SYSTEM_PROMPT = (
    "You are DeepSeek, a stock trading agent who reasons thoughtfully about position sizing, risk, and PnL. "
    "Read the provided market context carefully and respond ONLY with valid JSON that satisfies the schema."
)


def deepseek_plan_schema() -> dict[str, Any]:
    """Expose the stateful agent schema so DeepSeek responses can be validated."""
    return _stateful_schema()


def build_deepseek_messages(
    *,
    market_data: MarketDataBundle,
    target_date: date,
    account_snapshot: AccountSnapshot | None = None,
    account_payload: Mapping[str, Any] | None = None,
    symbols: Sequence[str] | None = None,
    include_market_history: bool = True,
) -> list[dict[str, str]]:
    """Assemble DeepSeek chat messages with a dedicated system prompt."""
    if account_payload is None:
        if account_snapshot is None:
            raise ValueError("account_snapshot or account_payload must be provided.")
        account_payload = account_snapshot.to_payload()

    prompt_text, payload = _build_stateful_prompt(
        market_data=market_data,
        account_payload=dict(account_payload),
        target_date=target_date,
        symbols=symbols,
        include_market_history=include_market_history,
    )
    payload_json = json.dumps(payload, ensure_ascii=False, indent=2)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text},
        {"role": "user", "content": payload_json},
    ]


__all__ = ["SYSTEM_PROMPT", "build_deepseek_messages", "deepseek_plan_schema"]
