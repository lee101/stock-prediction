"""Prompt construction utilities for the DeepSeek trading agent."""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any, Mapping, Sequence

from stockagent.agentsimulator.data_models import AccountSnapshot
from stockagent.agentsimulator.market_data import MarketDataBundle
from stockagent.agentsimulator.prompt_builder import (
    build_daily_plan_prompt as _build_stateful_prompt,
    plan_response_schema as _stateful_schema,
)

SYSTEM_PROMPT = (
    "You are a disciplined multi-asset trade planner. Produce precise limit-style instructions that respect capital, "
    "risk, and the enforced JSON schema. Respond with JSON only."
)


def deepseek_plan_schema() -> dict[str, Any]:
    """Expose the stateful agent schema so DeepSeek responses can be validated."""
    return _stateful_schema()


def _sanitize_market_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Remove absolute timestamps and replace them with relative labels."""
    sanitized = json.loads(json.dumps(payload))
    market_data = sanitized.get("market_data", {})
    for symbol, bars in market_data.items():
        for idx, entry in enumerate(bars):
            timestamp = entry.pop("timestamp", None)
            label = f"Day-{idx}"
            if isinstance(timestamp, str):
                try:
                    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    label = f"Day-{dt.strftime('%a')}"
                except ValueError:
                    pass
            entry["day_label"] = label
            entry["sequence_index"] = idx
    return sanitized


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

    # Remove explicit calendar references from the prompt.
    prompt_text = prompt_text.replace(target_date.isoformat(), "the upcoming session")

    execution_guidance = (
        "\nExecution guidance:\n"
        "- Provide limit-style entries and paired exits so the simulator executes only when markets touch those prices.\n"
        "- Intraday gross exposure can reach 4× when conviction warrants it, but positions must be reduced to 2× or lower by the close.\n"
        "- Borrowed capital accrues 6.75% annual interest on notional above available cash; ensure projected edge covers financing costs."
    )
    if execution_guidance not in prompt_text:
        prompt_text = f"{prompt_text}{execution_guidance}"

    prompt_text += (
        "\nHistorical payload entries use relative day labels (e.g. Day-Mon, Day-Tue) instead of calendar dates. "
        "Focus on return patterns rather than real-world timestamps."
    )

    sanitized_payload = _sanitize_market_payload(payload)
    payload_json = json.dumps(sanitized_payload, ensure_ascii=False, indent=2)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text},
        {"role": "user", "content": payload_json},
    ]


__all__ = ["SYSTEM_PROMPT", "build_deepseek_messages", "deepseek_plan_schema"]
