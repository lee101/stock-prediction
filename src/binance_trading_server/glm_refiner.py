from __future__ import annotations

import json
from typing import Any

from loguru import logger


def _build_refiner_prompt(plans: list[dict[str, Any]], market_context: dict[str, Any]) -> str:
    plans_json = json.dumps(plans, indent=2, default=str)
    ctx_json = json.dumps(market_context, indent=2, default=str)
    return f"""You are a crypto portfolio risk manager. Review and refine these trading plans.

PLANS:
{plans_json}

MARKET CONTEXT:
{ctx_json}

For each plan, adjust position sizes to manage portfolio-level risk:
- Reduce correlated exposure (e.g., BTC+ETH both long)
- Cap max allocation per symbol at 25%
- Flag any plan that looks suspicious (selling below recent buy)

Return JSON array of refined plans with same schema, adding "refined_qty" and "refinement_reason" fields."""


def _parse_refined_plans(response: Any) -> list[dict[str, Any]]:
    if hasattr(response, "direction"):
        return []
    if isinstance(response, str):
        try:
            parsed = json.loads(response)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
    return []


def refine_trading_plans(
    plans: list[dict[str, Any]],
    market_context: dict[str, Any],
    *,
    model: str = "glm-4-plus",
    max_retries: int = 2,
) -> list[dict[str, Any]]:
    if not plans:
        return plans
    try:
        from llm_hourly_trader.providers import call_llm
        prompt = _build_refiner_prompt(plans, market_context)
        result = call_llm(prompt, model=model)
        refined = _parse_refined_plans(result)
        if refined:
            return refined
    except Exception as e:
        logger.opt(exception=False).warning(f"GLM refiner failed, using original plans: {e}")
    return plans
