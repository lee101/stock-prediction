"""Agent that calls Claude for allocation predictions."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Mapping

import anthropic
from loguru import logger

from .data_formatter import PctLineData
from .prompt_builder import build_pctline_prompt, get_system_prompt

# Use Sonnet for speed/cost efficiency
MODEL = "claude-sonnet-4-20250514"


@dataclass
class AllocationPrediction:
    """Prediction for a single symbol."""
    symbol: str
    alloc: float  # 0-1 portfolio allocation
    pred_close_pct: float  # predicted % change in close
    high_pct: float  # predicted high % above close
    low_pct: float  # predicted low % below close
    last_close: float  # current close price

    @property
    def pred_close(self) -> float:
        """Predicted close price."""
        return self.last_close * (1 + self.pred_close_pct)

    @property
    def pred_high(self) -> float:
        """Predicted high price (for limit sell)."""
        return self.pred_close * (1 + self.high_pct)

    @property
    def pred_low(self) -> float:
        """Predicted low price (for limit buy)."""
        return self.pred_close * (1 + self.low_pct)


def _parse_json_response(text: str) -> dict[str, Any]:
    """Extract JSON from Claude's response."""
    # Try to find JSON in the response
    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Look for JSON in code blocks
    patterns = [
        r"```json\s*(.*?)\s*```",
        r"```\s*(.*?)\s*```",
        r"\{[^{}]*\}",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1) if "```" in pattern else match.group(0))
            except json.JSONDecodeError:
                continue

    raise ValueError(f"Could not parse JSON from response: {text[:200]}")


def generate_allocation_plan(
    pct_data: Mapping[str, PctLineData],
    prompt_version: str = "v1",
    max_lines: int = 500,
) -> dict[str, AllocationPrediction]:
    """Call Claude to get allocation predictions.

    Args:
        pct_data: Dict of symbol -> PctLineData
        prompt_version: Which prompt version to use (v1-v4)
        max_lines: Max historical lines per symbol

    Returns:
        Dict of symbol -> AllocationPrediction
    """
    client = anthropic.Anthropic()

    system_prompt = get_system_prompt(prompt_version)
    _, user_content = build_pctline_prompt(pct_data, max_lines)

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
        )

        response_text = response.content[0].text
        logger.debug(f"Claude response: {response_text[:500]}")

        parsed = _parse_json_response(response_text)

        # Convert to AllocationPrediction objects
        predictions = {}
        for symbol, pred in parsed.items():
            if symbol not in pct_data:
                continue

            predictions[symbol] = AllocationPrediction(
                symbol=symbol,
                alloc=float(pred.get("alloc", 0)),
                pred_close_pct=float(pred.get("pred_close_pct", 0)),
                high_pct=float(pred.get("high_pct", 0)),
                low_pct=float(pred.get("low_pct", 0)),
                last_close=pct_data[symbol].last_close,
            )

        return predictions

    except Exception as e:
        logger.error(f"Failed to generate allocation plan: {e}")
        # Return zero allocations on failure
        return {
            symbol: AllocationPrediction(
                symbol=symbol,
                alloc=0,
                pred_close_pct=0,
                high_pct=0,
                low_pct=0,
                last_close=data.last_close,
            )
            for symbol, data in pct_data.items()
        }


def simulate_pctline_agent(
    pct_data: Mapping[str, PctLineData],
    equity: float,
    prompt_version: str = "v1",
    max_lines: int = 500,
) -> tuple[dict[str, AllocationPrediction], list[dict]]:
    """Simulate agent and generate trade instructions.

    Args:
        pct_data: Dict of symbol -> PctLineData
        equity: Current portfolio equity
        prompt_version: Which prompt to use
        max_lines: Max historical lines

    Returns:
        (predictions, trade_instructions)
    """
    predictions = generate_allocation_plan(pct_data, prompt_version, max_lines)

    # Generate trade instructions from allocations
    instructions = []
    for symbol, pred in predictions.items():
        if pred.alloc <= 0.001:  # Skip near-zero allocations
            continue

        # Calculate position size
        notional = equity * pred.alloc

        # Calculate entry/exit prices from predictions
        entry_price = pred.last_close  # Enter at current close
        exit_price = pred.pred_high  # Exit at predicted high

        # Calculate quantity
        quantity = int(notional / entry_price) if entry_price > 0 else 0

        if quantity > 0:
            instructions.append({
                "symbol": symbol,
                "action": "buy",
                "quantity": quantity,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "allocation": pred.alloc,
                "pred_close_pct": pred.pred_close_pct,
                "notes": f"Alloc {pred.alloc:.1%}, pred close {pred.pred_close_pct:+.2%}",
            })

    return predictions, instructions
