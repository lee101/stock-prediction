"""Build terse prompts for pct-change line trading.

Output format is simple:
{symbol: {alloc: 0-1, pred_close_pct: float, high_pct: float, low_pct: float}}

Where:
- alloc: portfolio allocation 0 to 1 (sum across all stocks <= 1)
- pred_close_pct: predicted % change in close
- high_pct: predicted % above close for limit sell
- low_pct: predicted % below close for limit buy
"""

from __future__ import annotations

from typing import Mapping

from .data_formatter import PctLineData


SYSTEM_PROMPT_V1 = """You predict stock movements from price patterns.

Input: historical pct changes per day as: close_pct,high_pct,low_pct
- close_pct: daily close % change from prev close
- high_pct: intraday high % above close (positive)
- low_pct: intraday low % below close (negative)

Output JSON with allocation (0-1) and predictions per symbol.
Allocations must sum to <= 1. Predict next day's close_pct, high_pct, low_pct.

Be conservative. Only allocate to strong patterns. 0 allocation = skip.
Respond ONLY with valid JSON, no explanation."""


def build_pctline_prompt(
    pct_data: Mapping[str, PctLineData],
    max_lines: int = 500,
) -> tuple[str, str]:
    """Build system prompt and user content for pct-line agent.

    Args:
        pct_data: Dict of symbol -> PctLineData
        max_lines: Max lines per symbol to include

    Returns:
        (system_prompt, user_content)
    """
    user_parts = []

    for symbol, data in sorted(pct_data.items()):
        # Take only last max_lines
        lines = data.lines.split("\n")
        if len(lines) > max_lines:
            lines = lines[-max_lines:]

        user_parts.append(f"## {symbol} (last_close=${data.last_close:.2f})")
        user_parts.append("\n".join(lines))
        user_parts.append("")

    user_parts.append("---")
    user_parts.append("Output format: {\"SYMBOL\": {\"alloc\": 0.0-1.0, \"pred_close_pct\": %, \"high_pct\": %, \"low_pct\": %}}")
    user_parts.append("Allocations sum <= 1. Predict tomorrow's movements.")

    return SYSTEM_PROMPT_V1, "\n".join(user_parts)


# Alternative prompts for testing

SYSTEM_PROMPT_V2_MOMENTUM = """You are a momentum trader analyzing price patterns.

Input format per line: close_pct,high_pct,low_pct (daily % changes)

Look for:
- Trending patterns (consecutive same-direction moves)
- Breakout signals (expanding high/low range)
- Mean reversion after extremes

Output JSON allocations (0-1 per symbol, sum <= 1) and predictions.
Higher allocation = higher conviction. 0 = skip.
JSON only, no explanation."""


SYSTEM_PROMPT_V3_CONSERVATIVE = """You are an extremely conservative trader.

Input: daily price % changes (close_pct,high_pct,low_pct per line)

RULES:
- Only allocate when pattern is VERY clear
- Max 0.5 allocation per stock
- Total allocation often should be < 0.3 (hold cash)
- Predict conservatively - assume small moves

Most of the time, return low/zero allocations.
Output JSON only."""


SYSTEM_PROMPT_V4_PATTERN = """Price pattern analyzer for next-day prediction.

Input: close_pct,high_pct,low_pct per day (historical % changes)

Analyze:
1. Recent trend direction and strength
2. Volatility (range of high/low)
3. Reversal signals
4. Support/resistance patterns in pct space

Output JSON with:
- alloc: conviction 0-1 (higher = stronger pattern)
- pred_close_pct: expected close % change
- high_pct: expected intraday high above predicted close
- low_pct: expected intraday low below predicted close

Be data-driven. No explanation, just JSON."""


def get_system_prompt(version: str = "v1") -> str:
    """Get system prompt by version."""
    prompts = {
        "v1": SYSTEM_PROMPT_V1,
        "v2": SYSTEM_PROMPT_V2_MOMENTUM,
        "v3": SYSTEM_PROMPT_V3_CONSERVATIVE,
        "v4": SYSTEM_PROMPT_V4_PATTERN,
    }
    return prompts.get(version, SYSTEM_PROMPT_V1)
