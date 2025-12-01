"""Stage 1: Portfolio allocation across symbols.

Given historical price data and Chronos2 forecasts, determine:
- Which symbols to trade (allocation 0-1)
- Direction (long or short) - NOTE: Crypto is LONG ONLY
- Confidence score (0-1) for overall trade decision
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import anthropic
from loguru import logger


def _get_api_key() -> str:
    """Get API key from environment or env_real.py fallback."""
    key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
    if not key:
        # Try importing from env_real.py
        try:
            from env_real import CLAUDE_API_KEY
            if CLAUDE_API_KEY:
                return CLAUDE_API_KEY
        except ImportError:
            pass
        raise RuntimeError("API key required: set ANTHROPIC_API_KEY or CLAUDE_API_KEY")
    return key

from stockagent_pctline.data_formatter import PctLineData
from .async_api import async_api_call, async_api_call_with_thinking, parse_json_robust


# Models
MODEL_SONNET = "claude-sonnet-4-20250514"
MODEL_OPUS = "claude-opus-4-5-20251101"

# Default to Sonnet for speed, use Opus with thinking for better results
MODEL = MODEL_SONNET

# Extended thinking budget for Opus (max 64000 total output tokens)
# Budget must be < max_tokens, so we use 60000 thinking + 4000 response
THINKING_BUDGET = 60000


PORTFOLIO_SYSTEM_PROMPT = """You are an aggressive quantitative trader maximizing 1-day PnL.

INPUT:
- Historical price movements in BASIS POINTS (bps): close_bps, high_bps, low_bps per day
  (100 bps = 1%, so +250 bps = +2.5% move)
- Chronos2 neural forecast (predicted OHLC with confidence intervals)
- Current portfolio value
- Asset class indicator (crypto vs stock)

GOAL: MAXIMIZE expected daily PnL. Ignore risk concerns - pure profit optimization.

OUTPUT JSON FORMAT:
{
  "overall_confidence": 0.0-1.0,
  "reasoning": "brief explanation",
  "allocations": {
    "SYMBOL_X": {
      "alloc": 0.0-1.0,  // portfolio fraction (0 = skip)
      "direction": "long" or "short",
      "confidence": 0.0-1.0,
      "rationale": "1-line explanation",
      "leverage": 1.0-2.0  // for stocks only, 1.0 = no leverage
    }
  }
}

TRADING RULES:
- BE AGGRESSIVE - allocate heavily to your best ideas
- Overall confidence 0.5+ = trade, 0.7+ = go big
- Minimum 20% allocation to your best idea
- CRYPTO: LONG ONLY (no shorting available)
- STOCKS: Can short OR use leverage up to 2x (leverage costs 6.5% annually)
- Long when: oversold, rising momentum, support bounce
- Short when (stocks only): overbought, declining momentum, resistance hit
- High volatility = MORE opportunity, not risk
- Expected move > 50 bps = allocate

LEVERAGE GUIDANCE (stocks only):
- Use 1.5-2x leverage when very confident in direction
- Leverage cost: 6.5% annual = ~1.78 bps/day on leveraged portion
- Only leverage if expected return > leverage cost

PATTERN SIGNALS (all in basis points):
- Momentum: 3+ days same direction = continuation likely
- Mean reversion: large move (>300 bps) often reverses 100-200 bps
- Breakouts: price exceeding recent high_bps range
- Support/resistance: repeated bounces at similar levels

RESPOND WITH JSON ONLY."""


# Crypto-specific prompt (LONG ONLY)
PORTFOLIO_SYSTEM_PROMPT_CRYPTO = """You are an aggressive quantitative crypto trader maximizing 1-day PnL.

INPUT:
- Historical price movements in BASIS POINTS (bps): close_bps, high_bps, low_bps per day
  (100 bps = 1%, so +250 bps = +2.5% move)
- Chronos2 neural forecast (predicted OHLC with confidence intervals)
- Current portfolio value

GOAL: MAXIMIZE expected daily PnL. LONG POSITIONS ONLY (cannot short crypto).

OUTPUT JSON FORMAT:
{
  "overall_confidence": 0.0-1.0,
  "reasoning": "brief explanation",
  "allocations": {
    "SYMBOL_X": {
      "alloc": 0.0-1.0,  // portfolio fraction (0 = skip)
      "direction": "long",  // ALWAYS "long" for crypto
      "confidence": 0.0-1.0,
      "rationale": "1-line explanation"
    }
  }
}

TRADING RULES:
- BE AGGRESSIVE - allocate heavily to your best ideas
- Overall confidence 0.5+ = trade, 0.7+ = go big
- Minimum 20% allocation to your best idea
- LONG ONLY - if bearish, allocate 0% (skip symbol)
- Long when: oversold, rising momentum, support bounce, accumulation
- Skip when: overbought, declining momentum, distribution
- High volatility = MORE opportunity, not risk
- Expected move > 50 bps upward = allocate

PATTERN SIGNALS (all in basis points):
- Momentum: 3+ days same direction = continuation likely
- Mean reversion: large drop (>300 bps) often bounces 100-200 bps
- Breakouts: price exceeding recent high_bps range
- Support: repeated bounces at similar levels = buy opportunity

RESPOND WITH JSON ONLY."""


def is_crypto(symbol: str) -> bool:
    """Check if symbol is a crypto asset (ends with USD like BTCUSD, ETHUSD)."""
    return symbol.endswith("USD") and len(symbol) <= 7


@dataclass
class SymbolAllocation:
    """Allocation decision for a single symbol."""
    symbol: str
    alloc: float  # 0-1 portfolio fraction
    direction: str  # "long" or "short"
    confidence: float  # 0-1
    rationale: str
    leverage: float = 1.0  # 1.0-2.0 for stocks only


@dataclass
class PortfolioAllocation:
    """Complete portfolio allocation decision."""
    overall_confidence: float  # 0-1, if < threshold skip all trades
    reasoning: str
    allocations: dict[str, SymbolAllocation]

    def should_trade(self, min_confidence: float = 0.3) -> bool:
        """Check if we should execute any trades."""
        return self.overall_confidence >= min_confidence and len(self.allocations) > 0


def _format_chronos_forecast(forecast: Any) -> str:
    """Format Chronos2 forecast for prompt."""
    return (
        f"Median: O=${forecast.predicted_open:.2f} H=${forecast.predicted_high:.2f} "
        f"L=${forecast.predicted_low:.2f} C=${forecast.predicted_close:.2f}\n"
        f"10th pct: C=${forecast.low_close:.2f} | 90th pct: C=${forecast.high_close:.2f}\n"
        f"Expected return: {forecast.expected_return_pct:+.2%}, "
        f"Volatility range: {forecast.volatility_range_pct:.2%}"
    )


def _pct_to_bps(lines: str) -> str:
    """Convert pct-change lines to basis points format."""
    result = []
    for line in lines.split("\n"):
        if not line.strip():
            continue
        # Parse close_pct,high_pct,low_pct and multiply by 100 for bps
        parts = line.strip().split(",")
        if len(parts) >= 3:
            try:
                close_bps = float(parts[0]) * 100  # pct to bps
                high_bps = float(parts[1]) * 100
                low_bps = float(parts[2]) * 100
                result.append(f"{close_bps:+.0f},{high_bps:+.0f},{low_bps:+.0f}")
            except ValueError:
                result.append(line)
        else:
            result.append(line)
    return "\n".join(result)


def _build_allocation_prompt(
    pct_data: Mapping[str, PctLineData],
    chronos_forecasts: Mapping[str, Any] | None,
    equity: float,
    max_lines: int = 200,
    debias: bool = True,
) -> tuple[str, dict[str, str], bool]:
    """Build the user prompt for portfolio allocation.

    Returns:
        (prompt_text, symbol_map, all_crypto) where:
        - symbol_map maps debiased names to real symbols
        - all_crypto indicates if all symbols are crypto (for prompt selection)
    """
    # Check if all symbols are crypto (for prompt selection)
    all_crypto = all(is_crypto(sym) for sym in pct_data.keys())

    parts = [f"Portfolio Value: ${equity:,.2f}\n"]

    # Create symbol mapping for debiasing
    symbol_list = sorted(pct_data.keys())
    if debias:
        symbol_map = {f"SYMBOL_{chr(65+i)}": sym for i, sym in enumerate(symbol_list)}
        reverse_map = {sym: f"SYMBOL_{chr(65+i)}" for i, sym in enumerate(symbol_list)}
    else:
        symbol_map = {sym: sym for sym in symbol_list}
        reverse_map = symbol_map

    for symbol in symbol_list:
        data = pct_data[symbol]
        display_name = reverse_map[symbol]
        asset_type = "crypto" if is_crypto(symbol) else "stock"
        # Don't show last_close price to avoid bias from price level
        parts.append(f"## {display_name} ({asset_type})")

        # Add Chronos2 forecast if available (with expected return in bps)
        if chronos_forecasts and symbol in chronos_forecasts:
            forecast = chronos_forecasts[symbol]
            exp_ret_bps = forecast.expected_return_pct * 100  # Convert to bps
            vol_bps = forecast.volatility_range_pct * 100
            parts.append(f"Forecast: expected_return={exp_ret_bps:+.0f}bps, volatility={vol_bps:.0f}bps")

        # Add historical data as basis points
        lines = data.lines.split("\n")
        if len(lines) > max_lines:
            lines = lines[-max_lines:]
        bps_lines = _pct_to_bps("\n".join(lines))
        parts.append("Historical (close_bps,high_bps,low_bps):")
        parts.append(bps_lines)
        parts.append("")

    parts.append("---")
    parts.append("Decide portfolio allocation. JSON only.")

    return "\n".join(parts), symbol_map, all_crypto


def _parse_json_response(text: str) -> dict[str, Any]:
    """Extract JSON from response."""
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try code blocks
    patterns = [
        r"```json\s*(.*?)\s*```",
        r"```\s*(.*?)\s*```",
        r"\{[\s\S]*\}",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                json_str = match.group(1) if "```" in pattern else match.group(0)
                return json.loads(json_str)
            except json.JSONDecodeError:
                continue

    raise ValueError(f"Could not parse JSON from: {text[:300]}")


def allocate_portfolio(
    pct_data: Mapping[str, PctLineData],
    chronos_forecasts: Mapping[str, Any] | None = None,
    equity: float = 100_000.0,
    max_lines: int = 200,
    model: str = MODEL,
    use_thinking: bool = False,
    debias: bool = True,
) -> PortfolioAllocation:
    """Stage 1: Determine portfolio allocation across symbols.

    Args:
        pct_data: Dict of symbol -> PctLineData with historical pct changes
        chronos_forecasts: Optional Chronos2 forecasts per symbol
        equity: Current portfolio value
        max_lines: Max historical lines per symbol
        model: Claude model to use
        use_thinking: Enable extended thinking (uses Opus with 60000 token budget)
        debias: Use anonymous symbol names (SYMBOL_A, etc.) to avoid model bias

    Returns:
        PortfolioAllocation with decisions for each symbol
    """
    client = anthropic.Anthropic(api_key=_get_api_key())

    user_prompt, symbol_map, all_crypto = _build_allocation_prompt(
        pct_data, chronos_forecasts, equity, max_lines, debias=debias
    )

    # Select appropriate prompt based on asset class
    system_prompt = PORTFOLIO_SYSTEM_PROMPT_CRYPTO if all_crypto else PORTFOLIO_SYSTEM_PROMPT

    try:
        # Use extended thinking with Opus for better analysis
        if use_thinking:
            # Use streaming for extended thinking (required for >10 min operations)
            response_text = ""
            with client.messages.stream(
                model=MODEL_OPUS,
                max_tokens=64000,  # Max for Opus 4.5
                temperature=1,  # Required for thinking
                system=system_prompt,  # Actual system prompt
                messages=[{"role": "user", "content": user_prompt}],
                thinking={
                    "type": "enabled",
                    "budget_tokens": THINKING_BUDGET
                }
            ) as stream:
                for event in stream:
                    # Collect text events (skip thinking blocks)
                    if hasattr(event, "type") and event.type == "content_block_delta":
                        if hasattr(event.delta, "text"):
                            response_text += event.delta.text
            logger.info(f"Used extended thinking with {THINKING_BUDGET} token budget")
        else:
            response = client.messages.create(
                model=model,
                max_tokens=2048,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            response_text = response.content[0].text
        logger.debug(f"Portfolio allocation response: {response_text[:500]}")

        parsed = _parse_json_response(response_text)

        # Extract overall confidence
        overall_confidence = float(parsed.get("overall_confidence", 0.0))
        reasoning = str(parsed.get("reasoning", ""))

        # Parse allocations, mapping debiased names back to real symbols
        allocations: dict[str, SymbolAllocation] = {}
        for debiased_symbol, alloc_data in parsed.get("allocations", {}).items():
            # Map back from debiased symbol (SYMBOL_A) to real symbol (BTCUSD)
            real_symbol = symbol_map.get(debiased_symbol, debiased_symbol)

            if real_symbol not in pct_data:
                logger.warning(f"Unknown symbol in allocation: {debiased_symbol} -> {real_symbol}")
                continue

            alloc = float(alloc_data.get("alloc", 0.0))
            if alloc < 0.001:  # Skip tiny allocations
                continue

            direction = str(alloc_data.get("direction", "long")).lower()

            # ENFORCE: Crypto can only be long (no shorting)
            if is_crypto(real_symbol) and direction == "short":
                logger.warning(f"Crypto {real_symbol} cannot be shorted, skipping allocation")
                continue

            # Parse leverage (stocks only, default 1.0)
            leverage = 1.0
            if not is_crypto(real_symbol):
                leverage = min(2.0, max(1.0, float(alloc_data.get("leverage", 1.0))))

            allocations[real_symbol] = SymbolAllocation(
                symbol=real_symbol,
                alloc=alloc,
                direction=direction,
                confidence=float(alloc_data.get("confidence", 0.0)),
                rationale=str(alloc_data.get("rationale", "")),
                leverage=leverage,
            )

        return PortfolioAllocation(
            overall_confidence=overall_confidence,
            reasoning=reasoning,
            allocations=allocations,
        )

    except Exception as e:
        logger.error(f"Portfolio allocation failed: {e}")
        # Return empty allocation on failure
        return PortfolioAllocation(
            overall_confidence=0.0,
            reasoning=f"Error: {e}",
            allocations={},
        )


async def async_allocate_portfolio(
    client: anthropic.AsyncAnthropic,
    pct_data: Mapping[str, PctLineData],
    chronos_forecasts: Mapping[str, Any] | None = None,
    equity: float = 100_000.0,
    max_lines: int = 200,
    model: str = MODEL,
    use_thinking: bool = False,
    debias: bool = True,
) -> PortfolioAllocation:
    """Async Stage 1: Determine portfolio allocation across symbols.

    Args:
        client: Async Anthropic client
        pct_data: Dict of symbol -> PctLineData with historical pct changes
        chronos_forecasts: Optional Chronos2 forecasts per symbol
        equity: Current portfolio value
        max_lines: Max historical lines per symbol
        model: Claude model to use
        use_thinking: Enable extended thinking (uses Opus with 60000 token budget)
        debias: Use anonymous symbol names (SYMBOL_A, etc.) to avoid model bias

    Returns:
        PortfolioAllocation with decisions for each symbol
    """
    user_prompt, symbol_map, all_crypto = _build_allocation_prompt(
        pct_data, chronos_forecasts, equity, max_lines, debias=debias
    )

    # Select appropriate prompt based on asset class
    system_prompt = PORTFOLIO_SYSTEM_PROMPT_CRYPTO if all_crypto else PORTFOLIO_SYSTEM_PROMPT

    try:
        if use_thinking:
            response_text = await async_api_call_with_thinking(
                client=client,
                model=MODEL_OPUS,
                system=system_prompt,
                user_prompt=user_prompt,
                max_tokens=64000,
                thinking_budget=THINKING_BUDGET,
            )
            logger.info(f"Used async extended thinking with {THINKING_BUDGET} token budget")
        else:
            response_text = await async_api_call(
                client=client,
                model=model,
                system=system_prompt,
                user_prompt=user_prompt,
                max_tokens=2048,
            )
        logger.debug(f"Portfolio allocation response: {response_text[:500]}")

        parsed = parse_json_robust(response_text)

        # Extract overall confidence
        overall_confidence = float(parsed.get("overall_confidence", 0.0))
        reasoning = str(parsed.get("reasoning", ""))

        # Parse allocations, mapping debiased names back to real symbols
        allocations: dict[str, SymbolAllocation] = {}
        for debiased_symbol, alloc_data in parsed.get("allocations", {}).items():
            real_symbol = symbol_map.get(debiased_symbol, debiased_symbol)

            if real_symbol not in pct_data:
                logger.warning(f"Unknown symbol in allocation: {debiased_symbol} -> {real_symbol}")
                continue

            alloc = float(alloc_data.get("alloc", 0.0))
            if alloc < 0.001:
                continue

            direction = str(alloc_data.get("direction", "long")).lower()

            # ENFORCE: Crypto can only be long
            if is_crypto(real_symbol) and direction == "short":
                logger.warning(f"Crypto {real_symbol} cannot be shorted, skipping allocation")
                continue

            # Parse leverage (stocks only)
            leverage = 1.0
            if not is_crypto(real_symbol):
                leverage = min(2.0, max(1.0, float(alloc_data.get("leverage", 1.0))))

            allocations[real_symbol] = SymbolAllocation(
                symbol=real_symbol,
                alloc=alloc,
                direction=direction,
                confidence=float(alloc_data.get("confidence", 0.0)),
                rationale=str(alloc_data.get("rationale", "")),
                leverage=leverage,
            )

        return PortfolioAllocation(
            overall_confidence=overall_confidence,
            reasoning=reasoning,
            allocations=allocations,
        )

    except Exception as e:
        logger.error(f"Async portfolio allocation failed: {e}")
        return PortfolioAllocation(
            overall_confidence=0.0,
            reasoning=f"Error: {e}",
            allocations={},
        )
