"""Stage 2: Per-symbol specific buy/sell price prediction - HOURLY version.

Given a single symbol's HOURLY data and allocation decision, determine:
- Entry price (% from current close)
- Exit price (% from current close)
- Stop loss price
- Position direction confirmation
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Mapping

import anthropic
from loguru import logger

from stockagent_pctline.data_formatter import PctLineData
from .portfolio_allocator import SymbolAllocation, is_crypto
from .async_api import async_api_call, parse_json_robust


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


MODEL = "claude-sonnet-4-20250514"


# HOURLY-specific prompt (adjusted thresholds for smaller moves)
PRICE_SYSTEM_PROMPT = """You are an aggressive price analyst maximizing 1-HOUR PnL.

INPUT:
- Historical HOURLY price in BASIS POINTS (bps): close_bps, high_bps, low_bps per HOUR
  (100 bps = 1%, so +50 bps = +0.5% move)
- Chronos2 forecast (expected return in bps)
- Direction: long or short
- Allocation amount

TASK:
Set entry, exit, and stop prices in BASIS POINTS from current close for HOURLY trading.

OUTPUT JSON FORMAT:
{
  "entry_bps": int,       // bps from close (e.g., -30 = 0.3% below close)
  "exit_bps": int,        // bps from close (e.g., +80 = 0.8% above close)
  "stop_loss_bps": int,   // bps from close (e.g., -50 = 0.5% below close)
  "time_horizon": "intraday",  // always intraday for hourly
  "confidence": 0.0-1.0,
  "rationale": "brief explanation"
}

RULES FOR LONG (buy low, sell high) - HOURLY:
- entry_bps: -20 to -50 (buy on small dip) or 0 (market order)
- exit_bps: +50 to +150 (profit target - smaller for hourly)
- stop_loss_bps: -40 to -100 (loss limit)

RULES FOR SHORT (sell high, buy back low) - HOURLY:
- entry_bps: +20 to +50 (short on rally) or 0
- exit_bps: -50 to -150 (cover at profit)
- stop_loss_bps: +40 to +100 (cover at loss)

BE AGGRESSIVE (HOURLY mindset):
- Focus on maximizing expected return for the next hour
- Tighter targets for hourly vs daily
- Quick entries/exits

RESPOND WITH JSON ONLY."""


@dataclass
class PricePrediction:
    """Price prediction for a single symbol."""
    symbol: str
    direction: str  # "long" or "short"
    entry_pct: float  # % from current close for entry
    exit_pct: float  # % from current close for exit
    stop_loss_pct: float  # % from current close for stop
    time_horizon: str  # "intraday" for hourly
    confidence: float  # 0-1
    rationale: str
    last_close: float  # for calculating actual prices

    @property
    def entry_price(self) -> float:
        """Calculate actual entry price."""
        return self.last_close * (1 + self.entry_pct)

    @property
    def exit_price(self) -> float:
        """Calculate actual exit price."""
        return self.last_close * (1 + self.exit_pct)

    @property
    def stop_loss_price(self) -> float:
        """Calculate actual stop loss price."""
        return self.last_close * (1 + self.stop_loss_pct)


def _format_chronos_forecast_bps(forecast: Any) -> str:
    """Format Chronos2 forecast in basis points for prompt."""
    exp_ret_bps = forecast.expected_return_pct * 100  # Convert to bps
    vol_bps = forecast.volatility_range_pct * 100
    return f"Expected return: {exp_ret_bps:+.0f} bps, Volatility: {vol_bps:.0f} bps"


def _pct_to_bps(lines: str) -> str:
    """Convert pct-change lines to basis points format."""
    result = []
    for line in lines.split("\n"):
        if not line.strip():
            continue
        parts = line.strip().split(",")
        if len(parts) >= 3:
            try:
                close_bps = float(parts[0]) * 100
                high_bps = float(parts[1]) * 100
                low_bps = float(parts[2]) * 100
                result.append(f"{close_bps:+.0f},{high_bps:+.0f},{low_bps:+.0f}")
            except ValueError:
                result.append(line)
        else:
            result.append(line)
    return "\n".join(result)


def _build_price_prompt(
    symbol: str,
    pct_data: PctLineData,
    allocation: SymbolAllocation,
    chronos_forecast: Any | None,
    max_lines: int = 100,
) -> str:
    """Build the user prompt for price prediction."""
    parts = [
        f"Symbol: {symbol} ({'crypto' if is_crypto(symbol) else 'stock'})",
        f"Direction: {allocation.direction.upper()}",
        f"Allocation: {allocation.alloc:.0%} of portfolio",
        f"Allocation confidence: {allocation.confidence:.2f}",
        "",
    ]

    if chronos_forecast:
        parts.append(f"Chronos2 Forecast: {_format_chronos_forecast_bps(chronos_forecast)}")
        parts.append("")

    # Add historical data
    lines = pct_data.lines.split("\n")
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    bps_lines = _pct_to_bps("\n".join(lines))

    parts.append("Recent HOURLY history (close_bps,high_bps,low_bps):")
    parts.append(bps_lines)
    parts.append("")
    parts.append("---")
    parts.append("Set entry/exit/stop for next HOUR. JSON only.")

    return "\n".join(parts)


def _parse_price_response(text: str) -> dict[str, Any]:
    """Extract JSON from price prediction response."""
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

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


def predict_prices(
    symbol: str,
    pct_data: PctLineData,
    allocation: SymbolAllocation,
    chronos_forecast: Any | None = None,
    max_lines: int = 100,
    model: str = MODEL,
) -> PricePrediction:
    """Stage 2: Determine specific entry/exit/stop prices (HOURLY).

    Args:
        symbol: Symbol to predict prices for
        pct_data: Historical pct-change data
        allocation: Allocation decision from Stage 1
        chronos_forecast: Optional Chronos2 forecast
        max_lines: Max historical lines to include
        model: Claude model to use

    Returns:
        PricePrediction with entry/exit/stop levels
    """
    client = anthropic.Anthropic(api_key=_get_api_key())

    user_prompt = _build_price_prompt(
        symbol, pct_data, allocation, chronos_forecast, max_lines
    )

    try:
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=PRICE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        response_text = response.content[0].text
        logger.debug(f"Price prediction response for {symbol}: {response_text[:300]}")

        parsed = _parse_price_response(response_text)

        # Convert bps to pct
        entry_bps = int(parsed.get("entry_bps", 0))
        exit_bps = int(parsed.get("exit_bps", 0))
        stop_bps = int(parsed.get("stop_loss_bps", 0))

        return PricePrediction(
            symbol=symbol,
            direction=allocation.direction,
            entry_pct=entry_bps / 10000,  # bps to decimal
            exit_pct=exit_bps / 10000,
            stop_loss_pct=stop_bps / 10000,
            time_horizon=str(parsed.get("time_horizon", "intraday")),
            confidence=float(parsed.get("confidence", 0.5)),
            rationale=str(parsed.get("rationale", "")),
            last_close=pct_data.last_close,
        )

    except Exception as e:
        logger.error(f"Price prediction failed for {symbol}: {e}")
        # Return default values
        direction = allocation.direction
        if direction == "long":
            return PricePrediction(
                symbol=symbol,
                direction=direction,
                entry_pct=-0.002,  # -20 bps (smaller for hourly)
                exit_pct=0.008,   # +80 bps
                stop_loss_pct=-0.005,  # -50 bps
                time_horizon="intraday",
                confidence=0.3,
                rationale=f"Default hourly values (error: {e})",
                last_close=pct_data.last_close,
            )
        else:  # short
            return PricePrediction(
                symbol=symbol,
                direction=direction,
                entry_pct=0.002,  # +20 bps
                exit_pct=-0.008,  # -80 bps
                stop_loss_pct=0.005,  # +50 bps
                time_horizon="intraday",
                confidence=0.3,
                rationale=f"Default hourly values (error: {e})",
                last_close=pct_data.last_close,
            )


async def async_predict_prices(
    client: anthropic.AsyncAnthropic,
    symbol: str,
    pct_data: PctLineData,
    allocation: SymbolAllocation,
    chronos_forecast: Any | None = None,
    max_lines: int = 100,
    model: str = MODEL,
) -> PricePrediction:
    """Async Stage 2: Determine specific entry/exit/stop prices (HOURLY).

    Args:
        client: Async Anthropic client
        symbol: Symbol to predict prices for
        pct_data: Historical pct-change data
        allocation: Allocation decision from Stage 1
        chronos_forecast: Optional Chronos2 forecast
        max_lines: Max historical lines to include
        model: Claude model to use

    Returns:
        PricePrediction with entry/exit/stop levels
    """
    user_prompt = _build_price_prompt(
        symbol, pct_data, allocation, chronos_forecast, max_lines
    )

    try:
        response_text = await async_api_call(
            client=client,
            model=model,
            system=PRICE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=1024,
        )
        logger.debug(f"Price prediction response for {symbol}: {response_text[:300]}")

        parsed = parse_json_robust(response_text)

        # Convert bps to pct
        entry_bps = int(parsed.get("entry_bps", 0))
        exit_bps = int(parsed.get("exit_bps", 0))
        stop_bps = int(parsed.get("stop_loss_bps", 0))

        return PricePrediction(
            symbol=symbol,
            direction=allocation.direction,
            entry_pct=entry_bps / 10000,
            exit_pct=exit_bps / 10000,
            stop_loss_pct=stop_bps / 10000,
            time_horizon=str(parsed.get("time_horizon", "intraday")),
            confidence=float(parsed.get("confidence", 0.5)),
            rationale=str(parsed.get("rationale", "")),
            last_close=pct_data.last_close,
        )

    except Exception as e:
        logger.error(f"Async price prediction failed for {symbol}: {e}")
        direction = allocation.direction
        if direction == "long":
            return PricePrediction(
                symbol=symbol,
                direction=direction,
                entry_pct=-0.002,
                exit_pct=0.008,
                stop_loss_pct=-0.005,
                time_horizon="intraday",
                confidence=0.3,
                rationale=f"Default hourly values (error: {e})",
                last_close=pct_data.last_close,
            )
        else:
            return PricePrediction(
                symbol=symbol,
                direction=direction,
                entry_pct=0.002,
                exit_pct=-0.008,
                stop_loss_pct=0.005,
                time_horizon="intraday",
                confidence=0.3,
                rationale=f"Default hourly values (error: {e})",
                last_close=pct_data.last_close,
            )
