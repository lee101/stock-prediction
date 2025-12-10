"""Stage 2: Per-symbol specific buy/sell price prediction.

Given a single symbol's data and allocation decision, determine:
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
            from env_real import ANTHROPIC_API_KEY, CLAUDE_API_KEY
            key = ANTHROPIC_API_KEY or CLAUDE_API_KEY
            if key:
                return key
        except ImportError:
            pass
        raise RuntimeError("API key required: set ANTHROPIC_API_KEY or CLAUDE_API_KEY")
    return key


MODEL = "claude-sonnet-4-20250514"


PRICE_SYSTEM_PROMPT = """You are an aggressive price analyst maximizing 1-day PnL.

INPUT:
- Historical price in BASIS POINTS (bps): close_bps, high_bps, low_bps per day
  (100 bps = 1%, so +250 bps = +2.5% move)
- Chronos2 forecast (expected return in bps)
- Direction: long or short
- Allocation amount
- Asset type: crypto or stock

TASK:
Set entry, exit, and stop prices in BASIS POINTS from current close.

OUTPUT JSON FORMAT:
{
  "entry_bps": int,       // bps from close (e.g., -80 = 0.8% below close)
  "exit_bps": int,        // bps from close (e.g., +250 = 2.5% above close)
  "stop_loss_bps": int,   // bps from close (e.g., -150 = 1.5% below close)
  "time_horizon": "intraday" or "swing",
  "confidence": 0.0-1.0,
  "rationale": "brief explanation"
}

RULES FOR LONG (buy low, sell high):
- entry_bps: -50 to -100 (buy on small dip) or 0 (market order)
- exit_bps: +150 to +400 (profit target)
- stop_loss_bps: -100 to -250 (loss limit)

RULES FOR SHORT (sell high, buy back low) - STOCKS ONLY:
- entry_bps: +50 to +100 (short on rally) or 0
- exit_bps: -150 to -400 (cover at profit)
- stop_loss_bps: +100 to +250 (cover at loss)

TRADING FEES TO CONSIDER:
- STOCKS: ~10 bps round-trip (5 bps per side - factor in for tight trades)
- CRYPTO: ~16 bps round-trip (8 bps per side - must clear this hurdle)
  - For crypto, set exit_bps at least +30 bps to cover fees and profit

BE AGGRESSIVE:
- Focus on maximizing expected return, not minimizing risk
- Wider targets > conservative targets
- High volatility = opportunity
- For crypto: account for higher fees in targets

RESPOND WITH JSON ONLY."""


@dataclass
class PricePrediction:
    """Price prediction for a single symbol."""
    symbol: str
    direction: str  # "long" or "short"
    entry_pct: float  # % from current close for entry
    exit_pct: float  # % from current close for exit
    stop_loss_pct: float  # % from current close for stop
    time_horizon: str  # "intraday" or "swing"
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
    return f"expected_return={exp_ret_bps:+.0f}bps, volatility={vol_bps:.0f}bps"


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
    max_lines: int = 300,
) -> str:
    """Build the user prompt for price prediction using basis points."""
    asset_type = "crypto" if is_crypto(symbol) else "stock"
    fee_note = "~8 bps round-trip" if is_crypto(symbol) else "~0.5 bps round-trip"

    parts = [
        f"## ASSET",
        f"Type: {asset_type.upper()}",
        f"Direction: {allocation.direction.upper()}",
        f"Allocation: {allocation.alloc:.0%} of portfolio",
        f"Trading fees: {fee_note}",
        "",
    ]

    # Add Chronos2 forecast if available (in bps)
    if chronos_forecast:
        parts.append(f"Forecast: {_format_chronos_forecast_bps(chronos_forecast)}")
        parts.append("")

    # Add historical data as basis points
    lines = pct_data.lines.split("\n")
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    bps_lines = _pct_to_bps("\n".join(lines))

    parts.append("Historical (close_bps,high_bps,low_bps):")
    parts.append(bps_lines)
    parts.append("")
    parts.append("---")
    parts.append("Set entry/exit/stop in BASIS POINTS from close. JSON only.")

    return "\n".join(parts)


def _parse_json_response(text: str) -> dict[str, Any]:
    """Extract JSON from response, with robust fallback for truncated responses."""
    text = text.strip()

    # Try direct parse
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

    # Fallback: extract individual fields via regex (for truncated JSON)
    result = {}

    # Extract basis point values
    for field in ["entry_bps", "exit_bps", "stop_loss_bps"]:
        match = re.search(rf'"{field}":\s*([+-]?\d+)', text)
        if match:
            result[field] = int(match.group(1))

    # Extract legacy pct values
    for field in ["entry_pct", "exit_pct", "stop_loss_pct"]:
        match = re.search(rf'"{field}":\s*([+-]?\d+\.?\d*)', text)
        if match:
            result[field] = float(match.group(1))

    # Extract confidence
    match = re.search(r'"confidence":\s*(\d+\.?\d*)', text)
    if match:
        result["confidence"] = float(match.group(1))

    # Extract time_horizon
    match = re.search(r'"time_horizon":\s*"(\w+)"', text)
    if match:
        result["time_horizon"] = match.group(1)

    if result:
        return result

    raise ValueError(f"Could not parse JSON from: {text[:300]}")


def predict_price_single(
    symbol: str,
    pct_data: PctLineData,
    allocation: SymbolAllocation,
    chronos_forecast: Any | None = None,
    max_lines: int = 300,
    model: str = MODEL,
) -> PricePrediction:
    """Get price prediction for a single symbol.

    Args:
        symbol: Stock/crypto symbol
        pct_data: Historical pct-change data
        allocation: Portfolio allocation decision from Stage 1
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
        logger.debug(f"Price prediction for {symbol}: {response_text[:300]}")

        parsed = _parse_json_response(response_text)

        # Model returns basis points (e.g., -80 = -0.8%, +250 = +2.5%)
        # Convert bps to decimal: divide by 10000
        # Also support legacy pct format for backwards compatibility
        raw_entry = float(parsed.get("entry_bps", parsed.get("entry_pct", 0.0)))
        raw_exit = float(parsed.get("exit_bps", parsed.get("exit_pct", 0.0)))
        raw_stop = float(parsed.get("stop_loss_bps", parsed.get("stop_loss_pct", 0.0)))

        max_abs = max(abs(raw_entry), abs(raw_exit), abs(raw_stop))
        if max_abs > 15:  # Basis points (e.g., -80, +250)
            entry_pct = raw_entry / 10000.0
            exit_pct = raw_exit / 10000.0
            stop_loss_pct = raw_stop / 10000.0
        elif max_abs > 0.15:  # Percentage format (e.g., 2.5 = 2.5%)
            entry_pct = raw_entry / 100.0
            exit_pct = raw_exit / 100.0
            stop_loss_pct = raw_stop / 100.0
        else:  # Already decimal format (e.g., 0.025 = 2.5%)
            entry_pct = raw_entry
            exit_pct = raw_exit
            stop_loss_pct = raw_stop

        return PricePrediction(
            symbol=symbol,
            direction=allocation.direction,
            entry_pct=entry_pct,
            exit_pct=exit_pct,
            stop_loss_pct=stop_loss_pct,
            time_horizon=str(parsed.get("time_horizon", "intraday")),
            confidence=float(parsed.get("confidence", 0.0)),
            rationale=str(parsed.get("rationale", "")),
            last_close=pct_data.last_close,
        )

    except Exception as e:
        logger.error(f"Price prediction failed for {symbol}: {e}")
        # Return conservative default
        is_long = allocation.direction == "long"
        return PricePrediction(
            symbol=symbol,
            direction=allocation.direction,
            entry_pct=0.0,  # market order
            exit_pct=0.01 if is_long else -0.01,  # 1% target
            stop_loss_pct=-0.02 if is_long else 0.02,  # 2% stop
            time_horizon="intraday",
            confidence=0.0,
            rationale=f"Error fallback: {e}",
            last_close=pct_data.last_close,
        )


def predict_prices(
    pct_data: Mapping[str, PctLineData],
    allocations: Mapping[str, SymbolAllocation],
    chronos_forecasts: Mapping[str, Any] | None = None,
    max_lines: int = 300,
    model: str = MODEL,
) -> dict[str, PricePrediction]:
    """Get price predictions for all allocated symbols (parallel requests).

    Args:
        pct_data: Dict of symbol -> PctLineData
        allocations: Dict of symbol -> SymbolAllocation from Stage 1
        chronos_forecasts: Optional Chronos2 forecasts per symbol
        max_lines: Max historical lines per symbol
        model: Claude model to use

    Returns:
        Dict of symbol -> PricePrediction
    """
    predictions: dict[str, PricePrediction] = {}

    # Process each allocation
    for symbol, allocation in allocations.items():
        if symbol not in pct_data:
            continue

        chronos = chronos_forecasts.get(symbol) if chronos_forecasts else None

        prediction = predict_price_single(
            symbol=symbol,
            pct_data=pct_data[symbol],
            allocation=allocation,
            chronos_forecast=chronos,
            max_lines=max_lines,
            model=model,
        )
        predictions[symbol] = prediction

        logger.info(
            f"{symbol} ({allocation.direction}): "
            f"entry {prediction.entry_pct:+.2%}, "
            f"exit {prediction.exit_pct:+.2%}, "
            f"stop {prediction.stop_loss_pct:+.2%}, "
            f"conf {prediction.confidence:.2f}"
        )

    return predictions


async def async_predict_price_single(
    client: anthropic.AsyncAnthropic,
    symbol: str,
    pct_data: PctLineData,
    allocation: SymbolAllocation,
    chronos_forecast: Any | None = None,
    max_lines: int = 300,
    model: str = MODEL,
) -> PricePrediction:
    """Async get price prediction for a single symbol.

    Args:
        client: Async Anthropic client
        symbol: Stock/crypto symbol
        pct_data: Historical pct-change data
        allocation: Portfolio allocation decision from Stage 1
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

        logger.debug(f"Price prediction for {symbol}: {response_text[:300]}")

        parsed = parse_json_robust(response_text)

        # Model returns basis points (e.g., -80 = -0.8%, +250 = +2.5%)
        raw_entry = float(parsed.get("entry_bps", parsed.get("entry_pct", 0.0)))
        raw_exit = float(parsed.get("exit_bps", parsed.get("exit_pct", 0.0)))
        raw_stop = float(parsed.get("stop_loss_bps", parsed.get("stop_loss_pct", 0.0)))

        max_abs = max(abs(raw_entry), abs(raw_exit), abs(raw_stop))
        if max_abs > 15:  # Basis points
            entry_pct = raw_entry / 10000.0
            exit_pct = raw_exit / 10000.0
            stop_loss_pct = raw_stop / 10000.0
        elif max_abs > 0.15:  # Percentage format
            entry_pct = raw_entry / 100.0
            exit_pct = raw_exit / 100.0
            stop_loss_pct = raw_stop / 100.0
        else:  # Already decimal
            entry_pct = raw_entry
            exit_pct = raw_exit
            stop_loss_pct = raw_stop

        return PricePrediction(
            symbol=symbol,
            direction=allocation.direction,
            entry_pct=entry_pct,
            exit_pct=exit_pct,
            stop_loss_pct=stop_loss_pct,
            time_horizon=str(parsed.get("time_horizon", "intraday")),
            confidence=float(parsed.get("confidence", 0.0)),
            rationale=str(parsed.get("rationale", "")),
            last_close=pct_data.last_close,
        )

    except Exception as e:
        logger.error(f"Async price prediction failed for {symbol}: {e}")
        is_long = allocation.direction == "long"
        return PricePrediction(
            symbol=symbol,
            direction=allocation.direction,
            entry_pct=0.0,
            exit_pct=0.01 if is_long else -0.01,
            stop_loss_pct=-0.02 if is_long else 0.02,
            time_horizon="intraday",
            confidence=0.0,
            rationale=f"Error fallback: {e}",
            last_close=pct_data.last_close,
        )


async def async_predict_prices(
    client: anthropic.AsyncAnthropic,
    pct_data: Mapping[str, PctLineData],
    allocations: Mapping[str, SymbolAllocation],
    chronos_forecasts: Mapping[str, Any] | None = None,
    max_lines: int = 300,
    model: str = MODEL,
) -> dict[str, PricePrediction]:
    """Async get price predictions for all allocated symbols (parallel requests).

    Args:
        client: Async Anthropic client
        pct_data: Dict of symbol -> PctLineData
        allocations: Dict of symbol -> SymbolAllocation from Stage 1
        chronos_forecasts: Optional Chronos2 forecasts per symbol
        max_lines: Max historical lines per symbol
        model: Claude model to use

    Returns:
        Dict of symbol -> PricePrediction
    """
    # Build list of coroutines for parallel execution
    tasks = []
    symbols = []

    for symbol, allocation in allocations.items():
        if symbol not in pct_data:
            continue

        chronos = chronos_forecasts.get(symbol) if chronos_forecasts else None
        symbols.append(symbol)
        tasks.append(
            async_predict_price_single(
                client=client,
                symbol=symbol,
                pct_data=pct_data[symbol],
                allocation=allocation,
                chronos_forecast=chronos,
                max_lines=max_lines,
                model=model,
            )
        )

    # Run all predictions in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)

    predictions: dict[str, PricePrediction] = {}
    for symbol, result in zip(symbols, results):
        if isinstance(result, Exception):
            logger.error(f"Price prediction failed for {symbol}: {result}")
            allocation = allocations[symbol]
            is_long = allocation.direction == "long"
            predictions[symbol] = PricePrediction(
                symbol=symbol,
                direction=allocation.direction,
                entry_pct=0.0,
                exit_pct=0.01 if is_long else -0.01,
                stop_loss_pct=-0.02 if is_long else 0.02,
                time_horizon="intraday",
                confidence=0.0,
                rationale=f"Error: {result}",
                last_close=pct_data[symbol].last_close,
            )
        else:
            predictions[symbol] = result
            logger.info(
                f"{symbol} ({allocations[symbol].direction}): "
                f"entry {result.entry_pct:+.2%}, "
                f"exit {result.exit_pct:+.2%}, "
                f"stop {result.stop_loss_pct:+.2%}, "
                f"conf {result.confidence:.2f}"
            )

    return predictions
