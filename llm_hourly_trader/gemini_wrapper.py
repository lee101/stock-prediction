"""Gemini 3.1 Flash Lite wrapper for hourly trading decisions."""

from __future__ import annotations

import json
import os
import re
import time
import warnings
from dataclasses import dataclass
from typing import Optional

warnings.filterwarnings("ignore", message=".*is not a valid ThinkingLevel.*")

from google import genai
from google.genai import types

from llm_hourly_trader.cache import get_cached, set_cached


@dataclass
class TradePlan:
    direction: str  # "long", "short", or "hold"
    buy_price: float  # limit entry price (0 = no entry)
    sell_price: float  # limit exit / take-profit price (0 = no exit)
    confidence: float  # 0-1 how confident
    reasoning: str  # brief explanation


STRUCTURED_SCHEMA = genai.types.Schema(
    type=genai.types.Type.OBJECT,
    required=["direction", "buy_price", "sell_price", "confidence", "reasoning"],
    properties={
        "direction": genai.types.Schema(
            type=genai.types.Type.STRING,
            description="One of: long, short, hold",
        ),
        "buy_price": genai.types.Schema(
            type=genai.types.Type.STRING,
            description="Limit buy/entry price as a number string, or 0 if no entry",
        ),
        "sell_price": genai.types.Schema(
            type=genai.types.Type.STRING,
            description="Limit sell/take-profit price as a number string, or 0 if no exit",
        ),
        "confidence": genai.types.Schema(
            type=genai.types.Type.STRING,
            description="Confidence 0.0 to 1.0 as a string",
        ),
        "reasoning": genai.types.Schema(
            type=genai.types.Type.STRING,
            description="Brief reasoning for the trade decision",
        ),
    },
)


# ---------------------------------------------------------------------------
# Prompt variants
# ---------------------------------------------------------------------------

def _format_history_table(history_rows: list[dict], n: int = 24) -> str:
    lines = [
        "timestamp | open | high | low | close | volume",
        "---|---|---|---|---|---",
    ]
    for row in history_rows[-n:]:
        lines.append(
            f"{row['timestamp']} | {row['open']:.2f} | {row['high']:.2f} | "
            f"{row['low']:.2f} | {row['close']:.2f} | {row.get('volume', 0):.4f}"
        )
    return "\n".join(lines)


def _format_forecasts(
    forecast_1h: Optional[dict],
    forecast_24h: Optional[dict],
) -> str:
    lines = []
    if forecast_1h:
        lines.append(
            f"1h ahead: close={forecast_1h['predicted_close_p50']:.2f} "
            f"(p10={forecast_1h['predicted_close_p10']:.2f}, p90={forecast_1h['predicted_close_p90']:.2f}), "
            f"high={forecast_1h['predicted_high_p50']:.2f}, low={forecast_1h['predicted_low_p50']:.2f}"
        )
    if forecast_24h:
        lines.append(
            f"24h ahead: close={forecast_24h['predicted_close_p50']:.2f} "
            f"(p10={forecast_24h['predicted_close_p10']:.2f}, p90={forecast_24h['predicted_close_p90']:.2f}), "
            f"high={forecast_24h['predicted_high_p50']:.2f}, low={forecast_24h['predicted_low_p50']:.2f}"
        )
    if not lines:
        lines.append("(no forecasts available for this timestamp)")
    return "\n".join(lines)


def build_prompt(
    symbol: str,
    history_rows: list[dict],
    forecast_1h: Optional[dict],
    forecast_24h: Optional[dict],
    current_position: str,
    cash: float,
    equity: float,
    allowed_directions: list[str],
    asset_class: str = "crypto",
    maker_fee: float = 0.0008,
    variant: str = "default",
) -> str:
    """Build the prompt showing hourly history + Chronos forecasts."""
    last_close = history_rows[-1]["close"] if history_rows else 0
    dir_str = " or ".join(d.upper() for d in allowed_directions) + " or HOLD"
    fee_bps = maker_fee * 10000

    if variant == "conservative":
        return _prompt_conservative(
            symbol, history_rows, forecast_1h, forecast_24h,
            current_position, cash, equity, dir_str, fee_bps, last_close, asset_class,
        )
    elif variant == "aggressive":
        return _prompt_aggressive(
            symbol, history_rows, forecast_1h, forecast_24h,
            current_position, cash, equity, dir_str, fee_bps, last_close, asset_class,
        )
    else:
        return _prompt_default(
            symbol, history_rows, forecast_1h, forecast_24h,
            current_position, cash, equity, dir_str, fee_bps, last_close, asset_class,
        )


def _prompt_default(
    symbol, history_rows, forecast_1h, forecast_24h,
    current_position, cash, equity, dir_str, fee_bps, last_close, asset_class,
) -> str:
    asset_label = "cryptocurrency" if asset_class == "crypto" else "stock"
    return f"""You are a {asset_label} trader. You trade {symbol} on hourly bars.
Current position: {current_position}
Cash: ${cash:,.2f} | Portfolio equity: ${equity:,.2f}

## Recent hourly OHLCV (last 24 bars):
{_format_history_table(history_rows)}

## Chronos2 ML Forecasts:
{_format_forecasts(forecast_1h, forecast_24h)}

Current price: ${last_close:,.2f}

## Instructions:
Decide: {dir_str}.
If entering, set buy_price slightly below current for longs (or slightly above for shorts).
Set sell_price as your take-profit target.
If no good opportunity, direction=hold, prices=0.
Be selective - only trade when you see a clear edge.
Fees: {fee_bps:.0f}bp per side. You need to beat fees to profit."""


def _prompt_conservative(
    symbol, history_rows, forecast_1h, forecast_24h,
    current_position, cash, equity, dir_str, fee_bps, last_close, asset_class,
) -> str:
    asset_label = "cryptocurrency" if asset_class == "crypto" else "stock"
    return f"""You are a cautious {asset_label} trader focused on capital preservation.
You trade {symbol} on hourly bars. You ONLY trade when the evidence is very strong.

Current position: {current_position}
Cash: ${cash:,.2f} | Portfolio equity: ${equity:,.2f}

## Recent hourly OHLCV (last 24 bars):
{_format_history_table(history_rows)}

## Chronos2 ML Forecasts:
{_format_forecasts(forecast_1h, forecast_24h)}

Current price: ${last_close:,.2f}

## Rules:
- Decide: {dir_str}
- HOLD unless BOTH the price trend AND Chronos forecast agree on direction
- Require at least 0.3% expected move to justify entry (fees are {fee_bps:.0f}bp per side)
- Set tight take-profits (0.3-0.5% from entry)
- If in doubt, HOLD. Capital preservation > missing opportunities.
- confidence should reflect how strong the signal is (0.0-1.0)"""


def _prompt_aggressive(
    symbol, history_rows, forecast_1h, forecast_24h,
    current_position, cash, equity, dir_str, fee_bps, last_close, asset_class,
) -> str:
    asset_label = "cryptocurrency" if asset_class == "crypto" else "stock"
    return f"""You are an active {asset_label} trader seeking to maximize returns.
You trade {symbol} on hourly bars. You look for any reasonable opportunity.

Current position: {current_position}
Cash: ${cash:,.2f} | Portfolio equity: ${equity:,.2f}

## Recent hourly OHLCV (last 24 bars):
{_format_history_table(history_rows)}

## Chronos2 ML Forecasts:
{_format_forecasts(forecast_1h, forecast_24h)}

Current price: ${last_close:,.2f}

## Rules:
- Decide: {dir_str}
- Trade when you see momentum, mean-reversion, or forecast-aligned opportunities
- Set take-profits at 0.5-1.5% from entry
- Use high confidence (0.7-1.0) when trend and forecast align
- Fees: {fee_bps:.0f}bp per side
- Be decisive - missed opportunities cost money too"""


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------

def call_gemini_structured(
    prompt: str,
    model: str = "gemini-3.1-flash-lite-preview",
    max_retries: int = 5,
) -> TradePlan:
    """Call Gemini with structured JSON output."""
    # Check cache first
    cached = get_cached(model, prompt)
    if cached is not None:
        return TradePlan(**cached)

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=STRUCTURED_SCHEMA,
        temperature=0.3,
    )

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
                config=config,
            )
            data = json.loads(response.text)
            plan = TradePlan(
                direction=data.get("direction", "hold").lower().strip(),
                buy_price=float(data.get("buy_price", 0) or 0),
                sell_price=float(data.get("sell_price", 0) or 0),
                confidence=float(data.get("confidence", 0) or 0),
                reasoning=data.get("reasoning", ""),
            )
            set_cached(model, prompt, plan.__dict__)
            return plan
        except Exception as e:
            err_str = str(e)
            if "429" in err_str:
                # Extract retry delay from error if available
                import re as _re
                delay_match = _re.search(r"retry in (\d+\.?\d*)", err_str, _re.IGNORECASE)
                if delay_match:
                    wait = float(delay_match.group(1)) + 1
                else:
                    wait = 10 * (attempt + 1)
                if attempt < max_retries - 1:
                    time.sleep(wait)
                    continue
            elif attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            return TradePlan("hold", 0, 0, 0, f"API error: {e}")


def _parse_trade_plan_from_text(text: str) -> TradePlan:
    """Try to extract a TradePlan from freeform text containing JSON."""
    json_patterns = [
        r'\{[^{}]*"direction"[^{}]*\}',
        r'```json\s*(\{.*?\})\s*```',
    ]
    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in reversed(matches):
            try:
                data = json.loads(match)
                return TradePlan(
                    direction=str(data.get("direction", "hold")).lower().strip(),
                    buy_price=float(data.get("buy_price", 0) or 0),
                    sell_price=float(data.get("sell_price", 0) or 0),
                    confidence=float(data.get("confidence", 0) or 0),
                    reasoning=str(data.get("reasoning", "")),
                )
            except (json.JSONDecodeError, ValueError):
                continue

    text_lower = text.lower()
    if "long" in text_lower and "short" not in text_lower:
        return TradePlan("long", 0, 0, 0.3, "Parsed from text: bullish signal")
    elif "short" in text_lower and "long" not in text_lower:
        return TradePlan("short", 0, 0, 0.3, "Parsed from text: bearish signal")
    return TradePlan("hold", 0, 0, 0, "Could not parse response")
