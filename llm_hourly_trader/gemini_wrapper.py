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

try:
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover - exercised indirectly in prompt-only tests
    genai = None
    types = None

from llm_hourly_trader.cache import get_cached, set_cached


@dataclass
class TradePlan:
    direction: str  # "long", "short", or "hold"
    buy_price: float  # limit entry price (0 = no entry)
    sell_price: float  # limit exit / take-profit price (0 = no exit)
    confidence: float  # 0-1 how confident
    reasoning: str = ""  # brief explanation (optional, saves output tokens)
    allocation_pct: float = 0.0  # 0-100, how much of equity to allocate


if genai is not None:
    STRUCTURED_SCHEMA = genai.types.Schema(
        type=genai.types.Type.OBJECT,
        required=["direction", "buy_price", "sell_price", "confidence"],
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
        },
    )
else:  # pragma: no cover - simple import fallback
    STRUCTURED_SCHEMA = None


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


def _format_forecasts_with_mae_bands(
    forecast_1h: Optional[dict],
    forecast_24h: Optional[dict],
    forecast_error_1h: Optional[dict],
    forecast_error_24h: Optional[dict],
) -> str:
    def _fmt_row(label: str, forecast: Optional[dict], error_band: Optional[dict]) -> Optional[str]:
        if not forecast:
            return None

        close = forecast["predicted_close_p50"]
        high = forecast.get("predicted_high_p50", close)
        low = forecast.get("predicted_low_p50", close)
        if error_band and error_band.get("mae_pct", 0) > 0:
            band_pct = float(error_band["mae_pct"])
            band_width = close * band_pct / 100.0
            band_low = close - band_width
            band_high = close + band_width
            samples = int(error_band.get("samples", 0))
            return (
                f"{label} ahead: close={close:.2f} "
                f"(historical MAE band [{band_low:.2f}, {band_high:.2f}], "
                f"MAE={band_pct:.2f}%, n={samples}), "
                f"high={high:.2f}, low={low:.2f}"
            )

        return (
            f"{label} ahead: close={close:.2f} "
            f"(p10={forecast['predicted_close_p10']:.2f}, p90={forecast['predicted_close_p90']:.2f}), "
            f"high={high:.2f}, low={low:.2f}"
        )

    lines = []
    row_1h = _fmt_row("1h", forecast_1h, forecast_error_1h)
    row_24h = _fmt_row("24h", forecast_24h, forecast_error_24h)
    if row_1h:
        lines.append(row_1h)
    if row_24h:
        lines.append(row_24h)
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
    **kwargs,
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
    elif variant == "position_context":
        return _prompt_position_context(
            symbol, history_rows, forecast_1h, forecast_24h,
            current_position, cash, equity, dir_str, fee_bps, last_close, asset_class,
            **kwargs,
        )
    elif variant == "no_forecast":
        return _prompt_position_context(
            symbol, history_rows, None, None,
            current_position, cash, equity, dir_str, fee_bps, last_close, asset_class,
            **kwargs,
        )
    elif variant == "h1_only":
        return _prompt_position_context(
            symbol, history_rows, forecast_1h, None,
            current_position, cash, equity, dir_str, fee_bps, last_close, asset_class,
            **kwargs,
        )
    elif variant == "h24_only":
        return _prompt_position_context(
            symbol, history_rows, None, forecast_24h,
            current_position, cash, equity, dir_str, fee_bps, last_close, asset_class,
            **kwargs,
        )
    elif variant == "uncertainty_gated":
        return _prompt_uncertainty_gated(
            symbol, history_rows, forecast_1h, forecast_24h,
            current_position, cash, equity, dir_str, fee_bps, last_close, asset_class,
            **kwargs,
        )
    elif variant == "uncertainty_strict":
        return _prompt_uncertainty_strict(
            symbol, history_rows, forecast_1h, forecast_24h,
            current_position, cash, equity, dir_str, fee_bps, last_close, asset_class,
            **kwargs,
        )
    elif variant in ("freeform", "less_prescriptive"):
        return _prompt_freeform(
            symbol, history_rows, forecast_1h, forecast_24h,
            current_position, cash, equity, dir_str, fee_bps, last_close, asset_class,
            **kwargs,
        )
    elif variant in ("mae_bands", "historical_mae_bands"):
        return _prompt_mae_bands(
            symbol, history_rows, forecast_1h, forecast_24h,
            current_position, cash, equity, dir_str, fee_bps, last_close, asset_class,
            **kwargs,
        )
    elif variant == "fractional":
        return _prompt_fractional(
            symbol, history_rows, forecast_1h, forecast_24h,
            current_position, cash, equity, dir_str, fee_bps, last_close, asset_class,
            **kwargs,
        )
    elif variant == "anonymized":
        return _prompt_anonymized(
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


def _prompt_position_context(
    symbol, history_rows, forecast_1h, forecast_24h,
    current_position, cash, equity, dir_str, fee_bps, last_close, asset_class,
    **kwargs,
) -> str:
    asset_label = "cryptocurrency" if asset_class == "crypto" else "stock"
    pos_info = kwargs.get("position_info", {})
    pos_section = ""
    if pos_info and pos_info.get("qty", 0) > 0:
        entry = pos_info.get("entry_price", 0)
        held = pos_info.get("held_hours", 0)
        pnl_pct = (last_close - entry) / entry * 100 if entry > 0 else 0
        pnl_usd = pos_info.get("qty", 0) * (last_close - entry)
        pos_section = f"""
## CURRENT POSITION:
- Holding {pos_info['qty']:.6f} {symbol} @ ${entry:.2f} entry
- Held for {held:.0f} hours (max hold: 6h, auto-close at limit)
- Unrealized P&L: ${pnl_usd:+.2f} ({pnl_pct:+.2f}%)
- Current value: ${pos_info['qty'] * last_close:.2f}

EXIT DECISION: Consider whether to hold, tighten take-profit, or exit now.
If the position is profitable and momentum is fading, take profits.
If approaching max hold time (6h), set a realistic exit price.
"""
    else:
        pos_section = "\n## CURRENT POSITION: Flat (no position)\n"

    return f"""You are solving an optimization problem: maximize risk-adjusted returns trading {symbol} ({asset_label}).

CONSTRAINTS:
- LONG ONLY (spot market)
- 1-hour decision intervals, max 6-hour hold time
- Transaction cost: {fee_bps:.0f}bp per side
- Objective: maximize Sortino ratio
{pos_section}
Cash: ${cash:,.2f} | Portfolio equity: ${equity:,.2f}

## Recent hourly OHLCV (last 24 bars):
{_format_history_table(history_rows)}

## Chronos2 ML Forecasts:
{_format_forecasts(forecast_1h, forecast_24h)}

Current price: ${last_close:,.2f}

Decide: {dir_str}.
If holding a position: set sell_price for take-profit. If no position: set buy_price for entry.
Only enter when expected_return > fees. Respond with JSON: {{"direction": "long" or "hold", "buy_price": <entry or 0>, "sell_price": <take-profit or 0>, "confidence": <0-1>}}"""


def _prompt_fractional(
    symbol, history_rows, forecast_1h, forecast_24h,
    current_position, cash, equity, dir_str, fee_bps, last_close, asset_class,
    **kwargs,
) -> str:
    asset_label = "cryptocurrency" if asset_class == "crypto" else "stock"
    pos_info = kwargs.get("position_info", {})
    pos_section = ""
    if pos_info and pos_info.get("qty", 0) > 0:
        entry = pos_info.get("entry_price", 0)
        held = pos_info.get("held_hours", 0)
        pnl_pct = (last_close - entry) / entry * 100 if entry > 0 else 0
        pos_section = f"""
## CURRENT POSITION:
- Holding {pos_info['qty']:.6f} {symbol} @ ${entry:.2f} (P&L: {pnl_pct:+.2f}%, held {held:.0f}h)
"""
    else:
        pos_section = "\n## CURRENT POSITION: Flat\n"

    return f"""You are optimizing risk-adjusted returns trading {symbol} ({asset_label}).

CONSTRAINTS:
- LONG ONLY, 1-hour intervals, max 6h hold, {fee_bps:.0f}bp fees per side
- You can PARTIALLY exit positions (fractional sizing)
{pos_section}
Cash: ${cash:,.2f} | Equity: ${equity:,.2f}

## Recent OHLCV (last 24 bars):
{_format_history_table(history_rows)}

## Chronos2 ML Forecasts:
{_format_forecasts(forecast_1h, forecast_24h)}

Price: ${last_close:,.2f}

FRACTIONAL SIZING: You can scale in/out gradually.
- exit_pct: fraction of current position to sell (0.0 = hold all, 1.0 = sell all, 0.5 = sell half)
- For entries: confidence scales position size (0.3 = small, 1.0 = full allocation)

Respond with JSON: {{"direction": "long" or "hold", "buy_price": <entry or 0>, "sell_price": <exit price or 0>, "exit_pct": <0.0-1.0>, "confidence": <0-1>}}"""


def _ci_spread_pct(fc: Optional[dict], current_price: float) -> float:
    if not fc or "predicted_close_p90" not in fc or "predicted_close_p10" not in fc:
        return 0.0
    return (fc["predicted_close_p90"] - fc["predicted_close_p10"]) / current_price * 100


def _uncertainty_label(spread_pct: float) -> str:
    if spread_pct < 0.5:
        return "LOW"
    elif spread_pct < 2.0:
        return "MODERATE"
    return "HIGH"


def _prompt_uncertainty_gated(
    symbol, history_rows, forecast_1h, forecast_24h,
    current_position, cash, equity, dir_str, fee_bps, last_close, asset_class,
    **kwargs,
) -> str:
    asset_label = "cryptocurrency" if asset_class == "crypto" else "stock"
    pos_info = kwargs.get("position_info", {})
    pos_section = ""
    if pos_info and pos_info.get("qty", 0) > 0:
        entry = pos_info.get("entry_price", 0)
        held = pos_info.get("held_hours", 0)
        pnl_pct = (last_close - entry) / entry * 100 if entry > 0 else 0
        pnl_usd = pos_info.get("qty", 0) * (last_close - entry)
        pos_section = f"""
## CURRENT POSITION:
- Holding {pos_info['qty']:.6f} {symbol} @ ${entry:.2f} entry
- Held for {held:.0f} hours (max hold: 6h)
- Unrealized P&L: ${pnl_usd:+.2f} ({pnl_pct:+.2f}%)

EXIT DECISION: If profitable and momentum fading, take profits.
"""
    else:
        pos_section = "\n## CURRENT POSITION: Flat (no position)\n"

    ci_1h = _ci_spread_pct(forecast_1h, last_close)
    ci_24h = _ci_spread_pct(forecast_24h, last_close)
    unc_1h = _uncertainty_label(ci_1h)
    unc_24h = _uncertainty_label(ci_24h)

    return f"""You are solving an optimization problem: maximize risk-adjusted returns trading {symbol} ({asset_label}).

CONSTRAINTS:
- LONG ONLY (spot market), 1-hour intervals, max 6h hold, {fee_bps:.0f}bp fees
- Objective: maximize Sortino ratio
{pos_section}
Cash: ${cash:,.2f} | Portfolio equity: ${equity:,.2f}

## Recent hourly OHLCV (last 24 bars):
{_format_history_table(history_rows)}

## Chronos2 ML Forecasts:
{_format_forecasts(forecast_1h, forecast_24h)}

## FORECAST UNCERTAINTY:
1h CI spread: {ci_1h:.2f}% -> {unc_1h} uncertainty
24h CI spread: {ci_24h:.2f}% -> {unc_24h} uncertainty

RISK RULE: Scale your confidence inversely to uncertainty.
- LOW uncertainty: trade normally (confidence 0.5-1.0)
- MODERATE uncertainty: reduce confidence by 30%
- HIGH uncertainty: reduce confidence by 60%, consider holding

Current price: ${last_close:,.2f}

Decide: {dir_str}.
Respond with JSON: {{"direction": "long" or "hold", "buy_price": <entry or 0>, "sell_price": <take-profit or 0>, "confidence": <0-1>}}"""


def _prompt_uncertainty_strict(
    symbol, history_rows, forecast_1h, forecast_24h,
    current_position, cash, equity, dir_str, fee_bps, last_close, asset_class,
    **kwargs,
) -> str:
    asset_label = "cryptocurrency" if asset_class == "crypto" else "stock"
    pos_info = kwargs.get("position_info", {})
    pos_section = ""
    if pos_info and pos_info.get("qty", 0) > 0:
        entry = pos_info.get("entry_price", 0)
        held = pos_info.get("held_hours", 0)
        pnl_pct = (last_close - entry) / entry * 100 if entry > 0 else 0
        pnl_usd = pos_info.get("qty", 0) * (last_close - entry)
        pos_section = f"""
## CURRENT POSITION:
- Holding {pos_info['qty']:.6f} {symbol} @ ${entry:.2f} entry
- Held for {held:.0f} hours (max hold: 6h)
- Unrealized P&L: ${pnl_usd:+.2f} ({pnl_pct:+.2f}%)

EXIT DECISION: If profitable and momentum fading, take profits.
"""
    else:
        pos_section = "\n## CURRENT POSITION: Flat (no position)\n"

    ci_1h = _ci_spread_pct(forecast_1h, last_close)
    ci_24h = _ci_spread_pct(forecast_24h, last_close)
    unc_1h = _uncertainty_label(ci_1h)
    unc_24h = _uncertainty_label(ci_24h)

    return f"""You are solving an optimization problem: maximize risk-adjusted returns trading {symbol} ({asset_label}).

CONSTRAINTS:
- LONG ONLY (spot market), 1-hour intervals, max 6h hold, {fee_bps:.0f}bp fees
- Objective: maximize Sortino ratio
{pos_section}
Cash: ${cash:,.2f} | Portfolio equity: ${equity:,.2f}

## Recent hourly OHLCV (last 24 bars):
{_format_history_table(history_rows)}

## Chronos2 ML Forecasts:
{_format_forecasts(forecast_1h, forecast_24h)}

## FORECAST UNCERTAINTY:
1h CI spread: {ci_1h:.2f}% -> {unc_1h} uncertainty
24h CI spread: {ci_24h:.2f}% -> {unc_24h} uncertainty

STRICT RISK RULES:
- If 24h CI spread > 2% (HIGH uncertainty): you MUST hold. Do NOT enter new positions.
- If 1h CI spread > 2%: you MUST hold.
- Only enter when BOTH 1h AND 24h uncertainty are MODERATE or below.
- When both are LOW: trade with full confidence.

Current price: ${last_close:,.2f}

Decide: {dir_str}.
Respond with JSON: {{"direction": "long" or "hold", "buy_price": <entry or 0>, "sell_price": <take-profit or 0>, "confidence": <0-1>}}"""


def _prompt_freeform(
    symbol, history_rows, forecast_1h, forecast_24h,
    current_position, cash, equity, dir_str, fee_bps, last_close, asset_class,
    **kwargs,
) -> str:
    asset_label = "cryptocurrency" if asset_class == "crypto" else "stock"
    pos_info = kwargs.get("position_info", {})
    pos_section = ""
    if pos_info and pos_info.get("qty", 0) > 0:
        entry = pos_info.get("entry_price", 0)
        held = pos_info.get("held_hours", 0)
        pnl_pct = (last_close - entry) / entry * 100 if entry > 0 else 0
        pnl_usd = pos_info.get("qty", 0) * (last_close - entry)
        pos_section = f"""
## CURRENT POSITION:
- Holding {pos_info['qty']:.6f} {symbol} @ ${entry:.2f} entry
- Held for {held:.0f} hours (max hold: 6h)
- Unrealized P&L: ${pnl_usd:+.2f} ({pnl_pct:+.2f}%)
"""
    else:
        pos_section = "\n## CURRENT POSITION: Flat (no position)\n"

    return f"""You are trading {symbol} ({asset_label}) on 1-hour bars.

Objective:
- Maximize risk-adjusted returns after fees.

Hard constraints:
- LONG ONLY (spot market)
- 1-hour decision intervals
- Max hold time: 6 hours
- Fees: {fee_bps:.0f}bp per side
{pos_section}
Cash: ${cash:,.2f} | Portfolio equity: ${equity:,.2f}

## Recent hourly OHLCV (last 24 bars):
{_format_history_table(history_rows)}

## Chronos2 ML Forecasts:
{_format_forecasts(forecast_1h, forecast_24h)}

Current price: ${last_close:,.2f}

Use the data however you think is best.
- If flat: decide whether there is enough edge to enter.
- If long: decide whether to keep holding, set a realistic take-profit, or effectively stand aside.
- Do not force trades. Only act when the expected edge is strong enough to clear fees and near-term noise.

Decide: {dir_str}.
Respond with JSON: {{"direction": "long" or "hold", "buy_price": <entry or 0>, "sell_price": <take-profit or 0>, "confidence": <0-1>}}"""


def _prompt_mae_bands(
    symbol, history_rows, forecast_1h, forecast_24h,
    current_position, cash, equity, dir_str, fee_bps, last_close, asset_class,
    **kwargs,
) -> str:
    asset_label = "cryptocurrency" if asset_class == "crypto" else "stock"
    pos_info = kwargs.get("position_info", {})
    forecast_error_1h = kwargs.get("forecast_error_1h")
    forecast_error_24h = kwargs.get("forecast_error_24h")
    pos_section = ""
    if pos_info and pos_info.get("qty", 0) > 0:
        entry = pos_info.get("entry_price", 0)
        held = pos_info.get("held_hours", 0)
        pnl_pct = (last_close - entry) / entry * 100 if entry > 0 else 0
        pnl_usd = pos_info.get("qty", 0) * (last_close - entry)
        pos_section = f"""
## CURRENT POSITION:
- Holding {pos_info['qty']:.6f} {symbol} @ ${entry:.2f} entry
- Held for {held:.0f} hours (max hold: 6h)
- Unrealized P&L: ${pnl_usd:+.2f} ({pnl_pct:+.2f}%)
"""
    else:
        pos_section = "\n## CURRENT POSITION: Flat (no position)\n"

    return f"""You are trading {symbol} ({asset_label}) on 1-hour bars.

Objective:
- Maximize risk-adjusted returns after fees.

Hard constraints:
- LONG ONLY (spot market)
- 1-hour decision intervals
- Max hold time: 6 hours
- Fees: {fee_bps:.0f}bp per side
{pos_section}
Cash: ${cash:,.2f} | Portfolio equity: ${equity:,.2f}

## Recent hourly OHLCV (last 24 bars):
{_format_history_table(history_rows)}

## Chronos2 Forecasts With Historical Error Bands:
{_format_forecasts_with_mae_bands(forecast_1h, forecast_24h, forecast_error_1h, forecast_error_24h)}

## Error-band note:
- The MAE bands above come from resolved historical forecasts available before now.
- Treat them as empirical calibration for typical forecast error, not as hard rules.

Current price: ${last_close:,.2f}

Use the data however you think is best.
- If flat: decide whether there is enough edge to enter.
- If long: decide whether to keep holding, set a realistic take-profit, or effectively stand aside.
- Do not force trades. Only act when the expected edge is strong enough to clear fees and near-term noise.

Decide: {dir_str}.
Respond with JSON: {{"direction": "long" or "hold", "buy_price": <entry or 0>, "sell_price": <take-profit or 0>, "confidence": <0-1>}}"""


def _prompt_anonymized(
    symbol, history_rows, forecast_1h, forecast_24h,
    current_position, cash, equity, dir_str, fee_bps, last_close, asset_class,
) -> str:
    return f"""You are solving an optimization problem: maximize risk-adjusted returns on a spot trading account.

CONSTRAINTS:
- LONG ONLY (spot market, no shorting)
- 1-hour decision intervals, max 6-hour hold time
- Transaction cost: {fee_bps:.0f}bp per side
- Objective: maximize Sortino ratio

ASSET: Asset_X
Current position: {current_position}
Cash: ${cash:,.2f} | Portfolio equity: ${equity:,.2f}

## Recent hourly OHLCV (last 24 bars):
{_format_history_table(history_rows)}

## ML Forecasts:
{_format_forecasts(forecast_1h, forecast_24h)}

Current price: ${last_close:,.2f}

Decide: LONG or HOLD.
Focus purely on price action, momentum, and forecast signals.
Do NOT consider what asset this might be - analyze only the numbers.
If entering, set buy_price slightly below current. Set sell_price as take-profit.
Only trade when expected_return > fees.

Respond with JSON: {{"direction": "long" or "hold", "buy_price": <entry or 0>, "sell_price": <take-profit or 0>, "confidence": <0-1>}}"""


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------

def call_gemini_structured(
    prompt: str,
    model: str = "gemini-3.1-flash-lite-preview",
    max_retries: int = 5,
) -> TradePlan:
    """Call Gemini with structured JSON output."""
    if genai is None or types is None or STRUCTURED_SCHEMA is None:
        raise RuntimeError(
            "google-genai is required for call_gemini_structured; install it in the active environment."
        )

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
