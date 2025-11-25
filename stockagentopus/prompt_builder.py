"""Prompt construction utilities for the Claude Opus trading agent."""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any, Mapping, Sequence

from stockagent.agentsimulator.data_models import AccountSnapshot
from stockagent.agentsimulator.market_data import MarketDataBundle
from stockagent.agentsimulator.prompt_builder import (
    plan_response_schema as _stateful_schema,
)
from stockagent.constants import DEFAULT_SYMBOLS, TRADING_FEE, CRYPTO_TRADING_FEE

try:
    from stockagentdeepseek_neural.forecaster import NeuralForecast
except ImportError:
    NeuralForecast = None  # type: ignore

SYSTEM_PROMPT = """You are an expert quantitative trader with decades of experience in algorithmic trading and market microstructure. You analyze market data with precision and produce disciplined trading plans.

Your approach combines:
1. Technical analysis of price patterns and momentum
2. Risk management with strict position sizing
3. Neural forecast integration when available
4. Opportunistic limit-order execution

You respond ONLY with valid JSON matching the required schema. No explanations outside the JSON."""

EXECUTION_GUIDANCE = """
Execution Guidelines:
- Provide limit-style entries and paired exits for the simulator
- Intraday gross exposure can reach 4× capital when conviction is high
- Positions must be reduced to 2× or lower by market close
- Borrowed capital accrues 6.75% annual interest on notional above cash
- Ensure projected edge covers all financing and transaction costs
- Entry prices should be realistic within the day's expected range
- Exit prices should target achievable profit levels based on volatility

Risk Controls:
- Never risk more than 2% of capital on a single trade
- Scale into positions when uncertainty is high
- Cut losses quickly when thesis is invalidated
- Take profits incrementally on winning positions
"""


def opus_plan_schema() -> dict[str, Any]:
    """Return the JSON schema for trading plan responses."""
    return _stateful_schema()


def _sanitize_market_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Remove absolute timestamps and replace with relative labels."""
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


def _format_forecast_section(forecasts: Mapping[str, "NeuralForecast"]) -> str:
    """Format neural forecasts for the prompt."""
    if not forecasts:
        return ""

    lines = ["\n## Neural Price Forecasts\n"]
    for symbol in sorted(forecasts.keys()):
        forecast = forecasts[symbol]
        combined_bits = ", ".join(f"{key}={value:.2f}" for key, value in forecast.combined.items())
        best_label = forecast.best_model or "blended"
        source_label = f" ({forecast.selection_source})" if forecast.selection_source else ""
        lines.append(f"**{symbol}**: {combined_bits}")
        lines.append(f"  - Best model: {best_label}{source_label}")

        for name, summary in forecast.model_summaries.items():
            model_bits = ", ".join(f"{key}={value:.2f}" for key, value in summary.forecasts.items())
            lines.append(f"  - {name} (MAE={summary.average_price_mae:.4f}): {model_bits}")
        lines.append("")

    return "\n".join(lines)


def _format_calibration_section(calibration: Mapping[str, float] | None) -> str:
    """Format calibration metrics for the prompt."""
    if not calibration:
        return ""

    lines = ["\n## Signal Calibration Metrics\n"]
    symbols_seen = set()
    for key in calibration:
        if "_calibration_slope" in key:
            symbol = key.replace("_calibration_slope", "")
            symbols_seen.add(symbol)

    for symbol in sorted(symbols_seen):
        slope = calibration.get(f"{symbol}_calibration_slope", 1.0)
        intercept = calibration.get(f"{symbol}_calibration_intercept", 0.0)
        raw_move = calibration.get(f"{symbol}_raw_expected_move_pct", 0.0)
        cal_move = calibration.get(f"{symbol}_calibrated_expected_move_pct", 0.0)
        lines.append(f"**{symbol}**:")
        lines.append(f"  - Calibration: slope={slope:.4f}, intercept={intercept:.6f}")
        lines.append(f"  - Expected move: raw={raw_move:.4%}, calibrated={cal_move:.4%}")
        lines.append("")

    return "\n".join(lines)


def _build_opus_prompt(
    market_data: MarketDataBundle,
    account_payload: dict[str, Any],
    target_date: date,
    symbols: Sequence[str] | None = None,
    include_market_history: bool = True,
) -> tuple[str, dict[str, Any]]:
    """Build a prompt for Opus trading agent using only provided account state."""
    symbols = list(symbols) if symbols is not None else list(DEFAULT_SYMBOLS)
    market_payload = market_data.to_payload() if include_market_history else {"symbols": list(symbols)}

    equity = float(account_payload.get("equity") or 0.0)
    cash = float(account_payload.get("cash") or 0.0)
    positions = account_payload.get("positions", [])
    max_notional = max(25_000.0, equity * 0.05)

    # Build position summary from account_payload only (not external state)
    position_summary = ""
    if positions:
        pos_lines = []
        for pos in positions:
            sym = pos.get("symbol", "?")
            qty = pos.get("quantity", 0)
            market_value = pos.get("market_value", 0)
            pos_lines.append(f"  - {sym}: {qty} shares, ${market_value:,.2f}")
        position_summary = f"\n- Current positions:\n" + "\n".join(pos_lines)
    else:
        position_summary = "\n- No current positions (all cash)"

    # Get latest close prices from market data
    price_lines = []
    for symbol in symbols:
        frame = market_data.get_symbol_bars(symbol)
        if not frame.empty:
            last_close = float(frame["close"].iloc[-1])
            last_high = float(frame["high"].iloc[-1])
            last_low = float(frame["low"].iloc[-1])
            price_lines.append(f"  - {symbol}: close=${last_close:.2f}, range=[${last_low:.2f}-${last_high:.2f}]")

    price_summary = "\n".join(price_lines) if price_lines else "  (no price data)"

    prompt = f"""
You are a disciplined multi-asset execution planner. Build a one-day trading plan for the upcoming trading session.

CRITICAL: Entry and exit prices MUST be based on the historical data provided below. Do NOT use your own knowledge of current market prices.

Context:
- You may trade the following symbols only: {', '.join(symbols)}.
- Account equity: ${equity:,.2f}
- Available cash: ${cash:,.2f}{position_summary}

LATEST PRICES (use these as reference for your limit orders):
{price_summary}

- Historical context: the payload includes the last {market_data.lookback_days} trading days of OHLC data per symbol.
- Your task is to generate BUY instructions to open new positions with entry_price limits, and optionally paired EXIT instructions with exit_price targets.
- Valid execution windows are `market_open` (09:30 ET) and `market_close` (16:00 ET). Choose one per instruction.
- Assume round-trip trading fees of {TRADING_FEE:.4%} for equities and {CRYPTO_TRADING_FEE:.4%} for crypto.
- Max notional per new instruction is ${max_notional:,.0f}; prefer smaller sizes for risk management.

Structured output requirements:
- Produce JSON matching the provided schema exactly.
- Return a single JSON object with these fields: target_date, instructions, risk_notes, metadata.
- Use action="buy" to open new long positions with entry_price as limit price and exit_price as target.
- Use action="exit" only to close existing positions.
- Use action="hold" to maintain existing positions without changes.
- Provide realistic limit prices based on the LATEST PRICES above - entry_price should be near or slightly below recent close, exit_price should be 1-3% above entry.
- Include `risk_notes` summarizing risk considerations in under 3 sentences.
- Populate `metadata` with a `capital_allocation_plan` string explaining cash allocation.
- Return ONLY the JSON object; do not include markdown code fences or extra text.
- Every instruction must include: symbol, action, quantity, execution_session, entry_price, exit_price, exit_reason, notes.
""".strip()

    user_payload: dict[str, Any] = {
        "account": account_payload,
        "market_data": market_payload,
        "target_date": target_date.isoformat(),
    }

    return prompt, user_payload


def build_opus_messages(
    *,
    market_data: MarketDataBundle,
    target_date: date,
    account_snapshot: AccountSnapshot | None = None,
    account_payload: Mapping[str, Any] | None = None,
    symbols: Sequence[str] | None = None,
    include_market_history: bool = True,
    forecasts: Mapping[str, "NeuralForecast"] | None = None,
    calibration: Mapping[str, float] | None = None,
) -> list[dict[str, str]]:
    """Assemble Claude Opus chat messages with trading context."""
    if account_payload is None:
        if account_snapshot is None:
            raise ValueError("account_snapshot or account_payload must be provided.")
        account_payload = account_snapshot.to_payload()

    # Use our own prompt builder that doesn't load external state
    prompt_text, payload = _build_opus_prompt(
        market_data=market_data,
        account_payload=dict(account_payload),
        target_date=target_date,
        symbols=symbols,
        include_market_history=include_market_history,
    )

    if EXECUTION_GUIDANCE not in prompt_text:
        prompt_text = f"{prompt_text}\n{EXECUTION_GUIDANCE}"

    prompt_text += (
        "\n\nHistorical data uses relative day labels (Day-Mon, Day-Tue, etc.) "
        "rather than calendar dates. Focus on price patterns and returns."
    )

    if forecasts:
        prompt_text += _format_forecast_section(forecasts)

    if calibration:
        prompt_text += _format_calibration_section(calibration)

    sanitized_payload = _sanitize_market_payload(payload)

    if forecasts:
        sanitized_payload["neural_forecasts"] = {
            symbol: {
                "combined": forecast.combined,
                "best_model": forecast.best_model,
                "selection_source": forecast.selection_source,
                "models": {
                    name: {
                        "mae": summary.average_price_mae,
                        "forecasts": summary.forecasts,
                        "config": summary.config_name,
                    }
                    for name, summary in forecast.model_summaries.items()
                },
            }
            for symbol, forecast in forecasts.items()
        }

    payload_json = json.dumps(sanitized_payload, ensure_ascii=False, indent=2)

    return [
        {"role": "user", "content": prompt_text},
        {"role": "user", "content": f"Market data payload:\n```json\n{payload_json}\n```"},
    ]


__all__ = ["SYSTEM_PROMPT", "build_opus_messages", "opus_plan_schema"]
