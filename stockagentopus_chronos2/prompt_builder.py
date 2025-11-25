"""Prompt construction for Claude Opus with Chronos2 forecasting integration."""

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

from .forecaster import Chronos2Forecast

SYSTEM_PROMPT = """You are a disciplined quantitative trader focused on HIGH-CONVICTION trades only.

CRITICAL RULES:
1. Use ONLY the prices provided in the prompt - never use training data knowledge
2. ONLY TRADE when expected return > 1% (to cover round-trip fees + profit margin)
3. Entry price = Last Close (current market price) - ensures order fills
4. Exit price = Median forecast (realistic target that actually fills)
5. SKIP stocks with expected return < 1% - fees will eat the profit
6. Better to make fewer high-quality trades than many marginal ones
7. Position size proportional to conviction: higher expected return = larger position

TRADING PHILOSOPHY:
- Quality over quantity: Only take trades with clear edge
- Realistic exits: Median is more achievable than 90th percentile
- Capital preservation: Avoid trades where fees exceed expected profit
- Round-trip fees are ~0.006% (0.003% each way) - need 1%+ return to be worthwhile

You respond ONLY with valid JSON matching the required schema."""


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


def _build_opus_prompt(
    market_data: MarketDataBundle,
    account_payload: dict[str, Any],
    target_date: date,
    symbols: Sequence[str] | None = None,
    include_market_history: bool = True,
    chronos2_forecasts: Mapping[str, Chronos2Forecast] | None = None,
) -> tuple[str, dict[str, Any], str]:
    """Build a prompt for Opus trading agent with Chronos2 forecasts."""
    symbols = list(symbols) if symbols is not None else list(DEFAULT_SYMBOLS)
    market_payload = market_data.to_payload() if include_market_history else {"symbols": list(symbols)}

    equity = float(account_payload.get("equity") or 0.0)
    cash = float(account_payload.get("cash") or 0.0)
    positions = account_payload.get("positions", [])
    max_notional_per_trade = max(25_000.0, equity * 0.10)  # 10% per trade max
    total_available = equity * 0.80  # Use up to 80% of capital

    # Build position summary
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

    # Build Chronos2 forecast table with explicit quantiles
    forecast_table = []
    if chronos2_forecasts:
        forecast_table.append("")
        forecast_table.append("## CHRONOS2 NEURAL PRICE FORECASTS")
        forecast_table.append("")
        forecast_table.append("| Symbol | Last Close (ENTRY) | 10th Pct | Median (EXIT TARGET) | 90th Pct | Expected Return |")
        forecast_table.append("|--------|------------|------------------|--------|-----------------|-----------------|")

        for symbol in sorted(chronos2_forecasts.keys()):
            f = chronos2_forecasts[symbol]
            exp_ret = f"{f.expected_return_pct:+.2%}"
            forecast_table.append(
                f"| {symbol} | ${f.last_close:.2f} | ${f.low_close:.2f} | ${f.predicted_close:.2f} | ${f.high_close:.2f} | {exp_ret} |"
            )

        forecast_table.append("")
        forecast_table.append("IMPORTANT - HOW TO USE THESE FORECASTS:")
        forecast_table.append("- Last Close = CURRENT price (use this for entry)")
        forecast_table.append("- 10th/90th percentiles = PREDICTED price RANGE over next trading period")
        forecast_table.append("- Expected Return = potential profit from current price to median forecast")
        forecast_table.append("")
        forecast_table.append("TRADING STRATEGY:")
        forecast_table.append("- SET entry_price at or NEAR the Last Close (current market price)")
        forecast_table.append("- SET exit_price at the Median forecast (realistic, achievable target)")
        forecast_table.append("- ONLY TRADE when Expected Return > 1% (to cover fees + profit)")
        forecast_table.append("- SKIP stocks with Expected Return < 1% - not worth the fees")
        forecast_table.append("- Quality over quantity: fewer high-conviction trades beat many marginal ones")
        forecast_table.append("")

        # Rank stocks by expected return
        ranked = sorted(chronos2_forecasts.items(), key=lambda x: x[1].expected_return_pct, reverse=True)
        forecast_table.append("ALLOCATION RANKING (by expected return):")
        for rank, (sym, f) in enumerate(ranked, 1):
            if f.expected_return_pct >= 0.01:  # >= 1% expected return
                suggested_alloc = min(max_notional_per_trade, total_available * 0.5)
                action = "TRADE"
            elif f.expected_return_pct > 0:
                suggested_alloc = 0
                action = "SKIP (< 1%)"
            else:
                suggested_alloc = 0
                action = "SKIP (negative)"
            forecast_table.append(f"  {rank}. {sym}: exp_return={f.expected_return_pct:+.2%}, {action}, notional=${suggested_alloc:,.0f}")
        forecast_table.append("")

    forecast_section = "\n".join(forecast_table)

    prompt = f"""
Build a trading plan for {target_date.isoformat()} to maximize profit.

ACCOUNT STATUS:
- Equity: ${equity:,.2f}
- Cash: ${cash:,.2f}{position_summary}
- Max notional per trade: ${max_notional_per_trade:,.0f}
- Total deployable: ${total_available:,.0f}

SYMBOLS AVAILABLE: {', '.join(symbols)}
TRADING FEE: {TRADING_FEE:.4%} per trade (must be covered by profit)
{forecast_section}

OUTPUT REQUIREMENTS:
1. Return a JSON object with: target_date, instructions, risk_notes, metadata
2. Each instruction needs: symbol, action, quantity, execution_session, entry_price, exit_price, exit_reason, notes
3. entry_price = limit buy price (at/near Last Close for fills)
4. exit_price = target sell price (at Median forecast - realistic target)
5. action must be "buy", "exit", or "hold"
6. execution_session must be "market_open" or "market_close"
7. quantity = integer number of shares
8. ONLY include stocks with expected return >= 1% - others are not worth the fees
9. Include risk_notes (1-2 sentences on risks)
10. Include metadata.capital_allocation_plan explaining your allocation logic
11. If NO stocks have expected return >= 1%, return empty instructions array (sit out the day)

CRITICAL: Entry prices near Last Close will fill. Exit at Median is realistic.
""".strip()

    user_payload: dict[str, Any] = {
        "account": account_payload,
        "market_data": market_payload,
        "target_date": target_date.isoformat(),
    }

    return prompt, user_payload, forecast_section


def build_opus_messages(
    *,
    market_data: MarketDataBundle,
    target_date: date,
    account_snapshot: AccountSnapshot | None = None,
    account_payload: Mapping[str, Any] | None = None,
    symbols: Sequence[str] | None = None,
    include_market_history: bool = True,
    chronos2_forecasts: Mapping[str, Chronos2Forecast] | None = None,
) -> list[dict[str, str]]:
    """Assemble Claude Opus chat messages with Chronos2 forecasting context."""
    if account_payload is None:
        if account_snapshot is None:
            raise ValueError("account_snapshot or account_payload must be provided.")
        account_payload = account_snapshot.to_payload()

    prompt_text, payload, forecast_section = _build_opus_prompt(
        market_data=market_data,
        account_payload=dict(account_payload),
        target_date=target_date,
        symbols=symbols,
        include_market_history=include_market_history,
        chronos2_forecasts=chronos2_forecasts,
    )

    sanitized_payload = _sanitize_market_payload(payload)

    # Include Chronos2 forecasts in payload for model reference
    if chronos2_forecasts:
        sanitized_payload["chronos2_forecasts"] = {
            symbol: {
                "last_close": f.last_close,
                "predicted_close": f.predicted_close,
                "low_close_10pct": f.low_close,
                "high_close_90pct": f.high_close,
                "expected_return_pct": f.expected_return_pct,
                "volatility_range_pct": f.volatility_range_pct,
                "entry_target": f.low_close,  # Buy at 10th percentile
                "exit_target": f.high_close,  # Sell at 90th percentile
            }
            for symbol, f in chronos2_forecasts.items()
        }

    payload_json = json.dumps(sanitized_payload, ensure_ascii=False, indent=2)

    return [
        {"role": "user", "content": prompt_text},
        {"role": "user", "content": f"Market data payload:\n```json\n{payload_json}\n```"},
    ]


__all__ = ["SYSTEM_PROMPT", "build_opus_messages", "opus_plan_schema"]
