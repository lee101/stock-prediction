"""Cross-asset-aware prompt builder for unified trading.

Extends per-symbol prompts with portfolio context across both brokers,
including market transition guidance and rebalancing opportunities.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np

from unified_orchestrator.state import UnifiedPortfolioSnapshot, Position


def _build_time_context(snapshot: UnifiedPortfolioSnapshot) -> str:
    """Build time-awareness section: ET time, day name, market proximity."""
    now_et = datetime.now(ZoneInfo("America/New_York"))
    day_name = now_et.strftime("%A")
    time_str = now_et.strftime("%H:%M ET")

    lines = [f"TIME: {day_name} {time_str}"]
    if snapshot.regime == "CRYPTO_ONLY":
        if snapshot.minutes_to_open and snapshot.minutes_to_open < 180:
            lines.append(f"Stock market opens in {snapshot.minutes_to_open} min.")
        else:
            lines.append("Stock market closed.")
    elif snapshot.regime == "PRE_MARKET":
        lines.append(f"Stock market opens in {snapshot.minutes_to_open or '?'} min.")
    elif snapshot.regime == "STOCK_HOURS":
        lines.append(f"Stock market open, closes in {snapshot.minutes_to_close or '?'} min.")
    elif snapshot.regime == "POST_MARKET":
        lines.append("Stock market just closed.")
    return "\n".join(lines)


def _format_positions_summary(positions: dict[str, Position], label: str) -> str:
    """One-line summary of positions for a broker."""
    if not positions:
        return f"  {label}: No positions"
    parts = []
    for sym, pos in positions.items():
        pnl_pct = (pos.current_price - pos.avg_price) / pos.avg_price * 100 if pos.avg_price > 0 else 0
        if pos.broker == "binance":
            # Binance doesn't track avg price, show value
            parts.append(f"{sym} {pos.qty:.4f} (${pos.market_value:.0f})")
        else:
            parts.append(f"{sym} {pos.qty:.1f} @ ${pos.avg_price:.2f} ({pnl_pct:+.1f}%)")
    return f"  {label}: {', '.join(parts)}"


def build_portfolio_context(
    snapshot: UnifiedPortfolioSnapshot,
    best_stock_edges: Optional[dict[str, float]] = None,
    best_crypto_edges: Optional[dict[str, float]] = None,
) -> str:
    """Build the cross-asset portfolio section for the LLM prompt.

    Kept concise to minimize token usage. Only includes transition
    guidance when near market open/close.
    """
    lines = ["## Portfolio Overview"]

    # Alpaca — stocks + crypto, margin account
    equity = snapshot.total_stock_value
    long_val = sum(p.market_value for p in snapshot.alpaca_positions.values())
    leverage = long_val / equity if equity > 0 else 0
    lines.append(f"ALPACA: Equity ${equity:,.0f} | Cash ${snapshot.alpaca_cash:,.0f} | "
                 f"Buying power ${snapshot.alpaca_buying_power:,.0f}")
    lines.append(f"LEVERAGE: {leverage:.2f}x current | 4x max intraday | 2x max overnight")
    lines.append(f"MARGIN COST: 6.25% annual on leveraged portion "
                 f"(~${max(0, long_val - equity) * 0.0625 / 365:.2f}/day currently)")
    if snapshot.alpaca_positions:
        lines.append(_format_positions_summary(snapshot.alpaca_positions, "Positions"))

    # Market transition context
    if snapshot.regime == "PRE_MARKET" and snapshot.minutes_to_open is not None:
        lines.append(f"\nMARKET: Opens in {snapshot.minutes_to_open} min.")
        if best_stock_edges:
            best_sym = max(best_stock_edges, key=best_stock_edges.get)
            lines.append(f"Best stock signal: {best_sym} edge {best_stock_edges[best_sym]:.1%}")
        if snapshot.binance_positions:
            lines.append("You may exit crypto positions to free capital for better opportunities.")
            lines.append("Use backout_near_market to sell crypto at limit before market opens.")

    elif snapshot.regime == "STOCK_HOURS" and snapshot.minutes_to_close is not None:
        if snapshot.minutes_to_close <= 60:
            lines.append(f"\nMARKET: Closes in {snapshot.minutes_to_close} min.")
            lines.append("Consider crypto entries for off-hours if stock positions are closing.")
        elif snapshot.binance_positions:
            crypto_val = sum(p.market_value for p in snapshot.binance_positions.values())
            lines.append(f"\nCrypto positions: ${crypto_val:,.0f} running alongside stocks.")

    elif snapshot.regime == "POST_MARKET":
        lines.append("\nMARKET: Just closed. Stock orders will queue for next open.")
        lines.append("Crypto trading active — consider new entries with freed attention.")

    elif snapshot.regime == "CRYPTO_ONLY":
        if snapshot.minutes_to_open and snapshot.minutes_to_open < 120:
            lines.append(f"\nMARKET: Opens in {snapshot.minutes_to_open} min. Plan crypto exits if needed.")

    return "\n".join(lines)


def build_unified_prompt(
    symbol: str,
    history_rows: list[dict],
    current_price: float,
    snapshot: UnifiedPortfolioSnapshot,
    asset_class: str = "crypto",
    forecast_1h: Optional[dict] = None,
    forecast_24h: Optional[dict] = None,
    best_stock_edges: Optional[dict[str, float]] = None,
    best_crypto_edges: Optional[dict[str, float]] = None,
    held_position: Optional[Position] = None,
) -> str:
    """Build complete trading prompt with cross-asset awareness.

    Args:
        symbol: Symbol being analyzed (e.g. "BTCUSD" or "NVDA")
        history_rows: Recent OHLCV bars
        current_price: Current price
        snapshot: Full portfolio state
        asset_class: "crypto" or "stock"
        forecast_1h: Chronos 1h forecast dict
        forecast_24h: Chronos 24h forecast dict
        best_stock_edges: Best edge per stock symbol
        best_crypto_edges: Best edge per crypto symbol
    """
    asset_label = "cryptocurrency" if asset_class == "crypto" else "stock"
    instrument_style = "long only, spot" if asset_class == "crypto" else "margin equity (up to 4x intraday, 2x overnight, 6.25% margin interest)"
    fee_str = "10 bps per side (~20 bps round-trip) on Alpaca crypto"

    # Recent price history
    recent = history_rows[-12:]
    price_lines = []
    for row in recent:
        ts = str(row.get("timestamp", ""))[:16]
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        vol = float(row.get("volume", 0))
        price_lines.append(f"  {ts}: O={o:.2f} H={h:.2f} L={l:.2f} C={c:.2f} V={vol:.0f}")

    # Trend context
    closes = [float(r["close"]) for r in history_rows]
    highs = [float(r["high"]) for r in history_rows]
    lows = [float(r["low"]) for r in history_rows]
    trend_parts = []
    if len(closes) >= 25:
        ret_24h = (closes[-1] - closes[-25]) / closes[-25] * 100
        trend_parts.append(f"24h: {ret_24h:+.2f}%")
    if len(closes) >= 49:
        ret_48h = (closes[-1] - closes[-49]) / closes[-49] * 100
        trend_parts.append(f"48h: {ret_48h:+.2f}%")
    if len(highs) >= 24:
        atr = np.mean([highs[-i] - lows[-i] for i in range(1, 25)])
        atr_pct = atr / current_price * 100
        trend_parts.append(f"ATR: {atr_pct:.2f}%")
    trend_line = " | ".join(trend_parts) if trend_parts else "N/A"

    # Trend regime: is price above or below SMA48?
    trend_regime = ""
    if len(closes) >= 48:
        sma48 = np.mean(closes[-48:])
        if current_price > sma48 * 1.01:
            trend_regime = "UPTREND (price above SMA48)"
        elif current_price < sma48 * 0.99:
            trend_regime = "DOWNTREND (price below SMA48) — CAUTION: only enter with strong edge"
        else:
            trend_regime = "SIDEWAYS (price near SMA48)"

    # Support/resistance
    sr_line = ""
    if len(lows) >= 24:
        low_24 = min(lows[-24:])
        high_24 = max(highs[-24:])
        sr_line = f"  24h range: ${low_24:.2f} - ${high_24:.2f} ({(high_24 - low_24) / current_price * 100:.2f}% wide)"

    # Forecast section with confidence intervals
    fc_text = ""
    if forecast_1h:
        p50 = forecast_1h.get("predicted_close_p50", current_price)
        delta = (p50 - current_price) / current_price * 100
        p10 = forecast_1h.get("predicted_close_p10", p50)
        p90 = forecast_1h.get("predicted_close_p90", p50)
        spread = (p90 - p10) / current_price * 100
        fc_text += f"\n  1h forecast: ${p50:.2f} ({delta:+.2f}%), 80% CI: ${p10:.2f}-${p90:.2f} (±{spread/2:.2f}%)"
    if forecast_24h:
        p50 = forecast_24h.get("predicted_close_p50", current_price)
        delta = (p50 - current_price) / current_price * 100
        p10 = forecast_24h.get("predicted_close_p10", p50)
        p90 = forecast_24h.get("predicted_close_p90", p50)
        spread = (p90 - p10) / current_price * 100
        fc_text += f"\n  24h forecast: ${p50:.2f} ({delta:+.2f}%), 80% CI: ${p10:.2f}-${p90:.2f} (±{spread/2:.2f}%)"

    # Cross-asset context
    portfolio_ctx = build_portfolio_context(snapshot, best_stock_edges, best_crypto_edges)

    # Time awareness
    time_ctx = _build_time_context(snapshot)

    # Determine direction guidance based on regime
    if asset_class == "crypto":
        direction_guidance = "LONG or HOLD (spot market, no shorting)"
        direction_json_hint = '"long" or "hold"'
    else:
        direction_guidance = "LONG, SHORT, or HOLD"
        direction_json_hint = '"long", "short", or "hold"'

    # Transition-specific instructions
    transition_instructions = ""
    if snapshot.regime == "PRE_MARKET" and asset_class == "crypto":
        transition_instructions = """
TRANSITION CONTEXT: Market opens soon. If you have crypto positions with weak edge,
consider setting a tight sell limit to free capital. You can specify a multi-step plan:
1. Sell this crypto position at limit
2. The freed capital can be used for new crypto entries after market opens

Use direction="hold" with sell_price set to take-profit if you want to exit at a target."""

    elif snapshot.regime == "STOCK_HOURS" and snapshot.minutes_to_close and snapshot.minutes_to_close <= 60:
        transition_instructions = """
TRANSITION CONTEXT: Market closes soon. Stock positions will be managed separately.
Consider whether crypto positions should be adjusted for overnight holding."""

    # Position context
    position_ctx = ""
    if held_position and held_position.qty > 0:
        pnl_pct = (current_price - held_position.avg_price) / held_position.avg_price * 100
        position_ctx = (f"\nCURRENT POSITION: LONG {held_position.qty:.6f} @ avg ${held_position.avg_price:.2f} "
                        f"(${held_position.market_value:.0f}, {pnl_pct:+.2f}%)"
                        f"\n  ** You MUST set sell_price as your exit target for this position. **")

    return f"""You are an expert {asset_label} trader in a unified multi-asset portfolio system.

SYMBOL: {symbol} ({instrument_style})
CURRENT PRICE: ${current_price:.2f}
{sr_line}
FEES: {fee_str}{position_ctx}

{portfolio_ctx}

{time_ctx}

TREND CONTEXT: {trend_line}
{f"TREND REGIME: {trend_regime}" if trend_regime else ""}

CHRONOS2 FORECASTS:{fc_text if fc_text else " None available"}

LAST 12 HOURS:
{chr(10).join(price_lines)}
{transition_instructions}

TASK: You are a profitable swing trader. All orders must be LIMIT orders (never market).
- Set buy_price slightly below current (0.1-0.3% below for normal vol)
- Set sell_price at realistic target (1.5-2.5% above entry for new entries, or above current for exits)
- In DOWNTREND: prefer to HOLD or set very tight targets. Only enter with strong edge (>1.5% expected move).
- In UPTREND: wider targets (2-3%) capture more upside.
- Confidence: 0.6-0.9 for strong setups, 0.3-0.5 for marginal
- allocation_pct: How much of available capital to put on this trade (0-100%). Use 20-40% for normal conviction, 50-80% for high conviction, 0% for hold. In downtrends, prefer 10-25%.
- Enter trades 25-40% of the time. Hold when no clear edge.
- IMPORTANT: If you already hold a position, ALWAYS set sell_price to your take-profit exit target, even if direction is "hold". Every position must have an exit price.
- LEVERAGE: For stocks, we can use up to 4x intraday leverage (auto-deleverages to 2x before market close). Margin costs 6.25% annual on the leveraged portion. Factor this cost into expected returns — a 0.5% expected gain on a 4x leveraged position costs ~1.7bps/day in margin, so only leverage up on high-conviction setups with expected returns exceeding margin cost.

Respond with JSON: {{"direction": {direction_json_hint}, "buy_price": <limit entry or 0 if hold>, "sell_price": <ALWAYS set take-profit exit for held positions>, "confidence": <0-1>, "allocation_pct": <0-100>, "reasoning": "<brief>"}}"""
