"""Cross-asset-aware prompt builder for unified trading.

Extends per-symbol prompts with portfolio context across both brokers,
including market transition guidance and rebalancing opportunities.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from unified_orchestrator.state import UnifiedPortfolioSnapshot, Position


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

    # Alpaca
    lines.append(f"STOCKS (Alpaca, 2x margin): Cash ${snapshot.alpaca_cash:,.0f} | "
                 f"Buying power ${snapshot.alpaca_buying_power:,.0f}")
    if snapshot.alpaca_positions:
        lines.append(_format_positions_summary(snapshot.alpaca_positions, "Positions"))

    # Binance
    quote_str = f"FDUSD ${snapshot.binance_fdusd:,.0f}" if snapshot.binance_fdusd > 1 else ""
    if snapshot.binance_usdt > 1:
        quote_str += f"{' | ' if quote_str else ''}USDT ${snapshot.binance_usdt:,.0f}"
    lines.append(f"CRYPTO (Binance, 1x spot): {quote_str or 'no cash'}")
    if snapshot.binance_positions:
        lines.append(_format_positions_summary(snapshot.binance_positions, "Positions"))

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
    fee_str = "0 bps (FDUSD)" if "BTC" in symbol or "ETH" in symbol else "10 bps"

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
        trend_parts.append(f"ATR: {atr / current_price * 100:.2f}%")
    trend_line = " | ".join(trend_parts) if trend_parts else "N/A"

    # Support/resistance
    sr_line = ""
    if len(lows) >= 24:
        low_24 = min(lows[-24:])
        high_24 = max(highs[-24:])
        sr_line = f"  24h range: ${low_24:.2f} - ${high_24:.2f} ({(high_24 - low_24) / current_price * 100:.2f}% wide)"

    # Forecast section
    fc_text = ""
    if forecast_1h:
        delta = (forecast_1h.get("predicted_close_p50", current_price) - current_price) / current_price * 100
        fc_text += f"\n  1h forecast: close={forecast_1h['predicted_close_p50']:.2f} ({delta:+.2f}%)"
    if forecast_24h:
        delta = (forecast_24h.get("predicted_close_p50", current_price) - current_price) / current_price * 100
        fc_text += f"\n  24h forecast: close={forecast_24h['predicted_close_p50']:.2f} ({delta:+.2f}%)"

    # Cross-asset context
    portfolio_ctx = build_portfolio_context(snapshot, best_stock_edges, best_crypto_edges)

    # Determine direction guidance based on regime
    if asset_class == "crypto":
        direction_guidance = "LONG or HOLD (spot market, no shorting)"
    else:
        direction_guidance = "LONG, SHORT, or HOLD"

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

    return f"""You are an expert {asset_label} trader in a unified multi-asset portfolio system.

SYMBOL: {symbol} (long only, spot)
CURRENT PRICE: ${current_price:.2f}
{sr_line}
FEES: {fee_str}

{portfolio_ctx}

TREND CONTEXT: {trend_line}

CHRONOS2 FORECASTS:{fc_text if fc_text else " None available"}

LAST 12 HOURS:
{chr(10).join(price_lines)}
{transition_instructions}

TASK: You are a profitable swing trader. All orders must be LIMIT orders (never market).
- Set buy_price slightly below current (0.1-0.3% below for normal vol)
- Set sell_price at realistic target (0.5-2% above entry)
- Confidence: 0.6-0.9 for strong setups, 0.3-0.5 for marginal
- Enter trades 25-40% of the time. Hold when no clear edge.

Respond with JSON: {{"direction": "{direction_guidance.split(' or ')[0].lower()}" or "hold", "buy_price": <limit entry>, "sell_price": <take profit>, "confidence": <0-1>, "reasoning": "<brief>"}}"""
