"""Prompt builder for live Binance RL+LLM hybrid trader."""

from __future__ import annotations

import numpy as np


def _compute_trend_context(history_rows: list[dict]) -> dict:
    """Compute trend indicators from price history."""
    closes = [float(r["close"]) for r in history_rows]
    highs = [float(r["high"]) for r in history_rows]
    lows = [float(r["low"]) for r in history_rows]
    current = closes[-1]

    ctx = {}
    if len(closes) >= 25:
        ctx["ret_24h"] = (current - closes[-25]) / closes[-25] * 100
    if len(closes) >= 49:
        ctx["ret_48h"] = (current - closes[-49]) / closes[-49] * 100
    if len(closes) >= 72:
        ctx["ret_72h"] = (current - closes[-72]) / closes[-72] * 100

    if len(highs) >= 24:
        atr = np.mean([highs[-i] - lows[-i] for i in range(1, 25)])
        ctx["atr_pct"] = atr / current * 100

    if len(closes) >= 24:
        ctx["low_24h"] = min(lows[-24:])
        ctx["high_24h"] = max(highs[-24:])
        ctx["range_pct"] = (ctx["high_24h"] - ctx["low_24h"]) / current * 100

    if len(closes) >= 13:
        ups = sum(1 for i in range(-12, 0) if closes[i] > closes[i - 1])
        ctx["up_hours_12"] = ups

    return ctx


def build_live_prompt(
    symbol: str,
    history_rows: list[dict],
    current_price: float,
) -> str:
    """Build trading prompt from live Binance kline data.

    Args:
        symbol: Internal symbol name (e.g. "BTCUSD")
        history_rows: List of dicts with timestamp, open, high, low, close, volume
        current_price: Current live price
    """
    recent = history_rows[-12:]
    price_lines = []
    for row in recent:
        ts = str(row.get("timestamp", ""))[:16]
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        vol = float(row.get("volume", 0))
        price_lines.append(f"  {ts}: O={o:.2f} H={h:.2f} L={l:.2f} C={c:.2f} V={vol:.0f}")

    trend = _compute_trend_context(history_rows)
    trend_parts = []
    for key in ["ret_24h", "ret_48h", "ret_72h"]:
        if key in trend:
            label = key.replace("ret_", "")
            trend_parts.append(f"{label}: {trend[key]:+.2f}%")
    if "atr_pct" in trend:
        trend_parts.append(f"ATR: {trend['atr_pct']:.2f}%")
    if "up_hours_12" in trend:
        trend_parts.append(f"Up hrs (12h): {trend['up_hours_12']}/12")
    trend_line = " | ".join(trend_parts) if trend_parts else "N/A"

    sr_line = ""
    if "low_24h" in trend:
        sr_line = f"  24h range: ${trend['low_24h']:.2f} - ${trend['high_24h']:.2f} ({trend['range_pct']:.2f}% wide)"

    # Volume context
    volumes = [float(r.get("volume", 0)) for r in history_rows if r.get("volume")]
    vol_context = ""
    if len(volumes) >= 24:
        avg_vol_24 = np.mean(volumes[-24:])
        recent_vol = volumes[-1]
        vol_ratio = recent_vol / avg_vol_24 if avg_vol_24 > 0 else 1.0
        vol_context = f"\nVOLUME: Current={recent_vol:.0f}, 24h avg={avg_vol_24:.0f}, ratio={vol_ratio:.2f}x"

    return f"""You are an expert crypto trader analyzing live market data for a spot trading decision.

SYMBOL: {symbol} (long only, spot market)
CURRENT PRICE: ${current_price:.2f}
{sr_line}

TREND CONTEXT: {trend_line}
{vol_context}

LAST 12 HOURS:
{chr(10).join(price_lines)}

IMPORTANT: We can ONLY go long (spot market, no shorting). You should be actively looking for long entry opportunities.

TASK: You are a profitable swing trader. Enter long when ANY of these conditions are met:
1. Price is near recent support (24h low) with room to bounce
2. Momentum is turning up after a dip (buy the dip)
3. Price consolidating near highs (breakout setup)
4. Strong volume surge with upward price movement

SIZING & TARGETS:
- Set buy_price slightly below current price (0.1-0.3% below for normal vol, 0.3-0.5% below for high vol)
- Set sell_price at a realistic target (0.5-2% above entry, wider in high vol)
- Aim for 2:1+ reward:risk ratio
- Confidence: 0.6-0.9 for strong setups, 0.3-0.5 for marginal setups

You should be entering trades roughly 25-40% of the time. Hold only when price action clearly indicates continued downside.

Respond with JSON: {{"direction": "long" or "hold", "buy_price": <limit entry near support>, "sell_price": <take profit target>, "confidence": <0-1>, "reasoning": "<brief>"}}"""
