"""Prompt builder for live Binance per-symbol trader with Chronos2 forecasts."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


FORECAST_CACHE = Path(__file__).resolve().parent.parent / "binanceneural" / "forecast_cache"


def _compute_trend_context(history_rows: list[dict]) -> dict:
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


def load_latest_forecast(symbol: str, horizon: int, cache_root: Optional[Path] = None) -> Optional[dict]:
    root = cache_root or FORECAST_CACHE
    path = root / f"h{horizon}" / f"{symbol}.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if df.empty:
            return None
        row = df.iloc[-1]
        result = {}
        for col in df.columns:
            try:
                val = float(row[col])
                result[col] = val
            except (ValueError, TypeError):
                pass
        return result
    except Exception:
        return None


def _fmt_forecast(fc: Optional[dict], current_price: float) -> str:
    if not fc:
        return "  No forecast available"
    lines = []
    for key in ["predicted_close_p50", "predicted_high_p50", "predicted_low_p50"]:
        if key in fc:
            val = fc[key]
            delta_pct = (val - current_price) / current_price * 100
            label = key.replace("predicted_", "").replace("_p50", "")
            lines.append(f"  {label}: ${val:.2f} ({delta_pct:+.2f}%)")
    if "predicted_close_p90" in fc and "predicted_close_p10" in fc:
        spread = fc["predicted_close_p90"] - fc["predicted_close_p10"]
        spread_pct = spread / current_price * 100
        lines.append(f"  90% CI spread: ${spread:.2f} ({spread_pct:.2f}%)")
    return "\n".join(lines) if lines else "  No forecast available"


def build_live_prompt(
    symbol: str,
    history_rows: list[dict],
    current_price: float,
    fc_1h: Optional[dict] = None,
    fc_24h: Optional[dict] = None,
) -> str:
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
            trend_parts.append(f"{key.replace('ret_', '')}: {trend[key]:+.2f}%")
    if "atr_pct" in trend:
        trend_parts.append(f"ATR: {trend['atr_pct']:.2f}%")
    if "up_hours_12" in trend:
        trend_parts.append(f"Up hrs (12h): {trend['up_hours_12']}/12")
    trend_line = " | ".join(trend_parts) if trend_parts else "N/A"

    sr_line = ""
    if "low_24h" in trend:
        sr_line = f"  24h range: ${trend['low_24h']:.2f} - ${trend['high_24h']:.2f} ({trend['range_pct']:.2f}% wide)"

    volumes = [float(r.get("volume", 0)) for r in history_rows if r.get("volume")]
    vol_context = ""
    if len(volumes) >= 24:
        avg_vol_24 = np.mean(volumes[-24:])
        recent_vol = volumes[-1]
        vol_ratio = recent_vol / avg_vol_24 if avg_vol_24 > 0 else 1.0
        vol_context = f"\nVOLUME: Current={recent_vol:.0f}, 24h avg={avg_vol_24:.0f}, ratio={vol_ratio:.2f}x"

    fc_section = ""
    if fc_1h or fc_24h:
        fc_section = f"""
CHRONOS2 ML FORECASTS:
1-hour forecast:
{_fmt_forecast(fc_1h, current_price)}
24-hour forecast:
{_fmt_forecast(fc_24h, current_price)}
"""

    return f"""You are solving an optimization problem: maximize risk-adjusted returns on a crypto spot account.

CONSTRAINTS:
- LONG ONLY (spot market, no shorting)
- 1-hour decision intervals, max 6-hour hold time
- Transaction cost: 0.1% maker fee per trade
- Objective: maximize Sortino ratio (penalize downside deviation, reward upside)

SYMBOL: {symbol}
CURRENT PRICE: ${current_price:.2f}
{sr_line}

TREND CONTEXT: {trend_line}
{vol_context}
{fc_section}
LAST 12 HOURS:
{chr(10).join(price_lines)}

OPTIMIZATION TASK:
Find the optimal balance between:
1. ENTRY FREQUENCY: Too few trades = missed alpha. Too many = fee drag kills returns.
2. POSITION SIZING: Set entries that maximize fill probability while minimizing adverse selection.
3. DIRECTIONAL ACCURACY: A 55% hit rate with 2:1 R:R is highly profitable.
4. RISK ASYMMETRY: The Sortino ratio only penalizes downside. Take asymmetric bets where upside >> downside.

Think about the probability distribution of the next 1-6 hours:
- What is the expected move? (forecast + trend + momentum)
- What is the variance? (ATR, recent range)
- Is the risk/reward skewed favorably?

Only enter when expected_return > fees + slippage. Set buy_price and sell_price to capture the most likely profitable range.

Respond with JSON: {{"direction": "long" or "hold", "buy_price": <entry>, "sell_price": <take profit>, "confidence": <0-1>, "reasoning": "<brief>"}}"""
