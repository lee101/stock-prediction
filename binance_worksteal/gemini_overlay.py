"""Gemini LLM overlay for work-stealing daily strategy.

Enhances rule-based dip-buying with LLM recalibration of buy/sell prices.
Gemini sees: position context, Chronos2 forecasts, rule-based signals,
previous trades, market regime -- and outputs adjusted buy/sell prices.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from src.forecast_cache_lookup import load_latest_forecast_from_cache

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

FORECAST_CACHE = Path(__file__).resolve().parent.parent / "binanceneural" / "forecast_cache"


@dataclass
class DailyTradePlan:
    action: str  # "buy", "sell", "hold", "adjust"
    buy_price: float = 0.0
    sell_price: float = 0.0
    stop_price: float = 0.0
    confidence: float = 0.0
    reasoning: str = ""


if genai is not None:
    DAILY_SCHEMA = genai.types.Schema(
        type=genai.types.Type.OBJECT,
        required=["action", "buy_price", "sell_price", "stop_price", "confidence"],
        properties={
            "action": genai.types.Schema(
                type=genai.types.Type.STRING,
                description="One of: buy, sell, hold, adjust",
            ),
            "buy_price": genai.types.Schema(
                type=genai.types.Type.STRING,
                description="Limit buy price as number string, 0 if no buy",
            ),
            "sell_price": genai.types.Schema(
                type=genai.types.Type.STRING,
                description="Take-profit/sell price as number string, 0 if no sell",
            ),
            "stop_price": genai.types.Schema(
                type=genai.types.Type.STRING,
                description="Stop loss price as number string, 0 if no stop",
            ),
            "confidence": genai.types.Schema(
                type=genai.types.Type.STRING,
                description="Confidence 0.0 to 1.0",
            ),
            "reasoning": genai.types.Schema(
                type=genai.types.Type.STRING,
                description="Brief reasoning for the decision",
            ),
        },
    )
else:
    DAILY_SCHEMA = None


def load_forecast_daily(
    symbol: str,
    cache_root: Optional[Path] = None,
    *,
    as_of: pd.Timestamp | str | None = None,
) -> Optional[dict]:
    root = cache_root or FORECAST_CACHE
    return load_latest_forecast_from_cache(symbol, 24, root, as_of=as_of)


def build_daily_prompt(
    symbol: str,
    bars: pd.DataFrame,
    current_price: float,
    rule_signal: dict,
    position_info: Optional[dict] = None,
    recent_trades: Optional[List[dict]] = None,
    forecast_24h: Optional[dict] = None,
    universe_summary: Optional[str] = None,
    fee_bps: int = 10,
) -> str:
    # Last 20 daily bars
    recent = bars.tail(20)
    price_lines = []
    for _, row in recent.iterrows():
        ts = str(row["timestamp"])[:10]
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        vol = float(row.get("volume", 0))
        chg = (c - o) / o * 100
        price_lines.append(f"  {ts}: O={o:.2f} H={h:.2f} L={l:.2f} C={c:.2f} V={vol:.0f} ({chg:+.1f}%)")

    # Trend context
    closes = bars["close"].values
    trend_parts = []
    for n, label in [(1, "1d"), (5, "5d"), (10, "10d"), (20, "20d")]:
        if len(closes) > n:
            ret = (closes[-1] - closes[-n-1]) / closes[-n-1] * 100
            trend_parts.append(f"{label}: {ret:+.1f}%")
    if len(closes) >= 20:
        sma20 = np.mean(closes[-20:])
        trend_parts.append(f"SMA20: ${sma20:.2f} ({'above' if current_price > sma20 else 'below'})")
    if len(closes) >= 50:
        sma50 = np.mean(closes[-50:])
        trend_parts.append(f"SMA50: ${sma50:.2f}")

    # Volatility
    if len(bars) >= 14:
        highs = bars["high"].values[-14:]
        lows = bars["low"].values[-14:]
        prev_closes = bars["close"].values[-15:-1]
        tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_closes), np.abs(lows - prev_closes)))
        atr = np.mean(tr)
        atr_pct = atr / current_price * 100
        trend_parts.append(f"ATR14: ${atr:.2f} ({atr_pct:.1f}%)")

    # Recent high/low
    if len(bars) >= 20:
        high_20 = float(bars["high"].tail(20).max())
        low_20 = float(bars["low"].tail(20).min())
        dip_from_high = (current_price - high_20) / high_20 * 100
        trend_parts.append(f"20d high: ${high_20:.2f} (dip: {dip_from_high:+.1f}%)")
        trend_parts.append(f"20d low: ${low_20:.2f}")

    trend_line = " | ".join(trend_parts)

    # Forecast section
    fc_section = ""
    if forecast_24h:
        fc_lines = []
        for key in ["predicted_close_p50", "predicted_high_p50", "predicted_low_p50"]:
            if key in forecast_24h:
                val = forecast_24h[key]
                delta = (val - current_price) / current_price * 100
                label = key.replace("predicted_", "").replace("_p50", "")
                fc_lines.append(f"  {label}: ${val:.2f} ({delta:+.2f}%)")
        if "predicted_close_p90" in forecast_24h and "predicted_close_p10" in forecast_24h:
            spread = forecast_24h["predicted_close_p90"] - forecast_24h["predicted_close_p10"]
            fc_lines.append(f"  90% CI: ${forecast_24h['predicted_close_p10']:.2f} - ${forecast_24h['predicted_close_p90']:.2f} (spread: {spread/current_price*100:.2f}%)")
        fc_section = "\nCHRONOS2 24h FORECAST:\n" + "\n".join(fc_lines) + "\n"

    # Position section
    pos_section = "\nPOSITION: Flat (no open position)\n"
    if position_info and position_info.get("quantity", 0) > 0:
        entry = position_info["entry_price"]
        held = position_info.get("held_days", 0)
        pnl_pct = (current_price - entry) / entry * 100
        pnl_usd = position_info["quantity"] * (current_price - entry)
        peak = position_info.get("peak_price", entry)
        dd_from_peak = (current_price - peak) / peak * 100 if peak > 0 else 0
        pos_section = f"""
CURRENT POSITION:
  {position_info['quantity']:.6f} {symbol} @ ${entry:.2f} (held {held}d, max hold 14d)
  P&L: ${pnl_usd:+.2f} ({pnl_pct:+.2f}%)
  Peak since entry: ${peak:.2f} (drawdown from peak: {dd_from_peak:+.1f}%)
  Current target: ${position_info.get('target_sell', 0):.2f}
  Current stop: ${position_info.get('stop_price', 0):.2f}
"""

    # Recent trades section
    trades_section = ""
    if recent_trades:
        trade_lines = []
        for t in recent_trades[-5:]:  # last 5 trades
            trade_lines.append(
                f"  {t.get('timestamp','')[:10]} {t['side']:>5s} {t['symbol']} "
                f"@ ${t['price']:.2f} pnl=${t.get('pnl',0):+.2f} ({t.get('reason','')})"
            )
        trades_section = "\nRECENT TRADES:\n" + "\n".join(trade_lines) + "\n"

    # Rule-based signal
    signal_section = ""
    if rule_signal:
        sig_lines = []
        if rule_signal.get("buy_target"):
            sig_lines.append(f"  Buy target: ${rule_signal['buy_target']:.2f} (20% dip from high)")
        if rule_signal.get("dip_score"):
            sig_lines.append(f"  Dip proximity score: {rule_signal['dip_score']:.4f}")
        if rule_signal.get("ref_price"):
            sig_lines.append(f"  Reference high: ${rule_signal['ref_price']:.2f}")
        if rule_signal.get("sma_ok") is not None:
            sig_lines.append(f"  SMA-20 filter: {'PASS' if rule_signal['sma_ok'] else 'FAIL'}")
        signal_section = "\nRULE-BASED SIGNAL:\n" + "\n".join(sig_lines) + "\n"

    # Universe summary
    universe_section = ""
    if universe_summary:
        universe_section = f"\nUNIVERSE SNAPSHOT:\n{universe_summary}\n"

    fee_str = f"{fee_bps}bps" if fee_bps > 0 else "0bps (FDUSD)"

    return f"""You are a daily cryptocurrency trader optimizing a work-stealing dip-buying strategy across 30 symbols.

STRATEGY: Buy symbols that have dipped significantly from recent highs, sell on recovery.
CONSTRAINTS:
- LONG ONLY, daily bars, max 14-day hold
- Fees: {fee_str} per side
- Max 5 concurrent positions, shared $10,000 cash pool
- Objective: maximize Sortino ratio (reward upside, penalize downside)
- 25% max drawdown circuit breaker

SYMBOL: {symbol}
CURRENT PRICE: ${current_price:.2f}
{pos_section}
TREND: {trend_line}
{fc_section}{signal_section}{trades_section}{universe_section}
LAST 20 DAILY BARS:
{chr(10).join(price_lines)}

YOUR TASK:
The rule-based system identified this symbol as a candidate. You must recalibrate the execution:
1. Should we actually enter/exit this position? (consider momentum, forecast, regime)
2. At what EXACT price should we place the limit buy? (must be realistic intraday fill)
3. What take-profit and stop-loss are optimal given forecast uncertainty?
4. How confident are you? (0=skip, 1=max conviction)

Think about:
- Is the dip a genuine mean-reversion opportunity or a falling knife?
- Does the Chronos2 forecast support a recovery?
- Are we catching a dead cat bounce or a real bottom?
- Is the risk:reward ratio favorable after fees?

Respond JSON: {{"action": "buy"|"sell"|"hold"|"adjust", "buy_price": <limit buy>, "sell_price": <take-profit>, "stop_price": <stop-loss>, "confidence": <0-1>, "reasoning": "<brief>"}}"""


def call_gemini_daily(
    prompt: str,
    model: str = "gemini-2.5-flash",
    api_key: Optional[str] = None,
    temperature: float = 0.3,
    max_retries: int = 3,
) -> Optional[DailyTradePlan]:
    if genai is None:
        return None
    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        return None

    client = genai.Client(api_key=key)

    config = types.GenerateContentConfig(
        temperature=temperature,
        response_mime_type="application/json",
        response_schema=DAILY_SCHEMA,
    )

    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model, contents=prompt, config=config,
            )
            text = resp.text.strip()
            data = json.loads(text)
            return DailyTradePlan(
                action=str(data.get("action", "hold")),
                buy_price=float(data.get("buy_price", 0) or 0),
                sell_price=float(data.get("sell_price", 0) or 0),
                stop_price=float(data.get("stop_price", 0) or 0),
                confidence=float(data.get("confidence", 0) or 0),
                reasoning=str(data.get("reasoning", "")),
            )
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None
    return None
