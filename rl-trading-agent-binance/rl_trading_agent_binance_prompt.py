"""Prompt builder for live Binance per-symbol trader with Chronos2 forecasts."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from src.forecast_cache_lookup import load_latest_forecast_from_cache


FORECAST_CACHE = Path(__file__).resolve().parent.parent / "binanceneural" / "forecast_cache"

# Historical 1h forecast MAE by symbol (from validation)
FORECAST_MAE_1H = {
    "BTCUSD": 0.55,
    "ETHUSD": 0.75,
    "SOLUSD": 0.97,
    "DOGEUSD": 1.20,
    "AAVEUSD": 1.50,
}

FORECAST_MAE_4H = {
    "BTCUSD": 0.83,
    "ETHUSD": 1.13,
    "SOLUSD": 1.46,
    "DOGEUSD": 1.80,
    "AAVEUSD": 2.25,
}

FORECAST_MAE_12H = {
    "BTCUSD": 1.38,
    "ETHUSD": 1.88,
    "SOLUSD": 2.43,
    "DOGEUSD": 3.00,
    "AAVEUSD": 3.75,
}

FORECAST_MAE_BY_HORIZON = {
    1: FORECAST_MAE_1H,
    4: FORECAST_MAE_4H,
    12: FORECAST_MAE_12H,
}


def build_cross_asset_context(all_symbol_bars: dict) -> str:
    """Build cross-asset correlation and regime context.

    Args:
        all_symbol_bars: dict of {symbol: DataFrame with OHLCV columns}

    Returns:
        String block to insert into prompt.
    """
    if not all_symbol_bars:
        return ""

    returns_24h = {}
    for sym, df in all_symbol_bars.items():
        if df is None or (hasattr(df, "empty") and df.empty):
            continue
        if hasattr(df, "iloc") and len(df) >= 25:
            closes = df["close"].astype(float).values
            ret = (closes[-1] - closes[-25]) / closes[-25] * 100
            returns_24h[sym] = ret
        elif isinstance(df, list) and len(df) >= 25:
            closes = [float(r["close"]) for r in df]
            ret = (closes[-1] - closes[-25]) / closes[-25] * 100
            returns_24h[sym] = ret

    if len(returns_24h) < 2:
        return ""

    btc_key = None
    for k in returns_24h:
        if "BTC" in k.upper():
            btc_key = k
            break

    btc_ret = returns_24h.get(btc_key, 0.0) if btc_key else None
    alt_rets = {k: v for k, v in returns_24h.items() if k != btc_key}
    avg_alt_ret = np.mean(list(alt_rets.values())) if alt_rets else 0.0

    # regime classification
    threshold = 0.5
    if btc_ret is not None:
        if btc_ret > threshold and avg_alt_ret > threshold:
            regime = "risk-on"
        elif btc_ret < -threshold and avg_alt_ret < -threshold:
            regime = "risk-off"
        elif abs(btc_ret - avg_alt_ret) > 1.5:
            regime = "rotation"
        else:
            regime = "neutral"
    else:
        avg_all = np.mean(list(returns_24h.values()))
        if avg_all > threshold:
            regime = "risk-on"
        elif avg_all < -threshold:
            regime = "risk-off"
        else:
            regime = "neutral"

    # BTC-ETH rolling correlation (use hourly returns if enough data)
    eth_key = None
    for k in all_symbol_bars:
        if "ETH" in k.upper():
            eth_key = k
            break

    corr_str = "N/A"
    if btc_key and eth_key and btc_key in all_symbol_bars and eth_key in all_symbol_bars:
        btc_df = all_symbol_bars[btc_key]
        eth_df = all_symbol_bars[eth_key]
        try:
            if hasattr(btc_df, "iloc"):
                btc_c = btc_df["close"].astype(float).values
                eth_c = eth_df["close"].astype(float).values
            else:
                btc_c = np.array([float(r["close"]) for r in btc_df])
                eth_c = np.array([float(r["close"]) for r in eth_df])
            n = min(len(btc_c), len(eth_c), 25)
            if n >= 3:
                btc_r = np.diff(btc_c[-n:]) / btc_c[-n:-1]
                eth_r = np.diff(eth_c[-n:]) / eth_c[-n:-1]
                if np.std(btc_r) > 0 and np.std(eth_r) > 0:
                    corr = np.corrcoef(btc_r, eth_r)[0, 1]
                    if np.isfinite(corr):
                        corr_str = f"{corr:.2f}"
        except Exception:
            pass

    # dominance line
    dom_line = ""
    if btc_ret is not None:
        dom_line = f"BTC dominance: BTC {btc_ret:+.1f}% vs alts {avg_alt_ret:+.1f}%"
    else:
        dom_line = f"Avg return: {np.mean(list(returns_24h.values())):+.1f}%"

    # top movers
    sorted_movers = sorted(returns_24h.items(), key=lambda x: abs(x[1]), reverse=True)
    movers = sorted_movers[:5]
    mover_parts = []
    for sym, ret in movers:
        short = sym.replace("USD", "").replace("USDT", "")
        mover_parts.append(f"{short} {ret:+.1f}%")
    movers_line = ", ".join(mover_parts) if mover_parts else "N/A"

    lines = [
        "=== Market Regime ===",
        f"Regime: {regime} | {dom_line}",
        f"BTC-ETH correlation (24h): {corr_str}",
        f"Top movers (24h): {movers_line}",
    ]
    return "\n".join(lines)


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
    return load_latest_forecast_from_cache(symbol, int(horizon), root)


def _fmt_forecast(fc: Optional[dict], current_price: float, symbol: Optional[str] = None, horizon: Optional[int] = None) -> str:
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
    if symbol and horizon is not None:
        mae_table = FORECAST_MAE_BY_HORIZON.get(horizon, FORECAST_MAE_1H)
        if symbol in mae_table:
            lines.append(f"  Historical MAE: {mae_table[symbol]:.2f}%")
    elif symbol and symbol in FORECAST_MAE_1H:
        lines.append(f"  Historical MAE: {FORECAST_MAE_1H[symbol]:.2f}%")
    return "\n".join(lines) if lines else "  No forecast available"


def build_live_prompt(
    symbol: str,
    history_rows: list[dict],
    current_price: float,
    fc_1h: Optional[dict] = None,
    fc_24h: Optional[dict] = None,
    position_info: Optional[dict] = None,
    fee_bps: int = 10,
    leverage: float = 1.0,
    fc_4h: Optional[dict] = None,
    fc_12h: Optional[dict] = None,
    cross_asset_context: str = "",
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
    fc_parts = []
    for h, fc in [(1, fc_1h), (4, fc_4h), (12, fc_12h), (24, fc_24h)]:
        if fc:
            fc_parts.append(f"{h}-hour ahead:\n{_fmt_forecast(fc, current_price, symbol, horizon=h)}")
    if fc_parts:
        fc_section = "\nCHRONOS2 ML FORECASTS:\n" + "\n".join(fc_parts) + "\n"

    pos_section = "\nPOSITION: Flat (no position)\n"
    if position_info and position_info.get("qty", 0) > 0:
        entry = position_info.get("entry_price", 0)
        held = position_info.get("held_hours", 0)
        pnl_pct = (current_price - entry) / entry * 100 if entry > 0 else 0
        pnl_usd = position_info["qty"] * (current_price - entry)
        pos_section = f"""
CURRENT POSITION:
  Holding {position_info['qty']:.6f} {symbol} @ ${entry:.2f} entry
  Held {held:.0f}h (auto-close at 6h), P&L: ${pnl_usd:+.2f} ({pnl_pct:+.2f}%)
  EXIT DECISION: If profitable and momentum fading, take profits. If losing, set tight stop.
"""

    cross_section = ""
    if cross_asset_context:
        cross_section = f"\n{cross_asset_context}\n"

    fee_str = f"{fee_bps}bps per side" if fee_bps > 0 else "0bps (zero-fee FDUSD pair)"
    lev_str = f"{leverage:.0f}x" if leverage > 1 else "1x (no leverage)"
    return f"""You are solving an optimization problem: maximize risk-adjusted returns on a crypto spot account.

CONSTRAINTS:
- LONG ONLY (spot market, no shorting)
- Leverage: {lev_str}
- 1-hour decision intervals, max 6-hour hold time
- Transaction cost: {fee_str}
- Objective: maximize Sortino ratio (penalize downside deviation, reward upside)

SYMBOL: {symbol}
CURRENT PRICE: ${current_price:.2f}
{sr_line}
{pos_section}
TREND CONTEXT: {trend_line}
{vol_context}
{cross_section}{fc_section}
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


def build_live_prompt_freeform(
    symbol: str,
    history_rows: list[dict],
    current_price: float,
    fc_1h: Optional[dict] = None,
    fc_24h: Optional[dict] = None,
    position_info: Optional[dict] = None,
    fee_bps: int = 10,
    leverage: float = 1.0,
    forecast_error_1h: Optional[dict] = None,
    forecast_error_24h: Optional[dict] = None,
    fc_4h: Optional[dict] = None,
    fc_12h: Optional[dict] = None,
    forecast_error_4h: Optional[dict] = None,
    forecast_error_12h: Optional[dict] = None,
    cross_asset_context: str = "",
) -> str:
    recent = history_rows[-24:]
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
        sr_line = f"24h range: ${trend['low_24h']:.2f} - ${trend['high_24h']:.2f} ({trend['range_pct']:.2f}% wide)"

    fc_parts = []
    _horizon_specs = [
        (1, fc_1h, forecast_error_1h),
        (4, fc_4h, forecast_error_4h),
        (12, fc_12h, forecast_error_12h),
        (24, fc_24h, forecast_error_24h),
    ]
    for h, fc, fc_err in _horizon_specs:
        if not fc:
            continue
        fc_line = _fmt_forecast(fc, current_price, symbol, horizon=h)
        if fc_err and fc_err.get("mae_pct", 0) > 0:
            fc_line += f"\n  Historical MAE: {fc_err['mae_pct']:.2f}% (n={fc_err.get('samples', '?')})"
        fc_parts.append(f"{h}h ahead:\n{fc_line}")

    fc_section = ""
    if fc_parts:
        fc_section = "\n## Chronos2 ML Forecasts:\n" + "\n".join(fc_parts) + "\n"

    pos_section = "\n## Position: Flat (no position)\n"
    if position_info and position_info.get("qty", 0) > 0:
        entry = position_info.get("entry_price", 0)
        held = position_info.get("held_hours", 0)
        pnl_pct = (current_price - entry) / entry * 100 if entry > 0 else 0
        pnl_usd = position_info["qty"] * (current_price - entry)
        pos_section = f"""
## Current Position:
  Holding {position_info['qty']:.6f} {symbol} @ ${entry:.2f} entry
  Held {held:.0f}h (max hold: 6h), P&L: ${pnl_usd:+.2f} ({pnl_pct:+.2f}%)
"""

    fee_str = f"{fee_bps}bps per side" if fee_bps > 0 else "0bps (zero-fee FDUSD pair)"

    volumes = [float(r.get("volume", 0)) for r in history_rows if r.get("volume")]
    vol_context = ""
    if len(volumes) >= 24:
        avg_vol_24 = np.mean(volumes[-24:])
        recent_vol = volumes[-1]
        vol_ratio = recent_vol / avg_vol_24 if avg_vol_24 > 0 else 1.0
        vol_context = f"\nVolume: current={recent_vol:.0f}, 24h avg={avg_vol_24:.0f}, ratio={vol_ratio:.2f}x"

    cross_section = ""
    if cross_asset_context:
        cross_section = f"\n{cross_asset_context}\n"

    return f"""You are trading {symbol} (cryptocurrency) on 1-hour bars.

Objective: maximize risk-adjusted returns after fees.

Hard constraints:
- LONG ONLY (spot market)
- 1-hour decision intervals
- Max hold time: 6 hours
- Fees: {fee_str}
{pos_section}
Current price: ${current_price:.2f}
{sr_line}
Trend: {trend_line}{vol_context}
{cross_section}{fc_section}
## Last 24 hours:
{chr(10).join(price_lines)}

Use the data however you think is best.
- If flat: decide whether there is enough edge to enter.
- If long: decide whether to keep holding, set a realistic take-profit, or exit.
- Do not force trades. Only act when edge clearly exceeds fees.

Respond with JSON: {{"direction": "long" or "hold", "buy_price": <entry or 0>, "sell_price": <take-profit or 0>, "confidence": <0-1>, "reasoning": "<brief>"}}"""
