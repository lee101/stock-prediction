"""Portfolio allocation prompt builder for RL+Chronos2+LLM hybrid system.

Assembles rich context from multiple signal sources into a prompt for Gemini
to produce an optimized multi-asset allocation plan.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from rl_signal import (
    RLSignalGenerator,
    PortfolioSnapshot,
    RLSignal,
    SYMBOLS,
    ACTION_NAMES,
    _load_forecast_parquet,
)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

SYMBOL_BINANCE_MAP = {
    "BTCUSD": ("BTCFDUSD", "BTC", "FDUSD"),
    "ETHUSD": ("ETHFDUSD", "ETH", "FDUSD"),
    "SOLUSD": ("SOLUSDT", "SOL", "USDT"),
    "DOGEUSD": ("DOGEUSDT", "DOGE", "USDT"),
    "AAVEUSD": ("AAVEUSDT", "AAVE", "USDT"),
    "LINKUSD": ("LINKUSDT", "LINK", "USDT"),
    "XRPUSD": ("XRPUSDT", "XRP", "USDT"),
}


@dataclass
class SymbolContext:
    symbol: str
    price: float
    klines: pd.DataFrame  # OHLCV
    ret_1h: float = 0.0
    ret_24h: float = 0.0
    ret_72h: float = 0.0
    volatility_24h: float = 0.0
    atr_pct: float = 0.0
    trend_72h: float = 0.0
    drawdown_72h: float = 0.0
    # Chronos2
    fc_h1_close_delta: float = 0.0
    fc_h1_high_delta: float = 0.0
    fc_h1_low_delta: float = 0.0
    fc_h1_confidence: float = 0.0
    fc_h24_close_delta: float = 0.0
    fc_h24_high_delta: float = 0.0
    fc_h24_low_delta: float = 0.0
    fc_h24_confidence: float = 0.0
    # RL
    rl_long_prob: float = 0.0
    rl_short_prob: float = 0.0


@dataclass
class AllocationPlan:
    """Gemini's output: per-symbol allocation + reasoning."""
    allocations: dict[str, float] = field(default_factory=dict)  # symbol -> pct 0-100
    entry_prices: dict[str, float] = field(default_factory=dict)
    exit_prices: dict[str, float] = field(default_factory=dict)
    reasoning: str = ""
    timestamp: str = ""

    @property
    def cash_pct(self) -> float:
        return max(0.0, 100.0 - sum(self.allocations.values()))


@dataclass
class PlanOutcome:
    """What happened after executing a plan (for context in next cycle)."""
    plan: AllocationPlan
    pnl_usd: float = 0.0
    pnl_pct: float = 0.0
    fills: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Context gathering
# ---------------------------------------------------------------------------

def _fetch_klines(binance_pair: str, limit: int = 96) -> pd.DataFrame:
    from src.binan import binance_wrapper as bw
    try:
        klines = bw.get_client().get_klines(symbol=binance_pair, interval="1h", limit=limit)
    except Exception:
        alt_pair = binance_pair.replace("FDUSD", "USDT")
        klines = bw.get_client().get_klines(symbol=alt_pair, interval="1h", limit=limit)
    rows = []
    for k in klines:
        rows.append({
            "timestamp": pd.Timestamp(k[0], unit="ms", tz="UTC").floor("h"),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
        })
    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    return df[~df.index.duplicated(keep="last")]


def _compute_technicals(klines: pd.DataFrame) -> dict:
    if len(klines) < 2:
        return {}
    close = klines["close"]
    high = klines["high"]
    low = klines["low"]
    c = close.iloc[-1]
    out = {}
    if len(close) >= 2:
        out["ret_1h"] = (c - close.iloc[-2]) / close.iloc[-2]
    if len(close) >= 25:
        out["ret_24h"] = (c - close.iloc[-25]) / close.iloc[-25]
    if len(close) >= 73:
        out["ret_72h"] = (c - close.iloc[-73]) / close.iloc[-73]
    ret_1h_series = close.pct_change(1).dropna()
    if len(ret_1h_series) >= 24:
        out["volatility_24h"] = float(ret_1h_series.iloc[-24:].std())
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    if len(tr) >= 24:
        out["atr_pct"] = float(tr.iloc[-24:].mean() / c)
    if len(close) >= 73:
        out["trend_72h"] = (c - close.iloc[-73]) / close.iloc[-73]
    if len(close) >= 72:
        roll_max = close.iloc[-72:].max()
        out["drawdown_72h"] = (c - roll_max) / roll_max
    return out


def _get_forecast_latest(cache_root: Path, symbol: str, horizon: int) -> dict:
    fc = _load_forecast_parquet(cache_root, horizon, symbol)
    if fc.empty:
        return {}
    row = fc.iloc[-1]
    out = {}
    for c in fc.columns:
        if c in row.index:
            try:
                out[c] = float(row[c])
            except (TypeError, ValueError):
                pass
    return out


def gather_symbol_contexts(
    forecast_cache_root: Path,
) -> list[SymbolContext]:
    """Fetch all data for all 4 RL symbols."""
    contexts = []
    for sym in SYMBOLS:
        pair, base, quote = SYMBOL_BINANCE_MAP[sym]
        try:
            klines = _fetch_klines(pair, limit=96)
        except Exception as e:
            logger.warning(f"Failed klines for {sym}: {e}")
            continue

        price = float(klines["close"].iloc[-1])
        tech = _compute_technicals(klines)

        # Chronos2 forecasts
        fc_h1 = _get_forecast_latest(forecast_cache_root, sym, 1)
        fc_h24 = _get_forecast_latest(forecast_cache_root, sym, 24)

        def delta(fc, col):
            v = fc.get(col, 0)
            if v and price > 0:
                return (v - price) / price
            return 0.0

        def confidence(fc):
            p90 = fc.get("predicted_close_p90", 0)
            p10 = fc.get("predicted_close_p10", 0)
            if p90 and p10 and price > 0:
                return 1.0 / (1.0 + abs(p90 - p10) / price)
            return 0.0

        ctx = SymbolContext(
            symbol=sym,
            price=price,
            klines=klines,
            ret_1h=tech.get("ret_1h", 0),
            ret_24h=tech.get("ret_24h", 0),
            ret_72h=tech.get("ret_72h", 0),
            volatility_24h=tech.get("volatility_24h", 0),
            atr_pct=tech.get("atr_pct", 0),
            trend_72h=tech.get("trend_72h", 0),
            drawdown_72h=tech.get("drawdown_72h", 0),
            fc_h1_close_delta=delta(fc_h1, "predicted_close_p50"),
            fc_h1_high_delta=delta(fc_h1, "predicted_high_p50"),
            fc_h1_low_delta=delta(fc_h1, "predicted_low_p50"),
            fc_h1_confidence=confidence(fc_h1),
            fc_h24_close_delta=delta(fc_h24, "predicted_close_p50"),
            fc_h24_high_delta=delta(fc_h24, "predicted_high_p50"),
            fc_h24_low_delta=delta(fc_h24, "predicted_low_p50"),
            fc_h24_confidence=confidence(fc_h24),
        )
        contexts.append(ctx)
    return contexts


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _fmt_price(v: float) -> str:
    if v < 1:
        return f"{v:.5f}"
    if v < 100:
        return f"{v:.2f}"
    return f"{v:,.2f}"


def _format_klines_compact(klines: pd.DataFrame, n: int = 12) -> str:
    rows = klines.iloc[-n:]
    lines = []
    for ts, r in rows.iterrows():
        t = str(ts)[:16]
        o, h, l, c = _fmt_price(r['open']), _fmt_price(r['high']), _fmt_price(r['low']), _fmt_price(r['close'])
        lines.append(f"  {t}  O={o:<12s} H={h:<12s} L={l:<12s} C={c:<12s} V={r['volume']:.0f}")
    return "\n".join(lines)


def _softmax(logits: list[float]) -> list[float]:
    x = np.array(logits, dtype=np.float64)
    e = np.exp(x - x.max())
    return (e / e.sum()).tolist()


def build_allocation_prompt(
    contexts: list[SymbolContext],
    rl_signal: RLSignal,
    portfolio_value: float,
    cash_usd: float,
    positions: dict[str, float],  # base_asset -> qty
    prev_plan: Optional[AllocationPlan] = None,
    prev_outcome: Optional[PlanOutcome] = None,
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    probs = _softmax(rl_signal.logits)
    prob_labels = [f"{ACTION_NAMES[i]}={probs[i]:.0%}" for i in range(len(probs))]

    # Portfolio section
    pos_lines = []
    for ctx in contexts:
        _, base, _ = SYMBOL_BINANCE_MAP[ctx.symbol]
        qty = positions.get(base, 0)
        val = qty * ctx.price
        pct = val / max(portfolio_value, 1) * 100
        if val >= 1.0:
            pos_lines.append(f"  {ctx.symbol}: {qty:.6f} ({base}) = ${val:.2f} ({pct:.1f}%)")
    cash_pct = cash_usd / max(portfolio_value, 1) * 100
    pos_section = "\n".join(pos_lines) if pos_lines else "  (none)"

    # Per-symbol market context
    sym_sections = []
    for i, ctx in enumerate(contexts):
        fc_h1_str = "unavailable"
        if ctx.fc_h1_confidence > 0:
            fc_h1_str = (
                f"close {ctx.fc_h1_close_delta:+.2%} (conf={ctx.fc_h1_confidence:.2f}), "
                f"high {ctx.fc_h1_high_delta:+.2%}, low {ctx.fc_h1_low_delta:+.2%}"
            )
        fc_h24_str = "unavailable"
        if ctx.fc_h24_confidence > 0:
            fc_h24_str = (
                f"close {ctx.fc_h24_close_delta:+.2%} (conf={ctx.fc_h24_confidence:.2f}), "
                f"high {ctx.fc_h24_high_delta:+.2%}, low {ctx.fc_h24_low_delta:+.2%}"
            )

        rl_long_prob = probs[i + 1]  # actions 1-4 = long BTC/ETH/DOGE/AAVE
        rl_short_prob = probs[i + 5]  # actions 5-8 = short

        kline_table = _format_klines_compact(ctx.klines, n=12)

        sym_sections.append(f"""--- {ctx.symbol} ---
Price: ${_fmt_price(ctx.price)}
Returns: 1h={ctx.ret_1h:+.2%} | 24h={ctx.ret_24h:+.2%} | 72h={ctx.ret_72h:+.2%}
Volatility(24h): {ctx.volatility_24h:.2%} | ATR: {ctx.atr_pct:.2%} | Drawdown(72h): {ctx.drawdown_72h:.2%}
Chronos2 1h forecast:  {fc_h1_str}
Chronos2 24h forecast: {fc_h24_str}
RL policy: long={rl_long_prob:.0%} | short={rl_short_prob:.0%}
Last 12h:
{kline_table}""")

    market_section = "\n\n".join(sym_sections)

    # Previous plan section
    prev_section = "(first cycle, no previous plan)"
    if prev_plan:
        alloc_lines = []
        for s, pct in prev_plan.allocations.items():
            if pct > 0:
                ep = prev_plan.entry_prices.get(s, 0)
                xp = prev_plan.exit_prices.get(s, 0)
                alloc_lines.append(f"  {s}: {pct:.0f}% (entry=${ep:,.2f}, exit=${xp:,.2f})")
        alloc_str = "\n".join(alloc_lines) if alloc_lines else "  100% cash"
        prev_section = f"Plan @ {prev_plan.timestamp}:\n{alloc_str}\nReasoning: {prev_plan.reasoning[:200]}"
        if prev_outcome:
            prev_section += f"\nOutcome: PnL=${prev_outcome.pnl_usd:+.2f} ({prev_outcome.pnl_pct:+.2%})"
            if prev_outcome.fills:
                prev_section += f"\nFills: {', '.join(prev_outcome.fills[:5])}"

    # RL summary
    rl_rec = rl_signal.action_name
    rl_value = rl_signal.value

    prompt = f"""You are a quantitative portfolio manager optimizing hourly allocations across 4 crypto assets on Binance spot (long-only, no leverage, no shorting).

OBJECTIVE: Maximize risk-adjusted returns. Prioritize:
1. Sortino ratio (penalize downside, reward upside)
2. Smooth equity curve (avoid drawdowns)
3. Consistent positive returns over raw magnitude
Being in cash (0% allocated) is always valid. Only allocate when multiple signals align.

TIME: {now}

=== PORTFOLIO STATE ===
Total value: ${portfolio_value:,.2f}
Cash: ${cash_usd:,.2f} ({cash_pct:.1f}%)
Positions:
{pos_section}

=== PREVIOUS HOUR ===
{prev_section}

=== MARKET DATA & SIGNALS ===

{market_section}

=== RL POLICY ANALYSIS ===
Trained on 40K+ hours of historical data, 4-symbol portfolio rotator.
Recommendation: {rl_rec} | Value estimate: {rl_value:.3f}
Action probabilities: {', '.join(prob_labels)}
Note: RL policy can short but we are spot-only (long or cash).

=== ALLOCATION INSTRUCTIONS ===
Set target allocation % for each symbol (0-100). Sum of all allocations must be <= 100.
The remainder stays in cash (stablecoins).

For each symbol with allocation > 0, set:
- entry_price: limit buy price (slightly below current for favorable fill)
- exit_price: take-profit price (your target exit)

Guidelines:
- Fees: BTC/ETH on FDUSD = 0 fees. DOGE/AAVE on USDT = 10bps per side.
- Only allocate to symbols where Chronos2 forecast AND technicals AND RL signal show alignment.
- Prefer fewer concentrated bets over thin spread across many symbols.
- If all signals are mixed or bearish, stay mostly/fully in cash. Cash is a position.
- Think about correlation: BTC and ETH often move together, diversification into DOGE/AAVE can reduce risk.

Respond with JSON:
{{
  "btc_pct": <0-100>,
  "btc_entry": <price or 0>,
  "btc_exit": <price or 0>,
  "eth_pct": <0-100>,
  "eth_entry": <price or 0>,
  "eth_exit": <price or 0>,
  "doge_pct": <0-100>,
  "doge_entry": <price or 0>,
  "doge_exit": <price or 0>,
  "aave_pct": <0-100>,
  "aave_entry": <price or 0>,
  "aave_exit": <price or 0>,
  "reasoning": "<2-3 sentences explaining your allocation rationale>"
}}"""
    return prompt


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

ALLOC_FIELDS = {
    "BTCUSD": ("btc_pct", "btc_entry", "btc_exit"),
    "ETHUSD": ("eth_pct", "eth_entry", "eth_exit"),
    "DOGEUSD": ("doge_pct", "doge_entry", "doge_exit"),
    "AAVEUSD": ("aave_pct", "aave_entry", "aave_exit"),
}


def parse_allocation_response(text: str) -> AllocationPlan:
    """Parse Gemini's JSON response into an AllocationPlan."""
    # Try structured JSON first
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r'\{[^{}]*"btc_pct"[^{}]*\}', text, re.DOTALL)
        if not m:
            m = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0) if not m.lastindex else m.group(1))
            except json.JSONDecodeError:
                return AllocationPlan(reasoning="Failed to parse response")
        else:
            return AllocationPlan(reasoning="No JSON found in response")

    plan = AllocationPlan(
        reasoning=str(data.get("reasoning", "")),
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )

    for sym, (pct_key, entry_key, exit_key) in ALLOC_FIELDS.items():
        pct = float(data.get(pct_key, 0) or 0)
        if pct > 0:
            plan.allocations[sym] = min(pct, 100.0)
            plan.entry_prices[sym] = float(data.get(entry_key, 0) or 0)
            plan.exit_prices[sym] = float(data.get(exit_key, 0) or 0)

    # Clamp total to 100%
    total = sum(plan.allocations.values())
    if total > 100:
        scale = 100.0 / total
        plan.allocations = {s: v * scale for s, v in plan.allocations.items()}

    return plan


# ---------------------------------------------------------------------------
# Gemini API call
# ---------------------------------------------------------------------------

def call_gemini_allocation(
    prompt: str,
    model: str = "gemini-3.1-flash-lite-preview",
    max_retries: int = 3,
) -> AllocationPlan:
    """Call Gemini and parse allocation response."""
    from google import genai
    from google.genai import types

    schema = genai.types.Schema(
        type=genai.types.Type.OBJECT,
        required=["btc_pct", "eth_pct", "doge_pct", "aave_pct", "reasoning"],
        properties={
            "btc_pct": genai.types.Schema(type=genai.types.Type.STRING),
            "btc_entry": genai.types.Schema(type=genai.types.Type.STRING),
            "btc_exit": genai.types.Schema(type=genai.types.Type.STRING),
            "eth_pct": genai.types.Schema(type=genai.types.Type.STRING),
            "eth_entry": genai.types.Schema(type=genai.types.Type.STRING),
            "eth_exit": genai.types.Schema(type=genai.types.Type.STRING),
            "doge_pct": genai.types.Schema(type=genai.types.Type.STRING),
            "doge_entry": genai.types.Schema(type=genai.types.Type.STRING),
            "doge_exit": genai.types.Schema(type=genai.types.Type.STRING),
            "aave_pct": genai.types.Schema(type=genai.types.Type.STRING),
            "aave_entry": genai.types.Schema(type=genai.types.Type.STRING),
            "aave_exit": genai.types.Schema(type=genai.types.Type.STRING),
            "reasoning": genai.types.Schema(type=genai.types.Type.STRING),
        },
    )

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
    config_kwargs = dict(
        response_mime_type="application/json",
        response_schema=schema,
        temperature=0.3,
    )
    # Add thinking for models that support it
    if "lite" not in model:
        try:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=2048)
        except Exception:
            pass
    config = types.GenerateContentConfig(**config_kwargs)

    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
                config=config,
            )
            plan = parse_allocation_response(resp.text)
            logger.info(f"Gemini allocation: {plan.allocations} cash={plan.cash_pct:.0f}%")
            logger.info(f"Reasoning: {plan.reasoning[:150]}")
            return plan
        except Exception as e:
            err = str(e)
            if "429" in err:
                wait = 15 * (attempt + 1)
                logger.warning(f"Rate limited, retry in {wait}s")
                time.sleep(wait)
            elif attempt < max_retries - 1:
                time.sleep(3 * (attempt + 1))
            else:
                logger.error(f"Gemini call failed: {e}")
                return AllocationPlan(reasoning=f"API error: {e}")
    return AllocationPlan(reasoning="All retries exhausted")
