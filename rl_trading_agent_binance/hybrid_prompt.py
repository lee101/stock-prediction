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

import numpy as np
import pandas as pd


try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from rl_signal import (
    ACTION_NAMES,
    SYMBOLS,
    RLSignal,
    _load_forecast_parquet,
)

UTC = timezone.utc


SYMBOL_BINANCE_MAP = {
    "BTCUSD": ("BTCFDUSD", "BTC", "FDUSD"),
    "ETHUSD": ("ETHFDUSD", "ETH", "FDUSD"),
    "SOLUSD": ("SOLUSDT", "SOL", "USDT"),
    "LTCUSD": ("LTCUSDT", "LTC", "USDT"),
    "AVAXUSD": ("AVAXUSDT", "AVAX", "USDT"),
    "DOGEUSD": ("DOGEUSDT", "DOGE", "USDT"),
    "LINKUSD": ("LINKUSDT", "LINK", "USDT"),
    "ADAUSD": ("ADAUSDT", "ADA", "USDT"),
    "UNIUSD": ("UNIUSDT", "UNI", "USDT"),
    "AAVEUSD": ("AAVEUSDT", "AAVE", "USDT"),
    "ALGOUSD": ("ALGOUSDT", "ALGO", "USDT"),
    "DOTUSD": ("DOTUSDT", "DOT", "USDT"),
    "SHIBUSD": ("SHIBUSDT", "SHIB", "USDT"),
    "XRPUSD": ("XRPUSDT", "XRP", "USDT"),
    "MATICUSD": ("MATICUSDT", "MATIC", "USDT"),
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


def _fetch_klines(binance_pair: str, limit: int = 96) -> pd.DataFrame:
    from src.binan import binance_wrapper as bw

    try:
        klines = bw.get_client().get_klines(symbol=binance_pair, interval="1h", limit=limit)
    except Exception:
        alt_pair = binance_pair.replace("FDUSD", "USDT")
        klines = bw.get_client().get_klines(symbol=alt_pair, interval="1h", limit=limit)
    rows = []
    for k in klines:
        rows.append(
            {
                "timestamp": pd.Timestamp(k[0], unit="ms", tz="UTC").floor("h"),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            }
        )
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
    symbols: tuple[str, ...] | None = None,
) -> list[SymbolContext]:
    """Fetch data for all tradable RL symbols."""
    if symbols is None:
        symbols = SYMBOLS
    contexts = []
    for sym in symbols:
        if sym not in SYMBOL_BINANCE_MAP:
            continue
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
        o, h, l, c = _fmt_price(r["open"]), _fmt_price(r["high"]), _fmt_price(r["low"]), _fmt_price(r["close"])
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
    prev_plan: AllocationPlan | None = None,
    prev_outcome: PlanOutcome | None = None,
    rl_symbols: tuple[str, ...] | None = None,
    rl_action_names: list[str] | None = None,
) -> str:
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    probs = _softmax(rl_signal.logits)
    action_names = rl_action_names if rl_action_names else ACTION_NAMES
    prob_labels = [f"{action_names[i]}={probs[i]:.0%}" for i in range(min(len(probs), len(action_names)))]

    # Build symbol -> action index mapping for RL probs
    _rl_syms = rl_symbols if rl_symbols else SYMBOLS
    _n_rl = len(_rl_syms)
    sym_to_rl_idx = {sym: i for i, sym in enumerate(_rl_syms)}

    # Portfolio section
    pos_lines = []
    for ctx in contexts:
        if ctx.symbol not in SYMBOL_BINANCE_MAP:
            continue
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

        # Look up RL prob indices dynamically
        rl_idx = sym_to_rl_idx.get(ctx.symbol)
        if rl_idx is not None and (rl_idx + 1) < len(probs):
            rl_long_prob = probs[rl_idx + 1]
            rl_short_prob = probs[_n_rl + rl_idx + 1] if (_n_rl + rl_idx + 1) < len(probs) else 0.0
        else:
            rl_long_prob = 0.0
            rl_short_prob = 0.0

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

    n_ctx = len(contexts)
    ctx_syms = [ctx.symbol for ctx in contexts]

    # Build dynamic JSON schema for response
    json_fields = []
    for ctx in contexts:
        short = ctx.symbol.replace("USD", "").lower()
        json_fields.append(f'  "{short}_pct": <0-100>')
        json_fields.append(f'  "{short}_entry": <price or 0>')
        json_fields.append(f'  "{short}_exit": <price or 0>')
    json_fields.append('  "reasoning": "<2-3 sentences explaining your allocation rationale>"')
    json_template = "{{\n" + ",\n".join(json_fields) + "\n}}"

    # Determine fee info
    fee_info = []
    for ctx in contexts:
        if ctx.symbol in SYMBOL_BINANCE_MAP:
            _, _, quote = SYMBOL_BINANCE_MAP[ctx.symbol]
            fee = "0 fees" if quote == "FDUSD" else "10bps per side"
            fee_info.append(f"{ctx.symbol.replace('USD', '')} on {quote} = {fee}")

    prompt = f"""Optimize hourly trade plan across {n_ctx} assets.

OBJECTIVE: Maximize risk-adjusted returns. Prioritize:
Sortino ratio (penalize downside, reward upside)
Smooth equity curve (avoid drawdowns)
Consistent positive returns over raw magnitude

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
{_n_rl}-symbol portfolio rotator.
Recommendation: {rl_rec} | Value estimate: {rl_value:.3f}
Action probabilities: {", ".join(prob_labels[:20])}

=== ALLOCATION INSTRUCTIONS ===
Set target allocation % for each symbol (0-100). Sum of all allocations must be <= 500. max 5x leverage
The remainder stays in cash.

For each symbol with allocation > 0, set:
- entry_price: limit buy price (slightly below current for favorable fill)
- exit_price: take-profit price (your target exit)

Guidelines:
- Fees: {". ".join(fee_info)}.

Respond with JSON:
{json_template}"""
    return prompt


ALLOC_FIELDS = {
    "BTCUSD": ("btc_pct", "btc_entry", "btc_exit"),
    "ETHUSD": ("eth_pct", "eth_entry", "eth_exit"),
    "SOLUSD": ("sol_pct", "sol_entry", "sol_exit"),
    "LTCUSD": ("ltc_pct", "ltc_entry", "ltc_exit"),
    "AVAXUSD": ("avax_pct", "avax_entry", "avax_exit"),
    "DOGEUSD": ("doge_pct", "doge_entry", "doge_exit"),
    "LINKUSD": ("link_pct", "link_entry", "link_exit"),
    "ADAUSD": ("ada_pct", "ada_entry", "ada_exit"),
    "UNIUSD": ("uni_pct", "uni_entry", "uni_exit"),
    "AAVEUSD": ("aave_pct", "aave_entry", "aave_exit"),
    "ALGOUSD": ("algo_pct", "algo_entry", "algo_exit"),
    "DOTUSD": ("dot_pct", "dot_entry", "dot_exit"),
    "SHIBUSD": ("shib_pct", "shib_entry", "shib_exit"),
    "XRPUSD": ("xrp_pct", "xrp_entry", "xrp_exit"),
    "MATICUSD": ("matic_pct", "matic_entry", "matic_exit"),
}


def _alloc_fields_for_symbol(sym: str) -> tuple[str, str, str]:
    if sym in ALLOC_FIELDS:
        return ALLOC_FIELDS[sym]
    short = sym.replace("USD", "").lower()
    return (f"{short}_pct", f"{short}_entry", f"{short}_exit")


def _coerce_allocation_number(value: object) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return 0.0
    lowered = text.lower()
    if lowered in {"n/a", "na", "none", "null"}:
        return 0.0
    cleaned = text.replace("$", "").replace("%", "").replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def parse_allocation_response(text: str) -> AllocationPlan:
    """Parse Gemini's JSON response into an AllocationPlan."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r'\{[^{}]*"[a-z]+_pct"[^{}]*\}', text, re.DOTALL)
        if not m:
            m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0) if not m.lastindex else m.group(1))
            except json.JSONDecodeError:
                return AllocationPlan(reasoning="Failed to parse response")
        else:
            return AllocationPlan(reasoning="No JSON found in response")

    plan = AllocationPlan(
        reasoning=str(data.get("reasoning", "")),
        timestamp=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
    )

    for sym in ALLOC_FIELDS:
        pct_key, entry_key, exit_key = _alloc_fields_for_symbol(sym)
        pct = _coerce_allocation_number(data.get(pct_key, 0))
        if pct > 0:
            plan.allocations[sym] = min(pct, 100.0)
            plan.entry_prices[sym] = _coerce_allocation_number(data.get(entry_key, 0))
            plan.exit_prices[sym] = _coerce_allocation_number(data.get(exit_key, 0))

    # Also parse any _pct keys not in ALLOC_FIELDS
    for key, val in data.items():
        if key.endswith("_pct") and key != "reasoning":
            short = key.replace("_pct", "")
            sym = f"{short.upper()}USD"
            if sym not in plan.allocations:
                pct = _coerce_allocation_number(val)
                if pct > 0:
                    plan.allocations[sym] = min(pct, 100.0)
                    plan.entry_prices[sym] = _coerce_allocation_number(data.get(f"{short}_entry", 0))
                    plan.exit_prices[sym] = _coerce_allocation_number(data.get(f"{short}_exit", 0))

    total = sum(plan.allocations.values())
    if total > 100:
        scale = 100.0 / total
        plan.allocations = {s: v * scale for s, v in plan.allocations.items()}

    return plan


def _build_allocation_response_schema(
    genai_types: object,
    tradable_symbols: list[str],
):
    """Build a response schema whose required keys all exist in properties."""

    props = {
        "reasoning": genai_types.Schema(type=genai_types.Type.STRING),
    }
    required = ["reasoning"]
    for sym in tradable_symbols:
        pct_key, entry_key, exit_key = _alloc_fields_for_symbol(sym)
        props[pct_key] = genai_types.Schema(type=genai_types.Type.STRING)
        props[entry_key] = genai_types.Schema(type=genai_types.Type.STRING)
        props[exit_key] = genai_types.Schema(type=genai_types.Type.STRING)
        required.append(pct_key)

    return genai_types.Schema(
        type=genai_types.Type.OBJECT,
        required=required,
        properties=props,
    )


def call_gemini_allocation(
    prompt: str,
    model: str = "gemini-3.1-flash-lite-preview",
    max_retries: int = 3,
    tradable_symbols: list[str] | None = None,
) -> AllocationPlan:
    """Call Gemini and parse allocation response."""
    from google import genai
    from google.genai import types

    if tradable_symbols is None:
        tradable_symbols = ["BTCUSD", "ETHUSD", "DOGEUSD", "AAVEUSD"]
    schema = _build_allocation_response_schema(genai.types, tradable_symbols)

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set - generate a new key at Google AI Studio")
    client = genai.Client(api_key=api_key)
    config_kwargs = dict(
        response_mime_type="application/json",
        response_schema=schema,
        temperature=0.3,
    )
    # Add thinking for models that support it
    if "lite" not in model:
        try:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=2048)
        except Exception as e:
            logger.exception(e)
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
