"""
Prompt experiment v2: test optimization prompt (long-only) vs original prompt.
Then test winner with gemini-3.1-pro-preview.
"""
from __future__ import annotations
import argparse, json, sys, time, concurrent.futures
from pathlib import Path
from datetime import timedelta
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_hourly_trader.backtest import load_bars, load_forecasts, get_forecast_at, simulate, RESULTS_DIR
from llm_hourly_trader.config import SYMBOL_UNIVERSE, BacktestConfig, SymbolConfig
from llm_hourly_trader.gemini_wrapper import TradePlan
from llm_hourly_trader.providers import call_llm
from llm_hourly_trader.cache import get_cached, set_cached
from run_hybrid import (
    MLPPolicy, BINANCE6_SYMBOLS, TradingSignal, _compute_trend_context,
)
from pufferlib_market.export_data_hourly_forecast import (
    compute_features as compute_mktd_features, _read_hourly_prices, _read_forecast,
)
import torch


def _market_context(symbol, history_rows, current_price, fc_1h, fc_24h, rl_signal, prev_rl_signal, prev_outcome):
    recent = history_rows[-12:]
    price_lines = []
    for row in recent:
        ts = str(row.get("timestamp", ""))[-19:-6] if "timestamp" in row else "?"
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        price_lines.append(f"  {ts}: O={o:.2f} H={h:.2f} L={l:.2f} C={c:.2f}")
    trend = _compute_trend_context(history_rows)
    trend_parts = []
    for key in ["ret_24h", "ret_48h", "ret_72h"]:
        if key in trend:
            trend_parts.append(f"{key.replace('ret_', '')}: {trend[key]:+.2f}%")
    if "atr_pct" in trend:
        trend_parts.append(f"ATR: {trend['atr_pct']:.2f}%")
    if "up_hours_12" in trend:
        trend_parts.append(f"Up hrs (12h): {trend['up_hours_12']}/12")
    trend_line = " | ".join(trend_parts)
    sr_line = ""
    if "low_24h" in trend:
        sr_line = f"  24h range: ${trend['low_24h']:.2f} - ${trend['high_24h']:.2f} ({trend['range_pct']:.2f}% wide)"
    fc_text = ""
    if fc_1h:
        delta = (fc_1h['predicted_close_p50'] - current_price) / current_price * 100
        fc_text += f"\n  1h forecast: close={fc_1h['predicted_close_p50']:.2f} ({delta:+.2f}%)"
    if fc_24h:
        delta = (fc_24h['predicted_close_p50'] - current_price) / current_price * 100
        fc_text += f"\n  24h forecast: close={fc_24h['predicted_close_p50']:.2f} ({delta:+.2f}%)"
    rl_text = f"Direction: {rl_signal.action}, Confidence: {rl_signal.confidence:.1%}, Value: {rl_signal.value_estimate:.4f}"
    prev_text = "None (first hour)"
    if prev_rl_signal is not None:
        prev_text = f"RL: {prev_rl_signal.action} (conf={prev_rl_signal.confidence:.1%}), Outcome: {prev_outcome}"
    return f"""SYMBOL: {symbol}
CURRENT PRICE: ${current_price:.2f}
{sr_line}
TREND: {trend_line}
LAST 12H:
{chr(10).join(price_lines)}
FORECASTS:{fc_text}
RL MODEL (PPO): {rl_text}
PREVIOUS: {prev_text}"""


# Original prompt (from the +6.82% backtest)
PROMPT_ORIGINAL = """You are an expert crypto trader reviewing an AI trading model's recommendation.

{context}

IMPORTANT: We can ONLY go long (spot market, no shorting). You should be actively looking for long entry opportunities.

TASK: You are a profitable swing trader. Enter long when ANY of these conditions are met:
1. Chronos forecast predicts higher price in 1-24 hours
2. Price is near recent support (24h low) with room to bounce
3. Momentum is turning up after a dip (buy the dip)
4. Price consolidating near highs (breakout setup)

SIZING & TARGETS:
- Set buy_price slightly below current price (0.1-0.3% below for normal vol, 0.3-0.5% below for high vol)
- Set sell_price at a realistic target (0.5-2% above entry, wider in high vol)
- Aim for 2:1+ reward:risk ratio
- Confidence: 0.6-0.9 for strong setups, 0.3-0.5 for marginal setups

You should be entering trades roughly 25-40% of the time. Hold only when price action and forecasts clearly indicate continued downside.

Respond with JSON: {{"direction": "long" or "hold", "buy_price": <limit entry near support>, "sell_price": <take profit target>, "confidence": <0-1>, "reasoning": "<brief>"}}"""


# Optimization prompt (long-only version)
PROMPT_OPTIMIZATION_LONG = """You are solving an optimization problem: maximize risk-adjusted returns on a crypto spot account.

CONSTRAINTS:
- LONG ONLY (spot market, no shorting)
- 1-hour decision intervals, max 6-hour hold time
- Transaction cost: 0.1% maker fee per trade
- Objective: maximize Sortino ratio (penalize downside deviation, reward upside)

{context}

OPTIMIZATION TASK:
This is a fascinating mathematical challenge. You need to find the optimal balance between:
1. ENTRY FREQUENCY: Too few trades = missed alpha. Too many = fee drag kills returns.
2. POSITION SIZING: Set entries that maximize fill probability while minimizing adverse selection.
3. DIRECTIONAL ACCURACY: A 55% hit rate with 2:1 R:R is highly profitable.
4. RISK ASYMMETRY: The Sortino ratio only penalizes downside. Take asymmetric bets where upside >> downside.

Think deeply about the probability distribution of the next 1-6 hours:
- What is the expected move? (forecast + trend + momentum)
- What is the variance? (ATR, recent range)
- Is the risk/reward skewed favorably?
- What does the RL model's value estimate tell you about state quality?

Only enter when expected_return > fees + slippage. Set buy_price and sell_price to capture the most likely profitable range.

Respond with JSON: {{"direction": "long" or "hold", "buy_price": <entry>, "sell_price": <take profit>, "confidence": <0-1>, "reasoning": "<brief>"}}"""


PROMPTS = {
    "original": PROMPT_ORIGINAL,
    "optimization_long": PROMPT_OPTIMIZATION_LONG,
}


def build_prompt(variant, symbol, history_rows, fc_1h, fc_24h, rl_signal, prev_rl_signal, prev_outcome):
    current_price = float(history_rows[-1]["close"])
    context = _market_context(symbol, history_rows, current_price, fc_1h, fc_24h, rl_signal, prev_rl_signal, prev_outcome)
    return PROMPTS[variant].format(context=context)


def run_experiment(variant, symbols, days, model, parallel, checkpoint_path, thinking_level=None):
    cache_tag = f"prompt-v2-{variant}-{model}"
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt["model"]
    ns = len(BINANCE6_SYMBOLS)
    obs_size = ns * 16 + 5 + ns
    hidden = sd["encoder.0.weight"].shape[0]
    na = sd["actor.2.weight"].shape[0]
    rl = MLPPolicy(obs_size, na, hidden)
    rl.load_state_dict(sd)
    rl.eval()

    DR = REPO / "trainingdatahourly"
    FR = REPO / "binanceneural" / "forecast_cache"
    ab, af1, af24, amf = {}, {}, {}, {}
    for s in symbols:
        ab[s] = load_bars(s)
        af1[s] = load_forecasts(s, "h1")
        af24[s] = load_forecasts(s, "h24")
        try:
            p = _read_hourly_prices(s, DR)
            f1 = _read_forecast(s, FR, 1) if (FR / f"{s}_h1.parquet").exists() else pd.DataFrame()
            f24 = _read_forecast(s, FR, 24) if (FR / f"{s}_h24.parquet").exists() else pd.DataFrame()
            amf[s] = compute_mktd_features(p, f1, f24)
        except Exception:
            amf[s] = None

    abf = {}
    for b in BINANCE6_SYMBOLS:
        if b in amf and amf[b] is not None:
            abf[b] = amf[b]
        else:
            try:
                p = _read_hourly_prices(b, DR)
                f1 = _read_forecast(b, FR, 1) if (FR / f"{b}_h1.parquet").exists() else pd.DataFrame()
                f24 = _read_forecast(b, FR, 24) if (FR / f"{b}_h24.parquet").exists() else pd.DataFrame()
                abf[b] = compute_mktd_features(p, f1, f24)
            except Exception:
                abf[b] = None

    fe = [af1[s]["timestamp"].max() for s in symbols if not af1[s].empty]
    end_ts = min(fe)
    start_ts = end_ts - timedelta(days=days)
    sc = {s: SYMBOL_UNIVERSE.get(s, SymbolConfig(s, "crypto")) for s in symbols}

    def gfa(ts):
        f = np.zeros((ns, 16), dtype=np.float32)
        tf = pd.Timestamp(ts).floor("h")
        for i, b in enumerate(BINANCE6_SYMBOLS):
            m = abf.get(b)
            if m is None: continue
            if tf in m.index:
                f[i] = m.loc[tf].values[:16].astype(np.float32)
            else:
                bef = m.index[m.index <= tf]
                if len(bef) > 0:
                    f[i] = m.iloc[m.index.get_loc(bef[-1])].values[:16].astype(np.float32)
        return f

    def grs(fa, c):
        o = np.zeros(obs_size, dtype=np.float32)
        o[:ns*16] = fa.flatten()
        o[ns*16] = 1.0; o[ns*16+4] = 0.5
        ot = torch.from_numpy(o).unsqueeze(0)
        with torch.no_grad():
            lg, v = rl(ot)
            pr = torch.softmax(lg, -1)
            a = lg.argmax(-1).item()
            cf = pr[0, a].item()
            vl = v.item()
        if a == 0: return TradingSignal("flat", None, None, cf, vl, 0, 0)
        ai = a - 1
        sh = ai >= ns
        if sh: ai -= ns
        sy = BINANCE6_SYMBOLS[ai] if ai < ns else "?"
        return TradingSignal(f"{'short' if sh else 'long'}_{sy}", sy, "short" if sh else "long", cf, vl, 1, 0)

    tasks = []
    ps = {s: None for s in symbols}
    po = {s: "N/A" for s in symbols}
    for s in symbols:
        sb = ab[s]
        w = sb[(sb["timestamp"] >= start_ts) & (sb["timestamp"] <= end_ts)].copy()
        for i, (_, bar) in enumerate(w.iterrows()):
            ts = bar["timestamp"]
            h = sb[sb["timestamp"] <= ts].tail(72)
            if len(h) < 5:
                tasks.append((s, bar.to_dict(), None, None))
                continue
            f1 = get_forecast_at(af1[s], ts)
            f24 = get_forecast_at(af24[s], ts)
            fa = gfa(ts)
            rs = grs(fa, float(bar["close"]))
            prompt = build_prompt(variant, s, h.tail(72).to_dict("records"), f1, f24, rs, ps[s], po[s])
            tasks.append((s, bar.to_dict(), prompt, rs))
            ps[s] = rs
            if i > 0:
                pc = float(w.iloc[i-1]["close"])
                c = float(bar["close"])
                po[s] = f"{(c-pc)/pc*100:+.2f}% (${pc:.2f}->${c:.2f})"

    total = len(tasks)
    api = sum(1 for _, _, p, _ in tasks if p)
    print(f"  [{variant}] {total} bars, {api} API calls")

    def call(s, b, p, r, i):
        if p is None: return s, b, TradePlan("hold", 0, 0, 0, "no data"), r, i
        c = get_cached(cache_tag, p)
        if c: return s, b, TradePlan(**c), r, i
        plan = call_llm(p, model=model, thinking_level=thinking_level)
        set_cached(cache_tag, p, plan.__dict__)
        return s, b, plan, r, i

    results = [None] * total
    done = 0; t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as pool:
        futs = {pool.submit(call, *tasks[i], i): i for i in range(total)}
        for f in concurrent.futures.as_completed(futs):
            s, b, plan, rs, idx = f.result()
            results[idx] = (s, b, plan, rs)
            done += 1
            if done % max(1, total // 10) == 0:
                el = time.time() - t0
                rate = done / el if el > 0 else 0
                print(f"    [{variant}] {done}/{total} ({rate:.1f}/s)")

    # Long-only: filter to long/hold only
    ar, br = [], []
    stats = {"long": 0, "hold": 0, "short_filtered": 0}
    for s, b, plan, rs in results:
        ts = b["timestamp"] if isinstance(b["timestamp"], pd.Timestamp) else pd.Timestamp(b["timestamp"])
        if plan.direction == "short":
            plan = TradePlan("hold", 0, 0, 0, "short filtered")
            stats["short_filtered"] += 1
        elif plan.direction not in ("long", "hold"):
            plan = TradePlan("hold", 0, 0, 0, "invalid")
        stats[plan.direction] = stats.get(plan.direction, 0) + 1
        ar.append({"timestamp": ts, "symbol": s, "buy_price": plan.buy_price, "sell_price": plan.sell_price, "direction": plan.direction, "confidence": plan.confidence})
        br.append(b)

    cfg = BacktestConfig(initial_cash=10_000.0, max_hold_hours=6, max_position_pct=0.25, model=cache_tag)
    bd = pd.DataFrame(br); ad = pd.DataFrame(ar)
    bd["timestamp"] = pd.to_datetime(bd["timestamp"], utc=True)
    ad["timestamp"] = pd.to_datetime(ad["timestamp"], utc=True)

    result = simulate(bd, ad, cfg, sc)
    m = result["metrics"]
    trades = result["trades"]
    buys = sum(1 for t in trades if t["side"] in ("buy",))
    pnl = sum(t["realized_pnl"] for t in trades)
    fees = sum(t["fee"] for t in trades)

    print(f"\n  [{variant}] RESULTS:")
    print(f"    Return: {m['total_return_pct']:+.4f}%")
    print(f"    Sortino: {m['sortino']:.2f}")
    print(f"    Max DD: {m['max_drawdown_pct']:.2f}%")
    print(f"    Trades: {buys} entries, PnL=${pnl:+.2f}, fees=${fees:.2f}")
    print(f"    Signals: {stats}")
    print(f"    Final equity: ${m['final_equity']:,.2f}")

    return {"variant": variant, "model": model, **m, "entries": buys, "realized_pnl": pnl, "fees": fees, "signal_counts": stats}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", default=["BTCUSD", "ETHUSD", "SOLUSD"])
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--model", default="gemini-3.1-flash-lite-preview")
    p.add_argument("--thinking-level", default="HIGH")
    p.add_argument("--parallel", type=int, default=5)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--variants", nargs="+", default=list(PROMPTS.keys()))
    args = p.parse_args()

    print(f"\n{'='*70}")
    print(f"PROMPT V2 EXPERIMENT: {args.variants}")
    print(f"Model: {args.model}, Days: {args.days}, Long-only")
    print(f"{'='*70}\n")

    results = {}
    for v in args.variants:
        print(f"\n--- {v} ---")
        r = run_experiment(v, args.symbols, args.days, args.model, args.parallel, args.checkpoint, args.thinking_level)
        results[v] = r

    print(f"\n{'='*70}")
    print(f"COMPARISON (Long-only)")
    print(f"{'='*70}")
    print(f"{'Variant':<25} {'Return':>10} {'Sortino':>10} {'MaxDD':>10} {'Trades':>8} {'PnL':>10}")
    print("-" * 75)
    for v, r in sorted(results.items(), key=lambda x: -x[1]["total_return_pct"]):
        print(f"{v:<25} {r['total_return_pct']:>+9.2f}% {r['sortino']:>10.2f} {r['max_drawdown_pct']:>9.2f}% {r['entries']:>8} ${r['realized_pnl']:>+9.2f}")
    best = max(results.items(), key=lambda x: x[1]["sortino"])
    print(f"\nBEST: {best[0]} (Sortino={best[1]['sortino']:.2f})")

    out = RESULTS_DIR / "prompt_experiment_v2.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
