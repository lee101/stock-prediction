"""
Prompt experiment: test different prompt styles for RL+LLM hybrid with 5x leverage.
Runs 3 variants through the backtest simulator and compares results.
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

# ---------------------------------------------------------------------------
# Prompt variants
# ---------------------------------------------------------------------------

def _market_context_block(symbol, history_rows, current_price, fc_1h, fc_24h, rl_signal, prev_rl_signal, prev_outcome):
    """Shared market context block for all prompts."""
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
        delta_1h = (fc_1h['predicted_close_p50'] - current_price) / current_price * 100
        fc_text += f"\n  1h forecast: close={fc_1h['predicted_close_p50']:.2f} ({delta_1h:+.2f}%)"
    if fc_24h:
        delta_24h = (fc_24h['predicted_close_p50'] - current_price) / current_price * 100
        fc_text += f"\n  24h forecast: close={fc_24h['predicted_close_p50']:.2f} ({delta_24h:+.2f}%)"

    rl_text = f"Direction: {rl_signal.action}, Confidence: {rl_signal.confidence:.1%}"
    rl_text += f", Value estimate: {rl_signal.value_estimate:.4f}"

    prev_text = "None (first hour)"
    if prev_rl_signal is not None:
        prev_text = f"RL said: {prev_rl_signal.action} (conf={prev_rl_signal.confidence:.1%})"
        prev_text += f"\n  Outcome: {prev_outcome}"

    return f"""SYMBOL: {symbol}
CURRENT PRICE: ${current_price:.2f}
{sr_line}
TREND: {trend_line}
LAST 12H:
{chr(10).join(price_lines)}
CHRONOS2 FORECASTS:{fc_text}
RL MODEL (PPO): {rl_text}
PREVIOUS HOUR: {prev_text}"""


PROMPT_A_RISK_MANAGED = """You are an expert crypto trader managing a leveraged account.

ACCOUNT RULES:
- Up to 5x leverage (long or short)
- You can go LONG (buy, profit when price rises) or SHORT (sell, profit when price drops)
- Risk management is critical: never risk more than 2% of account on a single trade
- Use tighter stops in high volatility, wider in low volatility

{context}

TASK: Make a trading decision. You can go long, short, or hold.

ENTRY CRITERIA:
- Long: price near support, upward momentum, or bullish forecast
- Short: price near resistance, downward momentum, or bearish forecast
- Hold: no clear edge or conflicting signals

RISK RULES:
- Set stop-loss to limit risk to ~1-2% of account value
- buy_price: entry level (below current for longs, above current for shorts)
- sell_price: take-profit target
- Aim for 2:1+ reward:risk ratio
- Confidence 0.6-0.9 for strong setups, 0.3-0.5 marginal

Enter trades ~30-40% of the time. Protect capital first, then seek profit.

Respond with JSON: {{"direction": "long" or "short" or "hold", "buy_price": <entry>, "sell_price": <take profit>, "confidence": <0-1>, "reasoning": "<brief>"}}"""


PROMPT_B_MAX_PROFIT = """You are the world's most profitable crypto trader. Your goal is to MAXIMIZE RETURNS.

ACCOUNT RULES:
- Up to 5x leverage available (long or short)
- Every hour you don't trade is potential profit left on the table
- The best traders are aggressive when they see edge

{context}

TASK: MAXIMIZE PROFIT. You have 5x leverage - use it wisely.

You should be trading 40-60% of the time. Crypto moves fast and opportunities are everywhere.
- Go LONG when ANY bullish signal exists (forecast up, support bounce, momentum turning)
- Go SHORT when ANY bearish signal exists (forecast down, resistance rejection, momentum fading)
- Only hold when the market is truly flat with zero edge

Set aggressive but realistic targets:
- buy_price: tight entry (0.1-0.3% from current)
- sell_price: ambitious target (1-3% profit)
- High confidence (0.6+) on most trades - commit to your conviction

Respond with JSON: {{"direction": "long" or "short" or "hold", "buy_price": <entry>, "sell_price": <take profit>, "confidence": <0-1>, "reasoning": "<brief>"}}"""


PROMPT_C_OPTIMIZATION = """You are solving an optimization problem: maximize risk-adjusted returns on a leveraged crypto account.

CONSTRAINTS:
- 5x max leverage (long or short positions allowed)
- 1-hour decision intervals, max 6-hour hold time
- Transaction cost: 0.1% maker fee per trade
- Objective: maximize Sortino ratio (penalize downside deviation, reward upside)

{context}

OPTIMIZATION TASK:
This is a fascinating mathematical challenge. You need to find the optimal balance between:
1. ENTRY FREQUENCY: Too few trades = missed alpha. Too many = fee drag kills returns.
2. POSITION SIZING: Leverage amplifies both gains AND losses. The Kelly criterion suggests sizing proportional to edge/variance.
3. DIRECTIONAL ACCURACY: A 55% hit rate with 2:1 R:R is highly profitable. A 70% hit rate with 1:1 R:R is mediocre.
4. RISK ASYMMETRY: The Sortino ratio only penalizes downside. You should take asymmetric bets where upside >> downside.

Think deeply about the probability distribution of the next 1-6 hours. Consider:
- What is the expected move? (forecast + trend + momentum)
- What is the variance? (ATR, recent range)
- Is the risk/reward skewed favorably?
- What does the RL model's value estimate tell you about the state?

Only enter when expected_return > fees + slippage. Set buy_price and sell_price to capture the most likely profitable range.

Respond with JSON: {{"direction": "long" or "short" or "hold", "buy_price": <entry>, "sell_price": <take profit>, "confidence": <0-1>, "reasoning": "<brief>"}}"""


PROMPTS = {
    "risk_managed": PROMPT_A_RISK_MANAGED,
    "max_profit": PROMPT_B_MAX_PROFIT,
    "optimization": PROMPT_C_OPTIMIZATION,
}

SHORT_POLICY_ALLOW = "allow"
SHORT_POLICY_FILTER = "filter"
SHORT_POLICIES = {SHORT_POLICY_ALLOW, SHORT_POLICY_FILTER}


def build_prompt(variant, symbol, history_rows, fc_1h, fc_24h, rl_signal, prev_rl_signal, prev_outcome):
    current_price = float(history_rows[-1]["close"])
    context = _market_context_block(symbol, history_rows, current_price, fc_1h, fc_24h, rl_signal, prev_rl_signal, prev_outcome)
    return PROMPTS[variant].format(context=context)


def _symbol_config_for_short_policy(symbol: str, short_policy: str) -> SymbolConfig:
    if short_policy not in SHORT_POLICIES:
        raise ValueError(f"Unsupported short policy {short_policy!r}; expected one of {sorted(SHORT_POLICIES)}.")
    base = SYMBOL_UNIVERSE.get(symbol)
    asset_class = base.asset_class if base else "crypto"
    maker_fee = base.maker_fee if base else 0.0008
    if short_policy == SHORT_POLICY_ALLOW and asset_class == "crypto":
        directions = ["long", "short"]
    else:
        directions = list(base.allowed_directions) if base else ["long"]
        directions = [direction for direction in directions if direction != "short"]
        if not directions:
            directions = ["long"]
    return SymbolConfig(symbol, asset_class, directions, maker_fee)


def _normalize_plan_for_short_policy(
    plan: TradePlan,
    *,
    last_close: float,
    short_policy: str,
) -> tuple[TradePlan, bool]:
    if short_policy not in SHORT_POLICIES:
        raise ValueError(f"Unsupported short policy {short_policy!r}; expected one of {sorted(SHORT_POLICIES)}.")
    if plan.direction not in ("long", "short", "hold"):
        return TradePlan("hold", 0, 0, 0, "invalid direction"), False
    if short_policy == SHORT_POLICY_FILTER and plan.direction == "short":
        return TradePlan("hold", 0, 0, 0, "short filtered"), True
    if plan.direction != "short":
        return plan, False

    buy_price = plan.buy_price
    sell_price = plan.sell_price
    if buy_price > 0 and buy_price < last_close:
        buy_price = last_close * 1.001
    if sell_price > 0 and sell_price > last_close:
        sell_price = last_close * 0.999
    return TradePlan("short", buy_price, sell_price, plan.confidence, plan.reasoning), False


def run_experiment(
    variant: str,
    symbols: list[str],
    days: int,
    model: str,
    parallel: int,
    checkpoint_path: str,
    leverage: float = 5.0,
    thinking_level: str | None = None,
    short_policy: str = SHORT_POLICY_ALLOW,
):
    cache_tag = f"prompt-exp-{variant}-{model}"
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"]
    num_symbols = len(BINANCE6_SYMBOLS)
    obs_size = num_symbols * 16 + 5 + num_symbols
    hidden = state_dict["encoder.0.weight"].shape[0]
    num_actions = state_dict["actor.2.weight"].shape[0]

    rl_policy = MLPPolicy(obs_size, num_actions, hidden)
    rl_policy.load_state_dict(state_dict)
    rl_policy.eval()

    DATA_ROOT = REPO / "trainingdatahourly"
    FC_ROOT = REPO / "binanceneural" / "forecast_cache"
    all_bars, all_fc_h1, all_fc_h24, all_mktd_features = {}, {}, {}, {}

    for sym in symbols:
        all_bars[sym] = load_bars(sym)
        all_fc_h1[sym] = load_forecasts(sym, "h1")
        all_fc_h24[sym] = load_forecasts(sym, "h24")
        try:
            price_df = _read_hourly_prices(sym, DATA_ROOT)
            fc_h1_raw = _read_forecast(sym, FC_ROOT, 1) if (FC_ROOT / f"{sym}_h1.parquet").exists() else pd.DataFrame()
            fc_h24_raw = _read_forecast(sym, FC_ROOT, 24) if (FC_ROOT / f"{sym}_h24.parquet").exists() else pd.DataFrame()
            all_mktd_features[sym] = compute_mktd_features(price_df, fc_h1_raw, fc_h24_raw)
        except Exception:
            all_mktd_features[sym] = None

    all_b6_features = {}
    for b6 in BINANCE6_SYMBOLS:
        if b6 in all_mktd_features and all_mktd_features[b6] is not None:
            all_b6_features[b6] = all_mktd_features[b6]
        else:
            try:
                price_df = _read_hourly_prices(b6, DATA_ROOT)
                fc1 = _read_forecast(b6, FC_ROOT, 1) if (FC_ROOT / f"{b6}_h1.parquet").exists() else pd.DataFrame()
                fc24 = _read_forecast(b6, FC_ROOT, 24) if (FC_ROOT / f"{b6}_h24.parquet").exists() else pd.DataFrame()
                all_b6_features[b6] = compute_mktd_features(price_df, fc1, fc24)
            except Exception:
                all_b6_features[b6] = None

    fc_ends = [all_fc_h1[s]["timestamp"].max() for s in symbols if not all_fc_h1[s].empty]
    end_ts = min(fc_ends)
    start_ts = end_ts - timedelta(days=days)

    sym_configs = {s: _symbol_config_for_short_policy(s, short_policy) for s in symbols}

    def get_features_at(ts):
        features = np.zeros((len(BINANCE6_SYMBOLS), 16), dtype=np.float32)
        ts_floor = pd.Timestamp(ts).floor("h")
        for idx, b6 in enumerate(BINANCE6_SYMBOLS):
            mktd = all_b6_features.get(b6)
            if mktd is None:
                continue
            if ts_floor in mktd.index:
                features[idx] = mktd.loc[ts_floor].values[:16].astype(np.float32)
            else:
                before = mktd.index[mktd.index <= ts_floor]
                if len(before) > 0:
                    features[idx] = mktd.iloc[mktd.index.get_loc(before[-1])].values[:16].astype(np.float32)
        return features

    def get_rl_signal(features_all, close):
        obs = np.zeros(obs_size, dtype=np.float32)
        obs[:num_symbols * 16] = features_all.flatten()
        obs[num_symbols * 16] = 1.0
        obs[num_symbols * 16 + 4] = 0.5
        obs_t = torch.from_numpy(obs).unsqueeze(0)
        with torch.no_grad():
            logits, value = rl_policy(obs_t)
            probs = torch.softmax(logits, dim=-1)
            action = logits.argmax(dim=-1).item()
            conf = probs[0, action].item()
            val = value.item()
        if action == 0:
            return TradingSignal("flat", None, None, conf, val, 0, 0)
        idx = action - 1
        is_short = idx >= num_symbols
        if is_short:
            idx -= num_symbols
        sym = BINANCE6_SYMBOLS[idx] if idx < num_symbols else "?"
        d = "short" if is_short else "long"
        return TradingSignal(f"{d}_{sym}", sym, d, conf, val, 1, 0)

    # Build tasks
    tasks = []
    prev_sigs = {s: None for s in symbols}
    prev_outs = {s: "N/A" for s in symbols}

    for sym in symbols:
        sb = all_bars[sym]
        window = sb[(sb["timestamp"] >= start_ts) & (sb["timestamp"] <= end_ts)].copy()
        for i, (_, bar) in enumerate(window.iterrows()):
            ts = bar["timestamp"]
            hist = sb[sb["timestamp"] <= ts].tail(72)
            if len(hist) < 5:
                tasks.append((sym, bar.to_dict(), None, None))
                continue
            fc1 = get_forecast_at(all_fc_h1[sym], ts)
            fc24 = get_forecast_at(all_fc_h24[sym], ts)
            feats = get_features_at(ts)
            rl_sig = get_rl_signal(feats, float(bar["close"]))
            prompt = build_prompt(variant, sym, hist.tail(72).to_dict("records"), fc1, fc24, rl_sig, prev_sigs[sym], prev_outs[sym])
            tasks.append((sym, bar.to_dict(), prompt, rl_sig))
            prev_sigs[sym] = rl_sig
            if i > 0:
                pc = float(window.iloc[i-1]["close"])
                c = float(bar["close"])
                prev_outs[sym] = f"{(c-pc)/pc*100:+.2f}% (${pc:.2f}->${c:.2f})"

    total = len(tasks)
    api_calls = sum(1 for _, _, p, _ in tasks if p)
    print(f"  [{variant}] {total} bars, {api_calls} API calls")

    def do_call(sym, bar, prompt, rl_sig, idx):
        if prompt is None:
            return sym, bar, TradePlan("hold", 0, 0, 0, "no data"), rl_sig, idx
        cached = get_cached(cache_tag, prompt)
        if cached is not None:
            return sym, bar, TradePlan(**cached), rl_sig, idx
        plan = call_llm(prompt, model=model, thinking_level=thinking_level)
        set_cached(cache_tag, prompt, plan.__dict__)
        return sym, bar, plan, rl_sig, idx

    results = [None] * total
    done = 0
    t0 = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {pool.submit(do_call, *tasks[i], i): i for i in range(total)}
        for f in concurrent.futures.as_completed(futures):
            sym, bar, plan, rl_sig, idx = f.result()
            results[idx] = (sym, bar, plan, rl_sig)
            done += 1
            if done % max(1, total // 10) == 0:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                print(f"    [{variant}] {done}/{total} ({rate:.1f}/s)")

    action_rows, bar_rows = [], []
    stats = {"long": 0, "short": 0, "hold": 0, "short_filtered": 0}

    for sym, bar, plan, rl_sig in results:
        ts = bar["timestamp"] if isinstance(bar["timestamp"], pd.Timestamp) else pd.Timestamp(bar["timestamp"])
        last_close = float(bar["close"])
        plan, short_filtered = _normalize_plan_for_short_policy(
            plan,
            last_close=last_close,
            short_policy=short_policy,
        )
        if short_filtered:
            stats["short_filtered"] += 1

        stats[plan.direction] = stats.get(plan.direction, 0) + 1
        action_rows.append({
            "timestamp": ts, "symbol": sym,
            "buy_price": plan.buy_price, "sell_price": plan.sell_price,
            "direction": plan.direction, "confidence": plan.confidence,
        })
        bar_rows.append(bar)

    # Simulate with leverage (max_position_pct = 1/num_syms * leverage)
    max_pos = min(leverage / len(symbols), 1.0)
    config = BacktestConfig(initial_cash=10_000.0, max_hold_hours=6, max_position_pct=max_pos, model=cache_tag)
    bars_df = pd.DataFrame(bar_rows)
    actions_df = pd.DataFrame(action_rows)
    bars_df["timestamp"] = pd.to_datetime(bars_df["timestamp"], utc=True)
    actions_df["timestamp"] = pd.to_datetime(actions_df["timestamp"], utc=True)

    result = simulate(bars_df, actions_df, config, sym_configs)
    m = result["metrics"]
    trades = result["trades"]
    buys = sum(1 for t in trades if t["side"] in ("buy", "short"))
    pnl = sum(t["realized_pnl"] for t in trades)
    fees = sum(t["fee"] for t in trades)

    print(f"\n  [{variant}] RESULTS:")
    print(f"    Return: {m['total_return_pct']:+.4f}%")
    print(f"    Sortino: {m['sortino']:.2f}")
    print(f"    Max DD: {m['max_drawdown_pct']:.2f}%")
    print(f"    Trades: {buys} entries, PnL=${pnl:+.2f}, fees=${fees:.2f}")
    print(f"    Signals: {stats}")
    print(f"    Final equity: ${m['final_equity']:,.2f}")

    return {
        "variant": variant, "model": model, "leverage": leverage,
        "short_policy": short_policy,
        "days": days, "symbols": symbols,
        **m, "entries": buys, "realized_pnl": pnl, "fees": fees,
        "signal_counts": stats,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=["BTCUSD", "ETHUSD", "SOLUSD"])
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--model", default="gemini-3.1-flash-lite-preview")
    parser.add_argument("--thinking-level", default="HIGH")
    parser.add_argument("--parallel", type=int, default=5)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--leverage", type=float, default=5.0)
    parser.add_argument("--variants", nargs="+", default=list(PROMPTS.keys()))
    parser.add_argument("--short-policy", choices=sorted(SHORT_POLICIES), default=SHORT_POLICY_ALLOW)
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"PROMPT EXPERIMENT: {args.variants}")
    print(f"Model: {args.model}, Leverage: {args.leverage}x, Days: {args.days}, Short policy: {args.short_policy}")
    print(f"{'='*70}\n")

    all_results = {}
    for v in args.variants:
        print(f"\n--- Running variant: {v} ---")
        r = run_experiment(
            v, args.symbols, args.days, args.model, args.parallel,
            args.checkpoint, args.leverage, args.thinking_level, args.short_policy,
        )
        all_results[v] = r

    print(f"\n{'='*70}")
    print(f"COMPARISON")
    print(f"{'='*70}")
    print(f"{'Variant':<20} {'Return':>10} {'Sortino':>10} {'MaxDD':>10} {'Trades':>8} {'PnL':>10}")
    print("-" * 70)
    for v, r in sorted(all_results.items(), key=lambda x: -x[1]["total_return_pct"]):
        print(f"{v:<20} {r['total_return_pct']:>+9.2f}% {r['sortino']:>10.2f} {r['max_drawdown_pct']:>9.2f}% {r['entries']:>8} ${r['realized_pnl']:>+9.2f}")
    print(f"{'='*70}")

    best = max(all_results.items(), key=lambda x: x[1]["sortino"])
    print(f"\nBEST (by Sortino): {best[0]} -> {best[1]['sortino']:.2f}")

    out = RESULTS_DIR / "prompt_experiment.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
