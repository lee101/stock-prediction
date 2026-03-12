"""
Prompt experiment v4: RL+Chronos2+LLM integrated prompt.
Tests the full pipeline: RL signal + Chronos2 forecasts + price history -> Gemini optimization prompt.
Compares: pure RL rotator vs RL+Gemini hybrid vs Gemini-only (baseline from v2).
"""
from __future__ import annotations
import argparse, json, sys, time, concurrent.futures, os
from pathlib import Path
from datetime import timedelta
from collections import defaultdict
import numpy as np, pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_hourly_trader.backtest import load_bars, load_forecasts, get_forecast_at, RESULTS_DIR
from llm_hourly_trader.config import SYMBOL_UNIVERSE, BacktestConfig, SymbolConfig
from llm_hourly_trader.gemini_wrapper import TradePlan
from llm_hourly_trader.providers import call_llm
from llm_hourly_trader.cache import get_cached, set_cached
from rl_signal import (
    RLSignalGenerator, PortfolioSnapshot, TradingPolicy,
    SYMBOLS as RL_SYMBOLS, NUM_SYMBOLS, FEATURES_PER_SYM, OBS_SIZE, NUM_ACTIONS,
    ACTION_NAMES, compute_symbol_features, _load_forecast_parquet,
)


BACKTEST_SYMBOLS = ["BTCUSD", "ETHUSD"]  # RL model trades BTC/ETH (DOGE/AAVE excluded from live)

PROMPT_INTEGRATED = """You are optimizing hourly allocation for a crypto spot portfolio to maximize risk-adjusted returns (Sortino ratio).

PORTFOLIO STATE:
- Cash: ${cash:.2f} ({cash_pct:.0f}% of portfolio)
- Total value: ${total_value:.2f}
- Current position: {current_position}

RL MODEL RECOMMENDATION:
- Action: {rl_action} (value={rl_value:.3f})
- Logits: {rl_logits}
- The RL model was trained on 100M+ steps with PPO to optimize portfolio rotation.

{symbol_blocks}

PREVIOUS HOUR PLAN: {prev_plan}

CONSTRAINTS:
- LONG ONLY (spot market, no shorting)
- Max 1 position at a time (portfolio rotator)
- 1-hour decision intervals, max hold ~6 hours
- Transaction cost: 0% for BTC/ETH (FDUSD pairs), 0.1% for altcoins (USDT pairs)
- Objective: maximize Sortino ratio -- minimize downside risk while capturing upside

OPTIMIZATION TASK:
Given all the signals above (RL model, Chronos2 forecasts, price action, momentum), determine:
1. Should we HOLD current position, ROTATE to a different asset, or go FLAT (all cash)?
2. If entering/rotating, set buy_price to maximize fill probability (slightly below market)
3. Set sell_price as take-profit target (0.5-2% above entry based on volatility)
4. Confidence should reflect conviction: 0.7+ for strong setups, 0.4-0.6 for moderate

The RL model is a strong baseline -- only override it if price action or forecasts clearly disagree.
Weight the Chronos2 forecast heavily: if it predicts downside, prefer cash even if RL says long.

Respond with JSON: {{"direction": "long" or "hold", "symbol": "<BTCUSD or ETHUSD>", "buy_price": <entry>, "sell_price": <take profit>, "confidence": <0-1>, "reasoning": "<brief>"}}"""


def _build_symbol_block(sym, bars_df, fc_h1_df, fc_h24_df, ts):
    """Build context block for one symbol at timestamp ts."""
    hist = bars_df[bars_df.index <= ts].tail(24)
    if hist.empty:
        return ""
    price = float(hist.iloc[-1]["close"])
    lines = []
    for _, row in hist.tail(12).iterrows():
        t = str(row.name)[-19:-6] if hasattr(row.name, 'isoformat') else str(row.name)[:13]
        lines.append(f"  {t}: O={row['open']:.2f} H={row['high']:.2f} L={row['low']:.2f} C={row['close']:.2f}")

    # Trend
    closes = hist["close"].values
    ret_24h = (closes[-1] - closes[0]) / closes[0] * 100 if len(closes) > 1 else 0
    highs, lows = hist["high"].values, hist["low"].values
    atr = np.mean(highs[-min(24, len(highs)):] - lows[-min(24, len(lows)):])
    atr_pct = atr / price * 100

    # Forecasts
    fc_text = ""
    tf = pd.Timestamp(ts).floor("h")
    if not fc_h1_df.empty and tf in fc_h1_df.index:
        row_fc = fc_h1_df.loc[tf]
        p50 = float(row_fc.get("predicted_close_p50", 0))
        if p50 > 0:
            delta = (p50 - price) / price * 100
            fc_text += f"\n  Chronos2 1h forecast: ${p50:.2f} ({delta:+.2f}%)"
    if not fc_h24_df.empty and tf in fc_h24_df.index:
        row_fc = fc_h24_df.loc[tf]
        p50 = float(row_fc.get("predicted_close_p50", 0))
        if p50 > 0:
            delta = (p50 - price) / price * 100
            fc_text += f"\n  Chronos2 24h forecast: ${p50:.2f} ({delta:+.2f}%)"

    return f"""--- {sym} @ ${price:.2f} ---
Trend 24h: {ret_24h:+.2f}% | ATR: {atr_pct:.2f}%
Last 12h:
{chr(10).join(lines)}{fc_text}
"""


def simulate_integrated(
    bars_map, fc_h1_map, fc_h24_map,
    rl_gen, symbols, days, model, thinking_level,
    parallel, cache_tag, prev_plan_text="None",
):
    """Run integrated RL+LLM backtest."""
    # Find common timestamp range
    ends = []
    for s in symbols:
        if s in fc_h1_map and not fc_h1_map[s].empty:
            ends.append(fc_h1_map[s].index.max())
    if not ends:
        print("  No forecast data available")
        return None
    end_ts = min(ends)
    start_ts = end_ts - timedelta(days=days)

    # Build timestamp index from bars
    all_ts = set()
    for s in symbols:
        b = bars_map[s]
        w = b[(b.index >= start_ts) & (b.index <= end_ts)]
        all_ts.update(w.index.tolist())
    all_ts = sorted(all_ts)
    print(f"  {len(all_ts)} hourly bars, {len(symbols)} symbols")

    # State
    cash = 10000.0
    initial_cash = cash
    position_sym = None
    position_qty = 0.0
    position_entry = 0.0
    hold_hours = 0
    equity_history = []
    trades = []
    prev_plan = prev_plan_text
    api_calls = 0
    print(f"  {len(all_ts)} stateful decisions")

    # The prompt depends on current holdings and the previous plan, so it must
    # be built/evaluated sequentially to avoid look-ahead bias.
    for ti, ts in enumerate(all_ts):
        sym_blocks = []
        prices = {}
        for s in symbols:
            b = bars_map[s]
            if ts in b.index:
                prices[s] = float(b.loc[ts, "close"])
            block = _build_symbol_block(
                s, bars_map[s],
                fc_h1_map.get(s, pd.DataFrame()),
                fc_h24_map.get(s, pd.DataFrame()),
                ts,
            )
            if block:
                sym_blocks.append(block)

        current_price = prices.get(position_sym, position_entry) if position_sym else 0.0
        position_value = position_qty * current_price if position_sym else 0.0
        total_val = cash + position_value

        if not prices:
            equity_history.append(total_val)
            continue

        portfolio = PortfolioSnapshot(
            cash_usd=cash,
            total_value_usd=total_val,
            position_symbol=position_sym,
            position_value_usd=position_value,
            hold_hours=hold_hours,
        )

        klines_map = {}
        for s in RL_SYMBOLS:
            if s in bars_map:
                hist = bars_map[s][bars_map[s].index <= ts].tail(96)
                if not hist.empty:
                    klines_map[s] = hist

        try:
            rl_sig = rl_gen.get_signal(portfolio=portfolio, klines_map=klines_map)
        except Exception:
            rl_sig = None

        pos_str = "FLAT (all cash)"
        if position_sym and position_qty > 0:
            pnl = (current_price - position_entry) / position_entry * 100
            pos_str = (
                f"{position_sym}: {position_qty:.6f} @ ${position_entry:.2f} "
                f"(now ${current_price:.2f}, {pnl:+.2f}%, held {hold_hours}h)"
            )

        cash_pct = cash / max(total_val, 1) * 100
        prompt = PROMPT_INTEGRATED.format(
            cash=cash,
            cash_pct=cash_pct,
            total_value=total_val,
            current_position=pos_str,
            rl_action=rl_sig.action_name if rl_sig else "UNAVAILABLE",
            rl_value=rl_sig.value if rl_sig else 0,
            rl_logits=", ".join(f"{ACTION_NAMES[i]}={l:.2f}" for i, l in enumerate(rl_sig.logits)) if rl_sig else "N/A",
            symbol_blocks="\n".join(sym_blocks),
            prev_plan=prev_plan,
        )

        cached = get_cached(cache_tag, prompt)
        if cached:
            plan = TradePlan(**cached)
        else:
            plan = call_llm(prompt, model=model, thinking_level=thinking_level)
            set_cached(cache_tag, prompt, plan.__dict__)
            api_calls += 1

        # Check max hold / take-profit
        if position_sym and position_qty > 0:
            cur_price = prices.get(position_sym, position_entry)
            hold_hours += 1

            # Max hold exit
            if hold_hours >= 6:
                proceeds = position_qty * cur_price
                rpnl = (cur_price - position_entry) * position_qty
                cfg = SYMBOL_UNIVERSE.get(position_sym, SymbolConfig(position_sym, "crypto"))
                fee = proceeds * cfg.maker_fee
                cash += proceeds - fee
                trades.append({"timestamp": str(ts), "symbol": position_sym, "side": "close",
                              "price": cur_price, "quantity": position_qty, "realized_pnl": rpnl,
                              "fee": fee, "reason": "max_hold"})
                position_sym = None
                position_qty = 0
                position_entry = 0
                hold_hours = 0

        # Filter shorts
        if plan.direction == "short":
            plan = TradePlan("hold", 0, 0, 0, "short filtered")

        # Extract target symbol from plan
        target_sym = None
        if plan.direction == "long":
            # Try to get symbol from reasoning or default
            reasoning = plan.reasoning.lower() if plan.reasoning else ""
            if hasattr(plan, 'symbol') and plan.symbol:
                target_sym = plan.symbol
            elif "btc" in reasoning:
                target_sym = "BTCUSD"
            elif "eth" in reasoning:
                target_sym = "ETHUSD"
            elif rl_sig and rl_sig.target_symbol:
                target_sym = rl_sig.target_symbol
            else:
                # Default to first symbol with valid price
                for s in symbols:
                    if s in prices:
                        target_sym = s
                        break

        # Execute plan
        if plan.direction == "long" and target_sym and plan.confidence >= 0.4:
            buy_price = plan.buy_price if plan.buy_price > 0 else prices.get(target_sym, 0) * 0.998
            sell_price = plan.sell_price if plan.sell_price > 0 else buy_price * 1.01

            # Check if we need to rotate
            if position_sym and position_sym != target_sym and position_qty > 0:
                # Sell current
                cur_price = prices.get(position_sym, position_entry)
                proceeds = position_qty * cur_price
                rpnl = (cur_price - position_entry) * position_qty
                cfg = SYMBOL_UNIVERSE.get(position_sym, SymbolConfig(position_sym, "crypto"))
                fee = proceeds * cfg.maker_fee
                cash += proceeds - fee
                trades.append({"timestamp": str(ts), "symbol": position_sym, "side": "sell",
                              "price": cur_price, "quantity": position_qty, "realized_pnl": rpnl,
                              "fee": fee, "reason": "rotate"})
                position_sym = None
                position_qty = 0
                hold_hours = 0

            # Buy target (only if not already holding it)
            if position_sym != target_sym:
                target_price = prices.get(target_sym, 0)
                # Check fill: would limit order fill?
                bar = bars_map[target_sym]
                if ts in bar.index:
                    low = float(bar.loc[ts, "low"])
                    if low <= buy_price:
                        cfg = SYMBOL_UNIVERSE.get(target_sym, SymbolConfig(target_sym, "crypto"))
                        alloc = cash * 0.90 * plan.confidence
                        fee_rate = cfg.maker_fee
                        qty = alloc / (buy_price * (1 + fee_rate))
                        cost = qty * buy_price * (1 + fee_rate)
                        if cost <= cash and cost >= 10:
                            cash -= cost
                            position_sym = target_sym
                            position_qty = qty
                            position_entry = buy_price
                            hold_hours = 0
                            trades.append({"timestamp": str(ts), "symbol": target_sym, "side": "buy",
                                          "price": buy_price, "quantity": qty, "realized_pnl": 0,
                                          "fee": qty * buy_price * fee_rate, "reason": "entry"})

            # Check take-profit for existing position
            if position_sym and position_qty > 0 and sell_price > 0:
                bar = bars_map.get(position_sym)
                if bar is not None and ts in bar.index:
                    high = float(bar.loc[ts, "high"])
                    if high >= sell_price:
                        cfg = SYMBOL_UNIVERSE.get(position_sym, SymbolConfig(position_sym, "crypto"))
                        proceeds = position_qty * sell_price
                        rpnl = (sell_price - position_entry) * position_qty
                        fee = proceeds * cfg.maker_fee
                        cash += proceeds - fee
                        trades.append({"timestamp": str(ts), "symbol": position_sym, "side": "sell",
                                      "price": sell_price, "quantity": position_qty, "realized_pnl": rpnl,
                                      "fee": fee, "reason": "take_profit"})
                        position_sym = None
                        position_qty = 0
                        hold_hours = 0

        plan_snapshot = {
            "direction": plan.direction,
            "buy_price": round(plan.buy_price, 6),
            "sell_price": round(plan.sell_price, 6),
            "confidence": round(plan.confidence, 4),
            "reasoning": (plan.reasoning or "")[:120],
        }
        if target_sym:
            plan_snapshot["symbol"] = target_sym
        prev_plan = json.dumps(plan_snapshot, sort_keys=True)

        # Equity snapshot
        pos_val = position_qty * prices.get(position_sym, 0) if position_sym else 0
        equity_history.append(cash + pos_val)

        if (ti + 1) % max(1, len(all_ts) // 10) == 0:
            print(f"    {ti + 1}/{len(all_ts)} decisions | API calls={api_calls}")

    # Metrics
    equity = np.array(equity_history, dtype=float)
    if len(equity) < 2:
        return None
    total_return = (equity[-1] - equity[0]) / equity[0]
    returns = np.diff(equity) / np.clip(equity[:-1], 1e-8, None)
    downside = returns[returns < 0]
    ds_std = downside.std() if len(downside) else 0
    sortino = returns.mean() / ds_std * np.sqrt(8760) if ds_std > 0 else 0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = abs(dd.min()) * 100

    # Per-pair PnL
    pair_stats = defaultdict(lambda: {"pnl": 0.0, "fees": 0.0, "entries": 0, "wins": 0, "losses": 0})
    for t in trades:
        sym = t["symbol"]
        pair_stats[sym]["fees"] += t.get("fee", 0)
        if t["side"] in ("sell", "close"):
            rpnl = t.get("realized_pnl", 0)
            pair_stats[sym]["pnl"] += rpnl
            if rpnl > 0: pair_stats[sym]["wins"] += 1
            elif rpnl < 0: pair_stats[sym]["losses"] += 1
        elif t["side"] == "buy":
            pair_stats[sym]["entries"] += 1

    entries = sum(1 for t in trades if t["side"] == "buy")
    total_pnl = sum(t.get("realized_pnl", 0) for t in trades)
    total_fees = sum(t.get("fee", 0) for t in trades)

    print(f"\n  RESULTS:")
    print(f"    Return: {total_return*100:+.4f}%")
    print(f"    Sortino: {sortino:.2f}")
    print(f"    Max DD: {max_dd:.2f}%")
    print(f"    Trades: {entries} entries, PnL=${total_pnl:+.2f}, fees=${total_fees:.2f}")
    print(f"    Final equity: ${equity[-1]:,.2f}")
    print(f"    --- Per-Pair PnL ---")
    for sym in sorted(pair_stats.keys()):
        ps = pair_stats[sym]
        wr = ps["wins"] / max(1, ps["wins"] + ps["losses"]) * 100
        print(f"    {sym:>8}: PnL=${ps['pnl']:+8.2f}  fees=${ps['fees']:6.2f}  entries={ps['entries']:3d}  W/L={ps['wins']}/{ps['losses']} ({wr:.0f}%)")

    return {
        "total_return_pct": total_return * 100,
        "sortino": sortino,
        "max_drawdown_pct": max_dd,
        "final_equity": equity[-1],
        "entries": entries,
        "realized_pnl": total_pnl,
        "fees": total_fees,
        "per_pair": {k: {kk: round(vv, 2) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in pair_stats.items()},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", default=BACKTEST_SYMBOLS)
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--model", default="gemini-3.1-flash-lite-preview")
    p.add_argument("--thinking-level", default="HIGH")
    p.add_argument("--parallel", type=int, default=5)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--forecast-cache", default="binanceneural/forecast_cache")
    args = p.parse_args()

    print(f"\n{'='*70}")
    print(f"V4: RL+Chronos2+Gemini Integrated Test")
    print(f"Model: {args.model}, Days: {args.days}, Symbols: {args.symbols}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"{'='*70}\n")

    # Load RL generator
    rl_gen = RLSignalGenerator(
        checkpoint_path=args.checkpoint,
        forecast_cache_root=args.forecast_cache,
    )

    # Load bars + forecasts
    bars_map, fc_h1_map, fc_h24_map = {}, {}, {}
    for s in args.symbols:
        bars = load_bars(s)
        bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
        bars = bars.set_index("timestamp").sort_index()
        bars = bars[~bars.index.duplicated(keep="last")]
        bars_map[s] = bars

        fc1 = load_forecasts(s, "h1")
        if not fc1.empty:
            fc1["timestamp"] = pd.to_datetime(fc1["timestamp"], utc=True)
            fc1 = fc1.set_index("timestamp").sort_index()
            fc1 = fc1[~fc1.index.duplicated(keep="last")]
        fc_h1_map[s] = fc1

        fc24 = load_forecasts(s, "h24")
        if not fc24.empty:
            fc24["timestamp"] = pd.to_datetime(fc24["timestamp"], utc=True)
            fc24 = fc24.set_index("timestamp").sort_index()
            fc24 = fc24[~fc24.index.duplicated(keep="last")]
        fc_h24_map[s] = fc24

    cache_tag = f"prompt-v4-integrated-{args.model}"
    print("--- RL+Chronos2+Gemini Integrated ---")
    result = simulate_integrated(
        bars_map, fc_h1_map, fc_h24_map,
        rl_gen, args.symbols, args.days, args.model, args.thinking_level,
        args.parallel, cache_tag,
    )

    if result:
        out = RESULTS_DIR / "prompt_experiment_v4.json"
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
