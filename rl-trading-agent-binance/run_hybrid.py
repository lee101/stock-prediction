"""
RL + LLM Hybrid Trading Agent for Binance.

Symbols: BTC, ETH, SUI, SOL, AAVE, DOGE (USDT pairs on Binance)

Pipeline:
1. Load trained PPO model (binance6, h1024)
2. For each hour in the backtest window:
   a. Compute features from OHLCV + Chronos2 forecasts
   b. Get RL model's trading signal (action, confidence, value estimate)
   c. Build a prompt showing market data, forecasts, RL plan
   d. Ask LLM to refine the plan
   e. Use the LLM's refined plan for trading
3. Run market simulation on the refined plans

Usage:
  python -m rl-trading-agent-binance.run_hybrid --days 7 --model deepseek-chat --parallel 5
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from llm_hourly_trader.backtest import (
    load_bars, load_forecasts, get_forecast_at, simulate, _compute_metrics,
    RESULTS_DIR,
)
from llm_hourly_trader.config import SYMBOL_UNIVERSE, BacktestConfig, SymbolConfig
from llm_hourly_trader.gemini_wrapper import TradePlan
from llm_hourly_trader.providers import call_llm
from llm_hourly_trader.cache import get_cached, set_cached
from pufferlib_market.export_data_hourly_forecast import (
    compute_features as compute_mktd_features,
    _read_hourly_prices,
    _read_forecast,
)

import torch
import torch.nn as nn


class MLPPolicy(nn.Module):
    """MLP policy matching training checkpoint architecture."""
    def __init__(self, obs_size, num_actions, hidden=1024):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_size, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.actor = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, num_actions))
        self.critic = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))

    def forward(self, x):
        h = self.encoder(x)
        return self.actor(h), self.critic(h).squeeze(-1)


# Symbols the RL model was trained on (binance6)
BINANCE6_SYMBOLS = [
    "BTCUSD", "ETHUSD", "SUIUSD", "SOLUSD", "AAVEUSD", "DOGEUSD",
]

# Default backtest symbols (all 6)
BACKTEST_SYMBOLS = BINANCE6_SYMBOLS

# Checkpoint path - will be set after training completes
CHECKPOINT_DIR = REPO / "rl-trainingbinance" / "checkpoints"


def _find_best_checkpoint() -> Path:
    """Find the best checkpoint from the most recent training run."""
    candidates = sorted(CHECKPOINT_DIR.glob("*/best.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {CHECKPOINT_DIR}")
    return candidates[-1]


def _compute_trend_context(history_rows: list[dict]) -> dict:
    """Compute trend indicators from price history for prompt context."""
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
        ups = sum(1 for i in range(-12, 0) if closes[i] > closes[i-1])
        ctx["up_hours_12"] = ups

    return ctx


def build_hybrid_prompt(
    symbol: str,
    history_rows: list[dict],
    fc_1h: dict | None,
    fc_24h: dict | None,
    rl_signal,
    prev_rl_signal,
    prev_outcome: str,
) -> str:
    """Build prompt combining market data + RL plan for LLM refinement."""
    recent = history_rows[-12:]
    price_lines = []
    for row in recent:
        ts = str(row.get("timestamp", ""))[-19:-6] if "timestamp" in row else "?"
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        price_lines.append(f"  {ts}: O={o:.2f} H={h:.2f} L={l:.2f} C={c:.2f}")

    current_price = float(history_rows[-1]["close"])

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
    trend_line = " | ".join(trend_parts)

    sr_line = ""
    if "low_24h" in trend:
        sr_line = f"  24h range: ${trend['low_24h']:.2f} - ${trend['high_24h']:.2f} ({trend['range_pct']:.2f}% wide)"

    fc_text = ""
    if fc_1h:
        delta_1h = (fc_1h['predicted_close_p50'] - current_price) / current_price * 100
        fc_text += f"\n  1h forecast: close={fc_1h['predicted_close_p50']:.2f} ({delta_1h:+.2f}%), "
        fc_text += f"high={fc_1h['predicted_high_p50']:.2f}, low={fc_1h['predicted_low_p50']:.2f}"
    if fc_24h:
        delta_24h = (fc_24h['predicted_close_p50'] - current_price) / current_price * 100
        fc_text += f"\n  24h forecast: close={fc_24h['predicted_close_p50']:.2f} ({delta_24h:+.2f}%), "
        fc_text += f"high={fc_24h['predicted_high_p50']:.2f}, low={fc_24h['predicted_low_p50']:.2f}"

    rl_text = f"Direction: {rl_signal.action}, Confidence: {rl_signal.confidence:.1%}"
    rl_text += f", Value estimate: {rl_signal.value_estimate:.4f}"

    prev_text = "None (first hour)"
    if prev_rl_signal is not None:
        prev_text = f"RL said: {prev_rl_signal.action} (conf={prev_rl_signal.confidence:.1%})"
        prev_text += f"\n  Outcome: {prev_outcome}"

    return f"""You are an expert crypto trader reviewing an AI trading model's recommendation.

SYMBOL: {symbol} (long only, spot market, 0.1% maker fee on Binance)
CURRENT PRICE: ${current_price:.2f}
{sr_line}

TREND CONTEXT: {trend_line}

LAST 12 HOURS:
{chr(10).join(price_lines)}

CHRONOS2 FORECASTS:{fc_text}

RL MODEL (trained with PPO on Binance data):
  {rl_text}

PREVIOUS HOUR:
  {prev_text}

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


from dataclasses import dataclass


@dataclass
class TradingSignal:
    action: str
    symbol: str | None
    direction: str | None
    confidence: float
    value_estimate: float
    allocation_pct: float
    unrealized_pnl: float


def run_hybrid_backtest(
    symbols: list[str],
    days: int,
    model: str = "deepseek-chat",
    parallel: int = 1,
    rate_limit: float = 2.0,
    checkpoint_path: str | None = None,
) -> dict:
    """Run the RL+LLM hybrid backtest."""
    cache_model = f"binance-hybrid-v1-{model}"

    # Find checkpoint
    if checkpoint_path:
        ckpt_path = Path(checkpoint_path)
    else:
        ckpt_path = _find_best_checkpoint()

    print(f"\n{'='*70}")
    print(f"RL + LLM Hybrid Trading Agent (Binance)")
    print(f"RL checkpoint: {ckpt_path}")
    print(f"LLM: {model}")
    print(f"Symbols: {symbols}")
    print(f"Days: {days} | Parallel: {parallel}")
    print(f"{'='*70}\n")

    # Load RL model
    print("Loading RL model...")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = ckpt["model"]
    num_symbols = len(BINANCE6_SYMBOLS)
    obs_size = num_symbols * 16 + 5 + num_symbols
    hidden = state_dict["encoder.0.weight"].shape[0]
    num_actions = state_dict["actor.2.weight"].shape[0]
    print(f"  RL model: obs={obs_size}, actions={num_actions}, hidden={hidden}")

    rl_policy = MLPPolicy(obs_size, num_actions, hidden)
    rl_policy.load_state_dict(state_dict)
    rl_policy.eval()

    def get_rl_signal(features_all, prices_dict):
        """Get RL signal from features."""
        obs = np.zeros(obs_size, dtype=np.float32)
        obs[:num_symbols * 16] = features_all.flatten()
        obs[num_symbols * 16] = 1.0  # cash/10000
        obs[num_symbols * 16 + 4] = 0.5  # episode progress
        obs_t = torch.from_numpy(obs).unsqueeze(0)
        with torch.no_grad():
            logits, value = rl_policy(obs_t)
            probs = torch.softmax(logits, dim=-1)
            action = logits.argmax(dim=-1).item()
            confidence = probs[0, action].item()
            value_est = value.item()
        if action == 0:
            return TradingSignal("flat", None, None, confidence, value_est, 0.0, 0.0)
        action_idx = action - 1
        is_short = action_idx >= num_symbols
        if is_short:
            action_idx -= num_symbols
        sym = BINANCE6_SYMBOLS[action_idx] if action_idx < num_symbols else "?"
        direction = "short" if is_short else "long"
        return TradingSignal(f"{direction}_{sym}", sym, direction, confidence, value_est, 1.0, 0.0)

    # Load market data + MKTD features
    DATA_ROOT = REPO / "trainingdatahourly"
    FC_ROOT = REPO / "binanceneural" / "forecast_cache"
    all_bars = {}
    all_fc_h1 = {}
    all_fc_h24 = {}
    all_mktd_features = {}

    for sym in symbols:
        all_bars[sym] = load_bars(sym)
        all_fc_h1[sym] = load_forecasts(sym, "h1")
        all_fc_h24[sym] = load_forecasts(sym, "h24")
        fc_end = all_fc_h1[sym]["timestamp"].max() if not all_fc_h1[sym].empty else "none"
        print(f"  {sym}: {len(all_bars[sym])} bars, fc to {fc_end}")

        try:
            price_df = _read_hourly_prices(sym, DATA_ROOT)
            try:
                fc_h1_raw = _read_forecast(sym, FC_ROOT, 1)
            except FileNotFoundError:
                fc_h1_raw = pd.DataFrame()
            try:
                fc_h24_raw = _read_forecast(sym, FC_ROOT, 24)
            except FileNotFoundError:
                fc_h24_raw = pd.DataFrame()
            feat_df = compute_mktd_features(price_df, fc_h1_raw, fc_h24_raw)
            all_mktd_features[sym] = feat_df
            print(f"    MKTD features: {len(feat_df)} rows, last={feat_df.index[-1]}")
        except Exception as e:
            print(f"    MKTD features failed: {e}")
            all_mktd_features[sym] = None

    fc_ends = [all_fc_h1[sym]["timestamp"].max() for sym in symbols if not all_fc_h1[sym].empty]
    end_ts = min(fc_ends)
    start_ts = end_ts - timedelta(days=days)
    print(f"\n  Window: {start_ts} -> {end_ts}")

    sym_configs = {sym: SYMBOL_UNIVERSE.get(sym, SymbolConfig(sym, "crypto")) for sym in symbols}

    # Load MKTD features for ALL binance6 symbols
    all_b6_features = {}
    for b6_sym in BINANCE6_SYMBOLS:
        if b6_sym in all_mktd_features and all_mktd_features[b6_sym] is not None:
            all_b6_features[b6_sym] = all_mktd_features[b6_sym]
        else:
            try:
                price_df = _read_hourly_prices(b6_sym, DATA_ROOT)
                try:
                    fc_h1_raw = _read_forecast(b6_sym, FC_ROOT, 1)
                except FileNotFoundError:
                    fc_h1_raw = pd.DataFrame()
                try:
                    fc_h24_raw = _read_forecast(b6_sym, FC_ROOT, 24)
                except FileNotFoundError:
                    fc_h24_raw = pd.DataFrame()
                feat_df = compute_mktd_features(price_df, fc_h1_raw, fc_h24_raw)
                all_b6_features[b6_sym] = feat_df
            except Exception:
                all_b6_features[b6_sym] = None
    print(f"  Loaded MKTD features for {sum(1 for v in all_b6_features.values() if v is not None)}/{len(BINANCE6_SYMBOLS)} binance6 symbols")

    def get_all_features_at(ts):
        """Get feature vector for ALL 6 symbols at timestamp ts."""
        features = np.zeros((len(BINANCE6_SYMBOLS), 16), dtype=np.float32)
        ts_floor = ts.floor("h") if hasattr(ts, "floor") else pd.Timestamp(ts).floor("h")
        for idx, b6_sym in enumerate(BINANCE6_SYMBOLS):
            mktd = all_b6_features.get(b6_sym)
            if mktd is None:
                continue
            if ts_floor in mktd.index:
                features[idx] = mktd.loc[ts_floor].values[:16].astype(np.float32)
            else:
                before = mktd.index[mktd.index <= ts_floor]
                if len(before) > 0:
                    features[idx] = mktd.iloc[mktd.index.get_loc(before[-1])].values[:16].astype(np.float32)
        return features

    # Build all tasks
    print("\n  Computing RL signals and building prompts...")
    tasks = []
    prev_signals = {sym: None for sym in symbols}
    prev_outcomes = {sym: "N/A" for sym in symbols}

    for sym in symbols:
        sym_bars = all_bars[sym]
        window = sym_bars[(sym_bars["timestamp"] >= start_ts) & (sym_bars["timestamp"] <= end_ts)].copy()
        print(f"  {sym}: {len(window)} bars")

        for i, (_, bar) in enumerate(window.iterrows()):
            ts = bar["timestamp"]
            hist_slice = sym_bars[sym_bars["timestamp"] <= ts].tail(72)

            if len(hist_slice) < 5:
                tasks.append((sym, bar.to_dict(), None, None))
                continue

            fc_1h = get_forecast_at(all_fc_h1[sym], ts)
            fc_24h = get_forecast_at(all_fc_h24[sym], ts)
            close = float(bar["close"])

            features_all = get_all_features_at(ts)
            rl_signal = get_rl_signal(features_all, {sym: close})

            prev_signal = prev_signals[sym]
            prev_outcome = prev_outcomes[sym]

            history = hist_slice.tail(72).to_dict("records")
            prompt = build_hybrid_prompt(
                symbol=sym,
                history_rows=history,
                fc_1h=fc_1h,
                fc_24h=fc_24h,
                rl_signal=rl_signal,
                prev_rl_signal=prev_signal,
                prev_outcome=prev_outcome,
            )

            tasks.append((sym, bar.to_dict(), prompt, rl_signal))
            prev_signals[sym] = rl_signal
            if i > 0:
                prev_close = float(window.iloc[i-1]["close"])
                delta_pct = (close - prev_close) / prev_close * 100
                prev_outcomes[sym] = f"Price moved {delta_pct:+.2f}% (${prev_close:.2f} -> ${close:.2f})"

    total_tasks = len(tasks)
    total_api = sum(1 for _, _, p, _ in tasks if p is not None)
    print(f"\n  Total bars: {total_tasks}, API calls needed: {total_api}, parallel: {parallel}")

    # Dispatch LLM calls
    def _do_call(sym, bar, prompt, rl_signal, idx):
        if prompt is None:
            return sym, bar, TradePlan("hold", 0, 0, 0, "insufficient data"), rl_signal, idx
        cached = get_cached(cache_model, prompt)
        if cached is not None:
            return sym, bar, TradePlan(**cached), rl_signal, idx
        plan = call_llm(prompt, model=model)
        set_cached(cache_model, prompt, plan.__dict__)
        return sym, bar, plan, rl_signal, idx

    results = [None] * total_tasks
    t0 = time.time()
    done = 0

    if parallel > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = {pool.submit(_do_call, *tasks[i], i): i for i in range(total_tasks)}
            for f in concurrent.futures.as_completed(futures):
                sym, bar, plan, rl_sig, idx = f.result()
                results[idx] = (sym, bar, plan, rl_sig)
                done += 1
                if done % max(1, total_tasks // 20) == 0:
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (total_tasks - done) / rate if rate > 0 else 0
                    print(f"  Progress: {done}/{total_tasks} ({rate:.1f}/s, ETA {eta:.0f}s)")
    else:
        for i in range(total_tasks):
            sym, bar, plan, rl_sig, idx = _do_call(*tasks[i], i)
            results[i] = (sym, bar, plan, rl_sig)
            done += 1
            if rate_limit > 0 and tasks[i][2] is not None:
                time.sleep(rate_limit)
            if done % max(1, total_tasks // 20) == 0:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total_tasks - done) / rate if rate > 0 else 0
                print(f"  Progress: {done}/{total_tasks} ({rate:.1f}/s, ETA {eta:.0f}s)")

    # Build action/bar DataFrames
    action_rows = []
    bar_rows = []
    rl_actions = {"long": 0, "short": 0, "flat": 0}
    llm_overrides = 0

    for sym, bar, plan, rl_sig in results:
        ts = bar["timestamp"] if isinstance(bar["timestamp"], pd.Timestamp) else pd.Timestamp(bar["timestamp"])

        if plan.direction not in ["long", "hold"]:
            plan = TradePlan("hold", 0, 0, 0, "direction not allowed")

        if rl_sig is not None:
            rl_dir = rl_sig.direction or "flat"
            if rl_dir in rl_actions:
                rl_actions[rl_dir] += 1
            else:
                rl_actions["flat"] += 1
            if rl_sig.direction == "long" and plan.direction == "hold":
                llm_overrides += 1
            elif rl_sig.direction != "long" and plan.direction == "long":
                llm_overrides += 1

        action_rows.append({
            "timestamp": ts, "symbol": sym,
            "buy_price": plan.buy_price, "sell_price": plan.sell_price,
            "direction": plan.direction, "confidence": plan.confidence,
        })
        bar_rows.append(bar)

    config = BacktestConfig(
        initial_cash=10_000.0,
        max_hold_hours=6,
        max_position_pct=0.25,
        model=cache_model,
    )

    bars_df = pd.DataFrame(bar_rows)
    actions_df = pd.DataFrame(action_rows)
    bars_df["timestamp"] = pd.to_datetime(bars_df["timestamp"], utc=True)
    actions_df["timestamp"] = pd.to_datetime(actions_df["timestamp"], utc=True)

    print(f"\n  Running simulation ({len(bars_df)} bars)...")
    result = simulate(bars_df, actions_df, config, sym_configs)

    metrics = result["metrics"]
    all_trades = result["trades"]
    buys = sum(1 for t in all_trades if t["side"] in ("buy", "short"))
    exits = sum(1 for t in all_trades if t["side"] in ("sell", "cover", "close"))
    realized_pnl = sum(t["realized_pnl"] for t in all_trades)
    total_fees = sum(t["fee"] for t in all_trades)

    per_sym = {}
    for t in all_trades:
        s = t["symbol"]
        if s not in per_sym:
            per_sym[s] = {"entries": 0, "exits": 0, "realized_pnl": 0, "fees": 0}
        if t["side"] in ("buy", "short"):
            per_sym[s]["entries"] += 1
        else:
            per_sym[s]["exits"] += 1
        per_sym[s]["realized_pnl"] += t["realized_pnl"]
        per_sym[s]["fees"] += t["fee"]

    # Also run RL-only simulation for comparison
    rl_only_long = sum(1 for _, _, _, rl_sig in results if rl_sig and rl_sig.direction == "long")
    total_long = sum(1 for _, _, p, _ in results if p.direction == "long")
    entry_rate = total_long / len(results) * 100 if results else 0

    print(f"\n{'='*70}")
    print(f"RESULTS: RL+LLM Hybrid Binance ({model})")
    print(f"{'='*70}")
    print(f"  Window: {start_ts} -> {end_ts} ({days}d)")
    print(f"  RL signals: long={rl_actions['long']}, flat={rl_actions['flat']}, short={rl_actions['short']}")
    print(f"  LLM overrides: {llm_overrides}")
    print(f"  LLM entry rate: {total_long}/{len(results)} ({entry_rate:.1f}%)")
    print(f"  Total return: {metrics['total_return_pct']:+.4f}%")
    print(f"  Max drawdown: {metrics['max_drawdown_pct']:.4f}%")
    print(f"  Sortino: {metrics['sortino']:.4f}")
    print(f"  Entries: {buys}, Exits: {exits}")
    print(f"  Realized PnL: ${realized_pnl:+.2f}")
    print(f"  Total fees: ${total_fees:.2f}")
    print(f"  Final equity: ${metrics['final_equity']:,.2f}")
    print()
    for s, stats in sorted(per_sym.items()):
        print(f"  {s:10s}: {stats['entries']} entries, {stats['exits']} exits, "
              f"PnL=${stats['realized_pnl']:+.2f}, fees=${stats['fees']:.2f}")
    print(f"{'='*70}\n")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_data = {
        "experiment": "rl_llm_hybrid_binance",
        "rl_checkpoint": str(ckpt_path.name),
        "llm_model": model,
        "symbols": symbols,
        "days": days,
        "rl_signals": rl_actions,
        "llm_overrides": llm_overrides,
        "llm_entry_rate": f"{entry_rate:.1f}%",
        **metrics,
        "entries": buys,
        "exits": exits,
        "realized_pnl": realized_pnl,
        "total_fees": total_fees,
        "per_symbol": per_sym,
    }
    tag = f"binance_hybrid_{model.replace('-', '_')}_{days}d_{'_'.join(symbols)}"
    out = RESULTS_DIR / f"{tag}.json"
    with open(out, "w") as f:
        json.dump(result_data, f, indent=2, default=str)
    print(f"  Saved: {out}")

    return result_data


def run_rl_only_backtest(
    symbols: list[str],
    days: int,
    checkpoint_path: str | None = None,
) -> dict:
    """Run RL-only backtest (no LLM) for comparison."""
    if checkpoint_path:
        ckpt_path = Path(checkpoint_path)
    else:
        ckpt_path = _find_best_checkpoint()

    print(f"\n{'='*70}")
    print(f"RL-Only Backtest (Binance) - No LLM")
    print(f"RL checkpoint: {ckpt_path}")
    print(f"Symbols: {symbols}")
    print(f"Days: {days}")
    print(f"{'='*70}\n")

    # Load RL model
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
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

    # Load data
    all_bars = {}
    all_fc_h1 = {}
    all_fc_h24 = {}

    for sym in symbols:
        all_bars[sym] = load_bars(sym)
        all_fc_h1[sym] = load_forecasts(sym, "h1")
        all_fc_h24[sym] = load_forecasts(sym, "h24")

    # Load MKTD features for all binance6 symbols
    all_b6_features = {}
    for b6_sym in BINANCE6_SYMBOLS:
        try:
            price_df = _read_hourly_prices(b6_sym, DATA_ROOT)
            try:
                fc_h1_raw = _read_forecast(b6_sym, FC_ROOT, 1)
            except FileNotFoundError:
                fc_h1_raw = pd.DataFrame()
            try:
                fc_h24_raw = _read_forecast(b6_sym, FC_ROOT, 24)
            except FileNotFoundError:
                fc_h24_raw = pd.DataFrame()
            feat_df = compute_mktd_features(price_df, fc_h1_raw, fc_h24_raw)
            all_b6_features[b6_sym] = feat_df
        except Exception:
            all_b6_features[b6_sym] = None

    fc_ends = [all_fc_h1[sym]["timestamp"].max() for sym in symbols if not all_fc_h1[sym].empty]
    end_ts = min(fc_ends)
    start_ts = end_ts - timedelta(days=days)

    sym_configs = {sym: SYMBOL_UNIVERSE.get(sym, SymbolConfig(sym, "crypto")) for sym in symbols}

    def get_all_features_at(ts):
        features = np.zeros((len(BINANCE6_SYMBOLS), 16), dtype=np.float32)
        ts_floor = ts.floor("h") if hasattr(ts, "floor") else pd.Timestamp(ts).floor("h")
        for idx, b6_sym in enumerate(BINANCE6_SYMBOLS):
            mktd = all_b6_features.get(b6_sym)
            if mktd is None:
                continue
            if ts_floor in mktd.index:
                features[idx] = mktd.loc[ts_floor].values[:16].astype(np.float32)
            else:
                before = mktd.index[mktd.index <= ts_floor]
                if len(before) > 0:
                    features[idx] = mktd.iloc[mktd.index.get_loc(before[-1])].values[:16].astype(np.float32)
        return features

    # Run RL-only: use RL signal directly as trading action
    action_rows = []
    bar_rows = []
    rl_actions = {"long": 0, "short": 0, "flat": 0}

    for sym in symbols:
        sym_bars = all_bars[sym]
        window = sym_bars[(sym_bars["timestamp"] >= start_ts) & (sym_bars["timestamp"] <= end_ts)].copy()
        print(f"  {sym}: {len(window)} bars")

        for _, bar in window.iterrows():
            ts = bar["timestamp"]
            close = float(bar["close"])
            features_all = get_all_features_at(ts)

            obs = np.zeros(obs_size, dtype=np.float32)
            obs[:num_symbols * 16] = features_all.flatten()
            obs[num_symbols * 16] = 1.0
            obs[num_symbols * 16 + 4] = 0.5
            obs_t = torch.from_numpy(obs).unsqueeze(0)

            with torch.no_grad():
                logits, value = rl_policy(obs_t)
                action = logits.argmax(dim=-1).item()
                probs = torch.softmax(logits, dim=-1)
                confidence = probs[0, action].item()

            if action == 0:
                direction = "hold"
                buy_price = 0.0
                sell_price = 0.0
                rl_actions["flat"] += 1
            else:
                action_idx = action - 1
                is_short = action_idx >= num_symbols
                if is_short:
                    # Can't short on spot - treat as hold
                    direction = "hold"
                    buy_price = 0.0
                    sell_price = 0.0
                    rl_actions["short"] += 1
                else:
                    target_sym = BINANCE6_SYMBOLS[action_idx]
                    if target_sym == sym:
                        direction = "long"
                        buy_price = close * 0.999  # 0.1% below
                        sell_price = close * 1.01   # 1% above
                        rl_actions["long"] += 1
                    else:
                        direction = "hold"
                        buy_price = 0.0
                        sell_price = 0.0
                        rl_actions["long"] += 1

            action_rows.append({
                "timestamp": ts, "symbol": sym,
                "buy_price": buy_price, "sell_price": sell_price,
                "direction": direction, "confidence": confidence,
            })
            bar_rows.append(bar.to_dict())

    config = BacktestConfig(initial_cash=10_000.0, max_hold_hours=6, max_position_pct=0.25, model="rl-only")
    bars_df = pd.DataFrame(bar_rows)
    actions_df = pd.DataFrame(action_rows)
    bars_df["timestamp"] = pd.to_datetime(bars_df["timestamp"], utc=True)
    actions_df["timestamp"] = pd.to_datetime(actions_df["timestamp"], utc=True)

    result = simulate(bars_df, actions_df, config, sym_configs)
    metrics = result["metrics"]
    all_trades = result["trades"]
    buys = sum(1 for t in all_trades if t["side"] in ("buy", "short"))
    exits = sum(1 for t in all_trades if t["side"] in ("sell", "cover", "close"))
    realized_pnl = sum(t["realized_pnl"] for t in all_trades)

    total_long = sum(1 for r in action_rows if r["direction"] == "long")
    entry_rate = total_long / len(action_rows) * 100 if action_rows else 0

    print(f"\n{'='*70}")
    print(f"RESULTS: RL-Only Binance")
    print(f"{'='*70}")
    print(f"  Window: {start_ts} -> {end_ts} ({days}d)")
    print(f"  RL signals: long={rl_actions['long']}, flat={rl_actions['flat']}, short={rl_actions['short']}")
    print(f"  RL entry rate: {total_long}/{len(action_rows)} ({entry_rate:.1f}%)")
    print(f"  Total return: {metrics['total_return_pct']:+.4f}%")
    print(f"  Max drawdown: {metrics['max_drawdown_pct']:.4f}%")
    print(f"  Sortino: {metrics['sortino']:.4f}")
    print(f"  Entries: {buys}, Exits: {exits}")
    print(f"  Realized PnL: ${realized_pnl:+.2f}")
    print(f"  Final equity: ${metrics['final_equity']:,.2f}")
    print(f"{'='*70}\n")

    return {
        "experiment": "rl_only_binance",
        **metrics,
        "rl_signals": rl_actions,
        "entries": buys,
        "realized_pnl": realized_pnl,
    }


def main():
    parser = argparse.ArgumentParser(description="RL + LLM Hybrid Trading Agent (Binance)")
    parser.add_argument("--symbols", nargs="+", default=BACKTEST_SYMBOLS)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--model", type=str, default="deepseek-chat")
    parser.add_argument("--parallel", type=int, default=5)
    parser.add_argument("--rate-limit", type=float, default=0.0)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--rl-only", action="store_true", help="Run RL-only backtest for comparison")
    parser.add_argument("--compare", action="store_true", help="Run both RL-only and hybrid, compare")
    args = parser.parse_args()

    if args.rl_only:
        run_rl_only_backtest(args.symbols, args.days, args.checkpoint)
    elif args.compare:
        print("\n" + "="*70)
        print("COMPARISON: RL-Only vs RL+LLM Hybrid")
        print("="*70)
        rl_result = run_rl_only_backtest(args.symbols, args.days, args.checkpoint)
        hybrid_result = run_hybrid_backtest(
            args.symbols, args.days, args.model, args.parallel, args.rate_limit, args.checkpoint
        )
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)
        print(f"  RL-Only:  {rl_result['total_return_pct']:+.4f}% return, Sortino={rl_result['sortino']:.2f}")
        print(f"  Hybrid:   {hybrid_result['total_return_pct']:+.4f}% return, Sortino={hybrid_result['sortino']:.2f}")
        diff = hybrid_result['total_return_pct'] - rl_result['total_return_pct']
        print(f"  Delta:    {diff:+.4f}% ({'hybrid wins' if diff > 0 else 'rl wins'})")
        print("="*70)
    else:
        run_hybrid_backtest(
            args.symbols, args.days, args.model, args.parallel, args.rate_limit, args.checkpoint
        )


if __name__ == "__main__":
    main()
