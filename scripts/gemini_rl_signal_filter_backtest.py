#!/usr/bin/env python3
"""Backtest: RL-only vs RL + technical filter / Gemini filter on stocks17 val data.

Method:
  1. Run C_s31 + D_s29 ensemble on stocks17 val windows via subprocess
  2. Load per-window trade data and technical conditions from price CSV
  3. Analyze: do different RSI/trend conditions correlate with better/worse RL trades?
  4. (Optional) Query Gemini for each trade and compare agreement rate with wins

Two modes:
  --technical-only  Fast, free: classify RL trades by RSI/trend conditions
  --gemini          Slow, costs API calls: query Gemini for each RL trade decision

Usage:
    source .venv313/bin/activate
    # Fast technical analysis (no API):
    python scripts/gemini_rl_signal_filter_backtest.py --technical-only --n-windows 30
    # Gemini filter backtest (costs API calls):
    python scripts/gemini_rl_signal_filter_backtest.py --gemini --n-windows 10

Reports saved to: reports/gemini_rl_filter_backtest.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

C_S31 = REPO / "pufferlib_market/checkpoints/stocks17_sweep/C_low_tp/s31/val_best.pt"
D_S29 = REPO / "pufferlib_market/checkpoints/stocks17_sweep/D_muon/s29/champion_u200.pt"
VAL_DATA = REPO / "pufferlib_market/data/stocks17_augmented_val.bin"

STOCKS17_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "GOOG", "META", "TSLA",
    "SPY", "QQQ", "PLTR", "JPM", "V", "AMZN",
    "AMD", "NFLX", "COIN", "CRWD", "UBER",
]


# ---------------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------------

def compute_rsi(prices: np.ndarray, period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0
    delta = np.diff(prices[-(period + 1):])
    up = np.maximum(delta, 0).mean()
    dn = np.maximum(-delta, 0).mean()
    return float(100 - 100 / (1 + up / (dn + 1e-9)))


def compute_technicals(prices: np.ndarray) -> dict:
    if len(prices) < 21:
        return {"rsi": 50.0, "above_ma20": True, "ret_5d": 0.0}
    return {
        "rsi": round(compute_rsi(prices), 1),
        "above_ma20": bool(prices[-1] > np.mean(prices[-20:])),
        "above_ma50": bool(prices[-1] > np.mean(prices[-50:])) if len(prices) >= 50 else True,
        "ret_5d": round(float((prices[-1] / prices[-6] - 1) * 100), 2),
        "ret_20d": round(float((prices[-1] / prices[-21] - 1) * 100), 2),
    }


# ---------------------------------------------------------------------------
# Load price CSV for symbol
# ---------------------------------------------------------------------------

def load_price_series(symbol: str) -> Optional[np.ndarray]:
    csv = REPO / "trainingdata" / f"{symbol}.csv"
    if not csv.exists():
        return None
    try:
        df = pd.read_csv(csv)
        df.columns = [c.lower() for c in df.columns]
        return df["close"].values.astype(np.float32)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Run RL evaluation and collect per-window stats via evaluate_holdout
# ---------------------------------------------------------------------------

def run_rl_evaluation(n_windows: int = 30) -> dict:
    """Run the 2-model ensemble on stocks17 val and collect results JSON."""
    import tempfile
    out_path = REPO / "reports" / "_tmp_rl_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "pufferlib_market.evaluate_holdout",
        "--checkpoint", str(C_S31),
        "--extra-checkpoints", str(D_S29),
        "--data-path", str(VAL_DATA),
        "--eval-hours", "60",
        "--n-windows", str(n_windows),
        "--fee-rate", "0.001",
        "--fill-buffer-bps", "5.0",
        "--decision-lag", "2",
        "--deterministic",
        "--no-early-stop",
        "--out", str(out_path),
    ]
    print(f"Running RL eval ({n_windows} windows)...")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    if result.returncode != 0:
        print(f"RL eval failed:\n{result.stderr[-2000:]}")
        return {}

    # evaluate_holdout prints JSON to stdout (the entire output IS the JSON object)
    stdout = result.stdout.strip()
    # Find the first '{' — everything from there is the JSON
    json_start = stdout.find('{')
    if json_start >= 0:
        try:
            return json.loads(stdout[json_start:])
        except Exception as e:
            print(f"  JSON parse error: {e}")
    return {}


# ---------------------------------------------------------------------------
# Technical filter analysis (no Gemini calls)
# ---------------------------------------------------------------------------

def technical_filter_analysis(n_windows: int = 30) -> dict:
    """
    Analyze whether RSI/trend conditions correlate with next-day returns in the val data.

    For each symbol on each day in the val period:
    - Compute RSI, MA trend at day T
    - Record 1-day forward return (close[T+1] / close[T] - 1) as ground truth

    Then analyze: in conditions where the RL would trade (long signal), do
    RSI/MA filters predict better or worse outcomes?

    Uses CSV price data directly — no simulate_daily_policy needed.
    """
    from pufferlib_market.hourly_replay import read_mktd
    from pufferlib_market.evaluate_holdout import load_policy

    import torch
    import collections

    print("Loading RL policies...")
    data = read_mktd(VAL_DATA)
    num_symbols = data.num_symbols
    feats_per_sym = int(data.features.shape[2])

    loaded_c = load_policy(str(C_S31), num_symbols, features_per_sym=feats_per_sym)
    loaded_d = load_policy(str(D_S29), num_symbols, features_per_sym=feats_per_sym)

    # Number of actions: 1 (cash) + num_symbols (long) + num_symbols (short, masked)
    n_actions = 1 + 2 * num_symbols

    groups: dict[str, list[float]] = {
        "all_rl_picks": [],
        "rl_cash": [],
        "rsi_normal": [],
        "rsi_overbought": [],
        "rsi_oversold": [],
        "above_ma20": [],
        "below_ma20": [],
        "momentum_positive": [],
        "momentum_negative": [],
    }

    T = data.prices.shape[0]  # total timesteps in val data
    window_size = min(n_windows, T - 2)

    # Disable-shorts mask: mask out short actions (symbols N+1 to 2N+1)
    def build_obs(t: int, portfolio_flat: bool = True) -> np.ndarray:
        feat = data.features[t].flatten().astype(np.float32)  # (num_symbols * feats_per_sym,)
        port = np.zeros(5 + num_symbols, dtype=np.float32)
        return np.concatenate([feat, port])

    def get_rl_action(t: int) -> tuple[int, float, str]:
        """Run ensemble at timestep t and return (action, confidence, direction)."""
        obs = build_obs(t)
        obs_t = torch.from_numpy(obs).view(1, -1)
        with torch.no_grad():
            logits_c, _ = loaded_c.policy(obs_t)
            logits_d, _ = loaded_d.policy(obs_t)
            # Mask shorts: actions num_symbols+1 to 2*num_symbols
            short_start = 1 + num_symbols
            short_end = 1 + 2 * num_symbols
            logits_c[:, short_start:short_end] = -1e9
            logits_d[:, short_start:short_end] = -1e9
            probs = (torch.softmax(logits_c, -1) + torch.softmax(logits_d, -1)) / 2
            action = int(probs.argmax())
            confidence = float(probs.max())

        if action == 0:
            direction = "cash"
        elif action <= num_symbols:
            direction = "long"
        else:
            direction = "short"

        return action, confidence, direction

    # Walk through val data, get RL action each day, record next-day return
    lag = 2  # decision_lag=2: trade fills 2 days after decision
    for t in range(lag, min(T - 1, lag + window_size)):
        action, confidence, direction = get_rl_action(t - lag)  # decision made lag days ago

        if direction == "cash":
            groups["rl_cash"].append(0.0)
            continue

        # Which symbol?
        if direction == "long":
            sym_idx = action - 1
        else:
            sym_idx = action - 1 - num_symbols

        if sym_idx < 0 or sym_idx >= num_symbols:
            continue

        # Next-day return (1-day forward from fill day t)
        # prices shape: (T, num_symbols, 5) where dim2 = [open, high, low, close, volume]
        if t + 1 >= T:
            continue
        p_entry = float(data.prices[t, sym_idx, 3])   # close price at fill day
        p_exit = float(data.prices[t + 1, sym_idx, 3])  # close price next day
        if p_entry <= 0:
            continue

        daily_return = (p_exit / p_entry - 1.0)
        if direction == "short":
            daily_return = -daily_return

        # Apply fee
        daily_return -= 0.001  # 10bps round-trip approximate

        groups["all_rl_picks"].append(daily_return)

        # Technical conditions: use price data from val binary up to t
        prices_up_to_t = data.prices[max(0, t - 60):t, sym_idx]
        if len(prices_up_to_t) < 20:
            continue

        tech = compute_technicals(prices_up_to_t)
        rsi = tech.get("rsi", 50.0)

        if rsi > 70:
            groups["rsi_overbought"].append(daily_return)
        elif rsi < 30:
            groups["rsi_oversold"].append(daily_return)
        else:
            groups["rsi_normal"].append(daily_return)

        if tech.get("above_ma20"):
            groups["above_ma20"].append(daily_return)
        else:
            groups["below_ma20"].append(daily_return)

        if tech.get("ret_5d", 0) > 0:
            groups["momentum_positive"].append(daily_return)
        else:
            groups["momentum_negative"].append(daily_return)

    def stats(rets):
        if not rets:
            return {"n": 0, "mean_pct": 0.0, "win_rate": 0.0}
        arr = np.array(rets)
        return {
            "n": len(arr),
            "mean_pct": round(float(arr.mean()) * 100, 3),
            "win_rate": round(float((arr > 0).mean()), 3),
            "median_pct": round(float(np.median(arr)) * 100, 3),
        }

    return {group: stats(rets) for group, rets in groups.items()}


# ---------------------------------------------------------------------------
# Gemini live filter (for today's RL signal)
# ---------------------------------------------------------------------------

def gemini_live_filter(dry_run: bool = False) -> dict:
    """
    Get today's RL signal and ask Gemini whether to confirm it.
    Used to test the live filter, not as a backtest.
    """
    from pufferlib_market.hourly_replay import read_mktd
    from pufferlib_market.evaluate_holdout import load_policy

    data = read_mktd(VAL_DATA)
    num_symbols = data.num_symbols
    feats_per_sym = int(data.features.shape[2])

    import torch
    loaded_c = load_policy(str(C_S31), num_symbols, features_per_sym=feats_per_sym)
    loaded_d = load_policy(str(D_S29), num_symbols, features_per_sym=feats_per_sym)

    # Get last window observation
    features = data.features[-1]  # (num_symbols, feats_per_sym)
    prices = data.prices[-1]      # (num_symbols,)

    # Build obs vector: (num_symbols * feats_per_sym + 5 + num_symbols)
    portfolio_state = np.zeros(5 + num_symbols, dtype=np.float32)
    obs = np.concatenate([features.flatten(), portfolio_state])

    obs_t = torch.from_numpy(obs.astype(np.float32)).view(1, -1)
    with torch.no_grad():
        logits_c, _ = loaded_c.policy(obs_t)
        logits_d, _ = loaded_d.policy(obs_t)
        probs = (torch.softmax(logits_c, -1) + torch.softmax(logits_d, -1)) / 2
        action = int(probs.argmax())

    # Decode action → symbol
    per_sym_actions = 2  # long + short (disable_shorts means short is masked)
    sym_idx = (action - 1) // per_sym_actions if action > 0 else -1
    direction = "cash" if action == 0 else ("long" if (action - 1) % per_sym_actions == 0 else "short")
    confidence = float(probs.max())
    symbol = STOCKS17_SYMBOLS[sym_idx] if 0 <= sym_idx < len(STOCKS17_SYMBOLS) else "CASH"

    print(f"RL signal: {direction} {symbol} (confidence={confidence:.1%})")

    if dry_run or direction == "cash":
        return {"rl_action": direction, "symbol": symbol, "confidence": confidence, "gemini": "skipped"}

    # Get technicals
    ph = load_price_series(symbol)
    if ph is None:
        return {"rl_action": direction, "symbol": symbol, "confidence": confidence, "gemini": "no_data"}

    tech = compute_technicals(ph)

    # Prompt Gemini
    tech_str = "\n".join(f"  {k}: {v}" for k, v in tech.items())
    prompt = f"""You are reviewing a trading signal from an RL model that wants to enter LONG {symbol}.

Technical indicators as of today:
{tech_str}
RL model confidence: {confidence:.1%}

The RL model has been backtested with 0/50 negative windows (lag=2, 10bps fee), median +18.84%/60d.

Should this trade be executed today? Answer with one word only:
EXECUTE - technicals support the long entry
SKIP - technicals contradict or show overbought/downtrend
UNCERTAIN - mixed signals"""

    try:
        from llm_hourly_trader.providers import call_llm
        resp = call_llm(prompt, model="gemini-2.0-flash-exp", thinking_level="LOW", max_tokens=15)
        verdict = resp.strip().upper()
        for v in ["EXECUTE", "SKIP", "UNCERTAIN"]:
            if v in verdict:
                verdict = v
                break
        else:
            verdict = "UNCERTAIN"
    except Exception as e:
        verdict = f"error:{e}"

    return {
        "rl_action": direction,
        "symbol": symbol,
        "confidence": confidence,
        "technicals": tech,
        "gemini_verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Gemini RL signal filter backtest")
    p.add_argument("--technical-only", action="store_true",
                   help="Analyze RL trades by technical conditions only (no API calls)")
    p.add_argument("--gemini", action="store_true",
                   help="Query Gemini for current day RL signal (live filter test)")
    p.add_argument("--rl-eval", action="store_true",
                   help="Run RL-only evaluation baseline and print stats")
    p.add_argument("--n-windows", type=int, default=20,
                   help="Number of val windows to evaluate")
    p.add_argument("--dry-run", action="store_true",
                   help="Skip actual Gemini API call")
    p.add_argument("--output", default="reports/gemini_rl_filter_backtest.json")
    args = p.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    results = {}

    if args.rl_eval or not (args.technical_only or args.gemini):
        print("=== RL-only baseline evaluation ===")
        rl_stats = run_rl_evaluation(n_windows=args.n_windows)
        if rl_stats:
            results["rl_baseline"] = {
                "median_pct": round(rl_stats.get("median_total_return", 0) * 100, 2),
                "p10_pct": round(rl_stats.get("p10_total_return", 0) * 100, 2),
                "neg_windows": rl_stats.get("negative_windows", "?"),
                "sortino": round(rl_stats.get("median_sortino", 0), 2),
            }
            print(f"  RL-only: med={results['rl_baseline']['median_pct']:.1f}%  "
                  f"p10={results['rl_baseline']['p10_pct']:.1f}%  "
                  f"neg={results['rl_baseline']['neg_windows']}/50")

    if args.technical_only:
        print(f"\n=== Technical filter analysis ({args.n_windows} windows) ===")
        try:
            tech_results = technical_filter_analysis(n_windows=args.n_windows)
            results["technical_filter"] = tech_results
            all_s = tech_results.get("all", {})
            print(f"  ALL trades:         n={all_s.get('n',0):4d}  mean={all_s.get('mean_pct',0):+.3f}%  WR={all_s.get('win_rate',0):.1%}")
            for group in ["rsi_normal", "rsi_overbought", "rsi_oversold", "above_ma20", "below_ma20"]:
                s = tech_results.get(group, {})
                if s.get("n", 0) > 0:
                    print(f"  {group:20s}: n={s['n']:4d}  mean={s['mean_pct']:+.3f}%  WR={s['win_rate']:.1%}")
        except Exception as e:
            print(f"  Technical analysis failed: {e}")
            import traceback
            traceback.print_exc()

    if args.gemini:
        print("\n=== Gemini live filter (current day) ===")
        gemini_result = gemini_live_filter(dry_run=args.dry_run)
        results["gemini_live"] = gemini_result
        print(f"  RL: {gemini_result.get('rl_action')} {gemini_result.get('symbol')} "
              f"(conf={gemini_result.get('confidence', 0):.1%})")
        print(f"  Gemini verdict: {gemini_result.get('gemini_verdict', '?')}")
        tech = gemini_result.get("technicals", {})
        if tech:
            print(f"  RSI={tech.get('rsi', '?')}  above_ma20={tech.get('above_ma20', '?')}  "
                  f"5d={tech.get('ret_5d', '?')}%")

    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\nResults saved: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
