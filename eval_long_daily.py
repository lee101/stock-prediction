#!/usr/bin/env python3
"""Compare long-trained (30min) daily checkpoints against 5-min mass_daily counterparts.

Evaluates on 120d and 180d market sim windows to check whether longer training
improves or overfits compared to the 5-min budget models.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

# Disable early exit so we get full-period results
import src.market_sim_early_exit as _mse

def _no_early_exit(*args, **kwargs):
    return _mse.EarlyExitDecision(
        should_stop=False, progress_fraction=0.0,
        total_return=0.0, max_drawdown=0.0,
    )

_mse.evaluate_drawdown_vs_profit_early_exit = _no_early_exit

from pufferlib_market.hourly_replay import read_mktd, simulate_daily_policy
from pufferlib_market.metrics import annualize_total_return
from pufferlib_market.evaluate_multiperiod import load_policy, make_policy_fn
from pufferlib_market.evaluate_tail import _slice_tail

DATA = "pufferlib_market/data/crypto5_daily_val.bin"
BASE = Path("pufferlib_market/checkpoints")

# Long-trained checkpoints (30min budget)
LONG_CKPTS = {
    "long/tp0.15_s314": BASE / "long_daily/tp0.15_s314_30min/best.pt",
    "long/tp0.10_s42":  BASE / "long_daily/tp0.10_s42_30min/best.pt",
}

# Their 5-min mass_daily counterparts for comparison
SHORT_CKPTS = {
    "5min/tp0.15_s314": BASE / "mass_daily/tp0.15_s314/best.pt",
    "5min/tp0.10_s42":  BASE / "mass_daily/tp0.10_s42/best.pt",
}

# Also include the best known daily checkpoints for reference
REFERENCE_CKPTS = {
    "ref/trade_pen_05": BASE / "autoresearch_daily/trade_pen_05/best.pt",
}

PERIODS = {"120d": 120, "180d": 180}

FEE_RATE = 0.001
FILL_BUFFER_BPS = 8.0
MAX_LEVERAGE = 1.0
PERIODS_PER_YEAR = 365.0


def load_and_eval(ckpt_path, data, periods, device="cpu"):
    """Load a checkpoint and evaluate across multiple time periods."""
    device = torch.device(device)
    nsym = data.num_symbols

    policy, _, _ = load_policy(str(ckpt_path), nsym, device=device)
    policy_fn = make_policy_fn(policy, num_symbols=nsym, deterministic=True, device=device)

    results = {}
    for pname, steps in periods.items():
        if data.num_timesteps < steps + 1:
            results[pname] = {"error": "data too short"}
            continue
        tail = _slice_tail(data, steps=steps)
        sim = simulate_daily_policy(
            tail, policy_fn, max_steps=steps,
            fee_rate=FEE_RATE, fill_buffer_bps=FILL_BUFFER_BPS,
            max_leverage=MAX_LEVERAGE, periods_per_year=PERIODS_PER_YEAR,
        )
        ann = annualize_total_return(
            float(sim.total_return), periods=float(steps),
            periods_per_year=PERIODS_PER_YEAR,
        )
        results[pname] = {
            "return_pct": sim.total_return * 100,
            "annualized_pct": ann * 100,
            "sortino": sim.sortino,
            "max_dd_pct": sim.max_drawdown * 100,
            "trades": sim.num_trades,
            "wr": sim.win_rate,
            "hold": sim.avg_hold_steps,
        }
    return results


def main():
    data_path = Path(DATA)
    if not data_path.exists():
        print(f"ERROR: validation data not found at {DATA}")
        sys.exit(1)

    data = read_mktd(data_path)
    print(f"Loaded {DATA}: {data.num_timesteps} days, {data.num_symbols} symbols")
    print(f"Settings: {FILL_BUFFER_BPS}bps fill buffer, {FEE_RATE*100:.1f}% fee, deterministic\n")

    # Gather all checkpoints to evaluate
    all_ckpts = {}
    for group in [LONG_CKPTS, SHORT_CKPTS, REFERENCE_CKPTS]:
        for name, path in group.items():
            if path.exists():
                all_ckpts[name] = path
            else:
                print(f"  SKIP {name}: {path} not found")

    if not all_ckpts:
        print("No checkpoints found to evaluate.")
        sys.exit(1)

    print(f"Evaluating {len(all_ckpts)} checkpoints...\n")

    results_all = {}
    for name, ckpt in sorted(all_ckpts.items()):
        try:
            r = load_and_eval(ckpt, data, PERIODS)
            results_all[name] = r
        except Exception as e:
            print(f"  {name}: ERROR {e}")

    # Print comparison table
    header = f"{'Name':<25} {'120d Ret%':>10} {'120d Sort':>10} {'120d DD%':>9} {'180d Ret%':>10} {'180d Sort':>10} {'180d DD%':>9}"
    print(header)
    print("-" * len(header))

    for name in sorted(results_all.keys()):
        r = results_all[name]
        r120 = r.get("120d", {})
        r180 = r.get("180d", {})
        if "error" in r120 or "error" in r180:
            print(f"{name:<25} {'(error)':>10}")
            continue
        print(
            f"{name:<25} "
            f"{r120.get('return_pct', 0):>+9.1f}% "
            f"{r120.get('sortino', 0):>10.2f} "
            f"{r120.get('max_dd_pct', 0):>8.1f}% "
            f"{r180.get('return_pct', 0):>+9.1f}% "
            f"{r180.get('sortino', 0):>10.2f} "
            f"{r180.get('max_dd_pct', 0):>8.1f}%"
        )

    # Print summary: did long training help?
    print("\n=== Long vs Short Training Comparison ===")
    pairs = [
        ("tp0.15_s314", "long/tp0.15_s314", "5min/tp0.15_s314"),
        ("tp0.10_s42", "long/tp0.10_s42", "5min/tp0.10_s42"),
    ]
    for label, long_name, short_name in pairs:
        if long_name in results_all and short_name in results_all:
            for period in ["120d", "180d"]:
                lr = results_all[long_name].get(period, {})
                sr = results_all[short_name].get(period, {})
                if "error" in lr or "error" in sr:
                    continue
                diff = lr.get("return_pct", 0) - sr.get("return_pct", 0)
                verdict = "BETTER" if diff > 0 else "WORSE" if diff < 0 else "SAME"
                print(
                    f"  {label} {period}: long={lr.get('return_pct', 0):+.1f}% "
                    f"vs short={sr.get('return_pct', 0):+.1f}% "
                    f"=> {verdict} ({diff:+.1f}pp)"
                )

    # Save results
    out_path = "long_daily_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results_all, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
