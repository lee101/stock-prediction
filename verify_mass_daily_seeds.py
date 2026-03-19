#!/usr/bin/env python3
"""Verify mass_daily results across all seeds and trade_penalty values."""

import sys
import json
from pathlib import Path

import numpy as np
import torch

# Disable early exit
import src.market_sim_early_exit as _mse
_orig = _mse.evaluate_drawdown_vs_profit_early_exit
def _no_early_exit(*args, **kwargs):
    return _mse.EarlyExitDecision(should_stop=False, progress_fraction=0.0,
                                   total_return=0.0, max_drawdown=0.0)
_mse.evaluate_drawdown_vs_profit_early_exit = _no_early_exit

from pufferlib_market.hourly_replay import read_mktd, simulate_daily_policy
from pufferlib_market.metrics import annualize_total_return
from pufferlib_market.evaluate_tail import (
    TradingPolicy, _infer_num_actions, _infer_arch, _infer_hidden_size, _slice_tail,
)
from torch.distributions import Categorical

DATA = "pufferlib_market/data/crypto5_daily_val.bin"
BASE = Path("pufferlib_market/checkpoints")

# All mass_daily checkpoints
mass_ckpts = sorted(BASE.glob("mass_daily/*/best.pt"))
# All autoresearch_daily relevant checkpoints
ar_ckpts = [
    BASE / "autoresearch_daily/trade_pen_05/best.pt",
    BASE / "autoresearch_daily_combos/tp05_ent001/best.pt",
]


def load_and_eval(ckpt_path, data, periods_dict, device="cpu"):
    device = torch.device(device)
    nsym = data.num_symbols
    payload = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    sd = payload.get("model", payload)
    obs_size = nsym * 16 + 5 + nsym
    na = _infer_num_actions(sd, fallback=1 + 2 * nsym)
    arch = _infer_arch(sd)
    h = _infer_hidden_size(sd, arch=arch)
    policy = TradingPolicy(obs_size, na, hidden=h).to(device)
    policy.load_state_dict(sd)
    policy.eval()

    def policy_fn(obs):
        t = torch.from_numpy(obs.astype(np.float32)).to(device).view(1, -1)
        with torch.no_grad():
            logits, _ = policy(t)
        return int(torch.argmax(logits, dim=-1).item())

    results = {}
    for pname, steps in periods_dict.items():
        if data.num_timesteps < steps + 1:
            results[pname] = {"error": "data too short"}
            continue
        tail = _slice_tail(data, steps=steps)
        sim = simulate_daily_policy(tail, policy_fn, max_steps=steps,
                                     fee_rate=0.001, fill_buffer_bps=8.0,
                                     max_leverage=1.0, periods_per_year=365.0)
        ann = annualize_total_return(float(sim.total_return), periods=float(steps),
                                      periods_per_year=365.0)
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
    data = read_mktd(Path(DATA))
    periods = {"60d": 60, "90d": 90, "120d": 120, "180d": 180}

    all_ckpts = [(str(p.parent.name), p) for p in mass_ckpts]
    all_ckpts += [("autoresearch/" + p.parent.name, p) for p in ar_ckpts if p.exists()]

    print(f"Evaluating {len(all_ckpts)} checkpoints on crypto5 daily val ({data.num_timesteps} days)")
    print(f"Settings: 8bps fill, 0.1% fee, deterministic\n")

    results_120d = []

    for name, ckpt in all_ckpts:
        try:
            r = load_and_eval(ckpt, data, periods)
        except Exception as e:
            print(f"  {name}: ERROR {e}")
            continue
        r120 = r.get("120d", {})
        if "error" not in r120:
            results_120d.append((name, r))

    # Sort by 120d Sortino
    results_120d.sort(key=lambda x: x[1].get("120d", {}).get("sortino", -999), reverse=True)

    print(f"{'Name':<25} {'60d Ret%':>9} {'90d Ret%':>9} {'120d Ret%':>10} {'120d Sort':>10} {'120d DD%':>9} {'180d Ret%':>10}")
    print("-" * 95)
    for name, r in results_120d:
        r60 = r.get("60d", {})
        r90 = r.get("90d", {})
        r120 = r.get("120d", {})
        r180 = r.get("180d", {})
        print(f"{name:<25} "
              f"{r60.get('return_pct', 0):>+8.1f}% "
              f"{r90.get('return_pct', 0):>+8.1f}% "
              f"{r120.get('return_pct', 0):>+9.1f}% "
              f"{r120.get('sortino', 0):>10.2f} "
              f"{r120.get('max_dd_pct', 0):>8.1f}% "
              f"{r180.get('return_pct', 0):>+9.1f}%")

    # Save
    with open("mass_daily_seeds_comparison.json", "w") as f:
        json.dump({n: r for n, r in results_120d}, f, indent=2)
    print(f"\nSaved to mass_daily_seeds_comparison.json")


if __name__ == "__main__":
    main()
