#!/usr/bin/env python3
"""
Full 120-day market simulator comparison.
Runs each checkpoint on its matching val data with no early exit.
Uses the pure-Python market simulator (simulate_daily_policy) for realistic PnL.
"""

import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Disable early exit by monkey-patching
import src.market_sim_early_exit as _mse
_orig = _mse.evaluate_drawdown_vs_profit_early_exit
def _no_early_exit(*args, **kwargs):
    return _mse.EarlyExitDecision(should_stop=False, progress_fraction=0.0,
                                   total_return=0.0, max_drawdown=0.0)
_mse.evaluate_drawdown_vs_profit_early_exit = _no_early_exit

from pufferlib_market.hourly_replay import MktdData, read_mktd, simulate_daily_policy
from pufferlib_market.metrics import annualize_total_return
from pufferlib_market.evaluate_tail import (
    TradingPolicy, ResidualTradingPolicy,
    _infer_num_actions, _infer_arch, _infer_hidden_size,
    _infer_resmlp_blocks, _mask_all_shorts, _slice_tail,
)


def load_policy(ckpt_path, num_symbols, device):
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = payload.get("model", payload) if isinstance(payload, dict) else payload
    obs_size = num_symbols * 16 + 5 + num_symbols
    num_actions = _infer_num_actions(state_dict, fallback=1 + 2 * num_symbols)
    arch = _infer_arch(state_dict)
    hidden = _infer_hidden_size(state_dict, arch=arch)

    if arch == "resmlp":
        blocks = _infer_resmlp_blocks(state_dict)
        policy = ResidualTradingPolicy(obs_size, num_actions, hidden=hidden, num_blocks=blocks).to(device)
    else:
        policy = TradingPolicy(obs_size, num_actions, hidden=hidden).to(device)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy, num_actions


def make_policy_fn(policy, num_symbols, device, deterministic=True):
    from torch.distributions import Categorical
    def _fn(obs):
        obs_t = torch.from_numpy(obs.astype(np.float32, copy=False)).to(device).view(1, -1)
        with torch.no_grad():
            logits, _ = policy(obs_t)
        if deterministic:
            return int(torch.argmax(logits, dim=-1).item())
        return int(Categorical(logits=logits).sample().item())
    return _fn


def run_backtest(name, ckpt_path, data_path, periods, fee_rate=0.001, fill_buffer_bps=8.0,
                 periods_per_year=365.0, device_str="cpu"):
    device = torch.device(device_str)
    data = read_mktd(Path(data_path))
    nsym = data.num_symbols

    try:
        policy, num_actions = load_policy(ckpt_path, nsym, device)
    except Exception as e:
        return {p: {"error": str(e)} for p in periods}

    policy_fn = make_policy_fn(policy, nsym, device)

    results = {}
    for period_name, steps in sorted(periods.items(), key=lambda x: x[1]):
        if data.num_timesteps < steps + 1:
            results[period_name] = {"error": f"Data too short ({data.num_timesteps} < {steps+1})"}
            continue

        tail = _slice_tail(data, steps=steps)
        sim = simulate_daily_policy(
            tail, policy_fn, max_steps=steps,
            fee_rate=fee_rate, fill_buffer_bps=fill_buffer_bps,
            max_leverage=1.0, periods_per_year=periods_per_year,
        )

        ann = annualize_total_return(float(sim.total_return), periods=float(steps),
                                      periods_per_year=periods_per_year)
        results[period_name] = {
            "return_pct": sim.total_return * 100,
            "annualized_pct": ann * 100,
            "sortino": sim.sortino,
            "max_drawdown_pct": sim.max_drawdown * 100,
            "num_trades": sim.num_trades,
            "win_rate": sim.win_rate,
            "avg_hold": sim.avg_hold_steps,
        }
    return results


CANDIDATES = [
    # (name, checkpoint, val_data)
    ("mixed23/baseline_anneal_lr",
     "pufferlib_market/checkpoints/autoresearch_mixed23_daily/baseline_anneal_lr/best.pt",
     "pufferlib_market/data/mixed23_daily_val.bin"),

    ("mixed23/ent_anneal",
     "pufferlib_market/checkpoints/autoresearch_mixed23_daily/ent_anneal/best.pt",
     "pufferlib_market/data/mixed23_daily_val.bin"),

    ("mixed23/clip_anneal",
     "pufferlib_market/checkpoints/autoresearch_mixed23_daily/clip_anneal/best.pt",
     "pufferlib_market/data/mixed23_daily_val.bin"),

    ("mixed23/envs_256",
     "pufferlib_market/checkpoints/autoresearch_mixed23_daily/envs_256/best.pt",
     "pufferlib_market/data/mixed23_daily_val.bin"),

    ("crypto8/clip_anneal",
     "pufferlib_market/checkpoints/autoresearch_crypto8_daily/clip_anneal/best.pt",
     "pufferlib_market/data/crypto8_daily_val.bin"),

    ("crypto8/slip_10bps",
     "pufferlib_market/checkpoints/autoresearch_crypto8_daily/slip_10bps/best.pt",
     "pufferlib_market/data/crypto8_daily_val.bin"),

    ("crypto5/trade_pen_05 [PROD]",
     "pufferlib_market/checkpoints/autoresearch_daily/trade_pen_05/best.pt",
     "pufferlib_market/data/crypto5_daily_val.bin"),

    ("crypto5/cosine_lr",
     "pufferlib_market/checkpoints/autoresearch_daily/cosine_lr/best.pt",
     "pufferlib_market/data/crypto5_daily_val.bin"),

    ("mass/tp0.05_s123",
     "pufferlib_market/checkpoints/mass_daily/tp0.05_s123/best.pt",
     "pufferlib_market/data/crypto5_daily_val.bin"),

    ("mixed32/ent_anneal",
     "pufferlib_market/checkpoints/mixed32_daily_ent_anneal/best.pt",
     "pufferlib_market/data/mixed32_daily_val.bin"),
]

PERIODS = {"30d": 30, "60d": 60, "90d": 90, "120d": 120, "180d": 180}


def main():
    print("=" * 110)
    print("FULL MARKET SIMULATOR BACKTEST - DETERMINISTIC TAIL, NO EARLY EXIT")
    print("Settings: 8bps fill buffer, 0.1% fee, 1x leverage, periods_per_year=365")
    print("=" * 110)

    all_results = {}
    for name, ckpt, data in CANDIDATES:
        print(f"\n--- {name} ---")
        ckpt_path = Path(ckpt)
        if not ckpt_path.exists():
            print(f"  SKIP: checkpoint not found")
            continue
        results = run_backtest(name, str(ckpt_path), data, PERIODS)
        all_results[name] = results
        for p, r in sorted(results.items(), key=lambda x: PERIODS.get(x[0], 0)):
            if "error" in r:
                print(f"  {p}: ERROR - {r['error']}")
            else:
                print(f"  {p}: return={r['return_pct']:+.2f}% sortino={r['sortino']:.2f} "
                      f"maxDD={r['max_drawdown_pct']:.1f}% trades={r['num_trades']} "
                      f"WR={r['win_rate']:.1%} hold={r['avg_hold']:.1f}d")

    # Summary table for 120d
    print(f"\n{'='*110}")
    print(f"120-DAY PNL COMPARISON (deterministic tail slice)")
    print(f"{'='*110}")
    print(f"{'Name':<35} {'Return%':>9} {'Annual%':>9} {'Sortino':>8} {'MaxDD%':>7} {'Trades':>7} {'WR':>7} {'Hold':>6}")
    print(f"{'-'*35} {'-'*9} {'-'*9} {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*6}")

    items = []
    for name, periods in all_results.items():
        r = periods.get("120d", {})
        if "error" not in r and r:
            items.append((name, r))

    items.sort(key=lambda x: x[1].get("sortino", -999), reverse=True)
    for name, r in items:
        print(f"{name:<35} {r['return_pct']:>+8.2f}% {r['annualized_pct']:>+8.1f}% "
              f"{r['sortino']:>8.2f} {r['max_drawdown_pct']:>6.1f}% "
              f"{r['num_trades']:>7d} {r['win_rate']:>6.1%} {r['avg_hold']:>5.1f}d")

    # Also show full period breakdown for top 3
    print(f"\n{'='*110}")
    print(f"FULL PERIOD BREAKDOWN - TOP CANDIDATES")
    print(f"{'='*110}")
    top3 = [name for name, _ in items[:3]]
    for name in top3:
        print(f"\n  {name}:")
        print(f"  {'Period':<8} {'Return%':>9} {'Annual%':>9} {'Sortino':>8} {'MaxDD%':>7} {'Trades':>7}")
        for p in ["30d", "60d", "90d", "120d", "180d"]:
            r = all_results[name].get(p, {})
            if "error" in r:
                print(f"  {p:<8} {r['error']}")
            elif r:
                print(f"  {p:<8} {r['return_pct']:>+8.2f}% {r['annualized_pct']:>+8.1f}% "
                      f"{r['sortino']:>8.2f} {r['max_drawdown_pct']:>6.1f}% {r['num_trades']:>7d}")

    # Save
    with open("marketsim_120d_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to marketsim_120d_comparison.json")


if __name__ == "__main__":
    main()
