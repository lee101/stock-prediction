#!/usr/bin/env python3
"""Evaluate ensemble of mass_daily PPO checkpoints on 120-day market sim.

Tests three ensemble modes (logit_avg, majority_vote, softmax_avg) and
compares subsets (top-3, top-5) against the best single model.

Usage:
    source .venv313/bin/activate
    python eval_ensemble_marketsim.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

# Monkey-patch early exit before importing anything that uses it
import src.market_sim_early_exit as _mse

def _no_early_exit(*args, **kwargs):
    return _mse.EarlyExitDecision(
        should_stop=False,
        progress_fraction=0.0,
        total_return=0.0,
        max_drawdown=0.0,
    )

_mse.evaluate_drawdown_vs_profit_early_exit = _no_early_exit

from pufferlib_market.hourly_replay import read_mktd, simulate_daily_policy
from pufferlib_market.evaluate_tail import _slice_tail
from pufferlib_market.evaluate_multiperiod import load_policy, make_policy_fn
from pufferlib_market.ensemble_inference import EnsembleTrader, ENSEMBLE_MODES, load_policies
from pufferlib_market.metrics import annualize_total_return

# -- Configuration --
_REPO_ROOT = Path(__file__).resolve().parent
_MAIN_REPO = Path("/nvme0n1-disk/code/stock-prediction")

def _resolve_path(relative: str) -> Path:
    """Resolve a path, preferring the local directory but falling back to main repo."""
    local = _REPO_ROOT / relative
    if local.exists():
        return local
    main = _MAIN_REPO / relative
    if main.exists():
        return main
    return local

DATA_PATH = _resolve_path("pufferlib_market/data/crypto5_daily_val.bin")
CKPT_ROOT = _resolve_path("pufferlib_market/checkpoints/mass_daily")

# Top-5 mass_daily seeds ranked by Sortino (from prior evaluations)
TOP5_SEEDS = [
    "tp0.15_s314",
    "tp0.10_s42",
    "tp0.20_s123",
    "tp0.05_s123",
    "tp0.20_s2024",
]

# Sim parameters matching production
FEE_RATE = 0.001
FILL_BUFFER_BPS = 8.0
PERIODS_PER_YEAR = 365.0
MAX_LEVERAGE = 1.0


def _sim_kwargs() -> dict:
    return dict(
        fee_rate=FEE_RATE,
        fill_buffer_bps=FILL_BUFFER_BPS,
        max_leverage=MAX_LEVERAGE,
        periods_per_year=PERIODS_PER_YEAR,
    )


def _build_result(name: str, result, eval_steps: int, **extra) -> dict:
    ann_ret = annualize_total_return(
        float(result.total_return),
        periods=float(eval_steps),
        periods_per_year=PERIODS_PER_YEAR,
    )
    out = {
        "name": name,
        "total_return": float(result.total_return),
        "annualized_return": float(ann_ret),
        "sortino": float(result.sortino),
        "max_drawdown": float(result.max_drawdown),
        "num_trades": int(result.num_trades),
        "win_rate": float(result.win_rate),
    }
    out.update(extra)
    return out


def find_checkpoint(seed_name: str) -> str:
    """Find best.pt for a given seed directory."""
    ckpt = CKPT_ROOT / seed_name / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    return str(ckpt)


def evaluate_single(ckpt_path: str, data, eval_steps: int, device: torch.device) -> dict:
    """Evaluate a single checkpoint on the given data slice."""
    policy, _, _ = load_policy(ckpt_path, data.num_symbols, device=device)
    policy_fn = make_policy_fn(policy, num_symbols=data.num_symbols, deterministic=True, device=device)
    result = simulate_daily_policy(data, policy_fn, max_steps=eval_steps, **_sim_kwargs())
    return _build_result(Path(ckpt_path).parent.name, result, eval_steps)


def evaluate_ensemble(policies, data, eval_steps: int, mode: str, device_str: str, label: str) -> dict:
    """Evaluate an ensemble of pre-loaded policies."""
    ensemble = EnsembleTrader(policies=policies, device=device_str, mode=mode)
    policy_fn = ensemble.get_policy_fn(deterministic=True)
    result = simulate_daily_policy(data, policy_fn, max_steps=eval_steps, **_sim_kwargs())
    return _build_result(label, result, eval_steps, mode=mode, n_models=len(policies))


def format_table(results: list[dict]) -> str:
    """Format results as an aligned comparison table."""
    lines = []
    header = f"{'Name':<32} {'Return%':>9} {'AnnRet%':>9} {'Sortino':>8} {'MaxDD%':>7} {'Trades':>7} {'WinRate':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    for r in results:
        ret_pct = r["total_return"] * 100
        ann_pct = r["annualized_return"] * 100
        dd_pct = r["max_drawdown"] * 100
        wr_pct = r["win_rate"] * 100
        lines.append(
            f"{r['name']:<32} {ret_pct:>+8.2f}% {ann_pct:>+8.2f}% {r['sortino']:>8.2f} "
            f"{dd_pct:>6.2f}% {r['num_trades']:>7d} {wr_pct:>7.1f}%"
        )
    return "\n".join(lines)


def main():
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    if not DATA_PATH.exists():
        print(f"ERROR: Data file not found: {DATA_PATH}")
        sys.exit(1)

    data = read_mktd(DATA_PATH)
    print(f"Loaded data: {data.num_symbols} symbols, {data.num_timesteps} timesteps")
    print(f"Symbols: {data.symbols}")

    eval_steps = min(120, data.num_timesteps - 1)
    print(f"Evaluating on {eval_steps}-day window")
    print(f"Device: {device_str}")
    print()

    tail = _slice_tail(data, steps=eval_steps)

    # Find checkpoints and load policies once
    all_ckpts = []
    available_seeds = []
    for seed in TOP5_SEEDS:
        try:
            ckpt = find_checkpoint(seed)
            all_ckpts.append(ckpt)
            available_seeds.append(seed)
        except FileNotFoundError as e:
            print(f"WARNING: {e}")

    if len(all_ckpts) < 2:
        print("ERROR: Need at least 2 checkpoints for ensemble evaluation")
        sys.exit(1)

    print(f"Found {len(all_ckpts)} checkpoints: {available_seeds}")

    # Load all policies once to share across single-model and ensemble evaluations
    print("Loading policies...", flush=True)
    all_policies = load_policies(all_ckpts, tail.num_symbols, device)
    print()

    all_results = []

    # 1. Evaluate each single model
    print("=" * 70)
    print("SINGLE MODEL EVALUATION")
    print("=" * 70)
    single_results = []
    for i, (policy, seed) in enumerate(zip(all_policies, available_seeds)):
        t0 = time.time()
        policy_fn = make_policy_fn(policy, num_symbols=tail.num_symbols, deterministic=True, device=device)
        result = simulate_daily_policy(tail, policy_fn, max_steps=eval_steps, **_sim_kwargs())
        r = _build_result(seed, result, eval_steps)
        elapsed = time.time() - t0
        single_results.append(r)
        all_results.append(r)
        print(f"  [{i+1}/{len(all_policies)}] {seed}: Sortino={r['sortino']:.2f}, "
              f"Return={r['total_return']*100:+.2f}%, "
              f"MaxDD={r['max_drawdown']*100:.2f}% ({elapsed:.1f}s)")
    print()

    best_single = max(single_results, key=lambda r: r["sortino"])
    print(f"Best single model: {best_single['name']} (Sortino={best_single['sortino']:.2f})")
    print()

    # 2. Evaluate ensembles (policies already loaded, no redundant I/O)
    print("=" * 70)
    print("ENSEMBLE EVALUATION")
    print("=" * 70)

    subsets = []
    if len(all_policies) >= 3:
        subsets.append(("top3", all_policies[:3]))
    if len(all_policies) >= 5:
        subsets.append(("top5", all_policies[:5]))
    elif len(all_policies) > 3:
        subsets.append((f"top{len(all_policies)}", all_policies))

    ensemble_results = []
    for subset_name, policies in subsets:
        for mode in ENSEMBLE_MODES:
            label = f"ensemble_{subset_name}_{mode}"
            t0 = time.time()
            r = evaluate_ensemble(policies, tail, eval_steps, mode, device_str, label)
            elapsed = time.time() - t0
            ensemble_results.append(r)
            all_results.append(r)
            print(f"  {label}: Sortino={r['sortino']:.2f}, "
                  f"Return={r['total_return']*100:+.2f}%, "
                  f"MaxDD={r['max_drawdown']*100:.2f}% ({elapsed:.1f}s)")

    print()

    # 3. Summary comparison table
    print("=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(format_table(all_results))
    print()

    # 4. Verdict
    best_ensemble = max(ensemble_results, key=lambda r: r["sortino"]) if ensemble_results else None
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"Best single model:  {best_single['name']:<24} Sortino={best_single['sortino']:.2f}  "
          f"Return={best_single['total_return']*100:+.2f}%  MaxDD={best_single['max_drawdown']*100:.2f}%")
    if best_ensemble:
        print(f"Best ensemble:      {best_ensemble['name']:<24} Sortino={best_ensemble['sortino']:.2f}  "
              f"Return={best_ensemble['total_return']*100:+.2f}%  MaxDD={best_ensemble['max_drawdown']*100:.2f}%")
        delta_sortino = best_ensemble["sortino"] - best_single["sortino"]
        delta_return = (best_ensemble["total_return"] - best_single["total_return"]) * 100
        print()
        if best_ensemble["sortino"] > best_single["sortino"]:
            print(f"Ensemble WINS: +{delta_sortino:.2f} Sortino, {delta_return:+.2f}% return improvement")
        else:
            print(f"Single model WINS: ensemble is {delta_sortino:.2f} Sortino, {delta_return:+.2f}% return behind")

    output_path = Path("eval_ensemble_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
