#!/usr/bin/env python3
"""Seed diversity sweep for trade_penalty=0.15.

Trains tp=0.15 with 10 new seeds [1,2,3,4,5,6,8,9,10,11] (avoiding
existing seeds 42,7,123,314,2024 in mass_daily), then evaluates ALL 15
tp=0.15 checkpoints on the 120-day market sim.

Checkpoints go to pufferlib_market/checkpoints/mass_daily_v2/tp0.15_s{SEED}/best.pt
Results logged to pufferlib_market/tp15_seeds_leaderboard.csv
"""

from __future__ import annotations

import csv
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Monkey-patch: disable early-exit so the full sim window is evaluated
# ---------------------------------------------------------------------------
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
from pufferlib_market.metrics import annualize_total_return
from pufferlib_market.evaluate_multiperiod import load_policy, make_policy_fn
from pufferlib_market.evaluate_tail import _slice_tail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent
TRAIN_DATA = "pufferlib_market/data/crypto5_daily_train.bin"
VAL_DATA = "pufferlib_market/data/crypto5_daily_val.bin"

NEW_SEEDS = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11]
EXISTING_SEEDS = [42, 7, 123, 314, 2024]
ALL_SEEDS = EXISTING_SEEDS + NEW_SEEDS

TRADE_PENALTY = 0.15
TIME_BUDGET = 300  # seconds per seed

# Checkpoint paths
MASS_DAILY_BASE = Path("pufferlib_market/checkpoints/mass_daily")
MASS_DAILY_V2_BASE = Path("pufferlib_market/checkpoints/mass_daily_v2")

LEADERBOARD = Path("pufferlib_market/tp15_seeds_leaderboard.csv")

# Evaluation settings
EVAL_PERIODS = {"60d": 60, "90d": 90, "120d": 120, "180d": 180}
FEE_RATE = 0.001
FILL_BUFFER_BPS = 8.0
MAX_LEVERAGE = 1.0
PERIODS_PER_YEAR = 365.0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_seed(seed: int, time_budget: int = TIME_BUDGET) -> bool:
    """Train a single seed with the given time budget. Returns True on success."""
    ckpt_dir = str(MASS_DAILY_V2_BASE / f"tp0.15_s{seed}")
    best_pt = Path(ckpt_dir) / "best.pt"
    if best_pt.exists():
        print(f"  Seed {seed}: checkpoint already exists at {best_pt}, skipping training")
        return True

    cmd = [
        sys.executable, "-u", "-m", "pufferlib_market.train",
        "--data-path", TRAIN_DATA,
        "--total-timesteps", "999999999",
        "--max-steps", "720",
        "--hidden-size", "1024",
        "--lr", "3e-4",
        "--ent-coef", "0.05",
        "--gamma", "0.99",
        "--gae-lambda", "0.95",
        "--num-envs", "128",
        "--rollout-len", "256",
        "--ppo-epochs", "4",
        "--seed", str(seed),
        "--reward-scale", "10.0",
        "--reward-clip", "5.0",
        "--cash-penalty", "0.01",
        "--fee-rate", "0.001",
        "--trade-penalty", str(TRADE_PENALTY),
        "--anneal-lr",
        "--checkpoint-dir", ckpt_dir,
        "--periods-per-year", "365.0",
    ]

    print(f"  Seed {seed}: training for {time_budget}s ...")
    t0 = time.time()
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(REPO),
            preexec_fn=os.setsid,
        )
        try:
            while time.time() - t0 < time_budget:
                if proc.poll() is not None:
                    break
                try:
                    line = proc.stdout.readline()
                    if line:
                        decoded = line.decode("utf-8", errors="replace").strip()
                        if "ret=" in decoded:
                            print(f"    {decoded}")
                except Exception:
                    pass
            # Kill if still running after budget
            if proc.poll() is None:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    proc.wait(timeout=5)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass

        elapsed = time.time() - t0
        print(f"  Seed {seed}: done in {elapsed:.0f}s")

        if not best_pt.exists():
            print(f"  WARNING: no best.pt found at {best_pt}")
            return False
        return True

    except Exception as e:
        print(f"  Seed {seed}: ERROR during training: {e}")
        return False


def train_all_new_seeds():
    """Train all new seeds sequentially."""
    print(f"\n{'='*60}")
    print(f"Training {len(NEW_SEEDS)} new seeds with trade_penalty={TRADE_PENALTY}")
    print(f"Time budget: {TIME_BUDGET}s per seed")
    print(f"Checkpoint root: {MASS_DAILY_V2_BASE}")
    print(f"{'='*60}\n")

    results = {}
    for i, seed in enumerate(NEW_SEEDS):
        print(f"\n[{i+1}/{len(NEW_SEEDS)}] Seed {seed}")
        ok = train_seed(seed)
        results[seed] = ok

    successes = sum(1 for v in results.values() if v)
    print(f"\nTraining complete: {successes}/{len(NEW_SEEDS)} seeds succeeded")
    return results


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def load_and_eval(ckpt_path: Path, data, periods: dict, device: str = "cpu") -> dict:
    """Load a checkpoint and evaluate on multiple time periods."""
    device_t = torch.device(device)
    nsym = data.num_symbols

    policy, _, _ = load_policy(str(ckpt_path), nsym, device=device_t)
    policy_fn = make_policy_fn(policy, num_symbols=nsym, deterministic=True, device=device_t)

    results = {}
    for pname, steps in periods.items():
        if data.num_timesteps < steps + 1:
            results[pname] = {"error": "data too short"}
            continue
        tail = _slice_tail(data, steps=steps)
        sim = simulate_daily_policy(
            tail,
            policy_fn,
            max_steps=steps,
            fee_rate=FEE_RATE,
            fill_buffer_bps=FILL_BUFFER_BPS,
            max_leverage=MAX_LEVERAGE,
            periods_per_year=PERIODS_PER_YEAR,
        )
        ann = annualize_total_return(
            float(sim.total_return),
            periods=float(steps),
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


def get_checkpoint_path(seed: int) -> Path | None:
    """Find the best.pt for a tp0.15 seed, checking both mass_daily and mass_daily_v2."""
    # New seeds go to mass_daily_v2
    v2 = MASS_DAILY_V2_BASE / f"tp0.15_s{seed}" / "best.pt"
    if v2.exists():
        return v2
    # Original seeds in mass_daily
    v1 = MASS_DAILY_BASE / f"tp0.15_s{seed}" / "best.pt"
    if v1.exists():
        return v1
    return None


def evaluate_all_seeds():
    """Evaluate all 15 tp=0.15 seeds and write leaderboard."""
    data = read_mktd(Path(VAL_DATA))
    print(f"\n{'='*60}")
    print(f"Evaluating ALL tp=0.15 seeds on crypto5 daily val ({data.num_timesteps} days)")
    print(f"Settings: {FILL_BUFFER_BPS}bps fill, {FEE_RATE*100:.1f}% fee, deterministic")
    print(f"{'='*60}\n")

    rows = []
    for seed in ALL_SEEDS:
        ckpt = get_checkpoint_path(seed)
        if ckpt is None:
            print(f"  Seed {seed}: NO CHECKPOINT FOUND, skipping")
            continue
        source = "mass_daily_v2" if MASS_DAILY_V2_BASE in ckpt.parents else "mass_daily"
        try:
            r = load_and_eval(ckpt, data, EVAL_PERIODS)
        except Exception as e:
            print(f"  Seed {seed}: ERROR {e}")
            continue

        r120 = r.get("120d", {})
        if "error" in r120:
            print(f"  Seed {seed}: 120d eval error: {r120['error']}")
            continue

        row = {
            "seed": seed,
            "source": source,
            "60d_ret_pct": r.get("60d", {}).get("return_pct", 0),
            "90d_ret_pct": r.get("90d", {}).get("return_pct", 0),
            "120d_ret_pct": r120.get("return_pct", 0),
            "120d_sortino": r120.get("sortino", 0),
            "120d_max_dd_pct": r120.get("max_dd_pct", 0),
            "120d_trades": r120.get("trades", 0),
            "120d_wr": r120.get("wr", 0),
            "180d_ret_pct": r.get("180d", {}).get("return_pct", 0),
        }
        rows.append(row)
        print(
            f"  Seed {seed:>5} ({source:<14}): "
            f"120d ret={r120.get('return_pct', 0):+.1f}% "
            f"sortino={r120.get('sortino', 0):.2f} "
            f"DD={r120.get('max_dd_pct', 0):.1f}% "
            f"trades={r120.get('trades', 0)} "
            f"WR={r120.get('wr', 0):.1%}"
        )

    # Sort by 120d sortino descending
    rows.sort(key=lambda x: x["120d_sortino"], reverse=True)

    # Write leaderboard CSV
    if rows:
        fieldnames = list(rows[0].keys())
        LEADERBOARD.parent.mkdir(parents=True, exist_ok=True)
        with open(LEADERBOARD, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nLeaderboard saved to {LEADERBOARD}")

    # Print summary table
    print(f"\n{'Seed':<8} {'Source':<14} {'60d%':>8} {'90d%':>8} {'120d%':>9} {'Sort':>8} {'DD%':>8} {'180d%':>9}")
    print("-" * 80)
    for row in rows:
        print(
            f"{row['seed']:<8} {row['source']:<14} "
            f"{row['60d_ret_pct']:>+7.1f}% "
            f"{row['90d_ret_pct']:>+7.1f}% "
            f"{row['120d_ret_pct']:>+8.1f}% "
            f"{row['120d_sortino']:>8.2f} "
            f"{row['120d_max_dd_pct']:>7.1f}% "
            f"{row['180d_ret_pct']:>+8.1f}%"
        )

    # Summary stats
    if rows:
        rets = [r["120d_ret_pct"] for r in rows]
        sortinos = [r["120d_sortino"] for r in rows]
        print(f"\nSummary (n={len(rows)}):")
        print(f"  120d return: mean={np.mean(rets):+.1f}%, median={np.median(rets):+.1f}%, std={np.std(rets):.1f}%")
        print(f"  120d sortino: mean={np.mean(sortinos):.2f}, median={np.median(sortinos):.2f}")
        profitable = sum(1 for r in rets if r > 0)
        print(f"  Profitable seeds: {profitable}/{len(rows)} ({profitable/len(rows)*100:.0f}%)")

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global TIME_BUDGET, NEW_SEEDS

    import argparse

    parser = argparse.ArgumentParser(description="Seed diversity sweep for tp=0.15")
    parser.add_argument("--train-only", action="store_true", help="Only train, skip eval")
    parser.add_argument("--eval-only", action="store_true", help="Only eval, skip training")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated list of seeds to train (default: all new seeds)")
    parser.add_argument("--time-budget", type=int, default=TIME_BUDGET,
                        help=f"Training time budget per seed in seconds (default: {TIME_BUDGET})")
    args = parser.parse_args()

    TIME_BUDGET = args.time_budget

    if args.seeds:
        NEW_SEEDS = [int(s.strip()) for s in args.seeds.split(",")]
        print(f"Override: training seeds = {NEW_SEEDS}")

    if not args.eval_only:
        train_all_new_seeds()

    if not args.train_only:
        evaluate_all_seeds()


if __name__ == "__main__":
    main()
