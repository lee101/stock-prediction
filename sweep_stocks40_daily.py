#!/usr/bin/env python3
"""Stocks40 daily RL sweep: leverage x trade_penalty x seeds.

Trains on stocks40 daily data with leverage (1x, 2x) x trade_penalty (0.05, 0.10, 0.15)
x seeds (42, 314) = 12 configs, then evaluates top results via C-env and market sim.

Usage:
    python -u sweep_stocks40_daily.py
    python -u sweep_stocks40_daily.py --time-budget 30 --max-configs 1  # smoke test
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent

TRAIN_DATA = "pufferlib_market/data/stocks40_daily_train.bin"
VAL_DATA = "pufferlib_market/data/stocks40_daily_val.bin"
CHECKPOINT_ROOT = "pufferlib_market/checkpoints/stocks40_sweep"
LEADERBOARD_CSV = "pufferlib_market/stocks40_sweep_leaderboard.csv"

PERIODS_PER_YEAR = 252.0
MAX_STEPS = 720
SHORT_BORROW_APR = 0.0625
FEE_RATE = 0.001
HIDDEN_SIZE = 1024
LR = 3e-4
ENT_COEF = 0.05
REWARD_SCALE = 10.0
REWARD_CLIP = 5.0
CASH_PENALTY = 0.01
GAMMA = 0.99
GAE_LAMBDA = 0.95
NUM_ENVS = 128
ROLLOUT_LEN = 256
PPO_EPOCHS = 4
FILL_SLIPPAGE_BPS_TRAIN = 5.0
FILL_SLIPPAGE_BPS_EVAL = 8.0


@dataclass
class SweepConfig:
    leverage: float
    trade_penalty: float
    seed: int

    @property
    def name(self) -> str:
        return f"lev{self.leverage:.0f}_tp{self.trade_penalty:.2f}_s{self.seed}"


def build_configs() -> list[SweepConfig]:
    """leverage (1.0, 2.0) x trade_penalty (0.05, 0.10, 0.15) x seeds (42, 314) = 12 runs."""
    configs = []
    for lev in [1.0, 2.0]:
        for tp in [0.05, 0.10, 0.15]:
            for seed in [42, 314]:
                configs.append(SweepConfig(leverage=lev, trade_penalty=tp, seed=seed))
    return configs


def train_one(config: SweepConfig, time_budget: int, checkpoint_dir: str) -> dict:
    """Train a single config via subprocess with timeout."""
    cmd = [
        sys.executable, "-u", "-m", "pufferlib_market.train",
        "--data-path", TRAIN_DATA,
        "--total-timesteps", "999999999",
        "--max-steps", str(MAX_STEPS),
        "--periods-per-year", str(PERIODS_PER_YEAR),
        "--hidden-size", str(HIDDEN_SIZE),
        "--lr", str(LR),
        "--ent-coef", str(ENT_COEF),
        "--gamma", str(GAMMA),
        "--gae-lambda", str(GAE_LAMBDA),
        "--num-envs", str(NUM_ENVS),
        "--rollout-len", str(ROLLOUT_LEN),
        "--ppo-epochs", str(PPO_EPOCHS),
        "--anneal-lr",
        "--disable-shorts",
        "--max-leverage", str(config.leverage),
        "--short-borrow-apr", str(SHORT_BORROW_APR),
        "--trade-penalty", str(config.trade_penalty),
        "--fee-rate", str(FEE_RATE),
        "--reward-scale", str(REWARD_SCALE),
        "--reward-clip", str(REWARD_CLIP),
        "--cash-penalty", str(CASH_PENALTY),
        "--fill-slippage-bps", str(FILL_SLIPPAGE_BPS_TRAIN),
        "--checkpoint-dir", checkpoint_dir,
        "--seed", str(config.seed),
    ]

    print(f"\n  Training {config.name} for {time_budget}s...")
    print(f"  cmd: {' '.join(cmd)}")
    t0 = time.time()
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=str(REPO), preexec_fn=os.setsid,
        )
        stdout_lines = []
        while time.time() - t0 < time_budget:
            if proc.poll() is not None:
                break
            try:
                line = proc.stdout.readline()
                if line:
                    decoded = line.decode("utf-8", errors="replace").strip()
                    stdout_lines.append(decoded)
            except Exception:
                pass
        if proc.poll() is None:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait(timeout=5)
        elapsed = time.time() - t0

        train_return = train_sortino = train_wr = None
        total_steps = 0
        for line in reversed(stdout_lines):
            if "ret=" in line and train_return is None:
                try:
                    for part in line.split():
                        if part.startswith("ret="):
                            train_return = float(part.split("=")[1])
                        elif part.startswith("sortino="):
                            train_sortino = float(part.split("=")[1])
                        elif part.startswith("wr="):
                            train_wr = float(part.split("=")[1])
                        elif part.startswith("step="):
                            total_steps = int(part.split("=")[1].replace(",", ""))
                except Exception:
                    pass
                if train_return is not None:
                    break
    except Exception as e:
        return {"error": str(e)}

    print(f"  Training done: {elapsed:.0f}s, {total_steps:,} steps, ret={train_return}")
    return {
        "train_return": train_return,
        "train_sortino": train_sortino,
        "train_wr": train_wr,
        "train_steps": total_steps,
        "elapsed_s": elapsed,
    }


def eval_c_env(config: SweepConfig, checkpoint_dir: str) -> dict:
    """Evaluate using C-env evaluate module (100 episodes, deterministic)."""
    ckpt_path = _find_checkpoint(checkpoint_dir)
    if ckpt_path is None:
        return {"error": "no checkpoint"}

    cmd = [
        sys.executable, "-u", "-m", "pufferlib_market.evaluate",
        "--checkpoint", str(ckpt_path),
        "--data-path", VAL_DATA,
        "--deterministic",
        "--hidden-size", str(HIDDEN_SIZE),
        "--max-steps", str(MAX_STEPS),
        "--num-episodes", "100",
        "--seed", "42",
        "--fill-slippage-bps", str(FILL_SLIPPAGE_BPS_EVAL),
        "--periods-per-year", str(PERIODS_PER_YEAR),
        "--max-leverage", str(config.leverage),
        "--short-borrow-apr", str(SHORT_BORROW_APR),
        "--fee-rate", str(FEE_RATE),
        "--disable-shorts",
    ]

    print(f"  Evaluating {config.name} (C-env, 100 eps)...")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300, cwd=str(REPO),
        )
        output = result.stdout + result.stderr
        val_return = val_wr = val_sortino = val_profitable_pct = None
        for line in output.split("\n"):
            if "Return:" in line and "mean=" in line:
                try:
                    val_return = float(line.split("mean=")[1].split()[0])
                except Exception:
                    pass
            if "Win rate:" in line and "mean=" in line:
                try:
                    val_wr = float(line.split("mean=")[1].split()[0])
                except Exception:
                    pass
            if "Sortino:" in line and "mean=" in line:
                try:
                    val_sortino = float(line.split("mean=")[1].split()[0])
                except Exception:
                    pass
            if ">0:" in line:
                try:
                    pct_str = line.split("(")[1].split("%")[0]
                    val_profitable_pct = float(pct_str)
                except Exception:
                    pass

        print(f"  Val: ret={val_return}, sortino={val_sortino}, "
              f"wr={val_wr}, profitable={val_profitable_pct}%")

        return {
            "val_return": val_return,
            "val_sortino": val_sortino,
            "val_wr": val_wr,
            "val_profitable_pct": val_profitable_pct,
        }
    except subprocess.TimeoutExpired:
        return {"error": "eval timeout"}
    except Exception as e:
        return {"error": f"eval error: {e}"}


def _find_checkpoint(checkpoint_dir: str) -> Path | None:
    """Find the best available checkpoint in a directory."""
    d = Path(checkpoint_dir)
    for name in ("best.pt", "final.pt"):
        p = d / name
        if p.exists():
            return p
    pts = list(d.glob("*.pt"))
    if pts:
        return max(pts, key=lambda p: p.stat().st_mtime)
    return None


def run_market_sim(config: SweepConfig, checkpoint_dir: str, periods: dict[str, int]) -> dict:
    """Run market sim evaluation for given periods using pure-python simulator."""
    ckpt_path = _find_checkpoint(checkpoint_dir)
    if ckpt_path is None:
        return {"error": "no checkpoint"}

    # Disable early exit
    import src.market_sim_early_exit as _mse
    _orig = _mse.evaluate_drawdown_vs_profit_early_exit
    def _no_early_exit(*args, **kwargs):
        return _mse.EarlyExitDecision(should_stop=False, progress_fraction=0.0,
                                       total_return=0.0, max_drawdown=0.0)
    _mse.evaluate_drawdown_vs_profit_early_exit = _no_early_exit

    try:
        import numpy as np
        import torch
        from pufferlib_market.hourly_replay import read_mktd, simulate_daily_policy
        from pufferlib_market.metrics import annualize_total_return
        from pufferlib_market.evaluate_tail import (
            TradingPolicy, _infer_num_actions, _infer_arch, _infer_hidden_size, _slice_tail,
        )

        data = read_mktd(Path(VAL_DATA))
        device = torch.device("cpu")
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
        for pname, steps in periods.items():
            if data.num_timesteps < steps + 1:
                results[pname] = {"error": "data too short"}
                continue
            tail = _slice_tail(data, steps=steps)
            sim = simulate_daily_policy(
                tail, policy_fn, max_steps=steps,
                fee_rate=FEE_RATE, fill_buffer_bps=FILL_SLIPPAGE_BPS_EVAL,
                max_leverage=config.leverage, periods_per_year=PERIODS_PER_YEAR,
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
            }
        return results
    except Exception as e:
        return {"error": str(e)}
    finally:
        _mse.evaluate_drawdown_vs_profit_early_exit = _orig


def main():
    parser = argparse.ArgumentParser(description="Stocks40 daily RL sweep: leverage x trade_penalty x seeds")
    parser.add_argument("--time-budget", type=int, default=300,
                        help="Training time per config in seconds (default: 300)")
    parser.add_argument("--max-configs", type=int, default=0,
                        help="Max configs to run (0=all)")
    parser.add_argument("--skip-market-sim", action="store_true",
                        help="Skip market sim evaluation")
    parser.add_argument("--market-sim-top-n", type=int, default=5,
                        help="Run market sim on top N results")
    args = parser.parse_args()

    # Check data files exist
    train_path = REPO / TRAIN_DATA
    val_path = REPO / VAL_DATA
    if not train_path.exists():
        print(f"ERROR: Training data not found: {train_path}")
        print("Run export_stocks40_daily.sh first to generate the data.")
        sys.exit(1)
    if not val_path.exists():
        print(f"ERROR: Validation data not found: {val_path}")
        print("Run export_stocks40_daily.sh first to generate the data.")
        sys.exit(1)

    configs = build_configs()
    if args.max_configs > 0:
        configs = configs[:args.max_configs]

    print(f"Stocks40 daily sweep: {len(configs)} configs")
    print(f"  Train data: {train_path}")
    print(f"  Val data:   {val_path}")
    print(f"  Time budget: {args.time_budget}s per config")
    print(f"  Leverages: 1x, 2x | Trade penalties: 0.05, 0.10, 0.15")
    print(f"  Seeds: 42, 314 | periods_per_year={PERIODS_PER_YEAR}")
    print(f"  Long-only (--disable-shorts), fee={FEE_RATE}, borrow_apr={SHORT_BORROW_APR}")
    print(f"  reward_scale={REWARD_SCALE}, reward_clip={REWARD_CLIP}, cash_penalty={CASH_PENALTY}")
    print(f"  gamma={GAMMA}, gae_lambda={GAE_LAMBDA}, num_envs={NUM_ENVS}")

    ckpt_root = REPO / CHECKPOINT_ROOT
    ckpt_root.mkdir(parents=True, exist_ok=True)
    leaderboard_path = REPO / LEADERBOARD_CSV

    fieldnames = [
        "name", "leverage", "trade_penalty", "seed",
        "val_return", "val_sortino", "val_wr", "val_profitable_pct",
        "train_return", "train_sortino", "train_wr", "train_steps",
        "elapsed_s", "error",
    ]

    # Load already-completed configs and their val_return values
    existing: dict[str, float | None] = {}
    if leaderboard_path.exists():
        with open(leaderboard_path) as f:
            for row in csv.DictReader(f):
                name = row.get("name", "")
                vr = row.get("val_return")
                existing[name] = float(vr) if vr and vr != "None" else None

    best_val = -float("inf")
    results_for_sim = []

    for i, config in enumerate(configs):
        if config.name in existing:
            print(f"\n[{i+1}/{len(configs)}] SKIP {config.name} (already done)")
            vr = existing[config.name]
            if vr is not None:
                results_for_sim.append((config, vr))
            continue

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(configs)}] {config.name}")
        print(f"  leverage={config.leverage}, trade_penalty={config.trade_penalty}, seed={config.seed}")
        print(f"{'='*60}")

        ckpt_dir = str(ckpt_root / config.name)
        os.makedirs(ckpt_dir, exist_ok=True)

        # Train
        train_result = train_one(config, args.time_budget, ckpt_dir)
        if "error" in train_result:
            print(f"  TRAIN ERROR: {train_result['error']}")

        # Evaluate (C-env)
        eval_result = eval_c_env(config, ckpt_dir)

        # Write to leaderboard
        row = {
            "name": config.name,
            "leverage": config.leverage,
            "trade_penalty": config.trade_penalty,
            "seed": config.seed,
            "val_return": eval_result.get("val_return"),
            "val_sortino": eval_result.get("val_sortino"),
            "val_wr": eval_result.get("val_wr"),
            "val_profitable_pct": eval_result.get("val_profitable_pct"),
            "train_return": train_result.get("train_return"),
            "train_sortino": train_result.get("train_sortino"),
            "train_wr": train_result.get("train_wr"),
            "train_steps": train_result.get("train_steps"),
            "elapsed_s": train_result.get("elapsed_s"),
            "error": eval_result.get("error", train_result.get("error", "")),
        }

        write_header = not leaderboard_path.exists()
        with open(leaderboard_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        val_ret = eval_result.get("val_return")
        if val_ret is not None:
            results_for_sim.append((config, val_ret))
            if val_ret > best_val:
                best_val = val_ret
                print(f"  *** NEW BEST val_return={val_ret:.4f} ***")

    # Print leaderboard
    print(f"\n{'='*60}")
    print("LEADERBOARD (sorted by val_return)")
    print(f"{'='*60}")
    if leaderboard_path.exists():
        with open(leaderboard_path) as f:
            rows = list(csv.DictReader(f))
        rows = [r for r in rows if r.get("val_return") and r["val_return"] != "None"]
        rows.sort(key=lambda r: float(r["val_return"]), reverse=True)
        for r in rows:
            print(f"  {r['name']:30s} val_ret={float(r['val_return']):+.4f} "
                  f"sortino={r.get('val_sortino', 'N/A'):>8s} "
                  f"lev={r['leverage']:>4s} tp={r['trade_penalty']:>5s} s={r['seed']:>4s}")

    # Market sim evaluation on top N
    if not args.skip_market_sim and results_for_sim:
        results_for_sim.sort(key=lambda x: x[1], reverse=True)
        top_n = results_for_sim[:args.market_sim_top_n]

        print(f"\n{'='*60}")
        print(f"MARKET SIM EVALUATION (top {len(top_n)} configs)")
        print(f"{'='*60}")

        sim_periods = {"60d": 60, "90d": 90, "120d": 120}
        sim_results = []

        for config, val_ret in top_n:
            ckpt_dir = str(ckpt_root / config.name)
            print(f"\n  Market sim: {config.name} (val_ret={val_ret:+.4f})")
            sim = run_market_sim(config, ckpt_dir, sim_periods)
            if "error" in sim:
                print(f"    ERROR: {sim['error']}")
            else:
                sim_results.append((config.name, sim))
                for pname, metrics in sim.items():
                    if isinstance(metrics, dict) and "error" not in metrics:
                        print(f"    {pname}: ret={metrics['return_pct']:+.1f}%, "
                              f"ann={metrics['annualized_pct']:+.1f}%, "
                              f"sortino={metrics['sortino']:.2f}, "
                              f"dd={metrics['max_dd_pct']:.1f}%")

        # Summary table
        if sim_results:
            print(f"\n{'Name':<25} {'60d Ret%':>9} {'90d Ret%':>9} {'120d Ret%':>10} {'120d Sort':>10} {'120d DD%':>9}")
            print("-" * 80)
            for name, r in sim_results:
                r60 = r.get("60d", {})
                r90 = r.get("90d", {})
                r120 = r.get("120d", {})
                print(f"{name:<25} "
                      f"{r60.get('return_pct', 0):>+8.1f}% "
                      f"{r90.get('return_pct', 0):>+8.1f}% "
                      f"{r120.get('return_pct', 0):>+9.1f}% "
                      f"{r120.get('sortino', 0):>10.2f} "
                      f"{r120.get('max_dd_pct', 0):>8.1f}%")

            # Save market sim results
            sim_json_path = REPO / "pufferlib_market" / "stocks40_sweep_market_sim.json"
            with open(sim_json_path, "w") as f:
                json.dump({n: r for n, r in sim_results}, f, indent=2)
            print(f"\nMarket sim results saved to {sim_json_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
