#!/usr/bin/env python3
"""Sweep weight_decay x fill_slippage_bps neighborhood around crypto10/reg_combo_2.

The crypto10 autoresearch found reg_combo_2 as the clear winner (+27.8% val return,
holdout_robust_score=33.77). Its config:
  weight_decay=0.05, fill_slippage_bps=8.0, obs_norm=True
  everything else default (lr=3e-4, ent_coef=0.05, h1024, anneal_lr=True)

This script explores the neighborhood:
  - wd = {0.01, 0.05, 0.10}
  - slip = {0, 5, 8, 12} bps
  - seeds = {42, 314}
  - obs_norm = True (fixed, critical for this config)

Total: 3 x 4 x 2 = 24 configs, 300s budget each.

After training, evaluates on crypto10_daily_val.bin using the market simulator
with an 80-step window (val has only 87 timesteps).

Usage:
    source .venv313/bin/activate
    python -u sweep_crypto10_reg_combo.py

    # Smoke test (30s budget, first 2 configs):
    python -u sweep_crypto10_reg_combo.py --time-budget 30 --only-first 2
"""

from __future__ import annotations

import argparse
import csv
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# Disable early exit before importing simulate_daily_policy
import src.market_sim_early_exit as _mse


def _no_early_exit(*args, **kwargs):
    return _mse.EarlyExitDecision(
        should_stop=False, progress_fraction=0.0,
        total_return=0.0, max_drawdown=0.0,
    )


_mse.evaluate_drawdown_vs_profit_early_exit = _no_early_exit

from pufferlib_market.hourly_replay import MktdData, read_mktd, simulate_daily_policy
from pufferlib_market.evaluate_tail import (
    TradingPolicy, _infer_num_actions, _infer_arch, _infer_hidden_size,
    _infer_resmlp_blocks, ResidualTradingPolicy, _slice_tail,
)

REPO = Path(__file__).resolve().parent


@dataclass
class SweepConfig:
    description: str
    weight_decay: float = 0.05
    fill_slippage_bps: float = 8.0
    seed: int = 42
    hidden_size: int = 1024
    lr: float = 3e-4
    ent_coef: float = 0.05
    fee_rate: float = 0.001
    reward_scale: float = 10.0
    reward_clip: float = 5.0
    cash_penalty: float = 0.01
    max_steps: int = 720
    periods_per_year: float = 365.0


WD_VALUES = [0.01, 0.05, 0.10]
SLIP_VALUES = [0.0, 5.0, 8.0, 12.0]
SEEDS = [42, 314]

EXPERIMENTS: list[SweepConfig] = [
    SweepConfig(
        description=f"wd{wd:.2f}_slip{int(slip)}_s{seed}",
        weight_decay=wd,
        fill_slippage_bps=slip,
        seed=seed,
    )
    for wd in WD_VALUES
    for slip in SLIP_VALUES
    for seed in SEEDS
]

# Val data has 87 timesteps; use 80 for eval to be safe.
EVAL_MAX_STEPS = 80
EVAL_PERIODS = {"80d": EVAL_MAX_STEPS}

FIELDNAMES = [
    "description", "weight_decay", "fill_slippage_bps", "seed",
    "train_return", "train_sortino", "train_wr", "train_steps", "elapsed_s",
    "80d_ret_pct", "80d_sortino", "80d_dd_pct", "80d_trades", "80d_wr",
    "error",
]


def build_train_cmd(config: SweepConfig, train_data: str, checkpoint_dir: str) -> list[str]:
    return [
        sys.executable, "-u", "-m", "pufferlib_market.train",
        "--data-path", train_data,
        "--total-timesteps", "999999999",
        "--max-steps", str(config.max_steps),
        "--hidden-size", str(config.hidden_size),
        "--lr", str(config.lr),
        "--ent-coef", str(config.ent_coef),
        "--gamma", "0.99",
        "--gae-lambda", "0.95",
        "--num-envs", "128",
        "--rollout-len", "256",
        "--ppo-epochs", "4",
        "--seed", str(config.seed),
        "--reward-scale", str(config.reward_scale),
        "--reward-clip", str(config.reward_clip),
        "--cash-penalty", str(config.cash_penalty),
        "--fee-rate", str(config.fee_rate),
        "--weight-decay", str(config.weight_decay),
        "--fill-slippage-bps", str(config.fill_slippage_bps),
        "--obs-norm",
        "--anneal-lr",
        "--checkpoint-dir", checkpoint_dir,
        "--periods-per-year", str(config.periods_per_year),
    ]


def parse_train_output(stdout_lines: list[str]) -> dict:
    """Extract final training stats from stdout."""
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
    return {
        "train_return": train_return,
        "train_sortino": train_sortino,
        "train_wr": train_wr,
        "train_steps": total_steps,
    }


def find_checkpoint(checkpoint_dir: str) -> str | None:
    """Find best checkpoint in directory."""
    for name in ("best.pt", "final.pt"):
        p = Path(checkpoint_dir) / name
        if p.exists():
            return str(p)
    pts = list(Path(checkpoint_dir).glob("*.pt"))
    if pts:
        return str(max(pts, key=lambda p: p.stat().st_mtime))
    return None


def run_training(config: SweepConfig, train_data: str, checkpoint_dir: str,
                 time_budget: int) -> dict:
    """Run training with time budget, return training stats."""
    cmd = build_train_cmd(config, train_data, checkpoint_dir)
    print(f"  Training for {time_budget}s... (wd={config.weight_decay}, "
          f"slip={config.fill_slippage_bps}, seed={config.seed})")
    t0 = time.time()
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=str(REPO), preexec_fn=os.setsid,
        )
        stdout_lines: list[str] = []
        try:
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
                proc.wait(timeout=10)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
        elapsed = time.time() - t0
        stats = parse_train_output(stdout_lines)
        stats["elapsed_s"] = elapsed
        print(f"  Training done: {elapsed:.0f}s, {stats['train_steps']:,} steps, "
              f"ret={stats['train_return']}, sortino={stats['train_sortino']}")
        return stats
    except Exception as e:
        return {"error": str(e)}


def load_and_eval(ckpt_path: str, data: MktdData, periods_dict: dict,
                  device: str = "cpu") -> dict:
    """Load checkpoint and evaluate on market sim."""
    device_t = torch.device(device)
    nsym = data.num_symbols
    payload = torch.load(str(ckpt_path), map_location=device_t, weights_only=False)
    sd = payload.get("model", payload) if isinstance(payload, dict) else payload
    obs_size = nsym * 16 + 5 + nsym
    na = _infer_num_actions(sd, fallback=1 + 2 * nsym)
    arch = _infer_arch(sd)
    h = _infer_hidden_size(sd, arch=arch)

    if arch == "resmlp":
        num_blocks = _infer_resmlp_blocks(sd)
        policy = ResidualTradingPolicy(obs_size, na, hidden=h, num_blocks=num_blocks).to(device_t)
    else:
        policy = TradingPolicy(obs_size, na, hidden=h).to(device_t)
    policy.load_state_dict(sd)
    policy.eval()

    def policy_fn(obs):
        t = torch.from_numpy(obs.astype(np.float32)).to(device_t).view(1, -1)
        with torch.no_grad():
            logits, _ = policy(t)
        return int(torch.argmax(logits, dim=-1).item())

    results = {}
    for pname, steps in periods_dict.items():
        if data.num_timesteps < steps + 1:
            results[pname] = {"error": f"data too short ({data.num_timesteps} < {steps + 1})"}
            continue
        tail = _slice_tail(data, steps=steps)
        sim = simulate_daily_policy(
            tail, policy_fn, max_steps=steps,
            fee_rate=0.001, fill_buffer_bps=8.0,
            max_leverage=1.0, periods_per_year=365.0,
        )
        results[pname] = {
            "return_pct": sim.total_return * 100,
            "sortino": sim.sortino,
            "max_dd_pct": sim.max_drawdown * 100,
            "trades": sim.num_trades,
            "wr": sim.win_rate,
        }
    return results


def _write_row(leaderboard_path: Path, fieldnames: list[str], row: dict) -> None:
    """Append a row to the CSV leaderboard, writing header if needed."""
    write_header = not leaderboard_path.exists()
    with open(leaderboard_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _fmt(val) -> str:
    if val is None or val == "" or val == "None":
        return "N/A"
    try:
        return f"{float(val):+.1f}"
    except (ValueError, TypeError):
        return str(val)


def main():
    parser = argparse.ArgumentParser(
        description="Sweep wd x slip around crypto10/reg_combo_2 (obs_norm=True fixed)")
    parser.add_argument("--train-data",
                        default="pufferlib_market/data/crypto10_daily_train.bin")
    parser.add_argument("--val-data",
                        default="pufferlib_market/data/crypto10_daily_val.bin")
    parser.add_argument("--time-budget", type=int, default=300,
                        help="Training time budget per trial in seconds")
    parser.add_argument("--checkpoint-root",
                        default="pufferlib_market/checkpoints/crypto10_reg")
    parser.add_argument("--leaderboard",
                        default="pufferlib_market/crypto10_reg_sweep_leaderboard.csv")
    parser.add_argument("--only-first", type=int, default=0,
                        help="Only run the first N experiments (0 = all)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ckpt_root = Path(args.checkpoint_root)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    leaderboard_path = Path(args.leaderboard)

    # Load val data once for market sim eval
    val_data_path = Path(args.val_data)
    if not val_data_path.exists():
        print(f"ERROR: val data not found: {val_data_path}")
        sys.exit(1)
    val_data = read_mktd(val_data_path)
    print(f"Loaded val data: {val_data.num_symbols} symbols, "
          f"{val_data.num_timesteps} timesteps")

    # Verify train data exists
    train_data_path = Path(args.train_data)
    if not train_data_path.exists():
        print(f"ERROR: train data not found: {train_data_path}")
        sys.exit(1)

    # Load existing results to skip completed
    existing = set()
    if leaderboard_path.exists():
        with open(leaderboard_path) as f:
            for row in csv.DictReader(f):
                existing.add(row.get("description", ""))

    experiments = EXPERIMENTS
    if args.only_first > 0:
        experiments = experiments[:args.only_first]

    best_sortino = -float("inf")

    print("=" * 80)
    print(f"SWEEP: wd x slip neighborhood around crypto10/reg_combo_2")
    print(f"  {len(experiments)} configs, {args.time_budget}s budget each")
    print(f"  Train data: {args.train_data}")
    print(f"  Val data: {args.val_data} ({val_data.num_timesteps} timesteps)")
    print(f"  Checkpoints: {args.checkpoint_root}")
    print(f"  Leaderboard: {args.leaderboard}")
    print(f"  Eval: {EVAL_MAX_STEPS}-step market sim, 8bps fill, 0.001 fee")
    print(f"  Fixed: obs_norm=True, anneal_lr=True, h1024, lr=3e-4, ent=0.05")
    print(f"  Estimated total: ~{len(experiments) * args.time_budget / 60:.0f} min")
    print("=" * 80)

    for i, config in enumerate(experiments):
        if config.description in existing:
            print(f"\n[{i+1}/{len(experiments)}] SKIP {config.description} (already done)")
            continue

        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(experiments)}] {config.description}")
        print(f"  wd={config.weight_decay}, slip={config.fill_slippage_bps}, "
              f"seed={config.seed}")
        print(f"{'='*70}")

        ckpt_dir = str(ckpt_root / config.description)
        os.makedirs(ckpt_dir, exist_ok=True)

        # --- Train ---
        train_result = run_training(config, args.train_data, ckpt_dir,
                                    args.time_budget)
        if "error" in train_result and "train_return" not in train_result:
            row = {
                "description": config.description,
                "weight_decay": config.weight_decay,
                "fill_slippage_bps": config.fill_slippage_bps,
                "seed": config.seed,
                "error": train_result["error"],
            }
            _write_row(leaderboard_path, FIELDNAMES, row)
            continue

        # --- Find checkpoint ---
        ckpt = find_checkpoint(ckpt_dir)
        if ckpt is None:
            row = {
                "description": config.description,
                "weight_decay": config.weight_decay,
                "fill_slippage_bps": config.fill_slippage_bps,
                "seed": config.seed,
                "error": "no checkpoint",
                **{k: train_result.get(k) for k in
                   ["train_return", "train_sortino", "train_wr",
                    "train_steps", "elapsed_s"]},
            }
            _write_row(leaderboard_path, FIELDNAMES, row)
            continue

        # --- Market sim eval ---
        print(f"  Evaluating {ckpt} on market sim ({EVAL_MAX_STEPS}d)...")
        try:
            eval_results = load_and_eval(ckpt, val_data, EVAL_PERIODS,
                                         device=args.device)
        except Exception as e:
            print(f"  Eval ERROR: {e}")
            row = {
                "description": config.description,
                "weight_decay": config.weight_decay,
                "fill_slippage_bps": config.fill_slippage_bps,
                "seed": config.seed,
                "error": f"eval error: {e}",
                **{k: train_result.get(k) for k in
                   ["train_return", "train_sortino", "train_wr",
                    "train_steps", "elapsed_s"]},
            }
            _write_row(leaderboard_path, FIELDNAMES, row)
            continue

        # Build leaderboard row
        row = {
            "description": config.description,
            "weight_decay": config.weight_decay,
            "fill_slippage_bps": config.fill_slippage_bps,
            "seed": config.seed,
            **{k: train_result.get(k) for k in
               ["train_return", "train_sortino", "train_wr",
                "train_steps", "elapsed_s"]},
            "error": "",
        }
        for period_name in EVAL_PERIODS:
            pr = eval_results.get(period_name, {})
            if "error" in pr:
                row[f"{period_name}_ret_pct"] = None
                row[f"{period_name}_sortino"] = None
                row[f"{period_name}_dd_pct"] = None
                row[f"{period_name}_trades"] = None
                row[f"{period_name}_wr"] = None
            else:
                row[f"{period_name}_ret_pct"] = pr.get("return_pct")
                row[f"{period_name}_sortino"] = pr.get("sortino")
                row[f"{period_name}_dd_pct"] = pr.get("max_dd_pct")
                row[f"{period_name}_trades"] = pr.get("trades")
                row[f"{period_name}_wr"] = pr.get("wr")

        _write_row(leaderboard_path, FIELDNAMES, row)

        # Print summary
        r80 = eval_results.get("80d", {})
        sortino_80 = r80.get("sortino", -999)
        if "error" not in r80:
            print(f"  80d: ret={r80.get('return_pct', 0):+.1f}%, "
                  f"sortino={r80.get('sortino', 0):.2f}, "
                  f"dd={r80.get('max_dd_pct', 0):.1f}%, "
                  f"trades={r80.get('trades', 0)}, "
                  f"wr={r80.get('wr', 0):.1%}")

        if sortino_80 > best_sortino:
            best_sortino = sortino_80
            print(f"  *** NEW BEST 80d Sortino: {sortino_80:.2f} ***")

    # --- Print final leaderboard sorted by 80d Sortino ---
    print(f"\n{'='*80}")
    print("LEADERBOARD (sorted by 80d Sortino)")
    print(f"{'='*80}")
    all_rows: list[dict] = []
    if leaderboard_path.exists():
        with open(leaderboard_path) as f:
            all_rows = list(csv.DictReader(f))
        valid_rows = []
        for r in all_rows:
            s = r.get("80d_sortino", "")
            if s and s != "None":
                try:
                    valid_rows.append((float(s), r))
                except ValueError:
                    pass
        valid_rows.sort(key=lambda x: x[0], reverse=True)
        header = (f"  {'Description':28s} {'WD':>5s} {'Slip':>5s} {'Seed':>5s} "
                  f"{'80d%':>8s} {'Sort':>7s} {'DD%':>7s} "
                  f"{'Trades':>7s} {'WR':>7s}")
        print(header)
        print("  " + "-" * (len(header) - 2))
        for _, r in valid_rows:
            print(f"  {r['description']:28s} "
                  f"{r.get('weight_decay', ''):>5s} "
                  f"{r.get('fill_slippage_bps', ''):>5s} "
                  f"{r.get('seed', ''):>5s} "
                  f"{_fmt(r.get('80d_ret_pct')):>8s} "
                  f"{_fmt(r.get('80d_sortino')):>7s} "
                  f"{_fmt(r.get('80d_dd_pct')):>7s} "
                  f"{_fmt(r.get('80d_trades')):>7s} "
                  f"{_fmt(r.get('80d_wr')):>7s}")

    # --- Per-wd summary ---
    print(f"\n{'='*80}")
    print("PER-WD SUMMARY (mean across slip x seed)")
    print(f"{'='*80}")
    if all_rows:
        wd_groups: dict[str, list[dict]] = {}
        for r in all_rows:
            wd = r.get("weight_decay", "?")
            wd_groups.setdefault(wd, []).append(r)

        print(f"  {'WD':>6s} {'N':>4s} {'MeanRet%':>9s} {'MeanSort':>9s} {'MeanDD%':>9s}")
        print("  " + "-" * 45)
        summaries = []
        for wd_str, group in sorted(wd_groups.items()):
            rets, sorts, dds = [], [], []
            for r in group:
                try:
                    rets.append(float(r.get("80d_ret_pct", "nan")))
                except (ValueError, TypeError):
                    pass
                try:
                    sorts.append(float(r.get("80d_sortino", "nan")))
                except (ValueError, TypeError):
                    pass
                try:
                    dds.append(float(r.get("80d_dd_pct", "nan")))
                except (ValueError, TypeError):
                    pass
            mean_ret = np.mean(rets) if rets else float("nan")
            mean_sort = np.mean(sorts) if sorts else float("nan")
            mean_dd = np.mean(dds) if dds else float("nan")
            summaries.append((mean_sort, wd_str, len(group), mean_ret, mean_sort, mean_dd))
        summaries.sort(key=lambda x: x[0] if not np.isnan(x[0]) else float("-inf"),
                       reverse=True)
        for _, wd_str, n, mean_ret, mean_sort, mean_dd in summaries:
            print(f"  {wd_str:>6s} {n:>4d} {_fmt(mean_ret):>9s} "
                  f"{_fmt(mean_sort):>9s} {_fmt(mean_dd):>9s}")

    print(f"\nLeaderboard saved to: {leaderboard_path}")


if __name__ == "__main__":
    main()
