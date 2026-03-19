"""
Sweep mixed23 stock+crypto models with proper annualization (periods_per_year=252)
and optional leverage.

mixed23 has 23 symbols (15 stocks + 8 crypto). Since the C env uses one global
periods_per_year value and stocks dominate, we use 252 (stock trading days/year).

This compares against existing trade_pen_05 from autoresearch_mixed23_daily
which was trained with periods_per_year=365.

Sweep configs:
  - 4 seeds (42, 314, 123, 7) at 1x leverage, tp=0.15
  - 2 seeds (42, 314) at 2x leverage, tp=0.15
  - Comparison eval of existing tp=0.15 checkpoints from autoresearch_mixed23_daily

Usage:
    source .venv313/bin/activate
    python -u sweep_mixed_proper_annual.py

    # Smoke test (30s budget per trial):
    python -u sweep_mixed_proper_annual.py --time-budget 30 --only-first 1
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

REPO = Path(__file__).resolve().parent


@dataclass
class SweepConfig:
    description: str
    seed: int = 42
    max_leverage: float = 1.0
    short_borrow_apr: float = 0.0
    trade_penalty: float = 0.15
    periods_per_year: float = 252.0
    disable_shorts: bool = True
    hidden_size: int = 1024
    lr: float = 3e-4
    ent_coef: float = 0.05
    fee_rate: float = 0.001
    reward_scale: float = 10.0
    reward_clip: float = 5.0
    cash_penalty: float = 0.01
    max_steps: int = 720


# Sweep experiments
EXPERIMENTS: list[SweepConfig] = [
    # --- 1x leverage, 4 seeds ---
    SweepConfig(description="lev1x_tp015_s42", seed=42),
    SweepConfig(description="lev1x_tp015_s314", seed=314),
    SweepConfig(description="lev1x_tp015_s123", seed=123),
    SweepConfig(description="lev1x_tp015_s7", seed=7),
    # --- 2x leverage, 2 seeds ---
    SweepConfig(
        description="lev2x_tp015_s42", seed=42,
        max_leverage=2.0, short_borrow_apr=0.0625,
    ),
    SweepConfig(
        description="lev2x_tp015_s314", seed=314,
        max_leverage=2.0, short_borrow_apr=0.0625,
    ),
]

# Existing autoresearch_mixed23_daily checkpoints to compare against
COMPARISON_CKPTS: list[dict] = [
    {
        "description": "existing_tp005_ppy365",
        "checkpoint": "pufferlib_market/checkpoints/autoresearch_mixed23_daily/trade_pen_05/best.pt",
        "hidden_size": 1024,
        "max_steps": 720,
        "trade_penalty": 0.05,
        "note": "trained with periods_per_year=365",
    },
]

def build_train_cmd(config: SweepConfig, train_data: str, checkpoint_dir: str) -> list[str]:
    cmd = [
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
        "--trade-penalty", str(config.trade_penalty),
        "--max-leverage", str(config.max_leverage),
        "--short-borrow-apr", str(config.short_borrow_apr),
        "--checkpoint-dir", checkpoint_dir,
        "--periods-per-year", str(config.periods_per_year),
        "--anneal-lr",
    ]
    if config.disable_shorts:
        cmd.append("--disable-shorts")
    return cmd


def build_eval_cmd(
    checkpoint: str,
    val_data: str,
    hidden_size: int,
    max_steps: int,
    periods_per_year: float,
) -> list[str]:
    return [
        sys.executable, "-u", "-m", "pufferlib_market.evaluate",
        "--checkpoint", checkpoint,
        "--data-path", val_data,
        "--deterministic",
        "--hidden-size", str(hidden_size),
        "--max-steps", str(max_steps),
        "--num-episodes", "100",
        "--seed", "42",
        "--fill-slippage-bps", "8",
        "--periods-per-year", str(periods_per_year),
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


def parse_eval_output(output: str) -> dict:
    """Extract evaluation metrics from evaluator output."""
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
    return {
        "val_return": val_return,
        "val_sortino": val_sortino,
        "val_wr": val_wr,
        "val_profitable_pct": val_profitable_pct,
    }


def find_checkpoint(checkpoint_dir: str) -> str | None:
    """Find best checkpoint in directory."""
    ckpt_path = Path(checkpoint_dir) / "best.pt"
    if ckpt_path.exists():
        return str(ckpt_path)
    ckpt_path = Path(checkpoint_dir) / "final.pt"
    if ckpt_path.exists():
        return str(ckpt_path)
    pts = list(Path(checkpoint_dir).glob("*.pt"))
    if pts:
        return str(max(pts, key=lambda p: p.stat().st_mtime))
    return None


def run_training(config: SweepConfig, train_data: str, checkpoint_dir: str,
                 time_budget: int) -> dict:
    """Run training with time budget, return training stats."""
    cmd = build_train_cmd(config, train_data, checkpoint_dir)
    print(f"  Training for {time_budget}s...")
    t0 = time.time()
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=str(REPO), preexec_fn=os.setsid,
        )
        stdout_lines = []
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


def run_eval(checkpoint: str, val_data: str, hidden_size: int, max_steps: int,
             periods_per_year: float) -> dict:
    """Run evaluation and return metrics."""
    cmd = build_eval_cmd(checkpoint, val_data, hidden_size, max_steps, periods_per_year)
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=180, cwd=str(REPO),
        )
        output = result.stdout + result.stderr
        return parse_eval_output(output)
    except subprocess.TimeoutExpired:
        return {"error": "eval timeout"}
    except Exception as e:
        return {"error": f"eval error: {e}"}


def main():
    parser = argparse.ArgumentParser(
        description="Sweep mixed23 with proper annualization (252 days/year)")
    parser.add_argument("--train-data",
                        default="pufferlib_market/data/mixed23_daily_train.bin")
    parser.add_argument("--val-data",
                        default="pufferlib_market/data/mixed23_daily_val.bin")
    parser.add_argument("--time-budget", type=int, default=300,
                        help="Training time budget per trial in seconds")
    parser.add_argument("--checkpoint-root",
                        default="pufferlib_market/checkpoints/mixed23_252")
    parser.add_argument("--leaderboard",
                        default="pufferlib_market/sweep_mixed23_252_leaderboard.csv")
    parser.add_argument("--only-first", type=int, default=0,
                        help="Only run the first N experiments (0 = all)")
    args = parser.parse_args()

    ckpt_root = Path(args.checkpoint_root)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    leaderboard_path = Path(args.leaderboard)

    fieldnames = [
        "description", "seed", "max_leverage", "periods_per_year",
        "trade_penalty", "short_borrow_apr",
        "train_return", "train_sortino", "train_wr", "train_steps", "elapsed_s",
        "val_return", "val_sortino", "val_wr", "val_profitable_pct",
        "error",
    ]

    # Load existing results to skip
    existing = set()
    if leaderboard_path.exists():
        with open(leaderboard_path) as f:
            for row in csv.DictReader(f):
                existing.add(row.get("description", ""))

    experiments = EXPERIMENTS
    if args.only_first > 0:
        experiments = experiments[:args.only_first]

    best_val = -float("inf")

    # --- Phase 1: Train new models ---
    print("=" * 70)
    print("PHASE 1: Training new models with periods_per_year=252")
    print("=" * 70)

    for i, config in enumerate(experiments):
        if config.description in existing:
            print(f"\n[{i}] SKIP {config.description} (already done)")
            continue

        print(f"\n{'='*60}")
        print(f"[{i}] {config.description}  "
              f"(lev={config.max_leverage}x, seed={config.seed}, "
              f"ppy={config.periods_per_year})")
        print(f"{'='*60}")

        ckpt_dir = str(ckpt_root / config.description)
        os.makedirs(ckpt_dir, exist_ok=True)

        # Train
        train_result = run_training(config, args.train_data, ckpt_dir,
                                    args.time_budget)
        if "error" in train_result and "train_return" not in train_result:
            row = {
                "description": config.description,
                "seed": config.seed,
                "max_leverage": config.max_leverage,
                "periods_per_year": config.periods_per_year,
                "trade_penalty": config.trade_penalty,
                "short_borrow_apr": config.short_borrow_apr,
                "error": train_result["error"],
            }
            _write_row(leaderboard_path, fieldnames, row)
            continue

        # Find checkpoint and evaluate
        ckpt = find_checkpoint(ckpt_dir)
        if ckpt is None:
            row = {
                "description": config.description,
                "seed": config.seed,
                "max_leverage": config.max_leverage,
                "periods_per_year": config.periods_per_year,
                "trade_penalty": config.trade_penalty,
                "short_borrow_apr": config.short_borrow_apr,
                "error": "no checkpoint",
                **{k: train_result.get(k) for k in
                   ["train_return", "train_sortino", "train_wr",
                    "train_steps", "elapsed_s"]},
            }
            _write_row(leaderboard_path, fieldnames, row)
            continue

        print(f"  Evaluating {ckpt}...")
        eval_result = run_eval(
            ckpt, args.val_data, config.hidden_size,
            config.max_steps, config.periods_per_year,
        )

        row = {
            "description": config.description,
            "seed": config.seed,
            "max_leverage": config.max_leverage,
            "periods_per_year": config.periods_per_year,
            "trade_penalty": config.trade_penalty,
            "short_borrow_apr": config.short_borrow_apr,
            **{k: train_result.get(k) for k in
               ["train_return", "train_sortino", "train_wr",
                "train_steps", "elapsed_s"]},
            **{k: eval_result.get(k) for k in
               ["val_return", "val_sortino", "val_wr",
                "val_profitable_pct"]},
            "error": eval_result.get("error", ""),
        }
        _write_row(leaderboard_path, fieldnames, row)

        val_ret = eval_result.get("val_return")
        if val_ret is not None:
            print(f"  Val: ret={val_ret:.4f}, sortino={eval_result.get('val_sortino')}, "
                  f"wr={eval_result.get('val_wr')}, "
                  f"profitable={eval_result.get('val_profitable_pct')}%")
            if val_ret > best_val:
                best_val = val_ret
                print(f"  *** NEW BEST val_return={val_ret:.4f} ***")

    # --- Phase 2: Compare existing checkpoints ---
    print(f"\n{'='*70}")
    print("PHASE 2: Evaluating existing checkpoints for comparison")
    print(f"{'='*70}")

    for comp in COMPARISON_CKPTS:
        desc = comp["description"]
        if desc in existing:
            print(f"\n  SKIP {desc} (already done)")
            continue

        ckpt_path = str(REPO / comp["checkpoint"])
        if not Path(ckpt_path).exists():
            print(f"\n  SKIP {desc}: checkpoint not found at {ckpt_path}")
            continue

        print(f"\n  Evaluating {desc} with periods_per_year=252...")
        eval_result = run_eval(
            ckpt_path, args.val_data,
            comp["hidden_size"], comp["max_steps"],
            periods_per_year=252.0,
        )

        row = {
            "description": desc,
            "seed": "",
            "max_leverage": 1.0,
            "periods_per_year": 252.0,
            "trade_penalty": comp.get("trade_penalty", 0.05),
            "short_borrow_apr": 0.0,
            **{k: eval_result.get(k) for k in
               ["val_return", "val_sortino", "val_wr",
                "val_profitable_pct"]},
            "error": eval_result.get("error", comp.get("note", "")),
        }
        _write_row(leaderboard_path, fieldnames, row)

        val_ret = eval_result.get("val_return")
        if val_ret is not None:
            print(f"  Val: ret={val_ret:.4f}, sortino={eval_result.get('val_sortino')}, "
                  f"wr={eval_result.get('val_wr')}")

    # --- Print final leaderboard ---
    print(f"\n{'='*70}")
    print("LEADERBOARD (sorted by val_return)")
    print(f"{'='*70}")
    if leaderboard_path.exists():
        with open(leaderboard_path) as f:
            rows = list(csv.DictReader(f))
        rows = [r for r in rows
                if r.get("val_return") and r["val_return"] != "None"]
        rows.sort(key=lambda r: float(r["val_return"]), reverse=True)
        for r in rows:
            print(f"  {r['description']:30s} val_ret={float(r['val_return']):+.4f} "
                  f"sortino={r.get('val_sortino', ''):>8s} "
                  f"lev={r.get('max_leverage', ''):>4s} "
                  f"ppy={r.get('periods_per_year', ''):>6s} "
                  f"seed={r.get('seed', ''):>4s}")


def _write_row(leaderboard_path: Path, fieldnames: list[str], row: dict) -> None:
    """Append a row to the CSV leaderboard, writing header if needed."""
    write_header = not leaderboard_path.exists()
    with open(leaderboard_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
