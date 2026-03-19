#!/usr/bin/env python3
"""
Sweep: stocks daily RL with shorts ENABLED vs long-only baselines.

Grid: trade_penalty (0.05, 0.10, 0.15) x leverage (1.0, 2.0) x seeds (42, 314) = 12 shorts runs
Plus 3 long-only baselines (tp=0.05 lev=1.0 per seed + tp=0.05 lev=2.0 s42).

Training flags:
  - data: pufferlib_market/data/shortable10_daily_train.bin
  - max_steps=720, periods_per_year=252
  - hidden_size=1024, lr=3e-4, ent_coef=0.05, anneal_lr
  - short_borrow_apr=0.0625, fee_rate=0.001
  - NO --disable-shorts for shorts runs; --disable-shorts for baselines

After training, evaluates all checkpoints on val data and prints comparison.

Usage:
    source .venv313/bin/activate
    python -u sweep_stocks_shorts_daily.py [--time-budget 300]
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

TRAIN_DATA = "pufferlib_market/data/shortable10_daily_train.bin"
VAL_DATA = "pufferlib_market/data/shortable10_daily_val.bin"
CHECKPOINT_ROOT = "pufferlib_market/checkpoints/stocks_shorts"
LEADERBOARD_PATH = "pufferlib_market/stocks_shorts_daily_leaderboard.csv"


@dataclass
class SweepConfig:
    name: str
    trade_penalty: float
    max_leverage: float
    seed: int
    disable_shorts: bool


def build_configs() -> list[SweepConfig]:
    """Build the full sweep grid: 12 shorts + 3 long-only baselines."""
    configs: list[SweepConfig] = []

    trade_penalties = [0.05, 0.10, 0.15]
    leverages = [1.0, 2.0]
    seeds = [42, 314]

    # Shorts-enabled grid
    for tp in trade_penalties:
        for lev in leverages:
            for seed in seeds:
                name = f"shorts_lev{lev:.0f}_tp{tp:.2f}_s{seed}"
                configs.append(SweepConfig(
                    name=name,
                    trade_penalty=tp,
                    max_leverage=lev,
                    seed=seed,
                    disable_shorts=False,
                ))

    # Long-only baselines: tp=0.05 for each (lev, seed) combo
    for lev in leverages:
        for seed in seeds:
            if lev == 2.0 and seed == 314:
                continue  # 3 baselines total, not 4
            name = f"longonly_lev{lev:.0f}_tp0.05_s{seed}"
            configs.append(SweepConfig(
                name=name,
                trade_penalty=0.05,
                max_leverage=lev,
                seed=seed,
                disable_shorts=True,
            ))

    return configs


def run_training(config: SweepConfig, time_budget: int, checkpoint_dir: str) -> dict:
    """Train a single config, return parsed results."""
    cmd = [
        sys.executable, "-u", "-m", "pufferlib_market.train",
        "--data-path", TRAIN_DATA,
        "--total-timesteps", "999999999",
        "--max-steps", "720",
        "--periods-per-year", "252",
        "--hidden-size", "1024",
        "--lr", "3e-4",
        "--ent-coef", "0.05",
        "--fee-rate", "0.001",
        "--max-leverage", str(config.max_leverage),
        "--short-borrow-apr", "0.0625",
        "--trade-penalty", str(config.trade_penalty),
        "--checkpoint-dir", checkpoint_dir,
        "--seed", str(config.seed),
        "--anneal-lr",
    ]
    if config.disable_shorts:
        cmd.append("--disable-shorts")

    print(f"  Training for {time_budget}s ...")
    print(f"  cmd: {' '.join(cmd[-10:])}")
    t0 = time.time()
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=str(REPO), preexec_fn=os.setsid,
        )
        stdout_lines: list[str] = []
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
        elapsed = time.time() - t0
    except Exception as e:
        return {"error": str(e)}

    # Parse last training stats
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

    print(f"  Training done: {elapsed:.0f}s, {total_steps:,} steps, "
          f"ret={train_return}, sortino={train_sortino}")

    return {
        "train_return": train_return,
        "train_sortino": train_sortino,
        "train_wr": train_wr,
        "train_steps": total_steps,
        "elapsed_s": elapsed,
    }


def find_best_checkpoint(checkpoint_dir: str) -> str | None:
    """Find the best checkpoint in a directory."""
    ckpt_dir = Path(checkpoint_dir)
    for name in ["best.pt", "final.pt"]:
        p = ckpt_dir / name
        if p.exists():
            return str(p)
    pts = list(ckpt_dir.glob("*.pt"))
    if pts:
        return str(max(pts, key=lambda f: f.stat().st_mtime))
    return None


def run_eval(checkpoint: str, data_path: str, config: SweepConfig) -> dict:
    """Evaluate a checkpoint on val data."""
    cmd = [
        sys.executable, "-u", "-m", "pufferlib_market.evaluate",
        "--checkpoint", checkpoint,
        "--data-path", data_path,
        "--deterministic",
        "--hidden-size", "1024",
        "--max-steps", "720",
        "--periods-per-year", "252",
        "--num-episodes", "100",
        "--seed", "42",
        "--fill-slippage-bps", "8",
        "--fee-rate", "0.001",
        "--max-leverage", str(config.max_leverage),
        "--short-borrow-apr", "0.0625",
    ]
    if config.disable_shorts:
        cmd.append("--disable-shorts")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=180, cwd=str(REPO),
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
        return {"error": f"eval: {e}"}


def main():
    parser = argparse.ArgumentParser(description="Stocks shorts daily sweep")
    parser.add_argument("--time-budget", type=int, default=300,
                        help="Training time budget per run in seconds")
    args = parser.parse_args()

    # Check data dependency
    train_path = REPO / TRAIN_DATA
    val_path = REPO / VAL_DATA
    if not train_path.exists():
        print(f"ERROR: Training data not found: {train_path}")
        print("  Run Unit 2 first to export shortable10_daily_train.bin")
        sys.exit(1)
    if not val_path.exists():
        print(f"ERROR: Validation data not found: {val_path}")
        print("  Run Unit 2 first to export shortable10_daily_val.bin")
        sys.exit(1)

    configs = build_configs()
    print(f"Stocks shorts daily sweep: {len(configs)} configs, "
          f"{args.time_budget}s each")
    print(f"  Train data: {TRAIN_DATA}")
    print(f"  Val data:   {VAL_DATA}")

    ckpt_root = REPO / CHECKPOINT_ROOT
    ckpt_root.mkdir(parents=True, exist_ok=True)

    leaderboard_path = REPO / LEADERBOARD_PATH
    fieldnames = [
        "name", "disable_shorts", "max_leverage", "trade_penalty", "seed",
        "val_return", "val_sortino", "val_wr", "val_profitable_pct",
        "train_return", "train_sortino", "train_wr", "train_steps",
        "elapsed_s", "error",
    ]

    # Load already-completed runs
    existing: set[str] = set()
    if leaderboard_path.exists():
        with open(leaderboard_path) as f:
            for row in csv.DictReader(f):
                existing.add(row.get("name", ""))

    for i, config in enumerate(configs):
        if config.name in existing:
            print(f"\n[{i+1}/{len(configs)}] SKIP {config.name} (already done)")
            continue

        mode = "LONG-ONLY" if config.disable_shorts else "SHORTS"
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(configs)}] {config.name} ({mode})")
        print(f"  tp={config.trade_penalty} lev={config.max_leverage} "
              f"seed={config.seed}")
        print(f"{'='*60}")

        ckpt_dir = str(ckpt_root / config.name)
        os.makedirs(ckpt_dir, exist_ok=True)

        # Train
        train_result = run_training(config, args.time_budget, ckpt_dir)
        if "error" in train_result:
            print(f"  TRAIN ERROR: {train_result['error']}")

        # Evaluate
        eval_result: dict = {}
        ckpt_path = find_best_checkpoint(ckpt_dir)
        if ckpt_path:
            print(f"  Evaluating {Path(ckpt_path).name} ...")
            eval_result = run_eval(ckpt_path, str(val_path), config)
        else:
            eval_result = {"error": "no checkpoint found"}
            print(f"  No checkpoint found in {ckpt_dir}")

        # Write row
        error = train_result.get("error", "") or eval_result.get("error", "")
        row = {
            "name": config.name,
            "disable_shorts": config.disable_shorts,
            "max_leverage": config.max_leverage,
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
            "error": error,
        }

        write_header = not leaderboard_path.exists()
        with open(leaderboard_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    # Print final comparison
    print_leaderboard(leaderboard_path)


def print_leaderboard(leaderboard_path: Path) -> None:
    """Print sorted leaderboard with shorts vs long-only comparison."""
    if not leaderboard_path.exists():
        return

    with open(leaderboard_path) as f:
        rows = list(csv.DictReader(f))

    valid = [r for r in rows if r.get("val_return") and r["val_return"] != "None"]
    if not valid:
        print("\nNo valid results yet.")
        return

    valid.sort(key=lambda r: float(r["val_return"]), reverse=True)

    print(f"\n{'='*80}")
    print("LEADERBOARD — Stocks Shorts Daily (sorted by val_return)")
    print(f"{'='*80}")
    print(f"{'Name':42s} {'val_ret':>9s} {'sortino':>8s} {'wr':>7s} "
          f"{'prof%':>6s} {'mode':>8s}")
    print(f"{'-'*80}")

    for r in valid:
        mode = "long" if r["disable_shorts"] == "True" else "SHORTS"
        vr = float(r["val_return"])
        vs = r.get("val_sortino", "")
        vw = r.get("val_wr", "")
        vp = r.get("val_profitable_pct", "")
        print(f"  {r['name']:40s} {vr:+.4f}   {vs:>8s} {vw:>7s} "
              f"{vp:>6s} {mode:>8s}")

    # Summary comparison
    shorts_rows = [r for r in valid if r["disable_shorts"] == "False"]
    long_rows = [r for r in valid if r["disable_shorts"] == "True"]

    if shorts_rows and long_rows:
        shorts_avg = sum(float(r["val_return"]) for r in shorts_rows) / len(shorts_rows)
        long_avg = sum(float(r["val_return"]) for r in long_rows) / len(long_rows)
        print(f"\n  Shorts avg val_return: {shorts_avg:+.4f} ({len(shorts_rows)} runs)")
        print(f"  Long-only avg val_return: {long_avg:+.4f} ({len(long_rows)} runs)")
        diff = shorts_avg - long_avg
        print(f"  Difference (shorts - long): {diff:+.4f}")


if __name__ == "__main__":
    main()
