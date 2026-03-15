"""
Leverage + reward shaping sweep for FDUSD zero-fee crypto trading.

Tests leverage levels [1x, 2x, 3x, 5x] combined with Sortino-focused
reward shaping to find the highest risk-adjusted returns.

Usage:
    python -u pufferlib_market/sweep_leverage.py \
        --train-data pufferlib_market/data/fdusd3_hourly_train.bin \
        --val-data pufferlib_market/data/fdusd3_hourly_val.bin \
        --timeframe hourly
"""

from __future__ import annotations

import argparse
import csv
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


@dataclass
class LeverageConfig:
    description: str
    max_leverage: float = 1.0
    short_borrow_apr: float = 0.0
    disable_shorts: bool = True
    trade_penalty: float = 0.0
    downside_penalty: float = 0.0
    smooth_downside_penalty: float = 0.0
    drawdown_penalty: float = 0.0
    smoothness_penalty: float = 0.0
    fill_slippage_bps: float = 5.0
    fee_rate: float = 0.0
    hidden_size: int = 1024
    lr: float = 3e-4
    ent_coef: float = 0.05
    anneal_lr: bool = True
    max_steps: int = 720
    periods_per_year: float = 8760.0
    reward_scale: float = 10.0
    reward_clip: float = 5.0
    cash_penalty: float = 0.01


# Sweep configurations
EXPERIMENTS: list[LeverageConfig] = [
    # --- Leverage sweep (long-only, no reward shaping) ---
    LeverageConfig(description="lev1x_baseline", max_leverage=1.0),
    LeverageConfig(description="lev2x_baseline", max_leverage=2.0),
    LeverageConfig(description="lev3x_baseline", max_leverage=3.0),
    LeverageConfig(description="lev5x_baseline", max_leverage=5.0),

    # --- Leverage + trade penalty (best daily finding) ---
    LeverageConfig(description="lev1x_tp05", max_leverage=1.0, trade_penalty=0.05),
    LeverageConfig(description="lev2x_tp05", max_leverage=2.0, trade_penalty=0.05),
    LeverageConfig(description="lev3x_tp05", max_leverage=3.0, trade_penalty=0.05),
    LeverageConfig(description="lev5x_tp05", max_leverage=5.0, trade_penalty=0.05),

    # --- Leverage + downside penalty (Sortino shaping) ---
    LeverageConfig(description="lev2x_ds02", max_leverage=2.0, downside_penalty=0.2),
    LeverageConfig(description="lev2x_ds05", max_leverage=2.0, downside_penalty=0.5),
    LeverageConfig(description="lev3x_ds02", max_leverage=3.0, downside_penalty=0.2),
    LeverageConfig(description="lev3x_ds05", max_leverage=3.0, downside_penalty=0.5),
    LeverageConfig(description="lev5x_ds05", max_leverage=5.0, downside_penalty=0.5),

    # --- Leverage + drawdown penalty ---
    LeverageConfig(description="lev2x_dd01", max_leverage=2.0, drawdown_penalty=0.1),
    LeverageConfig(description="lev3x_dd01", max_leverage=3.0, drawdown_penalty=0.1),
    LeverageConfig(description="lev3x_dd03", max_leverage=3.0, drawdown_penalty=0.3),

    # --- Leverage + combined shaping (trade_pen + downside) ---
    LeverageConfig(description="lev2x_tp05_ds02", max_leverage=2.0, trade_penalty=0.05, downside_penalty=0.2),
    LeverageConfig(description="lev3x_tp05_ds02", max_leverage=3.0, trade_penalty=0.05, downside_penalty=0.2),
    LeverageConfig(description="lev3x_tp05_dd01", max_leverage=3.0, trade_penalty=0.05, drawdown_penalty=0.1),
    LeverageConfig(description="lev5x_tp05_ds05", max_leverage=5.0, trade_penalty=0.05, downside_penalty=0.5),

    # --- With shorts enabled (long+short, borrow fee) ---
    LeverageConfig(description="lev1x_short", max_leverage=1.0, disable_shorts=False, short_borrow_apr=0.0625),
    LeverageConfig(description="lev2x_short", max_leverage=2.0, disable_shorts=False, short_borrow_apr=0.0625),
    LeverageConfig(description="lev3x_short", max_leverage=3.0, disable_shorts=False, short_borrow_apr=0.0625),
    LeverageConfig(description="lev3x_short_tp05", max_leverage=3.0, disable_shorts=False,
                   short_borrow_apr=0.0625, trade_penalty=0.05),

    # --- Smooth downside (differentiable alternative) ---
    LeverageConfig(description="lev2x_sds03", max_leverage=2.0, smooth_downside_penalty=0.3),
    LeverageConfig(description="lev3x_sds05", max_leverage=3.0, smooth_downside_penalty=0.5),

    # --- Lower reward scale for leveraged (prevent gradient explosion) ---
    LeverageConfig(description="lev3x_rs5", max_leverage=3.0, reward_scale=5.0),
    LeverageConfig(description="lev5x_rs5", max_leverage=5.0, reward_scale=5.0),
    LeverageConfig(description="lev5x_rs3_clip3", max_leverage=5.0, reward_scale=3.0, reward_clip=3.0),
]


def run_trial(
    config: LeverageConfig,
    train_data: str,
    val_data: str,
    time_budget: int,
    checkpoint_dir: str,
) -> dict:
    cmd = [
        sys.executable, "-u", "-m", "pufferlib_market.train",
        "--data-path", train_data,
        "--total-timesteps", "999999999",
        "--max-steps", str(config.max_steps),
        "--hidden-size", str(config.hidden_size),
        "--lr", str(config.lr),
        "--ent-coef", str(config.ent_coef),
        "--fee-rate", str(config.fee_rate),
        "--max-leverage", str(config.max_leverage),
        "--short-borrow-apr", str(config.short_borrow_apr),
        "--trade-penalty", str(config.trade_penalty),
        "--downside-penalty", str(config.downside_penalty),
        "--smooth-downside-penalty", str(config.smooth_downside_penalty),
        "--drawdown-penalty", str(config.drawdown_penalty),
        "--fill-slippage-bps", str(config.fill_slippage_bps),
        "--reward-scale", str(config.reward_scale),
        "--reward-clip", str(config.reward_clip),
        "--cash-penalty", str(config.cash_penalty),
        "--checkpoint-dir", checkpoint_dir,
        "--periods-per-year", str(config.periods_per_year),
        "--seed", "42",
    ]
    if config.anneal_lr:
        cmd.append("--anneal-lr")
    if config.disable_shorts:
        cmd.append("--disable-shorts")

    print(f"\n  Training for {time_budget}s...")
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
                    stdout_lines.append(line.decode("utf-8", errors="replace").strip())
            except Exception:
                pass
        if proc.poll() is None:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=10)
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

    print(f"  Training done: {elapsed:.0f}s, {total_steps:,} steps")

    ckpt_path = Path(checkpoint_dir) / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = Path(checkpoint_dir) / "final.pt"
    if not ckpt_path.exists():
        pts = list(Path(checkpoint_dir).glob("*.pt"))
        if pts:
            ckpt_path = max(pts, key=lambda p: p.stat().st_mtime)
        else:
            return {"error": "no checkpoint", "train_return": train_return}

    print(f"  Evaluating...")
    eval_cmd = [
        sys.executable, "-u", "-m", "pufferlib_market.evaluate",
        "--checkpoint", str(ckpt_path),
        "--data-path", val_data,
        "--deterministic",
        "--hidden-size", str(config.hidden_size),
        "--max-steps", str(config.max_steps),
        "--num-episodes", "100",
        "--seed", "42",
        "--fill-slippage-bps", "8",
        "--periods-per-year", str(config.periods_per_year),
    ]

    try:
        result = subprocess.run(
            eval_cmd, capture_output=True, text=True, timeout=120, cwd=str(REPO),
        )
        eval_output = result.stdout + result.stderr
        val_return = val_wr = val_sortino = val_profitable_pct = None
        for line in eval_output.split("\n"):
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
            "train_return": train_return,
            "train_sortino": train_sortino,
            "train_wr": train_wr,
            "train_steps": total_steps,
            "val_return": val_return,
            "val_sortino": val_sortino,
            "val_wr": val_wr,
            "val_profitable_pct": val_profitable_pct,
            "elapsed_s": elapsed,
        }
    except subprocess.TimeoutExpired:
        return {"error": "eval timeout", "train_return": train_return}
    except Exception as e:
        return {"error": f"eval error: {e}", "train_return": train_return}


def main():
    parser = argparse.ArgumentParser(description="Leverage + reward shaping sweep")
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--val-data", required=True)
    parser.add_argument("--time-budget", type=int, default=300)
    parser.add_argument("--timeframe", choices=["hourly", "daily"], default="hourly")
    parser.add_argument("--leaderboard", default=None)
    parser.add_argument("--checkpoint-root", default=None)
    args = parser.parse_args()

    suffix = args.timeframe
    leaderboard = args.leaderboard or f"pufferlib_market/sweep_leverage_{suffix}_leaderboard.csv"
    ckpt_root = Path(args.checkpoint_root or f"pufferlib_market/checkpoints/leverage_sweep_{suffix}")
    ckpt_root.mkdir(parents=True, exist_ok=True)

    # Override for daily
    if args.timeframe == "daily":
        for exp in EXPERIMENTS:
            exp.periods_per_year = 365.0
            exp.max_steps = 90

    fieldnames = [
        "trial", "description", "val_return", "val_sortino", "val_wr",
        "val_profitable_pct", "train_return", "train_steps", "elapsed_s", "error",
        "max_leverage", "disable_shorts", "short_borrow_apr",
        "trade_penalty", "downside_penalty", "drawdown_penalty",
        "smooth_downside_penalty", "reward_scale", "reward_clip", "fee_rate",
    ]

    existing = set()
    leaderboard_path = Path(leaderboard)
    if leaderboard_path.exists():
        with open(leaderboard_path) as f:
            for row in csv.DictReader(f):
                existing.add(row.get("description", ""))

    best_val = -float("inf")
    for i, config in enumerate(EXPERIMENTS):
        if config.description in existing:
            print(f"\n[{i}] SKIP {config.description} (already done)")
            continue

        print(f"\n{'='*60}")
        print(f"[{i}] {config.description}")
        overrides = {k: v for k, v in asdict(config).items()
                     if k != "description" and v != getattr(LeverageConfig(description=""), k)}
        if overrides:
            print(f"  {overrides}")
        print(f"{'='*60}")

        ckpt_dir = str(ckpt_root / config.description)
        os.makedirs(ckpt_dir, exist_ok=True)

        result = run_trial(config, args.train_data, args.val_data,
                          args.time_budget, ckpt_dir)

        row = {
            "trial": i,
            "description": config.description,
            "val_return": result.get("val_return"),
            "val_sortino": result.get("val_sortino"),
            "val_wr": result.get("val_wr"),
            "val_profitable_pct": result.get("val_profitable_pct"),
            "train_return": result.get("train_return"),
            "train_steps": result.get("train_steps"),
            "elapsed_s": result.get("elapsed_s"),
            "error": result.get("error", ""),
            "max_leverage": config.max_leverage,
            "disable_shorts": config.disable_shorts,
            "short_borrow_apr": config.short_borrow_apr,
            "trade_penalty": config.trade_penalty,
            "downside_penalty": config.downside_penalty,
            "drawdown_penalty": config.drawdown_penalty,
            "smooth_downside_penalty": config.smooth_downside_penalty,
            "reward_scale": config.reward_scale,
            "reward_clip": config.reward_clip,
            "fee_rate": config.fee_rate,
        }

        write_header = not leaderboard_path.exists()
        with open(leaderboard_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        val_ret = result.get("val_return")
        if val_ret is not None and val_ret > best_val:
            best_val = val_ret
            print(f"  *** NEW BEST val_return={val_ret:.4f} ***")

    # Print final leaderboard
    print(f"\n{'='*60}")
    print("LEADERBOARD (sorted by val_return)")
    print(f"{'='*60}")
    if leaderboard_path.exists():
        with open(leaderboard_path) as f:
            rows = list(csv.DictReader(f))
        rows = [r for r in rows if r.get("val_return") and r["val_return"] != "None"]
        rows.sort(key=lambda r: float(r["val_return"]), reverse=True)
        for r in rows[:15]:
            print(f"  {r['description']:30s} val_ret={float(r['val_return']):+.4f} "
                  f"sortino={r['val_sortino']:>8s} lev={r['max_leverage']:>4s} "
                  f"shorts={'yes' if r['disable_shorts']=='False' else 'no':>3s}")


if __name__ == "__main__":
    main()
