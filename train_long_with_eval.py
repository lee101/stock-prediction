#!/usr/bin/env python3
"""Train for N segments with periodic val evaluation to detect overfitting.

Each segment runs training for a fixed wall-clock budget (default 5 min),
then evaluates the current best.pt checkpoint on validation data using the
pure-python market simulator, and logs the result.

Usage (smoke test — 2 segments):
    python train_long_with_eval.py --num-segments 2

Full run (12 segments = 60 min):
    python train_long_with_eval.py --num-segments 12
"""

from __future__ import annotations

import argparse
import csv
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ── Paths ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "pufferlib_market" / "data"
TRAIN_DATA = DATA_DIR / "crypto5_daily_train.bin"
VAL_DATA = DATA_DIR / "crypto5_daily_val.bin"
CHECKPOINT_DIR = REPO_ROOT / "pufferlib_market" / "checkpoints" / "long_eval" / "tp0.15_s314"
OUTPUT_CSV = REPO_ROOT / "long_training_curve.csv"
PYTHON = sys.executable

# ── Eval config ──────────────────────────────────────────────────────────
EVAL_PERIOD_DAYS = 120
FEE_RATE = 0.001
FILL_BUFFER_BPS = 8.0
PERIODS_PER_YEAR = 365.0


def _find_checkpoint(preference: str = "best") -> Path | None:
    """Find a checkpoint in CHECKPOINT_DIR.

    preference="best": best.pt > update_*.pt > final.pt  (for eval)
    preference="resume": final.pt > best.pt > update_*.pt (for resume — final has optimizer state)
    """
    best = CHECKPOINT_DIR / "best.pt"
    final = CHECKPOINT_DIR / "final.pt"
    candidates = sorted(CHECKPOINT_DIR.glob("update_*.pt"))
    latest_update = candidates[-1] if candidates else None

    if preference == "resume":
        order = [final, best, latest_update]
    else:
        order = [best, latest_update, final]

    for p in order:
        if p is not None and p.exists():
            return p
    return None


def _read_global_step(ckpt_path: str | Path) -> int:
    """Read global_step from a checkpoint without loading the full model."""
    try:
        payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        if isinstance(payload, dict):
            return int(payload.get("global_step", 0))
    except Exception:
        pass
    return 0


def run_training_segment(segment: int, segment_seconds: int) -> int:
    """Run one training segment, return the process exit code.

    The training process is given `segment_seconds` to run, then is sent
    SIGTERM so it can save its checkpoint before exiting.
    """
    resume_ckpt = _find_checkpoint("resume")

    cmd = [
        PYTHON, "-u", "-m", "pufferlib_market.train",
        "--data-path", str(TRAIN_DATA),
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
        "--seed", "314",
        "--reward-scale", "10.0",
        "--reward-clip", "5.0",
        "--cash-penalty", "0.01",
        "--fee-rate", str(FEE_RATE),
        "--trade-penalty", "0.15",
        "--anneal-lr",
        "--checkpoint-dir", str(CHECKPOINT_DIR),
        "--periods-per-year", str(PERIODS_PER_YEAR),
        "--save-every", "20",
    ]
    if resume_ckpt:
        cmd.extend(["--resume-from", str(resume_ckpt)])

    print(f"\n{'='*70}")
    print(f"SEGMENT {segment}: training for {segment_seconds}s")
    if resume_ckpt:
        print(f"  Resuming from: {resume_ckpt}")
    else:
        print("  Fresh start (no resume checkpoint)")
    print(f"{'='*70}\n")
    sys.stdout.flush()

    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    try:
        proc.wait(timeout=segment_seconds)
    except subprocess.TimeoutExpired:
        # Gracefully stop — send SIGTERM so train.py can flush final checkpoint
        print(f"\n  Segment {segment}: time budget exhausted, sending SIGTERM...")
        sys.stdout.flush()
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            print(f"  Segment {segment}: SIGTERM timed out, sending SIGKILL...")
            proc.kill()
            proc.wait(timeout=10)

    return proc.returncode or 0


def _load_val_data():
    """Load and cache validation data (called once, reused across segments)."""
    from pufferlib_market.hourly_replay import read_mktd
    return read_mktd(str(VAL_DATA))


# Module-level cache — populated on first eval, reused thereafter
_val_data_cache = None


def evaluate_checkpoint_on_val(ckpt_path: str | Path) -> dict:
    """Evaluate a checkpoint on the validation data using the pure-python sim.

    Returns dict with keys: total_return, sortino, max_drawdown, num_trades, win_rate.
    Reuses evaluate_multiperiod.evaluate_period for the actual simulation.
    """
    from pufferlib_market.evaluate_multiperiod import load_policy, evaluate_period

    global _val_data_cache
    if _val_data_cache is None:
        _val_data_cache = _load_val_data()
    data = _val_data_cache

    device = torch.device("cpu")
    num_symbols = data.num_symbols

    policy, _, num_actions = load_policy(
        str(ckpt_path), num_symbols, device=device,
    )

    result = evaluate_period(
        policy,
        data,
        EVAL_PERIOD_DAYS,
        num_symbols=num_symbols,
        fee_rate=FEE_RATE,
        fill_buffer_bps=FILL_BUFFER_BPS,
        max_leverage=1.0,
        periods_per_year=PERIODS_PER_YEAR,
        deterministic=True,
        device=device,
    )

    if "error" in result:
        return {
            "total_return": float("nan"),
            "sortino": float("nan"),
            "max_drawdown": float("nan"),
            "num_trades": 0,
            "win_rate": 0.0,
        }

    return {
        "total_return": float(result["total_return"]),
        "sortino": float(result["sortino"]),
        "max_drawdown": float(result["max_drawdown"]),
        "num_trades": int(result["num_trades"]),
        "win_rate": float(result["win_rate"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Long RL training with periodic val evaluation")
    parser.add_argument("--num-segments", type=int, default=12,
                        help="Number of training segments (default: 12 = 60 min)")
    parser.add_argument("--segment-seconds", type=int, default=300,
                        help="Wall-clock seconds per training segment (default: 300 = 5 min)")
    args = parser.parse_args()

    num_segments = args.num_segments
    segment_seconds = args.segment_seconds

    print(f"Long training with eval: {num_segments} segments x {segment_seconds}s")
    print(f"  Train data: {TRAIN_DATA}")
    print(f"  Val data:   {VAL_DATA}")
    print(f"  Checkpoint: {CHECKPOINT_DIR}")
    print(f"  Output CSV: {OUTPUT_CSV}")
    print(f"  Eval:       {EVAL_PERIOD_DAYS}d, fee={FEE_RATE}, fill_buf={FILL_BUFFER_BPS}bps")
    print()

    # Validate data files exist
    if not TRAIN_DATA.exists():
        print(f"ERROR: Train data not found: {TRAIN_DATA}")
        print("  Looking in parent repo...")
        alt = Path("/nvme0n1-disk/code/stock-prediction/pufferlib_market/data/crypto5_daily_train.bin")
        if alt.exists():
            print(f"  Found at {alt} — creating symlink")
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            if not TRAIN_DATA.exists():
                TRAIN_DATA.symlink_to(alt)
        else:
            sys.exit(1)

    if not VAL_DATA.exists():
        alt = Path("/nvme0n1-disk/code/stock-prediction/pufferlib_market/data/crypto5_daily_val.bin")
        if alt.exists():
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            if not VAL_DATA.exists():
                VAL_DATA.symlink_to(alt)
        else:
            print(f"ERROR: Val data not found: {VAL_DATA}")
            sys.exit(1)

    # Clean start: remove old checkpoints if any
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    wall_start = time.time()

    for seg in range(1, num_segments + 1):
        # ── Train ──
        exit_code = run_training_segment(seg, segment_seconds)

        # ── Read current training progress ──
        ckpt_path = _find_checkpoint("best")
        if ckpt_path is None:
            print(f"\n  Segment {seg}: No checkpoint found after training (exit={exit_code})")
            rows.append({
                "segment": seg,
                "elapsed_min": (time.time() - wall_start) / 60.0,
                "train_steps": 0,
                "val_return_120d": float("nan"),
                "val_sortino_120d": float("nan"),
                "val_maxdd_120d": float("nan"),
                "val_trades": 0,
                "val_winrate": 0.0,
                "train_exit_code": exit_code,
            })
            continue

        global_step = _read_global_step(ckpt_path)
        print(f"\n  Segment {seg}: evaluating {ckpt_path.name} (step={global_step:,d})...")
        sys.stdout.flush()

        # ── Eval ──
        try:
            result = evaluate_checkpoint_on_val(ckpt_path)
        except Exception as e:
            print(f"  Eval error: {e}")
            result = {
                "total_return": float("nan"),
                "sortino": float("nan"),
                "max_drawdown": float("nan"),
                "num_trades": 0,
                "win_rate": 0.0,
            }

        elapsed_min = (time.time() - wall_start) / 60.0
        row = {
            "segment": seg,
            "elapsed_min": round(elapsed_min, 1),
            "train_steps": global_step,
            "val_return_120d": round(result["total_return"] * 100, 3),
            "val_sortino_120d": round(result["sortino"], 3),
            "val_maxdd_120d": round(result["max_drawdown"] * 100, 3),
            "val_trades": result["num_trades"],
            "val_winrate": round(result["win_rate"] * 100, 1),
            "train_exit_code": exit_code,
        }
        rows.append(row)

        print(
            f"  [{seg}/{num_segments}] "
            f"elapsed={elapsed_min:.1f}min  steps={global_step:,d}  "
            f"val_ret={row['val_return_120d']:+.2f}%  "
            f"val_sortino={row['val_sortino_120d']:.2f}  "
            f"val_maxdd={row['val_maxdd_120d']:.2f}%  "
            f"trades={row['val_trades']}  wr={row['val_winrate']:.1f}%"
        )

    # ── Write CSV ──
    fieldnames = [
        "segment", "elapsed_min", "train_steps",
        "val_return_120d", "val_sortino_120d", "val_maxdd_120d",
        "val_trades", "val_winrate", "train_exit_code",
    ]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ── Print summary table ──
    total_elapsed = (time.time() - wall_start) / 60.0
    print(f"\n{'='*80}")
    print(f"LONG TRAINING CURVE — {num_segments} segments, {total_elapsed:.1f} min total")
    print(f"{'='*80}")
    print(f"{'Seg':>4} {'Elapsed':>8} {'Steps':>12} {'ValRet%':>9} {'Sortino':>8} {'MaxDD%':>7} {'Trades':>7} {'WR%':>6}")
    print("-" * 70)

    best_sortino = -999.0
    best_seg = -1
    for r in rows:
        s = r["val_sortino_120d"]
        if np.isfinite(s) and s > best_sortino:
            best_sortino = s
            best_seg = r["segment"]
        marker = " *" if r["segment"] == best_seg and np.isfinite(s) else ""
        print(
            f"{r['segment']:>4} "
            f"{r['elapsed_min']:>7.1f}m "
            f"{r['train_steps']:>12,d} "
            f"{r['val_return_120d']:>+8.2f}% "
            f"{r['val_sortino_120d']:>8.2f} "
            f"{r['val_maxdd_120d']:>6.2f}% "
            f"{r['val_trades']:>7d} "
            f"{r['val_winrate']:>5.1f}%"
            f"{marker}"
        )

    if best_seg > 0:
        best_row = rows[best_seg - 1]
        print(f"\nOptimal training budget: segment {best_seg} "
              f"({best_row['elapsed_min']:.1f} min, {best_row['train_steps']:,d} steps)")
        print(f"  Best val Sortino: {best_row['val_sortino_120d']:.3f}")
        print(f"  Val return: {best_row['val_return_120d']:+.2f}%")
        print(f"  Val max DD: {best_row['val_maxdd_120d']:.2f}%")

    # Detect overfitting: if best is not the last segment
    if best_seg > 0 and best_seg < num_segments:
        last_sortino = rows[-1]["val_sortino_120d"]
        if np.isfinite(last_sortino) and last_sortino < best_sortino * 0.8:
            print(f"\n  WARNING: Possible overfitting detected!")
            print(f"  Best Sortino at segment {best_seg}, but last segment Sortino={last_sortino:.3f}")
            print(f"  ({(1 - last_sortino / best_sortino) * 100:.1f}% degradation)")
    elif best_seg == num_segments:
        print(f"\n  No overfitting detected — performance still improving at segment {num_segments}")
        print(f"  Consider running more segments to find the peak.")

    print(f"\nResults saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
