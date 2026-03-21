"""Shared utilities for RL sweep scripts (champions, architecture, etc.)."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Default daily data files, newest first as fallback chain.
_DATA_CANDIDATES = [
    ("pufferlib_market/data/crypto15_daily_train.bin",
     "pufferlib_market/data/crypto15_daily_val.bin"),
    ("pufferlib_market/data/crypto10_daily_train.bin",
     "pufferlib_market/data/crypto10_daily_val.bin"),
]


def resolve_data_paths(train_override: str, val_override: str) -> tuple[str, str]:
    """Return (train, val) data paths, auto-detecting from known candidates."""
    if train_override and val_override:
        return train_override, val_override
    for train, val in _DATA_CANDIDATES:
        if (REPO / train).exists() and (REPO / val).exists():
            return train, val
    first_train, first_val = _DATA_CANDIDATES[0]
    print(f"[warning] No default data files found locally. Pass --data-train / --data-val.")
    return first_train, first_val


def build_dispatch_cmd(
    descriptions_csv: str,
    *,
    train_data: str,
    val_data: str,
    run_id: str,
    num_seeds: int,
    gpu_type: str,
    budget_limit: float,
    time_budget: int,
    max_trials: int,
    leaderboard: str,
    dry_run: bool,
) -> list[str]:
    """Build the argv list for dispatch_rl_training.py."""
    cmd = [
        sys.executable, "-u",
        str(REPO / "scripts" / "dispatch_rl_training.py"),
        "--data-train", train_data,
        "--data-val", val_data,
        "--run-id", run_id,
        "--time-budget", str(time_budget),
        "--max-trials", str(max_trials),
        "--descriptions", descriptions_csv,
        "--num-seeds", str(num_seeds),
        "--leaderboard", leaderboard,
        "--budget-limit", str(budget_limit),
    ]
    if gpu_type:
        cmd += ["--gpu-type", gpu_type]
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def merge_leaderboard_rows(source_csv: Path, dest_rows: list[dict]) -> None:
    """Append all rows from source_csv into dest_rows in-place."""
    if not source_csv.exists():
        return
    try:
        with open(source_csv) as f:
            for row in csv.DictReader(f):
                dest_rows.append(row)
    except Exception as exc:
        print(f"[warning] Could not read {source_csv}: {exc}")


def save_combined_leaderboard(rows: list[dict], path: Path, *, tag: str = "") -> None:
    """Write rows to a CSV, unioning all column names across rows."""
    if not rows:
        return
    # dict.fromkeys preserves insertion order (Python 3.7+) and deduplicates.
    all_keys = list(dict.fromkeys(k for row in rows for k in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    label = f"[{tag}] " if tag else ""
    print(f"{label}Combined leaderboard saved to: {path}")
