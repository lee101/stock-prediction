#!/usr/bin/env python3
"""Launch a diverse GPU sweep across 10 experiments on RTX 5090 pods.

Each experiment runs for 5 minutes (autoresearch timebox), configurable seeds.
Estimated cost per run: ~$1.00-1.50 on RTX 5090.
Total cost for 10 experiments x 3 seeds: ~$10-15.

Usage:
  python scripts/launch_gpu_sweep.py --dry-run
  python scripts/launch_gpu_sweep.py --budget-limit 15
  python scripts/launch_gpu_sweep.py --gpu-type 5090 --data-dir pufferlib_market/data/
  python scripts/launch_gpu_sweep.py --dry-run --num-seeds 1
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts._sweep_utils import build_dispatch_cmd, save_combined_leaderboard  # noqa: E402
from src.runpod_client import GPU_ALIASES, HOURLY_RATES  # noqa: E402

# Derive valid experiment names from the canonical source so this never goes stale.
try:
    from pufferlib_market.autoresearch_rl import TRIAL_CONFIGS as _TRIAL_CONFIGS
    VALID_EXPERIMENT_NAMES: frozenset[str] = frozenset(e["description"] for e in _TRIAL_CONFIGS)
except Exception:
    VALID_EXPERIMENT_NAMES = frozenset()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_GPU_TYPE = "5090"
DEFAULT_BUDGET_LIMIT = 15.0
DEFAULT_NUM_SEEDS = 3
DEFAULT_TIME_BUDGET = 300  # 5 min per trial
DEFAULT_DATA_DIR = "pufferlib_market/data"

# Mirrors dispatch_rl_training._SETUP_OVERHEAD_SECS
_SETUP_OVERHEAD_SECS = 1800

# ---------------------------------------------------------------------------
# Default sweep configs
# ---------------------------------------------------------------------------

DEFAULT_SWEEP_CONFIGS: list[dict] = [
    {
        "name": "trade_pen_05",
        "data_path": "mixed23_daily_train.bin",
        "val_path": "mixed23_daily_val.bin",
        "note": "proven best daily (Sortino 1.76)",
    },
    {
        "name": "slip_5bps",
        "data_path": "crypto12_hourly_train.bin",
        "val_path": "crypto12_hourly_val.bin",
        "note": "best hourly OOS",
    },
    {
        "name": "combined_smooth",
        "data_path": "mixed23_daily_train.bin",
        "val_path": "mixed23_daily_val.bin",
        "note": "low churn + production slippage",
    },
    {
        "name": "transformer_h256",
        "data_path": "mixed23_daily_train.bin",
        "val_path": "mixed23_daily_val.bin",
        "note": "transformer arch exploration",
    },
    {
        "name": "relu_sq_h1024",
        "data_path": "mixed23_daily_train.bin",
        "val_path": "mixed23_daily_val.bin",
        "note": "relu^2 activation variant",
    },
    {
        "name": "cosine_lr_tp05",
        "data_path": "mixed23_daily_train.bin",
        "val_path": "mixed23_daily_val.bin",
        "note": "cosine LR + trade penalty",
    },
    {
        "name": "depth_recur_h1024",
        "data_path": "mixed23_daily_train.bin",
        "val_path": "mixed23_daily_val.bin",
        "note": "depth recurrence arch",
    },
    {
        "name": "combo_best_daily",
        "data_path": "mixed32_daily_train.bin",
        "val_path": "mixed32_daily_val.bin",
        "note": "combo_best on mixed32 universe",
    },
    {
        "name": "calmar_focus",
        "data_path": "mixed23_daily_train.bin",
        "val_path": "mixed23_daily_val.bin",
        "note": "calmar ratio reward shaping",
    },
    {
        "name": "robust_champion",
        "data_path": "mixed23_daily_train.bin",
        "val_path": "mixed23_daily_val.bin",
        "note": "all lessons combined baseline",
    },
]


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------


def estimate_cost_per_experiment(gpu_type: str, num_seeds: int, time_budget_secs: int) -> float:
    """Estimate USD cost for one experiment (all seeds) including setup overhead."""
    resolved = GPU_ALIASES.get(gpu_type.lower(), gpu_type)
    rate = HOURLY_RATES.get(resolved, HOURLY_RATES.get(gpu_type, 0.0))
    total_secs = _SETUP_OVERHEAD_SECS + num_seeds * time_budget_secs
    return rate * (total_secs / 3600)


def estimate_total_cost(
    configs: list[dict],
    gpu_type: str,
    num_seeds: int,
    time_budget_secs: int,
) -> float:
    """Estimate total cost for all experiments."""
    return estimate_cost_per_experiment(gpu_type, num_seeds, time_budget_secs) * len(configs)


# ---------------------------------------------------------------------------
# Plan printing
# ---------------------------------------------------------------------------


def print_cost_table(
    configs: list[dict],
    gpu_type: str,
    num_seeds: int,
    time_budget_secs: int,
    data_dir: Path,
) -> float:
    """Print a formatted cost estimate table. Returns total estimated cost."""
    resolved = GPU_ALIASES.get(gpu_type.lower(), gpu_type)
    rate = HOURLY_RATES.get(resolved, HOURLY_RATES.get(gpu_type, 0.0))
    per_exp = estimate_cost_per_experiment(gpu_type, num_seeds, time_budget_secs)
    total = per_exp * len(configs)

    setup_min = _SETUP_OVERHEAD_SECS // 60
    train_min = num_seeds * time_budget_secs // 60

    print(f"GPU Sweep Plan — {len(configs)} experiments on {resolved}")
    print(f"  Rate: ${rate:.2f}/hr  |  Setup: {setup_min}min  |  Training: {num_seeds}s x {time_budget_secs//60}min = {train_min}min")
    print(f"  Cost per experiment: ~${per_exp:.2f}")
    print()
    print(f"  {'#':<3} {'Name':<28} {'Data':<32} {'Data exists?':<14} {'Note'}")
    print(f"  {'-'*3} {'-'*28} {'-'*32} {'-'*14} {'-'*30}")

    for i, cfg in enumerate(configs, 1):
        train_path = data_dir / cfg["data_path"]
        exists_str = "yes" if train_path.exists() else "MISSING"
        note = cfg.get("note", "")
        print(f"  {i:<3} {cfg['name']:<28} {cfg['data_path']:<32} {exists_str:<14} {note}")

    print()
    print(f"  Total: {len(configs)} experiments x ${per_exp:.2f} = ~${total:.2f}")
    return total


# ---------------------------------------------------------------------------
# Subprocess dispatch
# ---------------------------------------------------------------------------


def dispatch_experiment(
    cfg: dict,
    *,
    gpu_type: str,
    num_seeds: int,
    time_budget_secs: int,
    data_dir: Path,
    budget_limit: float,
    dry_run: bool,
    results_csv: Path,
) -> dict:
    """Dispatch a single experiment via dispatch_rl_training.py.

    Returns a result dict with keys: name, exit_code, data_exists.
    """
    train_path = data_dir / cfg["data_path"]
    val_path = data_dir / cfg["val_path"]
    name = cfg["name"]
    data_exists = train_path.exists()

    result: dict = {
        "name": name,
        "data_path": str(train_path),
        "val_path": str(val_path),
        "exit_code": -1,
        "data_exists": data_exists,
        "note": cfg.get("note", ""),
    }

    if not data_exists:
        print(f"[sweep] SKIP {name}: training data not found: {train_path}")
        result["exit_code"] = 1
        return result

    run_id = f"sweep_{name}_{time.strftime('%Y%m%d_%H%M%S')}"
    leaderboard = str(REPO / "pufferlib_market" / f"sweep_{name}_leaderboard.csv")

    cmd = build_dispatch_cmd(
        name,
        train_data=str(train_path),
        val_data=str(val_path),
        run_id=run_id,
        num_seeds=num_seeds,
        gpu_type=gpu_type,
        budget_limit=budget_limit,
        time_budget=time_budget_secs,
        max_trials=50,
        leaderboard=leaderboard,
        dry_run=dry_run,
    )

    print(f"\n[sweep] Launching: {name}")
    print(f"  cmd: {' '.join(cmd)}")

    proc = subprocess.run(cmd, cwd=str(REPO))
    result["exit_code"] = proc.returncode
    result["leaderboard"] = leaderboard

    save_combined_leaderboard([result], results_csv, tag="sweep")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a diverse GPU sweep across 10 experiments on RTX 5090 pods.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print cost estimate and plan without launching any pods.",
    )
    parser.add_argument(
        "--gpu-type", default=DEFAULT_GPU_TYPE,
        help="GPU type alias (5090, a100, h100, 4090, …).",
    )
    parser.add_argument(
        "--budget-limit", type=float, default=DEFAULT_BUDGET_LIMIT,
        help="Max total USD for the full sweep. 0 = no limit.",
    )
    parser.add_argument(
        "--num-seeds", type=int, default=DEFAULT_NUM_SEEDS,
        help="Number of seeds per experiment.",
    )
    parser.add_argument(
        "--data-dir", default=DEFAULT_DATA_DIR, metavar="DIR",
        help="Base directory containing training/validation .bin files.",
    )
    parser.add_argument(
        "--time-budget", type=int, default=DEFAULT_TIME_BUDGET,
        help="Autoresearch time budget in seconds per trial.",
    )
    parser.add_argument(
        "--configs", default="", metavar="NAMES",
        help=(
            "Comma-separated experiment names to run (overrides DEFAULT_SWEEP_CONFIGS). "
            "Must match names in autoresearch_rl TRIAL_CONFIGS. "
            "Example: trade_pen_05,slip_5bps,cosine_lr_tp05"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = REPO / data_dir

    if args.configs:
        names = [n.strip() for n in args.configs.split(",") if n.strip()]
        if VALID_EXPERIMENT_NAMES:
            unknown = [n for n in names if n not in VALID_EXPERIMENT_NAMES]
            if unknown:
                print(f"[sweep] Error: unknown experiment name(s): {unknown}")
                print(f"  Valid names from autoresearch_rl TRIAL_CONFIGS: {sorted(VALID_EXPERIMENT_NAMES)}")
                return 1
        configs = [{"name": n, "data_path": "mixed23_daily_train.bin", "val_path": "mixed23_daily_val.bin"} for n in names]
    else:
        configs = DEFAULT_SWEEP_CONFIGS

    total = print_cost_table(configs, args.gpu_type, args.num_seeds, args.time_budget, data_dir)

    if args.budget_limit > 0 and total > args.budget_limit:
        print(f"[sweep] Estimated cost ${total:.2f} exceeds budget limit ${args.budget_limit:.2f}.")
        print(f"  Use --budget-limit {total + 1:.0f} to allow, or --budget-limit 0 to disable.")
        if args.dry_run:
            print("[sweep] (dry-run) Would abort here in live mode.")
        else:
            return 1

    if args.dry_run:
        print("[sweep] DRY RUN — no pods launched.")
        return 0

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_csv = REPO / "pufferlib_market" / f"sweep_results_{timestamp}.csv"
    print(f"\n[sweep] Results will be written to: {results_csv}")

    all_results = []
    for cfg in configs:
        result = dispatch_experiment(
            cfg,
            gpu_type=args.gpu_type,
            num_seeds=args.num_seeds,
            time_budget_secs=args.time_budget,
            data_dir=data_dir,
            budget_limit=args.budget_limit / max(len(configs), 1),
            dry_run=False,
            results_csv=results_csv,
        )
        all_results.append(result)

    succeeded = sum(1 for r in all_results if r["exit_code"] == 0)
    failed = len(all_results) - succeeded
    print(f"\n[sweep] Complete: {succeeded}/{len(all_results)} succeeded, {failed} failed.")
    print(f"[sweep] Results CSV: {results_csv}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
