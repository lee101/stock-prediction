#!/usr/bin/env python3
"""Launch large model experiments (h2048/h4096) on A100 or H100 pods.

These architectures require more VRAM than consumer GPUs.
  h2048 needs ~16 GB VRAM — suitable for A100 80 GB PCIe.
  h4096 needs ~40 GB VRAM — suitable for A100/H100.

Usage:
  python scripts/launch_a100_large_models.py --dry-run
  python scripts/launch_a100_large_models.py --gpu-type a100 --budget-limit 25
  python scripts/launch_a100_large_models.py --gpu-type h100 --configs h4096_anneal
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_GPU_TYPE = "a100"
DEFAULT_BUDGET_LIMIT = 25.0
DEFAULT_NUM_SEEDS = 3
DEFAULT_TIME_BUDGET = 300  # 5 min per trial
DEFAULT_DATA_DIR = "pufferlib_market/data"

# Mirrors dispatch_rl_training._SETUP_OVERHEAD_SECS
_SETUP_OVERHEAD_SECS = 1800

# ---------------------------------------------------------------------------
# Large model configs — require A100 or H100
# ---------------------------------------------------------------------------

LARGE_MODEL_CONFIGS: list[dict] = [
    {
        "name": "h2048_anneal",
        "data_path": "mixed23_daily_train.bin",
        "val_path": "mixed23_daily_val.bin",
        "min_gpu": "a100",
        "vram_note": "~16 GB",
        "note": "h2048 scaling from h1024 baseline",
    },
    {
        "name": "h2048_anneal_tp05",
        "data_path": "mixed23_daily_train.bin",
        "val_path": "mixed23_daily_val.bin",
        "min_gpu": "a100",
        "vram_note": "~16 GB",
        "note": "h2048 + trade_pen_05 (best daily config)",
    },
    {
        "name": "h2048_resmlp_anneal",
        "data_path": "mixed23_daily_train.bin",
        "val_path": "mixed23_daily_val.bin",
        "min_gpu": "a100",
        "vram_note": "~16 GB",
        "note": "h2048 ResidualMLP variant",
    },
    {
        "name": "h4096_anneal",
        "data_path": "mixed23_daily_train.bin",
        "val_path": "mixed23_daily_val.bin",
        "min_gpu": "h100",
        "vram_note": "~40 GB",
        "note": "h4096 max-scale (H100 preferred)",
    },
]

VALID_LARGE_MODEL_NAMES: frozenset[str] = frozenset(c["name"] for c in LARGE_MODEL_CONFIGS)


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------


def estimate_cost_per_experiment(gpu_type: str, num_seeds: int, time_budget_secs: int) -> float:
    """Estimate USD cost for one experiment (all seeds) including setup overhead."""
    resolved = GPU_ALIASES.get(gpu_type.lower(), gpu_type)
    rate = HOURLY_RATES.get(resolved, HOURLY_RATES.get(gpu_type, 0.0))
    total_secs = _SETUP_OVERHEAD_SECS + num_seeds * time_budget_secs
    return rate * (total_secs / 3600)


def estimate_total_cost(configs: list[dict], gpu_type: str, num_seeds: int, time_budget_secs: int) -> float:
    """Estimate total cost across all experiments."""
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
    """Print formatted cost estimate table. Returns total estimated cost."""
    resolved = GPU_ALIASES.get(gpu_type.lower(), gpu_type)
    rate = HOURLY_RATES.get(resolved, HOURLY_RATES.get(gpu_type, 0.0))
    per_exp = estimate_cost_per_experiment(gpu_type, num_seeds, time_budget_secs)
    total = per_exp * len(configs)

    setup_min = _SETUP_OVERHEAD_SECS // 60
    train_min = num_seeds * time_budget_secs // 60

    print(f"A100/H100 Large Model Sweep — {len(configs)} experiments on {resolved}")
    print(f"  Rate: ${rate:.2f}/hr  |  Setup: {setup_min}min  |  Training: {num_seeds}s x {time_budget_secs//60}min = {train_min}min")
    print(f"  Cost per experiment: ~${per_exp:.2f}")
    print()
    print(f"  {'#':<3} {'Name':<28} {'Min GPU':<8} {'VRAM':<10} {'Data exists?':<14} {'Note'}")
    print(f"  {'-'*3} {'-'*28} {'-'*8} {'-'*10} {'-'*14} {'-'*35}")

    for i, cfg in enumerate(configs, 1):
        train_path = data_dir / cfg["data_path"]
        exists_str = "yes" if train_path.exists() else "MISSING"
        min_gpu = cfg.get("min_gpu", "a100")
        vram_note = cfg.get("vram_note", "")
        note = cfg.get("note", "")
        gpu_warn = " [H100 preferred]" if min_gpu == "h100" and gpu_type.lower() not in ("h100", "h100-sxm") else ""
        print(f"  {i:<3} {cfg['name']:<28} {min_gpu:<8} {vram_note:<10} {exists_str:<14} {note}{gpu_warn}")

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
    """Dispatch a single experiment via dispatch_rl_training.py."""
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
        "min_gpu": cfg.get("min_gpu", "a100"),
    }

    if not data_exists:
        print(f"[a100-sweep] SKIP {name}: training data not found: {train_path}")
        result["exit_code"] = 1
        return result

    run_id = f"a100_large_{name}_{time.strftime('%Y%m%d_%H%M%S')}"
    leaderboard = str(REPO / "pufferlib_market" / f"a100_large_{name}_leaderboard.csv")

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

    print(f"\n[a100-sweep] Launching: {name} (gpu={gpu_type}, min_gpu={cfg.get('min_gpu', 'a100')})")
    print(f"  cmd: {' '.join(cmd)}")

    proc = subprocess.run(cmd, cwd=str(REPO))
    result["exit_code"] = proc.returncode
    result["leaderboard"] = leaderboard

    save_combined_leaderboard([result], results_csv, tag="a100-sweep")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch large model experiments (h2048/h4096) on A100 or H100 pods.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print cost estimate and plan without launching any pods.",
    )
    parser.add_argument(
        "--gpu-type", default=DEFAULT_GPU_TYPE,
        help="GPU type alias (a100, a100-sxm, h100, h100-sxm).",
    )
    parser.add_argument(
        "--budget-limit", type=float, default=DEFAULT_BUDGET_LIMIT,
        help="Max total USD for this sweep. 0 = no limit.",
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
            "Comma-separated experiment names to run (overrides defaults). "
            f"Valid: {', '.join(sorted(VALID_LARGE_MODEL_NAMES))}. "
            "Example: h2048_anneal,h2048_anneal_tp05"
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
        unknown = [n for n in names if n not in VALID_LARGE_MODEL_NAMES]
        if unknown:
            print(f"[a100-sweep] Error: unknown large model config(s): {unknown}")
            print(f"  Valid names: {sorted(VALID_LARGE_MODEL_NAMES)}")
            return 1
        name_to_cfg = {c["name"]: c for c in LARGE_MODEL_CONFIGS}
        configs = [name_to_cfg[n] for n in names]
    else:
        configs = LARGE_MODEL_CONFIGS

    has_h4096 = any(c["name"] == "h4096_anneal" for c in configs)
    if has_h4096 and args.gpu_type.lower() not in ("h100", "h100-sxm"):
        print(f"[a100-sweep] Warning: h4096_anneal prefers H100 (requested: {args.gpu_type}).")
        print("  If A100 OOMs, re-run with --configs h2048_anneal,h2048_anneal_tp05,h2048_resmlp_anneal")
        print()

    total = print_cost_table(configs, args.gpu_type, args.num_seeds, args.time_budget, data_dir)

    if args.budget_limit > 0 and total > args.budget_limit:
        print(f"[a100-sweep] Estimated cost ${total:.2f} exceeds budget limit ${args.budget_limit:.2f}.")
        print(f"  Use --budget-limit {total + 1:.0f} to allow, or --budget-limit 0 to disable.")
        if args.dry_run:
            print("[a100-sweep] (dry-run) Would abort here in live mode.")
        else:
            return 1

    if args.dry_run:
        print("[a100-sweep] DRY RUN — no pods launched.")
        return 0

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_csv = REPO / "pufferlib_market" / f"a100_large_results_{timestamp}.csv"
    print(f"\n[a100-sweep] Results will be written to: {results_csv}")

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
    print(f"\n[a100-sweep] Complete: {succeeded}/{len(all_results)} succeeded, {failed} failed.")
    print(f"[a100-sweep] Results CSV: {results_csv}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
