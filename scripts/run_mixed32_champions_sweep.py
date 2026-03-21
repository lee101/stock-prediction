#!/usr/bin/env python3
"""Rerun top-5 daily configs on mixed32_latest + mixed23_latest data with 3 seeds.

Produces a baseline comparison leaderboard for the newest data by dispatching
each champion config through dispatch_rl_training.py.

Usage:
    # Dry-run to preview the plan
    python scripts/run_mixed32_champions_sweep.py --dry-run

    # Run on local GPU with budget guard
    python scripts/run_mixed32_champions_sweep.py --budget-limit 20

    # Run on specific remote GPU
    python scripts/run_mixed32_champions_sweep.py --gpu-type 5090 --budget-limit 50
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts._sweep_utils import (  # noqa: E402
    build_dispatch_cmd,
    merge_leaderboard_rows,
    resolve_data_paths,
    save_combined_leaderboard,
)

# Top-5 configs by val_sortino from autoresearch_daily_leaderboard.
CHAMPION_DESCRIPTIONS = [
    "trade_pen_05",
    "combined_smooth",
    "ent_anneal",
    "reg_combo_2",
    "clip_vloss",
]

OUTPUT_LEADERBOARD = "pufferlib_market/mixed32_champions_leaderboard.csv"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rerun top-5 daily configs on newest data with 3 seeds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-train", default="", help="Training .bin file (auto-detected if omitted)")
    p.add_argument("--data-val", default="", help="Validation .bin file (auto-detected if omitted)")
    p.add_argument("--num-seeds", type=int, default=3, help="Seeds per config")
    p.add_argument("--gpu-type", default="", help="GPU type for remote dispatch ('' = local)")
    p.add_argument("--budget-limit", type=float, default=20.0,
                   help="Max USD per config dispatch. 0 = no limit.")
    p.add_argument("--time-budget", type=int, default=300, help="Seconds per trial")
    p.add_argument("--max-trials", type=int, default=5,
                   help="Max trials per dispatch call")
    p.add_argument("--leaderboard", default=OUTPUT_LEADERBOARD,
                   help="Output combined leaderboard CSV path")
    p.add_argument("--dry-run", action="store_true", help="Print plan without executing")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    train_data, val_data = resolve_data_paths(args.data_train, args.data_val)
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    combined_rows: list[dict] = []

    print("Mixed32 Champions Sweep")
    print("=======================")
    print(f"Configs:    {len(CHAMPION_DESCRIPTIONS)}")
    print(f"Seeds:      {args.num_seeds}")
    print(f"Train data: {train_data}")
    print(f"Val data:   {val_data}")
    print(f"GPU type:   {args.gpu_type or 'local'}")
    print(f"Budget:     ${args.budget_limit:.2f} per config")
    print(f"Dry-run:    {args.dry_run}")
    print()

    for desc in CHAMPION_DESCRIPTIONS:
        run_id = f"mixed32_champ_{desc}_{run_stamp}"
        per_leaderboard = str(REPO / f"pufferlib_market/mixed32_champ_{desc}_leaderboard.csv")

        cmd = build_dispatch_cmd(
            desc,
            train_data=train_data,
            val_data=val_data,
            run_id=run_id,
            num_seeds=args.num_seeds,
            gpu_type=args.gpu_type,
            budget_limit=args.budget_limit,
            time_budget=args.time_budget,
            max_trials=args.max_trials,
            leaderboard=per_leaderboard,
            dry_run=args.dry_run,
        )

        print(f"--- {desc} ---")
        print(f"  cmd: {' '.join(cmd)}")

        if not args.dry_run:
            result = subprocess.run(cmd, cwd=str(REPO))
            if result.returncode != 0:
                print(f"[warning] dispatch returned exit code {result.returncode} for {desc}")
            merge_leaderboard_rows(Path(per_leaderboard), combined_rows)
        print()

    if not args.dry_run:
        save_combined_leaderboard(combined_rows, REPO / args.leaderboard, tag="champions")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
