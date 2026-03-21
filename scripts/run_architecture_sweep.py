#!/usr/bin/env python3
"""Sweep new architecture configs (transformer, GRU, depth-recurrence, relu_sq).

Smaller models run faster than h1024 MLP, so these can be evaluated on an
RTX 5090 in a single dispatch batch.  The sweep uses the same autoresearch
loop as the champions sweep but targets the architecture-specific experiment
descriptions added to EXPERIMENTS in autoresearch_rl.py.

Usage:
    # Dry-run preview
    python scripts/run_architecture_sweep.py --dry-run

    # Run locally
    python scripts/run_architecture_sweep.py

    # Run on RTX 5090 pod
    python scripts/run_architecture_sweep.py --gpu-type 5090 --budget-limit 30
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

# Each entry: (group_label, [description, ...])
# Configs within a group are dispatched together to reduce pod-provision overhead.
ARCH_GROUPS: list[tuple[str, list[str]]] = [
    ("transformer",      ["transformer_h256", "transformer_h256_tp05"]),
    ("gru",              ["gru_h256", "gru_h512"]),
    ("depth_recurrence", ["depth_recur_h512", "depth_recur_h1024"]),
    ("relu_sq",          ["relu_sq_h1024", "relu_sq_tp05"]),
    ("resmlp",           ["resmlp", "resmlp_wd"]),
]

OUTPUT_LEADERBOARD = "pufferlib_market/architecture_sweep_leaderboard.csv"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep new architecture configs on daily RL data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-train", default="", help="Training .bin file (auto-detected if omitted)")
    p.add_argument("--data-val", default="", help="Validation .bin file (auto-detected if omitted)")
    p.add_argument("--num-seeds", type=int, default=1, help="Seeds per config group")
    p.add_argument("--gpu-type", default="", help="GPU type for remote dispatch ('' = local)")
    p.add_argument("--budget-limit", type=float, default=30.0,
                   help="Max USD per dispatch group. 0 = no limit.")
    p.add_argument("--time-budget", type=int, default=300, help="Seconds per trial")
    p.add_argument(
        "--no-batch", dest="batch", action="store_false", default=True,
        help="Run each config as a separate dispatch call instead of grouping by architecture",
    )
    p.add_argument("--leaderboard", default=OUTPUT_LEADERBOARD,
                   help="Output combined leaderboard CSV path")
    p.add_argument("--dry-run", action="store_true", help="Print plan without executing")
    return p.parse_args(argv)


def _resolve_groups(*, batch: bool) -> list[tuple[str, list[str]]]:
    if batch:
        return ARCH_GROUPS
    return [(desc, [desc]) for _, descs in ARCH_GROUPS for desc in descs]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    train_data, val_data = resolve_data_paths(args.data_train, args.data_val)
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    combined_rows: list[dict] = []

    groups = _resolve_groups(batch=args.batch)
    total_configs = sum(len(descs) for _, descs in groups)

    print("Architecture Sweep")
    print("==================")
    print(f"Total configs:   {total_configs}")
    print(f"Dispatch groups: {len(groups)}")
    print(f"Seeds per group: {args.num_seeds}")
    print(f"Train data:  {train_data}")
    print(f"Val data:    {val_data}")
    print(f"GPU type:    {args.gpu_type or 'local'}")
    print(f"Budget:      ${args.budget_limit:.2f} per group")
    print(f"Dry-run:     {args.dry_run}")
    print()

    for group_label, descs in groups:
        descriptions_csv = ",".join(descs)
        run_id = f"arch_sweep_{group_label}_{run_stamp}"
        per_leaderboard = str(REPO / f"pufferlib_market/arch_sweep_{group_label}_leaderboard.csv")

        cmd = build_dispatch_cmd(
            descriptions_csv,
            train_data=train_data,
            val_data=val_data,
            run_id=run_id,
            num_seeds=args.num_seeds,
            gpu_type=args.gpu_type,
            budget_limit=args.budget_limit,
            time_budget=args.time_budget,
            max_trials=len(descs) * args.num_seeds,
            leaderboard=per_leaderboard,
            dry_run=args.dry_run,
        )

        print(f"--- Group: {group_label} ({len(descs)} configs) ---")
        print(f"  descriptions: {descriptions_csv}")
        print(f"  cmd: {' '.join(cmd)}")

        if not args.dry_run:
            result = subprocess.run(cmd, cwd=str(REPO))
            if result.returncode != 0:
                print(f"[warning] dispatch returned exit code {result.returncode} for group={group_label}")
            merge_leaderboard_rows(Path(per_leaderboard), combined_rows)
        print()

    if not args.dry_run:
        save_combined_leaderboard(combined_rows, REPO / args.leaderboard, tag="arch-sweep")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
