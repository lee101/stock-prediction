"""Targeted seed sweep for proven RL trading configs.

Runs the best-known config (tp05_slip5) across many seeds on a given dataset.
Supports both daily and hourly datasets.  Can be run locally (multi-GPU via
--gpu-ids) or dispatched to a RunPod pod via scripts/launch_runpod_scale.py.

Usage (local, one GPU):
    source .venv313/bin/activate
    python -m pufferlib_market.seed_sweep_rl \\
        --dataset crypto70_daily \\
        --seeds 1 100 \\
        --time-budget 300 \\
        --leaderboard sweepresults/crypto70_long_sweep.csv

Usage (local, specific GPU):
    CUDA_VISIBLE_DEVICES=1 python -m pufferlib_market.seed_sweep_rl \\
        --dataset crypto40_hourly --seeds 1 60 --time-budget 1800

Usage (on RunPod pod via launch_runpod_scale.py, called automatically):
    python -m pufferlib_market.seed_sweep_rl \\
        --dataset crypto70_daily --seeds 1 25 --time-budget 1800 \\
        --leaderboard pufferlib_market/c70_long_pod0.csv \\
        --checkpoint-root pufferlib_market/checkpoints/c70_long_pod0
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pufferlib_market.autoresearch_rl import TrialConfig, run_trial, build_config


# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------

DATASETS = {
    "crypto70_daily": {
        "train": "pufferlib_market/data/crypto70_daily_train.bin",
        "val": "pufferlib_market/data/crypto70_daily_val.bin",
        "periods_per_year": 365.0,
        "max_steps": 180,
        "fee_rate": 0.001,
        "holdout_eval_steps": 180,
        "holdout_n_windows": 20,
        "holdout_fill_buffer_bps": 5.0,
        # 48 sym × 677 steps × 500x = 16M step cap → ~10 min on H100
        "max_timesteps_per_sample": 500,
        "description": "48-sym Binance daily bars",
    },
    "crypto40_daily": {
        "train": "pufferlib_market/data/crypto40_daily_train.bin",
        "val": "pufferlib_market/data/crypto40_daily_val.bin",
        "periods_per_year": 365.0,
        "max_steps": 180,
        "fee_rate": 0.001,
        "holdout_eval_steps": 180,
        "holdout_n_windows": 10,
        "holdout_fill_buffer_bps": 5.0,
        # 25 sym × 797 steps × 500x = 9.96M step cap
        "max_timesteps_per_sample": 500,
        "description": "25-sym Binance daily bars (more train data: 797 vs 677 steps)",
    },
    "crypto40_hourly": {
        "train": "pufferlib_market/data/crypto40_hourly_train.bin",
        "val": "pufferlib_market/data/crypto40_hourly_val.bin",
        "periods_per_year": 8760.0,
        "max_steps": 720,
        "fee_rate": 0.001,
        "holdout_eval_steps": 720,
        "holdout_n_windows": 30,   # 4901 val steps / 720 per window = 6.8 non-overlapping
        "holdout_fill_buffer_bps": 5.0,
        # 25 sym × 18448 steps × 100x = 46M step cap → ~28 min on H100 with 128 envs
        "max_timesteps_per_sample": 100,
        "description": "25-sym Binance hourly bars (18448 train / 4901 val steps)",
    },
    "crypto34_hourly": {
        "train": "pufferlib_market/data/crypto34_hourly_train.bin",
        "val": "pufferlib_market/data/crypto34_hourly_val.bin",
        "periods_per_year": 8760.0,
        "max_steps": 720,
        "fee_rate": 0.001,
        "holdout_eval_steps": 720,
        "holdout_n_windows": 20,
        "holdout_fill_buffer_bps": 5.0,
        # 34 sym × 12685 steps × 100x = 43M step cap
        "max_timesteps_per_sample": 100,
        "description": "34-sym Binance hourly bars",
    },
}


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def make_seed_config(seed: int, dataset_key: str, overrides: dict) -> TrialConfig:
    """Build a TrialConfig for the proven tp05_slip5 base config with given seed."""
    ds = DATASETS[dataset_key]
    base = {
        "hidden_size": 1024,
        "lr": 3e-4,
        "anneal_lr": True,
        "ent_coef": 0.05,
        "trade_penalty": 0.05,
        "fill_slippage_bps": 5.0,
        "num_envs": 128,
        "use_bf16": True,
        "no_cuda_graph": True,
        "periods_per_year": ds["periods_per_year"],
        "max_steps": ds["max_steps"],
        "fee_rate": ds["fee_rate"],
        "seed": seed,
        "description": f"{dataset_key}_tp05_s{seed}",
    }
    base.update(overrides)
    return build_config(base)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

LEADERBOARD_FIELDS = [
    "description", "dataset", "seed", "val_return", "val_sortino", "val_wr",
    "holdout_median_return_pct", "holdout_p10_return_pct", "holdout_negative_rate",
    "holdout_worst_return_pct", "elapsed_s", "checkpoint_dir", "timestamp",
]


def _append_result(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=LEADERBOARD_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Main sweep loop
# ---------------------------------------------------------------------------

def run_sweep(
    dataset_key: str,
    seeds: list[int],
    time_budget: int,
    leaderboard: Path,
    checkpoint_root: Path,
    overrides: dict,
) -> None:
    ds = DATASETS[dataset_key]
    train_data = str(REPO / ds["train"])
    val_data = str(REPO / ds["val"])

    print(f"Seed sweep: {dataset_key}  seeds={seeds[0]}-{seeds[-1]}  budget={time_budget}s/seed")
    print(f"  train: {train_data}")
    print(f"  val:   {val_data}")
    print(f"  leaderboard: {leaderboard}")
    print(f"  n_seeds: {len(seeds)}")
    print(flush=True)

    # Skip seeds already in leaderboard
    done_seeds: set[int] = set()
    if leaderboard.exists():
        with open(leaderboard) as fh:
            for row in csv.DictReader(fh):
                try:
                    done_seeds.add(int(row["seed"]))
                except (KeyError, ValueError):
                    pass
        if done_seeds:
            print(f"  Skipping {len(done_seeds)} already-done seeds: {sorted(done_seeds)}")

    for i, seed in enumerate(seeds):
        if seed in done_seeds:
            continue

        config = make_seed_config(seed, dataset_key, overrides)
        ckpt_dir = str(checkpoint_root / config.description)
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

        print(f"\n[{i+1}/{len(seeds)}] seed={seed}  desc={config.description}", flush=True)
        t0 = time.time()

        result = run_trial(
            config=config,
            train_data=train_data,
            val_data=val_data,
            time_budget=time_budget,
            checkpoint_dir=ckpt_dir,
            holdout_data=val_data,
            holdout_eval_steps=ds["holdout_eval_steps"],
            holdout_n_windows=ds["holdout_n_windows"],
            holdout_fill_buffer_bps=ds["holdout_fill_buffer_bps"],
            rank_metric="val_return",
            use_poly_prune=False,    # don't prune in seed sweeps
            max_timesteps_per_sample=ds.get("max_timesteps_per_sample", 1000),
        )

        elapsed = time.time() - t0
        val_return = result.get("val_return", None)
        val_sortino = result.get("val_sortino", None)
        val_wr = result.get("val_wr", None)
        # run_trial returns holdout metrics as top-level keys (not nested under "holdout_metrics")
        hmed = result.get("holdout_median_return_pct", "")
        hp10 = result.get("holdout_p10_return_pct", "")
        hneg = result.get("holdout_negative_return_rate", "")
        hwst = result.get("holdout_return_worst_pct", "")

        row = {
            "description": config.description,
            "dataset": dataset_key,
            "seed": seed,
            "val_return": f"{val_return:.4f}" if val_return is not None else "",
            "val_sortino": f"{val_sortino:.2f}" if val_sortino is not None else "",
            "val_wr": f"{val_wr:.4f}" if val_wr is not None else "",
            "holdout_median_return_pct": hmed,
            "holdout_p10_return_pct": hp10,
            "holdout_negative_rate": hneg,
            "holdout_worst_return_pct": hwst,
            "elapsed_s": f"{elapsed:.1f}",
            "checkpoint_dir": ckpt_dir,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        _append_result(leaderboard, row)

        status = "OK" if result.get("error") is None else "ERR"
        print(f"  [{status}] val_ret={val_return}  val_sort={val_sortino}  "
              f"holdout_med={hmed or '?'}  "
              f"elapsed={elapsed:.0f}s", flush=True)

    print(f"\nSweep complete. Results in {leaderboard}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Targeted seed sweep for RL trading")
    parser.add_argument("--dataset", required=True, choices=list(DATASETS),
                        help="Dataset key (e.g. crypto70_daily, crypto40_hourly)")
    parser.add_argument("--seeds", nargs=2, type=int, metavar=("START", "END"),
                        default=[1, 60],
                        help="Inclusive seed range to sweep [start end]")
    parser.add_argument("--seed-list", nargs="+", type=int, default=None,
                        help="Explicit list of seeds (overrides --seeds range)")
    parser.add_argument("--time-budget", type=int, default=1800,
                        help="Training time budget per seed in seconds (default 1800=30min)")
    parser.add_argument("--leaderboard", type=str, default=None,
                        help="Output CSV path (default: sweepresults/{dataset}_seedsweep.csv)")
    parser.add_argument("--checkpoint-root", type=str, default=None,
                        help="Checkpoint root dir (default: pufferlib_market/checkpoints/{dataset}_seedsweep)")
    # Config overrides
    parser.add_argument("--trade-penalty", type=float, default=0.05)
    parser.add_argument("--fill-slippage-bps", type=float, default=5.0)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--ent-coef", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--use-bf16", action="store_true", default=True)
    parser.add_argument("--no-cuda-graph", action="store_true", default=False)
    parser.add_argument("--cuda-graph", action="store_true", default=False,
                        help="Enable CUDA graph PPO (safe on exclusive GPU like H100 RunPod pods)")
    args = parser.parse_args()

    if args.seed_list:
        seeds = args.seed_list
    else:
        seeds = list(range(args.seeds[0], args.seeds[1] + 1))

    leaderboard = Path(args.leaderboard) if args.leaderboard else \
        REPO / "sweepresults" / f"{args.dataset}_seedsweep.csv"
    checkpoint_root = Path(args.checkpoint_root) if args.checkpoint_root else \
        REPO / "pufferlib_market" / "checkpoints" / f"{args.dataset}_seedsweep"

    # --cuda-graph enables it; --no-cuda-graph disables; default is no-cuda-graph
    # (safe for shared GPUs but slower; H100 RunPod pods should pass --cuda-graph)
    use_cuda_graph = args.cuda_graph and not args.no_cuda_graph

    overrides = {
        "trade_penalty": args.trade_penalty,
        "fill_slippage_bps": args.fill_slippage_bps,
        "hidden_size": args.hidden_size,
        "ent_coef": args.ent_coef,
        "lr": args.lr,
        "num_envs": args.num_envs,
        "use_bf16": args.use_bf16,
        "no_cuda_graph": not use_cuda_graph,
        "cuda_graph_ppo": use_cuda_graph,
    }

    run_sweep(
        dataset_key=args.dataset,
        seeds=seeds,
        time_budget=args.time_budget,
        leaderboard=leaderboard,
        checkpoint_root=checkpoint_root,
        overrides=overrides,
    )


if __name__ == "__main__":
    main()
