#!/usr/bin/env python3
"""
Hyperparameter sweep for pufferlib trading.
Tests different configs and tracks best by validation Sortino.
"""
import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import itertools


@dataclass
class SweepConfig:
    name: str
    data_path: str
    hidden_size: int = 256
    alloc_bins: int = 5
    level_bins: int = 1
    lr: float = 3e-4
    downside_penalty: float = 2.0
    total_steps: int = 50_000_000
    arch: str = "resmlp"


SWEEP_CONFIGS = [
    # Baseline - current best
    SweepConfig("base_h256_a5", "pufferlib_market/data/stocks10_data.bin", 256, 5, 1, 3e-4, 2.0),
    # More allocation bins
    SweepConfig("h256_a10", "pufferlib_market/data/stocks10_data.bin", 256, 10, 1, 3e-4, 2.0),
    # Larger network
    SweepConfig("h512_a5", "pufferlib_market/data/stocks10_data.bin", 512, 5, 1, 1e-4, 2.0),
    # Lower LR
    SweepConfig("h256_a5_lr1e4", "pufferlib_market/data/stocks10_data.bin", 256, 5, 1, 1e-4, 2.0),
    # Higher downside penalty
    SweepConfig("h256_a5_dp4", "pufferlib_market/data/stocks10_data.bin", 256, 5, 1, 3e-4, 4.0),
    # Level bins for limit orders
    SweepConfig("h256_a5_l3", "pufferlib_market/data/stocks10_data.bin", 256, 5, 3, 3e-4, 2.0),
]


def run_training(cfg: SweepConfig, checkpoint_dir: str, gpu_id: int = 0) -> Dict[str, Any]:
    """Run a single training config."""
    cmd = [
        sys.executable, "-m", "pufferlib_market.train",
        "--data-path", cfg.data_path,
        "--checkpoint-dir", checkpoint_dir,
        "--hidden-size", str(cfg.hidden_size),
        "--action-allocation-bins", str(cfg.alloc_bins),
        "--action-level-bins", str(cfg.level_bins),
        "--lr", str(cfg.lr),
        "--downside-penalty", str(cfg.downside_penalty),
        "--total-timesteps", str(cfg.total_steps),
        "--arch", cfg.arch,
        "--reward-scale", "10.0",
        "--num-envs", "64",
    ]

    env = {"CUDA_VISIBLE_DEVICES": str(gpu_id)}

    print(f"\n{'='*60}")
    print(f"Starting: {cfg.name}")
    print(f"Config: h={cfg.hidden_size} a={cfg.alloc_bins} l={cfg.level_bins} lr={cfg.lr}")
    print(f"{'='*60}\n", flush=True)

    start = time.time()
    result = subprocess.run(cmd, env={**dict(__import__('os').environ), **env})
    elapsed = time.time() - start

    # Parse final results from checkpoint
    best_pt = Path(checkpoint_dir) / "best.pt"
    metrics = {"name": cfg.name, "elapsed": elapsed, "success": result.returncode == 0}

    if best_pt.exists():
        import torch
        ckpt = torch.load(best_pt, weights_only=True)
        metrics["best_return"] = ckpt.get("best_return", 0)

    return metrics


def run_sweep(configs: List[SweepConfig], base_dir: str, parallel: int = 1):
    """Run all configs, optionally in parallel on multiple GPUs."""
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, cfg in enumerate(configs):
        ckpt_dir = f"{base_dir}/{cfg.name}_{timestamp}"
        gpu_id = i % parallel

        metrics = run_training(cfg, ckpt_dir, gpu_id)
        results.append(metrics)

        # Save intermediate results
        with open(f"{base_dir}/sweep_results_{timestamp}.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n[{i+1}/{len(configs)}] {cfg.name}: return={metrics.get('best_return', 0):.4f}")

    # Print summary
    print("\n" + "="*60)
    print("SWEEP RESULTS")
    print("="*60)
    for r in sorted(results, key=lambda x: x.get("best_return", 0), reverse=True):
        print(f"{r['name']:20s} return={r.get('best_return', 0):+.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="experiments/sweep")
    parser.add_argument("--parallel", type=int, default=1, help="GPUs to use in parallel")
    parser.add_argument("--quick", action="store_true", help="Quick test with 1M steps")
    parser.add_argument("--configs", nargs="+", help="Specific configs to run")
    args = parser.parse_args()

    configs = SWEEP_CONFIGS
    if args.configs:
        configs = [c for c in SWEEP_CONFIGS if c.name in args.configs]

    if args.quick:
        for c in configs:
            c.total_steps = 1_000_000

    Path(args.base_dir).mkdir(parents=True, exist_ok=True)
    run_sweep(configs, args.base_dir, args.parallel)


if __name__ == "__main__":
    main()
