"""Auto-research improvement loop for PufferLib PPO trading.

Runs time-boxed training experiments sweeping hyperparameters,
evaluates each on validation data, and tracks the best configuration.

Usage:
  python -m pufferlib_market.auto_improve --data-path pufferlib_market/data/crypto12_data.bin \
      --budget-minutes 5 --num-trials 8
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


@dataclass
class TrialConfig:
    """One hyperparameter configuration to test."""
    lr: float = 3e-4
    ent_coef: float = 0.08
    ent_coef_end: float = 0.02
    clip_eps: float = 0.2
    clip_eps_end: float = 0.05
    weight_decay: float = 0.005
    lr_warmup_frac: float = 0.02
    lr_min_ratio: float = 0.05
    hidden_size: int = 1024
    num_envs: int = 64
    rollout_len: int = 256
    minibatch_size: int = 2048
    ppo_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    reward_scale: float = 10.0
    cash_penalty: float = 0.01
    obs_norm: bool = True
    lr_schedule: str = "cosine"
    anneal_ent: bool = True
    anneal_clip: bool = True
    clip_vloss: bool = True
    arch: str = "mlp"


# Known good baseline
BASELINE = TrialConfig()


def mutate_config(base: TrialConfig, exploration: float = 0.3) -> TrialConfig:
    """Create a mutated config from a base, varying some parameters."""
    import copy
    cfg = copy.deepcopy(base)

    # Each parameter has a probability of being mutated
    if random.random() < exploration:
        cfg.lr = random.choice([1e-4, 2e-4, 3e-4, 5e-4, 8e-4, 1e-3, 2e-3])
    if random.random() < exploration:
        cfg.ent_coef = random.choice([0.03, 0.05, 0.08, 0.10, 0.15])
    if random.random() < exploration:
        cfg.ent_coef_end = random.choice([0.005, 0.01, 0.02, 0.03])
    if random.random() < exploration:
        cfg.clip_eps = random.choice([0.1, 0.15, 0.2, 0.25, 0.3])
    if random.random() < exploration:
        cfg.clip_eps_end = random.choice([0.02, 0.05, 0.08, 0.1])
    if random.random() < exploration:
        cfg.weight_decay = random.choice([0.0, 0.001, 0.005, 0.01, 0.02])
    if random.random() < exploration:
        cfg.lr_warmup_frac = random.choice([0.01, 0.02, 0.05, 0.1])
    if random.random() < exploration:
        cfg.lr_min_ratio = random.choice([0.01, 0.05, 0.1, 0.2])
    if random.random() < exploration:
        cfg.num_envs = random.choice([32, 64, 128, 256])
    if random.random() < exploration:
        cfg.rollout_len = random.choice([128, 256, 512])
    if random.random() < exploration:
        cfg.ppo_epochs = random.choice([3, 4, 6, 8])
    if random.random() < exploration:
        cfg.gamma = random.choice([0.98, 0.99, 0.995])
    if random.random() < exploration:
        cfg.reward_scale = random.choice([5.0, 10.0, 20.0, 50.0])
    if random.random() < exploration:
        cfg.cash_penalty = random.choice([0.0, 0.005, 0.01, 0.02])
    if random.random() < exploration:
        cfg.obs_norm = random.choice([True, False])
    if random.random() < exploration:
        cfg.lr_schedule = random.choice(["cosine", "none"])
    if random.random() < exploration:
        cfg.anneal_ent = random.choice([True, False])
    if random.random() < exploration:
        cfg.anneal_clip = random.choice([True, False])
    if random.random() < exploration:
        cfg.clip_vloss = random.choice([True, False])
    if random.random() < exploration:
        cfg.arch = random.choice(["mlp", "resmlp"])

    return cfg


def run_trial(
    cfg: TrialConfig,
    data_path: str,
    total_timesteps: int,
    max_steps: int = 720,
    timeout_seconds: int = 300,
    trial_id: int = 0,
) -> dict:
    """Run a single training trial and return metrics."""
    ckpt_dir = f"/tmp/ppo_autoimprove_trial_{trial_id}_{int(time.time())}"

    cmd = [
        sys.executable, "-u", "-m", "pufferlib_market.train",
        "--data-path", data_path,
        "--max-steps", str(max_steps),
        "--total-timesteps", str(total_timesteps),
        "--hidden-size", str(cfg.hidden_size),
        "--lr", str(cfg.lr),
        "--ent-coef", str(cfg.ent_coef),
        "--num-envs", str(cfg.num_envs),
        "--rollout-len", str(cfg.rollout_len),
        "--minibatch-size", str(cfg.minibatch_size),
        "--ppo-epochs", str(cfg.ppo_epochs),
        "--gamma", str(cfg.gamma),
        "--gae-lambda", str(cfg.gae_lambda),
        "--reward-scale", str(cfg.reward_scale),
        "--cash-penalty", str(cfg.cash_penalty),
        "--clip-eps", str(cfg.clip_eps),
        "--weight-decay", str(cfg.weight_decay),
        "--checkpoint-dir", ckpt_dir,
        "--save-every", "99999",
        "--arch", cfg.arch,
    ]

    if cfg.lr_schedule != "none":
        cmd += ["--lr-schedule", cfg.lr_schedule,
                "--lr-warmup-frac", str(cfg.lr_warmup_frac),
                "--lr-min-ratio", str(cfg.lr_min_ratio)]
    if cfg.anneal_ent:
        cmd += ["--anneal-ent", "--ent-coef-end", str(cfg.ent_coef_end)]
    if cfg.anneal_clip:
        cmd += ["--anneal-clip", "--clip-eps-end", str(cfg.clip_eps_end)]
    if cfg.clip_vloss:
        cmd += ["--clip-vloss"]
    if cfg.obs_norm:
        cmd += ["--obs-norm"]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=str(REPO),
        )
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        output = ""

    # Parse best return from output
    best_return = -float("inf")
    best_sortino = 0.0
    best_wr = 0.0
    last_sps = 0.0

    for line in output.split("\n"):
        if "ret=" in line:
            try:
                # Extract ret=X.XXXX
                ret_str = line.split("ret=")[1].split()[0]
                ret = float(ret_str)
                if ret > best_return:
                    best_return = ret
                # Extract sortino
                if "sortino=" in line:
                    best_sortino = float(line.split("sortino=")[1].split()[0])
                if "wr=" in line:
                    best_wr = float(line.split("wr=")[1].split()[0])
            except (ValueError, IndexError):
                pass
        if "sps=" in line:
            try:
                last_sps = float(line.split("sps=")[1].split()[0])
            except (ValueError, IndexError):
                pass

    # Cleanup checkpoint dir
    import shutil
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir, ignore_errors=True)

    return {
        "trial_id": trial_id,
        "best_return": best_return,
        "best_sortino": best_sortino,
        "best_wr": best_wr,
        "sps": last_sps,
        "config": asdict(cfg),
    }


def main():
    parser = argparse.ArgumentParser(description="Auto-research improvement loop for PPO")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--budget-minutes", type=float, default=5.0,
                        help="Time budget per trial in minutes")
    parser.add_argument("--num-trials", type=int, default=8)
    parser.add_argument("--total-timesteps", type=int, default=5_000_000,
                        help="Steps per trial")
    parser.add_argument("--max-steps", type=int, default=720)
    parser.add_argument("--exploration", type=float, default=0.3,
                        help="Probability of mutating each parameter")
    parser.add_argument("--output", type=str, default="pufferlib_market/auto_improve_results.json")
    args = parser.parse_args()

    timeout = int(args.budget_minutes * 60)
    results = []
    best_config = BASELINE
    best_return = -float("inf")

    print(f"Auto-improve: {args.num_trials} trials, {args.budget_minutes}min budget each")
    print(f"Data: {args.data_path}")
    print(f"Steps per trial: {args.total_timesteps:,}")
    print()

    for trial in range(args.num_trials):
        if trial == 0:
            cfg = BASELINE  # First trial is always baseline
        else:
            cfg = mutate_config(best_config, exploration=args.exploration)

        print(f"\n{'=' * 60}")
        print(f"Trial {trial + 1}/{args.num_trials}")
        key_params = {k: v for k, v in asdict(cfg).items()
                      if v != asdict(BASELINE).get(k) or trial == 0}
        if trial == 0:
            key_params = {"baseline": True}
        print(f"Config changes: {key_params}")
        print(f"{'=' * 60}")

        result = run_trial(
            cfg, args.data_path, args.total_timesteps,
            max_steps=args.max_steps,
            timeout_seconds=timeout,
            trial_id=trial,
        )
        results.append(result)

        print(f"  Return: {result['best_return']:+.4f}")
        print(f"  Sortino: {result['best_sortino']:.2f}")
        print(f"  Win Rate: {result['best_wr']:.2f}")
        print(f"  SPS: {result['sps']:.0f}")

        if result["best_return"] > best_return:
            best_return = result["best_return"]
            best_config = cfg
            print(f"  *** NEW BEST ***")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "results": results,
            "best_config": asdict(best_config),
            "best_return": best_return,
        }, f, indent=2)

    # Summary
    print(f"\n{'=' * 60}")
    print("LEADERBOARD")
    print(f"{'=' * 60}")
    sorted_results = sorted(results, key=lambda r: r["best_return"], reverse=True)
    for i, r in enumerate(sorted_results[:5]):
        changes = {k: v for k, v in r["config"].items()
                   if v != asdict(BASELINE).get(k)}
        print(f"  #{i+1}: ret={r['best_return']:+.4f} sortino={r['best_sortino']:.1f} "
              f"wr={r['best_wr']:.2f} | {changes}")

    print(f"\nBest config saved to {args.output}")
    print(f"Best return: {best_return:+.4f}")


if __name__ == "__main__":
    main()
