"""Auto-research loop for Qwen GRPO trading plan training.

Runs timeboxed experiments, evaluates on holdout, tracks in leaderboard CSV.

Usage:
    python -m qwen_rl_trading.autoresearch_qwen --max-trials 10 --time-budget 600
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

log = logging.getLogger("qwen_autoresearch")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from .sweep_configs import EXPERIMENTS, QwenTrialConfig, build_trial_config
from .train_grpo import QwenGRPOConfig, train as train_grpo

LEADERBOARD_COLUMNS = [
    "description", "model_size", "lora_r", "group_size", "lr", "kl_coef",
    "n_symbols", "prompt_variant", "reward_type", "sft_warmstart",
    "val_mean_reward", "val_mean_sortino", "val_mean_return", "val_valid_json_pct",
    "training_time_s", "best_checkpoint", "seed", "error",
]


def _trial_config_to_grpo_config(tc: QwenTrialConfig, time_budget: int, output_dir: str) -> QwenGRPOConfig:
    return QwenGRPOConfig(
        model_size=tc.model_size,
        lora_r=tc.lora_r,
        lora_alpha=tc.lora_alpha,
        group_size=tc.group_size,
        lr=tc.lr,
        kl_coef=tc.kl_coef,
        max_completion_length=tc.max_completion_length,
        n_symbols=tc.n_symbols,
        reward_type=tc.reward_type,
        prompt_variant=tc.prompt_variant,
        sft_warmstart=tc.sft_warmstart,
        eval_horizon_hours=tc.eval_horizon_hours,
        seed=tc.seed,
        description=tc.description,
        time_budget=time_budget,
        output_dir=output_dir,
    )


def run_trial(trial_config: QwenTrialConfig, time_budget: int, checkpoint_root: Path) -> dict:
    """Run a single GRPO training trial within time_budget seconds."""
    output_dir = checkpoint_root / trial_config.description
    output_dir.mkdir(parents=True, exist_ok=True)

    grpo_config = _trial_config_to_grpo_config(trial_config, time_budget, str(output_dir))

    try:
        result = train_grpo(grpo_config)
        return result
    except Exception as e:
        log.error("trial %s failed: %s", trial_config.description, e)
        import traceback
        traceback.print_exc()
        return {
            "description": trial_config.description,
            "model_size": trial_config.model_size,
            "error": str(e),
            "training_time_s": 0,
            "val_mean_reward": None,
            "val_mean_sortino": None,
            "val_mean_return": None,
            "val_valid_json_pct": None,
            "best_checkpoint": None,
            "config": asdict(trial_config),
        }


def _append_leaderboard(path: Path, result: dict, trial_config: QwenTrialConfig):
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LEADERBOARD_COLUMNS)
        if write_header:
            writer.writeheader()
        row = {
            "description": result.get("description", trial_config.description),
            "model_size": trial_config.model_size,
            "lora_r": trial_config.lora_r,
            "group_size": trial_config.group_size,
            "lr": trial_config.lr,
            "kl_coef": trial_config.kl_coef,
            "n_symbols": trial_config.n_symbols,
            "prompt_variant": trial_config.prompt_variant,
            "reward_type": trial_config.reward_type,
            "sft_warmstart": trial_config.sft_warmstart,
            "val_mean_reward": result.get("val_mean_reward"),
            "val_mean_sortino": result.get("val_mean_sortino"),
            "val_mean_return": result.get("val_mean_return"),
            "val_valid_json_pct": result.get("val_valid_json_pct"),
            "training_time_s": result.get("training_time_s", 0),
            "best_checkpoint": result.get("best_checkpoint"),
            "seed": trial_config.seed,
            "error": result.get("error"),
        }
        writer.writerow(row)


def run_autoresearch(
    experiments: list[dict],
    time_budget: int = 600,
    max_trials: int = 30,
    checkpoint_root: Optional[Path] = None,
    leaderboard_path: Optional[Path] = None,
    descriptions: str = "",
):
    """Iterate through experiments, run trials, update leaderboard."""
    if checkpoint_root is None:
        checkpoint_root = REPO / "qwen_rl_trading" / "checkpoints"
    if leaderboard_path is None:
        leaderboard_path = REPO / "qwen_rl_trading" / "leaderboard.csv"

    checkpoint_root.mkdir(parents=True, exist_ok=True)
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)

    # filter experiments by description if specified
    if descriptions:
        desc_set = set(d.strip() for d in descriptions.split(","))
        experiments = [e for e in experiments if e.get("description") in desc_set]

    best_reward = -float("inf")
    best_description = None

    for i, exp in enumerate(experiments[:max_trials]):
        trial_config = build_trial_config(exp)
        log.info("=== trial %d/%d: %s ===", i + 1, min(max_trials, len(experiments)), trial_config.description)

        result = run_trial(trial_config, time_budget, checkpoint_root)
        _append_leaderboard(leaderboard_path, result, trial_config)

        reward = result.get("val_mean_reward")
        if reward is not None and reward > best_reward:
            best_reward = reward
            best_description = trial_config.description
            log.info("new best: %s reward=%.4f", best_description, best_reward)

        log.info("trial %d done: reward=%s sortino=%s",
                 i + 1, result.get("val_mean_reward"), result.get("val_mean_sortino"))

    log.info("autoresearch complete: %d trials, best=%s (%.4f)", len(experiments[:max_trials]),
             best_description, best_reward)
    return {"best_description": best_description, "best_reward": best_reward}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--time-budget", type=int, default=600)
    p.add_argument("--max-trials", type=int, default=30)
    p.add_argument("--checkpoint-root", default="qwen_rl_trading/checkpoints")
    p.add_argument("--leaderboard", default="qwen_rl_trading/leaderboard.csv")
    p.add_argument("--descriptions", default="", help="comma-separated experiment names to run")
    args = p.parse_args()

    run_autoresearch(
        EXPERIMENTS,
        time_budget=args.time_budget,
        max_trials=args.max_trials,
        checkpoint_root=Path(args.checkpoint_root),
        leaderboard_path=Path(args.leaderboard),
        descriptions=args.descriptions,
    )


if __name__ == "__main__":
    main()
