#!/usr/bin/env python3
"""Launch script for fresh RL training sweep on mixed-23 data.

Defines 5 training configs and generates shell commands that train
via pufferlib_market/train.py then evaluate via comprehensive_marketsim_eval.py.

Usage:
  python scripts/launch_mixed23_retrain.py --dry-run
  python scripts/launch_mixed23_retrain.py
"""
from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

REPO = Path(__file__).resolve().parents[1]

TRAIN_DATA = "pufferlib_market/data/mixed23_fresh_train.bin"
CHECKPOINT_ROOT = "pufferlib_market/checkpoints/mixed23_fresh_retrain_v2"
TOTAL_TIMESTEPS = 10_000_000


@dataclass
class TrainConfig:
    name: str
    lr: float = 3e-4
    lr_schedule: str = "none"
    lr_warmup_frac: float = 0.02
    lr_min_ratio: float = 0.05
    hidden_size: int = 256
    arch: str = "mlp"
    rollout_len: int = 256
    minibatch_size: int = 2048
    num_envs: int = 64
    total_timesteps: int = TOTAL_TIMESTEPS
    ppo_epochs: int = 4
    ent_coef: float = 0.01
    ent_coef_end: float = 0.01
    anneal_ent: bool = False
    max_grad_norm: float = 0.5
    clip_eps: float = 0.2
    clip_vloss: bool = True
    obs_norm: bool = True
    weight_decay: float = 0.0
    seed: int = 42
    save_every: int = 50
    extra_flags: list[str] = field(default_factory=list)


def build_configs() -> list[TrainConfig]:
    configs = []

    # (a) Cosine LR with 10% warmup + entropy anneal
    configs.append(TrainConfig(
        name="cosine_lr_ent_anneal",
        lr=3e-4,
        lr_schedule="cosine",
        lr_warmup_frac=0.10,
        lr_min_ratio=0.0,
        ent_coef=0.05,
        ent_coef_end=0.01,
        anneal_ent=True,
    ))

    # (b) Wider hidden (2048) + standard LR
    configs.append(TrainConfig(
        name="wide_h2048",
        hidden_size=2048,
        arch="resmlp",
    ))

    # (c) Deeper network (4 blocks resmlp) + lower LR
    configs.append(TrainConfig(
        name="deep_4block_lowlr",
        lr=1e-4,
        arch="resmlp",
    ))

    # (d) Gradient clipping at 0.5 (matches default) + higher entropy
    configs.append(TrainConfig(
        name="high_entropy_gc05",
        max_grad_norm=0.5,
        ent_coef=0.03,
    ))

    # (e) Longer rollouts (512 steps) + larger batch (4096)
    configs.append(TrainConfig(
        name="long_rollout_bigbatch",
        rollout_len=512,
        minibatch_size=4096,
    ))

    return configs


def config_to_train_cmd(cfg: TrainConfig, repo: Path) -> list[str]:
    ckpt_dir = str(repo / CHECKPOINT_ROOT / cfg.name)
    cmd = [
        "python", "-m", "pufferlib_market.train",
        "--data-path", str(repo / TRAIN_DATA),
        "--checkpoint-dir", ckpt_dir,
        "--lr", str(cfg.lr),
        "--lr-schedule", cfg.lr_schedule,
        "--lr-warmup-frac", str(cfg.lr_warmup_frac),
        "--lr-min-ratio", str(cfg.lr_min_ratio),
        "--hidden-size", str(cfg.hidden_size),
        "--rollout-len", str(cfg.rollout_len),
        "--minibatch-size", str(cfg.minibatch_size),
        "--num-envs", str(cfg.num_envs),
        "--total-timesteps", str(cfg.total_timesteps),
        "--ppo-epochs", str(cfg.ppo_epochs),
        "--ent-coef", str(cfg.ent_coef),
        "--ent-coef-end", str(cfg.ent_coef_end),
        "--max-grad-norm", str(cfg.max_grad_norm),
        "--clip-eps", str(cfg.clip_eps),
        "--weight-decay", str(cfg.weight_decay),
        "--seed", str(cfg.seed),
        "--save-every", str(cfg.save_every),
    ]
    if cfg.anneal_ent:
        cmd.append("--anneal-ent")
    if cfg.clip_vloss:
        cmd.append("--clip-vloss")
    if cfg.obs_norm:
        cmd.append("--obs-norm")
    if cfg.arch != "mlp":
        cmd.extend(["--arch", cfg.arch])
    for flag in cfg.extra_flags:
        if flag not in cmd:
            cmd.append(flag)
    return cmd


def config_to_eval_cmd(cfg: TrainConfig, repo: Path) -> list[str]:
    return [
        "python", "comprehensive_marketsim_eval.py",
        "--root", str(repo),
        "--output", str(repo / CHECKPOINT_ROOT / cfg.name / "eval_results.csv"),
        "--periods", "30,60,90,120",
    ]


def run_sweep(configs: list[TrainConfig], repo: Path, dry_run: bool = False) -> None:
    for i, cfg in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"  Config {i+1}/{len(configs)}: {cfg.name}")
        print(f"{'='*60}")

        train_cmd = config_to_train_cmd(cfg, repo)
        print(f"  Train: {' '.join(train_cmd)}")

        if not dry_run:
            subprocess.run(train_cmd, cwd=str(repo), check=True)

        eval_cmd = config_to_eval_cmd(cfg, repo)
        print(f"  Eval:  {' '.join(eval_cmd)}")

        if not dry_run:
            subprocess.run(eval_cmd, cwd=str(repo), check=False)

    print(f"\nSweep complete. Checkpoints in: {repo / CHECKPOINT_ROOT}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch mixed-23 RL retrain sweep")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    args = parser.parse_args(argv)

    configs = build_configs()
    print(f"Mixed-23 retrain sweep: {len(configs)} configs")
    for cfg in configs:
        print(f"  - {cfg.name}: lr={cfg.lr} hidden={cfg.hidden_size} arch={cfg.arch}")

    run_sweep(configs, REPO, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
