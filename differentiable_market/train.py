from __future__ import annotations

import argparse
from pathlib import Path

from .config import DataConfig, EnvironmentConfig, EvaluationConfig, TrainingConfig
from .trainer import DifferentiableMarketTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Differentiable market RL trainer")
    parser.add_argument("--data-root", type=Path, default=Path("trainingdata"), help="Root directory of OHLC CSV files")
    parser.add_argument("--data-glob", type=str, default="*.csv", help="Glob pattern for CSV selection")
    parser.add_argument("--max-assets", type=int, default=None, help="Limit number of assets loaded")
    parser.add_argument("--exclude", type=str, nargs="*", default=(), help="Symbols to exclude")
    parser.add_argument("--lookback", type=int, default=128, help="Training lookback window")
    parser.add_argument("--batch-windows", type=int, default=64, help="Number of sampled windows per step")
    parser.add_argument("--rollout-groups", type=int, default=4, help="GRPO rollout group size")
    parser.add_argument("--epochs", type=int, default=2000, help="Training iterations")
    parser.add_argument("--eval-interval", type=int, default=100, help="Steps between evaluations")
    parser.add_argument("--save-dir", type=Path, default=Path("differentiable_market") / "runs", help="Directory to store runs")
    parser.add_argument("--device", type=str, default="auto", help="Device override: auto/cpu/cuda")
    parser.add_argument("--dtype", type=str, default="auto", help="dtype override: auto/bfloat16/float32")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--no-muon", action="store_true", help="Disable Muon optimizer")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--microbatch-windows", type=int, default=None, help="Number of windows per micro-batch when accumulating gradients")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable GRU gradient checkpointing to save memory")
    parser.add_argument("--risk-aversion", type=float, default=None, help="Override risk aversion penalty")
    parser.add_argument("--drawdown-lambda", type=float, default=None, help="Penalty weight for maximum drawdown in objective")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_cfg = DataConfig(
        root=args.data_root,
        glob=args.data_glob,
        max_assets=args.max_assets,
        exclude_symbols=tuple(args.exclude),
    )
    env_cfg = EnvironmentConfig()
    if args.risk_aversion is not None:
        env_cfg.risk_aversion = args.risk_aversion
    if args.drawdown_lambda is not None:
        env_cfg.drawdown_lambda = args.drawdown_lambda
    train_cfg = TrainingConfig(
        lookback=args.lookback,
        batch_windows=args.batch_windows,
        rollout_groups=args.rollout_groups,
        epochs=args.epochs,
        eval_interval=args.eval_interval,
        save_dir=args.save_dir,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
        use_muon=not args.no_muon,
        use_compile=not args.no_compile,
        microbatch_windows=args.microbatch_windows,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    eval_cfg = EvaluationConfig(report_dir=Path("differentiable_market") / "evals")

    trainer = DifferentiableMarketTrainer(data_cfg, env_cfg, train_cfg, eval_cfg)
    trainer.fit()


if __name__ == "__main__":
    main()
