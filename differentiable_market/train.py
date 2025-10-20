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
    parser.add_argument("--include-cash", action="store_true", help="Append a zero-return cash asset to allow explicit de-risking")
    parser.add_argument("--soft-drawdown-lambda", type=float, default=None, help="Coefficient for soft drawdown penalty")
    parser.add_argument("--risk-budget-lambda", type=float, default=None, help="Coefficient for risk budget mismatch penalty")
    parser.add_argument(
        "--risk-budget-target",
        type=float,
        nargs="+",
        default=None,
        help="Target risk budget allocation per asset",
    )
    parser.add_argument("--trade-memory-lambda", type=float, default=None, help="Weight for trade memory regret penalty")
    parser.add_argument("--trade-memory-ema-decay", type=float, default=None, help="EMA decay for trade memory state")
    parser.add_argument("--use-taylor-features", action="store_true", help="Append Taylor positional features")
    parser.add_argument("--taylor-order", type=int, default=None, help="Taylor feature order when enabled")
    parser.add_argument("--taylor-scale", type=float, default=None, help="Taylor feature scale factor")
    parser.add_argument("--use-wavelet-features", action="store_true", help="Append Haar wavelet detail features")
    parser.add_argument("--wavelet-levels", type=int, default=None, help="Number of Haar wavelet pyramid levels")
    parser.add_argument(
        "--wavelet-padding-mode",
        type=str,
        choices=("reflect", "replicate", "constant"),
        default=None,
        help="Padding mode used when building Haar wavelet pyramid",
    )
    parser.add_argument("--init-checkpoint", type=Path, default=None, help="Optional policy checkpoint to warm-start training")
    parser.add_argument(
        "--best-k-checkpoints",
        type=int,
        default=3,
        help="Number of top evaluation checkpoints to keep on disk",
    )
    parser.add_argument("--use-wandb", action="store_true", help="Mirror metrics to Weights & Biases via wandboard logger")
    parser.add_argument("--wandb-project", type=str, default=None, help="Weights & Biases project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="Weights & Biases entity/team")
    parser.add_argument("--wandb-tags", type=str, nargs="*", default=None, help="Optional tags for the wandb run")
    parser.add_argument("--wandb-group", type=str, default=None, help="Optional wandb group")
    parser.add_argument("--wandb-notes", type=str, default=None, help="Free-form notes stored with the wandb run")
    parser.add_argument("--wandb-mode", type=str, default="auto", help="wandb mode: auto/off/online/offline")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Override wandb run name")
    parser.add_argument("--wandb-log-metrics", action="store_true", help="Echo mirrored metrics to the logger at INFO level")
    parser.add_argument("--wandb-metric-log-level", type=str, default="INFO", help="Log level for mirrored metric previews")
    parser.add_argument("--tensorboard-root", type=Path, default=None, help="Root directory for TensorBoard event files")
    parser.add_argument("--tensorboard-subdir", type=str, default=None, help="Sub-directory for this run inside the TensorBoard root")
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
        include_cash=args.include_cash,
        init_checkpoint=args.init_checkpoint,
        best_k_checkpoints=max(1, args.best_k_checkpoints),
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_tags=tuple(args.wandb_tags or ()),
        wandb_group=args.wandb_group,
        wandb_notes=args.wandb_notes,
        wandb_mode=args.wandb_mode,
        wandb_run_name=args.wandb_run_name,
        wandb_log_metrics=args.wandb_log_metrics,
        wandb_metric_log_level=args.wandb_metric_log_level,
        tensorboard_root=args.tensorboard_root if args.tensorboard_root is not None else Path("tensorboard_logs"),
        tensorboard_subdir=args.tensorboard_subdir,
    )
    if args.soft_drawdown_lambda is not None:
        train_cfg.soft_drawdown_lambda = args.soft_drawdown_lambda
    if args.risk_budget_lambda is not None:
        train_cfg.risk_budget_lambda = args.risk_budget_lambda
    if args.risk_budget_target is not None:
        train_cfg.risk_budget_target = tuple(args.risk_budget_target)
    if args.trade_memory_lambda is not None:
        train_cfg.trade_memory_lambda = args.trade_memory_lambda
    if args.trade_memory_ema_decay is not None:
        train_cfg.trade_memory_ema_decay = args.trade_memory_ema_decay
    if args.use_taylor_features:
        train_cfg.use_taylor_features = True
    if args.taylor_order is not None:
        train_cfg.taylor_order = args.taylor_order
    if args.taylor_scale is not None:
        train_cfg.taylor_scale = args.taylor_scale
    if args.use_wavelet_features:
        train_cfg.use_wavelet_features = True
    if args.wavelet_levels is not None:
        train_cfg.wavelet_levels = args.wavelet_levels
    if args.wavelet_padding_mode is not None:
        train_cfg.wavelet_padding_mode = args.wavelet_padding_mode
    eval_cfg = EvaluationConfig(report_dir=Path("differentiable_market") / "evals")

    trainer = DifferentiableMarketTrainer(data_cfg, env_cfg, train_cfg, eval_cfg)
    trainer.fit()


if __name__ == "__main__":
    main()
