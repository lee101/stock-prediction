from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import ContextManager, Optional

from wandboard import WandBoardLogger

from .config import FastForecasterConfig
from .trainer import FastForecasterTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FastForecaster for low validation MAE.")

    parser.add_argument(
        "--dataset",
        choices=["hourly", "daily", "custom"],
        default="hourly",
        help="Convenience dataset selector. Use custom with --data-dir for arbitrary roots.",
    )
    parser.add_argument("--data-dir", type=Path, default=None, help="Directory containing per-symbol CSV files.")
    parser.add_argument("--output-dir", type=Path, default=Path("FastForecaster") / "artifacts")
    parser.add_argument("--symbols", type=str, default="", help="Optional comma-separated symbol allowlist.")
    parser.add_argument("--max-symbols", type=int, default=24, help="Maximum symbol count (0 = all).")

    parser.add_argument("--lookback", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--train-stride", type=int, default=1)
    parser.add_argument("--eval-stride", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--min-rows-per-symbol", type=int, default=1024)
    parser.add_argument("--max-train-windows-per-symbol", type=int, default=80000)
    parser.add_argument("--max-eval-windows-per-symbol", type=int, default=10000)

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--return-loss-weight", type=float, default=0.20)
    parser.add_argument("--direction-loss-weight", type=float, default=0.02)
    parser.add_argument("--direction-margin-scale", type=float, default=16.0)
    parser.add_argument("--horizon-weight-power", type=float, default=0.35)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--ema-eval", dest="use_ema_eval", action="store_true", default=True)
    parser.add_argument("--no-ema-eval", dest="use_ema_eval", action="store_false")

    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ff-multiplier", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--qk-norm", dest="qk_norm", action="store_true", default=True)
    parser.add_argument("--no-qk-norm", dest="qk_norm", action="store_false")
    parser.add_argument("--qk-norm-eps", type=float, default=1e-6)

    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--torch-compile", dest="torch_compile", action="store_true", default=True)
    parser.add_argument("--no-torch-compile", dest="torch_compile", action="store_false")
    parser.add_argument("--compile-mode", type=str, default="max-autotune")
    parser.add_argument("--fused-optim", dest="use_fused_optimizer", action="store_true", default=True)
    parser.add_argument("--no-fused-optim", dest="use_fused_optimizer", action="store_false")
    parser.add_argument("--use-cpp-kernels", action="store_true", default=False)
    parser.add_argument(
        "--build-cpp-extension",
        action="store_true",
        default=False,
        help="Actually build optional C++/CUDA kernels. Off by default for startup speed.",
    )

    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", dest="pin_memory", action="store_true", default=True)
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default="")

    return parser.parse_args()


def _default_data_dir(dataset: str) -> Path:
    if dataset == "hourly":
        return Path("trainingdatahourly") / "stocks"
    if dataset == "daily":
        return Path("trainingdata")
    return Path("trainingdatahourly") / "stocks"


def _parse_symbols(raw: str) -> tuple[str, ...] | None:
    cleaned = tuple(sorted({item.strip().upper() for item in raw.split(",") if item.strip()}))
    return cleaned or None


def _parse_tags(raw: str) -> tuple[str, ...]:
    return tuple(sorted({item.strip() for item in raw.split(",") if item.strip()}))


def build_config(args: argparse.Namespace) -> FastForecasterConfig:
    data_dir = args.data_dir if args.data_dir is not None else _default_data_dir(args.dataset)
    min_rows_per_symbol = args.min_rows_per_symbol
    if args.dataset == "daily" and args.min_rows_per_symbol == 1024:
        min_rows_per_symbol = max(240, args.lookback + args.horizon + 64)
    return FastForecasterConfig(
        data_dir=data_dir,
        output_dir=args.output_dir,
        symbols=_parse_symbols(args.symbols),
        max_symbols=args.max_symbols,
        lookback=args.lookback,
        horizon=args.horizon,
        train_stride=args.train_stride,
        eval_stride=args.eval_stride,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        min_rows_per_symbol=min_rows_per_symbol,
        max_train_windows_per_symbol=args.max_train_windows_per_symbol,
        max_eval_windows_per_symbol=args.max_eval_windows_per_symbol,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        grad_accum_steps=args.grad_accum_steps,
        warmup_steps=args.warmup_steps,
        log_interval=args.log_interval,
        early_stopping_patience=args.early_stopping_patience,
        return_loss_weight=args.return_loss_weight,
        direction_loss_weight=args.direction_loss_weight,
        direction_margin_scale=args.direction_margin_scale,
        horizon_weight_power=args.horizon_weight_power,
        use_ema_eval=args.use_ema_eval,
        ema_decay=args.ema_decay,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_multiplier=args.ff_multiplier,
        dropout=args.dropout,
        qk_norm=args.qk_norm,
        qk_norm_eps=args.qk_norm_eps,
        precision=args.precision,
        torch_compile=args.torch_compile,
        compile_mode=args.compile_mode,
        use_fused_optimizer=args.use_fused_optimizer,
        use_cpp_kernels=args.use_cpp_kernels,
        build_cpp_extension=args.build_cpp_extension,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        seed=args.seed,
        device=args.device,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        wandb_tags=_parse_tags(args.wandb_tags),
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)

    logger_ctx: ContextManager[Optional[WandBoardLogger]]
    if config.wandb_project:
        run_name = config.wandb_run_name or f"fastforecaster_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        logger_ctx = WandBoardLogger(
            run_name=run_name,
            project=config.wandb_project,
            entity=config.wandb_entity,
            group=config.wandb_group,
            tags=config.wandb_tags,
            log_dir="tensorboard_logs",
            tensorboard_subdir=f"fastforecaster/{run_name}",
            enable_wandb=True,
            log_metrics=True,
            config={"fastforecaster": config.as_dict()},
        )
    else:
        logger_ctx = nullcontext()

    with logger_ctx as metrics_logger:
        trainer = FastForecasterTrainer(config, metrics_logger=metrics_logger)
        summary = trainer.train()

        if metrics_logger is not None:
            metrics_logger.log({f"summary/{k}": v for k, v in summary.items()}, step=config.epochs + 5)

    print("\n[fastforecaster] Done. Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
