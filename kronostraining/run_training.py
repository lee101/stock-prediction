from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import ContextManager, Optional

from .config import KronosTrainingConfig
from .trainer import KronosTrainer
from wandboard import WandBoardLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Kronos on the local training dataset.")
    parser.add_argument("--data-dir", type=Path, default=Path("trainingdata"), help="Path to training CSV directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("kronostraining") / "artifacts",
        help="Directory for checkpoints and metrics.",
    )
    parser.add_argument("--lookback", type=int, default=64, help="Historical window length.")
    parser.add_argument("--horizon", type=int, default=30, help="Forecast horizon in timesteps.")
    parser.add_argument("--validation-days", type=int, default=30, help="Number of unseen days for validation metrics.")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per step.")
    parser.add_argument("--learning-rate", type=float, default=4e-5, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay.")
    parser.add_argument("--model-name", type=str, default="NeoQuasar/Kronos-small", help="Base Kronos model identifier.")
    parser.add_argument(
        "--max-tokens-per-batch",
        type=int,
        default=262_144,
        help="Approximate token budget per optimisation step for dynamic batching.",
    )
    parser.add_argument(
        "--length-bucketing",
        type=int,
        nargs="+",
        default=[128, 256, 512],
        help="Allowed context lengths (tokens) for windowed batching.",
    )
    parser.add_argument(
        "--horizon-bucketing",
        type=int,
        nargs="+",
        default=[20, 32, 64],
        help="Allowed prediction horizons for windowed batching.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=20,
        help="Stride when extracting context windows. Smaller values increase overlap.",
    )
    parser.add_argument(
        "--pack-windows",
        dest="pack_windows",
        action="store_true",
        default=True,
        help="Pack batches by (context, horizon) bucket to keep shapes static.",
    )
    parser.add_argument(
        "--no-pack-windows",
        dest="pack_windows",
        action="store_false",
        help="Disable bucket packing (may increase recompilation).",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps before each optimizer update.",
    )
    parser.add_argument(
        "--bucket-warmup-steps",
        type=int,
        default=0,
        help="Optional warm-up iterations per (context, horizon) bucket prior to training.",
    )
    parser.add_argument(
        "--precision",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Autocast precision for the forward pass.",
    )
    parser.add_argument(
        "--torch-compile",
        dest="torch_compile",
        action="store_true",
        default=True,
        help="Enable torch.compile for the training step.",
    )
    parser.add_argument(
        "--no-torch-compile",
        dest="torch_compile",
        action="store_false",
        help="Disable torch.compile.",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="max-autotune",
        help="torch.compile mode to use when compilation is enabled.",
    )
    parser.add_argument(
        "--fused-optim",
        dest="use_fused_optimizer",
        action="store_true",
        default=True,
        help="Enable fused AdamW when supported by the build.",
    )
    parser.add_argument(
        "--no-fused-optim",
        dest="use_fused_optimizer",
        action="store_false",
        help="Disable fused optimizers even if available.",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default="NeoQuasar/Kronos-Tokenizer-base",
        help="Tokenizer identifier to pair with the model.",
    )
    parser.add_argument("--seed", type=int, default=1337, help="Random seed.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker count.")
    parser.add_argument("--eval-samples", type=int, default=4, help="Autoregressive sample count for evaluation.")
    parser.add_argument("--device", type=str, default=None, help="Explicit torch device, e.g. cuda:0.")
    parser.add_argument("--adapter", choices=["none", "lora"], default="none", help="Parameter-efficient adapter type.")
    parser.add_argument("--adapter-r", type=int, default=8, help="Adapter rank (LoRA).")
    parser.add_argument("--adapter-alpha", type=float, default=16.0, help="Adapter scaling factor.")
    parser.add_argument("--adapter-dropout", type=float, default=0.05, help="Adapter dropout before rank reduction.")
    parser.add_argument(
        "--adapter-targets",
        type=str,
        default="embedding.fusion_proj,transformer,dep_layer,head",
        help="Comma-separated substrings to select Linear layers for adapters.",
    )
    parser.add_argument(
        "--adapter-output-dir",
        type=Path,
        default=None,
        help="Directory root for saving adapter weights (defaults to output_dir/adapters).",
    )
    parser.add_argument("--adapter-name", type=str, default=None, help="Adapter subdirectory name (e.g., ticker).")
    parser.add_argument(
        "--freeze-backbone",
        dest="freeze_backbone",
        action="store_true",
        default=True,
        help="Freeze base Kronos weights when adapters are enabled.",
    )
    parser.add_argument(
        "--no-freeze-backbone",
        dest="freeze_backbone",
        action="store_false",
        help="Allow base Kronos weights to train alongside adapters.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Weights & Biases project name for logging metrics (enables WandBoard).",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Optional WandB run name; defaults to kronos_<timestamp> when project is provided.",
    )
    return parser.parse_args()


def _parse_targets(raw: str) -> tuple[str, ...]:
    return tuple(sorted({item.strip() for item in raw.split(",") if item.strip()}))


def build_config(args: argparse.Namespace) -> KronosTrainingConfig:
    return KronosTrainingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_name,
        lookback_window=args.lookback,
        prediction_length=args.horizon,
        validation_days=args.validation_days,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        num_workers=args.num_workers,
        eval_sample_count=args.eval_samples,
        device=args.device,
        max_tokens_per_batch=args.max_tokens_per_batch,
        length_buckets=tuple(args.length_bucketing),
        horizon_buckets=tuple(args.horizon_bucketing),
        window_stride=args.stride,
        pack_windows=args.pack_windows,
        grad_accum_steps=args.grad_accum_steps,
        bucket_warmup_steps=args.bucket_warmup_steps,
        precision=args.precision,
        torch_compile=args.torch_compile,
        compile_mode=args.compile_mode,
        use_fused_optimizer=args.use_fused_optimizer,
        adapter_type=args.adapter,
        adapter_rank=args.adapter_r,
        adapter_alpha=args.adapter_alpha,
        adapter_dropout=args.adapter_dropout,
        adapter_targets=_parse_targets(args.adapter_targets),
        adapter_output_dir=args.adapter_output_dir,
        adapter_name=args.adapter_name,
        freeze_backbone=args.freeze_backbone,
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)
    logger_ctx: ContextManager[Optional[WandBoardLogger]]
    if args.wandb_project:
        run_name = args.wandb_run_name or f"kronos_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        logger_ctx = WandBoardLogger(
            run_name=run_name,
            project=args.wandb_project,
            log_dir="tensorboard_logs",
            tensorboard_subdir=f"kronos/{run_name}",
            enable_wandb=True,
            log_metrics=True,
            config={"kronos": config.as_dict()},
        )
    else:
        logger_ctx = nullcontext()

    with logger_ctx as metrics_logger:
        trainer = KronosTrainer(config, metrics_logger=metrics_logger)
        summary = trainer.train()
        metrics = trainer.evaluate_holdout()
        if metrics_logger is not None:
            payload = {f"summary/{key}": value for key, value in summary.items()}
            payload.update(
                {f"holdout/{key}": value for key, value in metrics.get("aggregate", {}).items()}
            )
            metrics_logger.log(payload, step=config.epochs + 3)

    print("\n[kronos] Training summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print("\n[kronos] Validation aggregate metrics:")
    for key, value in metrics["aggregate"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
