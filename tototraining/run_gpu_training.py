#!/usr/bin/env python3
"""
Launch a longer Toto training run on GPU using the enhanced trainer.

This script configures a moderately deeper model, runs for additional epochs,
and keeps the top-4 checkpoints by validation loss for later evaluation.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch

try:
    from .toto_trainer import TrainerConfig, DataLoaderConfig, TotoTrainer
except ImportError:
    # Fallback for script-style execution when package-relative imports are unavailable
    import sys

    package_dir = Path(__file__).resolve().parent
    parent_dir = package_dir.parent
    for path in (package_dir, parent_dir):
        str_path = str(path)
        if str_path not in sys.path:
            sys.path.insert(0, str_path)
    from toto_trainer import TrainerConfig, DataLoaderConfig, TotoTrainer


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch a Toto GPU training run with configurable overrides",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--compile", dest="compile", action="store_true", default=None,
                        help="Enable torch.compile for the model")
    parser.add_argument("--no-compile", dest="compile", action="store_false",
                        help="Disable torch.compile")
    parser.add_argument("--optim", "--optimizer", dest="optimizer", type=str,
                        help="Optimizer to use (e.g. muon_mix, adamw)")
    parser.add_argument("--device-bs", "--device_bs", dest="device_batch_size", type=int,
                        help="Per-device batch size")
    parser.add_argument("--grad-accum", "--grad_accum", dest="accumulation_steps", type=int,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", "--learning-rate", dest="learning_rate", type=float,
                        help="Learning rate")
    parser.add_argument("--warmup-steps", "--warmup_steps", dest="warmup_steps", type=int,
                        help="Number of warmup steps")
    parser.add_argument("--max-epochs", "--max_epochs", dest="max_epochs", type=int,
                        help="Maximum epochs to train")
    parser.add_argument("--report", dest="report_path", type=Path,
                        help="Optional path to write a Markdown training report")
    parser.add_argument("--metrics-frequency", "--metrics_frequency", dest="metrics_log_frequency", type=int,
                        help="Log train metrics every N batches")
    parser.add_argument("--no-freeze-backbone", dest="freeze_backbone", action="store_false", default=None,
                        help="Unfreeze the Toto backbone for finetuning")
    parser.add_argument("--seed", dest="random_seed", type=int,
                        help="Override the random seed")
    parser.add_argument("--save-dir", dest="save_dir", type=Path,
                        help="Override the checkpoint directory root")
    parser.add_argument("--summary-only", dest="summary_only", action="store_true",
                        help="Print the effective configuration and exit without training")
    return parser


def _apply_overrides(trainer_config: TrainerConfig, args: argparse.Namespace) -> None:
    overrides: Dict[str, Any] = {
        "compile": args.compile,
        "optimizer": args.optimizer,
        "device_batch_size": args.device_batch_size,
        "accumulation_steps": args.accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "max_epochs": args.max_epochs,
        "metrics_log_frequency": args.metrics_log_frequency,
        "random_seed": args.random_seed,
    }

    for field_name, maybe_value in overrides.items():
        if maybe_value is not None:
            setattr(trainer_config, field_name, maybe_value)

    if args.device_batch_size is not None:
        trainer_config.batch_size = args.device_batch_size

    if args.freeze_backbone is not None:
        trainer_config.freeze_backbone = args.freeze_backbone


def _print_run_header(save_dir: Path, trainer_config: TrainerConfig, loader_config: DataLoaderConfig) -> None:
    effective_global = (
        trainer_config.batch_size
        * max(1, trainer_config.accumulation_steps)
        * (trainer_config.world_size if trainer_config.distributed else 1)
    )

    header_lines = [
        "================ Toto GPU Training ================",
        f"Timestamp             : {datetime.now().isoformat(timespec='seconds')}",
        f"Checkpoints Directory : {save_dir}",
        f"torch.compile         : {trainer_config.compile}",
        f"Optimizer             : {trainer_config.optimizer}",
        f"Learning Rate         : {trainer_config.learning_rate}",
        f"Warmup Steps          : {trainer_config.warmup_steps}",
        f"Max Epochs            : {trainer_config.max_epochs}",
        f"Per-Device Batch Size : {trainer_config.batch_size}",
        f"Grad Accumulation     : {trainer_config.accumulation_steps}",
        f"Effective Global Batch: {effective_global}",
        f"Freeze Backbone       : {trainer_config.freeze_backbone}",
        f"Training Data Path    : {loader_config.train_data_path}",
        f"Test Data Path        : {loader_config.test_data_path}",
        "====================================================",
    ]
    print("\n".join(header_lines))


def _write_report(report_path: Path, trainer_config: TrainerConfig, val_metrics: Dict[str, float],
                  test_metrics: Dict[str, float]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Toto GPU Training Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Trainer Configuration",
        "",
    ]

    excluded_keys: Iterable[str] = {"save_dir", "log_file", "export_pretrained_dir"}
    for key, value in sorted(asdict(trainer_config).items()):
        if key in excluded_keys:
            continue
        lines.append(f"- **{key}**: {value}")

    lines.extend([
        "",
        "## Validation Metrics",
        "",
    ])
    if val_metrics:
        for key, value in sorted(val_metrics.items()):
            lines.append(f"- **{key}**: {value}")
    else:
        lines.append("- Validation metrics not available")

    lines.extend([
        "",
        "## Test Metrics",
        "",
    ])
    if test_metrics:
        for key, value in sorted(test_metrics.items()):
            lines.append(f"- **{key}**: {value}")
    else:
        lines.append("- Test metrics not available")

    report_path.write_text("\n".join(lines))
    print(f"Wrote report to {report_path}")


def main(argv: Optional[Iterable[str]] = None) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for this training run.")

    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    save_root = Path("tototraining") / "checkpoints" / "gpu_run"
    if args.save_dir is not None:
        save_root = args.save_dir
    save_dir = save_root
    save_dir.mkdir(parents=True, exist_ok=True)

    trainer_config = TrainerConfig(
        patch_size=64,
        stride=64,
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        mlp_hidden_dim=1536,
        dropout=0.1,
        spacewise_every_n_layers=2,
        scaler_cls="<class 'model.scaler.StdMeanScaler'>",
        output_distribution_classes=["<class 'model.distribution.StudentTOutput'>"],
        learning_rate=3e-4,
        min_lr=1e-6,
        weight_decay=0.01,
        batch_size=4,
        accumulation_steps=4,
        max_epochs=24,
        warmup_epochs=0,
        warmup_steps=2000,
        optimizer="muon_mix",
        scheduler="cosine",
        gradient_clip_val=0.1,
        use_mixed_precision=True,
        compile=True,
        require_gpu=True,
        distributed=False,
        save_dir=str(save_dir),
        save_every_n_epochs=1,
        keep_last_n_checkpoints=8,
        best_k_checkpoints=4,
        validation_frequency=1,
        early_stopping_patience=8,
        early_stopping_delta=1e-4,
        compute_train_metrics=True,
        compute_val_metrics=True,
        metrics_log_frequency=10,
        gradient_checkpointing=False,
        memory_efficient_attention=False,
        pin_memory=False,
        log_level="INFO",
        log_file=str(save_dir / "training.log"),
        wandb_project=None,
        experiment_name="toto_gpu_run",
        log_to_tensorboard=False,
        tensorboard_log_dir="tensorboard_logs",
        export_pretrained_dir=str(save_dir / "hf_export"),
        export_on_best=False,
        random_seed=1337,
        pretrained_model_id="Datadog/Toto-Open-Base-1.0",
        freeze_backbone=False,
        trainable_param_substrings=None,
    )

    _apply_overrides(trainer_config, args)

    loader_config = DataLoaderConfig(
        train_data_path="trainingdata/train",
        test_data_path="trainingdata/test",
        patch_size=trainer_config.patch_size,
        stride=trainer_config.stride,
        sequence_length=192,
        prediction_length=24,
        normalization_method="robust",
        handle_missing="interpolate",
        outlier_threshold=3.0,
        batch_size=trainer_config.batch_size,
        validation_split=0.2,
        test_split_days=30,
        cv_folds=3,
        cv_gap=24,
        min_sequence_length=256,
        max_symbols=128,
        ohlc_features=["Open", "High", "Low", "Close"],
        additional_features=[],
        target_feature="Close",
        add_technical_indicators=False,
        rsi_period=14,
        ma_periods=[5, 10],
        enable_augmentation=True,
        price_noise_std=0.006,
        volume_noise_std=0.02,
        feature_dropout_prob=0.01,
        time_mask_prob=0.05,
        time_mask_max_span=4,
        random_scaling_range=(0.9975, 1.0025),
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        random_seed=1337,
    )

    loader_config.batch_size = trainer_config.batch_size
    loader_config.random_seed = trainer_config.random_seed

    if args.summary_only:
        print(json.dumps({"trainer_config": asdict(trainer_config)}, indent=2))
        return

    _print_run_header(save_dir, trainer_config, loader_config)

    trainer = TotoTrainer(trainer_config, loader_config)
    trainer.prepare_data()
    trainer.setup_model()
    trainer.train()

    val_metrics = trainer.evaluate("val")
    test_metrics = trainer.evaluate("test")

    summary_path = save_dir / "final_metrics.json"
    summary_path.write_text(
        json.dumps(
            {
                "val": val_metrics,
                "test": test_metrics,
            },
            indent=2,
        )
    )
    print("FINAL_VAL_METRICS", val_metrics)
    print("FINAL_TEST_METRICS", test_metrics)
    print(f"Saved metrics summary to {summary_path}")

    if args.report_path:
        _write_report(args.report_path, trainer_config, val_metrics, test_metrics)


if __name__ == "__main__":
    main()
