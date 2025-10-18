#!/usr/bin/env python3
"""
Launch a longer Toto training run on GPU using the enhanced trainer.

This script configures a moderately deeper model, runs for additional epochs,
and keeps the top-4 checkpoints by validation loss for later evaluation.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Sequence

import torch

try:
    from .toto_trainer import TrainerConfig, DataLoaderConfig, TotoTrainer
except ImportError:  # pragma: no cover - fallback for script execution from repo root
    import sys

    package_dir = Path(__file__).resolve().parent
    parent_dir = package_dir.parent
    for path in (package_dir, parent_dir):
        str_path = str(path)
        if str_path not in sys.path:
            sys.path.insert(0, str_path)
    from toto_trainer import TrainerConfig, DataLoaderConfig, TotoTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__ or "Toto training launcher.")
    parser.add_argument(
        "--compile",
        dest="compile",
        action="store_true",
        help="Enable torch.compile. Defaults to enabled when CUDA is available.",
    )
    parser.add_argument(
        "--no-compile",
        dest="compile",
        action="store_false",
        help="Disable torch.compile even if CUDA is available.",
    )
    parser.set_defaults(compile=None)
    parser.add_argument("--optim", default=None, help="Optimizer name to use (e.g. muon_mix, adamw).")
    parser.add_argument("--device_bs", type=int, default=None, help="Per-device batch size.")
    parser.add_argument("--grad_accum", type=int, default=None, help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--warmup_steps", type=int, default=None, help="Number of warmup steps.")
    parser.add_argument("--max_epochs", type=int, default=None, help="Maximum training epochs.")
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional path to write a Markdown training summary report.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Override experiment name used in logs and checkpoints.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Optional override for checkpoint directory.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint in the save directory.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Resume from a specific checkpoint path.",
    )
    return parser.parse_args()


def _format_metric_table(metrics: dict[str, float]) -> Sequence[str]:
    if not metrics:
        return ["(no metrics recorded)"]
    rows = ["| metric | value |", "| --- | --- |"]
    for key in sorted(metrics):
        rows.append(f"| {key} | {metrics[key]:.6g} |")
    return rows


def main() -> None:
    args = parse_args()

    has_cuda = torch.cuda.is_available()
    if not has_cuda:
        print(
            "CUDA not available; falling back to CPU configuration with reduced model size.",
            flush=True,
        )

    default_batch_size = 4
    default_grad_accum = 4
    default_lr = 3e-4
    default_warmup_steps = 2000
    default_max_epochs = 24

    batch_size = args.device_bs if args.device_bs is not None else default_batch_size
    accumulation_steps = args.grad_accum if args.grad_accum is not None else default_grad_accum
    learning_rate = args.lr if args.lr is not None else default_lr
    warmup_steps = args.warmup_steps if args.warmup_steps is not None else default_warmup_steps
    max_epochs = args.max_epochs if args.max_epochs is not None else default_max_epochs
    optimizer = args.optim if args.optim is not None else "muon_mix"
    compile_flag = (
        has_cuda if args.compile is None else args.compile
    )

    if not has_cuda:
        if args.device_bs is None:
            batch_size = max(1, min(batch_size, 2))
        if args.grad_accum is None:
            accumulation_steps = max(1, accumulation_steps // 2)
        if args.lr is None:
            learning_rate = min(learning_rate, 2e-4)
        if args.warmup_steps is None:
            warmup_steps = min(warmup_steps, 500)
        if args.max_epochs is None:
            max_epochs = min(max_epochs, 6)
        if args.compile is None:
            compile_flag = False

    experiment_name = args.run_name or ("toto_gpu_run" if has_cuda else "toto_cpu_run")
    default_dir_name = "gpu_run" if has_cuda else "cpu_run"
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    base_dir = args.save_dir or (Path("tototraining") / "checkpoints" / default_dir_name)
    if args.resume:
        save_dir = base_dir
    else:
        if base_dir.exists():
            save_dir = base_dir / timestamp
        else:
            save_dir = base_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        latest_symlink = base_dir / "latest"
        try:
            if latest_symlink.is_symlink() or latest_symlink.exists():
                latest_symlink.unlink()
            latest_symlink.symlink_to(save_dir)
        except OSError:
            pass

    trainer_config = TrainerConfig(
        patch_size=64,
        stride=64,
        embed_dim=512 if not has_cuda else 768,
        num_layers=8 if not has_cuda else 12,
        num_heads=8 if not has_cuda else 12,
        mlp_hidden_dim=1024 if not has_cuda else 1536,
        dropout=0.1,
        spacewise_every_n_layers=2,
        scaler_cls="<class 'model.scaler.StdMeanScaler'>",
        output_distribution_classes=["<class 'model.distribution.StudentTOutput'>"],
        learning_rate=learning_rate,
        min_lr=1e-6,
        weight_decay=0.01,
        batch_size=batch_size,
        accumulation_steps=accumulation_steps,
        max_epochs=max_epochs,
        warmup_epochs=0,
        warmup_steps=warmup_steps,
        optimizer=optimizer,
        scheduler="cosine",
        gradient_clip_val=0.1,
        use_mixed_precision=has_cuda,
        compile=compile_flag,
        require_gpu=has_cuda,
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
        pin_memory=has_cuda,
        log_level="INFO",
        log_file=str(save_dir / "training.log"),
        wandb_project=None,
        experiment_name=experiment_name,
        log_to_tensorboard=False,
        tensorboard_log_dir="tensorboard_logs",
        export_pretrained_dir=str(save_dir / "hf_export"),
        export_on_best=False,
        random_seed=1337,
        pretrained_model_id="Datadog/Toto-Open-Base-1.0",
        freeze_backbone=True,
        trainable_param_substrings=["output_distribution", "loc_proj", "scale_proj", "df"],
        resume_from_checkpoint=str(args.resume_from) if args.resume_from else None,
    )

    loader_config = DataLoaderConfig(
        train_data_path="trainingdata/train",
        test_data_path="trainingdata/test",
        patch_size=trainer_config.patch_size,
        stride=trainer_config.stride,
        sequence_length=192,
        prediction_length=24,
        normalization_method="none",
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
        price_noise_std=0.0125,
        volume_noise_std=0.05,
        feature_dropout_prob=0.02,
        time_mask_prob=0.1,
        time_mask_max_span=6,
        random_scaling_range=(0.995, 1.005),
        num_workers=4 if has_cuda else max(1, min(2, (torch.get_num_threads() or 2))),
        pin_memory=has_cuda,
        drop_last=False,
        random_seed=1337,
    )

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

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().isoformat(timespec="seconds")
        lines = [
            f"# Toto Training Summary — {experiment_name}",
            "",
            f"- Timestamp (UTC): {timestamp}",
            f"- Device: {'CUDA' if has_cuda else 'CPU'}",
            f"- torch.compile: {trainer_config.compile}",
            f"- Optimizer: {trainer_config.optimizer}",
            f"- Learning rate: {trainer_config.learning_rate}",
            f"- Batch size: {trainer_config.batch_size}",
            f"- Grad accumulation: {trainer_config.accumulation_steps}",
            f"- Max epochs: {trainer_config.max_epochs}",
            "",
            "## Validation Metrics",
        ]
        lines.extend(_format_metric_table(val_metrics))
        lines.extend(["", "## Test Metrics"])
        lines.extend(_format_metric_table(test_metrics))
        report_content = "\n".join(lines) + "\n"
        args.report.write_text(report_content)
        print(f"Wrote Markdown report to {args.report}")


if __name__ == "__main__":
    main()
