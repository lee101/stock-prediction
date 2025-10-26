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
from typing import Dict, Iterable, Optional, Sequence

try:
    from .injection import get_torch
except Exception:  # pragma: no cover - script execution fallback
    try:
        from injection import get_torch  # type: ignore
    except Exception:
        def get_torch():
            import torch as _torch  # type: ignore

            return _torch

torch = get_torch()

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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__ or "Toto training launcher.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
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
    parser.add_argument(
        "--optim",
        "--optimizer",
        dest="optimizer",
        type=str,
        help="Optimizer name to use (e.g. muon_mix, adamw).",
    )
    parser.add_argument(
        "--device-bs",
        "--device_bs",
        dest="device_batch_size",
        type=int,
        help="Per-device batch size.",
    )
    parser.add_argument(
        "--grad-accum",
        "--grad_accum",
        dest="accumulation_steps",
        type=int,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        dest="learning_rate",
        type=float,
        help="Learning rate.",
    )
    parser.add_argument(
        "--warmup-steps",
        "--warmup_steps",
        dest="warmup_steps",
        type=int,
        help="Number of warmup steps.",
    )
    parser.add_argument(
        "--max-epochs",
        "--max_epochs",
        dest="max_epochs",
        type=int,
        help="Maximum training epochs.",
    )
    parser.add_argument(
        "--report",
        "--report-path",
        dest="report_path",
        type=Path,
        help="Optional path to write a Markdown training summary report.",
    )
    parser.add_argument(
        "--run-name",
        dest="run_name",
        type=str,
        help="Override experiment name used in logs and checkpoints.",
    )
    parser.add_argument(
        "--save-dir",
        dest="save_dir",
        type=Path,
        help="Optional override for checkpoint directory.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint in the save directory.",
    )
    parser.add_argument(
        "--resume-from",
        dest="resume_from",
        type=Path,
        help="Resume from a specific checkpoint path.",
    )
    parser.add_argument(
        "--metrics-frequency",
        "--metrics_frequency",
        dest="metrics_log_frequency",
        type=int,
        help="Log train metrics every N batches.",
    )
    parser.add_argument(
        "--no-freeze-backbone",
        dest="freeze_backbone",
        action="store_false",
        help="Unfreeze the Toto backbone for finetuning.",
    )
    parser.add_argument(
        "--freeze-backbone",
        dest="freeze_backbone",
        action="store_true",
        help="Freeze the Toto backbone during finetuning.",
    )
    parser.add_argument(
        "--seed",
        "--random-seed",
        dest="random_seed",
        type=int,
        help="Override the random seed.",
    )
    parser.add_argument(
        "--summary-only",
        dest="summary_only",
        action="store_true",
        help="Print the effective configuration and exit without training.",
    )
    parser.set_defaults(freeze_backbone=None)
    return parser


def _format_metric_table(metrics: Dict[str, float]) -> Sequence[str]:
    if not metrics:
        return ["(no metrics recorded)"]
    rows = ["| metric | value |", "| --- | --- |"]
    for key in sorted(metrics):
        rows.append(f"| {key} | {metrics[key]:.6g} |")
    return rows


def _apply_overrides(trainer_config: TrainerConfig, args: argparse.Namespace) -> None:
    overrides: Dict[str, Optional[object]] = {
        "compile": args.compile,
        "optimizer": args.optimizer,
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
        trainer_config.device_batch_size = args.device_batch_size

    if args.freeze_backbone is not None:
        trainer_config.freeze_backbone = args.freeze_backbone

    if trainer_config.freeze_backbone:
        if not getattr(trainer_config, "trainable_param_substrings", None):
            trainer_config.trainable_param_substrings = [
                "output_distribution",
                "loc_proj",
                "scale_proj",
                "df",
            ]
    else:
        trainer_config.trainable_param_substrings = None


def _print_run_header(
    save_dir: Path,
    trainer_config: TrainerConfig,
    loader_config: DataLoaderConfig,
) -> None:
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


def _write_markdown_report(
    report_path: Path,
    experiment_name: str,
    device_label: str,
    trainer_config: TrainerConfig,
    val_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().isoformat(timespec="seconds")
    lines = [
        f"# Toto Training Summary — {experiment_name}",
        "",
        f"- Timestamp (UTC): {timestamp}",
        f"- Device: {device_label}",
        f"- torch.compile: {trainer_config.compile}",
        f"- Optimizer: {trainer_config.optimizer}",
        f"- Learning rate: {trainer_config.learning_rate}",
        f"- Batch size: {trainer_config.batch_size}",
        f"- Grad accumulation: {trainer_config.accumulation_steps}",
        f"- Max epochs: {trainer_config.max_epochs}",
        "",
        "## Trainer Configuration",
        "",
    ]

    excluded_keys: Iterable[str] = {"save_dir", "log_file", "export_pretrained_dir"}
    for key, value in sorted(asdict(trainer_config).items()):
        if key in excluded_keys:
            continue
        lines.append(f"- **{key}**: {value}")

    lines.extend(["", "## Validation Metrics"])
    lines.extend(_format_metric_table(val_metrics))
    lines.extend(["", "## Test Metrics"])
    lines.extend(_format_metric_table(test_metrics))

    report_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote Markdown report to {report_path}")


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

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

    batch_size = (
        args.device_batch_size if args.device_batch_size is not None else default_batch_size
    )
    accumulation_steps = (
        args.accumulation_steps if args.accumulation_steps is not None else default_grad_accum
    )
    learning_rate = args.learning_rate if args.learning_rate is not None else default_lr
    warmup_steps = args.warmup_steps if args.warmup_steps is not None else default_warmup_steps
    max_epochs = args.max_epochs if args.max_epochs is not None else default_max_epochs
    optimizer = args.optimizer if args.optimizer is not None else "muon_mix"
    compile_flag = has_cuda if args.compile is None else args.compile

    if not has_cuda:
        if args.device_batch_size is None:
            batch_size = max(1, min(batch_size, 2))
        if args.accumulation_steps is None:
            accumulation_steps = max(1, accumulation_steps // 2)
        if args.learning_rate is None:
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

    resume_flag = bool(args.resume or args.resume_from)
    if resume_flag:
        save_dir = base_dir
    else:
        if args.save_dir is None or (base_dir.exists() and base_dir.is_dir()):
            save_dir = base_dir / timestamp
        else:
            save_dir = base_dir

    save_dir.parent.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    if not resume_flag and save_dir.parent != save_dir:
        latest_symlink = save_dir.parent / "latest"
        try:
            if latest_symlink.is_symlink() or latest_symlink.exists():
                latest_symlink.unlink()
            latest_symlink.symlink_to(save_dir)
        except OSError:
            pass

    metrics_frequency = (
        args.metrics_log_frequency if args.metrics_log_frequency is not None else 10
    )
    seed = args.random_seed if args.random_seed is not None else 1337
    device_label = "CUDA" if has_cuda else "CPU"

    resume_checkpoint = str(args.resume_from) if args.resume_from else None
    worker_count = 4 if has_cuda else max(1, min(2, torch.get_num_threads() or 2))
    pin_memory_flag = has_cuda
    if has_cuda:
        price_noise_std = 0.0125
        volume_noise_std = 0.05
        feature_dropout_prob = 0.02
        time_mask_prob = 0.1
        time_mask_max_span = 6
        scaling_range = (0.995, 1.005)
    else:
        price_noise_std = 0.006
        volume_noise_std = 0.02
        feature_dropout_prob = 0.01
        time_mask_prob = 0.05
        time_mask_max_span = 4
        scaling_range = (0.9975, 1.0025)

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
        device_batch_size=batch_size,
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
        metrics_log_frequency=metrics_frequency,
        gradient_checkpointing=False,
        memory_efficient_attention=False,
        pin_memory=pin_memory_flag,
        log_level="INFO",
        log_file=str(save_dir / "training.log"),
        wandb_project=None,
        experiment_name=experiment_name,
        log_to_tensorboard=False,
        tensorboard_log_dir="tensorboard_logs",
        export_pretrained_dir=str(save_dir / "hf_export"),
        export_on_best=False,
        random_seed=seed,
        pretrained_model_id="Datadog/Toto-Open-Base-1.0",
        freeze_backbone=False,
        trainable_param_substrings=None,
        resume_from_checkpoint=resume_checkpoint,
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
        price_noise_std=price_noise_std,
        volume_noise_std=volume_noise_std,
        feature_dropout_prob=feature_dropout_prob,
        time_mask_prob=time_mask_prob,
        time_mask_max_span=time_mask_max_span,
        random_scaling_range=scaling_range,
        num_workers=worker_count,
        pin_memory=pin_memory_flag,
        drop_last=False,
        random_seed=seed,
    )

    loader_config.batch_size = trainer_config.batch_size
    loader_config.random_seed = trainer_config.random_seed

    if args.summary_only:
        summary = {
            "save_dir": str(save_dir),
            "device": device_label,
            "trainer_config": asdict(trainer_config),
            "loader_config": asdict(loader_config),
        }
        print(json.dumps(summary, indent=2))
        return

    _print_run_header(save_dir, trainer_config, loader_config)

    trainer = TotoTrainer(trainer_config, loader_config)
    trainer.prepare_data()
    trainer.setup_model()
    trainer.train()

    val_metrics = trainer.evaluate("val") or {}
    test_metrics = trainer.evaluate("test") or {}

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
        _write_markdown_report(
            args.report_path,
            experiment_name,
            device_label,
            trainer_config,
            val_metrics,
            test_metrics,
        )


if __name__ == "__main__":
    main()
