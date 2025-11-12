#!/usr/bin/env python3
"""
Improved Toto Training Script - Aligned with Datadog Toto Paper

This script uses hyperparameters aligned with the official Datadog Toto paper:
- Patch size: 32 (not 64)
- Context length: 512+ (not 192)
- Proper gradient clipping (1.0 not 0.1)
- Longer training (100+ epochs)
- Better loss functions (quantile)
- All modern optimizations enabled

Usage:
    # Quick test run (10 epochs):
    python tototraining/run_improved_training.py --max-epochs 10 --run-name quick_test

    # Full training run (100 epochs):
    python tototraining/run_improved_training.py --max-epochs 100 --run-name full_v1

    # With WandB logging:
    python tototraining/run_improved_training.py --wandb-project stock-toto --run-name experiment_1

    # Resume from checkpoint:
    python tototraining/run_improved_training.py --resume
"""
from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Iterable

try:
    from .injection import get_torch
except Exception:
    try:
        from injection import get_torch  # type: ignore
    except Exception:
        def get_torch():
            import torch as _torch  # type: ignore
            return _torch

torch = get_torch()

try:
    from .toto_trainer import TrainerConfig, DataLoaderConfig, TotoTrainer
except ImportError:
    import sys
    package_dir = Path(__file__).resolve().parent
    parent_dir = package_dir.parent
    for path in (package_dir, parent_dir):
        str_path = str(path)
        if str_path not in sys.path:
            sys.path.insert(0, str_path)
    from toto_trainer import TrainerConfig, DataLoaderConfig, TotoTrainer

from wandboard import WandBoardLogger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--max-epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--device-bs", type=int, default=8, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=5000, help="Warmup steps")
    parser.add_argument("--context-length", type=int, default=512, help="Input sequence length")
    parser.add_argument("--pred-length", type=int, default=64, help="Prediction length")
    parser.add_argument("--patch-size", type=int, default=32, help="Patch size (32 recommended)")
    parser.add_argument("--stride", type=int, default=None, help="Patch stride (default: same as patch_size)")
    parser.add_argument("--use-quantile-loss", action="store_true", default=True, help="Use quantile loss")
    parser.add_argument("--no-quantile-loss", dest="use_quantile_loss", action="store_false")
    parser.add_argument("--enable-cuda-graphs", action="store_true", help="Enable CUDA graphs for speedup")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--wandb-project", type=str, help="WandB project name")
    parser.add_argument("--run-name", type=str, help="Experiment name")
    parser.add_argument("--save-dir", type=Path, help="Checkpoint directory")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--resume-from", type=Path, help="Resume from specific checkpoint")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--compile", action="store_true", default=True, help="Use torch.compile")
    parser.add_argument("--no-compile", dest="compile", action="store_false")
    parser.add_argument("--max-symbols", type=int, help="Max symbols to train on (default: all)")
    parser.add_argument("--augmentation", action="store_true", default=True, help="Enable data augmentation")
    parser.add_argument("--no-augmentation", dest="augmentation", action="store_false")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    has_cuda = torch.cuda.is_available()
    if not has_cuda:
        print("⚠️  CUDA not available. Training will be slow. Consider using a GPU.", flush=True)

    # Determine stride
    stride = args.stride if args.stride is not None else args.patch_size

    # Setup directories
    experiment_name = args.run_name or f"toto_improved_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    base_dir = args.save_dir or (Path("tototraining") / "checkpoints" / "improved")

    resume_flag = bool(args.resume or args.resume_from)
    if resume_flag:
        save_dir = base_dir
    else:
        save_dir = base_dir / timestamp

    save_dir.parent.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create latest symlink
    if not resume_flag and save_dir.parent != save_dir:
        latest_symlink = save_dir.parent / "latest"
        try:
            if latest_symlink.is_symlink() or latest_symlink.exists():
                latest_symlink.unlink()
            latest_symlink.symlink_to(save_dir)
        except OSError:
            pass

    resume_checkpoint = str(args.resume_from) if args.resume_from else None

    # Augmentation settings for GPU
    if has_cuda and args.augmentation:
        price_noise_std = 0.015
        volume_noise_std = 0.05
        feature_dropout_prob = 0.02
        time_mask_prob = 0.1
        time_mask_max_span = 8
        scaling_range = (0.99, 1.01)
    else:
        price_noise_std = 0.0
        volume_noise_std = 0.0
        feature_dropout_prob = 0.0
        time_mask_prob = 0.0
        time_mask_max_span = 0
        scaling_range = (1.0, 1.0)

    # ======================================================================
    # IMPROVED CONFIGURATION - ALIGNED WITH TOTO PAPER
    # ======================================================================

    trainer_config = TrainerConfig(
        # Model architecture - aligned with Toto paper
        patch_size=args.patch_size,  # 32 per paper (not 64!)
        stride=stride,  # 32 or less for overlap
        embed_dim=768 if has_cuda else 512,
        num_layers=12 if has_cuda else 8,
        num_heads=12 if has_cuda else 8,
        mlp_hidden_dim=1536 if has_cuda else 1024,
        dropout=0.1,
        spacewise_every_n_layers=2,
        scaler_cls="<class 'model.scaler.StdMeanScaler'>",
        output_distribution_classes=["<class 'model.distribution.StudentTOutput'>"],

        # Optimization - much better settings
        learning_rate=args.lr,
        min_lr=1e-6,
        weight_decay=0.01,
        batch_size=args.device_bs,
        device_batch_size=args.device_bs,
        accumulation_steps=args.grad_accum,
        max_epochs=args.max_epochs,
        warmup_epochs=0,
        warmup_steps=args.warmup_steps,
        optimizer="muon_mix",  # state-of-the-art for transformers
        scheduler="cosine",
        gradient_clip_val=1.0,  # FIXED: was 0.1 (too aggressive!)

        # Modern acceleration
        use_mixed_precision=has_cuda,
        compile=args.compile and has_cuda,
        require_gpu=has_cuda,
        use_cuda_graphs=args.enable_cuda_graphs and has_cuda,
        cuda_graph_warmup=10 if args.enable_cuda_graphs else 3,

        # Memory optimization
        gradient_checkpointing=args.gradient_checkpointing,
        memory_efficient_attention=False,  # disabled for CUDA graphs
        pin_memory=has_cuda,

        # Training settings
        distributed=False,
        save_dir=str(save_dir),
        save_every_n_epochs=1,
        keep_last_n_checkpoints=8,
        best_k_checkpoints=4,
        validation_frequency=1,
        early_stopping_patience=15,  # more patient
        early_stopping_delta=1e-5,
        compute_train_metrics=True,
        compute_val_metrics=True,
        metrics_log_frequency=20,

        # Loss function - quantile is better for forecasting
        loss_type="quantile" if args.use_quantile_loss else "huber",
        huber_delta=0.01,
        quantile_levels=[0.1, 0.25, 0.5, 0.75, 0.9] if args.use_quantile_loss else None,

        # EMA for better generalization
        ema_decay=0.9999,
        ema_eval=True,

        # Logging
        log_level="INFO",
        log_file=str(save_dir / "training.log"),
        wandb_project=args.wandb_project,
        experiment_name=experiment_name,
        log_to_tensorboard=False,
        tensorboard_log_dir=str(save_dir / "tensorboard"),
        export_pretrained_dir=str(save_dir / "hf_export"),
        export_on_best=True,

        # Model initialization
        random_seed=args.seed,
        pretrained_model_id="Datadog/Toto-Open-Base-1.0",
        freeze_backbone=False,
        trainable_param_substrings=None,
        resume_from_checkpoint=resume_checkpoint,
    )

    # Data loader config - MUCH better than before
    loader_config = DataLoaderConfig(
        train_data_path="trainingdata/train",
        test_data_path="trainingdata/test",

        # Aligned with model
        patch_size=args.patch_size,
        stride=stride,
        sequence_length=args.context_length,  # 512+ per paper (not 192!)
        prediction_length=args.pred_length,  # predict more steps

        # Preprocessing
        normalization_method="robust",
        handle_missing="interpolate",
        outlier_threshold=3.0,

        # Training data
        batch_size=args.device_bs,
        validation_split=0.2,
        test_split_days=30,
        cv_folds=3,
        cv_gap=24,
        min_sequence_length=args.context_length + args.pred_length + 50,
        max_symbols=args.max_symbols,  # Use all available data by default

        # Features
        ohlc_features=["Open", "High", "Low", "Close"],
        additional_features=["Volume"],
        target_feature="Close",
        add_technical_indicators=False,

        # Augmentation
        enable_augmentation=args.augmentation and has_cuda,
        price_noise_std=price_noise_std,
        volume_noise_std=volume_noise_std,
        feature_dropout_prob=feature_dropout_prob,
        time_mask_prob=time_mask_prob,
        time_mask_max_span=time_mask_max_span,
        random_scaling_range=scaling_range,

        # Data loading
        num_workers=6 if has_cuda else 2,
        pin_memory=has_cuda,
        drop_last=True,
        random_seed=args.seed,
    )

    # Print configuration summary
    print("=" * 80)
    print(f"🚀 IMPROVED TOTO TRAINING - {experiment_name}")
    print("=" * 80)
    print(f"Timestamp:              {datetime.now().isoformat()}")
    print(f"Device:                 {'CUDA' if has_cuda else 'CPU'}")
    print(f"Checkpoint Dir:         {save_dir}")
    print()
    print("KEY IMPROVEMENTS vs Previous:")
    print(f"  ✅ Patch size:         32 (was 64)")
    print(f"  ✅ Context length:     {args.context_length} (was 192)")
    print(f"  ✅ Gradient clip:      1.0 (was 0.1 - too aggressive!)")
    print(f"  ✅ Max epochs:         {args.max_epochs} (was 24)")
    print(f"  ✅ Loss function:      {'Quantile' if args.use_quantile_loss else 'Huber'}")
    print(f"  ✅ Effective batch:    {args.device_bs * args.grad_accum} (device_bs × grad_accum)")
    print(f"  ✅ CUDA graphs:        {args.enable_cuda_graphs}")
    print(f"  ✅ Optimizer:          muon_mix (state-of-the-art)")
    print(f"  ✅ torch.compile:      {trainer_config.compile}")
    print(f"  ✅ Data augmentation:  {args.augmentation}")
    print("=" * 80)
    print()

    # Setup WandB if requested
    logger_ctx = (
        WandBoardLogger(
            run_name=experiment_name,
            project=trainer_config.wandb_project,
            log_dir="tensorboard_logs",
            tensorboard_subdir=f"toto/{experiment_name}",
            enable_wandb=True,
            log_metrics=True,
            config={
                "toto_trainer": asdict(trainer_config),
                "toto_dataloader": asdict(loader_config),
                "improvements": {
                    "patch_size_fixed": "32 (was 64)",
                    "context_length_increased": f"{args.context_length} (was 192)",
                    "gradient_clip_fixed": "1.0 (was 0.1)",
                    "epochs_increased": f"{args.max_epochs} (was 24)",
                    "loss_improved": "quantile" if args.use_quantile_loss else "huber",
                },
            },
        )
        if trainer_config.wandb_project
        else nullcontext()
    )

    # Run training
    with logger_ctx as metrics_logger:
        trainer = TotoTrainer(trainer_config, loader_config, metrics_logger=metrics_logger)

        print("📊 Preparing data...")
        trainer.prepare_data()

        print("🏗️  Setting up model...")
        trainer.setup_model()

        print("🎯 Starting training...")
        print()
        trainer.train()

        # Evaluate
        print()
        print("=" * 80)
        print("📈 FINAL EVALUATION")
        print("=" * 80)

        val_metrics = trainer.evaluate("val") or {}
        test_metrics = trainer.evaluate("test") or {}

        if metrics_logger is not None:
            metrics_logger.log(
                {
                    **{f"final/val/{k}": v for k, v in val_metrics.items()},
                    **{f"final/test/{k}": v for k, v in test_metrics.items()},
                },
                step=trainer.current_epoch + 1,
            )

    # Save final metrics
    summary_path = save_dir / "final_metrics.json"
    summary_path.write_text(
        json.dumps(
            {
                "val": val_metrics,
                "test": test_metrics,
                "config": {
                    "patch_size": args.patch_size,
                    "context_length": args.context_length,
                    "max_epochs": args.max_epochs,
                    "loss_type": trainer_config.loss_type,
                },
            },
            indent=2,
        )
    )

    print()
    print("=" * 80)
    print("✅ TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Checkpoints saved to:   {save_dir}")
    print(f"Final metrics:          {summary_path}")
    print()
    print("VALIDATION METRICS:")
    for k, v in val_metrics.items():
        print(f"  {k:25s} {v:.6f}")
    print()
    print("TEST METRICS:")
    for k, v in test_metrics.items():
        print(f"  {k:25s} {v:.6f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
