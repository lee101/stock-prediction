#!/usr/bin/env python3
"""Train neuraldailyv4 model with Chronos-2 inspired architecture.

Key features:
- Patching: 5-day patches for faster processing
- Multi-window: Direct prediction of multiple future windows
- Quantile outputs: Price distribution instead of point estimates
- Trimmed mean: Robust aggregation across windows
- Learned position sizing: Dynamic sizing based on uncertainty

Usage:
    python train_v4.py
    python train_v4.py --epochs 300 --batch-size 32
    python train_v4.py --dry-run 100  # Quick test
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from neuraldailyv4.config import DailyTrainingConfigV4, DailyDatasetConfigV4
from neuraldailyv4.data import DailyDataModuleV4
from neuraldailyv4.trainer import NeuralDailyTrainerV4


def parse_args():
    parser = argparse.ArgumentParser(description="Train V4 neural trading model")

    # Training params
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0003, help="Base learning rate")
    parser.add_argument("--dry-run", type=int, default=None, help="Stop after N steps")

    # V4 specific params
    parser.add_argument("--patch-size", type=int, default=5, help="Patch size (days)")
    parser.add_argument("--num-windows", type=int, default=4, help="Number of prediction windows")
    parser.add_argument("--window-size", type=int, default=5, help="Days per window")
    parser.add_argument("--num-quantiles", type=int, default=3, help="Number of quantiles")
    parser.add_argument("--trim-fraction", type=float, default=0.25, help="Trimmed mean fraction")

    # Model params (wide & shallow from experiments)
    parser.add_argument("--dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--heads", type=int, default=8, help="Number of heads")

    # Data params
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length")
    parser.add_argument("--crypto-only", action="store_true", help="Train only on crypto")
    parser.add_argument("--equity-only", action="store_true", help="Train only on equities")

    # Run params
    parser.add_argument("--run-name", type=str, default=None, help="Run name")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Device")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--log-every", type=int, default=10, help="Log every N epochs")

    return parser.parse_args()


def main():
    args = parse_args()

    # Generate run name
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"v4_{timestamp}"

    logger.info(f"Starting V4 training run: {args.run_name}")
    logger.info(f"Patch size: {args.patch_size}, Windows: {args.num_windows}, Quantiles: {args.num_quantiles}")

    # Select symbols
    if args.crypto_only:
        symbols = ("BTCUSD", "ETHUSD")
        logger.info("Training on crypto only")
    elif args.equity_only:
        symbols = (
            "SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META",
        )
        logger.info("Training on equities only")
    else:
        symbols = (
            "SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META",
            "BTCUSD", "ETHUSD",
        )

    logger.info(f"Training on {len(symbols)} symbols: {symbols}")

    # Lookahead = num_windows * window_size
    lookahead_days = args.num_windows * args.window_size

    # Dataset config
    dataset_config = DailyDatasetConfigV4(
        symbols=symbols,
        sequence_length=args.seq_len,
        lookahead_days=lookahead_days,
        validation_days=40,
        min_history_days=300,
        include_weekly_features=True,
        crypto_only=args.crypto_only,
    )

    # Training config
    training_config = DailyTrainingConfigV4(
        # Training
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        sequence_length=args.seq_len,
        lookahead_days=lookahead_days,

        # V4 specific
        patch_size=args.patch_size,
        num_windows=args.num_windows,
        window_size=args.window_size,
        num_quantiles=args.num_quantiles,
        trim_fraction=args.trim_fraction,

        # Model (wide & shallow)
        transformer_dim=args.dim,
        transformer_layers=args.layers,
        transformer_heads=args.heads,
        transformer_kv_heads=args.heads // 2,

        # Simulation
        max_hold_days=lookahead_days,
        min_hold_days=1,
        maker_fee=0.0008,
        equity_max_leverage=2.0,
        crypto_max_leverage=1.0,

        # Temperature
        initial_temperature=0.01,
        final_temperature=0.0001,
        temp_warmup_epochs=10,
        temp_anneal_epochs=min(150, args.epochs - 10),

        # Optimizer
        optimizer_name="dual",
        matrix_lr=0.02,
        embed_lr=0.2,
        head_lr=0.004,
        grad_clip=1.0,

        # Loss weights
        return_loss_weight=1.0,
        sharpe_loss_weight=0.1,
        forced_exit_penalty=0.1,
        quantile_calibration_weight=0.05,
        position_regularization=0.01,

        # Run settings
        run_name=args.run_name,
        checkpoint_root="neuraldailyv4/checkpoints",
        device=args.device,
        use_compile=args.compile,
        use_amp=True,
        amp_dtype="bfloat16",
        use_tf32=True,

        # Dry run
        dry_train_steps=args.dry_run,

        # Resume
        preload_checkpoint_path=args.resume,

        # Dataset
        dataset=dataset_config,
    )

    # Create data module
    logger.info("Preparing data module...")
    data_module = DailyDataModuleV4(dataset_config)

    # Create trainer
    logger.info("Creating trainer...")
    trainer = NeuralDailyTrainerV4(training_config, data_module)

    # Print model info
    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    logger.info(f"Model: {total_params:,} total params, {trainable_params:,} trainable")

    # Training
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)

    try:
        history = trainer.train(log_every=args.log_every)

        # Print final summary
        logger.info("=" * 60)
        logger.info("Training complete!")
        logger.info("=" * 60)

        if history["val_sharpe"]:
            best_idx = max(range(len(history["val_sharpe"])), key=lambda i: history["val_sharpe"][i])
            logger.info(f"Best validation Sharpe: {history['val_sharpe'][best_idx]:.4f} (epoch {best_idx + 1})")
            logger.info(f"Best TP rate: {history['val_tp_rate'][best_idx]:.2%}")
            logger.info(f"Best avg hold: {history['val_avg_hold'][best_idx]:.1f} days")

        logger.info(f"Checkpoints saved to: {trainer.checkpoint_dir}")

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
