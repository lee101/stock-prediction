#!/usr/bin/env python3
"""Train neuraldailyv4.1 model with fixed train=inference alignment.

V4.1 Key fixes:
- Aggregate predictions BEFORE simulation (matches inference)
- Minimum 2% spread enforcement
- Position size regularization to prevent overfitting
- Better loss weights for risk/reward

Usage:
    python train_v41.py --equity-only --epochs 200
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
from neuraldailyv4.trainer_v41 import NeuralDailyTrainerV41


def parse_args():
    parser = argparse.ArgumentParser(description="Train V4.1 neural trading model")

    # Training params
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--dry-run", type=int, default=None, help="Stop after N steps")

    # V4.1 params
    parser.add_argument("--patch-size", type=int, default=5, help="Patch size (days)")
    parser.add_argument("--num-windows", type=int, default=4, help="Number of windows")
    parser.add_argument("--window-size", type=int, default=5, help="Days per window")
    parser.add_argument("--min-spread", type=float, default=0.02, help="Min spread (2%)")

    # Model params
    parser.add_argument("--dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--layers", type=int, default=4, help="Number of layers")

    # Data params
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length")
    parser.add_argument("--crypto-only", action="store_true", help="Train only on crypto")
    parser.add_argument("--equity-only", action="store_true", help="Train only on equities")

    # Run params
    parser.add_argument("--run-name", type=str, default=None, help="Run name")
    parser.add_argument("--log-every", type=int, default=10, help="Log every N epochs")

    return parser.parse_args()


def main():
    args = parse_args()

    # Generate run name
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"v41_{timestamp}"

    logger.info(f"Starting V4.1 training run: {args.run_name}")
    logger.info("V4.1 KEY FIXES:")
    logger.info("  - Aggregate predictions BEFORE simulation (train=inference)")
    logger.info(f"  - Minimum spread: {args.min_spread:.1%}")
    logger.info("  - Position size regularization")

    # Select symbols
    if args.crypto_only:
        symbols = ("BTCUSD", "ETHUSD")
    elif args.equity_only:
        symbols = ("SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META")
    else:
        symbols = (
            "SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META",
            "BTCUSD", "ETHUSD",
        )

    logger.info(f"Training on {len(symbols)} symbols: {symbols}")

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

    # Training config with V4.1 improvements
    training_config = DailyTrainingConfigV4(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        sequence_length=args.seq_len,
        lookahead_days=lookahead_days,

        patch_size=args.patch_size,
        num_windows=args.num_windows,
        window_size=args.window_size,
        num_quantiles=3,
        trim_fraction=0.25,

        transformer_dim=args.dim,
        transformer_layers=args.layers,
        transformer_heads=8,
        transformer_kv_heads=4,

        max_hold_days=lookahead_days,
        min_hold_days=1,
        maker_fee=0.0008,
        equity_max_leverage=2.0,
        crypto_max_leverage=1.0,

        # Temperature (less aggressive annealing)
        initial_temperature=0.005,
        final_temperature=0.0001,
        temp_warmup_epochs=20,
        temp_anneal_epochs=min(100, args.epochs - 20),

        # Optimizer (simpler, with weight decay)
        optimizer_name="adamw",
        grad_clip=1.0,

        # V4.1: Better loss weights
        return_loss_weight=1.0,
        sharpe_loss_weight=0.5,
        forced_exit_penalty=0.5,
        quantile_calibration_weight=0.0,  # Disabled for V4.1
        position_regularization=0.1,

        run_name=args.run_name,
        checkpoint_root="neuraldailyv4/checkpoints",
        use_amp=True,
        amp_dtype="bfloat16",
        use_tf32=True,
        dry_train_steps=args.dry_run,
        dataset=dataset_config,
    )

    # Create data module
    logger.info("Preparing data module...")
    data_module = DailyDataModuleV4(dataset_config)

    # Create V4.1 trainer
    logger.info("Creating V4.1 trainer...")
    trainer = NeuralDailyTrainerV41(training_config, data_module)

    # Print model info
    total_params = sum(p.numel() for p in trainer.model.parameters())
    logger.info(f"Model: {total_params:,} parameters")

    # Training
    logger.info("=" * 60)
    logger.info("Starting V4.1 training...")
    logger.info("=" * 60)

    try:
        history = trainer.train(log_every=args.log_every)

        logger.info("=" * 60)
        logger.info("Training complete!")
        logger.info("=" * 60)

        if history.get("val_sharpe"):
            best_idx = max(range(len(history["val_sharpe"])), key=lambda i: history["val_sharpe"][i])
            logger.info(f"Best validation Sharpe: {history['val_sharpe'][best_idx]:.4f} (epoch {best_idx + 1})")
            logger.info(f"Best TP rate: {history['val_tp_rate'][best_idx]:.2%}")
            logger.info(f"Best spread: {history['val_spread'][best_idx]:.2%}")

        logger.info(f"Checkpoints saved to: {trainer.checkpoint_dir}")

    except KeyboardInterrupt:
        logger.warning("Training interrupted")
        sys.exit(1)


if __name__ == "__main__":
    main()
