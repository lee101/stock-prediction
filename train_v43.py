#!/usr/bin/env python3
"""Train neuraldailyv4.3 model with HARD fills (no soft simulation).

V4.3 Key insight: Epoch 1 had best backtest performance!
- The soft fills during training cause overfitting
- Use hard fills (temperature=0) to match inference exactly
- Train for very few epochs (5-10)

Usage:
    python train_v43.py --equity-only --epochs 5
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
    parser = argparse.ArgumentParser(description="Train V4.3 - hard fills")

    # Training params
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs (few!)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate (lower)")
    parser.add_argument("--dry-run", type=int, default=None, help="Stop after N steps")

    # V4.3 params - same as V4.2 but with hard fills
    parser.add_argument("--num-windows", type=int, default=2, help="Number of windows")
    parser.add_argument("--window-size", type=int, default=3, help="Days per window")

    # Model params - smaller to reduce overfitting
    parser.add_argument("--dim", type=int, default=128, help="Hidden dimension (smaller)")
    parser.add_argument("--layers", type=int, default=2, help="Number of layers (fewer)")

    # Data params
    parser.add_argument("--seq-len", type=int, default=64, help="Sequence length (shorter)")
    parser.add_argument("--equity-only", action="store_true", help="Train only on equities")

    # Temperature - the key change!
    parser.add_argument("--temperature", type=float, default=0.0, help="Fill temperature (0=hard)")

    # Run params
    parser.add_argument("--run-name", type=str, default=None, help="Run name")

    return parser.parse_args()


def main():
    args = parse_args()

    # Generate run name
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"v43_hard_{timestamp}"

    logger.info(f"Starting V4.3 training run: {args.run_name}")
    logger.info("V4.3 KEY CHANGES:")
    logger.info(f"  - HARD FILLS: temperature={args.temperature} (match inference exactly)")
    logger.info(f"  - Few epochs: {args.epochs} (early stopping insight)")
    logger.info(f"  - Smaller model: {args.dim}d x {args.layers}L (reduce overfitting)")
    logger.info(f"  - Lower LR: {args.lr}")

    # Select symbols - exclude META/MSFT
    if args.equity_only:
        symbols = ("SPY", "QQQ", "AAPL", "GOOGL", "AMZN", "NVDA", "TSLA")
    else:
        symbols = (
            "SPY", "QQQ", "AAPL", "GOOGL", "AMZN", "NVDA", "TSLA",
            "BTCUSD", "ETHUSD",
        )

    logger.info(f"Training on {len(symbols)} symbols: {symbols}")
    logger.info("  (Excluding META/MSFT - poor backtest performance)")

    lookahead_days = args.num_windows * args.window_size

    # Dataset config
    dataset_config = DailyDatasetConfigV4(
        symbols=symbols,
        sequence_length=args.seq_len,
        lookahead_days=lookahead_days,
        validation_days=40,
        min_history_days=300,
        include_weekly_features=True,
    )

    # Training config with HARD fills
    training_config = DailyTrainingConfigV4(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        sequence_length=args.seq_len,
        lookahead_days=lookahead_days,

        patch_size=5,
        num_windows=args.num_windows,
        window_size=args.window_size,
        num_quantiles=3,
        trim_fraction=0.25,

        transformer_dim=args.dim,
        transformer_layers=args.layers,
        transformer_heads=4,
        transformer_kv_heads=2,

        max_hold_days=lookahead_days,
        min_hold_days=1,
        maker_fee=0.0008,
        equity_max_leverage=2.0,
        crypto_max_leverage=1.0,

        # V4.3: HARD FILLS - no temperature annealing
        initial_temperature=args.temperature,
        final_temperature=args.temperature,
        temp_warmup_epochs=0,
        temp_anneal_epochs=0,

        # Optimizer with more regularization
        optimizer_name="adamw",
        grad_clip=0.5,  # Tighter clipping

        # Loss weights - emphasize Sharpe and forced exit penalty
        return_loss_weight=1.0,
        sharpe_loss_weight=1.0,  # Higher
        forced_exit_penalty=1.0,  # Higher
        quantile_calibration_weight=0.0,
        position_regularization=0.2,  # Stronger

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

    # Create trainer
    logger.info("Creating V4.3 trainer...")
    trainer = NeuralDailyTrainerV41(training_config, data_module)

    # Print model info
    total_params = sum(p.numel() for p in trainer.model.parameters())
    logger.info(f"Model: {total_params:,} parameters")

    # Training
    logger.info("=" * 60)
    logger.info("Starting V4.3 training with HARD fills...")
    logger.info("=" * 60)

    try:
        history = trainer.train(log_every=1)  # Log every epoch

        logger.info("=" * 60)
        logger.info("Training complete!")
        logger.info("=" * 60)

        if history.get("val_sharpe"):
            best_idx = max(range(len(history["val_sharpe"])), key=lambda i: history["val_sharpe"][i])
            logger.info(f"Best validation Sharpe: {history['val_sharpe'][best_idx]:.4f} (epoch {best_idx + 1})")

        logger.info(f"Checkpoints saved to: {trainer.checkpoint_dir}")

    except KeyboardInterrupt:
        logger.warning("Training interrupted")
        sys.exit(1)


if __name__ == "__main__":
    main()
