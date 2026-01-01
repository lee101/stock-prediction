#!/usr/bin/env python3
"""Train neuraldailyv3timed model with explicit exit timing.

This trains a model that learns:
- Entry prices (buy_price)
- Take profit targets (sell_price)
- Exit timing (exit_days: 1-10 days max hold)

The model is validated on 10-day trade episode simulations to ensure
profitability before deployment.

Usage:
    python train_v3timed.py
    python train_v3timed.py --epochs 300 --batch-size 64
    python train_v3timed.py --dry-run 100  # Quick test
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from neuraldailyv3timed.config import DailyTrainingConfigV3, DailyDatasetConfigV3
from neuraldailyv3timed.data import DailyDataModuleV3
from neuraldailyv3timed.trainer import NeuralDailyTrainerV3


def parse_args():
    parser = argparse.ArgumentParser(description="Train V3 Timed neural trading model")

    # Training params
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0003, help="Base learning rate")
    parser.add_argument("--dry-run", type=int, default=None, help="Stop after N steps (for testing)")

    # Model params
    parser.add_argument("--dim", type=int, default=256, help="Transformer hidden dimension")
    parser.add_argument("--layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--heads", type=int, default=8, help="Number of attention heads")

    # V3 specific params
    parser.add_argument("--max-hold-days", type=int, default=20, help="Maximum hold duration (1-20)")
    parser.add_argument("--lookahead-days", type=int, default=20, help="Days of future data for simulation")
    parser.add_argument("--forced-exit-penalty", type=float, default=0.1, help="Penalty for hitting deadline")
    parser.add_argument("--hold-time-penalty", type=float, default=0.02, help="Penalty for long holds")

    # Data params
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length (history)")
    parser.add_argument("--crypto-only", action="store_true", help="Train only on crypto")

    # Run params
    parser.add_argument("--run-name", type=str, default=None, help="Run name for checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")

    return parser.parse_args()


def main():
    args = parse_args()

    # Generate run name if not provided
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"v3timed_{timestamp}"

    logger.info(f"Starting V3 Timed training run: {args.run_name}")
    logger.info(f"Max hold days: {args.max_hold_days}, Lookahead: {args.lookahead_days}")

    # All symbols to train on
    all_symbols = (
        # Major indices
        "SPY", "QQQ",
        # Tech giants
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META",
        # Crypto
        "BTCUSD", "ETHUSD",
    )

    if args.crypto_only:
        all_symbols = ("BTCUSD", "ETHUSD")
        logger.info("Training on crypto only")

    logger.info(f"Training on {len(all_symbols)} symbols: {all_symbols}")

    # Dataset config with lookahead
    dataset_config = DailyDatasetConfigV3(
        symbols=all_symbols,
        sequence_length=args.seq_len,
        lookahead_days=args.lookahead_days,
        validation_days=40,  # Last 40 days for validation
        min_history_days=300,
        include_weekly_features=True,
        crypto_only=args.crypto_only,
    )

    # Training config
    training_config = DailyTrainingConfigV3(
        # Training
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        sequence_length=args.seq_len,
        lookahead_days=args.lookahead_days,

        # Model architecture
        transformer_dim=args.dim,
        transformer_layers=args.layers,
        transformer_heads=args.heads,
        transformer_kv_heads=args.heads // 2,  # MQA

        # V3 exit timing
        max_hold_days=args.max_hold_days,
        min_hold_days=1,
        forced_exit_slippage=0.001,  # 10bps slippage on forced exits
        min_exit_days=1.0,
        max_exit_days=float(args.max_hold_days),

        # Loss weights
        return_weight=0.08,
        forced_exit_penalty=args.forced_exit_penalty,
        risk_penalty=0.05,
        hold_time_penalty=args.hold_time_penalty,

        # Simulation
        maker_fee=0.0008,  # 8bps
        equity_max_leverage=2.0,
        crypto_max_leverage=1.0,

        # Temperature annealing
        initial_temperature=0.01,
        final_temperature=0.0001,
        temp_warmup_epochs=10,
        temp_anneal_epochs=150,

        # Optimizer
        optimizer_name="dual",  # Muon + AdamW
        matrix_lr=0.02,
        embed_lr=0.2,
        head_lr=0.004,
        grad_clip=1.0,

        # Scheduler
        warmup_steps=100,
        use_cosine_schedule=True,

        # Run settings
        run_name=args.run_name,
        checkpoint_root="neuraldailyv3timed/checkpoints",
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
    data_module = DailyDataModuleV3(dataset_config)

    # Create trainer
    logger.info("Creating trainer...")
    trainer = NeuralDailyTrainerV3(training_config, data_module)

    # Print model info
    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    logger.info(f"Model: {total_params:,} total params, {trainable_params:,} trainable")

    # Training
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)

    try:
        history = trainer.train()

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
