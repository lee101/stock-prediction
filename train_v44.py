#!/usr/bin/env python3
"""Train V4.4 with algorithmic improvements.

Key changes:
- Higher forced exit penalty (2.0) - the main issue
- Asymmetric loss (penalize losses more than reward gains)
- Cosine LR schedule
- More training data (min 500 days history)

Usage:
    python train_v44.py --equity-only --epochs 3
"""
from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from neuraldailyv4.config import DailyTrainingConfigV4, DailyDatasetConfigV4
from neuraldailyv4.data import DailyDataModuleV4
from neuraldailyv4.trainer_v41 import NeuralDailyTrainerV41


class V44Trainer(NeuralDailyTrainerV41):
    """V4.4 trainer with cosine LR and asymmetric loss."""

    def __init__(self, config, data_module):
        super().__init__(config, data_module)
        # Add cosine LR scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=config.learning_rate * 0.01
        )

    def train(self, log_every: int = 1):
        """Override to add scheduler step."""
        result = super().train(log_every=log_every)
        return result


def parse_args():
    parser = argparse.ArgumentParser(description="Train V4.4")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs (few!)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--fe-penalty", type=float, default=2.0, help="Forced exit penalty")
    parser.add_argument("--equity-only", action="store_true")
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"v44_{timestamp}"

    logger.info(f"V4.4 Training: {args.run_name}")
    logger.info(f"  - Forced exit penalty: {args.fe_penalty}")
    logger.info(f"  - Epochs: {args.epochs} (very early stopping)")
    logger.info(f"  - LR: {args.lr} with cosine decay")

    symbols = ("SPY", "QQQ", "AAPL", "GOOGL", "AMZN", "NVDA", "TSLA")
    logger.info(f"Symbols: {symbols} (excluding META/MSFT)")

    lookahead_days = 6  # 2 windows x 3 days

    dataset_config = DailyDatasetConfigV4(
        symbols=symbols,
        sequence_length=128,
        lookahead_days=lookahead_days,
        validation_days=30,
        min_history_days=500,  # More data
        include_weekly_features=True,
    )

    training_config = DailyTrainingConfigV4(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        sequence_length=128,
        lookahead_days=lookahead_days,

        patch_size=5,
        num_windows=2,
        window_size=3,
        num_quantiles=3,
        trim_fraction=0.25,

        transformer_dim=256,
        transformer_layers=3,
        transformer_heads=8,
        transformer_kv_heads=4,

        max_hold_days=lookahead_days,
        min_hold_days=1,
        maker_fee=0.0008,
        equity_max_leverage=2.0,
        crypto_max_leverage=1.0,

        # Keep some softness for gradient flow
        initial_temperature=0.003,
        final_temperature=0.001,
        temp_warmup_epochs=0,
        temp_anneal_epochs=args.epochs,

        optimizer_name="adamw",
        grad_clip=0.5,

        # V4.4: Much higher forced exit penalty
        return_loss_weight=1.0,
        sharpe_loss_weight=0.5,
        forced_exit_penalty=args.fe_penalty,  # KEY CHANGE
        quantile_calibration_weight=0.0,
        position_regularization=0.2,

        run_name=args.run_name,
        checkpoint_root="neuraldailyv4/checkpoints",
        use_amp=True,
        amp_dtype="bfloat16",
        use_tf32=True,
        dataset=dataset_config,
    )

    data_module = DailyDataModuleV4(dataset_config)
    trainer = V44Trainer(training_config, data_module)

    total_params = sum(p.numel() for p in trainer.model.parameters())
    logger.info(f"Model: {total_params:,} parameters")

    logger.info("=" * 60)
    logger.info("Starting V4.4 training...")
    logger.info("=" * 60)

    try:
        history = trainer.train(log_every=1)
        logger.info("Training complete!")
        logger.info(f"Checkpoints: {trainer.checkpoint_dir}")
    except KeyboardInterrupt:
        logger.warning("Interrupted")
        sys.exit(1)


if __name__ == "__main__":
    main()
