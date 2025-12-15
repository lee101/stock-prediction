#!/usr/bin/env python3
"""V2 Neural Daily Training entry point.

Key V2 features:
- Temperature annealing from soft to binary fills
- Modern transformer architecture (RoPE, MQA, QK-norm)
- Dual optimizer (Muon + AdamW)
- Unified simulation matching inference
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from loguru import logger

from neuraldailyv2 import (
    DailyDatasetConfigV2,
    DailyTrainingConfigV2,
    NeuralDailyTrainerV2,
)
from neuraldailyv2.data import DailyDataModuleV2


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train NeuralDaily V2 model.")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.0003)

    # Model architecture
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Temperature annealing
    parser.add_argument("--initial-temp", type=float, default=0.01)
    parser.add_argument("--final-temp", type=float, default=0.0001)
    parser.add_argument("--temp-warmup", type=int, default=10)
    parser.add_argument("--temp-anneal", type=int, default=150)

    # Leverage limits
    parser.add_argument("--equity-max-leverage", type=float, default=2.0)
    parser.add_argument("--crypto-max-leverage", type=float, default=1.0)

    # Loss parameters
    parser.add_argument("--maker-fee", type=float, default=0.0008)
    parser.add_argument("--return-weight", type=float, default=0.08)
    parser.add_argument("--leverage-fee-rate", type=float, default=0.065)

    # Optimizer
    parser.add_argument("--optimizer", choices=["dual", "adamw", "muon"], default="dual")
    parser.add_argument("--matrix-lr", type=float, default=0.02)
    parser.add_argument("--embed-lr", type=float, default=0.2)
    parser.add_argument("--head-lr", type=float, default=0.004)

    # Data
    parser.add_argument("--data-root", default="trainingdata/train")
    parser.add_argument("--forecast-cache", default="strategytraining/forecast_cache")
    parser.add_argument("--validation-days", type=int, default=40)
    parser.add_argument("--symbols", nargs="*", help="Symbols to train on.")
    parser.add_argument("--crypto-only", action="store_true")

    # Training settings
    parser.add_argument("--run-name", default="neuraldailyv2")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--use-compile", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--checkpoint", help="Checkpoint to resume from.")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--dry-run", type=int, help="Run only N steps.")

    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_config(args: argparse.Namespace) -> Tuple[DailyTrainingConfigV2, DailyDatasetConfigV2]:
    """Build training and dataset configs from args."""

    # Default symbols if not specified
    default_symbols = (
        "SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META",
        "V", "MA", "JPM", "GS", "BAC", "WMT", "COST", "HD", "MCD",
        "BTCUSD", "ETHUSD",
    )
    symbols = tuple(s.upper() for s in (args.symbols or default_symbols))

    dataset_config = DailyDatasetConfigV2(
        symbols=symbols,
        data_root=Path(args.data_root),
        forecast_cache_dir=Path(args.forecast_cache),
        sequence_length=args.sequence_length,
        validation_days=args.validation_days,
        crypto_only=args.crypto_only,
    )

    training_config = DailyTrainingConfigV2(
        epochs=args.epochs,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        learning_rate=args.learning_rate,
        transformer_dim=args.hidden_dim,
        transformer_layers=args.num_layers,
        transformer_heads=args.num_heads,
        transformer_kv_heads=args.num_kv_heads,
        transformer_dropout=args.dropout,
        initial_temperature=args.initial_temp,
        final_temperature=args.final_temp,
        temp_warmup_epochs=args.temp_warmup,
        temp_anneal_epochs=args.temp_anneal,
        equity_max_leverage=args.equity_max_leverage,
        crypto_max_leverage=args.crypto_max_leverage,
        maker_fee=args.maker_fee,
        return_weight=args.return_weight,
        leverage_fee_rate=args.leverage_fee_rate,
        optimizer_name=args.optimizer,
        matrix_lr=args.matrix_lr,
        embed_lr=args.embed_lr,
        head_lr=args.head_lr,
        run_name=args.run_name,
        device=args.device,
        seed=args.seed,
        use_compile=args.use_compile,
        use_amp=not args.no_amp,
        preload_checkpoint_path=args.checkpoint,
        force_retrain=args.force_retrain,
        dry_train_steps=args.dry_run,
        dataset=dataset_config,
    )

    return training_config, dataset_config


def main() -> int:
    """Main entry point."""
    args = parse_args()
    set_seed(args.seed)

    logger.info("=" * 60)
    logger.info("NeuralDaily V2 Training")
    logger.info("=" * 60)

    # Build configs
    training_config, dataset_config = build_config(args)

    logger.info(f"Run name: {training_config.run_name}")
    logger.info(f"Epochs: {training_config.epochs}")
    logger.info(f"Temperature: {training_config.initial_temperature} -> {training_config.final_temperature}")
    logger.info(f"Leverage limits: stocks={training_config.equity_max_leverage}x, crypto={training_config.crypto_max_leverage}x")
    logger.info(f"Symbols: {len(dataset_config.symbols)}")

    # Create data module
    data_module = DailyDataModuleV2(dataset_config)

    # Create trainer
    trainer = NeuralDailyTrainerV2(training_config, data_module)

    # Run training
    try:
        history = trainer.train()
        logger.info("Training complete!")
        logger.info(f"Best val sortino: {trainer.best_val_score:.4f}")
        return 0
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
        return 1
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
