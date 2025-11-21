#!/usr/bin/env python3
"""Training script for all available symbols with improved architecture."""

import argparse
from pathlib import Path
from neuraldailytraining.symbol_groups import list_training_symbols
from neuraldailytraining.config import DailyDatasetConfig, DailyTrainingConfig
from neuraldailytraining.data import DailyDataModule
from neuraldailytraining.trainer import NeuralDailyTrainer


def main():
    parser = argparse.ArgumentParser(description="Train on all symbols with improved architecture")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--validation-days", type=int, default=40)
    parser.add_argument("--data-root", type=str, default="trainingdata/train")
    parser.add_argument("--forecast-cache", type=str, default="strategytraining/forecast_cache")
    parser.add_argument("--checkpoint-root", type=str, default="neuraldailytraining/checkpoints")
    args = parser.parse_args()

    # Get all available symbols
    symbols = list_training_symbols(args.data_root)
    print(f"Training on {len(symbols)} symbols")

    # Create dataset config
    dataset_config = DailyDatasetConfig(
        symbols=symbols,
        data_root=args.data_root,
        forecast_cache_dir=args.forecast_cache,
        sequence_length=args.sequence_length,
        validation_days=args.validation_days,
    )

    # Create training config
    training_config = DailyTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        sequence_length=args.sequence_length,
        device=args.device,
        checkpoint_root=Path(args.checkpoint_root),
        transformer_dim=256,
        transformer_layers=4,
        transformer_heads=8,
        transformer_dropout=0.1,
        equity_max_leverage=2.0,
        crypto_max_leverage=1.0,
        leverage_fee_rate=0.065,
        dataset=dataset_config,
    )

    # Create data module
    print("Loading data...")
    data_module = DailyDataModule(dataset_config)
    print(f"Training samples: {len(data_module.train_dataset)}")
    print(f"Validation samples: {len(data_module.val_dataset)}")

    # Create trainer and train
    trainer = NeuralDailyTrainer(training_config, data_module)
    trainer.train()


if __name__ == "__main__":
    main()
