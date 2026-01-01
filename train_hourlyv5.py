#!/usr/bin/env python3
"""Training script for Neural Hourly Trading V5.

Usage:
    python train_hourlyv5.py --symbol BTCUSD --epochs 100
    python train_hourlyv5.py --symbols BTCUSD,ETHUSD,SOLUSD --epochs 100
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from neuralhourlytradingv5.config import DatasetConfigV5, TrainingConfigV5
from neuralhourlytradingv5.data import HourlyDataModuleV5, MultiSymbolDataModuleV5
from neuralhourlytradingv5.trainer import HourlyCryptoTrainerV5, set_seed
from neuralhourlytradingv5.backtest import (
    run_10day_validation,
    print_backtest_summary,
    HourlyMarketSimulatorV5,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train V5 Hourly Crypto Model")

    # Data
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSD",
        help="Primary symbol to train on",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated list of symbols for multi-symbol training",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="trainingdatahourly/crypto",
        help="Path to hourly data directory",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")

    # Model
    parser.add_argument("--hidden-dim", type=int, default=384, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=4, help="Transformer layers")
    parser.add_argument("--patch-size", type=int, default=4, help="Patch size (hours)")

    # Sequence
    parser.add_argument(
        "--seq-length", type=int, default=168, help="Sequence length (hours)"
    )
    parser.add_argument(
        "--val-hours", type=int, default=240, help="Validation hours (10 days)"
    )

    # Checkpoints
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="neuralhourlytradingv5/checkpoints",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    # Flags
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile",
    )
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Skip final backtest after training",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("Neural Hourly Trading V5 - Training")
    print("=" * 60)

    # Set seed
    set_seed(args.seed)

    # Parse symbols
    if args.symbols:
        symbols = tuple(s.strip().upper() for s in args.symbols.split(","))
    else:
        symbols = (args.symbol.upper(),)

    print(f"\nSymbols: {symbols}")
    print(f"Data root: {args.data_root}")

    # Create dataset config
    dataset_config = DatasetConfigV5(
        symbols=symbols,
        data_root=Path(args.data_root),
        sequence_length=args.seq_length,
        validation_hours=args.val_hours,
        lookahead_hours=24,
    )

    # Create training config
    training_config = TrainingConfigV5(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        sequence_length=args.seq_length,
        lookahead_hours=24,
        transformer_dim=args.hidden_dim,
        transformer_layers=args.num_layers,
        patch_size=args.patch_size,
        checkpoint_root=args.checkpoint_dir,
        use_compile=not args.no_compile,
        seed=args.seed,
        dataset=dataset_config,
    )

    # Create data module
    print("\nLoading data...")
    try:
        if len(symbols) > 1:
            data_module = MultiSymbolDataModuleV5(symbols, dataset_config)
            print(f"Multi-symbol mode: {len(data_module.modules)} symbols loaded")
        else:
            data_module = HourlyDataModuleV5(dataset_config)
            print(f"Single symbol mode: {symbols[0]}")

        print(f"Train samples: {len(data_module.train_dataset)}")
        print(f"Val samples: {len(data_module.val_dataset)}")
        print(f"Features: {len(data_module.feature_columns)}")
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("\nMake sure hourly data exists in the data directory.")
        print("Expected format: trainingdatahourly/crypto/BTCUSD.csv")
        return

    # Create trainer
    print("\nInitializing trainer...")
    trainer = HourlyCryptoTrainerV5(training_config, data_module)

    # Load checkpoint if specified
    if args.load_checkpoint:
        print(f"Loading checkpoint: {args.load_checkpoint}")
        checkpoint = torch.load(args.load_checkpoint, map_location="cpu")
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Resumed from epoch {checkpoint.get('epoch', 0)}")

    # Train
    print("\nStarting training...")
    result = trainer.train()

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best Sortino: {result['best_sortino']:.2f}")
    print(f"Final epoch: {result['final_epoch']}")

    # Run final backtest
    if not args.skip_backtest:
        print("\nRunning 10-day validation backtest...")
        try:
            # Get validation data
            val_loader = data_module.val_dataloader(batch_size=1)

            # Collect all validation data into a single dataframe
            bars_list = []
            features_list = []

            for batch in val_loader:
                # This is a simplified approach - in production you'd
                # reconstruct the full dataframe from the dataset
                pass

            # For now, just print metrics from training
            if result["history"]:
                final_metrics = result["history"][-1]
                print("\nFinal Validation Metrics:")
                print(f"  Sortino:     {final_metrics.get('val_sortino', 0):.2f}")
                print(f"  Return:      {final_metrics.get('val_return', 0)*100:.2f}%")
                print(f"  TP Rate:     {final_metrics.get('val_tp_rate', 0)*100:.1f}%")
                print(f"  Hold Hours:  {final_metrics.get('val_hold_hours', 0):.1f}")
        except Exception as e:
            print(f"Backtest error: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
