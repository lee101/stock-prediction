#!/usr/bin/env python3
"""Training script for Neural Hourly Stocks V5.

Trains on all 280 available stock symbols with market hours filtering.

Usage:
    python train_stocks_hourlyv5.py --epochs 100
    python train_stocks_hourlyv5.py --symbols AAPL,MSFT,NVDA --epochs 50
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from neuralhourlystocksv5.config import DatasetConfigStocksV5, TrainingConfigStocksV5
from neuralhourlystocksv5.data import (
    HourlyStockDataModuleV5,
    MultiSymbolStockDataModuleV5,
    get_all_stock_symbols,
)
from neuralhourlystocksv5.trainer import HourlyStockTrainerV5, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train V5 Hourly Stock Model")

    # Data
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated list of symbols (default: all available stocks)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="trainingdatahourly/stocks",
        help="Path to hourly stock data directory",
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
        "--seq-length", type=int, default=168, help="Sequence length (bars)"
    )
    parser.add_argument(
        "--val-hours", type=int, default=70, help="Validation hours (~10 trading days)"
    )

    # Checkpoints
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="neuralhourlystocksv5/checkpoints",
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
    parser.add_argument(
        "--no-market-hours-filter",
        action="store_true",
        help="Disable market hours filtering (use all data)",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=None,
        help="Limit to N symbols for faster testing",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("Neural Hourly Stocks V5 - Training")
    print("=" * 60)

    # Set seed
    set_seed(args.seed)

    # Get symbols
    data_root = Path(args.data_root)
    if args.symbols:
        symbols = tuple(s.strip().upper() for s in args.symbols.split(","))
    else:
        # Load all available symbols
        symbols = tuple(get_all_stock_symbols(data_root))
        if not symbols:
            print(f"No stock data found in {data_root}")
            return

    if args.max_symbols:
        symbols = symbols[: args.max_symbols]

    print(f"\nSymbols: {len(symbols)} stocks")
    if len(symbols) <= 10:
        print(f"  {', '.join(symbols)}")
    else:
        print(f"  {', '.join(symbols[:5])} ... {', '.join(symbols[-5:])}")
    print(f"Data root: {data_root}")
    print(f"Market hours filter: {not args.no_market_hours_filter}")

    # Create dataset config
    dataset_config = DatasetConfigStocksV5(
        symbols=symbols,
        data_root=data_root,
        sequence_length=args.seq_length,
        validation_hours=args.val_hours,
        lookahead_hours=24,
        market_hours_only=not args.no_market_hours_filter,
    )

    # Create training config
    training_config = TrainingConfigStocksV5(
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
            data_module = MultiSymbolStockDataModuleV5(symbols, dataset_config)
            print(f"Multi-symbol mode: {len(data_module.modules)} symbols loaded")
            if data_module.failed_symbols:
                print(f"Failed to load: {len(data_module.failed_symbols)} symbols")
        else:
            data_module = HourlyStockDataModuleV5(dataset_config)
            print(f"Single symbol mode: {symbols[0]}")

        print(f"Train samples: {len(data_module.train_dataset)}")
        print(f"Val samples: {len(data_module.val_dataset)}")
        print(f"Features: {len(data_module.feature_columns)}")
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("\nMake sure hourly stock data exists in the data directory.")
        print("Expected format: trainingdatahourly/stocks/AAPL.csv")
        return
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Create trainer
    print("\nInitializing trainer...")
    trainer = HourlyStockTrainerV5(training_config, data_module)

    # Load checkpoint if specified
    if args.load_checkpoint:
        print(f"Loading checkpoint: {args.load_checkpoint}")
        checkpoint = torch.load(args.load_checkpoint, map_location="cpu", weights_only=False)
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Resumed from epoch {checkpoint.get('epoch', 0)}")

    # Train
    print("\nStarting training...")
    print(f"Device: {training_config.device or 'auto'}")
    print(f"Compile: {training_config.use_compile}")
    print(f"Maker fee: {training_config.maker_fee * 10000:.1f} bps")
    print()

    result = trainer.train()

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best Sortino: {result['best_sortino']:.2f}")
    print(f"Final epoch: {result['final_epoch']}")

    # Run final backtest
    if not args.skip_backtest and result["history"]:
        print("\nFinal Validation Metrics:")
        final_metrics = result["history"][-1]
        print(f"  Sortino:     {final_metrics.get('val_sortino', 0):.2f}")
        print(f"  Return:      {final_metrics.get('val_return', 0)*100:.2f}%")
        print(f"  TP Rate:     {final_metrics.get('val_tp_rate', 0)*100:.1f}%")
        print(f"  Hold Hours:  {final_metrics.get('val_hold_hours', 0):.1f}")

    print("\nDone!")
    print(f"Checkpoint saved to: {training_config.checkpoint_root}")


if __name__ == "__main__":
    main()
