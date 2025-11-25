#!/usr/bin/env python3
"""Training script for neural daily model with simulator validation after each epoch.

This script trains the neural daily model and validates each checkpoint against the
actual simulator to catch overfitting early. The simulator validation is the ground
truth for whether the model will work in production.

Usage:
    python train_neural_daily_with_validation.py --epochs 10 --symbols AAPL MSFT NVDA

    # Train 1 epoch, validate with simulator, repeat
    python train_neural_daily_with_validation.py --epochs 1 --validate-after-epoch
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train neural daily model with simulator validation.")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--transformer-dim", type=int, default=256)
    parser.add_argument("--transformer-layers", type=int, default=4)
    parser.add_argument("--transformer-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Dataset parameters
    parser.add_argument("--symbols", nargs="*", help="Symbols to train on.")
    parser.add_argument("--data-root", default="trainingdata/train")
    parser.add_argument("--forecast-cache", default="strategytraining/forecast_cache")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--validation-days", type=int, default=90)
    parser.add_argument("--crypto-only", action="store_true")

    # Augmentation parameters
    parser.add_argument("--symbol-dropout-rate", type=float, default=0.1)
    parser.add_argument("--permutation-rate", type=float, default=0.5)
    parser.add_argument("--price-scale-prob", type=float, default=0.2)

    # Leverage and fees
    parser.add_argument("--maker-fee", type=float, default=0.0008)
    parser.add_argument("--equity-max-leverage", type=float, default=2.0)
    parser.add_argument("--crypto-max-leverage", type=float, default=1.0)
    parser.add_argument("--leverage-fee-rate", type=float, default=0.065)

    # Validation parameters
    parser.add_argument("--validate-after-epoch", action="store_true",
                        help="Run simulator validation after each epoch.")
    parser.add_argument("--sim-days", type=int, default=10,
                        help="Number of days for simulator validation.")
    parser.add_argument("--sim-start-date", help="Start date for simulator validation.")
    parser.add_argument("--early-stop-patience", type=int, default=5,
                        help="Stop training if simulator PnL doesn't improve for N epochs.")
    parser.add_argument("--min-sim-pnl", type=float, default=-0.05,
                        help="Minimum simulator PnL to continue training.")

    # Output parameters
    parser.add_argument("--run-name", help="Name for this training run.")
    parser.add_argument("--checkpoint-root", default="neuraldailytraining/checkpoints")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Quick test with minimal steps.")

    return parser.parse_args()


def run_simulator_validation(
    checkpoint_path: Path,
    symbols: Sequence[str],
    *,
    data_root: str,
    forecast_cache: str,
    sequence_length: int,
    validation_days: int,
    sim_days: int,
    sim_start_date: str | None,
    maker_fee: float,
) -> dict:
    """Run simulator on checkpoint and return results."""
    from neuraldailytraining import DailyTradingRuntime
    from neuraldailytraining.config import DailyDatasetConfig
    from neuraldailymarketsimulator.simulator import NeuralDailyMarketSimulator

    dataset_cfg = DailyDatasetConfig(
        symbols=tuple(symbols),
        data_root=Path(data_root),
        forecast_cache_dir=Path(forecast_cache),
        sequence_length=sequence_length,
        validation_days=validation_days,
    )

    runtime = DailyTradingRuntime(
        checkpoint_path,
        dataset_config=dataset_cfg,
        confidence_threshold=None,  # Don't filter by confidence for validation
    )

    simulator = NeuralDailyMarketSimulator(
        runtime,
        symbols,
        stock_fee=maker_fee,
        crypto_fee=maker_fee,
        initial_cash=1.0,
    )

    results, summary = simulator.run(start_date=sim_start_date, days=sim_days)

    return {
        "pnl": summary["pnl"],
        "sortino": summary["sortino"],
        "final_equity": summary["final_equity"],
        "max_leverage": summary["max_leverage"],
        "total_leverage_costs": summary["total_leverage_costs"],
        "daily_returns": [r.daily_return for r in results],
    }


def train_with_validation(args: argparse.Namespace) -> None:
    """Main training loop with simulator validation."""
    from neuraldailytraining import (
        DailyDataModule,
        DailyDatasetConfig,
        DailyTrainingConfig,
        NeuralDailyTrainer,
    )

    # Setup symbols
    if args.symbols:
        symbols = tuple(s.upper() for s in args.symbols)
    else:
        symbols = DailyDatasetConfig().symbols

    print(f"Training on {len(symbols)} symbols")

    # Build configs
    dataset_cfg = DailyDatasetConfig(
        symbols=symbols,
        data_root=Path(args.data_root),
        forecast_cache_dir=Path(args.forecast_cache),
        sequence_length=args.sequence_length,
        val_fraction=args.val_fraction,
        validation_days=args.validation_days,
        symbol_dropout_rate=args.symbol_dropout_rate,
        crypto_only=args.crypto_only,
    )

    run_name = args.run_name or time.strftime("neuraldaily_validated_%Y%m%d_%H%M%S")

    training_cfg = DailyTrainingConfig(
        epochs=1,  # Train 1 epoch at a time for validation
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        learning_rate=args.learning_rate,
        transformer_dim=args.transformer_dim,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        transformer_dropout=args.dropout,
        maker_fee=args.maker_fee,
        equity_max_leverage=args.equity_max_leverage,
        crypto_max_leverage=args.crypto_max_leverage,
        leverage_fee_rate=args.leverage_fee_rate,
        permutation_rate=args.permutation_rate,
        price_scale_probability=args.price_scale_prob,
        run_name=run_name,
        checkpoint_root=Path(args.checkpoint_root),
        device=args.device,
        dry_train_steps=100 if args.dry_run else None,
        dataset=dataset_cfg,
    )

    # Initialize data module once
    data_module = DailyDataModule(dataset_cfg)
    print(f"Dataset: {len(data_module.train_dataset)} train samples, {len(data_module.val_dataset)} val samples")

    # Training loop with validation
    best_sim_pnl = float("-inf")
    best_checkpoint = None
    patience_counter = 0
    validation_history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")

        # Update epoch in config
        training_cfg.epochs = 1
        if epoch > 1:
            # Load previous checkpoint to continue training
            training_cfg.preload_checkpoint_path = best_checkpoint or last_checkpoint

        # Train one epoch
        trainer = NeuralDailyTrainer(training_cfg, data_module)
        artifacts = trainer.train()

        last_checkpoint = artifacts.best_checkpoint
        print(f"Checkpoint saved: {last_checkpoint}")

        # Get training metrics
        if artifacts.history:
            train_entry = artifacts.history[-1]
            print(f"Train - Score: {train_entry.train_score:.4f}, Return: {train_entry.train_return:.4f}")
            print(f"Val   - Score: {train_entry.val_score:.4f}, Return: {train_entry.val_return:.4f}")
            if train_entry.binary_return is not None:
                print(f"Binary Val - Sortino: {train_entry.binary_sortino:.4f}, Return: {train_entry.binary_return:.4f}")

        # Run simulator validation
        if args.validate_after_epoch and last_checkpoint:
            print(f"\nRunning simulator validation ({args.sim_days} days)...")
            try:
                sim_results = run_simulator_validation(
                    last_checkpoint,
                    list(symbols),
                    data_root=args.data_root,
                    forecast_cache=args.forecast_cache,
                    sequence_length=args.sequence_length,
                    validation_days=args.validation_days,
                    sim_days=args.sim_days,
                    sim_start_date=args.sim_start_date,
                    maker_fee=args.maker_fee,
                )

                sim_pnl = sim_results["pnl"]
                sim_sortino = sim_results["sortino"]

                print(f"Simulator - PnL: {sim_pnl:.4f}, Sortino: {sim_sortino:.4f}, "
                      f"Max Leverage: {sim_results['max_leverage']:.2f}x")

                validation_entry = {
                    "epoch": epoch,
                    "train_score": train_entry.train_score if artifacts.history else 0,
                    "val_score": train_entry.val_score if artifacts.history else 0,
                    "sim_pnl": sim_pnl,
                    "sim_sortino": sim_sortino,
                    "checkpoint": str(last_checkpoint),
                }
                validation_history.append(validation_entry)

                # Check for improvement
                if sim_pnl > best_sim_pnl:
                    print(f"  ✓ New best simulator PnL: {sim_pnl:.4f} (prev: {best_sim_pnl:.4f})")
                    best_sim_pnl = sim_pnl
                    best_checkpoint = last_checkpoint
                    patience_counter = 0
                else:
                    patience_counter += 1
                    print(f"  ✗ No improvement (patience: {patience_counter}/{args.early_stop_patience})")

                # Early stopping checks
                if sim_pnl < args.min_sim_pnl:
                    print(f"\n⚠ Simulator PnL {sim_pnl:.4f} below threshold {args.min_sim_pnl}. "
                          "Model may be overfitting.")

                if patience_counter >= args.early_stop_patience:
                    print(f"\n⚠ Early stopping: No improvement for {args.early_stop_patience} epochs")
                    break

            except Exception as e:
                print(f"Simulator validation failed: {e}")
                import traceback
                traceback.print_exc()

        # Update run name for next epoch to use same directory
        training_cfg.run_name = run_name

    # Write validation summary
    summary_path = Path(args.checkpoint_root) / run_name / "validation_summary.json"
    summary = {
        "total_epochs": epoch,
        "best_sim_pnl": best_sim_pnl,
        "best_checkpoint": str(best_checkpoint) if best_checkpoint else None,
        "validation_history": validation_history,
        "config": {
            "symbols": list(symbols),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "sim_days": args.sim_days,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nValidation summary written to {summary_path}")

    print(f"\n{'='*60}")
    print("Training Complete")
    print(f"{'='*60}")
    print(f"Best checkpoint: {best_checkpoint}")
    print(f"Best simulator PnL: {best_sim_pnl:.4f}")

    # Show how to run final validation
    if best_checkpoint:
        print(f"\nTo validate the best checkpoint:")
        print(f"  python -m neuraldailymarketsimulator.simulator \\")
        print(f"    --checkpoint {best_checkpoint} \\")
        print(f"    --days 10 --start-date 2025-11-11")


def main():
    args = parse_args()
    train_with_validation(args)


if __name__ == "__main__":
    main()
