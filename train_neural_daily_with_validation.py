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
    parser.add_argument(
        "--symbol-source",
        choices=("default", "broad_mixed"),
        default="default",
        help="How to resolve symbols when --symbols is not provided.",
    )
    parser.add_argument("--max-symbols", type=int, default=48, help="Cap for auto-selected broad universes.")
    parser.add_argument(
        "--per-group-cap",
        type=int,
        default=2,
        help="Max extra symbols to add from each group when using --symbol-source broad_mixed.",
    )
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
    parser.add_argument(
        "--smoothness-penalty",
        type=float,
        default=0.0,
        help="Penalty on jagged per-step returns during training.",
    )

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
    parser.add_argument(
        "--selection-metric",
        choices=("pnl", "sortino", "goodness"),
        default="goodness",
        help="Metric used to keep the best validated checkpoint.",
    )

    # Output parameters
    parser.add_argument("--run-name", help="Name for this training run.")
    parser.add_argument("--checkpoint-root", default="neuraldailytraining/checkpoints")
    parser.add_argument("--device", default=None)
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-train-batches-per-epoch", type=int, default=None)
    parser.add_argument("--max-val-batches-per-epoch", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Quick test with minimal steps.")

    return parser.parse_args()


def _epoch_run_name(base_run_name: str, epoch: int) -> str:
    return f"{base_run_name}_epoch{int(epoch):03d}"


def _validation_metric_value(summary: dict, metric: str) -> float:
    field_map = {
        "pnl": "pnl",
        "sortino": "sortino",
        "goodness": "goodness_score",
    }
    field = field_map.get(metric, metric)
    return float(summary.get(field, 0.0) or 0.0)


def _resolve_training_symbols(args: argparse.Namespace) -> tuple[str, ...]:
    if args.symbols:
        return tuple(s.upper() for s in args.symbols)

    if args.symbol_source == "broad_mixed":
        from neuraldailytraining.symbol_groups import build_broad_symbol_universe

        broad_symbols = build_broad_symbol_universe(
            data_root=args.data_root,
            max_symbols=args.max_symbols,
            per_group_cap=args.per_group_cap,
        )
        if broad_symbols:
            return tuple(broad_symbols)

    from neuraldailytraining.config import DailyDatasetConfig

    return tuple(DailyDatasetConfig().symbols)


def _resolve_dataset_window(args: argparse.Namespace) -> tuple[float, int]:
    effective_val_fraction = float(args.val_fraction)
    effective_validation_days = int(args.validation_days)
    if args.dry_run:
        effective_val_fraction = min(effective_val_fraction, 0.05)
        effective_validation_days = min(effective_validation_days, 20)
    return effective_val_fraction, effective_validation_days


def _resolve_epoch_batch_limits(args: argparse.Namespace) -> tuple[int | None, int | None]:
    effective_max_train_batches = args.max_train_batches_per_epoch
    effective_max_val_batches = args.max_val_batches_per_epoch
    if args.dry_run:
        if effective_max_train_batches is None:
            effective_max_train_batches = 100
        if effective_max_val_batches is None:
            effective_max_val_batches = 8
    return effective_max_train_batches, effective_max_val_batches


def _resolve_simulation_start_date(simulator, sim_days: int, sim_start_date: str | None) -> str | None:
    if sim_start_date:
        return sim_start_date
    available_dates = simulator._available_dates()  # type: ignore[attr-defined]
    if not available_dates:
        return None
    start_idx = max(len(available_dates) - max(int(sim_days), 1), 0)
    return str(available_dates[start_idx])


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

    resolved_start_date = _resolve_simulation_start_date(simulator, sim_days, sim_start_date)
    results, summary = simulator.run(start_date=resolved_start_date, days=sim_days)

    return {
        "start_date": resolved_start_date,
        "pnl": summary["pnl"],
        "total_return": summary.get("total_return", 0.0),
        "sortino": summary["sortino"],
        "max_drawdown": summary.get("max_drawdown", 0.0),
        "pnl_smoothness": summary.get("pnl_smoothness", 0.0),
        "goodness_score": summary.get("goodness_score", 0.0),
        "trade_count": summary.get("trade_count", 0),
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

    symbols = _resolve_training_symbols(args)

    print(f"Training on {len(symbols)} symbols")

    # Build configs
    effective_val_fraction, effective_validation_days = _resolve_dataset_window(args)
    effective_max_train_batches, effective_max_val_batches = _resolve_epoch_batch_limits(args)

    dataset_cfg = DailyDatasetConfig(
        symbols=symbols,
        data_root=Path(args.data_root),
        forecast_cache_dir=Path(args.forecast_cache),
        sequence_length=args.sequence_length,
        val_fraction=effective_val_fraction,
        validation_days=effective_validation_days,
        symbol_dropout_rate=args.symbol_dropout_rate,
        crypto_only=args.crypto_only,
    )
    if args.dry_run:
        print(
            f"Dry run enabled: using val_fraction={effective_val_fraction:.3f} "
            f"validation_days={effective_validation_days} "
            f"max_train_batches={effective_max_train_batches} "
            f"max_val_batches={effective_max_val_batches}"
        )

    base_run_name = args.run_name or time.strftime("neuraldaily_validated_%Y%m%d_%H%M%S")
    summary_dir = Path(args.checkpoint_root) / base_run_name
    summary_dir.mkdir(parents=True, exist_ok=True)

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
        smoothness_penalty=args.smoothness_penalty,
        permutation_rate=args.permutation_rate,
        price_scale_probability=args.price_scale_prob,
        run_name=_epoch_run_name(base_run_name, 1),
        checkpoint_root=Path(args.checkpoint_root),
        device=args.device,
        use_amp=args.use_amp,
        use_compile=args.use_compile,
        max_train_batches_per_epoch=effective_max_train_batches,
        max_val_batches_per_epoch=effective_max_val_batches,
        dry_train_steps=100 if args.dry_run else None,
        dataset=dataset_cfg,
    )

    # Initialize data module once
    data_module = DailyDataModule(dataset_cfg)
    print(f"Dataset: {len(data_module.train_dataset)} train samples, {len(data_module.val_dataset)} val samples")

    # Training loop with validation
    best_selection_value = float("-inf")
    best_checkpoint = None
    patience_counter = 0
    validation_history = []
    last_checkpoint = None

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")

        training_cfg.epochs = 1
        training_cfg.run_name = _epoch_run_name(base_run_name, epoch)
        if epoch > 1:
            training_cfg.preload_checkpoint_path = best_checkpoint or last_checkpoint
        else:
            training_cfg.preload_checkpoint_path = None

        trainer = NeuralDailyTrainer(training_cfg, data_module)
        artifacts = trainer.train()

        last_checkpoint = artifacts.best_checkpoint
        print(f"Checkpoint saved: {last_checkpoint}")
        if not args.validate_after_epoch and last_checkpoint is not None:
            best_checkpoint = last_checkpoint
            best_selection_value = float(epoch)

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
                selection_value = _validation_metric_value(sim_results, args.selection_metric)

                print(
                    f"Simulator - PnL: {sim_pnl:.4f}, Sortino: {sim_sortino:.4f}, "
                    f"Drawdown: {sim_results['max_drawdown']:.4f}, "
                    f"Smoothness: {sim_results['pnl_smoothness']:.6f}, "
                    f"Goodness: {sim_results['goodness_score']:.4f}, "
                    f"Max Leverage: {sim_results['max_leverage']:.2f}x"
                )

                validation_entry = {
                    "epoch": epoch,
                    "epoch_run_name": training_cfg.run_name,
                    "train_score": train_entry.train_score if artifacts.history else 0,
                    "val_score": train_entry.val_score if artifacts.history else 0,
                    "sim_pnl": sim_pnl,
                    "sim_sortino": sim_sortino,
                    "sim_total_return": sim_results["total_return"],
                    "sim_max_drawdown": sim_results["max_drawdown"],
                    "sim_pnl_smoothness": sim_results["pnl_smoothness"],
                    "sim_goodness_score": sim_results["goodness_score"],
                    "selection_metric": args.selection_metric,
                    "selection_value": selection_value,
                    "checkpoint": str(last_checkpoint),
                }
                validation_history.append(validation_entry)

                if selection_value > best_selection_value:
                    print(
                        f"  ✓ New best {args.selection_metric}: {selection_value:.4f} "
                        f"(prev: {best_selection_value:.4f})"
                    )
                    best_selection_value = selection_value
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

    # Write validation summary
    summary_path = summary_dir / "validation_summary.json"
    summary = {
        "total_epochs": epoch,
        "best_selection_metric": args.selection_metric,
        "best_selection_value": best_selection_value,
        "best_checkpoint": str(best_checkpoint) if best_checkpoint else None,
        "validation_history": validation_history,
        "config": {
            "symbols": list(symbols),
            "symbol_source": args.symbol_source,
            "max_symbols": args.max_symbols,
            "per_group_cap": args.per_group_cap,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "sim_days": args.sim_days,
            "smoothness_penalty": args.smoothness_penalty,
            "selection_metric": args.selection_metric,
            "max_train_batches_per_epoch": effective_max_train_batches,
            "max_val_batches_per_epoch": effective_max_val_batches,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nValidation summary written to {summary_path}")

    print(f"\n{'='*60}")
    print("Training Complete")
    print(f"{'='*60}")
    print(f"Best checkpoint: {best_checkpoint}")
    print(f"Best {args.selection_metric}: {best_selection_value:.4f}")

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
