#!/usr/bin/env python3
"""V9: Backtest-Validated Training.

Key innovation: Use actual backtest performance for model selection instead of
simulation metrics. This addresses the gap between training optimization and
real-world performance.

Training loop:
1. Train for N epochs (e.g., 5)
2. Run actual backtest on recent data (e.g., 60 days)
3. Track backtest Sortino, return, win rate
4. Save checkpoint if backtest metrics improve
5. Early stop if no backtest improvement

This is more computationally expensive but gives honest evaluation.

Usage:
    python train_v9_backtest_validated.py --epochs 100 --eval-every 5
    python train_v9_backtest_validated.py --epochs 100 --backtest-days 90
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from neuraldailyv4.config import DailyTrainingConfigV4, DailyDatasetConfigV4
from neuraldailyv4.data import DailyDataModuleV4
from neuraldailyv4.trainer import NeuralDailyTrainerV4


# Best symbols from V8 analysis (high Sortino, positive returns)
V9_SYMBOLS = (
    # ETFs - consistently profitable
    "SPY", "QQQ", "IWM", "XLF", "XLK", "DIA",
    # Mega cap tech
    "AAPL", "GOOGL", "NVDA",
    # Financials
    "V", "MA",
    # Consumer
    "MCD", "TGT",
    # Semiconductors - highest returns (MRVL excluded - loses money)
    "LRCX", "AMAT", "QCOM", "AMD",
    # Healthcare
    "LLY", "PFE",
    # Software (only ADBE works)
    "ADBE",
)

# Symbols to exclude (negative Sortino in 90-day backtest)
V9_EXCLUDED = {
    "ORCL", "NOW", "DDOG", "PYPL", "COIN", "INTC",
    "AVGO", "COST", "CRM", "HD", "MRVL",
    "META", "MSFT", "AMZN", "NFLX",
    "BTCUSD", "AVAXUSD",
}


def calculate_sortino(returns: List[float], target: float = 0.0) -> float:
    """Calculate Sortino ratio (penalizes downside volatility only)."""
    if not returns or len(returns) < 2:
        return 0.0
    returns_arr = np.array(returns)
    excess = returns_arr - target
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float('inf') if np.mean(excess) > 0 else 0.0
    downside_std = np.std(downside)
    if downside_std < 1e-8:
        return float('inf') if np.mean(excess) > 0 else 0.0
    return float(np.mean(excess) / downside_std)


def run_backtest_evaluation(
    checkpoint_path: Path,
    symbols: List[str],
    days: int = 60,
) -> Dict[str, float]:
    """Run actual backtest and return metrics."""
    from backtest_v4 import run_backtest
    from neuraldailyv4.runtime import DailyTradingRuntimeV4

    try:
        # Load runtime from checkpoint
        runtime = DailyTradingRuntimeV4(
            checkpoint_path,
            trade_crypto=False,
            max_exit_days=2,  # Use optimal exit days
        )

        end_date = pd.Timestamp.now(tz="UTC")
        start_date = end_date - timedelta(days=days)

        # Run backtest
        trades = run_backtest(
            runtime,
            list(symbols),
            start_date,
            end_date,
            verbose=False,
            stop_loss_pct=None,  # No stop loss
        )

        if not trades:
            return {
                "backtest_return": 0.0,
                "backtest_sharpe": 0.0,
                "backtest_sortino": 0.0,
                "backtest_win_rate": 0.0,
                "backtest_trades": 0,
                "backtest_tp_rate": 0.0,
            }

        returns = [t["return_pct"] for t in trades]
        total_return = sum(returns)
        avg_return = np.mean(returns)
        std_return = np.std(returns) if len(returns) > 1 else 1e-8
        sharpe = avg_return / std_return if std_return > 0 else 0.0
        sortino = calculate_sortino(returns)
        win_rate = sum(1 for r in returns if r > 0) / len(returns)
        tp_rate = sum(1 for t in trades if t.get("tp_hit")) / len(trades)

        # Cap extreme values
        sortino = min(sortino, 100.0) if sortino != float('inf') else 100.0

        return {
            "backtest_return": total_return,
            "backtest_sharpe": sharpe,
            "backtest_sortino": sortino,
            "backtest_win_rate": win_rate,
            "backtest_trades": len(trades),
            "backtest_tp_rate": tp_rate,
        }

    except Exception as e:
        logger.error(f"Backtest evaluation failed: {e}")
        return {
            "backtest_return": -999.0,
            "backtest_sharpe": -999.0,
            "backtest_sortino": -999.0,
            "backtest_win_rate": 0.0,
            "backtest_trades": 0,
            "backtest_tp_rate": 0.0,
        }


class BacktestValidatedTrainer:
    """Trainer that validates using actual backtests."""

    def __init__(
        self,
        config: DailyTrainingConfigV4,
        data_module: DailyDataModuleV4,
        backtest_symbols: Tuple[str, ...],
        backtest_days: int = 60,
        eval_every: int = 5,
    ):
        self.config = config
        self.data_module = data_module
        self.backtest_symbols = backtest_symbols
        self.backtest_days = backtest_days
        self.eval_every = eval_every

        # Create base trainer
        self.trainer = NeuralDailyTrainerV4(config, data_module)

        # Backtest tracking
        self.best_backtest_sortino = float("-inf")
        self.best_backtest_epoch = 0
        self.backtest_history = []

    def _save_checkpoint(self, epoch: int, metrics: Dict) -> Path:
        """Save checkpoint with backtest metrics."""
        checkpoint_path = self.trainer.checkpoint_dir / f"epoch_{epoch:04d}.pt"

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.trainer.model.state_dict(),
            "metrics": metrics,
            "config": {
                "policy": self.trainer.model.config.__dict__,
                "simulation": self.trainer.sim_config.__dict__,
                "training": {
                    k: v for k, v in self.config.__dict__.items()
                    if not k.startswith("_") and k != "dataset"
                },
            },
            "normalizer": self.data_module.normalizer,
            "feature_columns": self.data_module.feature_columns,
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Update manifest
        self._update_manifest(epoch, metrics)

        return checkpoint_path

    def _update_manifest(self, epoch: int, metrics: Dict) -> None:
        """Update checkpoint manifest with backtest metrics."""
        manifest_path = self.trainer.checkpoint_dir / "manifest.json"

        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
        else:
            manifest = {
                "run_name": self.config.run_name,
                "checkpoints": [],
            }

        manifest["best_epoch"] = self.best_backtest_epoch
        manifest["best_backtest_sortino"] = self.best_backtest_sortino
        manifest["backtest_history"] = self.backtest_history

        checkpoint_name = f"epoch_{epoch:04d}.pt"
        if checkpoint_name not in manifest["checkpoints"]:
            manifest["checkpoints"].append(checkpoint_name)

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    def train(
        self,
        epochs: int,
        early_stopping_patience: int = 20,
    ) -> Dict[str, List]:
        """Train with periodic backtest validation."""
        config = self.config
        model = self.trainer.model
        optimizer = self.trainer.optimizer
        scaler = self.trainer.scaler

        train_loader = self.data_module.train_dataloader(config.batch_size, config.num_workers)
        val_loader = self.data_module.val_dataloader(config.batch_size, config.num_workers)

        logger.info(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
        logger.info(f"Backtest validation every {self.eval_every} epochs on {len(self.backtest_symbols)} symbols")
        logger.info(f"Backtest period: {self.backtest_days} days")

        history = defaultdict(list)
        epochs_without_improvement = 0

        start_time = time.time()

        for epoch in range(epochs):
            epoch_start = time.time()

            # Get temperature
            temperature = self.trainer.temp_schedule.get_temperature(epoch, epochs)

            # Training epoch
            train_metrics = self.trainer._run_epoch(
                train_loader, training=True, temperature=temperature, quiet=True
            )

            # Validation epoch
            with torch.no_grad():
                val_metrics = self.trainer._run_epoch(
                    val_loader, training=False, temperature=0.0, quiet=True
                )

            epoch_time = time.time() - epoch_start

            # Track history
            history["train_loss"].append(train_metrics["loss"])
            history["train_sharpe"].append(train_metrics["sharpe"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_sharpe"].append(val_metrics["sharpe"])

            # Periodic backtest evaluation
            if (epoch + 1) % self.eval_every == 0 or epoch == 0:
                # Save temporary checkpoint for backtest
                temp_checkpoint = self._save_checkpoint(epoch + 1, {
                    **val_metrics,
                    "train_loss": train_metrics["loss"],
                    "train_sharpe": train_metrics["sharpe"],
                })

                # Run backtest
                logger.info(f"Running backtest evaluation at epoch {epoch + 1}...")
                backtest_metrics = run_backtest_evaluation(
                    temp_checkpoint,
                    self.backtest_symbols,
                    self.backtest_days,
                )

                # Track backtest metrics
                for k, v in backtest_metrics.items():
                    history[k].append(v)

                self.backtest_history.append({
                    "epoch": epoch + 1,
                    **backtest_metrics,
                })

                # Log results
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} ({epoch_time:.1f}s) | "
                    f"Train Sharpe: {train_metrics['sharpe']:.3f} | "
                    f"Val Sharpe: {val_metrics['sharpe']:.3f} | "
                    f"BACKTEST: Return={backtest_metrics['backtest_return']:.2%}, "
                    f"Sortino={backtest_metrics['backtest_sortino']:.3f}, "
                    f"WinRate={backtest_metrics['backtest_win_rate']:.1%}, "
                    f"Trades={backtest_metrics['backtest_trades']}"
                )

                # Check if best backtest performance
                current_sortino = backtest_metrics["backtest_sortino"]
                if current_sortino > self.best_backtest_sortino:
                    self.best_backtest_sortino = current_sortino
                    self.best_backtest_epoch = epoch + 1
                    epochs_without_improvement = 0
                    logger.info(f"*** NEW BEST backtest Sortino: {current_sortino:.3f} ***")
                    self._update_manifest(epoch + 1, backtest_metrics)
                else:
                    epochs_without_improvement += self.eval_every
                    if epochs_without_improvement >= early_stopping_patience:
                        logger.warning(
                            f"Early stopping: No backtest improvement for "
                            f"{epochs_without_improvement} epochs"
                        )
                        break
            else:
                # Just log training progress
                if (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch {epoch + 1}/{epochs} ({epoch_time:.1f}s) | "
                        f"Train: loss={train_metrics['loss']:.4f}, sharpe={train_metrics['sharpe']:.3f} | "
                        f"Val: loss={val_metrics['loss']:.4f}, sharpe={val_metrics['sharpe']:.3f}"
                    )

        total_time = time.time() - start_time
        logger.info(f"\nTraining complete in {total_time / 60:.1f} minutes")
        logger.info(f"Best backtest Sortino: {self.best_backtest_sortino:.3f} at epoch {self.best_backtest_epoch}")

        return dict(history)


def main():
    parser = argparse.ArgumentParser(description="V9 Backtest-Validated Training")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--eval-every", type=int, default=5,
                        help="Run backtest every N epochs")
    parser.add_argument("--backtest-days", type=int, default=60,
                        help="Backtest period in days")
    parser.add_argument("--early-stopping", type=int, default=30,
                        help="Stop if no backtest improvement for N epochs")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    args = parser.parse_args()

    if args.run_name is None:
        args.run_name = f"v9_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    logger.info("=" * 70)
    logger.info("V9 BACKTEST-VALIDATED TRAINING")
    logger.info("=" * 70)
    logger.info(f"Key innovation: Model selection based on ACTUAL backtest performance")
    logger.info(f"Training on {len(V9_SYMBOLS)} proven profitable symbols")
    logger.info(f"Backtest evaluation every {args.eval_every} epochs")
    logger.info(f"Backtest period: {args.backtest_days} days")
    logger.info("")

    # Dataset config
    dataset_config = DailyDatasetConfigV4(
        symbols=V9_SYMBOLS,
        sequence_length=256,
        lookahead_days=8,
        validation_days=80,
        min_history_days=300,
        include_weekly_features=True,
        grouping_strategy="correlation",
        correlation_min_corr=0.6,
        correlation_max_group_size=12,
        exclude_symbols=list(V9_EXCLUDED),
    )

    # Training config
    training_config = DailyTrainingConfigV4(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        sequence_length=256,
        lookahead_days=8,
        patch_size=5,
        num_windows=4,
        window_size=2,
        num_quantiles=3,
        transformer_dim=512,
        transformer_layers=4,
        transformer_heads=8,
        transformer_kv_heads=4,
        max_hold_days=8,
        min_hold_days=1,
        maker_fee=0.0008,

        # Loss weights
        return_loss_weight=1.0,
        sharpe_loss_weight=0.3,
        forced_exit_penalty=0.25,
        quantile_calibration_weight=0.05,
        position_regularization=0.20,
        quantile_ordering_weight=0.1,
        exit_days_penalty_weight=0.05,

        # Temperature
        initial_temperature=0.01,
        final_temperature=0.0001,
        temp_warmup_epochs=10,
        temp_anneal_epochs=150,

        # Optimizer
        optimizer_name="dual",
        matrix_lr=0.02,
        embed_lr=0.2,
        head_lr=0.004,

        # Training
        run_name=args.run_name,
        checkpoint_root="neuraldailyv4/checkpoints",
        use_amp=True,
        amp_dtype="bfloat16",
        use_tf32=True,
        use_cross_attention=True,
        dataset=dataset_config,
    )

    # Create data module
    logger.info("Loading data module...")
    data_module = DailyDataModuleV4(dataset_config)

    # Create backtest-validated trainer
    logger.info("Creating backtest-validated trainer...")
    trainer = BacktestValidatedTrainer(
        training_config,
        data_module,
        backtest_symbols=V9_SYMBOLS,
        backtest_days=args.backtest_days,
        eval_every=args.eval_every,
    )

    # Log model info
    params = sum(p.numel() for p in trainer.trainer.model.parameters())
    logger.info(f"Model parameters: {params:,}")
    logger.info(f"Feature columns: {len(data_module.feature_columns)}")
    logger.info("")

    # Train
    logger.info(f"Starting training for {args.epochs} epochs...")
    history = trainer.train(
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping,
    )

    # Final summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Best backtest Sortino: {trainer.best_backtest_sortino:.4f}")
    logger.info(f"Best epoch: {trainer.best_backtest_epoch}")
    logger.info(f"Checkpoint dir: {trainer.trainer.checkpoint_dir}")

    # Print backtest history
    logger.info("\nBacktest History:")
    for entry in trainer.backtest_history:
        logger.info(
            f"  Epoch {entry['epoch']:3d}: "
            f"Return={entry['backtest_return']:7.2%}, "
            f"Sortino={entry['backtest_sortino']:6.3f}, "
            f"WinRate={entry['backtest_win_rate']:5.1%}, "
            f"Trades={entry['backtest_trades']:3d}"
        )

    logger.info("")
    logger.info("Next steps:")
    best_ckpt = trainer.trainer.checkpoint_dir / f"epoch_{trainer.best_backtest_epoch:04d}.pt"
    logger.info(f"  1. Full backtest: python backtest_v4.py --checkpoint {best_ckpt} --days 90")
    logger.info(f"  2. Symbol analysis: python run_symbol_experiments.py --checkpoint {best_ckpt}")
    logger.info(f"  3. Deploy: update trade_v4_e2e.py checkpoint path")


if __name__ == "__main__":
    main()
