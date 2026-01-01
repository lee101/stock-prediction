#!/usr/bin/env python3
"""Walk-Forward Validation: Test weekly retraining performance.

This simulates realistic continual learning:
1. Train on all data up to week N
2. Backtest on week N+1 only (truly out-of-sample)
3. Repeat for multiple weeks
4. Report per-week and aggregate performance

This answers: "If we retrain weekly, how well does the model perform
on the next week's data?"

Usage:
    python walk_forward_validation.py --weeks 8 --epochs 30
    python walk_forward_validation.py --weeks 12 --epochs 50 --config quick
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


# Symbols proven profitable in backtests (V11+: includes crypto for holiday trading)
WALK_FORWARD_SYMBOLS = (
    # Equities
    "SPY", "QQQ", "IWM", "XLF", "XLK", "DIA",
    "AAPL", "GOOGL", "NVDA",
    "V", "MA",
    "AMD", "LRCX", "AMAT", "QCOM",
    "MCD", "TGT",
    "LLY", "PFE",
    "ADBE",
    # Crypto (for holiday trading when NYSE is closed)
    "ETHUSD",  # ETH has shown better performance than BTC
)

EXCLUDED_SYMBOLS = {
    "ORCL", "NOW", "DDOG", "PYPL", "COIN", "INTC",
    "AVGO", "COST", "CRM", "HD", "MRVL",
    "META", "MSFT", "AMZN", "NFLX",
    "BTCUSD", "AVAXUSD",  # BTC and AVAX excluded due to poor model performance
}


def calculate_sortino(returns: List[float], target: float = 0.0) -> float:
    """Calculate Sortino ratio."""
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


def run_week_backtest(
    checkpoint_path: Path,
    symbols: List[str],
    start_date: str,
    end_date: str,
) -> Dict[str, float]:
    """Run backtest for a specific week."""
    from backtest_v4 import run_backtest
    from neuraldailyv4.runtime import DailyTradingRuntimeV4

    try:
        runtime = DailyTradingRuntimeV4(
            checkpoint_path,
            trade_crypto=False,  # Normal trading: stocks only
            trade_crypto_on_holidays=True,  # V11+: Enable crypto when NYSE is closed
            max_exit_days=2,
        )

        # Convert string dates to Timestamp objects
        start_ts = pd.Timestamp(start_date, tz="UTC")
        end_ts = pd.Timestamp(end_date, tz="UTC")

        # Run backtest for specific date range
        trades = run_backtest(
            runtime,
            symbols,
            start_date=start_ts,
            end_date=end_ts,
        )

        if not trades:
            return {
                "total_return": 0.0,
                "sortino": 0.0,
                "sharpe": 0.0,
                "win_rate": 0.0,
                "num_trades": 0,
                "tp_rate": 0.0,
            }

        returns = [t["return_pct"] for t in trades]
        wins = sum(1 for r in returns if r > 0)
        tp_hits = sum(1 for t in trades if t.get("exit_type") == "take_profit")

        total_return = sum(returns)
        avg_return = np.mean(returns)
        std_return = np.std(returns) if len(returns) > 1 else 0.001

        return {
            "total_return": total_return,
            "sortino": calculate_sortino(returns),
            "sharpe": avg_return / std_return if std_return > 0 else 0.0,
            "win_rate": wins / len(trades) * 100,
            "num_trades": len(trades),
            "tp_rate": tp_hits / len(trades) * 100 if trades else 0.0,
        }

    except Exception as e:
        logger.warning(f"Backtest failed: {e}")
        return {
            "total_return": 0.0,
            "sortino": 0.0,
            "sharpe": 0.0,
            "win_rate": 0.0,
            "num_trades": 0,
            "tp_rate": 0.0,
        }


def train_model(
    end_date: str,
    epochs: int,
    config_preset: str,
    run_name: str,
) -> Tuple[Path, Dict]:
    """Train model on data up to end_date."""

    # Config presets for different training strategies
    if config_preset == "quick":
        # Quick training - fewer epochs, faster iteration
        training_epochs = min(epochs, 20)
        early_stopping = 10
        batch_size = 64
    elif config_preset == "standard":
        # Standard training
        training_epochs = epochs
        early_stopping = 15
        batch_size = 64
    elif config_preset == "thorough":
        # Thorough training - more epochs
        training_epochs = max(epochs, 50)
        early_stopping = 25
        batch_size = 32
    else:
        training_epochs = epochs
        early_stopping = 20
        batch_size = 64

    dataset_config = DailyDatasetConfigV4(
        symbols=WALK_FORWARD_SYMBOLS,
        sequence_length=256,
        lookahead_days=8,
        validation_days=40,  # Smaller for faster training
        min_history_days=300,
        include_weekly_features=True,
        grouping_strategy="correlation",
        end_date=end_date,  # Train only on data up to this date
        exclude_symbols=list(EXCLUDED_SYMBOLS),
    )

    training_config = DailyTrainingConfigV4(
        epochs=training_epochs,
        batch_size=batch_size,
        learning_rate=0.0003,
        weight_decay=0.01,
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
        return_loss_weight=1.0,
        sharpe_loss_weight=0.3,
        forced_exit_penalty=0.25,
        quantile_calibration_weight=0.05,
        position_regularization=0.10,  # Reduced from 0.20 since bug fix makes it actually work
        quantile_ordering_weight=0.1,
        exit_days_penalty_weight=0.05,
        initial_temperature=0.01,
        final_temperature=0.0001,
        temp_warmup_epochs=5,
        temp_anneal_epochs=training_epochs - 5,
        optimizer_name="dual",
        matrix_lr=0.02,
        embed_lr=0.2,
        head_lr=0.004,
        run_name=run_name,
        checkpoint_root="neuraldailyv4/checkpoints/walk_forward",
        use_amp=True,
        amp_dtype="bfloat16",
        use_tf32=True,
        use_cross_attention=True,
        dataset=dataset_config,
    )

    # Create data module and trainer
    data_module = DailyDataModuleV4(dataset_config)
    trainer = NeuralDailyTrainerV4(training_config, data_module)

    # Train
    start_time = time.time()
    history = trainer.train(log_every=10, early_stopping_patience=early_stopping)
    train_time = time.time() - start_time

    # Get best checkpoint
    best_checkpoint = trainer.checkpoint_dir / f"epoch_{trainer.best_epoch:04d}.pt"

    return best_checkpoint, {
        "best_epoch": trainer.best_epoch,
        "best_val_sharpe": trainer.best_val_sharpe,
        "train_time_minutes": train_time / 60,
        "total_epochs": len(history.get("train_loss", [])),
    }


def run_walk_forward(
    num_weeks: int = 8,
    epochs_per_week: int = 30,
    config_preset: str = "standard",
    start_offset_weeks: int = 0,
) -> Dict:
    """Run walk-forward validation over multiple weeks."""

    results = {
        "config": {
            "num_weeks": num_weeks,
            "epochs_per_week": epochs_per_week,
            "config_preset": config_preset,
            "symbols": list(WALK_FORWARD_SYMBOLS),
        },
        "weekly_results": [],
        "aggregate": {},
    }

    # Calculate week dates going backwards from today
    today = datetime.now()

    all_returns = []
    all_sortinos = []
    all_win_rates = []
    total_trades = 0

    for week_idx in range(num_weeks):
        # Week N+1 is the test week (most recent first)
        test_week_end = today - timedelta(weeks=week_idx + start_offset_weeks)
        test_week_start = test_week_end - timedelta(days=7)

        # Training ends the day before test week starts
        train_end = test_week_start - timedelta(days=1)

        test_start_str = test_week_start.strftime("%Y-%m-%d")
        test_end_str = test_week_end.strftime("%Y-%m-%d")
        train_end_str = train_end.strftime("%Y-%m-%d")

        logger.info("=" * 70)
        logger.info(f"WEEK {week_idx + 1}/{num_weeks}")
        logger.info(f"Training on data up to: {train_end_str}")
        logger.info(f"Testing on: {test_start_str} to {test_end_str}")
        logger.info("=" * 70)

        # Train model
        run_name = f"wf_week{week_idx + 1}_{train_end_str}"

        try:
            checkpoint, train_info = train_model(
                end_date=train_end_str,
                epochs=epochs_per_week,
                config_preset=config_preset,
                run_name=run_name,
            )
            logger.info(f"Training complete: {train_info['train_time_minutes']:.1f} min, "
                       f"best epoch {train_info['best_epoch']}")

            # Test on next week
            week_metrics = run_week_backtest(
                checkpoint,
                list(WALK_FORWARD_SYMBOLS),
                test_start_str,
                test_end_str,
            )

            logger.info(f"Week {week_idx + 1} Results:")
            logger.info(f"  Return: {week_metrics['total_return']:.2f}%")
            logger.info(f"  Sortino: {week_metrics['sortino']:.3f}")
            logger.info(f"  Win Rate: {week_metrics['win_rate']:.1f}%")
            logger.info(f"  Trades: {week_metrics['num_trades']}")

            week_result = {
                "week": week_idx + 1,
                "train_end": train_end_str,
                "test_start": test_start_str,
                "test_end": test_end_str,
                "checkpoint": str(checkpoint),
                "train_info": train_info,
                "metrics": week_metrics,
            }

            results["weekly_results"].append(week_result)

            # Aggregate stats
            all_returns.append(week_metrics["total_return"])
            all_sortinos.append(week_metrics["sortino"])
            all_win_rates.append(week_metrics["win_rate"])
            total_trades += week_metrics["num_trades"]

        except Exception as e:
            logger.error(f"Week {week_idx + 1} failed: {e}")
            results["weekly_results"].append({
                "week": week_idx + 1,
                "error": str(e),
            })

    # Aggregate results
    if all_returns:
        results["aggregate"] = {
            "total_weeks": len(all_returns),
            "avg_weekly_return": np.mean(all_returns),
            "std_weekly_return": np.std(all_returns),
            "total_return": sum(all_returns),
            "avg_sortino": np.mean([s for s in all_sortinos if s != float('inf')]),
            "avg_win_rate": np.mean(all_win_rates),
            "total_trades": total_trades,
            "positive_weeks": sum(1 for r in all_returns if r > 0),
            "negative_weeks": sum(1 for r in all_returns if r <= 0),
        }

        logger.info("")
        logger.info("=" * 70)
        logger.info("WALK-FORWARD SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Weeks tested: {len(all_returns)}")
        logger.info(f"Total return: {sum(all_returns):.2f}%")
        logger.info(f"Avg weekly return: {np.mean(all_returns):.2f}% (std: {np.std(all_returns):.2f}%)")
        logger.info(f"Positive weeks: {sum(1 for r in all_returns if r > 0)}/{len(all_returns)}")
        logger.info(f"Avg Sortino: {np.mean([s for s in all_sortinos if s != float('inf')]):.3f}")
        logger.info(f"Total trades: {total_trades}")

    # Save results
    output_dir = Path("experiments/walk_forward")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"wf_{config_preset}_{num_weeks}weeks_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Validation")
    parser.add_argument("--weeks", type=int, default=8, help="Number of weeks to test")
    parser.add_argument("--epochs", type=int, default=30, help="Epochs per training run")
    parser.add_argument("--config", type=str, default="standard",
                       choices=["quick", "standard", "thorough"],
                       help="Training config preset")
    parser.add_argument("--offset", type=int, default=0,
                       help="Weeks to offset from today (0=most recent)")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("WALK-FORWARD VALIDATION")
    logger.info("=" * 70)
    logger.info(f"Testing {args.weeks} weeks with {args.epochs} epochs each")
    logger.info(f"Config preset: {args.config}")
    logger.info(f"Symbols: {len(WALK_FORWARD_SYMBOLS)}")
    logger.info("")

    results = run_walk_forward(
        num_weeks=args.weeks,
        epochs_per_week=args.epochs,
        config_preset=args.config,
        start_offset_weeks=args.offset,
    )

    # Print final verdict
    if results.get("aggregate"):
        agg = results["aggregate"]
        positive_rate = agg["positive_weeks"] / agg["total_weeks"] * 100

        logger.info("")
        logger.info("=" * 70)
        logger.info("VERDICT")
        logger.info("=" * 70)

        if positive_rate >= 60 and agg["avg_weekly_return"] > 0:
            logger.info("✓ PASSED: Model is profitable in walk-forward testing")
            logger.info(f"  {positive_rate:.0f}% positive weeks, {agg['avg_weekly_return']:.2f}% avg return")
        elif positive_rate >= 50:
            logger.info("~ MARGINAL: Model shows some edge but needs improvement")
            logger.info(f"  {positive_rate:.0f}% positive weeks, {agg['avg_weekly_return']:.2f}% avg return")
        else:
            logger.info("✗ FAILED: Model does not generalize well to future weeks")
            logger.info(f"  Only {positive_rate:.0f}% positive weeks")


if __name__ == "__main__":
    main()
