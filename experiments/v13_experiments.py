#!/usr/bin/env python3
"""V13 Experiments: Testing position sizing strategies.

Based on analysis of V10 vs V11/V12 performance drop:
- V10: 66.39% return (position_reg "bug" allowed free sizing)
- V11: 56.59% return (position_reg fix pushed toward 50%)
- V12: 37.63% return (same fix + NaN handling)

Hypothesis: The "bug" in V10 was actually beneficial - it allowed the model
to express high conviction through position sizing. The "fix" forces
positions toward 50%, reducing the model's ability to make concentrated bets.

Experiments:
1. v13a: Remove position regularization entirely (0.0)
2. v13b: Very low position regularization (0.01)
3. v13c: Remove utilization penalty (0.0)
4. v13d: Combine v13a + v13c (remove both)
5. v13e: Use confidence to scale position (instead of regularizing)

Run with:
    python experiments/v13_experiments.py --experiment v13a
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from neuraldailyv4.config import DailyTrainingConfigV4, DailyDatasetConfigV4
from neuraldailyv4.data import DailyDataModuleV4
from neuraldailyv4.trainer import NeuralDailyTrainerV4
from backtest_v4 import run_backtest, print_summary, is_crypto_symbol
from neuraldailyv4.runtime import DailyTradingRuntimeV4


# Symbols that work well with V10
EXPERIMENT_SYMBOLS = (
    "SPY", "QQQ", "IWM", "XLF", "XLK", "DIA",
    "AAPL", "GOOGL", "NVDA",
    "V", "MA",
    "AMD", "LRCX", "AMAT", "QCOM",
    "MCD", "TGT",
    "LLY", "PFE",
    "ADBE",
)


def get_experiment_config(experiment_name: str) -> dict:
    """Get hyperparameters for each experiment."""

    base_config = {
        # Standard training params
        "epochs": 30,
        "batch_size": 64,
        "learning_rate": 0.0003,
        "weight_decay": 0.01,
        "sequence_length": 256,
        "lookahead_days": 8,
        "patch_size": 5,
        "num_windows": 4,
        "window_size": 2,
        "num_quantiles": 3,
        "transformer_dim": 512,
        "transformer_layers": 4,
        "max_hold_days": 8,
        "min_hold_days": 1,
        "maker_fee": 0.0008,

        # Loss weights (baseline from V10)
        "return_loss_weight": 1.0,
        "sharpe_loss_weight": 0.3,
        "forced_exit_penalty": 0.25,
        "quantile_calibration_weight": 0.05,
        "position_regularization": 0.10,  # Default (V11)
        "quantile_ordering_weight": 0.1,
        "exit_days_penalty_weight": 0.05,
        "utilization_loss_weight": 0.1,  # Default (V11)
        "utilization_target": 0.5,
    }

    experiments = {
        "v13a": {
            **base_config,
            "position_regularization": 0.0,  # Remove position reg entirely
            "description": "No position regularization (free sizing)",
        },
        "v13b": {
            **base_config,
            "position_regularization": 0.01,  # Very low
            "description": "Minimal position regularization (0.01)",
        },
        "v13c": {
            **base_config,
            "utilization_loss_weight": 0.0,  # Remove utilization penalty
            "description": "No utilization penalty (don't force portfolio usage)",
        },
        "v13d": {
            **base_config,
            "position_regularization": 0.0,
            "utilization_loss_weight": 0.0,
            "description": "No position reg + no utilization (V10-like)",
        },
        "v13e": {
            **base_config,
            "position_regularization": 0.0,
            "utilization_loss_weight": 0.0,
            "sharpe_loss_weight": 0.5,  # Higher sharpe weight
            "description": "V10-like + higher Sharpe emphasis",
        },
        "v13f": {
            **base_config,
            "position_regularization": 0.02,  # Slightly higher than v13b
            "utilization_loss_weight": 0.05,  # Halved
            "description": "Balanced: low pos_reg (0.02) + half utilization (0.05)",
        },
    }

    if experiment_name not in experiments:
        raise ValueError(f"Unknown experiment: {experiment_name}. Available: {list(experiments.keys())}")

    return experiments[experiment_name]


def train_and_backtest(
    experiment_name: str,
    epochs: int = 30,
    backtest_days: int = 60,
) -> dict:
    """Train a model with experiment config and backtest it."""

    config_params = get_experiment_config(experiment_name)
    logger.info(f"Running experiment: {experiment_name}")
    logger.info(f"Description: {config_params.get('description', 'N/A')}")
    logger.info(f"Key params:")
    logger.info(f"  position_regularization: {config_params['position_regularization']}")
    logger.info(f"  utilization_loss_weight: {config_params['utilization_loss_weight']}")

    # Override epochs if specified
    config_params["epochs"] = epochs

    # Create dataset config
    dataset_config = DailyDatasetConfigV4(
        symbols=EXPERIMENT_SYMBOLS,
        sequence_length=256,
        lookahead_days=8,
        validation_days=40,
        min_history_days=300,
        include_weekly_features=True,
        grouping_strategy="correlation",
    )

    # Create training config
    run_name = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    training_config = DailyTrainingConfigV4(
        epochs=config_params["epochs"],
        batch_size=config_params["batch_size"],
        learning_rate=config_params["learning_rate"],
        weight_decay=config_params["weight_decay"],
        sequence_length=config_params["sequence_length"],
        lookahead_days=config_params["lookahead_days"],
        patch_size=config_params["patch_size"],
        num_windows=config_params["num_windows"],
        window_size=config_params["window_size"],
        num_quantiles=config_params["num_quantiles"],
        transformer_dim=config_params["transformer_dim"],
        transformer_layers=config_params["transformer_layers"],
        transformer_heads=8,
        transformer_kv_heads=4,
        max_hold_days=config_params["max_hold_days"],
        min_hold_days=config_params["min_hold_days"],
        maker_fee=config_params["maker_fee"],
        return_loss_weight=config_params["return_loss_weight"],
        sharpe_loss_weight=config_params["sharpe_loss_weight"],
        forced_exit_penalty=config_params["forced_exit_penalty"],
        quantile_calibration_weight=config_params["quantile_calibration_weight"],
        position_regularization=config_params["position_regularization"],
        quantile_ordering_weight=config_params["quantile_ordering_weight"],
        exit_days_penalty_weight=config_params["exit_days_penalty_weight"],
        utilization_loss_weight=config_params["utilization_loss_weight"],
        utilization_target=config_params["utilization_target"],
        initial_temperature=0.01,
        final_temperature=0.0001,
        temp_warmup_epochs=5,
        temp_anneal_epochs=epochs - 5,
        optimizer_name="dual",
        matrix_lr=0.02,
        embed_lr=0.2,
        head_lr=0.004,
        run_name=run_name,
        checkpoint_root="neuraldailyv4/checkpoints/experiments",
        use_amp=True,
        amp_dtype="bfloat16",
        use_tf32=True,
        use_cross_attention=True,
        dataset=dataset_config,
    )

    # Train
    logger.info("Starting training...")
    start_time = time.time()

    data_module = DailyDataModuleV4(dataset_config)
    trainer = NeuralDailyTrainerV4(training_config, data_module)
    history = trainer.train(log_every=5, early_stopping_patience=15)

    train_time = time.time() - start_time
    logger.info(f"Training complete in {train_time/60:.1f} minutes")
    logger.info(f"Best epoch: {trainer.best_epoch}, Best val sharpe: {trainer.best_val_sharpe:.4f}")

    # Get best checkpoint
    checkpoint_path = trainer.checkpoint_dir / f"epoch_{trainer.best_epoch:04d}.pt"

    # Backtest
    logger.info(f"Running {backtest_days}-day backtest...")
    runtime = DailyTradingRuntimeV4(checkpoint_path, trade_crypto=False, max_exit_days=2)

    end_date = pd.Timestamp.now(tz="UTC")
    start_date = end_date - timedelta(days=backtest_days)

    trades = run_backtest(
        runtime,
        list(EXPERIMENT_SYMBOLS),
        start_date,
        end_date,
        verbose=False,
    )

    # Compute metrics
    if trades:
        returns = [t["return_pct"] for t in trades]
        total_return = sum(returns)
        avg_return = np.mean(returns)
        std_return = np.std(returns) if len(returns) > 1 else 0.001
        sharpe = avg_return / std_return if std_return > 0 else 0
        win_rate = sum(1 for r in returns if r > 0) / len(returns)
        tp_rate = sum(1 for t in trades if t.get("tp_hit")) / len(trades)
    else:
        total_return = avg_return = sharpe = win_rate = tp_rate = 0.0
        std_return = 0.001

    results = {
        "experiment": experiment_name,
        "description": config_params.get("description", ""),
        "config": {
            "position_regularization": config_params["position_regularization"],
            "utilization_loss_weight": config_params["utilization_loss_weight"],
            "sharpe_loss_weight": config_params["sharpe_loss_weight"],
        },
        "training": {
            "best_epoch": trainer.best_epoch,
            "best_val_sharpe": trainer.best_val_sharpe,
            "train_time_minutes": train_time / 60,
        },
        "backtest": {
            "total_return": total_return,
            "avg_return": avg_return,
            "std_return": std_return,
            "sharpe": sharpe,
            "win_rate": win_rate,
            "tp_rate": tp_rate,
            "num_trades": len(trades),
            "backtest_days": backtest_days,
        },
        "checkpoint": str(checkpoint_path),
    }

    # Print summary
    print_summary(trades)

    logger.info("=" * 60)
    logger.info("EXPERIMENT RESULTS")
    logger.info("=" * 60)
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Total Return: {total_return:.2%}")
    logger.info(f"Sharpe: {sharpe:.3f}")
    logger.info(f"Win Rate: {win_rate:.1%}")
    logger.info(f"Trades: {len(trades)}")

    # Save results
    output_dir = Path("experiments/v13_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{run_name}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="V13 Experiments")
    parser.add_argument("--experiment", type=str, required=True,
                       choices=["v13a", "v13b", "v13c", "v13d", "v13e", "v13f", "all"],
                       help="Experiment to run")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--backtest-days", type=int, default=60, help="Backtest days")
    args = parser.parse_args()

    if args.experiment == "all":
        experiments = ["v13a", "v13b", "v13c", "v13d", "v13e", "v13f"]
        all_results = []
        for exp in experiments:
            logger.info(f"\n{'='*60}\nRunning {exp}\n{'='*60}\n")
            result = train_and_backtest(exp, args.epochs, args.backtest_days)
            all_results.append(result)

        # Summary comparison
        logger.info("\n" + "=" * 80)
        logger.info("EXPERIMENT COMPARISON SUMMARY")
        logger.info("=" * 80)
        for r in all_results:
            logger.info(
                f"{r['experiment']:6} | "
                f"Return: {r['backtest']['total_return']:6.1%} | "
                f"Sharpe: {r['backtest']['sharpe']:.3f} | "
                f"Win: {r['backtest']['win_rate']:.1%} | "
                f"Trades: {r['backtest']['num_trades']}"
            )
    else:
        train_and_backtest(args.experiment, args.epochs, args.backtest_days)


if __name__ == "__main__":
    main()
