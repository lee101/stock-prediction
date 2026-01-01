#!/usr/bin/env python3
"""Benchmark script for Neural Hourly Trading V5 models.

Runs walk-forward backtest on validation data and reports PnL metrics.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import torch

from neuralhourlytradingv5.config import SimulationConfigV5
from neuralhourlytradingv5.data import HourlyDataModuleV5, FeatureNormalizer, HOURLY_FEATURES_V5
from neuralhourlytradingv5.model import HourlyCryptoPolicyV5
from neuralhourlytradingv5.backtest import (
    HourlyMarketSimulatorV5,
    run_10day_validation,
    print_backtest_summary,
)


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: str = "cuda",
) -> tuple:
    """Load model and normalizer from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Get config
    policy_config = checkpoint["config"]["policy"]

    # Create model
    model = HourlyCryptoPolicyV5(policy_config)

    # Handle torch.compile prefix in state dict keys
    state_dict = checkpoint["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {
            k.replace("_orig_mod.", ""): v for k, v in state_dict.items()
        }

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Get normalizer
    normalizer = FeatureNormalizer.from_dict(checkpoint["normalizer"])
    feature_columns = checkpoint["feature_columns"]

    return model, normalizer, feature_columns, checkpoint


def run_backtest(
    checkpoint_path: str,
    symbol: str = "BTCUSD",
    data_root: str = "trainingdatahourly/crypto",
    device: str = "cuda",
) -> Dict:
    """Run backtest on a checkpoint and return results."""
    print(f"\n{'='*60}")
    print(f"Backtesting: {Path(checkpoint_path).name}")
    print(f"Symbol: {symbol}")
    print(f"{'='*60}")

    # Load model
    model, normalizer, feature_columns, checkpoint = load_model_from_checkpoint(
        checkpoint_path, device
    )

    # Get training config
    config = checkpoint["config"]
    training_config = config.get("training", None)

    symbols = None
    if training_config and hasattr(training_config, "dataset") and training_config.dataset:
        symbols = training_config.dataset.symbols

    print(f"Trained on symbols: {symbols}")
    print(f"Features: {len(feature_columns)}")

    # Load data
    data_path = Path(data_root) / f"{symbol}.csv"
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        return None

    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"Data rows: {len(df)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Prepare features
    # Add missing features with defaults
    for feat in HOURLY_FEATURES_V5:
        if feat not in df.columns:
            if feat.startswith("chronos_"):
                df[feat] = 0.0
            elif feat == "volume_z":
                df[feat] = 0.0
            else:
                df[feat] = 0.0

    # Create features array
    features = df[list(feature_columns)].values

    # Normalize features
    features_normalized = normalizer.transform(features)

    # Get validation period (last 10 days + sequence length)
    sequence_length = 168  # 1 week
    validation_hours = 240  # 10 days

    start_idx = max(0, len(df) - validation_hours - sequence_length - 24)

    val_bars = df.iloc[start_idx:].reset_index(drop=True)
    val_features = features_normalized[start_idx:]

    print(f"Validation period: {val_bars['timestamp'].iloc[0]} to {val_bars['timestamp'].iloc[-1]}")
    print(f"Validation hours: {len(val_bars)}")

    # Run backtest
    sim_config = SimulationConfigV5(
        initial_cash=10000.0,
        maker_fee=0.0008,
        forced_exit_slippage=0.001,
    )

    result = run_10day_validation(
        model=model,
        bars=val_bars,
        features=val_features,
        normalizer=normalizer,
        config=sim_config,
        sequence_length=sequence_length,
    )

    # Print results
    print_backtest_summary(result)

    # Calculate additional metrics
    initial_equity = sim_config.initial_cash
    final_equity = result.equity_curve.iloc[-1]
    total_pnl = final_equity - initial_equity
    total_return = (final_equity / initial_equity - 1) * 100

    print(f"\nPNL Summary:")
    print(f"  Initial Equity: ${initial_equity:,.2f}")
    print(f"  Final Equity:   ${final_equity:,.2f}")
    print(f"  Total PnL:      ${total_pnl:+,.2f}")
    print(f"  Total Return:   {total_return:+.2f}%")

    return {
        "checkpoint": checkpoint_path,
        "symbol": symbol,
        "trained_symbols": symbols,
        "initial_equity": initial_equity,
        "final_equity": final_equity,
        "total_pnl": total_pnl,
        "total_return": total_return,
        "sortino": result.metrics.get("sortino", 0),
        "num_trades": result.metrics.get("num_trades", 0),
        "win_rate": result.metrics.get("win_rate", 0),
        "tp_rate": result.metrics.get("tp_rate", 0),
        "avg_hold_hours": result.metrics.get("avg_hold_hours", 0),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark V5 Hourly Model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSD",
        help="Symbol to backtest on",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="trainingdatahourly/crypto",
        help="Path to data directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )

    args = parser.parse_args()

    result = run_backtest(
        checkpoint_path=args.checkpoint,
        symbol=args.symbol,
        data_root=args.data_root,
        device=args.device,
    )

    if result:
        print("\n" + "="*60)
        print("BENCHMARK COMPLETE")
        print("="*60)


if __name__ == "__main__":
    main()
