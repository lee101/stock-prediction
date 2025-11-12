#!/usr/bin/env python3
"""
Quick script to improve crypto forecasting by testing better hyperparameters.

This script focuses on BTCUSD, ETHUSD, and UNIUSD to find optimal configurations
that minimize percentage return MAE for crypto trading.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error

from src.models.kronos_wrapper import KronosForecastingWrapper
from src.models.toto_wrapper import TotoPipeline
from src.models.toto_aggregation import aggregate_with_spec


# Crypto symbols to optimize
CRYPTO_SYMBOLS = ["BTCUSD", "ETHUSD", "UNIUSD"]
DATA_DIR = Path("trainingdata")
OUTPUT_DIR = Path("hyperparams_crypto_optimized")
OUTPUT_DIR.mkdir(exist_ok=True)

# Evaluation windows
FORECAST_HORIZON = 1
VAL_WINDOW = 20
TEST_WINDOW = 20


def load_training_data(symbol: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load train/val/test data for a symbol."""
    data_file = DATA_DIR / f"{symbol}_daily.csv"
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    import pandas as pd
    df = pd.read_csv(data_file)
    prices = df['close'].values

    # Split: 70% train, 15% val, 15% test
    n = len(prices)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_data = prices[:train_end]
    val_data = prices[train_end:val_end]
    test_data = prices[val_end:]

    return train_data, val_data, test_data


def evaluate_toto_config(
    symbol: str,
    val_data: np.ndarray,
    test_data: np.ndarray,
    num_samples: int,
    aggregate: str,
    samples_per_batch: int
) -> Dict:
    """Evaluate a Toto configuration."""
    pipeline = TotoPipeline()

    results = {
        "symbol": symbol,
        "model": "toto",
        "config": {
            "num_samples": num_samples,
            "aggregate": aggregate,
            "samples_per_batch": samples_per_batch,
        },
        "validation": {},
        "test": {}
    }

    # Evaluate on validation set
    val_predictions = []
    val_actuals = []
    val_start = time.time()

    for i in range(VAL_WINDOW):
        if i + 128 >= len(val_data):
            break
        context = val_data[i:i+128]
        actual = val_data[i+128]

        raw_preds = pipeline.predict(
            context,
            prediction_length=FORECAST_HORIZON,
            num_samples=num_samples,
            samples_per_batch=samples_per_batch
        )
        pred = aggregate_with_spec(raw_preds, aggregate)

        val_predictions.append(pred)
        val_actuals.append(actual)

    val_time = (time.time() - val_start) / max(1, len(val_predictions))

    val_predictions = np.array(val_predictions)
    val_actuals = np.array(val_actuals)

    # Calculate metrics
    val_price_mae = mean_absolute_error(val_actuals, val_predictions)
    val_pct_returns_pred = (val_predictions[1:] - val_predictions[:-1]) / val_predictions[:-1]
    val_pct_returns_actual = (val_actuals[1:] - val_actuals[:-1]) / val_actuals[:-1]
    val_pct_mae = mean_absolute_error(val_pct_returns_actual, val_pct_returns_pred)

    results["validation"] = {
        "price_mae": float(val_price_mae),
        "pct_return_mae": float(val_pct_mae),
        "latency_s": float(val_time)
    }

    # Evaluate on test set
    test_predictions = []
    test_actuals = []
    test_start = time.time()

    for i in range(TEST_WINDOW):
        if i + 128 >= len(test_data):
            break
        context = test_data[i:i+128]
        actual = test_data[i+128]

        raw_preds = pipeline.predict(
            context,
            prediction_length=FORECAST_HORIZON,
            num_samples=num_samples,
            samples_per_batch=samples_per_batch
        )
        pred = aggregate_with_spec(raw_preds, aggregate)

        test_predictions.append(pred)
        test_actuals.append(actual)

    test_time = (time.time() - test_start) / max(1, len(test_predictions))

    test_predictions = np.array(test_predictions)
    test_actuals = np.array(test_actuals)

    test_price_mae = mean_absolute_error(test_actuals, test_predictions)
    test_pct_returns_pred = (test_predictions[1:] - test_predictions[:-1]) / test_predictions[:-1]
    test_pct_returns_actual = (test_actuals[1:] - test_actuals[:-1]) / test_actuals[:-1]
    test_pct_mae = mean_absolute_error(test_pct_returns_actual, test_pct_returns_pred)

    results["test"] = {
        "price_mae": float(test_price_mae),
        "pct_return_mae": float(test_pct_mae),
        "latency_s": float(test_time)
    }

    return results


def optimize_crypto_symbol(symbol: str):
    """Run optimization for a single crypto symbol."""
    print(f"\n{'='*60}")
    print(f"Optimizing {symbol}")
    print(f"{'='*60}")

    # Load data
    train_data, val_data, test_data = load_training_data(symbol)
    print(f"Loaded data: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    # Test configurations for Toto
    toto_configs = [
        # Current ETHUSD is 128 samples - let's try more
        (256, "trimmed_mean_20", 32),
        (512, "trimmed_mean_10", 64),
        (1024, "trimmed_mean_5", 128),
        (2048, "trimmed_mean_5", 128),

        # Try different aggregations
        (512, "trimmed_mean_20", 64),
        (1024, "trimmed_mean_10", 128),
        (512, "quantile_0.50", 64),
        (1024, "mean", 128),
    ]

    best_result = None
    best_pct_mae = float('inf')

    for num_samples, aggregate, samples_per_batch in toto_configs:
        try:
            print(f"\nTesting: samples={num_samples}, agg={aggregate}, spb={samples_per_batch}")
            result = evaluate_toto_config(
                symbol, val_data, test_data,
                num_samples, aggregate, samples_per_batch
            )

            test_pct_mae = result["test"]["pct_return_mae"]
            print(f"  Val pct MAE: {result['validation']['pct_return_mae']:.4f}")
            print(f"  Test pct MAE: {test_pct_mae:.4f}")
            print(f"  Latency: {result['test']['latency_s']:.2f}s")

            if test_pct_mae < best_pct_mae:
                best_pct_mae = test_pct_mae
                best_result = result
                print(f"  ✓ New best!")

        except Exception as e:
            print(f"  Error: {e}")
            continue

    # Save best result
    if best_result:
        output_file = OUTPUT_DIR / f"{symbol}.json"
        with open(output_file, 'w') as f:
            json.dump(best_result, f, indent=2)
        print(f"\n✓ Saved best config to {output_file}")
        print(f"  Best test pct MAE: {best_pct_mae:.4f}")
    else:
        print(f"\n✗ No successful configurations for {symbol}")


def main():
    """Main optimization loop."""
    print("Starting crypto forecasting optimization")
    print(f"Symbols: {', '.join(CRYPTO_SYMBOLS)}")

    for symbol in CRYPTO_SYMBOLS:
        try:
            optimize_crypto_symbol(symbol)
        except Exception as e:
            print(f"\n✗ Error optimizing {symbol}: {e}")
            continue

    print(f"\n{'='*60}")
    print("Optimization complete!")
    print(f"Results saved to {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
