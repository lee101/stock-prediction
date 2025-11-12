#!/usr/bin/env python3
"""
Test EXTREME configs - push the limits to see what's possible.
Try very high sample counts and different strategies.
"""
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error

from src.models.toto_wrapper import TotoPipeline
from src.models.toto_aggregation import aggregate_with_spec

def quick_test(symbol, test_data, num_samples, aggregate, spb):
    """Ultra-fast test with minimal iterations."""
    pipeline = TotoPipeline()

    predictions = []
    actuals = []
    start_time = time.time()

    # Only 10 predictions for speed
    for i in range(10):
        if i + 128 >= len(test_data):
            break

        context = test_data[i:i+128]
        actual = test_data[i+128]

        raw_preds = pipeline.predict(
            context,
            prediction_length=1,
            num_samples=num_samples,
            samples_per_batch=spb
        )
        pred = aggregate_with_spec(raw_preds, aggregate)

        predictions.append(pred)
        actuals.append(actual)

    total_time = time.time() - start_time

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    pct_returns_pred = (predictions[1:] - predictions[:-1]) / predictions[:-1]
    pct_returns_actual = (actuals[1:] - actuals[:-1]) / actuals[:-1]
    pct_return_mae = mean_absolute_error(pct_returns_actual, pct_returns_pred)

    return {
        "num_samples": num_samples,
        "aggregate": aggregate,
        "pct_return_mae": float(pct_return_mae),
        "latency_s": float(total_time / len(predictions))
    }

# Load ETHUSD data (worst performer)
import pandas as pd
df = pd.read_csv("data/ETHUSD/ETHUSD-2025-11-04.csv")
prices = df['Close'].values
test_data = prices[int(len(prices)*0.85):]

print("="*70)
print("EXTREME CONFIG TESTING - ETHUSD")
print("="*70)
print("Testing AGGRESSIVE sample counts and aggregations...")
print()

extreme_configs = [
    # Baseline
    (128, "trimmed_mean_20", 32),

    # Moderate
    (512, "trimmed_mean_10", 64),
    (1024, "trimmed_mean_5", 128),

    # High
    (2048, "trimmed_mean_5", 128),
    (3072, "trimmed_mean_3", 128),
    (4096, "trimmed_mean_3", 256),

    # Try quantiles
    (1024, "quantile_0.50", 128),
    (2048, "quantile_0.50", 128),

    # Try mean (no trimming)
    (2048, "mean", 128),
]

results = []

for num_samples, aggregate, spb in extreme_configs:
    print(f"Testing: {num_samples:4d} samples, {aggregate:20s}...", end=" ")

    try:
        result = quick_test("ETHUSD", test_data, num_samples, aggregate, spb)
        print(f"→ {result['pct_return_mae']*100:5.2f}% MAE, {result['latency_s']:.2f}s")
        results.append(result)
    except Exception as e:
        print(f"✗ Error: {e}")

# Find best
best = min(results, key=lambda x: x["pct_return_mae"])

print()
print("="*70)
print("BEST CONFIG FOUND:")
print(f"  Samples: {best['num_samples']}")
print(f"  Aggregate: {best['aggregate']}")
print(f"  MAE: {best['pct_return_mae']*100:.2f}%")
print(f"  Latency: {best['latency_s']:.2f}s")
print("="*70)

# Save
with open("results/extreme_configs_ethusd.json", 'w') as f:
    json.dump({"results": results, "best": best}, f, indent=2)

print(f"\n✓ Saved to: results/extreme_configs_ethusd.json")
