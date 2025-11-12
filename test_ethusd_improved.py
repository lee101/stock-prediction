#!/usr/bin/env python3
"""
Quick test: ETHUSD with improved config (1024 samples vs current 128).
Expected: 30-40% improvement (3.75% → 2.4% MAE)
"""
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error

from src.models.toto_wrapper import TotoPipeline
from src.models.toto_aggregation import aggregate_with_spec

# Load training data
import pandas as pd
data_file = Path("data/ETHUSD/ETHUSD-2025-11-04.csv")
df = pd.read_csv(data_file)
prices = df['Close'].values

# Split data
n = len(prices)
train_end = int(n * 0.70)
val_end = int(n * 0.85)
val_data = prices[train_end:val_end]
test_data = prices[val_end:]

print("="*60)
print("ETHUSD FORECASTING TEST - IMPROVED CONFIG")
print("="*60)
print(f"Data loaded: {len(prices)} prices")
print(f"Test window: {len(test_data)} prices")

# Current config (baseline)
current_config = {
    "num_samples": 128,
    "aggregate": "trimmed_mean_20",
    "samples_per_batch": 32
}

# Improved config (from our optimization)
improved_config = {
    "num_samples": 1024,
    "aggregate": "trimmed_mean_5",
    "samples_per_batch": 128
}

def test_config(config_name, num_samples, aggregate, samples_per_batch):
    """Test a configuration and return metrics."""
    print(f"\n{'='*60}")
    print(f"Testing: {config_name}")
    print(f"  Samples: {num_samples}, Aggregate: {aggregate}, SPB: {samples_per_batch}")
    print(f"{'='*60}")

    pipeline = TotoPipeline()

    predictions = []
    actuals = []
    latencies = []

    TEST_WINDOW = 20
    CONTEXT_LENGTH = 128

    for i in range(TEST_WINDOW):
        if i + CONTEXT_LENGTH >= len(test_data):
            break

        context = test_data[i:i+CONTEXT_LENGTH]
        actual = test_data[i+CONTEXT_LENGTH]

        start_time = time.time()
        raw_preds = pipeline.predict(
            context,
            prediction_length=1,
            num_samples=num_samples,
            samples_per_batch=samples_per_batch
        )
        pred = aggregate_with_spec(raw_preds, aggregate)
        latency = time.time() - start_time

        predictions.append(pred)
        actuals.append(actual)
        latencies.append(latency)

        if (i + 1) % 5 == 0:
            print(f"  Progress: {i+1}/{TEST_WINDOW} predictions...")

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Calculate metrics
    price_mae = mean_absolute_error(actuals, predictions)

    pct_returns_pred = (predictions[1:] - predictions[:-1]) / predictions[:-1]
    pct_returns_actual = (actuals[1:] - actuals[:-1]) / actuals[:-1]
    pct_return_mae = mean_absolute_error(pct_returns_actual, pct_returns_pred)

    avg_latency = np.mean(latencies)

    print(f"\n  Results:")
    print(f"    Price MAE: ${price_mae:.2f}")
    print(f"    Pct Return MAE: {pct_return_mae:.4f} ({pct_return_mae*100:.2f}%)")
    print(f"    Avg Latency: {avg_latency:.2f}s")

    return {
        "config_name": config_name,
        "price_mae": float(price_mae),
        "pct_return_mae": float(pct_return_mae),
        "latency_s": float(avg_latency),
        "num_samples": num_samples,
        "aggregate": aggregate
    }

# Test baseline
print("\n" + "="*60)
print("BASELINE TEST (Current Config)")
print("="*60)
baseline_result = test_config(
    "Current (128 samples)",
    current_config["num_samples"],
    current_config["aggregate"],
    current_config["samples_per_batch"]
)

# Test improved
print("\n" + "="*60)
print("IMPROVED CONFIG TEST")
print("="*60)
improved_result = test_config(
    "Improved (1024 samples)",
    improved_config["num_samples"],
    improved_config["aggregate"],
    improved_config["samples_per_batch"]
)

# Compare
print("\n" + "="*60)
print("COMPARISON & RESULTS")
print("="*60)
print(f"\nBaseline (128 samples, trimmed_mean_20):")
print(f"  Pct Return MAE: {baseline_result['pct_return_mae']*100:.2f}%")
print(f"  Latency: {baseline_result['latency_s']:.2f}s")

print(f"\nImproved (1024 samples, trimmed_mean_5):")
print(f"  Pct Return MAE: {improved_result['pct_return_mae']*100:.2f}%")
print(f"  Latency: {improved_result['latency_s']:.2f}s")

improvement_pct = ((baseline_result['pct_return_mae'] - improved_result['pct_return_mae'])
                   / baseline_result['pct_return_mae'] * 100)

print(f"\nImprovement: {improvement_pct:.1f}%")
if improvement_pct > 0:
    print(f"✓ BETTER by {improvement_pct:.1f}%!")
else:
    print(f"✗ WORSE by {abs(improvement_pct):.1f}%")

# Save results
results = {
    "baseline": baseline_result,
    "improved": improved_result,
    "improvement_pct": float(improvement_pct)
}

output_file = Path("results/ethusd_improved_test.json")
output_file.parent.mkdir(exist_ok=True)
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved to: {output_file}")
print("="*60)
