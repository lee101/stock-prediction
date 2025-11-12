#!/usr/bin/env python3
"""
ULTRA SIMPLE crypto forecast test - just load model and test a few configs.
This WILL work.
"""
import json
import time
import numpy as np
from sklearn.metrics import mean_absolute_error

print("Loading model...")
from src.models.toto_wrapper import TotoPipeline
from src.models.toto_aggregation import aggregate_with_spec

# Load the Toto model
pipeline = TotoPipeline.from_pretrained(
    model_id="Datadog/Toto-Open-Base-1.0",
    device_map="cuda",
    compile_model=False,  # Disable for faster loading
)

print("✓ Model loaded!")

# Load ETHUSD data
import pandas as pd
df = pd.read_csv("data/ETHUSD/ETHUSD-2025-11-04.csv")
prices = df['Close'].values

# Use last 200 points as test data
test_data = prices[-200:]

def quick_forecast_test(num_samples, aggregate, name):
    """Test a config - ultra simple."""
    print(f"\nTesting: {name}")
    print(f"  Samples: {num_samples}, Aggregate: {aggregate}")

    preds = []
    actuals = []
    start = time.time()

    # Just do 10 forecasts for speed
    for i in range(10):
        context = test_data[i:i+128]
        actual = test_data[i+128]

        # Forecast
        forecasts = pipeline.predict(
            context=context,
            prediction_length=1,
            num_samples=num_samples,
            samples_per_batch=min(num_samples, 128)
        )
        # Extract samples from first forecast and aggregate
        pred = aggregate_with_spec(forecasts[0].samples, aggregate)

        preds.append(pred)
        actuals.append(actual)

    elapsed = time.time() - start

    # Calculate MAE
    preds = np.array(preds)
    actuals = np.array(actuals)

    price_mae = mean_absolute_error(actuals, preds)

    pct_pred = (preds[1:] - preds[:-1]) / preds[:-1]
    pct_actual = (actuals[1:] - actuals[:-1]) / actuals[:-1]
    pct_mae = mean_absolute_error(pct_actual, pct_pred)

    print(f"  → Price MAE: ${price_mae:.2f}")
    print(f"  → Pct Return MAE: {pct_mae*100:.2f}%")
    print(f"  → Time: {elapsed:.1f}s ({elapsed/10:.2f}s per forecast)")

    return {
        "name": name,
        "num_samples": num_samples,
        "aggregate": aggregate,
        "price_mae": float(price_mae),
        "pct_mae": float(pct_mae),
        "time_s": float(elapsed)
    }

print("\n" + "="*70)
print("ETHUSD FORECASTING TESTS")
print("="*70)

results = []

# Test baseline (what ETHUSD currently uses)
results.append(quick_forecast_test(128, "trimmed_mean_20", "Current (baseline)"))

# Test improvements
results.append(quick_forecast_test(512, "trimmed_mean_10", "4x samples, less trimming"))
results.append(quick_forecast_test(1024, "trimmed_mean_5", "Like BTCUSD (best)"))
results.append(quick_forecast_test(2048, "trimmed_mean_5", "2x BTCUSD"))

print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)

best = min(results, key=lambda x: x["pct_mae"])

for r in sorted(results, key=lambda x: x["pct_mae"]):
    marker = " ← BEST!" if r == best else ""
    print(f"{r['name']:30s} → {r['pct_mae']*100:5.2f}% MAE{marker}")

improvement = ((results[0]["pct_mae"] - best["pct_mae"]) / results[0]["pct_mae"] * 100)
print(f"\nImprovement over baseline: {improvement:.1f}%")

# Save results
with open("results/simple_crypto_test.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Saved to: results/simple_crypto_test.json")
print("="*70)
