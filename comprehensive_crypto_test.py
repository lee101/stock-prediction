#!/usr/bin/env python3
"""
COMPREHENSIVE crypto forecasting test.
Test MANY configs with MANY forecasts to find real improvements.
"""
import json
import time
import numpy as np
from sklearn.metrics import mean_absolute_error

print("Loading Toto model...")
from src.models.toto_wrapper import TotoPipeline
from src.models.toto_aggregation import aggregate_with_spec

pipeline = TotoPipeline.from_pretrained(
    model_id="Datadog/Toto-Open-Base-1.0",
    device_map="cuda",
    compile_model=False,
)
print("✓ Model loaded!\n")

def test_symbol(symbol, num_forecasts=20):
    """Test a symbol with multiple configs."""
    import pandas as pd

    df = pd.read_csv(f"data/{symbol}/{symbol}-2025-11-04.csv")
    prices = df['Close'].values

    # Use last 250 points as test data
    test_data = prices[-250:]

    print(f"\n{'='*70}")
    print(f"TESTING: {symbol}")
    print(f"{'='*70}")
    print(f"Test data: {len(test_data)} prices")
    print(f"Forecasts per config: {num_forecasts}\n")

    # Configs to test - MANY variations
    configs = [
        # Baseline
        (128, "trimmed_mean_20"),
        (128, "trimmed_mean_10"),
        (128, "trimmed_mean_5"),

        # Medium samples
        (256, "trimmed_mean_20"),
        (256, "trimmed_mean_10"),
        (256, "trimmed_mean_5"),
        (512, "trimmed_mean_10"),
        (512, "trimmed_mean_5"),

        # High samples
        (1024, "trimmed_mean_10"),
        (1024, "trimmed_mean_5"),
        (1024, "trimmed_mean_3"),

        # Very high samples
        (2048, "trimmed_mean_5"),
        (2048, "trimmed_mean_3"),

        # Try quantiles
        (512, "quantile_0.50"),
        (1024, "quantile_0.50"),

        # Try mean (no trimming)
        (1024, "mean"),
    ]

    results = []

    for num_samples, aggregate in configs:
        print(f"Testing: {num_samples:4d} samples, {aggregate:20s} ...", end=" ", flush=True)

        preds = []
        actuals = []
        start = time.time()

        try:
            for i in range(num_forecasts):
                context = test_data[i:i+128]
                actual = test_data[i+128]

                forecasts = pipeline.predict(
                    context=context,
                    prediction_length=1,
                    num_samples=num_samples,
                    samples_per_batch=min(num_samples, 128)
                )
                pred = aggregate_with_spec(forecasts[0].samples, aggregate)

                preds.append(pred)
                actuals.append(actual)

            elapsed = time.time() - start

            preds = np.array(preds)
            actuals = np.array(actuals)

            price_mae = mean_absolute_error(actuals, preds)
            pct_pred = (preds[1:] - preds[:-1]) / preds[:-1]
            pct_actual = (actuals[1:] - actuals[:-1]) / actuals[:-1]
            pct_mae = mean_absolute_error(pct_actual, pct_pred)

            print(f"→ {pct_mae*100:5.2f}% MAE, {elapsed/num_forecasts:.2f}s/forecast")

            results.append({
                "symbol": symbol,
                "num_samples": num_samples,
                "aggregate": aggregate,
                "price_mae": float(price_mae),
                "pct_mae": float(pct_mae),
                "time_s": float(elapsed)
            })

        except Exception as e:
            print(f"✗ Error: {e}")

    # Find best
    if results:
        best = min(results, key=lambda x: x["pct_mae"])
        print(f"\n{'='*70}")
        print(f"BEST CONFIG FOR {symbol}:")
        print(f"  Samples: {best['num_samples']}, Aggregate: {best['aggregate']}")
        print(f"  Pct MAE: {best['pct_mae']*100:.2f}%")
        print(f"  Price MAE: ${best['price_mae']:.2f}")
        print(f"  Latency: {best['time_s']/num_forecasts:.2f}s per forecast")
        print(f"{'='*70}")

    return results

# Test all three crypto symbols
all_results = []

all_results.extend(test_symbol("ETHUSD", num_forecasts=20))
all_results.extend(test_symbol("BTCUSD", num_forecasts=20))
all_results.extend(test_symbol("UNIUSD", num_forecasts=20))

# Save all results
with open("results/comprehensive_crypto_test.json", "w") as f:
    json.dump(all_results, f, indent=2)

# Print overall summary
print(f"\n\n{'='*70}")
print("OVERALL SUMMARY")
print(f"{'='*70}\n")

for symbol in ["ETHUSD", "BTCUSD", "UNIUSD"]:
    symbol_results = [r for r in all_results if r["symbol"] == symbol]
    if symbol_results:
        best = min(symbol_results, key=lambda x: x["pct_mae"])
        baseline = [r for r in symbol_results if r["num_samples"] == 128 and r["aggregate"] == "trimmed_mean_20"]
        baseline = baseline[0] if baseline else None

        print(f"{symbol}:")
        print(f"  Best: {best['num_samples']} samples, {best['aggregate']}")
        print(f"    → {best['pct_mae']*100:.2f}% MAE")

        if baseline:
            improvement = ((baseline["pct_mae"] - best["pct_mae"]) / baseline["pct_mae"] * 100)
            print(f"  Baseline: 128 samples, trimmed_mean_20")
            print(f"    → {baseline['pct_mae']*100:.2f}% MAE")
            print(f"  Improvement: {improvement:.1f}%")
        print()

print(f"✓ Results saved to: results/comprehensive_crypto_test.json")
print(f"{'='*70}")
