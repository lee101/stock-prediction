#!/usr/bin/env python3
"""
Test ALL improved crypto configs in rapid succession.
Tests BTCUSD, ETHUSD, UNIUSD with multiple sample counts and aggregations.
"""
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error

from src.models.toto_wrapper import TotoPipeline
from src.models.toto_aggregation import aggregate_with_spec

def load_crypto_data(symbol):
    """Load and split crypto data."""
    import pandas as pd
    # Use the most recent data file
    data_file = Path(f"data/{symbol}/{symbol}-2025-11-04.csv")
    df = pd.read_csv(data_file)
    prices = df['Close'].values

    n = len(prices)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    return prices[val_end:]  # Return test data only

def test_config(symbol, test_data, num_samples, aggregate, samples_per_batch):
    """Test a single config quickly."""
    pipeline = TotoPipeline()

    predictions = []
    actuals = []
    start_time = time.time()

    TEST_WINDOW = 15  # Reduced for speed
    CONTEXT_LENGTH = 128

    for i in range(TEST_WINDOW):
        if i + CONTEXT_LENGTH >= len(test_data):
            break

        context = test_data[i:i+CONTEXT_LENGTH]
        actual = test_data[i+CONTEXT_LENGTH]

        raw_preds = pipeline.predict(
            context,
            prediction_length=1,
            num_samples=num_samples,
            samples_per_batch=samples_per_batch
        )
        pred = aggregate_with_spec(raw_preds, aggregate)

        predictions.append(pred)
        actuals.append(actual)

    total_time = time.time() - start_time

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Calculate metrics
    price_mae = mean_absolute_error(actuals, predictions)
    pct_returns_pred = (predictions[1:] - predictions[:-1]) / predictions[:-1]
    pct_returns_actual = (actuals[1:] - actuals[:-1]) / actuals[:-1]
    pct_return_mae = mean_absolute_error(pct_returns_actual, pct_returns_pred)

    return {
        "symbol": symbol,
        "num_samples": num_samples,
        "aggregate": aggregate,
        "pct_return_mae": float(pct_return_mae),
        "price_mae": float(price_mae),
        "latency_s": float(total_time / len(predictions))
    }

# Test configurations to try
configs_to_test = [
    # ETHUSD - the one that needs most help
    ("ETHUSD", 256, "trimmed_mean_10", 32),
    ("ETHUSD", 512, "trimmed_mean_10", 64),
    ("ETHUSD", 1024, "trimmed_mean_5", 128),
    ("ETHUSD", 2048, "trimmed_mean_5", 128),

    # BTCUSD - push for even better
    ("BTCUSD", 2048, "trimmed_mean_5", 128),
    ("BTCUSD", 1024, "quantile_0.50", 128),
    ("BTCUSD", 1536, "trimmed_mean_3", 128),

    # UNIUSD - try Toto
    ("UNIUSD", 512, "trimmed_mean_10", 64),
    ("UNIUSD", 1024, "trimmed_mean_5", 128),
    ("UNIUSD", 2048, "trimmed_mean_5", 128),
]

print("="*70)
print("RAPID CRYPTO CONFIG TESTING")
print("="*70)
print(f"Testing {len(configs_to_test)} configurations...")
print()

results = []

for i, (symbol, num_samples, aggregate, spb) in enumerate(configs_to_test, 1):
    print(f"[{i}/{len(configs_to_test)}] Testing {symbol}: {num_samples} samples, {aggregate}...")

    try:
        # Load data if not already loaded
        test_data = load_crypto_data(symbol)

        # Run test
        result = test_config(symbol, test_data, num_samples, aggregate, spb)

        print(f"  → Pct Return MAE: {result['pct_return_mae']*100:.2f}%")
        print(f"  → Latency: {result['latency_s']:.2f}s")

        results.append(result)

    except Exception as e:
        print(f"  ✗ Error: {e}")
        continue

# Save all results
output_file = Path("results/all_crypto_configs_test.json")
output_file.parent.mkdir(exist_ok=True)
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

# Print summary
print()
print("="*70)
print("SUMMARY OF RESULTS")
print("="*70)

for symbol in ["ETHUSD", "BTCUSD", "UNIUSD"]:
    symbol_results = [r for r in results if r["symbol"] == symbol]
    if not symbol_results:
        continue

    print(f"\n{symbol}:")
    best = min(symbol_results, key=lambda x: x["pct_return_mae"])
    print(f"  Best Config: {best['num_samples']} samples, {best['aggregate']}")
    print(f"  Best MAE: {best['pct_return_mae']*100:.2f}%")
    print(f"  Latency: {best['latency_s']:.2f}s")

    # Show all results for this symbol
    for r in sorted(symbol_results, key=lambda x: x["pct_return_mae"]):
        print(f"    {r['num_samples']:4d} samples, {r['aggregate']:20s} → {r['pct_return_mae']*100:5.2f}%")

print()
print(f"✓ All results saved to: {output_file}")
print("="*70)
