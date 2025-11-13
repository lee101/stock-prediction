#!/usr/bin/env python3
"""
Test Chronos2 compilation with real trading data.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.chronos2_wrapper import Chronos2OHLCWrapper

print("=" * 60)
print("Testing Chronos2 Compilation with Real Data")
print("=" * 60)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}\n")

trainingdata_dir = Path("trainingdata")

if not trainingdata_dir.exists():
    print("❌ trainingdata/ directory not found")
    print("Skipping real data tests")
    sys.exit(0)

# Test symbols
test_symbols = ["BTCUSD", "ETHUSD", "SOLUSD", "AAPL", "TSLA", "SPY", "NVDA"]

results = {}

for symbol in test_symbols:
    csv_path = trainingdata_dir / f"{symbol}.csv"

    if not csv_path.exists():
        print(f"⊘ {symbol}: not found")
        continue

    print(f"\nTesting {symbol}...")
    print("-" * 60)

    try:
        # Load data
        df = pd.read_csv(csv_path)

        # Ensure required columns
        required_cols = ["timestamp", "open", "high", "low", "close"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print(f"  ✗ Missing columns: {missing}")
            results[symbol] = False
            continue

        # Process timestamps
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

        if len(df) < 200:
            print(f"  ✗ Insufficient data: {len(df)} rows")
            results[symbol] = False
            continue

        # Use last 200 points
        df = df.tail(200).reset_index(drop=True)
        print(f"  Data points: {len(df)}")

        # Split context/prediction
        context = df.iloc[:-16]
        prediction_length = 16
        context_length = 128

        # Test eager mode
        print("  Running eager mode...")
        eager_wrapper = Chronos2OHLCWrapper.from_pretrained(
            model_id="amazon/chronos-2",
            device_map=device,
            default_context_length=context_length,
            torch_compile=False,
        )

        eager_result = eager_wrapper.predict_ohlc(
            context_df=context,
            symbol=symbol,
            prediction_length=prediction_length,
            context_length=context_length,
        )
        eager_preds = eager_result.median["close"].values
        eager_wrapper.unload()

        # Test compiled mode
        print("  Running compiled mode...")
        compiled_wrapper = Chronos2OHLCWrapper.from_pretrained(
            model_id="amazon/chronos-2",
            device_map=device,
            default_context_length=context_length,
            torch_compile=True,
            compile_mode="reduce-overhead",
        )

        compiled_result = compiled_wrapper.predict_ohlc(
            context_df=context,
            symbol=symbol,
            prediction_length=prediction_length,
            context_length=context_length,
        )
        compiled_preds = compiled_result.median["close"].values
        compiled_wrapper.unload()

        # Compare
        if np.isnan(eager_preds).any():
            print(f"  ✗ Eager mode produced NaN")
            results[symbol] = False
            continue

        if np.isnan(compiled_preds).any():
            print(f"  ✗ Compiled mode produced NaN")
            results[symbol] = False
            continue

        mae_diff = float(np.mean(np.abs(eager_preds - compiled_preds)))
        mean_price = float(np.mean(np.abs(eager_preds)))
        relative_diff = mae_diff / mean_price if mean_price > 0 else 0

        print(f"  MAE difference: {mae_diff:.6f}")
        print(f"  Relative diff: {relative_diff:.2%}")

        # Check for price scale issues
        if mean_price < 1e-10:
            print(f"  ⚠️ Warning: Very small predictions (mean={mean_price:.2e})")

        # Validate
        if mae_diff > 0.01 and relative_diff > 0.05:
            print(f"  ✗ Large difference detected")
            results[symbol] = False
        else:
            print(f"  ✅ Passed")
            results[symbol] = True

        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"  ✗ Error: {str(e)[:100]}")
        results[symbol] = False
        import traceback
        traceback.print_exc()

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

if not results:
    print("No symbols tested")
    sys.exit(0)

passed = sum(1 for v in results.values() if v)
total = len(results)

print(f"\nPassed: {passed}/{total}")
print()

for symbol, result in results.items():
    status = "✅" if result else "❌"
    print(f"  {status} {symbol}")

if passed == total:
    print("\n✅ All real data tests passed!")
    sys.exit(0)
elif passed > 0:
    print(f"\n⚠️  {total - passed}/{total} symbols failed")
    sys.exit(1)
else:
    print("\n❌ All symbols failed")
    sys.exit(1)
