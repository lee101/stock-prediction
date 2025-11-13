#!/usr/bin/env python3
"""
Quick sanity test for Chronos2 compilation.
Runs a few scenarios to verify basic functionality.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.chronos2_wrapper import Chronos2OHLCWrapper

print("=" * 60)
print("Quick Chronos2 Compilation Test")
print("=" * 60)


def create_test_data(n_points=200):
    """Create simple test data."""
    np.random.seed(42)
    returns = np.random.randn(n_points) * 0.02
    prices = 100.0 * np.exp(np.cumsum(returns))

    return pd.DataFrame({
        "timestamp": pd.date_range(start="2024-01-01", periods=n_points, freq="D"),
        "open": prices * (1 + np.random.randn(n_points) * 0.002),
        "high": prices * (1 + np.abs(np.random.randn(n_points)) * 0.005),
        "low": prices * (1 - np.abs(np.random.randn(n_points)) * 0.005),
        "close": prices,
        "symbol": "TEST",
    })


def run_test(mode: str, compile_enabled: bool):
    """Run a single test."""
    print(f"\nTesting {mode}...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = create_test_data()
    context = data.iloc[:-16]

    start = time.time()

    wrapper = Chronos2OHLCWrapper.from_pretrained(
        model_id="amazon/chronos-2",
        device_map=device,
        default_context_length=128,
        torch_compile=compile_enabled,
        compile_mode="reduce-overhead" if compile_enabled else None,
    )

    load_time = time.time() - start
    print(f"  Load time: {load_time:.2f}s")

    # First prediction (includes compilation if enabled)
    start = time.time()
    result1 = wrapper.predict_ohlc(
        context_df=context,
        symbol="TEST",
        prediction_length=16,
        context_length=128,
    )
    first_time = time.time() - start
    print(f"  First prediction: {first_time:.2f}s")

    # Second prediction (uses cached compiled model)
    start = time.time()
    result2 = wrapper.predict_ohlc(
        context_df=context,
        symbol="TEST",
        prediction_length=16,
        context_length=128,
    )
    second_time = time.time() - start
    print(f"  Second prediction: {second_time:.2f}s")

    # Check consistency
    preds1 = result1.median["close"].values
    preds2 = result2.median["close"].values
    consistency_mae = np.mean(np.abs(preds1 - preds2))

    print(f"  Consistency MAE: {consistency_mae:.6f}")
    print(f"  ✓ {mode} passed")

    wrapper.unload()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return preds1


try:
    # Test eager mode
    eager_preds = run_test("Eager mode", compile_enabled=False)

    # Test compiled mode
    compiled_preds = run_test("Compiled mode", compile_enabled=True)

    # Compare modes
    print("\n" + "=" * 60)
    print("Comparing modes...")
    mae_diff = np.mean(np.abs(eager_preds - compiled_preds))
    print(f"MAE difference: {mae_diff:.6f}")

    if mae_diff < 1e-2:
        print("✅ PASS - Modes produce similar results")
        sys.exit(0)
    else:
        print(f"⚠️  WARNING - Large MAE difference: {mae_diff:.6f}")
        sys.exit(1)

except Exception as e:
    print(f"\n❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
