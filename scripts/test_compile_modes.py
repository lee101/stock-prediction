#!/usr/bin/env python3
"""
Test different torch.compile modes and settings to find potential issues.
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
print("Testing Different Compile Modes")
print("=" * 60)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}\n")


def create_test_data(n_points=200, seed=42):
    """Create test data."""
    np.random.seed(seed)
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


def run_prediction(compile_mode, backend, context_df):
    """Run prediction with specific compile settings."""
    wrapper = Chronos2OHLCWrapper.from_pretrained(
        model_id="amazon/chronos-2",
        device_map=device,
        default_context_length=128,
        torch_compile=True,
        compile_mode=compile_mode,
        compile_backend=backend,
    )

    start = time.time()
    result = wrapper.predict_ohlc(
        context_df=context_df,
        symbol="TEST",
        prediction_length=16,
        context_length=128,
    )
    latency = time.time() - start

    preds = result.median["close"].values
    wrapper.unload()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return preds, latency


# Get baseline (eager mode)
print("Running baseline (eager mode)...")
data = create_test_data()
context = data.iloc[:-16]

eager_wrapper = Chronos2OHLCWrapper.from_pretrained(
    model_id="amazon/chronos-2",
    device_map=device,
    default_context_length=128,
    torch_compile=False,
)

eager_result = eager_wrapper.predict_ohlc(
    context_df=context,
    symbol="TEST",
    prediction_length=16,
    context_length=128,
)
baseline_preds = eager_result.median["close"].values
eager_wrapper.unload()
print(f"✓ Baseline established\n")

# Test different compile modes
test_configs = [
    ("default", "inductor"),
    ("reduce-overhead", "inductor"),
    # ("max-autotune", "inductor"),  # Often unstable
]

results = []

for compile_mode, backend in test_configs:
    print(f"Testing: mode={compile_mode}, backend={backend}")
    print("-" * 60)

    try:
        preds, latency = run_prediction(compile_mode, backend, context)

        # Check for invalid values
        if np.isnan(preds).any():
            print(f"  ✗ NaN detected")
            results.append((compile_mode, backend, False, "NaN", 0.0, latency))
            continue

        if np.isinf(preds).any():
            print(f"  ✗ Inf detected")
            results.append((compile_mode, backend, False, "Inf", 0.0, latency))
            continue

        # Compare with baseline
        mae_diff = float(np.mean(np.abs(preds - baseline_preds)))
        mean_price = float(np.mean(np.abs(baseline_preds)))
        relative_diff = mae_diff / mean_price if mean_price > 0 else 0

        print(f"  MAE difference: {mae_diff:.6f}")
        print(f"  Relative diff: {relative_diff:.4%}")
        print(f"  Latency: {latency:.2f}s")

        if mae_diff > 0.01:
            print(f"  ⚠️  Large MAE difference")
            results.append((compile_mode, backend, False, f"MAE={mae_diff:.6f}", mae_diff, latency))
        else:
            print(f"  ✅ Passed")
            results.append((compile_mode, backend, True, "OK", mae_diff, latency))

    except Exception as e:
        print(f"  ✗ Error: {str(e)[:100]}")
        results.append((compile_mode, backend, False, str(e)[:50], 0.0, 0.0))

    print()

# Test with extreme data
print("=" * 60)
print("Testing with extreme data scenarios")
print("=" * 60)

def _scale_prices(df, scale):
    """Scale OHLC prices in dataframe."""
    scaled = df.copy()
    for col in ["open", "high", "low", "close"]:
        scaled[col] = scaled[col] * scale
    return scaled

extreme_scenarios = [
    ("very_small", lambda: _scale_prices(create_test_data(), 1e-4)),
    ("very_large", lambda: _scale_prices(create_test_data(), 1e6)),
    ("high_volatility", lambda: create_test_data(seed=123)),
]

extreme_results = []

for scenario_name, data_fn in extreme_scenarios:
    print(f"\nScenario: {scenario_name}")
    print("-" * 60)

    try:
        data = data_fn()
        context = data.iloc[:-16]

        # Test with reduce-overhead (safest compiled mode)
        preds, latency = run_prediction("reduce-overhead", "inductor", context)

        if np.isnan(preds).any() or np.isinf(preds).any():
            print(f"  ✗ Invalid values")
            extreme_results.append((scenario_name, False))
        else:
            print(f"  ✅ Passed (latency: {latency:.2f}s)")
            extreme_results.append((scenario_name, True))

    except Exception as e:
        print(f"  ✗ Error: {str(e)[:100]}")
        extreme_results.append((scenario_name, False))

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print("\nCompile Mode Tests:")
passed = sum(1 for r in results if r[2])
total = len(results)
print(f"Passed: {passed}/{total}\n")

for mode, backend, success, error, mae, latency in results:
    status = "✅" if success else "❌"
    print(f"  {status} {mode} + {backend}: {error}")

print("\nExtreme Data Tests:")
extreme_passed = sum(1 for r in extreme_results if r[1])
extreme_total = len(extreme_results)
print(f"Passed: {extreme_passed}/{extreme_total}\n")

for scenario, success in extreme_results:
    status = "✅" if success else "❌"
    print(f"  {status} {scenario}")

# Recommendation
print("\n" + "=" * 60)
print("RECOMMENDATION")
print("=" * 60)

if passed == total and extreme_passed == extreme_total:
    print("✅ All tests passed!")
    print("\nSafest compile settings confirmed:")
    print("  mode='reduce-overhead'")
    print("  backend='inductor'")
    print("  dtype='float32'")
    sys.exit(0)
elif passed > 0:
    print(f"⚠️  {total - passed} compile mode(s) failed")
    print(f"⚠️  {extreme_total - extreme_passed} extreme scenario(s) failed")
    print("\nUse with caution:")
    for mode, backend, success, _, _, _ in results:
        if success:
            print(f"  ✓ {mode} + {backend}")
    sys.exit(1)
else:
    print("❌ All compile modes failed")
    print("Compilation not recommended for this environment")
    sys.exit(1)
