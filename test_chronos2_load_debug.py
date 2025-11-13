"""
Load test and debug for Chronos2 torch.compile.
Tests if compilation is actually working and providing benefits.
"""
import os
import sys
import time
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))

os.environ["ONLY_CHRONOS2"] = "1"
os.environ["REAL_TESTING"] = "1"

print("="*80)
print("CHRONOS2 TORCH.COMPILE DEBUG & LOAD TEST")
print("="*80)

def load_test_data(n_rows=200):
    """Load BTCUSD test data."""
    data_path = Path(__file__).parent / "trainingdata" / "BTCUSD.csv"
    df = pd.read_csv(data_path).tail(n_rows).copy()
    df = df.reset_index(drop=True)
    df.columns = [col.lower() for col in df.columns]
    df["timestamp"] = pd.date_range("2024-01-01", periods=len(df), freq="D")
    df["symbol"] = "BTCUSD"
    return df

def test_compiled_mode(num_iterations=10, warmup_iterations=2):
    """Test torch compiled mode with load testing."""
    print(f"\n{'='*80}")
    print("TESTING TORCH_COMPILED=1 (Compiled Mode)")
    print(f"{'='*80}")

    os.environ["TORCH_COMPILED"] = "1"

    # Force reimport
    if "backtest_test3_inline" in sys.modules:
        del sys.modules["backtest_test3_inline"]

    from backtest_test3_inline import (
        load_chronos2_wrapper,
        resolve_chronos2_params,
    )

    # Load data
    df = load_test_data(200)
    print(f"✓ Loaded {len(df)} rows of BTCUSD data")

    # Load model
    print(f"\nLoading model with TORCH_COMPILED=1...")
    params = resolve_chronos2_params("BTCUSD")

    # Capture warnings during model load
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        wrapper = load_chronos2_wrapper(params)
        if w:
            print(f"⚠ Warnings during model load: {len(w)}")
            for warning in w[:3]:  # Show first 3
                print(f"  - {warning.category.__name__}: {warning.message}")

    print(f"✓ Model loaded")

    # Check if torch.compile was actually applied
    print(f"\nChecking compilation status...")
    print(f"  torch.compile available: {hasattr(torch, '_dynamo')}")
    if hasattr(wrapper, 'pipeline') and hasattr(wrapper.pipeline, 'model'):
        model = wrapper.pipeline.model
        print(f"  Model type: {type(model).__name__}")
        print(f"  Is compiled: {hasattr(model, '_orig_mod')}")  # torch.compile wraps models

    # Warmup runs (compilation happens here)
    print(f"\n{'='*80}")
    print(f"WARMUP PHASE ({warmup_iterations} iterations)")
    print(f"{'='*80}")
    print("Compilation happens during warmup - expect warnings here")

    warmup_times = []
    for i in range(warmup_iterations):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            start = time.time()
            result = wrapper.predict_ohlc(
                context_df=df.copy(),
                symbol="BTCUSD",
                prediction_length=7,
                context_length=min(params["context_length"], len(df)),
                batch_size=params["batch_size"],
            )
            elapsed = time.time() - start
            warmup_times.append(elapsed)

            # Count warning types
            warning_types = {}
            for warning in w:
                msg = str(warning.message)
                if "cudagraphs" in msg:
                    warning_types["cudagraphs"] = warning_types.get("cudagraphs", 0) + 1
                elif "symbolic_shapes" in msg:
                    warning_types["symbolic_shapes"] = warning_types.get("symbolic_shapes", 0) + 1
                else:
                    warning_types["other"] = warning_types.get("other", 0) + 1

            print(f"  Warmup {i+1}/{warmup_iterations}: {elapsed:.3f}s", end="")
            if warning_types:
                print(f" - Warnings: {dict(warning_types)}")
            else:
                print(" - No warnings")

    print(f"\nWarmup times: {[f'{t:.3f}s' for t in warmup_times]}")

    # Main load test
    print(f"\n{'='*80}")
    print(f"LOAD TEST ({num_iterations} iterations)")
    print(f"{'='*80}")
    print("Compiled graph should be cached now - should be faster")

    times = []
    errors = []
    warning_counts = []

    for i in range(num_iterations):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            try:
                start = time.time()
                result = wrapper.predict_ohlc(
                    context_df=df.copy(),
                    symbol="BTCUSD",
                    prediction_length=7,
                    context_length=min(params["context_length"], len(df)),
                    batch_size=params["batch_size"],
                )
                elapsed = time.time() - start
                times.append(elapsed)

                # Verify result
                median = result.quantile_frames[0.5]
                pred_close = median["close"].values
                if len(pred_close) != 7:
                    errors.append(f"Wrong prediction length: {len(pred_close)}")
                if np.any(np.isnan(pred_close)):
                    errors.append("NaN in predictions")

                warning_counts.append(len(w))

                print(f"  Iteration {i+1}/{num_iterations}: {elapsed:.3f}s - {len(w)} warnings")

            except Exception as e:
                errors.append(str(e))
                print(f"  Iteration {i+1}/{num_iterations}: FAILED - {str(e)[:60]}")

    # Results
    print(f"\n{'='*80}")
    print("RESULTS - COMPILED MODE")
    print(f"{'='*80}")

    if len(times) > 0:
        print(f"Success rate:       {len(times)}/{num_iterations} ({len(times)/num_iterations*100:.1f}%)")
        print(f"")
        print(f"TIMING:")
        print(f"  Warmup avg:       {np.mean(warmup_times):.3f}s")
        print(f"  Main test avg:    {np.mean(times):.3f}s ± {np.std(times):.3f}s")
        print(f"  Min:              {np.min(times):.3f}s")
        print(f"  Max:              {np.max(times):.3f}s")
        print(f"  Speedup:          {np.mean(warmup_times) / np.mean(times):.2f}x (warmup vs steady)")
        print(f"")
        print(f"WARNINGS:")
        print(f"  Total warnings:   {sum(warning_counts)}")
        print(f"  Avg per iter:     {np.mean(warning_counts):.1f}")
        print(f"  Warnings in last 3: {warning_counts[-3:] if len(warning_counts) >= 3 else warning_counts}")

        if sum(warning_counts[-3:]) > 0:
            print(f"  ⚠ Still getting warnings after warmup - compilation may not be optimal")
        else:
            print(f"  ✓ No warnings in final iterations - compilation successful")

    if errors:
        print(f"")
        print(f"ERRORS:")
        for err in errors[:5]:
            print(f"  - {err}")

    print(f"{'='*80}")

    return {
        "mode": "compiled",
        "success_rate": len(times) / num_iterations,
        "avg_time": np.mean(times) if times else None,
        "std_time": np.std(times) if times else None,
        "warmup_time": np.mean(warmup_times),
        "speedup": np.mean(warmup_times) / np.mean(times) if times else None,
        "warning_count": sum(warning_counts),
        "errors": errors,
    }

def test_eager_mode(num_iterations=10):
    """Test eager mode for comparison."""
    print(f"\n{'='*80}")
    print("TESTING TORCH_COMPILED=0 (Eager Mode - Baseline)")
    print(f"{'='*80}")

    os.environ["TORCH_COMPILED"] = "0"

    # Force reimport
    if "backtest_test3_inline" in sys.modules:
        del sys.modules["backtest_test3_inline"]

    from backtest_test3_inline import (
        load_chronos2_wrapper,
        resolve_chronos2_params,
    )

    # Load data
    df = load_test_data(200)

    # Load model
    print(f"Loading model with TORCH_COMPILED=0...")
    params = resolve_chronos2_params("BTCUSD")
    wrapper = load_chronos2_wrapper(params)
    print(f"✓ Model loaded")

    # Run test
    print(f"\nRunning {num_iterations} iterations...")

    times = []
    errors = []

    for i in range(num_iterations):
        try:
            start = time.time()
            result = wrapper.predict_ohlc(
                context_df=df.copy(),
                symbol="BTCUSD",
                prediction_length=7,
                context_length=min(params["context_length"], len(df)),
                batch_size=params["batch_size"],
            )
            elapsed = time.time() - start
            times.append(elapsed)

            print(f"  Iteration {i+1}/{num_iterations}: {elapsed:.3f}s")

        except Exception as e:
            errors.append(str(e))
            print(f"  Iteration {i+1}/{num_iterations}: FAILED - {str(e)[:60]}")

    # Results
    print(f"\n{'='*80}")
    print("RESULTS - EAGER MODE")
    print(f"{'='*80}")

    if len(times) > 0:
        print(f"Success rate:       {len(times)}/{num_iterations} ({len(times)/num_iterations*100:.1f}%)")
        print(f"")
        print(f"TIMING:")
        print(f"  Avg time:         {np.mean(times):.3f}s ± {np.std(times):.3f}s")
        print(f"  Min:              {np.min(times):.3f}s")
        print(f"  Max:              {np.max(times):.3f}s")

    if errors:
        print(f"")
        print(f"ERRORS:")
        for err in errors[:5]:
            print(f"  - {err}")

    print(f"{'='*80}")

    return {
        "mode": "eager",
        "success_rate": len(times) / num_iterations,
        "avg_time": np.mean(times) if times else None,
        "std_time": np.std(times) if times else None,
        "errors": errors,
    }

def main():
    """Run debug load test."""
    num_iterations = 10
    warmup_iterations = 2

    print(f"\nTest configuration:")
    print(f"  Warmup iterations: {warmup_iterations}")
    print(f"  Test iterations: {num_iterations}")
    print(f"  Data: BTCUSD (200 rows)")
    print(f"  Prediction length: 7 days")

    # Test compiled mode first
    compiled_results = test_compiled_mode(num_iterations, warmup_iterations)

    print(f"\n{'='*80}")
    print("Waiting 5 seconds before eager mode test...")
    print(f"{'='*80}")
    time.sleep(5)

    # Test eager mode for baseline
    eager_results = test_eager_mode(num_iterations)

    # Final comparison
    print(f"\n{'='*80}")
    print("FINAL COMPARISON")
    print(f"{'='*80}")

    if compiled_results["avg_time"] and eager_results["avg_time"]:
        speedup = eager_results["avg_time"] / compiled_results["avg_time"]
        print(f"\nSpeedup (eager baseline / compiled): {speedup:.2f}x")
        print(f"  Eager:    {eager_results['avg_time']:.3f}s")
        print(f"  Compiled: {compiled_results['avg_time']:.3f}s")

        if speedup > 1.2:
            print(f"\n✅ Compiled mode is {speedup:.2f}x faster")
        elif speedup > 0.95:
            print(f"\n➡️ Similar performance ({speedup:.2f}x)")
        else:
            print(f"\n❌ Compiled mode is SLOWER ({speedup:.2f}x)")

    if compiled_results["warning_count"] > 0:
        print(f"\n⚠ Compiled mode had {compiled_results['warning_count']} warnings")
        print(f"   This suggests torch.compile is not fully optimizing")
        print(f"   Common causes:")
        print(f"   - Dynamic shapes (batch size changes)")
        print(f"   - Mutated inputs (in-place operations)")
        print(f"   - Unsupported operations")

    # Recommendation
    print(f"\n{'='*80}")
    print("RECOMMENDATION")
    print(f"{'='*80}")

    if compiled_results["success_rate"] < 1.0:
        print("❌ COMPILED MODE UNSTABLE")
        print(f"   Use TORCH_COMPILED=0 (eager mode)")
    elif compiled_results["avg_time"] and eager_results["avg_time"]:
        speedup = eager_results["avg_time"] / compiled_results["avg_time"]
        if speedup > 1.3 and compiled_results["warning_count"] == 0:
            print("✅ COMPILED MODE WORKING WELL")
            print(f"   {speedup:.2f}x speedup with no warnings")
            print(f"   Keep TORCH_COMPILED=1")
        elif speedup > 1.3:
            print("⚠️ COMPILED MODE FASTER BUT WITH WARNINGS")
            print(f"   {speedup:.2f}x speedup but {compiled_results['warning_count']} warnings")
            print(f"   torch.compile is skipping optimizations")
            print(f"   Consider investigating the warnings")
        else:
            print("❌ COMPILED MODE NOT PROVIDING BENEFIT")
            print(f"   Only {speedup:.2f}x speedup")
            print(f"   Consider using TORCH_COMPILED=0")

    print(f"{'='*80}")

if __name__ == "__main__":
    main()
