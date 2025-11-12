"""
Performance and accuracy comparison between compiled and eager modes for Chronos2.
Tests both modes on real training data to measure:
1. Prediction latency
2. Mean Absolute Error (MAE)
3. Stability (success rate across multiple runs)
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Set environment variables
os.environ["ONLY_CHRONOS2"] = "1"
os.environ["REAL_TESTING"] = "1"

print("="*80)
print("CHRONOS2 PERFORMANCE TEST: COMPILED vs EAGER")
print("="*80)

def load_training_data(symbol="BTCUSD", n_rows=500):
    """Load training data for testing."""
    data_path = Path(__file__).parent / "trainingdata" / f"{symbol}.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    df = pd.read_csv(data_path)
    df = df.tail(n_rows).copy()
    df = df.reset_index(drop=True)
    df.columns = [col.lower() for col in df.columns]
    df["timestamp"] = pd.date_range(start="2024-01-01", periods=len(df), freq="D")
    df["symbol"] = symbol

    return df

def test_mode(mode_name, torch_compiled, num_runs=5, prediction_length=7):
    """Test a specific mode (compiled or eager)."""
    print(f"\n{'='*80}")
    print(f"Testing: {mode_name}")
    print(f"{'='*80}")

    # Set environment variable
    os.environ["TORCH_COMPILED"] = torch_compiled

    # Force reimport to pick up new env var
    if "backtest_test3_inline" in sys.modules:
        del sys.modules["backtest_test3_inline"]

    from backtest_test3_inline import (
        load_chronos2_wrapper,
        resolve_chronos2_params,
    )

    # Load data
    print(f"Loading BTCUSD training data...")
    df = load_training_data("BTCUSD", n_rows=500)
    print(f"  Data shape: {df.shape}")

    # Split into context and ground truth
    split_idx = len(df) - prediction_length
    context_df = df.iloc[:split_idx].copy()
    ground_truth = df.iloc[split_idx:].copy()

    print(f"  Context length: {len(context_df)}")
    print(f"  Ground truth length: {len(ground_truth)}")

    # Load model
    print(f"\nLoading Chronos2 wrapper (TORCH_COMPILED={torch_compiled})...")
    params = resolve_chronos2_params("BTCUSD")
    wrapper = load_chronos2_wrapper(params)
    print(f"  ✓ Wrapper loaded")

    # Run predictions
    results = {
        "latencies": [],
        "predictions": [],
        "errors": [],
        "success_count": 0,
    }

    print(f"\nRunning {num_runs} prediction iterations...")
    for i in range(num_runs):
        try:
            start_time = time.time()

            result = wrapper.predict_ohlc(
                context_df=context_df.copy(),
                symbol="BTCUSD",
                prediction_length=prediction_length,
                context_length=min(params["context_length"], len(context_df)),
                batch_size=params["batch_size"],
            )

            latency = time.time() - start_time

            # Extract median predictions
            median_frame = result.quantile_frames[0.5]
            close_predictions = median_frame["close"].values

            # Calculate MAE
            ground_truth_close = ground_truth["close"].values
            mae = np.mean(np.abs(close_predictions - ground_truth_close))
            mae_percent = (mae / np.mean(ground_truth_close)) * 100

            results["latencies"].append(latency)
            results["predictions"].append(close_predictions)
            results["errors"].append(mae_percent)
            results["success_count"] += 1

            print(f"  Run {i+1}/{num_runs}: latency={latency:.3f}s, MAE={mae_percent:.2f}%")

        except Exception as e:
            print(f"  Run {i+1}/{num_runs}: FAILED - {str(e)[:100]}")
            results["errors"].append(None)

    # Calculate statistics
    if results["success_count"] > 0:
        avg_latency = np.mean(results["latencies"])
        std_latency = np.std(results["latencies"])
        valid_maes = [e for e in results["errors"] if e is not None]
        avg_mae = np.mean(valid_maes) if valid_maes else None
        std_mae = np.std(valid_maes) if valid_maes else None
        success_rate = (results["success_count"] / num_runs) * 100

        print(f"\n{'='*80}")
        print(f"RESULTS: {mode_name}")
        print(f"{'='*80}")
        print(f"Success rate:    {success_rate:.1f}% ({results['success_count']}/{num_runs})")
        print(f"Avg latency:     {avg_latency:.3f}s ± {std_latency:.3f}s")
        if avg_mae is not None:
            print(f"Avg MAE:         {avg_mae:.2f}% ± {std_mae:.2f}%")
        print(f"{'='*80}")

        return {
            "mode": mode_name,
            "torch_compiled": torch_compiled,
            "success_rate": success_rate,
            "avg_latency": avg_latency,
            "std_latency": std_latency,
            "avg_mae": avg_mae,
            "std_mae": std_mae,
            "raw_results": results,
        }
    else:
        print(f"\n{'='*80}")
        print(f"RESULTS: {mode_name}")
        print(f"{'='*80}")
        print(f"Success rate:    0% (ALL RUNS FAILED)")
        print(f"{'='*80}")

        return {
            "mode": mode_name,
            "torch_compiled": torch_compiled,
            "success_rate": 0,
            "avg_latency": None,
            "std_latency": None,
            "avg_mae": None,
            "std_mae": None,
            "raw_results": results,
        }

def main():
    """Run performance comparison."""
    num_runs = 5
    prediction_length = 7

    print(f"\nTest configuration:")
    print(f"  Symbol: BTCUSD")
    print(f"  Training data rows: 500")
    print(f"  Prediction length: {prediction_length}")
    print(f"  Iterations per mode: {num_runs}")

    # Test eager mode (non-compiled)
    eager_results = test_mode(
        mode_name="EAGER MODE (TORCH_COMPILED=0)",
        torch_compiled="0",
        num_runs=num_runs,
        prediction_length=prediction_length,
    )

    # Clear cache and wait before next test
    print("\n" + "="*80)
    print("Waiting 5 seconds before compiled mode test...")
    print("="*80)
    time.sleep(5)

    # Test compiled mode
    compiled_results = test_mode(
        mode_name="COMPILED MODE (TORCH_COMPILED=1)",
        torch_compiled="1",
        num_runs=num_runs,
        prediction_length=prediction_length,
    )

    # Comparison
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)

    print(f"\n{'Mode':<30} {'Success Rate':<15} {'Avg Latency':<15} {'Avg MAE %':<15}")
    print("-" * 75)

    for results in [eager_results, compiled_results]:
        mode = results["mode"]
        success = f"{results['success_rate']:.1f}%"
        latency = f"{results['avg_latency']:.3f}s" if results['avg_latency'] else "N/A"
        mae = f"{results['avg_mae']:.2f}%" if results['avg_mae'] else "N/A"
        print(f"{mode:<30} {success:<15} {latency:<15} {mae:<15}")

    # Calculate improvements
    if eager_results["success_rate"] > 0 and compiled_results["success_rate"] > 0:
        print("\n" + "="*80)
        print("PERFORMANCE DIFFERENCE")
        print("="*80)

        if eager_results["avg_latency"] and compiled_results["avg_latency"]:
            speedup = eager_results["avg_latency"] / compiled_results["avg_latency"]
            speedup_pct = (speedup - 1) * 100
            print(f"Speedup: {speedup:.2f}x ({speedup_pct:+.1f}%)")

        if eager_results["avg_mae"] and compiled_results["avg_mae"]:
            mae_diff = compiled_results["avg_mae"] - eager_results["avg_mae"]
            mae_diff_pct = (mae_diff / eager_results["avg_mae"]) * 100
            print(f"MAE difference: {mae_diff:+.2f}pp ({mae_diff_pct:+.1f}%)")

    # Recommendation
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)

    if eager_results["success_rate"] == 100 and compiled_results["success_rate"] < 100:
        print("❌ COMPILED MODE UNSTABLE - Use TORCH_COMPILED=0 (eager mode)")
        print("   Compiled mode has failures, eager mode is 100% reliable")
    elif eager_results["success_rate"] == 100 and compiled_results["success_rate"] == 100:
        if compiled_results["avg_latency"] < eager_results["avg_latency"]:
            speedup = eager_results["avg_latency"] / compiled_results["avg_latency"]
            print(f"✅ COMPILED MODE FASTER - Consider using TORCH_COMPILED=1")
            print(f"   {speedup:.2f}x speedup with same stability")
        else:
            print("✅ EAGER MODE RECOMMENDED - TORCH_COMPILED=0")
            print("   Similar or better performance without compilation overhead")
    else:
        print("⚠️  BOTH MODES HAVE ISSUES - Further investigation needed")

    print("="*80)

if __name__ == "__main__":
    main()
