#!/usr/bin/env python3
"""
Quick test to compare compiled vs eager prediction accuracy without full backtests.
This is faster and more practical for immediate decisions.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.toto_wrapper import TotoPipeline


def test_prediction_accuracy(mode: str, num_iterations: int = 3):
    """Test prediction accuracy in compiled or eager mode."""

    compiled = (mode == "compiled")

    # Generate test series
    np.random.seed(42)
    test_series = np.linspace(100, 110, 512) + np.random.randn(512) * 2

    predictions = []
    times = []

    print(f"\n{'='*60}")
    print(f"Testing {mode.upper()} mode ({num_iterations} iterations)")
    print(f"{'='*60}")

    for i in range(num_iterations):
        print(f"Iteration {i+1}/{num_iterations}...", end=" ", flush=True)

        # Set environment
        os.environ["TOTO_DISABLE_COMPILE"] = "0" if compiled else "1"
        if compiled:
            os.environ["TOTO_COMPILE_MODE"] = "max-autotune"

        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        start = time.perf_counter()

        try:
            pipeline = TotoPipeline.from_pretrained(
                "Datadog/Toto-Open-Base-1.0",
                device_map="cuda" if torch.cuda.is_available() else "cpu",
                torch_compile=compiled,
                compile_mode="max-autotune" if compiled else None,
            )

            pred = pipeline.predict(
                context=test_series,
                prediction_length=1,
                num_samples=128,
            )[0].numpy()

            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
            predictions.append(pred.mean())

            print(f"✓ {elapsed:.0f}ms, pred={pred.mean():.2f}")

            # Cleanup
            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"✗ Error: {e}")
            return None, None

    avg_pred = np.mean(predictions)
    std_pred = np.std(predictions)
    avg_time = np.mean(times)

    print(f"\nResults:")
    print(f"  Avg prediction: {avg_pred:.4f} ± {std_pred:.4f}")
    print(f"  Avg time: {avg_time:.0f}ms")

    return avg_pred, avg_time


def main():
    print("="*60)
    print("QUICK COMPILE ACCURACY TEST")
    print("="*60)

    # Test eager mode
    eager_pred, eager_time = test_prediction_accuracy("eager", num_iterations=3)

    if eager_pred is None:
        print("\n❌ Eager mode failed, cannot proceed")
        return

    # Test compiled mode
    compiled_pred, compiled_time = test_prediction_accuracy("compiled", num_iterations=3)

    if compiled_pred is None:
        print("\n❌ Compiled mode failed")
        print("\n🔴 RECOMMENDATION: Use EAGER mode (compile is broken)")
        print("\nConfiguration:")
        print("  export TOTO_DISABLE_COMPILE=1")
        return

    # Compare results
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)

    pred_delta = abs(compiled_pred - eager_pred)
    pred_delta_pct = (pred_delta / eager_pred * 100) if eager_pred != 0 else 0

    speedup = eager_time / compiled_time if compiled_time > 0 else 0

    print(f"\nAccuracy:")
    print(f"  Eager:    {eager_pred:.4f}")
    print(f"  Compiled: {compiled_pred:.4f}")
    print(f"  Delta:    {pred_delta:.4f} ({pred_delta_pct:.2f}%)")

    print(f"\nPerformance:")
    print(f"  Eager:    {eager_time:.0f}ms")
    print(f"  Compiled: {compiled_time:.0f}ms")
    print(f"  Speedup:  {speedup:.2f}x")

    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)

    # Decision logic
    accuracy_ok = pred_delta_pct < 1.0
    performance_better = speedup > 1.2  # Must be at least 20% faster

    print(f"\nAccuracy: {'✓' if accuracy_ok else '✗'} Delta {pred_delta_pct:.2f}% (threshold: <1%)")
    print(f"Performance: {'✓' if performance_better else '✗'} Speedup {speedup:.2f}x (threshold: >1.2x)")

    if accuracy_ok and performance_better:
        print("\n🟢 RECOMMENDATION: Use COMPILED mode")
        print("   - Accurate predictions (delta <1%)")
        print("   - Better performance")
        print("\nConfiguration:")
        print("  export TOTO_DISABLE_COMPILE=0")
        print("  export TOTO_COMPILE_MODE=max-autotune")

        # Set it
        with open(PROJECT_ROOT / ".env.compile", "w") as f:
            f.write("# Torch compile configuration - COMPILED MODE\n")
            f.write("export TOTO_DISABLE_COMPILE=0\n")
            f.write("export TOTO_COMPILE_MODE=max-autotune\n")
            f.write("export TOTO_COMPILE_BACKEND=inductor\n")
        print(f"\n✓ Saved to .env.compile")

    elif accuracy_ok and not performance_better:
        print("\n🟡 RECOMMENDATION: Use EAGER mode")
        print("   - Accurate predictions")
        print("   - But compiled is not faster (likely recompilation overhead)")
        print("\nConfiguration:")
        print("  export TOTO_DISABLE_COMPILE=1")

        # Set it
        with open(PROJECT_ROOT / ".env.compile", "w") as f:
            f.write("# Torch compile configuration - EAGER MODE\n")
            f.write("export TOTO_DISABLE_COMPILE=1\n")
        print(f"\n✓ Saved to .env.compile")

    else:
        print("\n🔴 RECOMMENDATION: Use EAGER mode")
        print("   - Compiled mode has accuracy issues")
        print("\nConfiguration:")
        print("  export TOTO_DISABLE_COMPILE=1")

        # Set it
        with open(PROJECT_ROOT / ".env.compile", "w") as f:
            f.write("# Torch compile configuration - EAGER MODE\n")
            f.write("export TOTO_DISABLE_COMPILE=1\n")
        print(f"\n✓ Saved to .env.compile")

    print("\nTo apply:")
    print("  source .env.compile")
    print("  python trade_stock_e2e.py")


if __name__ == "__main__":
    main()
