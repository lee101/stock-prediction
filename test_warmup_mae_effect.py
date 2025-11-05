#!/usr/bin/env python3
"""
Test whether warmup runs affect MAE predictions in compiled Toto.

Critical question: Does the first inference (cold start) produce different
predictions than subsequent inferences (warm start)?

If YES: Warmup is REQUIRED before production predictions
If NO: Warmup is optional (just for performance)

Test design:
1. Load compiled model (fresh)
2. Run prediction WITHOUT warmup (cold start)
3. Run same prediction again (warm, after compilation)
4. Run same prediction again (warm, stable)
5. Compare MAE across all runs

Expected result:
- If compilation is deterministic: All MAEs should be identical
- If compilation affects predictions: Cold start MAE will differ
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

# Disable logs for cleaner output
os.environ.setdefault("TORCH_LOGS", "")

print("=" * 80)
print("WARMUP MAE EFFECT TEST")
print("=" * 80)
print()
print("Testing whether warmup runs affect MAE predictions...")
print()

from src.models.toto_wrapper import TotoPipeline


def load_real_data(symbol: str, context_length: int = 512) -> torch.Tensor:
    """Load real training data."""
    csv_path = Path("trainingdata") / f"{symbol}.csv"
    df = pd.read_csv(csv_path)

    if 'close' in df.columns:
        prices = df['close'].values
    elif 'Close' in df.columns:
        prices = df['Close'].values
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        prices = df[numeric_cols[-1]].values

    if len(prices) >= context_length:
        context = prices[-context_length:]
    else:
        context = np.pad(prices, (context_length - len(prices), 0), mode='mean')

    return torch.from_numpy(context.astype(np.float32)).float()


def reset_cuda():
    """Reset CUDA state."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def test_warmup_effect(
    symbol: str,
    compile_mode: str,
    num_sequential_runs: int = 5,
) -> Tuple[List[float], List[np.ndarray]]:
    """
    Test warmup effect by running predictions sequentially without warmup.

    Returns:
        maes: List of MAE values for each run
        all_samples: List of prediction arrays for each run
    """
    print(f"\nTesting {symbol} with {compile_mode} mode")
    print(f"Running {num_sequential_runs} sequential predictions...")
    print()

    # Load data once
    context = load_real_data(symbol, context_length=512)

    # Load pipeline fresh (no warmup)
    reset_cuda()

    pipeline = TotoPipeline.from_pretrained(
        model_id="Datadog/Toto-Open-Base-1.0",
        device_map="cuda",
        torch_dtype=torch.float32,
        torch_compile=True,
        compile_mode=compile_mode,
        compile_backend="inductor",
        warmup_sequence=0,  # NO WARMUP
        cache_policy="prefer",
    )

    print(f"Pipeline loaded (torch_compile={pipeline.compiled})")
    print()

    maes = []
    all_samples = []

    # Run predictions sequentially WITHOUT any warmup
    for run in range(num_sequential_runs):
        print(f"Run {run + 1}/{num_sequential_runs} (no warmup)...", end=" ", flush=True)

        # Use SAME context every time
        forecasts = pipeline.predict(
            context=context,
            prediction_length=8,
            num_samples=256,
            samples_per_batch=128,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Extract samples
        samples = forecasts[0].numpy()
        mae = np.mean(np.abs(samples))
        mean_pred = np.mean(samples)

        maes.append(mae)
        all_samples.append(samples)

        print(f"MAE={mae:.6f}, Mean={mean_pred:.4f}")

    # Clean up
    pipeline.unload()
    del pipeline
    reset_cuda()

    return maes, all_samples


def analyze_warmup_effect(maes: List[float], all_samples: List[np.ndarray], symbol: str):
    """Analyze whether warmup affects predictions."""
    print()
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print()

    # MAE statistics
    mae_array = np.array(maes)
    mae_mean = np.mean(mae_array)
    mae_std = np.std(mae_array)
    mae_min = np.min(mae_array)
    mae_max = np.max(mae_array)
    mae_range = mae_max - mae_min

    print(f"Symbol: {symbol}")
    print(f"MAE across {len(maes)} runs:")
    print(f"  Mean:  {mae_mean:.6f}")
    print(f"  Std:   {mae_std:.6e}")
    print(f"  Min:   {mae_min:.6f}")
    print(f"  Max:   {mae_max:.6f}")
    print(f"  Range: {mae_range:.6e}")
    print()

    # Compare cold start (run 1) vs warm runs (runs 2-5)
    cold_start_mae = maes[0]
    warm_maes = maes[1:]
    warm_mae_mean = np.mean(warm_maes)
    warm_mae_std = np.std(warm_maes)

    cold_vs_warm_diff = abs(cold_start_mae - warm_mae_mean)
    cold_vs_warm_pct = (cold_vs_warm_diff / cold_start_mae) * 100

    print("Cold Start vs Warm Runs:")
    print(f"  Cold start (run 1):      {cold_start_mae:.6f}")
    print(f"  Warm mean (runs 2-{len(maes)}):    {warm_mae_mean:.6f} ± {warm_mae_std:.6e}")
    print(f"  Difference:              {cold_vs_warm_diff:.6e} ({cold_vs_warm_pct:.4f}%)")
    print()

    # Sample-level comparison (cold vs warm)
    cold_samples = all_samples[0]
    warm_samples = all_samples[1]  # Second run (first warm)

    sample_mae_diff = np.mean(np.abs(cold_samples - warm_samples))
    sample_correlation = np.corrcoef(cold_samples.flatten(), warm_samples.flatten())[0, 1]

    print("Sample-level comparison (run 1 vs run 2):")
    print(f"  MAE difference:  {sample_mae_diff:.6e}")
    print(f"  Correlation:     {sample_correlation:.6f}")
    print()

    # Consecutive run stability (warm runs only)
    if len(all_samples) >= 3:
        consecutive_diffs = []
        for i in range(1, len(all_samples) - 1):
            diff = np.mean(np.abs(all_samples[i] - all_samples[i + 1]))
            consecutive_diffs.append(diff)

        consecutive_mean = np.mean(consecutive_diffs)
        consecutive_std = np.std(consecutive_diffs)

        print("Consecutive run stability (warm runs):")
        print(f"  Mean difference: {consecutive_mean:.6e} ± {consecutive_std:.6e}")
        print()

    # Verdict
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)
    print()

    # Tolerance for "significant" difference
    tolerance_pct = 0.1  # 0.1% tolerance

    if cold_vs_warm_pct > tolerance_pct:
        print(f"⚠️  WARMUP REQUIRED")
        print()
        print(f"   Cold start MAE differs by {cold_vs_warm_pct:.4f}% from warm runs.")
        print(f"   This exceeds tolerance of {tolerance_pct}%.")
        print()
        print("   RECOMMENDATION:")
        print("   - MUST run 2-3 warmup inferences before production predictions")
        print("   - Warmup affects prediction accuracy, not just performance")
        print()
        return False  # Warmup required
    else:
        print(f"✅ WARMUP OPTIONAL (for performance only)")
        print()
        print(f"   Cold start MAE differs by only {cold_vs_warm_pct:.4f}% from warm runs.")
        print(f"   This is within tolerance of {tolerance_pct}%.")
        print()
        print("   RECOMMENDATION:")
        print("   - Warmup recommended for performance but not required for accuracy")
        print("   - First inference may be slower but produces correct predictions")
        print()
        return True  # Warmup optional


def main():
    # Test symbols
    test_cases = [
        ("BTCUSD", "reduce-overhead"),
        ("ETHUSD", "reduce-overhead"),
        ("AAPL", "default"),
    ]

    results = {}

    for symbol, compile_mode in test_cases:
        print()
        print("=" * 80)
        print(f"TESTING: {symbol} ({compile_mode} mode)")
        print("=" * 80)

        maes, all_samples = test_warmup_effect(
            symbol=symbol,
            compile_mode=compile_mode,
            num_sequential_runs=5,
        )

        warmup_optional = analyze_warmup_effect(maes, all_samples, symbol)
        results[symbol] = {
            "compile_mode": compile_mode,
            "maes": maes,
            "warmup_optional": warmup_optional,
        }

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    summary_df = pd.DataFrame([
        {
            "Symbol": symbol,
            "Mode": data["compile_mode"],
            "Cold Start MAE": f"{data['maes'][0]:.6f}",
            "Warm Mean MAE": f"{np.mean(data['maes'][1:]):.6f}",
            "Difference %": f"{(abs(data['maes'][0] - np.mean(data['maes'][1:])) / data['maes'][0] * 100):.4f}%",
            "Warmup Optional": "✓" if data["warmup_optional"] else "✗ REQUIRED",
        }
        for symbol, data in results.items()
    ])

    print(summary_df.to_string(index=False))
    print()

    # Final recommendation
    all_optional = all(data["warmup_optional"] for data in results.values())

    print("=" * 80)
    print("FINAL RECOMMENDATION")
    print("=" * 80)
    print()

    if all_optional:
        print("✅ Warmup is OPTIONAL for all tested symbols")
        print()
        print("   Cold start predictions are accurate.")
        print("   Warmup improves performance but doesn't affect MAE.")
        print()
        print("   For production:")
        print("   - Warmup recommended (2-3 runs) for best performance")
        print("   - Can skip warmup if immediate prediction needed")
    else:
        print("⚠️  Warmup is REQUIRED for some symbols")
        print()
        print("   Cold start predictions differ from warm predictions.")
        print()
        print("   For production:")
        print("   - MUST run 2-3 warmup inferences before real predictions")
        print("   - Add warmup to startup sequence")
        print()
        print("   Required symbols:")
        for symbol, data in results.items():
            if not data["warmup_optional"]:
                print(f"     - {symbol} ({data['compile_mode']} mode)")

    print()

    return 0 if all_optional else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
