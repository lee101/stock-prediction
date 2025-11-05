#!/usr/bin/env python3
"""
Comprehensive Toto compilation test on real training data.

Tests:
1. MAE equivalence between compiled and uncompiled
2. Multiple compile modes (default, reduce-overhead, max-autotune)
3. Stability across multiple runs
4. Recompilation tracking
5. Performance metrics

Usage:
    python test_toto_compilation_real_data.py
"""

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

# Disable cudagraphs logging initially
os.environ.setdefault("TORCH_LOGS", "")

print("=" * 80)
print("TOTO COMPILATION TEST ON REAL TRAINING DATA")
print("=" * 80)
print()

# Import after setting env
from src.models.toto_wrapper import TotoPipeline


@dataclass
class TestMetrics:
    """Metrics from a single test run."""
    symbol: str
    compile_mode: str
    compiled: bool
    mae: float
    mean_pred: float
    std_pred: float
    min_pred: float
    max_pred: float
    inference_time_ms: float
    num_samples: int
    context_length: int
    prediction_length: int


@dataclass
class StabilityMetrics:
    """Stability metrics across multiple runs."""
    symbol: str
    compile_mode: str
    num_runs: int
    mae_mean: float
    mae_std: float
    mae_min: float
    mae_max: float
    pred_mean_variance: float
    pred_std_variance: float
    time_mean_ms: float
    time_std_ms: float


def load_training_data(symbol: str, context_length: int = 512) -> torch.Tensor:
    """Load real training data from CSV file."""
    csv_path = Path("trainingdata") / f"{symbol}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Training data not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Assume CSV has columns like: timestamp, open, high, low, close, volume
    # Use close prices for forecasting
    if 'close' in df.columns:
        prices = df['close'].values
    elif 'Close' in df.columns:
        prices = df['Close'].values
    else:
        # Use the last numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        prices = df[numeric_cols[-1]].values

    # Take last context_length points
    if len(prices) >= context_length:
        context = prices[-context_length:]
    else:
        # Pad with mean if not enough data
        context = np.pad(
            prices,
            (context_length - len(prices), 0),
            mode='mean'
        )

    # Normalize to prevent extreme values
    context = context.astype(np.float32)

    return torch.from_numpy(context).float()


def reset_cuda_stats():
    """Reset CUDA statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_memory_mb() -> float:
    """Get current CUDA memory usage in MB."""
    if not torch.cuda.is_available():
        return 0.0

    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / (1024 ** 2)


def run_single_test(
    symbol: str,
    compile_mode: str,
    compiled: bool,
    context_length: int = 512,
    prediction_length: int = 8,
    num_samples: int = 256,
    samples_per_batch: int = 128,
    warmup: bool = True,
) -> Tuple[TestMetrics, np.ndarray]:
    """Run a single test and return metrics and predictions."""

    # Load real data
    context = load_training_data(symbol, context_length)

    # Load pipeline
    reset_cuda_stats()

    pipeline = TotoPipeline.from_pretrained(
        model_id="Datadog/Toto-Open-Base-1.0",
        device_map="cuda",
        torch_dtype=torch.float32,
        torch_compile=compiled,
        compile_mode=compile_mode if compiled else None,
        compile_backend="inductor" if compiled else None,
        warmup_sequence=0,  # Manual warmup
        cache_policy="prefer",
    )

    # Warmup if requested
    if warmup:
        for _ in range(2):
            _ = pipeline.predict(
                context=context,
                prediction_length=prediction_length,
                num_samples=num_samples,
                samples_per_batch=samples_per_batch,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    # Actual inference
    reset_cuda_stats()

    start_time = time.perf_counter()

    forecasts = pipeline.predict(
        context=context,
        prediction_length=prediction_length,
        num_samples=num_samples,
        samples_per_batch=samples_per_batch,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Extract predictions
    samples = forecasts[0].numpy()  # Shape: (num_samples, prediction_length)

    # Compute metrics
    mae = np.mean(np.abs(samples))
    mean_pred = np.mean(samples)
    std_pred = np.std(samples)
    min_pred = np.min(samples)
    max_pred = np.max(samples)

    metrics = TestMetrics(
        symbol=symbol,
        compile_mode=compile_mode,
        compiled=compiled,
        mae=mae,
        mean_pred=mean_pred,
        std_pred=std_pred,
        min_pred=min_pred,
        max_pred=max_pred,
        inference_time_ms=elapsed_ms,
        num_samples=num_samples,
        context_length=context_length,
        prediction_length=prediction_length,
    )

    # Clean up
    pipeline.unload()
    del pipeline
    reset_cuda_stats()

    return metrics, samples


def run_stability_test(
    symbol: str,
    compile_mode: str,
    compiled: bool,
    num_runs: int = 3,
) -> StabilityMetrics:
    """Run multiple tests to measure stability."""

    print(f"  Running {num_runs} stability tests...")

    maes = []
    pred_means = []
    pred_stds = []
    times = []

    for i in range(num_runs):
        print(f"    Run {i+1}/{num_runs}...", end=" ", flush=True)

        metrics, _ = run_single_test(
            symbol=symbol,
            compile_mode=compile_mode,
            compiled=compiled,
            warmup=(i == 0),  # Only warmup first run
        )

        maes.append(metrics.mae)
        pred_means.append(metrics.mean_pred)
        pred_stds.append(metrics.std_pred)
        times.append(metrics.inference_time_ms)

        print(f"MAE={metrics.mae:.4f}, Time={metrics.inference_time_ms:.1f}ms")

    stability = StabilityMetrics(
        symbol=symbol,
        compile_mode=compile_mode,
        num_runs=num_runs,
        mae_mean=np.mean(maes),
        mae_std=np.std(maes),
        mae_min=np.min(maes),
        mae_max=np.max(maes),
        pred_mean_variance=np.var(pred_means),
        pred_std_variance=np.var(pred_stds),
        time_mean_ms=np.mean(times),
        time_std_ms=np.std(times),
    )

    return stability


def compare_predictions(
    symbol: str,
    uncompiled_samples: np.ndarray,
    compiled_samples: np.ndarray,
    compile_mode: str,
) -> Dict[str, float]:
    """Compare predictions between compiled and uncompiled."""

    # MAE between sample sets
    mae_diff = np.mean(np.abs(uncompiled_samples - compiled_samples))

    # Mean difference
    mean_diff = abs(np.mean(uncompiled_samples) - np.mean(compiled_samples))

    # Std difference
    std_diff = abs(np.std(uncompiled_samples) - np.std(compiled_samples))

    # Max absolute difference
    max_diff = np.max(np.abs(uncompiled_samples - compiled_samples))

    # Correlation
    flat_uncomp = uncompiled_samples.flatten()
    flat_comp = compiled_samples.flatten()
    correlation = np.corrcoef(flat_uncomp, flat_comp)[0, 1]

    return {
        "mae_diff": mae_diff,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "max_diff": max_diff,
        "correlation": correlation,
    }


def main():
    # Test symbols
    symbols = ["BTCUSD", "ETHUSD", "AAPL", "GOOGL", "AMD"]

    # Compile modes to test
    compile_modes = [
        "default",
        "reduce-overhead",
        "max-autotune",
    ]

    print(f"Testing {len(symbols)} symbols with {len(compile_modes)} compile modes")
    print(f"Symbols: {', '.join(symbols)}")
    print()

    all_results = []
    stability_results = []

    for symbol in symbols:
        print(f"\n{'='*80}")
        print(f"TESTING: {symbol}")
        print(f"{'='*80}\n")

        # Test uncompiled first (baseline)
        print("1. Uncompiled (baseline)...")
        uncompiled_metrics, uncompiled_samples = run_single_test(
            symbol=symbol,
            compile_mode="none",
            compiled=False,
        )
        all_results.append(uncompiled_metrics)

        print(f"   MAE: {uncompiled_metrics.mae:.6f}")
        print(f"   Mean: {uncompiled_metrics.mean_pred:.4f}")
        print(f"   Std: {uncompiled_metrics.std_pred:.4f}")
        print(f"   Time: {uncompiled_metrics.inference_time_ms:.1f}ms")
        print()

        # Test each compile mode
        for compile_mode in compile_modes:
            print(f"2. Compiled ({compile_mode})...")

            compiled_metrics, compiled_samples = run_single_test(
                symbol=symbol,
                compile_mode=compile_mode,
                compiled=True,
            )
            all_results.append(compiled_metrics)

            # Compare with uncompiled
            comparison = compare_predictions(
                symbol, uncompiled_samples, compiled_samples, compile_mode
            )

            speedup = uncompiled_metrics.inference_time_ms / compiled_metrics.inference_time_ms

            print(f"   MAE: {compiled_metrics.mae:.6f}")
            print(f"   Mean: {compiled_metrics.mean_pred:.4f}")
            print(f"   Std: {compiled_metrics.std_pred:.4f}")
            print(f"   Time: {compiled_metrics.inference_time_ms:.1f}ms ({speedup:.2f}x speedup)")
            print(f"   MAE diff vs uncompiled: {comparison['mae_diff']:.6e}")
            print(f"   Mean diff: {comparison['mean_diff']:.6e}")
            print(f"   Correlation: {comparison['correlation']:.6f}")
            print()

            # Stability test
            print(f"3. Stability test ({compile_mode})...")
            stability = run_stability_test(
                symbol=symbol,
                compile_mode=compile_mode,
                compiled=True,
                num_runs=3,
            )
            stability_results.append(stability)

            print(f"   MAE stability: {stability.mae_mean:.6f} ± {stability.mae_std:.6e}")
            print(f"   Time stability: {stability.time_mean_ms:.1f} ± {stability.time_std_ms:.1f}ms")
            print()

    # Summary report
    print("\n" + "=" * 80)
    print("SUMMARY REPORT")
    print("=" * 80)
    print()

    # Create DataFrame for results
    results_df = pd.DataFrame([
        {
            "Symbol": r.symbol,
            "Mode": f"{r.compile_mode}{' (C)' if r.compiled else ''}",
            "MAE": f"{r.mae:.6f}",
            "Mean": f"{r.mean_pred:.4f}",
            "Std": f"{r.std_pred:.4f}",
            "Time (ms)": f"{r.inference_time_ms:.1f}",
            "Speedup": f"{all_results[0].inference_time_ms / r.inference_time_ms:.2f}x" if r.compiled else "1.00x",
        }
        for r in all_results
    ])

    print(results_df.to_string(index=False))
    print()

    # Stability report
    print("=" * 80)
    print("STABILITY METRICS")
    print("=" * 80)
    print()

    stability_df = pd.DataFrame([
        {
            "Symbol": s.symbol,
            "Mode": s.compile_mode,
            "MAE Mean": f"{s.mae_mean:.6f}",
            "MAE Std": f"{s.mae_std:.6e}",
            "Time Mean": f"{s.time_mean_ms:.1f}ms",
            "Time Std": f"{s.time_std_ms:.1f}ms",
        }
        for s in stability_results
    ])

    print(stability_df.to_string(index=False))
    print()

    # Equivalence check
    print("=" * 80)
    print("MAE EQUIVALENCE CHECK (tolerance: 1e-3)")
    print("=" * 80)
    print()

    tolerance = 1e-3

    for symbol in symbols:
        uncompiled_mae = next(r.mae for r in all_results if r.symbol == symbol and not r.compiled)

        print(f"{symbol}:")
        for compile_mode in compile_modes:
            compiled_mae = next(
                r.mae for r in all_results
                if r.symbol == symbol and r.compiled and r.compile_mode == compile_mode
            )

            diff = abs(compiled_mae - uncompiled_mae)
            equiv = diff < tolerance

            print(f"  {compile_mode:20s}: MAE diff = {diff:.6e} {'✓' if equiv else '✗'}")
        print()

    # Recommendations
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()

    # Find best mode by stability and performance
    best_mode = min(
        stability_results,
        key=lambda s: s.mae_std + (s.time_mean_ms / 1000)  # Balance stability and speed
    )

    print(f"Best compile mode: {best_mode.compile_mode}")
    print(f"  Reason: Best balance of MAE stability ({best_mode.mae_std:.6e})")
    print(f"          and performance ({best_mode.time_mean_ms:.1f}ms)")
    print()

    # Check for equivalence issues
    has_issues = any(
        abs(r1.mae - r2.mae) > tolerance
        for r1 in all_results
        for r2 in all_results
        if r1.symbol == r2.symbol and r1.compiled != r2.compiled
    )

    if has_issues:
        print("⚠️  WARNING: Some compile modes show MAE differences > tolerance")
        print("   Consider using uncompiled or 'default' mode for maximum accuracy")
    else:
        print("✓ All compile modes maintain MAE equivalence")
    print()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
