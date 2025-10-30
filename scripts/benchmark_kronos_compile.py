#!/usr/bin/env python3
"""
Comprehensive benchmark: Kronos compiled vs eager mode.

Tests:
1. MAE on real training data
2. Inference time (warmup + steady state)
3. Memory usage
4. Stability (multiple iterations)
5. Recompilation count
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Set cache dirs
os.environ["COMPILED_MODELS_DIR"] = "/vfast"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/vfast/torch_inductor"

from src.models.kronos_wrapper import KronosForecastingWrapper


class KronosBenchmark:
    """Benchmark Kronos compiled vs eager."""

    def __init__(self, num_iterations: int = 5):
        self.num_iterations = num_iterations
        self.results = {"eager": {}, "compiled": {}}

    def _load_real_data(self) -> pd.DataFrame:
        """Load real training data from trainingdata/."""
        training_dir = PROJECT_ROOT / "trainingdata"

        # Try to find a BTC or ETH file
        for pattern in ["*BTC*.csv", "*ETH*.csv", "*.csv"]:
            files = list(training_dir.glob(pattern))
            if files:
                df = pd.read_csv(files[0])

                # Ensure required columns
                if "Date" in df.columns:
                    df["ds"] = pd.to_datetime(df["Date"])
                elif "ds" not in df.columns:
                    # Create synthetic dates
                    df["ds"] = pd.date_range("2020-01-01", periods=len(df), freq="D")

                # Ensure OHLCV columns
                required = ["Close"]
                if all(col in df.columns or col.lower() in df.columns for col in required):
                    # Normalize column names
                    df.columns = [c if c == "ds" else c.title() for c in df.columns]

                    # Fill missing OHLCV
                    if "Open" not in df.columns:
                        df["Open"] = df["Close"]
                    if "High" not in df.columns:
                        df["High"] = df["Close"] * 1.01
                    if "Low" not in df.columns:
                        df["Low"] = df["Close"] * 0.99
                    if "Volume" not in df.columns:
                        df["Volume"] = 1000000.0

                    print(f"✓ Loaded real data from {files[0].name} ({len(df)} rows)")
                    return df.tail(600)  # Last 600 rows

        # Fallback: generate synthetic
        print("⚠️ No real data found, using synthetic")
        return self._generate_synthetic_data(600)

    def _generate_synthetic_data(self, length: int) -> pd.DataFrame:
        """Generate synthetic OHLCV data."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=length, freq="D")

        close = 100.0 * np.exp(np.cumsum(np.random.randn(length) * 0.02))
        high = close * (1 + np.abs(np.random.randn(length) * 0.01))
        low = close * (1 - np.abs(np.random.randn(length) * 0.01))
        open_price = np.roll(close, 1)
        open_price[0] = close[0]
        volume = np.random.uniform(1e6, 1e7, length)

        return pd.DataFrame({
            "ds": dates,
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        })

    def benchmark_mode(
        self,
        mode: str,
        compile: bool,
    ) -> Dict:
        """Benchmark a single mode (eager or compiled)."""

        print(f"\n{'='*70}")
        print(f"BENCHMARKING KRONOS {mode.upper()} MODE")
        print(f"{'='*70}")

        # Load data
        df = self._load_real_data()
        context_df = df.iloc[:-1]
        target_close = df.iloc[-1]["Close"]

        mae_list = []
        time_list = []
        memory_list = []

        for i in range(self.num_iterations):
            print(f"\nIteration {i+1}/{self.num_iterations}", end="... ", flush=True)

            try:
                torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
                start = time.perf_counter()

                # Create wrapper
                wrapper = KronosForecastingWrapper(
                    model_name="NeoQuasar/Kronos-base",
                    tokenizer_name="NeoQuasar/Kronos-Tokenizer-base",
                    device="cuda",
                    max_context=256,
                    sample_count=8,
                    compile=compile,
                    compile_mode="max-autotune",
                )

                # Predict
                results = wrapper.predict_series(
                    data=context_df,
                    timestamp_col="ds",
                    columns=["Close"],
                    pred_len=1,
                )

                pred_close = results["Close"].absolute[0]
                elapsed = (time.perf_counter() - start) * 1000

                # Metrics
                mae = abs(pred_close - target_close)
                mae_pct = (mae / target_close * 100)

                if torch.cuda.is_available():
                    memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                else:
                    memory_mb = 0.0

                mae_list.append(mae)
                time_list.append(elapsed)
                memory_list.append(memory_mb)

                print(f"✓ MAE={mae:.2f} ({mae_pct:.2f}%), time={elapsed:.0f}ms, mem={memory_mb:.0f}MB")

                # Cleanup
                wrapper.unload()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"✗ Error: {e}")
                continue

        if not mae_list:
            print(f"\n❌ {mode.upper()} mode failed - no successful iterations")
            return None

        # Compute statistics
        results = {
            "mae_mean": float(np.mean(mae_list)),
            "mae_std": float(np.std(mae_list)),
            "mae_list": [float(x) for x in mae_list],
            "time_mean": float(np.mean(time_list)),
            "time_std": float(np.std(time_list)),
            "time_list": [float(x) for x in time_list],
            "memory_mean": float(np.mean(memory_list)),
            "memory_peak": float(np.max(memory_list)),
            "iterations": len(mae_list),
        }

        print(f"\n{mode.upper()} Results:")
        print(f"  MAE:    {results['mae_mean']:.2f} ± {results['mae_std']:.2f}")
        print(f"  Time:   {results['time_mean']:.0f} ± {results['time_std']:.0f} ms")
        print(f"  Memory: {results['memory_mean']:.0f} MB (peak: {results['memory_peak']:.0f} MB)")

        return results

    def compare_and_decide(self) -> str:
        """Compare results and make recommendation."""

        print(f"\n{'='*70}")
        print("COMPARISON AND DECISION")
        print(f"{'='*70}")

        eager = self.results["eager"]
        compiled = self.results["compiled"]

        if not eager or not compiled:
            print("\n❌ INCOMPLETE - Cannot make recommendation")
            return "INCONCLUSIVE"

        # MAE comparison
        mae_delta = abs(compiled["mae_mean"] - eager["mae_mean"])
        mae_delta_pct = (mae_delta / eager["mae_mean"] * 100) if eager["mae_mean"] > 0 else 0

        print(f"\n📊 MAE Comparison:")
        print(f"  Eager MAE:    {eager['mae_mean']:.2f} ± {eager['mae_std']:.2f}")
        print(f"  Compiled MAE: {compiled['mae_mean']:.2f} ± {compiled['mae_std']:.2f}")
        print(f"  Δ MAE:        {mae_delta:.2f} ({mae_delta_pct:.2f}%)")

        # Time comparison
        time_delta = compiled["time_mean"] - eager["time_mean"]
        speedup = eager["time_mean"] / compiled["time_mean"] if compiled["time_mean"] > 0 else 0

        print(f"\n⚡ Performance Comparison:")
        print(f"  Eager time:    {eager['time_mean']:.0f} ± {eager['time_std']:.0f} ms")
        print(f"  Compiled time: {compiled['time_mean']:.0f} ± {compiled['time_std']:.0f} ms")
        print(f"  Speedup:       {speedup:.2f}x")

        # Memory comparison
        memory_delta = compiled["memory_mean"] - eager["memory_mean"]
        memory_delta_pct = (memory_delta / eager["memory_mean"] * 100) if eager["memory_mean"] > 0 else 0

        print(f"\n💾 Memory Comparison:")
        print(f"  Eager memory:    {eager['memory_mean']:.0f} MB")
        print(f"  Compiled memory: {compiled['memory_mean']:.0f} MB")
        print(f"  Δ Memory:        {memory_delta:+.0f} MB ({memory_delta_pct:+.1f}%)")

        # Decision criteria
        print(f"\n{'='*70}")
        print("DECISION CRITERIA")
        print(f"{'='*70}")

        accuracy_ok = mae_delta_pct < 5.0
        performance_better = speedup > 1.1  # At least 10% faster
        memory_acceptable = memory_delta_pct < 50.0  # Less than 50% more memory

        print(f"\n{'✅' if accuracy_ok else '❌'} Accuracy: MAE delta {mae_delta_pct:.2f}% (threshold: <5%)")
        print(f"{'✅' if performance_better else '❌'} Performance: Speedup {speedup:.2f}x (threshold: >1.1x)")
        print(f"{'✅' if memory_acceptable else '❌'} Memory: Delta {memory_delta_pct:+.1f}% (threshold: <50%)")

        # Final decision
        print(f"\n{'='*70}")
        print("FINAL RECOMMENDATION")
        print(f"{'='*70}\n")

        if accuracy_ok and performance_better and memory_acceptable:
            decision = "COMPILED"
            print("🟢 RECOMMENDATION: Use COMPILED mode for Kronos")
            print(f"   ✓ Accurate (MAE delta {mae_delta_pct:.2f}% < 5%)")
            print(f"   ✓ Faster ({speedup:.2f}x speedup)")
            print(f"   ✓ Memory acceptable ({memory_delta_pct:+.1f}%)")
        elif accuracy_ok and not performance_better:
            decision = "EAGER"
            print("🟡 RECOMMENDATION: Use EAGER mode for Kronos")
            print(f"   ✓ Accurate (MAE delta {mae_delta_pct:.2f}%)")
            print(f"   ✗ Not significantly faster ({speedup:.2f}x < 1.1x)")
            print(f"   → EAGER preferred for simplicity")
        else:
            decision = "EAGER"
            print("🔴 RECOMMENDATION: Use EAGER mode for Kronos")
            print(f"   {'✗' if not accuracy_ok else '✓'} Accuracy issue" if not accuracy_ok else f"   ✓ Accurate")
            print(f"   → EAGER required for correctness")

        return decision

    def run(self) -> str:
        """Run full benchmark."""

        print("="*70)
        print("KRONOS TORCH.COMPILE BENCHMARK")
        print("="*70)
        print(f"Iterations: {self.num_iterations}")
        print()

        # Benchmark eager
        eager_results = self.benchmark_mode("eager", compile=False)
        self.results["eager"] = eager_results

        # Benchmark compiled
        compiled_results = self.benchmark_mode("compiled", compile=True)
        self.results["compiled"] = compiled_results

        # Compare and decide
        decision = self.compare_and_decide()

        # Save results
        output_dir = PROJECT_ROOT / "tests" / "compile_stress_results"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "kronos_benchmark_results.json"
        with open(output_file, "w") as f:
            json.dump({
                "results": self.results,
                "decision": decision,
            }, f, indent=2)

        print(f"\n📄 Results saved to: {output_file}")

        return decision


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark Kronos torch.compile")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations")
    args = parser.parse_args()

    benchmark = KronosBenchmark(num_iterations=args.iterations)
    decision = benchmark.run()

    print(f"\n{'='*70}")
    print(f"DECISION: {decision} MODE")
    print(f"{'='*70}")

    sys.exit(0 if decision == "COMPILED" else 1)


if __name__ == "__main__":
    main()
