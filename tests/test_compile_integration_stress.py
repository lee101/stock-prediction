#!/usr/bin/env python3
"""
Integration stress test for torch.compile reliability in production.

This test suite validates:
1. Compiled vs non-compiled Toto model accuracy (MAE, predictions)
2. Compiled vs non-compiled Kronos model accuracy (if applicable)
3. Performance metrics (inference time, memory usage)
4. Recompilation behavior under varied inputs
5. Multi-iteration stability
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.kronos_wrapper import KronosForecastingWrapper
from src.models.toto_wrapper import TotoPipeline


@dataclass
class AccuracyMetrics:
    """Accuracy metrics for a single test run."""
    mae: float
    rmse: float
    mape: float  # Mean Absolute Percentage Error
    prediction_mean: float
    prediction_std: float
    target_mean: float


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single test run."""
    inference_time_ms: float
    peak_memory_mb: float
    recompilations: int  # Number of recompilations detected


@dataclass
class TestResult:
    """Combined result for a single test configuration."""
    model_name: str
    compile_mode: str  # "compiled", "eager", or compile mode like "max-autotune"
    accuracy: AccuracyMetrics
    performance: PerformanceMetrics
    iteration: int


class CompileStressTestRunner:
    """Runner for compile integration stress tests."""

    def __init__(
        self,
        *,
        device: str = "cuda",
        num_iterations: int = 5,
        context_length: int = 512,
        pred_length: int = 1,
        num_samples: int = 128,
        output_dir: Optional[Path] = None,
    ):
        self.device = device
        self.num_iterations = num_iterations
        self.context_length = context_length
        self.pred_length = pred_length
        self.num_samples = num_samples
        self.output_dir = output_dir or (PROJECT_ROOT / "tests" / "compile_stress_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _generate_synthetic_series(self, length: int, seed: int = 42) -> np.ndarray:
        """Generate synthetic stock price series with trend and noise."""
        np.random.seed(seed)
        trend = np.linspace(100, 110, length)
        noise = np.random.normal(0, 2, length)
        return trend + noise

    def _compute_accuracy_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> AccuracyMetrics:
        """Compute accuracy metrics comparing predictions to targets."""
        mae = float(np.mean(np.abs(predictions - targets)))
        rmse = float(np.sqrt(np.mean((predictions - targets) ** 2)))

        # MAPE - avoid division by zero
        nonzero_mask = targets != 0
        if nonzero_mask.any():
            mape = float(np.mean(np.abs((predictions[nonzero_mask] - targets[nonzero_mask]) / targets[nonzero_mask])) * 100)
        else:
            mape = 0.0

        return AccuracyMetrics(
            mae=mae,
            rmse=rmse,
            mape=mape,
            prediction_mean=float(np.mean(predictions)),
            prediction_std=float(np.std(predictions)),
            target_mean=float(np.mean(targets)),
        )

    def _measure_memory_usage(self) -> float:
        """Measure current GPU memory usage in MB."""
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated() / 1024 / 1024
        return 0.0

    def _detect_recompilations(self) -> int:
        """
        Attempt to detect torch.compile recompilations.
        This is a heuristic based on torch._dynamo stats if available.
        """
        try:
            if hasattr(torch, "_dynamo"):
                # Try to get recompile count from dynamo
                stats = getattr(torch._dynamo.utils, "counters", None)
                if stats:
                    frames = stats.get("frames", {})
                    return frames.get("total_recompilations", 0)
        except Exception:
            pass
        return 0

    def test_toto_compiled_vs_eager(
        self,
        test_series: np.ndarray,
        targets: np.ndarray,
    ) -> tuple[List[TestResult], List[TestResult]]:
        """
        Test Toto model in both compiled and eager modes.

        Returns:
            (compiled_results, eager_results)
        """
        compiled_results = []
        eager_results = []

        # Test compiled mode
        print("\n=== Testing Toto COMPILED mode ===")
        for iteration in range(self.num_iterations):
            print(f"Iteration {iteration + 1}/{self.num_iterations} (compiled)")

            # Force new pipeline each iteration to test recompilation
            torch.manual_seed(42 + iteration)
            torch.cuda.reset_peak_memory_stats()

            start_time = time.perf_counter()
            pipeline = TotoPipeline.from_pretrained(
                "Datadog/Toto-Open-Base-1.0",
                device_map=self.device,
                torch_compile=True,
                compile_mode="max-autotune",
                compile_backend="inductor",
            )

            predictions = pipeline.predict(
                context=test_series,
                prediction_length=self.pred_length,
                num_samples=self.num_samples,
            )[0].numpy()

            torch.cuda.synchronize() if self.device.startswith("cuda") else None
            inference_time = (time.perf_counter() - start_time) * 1000

            pred_mean = predictions.mean()
            accuracy = self._compute_accuracy_metrics(
                np.array([pred_mean]),
                targets,
            )

            performance = PerformanceMetrics(
                inference_time_ms=inference_time,
                peak_memory_mb=self._measure_memory_usage(),
                recompilations=self._detect_recompilations(),
            )

            compiled_results.append(TestResult(
                model_name="Toto",
                compile_mode="max-autotune",
                accuracy=accuracy,
                performance=performance,
                iteration=iteration,
            ))

            # Clean up
            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Test eager mode
        print("\n=== Testing Toto EAGER mode ===")
        for iteration in range(self.num_iterations):
            print(f"Iteration {iteration + 1}/{self.num_iterations} (eager)")

            torch.manual_seed(42 + iteration)
            torch.cuda.reset_peak_memory_stats()

            start_time = time.perf_counter()
            pipeline = TotoPipeline.from_pretrained(
                "Datadog/Toto-Open-Base-1.0",
                device_map=self.device,
                torch_compile=False,
            )

            predictions = pipeline.predict(
                context=test_series,
                prediction_length=self.pred_length,
                num_samples=self.num_samples,
            )[0].numpy()

            torch.cuda.synchronize() if self.device.startswith("cuda") else None
            inference_time = (time.perf_counter() - start_time) * 1000

            pred_mean = predictions.mean()
            accuracy = self._compute_accuracy_metrics(
                np.array([pred_mean]),
                targets,
            )

            performance = PerformanceMetrics(
                inference_time_ms=inference_time,
                peak_memory_mb=self._measure_memory_usage(),
                recompilations=0,  # No recompilations in eager mode
            )

            eager_results.append(TestResult(
                model_name="Toto",
                compile_mode="eager",
                accuracy=accuracy,
                performance=performance,
                iteration=iteration,
            ))

            # Clean up
            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return compiled_results, eager_results

    def test_kronos_compiled_vs_eager(
        self,
        test_df: pd.DataFrame,
        targets: np.ndarray,
    ) -> tuple[List[TestResult], List[TestResult]]:
        """
        Test Kronos model (currently only supports eager mode).

        Returns:
            (compiled_results, eager_results) - compiled will be empty for now
        """
        compiled_results = []
        eager_results = []

        print("\n=== Testing Kronos EAGER mode ===")
        for iteration in range(self.num_iterations):
            print(f"Iteration {iteration + 1}/{self.num_iterations} (kronos eager)")

            torch.cuda.reset_peak_memory_stats()

            start_time = time.perf_counter()
            wrapper = KronosForecastingWrapper(
                model_name="NeoQuasar/Kronos-base",
                tokenizer_name="NeoQuasar/Kronos-Tokenizer-base",
                device=self.device,
                max_context=self.context_length,
                sample_count=self.num_samples,
            )

            results = wrapper.predict_series(
                data=test_df,
                timestamp_col="ds",
                columns=["Close"],
                pred_len=self.pred_length,
            )

            predictions = results["Close"].absolute
            inference_time = (time.perf_counter() - start_time) * 1000

            accuracy = self._compute_accuracy_metrics(predictions, targets)

            performance = PerformanceMetrics(
                inference_time_ms=inference_time,
                peak_memory_mb=self._measure_memory_usage(),
                recompilations=0,
            )

            eager_results.append(TestResult(
                model_name="Kronos",
                compile_mode="eager",
                accuracy=accuracy,
                performance=performance,
                iteration=iteration,
            ))

            # Clean up
            wrapper.unload()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return compiled_results, eager_results

    def save_results(
        self,
        all_results: List[TestResult],
        filename: str = "compile_stress_test_results.json",
    ) -> None:
        """Save test results to JSON file."""
        output_path = self.output_dir / filename

        # Convert to serializable format
        serializable_results = []
        for result in all_results:
            serializable_results.append({
                "model_name": result.model_name,
                "compile_mode": result.compile_mode,
                "iteration": result.iteration,
                "accuracy": asdict(result.accuracy),
                "performance": asdict(result.performance),
            })

        with open(output_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        print(f"\nResults saved to {output_path}")

    def generate_report(
        self,
        all_results: List[TestResult],
        filename: str = "compile_stress_test_report.md",
    ) -> None:
        """Generate a markdown report comparing compiled vs eager modes."""
        output_path = self.output_dir / filename

        lines = ["# Compile Integration Stress Test Report", ""]
        lines.append(f"**Test Configuration:**")
        lines.append(f"- Device: {self.device}")
        lines.append(f"- Iterations: {self.num_iterations}")
        lines.append(f"- Context Length: {self.context_length}")
        lines.append(f"- Prediction Length: {self.pred_length}")
        lines.append(f"- Num Samples: {self.num_samples}")
        lines.append("")

        # Group results by model and compile mode
        grouped: Dict[str, Dict[str, List[TestResult]]] = {}
        for result in all_results:
            if result.model_name not in grouped:
                grouped[result.model_name] = {}
            if result.compile_mode not in grouped[result.model_name]:
                grouped[result.model_name][result.compile_mode] = []
            grouped[result.model_name][result.compile_mode].append(result)

        for model_name, modes in grouped.items():
            lines.append(f"## {model_name} Model")
            lines.append("")

            # Accuracy comparison
            lines.append("### Accuracy Metrics")
            lines.append("")
            lines.append("| Compile Mode | MAE (avg) | RMSE (avg) | MAPE (avg) | Prediction Mean | Std Dev |")
            lines.append("|--------------|-----------|------------|------------|-----------------|---------|")

            for mode, results in modes.items():
                avg_mae = np.mean([r.accuracy.mae for r in results])
                avg_rmse = np.mean([r.accuracy.rmse for r in results])
                avg_mape = np.mean([r.accuracy.mape for r in results])
                avg_pred_mean = np.mean([r.accuracy.prediction_mean for r in results])
                avg_pred_std = np.mean([r.accuracy.prediction_std for r in results])

                lines.append(
                    f"| {mode} | {avg_mae:.4f} | {avg_rmse:.4f} | {avg_mape:.2f}% | {avg_pred_mean:.2f} | {avg_pred_std:.2f} |"
                )

            lines.append("")

            # Performance comparison
            lines.append("### Performance Metrics")
            lines.append("")
            lines.append("| Compile Mode | Inference Time (ms) | Peak Memory (MB) | Recompilations |")
            lines.append("|--------------|---------------------|------------------|----------------|")

            for mode, results in modes.items():
                avg_time = np.mean([r.performance.inference_time_ms for r in results])
                avg_memory = np.mean([r.performance.peak_memory_mb for r in results])
                total_recompiles = sum([r.performance.recompilations for r in results])

                lines.append(
                    f"| {mode} | {avg_time:.2f} | {avg_memory:.2f} | {total_recompiles} |"
                )

            lines.append("")

            # Accuracy delta (if both compiled and eager exist)
            if len(modes) >= 2:
                compiled_key = next((k for k in modes.keys() if k != "eager"), None)
                if compiled_key and "eager" in modes:
                    compiled_mae = np.mean([r.accuracy.mae for r in modes[compiled_key]])
                    eager_mae = np.mean([r.accuracy.mae for r in modes["eager"]])
                    mae_delta = compiled_mae - eager_mae
                    mae_delta_pct = (mae_delta / eager_mae * 100) if eager_mae != 0 else 0

                    lines.append("### Accuracy Delta (Compiled - Eager)")
                    lines.append("")
                    lines.append(f"- MAE Delta: {mae_delta:.4f} ({mae_delta_pct:+.2f}%)")

                    if abs(mae_delta_pct) > 5.0:
                        lines.append(f"- ⚠️ **WARNING**: MAE delta exceeds 5% threshold!")
                    else:
                        lines.append(f"- ✅ MAE delta within acceptable range (<5%)")

                    lines.append("")

        # Overall recommendations
        lines.append("## Recommendations")
        lines.append("")

        # Check for issues
        issues = []
        for model_name, modes in grouped.items():
            for mode, results in modes.items():
                total_recompiles = sum([r.performance.recompilations for r in results])
                if total_recompiles > 10:
                    issues.append(f"- {model_name} {mode}: Excessive recompilations ({total_recompiles})")

                avg_time = np.mean([r.performance.inference_time_ms for r in results])
                if "compiled" in modes and "eager" in modes:
                    if mode != "eager":
                        eager_time = np.mean([r.performance.inference_time_ms for r in modes["eager"]])
                        if avg_time > eager_time:
                            issues.append(f"- {model_name} {mode}: Compiled slower than eager ({avg_time:.2f}ms vs {eager_time:.2f}ms)")

        if issues:
            lines.append("### Issues Detected")
            lines.extend(issues)
            lines.append("")
        else:
            lines.append("No major issues detected. ✅")
            lines.append("")

        with open(output_path, "w") as f:
            f.write("\n".join(lines))

        print(f"Report saved to {output_path}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.slow
def test_toto_compile_stress():
    """Integration test for Toto compiled vs eager modes."""
    runner = CompileStressTestRunner(
        device="cuda",
        num_iterations=3,  # Reduced for CI
        context_length=256,
        num_samples=64,
    )

    # Generate test data
    series = runner._generate_synthetic_series(256)
    targets = np.array([series[-1] * 1.01])  # Predict 1% increase

    compiled_results, eager_results = runner.test_toto_compiled_vs_eager(series, targets)

    all_results = compiled_results + eager_results
    runner.save_results(all_results, "toto_compile_stress_results.json")
    runner.generate_report(all_results, "toto_compile_stress_report.md")

    # Assertions
    assert len(compiled_results) == runner.num_iterations
    assert len(eager_results) == runner.num_iterations

    # Check that MAE doesn't diverge too much
    compiled_mae = np.mean([r.accuracy.mae for r in compiled_results])
    eager_mae = np.mean([r.accuracy.mae for r in eager_results])
    mae_delta_pct = abs(compiled_mae - eager_mae) / eager_mae * 100 if eager_mae != 0 else 0

    print(f"\nToto MAE Delta: {mae_delta_pct:.2f}%")
    assert mae_delta_pct < 10.0, f"MAE diverged too much: {mae_delta_pct:.2f}%"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.slow
@pytest.mark.skip(reason="Kronos requires external dependencies")
def test_kronos_stress():
    """Integration test for Kronos eager mode."""
    runner = CompileStressTestRunner(
        device="cuda",
        num_iterations=3,
        context_length=256,
        num_samples=8,
    )

    # Generate test data as DataFrame
    series = runner._generate_synthetic_series(256)
    df = pd.DataFrame({
        "ds": pd.date_range("2020-01-01", periods=len(series), freq="D"),
        "Close": series,
    })
    targets = np.array([series[-1] * 1.01])

    _, eager_results = runner.test_kronos_compiled_vs_eager(df, targets)

    runner.save_results(eager_results, "kronos_stress_results.json")
    runner.generate_report(eager_results, "kronos_stress_report.md")

    assert len(eager_results) == runner.num_iterations


if __name__ == "__main__":
    # Run full stress test
    print("Running full compile integration stress test...")

    runner = CompileStressTestRunner(
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_iterations=5,
        context_length=512,
        num_samples=128,
    )

    # Test Toto
    series = runner._generate_synthetic_series(512)
    targets = np.array([series[-1] * 1.01])

    print("\n" + "="*80)
    print("TOTO MODEL STRESS TEST")
    print("="*80)
    compiled_results, eager_results = runner.test_toto_compiled_vs_eager(series, targets)
    toto_results = compiled_results + eager_results

    # Test Kronos (if available)
    # df = pd.DataFrame({
    #     "ds": pd.date_range("2020-01-01", periods=len(series), freq="D"),
    #     "Close": series,
    # })
    # print("\n" + "="*80)
    # print("KRONOS MODEL STRESS TEST")
    # print("="*80)
    # _, kronos_results = runner.test_kronos_compiled_vs_eager(df, targets)

    all_results = toto_results  # + kronos_results
    runner.save_results(all_results)
    runner.generate_report(all_results)

    print("\n" + "="*80)
    print("STRESS TEST COMPLETE")
    print("="*80)
