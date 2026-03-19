#!/usr/bin/env python3
"""
Aggressive stress testing for Chronos2 torch.compile to uncover edge cases.

This script tries to break compilation by:
- Testing many random data patterns
- Trying different input shapes and sizes
- Testing edge cases that have historically caused issues
- Running extensive iterations to find intermittent failures
- Testing with real training data if available
"""

import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chronos_compile_config import ChronosCompileConfig, apply_production_compiled
from src.models.chronos2_wrapper import Chronos2OHLCWrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("stress_test")

# Test configuration
CONTEXT_LENGTHS = [64, 128, 256, 512, 1024]
PREDICTION_LENGTHS = [7, 16, 32, 64]
NUM_ITERATIONS = 10
MAE_TOLERANCE = 1e-2


class TestResult:
    """Store test results."""

    def __init__(self, name: str):
        self.name = name
        self.passed = 0
        self.failed = 0
        self.errors: List[str] = []
        self.mae_diffs: List[float] = []
        self.latencies_eager: List[float] = []
        self.latencies_compiled: List[float] = []

    def add_pass(self, mae_diff: float = 0.0, latency_eager: float = 0.0, latency_compiled: float = 0.0):
        self.passed += 1
        self.mae_diffs.append(mae_diff)
        self.latencies_eager.append(latency_eager)
        self.latencies_compiled.append(latency_compiled)

    def add_fail(self, error: str):
        self.failed += 1
        self.errors.append(error)

    @property
    def total(self) -> int:
        return self.passed + self.failed

    @property
    def success_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0

    def summary(self) -> str:
        status = "✅" if self.failed == 0 else "❌"
        avg_mae = np.mean(self.mae_diffs) if self.mae_diffs else 0
        avg_speedup = 0.0
        if self.latencies_eager and self.latencies_compiled:
            avg_speedup = np.mean(self.latencies_eager) / np.mean(self.latencies_compiled)

        lines = [
            f"{status} {self.name}",
            f"   Success: {self.passed}/{self.total} ({self.success_rate:.1%})",
        ]
        if self.mae_diffs:
            lines.append(f"   Avg MAE diff: {avg_mae:.6f}")
        if avg_speedup > 0:
            lines.append(f"   Avg speedup: {avg_speedup:.2f}x")
        if self.errors:
            lines.append(f"   Errors: {len(self.errors)}")
            for i, error in enumerate(self.errors[:3], 1):
                lines.append(f"     {i}. {error[:100]}")
            if len(self.errors) > 3:
                lines.append(f"     ... and {len(self.errors) - 3} more")

        return "\n".join(lines)


def _get_device() -> str:
    """Get available device."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _create_data(
    n_points: int,
    scenario: str = "normal",
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Create test data with various patterns."""
    if seed is not None:
        np.random.seed(seed)

    base_price = 100.0
    timestamps = pd.date_range(start="2024-01-01", periods=n_points, freq="D")

    if scenario == "random_walk":
        returns = np.random.randn(n_points) * 0.02
        prices = base_price * np.exp(np.cumsum(returns))

    elif scenario == "trending_up":
        trend = np.linspace(0, 0.5, n_points)
        noise = np.random.randn(n_points) * 0.01
        prices = base_price * np.exp(trend + noise)

    elif scenario == "trending_down":
        trend = np.linspace(0, -0.5, n_points)
        noise = np.random.randn(n_points) * 0.01
        prices = base_price * np.exp(trend + noise)

    elif scenario == "high_vol":
        returns = np.random.randn(n_points) * 0.1
        prices = base_price * np.exp(np.cumsum(returns))

    elif scenario == "low_vol":
        returns = np.random.randn(n_points) * 0.001
        prices = base_price * np.exp(np.cumsum(returns))

    elif scenario == "jumps":
        returns = np.random.randn(n_points) * 0.02
        # Add random jumps
        jump_indices = np.random.choice(n_points, size=5, replace=False)
        returns[jump_indices] += np.random.choice([-0.2, 0.2], size=5)
        prices = base_price * np.exp(np.cumsum(returns))

    elif scenario == "mean_reverting":
        prices = np.zeros(n_points)
        prices[0] = base_price
        for i in range(1, n_points):
            mean_reversion = -0.1 * (prices[i - 1] - base_price)
            shock = np.random.randn() * 2
            prices[i] = prices[i - 1] + mean_reversion + shock

    elif scenario == "cyclic":
        cycle = np.sin(np.linspace(0, 4 * np.pi, n_points)) * 20
        noise = np.random.randn(n_points) * 2
        prices = base_price + cycle + noise

    elif scenario == "regime_change":
        n_half = n_points // 2
        # Low vol first half
        returns1 = np.random.randn(n_half) * 0.005
        # High vol second half
        returns2 = np.random.randn(n_points - n_half) * 0.05
        returns = np.concatenate([returns1, returns2])
        prices = base_price * np.exp(np.cumsum(returns))

    elif scenario == "outliers":
        returns = np.random.randn(n_points) * 0.02
        # Add outliers
        outlier_indices = np.random.choice(n_points, size=3, replace=False)
        returns[outlier_indices] = np.random.choice([-0.5, 0.5], size=3)
        prices = base_price * np.exp(np.cumsum(returns))

    elif scenario == "gaps":
        # Simulate data with gaps (NaN handling)
        returns = np.random.randn(n_points) * 0.02
        prices = base_price * np.exp(np.cumsum(returns))
        # Create gaps
        gap_indices = np.random.choice(n_points // 2, size=5, replace=False)
        prices[gap_indices] = np.nan

    else:  # "normal"
        returns = np.random.randn(n_points) * 0.02
        prices = base_price * np.exp(np.cumsum(returns))

    # Create OHLC
    opens = prices * (1 + np.random.randn(n_points) * 0.002)
    closes = prices * (1 + np.random.randn(n_points) * 0.002)
    highs = np.maximum(opens, closes) * (1 + np.abs(np.random.randn(n_points)) * 0.005)
    lows = np.minimum(opens, closes) * (1 - np.abs(np.random.randn(n_points)) * 0.005)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "symbol": "TEST",
    })

    # Drop NaN if present
    if scenario != "gaps":
        df = df.dropna()

    return df


def _run_prediction(
    wrapper: Chronos2OHLCWrapper,
    context_df: pd.DataFrame,
    prediction_length: int,
    context_length: int,
) -> Tuple[np.ndarray, float]:
    """Run prediction and return close prices and latency."""
    start = time.perf_counter()

    result = wrapper.predict_ohlc(
        context_df=context_df,
        symbol="TEST",
        prediction_length=prediction_length,
        context_length=min(context_length, len(context_df)),
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    latency = time.perf_counter() - start
    preds = result.median["close"].values

    return preds, latency


def _cleanup_wrapper(wrapper: Chronos2OHLCWrapper):
    """Clean up wrapper and free memory."""
    try:
        wrapper.unload()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        logger.debug(f"Cleanup error: {e}")


def test_scenario(
    scenario: str,
    context_length: int,
    prediction_length: int,
    device: str,
    num_iterations: int = NUM_ITERATIONS,
) -> TestResult:
    """Test a specific scenario with multiple iterations."""
    result = TestResult(f"{scenario} (ctx={context_length}, pred={prediction_length})")

    logger.info(f"Testing {result.name}...")

    for i in range(num_iterations):
        try:
            # Create data
            data = _create_data(
                n_points=context_length + prediction_length + 10,
                scenario=scenario,
                seed=42 + i,
            )

            if len(data) < context_length + prediction_length:
                raise ValueError(f"Insufficient data: {len(data)} rows")

            context = data.iloc[:-prediction_length]

            # Run eager mode
            eager_wrapper = Chronos2OHLCWrapper.from_pretrained(
                model_id="amazon/chronos-2",
                device_map=device,
                default_context_length=context_length,
                torch_compile=False,
            )

            try:
                eager_preds, latency_eager = _run_prediction(
                    eager_wrapper, context, prediction_length, context_length
                )
            finally:
                _cleanup_wrapper(eager_wrapper)

            # Run compiled mode
            compiled_wrapper = Chronos2OHLCWrapper.from_pretrained(
                model_id="amazon/chronos-2",
                device_map=device,
                default_context_length=context_length,
                torch_compile=True,
                compile_mode="reduce-overhead",
                compile_backend="inductor",
            )

            try:
                compiled_preds, latency_compiled = _run_prediction(
                    compiled_wrapper, context, prediction_length, context_length
                )
            finally:
                _cleanup_wrapper(compiled_wrapper)

            # Compare
            if np.isnan(eager_preds).any() or np.isnan(compiled_preds).any():
                raise ValueError("NaN in predictions")

            if np.isinf(eager_preds).any() or np.isinf(compiled_preds).any():
                raise ValueError("Inf in predictions")

            mae_diff = float(np.mean(np.abs(eager_preds - compiled_preds)))

            if mae_diff > MAE_TOLERANCE:
                raise ValueError(f"MAE diff {mae_diff:.6f} exceeds tolerance {MAE_TOLERANCE}")

            result.add_pass(mae_diff, latency_eager, latency_compiled)

            logger.debug(
                f"  Iter {i+1}/{num_iterations}: MAE={mae_diff:.6f}, "
                f"speedup={latency_eager/latency_compiled:.2f}x"
            )

        except Exception as e:
            error_msg = f"Iter {i+1}: {str(e)}"
            result.add_fail(error_msg)
            logger.warning(f"  {error_msg}")
            logger.debug(traceback.format_exc())

    return result


def test_real_data(device: str) -> Optional[TestResult]:
    """Test with real training data if available."""
    result = TestResult("Real training data")

    trainingdata_dir = Path("trainingdata")
    if not trainingdata_dir.exists():
        logger.info("Skipping real data tests (trainingdata/ not found)")
        return None

    # Try to load some real symbols
    symbols = ["BTCUSD", "ETHUSD", "SOLUSD", "AAPL", "TSLA"]
    tested_any = False

    for symbol in symbols:
        csv_path = trainingdata_dir / f"{symbol}.csv"
        if not csv_path.exists():
            continue

        logger.info(f"Testing with real data: {symbol}")
        tested_any = True

        try:
            df = pd.read_csv(csv_path)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

            if len(df) < 200:
                logger.warning(f"  Insufficient data for {symbol}: {len(df)} rows")
                continue

            # Use last 200 points
            df = df.tail(200).reset_index(drop=True)

            context = df.iloc[:-16]
            prediction_length = 16
            context_length = 128

            # Eager
            eager_wrapper = Chronos2OHLCWrapper.from_pretrained(
                model_id="amazon/chronos-2",
                device_map=device,
                default_context_length=context_length,
                torch_compile=False,
            )

            try:
                eager_preds, latency_eager = _run_prediction(
                    eager_wrapper, context, prediction_length, context_length
                )
            finally:
                _cleanup_wrapper(eager_wrapper)

            # Compiled
            compiled_wrapper = Chronos2OHLCWrapper.from_pretrained(
                model_id="amazon/chronos-2",
                device_map=device,
                default_context_length=context_length,
                torch_compile=True,
                compile_mode="reduce-overhead",
                compile_backend="inductor",
            )

            try:
                compiled_preds, latency_compiled = _run_prediction(
                    compiled_wrapper, context, prediction_length, context_length
                )
            finally:
                _cleanup_wrapper(compiled_wrapper)

            # Compare
            mae_diff = float(np.mean(np.abs(eager_preds - compiled_preds)))

            if mae_diff > MAE_TOLERANCE:
                raise ValueError(f"MAE diff {mae_diff:.6f} exceeds tolerance")

            result.add_pass(mae_diff, latency_eager, latency_compiled)
            logger.info(f"  ✓ {symbol}: MAE={mae_diff:.6f}")

        except Exception as e:
            error_msg = f"{symbol}: {str(e)}"
            result.add_fail(error_msg)
            logger.warning(f"  ✗ {error_msg}")

    if not tested_any:
        return None

    return result


def main():
    """Run stress tests."""
    logger.info("=" * 80)
    logger.info("Chronos2 Compilation Stress Test")
    logger.info("=" * 80)

    device = _get_device()
    logger.info(f"Device: {device}")
    logger.info(f"Iterations per test: {NUM_ITERATIONS}")
    logger.info(f"MAE tolerance: {MAE_TOLERANCE}")

    # Test scenarios
    scenarios = [
        "random_walk",
        "trending_up",
        "trending_down",
        "high_vol",
        "low_vol",
        "jumps",
        "mean_reverting",
        "cyclic",
        "regime_change",
        "outliers",
    ]

    # Test configurations
    configs = [
        (128, 16),  # Standard
        (256, 32),  # Larger
        (64, 7),    # Smaller
    ]

    all_results: List[TestResult] = []

    # Run scenario tests
    logger.info("\n" + "=" * 80)
    logger.info("Testing synthetic data scenarios")
    logger.info("=" * 80)

    for context_length, prediction_length in configs:
        for scenario in scenarios:
            result = test_scenario(
                scenario=scenario,
                context_length=context_length,
                prediction_length=prediction_length,
                device=device,
                num_iterations=NUM_ITERATIONS,
            )
            all_results.append(result)

    # Test with real data
    logger.info("\n" + "=" * 80)
    logger.info("Testing with real training data")
    logger.info("=" * 80)

    real_result = test_real_data(device)
    if real_result:
        all_results.append(real_result)

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("STRESS TEST SUMMARY")
    logger.info("=" * 80)

    total_passed = sum(r.passed for r in all_results)
    total_failed = sum(r.failed for r in all_results)
    total_tests = total_passed + total_failed

    logger.info(f"\nOverall: {total_passed}/{total_tests} passed ({total_passed/total_tests:.1%})")
    logger.info(f"Failed: {total_failed}")

    logger.info("\nDetailed Results:")
    logger.info("-" * 80)

    for result in all_results:
        logger.info(result.summary())

    # Find worst cases
    logger.info("\n" + "=" * 80)
    logger.info("WORST CASES")
    logger.info("=" * 80)

    failed_results = [r for r in all_results if r.failed > 0]
    if failed_results:
        failed_results.sort(key=lambda r: r.failed, reverse=True)
        for result in failed_results[:5]:
            logger.info(result.summary())
    else:
        logger.info("✅ No failures detected!")

    # Summary statistics
    all_mae_diffs = [mae for r in all_results for mae in r.mae_diffs]
    if all_mae_diffs:
        logger.info("\n" + "=" * 80)
        logger.info("NUMERICAL ACCURACY")
        logger.info("=" * 80)
        logger.info(f"Mean MAE diff: {np.mean(all_mae_diffs):.6f}")
        logger.info(f"Median MAE diff: {np.median(all_mae_diffs):.6f}")
        logger.info(f"Max MAE diff: {np.max(all_mae_diffs):.6f}")
        logger.info(f"95th percentile: {np.percentile(all_mae_diffs, 95):.6f}")

    # Final verdict
    logger.info("\n" + "=" * 80)
    logger.info("VERDICT")
    logger.info("=" * 80)

    if total_failed == 0:
        logger.info("✅ ALL TESTS PASSED - Compilation appears stable")
        return 0
    elif total_failed / total_tests < 0.05:
        logger.info(f"⚠️  {total_failed} failures detected ({total_failed/total_tests:.1%})")
        logger.info("Compilation mostly stable but has some edge cases")
        return 1
    else:
        logger.info(f"❌ {total_failed} failures detected ({total_failed/total_tests:.1%})")
        logger.info("Compilation has significant issues - recommend keeping disabled")
        return 1


if __name__ == "__main__":
    sys.exit(main())
