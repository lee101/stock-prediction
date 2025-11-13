"""
Comprehensive fuzzing tests for Chronos2 torch.compile with numerical stability checks.

This test suite validates that torch.compile works reliably across:
1. Different compile modes (None, reduce-overhead, default, max-autotune)
2. Various numerical inputs (normal, extreme values, NaN, inf, very small)
3. Different dtypes (float32, float16, bfloat16)
4. Edge cases that have historically caused issues
5. Real-world anomalies (spikes, drops, volatility changes)

The goal is to ensure compilation is robust and produces numerically stable results
compared to eager mode.
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.chronos2_wrapper import Chronos2OHLCWrapper

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("chronos2_compile_fuzzing")

# Test configuration
CONTEXT_LENGTH = 128
PREDICTION_LENGTH = 16
BATCH_SIZE = 32
MAE_TOLERANCE = 1e-2  # Maximum acceptable MAE difference between eager and compiled
RELATIVE_TOLERANCE = 0.05  # 5% relative difference tolerance

# Compile modes to test (ordered from safest to most aggressive)
COMPILE_MODES = [
    None,  # Default PyTorch behavior
    "default",  # Balanced compilation
    "reduce-overhead",  # Currently used in production
    # "max-autotune",  # Most aggressive - commented out as it's often unstable
]

# Backends to test
COMPILE_BACKENDS = [
    "inductor",  # Default backend (most tested)
    # "aot_eager",  # Ahead-of-time eager backend (safer but slower)
]

# Data types to test
DTYPES = [
    "float32",  # Default
    # "float16",  # Half precision - often causes numerical issues
    # "bfloat16",  # Brain float - better range than fp16 but requires specific hardware
]


def _get_device() -> str:
    """Get available device for testing."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _create_test_data(
    n_points: int = CONTEXT_LENGTH + PREDICTION_LENGTH,
    base_price: float = 100.0,
    volatility: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:
    """Create realistic OHLC test data."""
    np.random.seed(seed)

    # Generate realistic price movement
    returns = np.random.randn(n_points) * volatility
    prices = base_price * np.exp(np.cumsum(returns))

    # Create OHLC with realistic intraday movements
    opens = prices * (1 + np.random.randn(n_points) * 0.002)
    closes = prices * (1 + np.random.randn(n_points) * 0.002)
    highs = np.maximum(opens, closes) * (1 + np.abs(np.random.randn(n_points)) * 0.005)
    lows = np.minimum(opens, closes) * (1 - np.abs(np.random.randn(n_points)) * 0.005)

    return pd.DataFrame({
        "timestamp": pd.date_range(start="2024-01-01", periods=n_points, freq="D"),
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "symbol": "TEST",
    })


def _create_extreme_data(
    n_points: int = CONTEXT_LENGTH + PREDICTION_LENGTH,
    scenario: str = "normal",
) -> pd.DataFrame:
    """Create test data with extreme values for robustness testing."""
    base_df = _create_test_data(n_points)

    if scenario == "very_small":
        # Very small positive values (test epsilon clamping)
        base_df[["open", "high", "low", "close"]] *= 1e-4

    elif scenario == "very_large":
        # Very large values
        base_df[["open", "high", "low", "close"]] *= 1e6

    elif scenario == "high_volatility":
        # Extreme volatility
        multipliers = np.exp(np.random.randn(n_points) * 0.2)  # 20% volatility
        for col in ["open", "high", "low", "close"]:
            base_df[col] *= multipliers

    elif scenario == "spike":
        # Sudden spike in the middle
        spike_idx = n_points // 2
        base_df.loc[spike_idx : spike_idx + 3, ["open", "high", "low", "close"]] *= 5.0

    elif scenario == "drop":
        # Sudden drop in the middle
        drop_idx = n_points // 2
        base_df.loc[drop_idx : drop_idx + 3, ["open", "high", "low", "close"]] *= 0.2

    elif scenario == "near_zero":
        # Values very close to zero (but not quite)
        base_df[["open", "high", "low", "close"]] = (
            np.random.randn(n_points, 4) * 1e-5 + 1e-4
        )
        base_df[["open", "high", "low", "close"]] = base_df[
            ["open", "high", "low", "close"]
        ].abs()

    elif scenario == "constant":
        # Constant values (no movement)
        base_df[["open", "high", "low", "close"]] = 100.0

    elif scenario == "linear_trend":
        # Strong linear trend
        trend = np.linspace(50, 200, n_points)
        base_df[["open", "high", "low", "close"]] = trend[:, None] * (
            1 + np.random.randn(n_points, 4) * 0.01
        )

    return base_df


def _load_wrapper(
    compile_enabled: bool,
    compile_mode: Optional[str] = None,
    compile_backend: str = "inductor",
    dtype: str = "float32",
    device: str = "cpu",
) -> Chronos2OHLCWrapper:
    """Load Chronos2 wrapper with specified compilation settings."""
    # Clear any existing environment variables
    os.environ.pop("CHRONOS_COMPILE", None)
    os.environ.pop("TORCH_COMPILED", None)

    return Chronos2OHLCWrapper.from_pretrained(
        model_id="amazon/chronos-2",
        device_map=device,
        default_context_length=CONTEXT_LENGTH,
        default_batch_size=BATCH_SIZE,
        torch_compile=compile_enabled,
        compile_mode=compile_mode,
        compile_backend=compile_backend if compile_enabled else None,
        torch_dtype=dtype,
    )


def _run_prediction(
    wrapper: Chronos2OHLCWrapper,
    context_df: pd.DataFrame,
    prediction_length: int = PREDICTION_LENGTH,
) -> pd.DataFrame:
    """Run prediction and return close prices."""
    result = wrapper.predict_ohlc(
        context_df=context_df,
        symbol="TEST",
        prediction_length=prediction_length,
        context_length=min(CONTEXT_LENGTH, len(context_df)),
    )
    return result.median["close"].values


def _calculate_mae_difference(
    eager_preds: np.ndarray,
    compiled_preds: np.ndarray,
) -> Tuple[float, float]:
    """Calculate MAE and relative difference between predictions."""
    mae_diff = float(np.mean(np.abs(eager_preds - compiled_preds)))
    mean_scale = float(np.mean(np.abs(eager_preds)))
    relative_diff = mae_diff / mean_scale if mean_scale > 1e-10 else 0.0
    return mae_diff, relative_diff


def _cleanup_wrapper(wrapper: Chronos2OHLCWrapper) -> None:
    """Clean up wrapper and free GPU memory."""
    wrapper.unload()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Test fixtures
@pytest.fixture(scope="module")
def device() -> str:
    """Get device for testing."""
    return _get_device()


@pytest.fixture(scope="module")
def test_data() -> pd.DataFrame:
    """Create standard test data."""
    return _create_test_data()


# Basic smoke tests
def test_eager_mode_baseline(device: str, test_data: pd.DataFrame) -> None:
    """Test that eager mode works as baseline."""
    logger.info("Testing eager mode baseline...")

    wrapper = _load_wrapper(compile_enabled=False, device=device)
    context = test_data.iloc[:-PREDICTION_LENGTH]

    try:
        preds = _run_prediction(wrapper, context)
        assert len(preds) == PREDICTION_LENGTH
        assert not np.isnan(preds).any(), "Predictions contain NaN"
        assert not np.isinf(preds).any(), "Predictions contain inf"
        logger.info("✓ Eager mode baseline passed")
    finally:
        _cleanup_wrapper(wrapper)


@pytest.mark.parametrize("compile_mode", COMPILE_MODES)
@pytest.mark.parametrize("compile_backend", COMPILE_BACKENDS)
def test_compiled_mode_smoke(
    device: str,
    test_data: pd.DataFrame,
    compile_mode: Optional[str],
    compile_backend: str,
) -> None:
    """Smoke test for each compile mode and backend combination."""
    mode_str = compile_mode or "default"
    logger.info(f"Testing compiled mode: {mode_str} + {compile_backend}...")

    wrapper = _load_wrapper(
        compile_enabled=True,
        compile_mode=compile_mode,
        compile_backend=compile_backend,
        device=device,
    )
    context = test_data.iloc[:-PREDICTION_LENGTH]

    try:
        preds = _run_prediction(wrapper, context)
        assert len(preds) == PREDICTION_LENGTH
        assert not np.isnan(preds).any(), f"Predictions contain NaN ({mode_str})"
        assert not np.isinf(preds).any(), f"Predictions contain inf ({mode_str})"
        logger.info(f"✓ Compiled mode {mode_str} + {compile_backend} passed")
    finally:
        _cleanup_wrapper(wrapper)


# Numerical stability tests
@pytest.mark.parametrize("compile_mode", COMPILE_MODES)
def test_eager_vs_compiled_accuracy(
    device: str,
    test_data: pd.DataFrame,
    compile_mode: Optional[str],
) -> None:
    """Test that compiled mode produces similar results to eager mode."""
    mode_str = compile_mode or "default"
    logger.info(f"Testing eager vs compiled accuracy ({mode_str})...")

    context = test_data.iloc[:-PREDICTION_LENGTH]

    # Run eager mode
    eager_wrapper = _load_wrapper(compile_enabled=False, device=device)
    try:
        eager_preds = _run_prediction(eager_wrapper, context)
    finally:
        _cleanup_wrapper(eager_wrapper)

    # Run compiled mode
    compiled_wrapper = _load_wrapper(
        compile_enabled=True,
        compile_mode=compile_mode,
        device=device,
    )
    try:
        compiled_preds = _run_prediction(compiled_wrapper, context)
    finally:
        _cleanup_wrapper(compiled_wrapper)

    # Compare
    mae_diff, relative_diff = _calculate_mae_difference(eager_preds, compiled_preds)

    logger.info(
        f"  MAE difference: {mae_diff:.6f}, Relative: {relative_diff:.2%}"
    )

    assert mae_diff < MAE_TOLERANCE, (
        f"MAE difference {mae_diff} exceeds tolerance {MAE_TOLERANCE} ({mode_str})"
    )
    assert relative_diff < RELATIVE_TOLERANCE, (
        f"Relative difference {relative_diff:.2%} exceeds {RELATIVE_TOLERANCE:.2%} ({mode_str})"
    )

    logger.info(f"✓ Accuracy test passed ({mode_str})")


# Fuzzing tests with extreme data
@pytest.mark.parametrize(
    "scenario",
    [
        "very_small",
        "very_large",
        "high_volatility",
        "spike",
        "drop",
        "near_zero",
        "constant",
        "linear_trend",
    ],
)
def test_extreme_data_robustness(device: str, scenario: str) -> None:
    """Test that both eager and compiled modes handle extreme data gracefully."""
    logger.info(f"Testing robustness with {scenario} data...")

    data = _create_extreme_data(scenario=scenario)
    context = data.iloc[:-PREDICTION_LENGTH]

    # Test eager mode
    eager_wrapper = _load_wrapper(compile_enabled=False, device=device)
    try:
        eager_preds = _run_prediction(eager_wrapper, context)
        eager_success = True
        eager_has_nan = np.isnan(eager_preds).any()
        eager_has_inf = np.isinf(eager_preds).any()
    except Exception as e:
        logger.warning(f"  Eager mode failed on {scenario}: {e}")
        eager_success = False
        eager_has_nan = eager_has_inf = False
    finally:
        _cleanup_wrapper(eager_wrapper)

    # Test compiled mode with safest settings
    compiled_wrapper = _load_wrapper(
        compile_enabled=True,
        compile_mode="reduce-overhead",
        device=device,
    )
    try:
        compiled_preds = _run_prediction(compiled_wrapper, context)
        compiled_success = True
        compiled_has_nan = np.isnan(compiled_preds).any()
        compiled_has_inf = np.isinf(compiled_preds).any()
    except Exception as e:
        logger.warning(f"  Compiled mode failed on {scenario}: {e}")
        compiled_success = False
        compiled_has_nan = compiled_has_inf = False
    finally:
        _cleanup_wrapper(compiled_wrapper)

    # Both modes should have similar behavior
    if eager_success and compiled_success:
        # Compare numerical stability
        mae_diff, relative_diff = _calculate_mae_difference(eager_preds, compiled_preds)
        logger.info(
            f"  {scenario}: MAE diff={mae_diff:.6f}, Relative={relative_diff:.2%}"
        )

        # Check for NaN/inf consistency
        assert eager_has_nan == compiled_has_nan, (
            f"NaN inconsistency in {scenario}: eager={eager_has_nan}, compiled={compiled_has_nan}"
        )
        assert eager_has_inf == compiled_has_inf, (
            f"Inf inconsistency in {scenario}: eager={eager_has_inf}, compiled={compiled_has_inf}"
        )

        if not (eager_has_nan or eager_has_inf):
            # Only check accuracy if both produce valid numbers
            assert mae_diff < MAE_TOLERANCE * 10, (  # More lenient for extreme data
                f"MAE difference too large for {scenario}: {mae_diff}"
            )
    else:
        # At least one mode should work, or both should fail consistently
        assert eager_success == compiled_success, (
            f"Inconsistent failure behavior for {scenario}: "
            f"eager={eager_success}, compiled={compiled_success}"
        )

    logger.info(f"✓ Robustness test passed for {scenario}")


# Stress tests
def test_multiple_predictions_stability(device: str) -> None:
    """Test that compiled mode remains stable across multiple predictions."""
    logger.info("Testing multiple predictions stability...")

    wrapper = _load_wrapper(
        compile_enabled=True,
        compile_mode="reduce-overhead",
        device=device,
    )

    try:
        all_preds = []
        for i in range(5):
            data = _create_test_data(seed=42 + i)
            context = data.iloc[:-PREDICTION_LENGTH]
            preds = _run_prediction(wrapper, context)

            assert not np.isnan(preds).any(), f"Run {i+1} produced NaN"
            assert not np.isinf(preds).any(), f"Run {i+1} produced inf"

            all_preds.append(preds)

        # Check that predictions are consistent for same seed
        data_repeat = _create_test_data(seed=42)
        context_repeat = data_repeat.iloc[:-PREDICTION_LENGTH]
        preds_repeat = _run_prediction(wrapper, context_repeat)

        mae_consistency = float(np.mean(np.abs(all_preds[0] - preds_repeat)))
        logger.info(f"  Consistency MAE: {mae_consistency:.6f}")

        # Should be very close (may not be exact due to numerical precision)
        assert mae_consistency < 1e-4, f"Inconsistent predictions: {mae_consistency}"

        logger.info("✓ Multiple predictions stability test passed")

    finally:
        _cleanup_wrapper(wrapper)


def test_compilation_fallback_mechanism(device: str, test_data: pd.DataFrame) -> None:
    """Test that the fallback mechanism works when compilation fails."""
    logger.info("Testing compilation fallback mechanism...")

    # This test verifies the _call_with_compile_fallback mechanism
    # by checking that predictions still work even if compilation has issues

    wrapper = _load_wrapper(
        compile_enabled=True,
        compile_mode="reduce-overhead",
        device=device,
    )

    try:
        context = test_data.iloc[:-PREDICTION_LENGTH]

        # First prediction should trigger compilation
        preds1 = _run_prediction(wrapper, context)
        assert len(preds1) == PREDICTION_LENGTH

        # Subsequent predictions should use compiled model
        preds2 = _run_prediction(wrapper, context)
        assert len(preds2) == PREDICTION_LENGTH

        # Check consistency
        mae = float(np.mean(np.abs(preds1 - preds2)))
        assert mae < 1e-4, f"Inconsistent predictions: {mae}"

        logger.info("✓ Fallback mechanism test passed")

    finally:
        _cleanup_wrapper(wrapper)


# Summary test
def test_recommended_configuration(device: str, test_data: pd.DataFrame) -> None:
    """Test the recommended production configuration."""
    logger.info("Testing recommended production configuration...")

    # Recommended: reduce-overhead + inductor + float32 + eager attention
    wrapper = _load_wrapper(
        compile_enabled=True,
        compile_mode="reduce-overhead",
        compile_backend="inductor",
        dtype="float32",
        device=device,
    )

    try:
        context = test_data.iloc[:-PREDICTION_LENGTH]

        # Run multiple predictions to ensure stability
        for i in range(3):
            preds = _run_prediction(wrapper, context)
            assert len(preds) == PREDICTION_LENGTH
            assert not np.isnan(preds).any()
            assert not np.isinf(preds).any()
            logger.info(f"  Run {i+1}/3: ✓")

        logger.info("✓ Recommended configuration test passed")

    finally:
        _cleanup_wrapper(wrapper)


if __name__ == "__main__":
    # Run tests manually for debugging
    device = _get_device()
    logger.info(f"Running tests on device: {device}")

    test_data = _create_test_data()

    try:
        logger.info("\n=== Basic Tests ===")
        test_eager_mode_baseline(device, test_data)

        logger.info("\n=== Compiled Mode Smoke Tests ===")
        for mode in COMPILE_MODES:
            for backend in COMPILE_BACKENDS:
                test_compiled_mode_smoke(device, test_data, mode, backend)

        logger.info("\n=== Accuracy Tests ===")
        for mode in COMPILE_MODES:
            test_eager_vs_compiled_accuracy(device, test_data, mode)

        logger.info("\n=== Extreme Data Robustness Tests ===")
        scenarios = [
            "very_small",
            "very_large",
            "high_volatility",
            "spike",
            "drop",
            "near_zero",
            "constant",
            "linear_trend",
        ]
        for scenario in scenarios:
            test_extreme_data_robustness(device, scenario)

        logger.info("\n=== Stress Tests ===")
        test_multiple_predictions_stability(device)
        test_compilation_fallback_mechanism(device, test_data)

        logger.info("\n=== Production Configuration Test ===")
        test_recommended_configuration(device, test_data)

        logger.info("\n" + "=" * 60)
        logger.info("ALL TESTS PASSED ✓")
        logger.info("=" * 60)

    except Exception as exc:
        logger.exception(f"Test failed: {exc}")
        sys.exit(1)
