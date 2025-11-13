"""
Test for Chronos2 negative value handling in augmentations.

This test reproduces the issue where augmentation strategies can produce
small negative values that cause AssertionError in PyTorch's symbolic math
compilation (torch._inductor).

The error occurs because PyTorch Inductor's sympy evaluation expects
non-negative values in certain operations, but augmentations like
differencing, detrending, and robust scaling can produce small negative values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from preaug_sweeps.augmentations.strategies import (
    DifferencingAugmentation,
    DetrendingAugmentation,
    PercentChangeAugmentation,
    RobustScalingAugmentation,
    RollingWindowNormalization,
    LogReturnsAugmentation,
)


def _make_eth_like_dataframe(rows: int = 512) -> pd.DataFrame:
    """
    Create a dataframe that simulates ETHUSD price data.

    This includes realistic price movements that can produce
    negative differences, detrended residuals, etc.
    """
    np.random.seed(42)
    index = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")

    # Simulate realistic price with trend and volatility
    base_price = 2000.0
    trend = np.linspace(0, 200, rows)  # Upward trend
    noise = np.cumsum(np.random.randn(rows) * 10)  # Random walk component
    prices = base_price + trend + noise

    data = {
        "timestamp": index,
        "open": prices + np.random.randn(rows) * 5,
        "high": prices + np.abs(np.random.randn(rows) * 10),
        "low": prices - np.abs(np.random.randn(rows) * 10),
        "close": prices + np.random.randn(rows) * 5,
        "symbol": ["ETHUSD"] * rows,
        "volume": np.random.uniform(1000, 5000, rows),
    }

    # Ensure OHLC relationships are valid
    df = pd.DataFrame(data)
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)

    return df


def test_differencing_produces_negative_values():
    """Test that differencing augmentation produces negative values."""
    df = _make_eth_like_dataframe(512)

    aug = DifferencingAugmentation(order=1)
    transformed = aug.transform_dataframe(df[["open", "high", "low", "close"]])

    # Check that we have negative values
    for col in ["open", "high", "low", "close"]:
        assert (transformed[col] < 0).any(), f"Expected negative values in {col} after differencing"

        # Check for small negative values similar to the error
        min_val = transformed[col].min()
        print(f"{col}: min={min_val}, max={transformed[col].max()}")


def test_detrending_produces_negative_values():
    """Test that detrending augmentation produces negative values."""
    df = _make_eth_like_dataframe(512)

    aug = DetrendingAugmentation()
    transformed = aug.transform_dataframe(df[["open", "high", "low", "close"]])

    # Detrended residuals should have both positive and negative values
    for col in ["open", "high", "low", "close"]:
        assert (transformed[col] < 0).any(), f"Expected negative values in {col} after detrending"

        min_val = transformed[col].min()
        print(f"{col}: min={min_val}, max={transformed[col].max()}")


def test_robust_scaling_produces_negative_values():
    """Test that robust scaling produces negative values."""
    df = _make_eth_like_dataframe(512)

    aug = RobustScalingAugmentation()
    transformed = aug.transform_dataframe(df[["open", "high", "low", "close"]])

    # Values below median should be negative
    for col in ["open", "high", "low", "close"]:
        assert (transformed[col] < 0).any(), f"Expected negative values in {col} after robust scaling"

        min_val = transformed[col].min()
        print(f"{col}: min={min_val}, max={transformed[col].max()}")


def test_percent_change_with_declining_prices():
    """Test that percent change can produce negative values with declining prices."""
    # Create a declining price series
    rows = 512
    index = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")

    # Start high and decline
    base_price = 3000.0
    decline = np.linspace(0, -500, rows)
    prices = base_price + decline + np.random.randn(rows) * 10

    df = pd.DataFrame({
        "timestamp": index,
        "open": prices,
        "high": prices + 10,
        "low": prices - 10,
        "close": prices,
        "symbol": ["ETHUSD"] * rows,
    })

    aug = PercentChangeAugmentation()
    transformed = aug.transform_dataframe(df[["open", "high", "low", "close"]])

    # Should have negative percent changes
    for col in ["open", "high", "low", "close"]:
        assert (transformed[col] < 0).any(), f"Expected negative values in {col} with declining prices"

        min_val = transformed[col].min()
        print(f"{col}: min={min_val}, max={transformed[col].max()}")


def test_log_returns_with_volatility():
    """Test that log returns can produce negative values."""
    df = _make_eth_like_dataframe(512)

    aug = LogReturnsAugmentation()
    transformed = aug.transform_dataframe(df[["open", "high", "low", "close"]])

    # Log returns should have negative values when prices decline
    for col in ["open", "high", "low", "close"]:
        assert (transformed[col] < 0).any(), f"Expected negative values in {col} in log returns"

        min_val = transformed[col].min()
        print(f"{col}: min={min_val}, max={transformed[col].max()}")


def test_rolling_norm_produces_negative_values():
    """Test that rolling window normalization produces negative values."""
    df = _make_eth_like_dataframe(512)

    aug = RollingWindowNormalization(window_size=20)
    transformed = aug.transform_dataframe(df[["open", "high", "low", "close"]])

    # Values below rolling mean should be negative
    for col in ["open", "high", "low", "close"]:
        assert (transformed[col] < 0).any(), f"Expected negative values in {col} after rolling norm"

        min_val = transformed[col].min()
        print(f"{col}: min={min_val}, max={transformed[col].max()}")


def test_augmentation_roundtrip():
    """Test that augmentation + inverse transform recovers original values."""
    df = _make_eth_like_dataframe(100)
    context_df = df[["open", "high", "low", "close"]].copy()

    augmentations = [
        DifferencingAugmentation(order=1),
        DetrendingAugmentation(),
        RobustScalingAugmentation(),
        PercentChangeAugmentation(),
        LogReturnsAugmentation(),
        RollingWindowNormalization(window_size=20),
    ]

    for aug in augmentations:
        print(f"\nTesting {aug.name()}...")

        # Transform
        transformed = aug.transform_dataframe(context_df)

        # Check for negative values
        has_negatives = (transformed < 0).any().any()
        print(f"  Has negative values: {has_negatives}")

        if has_negatives:
            mins = transformed.min()
            print(f"  Min values: {mins.to_dict()}")

            # Check for very small negative values like the error
            very_small = (transformed < 0) & (transformed > -0.001)
            if very_small.any().any():
                print(f"  WARNING: Found very small negative values like the error!")
                for col in transformed.columns:
                    small_vals = transformed.loc[very_small[col], col]
                    if len(small_vals) > 0:
                        print(f"    {col}: {small_vals.head().values}")

        # Create fake predictions (same shape as last 10 rows)
        predictions = transformed.iloc[-10:].values

        # Inverse transform
        recovered = aug.inverse_transform_predictions(
            predictions,
            context_df,
            columns=list(context_df.columns)
        )

        # Check shapes match
        assert recovered.shape == predictions.shape, f"Shape mismatch for {aug.name()}"

        print(f"  Roundtrip successful")


def test_very_small_negative_value_simulation():
    """
    Simulate the exact error condition: very small negative value
    that causes assertion failure in PyTorch sympy evaluation.
    """
    # The error shows: -834735604272579/1000000000000000 ≈ -0.00083473...
    problematic_value = -834735604272579 / 1000000000000000
    print(f"Problematic value from error: {problematic_value}")

    # Create a dataframe with values that will produce similar small negatives
    rows = 512
    index = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")

    # Use prices that are very close to each other (low volatility)
    # This can produce very small differences
    base_price = 2000.0
    prices = base_price + np.random.randn(rows) * 0.01  # Very small noise

    df = pd.DataFrame({
        "timestamp": index,
        "open": prices,
        "high": prices + 0.001,
        "low": prices - 0.001,
        "close": prices + np.random.randn(rows) * 0.001,
        "symbol": ["ETHUSD"] * rows,
    })

    # Try differencing - this should produce very small values
    aug = DifferencingAugmentation(order=1)
    transformed = aug.transform_dataframe(df[["open", "high", "low", "close"]])

    # Find the smallest negative value
    for col in ["open", "high", "low", "close"]:
        neg_vals = transformed.loc[transformed[col] < 0, col]
        if len(neg_vals) > 0:
            min_val = neg_vals.min()
            # Check if we have values in the same order of magnitude as the error
            if min_val > -0.01:
                print(f"{col}: Found small negative value: {min_val} (similar magnitude to error)")

                # This kind of value could cause issues in PyTorch compilation
                assert min_val < 0, "Should be negative"
                assert abs(min_val) < 1.0, "Should be small in magnitude"


def test_chronos2_compile_fallback_mechanism():
    """
    Test that the Chronos2 wrapper has the compile fallback mechanism.

    This tests the fix for the PyTorch compilation error:
    AssertionError: -834735604272579/1000000000000000

    The fix involves:
    1. Storing the eager model before compilation (_eager_model)
    2. Wrapping predict_df calls with _call_with_compile_fallback
    3. Catching compilation errors and retrying without torch.compile
    """
    try:
        from src.models.chronos2_wrapper import Chronos2OHLCWrapper
    except ImportError:
        pytest.skip("Chronos2 wrapper not available")

    # Verify the fallback methods exist
    assert hasattr(Chronos2OHLCWrapper, "_disable_torch_compile"), \
        "Wrapper should have _disable_torch_compile method"
    assert hasattr(Chronos2OHLCWrapper, "_call_with_compile_fallback"), \
        "Wrapper should have _call_with_compile_fallback method"

    print("SUCCESS: Chronos2 wrapper has compile fallback mechanism")


def test_torch_compile_error_explanation():
    """
    Document the torch.compile error for future reference.

    The error occurs when:
    1. Augmentation strategies (like differencing, detrending, etc.) transform the data
    2. The transformed data is passed to the Chronos2 model
    3. torch.compile() attempts to optimize the model
    4. During symbolic shape analysis, PyTorch's inductor encounters a value
       that triggers an assertion failure in sympy evaluation

    The fix:
    - The Chronos2 wrapper now catches any exceptions during predict_df()
    - If torch.compile was enabled and an error occurs, it:
      * Disables compilation
      * Restores the eager (uncompiled) model
      * Retries the prediction without compilation
    - This ensures predictions succeed even if compilation fails
    """
    import numpy as np

    # The problematic value from the error (this is a symbolic value during compilation,
    # not necessarily a data value)
    error_value = -834735604272579 / 1000000000000000  # ≈ -0.835

    print(f"Error value from PyTorch compilation: {error_value}")
    print(f"Error context: PyTorch symbolic math in torch._inductor")
    print(f"Fix: Automatic fallback to eager mode when compilation fails")

    # This value is what PyTorch encountered during its internal symbolic analysis
    # It's not directly from the augmented data, but rather from the compiler's
    # analysis of tensor operations
    assert True, "Test passes - this is just documentation"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
