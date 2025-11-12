"""
Comprehensive integration tests for Chronos2 using real training data.
This verifies end-to-end that Chronos2 predictions work correctly.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set environment variable BEFORE importing backtest module
os.environ["ONLY_CHRONOS2"] = "1"
os.environ["REAL_TESTING"] = "1"

from backtest_test3_inline import (
    load_chronos2_wrapper,
    resolve_best_model,
    resolve_chronos2_params,
)
from src.models.chronos2_wrapper import Chronos2OHLCWrapper


@pytest.fixture
def btcusd_data():
    """Load real BTCUSD training data."""
    data_path = Path(__file__).parent.parent / "trainingdata" / "BTCUSD.csv"
    if not data_path.exists():
        pytest.skip(f"Training data not found: {data_path}")

    df = pd.read_csv(data_path)
    # Ensure proper column names
    required_cols = ["Open", "High", "Low", "Close"]
    for col in required_cols:
        if col not in df.columns:
            pytest.skip(f"Missing required column: {col}")

    return df


@pytest.fixture
def aapl_data():
    """Load real AAPL training data."""
    data_path = Path(__file__).parent.parent / "trainingdata" / "AAPL.csv"
    if not data_path.exists():
        pytest.skip(f"Training data not found: {data_path}")

    df = pd.read_csv(data_path)
    required_cols = ["Open", "High", "Low", "Close"]
    for col in required_cols:
        if col not in df.columns:
            pytest.skip(f"Missing required column: {col}")

    return df


def test_resolve_best_model_returns_chronos2():
    """Test that resolve_best_model returns 'chronos2' when ONLY_CHRONOS2 is set."""
    model = resolve_best_model("BTCUSD")
    assert model == "chronos2", f"Expected 'chronos2', got '{model}'"

    model = resolve_best_model("AAPL")
    assert model == "chronos2", f"Expected 'chronos2', got '{model}'"


def test_resolve_chronos2_params():
    """Test that chronos2 params can be resolved."""
    params = resolve_chronos2_params("BTCUSD")

    assert isinstance(params, dict), "Params should be a dictionary"
    assert "model_id" in params, "Missing model_id"
    assert "context_length" in params, "Missing context_length"
    assert "prediction_length" in params, "Missing prediction_length"
    assert "quantile_levels" in params, "Missing quantile_levels"
    assert "batch_size" in params, "Missing batch_size"

    # Verify types and ranges
    assert isinstance(params["model_id"], str)
    assert isinstance(params["context_length"], int)
    assert params["context_length"] > 0
    assert isinstance(params["prediction_length"], int)
    assert params["prediction_length"] > 0
    assert isinstance(params["quantile_levels"], (list, tuple))
    assert len(params["quantile_levels"]) > 0
    assert all(0 < q < 1 for q in params["quantile_levels"])


def test_load_chronos2_wrapper():
    """Test that Chronos2 wrapper loads successfully."""
    params = resolve_chronos2_params("BTCUSD")

    try:
        wrapper = load_chronos2_wrapper(params)
    except Exception as e:
        pytest.fail(f"Failed to load Chronos2 wrapper: {e}")

    assert wrapper is not None, "Wrapper should not be None"
    assert hasattr(wrapper, "predict_ohlc"), "Wrapper should have predict_ohlc method"
    assert hasattr(wrapper, "pipeline"), "Wrapper should have pipeline attribute"


def test_chronos2_prediction_with_real_btcusd_data(btcusd_data):
    """Test Chronos2 predictions on real BTCUSD data."""
    # Prepare data
    df = btcusd_data.tail(200).copy()  # Use last 200 rows
    df = df.reset_index(drop=True)
    df.columns = [col.lower() for col in df.columns]
    df["timestamp"] = pd.date_range(start="2024-01-01", periods=len(df), freq="D")
    df["symbol"] = "BTCUSD"

    # Load wrapper
    params = resolve_chronos2_params("BTCUSD")
    wrapper = load_chronos2_wrapper(params)

    # Make prediction
    try:
        result = wrapper.predict_ohlc(
            context_df=df,
            symbol="BTCUSD",
            prediction_length=7,
            context_length=min(params["context_length"], len(df)),
            batch_size=params["batch_size"],
        )
    except Exception as e:
        pytest.fail(f"Prediction failed: {e}")

    # Verify result structure
    assert result is not None, "Result should not be None"
    assert hasattr(result, "quantile_frames"), "Result should have quantile_frames"
    assert 0.5 in result.quantile_frames, "Should have median (0.5) quantile"

    median_frame = result.quantile_frames[0.5]

    # Verify all OHLC columns are present
    for col in ["open", "high", "low", "close"]:
        assert col in median_frame.columns, f"Missing column: {col}"

    # Verify prediction length
    assert len(median_frame) == 7, f"Expected 7 predictions, got {len(median_frame)}"

    # Verify predictions are reasonable (not NaN, not infinite, within reasonable bounds)
    for col in ["open", "high", "low", "close"]:
        values = median_frame[col].values
        assert not np.any(np.isnan(values)), f"{col} contains NaN values"
        assert not np.any(np.isinf(values)), f"{col} contains infinite values"
        assert np.all(values > 0), f"{col} contains non-positive values"

        # Check predictions are within 50% of last known price (reasonable for 7-day forecast)
        last_price = df[col].iloc[-1]
        max_deviation = 0.5  # 50%
        assert np.all(values > last_price * (1 - max_deviation)), \
            f"{col} predictions too low (>50% below last price)"
        assert np.all(values < last_price * (1 + max_deviation)), \
            f"{col} predictions too high (>50% above last price)"

    # Verify OHLC relationships (high >= low for each prediction)
    assert np.all(median_frame["high"].values >= median_frame["low"].values), \
        "High should be >= Low for all predictions"

    print(f"✓ BTCUSD predictions look good!")
    print(f"  Last close: {df['close'].iloc[-1]:.2f}")
    print(f"  Predicted closes: {median_frame['close'].values}")


def test_chronos2_prediction_with_real_aapl_data(aapl_data):
    """Test Chronos2 predictions on real AAPL data."""
    # Prepare data
    df = aapl_data.tail(200).copy()
    df = df.reset_index(drop=True)
    df.columns = [col.lower() for col in df.columns]
    df["timestamp"] = pd.date_range(start="2024-01-01", periods=len(df), freq="D")
    df["symbol"] = "AAPL"

    # Load wrapper
    params = resolve_chronos2_params("AAPL")
    wrapper = load_chronos2_wrapper(params)

    # Make prediction
    try:
        result = wrapper.predict_ohlc(
            context_df=df,
            symbol="AAPL",
            prediction_length=7,
            context_length=min(params["context_length"], len(df)),
            batch_size=params["batch_size"],
        )
    except Exception as e:
        pytest.fail(f"Prediction failed: {e}")

    # Verify result
    assert result is not None
    assert 0.5 in result.quantile_frames

    median_frame = result.quantile_frames[0.5]

    # Verify OHLC columns
    for col in ["open", "high", "low", "close"]:
        assert col in median_frame.columns, f"Missing column: {col}"

    # Verify reasonable predictions
    for col in ["open", "high", "low", "close"]:
        values = median_frame[col].values
        assert not np.any(np.isnan(values)), f"{col} contains NaN"
        assert not np.any(np.isinf(values)), f"{col} contains inf"
        assert np.all(values > 0), f"{col} contains non-positive values"

        last_price = df[col].iloc[-1]
        assert np.all(values > last_price * 0.5), f"{col} predictions unreasonably low"
        assert np.all(values < last_price * 1.5), f"{col} predictions unreasonably high"

    print(f"✓ AAPL predictions look good!")
    print(f"  Last close: {df['close'].iloc[-1]:.2f}")
    print(f"  Predicted closes: {median_frame['close'].values}")


def test_percentage_return_conversion(btcusd_data):
    """Test that absolute predictions can be converted to percentage returns."""
    df = btcusd_data.tail(200).copy()
    df = df.reset_index(drop=True)
    df.columns = [col.lower() for col in df.columns]
    df["timestamp"] = pd.date_range(start="2024-01-01", periods=len(df), freq="D")
    df["symbol"] = "BTCUSD"

    params = resolve_chronos2_params("BTCUSD")
    wrapper = load_chronos2_wrapper(params)

    result = wrapper.predict_ohlc(
        context_df=df,
        symbol="BTCUSD",
        prediction_length=7,
        context_length=min(params["context_length"], len(df)),
        batch_size=params["batch_size"],
    )

    median_frame = result.quantile_frames[0.5]
    close_predictions = median_frame["close"].values

    # Convert to percentage returns (like backtest does)
    current_last_price = float(df["close"].iloc[-1])
    pct_returns = []
    prev_price = current_last_price

    for pred_price in close_predictions:
        pct_change = (pred_price - prev_price) / prev_price if prev_price != 0 else 0.0
        pct_returns.append(pct_change)
        prev_price = pred_price

    # Verify returns are reasonable
    assert len(pct_returns) == 7
    assert all(isinstance(r, float) for r in pct_returns)
    assert all(-0.5 < r < 0.5 for r in pct_returns), \
        f"Returns outside reasonable range: {pct_returns}"

    # Verify we can convert to tensor
    pct_tensor = torch.tensor(pct_returns, dtype=torch.float32)
    assert pct_tensor.shape == (7,)
    assert not torch.any(torch.isnan(pct_tensor))

    print(f"✓ Percentage return conversion works!")
    print(f"  Returns: {pct_returns}")


def test_no_invalid_backend_error(btcusd_data):
    """Test that we don't get 'Invalid backend' error."""
    df = btcusd_data.tail(100).copy()
    df = df.reset_index(drop=True)
    df.columns = [col.lower() for col in df.columns]
    df["timestamp"] = pd.date_range(start="2024-01-01", periods=len(df), freq="D")
    df["symbol"] = "BTCUSD"

    params = resolve_chronos2_params("BTCUSD")
    wrapper = load_chronos2_wrapper(params)

    # Run prediction multiple times to ensure no intermittent errors
    for i in range(3):
        try:
            result = wrapper.predict_ohlc(
                context_df=df,
                symbol="BTCUSD",
                prediction_length=7,
                context_length=min(params["context_length"], len(df)),
                batch_size=params["batch_size"],
            )
            assert result is not None, f"Prediction {i+1} returned None"
        except RuntimeError as e:
            if "Invalid backend" in str(e):
                pytest.fail(f"Got 'Invalid backend' error on attempt {i+1}: {e}")
            else:
                raise

    print(f"✓ No 'Invalid backend' errors in 3 prediction attempts!")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s", "--tb=short"])
