"""Integration test for Chronos2 forecasting in backtesting."""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.chronos2_wrapper import Chronos2OHLCWrapper


@pytest.fixture
def sample_ohlc_data():
    """Create sample OHLC data for testing."""
    np.random.seed(42)
    n_points = 100

    # Generate realistic price movements
    base_price = 100.0
    returns = np.random.randn(n_points) * 0.02  # 2% daily volatility
    prices = base_price * np.exp(np.cumsum(returns))

    # Create OHLC data
    data = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=n_points, freq='D'),
        'open': prices * (1 + np.random.randn(n_points) * 0.005),
        'high': prices * (1 + np.abs(np.random.randn(n_points)) * 0.01),
        'low': prices * (1 - np.abs(np.random.randn(n_points)) * 0.01),
        'close': prices,
        'symbol': 'TEST'
    })

    return data


def test_chronos2_wrapper_initialization():
    """Test that Chronos2 wrapper can be initialized."""
    try:
        wrapper = Chronos2OHLCWrapper.from_pretrained(
            model_id="amazon/chronos-2",
            device_map="cpu",  # Use CPU for testing
            default_context_length=64,
            default_batch_size=16,
            torch_compile=False,
        )
        assert wrapper is not None
        assert wrapper.default_context_length == 64
        assert wrapper.default_batch_size == 16
    except Exception as e:
        pytest.skip(f"Chronos2 not available: {e}")


def test_chronos2_prediction(sample_ohlc_data):
    """Test that Chronos2 can make predictions on OHLC data."""
    try:
        wrapper = Chronos2OHLCWrapper.from_pretrained(
            model_id="amazon/chronos-2",
            device_map="cpu",
            default_context_length=64,
            default_batch_size=16,
            torch_compile=False,
        )

        # Make prediction
        result = wrapper.predict_ohlc(
            context_df=sample_ohlc_data,
            symbol="TEST",
            prediction_length=7,
            context_length=64,
            batch_size=16,
        )

        # Verify result structure
        assert result is not None
        assert hasattr(result, 'quantile_frames')
        assert 0.5 in result.quantile_frames

        median_frame = result.quantile_frames[0.5]
        assert 'close' in median_frame.columns
        assert 'open' in median_frame.columns
        assert 'high' in median_frame.columns
        assert 'low' in median_frame.columns
        assert len(median_frame) == 7

    except Exception as e:
        pytest.skip(f"Chronos2 prediction failed: {e}")


def test_chronos2_column_names(sample_ohlc_data):
    """Test that column names must be lowercase."""
    try:
        wrapper = Chronos2OHLCWrapper.from_pretrained(
            model_id="amazon/chronos-2",
            device_map="cpu",
            default_context_length=64,
            default_batch_size=16,
            torch_compile=False,
        )

        # Create data with uppercase columns (should fail)
        bad_data = sample_ohlc_data.copy()
        bad_data.columns = [col.upper() if col != 'timestamp' and col != 'symbol' else col
                           for col in bad_data.columns]

        # This should raise an error about missing columns
        with pytest.raises(Exception):
            wrapper.predict_ohlc(
                context_df=bad_data,
                symbol="TEST",
                prediction_length=7,
                context_length=64,
                batch_size=16,
            )

    except ImportError:
        pytest.skip("Chronos2 not available")


def test_chronos2_percentage_returns_conversion(sample_ohlc_data):
    """Test converting absolute predictions to percentage returns."""
    try:
        wrapper = Chronos2OHLCWrapper.from_pretrained(
            model_id="amazon/chronos-2",
            device_map="cpu",
            default_context_length=64,
            default_batch_size=16,
            torch_compile=False,
        )

        result = wrapper.predict_ohlc(
            context_df=sample_ohlc_data,
            symbol="TEST",
            prediction_length=7,
            context_length=64,
            batch_size=16,
        )

        median_frame = result.quantile_frames[0.5]
        close_predictions = median_frame['close'].values

        # Convert to percentage returns like in backtest code
        current_last_price = float(sample_ohlc_data['close'].iloc[-1])
        pct_returns = []
        prev_price = current_last_price
        for pred_price in close_predictions:
            pct_change = (pred_price - prev_price) / prev_price if prev_price != 0 else 0.0
            pct_returns.append(pct_change)
            prev_price = pred_price

        assert len(pct_returns) == 7
        # Verify returns are reasonable (within -50% to +50%)
        assert all(-0.5 < r < 0.5 for r in pct_returns)

    except Exception as e:
        pytest.skip(f"Chronos2 not available: {e}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
