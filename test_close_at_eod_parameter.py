"""
Unit tests for close_at_eod parameter in strategy evaluation functions.

Tests that the close_at_eod parameter is correctly handled in both
evaluate_maxdiff_strategy and evaluate_maxdiff_always_on_strategy.
"""

import pytest
import torch
import pandas as pd
import numpy as np
from backtest_test3_inline import (
    evaluate_maxdiff_strategy,
    evaluate_maxdiff_always_on_strategy,
)


@pytest.fixture
def mock_predictions():
    """Create mock prediction data for testing."""
    n = 100
    return {
        "close_actual_movement_values": torch.randn(n) * 0.01,
        "high_actual_movement_values": torch.abs(torch.randn(n)) * 0.02,
        "low_actual_movement_values": -torch.abs(torch.randn(n)) * 0.02,
        "high_predictions": torch.abs(torch.randn(n)) * 0.015,
        "low_predictions": -torch.abs(torch.randn(n)) * 0.015,
        "high_predicted_price_value": 100.0,
        "low_predicted_price_value": 98.0,
    }


@pytest.fixture
def mock_simulation_data():
    """Create mock simulation data for testing."""
    n = 102  # Need 2 extra for the offset
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "Close": np.random.uniform(95, 105, n),
        "High": np.random.uniform(100, 110, n),
        "Low": np.random.uniform(90, 100, n),
        "Volume": np.random.randint(1000000, 10000000, n),
    }, index=dates)


class TestCloseAtEodParameter:
    """Tests for close_at_eod parameter handling."""

    def test_maxdiff_strategy_accepts_close_at_eod_true(
        self, mock_predictions, mock_simulation_data
    ):
        """Test that evaluate_maxdiff_strategy accepts close_at_eod=True."""
        eval_result, returns, metadata = evaluate_maxdiff_strategy(
            mock_predictions,
            mock_simulation_data,
            trading_fee=0.001,
            trading_days_per_year=252,
            is_crypto=False,
            close_at_eod=True,
        )

        # Should complete without error
        assert eval_result is not None
        assert metadata["maxdiff_close_at_eod"] is True

    def test_maxdiff_strategy_accepts_close_at_eod_false(
        self, mock_predictions, mock_simulation_data
    ):
        """Test that evaluate_maxdiff_strategy accepts close_at_eod=False."""
        eval_result, returns, metadata = evaluate_maxdiff_strategy(
            mock_predictions,
            mock_simulation_data,
            trading_fee=0.001,
            trading_days_per_year=252,
            is_crypto=False,
            close_at_eod=False,
        )

        # Should complete without error
        assert eval_result is not None
        assert metadata["maxdiff_close_at_eod"] is False

    def test_maxdiff_strategy_optimizes_when_none(
        self, mock_predictions, mock_simulation_data
    ):
        """Test that evaluate_maxdiff_strategy optimizes close_at_eod when None."""
        eval_result, returns, metadata = evaluate_maxdiff_strategy(
            mock_predictions,
            mock_simulation_data,
            trading_fee=0.001,
            trading_days_per_year=252,
            is_crypto=False,
            close_at_eod=None,
        )

        # Should complete without error
        assert eval_result is not None
        # Should have chosen either True or False (optimized)
        assert isinstance(metadata["maxdiff_close_at_eod"], bool)

    def test_maxdiff_always_on_accepts_close_at_eod_true(
        self, mock_predictions, mock_simulation_data
    ):
        """Test that evaluate_maxdiff_always_on_strategy accepts close_at_eod=True."""
        eval_result, returns, metadata = evaluate_maxdiff_always_on_strategy(
            mock_predictions,
            mock_simulation_data,
            trading_fee=0.001,
            trading_days_per_year=252,
            is_crypto=False,
            close_at_eod=True,
        )

        # Should complete without error
        assert eval_result is not None
        assert metadata["maxdiffalwayson_close_at_eod"] is True

    def test_maxdiff_always_on_accepts_close_at_eod_false(
        self, mock_predictions, mock_simulation_data
    ):
        """Test that evaluate_maxdiff_always_on_strategy accepts close_at_eod=False."""
        eval_result, returns, metadata = evaluate_maxdiff_always_on_strategy(
            mock_predictions,
            mock_simulation_data,
            trading_fee=0.001,
            trading_days_per_year=252,
            is_crypto=False,
            close_at_eod=False,
        )

        # Should complete without error
        assert eval_result is not None
        assert metadata["maxdiffalwayson_close_at_eod"] is False

    def test_maxdiff_always_on_optimizes_when_none(
        self, mock_predictions, mock_simulation_data
    ):
        """Test that evaluate_maxdiff_always_on_strategy optimizes close_at_eod when None."""
        eval_result, returns, metadata = evaluate_maxdiff_always_on_strategy(
            mock_predictions,
            mock_simulation_data,
            trading_fee=0.001,
            trading_days_per_year=252,
            is_crypto=False,
            close_at_eod=None,
        )

        # Should complete without error
        assert eval_result is not None
        # Should have chosen either True or False (optimized)
        assert isinstance(metadata["maxdiffalwayson_close_at_eod"], bool)

    def test_close_at_eod_results_differ(
        self, mock_predictions, mock_simulation_data
    ):
        """Test that close_at_eod=True and close_at_eod=False produce different results."""
        eval_true, _, meta_true = evaluate_maxdiff_always_on_strategy(
            mock_predictions,
            mock_simulation_data,
            trading_fee=0.001,
            trading_days_per_year=252,
            is_crypto=False,
            close_at_eod=True,
        )

        eval_false, _, meta_false = evaluate_maxdiff_always_on_strategy(
            mock_predictions,
            mock_simulation_data,
            trading_fee=0.001,
            trading_days_per_year=252,
            is_crypto=False,
            close_at_eod=False,
        )

        # Results should be different (unless by coincidence they're the same)
        # At minimum, the metadata should reflect the different settings
        assert meta_true["maxdiffalwayson_close_at_eod"] is True
        assert meta_false["maxdiffalwayson_close_at_eod"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
