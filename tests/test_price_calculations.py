"""Unit tests for src/price_calculations.py"""

import numpy as np
import pytest

from src.price_calculations import (
    compute_close_to_extreme_movements,
    compute_price_range_pct,
    safe_price_ratio,
)


class TestComputeCloseToExtremeMovements:
    """Tests for compute_close_to_extreme_movements function."""

    def test_basic_calculation(self):
        """Test basic percentage movement calculation."""
        close = np.array([100.0, 200.0, 50.0])
        high = np.array([110.0, 210.0, 55.0])
        low = np.array([95.0, 190.0, 48.0])

        high_pct, low_pct = compute_close_to_extreme_movements(close, high, low)

        # High: |1 - 110/100| = 0.1, |1 - 210/200| = 0.05, |1 - 55/50| = 0.1
        assert np.allclose(high_pct, [0.1, 0.05, 0.1], atol=1e-6)

        # Low: |1 - 95/100| = 0.05, |1 - 190/200| = 0.05, |1 - 48/50| = 0.04
        assert np.allclose(low_pct, [0.05, 0.05, 0.04], atol=1e-6)

    def test_zero_close_price(self):
        """Test handling of zero close prices."""
        close = np.array([100.0, 0.0, 50.0])
        high = np.array([110.0, 10.0, 55.0])
        low = np.array([95.0, 5.0, 48.0])

        high_pct, low_pct = compute_close_to_extreme_movements(close, high, low)

        # When close is 0, the division result is 0 (from out=zeros_like)
        # so |1 - 0| = 1.0 (100% movement)
        assert high_pct[1] == 1.0
        assert low_pct[1] == 1.0

        # Other elements should be calculated normally
        assert np.allclose(high_pct[[0, 2]], [0.1, 0.1], atol=1e-6)
        assert np.allclose(low_pct[[0, 2]], [0.05, 0.04], atol=1e-6)

    def test_nan_handling(self):
        """Test that NaN values are replaced with 0.0."""
        close = np.array([100.0, np.nan, 50.0])
        high = np.array([110.0, 210.0, np.nan])
        low = np.array([95.0, 190.0, 48.0])

        high_pct, low_pct = compute_close_to_extreme_movements(close, high, low)

        # NaN should be replaced with 0.0
        assert not np.isnan(high_pct).any()
        assert not np.isnan(low_pct).any()

    def test_inf_handling(self):
        """Test that inf values are replaced with 0.0."""
        close = np.array([100.0, 0.0, 50.0])
        high = np.array([np.inf, 210.0, 55.0])
        low = np.array([95.0, -np.inf, 48.0])

        high_pct, low_pct = compute_close_to_extreme_movements(close, high, low)

        # Inf should be replaced with 0.0
        assert not np.isinf(high_pct).any()
        assert not np.isinf(low_pct).any()

    def test_empty_arrays(self):
        """Test handling of empty arrays."""
        close = np.array([])
        high = np.array([])
        low = np.array([])

        high_pct, low_pct = compute_close_to_extreme_movements(close, high, low)

        assert len(high_pct) == 0
        assert len(low_pct) == 0

    def test_single_element(self):
        """Test with single element arrays."""
        close = np.array([100.0])
        high = np.array([105.0])
        low = np.array([98.0])

        high_pct, low_pct = compute_close_to_extreme_movements(close, high, low)

        assert np.allclose(high_pct, [0.05], atol=1e-6)
        assert np.allclose(low_pct, [0.02], atol=1e-6)

    def test_no_movement(self):
        """Test when high/low equal close (no movement)."""
        close = np.array([100.0, 200.0])
        high = np.array([100.0, 200.0])
        low = np.array([100.0, 200.0])

        high_pct, low_pct = compute_close_to_extreme_movements(close, high, low)

        assert np.allclose(high_pct, [0.0, 0.0], atol=1e-6)
        assert np.allclose(low_pct, [0.0, 0.0], atol=1e-6)


class TestComputePriceRangePct:
    """Tests for compute_price_range_pct function."""

    def test_basic_range_calculation(self):
        """Test basic range percentage calculation."""
        high = np.array([110.0, 220.0])
        low = np.array([90.0, 180.0])
        close = np.array([100.0, 200.0])

        ranges = compute_price_range_pct(high, low, close)

        # (110-90)/100 = 0.2, (220-180)/200 = 0.2
        assert np.allclose(ranges, [0.2, 0.2], atol=1e-6)

    def test_zero_reference(self):
        """Test handling of zero reference values."""
        high = np.array([110.0, 20.0])
        low = np.array([90.0, 10.0])
        close = np.array([100.0, 0.0])

        ranges = compute_price_range_pct(high, low, close)

        # Second element should be 0.0 due to division by zero
        assert ranges[1] == 0.0
        assert np.allclose(ranges[0], 0.2, atol=1e-6)

    def test_nan_and_inf_handling(self):
        """Test NaN and inf value handling."""
        high = np.array([np.nan, np.inf, 110.0])
        low = np.array([90.0, 180.0, np.nan])
        close = np.array([100.0, 200.0, 100.0])

        ranges = compute_price_range_pct(high, low, close)

        # All NaN/inf should be replaced with 0.0
        assert not np.isnan(ranges).any()
        assert not np.isinf(ranges).any()


class TestSafePriceRatio:
    """Tests for safe_price_ratio function."""

    def test_basic_ratio(self):
        """Test basic ratio calculation."""
        nums = np.array([100.0, 200.0, 50.0])
        denoms = np.array([50.0, 100.0, 25.0])

        ratios = safe_price_ratio(nums, denoms)

        assert np.allclose(ratios, [2.0, 2.0, 2.0], atol=1e-6)

    def test_division_by_zero_uses_default(self):
        """Test division by zero returns default."""
        nums = np.array([100.0, 200.0, 50.0])
        denoms = np.array([50.0, 0.0, 25.0])

        ratios = safe_price_ratio(nums, denoms, default=1.0)

        # Middle element should be default
        assert ratios[1] == 1.0
        assert np.allclose(ratios[[0, 2]], [2.0, 2.0], atol=1e-6)

    def test_custom_default(self):
        """Test custom default value."""
        nums = np.array([100.0, 200.0])
        denoms = np.array([0.0, 0.0])

        ratios = safe_price_ratio(nums, denoms, default=99.0)

        assert np.allclose(ratios, [99.0, 99.0], atol=1e-6)

    def test_nan_handling(self):
        """Test NaN values are replaced with default."""
        nums = np.array([np.nan, 200.0, 50.0])
        denoms = np.array([50.0, np.nan, 25.0])

        ratios = safe_price_ratio(nums, denoms, default=1.0)

        # NaN should be replaced with default
        assert not np.isnan(ratios).any()
        assert ratios[0] == 1.0
        assert ratios[1] == 1.0
        assert np.allclose(ratios[2], 2.0, atol=1e-6)

    def test_inf_handling(self):
        """Test inf values are replaced with default."""
        nums = np.array([np.inf, 200.0, 50.0])
        denoms = np.array([50.0, np.nan, 25.0])  # Use nan for denominator test

        ratios = safe_price_ratio(nums, denoms, default=1.0)

        # Inf/NaN should be replaced with default
        assert not np.isinf(ratios).any()
        assert not np.isnan(ratios).any()
        assert ratios[0] == 1.0
        assert ratios[1] == 1.0

    def test_empty_arrays(self):
        """Test with empty arrays."""
        nums = np.array([])
        denoms = np.array([])

        ratios = safe_price_ratio(nums, denoms)

        assert len(ratios) == 0

    def test_negative_ratios(self):
        """Test negative ratio calculation."""
        nums = np.array([-100.0, 200.0])
        denoms = np.array([50.0, -100.0])

        ratios = safe_price_ratio(nums, denoms)

        assert np.allclose(ratios, [-2.0, -2.0], atol=1e-6)
