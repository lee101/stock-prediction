"""Tests for Alpaca fractional order handling."""

import pytest


def test_get_time_in_force_for_qty():
    """Test that fractional quantities get 'day' and whole numbers get 'gtc'."""
    # Import here to avoid importing alpaca_wrapper globally
    import sys
    from pathlib import Path

    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from alpaca_wrapper import _get_time_in_force_for_qty

    # Whole numbers should get 'gtc'
    assert _get_time_in_force_for_qty(1.0) == "gtc"
    assert _get_time_in_force_for_qty(10.0) == "gtc"
    assert _get_time_in_force_for_qty(100) == "gtc"
    assert _get_time_in_force_for_qty(4352) == "gtc"

    # Fractional numbers should get 'day'
    assert _get_time_in_force_for_qty(0.5) == "day"
    assert _get_time_in_force_for_qty(1.23) == "day"
    assert _get_time_in_force_for_qty(10.001) == "day"
    assert _get_time_in_force_for_qty(8040.297715) == "day"  # From the error log

    # Edge cases
    assert _get_time_in_force_for_qty(0.0) == "gtc"  # Zero is whole

    # Invalid input should default to 'day' (safer)
    assert _get_time_in_force_for_qty(None) == "day"
    assert _get_time_in_force_for_qty("invalid") == "day"


def test_fractional_vs_whole_detection():
    """Test edge cases for fractional detection."""
    from alpaca_wrapper import _get_time_in_force_for_qty

    # Very small fractions
    assert _get_time_in_force_for_qty(0.000001) == "day"

    # Numbers that might have floating point precision issues
    assert _get_time_in_force_for_qty(0.1 + 0.2) == "day"  # Famous 0.30000000000000004

    # Large whole numbers
    assert _get_time_in_force_for_qty(1000000.0) == "gtc"

    # Large fractional numbers
    assert _get_time_in_force_for_qty(1000000.1) == "day"
