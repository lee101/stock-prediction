"""Tests for Alpaca fractional order handling."""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from alpaca_wrapper import _get_time_in_force_for_qty


def test_get_time_in_force_for_qty():
    """Test that fractional quantities get 'day' and whole numbers get 'gtc'."""

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


def test_crypto_symbols_always_gtc():
    """Crypto always gets gtc regardless of fractional quantity.

    Regression test: open_market_order_violently and close_position_violently
    used to hardcode 'gtc' for ALL market orders.  For fractional stock
    positions that caused Alpaca to silently reject the order.  The fix passes
    qty+symbol through _get_time_in_force_for_qty which returns 'gtc' for
    crypto (24/7 markets don't support 'day') and 'day' for fractional stocks.
    """
    # Fractional crypto → still gtc (crypto markets are 24/7)
    assert _get_time_in_force_for_qty(0.00123456, "BTCUSD") == "gtc"
    assert _get_time_in_force_for_qty(0.5, "ETHUSD") == "gtc"
    assert _get_time_in_force_for_qty(12.345, "SOLUSD") == "gtc"

    # Whole crypto → gtc
    assert _get_time_in_force_for_qty(1.0, "BTCUSD") == "gtc"

    # Fractional stock → day (the broken case before the fix)
    assert _get_time_in_force_for_qty(8040.297715, "AAPL") == "day"
    assert _get_time_in_force_for_qty(0.5, "MSFT") == "day"

    # Whole stock → gtc
    assert _get_time_in_force_for_qty(10.0, "AAPL") == "gtc"


def test_market_order_tif_uses_helper_not_hardcoded_gtc():
    """Document that open_market_order_violently now derives time_in_force
    via _get_time_in_force_for_qty instead of hardcoding 'gtc'.

    We verify the helper returns the right values for the scenarios that would
    have failed before the fix: live equity market orders with fractional qty.
    """
    # Stock, fractional → must be 'day' (Alpaca rejects fractional equity market
    # orders with time_in_force='gtc')
    assert _get_time_in_force_for_qty(0.75, "NVDA") == "day"
    assert _get_time_in_force_for_qty(5.123, "TSLA") == "day"

    # Stock, whole share → gtc is fine for equity market orders
    assert _get_time_in_force_for_qty(5.0, "TSLA") == "gtc"

    # Crypto → gtc always (crypto doesn't support 'day')
    assert _get_time_in_force_for_qty(0.01, "LTCUSD") == "gtc"
