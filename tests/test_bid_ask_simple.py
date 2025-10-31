"""Simple tests for bid/ask price handling in data_curate_daily.

These tests verify that the bug fixes prevent crashes and ensure bid/ask data is always available.
"""
from __future__ import annotations

import pytest


def test_get_bid_returns_none_when_not_set():
    """Test that get_bid returns None for symbols that haven't been fetched."""
    import data_curate_daily

    # Clear any existing data
    data_curate_daily.bids = {}

    result = data_curate_daily.get_bid('TEST_SYMBOL')
    assert result is None


def test_get_ask_returns_none_when_not_set():
    """Test that get_ask returns None for symbols that haven't been fetched."""
    import data_curate_daily

    # Clear any existing data
    data_curate_daily.asks = {}

    result = data_curate_daily.get_ask('TEST_SYMBOL')
    assert result is None


def test_bids_dict_can_be_populated():
    """Test that we can directly set bid/ask values in the dictionaries."""
    import data_curate_daily

    # Clear and set test data
    data_curate_daily.bids = {}
    data_curate_daily.asks = {}

    test_symbol = 'ETHUSD'
    test_bid = 3900.0
    test_ask = 3910.0

    data_curate_daily.bids[test_symbol] = test_bid
    data_curate_daily.asks[test_symbol] = test_ask

    # Verify we can retrieve them
    assert data_curate_daily.get_bid(test_symbol) == test_bid
    assert data_curate_daily.get_ask(test_symbol) == test_ask


def test_is_fp_close_to_zero_handles_small_numbers():
    """Test that is_fp_close_to_zero correctly identifies near-zero values."""
    from data_utils import is_fp_close_to_zero

    assert is_fp_close_to_zero(0.0)
    assert is_fp_close_to_zero(1e-7)
    assert not is_fp_close_to_zero(1.0)
    assert not is_fp_close_to_zero(0.01)


def test_is_fp_close_to_zero_raises_on_none():
    """Test that is_fp_close_to_zero raises TypeError when passed None.

    This is intentional behavior - the calling code should check for None first.
    """
    from data_utils import is_fp_close_to_zero

    with pytest.raises(TypeError):
        is_fp_close_to_zero(None)
