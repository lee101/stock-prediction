"""Tests for symbol filtering utilities."""
from __future__ import annotations

import os

import pytest

from src.symbol_filtering import filter_symbols_by_tradable_pairs, get_filter_info


@pytest.fixture(autouse=True)
def clean_env():
    """Clean up TRADABLE_PAIRS env var before and after each test."""
    if "TRADABLE_PAIRS" in os.environ:
        del os.environ["TRADABLE_PAIRS"]
    yield
    if "TRADABLE_PAIRS" in os.environ:
        del os.environ["TRADABLE_PAIRS"]


def test_filter_symbols_no_env_var():
    """When TRADABLE_PAIRS is not set, should return original symbols."""
    symbols = ["BTCUSD", "ETHUSD", "AAPL", "MSFT"]
    result = filter_symbols_by_tradable_pairs(symbols)
    assert result == symbols


def test_filter_symbols_empty_env_var():
    """When TRADABLE_PAIRS is empty string, should return original symbols."""
    os.environ["TRADABLE_PAIRS"] = ""
    symbols = ["BTCUSD", "ETHUSD", "AAPL"]
    result = filter_symbols_by_tradable_pairs(symbols)
    assert result == symbols


def test_filter_symbols_whitespace_only():
    """When TRADABLE_PAIRS is whitespace only, should return original symbols."""
    os.environ["TRADABLE_PAIRS"] = "  ,  , "
    symbols = ["BTCUSD", "ETHUSD"]
    result = filter_symbols_by_tradable_pairs(symbols)
    assert result == symbols


def test_filter_symbols_basic():
    """Should filter to only allowed pairs."""
    os.environ["TRADABLE_PAIRS"] = "BTCUSD,ETHUSD"
    symbols = ["BTCUSD", "ETHUSD", "AAPL", "MSFT"]
    result = filter_symbols_by_tradable_pairs(symbols)
    assert result == ["BTCUSD", "ETHUSD"]


def test_filter_symbols_case_insensitive():
    """Should handle case insensitive matching."""
    os.environ["TRADABLE_PAIRS"] = "btcusd,ETHUSD,AaPl"
    symbols = ["BTCUSD", "ETHUSD", "AAPL", "MSFT"]
    result = filter_symbols_by_tradable_pairs(symbols)
    assert result == ["BTCUSD", "ETHUSD", "AAPL"]


def test_filter_symbols_with_whitespace():
    """Should handle whitespace around symbols."""
    os.environ["TRADABLE_PAIRS"] = " BTCUSD , ETHUSD , AAPL "
    symbols = ["BTCUSD", "ETHUSD", "AAPL", "MSFT"]
    result = filter_symbols_by_tradable_pairs(symbols)
    assert result == ["BTCUSD", "ETHUSD", "AAPL"]


def test_filter_symbols_preserves_order():
    """Should preserve original symbol order."""
    os.environ["TRADABLE_PAIRS"] = "AAPL,BTCUSD"
    symbols = ["BTCUSD", "ETHUSD", "AAPL", "MSFT"]
    result = filter_symbols_by_tradable_pairs(symbols)
    assert result == ["BTCUSD", "AAPL"]


def test_filter_symbols_no_matches():
    """When filter matches nothing, should return original symbols."""
    os.environ["TRADABLE_PAIRS"] = "XXXUSD,YYYUSD"
    symbols = ["BTCUSD", "ETHUSD", "AAPL"]
    result = filter_symbols_by_tradable_pairs(symbols)
    assert result == symbols


def test_filter_symbols_no_matches_with_fallback():
    """When filter matches nothing and fallback provided, should use fallback."""
    os.environ["TRADABLE_PAIRS"] = "XXXUSD,YYYUSD"
    symbols = ["BTCUSD", "ETHUSD"]
    fallback = ["AAPL", "MSFT"]
    result = filter_symbols_by_tradable_pairs(symbols, fallback_symbols=fallback)
    assert result == fallback


def test_filter_symbols_partial_match():
    """Should only include symbols that exist in original list."""
    os.environ["TRADABLE_PAIRS"] = "BTCUSD,ETHUSD,XXXUSD"
    symbols = ["BTCUSD", "ETHUSD", "AAPL"]
    result = filter_symbols_by_tradable_pairs(symbols)
    assert result == ["BTCUSD", "ETHUSD"]


def test_filter_symbols_custom_env_var():
    """Should support custom environment variable name."""
    os.environ["CUSTOM_PAIRS"] = "BTCUSD,ETHUSD"
    symbols = ["BTCUSD", "ETHUSD", "AAPL"]
    result = filter_symbols_by_tradable_pairs(symbols, env_var_name="CUSTOM_PAIRS")
    assert result == ["BTCUSD", "ETHUSD"]


def test_filter_symbols_single_symbol():
    """Should handle single symbol in env var."""
    os.environ["TRADABLE_PAIRS"] = "BTCUSD"
    symbols = ["BTCUSD", "ETHUSD", "AAPL"]
    result = filter_symbols_by_tradable_pairs(symbols)
    assert result == ["BTCUSD"]


def test_filter_symbols_duplicates_in_env():
    """Should handle duplicate symbols in env var."""
    os.environ["TRADABLE_PAIRS"] = "BTCUSD,BTCUSD,ETHUSD"
    symbols = ["BTCUSD", "ETHUSD", "AAPL"]
    result = filter_symbols_by_tradable_pairs(symbols)
    assert result == ["BTCUSD", "ETHUSD"]


def test_get_filter_info_no_filtering():
    """Should show no filtering when lists are the same."""
    original = ["A", "B", "C"]
    filtered = ["A", "B", "C"]
    info = get_filter_info(original, filtered)
    assert info == {
        "original_count": 3,
        "filtered_count": 3,
        "removed_count": 0,
        "was_filtered": False,
    }


def test_get_filter_info_with_filtering():
    """Should show filtering stats when symbols removed."""
    original = ["A", "B", "C", "D"]
    filtered = ["A", "B"]
    info = get_filter_info(original, filtered)
    assert info == {
        "original_count": 4,
        "filtered_count": 2,
        "removed_count": 2,
        "was_filtered": True,
    }


def test_get_filter_info_all_removed():
    """Should handle case where all symbols filtered out."""
    original = ["A", "B", "C"]
    filtered = []
    info = get_filter_info(original, filtered)
    assert info == {
        "original_count": 3,
        "filtered_count": 0,
        "removed_count": 3,
        "was_filtered": True,
    }


def test_get_filter_info_empty_original():
    """Should handle empty original list."""
    original = []
    filtered = []
    info = get_filter_info(original, filtered)
    assert info == {
        "original_count": 0,
        "filtered_count": 0,
        "removed_count": 0,
        "was_filtered": False,
    }
