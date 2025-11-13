"""Symbol filtering utilities for trade execution scripts."""
from __future__ import annotations

import os
from typing import List, Optional, Sequence


def filter_symbols_by_tradable_pairs(
    symbols: Sequence[str],
    env_var_name: str = "TRADABLE_PAIRS",
    fallback_symbols: Optional[Sequence[str]] = None,
) -> List[str]:
    """
    Filter a list of symbols based on the TRADABLE_PAIRS environment variable.

    Args:
        symbols: The list of symbols to filter
        env_var_name: Name of the environment variable to read (default: "TRADABLE_PAIRS")
        fallback_symbols: Symbols to use if filter excludes everything (default: original symbols)

    Returns:
        Filtered list of symbols, or original/fallback if env var not set or filter excludes all

    Examples:
        >>> os.environ["TRADABLE_PAIRS"] = "BTCUSD,ETHUSD"
        >>> filter_symbols_by_tradable_pairs(["BTCUSD", "ETHUSD", "AAPL"])
        ['BTCUSD', 'ETHUSD']

        >>> # Case insensitive and whitespace handling
        >>> os.environ["TRADABLE_PAIRS"] = " btcusd , ETHUSD "
        >>> filter_symbols_by_tradable_pairs(["BTCUSD", "ETHUSD", "AAPL"])
        ['BTCUSD', 'ETHUSD']

        >>> # No filtering if env var not set
        >>> del os.environ["TRADABLE_PAIRS"]
        >>> filter_symbols_by_tradable_pairs(["BTCUSD", "AAPL"])
        ['BTCUSD', 'AAPL']
    """
    symbols_list = list(symbols)
    tradable_pairs_env = os.getenv(env_var_name)

    if not tradable_pairs_env:
        return symbols_list

    allowed_pairs = {
        pair.strip().upper()
        for pair in tradable_pairs_env.split(",")
        if pair.strip()
    }

    if not allowed_pairs:
        return symbols_list

    filtered = [s for s in symbols_list if s.upper() in allowed_pairs]

    if not filtered:
        # If filter excludes everything, use fallback or original
        return list(fallback_symbols) if fallback_symbols else symbols_list

    return filtered


def get_filter_info(
    original_symbols: Sequence[str],
    filtered_symbols: Sequence[str],
) -> dict:
    """
    Get information about the filtering result for logging.

    Args:
        original_symbols: The original symbol list before filtering
        filtered_symbols: The symbol list after filtering

    Returns:
        Dictionary with filtering statistics

    Examples:
        >>> get_filter_info(["A", "B", "C"], ["A", "B"])
        {'original_count': 3, 'filtered_count': 2, 'removed_count': 1, 'was_filtered': True}
    """
    original = list(original_symbols)
    filtered = list(filtered_symbols)

    return {
        "original_count": len(original),
        "filtered_count": len(filtered),
        "removed_count": len(original) - len(filtered),
        "was_filtered": set(original) != set(filtered),
    }
