"""Utility functions for symbol handling and identification."""

from __future__ import annotations

from src.fixtures import all_crypto_symbols, all_stock_symbols


def is_crypto_symbol(symbol):
    """Check if a symbol is crypto, handling both formats (BTC/USD and BTCUSD).

    This function provides foolproof crypto detection regardless of whether the symbol
    uses slashes or not. It handles:
    - Direct matches: "BTCUSD" in all_crypto_symbols
    - Slash format: "BTC/USD" -> "BTCUSD" in all_crypto_symbols
    - Split format: "BTC/USD" -> check if "BTCUSD" exists
    - Stable-quote pairs: "BTCUSDT", "BTC/FDUSD", "ETH-USDC"

    Args:
        symbol: The symbol to check (e.g., "BTC/USD", "BTCUSD", "AAPL")

    Returns:
        bool: True if the symbol is crypto, False otherwise

    Examples:
        >>> is_crypto_symbol("BTCUSD")
        True
        >>> is_crypto_symbol("BTC/USD")
        True
        >>> is_crypto_symbol("AAPL")
        False
    """
    if not symbol:
        return False

    normalized = symbol.replace("/", "").replace("-", "").upper()

    # Known-stock override (prevents false positives like MU/U when supporting Binance's 'U' stable-quote).
    if normalized in all_stock_symbols:
        return False

    # Direct match (e.g., "BTCUSD")
    if normalized in all_crypto_symbols:
        return True

    # Stable-quote pairs: treat as crypto only when the base is a known crypto (e.g., BTCFDUSD -> BTCUSD exists).
    stable_quotes = ("USDT", "USDC", "FDUSD", "BUSD", "TUSD", "DAI", "U")
    stable_tokens = set(stable_quotes)
    for quote in stable_quotes:
        if normalized == quote:
            # Quote tokens themselves (USDT/FDUSD/...) are crypto unless shadowed by a known stock above.
            # NOTE: "U" is ambiguous with the Unity stock ticker; treat bare "U" as non-crypto.
            if quote == "U":
                continue
            return True
        if normalized.endswith(quote) and len(normalized) > len(quote):
            base = normalized[: -len(quote)]
            if not base:
                continue
            if base in stable_tokens:
                return True
            if f"{base}USD" in all_crypto_symbols:
                return True

    # Check if it ends with USD and the base is a known crypto
    # This handles cases like "BTC/USD" where we strip to "BTC" and check if "BTCUSD" exists
    if "/" in symbol and symbol.upper().endswith("/USD"):
        base = symbol.split("/")[0]
        if f"{base}USD".upper() in all_crypto_symbols:
            return True

    return False
