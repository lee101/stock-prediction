"""Utility functions for symbol handling and identification."""

from src.fixtures import all_crypto_symbols


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

    # Direct match (e.g., "BTCUSD")
    if symbol in all_crypto_symbols:
        return True

    normalized = symbol.replace("/", "").replace("-", "").upper()

    # Remove slash and try again (e.g., "BTC/USD" -> "BTCUSD")
    if normalized in all_crypto_symbols:
        return True

    stable_quotes = ("USD", "USDT", "USDC", "FDUSD", "BUSD", "TUSD", "DAI", "U")
    if normalized in stable_quotes:
        return True
    for quote in stable_quotes:
        if normalized.endswith(quote) and len(normalized) > len(quote):
            return True

    # Check if it ends with USD and the base is a known crypto
    # This handles cases like "BTC/USD" where we strip to "BTC" and check if "BTCUSD" exists
    if "/" in symbol and symbol.upper().endswith("/USD"):
        base = symbol.split("/")[0]
        if f"{base}USD".upper() in all_crypto_symbols:
            return True

    return False
