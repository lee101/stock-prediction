#!/usr/bin/env python3
"""Test script to verify is_crypto_symbol works with both formats."""

from src.symbol_utils import is_crypto_symbol


def test_is_crypto_symbol():
    """Test the is_crypto_symbol function with various formats."""
    print("Testing is_crypto_symbol function...")

    # Test cases: (symbol, expected_result, description)
    test_cases = [
        # Crypto symbols without slash (direct match)
        ("BTCUSD", True, "BTCUSD - direct match"),
        ("ETHUSD", True, "ETHUSD - direct match"),
        ("UNIUSD", True, "UNIUSD - direct match"),
        ("BNBUSD", True, "BNBUSD - direct match"),
        ("BTCUSDT", True, "BTCUSDT - stable quote"),
        ("ETHUSDT", True, "ETHUSDT - stable quote"),
        ("BTCFDUSD", True, "BTCFDUSD - stable quote"),

        # Crypto symbols with slash (should be detected)
        ("BTC/USD", True, "BTC/USD - with slash"),
        ("ETH/USD", True, "ETH/USD - with slash"),
        ("UNI/USD", True, "UNI/USD - with slash"),
        ("BNB/USD", True, "BNB/USD - with slash"),
        ("BTC/USDT", True, "BTC/USDT - stable quote slash"),
        ("BTC/FDUSD", True, "BTC/FDUSD - stable quote slash"),
        ("BTC-USDT", True, "BTC-USDT - stable quote dash"),

        # Non-crypto symbols
        ("AAPL", False, "AAPL - stock"),
        ("GOOG", False, "GOOG - stock"),
        ("TSLA", False, "TSLA - stock"),
        ("MSFT", False, "MSFT - stock"),

        # Edge cases
        ("", False, "Empty string"),
        (None, False, "None"),
        ("BTCEUR", False, "BTC with EUR (not in our list)"),
        ("BTC/EUR", False, "BTC/EUR (not in our list)"),
    ]

    passed = 0
    failed = 0

    for symbol, expected, description in test_cases:
        try:
            result = is_crypto_symbol(symbol)
            if result == expected:
                print(f"✓ PASS: {description:40s} -> {result}")
                passed += 1
            else:
                print(f"✗ FAIL: {description:40s} -> Expected {expected}, got {result}")
                failed += 1
        except Exception as e:
            print(f"✗ ERROR: {description:40s} -> {e}")
            failed += 1

    print(f"\n{'='*70}")
    print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print(f"{'='*70}")

    if failed == 0:
        print("\n✓ All tests passed! The is_crypto_symbol function is working correctly.")
        print("  - Handles both 'BTC/USD' and 'BTCUSD' formats")
        print("  - Correctly identifies non-crypto symbols")
        print("  - Handles edge cases properly")
        return True
    else:
        print(f"\n✗ {failed} test(s) failed!")
        return False


if __name__ == "__main__":
    success = test_is_crypto_symbol()
    exit(0 if success else 1)
