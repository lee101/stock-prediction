#!/usr/bin/env python3
"""Test script to validate MAXDIFF simple sizing fix."""

import os
import sys
sys.path.insert(0, '.')

# Mock to avoid loading full env
os.environ.setdefault('PAPER', '1')

from trade_stock_e2e import _get_simple_qty, all_crypto_symbols
from unittest.mock import Mock, patch


def test_crypto_sizing():
    """Test crypto uses equity/2 with no leverage."""
    print("Testing crypto sizing...")

    with patch('trade_stock_e2e.alpaca_wrapper') as mock_alpaca:
        mock_alpaca.equity = 100000.0
        mock_alpaca.total_buying_power = 200000.0

        # For crypto, should use equity/2 regardless of buying power
        symbol = "BTCUSD"
        entry_price = 100000.0
        positions = []

        qty = _get_simple_qty(symbol, entry_price, positions)
        expected_value = 100000.0 / 2.0  # equity/2
        expected_qty = expected_value / entry_price  # 0.5 BTC

        print(f"  Symbol: {symbol}")
        print(f"  Equity: ${mock_alpaca.equity:,.2f}")
        print(f"  Entry Price: ${entry_price:,.2f}")
        print(f"  Expected Qty: {expected_qty:.4f}")
        print(f"  Actual Qty: {qty:.4f}")
        print(f"  Expected Value: ${expected_value:,.2f}")
        print(f"  Actual Value: ${qty * entry_price:,.2f}")

        assert abs(qty - expected_qty) < 0.001, f"Expected {expected_qty}, got {qty}"
        print("  ✓ PASSED\n")


def test_crypto_sizing_realistic():
    """Test with realistic account values."""
    print("Testing crypto sizing with realistic values...")

    with patch('trade_stock_e2e.alpaca_wrapper') as mock_alpaca:
        # Your actual account values
        mock_alpaca.equity = 95860.93
        mock_alpaca.total_buying_power = 63965.06

        # BTCUSD
        symbol = "BTCUSD"
        entry_price = 101937.895
        positions = []

        qty = _get_simple_qty(symbol, entry_price, positions)
        expected_value = 95860.93 / 2.0  # ~$47,930
        expected_qty = expected_value / entry_price

        print(f"  Symbol: {symbol}")
        print(f"  Equity: ${mock_alpaca.equity:,.2f}")
        print(f"  Entry Price: ${entry_price:,.2f}")
        print(f"  Expected Qty: {expected_qty:.6f} BTC")
        print(f"  Actual Qty: {qty:.6f} BTC")
        print(f"  Expected Value: ${expected_value:,.2f}")
        print(f"  Actual Value: ${qty * entry_price:,.2f}")

        # Note: crypto rounds down to 3 decimals
        expected_qty_rounded = int(expected_qty * 1000) / 1000.0
        assert qty == expected_qty_rounded, f"Expected {expected_qty_rounded}, got {qty}"
        print("  ✓ PASSED\n")

        # ETHUSD
        symbol = "ETHUSD"
        entry_price = 3158.32

        qty = _get_simple_qty(symbol, entry_price, positions)
        expected_value = 95860.93 / 2.0
        expected_qty = expected_value / entry_price
        expected_qty_rounded = int(expected_qty * 1000) / 1000.0

        print(f"  Symbol: {symbol}")
        print(f"  Equity: ${mock_alpaca.equity:,.2f}")
        print(f"  Entry Price: ${entry_price:,.2f}")
        print(f"  Expected Qty: {expected_qty:.6f} ETH")
        print(f"  Actual Qty: {qty:.6f} ETH")
        print(f"  Expected Value: ${expected_value:,.2f}")
        print(f"  Actual Value: ${qty * entry_price:,.2f}")

        assert qty == expected_qty_rounded, f"Expected {expected_qty_rounded}, got {qty}"
        print("  ✓ PASSED\n")


def test_stock_sizing():
    """Test stock uses buying_power*risk/2."""
    print("Testing stock sizing...")

    with patch('trade_stock_e2e.alpaca_wrapper') as mock_alpaca, \
         patch('src.portfolio_risk.get_global_risk_threshold', return_value=2.0):

        mock_alpaca.equity = 100000.0
        mock_alpaca.total_buying_power = 200000.0

        symbol = "AAPL"
        entry_price = 200.0
        positions = []

        qty = _get_simple_qty(symbol, entry_price, positions)
        # For stocks: buying_power * global_risk / 2
        # 200000 * 2.0 / 2 = 200000
        expected_value = 200000.0 * 2.0 / 2.0
        expected_qty = int(expected_value / entry_price)  # rounds down to whole number

        print(f"  Symbol: {symbol}")
        print(f"  Buying Power: ${mock_alpaca.total_buying_power:,.2f}")
        print(f"  Global Risk: 2.0")
        print(f"  Entry Price: ${entry_price:,.2f}")
        print(f"  Expected Qty: {expected_qty} shares")
        print(f"  Actual Qty: {qty} shares")
        print(f"  Expected Value: ${expected_value:,.2f}")
        print(f"  Actual Value: ${qty * entry_price:,.2f}")

        assert qty == expected_qty, f"Expected {expected_qty}, got {qty}"
        print("  ✓ PASSED\n")


def test_old_vs_new_comparison():
    """Compare old get_qty vs new _get_simple_qty for MAXDIFF strategies."""
    print("Comparing old complex sizing vs new simple sizing...")

    from src.sizing_utils import get_qty

    with patch('trade_stock_e2e.alpaca_wrapper') as mock_alpaca_trade, \
         patch('alpaca_wrapper.alpaca_wrapper') as mock_alpaca_sizing:

        # Set up both mocks with same values
        for mock in [mock_alpaca_trade, mock_alpaca_sizing]:
            mock.equity = 95860.93
            mock.total_buying_power = 63965.06

        symbol = "BTCUSD"
        entry_price = 101937.895
        positions = []

        old_qty = get_qty(symbol, entry_price, positions)
        new_qty = _get_simple_qty(symbol, entry_price, positions)

        print(f"  Symbol: {symbol}")
        print(f"  Entry Price: ${entry_price:,.2f}")
        print(f"  OLD complex sizing: {old_qty:.6f} BTC (${old_qty * entry_price:,.2f})")
        print(f"  NEW simple sizing: {new_qty:.6f} BTC (${new_qty * entry_price:,.2f})")
        print(f"  Difference: {new_qty - old_qty:.6f} BTC (${(new_qty - old_qty) * entry_price:,.2f})")
        print(f"  Improvement: {((new_qty - old_qty) / old_qty * 100) if old_qty > 0 else float('inf'):.1f}%")
        print()


if __name__ == "__main__":
    print("=" * 60)
    print("MAXDIFF SIMPLE SIZING TEST SUITE")
    print("=" * 60)
    print()

    try:
        test_crypto_sizing()
        test_crypto_sizing_realistic()
        test_stock_sizing()
        test_old_vs_new_comparison()

        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
