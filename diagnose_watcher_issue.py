#!/usr/bin/env python
"""
Diagnose why BTCUSD and ETHUSD entry watchers are stuck.

This script will:
1. Check current positions and orders
2. Test order submission for the stuck symbols
3. Check cash/buying power
4. Identify why orders are returning None
"""

import sys
import alpaca_wrapper
from src.logging_utils import setup_logging

logger = setup_logging("diagnose_watcher")

def diagnose():
    print("="*80)
    print("WATCHER DIAGNOSTIC REPORT")
    print("="*80)

    # 1. Check account status
    print("\n1. Account Status:")
    print(f"   Cash: ${alpaca_wrapper.cash:,.2f}")
    print(f"   Equity: ${alpaca_wrapper.equity:,.2f}")
    print(f"   Buying Power: ${alpaca_wrapper.total_buying_power:,.2f}")

    # 2. Check existing positions
    print("\n2. Existing Positions:")
    positions = alpaca_wrapper.get_all_positions()
    if positions:
        for pos in positions:
            print(f"   {pos.symbol}: {pos.side} qty={pos.qty} value=${pos.market_value}")
    else:
        print("   No positions")

    # 3. Check open orders
    print("\n3. Open Orders:")
    orders = alpaca_wrapper.get_orders()
    if orders:
        for order in orders:
            print(f"   {order.symbol}: {order.side} {order.qty} @ {order.limit_price} status={order.status}")
    else:
        print("   No open orders")

    # 4. Test problematic symbols
    test_symbols = [
        ("BTCUSD", "buy", 100319.31, 0.00225108),
        ("ETHUSD", "buy", 3158.31, 1.16262641),
    ]

    print("\n4. Testing Order Submission:")
    for symbol, side, limit_price, qty in test_symbols:
        notional = limit_price * qty
        print(f"\n   {symbol}:")
        print(f"      Side: {side}")
        print(f"      Quantity: {qty}")
        print(f"      Limit Price: ${limit_price:.2f}")
        print(f"      Notional: ${notional:.2f}")
        print(f"      Has Cash: {notional <= alpaca_wrapper.cash}")

        # Check if position already exists
        has_position = alpaca_wrapper.has_current_open_position(symbol, side)
        print(f"      Has Position: {has_position}")

        if has_position:
            print(f"      ⚠ Cannot submit - position already open!")
            continue

        if notional > alpaca_wrapper.cash:
            print(f"      ⚠ Cannot submit - insufficient funds!")
            continue

        # Try to understand why it would return None
        print(f"      ✓ All checks pass - order should succeed")
        print(f"      Note: NOT actually submitting order (diagnostic mode)")

    print("\n" + "="*80)
    print("ANALYSIS:")
    print("="*80)

    btc_notional = 100319.31 * 0.00225108
    eth_notional = 3158.31 * 1.16262641

    print(f"\nBTCUSD order notional: ${btc_notional:.2f}")
    print(f"ETHUSD order notional: ${eth_notional:.2f}")
    print(f"Total needed: ${btc_notional + eth_notional:.2f}")
    print(f"Cash available: ${alpaca_wrapper.cash:.2f}")

    if btc_notional + eth_notional > alpaca_wrapper.cash:
        print("\n⚠ ISSUE FOUND: Combined orders exceed available cash!")
        print("   Orders may be competing for the same cash pool.")
        print("   Work stealing coordinator should handle this, but may be failing.")
    else:
        print("\n✓ Cash is sufficient for both orders")
        print("   Issue is likely:")
        print("   - Silent exception in alpaca_wrapper.open_order_at_price_or_all()")
        print("   - API error not being logged properly")
        print("   - Crypto symbols not properly mapped")

    # Check if these are crypto symbols
    print(f"\nSymbol mapping check:")
    for symbol in ["BTCUSD", "ETHUSD"]:
        remapped = alpaca_wrapper.remap_symbols(symbol)
        print(f"   {symbol} → {remapped}")

    return 0

if __name__ == "__main__":
    sys.exit(diagnose())
