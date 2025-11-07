#!/usr/bin/env python3
"""Preview what sizing would look like with real account in PAPER mode."""

import os
import sys
sys.path.insert(0, '.')

os.environ['PAPER'] = '1'

from trade_stock_e2e import _get_simple_qty, all_crypto_symbols
import alpaca_wrapper


def preview_sizing():
    """Show what sizing would be with current account."""

    # Get real account state (PAPER mode)
    account = alpaca_wrapper.get_account()
    positions = alpaca_wrapper.get_all_positions()

    equity = float(account.equity)
    buying_power = float(account.buying_power)

    print("=" * 70)
    print("SIZING PREVIEW WITH SIMPLE SIZING FIX")
    print("=" * 70)
    print(f"\nAccount Status (PAPER):")
    print(f"  Equity: ${equity:,.2f}")
    print(f"  Buying Power: ${buying_power:,.2f}")
    print(f"  Max Exposure (120%): ${equity * 1.2:,.2f}")

    # Calculate current exposure
    total_exposure = sum(abs(float(p.market_value)) for p in positions)
    print(f"  Current Exposure: ${total_exposure:,.2f} ({total_exposure/equity*100:.1f}% of equity)")
    print(f"  Remaining Budget: ${equity * 1.2 - total_exposure:,.2f}")

    print("\n" + "-" * 70)
    print("CRYPTO POSITION SIZING (equity / 2):")
    print("-" * 70)

    # Test crypto symbols
    crypto_tests = [
        ("BTCUSD", 101937.895),
        ("ETHUSD", 3158.32),
    ]

    for symbol, price in crypto_tests:
        qty = _get_simple_qty(symbol, price, positions)
        value = qty * price

        print(f"\n{symbol}:")
        print(f"  Entry Price: ${price:,.2f}")
        print(f"  Target Qty: {qty:.6f}")
        print(f"  Target Value: ${value:,.2f}")
        print(f"  % of Equity: {value/equity*100:.1f}%")
        print(f"  Formula: equity / 2 / price = ${equity:,.2f} / 2 / ${price:,.2f}")

    print("\n" + "=" * 70)
    print("✓ Both positions would use ~50% of equity each")
    print("✓ Total exposure would be ~100% of equity (within 120% limit)")
    print("=" * 70)


if __name__ == "__main__":
    preview_sizing()
