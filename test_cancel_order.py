#!/usr/bin/env python3
"""Test script to verify cancel_order works with both order objects and UUIDs."""

import os
import sys
import time

# Set PAPER mode
os.environ['PAPER'] = '1'

import alpaca_wrapper


def test_cancel_order():
    """Test canceling orders using both order objects and direct IDs."""
    print("Testing cancel_order function with PAPER=1...")

    # Get current open orders
    print("\n1. Fetching current open orders...")
    orders = alpaca_wrapper.get_orders()
    print(f"   Found {len(orders)} open orders")

    if not orders:
        print("\n   Creating a test order to cancel...")
        # Try to create a test limit order that won't fill immediately
        try:
            # Place a limit order far from current price so it stays open
            test_order = alpaca_wrapper.alpaca_api.submit_order(
                symbol="AAPL",
                qty=1,
                side="buy",
                type="limit",
                time_in_force="day",
                limit_price=1.0  # Very low price, won't fill
            )
            print(f"   Created test order: {test_order.id}")
            time.sleep(1)  # Give it a moment to register
            orders = alpaca_wrapper.get_orders()
        except Exception as e:
            print(f"   Failed to create test order: {e}")
            print("   Skipping test - no orders available")
            return

    if not orders:
        print("   Still no orders available to test")
        return

    # Test 1: Cancel using order object (old way)
    print("\n2. Testing cancel with order object...")
    test_order = orders[0]
    print(f"   Order ID: {test_order.id}")
    print(f"   Symbol: {test_order.symbol}")

    try:
        alpaca_wrapper.cancel_order(test_order)
        print("   ✓ Successfully cancelled using order object")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return

    # Wait a bit and create another test order for the second test
    time.sleep(2)
    print("\n3. Creating another test order for UUID test...")
    try:
        test_order2 = alpaca_wrapper.alpaca_api.submit_order(
            symbol="AAPL",
            qty=1,
            side="buy",
            type="limit",
            time_in_force="day",
            limit_price=1.0
        )
        order_id = test_order2.id
        print(f"   Created order: {order_id}")
        time.sleep(1)
    except Exception as e:
        print(f"   Failed to create second test order: {e}")
        print("   Checking if we have other orders to test...")
        orders = alpaca_wrapper.get_orders()
        if orders:
            order_id = orders[0].id
            print(f"   Using existing order: {order_id}")
        else:
            print("   No orders available for UUID test")
            return

    # Test 2: Cancel using UUID directly (new way - the fix)
    print("\n4. Testing cancel with UUID directly (the fix)...")
    print(f"   Order ID: {order_id}")

    try:
        alpaca_wrapper.cancel_order(order_id)
        print("   ✓ Successfully cancelled using UUID directly")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return

    print("\n✓ All tests passed! The cancel_order fix is working correctly.")
    print("  - Can cancel using order objects (backward compatibility)")
    print("  - Can cancel using UUID/string IDs (the fix)")


if __name__ == "__main__":
    try:
        test_cancel_order()
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
