#!/usr/bin/env python3
"""
Debug script to investigate order discrepancies between API and UI.
"""
import alpaca_wrapper
from datetime import datetime, timezone
import pytz

def main():
    print("=" * 80)
    print("ORDER DIAGNOSTIC REPORT")
    print("=" * 80)

    # Get orders from API
    try:
        orders = alpaca_wrapper.get_orders()
        print(f"\n✓ Successfully retrieved {len(orders)} orders from Alpaca API\n")
    except Exception as e:
        print(f"\n✗ Failed to retrieve orders: {e}\n")
        return

    if not orders:
        print("No open orders found.")
        return

    # Detailed order information
    for idx, order in enumerate(orders, 1):
        print(f"\n{'=' * 80}")
        print(f"ORDER {idx}/{len(orders)}")
        print(f"{'=' * 80}")

        # Basic order info
        print(f"  ID: {order.id}")
        print(f"  Symbol: {order.symbol}")
        print(f"  Side: {order.side}")
        print(f"  Qty: {order.qty}")
        print(f"  Type: {order.type}")
        print(f"  Status: {order.status}")

        # Price information
        if hasattr(order, 'limit_price') and order.limit_price:
            print(f"  Limit Price: ${order.limit_price}")
        if hasattr(order, 'stop_price') and order.stop_price:
            print(f"  Stop Price: ${order.stop_price}")
        if hasattr(order, 'filled_avg_price') and order.filled_avg_price:
            print(f"  Filled Avg Price: ${order.filled_avg_price}")

        # Quantity details
        if hasattr(order, 'filled_qty'):
            print(f"  Filled Qty: {order.filled_qty}")

        # Timestamps
        if hasattr(order, 'submitted_at') and order.submitted_at:
            submitted = order.submitted_at
            now = datetime.now(timezone.utc)
            age = now - submitted
            print(f"  Submitted: {submitted.astimezone(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S %Z')}")
            print(f"  Age: {age.days} days, {age.seconds // 3600} hours ago")

        if hasattr(order, 'updated_at') and order.updated_at:
            print(f"  Last Updated: {order.updated_at.astimezone(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S %Z')}")

        if hasattr(order, 'expired_at') and order.expired_at:
            print(f"  Expired: {order.expired_at}")

        if hasattr(order, 'canceled_at') and order.canceled_at:
            print(f"  Canceled: {order.canceled_at}")

        # Order attributes
        if hasattr(order, 'time_in_force'):
            print(f"  Time in Force: {order.time_in_force}")
        if hasattr(order, 'extended_hours'):
            print(f"  Extended Hours: {order.extended_hours}")

        # Additional status info
        if hasattr(order, 'replaced_by') and order.replaced_by:
            print(f"  ⚠ Replaced By: {order.replaced_by}")
        if hasattr(order, 'replaces') and order.replaces:
            print(f"  ⚠ Replaces: {order.replaces}")

        # Print all attributes for debugging
        print("\n  All Order Attributes:")
        for attr in dir(order):
            if not attr.startswith('_'):
                try:
                    value = getattr(order, attr)
                    if not callable(value):
                        print(f"    {attr}: {value}")
                except Exception:
                    pass

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    old_orders = []
    for order in orders:
        if hasattr(order, 'submitted_at') and order.submitted_at:
            age_days = (datetime.now(timezone.utc) - order.submitted_at).days
            if age_days > 2:
                old_orders.append((order, age_days))

    if old_orders:
        print(f"\n⚠ Found {len(old_orders)} orders older than 2 days:")
        for order, age in old_orders:
            print(f"  - {order.symbol} {order.side} {order.qty} @ {order.limit_price if hasattr(order, 'limit_price') else 'N/A'} ({age} days old)")
        print("\nThese old orders may be:")
        print("  1. Stale orders that should be canceled")
        print("  2. Orders with limit prices too far from market")
        print("  3. Orders that won't fill at current market prices")
        print("\nTo cancel these orders, run:")
        print("  python debug_orders.py --cancel-old")

    print()

if __name__ == "__main__":
    import sys
    if "--cancel-old" in sys.argv:
        print("\n⚠ Canceling old orders...")
        try:
            orders = alpaca_wrapper.get_orders()
            canceled_count = 0
            for order in orders:
                if hasattr(order, 'submitted_at') and order.submitted_at:
                    age_days = (datetime.now(timezone.utc) - order.submitted_at).days
                    if age_days > 2:
                        print(f"  Canceling: {order.symbol} {order.side} (ID: {order.id})")
                        alpaca_wrapper.cancel_order(order)
                        canceled_count += 1
            print(f"\n✓ Canceled {canceled_count} old orders")
        except Exception as e:
            print(f"\n✗ Error canceling orders: {e}")
    else:
        main()
