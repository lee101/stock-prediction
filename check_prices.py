#!/usr/bin/env python3
"""Check current market prices for crypto symbols"""
import alpaca_wrapper

symbols = ['BTCUSD', 'ETHUSD', 'UNIUSD']
old_orders = {
    'BTCUSD': 100319.31,
    'ETHUSD': 3158.32,
    'UNIUSD': 4.67
}

print("Current Market Prices vs Old Order Limits:\n")
for symbol in symbols:
    try:
        quote = alpaca_wrapper.latest_data(symbol)
        bid = float(quote.bid_price)
        ask = float(quote.ask_price)
        mid = (bid + ask) / 2
        old_limit = old_orders[symbol]

        # Calculate how far off the order is
        pct_diff = ((old_limit - mid) / mid) * 100

        print(f"{symbol}:")
        print(f"  Current: ${mid:,.2f} (bid: ${bid:,.2f}, ask: ${ask:,.2f})")
        print(f"  Old Limit: ${old_limit:,.2f}")
        print(f"  Difference: {pct_diff:+.2f}% {'(too low - will fill immediately!)' if pct_diff < 0 else '(too high - won\'t fill)'}")
        print()
    except Exception as e:
        print(f"{symbol}: Error getting quote - {e}\n")
