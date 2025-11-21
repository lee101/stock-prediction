#!/usr/bin/env python3
from pprint import pprint

import alpaca_wrapper

client = alpaca_wrapper.alpaca_api
orders = client.get_orders()
print(f"open orders: {len(orders)}")
for o in orders:
    print(o.symbol, o.side, o.qty, o.limit_price, o.status)
positions = alpaca_wrapper.get_all_positions()
print(f"positions: {len(positions)}")
for p in positions:
    print(p.symbol, p.qty, p.market_value, p.side)
