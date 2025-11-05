#!/usr/bin/env python3
"""
Test that build_portfolio() respects per-strategy returns.

This verifies BTCUSD with maxdiffalwayson gets included even though
simple_return is negative.
"""

import sys
sys.path.insert(0, '.')

# Mock the necessary config values
import trade_stock_e2e
trade_stock_e2e.SIMPLIFIED_MODE = False

# Test data mimicking BTCUSD analysis results
test_results = {
    "BTCUSD": {
        "symbol": "BTCUSD",
        "strategy": "maxdiffalwayson",
        "side": "buy",
        "avg_return": 0.016,  # ✅ Positive
        "simple_return": -0.0056,  # ❌ Negative
        "all_signals_return": -0.0079,
        "unprofit_shutdown_return": -0.011,
        "takeprofit_return": -0.0361,
        "highlow_return": -0.0177,
        "maxdiff_return": 0.0466,
        "maxdiffalwayson_return": 0.0953,  # ✅✅✅ Excellent!
        "composite_score": 0.014,
        "edge_strength": 0.0,
        "trade_blocked": False,
        "predicted_movement": -806.373,
    },
    "ETHUSD": {
        "symbol": "ETHUSD",
        "strategy": "maxdiffalwayson",
        "side": "buy",
        "avg_return": 0.024,
        "simple_return": -0.0177,
        "maxdiffalwayson_return": 0.144,  # ✅ Also good
        "composite_score": 0.022,
        "edge_strength": 0.0,
        "trade_blocked": False,
        "predicted_movement": -47.074,
    },
    "GOOG": {
        "symbol": "GOOG",
        "strategy": "simple",
        "side": "buy",
        "avg_return": 0.005,
        "simple_return": 0.003,  # ✅ Positive
        "composite_score": 0.005,
        "edge_strength": 0.001,
        "trade_blocked": False,
        "predicted_movement": 2.5,
    }
}

# Test build_portfolio()
from trade_stock_e2e import build_portfolio

print("Testing build_portfolio() with strategy-aware filtering...\n")
print("Test Results:")
for symbol, data in test_results.items():
    strat = data['strategy']
    strat_return = data.get(f"{strat}_return", 0)
    print(f"  {symbol}: strategy={strat} return={strat_return:.4f} (simple_return={data.get('simple_return', 0):.4f})")

print("\n" + "="*80)
print("Building portfolio (min=3, max=10, expanded=8)...")
print("="*80 + "\n")

picks = build_portfolio(
    test_results,
    min_positions=3,
    max_positions=10,
    max_expanded=8
)

print(f"Portfolio selected {len(picks)} symbols:\n")
for symbol, data in picks.items():
    strat = data['strategy']
    strat_return = data.get(f"{strat}_return", 0)
    print(f"  ✅ {symbol}: strategy={strat} return={strat_return:.4f}")

print("\n" + "="*80)
if "BTCUSD" in picks:
    print("✅ SUCCESS: BTCUSD included despite negative simple_return!")
    print(f"   Reason: maxdiffalwayson_return = {test_results['BTCUSD']['maxdiffalwayson_return']:.4f}")
else:
    print("❌ FAIL: BTCUSD excluded - strategy-aware filtering not working")
    print(f"   BTCUSD maxdiffalwayson_return = {test_results['BTCUSD']['maxdiffalwayson_return']:.4f}")
print("="*80)
