#!/usr/bin/env python3
"""Debug correlation-aware strategy."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from marketsimulator.sizing_strategies import (
    CorrelationAwareStrategy,
    MarketContext,
)
from trainingdata.load_correlation_utils import load_correlation_matrix

# Load data
corr_data = load_correlation_matrix()
print(f"Loaded correlation data with {len(corr_data['symbols'])} symbols")

# Create strategy
strategy = CorrelationAwareStrategy(corr_data=corr_data)
print(f"Created strategy: {strategy.name}")
print(f"Covariance matrix shape: {strategy.covariance_matrix.shape if strategy.covariance_matrix is not None else 'None'}")
print(f"Number of symbols: {len(strategy.symbols)}")

# Create test contexts
symbols = ['BTCUSD', 'ETHUSD', 'AAPL']
contexts = {}

for sym in symbols:
    contexts[sym] = MarketContext(
        symbol=sym,
        predicted_return=0.01,  # 1% expected return
        predicted_volatility=0.02,  # 2% volatility
        current_price=100.0,
        equity=100000,
        is_crypto=sym.endswith('USD'),
    )

print(f"\nTest contexts created for: {list(contexts.keys())}")

# Check if symbols are in the correlation matrix
print("\nChecking symbol availability:")
for sym in symbols:
    in_matrix = sym in strategy.symbols
    print(f"  {sym}: {'✓' if in_matrix else '✗ NOT FOUND'}")

# Test sizing
for sym in symbols:
    print(f"\nTesting {sym}:")
    try:
        # Check active symbols
        active_symbols = [s for s in contexts.keys() if s in strategy.symbols]
        print(f"  Active symbols in correlation matrix: {active_symbols}")

        result = strategy.calculate_size(contexts[sym], portfolio_context=contexts)
        print(f"  Position fraction: {result.position_fraction}")
        print(f"  Position value: {result.position_value}")
        print(f"  Quantity: {result.quantity}")
        print(f"  Rationale: {result.rationale}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
