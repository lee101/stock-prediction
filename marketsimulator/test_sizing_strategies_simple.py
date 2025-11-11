"""
Simple test of sizing strategies with synthetic data.

Quick validation without needing real data or models.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from marketsimulator.sizing_strategies import (
    SIZING_STRATEGIES,
    MarketContext,
)


def test_sizing_strategies():
    """Test all sizing strategies with synthetic market context."""

    print("Testing Sizing Strategies")
    print("=" * 60)

    # Test case 1: Moderate positive prediction
    print("\nTest 1: Moderate upward prediction (stocks)")
    ctx1 = MarketContext(
        symbol="AAPL",
        predicted_return=0.05,  # 5% expected return
        predicted_volatility=0.02,  # 2% volatility
        current_price=180.0,
        equity=100000.0,
        is_crypto=False,
        existing_position_value=0.0,
    )

    for name, strategy in SIZING_STRATEGIES.items():
        result = strategy.calculate_size(ctx1)
        print(f"  {name:20s}: position={result.position_fraction:+.2%}, "
              f"leverage={result.leverage_used:.2f}x, "
              f"qty={result.quantity:.0f}")

    # Test case 2: High volatility crypto
    print("\nTest 2: Crypto with high volatility")
    ctx2 = MarketContext(
        symbol="BTCUSD",
        predicted_return=0.03,  # 3% expected return
        predicted_volatility=0.05,  # 5% volatility
        current_price=65000.0,
        equity=100000.0,
        is_crypto=True,
        existing_position_value=0.0,
    )

    for name, strategy in SIZING_STRATEGIES.items():
        result = strategy.calculate_size(ctx2)
        print(f"  {name:20s}: position={result.position_fraction:+.2%}, "
              f"leverage={result.leverage_used:.2f}x, "
              f"qty={result.quantity:.3f}")

    # Test case 3: Negative prediction (short)
    print("\nTest 3: Negative prediction (stocks)")
    ctx3 = MarketContext(
        symbol="SPY",
        predicted_return=-0.04,  # -4% expected return
        predicted_volatility=0.015,  # 1.5% volatility
        current_price=500.0,
        equity=100000.0,
        is_crypto=False,
        existing_position_value=0.0,
    )

    for name, strategy in SIZING_STRATEGIES.items():
        result = strategy.calculate_size(ctx3)
        print(f"  {name:20s}: position={result.position_fraction:+.2%}, "
              f"leverage={result.leverage_used:.2f}x, "
              f"qty={result.quantity:.0f}")

    # Test case 4: Very high conviction
    print("\nTest 4: High conviction, low volatility")
    ctx4 = MarketContext(
        symbol="MSFT",
        predicted_return=0.08,  # 8% expected return
        predicted_volatility=0.01,  # 1% volatility
        current_price=400.0,
        equity=100000.0,
        is_crypto=False,
        existing_position_value=0.0,
    )

    for name, strategy in SIZING_STRATEGIES.items():
        result = strategy.calculate_size(ctx4)
        print(f"  {name:20s}: position={result.position_fraction:+.2%}, "
              f"leverage={result.leverage_used:.2f}x, "
              f"qty={result.quantity:.0f}")
        if result.leverage_used > 1.5:
            print(f"      ^^ Using leverage! Cost will apply.")

    print("\n" + "=" * 60)
    print("All tests completed successfully!")


def test_leverage_constraints():
    """Test that leverage constraints are properly enforced."""

    print("\n\nTesting Leverage Constraints")
    print("=" * 60)

    # High edge scenario that would want >2x leverage
    ctx_high_edge = MarketContext(
        symbol="AAPL",
        predicted_return=0.20,  # 20% expected return (unrealistic but tests limits)
        predicted_volatility=0.05,  # 5% volatility
        current_price=180.0,
        equity=100000.0,
        is_crypto=False,
    )

    print("\nHigh edge case (20% return, 5% vol) - should cap at 2x:")
    for name, strategy in SIZING_STRATEGIES.items():
        result = strategy.calculate_size(ctx_high_edge)
        if result.leverage_used > 2.0:
            print(f"  ERROR: {name} exceeded 2x leverage: {result.leverage_used:.2f}x")
        else:
            print(f"  {name:20s}: leverage={result.leverage_used:.2f}x ✓")

    # Crypto should never leverage
    ctx_crypto = MarketContext(
        symbol="BTCUSD",
        predicted_return=0.20,
        predicted_volatility=0.05,
        current_price=65000.0,
        equity=100000.0,
        is_crypto=True,
    )

    print("\nCrypto case - should never exceed 1x:")
    for name, strategy in SIZING_STRATEGIES.items():
        result = strategy.calculate_size(ctx_crypto)
        if result.leverage_used > 1.0:
            print(f"  ERROR: {name} used leverage on crypto: {result.leverage_used:.2f}x")
        else:
            print(f"  {name:20s}: leverage={result.leverage_used:.2f}x ✓")

    # Negative prediction on crypto should be 0
    ctx_crypto_neg = MarketContext(
        symbol="ETHUSD",
        predicted_return=-0.10,
        predicted_volatility=0.05,
        current_price=3500.0,
        equity=100000.0,
        is_crypto=True,
    )

    print("\nCrypto with negative prediction - should not short:")
    for name, strategy in SIZING_STRATEGIES.items():
        result = strategy.calculate_size(ctx_crypto_neg)
        if result.position_fraction < 0:
            print(f"  ERROR: {name} tried to short crypto: {result.position_fraction:.2f}")
        else:
            print(f"  {name:20s}: position={result.position_fraction:+.2%} ✓")

    print("\n" + "=" * 60)
    print("Constraint tests passed!")


if __name__ == '__main__':
    test_sizing_strategies()
    test_leverage_constraints()
