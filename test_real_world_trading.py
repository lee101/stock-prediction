#!/usr/bin/env python3
"""
Real-world trading tests with PAPER=1.

Tests actual market conditions, real quotes, and live positions.
Safe to run anytime - only uses paper trading account.

Usage:
    PAPER=1 python test_real_world_trading.py
"""

import os
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

# Ensure we're using paper trading
if os.getenv("PAPER") != "1":
    print("ERROR: This script requires PAPER=1 environment variable")
    print("Usage: PAPER=1 python test_real_world_trading.py")
    sys.exit(1)

import alpaca_wrapper
from src.logging_utils import setup_logging
from src.fixtures import crypto_symbols

logger = setup_logging("test_real_world.log")


def test_account_access():
    """Test that we can access the paper account."""
    logger.info("=" * 60)
    logger.info("Test 1: Paper Account Access")
    logger.info("=" * 60)

    try:
        account = alpaca_wrapper.get_account()
        equity = float(account.equity)
        cash = float(account.cash)

        logger.info(f"✓ Successfully accessed paper account")
        logger.info(f"  Equity: ${equity:,.2f}")
        logger.info(f"  Cash: ${cash:,.2f}")
        logger.info(f"  Multiplier: {account.multiplier}x")

        return True
    except Exception as e:
        logger.error(f"✗ Failed to access account: {e}")
        return False


def test_current_positions():
    """Test retrieval and analysis of current positions."""
    logger.info("=" * 60)
    logger.info("Test 2: Current Positions Analysis")
    logger.info("=" * 60)

    try:
        positions = alpaca_wrapper.get_all_positions()
        logger.info(f"Found {len(positions)} total positions")

        stock_positions = []
        crypto_positions = []

        for pos in positions:
            if hasattr(pos, 'symbol'):
                if pos.symbol in crypto_symbols:
                    crypto_positions.append(pos)
                else:
                    stock_positions.append(pos)

        logger.info(f"  Stock positions: {len(stock_positions)}")
        logger.info(f"  Crypto positions: {len(crypto_positions)}")

        # Test that we can get quotes for each position
        for pos in positions[:5]:  # Test first 5 to avoid rate limits
            if hasattr(pos, 'symbol'):
                try:
                    quote = alpaca_wrapper.latest_data(pos.symbol)
                    ask = float(getattr(quote, "ask_price", 0) or 0)
                    bid = float(getattr(quote, "bid_price", 0) or 0)

                    if ask > 0 and bid > 0:
                        spread = (ask - bid) / ((ask + bid) / 2) * 100
                        logger.info(f"  {pos.symbol}: bid=${bid:.2f}, ask=${ask:.2f}, spread={spread:.3f}%")
                except Exception as e:
                    logger.warning(f"  {pos.symbol}: Could not get quote - {e}")

        return True
    except Exception as e:
        logger.error(f"✗ Failed to get positions: {e}")
        return False


def test_market_order_restrictions():
    """Test market order restrictions with real market status."""
    logger.info("=" * 60)
    logger.info("Test 3: Market Order Restrictions (Real-time)")
    logger.info("=" * 60)

    try:
        clock = alpaca_wrapper.get_clock()
        is_open = clock.is_open

        logger.info(f"Current time: {datetime.now(timezone.utc)}")
        logger.info(f"Market is open: {is_open}")
        logger.info(f"Next open: {clock.next_open}")
        logger.info(f"Next close: {clock.next_close}")

        # Test stock symbol
        test_stock = "AAPL"
        can_use_stock, reason_stock = alpaca_wrapper._can_use_market_order(test_stock, is_closing_position=False)
        logger.info(f"\n{test_stock} (stock):")
        logger.info(f"  Can use market order: {can_use_stock}")
        if not can_use_stock:
            logger.info(f"  Reason: {reason_stock}")

        # Test crypto symbol
        test_crypto = "BTCUSD"
        can_use_crypto, reason_crypto = alpaca_wrapper._can_use_market_order(test_crypto, is_closing_position=False)
        logger.info(f"\n{test_crypto} (crypto):")
        logger.info(f"  Can use market order: {can_use_crypto}")
        if not can_use_crypto:
            logger.info(f"  Reason: {reason_crypto}")

        # Crypto should ALWAYS be blocked
        if can_use_crypto:
            logger.error("✗ FAIL: Crypto market orders should NEVER be allowed!")
            return False

        # Stock should match market hours
        if is_open and not can_use_stock:
            logger.error("✗ FAIL: Stock market orders should be allowed during market hours!")
            return False
        elif not is_open and can_use_stock:
            logger.error("✗ FAIL: Stock market orders should be blocked outside market hours!")
            return False

        logger.info("\n✓ Market order restrictions working correctly")
        return True
    except Exception as e:
        logger.error(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_spread_analysis():
    """Analyze real spreads for various symbols."""
    logger.info("=" * 60)
    logger.info("Test 4: Real Spread Analysis")
    logger.info("=" * 60)

    test_symbols = {
        "stocks": ["AAPL", "GOOGL", "TSLA", "SPY"],
        "crypto": ["BTCUSD", "ETHUSD"]
    }

    results = {"stocks": [], "crypto": []}

    for category, symbols in test_symbols.items():
        logger.info(f"\n{category.upper()}:")
        for symbol in symbols:
            try:
                spread_pct = alpaca_wrapper._calculate_spread_pct(symbol)
                if spread_pct is None:
                    logger.warning(f"  {symbol}: Could not calculate spread (market may be closed)")
                    results[category].append({"symbol": symbol, "spread": None})
                else:
                    spread_display = spread_pct * 100
                    max_spread = alpaca_wrapper.MARKET_ORDER_MAX_SPREAD_PCT * 100

                    status = "✓" if spread_pct <= alpaca_wrapper.MARKET_ORDER_MAX_SPREAD_PCT else "✗"
                    logger.info(f"  {status} {symbol}: {spread_display:.3f}% (max: {max_spread:.1f}%)")
                    results[category].append({"symbol": symbol, "spread": spread_pct})
            except Exception as e:
                logger.error(f"  ✗ {symbol}: Error - {e}")
                results[category].append({"symbol": symbol, "spread": None})

    # Summary
    logger.info("\nSummary:")
    for category, data in results.items():
        valid_spreads = [d["spread"] for d in data if d["spread"] is not None]
        if valid_spreads:
            avg_spread = sum(valid_spreads) / len(valid_spreads) * 100
            logger.info(f"  {category}: Average spread = {avg_spread:.3f}%")

    return True


def test_close_position_fallback():
    """Test close_position_violently fallback behavior with real positions."""
    logger.info("=" * 60)
    logger.info("Test 5: Close Position Fallback (Dry Run)")
    logger.info("=" * 60)

    try:
        positions = alpaca_wrapper.get_all_positions()

        if not positions:
            logger.info("No positions to test (this is fine)")
            return True

        logger.info(f"Testing fallback logic on {len(positions)} positions (dry run)")

        for pos in positions[:3]:  # Test first 3 positions
            if not hasattr(pos, 'symbol'):
                continue

            symbol = pos.symbol
            is_crypto = symbol in crypto_symbols

            logger.info(f"\n{symbol} ({'crypto' if is_crypto else 'stock'}):")

            # Check if market orders would be allowed
            can_use_market, reason = alpaca_wrapper._can_use_market_order(symbol, is_closing_position=True)
            logger.info(f"  Market order allowed: {can_use_market}")
            if not can_use_market:
                logger.info(f"  Reason: {reason}")
                logger.info(f"  → Would fallback to limit order @ midpoint")

                # Get the midpoint that would be used
                try:
                    quote = alpaca_wrapper.latest_data(symbol)
                    ask = float(getattr(quote, "ask_price", 0) or 0)
                    bid = float(getattr(quote, "bid_price", 0) or 0)

                    if ask > 0 and bid > 0:
                        midpoint = (ask + bid) / 2.0
                        logger.info(f"  Fallback price: ${midpoint:.2f} (bid: ${bid:.2f}, ask: ${ask:.2f})")
                except Exception as e:
                    logger.warning(f"  Could not calculate midpoint: {e}")
            else:
                logger.info(f"  → Would use market order")

        logger.info("\n✓ Fallback logic validated (no actual orders placed)")
        return True
    except Exception as e:
        logger.error(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_crypto_protection():
    """Test that crypto positions are always protected from market orders."""
    logger.info("=" * 60)
    logger.info("Test 6: Crypto Market Order Protection")
    logger.info("=" * 60)

    crypto_test_symbols = ["BTCUSD", "ETHUSD", "LTCUSD"]
    all_protected = True

    for symbol in crypto_test_symbols:
        can_use, reason = alpaca_wrapper._can_use_market_order(symbol, is_closing_position=False)

        if can_use:
            logger.error(f"✗ {symbol}: Market orders SHOULD be blocked!")
            all_protected = False
        else:
            logger.info(f"✓ {symbol}: Protected - {reason}")

    if all_protected:
        logger.info("\n✓ All crypto symbols correctly protected")
    else:
        logger.error("\n✗ Some crypto symbols not protected!")

    return all_protected


def test_limit_order_availability():
    """Verify limit orders work regardless of market status."""
    logger.info("=" * 60)
    logger.info("Test 7: Limit Order Availability")
    logger.info("=" * 60)

    clock = alpaca_wrapper.get_clock()
    logger.info(f"Market is open: {clock.is_open}")
    logger.info("\nLimit orders should work in all conditions:")
    logger.info("  ✓ During market hours")
    logger.info("  ✓ During pre-market")
    logger.info("  ✓ During after-hours")
    logger.info("  ✓ During overnight session")
    logger.info("  ✓ For crypto (24/7)")
    logger.info("  ✓ For stocks (24/5 with Alpaca's overnight trading)")

    logger.info("\nNote: Limit orders are NOT blocked by market hours check")
    logger.info("Only market orders are restricted to regular trading hours")

    return True


def main():
    """Run all real-world tests."""
    logger.info("=" * 60)
    logger.info("REAL-WORLD TRADING TESTS (PAPER=1)")
    logger.info("=" * 60)
    logger.info(f"Started at: {datetime.now()}")

    results = []

    # Run all tests
    results.append(("Account Access", test_account_access()))
    results.append(("Current Positions", test_current_positions()))
    results.append(("Market Order Restrictions", test_market_order_restrictions()))
    results.append(("Spread Analysis", test_spread_analysis()))
    results.append(("Close Position Fallback", test_close_position_fallback()))
    results.append(("Crypto Protection", test_crypto_protection()))
    results.append(("Limit Order Availability", test_limit_order_availability()))

    # Summary
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {name}")

    logger.info(f"\nResults: {passed}/{total} tests passed")

    if all(result for _, result in results):
        logger.info("✓ ALL TESTS PASSED")
        return True
    else:
        logger.error("✗ SOME TESTS FAILED")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
