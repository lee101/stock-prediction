#!/usr/bin/env python3
"""Integration test for PAXGUSD data fetching from Alpaca API.

This test verifies that we can successfully fetch both live and historical
data for PAXGUSD using the Alpaca Paper trading API.

Run with: PAPER=1 pytest tests/test_paxgusd_data_integration.py -v
"""
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Ensure we're using PAPER mode
os.environ['PAPER'] = '1'

from alpaca_wrapper import (
    latest_data,
    download_symbol_history,
    crypto_client,
)
from alpaca.data import CryptoLatestQuoteRequest, CryptoBarsRequest, TimeFrame, TimeFrameUnit


class TestPAXGUSDDataFetching:
    """Test suite for PAXGUSD data fetching functionality."""

    @pytest.mark.integration
    def test_paxgusd_latest_quote(self):
        """Test fetching latest quote for PAXGUSD."""
        print("\n" + "="*80)
        print("Testing PAXGUSD latest quote...")
        print("="*80)

        try:
            # PAXGUSD should be remapped to PAX/USD for Alpaca crypto API
            symbol = "PAXGUSD"
            quote = latest_data(symbol)

            print(f"\nSuccessfully fetched quote for {symbol}:")
            print(f"  Bid Price: ${getattr(quote, 'bid_price', None)}")
            print(f"  Ask Price: ${getattr(quote, 'ask_price', None)}")
            print(f"  Timestamp: {getattr(quote, 'timestamp', None)}")

            # Verify we got valid data
            bid_price = float(getattr(quote, 'bid_price', 0) or 0)
            ask_price = float(getattr(quote, 'ask_price', 0) or 0)

            assert bid_price > 0, f"Bid price should be positive, got {bid_price}"
            assert ask_price > 0, f"Ask price should be positive, got {ask_price}"
            assert ask_price >= bid_price, f"Ask ({ask_price}) should be >= bid ({bid_price})"

            # Calculate spread
            if bid_price > 0:
                spread_pct = ((ask_price - bid_price) / bid_price) * 100
                print(f"  Spread: {spread_pct:.4f}%")
                assert spread_pct < 10, f"Spread seems too high: {spread_pct:.4f}%"

            print("\n✓ Latest quote test PASSED")
            return True

        except Exception as e:
            print(f"\n✗ Latest quote test FAILED: {e}")
            import traceback
            traceback.print_exc()
            pytest.fail(f"Failed to fetch latest quote for PAXGUSD: {e}")

    @pytest.mark.integration
    def test_paxgusd_historical_bars(self):
        """Test fetching historical bars for PAXGUSD."""
        print("\n" + "="*80)
        print("Testing PAXGUSD historical bars...")
        print("="*80)

        try:
            # Test with direct crypto client
            end_dt = datetime.now(timezone.utc)
            start_dt = end_dt - timedelta(days=7)  # Last 7 days

            # PAXGUSD needs to be remapped to PAX/USD
            request = CryptoBarsRequest(
                symbol_or_symbols="PAX/USD",  # Alpaca format
                timeframe=TimeFrame(1, TimeFrameUnit.Hour),
                start=start_dt,
                end=end_dt,
            )

            bars = crypto_client.get_crypto_bars(request).df

            print(f"\nSuccessfully fetched historical bars:")
            print(f"  Symbol: PAXGUSD (PAX/USD)")
            print(f"  Date range: {start_dt.date()} to {end_dt.date()}")
            print(f"  Total bars: {len(bars)}")

            if len(bars) > 0:
                print(f"  First bar: {bars.index[0]}")
                print(f"  Last bar: {bars.index[-1]}")
                print(f"\nSample data (first 3 rows):")
                print(bars.head(3))

                # Verify data structure
                assert 'open' in bars.columns or 'open' in [c.lower() for c in bars.columns], "Should have 'open' column"
                assert 'high' in bars.columns or 'high' in [c.lower() for c in bars.columns], "Should have 'high' column"
                assert 'low' in bars.columns or 'low' in [c.lower() for c in bars.columns], "Should have 'low' column"
                assert 'close' in bars.columns or 'close' in [c.lower() for c in bars.columns], "Should have 'close' column"
                assert 'volume' in bars.columns or 'volume' in [c.lower() for c in bars.columns], "Should have 'volume' column"

                print("\n✓ Historical bars test PASSED")
            else:
                print("\n⚠ WARNING: No historical bars returned")
                pytest.skip("No historical data available for PAXGUSD")

            return True

        except Exception as e:
            print(f"\n✗ Historical bars test FAILED: {e}")
            import traceback
            traceback.print_exc()
            pytest.fail(f"Failed to fetch historical bars for PAXGUSD: {e}")

    @pytest.mark.integration
    def test_paxgusd_download_symbol_history(self):
        """Test the download_symbol_history wrapper function for PAXGUSD."""
        print("\n" + "="*80)
        print("Testing PAXGUSD via download_symbol_history()...")
        print("="*80)

        try:
            symbol = "PAXGUSD"
            end_dt = datetime.now(timezone.utc)
            start_dt = end_dt - timedelta(days=30)  # Last 30 days

            df = download_symbol_history(
                symbol=symbol,
                start=start_dt,
                end=end_dt,
                include_latest=True,
                timeframe=TimeFrame(1, TimeFrameUnit.Day)
            )

            print(f"\nSuccessfully fetched history for {symbol}:")
            print(f"  Total rows: {len(df)}")

            if len(df) > 0:
                print(f"  Date range: {df.index[0]} to {df.index[-1]}")
                print(f"\nColumns: {list(df.columns)}")
                print(f"\nSample data (first 3 rows):")
                print(df.head(3))
                print(f"\nSample data (last 3 rows):")
                print(df.tail(3))

                # Verify data
                assert 'close' in df.columns, "Should have 'close' column"
                assert len(df) > 0, "Should have at least some data"
                assert df['close'].notna().any(), "Should have non-null close prices"

                # Check for reasonable prices
                close_prices = df['close'].dropna()
                if len(close_prices) > 0:
                    avg_price = close_prices.mean()
                    print(f"\nAverage close price: ${avg_price:.2f}")
                    assert avg_price > 0, f"Average price should be positive, got {avg_price}"

                print("\n✓ download_symbol_history test PASSED")
            else:
                print("\n⚠ WARNING: No data returned")
                pytest.skip("No historical data available for PAXGUSD")

            return True

        except Exception as e:
            print(f"\n✗ download_symbol_history test FAILED: {e}")
            import traceback
            traceback.print_exc()
            pytest.fail(f"Failed to download symbol history for PAXGUSD: {e}")

    @pytest.mark.integration
    def test_paxgusd_symbol_remapping(self):
        """Test that PAXGUSD is correctly identified as crypto and remapped."""
        print("\n" + "="*80)
        print("Testing PAXGUSD symbol remapping...")
        print("="*80)

        from src.stock_utils import remap_symbols

        # Test symbol remapping
        symbol = "PAXGUSD"
        remapped = remap_symbols(symbol)

        print(f"\nOriginal symbol: {symbol}")
        print(f"Remapped symbol: {remapped}")

        # PAXGUSD should be remapped to PAX/USD for Alpaca crypto API
        expected = "PAX/USD"
        assert remapped == expected, f"Expected '{expected}', got '{remapped}'"

        print(f"\n✓ Symbol remapping test PASSED")


def main():
    """Run the tests manually."""
    import sys

    print("\n" + "="*80)
    print("PAXGUSD Data Integration Test Suite")
    print("Mode: PAPER=1 (Paper Trading)")
    print("="*80)

    test_instance = TestPAXGUSDDataFetching()

    tests = [
        ("Latest Quote", test_instance.test_paxgusd_latest_quote),
        ("Historical Bars", test_instance.test_paxgusd_historical_bars),
        ("Download Symbol History", test_instance.test_paxgusd_download_symbol_history),
        ("Symbol Remapping", test_instance.test_paxgusd_symbol_remapping),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            print(f"\n\nRunning: {test_name}")
            print("-" * 80)
            test_func()
            results.append((test_name, "PASSED"))
        except Exception as e:
            results.append((test_name, f"FAILED: {e}"))

    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    for test_name, result in results:
        status = "✓" if result == "PASSED" else "✗"
        print(f"{status} {test_name}: {result}")

    # Exit with appropriate code
    failed = [r for r in results if r[1] != "PASSED"]
    if failed:
        print(f"\n{len(failed)} test(s) failed")
        sys.exit(1)
    else:
        print(f"\nAll {len(results)} tests passed!")
        sys.exit(0)


if __name__ == '__main__':
    main()
