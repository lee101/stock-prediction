#!/usr/bin/env python3
"""Test ETHUSD bid/ask fix to verify the issue is resolved."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from alpaca.data import StockHistoricalDataClient
from data_curate_daily import download_exchange_latest_data, get_bid, get_ask, get_spread
from env_real import ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD


def test_ethusd_with_add_latest_true():
    """Test ETHUSD with ADD_LATEST=True (real API call)."""
    print("\n" + "="*80)
    print("TEST 1: ETHUSD with ADD_LATEST=True (real API)")
    print("="*80)

    # Set ADD_LATEST to True
    import data_curate_daily
    data_curate_daily.ADD_LATEST = True

    # Clear any existing data
    data_curate_daily.bids = {}
    data_curate_daily.asks = {}
    data_curate_daily.spreads = {}

    # Create client and fetch data
    client = StockHistoricalDataClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD)
    symbol = 'ETHUSD'

    print(f"\nFetching data for {symbol}...")
    try:
        result = download_exchange_latest_data(client, symbol)
        print(f"✓ download_exchange_latest_data completed successfully")
        print(f"  Data shape: {result.shape}")
        print(f"  Latest timestamp: {result.index[-1]}")
        print(f"  Latest close: {result.iloc[-1]['close']}")
    except Exception as e:
        print(f"✗ Error during download: {e}")
        return False

    # Check bid/ask
    bid = get_bid(symbol)
    ask = get_ask(symbol)
    spread = get_spread(symbol)

    print(f"\nBid/Ask Results:")
    print(f"  Bid: {bid}")
    print(f"  Ask: {ask}")
    print(f"  Spread: {spread}")

    # Verify
    if bid is None:
        print(f"\n✗ FAILED: Bid is None!")
        return False
    if ask is None:
        print(f"\n✗ FAILED: Ask is None!")
        return False
    if bid <= 0:
        print(f"\n✗ FAILED: Bid is not positive: {bid}")
        return False
    if ask <= 0:
        print(f"\n✗ FAILED: Ask is not positive: {ask}")
        return False

    spread_pct = ((ask - bid) / bid) * 100
    print(f"\n✓ SUCCESS!")
    print(f"  Bid: ${bid:,.2f}")
    print(f"  Ask: ${ask:,.2f}")
    print(f"  Spread: {spread_pct:.4f}%")

    return True


def test_ethusd_with_add_latest_false():
    """Test ETHUSD with ADD_LATEST=False (synthetic fallback)."""
    print("\n" + "="*80)
    print("TEST 2: ETHUSD with ADD_LATEST=False (synthetic fallback)")
    print("="*80)

    # Set ADD_LATEST to False
    import data_curate_daily
    data_curate_daily.ADD_LATEST = False

    # Clear any existing data
    data_curate_daily.bids = {}
    data_curate_daily.asks = {}
    data_curate_daily.spreads = {}

    # Create client and fetch data
    client = StockHistoricalDataClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD)
    symbol = 'ETHUSD'

    print(f"\nFetching data for {symbol}...")
    try:
        result = download_exchange_latest_data(client, symbol)
        print(f"✓ download_exchange_latest_data completed successfully")
        print(f"  Data shape: {result.shape}")
        print(f"  Latest timestamp: {result.index[-1]}")
        print(f"  Latest close: {result.iloc[-1]['close']}")
    except Exception as e:
        print(f"✗ Error during download: {e}")
        return False

    # Check bid/ask
    bid = get_bid(symbol)
    ask = get_ask(symbol)
    spread = get_spread(symbol)

    print(f"\nBid/Ask Results:")
    print(f"  Bid: {bid}")
    print(f"  Ask: {ask}")
    print(f"  Spread: {spread}")

    # Verify
    if bid is None:
        print(f"\n✗ FAILED: Bid is None!")
        return False
    if ask is None:
        print(f"\n✗ FAILED: Ask is None!")
        return False
    if bid <= 0:
        print(f"\n✗ FAILED: Bid is not positive: {bid}")
        return False
    if ask <= 0:
        print(f"\n✗ FAILED: Ask is not positive: {ask}")
        return False

    last_close = result.iloc[-1]['close']

    # With ADD_LATEST=False, should use synthetic (bid=ask=last_close)
    if bid != last_close:
        print(f"\n⚠ WARNING: Expected bid to equal last_close ({last_close}), got {bid}")
    if ask != last_close:
        print(f"\n⚠ WARNING: Expected ask to equal last_close ({last_close}), got {ask}")

    print(f"\n✓ SUCCESS!")
    print(f"  Bid: ${bid:,.2f}")
    print(f"  Ask: ${ask:,.2f}")
    print(f"  Last Close: ${last_close:,.2f}")
    print(f"  Spread: {spread} (should be 1.0 for 0%)")

    return True


def test_multiple_symbols():
    """Test multiple symbols to ensure no interference."""
    print("\n" + "="*80)
    print("TEST 3: Multiple symbols (ETHUSD, BTCUSD, LTCUSD)")
    print("="*80)

    import data_curate_daily
    data_curate_daily.ADD_LATEST = True

    # Clear any existing data
    data_curate_daily.bids = {}
    data_curate_daily.asks = {}
    data_curate_daily.spreads = {}

    client = StockHistoricalDataClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD)
    symbols = ['ETHUSD', 'BTCUSD', 'LTCUSD']

    results = {}
    for symbol in symbols:
        print(f"\nFetching {symbol}...", end=' ')
        try:
            download_exchange_latest_data(client, symbol)
            bid = get_bid(symbol)
            ask = get_ask(symbol)

            if bid is None or ask is None:
                print(f"✗ FAILED: Bid or Ask is None")
                results[symbol] = False
            else:
                spread_pct = ((ask - bid) / bid) * 100 if bid > 0 else 0
                print(f"✓ bid=${bid:,.2f} ask=${ask:,.2f} spread={spread_pct:.4f}%")
                results[symbol] = True
        except Exception as e:
            print(f"✗ ERROR: {e}")
            results[symbol] = False

    all_passed = all(results.values())
    print(f"\n{'✓ SUCCESS' if all_passed else '✗ FAILED'}: {sum(results.values())}/{len(results)} symbols passed")

    return all_passed


def main():
    """Run all tests."""
    print("\n" + "#"*80)
    print("# ETHUSD BID/ASK FIX VERIFICATION")
    print("#"*80)

    results = []

    # Test 1: ADD_LATEST=True
    try:
        results.append(('ADD_LATEST=True', test_ethusd_with_add_latest_true()))
    except Exception as e:
        print(f"\n✗ Test 1 crashed: {e}")
        import traceback
        traceback.print_exc()
        results.append(('ADD_LATEST=True', False))

    # Test 2: ADD_LATEST=False
    try:
        results.append(('ADD_LATEST=False', test_ethusd_with_add_latest_false()))
    except Exception as e:
        print(f"\n✗ Test 2 crashed: {e}")
        import traceback
        traceback.print_exc()
        results.append(('ADD_LATEST=False', False))

    # Test 3: Multiple symbols
    try:
        results.append(('Multiple symbols', test_multiple_symbols()))
    except Exception as e:
        print(f"\n✗ Test 3 crashed: {e}")
        import traceback
        traceback.print_exc()
        results.append(('Multiple symbols', False))

    # Final report
    print("\n" + "="*80)
    print("FINAL REPORT")
    print("="*80)

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name:20s}: {status}")

    all_passed = all(passed for _, passed in results)
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    print("="*80 + "\n")

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
