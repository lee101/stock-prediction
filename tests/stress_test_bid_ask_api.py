#!/usr/bin/env python3
"""Stress test for bid/ask API to identify which symbols have issues.

This script tests the latest_data API for various stock and crypto symbols
to understand patterns in failures, zero values, and successful responses.
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from alpaca_wrapper import latest_data
from src.fixtures import crypto_symbols


# Test symbols - mix of popular stocks and crypto
TEST_STOCKS = [
    'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA',
    'NVDA', 'META', 'NFLX', 'AMD', 'COIN',
    'SPY', 'QQQ', 'DIA', 'IWM', 'VTI',
]

TEST_CRYPTO = [
    'BTCUSD', 'ETHUSD', 'LTCUSD', 'UNIUSD', 'PAXGUSD',
    'ADAUSD', 'SOLUSD', 'DOGEUSD', 'MATICUSD', 'AVAXUSD',
]

# Extended test - use all crypto symbols from fixtures
USE_ALL_CRYPTO = True


def test_symbol(symbol: str) -> dict[str, Any]:
    """Test a single symbol and return detailed results."""
    result = {
        'symbol': symbol,
        'status': 'unknown',
        'bid': None,
        'ask': None,
        'error': None,
        'timestamp': None,
        'response_time_ms': None,
    }

    start_time = time.time()
    try:
        quote = latest_data(symbol)
        response_time = (time.time() - start_time) * 1000
        result['response_time_ms'] = round(response_time, 2)

        # Extract bid/ask
        bid = float(getattr(quote, 'bid_price', 0) or 0)
        ask = float(getattr(quote, 'ask_price', 0) or 0)
        result['bid'] = bid
        result['ask'] = ask

        # Get timestamp if available
        if hasattr(quote, 'timestamp'):
            result['timestamp'] = quote.timestamp

        # Categorize the result
        if bid == 0 and ask == 0:
            result['status'] = 'both_zero'
        elif bid == 0:
            result['status'] = 'bid_zero'
        elif ask == 0:
            result['status'] = 'ask_zero'
        elif bid > 0 and ask > 0:
            spread_pct = ((ask - bid) / bid) * 100
            result['spread_pct'] = round(spread_pct, 4)
            result['status'] = 'success'
        else:
            result['status'] = 'invalid'

    except Exception as e:
        response_time = (time.time() - start_time) * 1000
        result['response_time_ms'] = round(response_time, 2)
        result['status'] = 'error'
        result['error'] = str(e)

    return result


def run_stress_test(symbols: list[str], delay_ms: int = 100) -> dict[str, Any]:
    """Run stress test on all symbols with optional delay between requests."""
    results = []
    stats = defaultdict(int)

    print(f"\n{'='*80}")
    print(f"Starting stress test at {datetime.now(timezone.utc).isoformat()}")
    print(f"Testing {len(symbols)} symbols with {delay_ms}ms delay between requests")
    print(f"{'='*80}\n")

    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] Testing {symbol}...", end=' ')
        sys.stdout.flush()

        result = test_symbol(symbol)
        results.append(result)
        stats[result['status']] += 1

        # Print status
        status_emoji = {
            'success': '✓',
            'both_zero': '✗',
            'bid_zero': '⚠',
            'ask_zero': '⚠',
            'error': '✗',
            'invalid': '?',
        }
        emoji = status_emoji.get(result['status'], '?')
        print(f"{emoji} {result['status']}", end='')

        if result['status'] == 'success' and 'spread_pct' in result:
            print(f" (spread: {result['spread_pct']}%)", end='')
        elif result['status'] == 'error':
            print(f" ({result['error'][:50]})", end='')

        print(f" [{result['response_time_ms']}ms]")

        # Delay between requests to avoid rate limiting
        if i < len(symbols) and delay_ms > 0:
            time.sleep(delay_ms / 1000)

    return {
        'results': results,
        'stats': dict(stats),
        'total': len(symbols),
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }


def print_report(test_data: dict[str, Any]):
    """Print a detailed report of the stress test results."""
    results = test_data['results']
    stats = test_data['stats']
    total = test_data['total']

    print(f"\n{'='*80}")
    print("STRESS TEST REPORT")
    print(f"{'='*80}\n")

    # Overall statistics
    print("Overall Statistics:")
    print(f"  Total symbols tested: {total}")
    success_count = stats.get('success', 0)
    success_rate = (success_count / total * 100) if total > 0 else 0
    print(f"  Successful: {success_count} ({success_rate:.1f}%)")
    print(f"  Both zero: {stats.get('both_zero', 0)}")
    print(f"  Bid zero: {stats.get('bid_zero', 0)}")
    print(f"  Ask zero: {stats.get('ask_zero', 0)}")
    print(f"  Errors: {stats.get('error', 0)}")
    print(f"  Invalid: {stats.get('invalid', 0)}")

    # Average response time
    response_times = [r['response_time_ms'] for r in results if r['response_time_ms'] is not None]
    if response_times:
        avg_response = sum(response_times) / len(response_times)
        print(f"  Avg response time: {avg_response:.2f}ms")

    # Failed symbols
    print("\n" + "-"*80)
    failed = [r for r in results if r['status'] != 'success']
    if failed:
        print(f"\nFailed Symbols ({len(failed)}):")
        for r in failed:
            print(f"  {r['symbol']:12s} - {r['status']:12s}", end='')
            if r['error']:
                print(f" - {r['error'][:60]}")
            elif r['status'] in ['both_zero', 'bid_zero', 'ask_zero']:
                print(f" - bid={r['bid']}, ask={r['ask']}")
            else:
                print()
    else:
        print("\n✓ All symbols returned valid bid/ask data!")

    # Successful symbols with spreads
    print("\n" + "-"*80)
    successful = [r for r in results if r['status'] == 'success']
    if successful:
        print(f"\nSuccessful Symbols ({len(successful)}):")
        # Sort by spread percentage
        successful_sorted = sorted(successful, key=lambda x: x.get('spread_pct', 0))

        for r in successful_sorted[:10]:  # Show first 10
            spread = r.get('spread_pct', 0)
            print(f"  {r['symbol']:12s} - bid={r['bid']:10.4f} ask={r['ask']:10.4f} spread={spread:.4f}%")

        if len(successful) > 10:
            print(f"  ... and {len(successful) - 10} more")

        # Spread statistics
        spreads = [r['spread_pct'] for r in successful]
        print(f"\n  Spread statistics:")
        print(f"    Min: {min(spreads):.4f}%")
        print(f"    Max: {max(spreads):.4f}%")
        print(f"    Avg: {sum(spreads)/len(spreads):.4f}%")

    # Pattern analysis
    print("\n" + "-"*80)
    print("\nPattern Analysis:")

    # Analyze by asset type
    crypto_results = [r for r in results if 'USD' in r['symbol'] or r['symbol'] in crypto_symbols]
    stock_results = [r for r in results if r not in crypto_results]

    if crypto_results:
        crypto_success = sum(1 for r in crypto_results if r['status'] == 'success')
        crypto_total = len(crypto_results)
        crypto_rate = (crypto_success / crypto_total * 100) if crypto_total > 0 else 0
        print(f"  Crypto: {crypto_success}/{crypto_total} successful ({crypto_rate:.1f}%)")

    if stock_results:
        stock_success = sum(1 for r in stock_results if r['status'] == 'success')
        stock_total = len(stock_results)
        stock_rate = (stock_success / stock_total * 100) if stock_total > 0 else 0
        print(f"  Stocks: {stock_success}/{stock_total} successful ({stock_rate:.1f}%)")

    # Most common errors
    errors = [r['error'] for r in results if r['error']]
    if errors:
        error_counts = defaultdict(int)
        for err in errors:
            # Group similar errors
            if 'not found' in err.lower() or '404' in err:
                error_counts['not_found'] += 1
            elif 'rate limit' in err.lower() or '429' in err:
                error_counts['rate_limit'] += 1
            elif 'unauthorized' in err.lower() or '401' in err:
                error_counts['unauthorized'] += 1
            elif 'timeout' in err.lower():
                error_counts['timeout'] += 1
            else:
                error_counts['other'] += 1

        print(f"\n  Error types:")
        for err_type, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            print(f"    {err_type}: {count}")

    print(f"\n{'='*80}\n")


def main():
    """Run the stress test."""
    import argparse

    parser = argparse.ArgumentParser(description='Stress test bid/ask API')
    parser.add_argument('--stocks-only', action='store_true', help='Test stocks only')
    parser.add_argument('--crypto-only', action='store_true', help='Test crypto only')
    parser.add_argument('--delay', type=int, default=100, help='Delay between requests in ms (default: 100)')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to test')
    args = parser.parse_args()

    # Build symbol list
    symbols = []
    if args.symbols:
        symbols = args.symbols
    else:
        if not args.crypto_only:
            symbols.extend(TEST_STOCKS)
        if not args.stocks_only:
            if USE_ALL_CRYPTO:
                symbols.extend(sorted(set(crypto_symbols)))
            else:
                symbols.extend(TEST_CRYPTO)

    # Remove duplicates while preserving order
    seen = set()
    symbols = [s for s in symbols if not (s in seen or seen.add(s))]

    # Run the test
    test_data = run_stress_test(symbols, delay_ms=args.delay)

    # Print report
    print_report(test_data)

    # Save results to file
    import json
    output_file = Path(__file__).parent / 'bid_ask_stress_test_results.json'
    with open(output_file, 'w') as f:
        json.dump(test_data, f, indent=2, default=str)
    print(f"Full results saved to: {output_file}")


if __name__ == '__main__':
    main()
