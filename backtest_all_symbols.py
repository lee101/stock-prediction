#!/usr/bin/env python3
"""Comprehensive backtest across all symbols in small groups to ensure each gets trades.

Tests symbols in groups of 3-4 to force Claude to pick from each group, then aggregates
results to determine which symbols to KEEP vs RULE OUT.
"""
import asyncio
import sys
sys.path.insert(0, '.')

from datetime import date, timedelta
from collections import defaultdict

from stockagent_twostage.backtest import run_backtest_async


# Symbols already tested and confirmed
ALREADY_TESTED = {
    # KEEP (profitable)
    "NFLX", "TSLA", "AVGO", "AMZN", "NVDA", "META", "AMD", "AAPL", "COST", "ADBE",
    "UNIUSD", "ETHUSD", "SKYUSD",
    # RULE OUT (unprofitable)
    "MSFT", "GOOG", "CRM",
    # INCONCLUSIVE (no trades but tested)
    "BTCUSD", "SPY", "QQQ", "BNBUSD",
}

# Remaining crypto to test
CRYPTO_GROUPS = [
    (["LINKUSD", "SOLUSD", "DOGEUSD"], "Crypto Group 1"),
    (["XRPUSD", "LTCUSD", "DOTUSD"], "Crypto Group 2"),
    (["MATICUSD", "AAVEUSD", "TRXUSD"], "Crypto Group 3"),
]

# Stock groups to test - organized by sector for fair comparison
STOCK_GROUPS = [
    # Tech stocks not yet tested
    (["ORCL", "CSCO", "QCOM"], "Tech Group 1"),
    (["TXN", "MU", "AMAT"], "Tech Group 2"),
    (["INTC", "GOOGL", "IBM"], "Tech Group 3"),

    # Financials
    (["JPM", "BAC", "WFC"], "Financials Group 1"),
    (["GS", "MS", "C"], "Financials Group 2"),
    (["AXP", "V", "MA"], "Financials Group 3"),

    # Healthcare
    (["JNJ", "UNH", "PFE"], "Healthcare Group 1"),
    (["MRK", "ABBV", "LLY"], "Healthcare Group 2"),
    (["TMO", "DHR", "ABT"], "Healthcare Group 3"),

    # Consumer
    (["WMT", "HD", "LOW"], "Consumer Group 1"),
    (["TGT", "SBUX", "MCD"], "Consumer Group 2"),
    (["NKE", "PG", "KO"], "Consumer Group 3"),

    # Industrial/Energy
    (["CAT", "BA", "GE"], "Industrial Group 1"),
    (["HON", "UPS", "DE"], "Industrial Group 2"),
    (["XOM", "CVX", "COP"], "Energy Group"),

    # ETFs
    (["IWM", "DIA", "VTI"], "ETF Group 1"),
    (["EEM", "GLD", "SLV"], "ETF Group 2"),
    (["XLE", "XLF", "XLK"], "ETF Group 3"),

    # High-growth / Volatile
    (["SHOP", "SQ", "PLTR"], "Growth Group 1"),
    (["COIN", "HOOD", "SOFI"], "Growth Group 2"),
    (["ROKU", "SNOW", "NET"], "Growth Group 3"),
    (["DDOG", "CRWD", "ZS"], "Growth Group 4"),
    (["PANW", "NOW", "TEAM"], "Growth Group 5"),
]


async def run_group(symbols, group_name, end_date, days=15):
    """Run backtest for a group of symbols"""
    start_date = end_date - timedelta(days=days)

    print(f"\n{'='*60}")
    print(f"Testing {group_name}: {symbols}")
    print(f"Period: {start_date} to {end_date}")
    print(f"{'='*60}")

    try:
        summary = await run_backtest_async(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            initial_capital=100_000,
            min_confidence=0.3,
            max_lines=200,
            use_chronos=True,
            chronos_device="cuda",
            use_thinking=False,
            max_parallel_days=5,
        )
        return summary
    except Exception as e:
        print(f"Error testing {group_name}: {e}")
        return None


def extract_results(summary, all_results):
    """Extract per-symbol PnL from backtest summary"""
    if not summary:
        return

    for day_result in summary.get("daily_results", []):
        for trade in day_result.get("trades", []):
            sym = trade.get("symbol", "UNKNOWN")
            pnl = trade.get("net_pnl", 0)
            if sym not in all_results:
                all_results[sym] = {"pnl": 0, "trades": 0, "wins": 0}
            all_results[sym]["pnl"] += pnl
            all_results[sym]["trades"] += 1
            if pnl > 0:
                all_results[sym]["wins"] += 1


async def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--crypto-only", action="store_true", help="Only test crypto symbols")
    parser.add_argument("--stocks-only", action="store_true", help="Only test stock symbols")
    parser.add_argument("--days", type=int, default=15, help="Number of days to backtest")
    parser.add_argument("--end-date", type=str, default="2025-11-30", help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    end_date = date.fromisoformat(args.end_date)
    all_results = {}

    groups_to_test = []
    if not args.stocks_only:
        groups_to_test.extend(CRYPTO_GROUPS)
    if not args.crypto_only:
        groups_to_test.extend(STOCK_GROUPS)

    print(f"Testing {len(groups_to_test)} groups over {args.days} days ending {end_date}")
    print(f"Already tested: {len(ALREADY_TESTED)} symbols")

    for symbols, group_name in groups_to_test:
        # Skip symbols already tested
        untested = [s for s in symbols if s not in ALREADY_TESTED]
        if not untested:
            print(f"\nSkipping {group_name} - all symbols already tested")
            continue

        summary = await run_group(untested, group_name, end_date, args.days)
        extract_results(summary, all_results)

    # Print final summary
    print("\n" + "=" * 80)
    print("COMPREHENSIVE BACKTEST RESULTS")
    print("=" * 80)

    sorted_results = sorted(all_results.items(), key=lambda x: x[1]["pnl"], reverse=True)
    total_pnl = 0
    keep_symbols = []
    rule_out_symbols = []

    for sym, data in sorted_results:
        pnl = data["pnl"]
        trades = data["trades"]
        wins = data["wins"]
        win_rate = wins / trades * 100 if trades > 0 else 0
        total_pnl += pnl
        status = "KEEP" if pnl > 0 else "RULE OUT"
        if pnl > 0:
            keep_symbols.append(sym)
        else:
            rule_out_symbols.append(sym)
        print(f"{sym:10} | PnL: ${pnl:>12,.2f} | Trades: {trades:>3} | Win Rate: {win_rate:>5.1f}% | {status}")

    print("-" * 80)
    print(f"{'TOTAL':10} | PnL: ${total_pnl:>12,.2f}")
    print("=" * 80)

    # Symbols that still got no trades
    all_requested = set()
    for symbols, _ in groups_to_test:
        all_requested.update(symbols)
    tested = set(all_results.keys())
    missing = all_requested - tested - ALREADY_TESTED
    if missing:
        print(f"\nSymbols still with no trades: {sorted(missing)}")

    # Summary lists
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nKEEP ({len(keep_symbols)}): {sorted(keep_symbols)}")
    print(f"\nRULE OUT ({len(rule_out_symbols)}): {sorted(rule_out_symbols)}")
    print(f"\nNO TRADES ({len(missing)}): {sorted(missing)}")


if __name__ == "__main__":
    asyncio.run(main())
