#!/usr/bin/env python3
"""
Aggregate PNL by stock pair across ALL strategies to find the best overall performers.
"""

import json
from collections import defaultdict

def main():
    # Load the analysis
    with open('strategytraining/pnl_by_stock_pairs_analysis.json') as f:
        data = json.load(f)

    # Aggregate by symbol across all strategies
    symbol_aggregates = defaultdict(lambda: {
        'total_pnl': 0,
        'total_trades': 0,
        'strategies_profitable': 0,
        'strategies_total': 0,
        'win_rates': [],
        'sharpe_ratios': [],
        'strategy_pnls': {}
    })

    for strategy, symbols in data.items():
        for s in symbols:
            symbol = s['symbol']
            pnl = s['total_pnl'] if s['total_pnl'] is not None else 0

            symbol_aggregates[symbol]['total_pnl'] += pnl
            symbol_aggregates[symbol]['total_trades'] += s['num_trades']
            symbol_aggregates[symbol]['strategies_total'] += 1

            if pnl > 0:
                symbol_aggregates[symbol]['strategies_profitable'] += 1

            if s['win_rate'] is not None:
                symbol_aggregates[symbol]['win_rates'].append(s['win_rate'])

            if s['sharpe_ratio'] is not None:
                symbol_aggregates[symbol]['sharpe_ratios'].append(s['sharpe_ratio'])

            symbol_aggregates[symbol]['strategy_pnls'][strategy] = pnl

    # Convert to list and calculate averages
    results = []
    for symbol, data in symbol_aggregates.items():
        avg_win_rate = sum(data['win_rates']) / len(data['win_rates']) if data['win_rates'] else 0
        avg_sharpe = sum(data['sharpe_ratios']) / len(data['sharpe_ratios']) if data['sharpe_ratios'] else 0

        results.append({
            'symbol': symbol,
            'total_pnl': data['total_pnl'],
            'total_trades': data['total_trades'],
            'avg_win_rate': avg_win_rate,
            'avg_sharpe': avg_sharpe,
            'strategies_profitable': data['strategies_profitable'],
            'strategies_total': data['strategies_total'],
            'strategy_pnls': data['strategy_pnls']
        })

    # Sort by total PNL
    results.sort(key=lambda x: x['total_pnl'], reverse=True)

    # Print full report
    print("=" * 120)
    print("STOCK PAIRS RANKED BY TOTAL PNL ACROSS ALL STRATEGIES")
    print("=" * 120)
    print()

    print(f"{'Rank':<6} {'Symbol':<12} {'Total PNL':>15} {'Trades':>8} {'Avg Win%':>10} "
          f"{'Avg Sharpe':>12} {'Prof Strats':>12}")
    print("-" * 120)

    for idx, s in enumerate(results, 1):
        print(f"{idx:<6} {s['symbol']:<12} ${s['total_pnl']:>14,.2f} {s['total_trades']:>8,.0f} "
              f"{s['avg_win_rate']*100:>9.1f}% {s['avg_sharpe']:>12.2f} "
              f"{s['strategies_profitable']}/{s['strategies_total']}")

    # Print top 40 in detail
    print("\n\n" + "=" * 120)
    print("TOP 40 STOCK PAIRS - DETAILED BREAKDOWN")
    print("=" * 120)

    for idx, s in enumerate(results[:40], 1):
        print(f"\n{idx}. {s['symbol']} - Total PNL: ${s['total_pnl']:,.2f} | Trades: {s['total_trades']:,.0f} | "
              f"Avg Win Rate: {s['avg_win_rate']*100:.1f}% | Avg Sharpe: {s['avg_sharpe']:.2f}")
        print(f"   Profitable in {s['strategies_profitable']}/{s['strategies_total']} strategies")

        # Show PNL by strategy
        strategy_pnls = [(strat, pnl) for strat, pnl in s['strategy_pnls'].items()]
        strategy_pnls.sort(key=lambda x: x[1], reverse=True)

        print("   Strategy breakdown:")
        for strat, pnl in strategy_pnls:
            symbol_indicator = "✓" if pnl > 0 else "✗"
            print(f"      {symbol_indicator} {strat:<25} ${pnl:>12,.2f}")

    # Print top 40 as a list for easy copy
    print("\n\n" + "=" * 120)
    print("TOP 40 SYMBOLS (for easy reference)")
    print("=" * 120)
    print()

    top_40_symbols = [s['symbol'] for s in results[:40]]
    print("Symbols list:")
    print(top_40_symbols)
    print()

    print("Comma-separated:")
    print(", ".join(top_40_symbols))
    print()

    print("Python list:")
    print(repr(top_40_symbols))

    # Save to JSON
    output_file = "strategytraining/top_40_stock_pairs_by_total_pnl.json"
    with open(output_file, 'w') as f:
        json.dump({
            'top_40': results[:40],
            'all_symbols': results
        }, f, indent=2)

    print(f"\n\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
