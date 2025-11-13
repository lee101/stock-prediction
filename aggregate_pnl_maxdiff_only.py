#!/usr/bin/env python3
"""
Aggregate PNL by stock pair for ONLY maxdiff and maxdiffalwayson strategies.
"""

import json
from collections import defaultdict

def main():
    # Load the analysis
    with open('strategytraining/pnl_by_stock_pairs_analysis.json') as f:
        data = json.load(f)

    print("Available strategies in dataset:")
    for strategy in sorted(data.keys()):
        print(f"  - {strategy}")
    print()

    # Filter to only maxdiff strategies
    maxdiff_strategies = [s for s in data.keys() if 'maxdiff' in s.lower()]

    print(f"Maxdiff strategies found: {maxdiff_strategies}")
    print()

    if not maxdiff_strategies:
        print("No maxdiff strategies found in the dataset!")
        return

    # Aggregate by symbol across maxdiff strategies only
    symbol_aggregates = defaultdict(lambda: {
        'total_pnl': 0,
        'total_trades': 0,
        'strategies_profitable': 0,
        'strategies_total': 0,
        'win_rates': [],
        'sharpe_ratios': [],
        'strategy_pnls': {}
    })

    for strategy in maxdiff_strategies:
        symbols = data[strategy]
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
    print(f"STOCK PAIRS RANKED BY TOTAL PNL - MAXDIFF STRATEGIES ONLY ({len(maxdiff_strategies)} strategies)")
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
    print("TOP 40 STOCK PAIRS - MAXDIFF STRATEGIES DETAILED BREAKDOWN")
    print("=" * 120)

    for idx, s in enumerate(results[:40], 1):
        print(f"\n{idx}. {s['symbol']} - Total PNL: ${s['total_pnl']:,.2f} | Trades: {s['total_trades']:,.0f} | "
              f"Avg Win Rate: {s['avg_win_rate']*100:.1f}% | Avg Sharpe: {s['avg_sharpe']:.2f}")
        print(f"   Profitable in {s['strategies_profitable']}/{s['strategies_total']} maxdiff strategies")

        # Show PNL by strategy
        strategy_pnls = [(strat, pnl) for strat, pnl in s['strategy_pnls'].items()]
        strategy_pnls.sort(key=lambda x: x[1], reverse=True)

        print("   Maxdiff strategy breakdown:")
        for strat, pnl in strategy_pnls:
            symbol_indicator = "✓" if pnl > 0 else "✗"
            print(f"      {symbol_indicator} {strat:<30} ${pnl:>12,.2f}")

    # Print top 40 as a list for easy copy
    print("\n\n" + "=" * 120)
    print("TOP 40 SYMBOLS FOR MAXDIFF STRATEGIES (for easy reference)")
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

    # Summary stats
    print("\n\n" + "=" * 120)
    print("SUMMARY STATISTICS - MAXDIFF STRATEGIES")
    print("=" * 120)
    print()

    total_pnl = sum(s['total_pnl'] for s in results)
    total_trades = sum(s['total_trades'] for s in results)
    profitable_symbols = sum(1 for s in results if s['total_pnl'] > 0)
    top_40_pnl = sum(s['total_pnl'] for s in results[:40])

    print(f"Total PNL (all symbols): ${total_pnl:,.2f}")
    print(f"Total PNL (top 40): ${top_40_pnl:,.2f} ({top_40_pnl/total_pnl*100:.1f}% of total)")
    print(f"Total trades: {total_trades:,}")
    print(f"Profitable symbols: {profitable_symbols}/{len(results)}")
    print(f"Top 40 avg PNL per symbol: ${top_40_pnl/40:,.2f}")

    # Save to JSON
    output_file = "strategytraining/top_40_maxdiff_only.json"
    with open(output_file, 'w') as f:
        json.dump({
            'strategies_included': maxdiff_strategies,
            'top_40': results[:40],
            'all_symbols': results,
            'summary': {
                'total_pnl': total_pnl,
                'top_40_pnl': top_40_pnl,
                'total_trades': total_trades,
                'profitable_symbols': profitable_symbols,
                'total_symbols': len(results)
            }
        }, f, indent=2)

    print(f"\n\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
