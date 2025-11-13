#!/usr/bin/env python3
"""
Create a concise summary showing top performers for each strategy.
"""

import json
import sys

def main():
    # Load the analysis
    with open('strategytraining/pnl_by_stock_pairs_analysis.json') as f:
        data = json.load(f)

    print("=" * 100)
    print("TOP 10 STOCK PAIRS BY PNL FOR EACH STRATEGY")
    print("=" * 100)
    print()

    for strategy in sorted(data.keys()):
        symbols = data[strategy]

        print(f"\n{'='*100}")
        print(f"STRATEGY: {strategy}")
        print(f"{'='*100}")

        # Calculate summary stats
        total_pnl = sum(s['total_pnl'] for s in symbols if s['total_pnl'] is not None)
        positive_count = sum(1 for s in symbols if s['total_pnl'] and s['total_pnl'] > 0)
        total_count = sum(1 for s in symbols if s['total_pnl'] is not None)

        print(f"\nOverall: Total PNL = ${total_pnl:,.2f} | {positive_count}/{total_count} profitable symbols")
        print()

        # Show top 10 profitable
        print("TOP 10 PROFITABLE:")
        print(f"{'Rank':<6} {'Symbol':<12} {'Total PNL':>14} {'Trades':>8} {'Win Rate':>10} {'Sharpe':>8}")
        print("-" * 80)

        top_10 = [s for s in symbols if s['total_pnl'] and s['total_pnl'] > 0][:10]
        for idx, s in enumerate(top_10, 1):
            win_rate = s['win_rate'] * 100 if s['win_rate'] else 0
            sharpe = s['sharpe_ratio'] if s['sharpe_ratio'] else 0
            print(f"{idx:<6} {s['symbol']:<12} ${s['total_pnl']:>13,.2f} {s['num_trades']:>8,.0f} "
                  f"{win_rate:>9.1f}% {sharpe:>8.2f}")

        # Show worst 5
        print()
        print("WORST 5 PERFORMERS:")
        print(f"{'Rank':<6} {'Symbol':<12} {'Total PNL':>14} {'Trades':>8} {'Win Rate':>10} {'Sharpe':>8}")
        print("-" * 80)

        worst_5 = [s for s in reversed(symbols) if s['total_pnl']][:5]
        for idx, s in enumerate(reversed(worst_5), 1):
            win_rate = s['win_rate'] * 100 if s['win_rate'] else 0
            sharpe = s['sharpe_ratio'] if s['sharpe_ratio'] else 0
            print(f"{idx:<6} {s['symbol']:<12} ${s['total_pnl']:>13,.2f} {s['num_trades']:>8,.0f} "
                  f"{win_rate:>9.1f}% {sharpe:>8.2f}")
        print()

    # Create comparison table
    print("\n" + "=" * 100)
    print("STRATEGY COMPARISON - TOTAL PNL")
    print("=" * 100)
    print()

    strategy_totals = []
    for strategy, symbols in data.items():
        total_pnl = sum(s['total_pnl'] for s in symbols if s['total_pnl'] is not None)
        positive_count = sum(1 for s in symbols if s['total_pnl'] and s['total_pnl'] > 0)
        total_count = sum(1 for s in symbols if s['total_pnl'] is not None)
        avg_pnl = total_pnl / total_count if total_count > 0 else 0

        strategy_totals.append({
            'strategy': strategy,
            'total_pnl': total_pnl,
            'positive': positive_count,
            'total': total_count,
            'avg_pnl': avg_pnl
        })

    # Sort by total PNL
    strategy_totals.sort(key=lambda x: x['total_pnl'], reverse=True)

    print(f"{'Rank':<6} {'Strategy':<25} {'Total PNL':>15} {'Profitable':>12} {'Avg PNL/Symbol':>16}")
    print("-" * 100)

    for idx, s in enumerate(strategy_totals, 1):
        print(f"{idx:<6} {s['strategy']:<25} ${s['total_pnl']:>14,.2f} "
              f"{s['positive']:>5}/{s['total']:<5} ${s['avg_pnl']:>15,.2f}")

if __name__ == "__main__":
    main()
