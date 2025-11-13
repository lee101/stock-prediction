#!/usr/bin/env python3
"""
Show the worst performing symbols in maxdiff strategy.
"""

import json

def main():
    # Load the maxdiff analysis
    with open('strategytraining/top_40_maxdiff_only.json') as f:
        data = json.load(f)

    all_symbols = data['all_symbols']

    # Sort by PNL (ascending to get worst first)
    worst_symbols = sorted(all_symbols, key=lambda x: x['total_pnl'])

    print("=" * 120)
    print("WORST 40 PERFORMERS - MAXDIFF STRATEGY")
    print("=" * 120)
    print()

    print(f"{'Rank':<6} {'Symbol':<12} {'Total PNL':>15} {'Trades':>8} {'Avg Win%':>10} {'Avg Sharpe':>12}")
    print("-" * 120)

    for idx, s in enumerate(worst_symbols[:40], 1):
        print(f"{idx:<6} {s['symbol']:<12} ${s['total_pnl']:>14,.2f} {s['total_trades']:>8,.0f} "
              f"{s['avg_win_rate']*100:>9.1f}% {s['avg_sharpe']:>12.2f}")

    # Show summary of losses
    print()
    print("=" * 120)
    print("LOSS SUMMARY")
    print("=" * 120)
    print()

    total_losses = sum(s['total_pnl'] for s in all_symbols if s['total_pnl'] < 0)
    losing_symbols = [s for s in all_symbols if s['total_pnl'] < 0]
    worst_40_losses = sum(s['total_pnl'] for s in worst_symbols[:40])

    print(f"Total losses across all symbols: ${total_losses:,.2f}")
    print(f"Number of losing symbols: {len(losing_symbols)}/{len(all_symbols)}")
    print(f"Worst 40 losses: ${worst_40_losses:,.2f} ({abs(worst_40_losses/total_losses)*100:.1f}% of total losses)")
    print(f"Average loss per losing symbol: ${total_losses/len(losing_symbols):,.2f}")
    print()

    # Show worst 40 as a list
    print("=" * 120)
    print("WORST 40 SYMBOLS (for reference - AVOID THESE)")
    print("=" * 120)
    print()

    worst_40_symbols = [s['symbol'] for s in worst_symbols[:40]]
    print("Symbols list:")
    print(worst_40_symbols)
    print()

    print("Comma-separated:")
    print(", ".join(worst_40_symbols))
    print()

    print("Python list:")
    print(repr(worst_40_symbols))

    # Show detailed breakdown
    print("\n\n" + "=" * 120)
    print("WORST 40 - DETAILED BREAKDOWN")
    print("=" * 120)

    for idx, s in enumerate(worst_symbols[:40], 1):
        print(f"\n{idx}. {s['symbol']} - Total PNL: ${s['total_pnl']:,.2f} | Trades: {s['total_trades']:,.0f}")
        print(f"   Win Rate: {s['avg_win_rate']*100:.1f}% | Sharpe: {s['avg_sharpe']:.2f}")

if __name__ == "__main__":
    main()
