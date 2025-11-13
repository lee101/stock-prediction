#!/usr/bin/env python3
"""
Test top 5 sizing strategies from fast precomputed test in marketsimulator.

Based on fast test results:
- VolAdjusted_10pct: Best Sharpe (2.12)
- VolAdjusted_15pct: 2nd best Sharpe (2.02)
- Kelly_50pct: Best return (4909%)
- Fixed_25pct: 3rd best Sharpe (1.93)
- Naive_50pct: Baseline (1.42 Sharpe)

Usage:
    python experiments/test_top5_sizing_strategies.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.test_sizing_strategies_comparison import (
    SimplifiedBacktester,
    StrategyResult,
)
from marketsimulator.sizing_strategies import (
    FixedFractionStrategy,
    KellyStrategy,
    VolatilityAdjustedStrategy,
)
from trainingdata.load_correlation_utils import load_correlation_matrix

import pandas as pd
from datetime import datetime
import json


def main():
    print("=" * 80)
    print("TOP 5 SIZING STRATEGIES - MARKETSIMULATOR TEST")
    print("=" * 80)
    print()

    # Test symbols - same as fast test
    symbols = ['BTCUSD', 'ETHUSD', 'MSFT', 'NVDA', 'AAPL', 'SPY']
    print(f"Testing on {len(symbols)} symbols: {', '.join(symbols)}")
    print(f"Simulation period: 10 days")
    print(f"Initial capital: $100,000")
    print()

    # Load correlation data
    print("Loading correlation and volatility data...")
    try:
        corr_data = load_correlation_matrix()
        print(f"✓ Loaded correlation matrix with {len(corr_data['symbols'])} symbols")
        print()
    except Exception as e:
        print(f"⚠️  Could not load correlation data: {e}")
        corr_data = None
        print()

    # Initialize backtester
    backtester = SimplifiedBacktester(symbols, corr_data=corr_data)

    # Top 5 strategies from fast test
    strategies = [
        # Winner: Best risk-adjusted
        (VolatilityAdjustedStrategy(corr_data=corr_data, target_vol_contribution=0.10), "VolAdjusted_10pct"),

        # Runner-up: Great risk-adjusted
        (VolatilityAdjustedStrategy(corr_data=corr_data, target_vol_contribution=0.15), "VolAdjusted_15pct"),

        # Best raw return
        (KellyStrategy(fraction=0.5, cap=1.0), "Kelly_50pct"),

        # Conservative with good Sharpe
        (FixedFractionStrategy(0.25), "Fixed_25pct"),

        # Baseline
        (FixedFractionStrategy(0.5), "Naive_50pct_Baseline"),
    ]

    # Run strategies
    print("Running top 5 strategies on market simulator...")
    print()

    results = []
    for strategy, name in strategies:
        result = backtester.run_strategy(strategy, name)
        results.append(result)
        print(f"  ✓ Completed: {name}")

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()

    # Create results table
    df_data = []
    for r in results:
        df_data.append({
            'Strategy': r.strategy_name,
            'Final Equity': f"${r.final_equity:,.0f}",
            'Return': f"{r.total_return_pct:.2%}",
            'Sharpe': f"{r.sharpe_ratio:.2f}",
            'Max DD': f"{r.max_drawdown_pct:.2%}",
            'Volatility': f"{r.volatility:.2%}",
            'Trades': r.num_trades,
            'Avg Size': f"{r.avg_position_size:.2%}",
        })

    df = pd.DataFrame(df_data)
    print(df.to_string(index=False))
    print()

    # Rankings
    print("Ranking by Sharpe Ratio:")
    print("-" * 80)
    sorted_results = sorted(results, key=lambda x: x.sharpe_ratio, reverse=True)
    for i, r in enumerate(sorted_results, 1):
        print(f"  {i}. {r.strategy_name:30s} Sharpe: {r.sharpe_ratio:6.2f}  "
              f"Return: {r.total_return_pct:6.2%}  DD: {r.max_drawdown_pct:6.2%}")

    print()
    print("Ranking by Total Return:")
    print("-" * 80)
    sorted_results = sorted(results, key=lambda x: x.total_return_pct, reverse=True)
    for i, r in enumerate(sorted_results, 1):
        print(f"  {i}. {r.strategy_name:30s} Return: {r.total_return_pct:6.2%}  "
              f"Sharpe: {r.sharpe_ratio:6.2f}  DD: {r.max_drawdown_pct:6.2%}")

    print()

    # Save results
    output_file = Path("experiments/top5_sizing_marketsim_results.json")
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'symbols': symbols,
        'initial_cash': backtester.initial_cash,
        'days': backtester.days,
        'results': [
            {
                'strategy': r.strategy_name,
                'final_equity': r.final_equity,
                'return_pct': r.total_return_pct,
                'max_dd_pct': r.max_drawdown_pct,
                'sharpe': r.sharpe_ratio,
                'volatility': r.volatility,
                'trades': r.num_trades,
                'avg_size': r.avg_position_size,
            }
            for r in results
        ]
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to: {output_file}")
    print()

    # Key findings
    baseline = next((r for r in results if 'Baseline' in r.strategy_name), None)
    if baseline:
        print("=" * 80)
        print("KEY FINDINGS")
        print("=" * 80)
        print(f"Baseline (Naive 50%):  Return {baseline.total_return_pct:6.2%}  "
              f"Sharpe {baseline.sharpe_ratio:5.2f}  DD {baseline.max_drawdown_pct:6.2%}")
        print()

        # Best Sharpe
        best_sharpe = max(results, key=lambda x: x.sharpe_ratio)
        if best_sharpe.strategy_name != baseline.strategy_name:
            print(f"Best Risk-Adjusted ({best_sharpe.strategy_name}):")
            print(f"  Return: {best_sharpe.total_return_pct:6.2%} ({best_sharpe.total_return_pct - baseline.total_return_pct:+.2%} vs baseline)")
            print(f"  Sharpe: {best_sharpe.sharpe_ratio:6.2f} ({best_sharpe.sharpe_ratio - baseline.sharpe_ratio:+.2f} vs baseline)")
            print(f"  Max DD: {best_sharpe.max_drawdown_pct:6.2%} ({best_sharpe.max_drawdown_pct - baseline.max_drawdown_pct:+.2%} vs baseline)")
            print()

        # Best return
        best_return = max(results, key=lambda x: x.total_return_pct)
        if best_return.strategy_name != best_sharpe.strategy_name:
            print(f"Highest Return ({best_return.strategy_name}):")
            print(f"  Return: {best_return.total_return_pct:6.2%} ({best_return.total_return_pct - baseline.total_return_pct:+.2%} vs baseline)")
            print(f"  Sharpe: {best_return.sharpe_ratio:6.2f} ({best_return.sharpe_ratio - baseline.sharpe_ratio:+.2f} vs baseline)")
            print(f"  Max DD: {best_return.max_drawdown_pct:6.2%} ({best_return.max_drawdown_pct - baseline.max_drawdown_pct:+.2%} vs baseline)")

    print()
    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    # Recommend based on Sharpe
    best = max(results, key=lambda x: x.sharpe_ratio)
    print(f"For production use, recommend: {best.strategy_name}")
    print(f"  • Risk-adjusted return (Sharpe): {best.sharpe_ratio:.2f}")
    print(f"  • Total return: {best.total_return_pct:.2%}")
    print(f"  • Max drawdown: {best.max_drawdown_pct:.2%}")
    print(f"  • Average position size: {best.avg_position_size:.2%}")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
