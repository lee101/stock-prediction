#!/usr/bin/env python3
"""
Analyze historical PNL data from strategytraining datasets to show
which stock pairs perform best for each strategy.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

def load_strategy_performance(parquet_path: str) -> pd.DataFrame:
    """Load strategy performance data from parquet file."""
    return pd.read_parquet(parquet_path)

def analyze_pnl_by_strategy_and_symbol(df: pd.DataFrame) -> dict:
    """
    Analyze PNL by strategy and symbol.

    Returns a dict with structure:
    {
        'strategy': [
            {'symbol': 'BTCUSD', 'total_pnl': 1234.56, 'num_trades': 10, ...},
            ...
        ]
    }
    """
    results = {}

    # Get unique strategies
    strategies = df['strategy'].unique()

    for strategy in sorted(strategies):
        strategy_df = df[df['strategy'] == strategy]

        # Group by symbol and aggregate metrics
        symbol_stats = strategy_df.groupby('symbol').agg({
            'total_pnl': 'sum',
            'num_trades': 'sum',
            'win_rate': 'mean',
            'max_drawdown': 'min',  # Most negative drawdown
            'sharpe_ratio': 'mean',
            'total_return': 'sum'
        }).reset_index()

        # Sort by total_pnl descending
        symbol_stats = symbol_stats.sort_values('total_pnl', ascending=False)

        # Convert to list of dicts
        results[strategy] = symbol_stats.to_dict('records')

    return results

def print_full_report(results: dict):
    """Print a comprehensive report of all strategies and their stock pairs."""
    print("=" * 100)
    print("PNL BY STOCK PAIRS FOR EACH STRATEGY")
    print("=" * 100)
    print()

    for strategy, symbols in results.items():
        print(f"\n{'='*100}")
        print(f"STRATEGY: {strategy}")
        print(f"{'='*100}")
        print()

        if not symbols:
            print("  No data available")
            continue

        # Header
        print(f"{'Rank':<6} {'Symbol':<12} {'Total PNL':>12} {'Trades':>8} {'Win Rate':>10} "
              f"{'Sharpe':>8} {'Max DD':>10} {'Total Return':>12}")
        print("-" * 100)

        # Print each symbol
        for idx, symbol_data in enumerate(symbols, 1):
            symbol = symbol_data['symbol']
            pnl = symbol_data['total_pnl']
            trades = symbol_data['num_trades']
            win_rate = symbol_data['win_rate'] * 100 if not pd.isna(symbol_data['win_rate']) else 0
            sharpe = symbol_data['sharpe_ratio'] if not pd.isna(symbol_data['sharpe_ratio']) else 0
            max_dd = symbol_data['max_drawdown'] if not pd.isna(symbol_data['max_drawdown']) else 0
            total_return = symbol_data['total_return'] if not pd.isna(symbol_data['total_return']) else 0

            print(f"{idx:<6} {symbol:<12} ${pnl:>11,.2f} {trades:>8,.0f} {win_rate:>9.1f}% "
                  f"{sharpe:>8.2f} {max_dd:>9.1f}% ${total_return:>11,.2f}")

        # Summary stats
        print("-" * 100)
        total_pnl = sum(s['total_pnl'] for s in symbols)
        total_trades = sum(s['num_trades'] for s in symbols)
        avg_win_rate = np.mean([s['win_rate'] for s in symbols if not pd.isna(s['win_rate'])]) * 100
        positive_symbols = sum(1 for s in symbols if s['total_pnl'] > 0)

        print(f"{'TOTAL':<6} {len(symbols)} symbols {'':<1} ${total_pnl:>11,.2f} {total_trades:>8,.0f} "
              f"{avg_win_rate:>9.1f}% {'':>8} {'':>10} "
              f"(+{positive_symbols}/{len(symbols)} profitable)")
        print()

def save_to_json(results: dict, output_path: str):
    """Save results to JSON file for further analysis."""
    # Convert numpy types to native Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif pd.isna(obj):
            return None
        return obj

    results_converted = convert_types(results)

    with open(output_path, 'w') as f:
        json.dump(results_converted, f, indent=2)

    print(f"\n\nResults saved to: {output_path}")

def main():
    # Find the most recent full strategy dataset
    datasets_dir = Path("strategytraining/datasets")

    # Look for the most recent full strategy dataset
    pattern = "full_strategy_dataset_*_strategy_performance.parquet"
    files = sorted(datasets_dir.glob(pattern), reverse=True)

    if not files:
        print("No strategy performance datasets found!")
        return

    most_recent = files[0]
    print(f"Loading data from: {most_recent}")
    print()

    # Load metadata to understand the dataset
    metadata_file = most_recent.with_name(most_recent.name.replace('_strategy_performance.parquet', '_metadata.json'))
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)
        print("Dataset metadata:")
        print(f"  Timestamp: {metadata.get('timestamp', 'N/A')}")
        print(f"  Total strategies: {metadata.get('total_strategies', 'N/A')}")
        print(f"  Total symbols: {metadata.get('total_symbols', 'N/A')}")
        print()

    # Load and analyze
    df = load_strategy_performance(str(most_recent))

    print(f"Loaded {len(df)} strategy-symbol combinations")
    print(f"Unique strategies: {df['strategy'].nunique()}")
    print(f"Unique symbols: {df['symbol'].nunique()}")
    print()

    # Analyze
    results = analyze_pnl_by_strategy_and_symbol(df)

    # Print report
    print_full_report(results)

    # Save to JSON
    output_file = "strategytraining/pnl_by_stock_pairs_analysis.json"
    save_to_json(results, output_file)

if __name__ == "__main__":
    main()
