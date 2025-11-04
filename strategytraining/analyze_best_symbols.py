"""
Analyze best performing symbols by annualized PnL

Creates best_performing_symbols.csv with symbol rankings
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


def is_crypto(symbol: str) -> bool:
    """Check if symbol is cryptocurrency"""
    return '-USD' in symbol.upper() or symbol.upper().endswith('USD')


def calculate_annualized_pnl(row):
    """
    Calculate annualized PnL based on symbol type and window duration

    Stocks: 252 trading days per year
    Crypto: 365 days per year
    """
    is_crypto_symbol = row['is_crypto']

    # Get window duration from timestamps
    try:
        start_time = pd.to_datetime(row['start_time'])
        end_time = pd.to_datetime(row['end_time'])
        window_days = (end_time - start_time).total_seconds() / 86400
    except:
        # Fallback: assume 7 day window
        window_days = 7

    if window_days <= 0:
        return 0

    # Calculate annualization factor
    trading_days_per_year = 365 if is_crypto_symbol else 252
    annualization_factor = trading_days_per_year / window_days

    # Annualize the PnL
    total_pnl = row['total_pnl']
    annualized_pnl = total_pnl * annualization_factor

    return annualized_pnl


def main():
    print("="*80)
    print("ANALYZING BEST PERFORMING SYMBOLS")
    print("="*80)

    # Find latest dataset
    datasets_dir = Path('strategytraining/datasets')

    # Look for strategy performance files
    perf_files = list(datasets_dir.glob('*_strategy_performance.parquet'))

    if not perf_files:
        print("No datasets found yet. Run collection first.")
        return 1

    # Use the most recent file
    latest_file = max(perf_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading: {latest_file.name}")

    # Load data
    df = pd.read_parquet(latest_file)

    print(f"\nData loaded:")
    print(f"  Total records: {len(df)}")
    print(f"  Symbols: {df['symbol'].nunique()}")
    print(f"  Strategies: {df['strategy'].nunique()}")
    print(f"  Windows: {df['window_num'].nunique()}")

    # Calculate annualized PnL for each record
    df['annualized_pnl'] = df.apply(calculate_annualized_pnl, axis=1)

    # Group by symbol and aggregate
    symbol_stats = df.groupby('symbol').agg({
        'total_pnl': 'sum',
        'annualized_pnl': 'mean',  # Average annualized PnL across all windows
        'sharpe_ratio': 'mean',
        'win_rate': 'mean',
        'max_drawdown': 'mean',
        'num_trades': 'sum',
        'window_num': 'count',  # Number of strategy-window combinations
        'is_crypto': 'max'  # Use max to get True if any is True
    }).reset_index()

    # Re-detect crypto properly
    symbol_stats['is_crypto'] = symbol_stats['symbol'].apply(is_crypto)

    # Rename columns for clarity
    symbol_stats.columns = [
        'symbol',
        'total_pnl_all_windows',
        'avg_annualized_pnl_per_window',
        'avg_sharpe_ratio',
        'avg_win_rate',
        'avg_max_drawdown',
        'total_trades',
        'num_strategy_windows',
        'is_crypto'
    ]

    # Calculate total annualized PnL (sum of annualized across all strategy-windows)
    total_annualized = df.groupby('symbol')['annualized_pnl'].sum().reset_index()
    total_annualized.columns = ['symbol', 'total_annualized_pnl']

    symbol_stats = symbol_stats.merge(total_annualized, on='symbol')

    # Sort by average annualized PnL per window (better metric than total)
    symbol_stats = symbol_stats.sort_values('avg_annualized_pnl_per_window', ascending=False)

    # Format output
    output_df = symbol_stats[[
        'symbol',
        'avg_annualized_pnl_per_window',
        'total_annualized_pnl',
        'total_pnl_all_windows',
        'avg_sharpe_ratio',
        'avg_win_rate',
        'avg_max_drawdown',
        'total_trades',
        'num_strategy_windows',
        'is_crypto'
    ]].copy()

    # Round numeric columns
    output_df['avg_annualized_pnl_per_window'] = output_df['avg_annualized_pnl_per_window'].round(2)
    output_df['total_annualized_pnl'] = output_df['total_annualized_pnl'].round(2)
    output_df['total_pnl_all_windows'] = output_df['total_pnl_all_windows'].round(2)
    output_df['avg_sharpe_ratio'] = output_df['avg_sharpe_ratio'].round(3)
    output_df['avg_win_rate'] = output_df['avg_win_rate'].round(3)
    output_df['avg_max_drawdown'] = output_df['avg_max_drawdown'].round(3)

    # Save to CSV
    output_file = 'strategytraining/best_performing_symbols.csv'
    output_df.to_csv(output_file, index=False)

    print(f"\n{'='*80}")
    print("BEST PERFORMING SYMBOLS")
    print(f"{'='*80}")
    print(output_df.to_string(index=False))

    print(f"\n{'='*80}")
    print(f"Saved to: {output_file}")
    print(f"{'='*80}")

    # Summary stats
    print(f"\nSummary:")
    print(f"  Total symbols analyzed: {len(output_df)}")
    print(f"  Stocks: {(~output_df['is_crypto']).sum()}")
    print(f"  Crypto: {output_df['is_crypto'].sum()}")
    print(f"\nTop 5 by avg annualized PnL per window:")
    for idx, row in output_df.head(5).iterrows():
        crypto_flag = "🪙" if row['is_crypto'] else "📈"
        print(f"  {crypto_flag} {row['symbol']:12s} ${row['avg_annualized_pnl_per_window']:>12,.2f}/window "
              f"(Sharpe: {row['avg_sharpe_ratio']:>6.3f}, WinRate: {row['avg_win_rate']:>5.1%})")

    print(f"\nBottom 5:")
    for idx, row in output_df.tail(5).iterrows():
        crypto_flag = "🪙" if row['is_crypto'] else "📈"
        print(f"  {crypto_flag} {row['symbol']:12s} ${row['avg_annualized_pnl_per_window']:>12,.2f}/window "
              f"(Sharpe: {row['avg_sharpe_ratio']:>6.3f}, WinRate: {row['avg_win_rate']:>5.1%})")

    return 0


if __name__ == '__main__':
    exit(main())
