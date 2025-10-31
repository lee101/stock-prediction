"""
Example Usage: Position Sizing Dataset Collection

Demonstrates how to use the dataset collector programmatically.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategytraining import DatasetCollector, DatasetAnalyzer


def example_basic_collection():
    """Example 1: Basic dataset collection"""

    print("\n" + "="*80)
    print("Example 1: Basic Collection")
    print("="*80)

    # Create collector
    collector = DatasetCollector(
        data_dir='trainingdata/train',
        output_dir='strategytraining/datasets',
        window_days=7,
        stride_days=7
    )

    # Collect for specific symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    results = collector.collect_all_symbols(symbols=symbols)

    # Save dataset
    if len(collector.trades_data) > 0:
        paths = collector.save_dataset(dataset_name='example_basic')
        print(f"\nDataset saved: {paths['trades_path']}")
        return paths
    else:
        print("\nNo data collected")
        return None


def example_custom_windows():
    """Example 2: Custom window configuration"""

    print("\n" + "="*80)
    print("Example 2: Custom Windows (14-day with 7-day stride)")
    print("="*80)

    # Longer windows, overlapping
    collector = DatasetCollector(
        data_dir='trainingdata/train',
        output_dir='strategytraining/datasets',
        window_days=14,  # 2 weeks
        stride_days=7    # 1 week overlap
    )

    symbols = ['BTC-USD', 'ETH-USD']
    results = collector.collect_all_symbols(symbols=symbols)

    if len(collector.trades_data) > 0:
        paths = collector.save_dataset(dataset_name='example_long_windows')
        print(f"\nDataset saved: {paths['trades_path']}")
        return paths
    else:
        print("\nNo data collected")
        return None


def example_analysis(dataset_path: str):
    """Example 3: Analyze collected dataset"""

    print("\n" + "="*80)
    print("Example 3: Dataset Analysis")
    print("="*80)

    # Create analyzer
    analyzer = DatasetAnalyzer(dataset_path)

    # Get summary statistics
    stats = analyzer.get_summary_statistics()

    print(f"\nDataset Statistics:")
    print(f"  Total trades: {stats['total_trades']}")
    print(f"  Total windows: {stats['total_windows']}")
    print(f"  Symbols: {stats['unique_symbols']}")

    if 'trade_stats' in stats:
        print(f"\n  Avg PnL: ${stats['trade_stats']['avg_pnl']:.2f}")
        print(f"  Win rate: {stats['trade_stats']['win_rate']:.1%}")

    # Compare stocks vs crypto
    comparison = analyzer.compare_stocks_vs_crypto()

    print(f"\nStocks vs Crypto:")
    print(f"  Stocks - Avg Return: {comparison['stocks']['avg_return']:.2%}")
    print(f"  Crypto - Avg Return: {comparison['crypto']['avg_return']:.2%}")

    # Get best/worst windows
    best_worst = analyzer.get_best_worst_windows(n=3)

    print(f"\nTop 3 Windows:")
    for i, window in enumerate(best_worst['best'], 1):
        print(f"  {i}. {window['symbol']} - Return: {window['total_return']:.2%}")

    return stats


def example_symbol_specific(dataset_path: str, symbol: str):
    """Example 4: Symbol-specific analysis"""

    print("\n" + "="*80)
    print(f"Example 4: Analyzing {symbol}")
    print("="*80)

    analyzer = DatasetAnalyzer(dataset_path)
    analysis = analyzer.analyze_by_symbol(symbol)

    if analysis:
        print(f"\n{symbol} Analysis:")
        print(f"  Windows: {analysis['num_windows']}")
        print(f"  Trades: {analysis['num_trades']}")

        if 'trade_stats' in analysis:
            print(f"  Total PnL: ${analysis['trade_stats']['total_pnl']:,.2f}")
            print(f"  Win Rate: {analysis['trade_stats']['win_rate']:.1%}")
            print(f"  Profit Factor: {analysis['trade_stats']['profit_factor']:.2f}")
            print(f"  Avg Winner: ${analysis['trade_stats']['avg_winner']:.2f}")
            print(f"  Avg Loser: ${analysis['trade_stats']['avg_loser']:.2f}")

        return analysis
    else:
        print(f"\n{symbol} not found in dataset")
        return None


def example_programmatic_access(dataset_path: str):
    """Example 5: Direct dataframe access"""

    print("\n" + "="*80)
    print("Example 5: Direct DataFrame Access")
    print("="*80)

    analyzer = DatasetAnalyzer(dataset_path)

    # Access raw dataframes
    print(f"\nTrades DataFrame shape: {analyzer.trades_df.shape}")
    print(f"Columns: {list(analyzer.trades_df.columns)}")

    # Filter for specific conditions
    profitable_trades = analyzer.trades_df[analyzer.trades_df['pnl'] > 0]
    print(f"\nProfitable trades: {len(profitable_trades)} / {len(analyzer.trades_df)}")

    # Group by symbol
    pnl_by_symbol = analyzer.trades_df.groupby('symbol')['pnl'].sum().sort_values(ascending=False)
    print(f"\nTop 3 symbols by total PnL:")
    for symbol, pnl in pnl_by_symbol.head(3).items():
        print(f"  {symbol}: ${pnl:,.2f}")

    # Analyze trade duration
    avg_duration_by_symbol = analyzer.trades_df.groupby('symbol')['duration_bars'].mean()
    print(f"\nAverage trade duration (bars):")
    for symbol, duration in avg_duration_by_symbol.items():
        print(f"  {symbol}: {duration:.1f}")


def main():
    """Run all examples"""

    print("\n" + "="*80)
    print("POSITION SIZING DATASET COLLECTION - EXAMPLES")
    print("="*80)

    # Example 1: Basic collection
    paths1 = example_basic_collection()

    # Example 2: Custom windows
    paths2 = example_custom_windows()

    # Examples 3-5 require a dataset to exist
    if paths1:
        # Extract base path (without suffix)
        base_path = paths1['trades_path'].replace('_trades.parquet', '')

        # Example 3: Analysis
        example_analysis(base_path)

        # Example 4: Symbol-specific (use first symbol from dataset)
        example_symbol_specific(base_path, 'AAPL')

        # Example 5: Programmatic access
        example_programmatic_access(base_path)

    print("\n" + "="*80)
    print("EXAMPLES COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
