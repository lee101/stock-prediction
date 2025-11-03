"""
Quick test for multi-strategy dataset collection

Tests with a small set of symbols to verify the pipeline works.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from collect_strategy_pnl_dataset import StrategyPnLCollector


def main():
    print("="*80)
    print("QUICK TEST: Multi-Strategy PnL Dataset Collection")
    print("="*80)

    # Test with minimal symbols
    test_symbols = ['AAPL', 'BTC-USD']

    print(f"\nTesting with symbols: {test_symbols}")
    print("This will collect PnL for ALL strategies on each symbol")

    # Create collector
    collector = StrategyPnLCollector(
        data_dir='trainingdata/train',
        output_dir='strategytraining/datasets',
        window_days=7,
        stride_days=7,
        min_data_points=500
    )

    # Collect data
    print("\nStarting collection...")
    results = collector.collect_all_symbols(symbols=test_symbols)

    # Summary
    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)

    for result in results:
        print(f"\n{result['symbol']}:")
        print(f"  Windows: {result['num_windows']}")
        print(f"  Strategy-Window Records: {result['num_strategy_results']}")
        print(f"  Total Trades: {result['num_trades']}")

    # Save
    if len(collector.strategy_performance) > 0:
        paths = collector.save_dataset(dataset_name='test_strategy_dataset')

        print("\n" + "="*80)
        print("TEST SUCCESS!")
        print("="*80)
        print(f"\nAnalyze with:")
        print(f"  python strategytraining/analyze_strategy_dataset.py {paths['base_path']}")
        return 0
    else:
        print("\nNo data collected")
        return 1


if __name__ == '__main__':
    exit(main())
