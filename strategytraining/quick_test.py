"""
Quick test script to verify dataset collection works

Tests with a small set of symbols (2 stocks, 1 crypto) to ensure
the pipeline is working correctly.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from collect_position_sizing_dataset import DatasetCollector


def main():
    print("="*80)
    print("QUICK TEST: Position Sizing Dataset Collection")
    print("="*80)

    # Test with minimal symbols
    test_symbols = ['AAPL', 'MSFT', 'BTC-USD']

    print(f"\nTesting with symbols: {test_symbols}")

    # Create collector with test settings
    collector = DatasetCollector(
        data_dir='trainingdata/train',
        output_dir='strategytraining/datasets',
        window_days=7,
        stride_days=7,
        min_data_points=500  # Lower threshold for testing
    )

    # Collect data
    print("\nStarting data collection...")
    results = collector.collect_all_symbols(
        symbols=test_symbols,
        max_symbols=None
    )

    # Summary
    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)

    for result in results:
        print(f"\n{result['symbol']}:")
        print(f"  Windows: {result['num_windows']}")
        print(f"  Trades: {result['num_trades']}")
        print(f"  Data points: {result['total_data_points']}")

    # Save if we got data
    if len(collector.trades_data) > 0:
        print("\n" + "="*80)
        print("SAVING DATASET")
        print("="*80)

        paths = collector.save_dataset(dataset_name='test_dataset')

        print("\nTest successful! Files created:")
        for key, path in paths.items():
            print(f"  {key}: {path}")

        print("\nYou can analyze the test dataset with:")
        print(f"  python strategytraining/analyze_dataset.py {paths['trades_path'].replace('_trades.parquet', '')}")

    else:
        print("\nNo data collected - check your data directory and symbols")
        return 1

    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
