"""
Full Strategy PnL Dataset Collection - Resumable Multi-Day Collection

This script runs the full collection across all symbols with progress tracking
and automatic resume capability.
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from strategytraining.collect_strategy_pnl_dataset import StrategyPnLCollector


def main():
    print("="*80)
    print("FULL STRATEGY PNL DATASET COLLECTION")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Initialize collector
    collector = StrategyPnLCollector(
        data_dir='trainingdata/train',
        output_dir='strategytraining/datasets',
        window_days=7,
        stride_days=7,
        min_data_points=500  # Lower threshold to include more symbols
    )

    # Get all available symbols
    all_symbols = collector.get_available_symbols()
    print(f"Total symbols found: {len(all_symbols)}")

    # Check for existing progress
    progress_file = Path('strategytraining/datasets/collection_progress.json')
    completed_symbols = set()

    if progress_file.exists():
        with open(progress_file, 'r') as f:
            progress_data = json.load(f)
            completed_symbols = set(progress_data.get('completed_symbols', []))
            print(f"\nResuming from previous run:")
            print(f"  Already completed: {len(completed_symbols)} symbols")
            print(f"  Remaining: {len(all_symbols) - len(completed_symbols)} symbols")

    # Filter out completed symbols
    remaining_symbols = [s for s in all_symbols if s not in completed_symbols]

    if not remaining_symbols:
        print("\n✓ All symbols already processed!")
        return 0

    print(f"\nProcessing {len(remaining_symbols)} symbols")
    print(f"Estimated time: {len(remaining_symbols) * 15} - {len(remaining_symbols) * 20} minutes")
    print(f"              = {len(remaining_symbols) * 15 / 60:.1f} - {len(remaining_symbols) * 20 / 60:.1f} hours")
    print()

    # Process each symbol
    results = []
    start_time = time.time()

    for idx, symbol in enumerate(remaining_symbols, 1):
        symbol_start = time.time()

        print(f"\n{'='*80}")
        print(f"[{idx}/{len(remaining_symbols)}] Processing {symbol}")
        print(f"Progress: {len(completed_symbols) + idx - 1}/{len(all_symbols)} total symbols")
        elapsed = time.time() - start_time
        if idx > 1:
            avg_time = elapsed / (idx - 1)
            remaining = avg_time * (len(remaining_symbols) - idx + 1)
            print(f"Elapsed: {elapsed/3600:.1f}h, Estimated remaining: {remaining/3600:.1f}h")
        print(f"{'='*80}")

        try:
            result = collector.collect_symbol_data(symbol)

            if result:
                results.append(result)
                completed_symbols.add(symbol)

                symbol_time = time.time() - symbol_start
                print(f"\n✓ {symbol} completed in {symbol_time/60:.1f} minutes")
                print(f"  Windows: {result['num_windows']}")
                print(f"  Strategy-Window Results: {result['num_strategy_results']}")
                print(f"  Trades: {result['num_trades']}")

                # Save progress after each symbol
                progress_data = {
                    'completed_symbols': list(completed_symbols),
                    'last_updated': datetime.now().isoformat(),
                    'total_symbols': len(all_symbols),
                    'completed_count': len(completed_symbols)
                }
                progress_file.parent.mkdir(parents=True, exist_ok=True)
                with open(progress_file, 'w') as f:
                    json.dump(progress_data, f, indent=2)

                # Save incremental dataset every 10 symbols
                if len(completed_symbols) % 10 == 0:
                    print(f"\n📊 Saving incremental dataset ({len(completed_symbols)} symbols)...")
                    collector.save_dataset(dataset_name=f'incremental_strategy_dataset_{len(completed_symbols)}')
            else:
                print(f"\n⚠ {symbol} skipped (insufficient data or error)")

        except KeyboardInterrupt:
            print("\n\n⚠ Interrupted by user. Progress saved!")
            print(f"Completed: {len(completed_symbols)}/{len(all_symbols)} symbols")
            break
        except Exception as e:
            print(f"\n✗ Error processing {symbol}: {e}")
            import traceback
            traceback.print_exc()
            # Continue with next symbol
            continue

    # Final save
    if len(collector.strategy_performance) > 0:
        print(f"\n{'='*80}")
        print("SAVING FINAL DATASET")
        print(f"{'='*80}")
        print(f"Total symbols processed: {len(completed_symbols)}")
        print(f"Total strategy-window results: {len(collector.strategy_performance)}")
        print(f"Total trades: {len(collector.strategy_trades)}")

        paths = collector.save_dataset(dataset_name='full_strategy_dataset')

        print(f"\n{'='*80}")
        print("COLLECTION COMPLETE!")
        print(f"{'='*80}")
        print(f"Total time: {(time.time() - start_time)/3600:.1f} hours")
        print(f"Dataset saved to: {paths['base_path']}")
        print(f"\nAnalyze with:")
        print(f"  python strategytraining/analyze_strategy_dataset.py {paths['base_path']}")

        return 0
    else:
        print("\nNo data collected")
        return 1


if __name__ == '__main__':
    exit(main())
