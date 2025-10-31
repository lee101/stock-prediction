"""
Merge multiple datasets collected in batches into a single unified dataset.

This is useful when collecting data in batches to avoid memory issues.
"""

import pandas as pd
import json
from pathlib import Path
from typing import List
import argparse
from datetime import datetime


def find_batch_datasets(datasets_dir: str, pattern: str = "batch_*") -> List[str]:
    """Find all batch datasets in the directory"""

    datasets_path = Path(datasets_dir)
    if not datasets_path.exists():
        print(f"Directory {datasets_dir} not found")
        return []

    # Find all batch metadata files
    metadata_files = list(datasets_path.glob(f"{pattern}_metadata.json"))

    # Extract base paths (without suffix)
    base_paths = []
    for metadata_file in metadata_files:
        base_path = str(metadata_file).replace("_metadata.json", "")
        base_paths.append(base_path)

    return sorted(base_paths)


def merge_datasets(
    dataset_paths: List[str],
    output_name: str,
    output_dir: str = "strategytraining/datasets"
):
    """Merge multiple datasets into one"""

    if not dataset_paths:
        print("No datasets to merge")
        return None

    print(f"Merging {len(dataset_paths)} datasets...")

    # Lists to accumulate data
    all_trades = []
    all_summaries = []
    all_positions = []
    all_symbols = set()

    # Load and merge each dataset
    for i, base_path in enumerate(dataset_paths, 1):
        print(f"\nLoading dataset {i}/{len(dataset_paths)}: {Path(base_path).name}")

        try:
            # Load trades
            trades_path = f"{base_path}_trades.parquet"
            if Path(trades_path).exists():
                trades = pd.read_parquet(trades_path)
                all_trades.append(trades)
                print(f"  Trades: {len(trades)}")

            # Load summaries
            summaries_path = f"{base_path}_summaries.parquet"
            if Path(summaries_path).exists():
                summaries = pd.read_parquet(summaries_path)
                all_summaries.append(summaries)
                print(f"  Windows: {len(summaries)}")

            # Load positions
            positions_path = f"{base_path}_positions.parquet"
            if Path(positions_path).exists():
                positions = pd.read_parquet(positions_path)
                all_positions.append(positions)
                print(f"  Positions: {len(positions)}")

            # Load metadata to get symbols
            metadata_path = f"{base_path}_metadata.json"
            if Path(metadata_path).exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    all_symbols.update(metadata.get('symbols', []))

        except Exception as e:
            print(f"  Error loading {base_path}: {e}")
            continue

    # Concatenate all dataframes
    print("\nMerging dataframes...")

    merged_trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    merged_summaries = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()
    merged_positions = pd.concat(all_positions, ignore_index=True) if all_positions else pd.DataFrame()

    print(f"  Total trades: {len(merged_trades)}")
    print(f"  Total windows: {len(merged_summaries)}")
    print(f"  Total positions: {len(merged_positions)}")
    print(f"  Unique symbols: {len(all_symbols)}")

    # Save merged dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(output_dir) / f"{output_name}_{timestamp}"

    print(f"\nSaving merged dataset: {output_base}")

    # Save trades
    trades_path = f"{output_base}_trades.parquet"
    merged_trades.to_parquet(trades_path, index=False)
    print(f"  Saved trades: {trades_path}")

    # Save summaries
    summaries_path = f"{output_base}_summaries.parquet"
    merged_summaries.to_parquet(summaries_path, index=False)
    print(f"  Saved summaries: {summaries_path}")

    # Save positions
    positions_path = f"{output_base}_positions.parquet"
    merged_positions.to_parquet(positions_path, index=False)
    print(f"  Saved positions: {positions_path}")

    # Create merged metadata
    merged_metadata = {
        'dataset_name': output_name,
        'created_at': timestamp,
        'merged_from': [Path(p).name for p in dataset_paths],
        'num_source_datasets': len(dataset_paths),
        'num_trades': len(merged_trades),
        'num_windows': len(merged_summaries),
        'num_position_snapshots': len(merged_positions),
        'symbols': sorted(list(all_symbols))
    }

    metadata_path = f"{output_base}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(merged_metadata, f, indent=2)
    print(f"  Saved metadata: {metadata_path}")

    print("\nMerge complete!")

    return {
        'trades_path': trades_path,
        'summaries_path': summaries_path,
        'positions_path': positions_path,
        'metadata_path': metadata_path,
        'base_path': str(output_base)
    }


def main():
    parser = argparse.ArgumentParser(description="Merge batch datasets")
    parser.add_argument('--datasets-dir', default='strategytraining/datasets',
                       help='Directory containing batch datasets')
    parser.add_argument('--pattern', default='batch_*',
                       help='Pattern to match batch dataset names')
    parser.add_argument('--output-name', default='merged_dataset',
                       help='Name for merged dataset')
    parser.add_argument('--datasets', nargs='+', default=None,
                       help='Specific dataset paths to merge (without suffix)')

    args = parser.parse_args()

    # Find or use specified datasets
    if args.datasets:
        dataset_paths = args.datasets
    else:
        dataset_paths = find_batch_datasets(args.datasets_dir, args.pattern)

    if not dataset_paths:
        print(f"No datasets found matching pattern '{args.pattern}' in {args.datasets_dir}")
        return 1

    print(f"Found {len(dataset_paths)} datasets to merge")

    # Merge
    result = merge_datasets(
        dataset_paths,
        output_name=args.output_name,
        output_dir=args.datasets_dir
    )

    if result:
        print(f"\nMerged dataset base path:")
        print(f"  {result['base_path']}")
        print(f"\nAnalyze with:")
        print(f"  python strategytraining/analyze_dataset.py {result['base_path']}")
        return 0
    else:
        print("\nMerge failed")
        return 1


if __name__ == '__main__':
    exit(main())
