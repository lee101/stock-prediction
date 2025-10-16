#!/usr/bin/env python3
"""
Extract latest training data for each stock pair from the data/ directory.
Creates organized training data with proper train/test split.
"""

import os
import pandas as pd
from collections import defaultdict
from datetime import datetime, timedelta
import shutil
from pathlib import Path

def find_all_stock_symbols():
    """Find all unique stock symbols from CSV files in data directories."""
    symbols = set()
    data_dir = Path('data')
    
    for timestamp_dir in data_dir.iterdir():
        if timestamp_dir.is_dir() and timestamp_dir.name.startswith('2024'):
            for csv_file in timestamp_dir.glob('*.csv'):
                # Extract symbol from filename (e.g., "AAPL-2024-12-28.csv" -> "AAPL")
                symbol = csv_file.stem.split('-')[0]
                symbols.add(symbol)
    
    return sorted(symbols)

def find_latest_data_for_symbol(symbol):
    """Find the latest data file for a given symbol."""
    data_dir = Path('data')
    latest_file = None
    latest_date = None
    
    for timestamp_dir in sorted(data_dir.iterdir(), reverse=True):
        if timestamp_dir.is_dir() and timestamp_dir.name.startswith('2024'):
            csv_files = list(timestamp_dir.glob(f'{symbol}-*.csv'))
            if csv_files:
                csv_file = csv_files[0]  # Should only be one per symbol per timestamp
                # Extract date from filename
                try:
                    date_str = csv_file.stem.split('-', 1)[1]  # e.g., "2024-12-28"
                    file_date = datetime.strptime(date_str, '%Y-%m-%d')
                    
                    if latest_date is None or file_date > latest_date:
                        latest_date = file_date
                        latest_file = csv_file
                except ValueError:
                    continue
    
    return latest_file, latest_date

def create_train_test_split(data, test_days=30):
    """Split data into train/test with test being last N days."""
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values('date')
        
        # Get the latest date and calculate cutoff
        latest_date = data['date'].max()
        cutoff_date = latest_date - timedelta(days=test_days)
        
        train_data = data[data['date'] <= cutoff_date]
        test_data = data[data['date'] > cutoff_date]
        
        return train_data, test_data
    else:
        # If no date column, use last N% of rows
        test_size = len(data) * test_days // 100 if test_days < 1 else test_days
        test_size = min(test_size, len(data) // 4)  # Max 25% for test
        
        train_data = data.iloc[:-test_size]
        test_data = data.iloc[-test_size:]
        
        return train_data, test_data

def main():
    print("Finding all stock symbols...")
    symbols = find_all_stock_symbols()
    print(f"Found {len(symbols)} unique symbols: {symbols[:10]}...")
    
    # Create trainingdata directory structure
    training_dir = Path('trainingdata')
    training_dir.mkdir(exist_ok=True)
    (training_dir / 'train').mkdir(exist_ok=True)
    (training_dir / 'test').mkdir(exist_ok=True)
    
    symbol_info = []
    
    for symbol in symbols:
        print(f"Processing {symbol}...")
        latest_file, latest_date = find_latest_data_for_symbol(symbol)
        
        if latest_file is None:
            print(f"  No data found for {symbol}")
            continue
            
        try:
            # Load the data
            data = pd.read_csv(latest_file)
            print(f"  Latest data: {latest_file} ({len(data)} rows)")
            
            # Create train/test split
            train_data, test_data = create_train_test_split(data, test_days=30)
            
            # Save train and test data
            train_file = training_dir / 'train' / f'{symbol}.csv'
            test_file = training_dir / 'test' / f'{symbol}.csv'
            
            train_data.to_csv(train_file, index=False)
            test_data.to_csv(test_file, index=False)
            
            symbol_info.append({
                'symbol': symbol,
                'latest_date': latest_date.strftime('%Y-%m-%d') if latest_date else 'Unknown',
                'total_rows': len(data),
                'train_rows': len(train_data),
                'test_rows': len(test_data),
                'source_file': str(latest_file)
            })
            
            print(f"  Train: {len(train_data)} rows, Test: {len(test_data)} rows")
            
        except Exception as e:
            print(f"  Error processing {symbol}: {e}")
    
    # Save summary
    summary_df = pd.DataFrame(symbol_info)
    summary_df.to_csv(training_dir / 'data_summary.csv', index=False)
    
    print(f"\nCompleted! Processed {len(symbol_info)} symbols.")
    print(f"Training data saved to: {training_dir}")
    print(f"Summary saved to: {training_dir / 'data_summary.csv'}")
    
    # Print summary statistics
    if symbol_info:
        total_train_rows = sum(info['train_rows'] for info in symbol_info)
        total_test_rows = sum(info['test_rows'] for info in symbol_info)
        print(f"\nSummary:")
        print(f"  Total symbols: {len(symbol_info)}")
        print(f"  Total train rows: {total_train_rows:,}")
        print(f"  Total test rows: {total_test_rows:,}")

if __name__ == "__main__":
    main()