#!/usr/bin/env python3
"""
Debug data loading to understand the issue
"""

from toto_ohlc_dataloader import TotoOHLCDataLoader, DataLoaderConfig
from pathlib import Path

def debug_data_loading():
    """Debug the data loading process"""
    print("🔍 Debugging Data Loading")
    
    # Check directory structure
    train_path = Path("trainingdata/train")
    test_path = Path("trainingdata/test")
    
    print(f"Train path exists: {train_path.exists()}")
    print(f"Test path exists: {test_path.exists()}")
    
    if train_path.exists():
        csv_files = list(train_path.glob("*.csv"))
        print(f"Train CSV files: {len(csv_files)}")
        for f in csv_files[:5]:  # Show first 5
            print(f"  - {f.name}")
    
    if test_path.exists():
        csv_files = list(test_path.glob("*.csv"))
        print(f"Test CSV files: {len(csv_files)}")
        for f in csv_files[:5]:  # Show first 5
            print(f"  - {f.name}")
    
    # Test with minimal config
    config = DataLoaderConfig(
        batch_size=2,
        sequence_length=24,
        prediction_length=6,
        max_symbols=2,
        num_workers=0,
        validation_split=0.0,  # No validation split
        min_sequence_length=50  # Lower minimum
    )
    
    print("\n📊 Testing data loading with minimal config")
    dataloader = TotoOHLCDataLoader(config)
    
    # Load data step by step
    train_data, val_data, test_data = dataloader.load_data()
    
    print(f"Train data symbols: {len(train_data)}")
    print(f"Val data symbols: {len(val_data)}")  
    print(f"Test data symbols: {len(test_data)}")
    
    if train_data:
        for symbol, df in train_data.items():
            print(f"  {symbol}: {len(df)} rows")
    
    if val_data:
        for symbol, df in val_data.items():
            print(f"  {symbol} (val): {len(df)} rows")
    
    # Test with even more minimal config
    print("\n📊 Testing with even more minimal requirements")
    config.min_sequence_length = 20
    config.sequence_length = 12
    config.prediction_length = 3
    
    dataloader2 = TotoOHLCDataLoader(config)
    train_data2, val_data2, test_data2 = dataloader2.load_data()
    
    print(f"Train data symbols (minimal): {len(train_data2)}")
    print(f"Val data symbols (minimal): {len(val_data2)}")  
    print(f"Test data symbols (minimal): {len(test_data2)}")

if __name__ == "__main__":
    debug_data_loading()