#!/usr/bin/env python3
"""
Generate sample OHLC data for testing the dataloader
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_ohlc_data(symbol: str, 
                      days: int = 100, 
                      freq: str = '1H',
                      base_price: float = 100.0) -> pd.DataFrame:
    """Generate realistic OHLC data with proper relationships"""
    
    # Create time index
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    timestamps = pd.date_range(start=start_time, end=end_time, freq=freq)
    
    n_points = len(timestamps)
    
    # Generate realistic price movements using random walk
    np.random.seed(hash(symbol) % 2**32)  # Consistent seed per symbol
    
    # Generate returns with some autocorrelation
    returns = np.random.normal(0, 0.02, n_points)  # 2% daily volatility
    
    # Add some trend
    trend = np.linspace(-0.1, 0.1, n_points) / n_points
    returns += trend
    
    # Create close prices
    close_prices = np.zeros(n_points)
    close_prices[0] = base_price
    
    for i in range(1, n_points):
        close_prices[i] = close_prices[i-1] * (1 + returns[i])
    
    # Generate OHLC with realistic relationships
    data = []
    for i, close in enumerate(close_prices):
        # Previous close (or current for first point)
        prev_close = close if i == 0 else close_prices[i-1]
        
        # Random intraday volatility
        volatility = abs(np.random.normal(0, 0.01))
        
        # High/Low around the close price
        high_factor = 1 + np.random.uniform(0, volatility)
        low_factor = 1 - np.random.uniform(0, volatility)
        
        high = max(close, prev_close) * high_factor
        low = min(close, prev_close) * low_factor
        
        # Open price (close to previous close with some gap)
        open_gap = np.random.normal(0, 0.005)  # 0.5% gap on average
        open_price = prev_close * (1 + open_gap)
        
        # Ensure OHLC relationships are maintained
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        # Volume (random with some correlation to price movement)
        price_change = abs((close - prev_close) / prev_close)
        base_volume = 1000000
        volume = int(base_volume * (1 + price_change * 10) * np.random.uniform(0.5, 2.0))
        
        data.append({
            'timestamp': timestamps[i],
            'Open': round(open_price, 2),
            'High': round(high, 2),
            'Low': round(low, 2),
            'Close': round(close, 2),
            'Volume': volume
        })
    
    return pd.DataFrame(data)

def main():
    """Generate sample data for testing"""
    print("🔧 Generating sample OHLC data...")
    
    # Create directories
    os.makedirs("trainingdata/train", exist_ok=True)
    os.makedirs("trainingdata/test", exist_ok=True)
    
    # Popular stock symbols for testing
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX']
    
    # Generate training data (longer history)
    for symbol in symbols:
        df = generate_ohlc_data(symbol, days=200, base_price=50 + hash(symbol) % 200)
        
        # Split: most data for training, last 30 days for test
        split_date = df['timestamp'].max() - timedelta(days=30)
        
        train_df = df[df['timestamp'] <= split_date].copy()
        test_df = df[df['timestamp'] > split_date].copy()
        
        # Save training data
        train_file = f"trainingdata/train/{symbol}.csv"
        train_df.to_csv(train_file, index=False)
        print(f"✅ Created {train_file}: {len(train_df)} rows")
        
        # Save test data
        if len(test_df) > 0:
            test_file = f"trainingdata/test/{symbol}.csv"
            test_df.to_csv(test_file, index=False)
            print(f"✅ Created {test_file}: {len(test_df)} rows")
    
    print("✅ Sample data generation completed!")
    
    # Show sample data
    sample_file = "trainingdata/train/AAPL.csv"
    if os.path.exists(sample_file):
        sample_df = pd.read_csv(sample_file)
        print(f"\n📊 Sample data from {sample_file}:")
        print(sample_df.head())
        print(f"Shape: {sample_df.shape}")
        print(f"Date range: {sample_df['timestamp'].min()} to {sample_df['timestamp'].max()}")

if __name__ == "__main__":
    main()