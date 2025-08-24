#!/usr/bin/env python3
"""
Data utilities for HuggingFace-style training
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import yfinance as yf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import warnings
warnings.filterwarnings('ignore')


class StockDataProcessor:
    """Advanced stock data processor with multiple features"""
    
    def __init__(self, sequence_length=60, prediction_horizon=5, features=None):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.features = features or ['open', 'high', 'low', 'close', 'volume']
        self.scalers = {}
        self.feature_names = []
        
    def add_technical_indicators(self, df):
        """Add technical indicators to the dataframe"""
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        rolling_mean = df['close'].rolling(window=20).mean()
        rolling_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = rolling_mean + (rolling_std * 2)
        df['bb_lower'] = rolling_mean - (rolling_std * 2)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['close'] - df['bb_lower']) / df['bb_width']
        
        # Price-based features
        df['price_change'] = df['close'].pct_change()
        df['price_change_2'] = df['close'].pct_change(periods=2)
        df['price_change_5'] = df['close'].pct_change(periods=5)
        
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Volume features
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Volatility
        df['volatility'] = df['close'].rolling(window=20).std()
        df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(window=60).mean()
        
        # Support and resistance levels
        df['resistance'] = df['high'].rolling(window=20).max()
        df['support'] = df['low'].rolling(window=20).min()
        df['resistance_distance'] = (df['resistance'] - df['close']) / df['close']
        df['support_distance'] = (df['close'] - df['support']) / df['close']
        
        return df
    
    def prepare_features(self, df):
        """Prepare and select features for training"""
        
        # Add technical indicators
        df = self.add_technical_indicators(df)
        
        # Select features
        feature_columns = []
        
        # Basic OHLCV
        basic_features = ['open', 'high', 'low', 'close', 'volume']
        feature_columns.extend(basic_features)
        
        # Technical indicators
        technical_features = [
            'ma_5', 'ma_10', 'ma_20', 'ma_50',
            'ema_5', 'ema_10', 'ema_20', 'ema_50',
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_lower', 'bb_width', 'bb_position',
            'price_change', 'price_change_2', 'price_change_5',
            'high_low_ratio', 'close_open_ratio',
            'volume_ratio', 'volatility', 'volatility_ratio',
            'resistance_distance', 'support_distance'
        ]
        feature_columns.extend(technical_features)
        
        # Filter out features that don't exist
        feature_columns = [col for col in feature_columns if col in df.columns]
        
        self.feature_names = feature_columns
        
        # Handle missing values
        df = df[feature_columns].fillna(method='ffill').fillna(method='bfill')
        
        return df[feature_columns].values
    
    def fit_scalers(self, data):
        """Fit scalers on training data"""
        
        # Standard scaler for most features
        self.scalers['standard'] = StandardScaler()
        
        # MinMax scaler for bounded features (like RSI)
        self.scalers['minmax'] = MinMaxScaler()
        
        # Fit standard scaler on all features
        self.scalers['standard'].fit(data)
        
        return self
    
    def transform(self, data):
        """Transform data using fitted scalers"""
        if 'standard' not in self.scalers:
            raise ValueError("Scalers not fitted. Call fit_scalers first.")
        
        return self.scalers['standard'].transform(data)
    
    def inverse_transform(self, data):
        """Inverse transform data"""
        return self.scalers['standard'].inverse_transform(data)
    
    def save_scalers(self, path):
        """Save scalers to disk"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'scalers': self.scalers,
            'feature_names': self.feature_names,
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon
        }, path)
    
    def load_scalers(self, path):
        """Load scalers from disk"""
        data = joblib.load(path)
        self.scalers = data['scalers']
        self.feature_names = data['feature_names']
        self.sequence_length = data['sequence_length']
        self.prediction_horizon = data['prediction_horizon']
        return self


def download_stock_data(symbols, start_date='2015-01-01', end_date=None):
    """
    Download stock data from Yahoo Finance
    
    Args:
        symbols: List of stock symbols or single symbol
        start_date: Start date for data
        end_date: End date for data (default: today)
        
    Returns:
        Dictionary of dataframes or single dataframe
    """
    
    if isinstance(symbols, str):
        symbols = [symbols]
    
    data = {}
    
    for symbol in symbols:
        try:
            print(f"Downloading data for {symbol}...")
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            
            if len(df) > 0:
                # Clean column names
                df.columns = [col.lower().replace(' ', '_') for col in df.columns]
                df.index.name = 'date'
                df.reset_index(inplace=True)
                
                data[symbol] = df
                print(f"Downloaded {len(df)} records for {symbol}")
            else:
                print(f"No data found for {symbol}")
                
        except Exception as e:
            print(f"Error downloading {symbol}: {e}")
    
    # Always return a dictionary for consistency
    return data


def create_sequences(data, sequence_length, prediction_horizon, target_column='close'):
    """
    Create sequences for time series prediction
    
    Args:
        data: Input data array
        sequence_length: Length of input sequences
        prediction_horizon: Number of steps to predict
        target_column: Index of target column (default: 3 for close price)
        
    Returns:
        Tuple of (sequences, targets, action_labels)
    """
    
    if len(data) < sequence_length + prediction_horizon:
        raise ValueError(f"Data too short: {len(data)} < {sequence_length + prediction_horizon}")
    
    sequences = []
    targets = []
    action_labels = []
    
    for i in range(len(data) - sequence_length - prediction_horizon + 1):
        # Input sequence
        seq = data[i:i + sequence_length]
        sequences.append(seq)
        
        # Target sequence (future prices)
        target_start = i + sequence_length
        target_end = target_start + prediction_horizon
        target = data[target_start:target_end]
        targets.append(target)
        
        # Action label (buy/hold/sell based on next price movement)
        current_price = data[i + sequence_length - 1, 3]  # Last close price in sequence
        next_price = data[i + sequence_length, 3]  # Next close price
        
        price_change = (next_price - current_price) / current_price
        
        if price_change > 0.01:  # 1% threshold
            action_label = 0  # Buy
        elif price_change < -0.01:
            action_label = 2  # Sell
        else:
            action_label = 1  # Hold
            
        action_labels.append(action_label)
    
    return np.array(sequences), np.array(targets), np.array(action_labels)


def split_data(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split data into train/validation/test sets
    
    Args:
        data: Input data
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    return train_data, val_data, test_data


def augment_data(data, noise_factor=0.01, scaling_factor=0.05):
    """
    Augment time series data with noise and scaling
    
    Args:
        data: Input data array
        noise_factor: Standard deviation of Gaussian noise
        scaling_factor: Standard deviation of scaling factor
        
    Returns:
        Augmented data
    """
    
    augmented = data.copy()
    
    # Add Gaussian noise
    noise = np.random.normal(0, noise_factor, data.shape)
    augmented += noise
    
    # Random scaling
    scaling = np.random.normal(1.0, scaling_factor, (data.shape[0], 1))
    augmented *= scaling
    
    return augmented


def load_training_data(data_dir="trainingdata", symbols=None, start_date='2015-01-01'):
    """
    Load training data from various sources
    
    Args:
        data_dir: Directory containing CSV files
        symbols: List of symbols to download if no local data
        start_date: Start date for downloading data
        
    Returns:
        Processed data array
    """
    
    data_path = Path(data_dir)
    
    # Try to load from local CSV files first
    if data_path.exists():
        csv_files = list(data_path.glob("*.csv"))
        if csv_files:
            print(f"Found {len(csv_files)} CSV files in {data_dir}")
            
            # Load and combine all CSV files
            all_data = []
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    print(f"Loaded {csv_file.name}: {len(df)} records")
                    
                    # Assume standard OHLCV format
                    if 'close' in df.columns.str.lower():
                        processor = StockDataProcessor()
                        features = processor.prepare_features(df)
                        all_data.append(features)
                        
                except Exception as e:
                    print(f"Error loading {csv_file}: {e}")
            
            if all_data:
                # Combine all data
                combined_data = np.vstack(all_data)
                print(f"Combined data shape: {combined_data.shape}")
                return combined_data
    
    # If no local data, try to download
    if symbols:
        print(f"No local data found. Downloading symbols: {symbols}")
        data_dict = download_stock_data(symbols, start_date)
        
        if data_dict and len(data_dict) > 0:
            all_data = []
            processor = StockDataProcessor()
            
            for symbol, df in data_dict.items():
                features = processor.prepare_features(df)
                all_data.append(features)
            
            combined_data = np.vstack(all_data)
            print(f"Downloaded data shape: {combined_data.shape}")
            
            # Save for future use
            data_path.mkdir(parents=True, exist_ok=True)
            for symbol, df in data_dict.items():
                df.to_csv(data_path / f"{symbol}.csv", index=False)
            
            return combined_data
    
    # Generate synthetic data as fallback
    print("No data sources available. Generating synthetic data...")
    return generate_synthetic_data()


def generate_synthetic_data(length=10000, n_features=25):
    """
    Generate synthetic stock-like data for testing
    
    Args:
        length: Number of time steps
        n_features: Number of features
        
    Returns:
        Synthetic data array
    """
    
    np.random.seed(42)
    
    # Generate realistic stock price movements
    initial_price = 100.0
    returns = np.random.normal(0.0005, 0.02, length)  # 0.05% daily return, 2% volatility
    prices = [initial_price]
    
    for i in range(1, length):
        new_price = prices[-1] * (1 + returns[i])
        prices.append(max(new_price, 0.01))  # Prevent negative prices
    
    prices = np.array(prices)
    
    # Generate OHLCV data
    data = []
    for i in range(len(prices)):
        price = prices[i]
        
        # Generate realistic OHLC from close price
        volatility = abs(np.random.normal(0, 0.01))
        high = price * (1 + volatility)
        low = price * (1 - volatility)
        open_price = np.random.uniform(low, high)
        
        # Volume (random but realistic)
        volume = np.random.exponential(1000000)
        
        # Additional synthetic features
        features = [open_price, high, low, price, volume]
        
        # Add more synthetic technical indicators
        for j in range(n_features - 5):
            features.append(np.random.normal(0, 1))
        
        data.append(features)
    
    data = np.array(data)
    print(f"Generated synthetic data: {data.shape}")
    
    return data


class DataCollator:
    """Data collator for batching sequences"""
    
    def __init__(self, pad_token_id=0):
        self.pad_token_id = pad_token_id
    
    def __call__(self, examples):
        """Collate examples into a batch"""
        
        batch = {}
        
        # Get max sequence length in batch
        max_len = max(example['input_ids'].shape[0] for example in examples)
        
        # Pad sequences
        input_ids = []
        attention_masks = []
        labels = []
        action_labels = []
        
        for example in examples:
            seq_len = example['input_ids'].shape[0]
            
            # Pad input_ids
            padded_input = torch.zeros(max_len, example['input_ids'].shape[1])
            padded_input[:seq_len] = example['input_ids']
            input_ids.append(padded_input)
            
            # Create attention mask
            attention_mask = torch.zeros(max_len)
            attention_mask[:seq_len] = 1
            attention_masks.append(attention_mask)
            
            labels.append(example['labels'])
            action_labels.append(example['action_labels'])
        
        batch['input_ids'] = torch.stack(input_ids)
        batch['attention_mask'] = torch.stack(attention_masks)
        batch['labels'] = torch.stack(labels)
        batch['action_labels'] = torch.stack(action_labels)
        
        return batch