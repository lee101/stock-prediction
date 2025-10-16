#!/usr/bin/env python3
"""
Robust Data Pipeline with Error Handling and Validation
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
import logging
from collections import defaultdict
from data_utils import load_local_stock_data
from sklearn.preprocessing import RobustScaler
import ta  # Technical analysis library

warnings.filterwarnings('ignore')

class DataValidator:
    """Validates and cleans financial data"""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_cols: List[str] = None) -> pd.DataFrame:
        """Validate and clean dataframe"""
        if required_cols is None:
            required_cols = ['open', 'high', 'low', 'close', 'volume']
        
        # Ensure columns exist
        df.columns = df.columns.str.lower()
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Remove non-numeric data
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with NaN in critical columns
        df = df.dropna(subset=required_cols)
        
        # Validate price relationships
        invalid_rows = (
            (df['high'] < df['low']) |
            (df['high'] < df['close']) |
            (df['low'] > df['close']) |
            (df['volume'] < 0)
        )
        
        if invalid_rows.any():
            logging.warning(f"Removing {invalid_rows.sum()} invalid rows")
            df = df[~invalid_rows]
        
        # Sort by date if available
        if 'date' in df.columns:
            df = df.sort_values('date')
        
        return df
    
    @staticmethod
    def validate_sequence(sequence: np.ndarray, name: str = "sequence") -> np.ndarray:
        """Validate sequence data"""
        if not isinstance(sequence, np.ndarray):
            sequence = np.array(sequence)
        
        # Check for NaN or Inf
        if np.any(np.isnan(sequence)):
            logging.warning(f"{name} contains NaN values, replacing with 0")
            sequence = np.nan_to_num(sequence, nan=0.0)
        
        if np.any(np.isinf(sequence)):
            logging.warning(f"{name} contains Inf values, clipping")
            sequence = np.clip(sequence, -1e10, 1e10)
        
        return sequence


class EnhancedStockDataset(Dataset):
    """Enhanced dataset with robust error handling"""
    
    def __init__(self, 
                 data: np.ndarray,
                 sequence_length: int = 60,
                 prediction_horizon: int = 5,
                 augment: bool = False,
                 cache_size: int = 1000):
        
        self.validator = DataValidator()
        self.data = self.validator.validate_sequence(data, "input_data")
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.augment = augment
        
        # Cache for processed samples
        self.cache = {}
        self.cache_size = cache_size
        
        # Validate data shape
        if len(self.data) < sequence_length + prediction_horizon:
            raise ValueError(
                f"Insufficient data: {len(self.data)} < "
                f"{sequence_length + prediction_horizon}"
            )
        
        # Pre-compute valid indices
        self.valid_indices = []
        for i in range(len(self.data) - sequence_length - prediction_horizon + 1):
            # Check if this index produces valid data
            end_idx = i + sequence_length + prediction_horizon
            if end_idx <= len(self.data):
                self.valid_indices.append(i)
        
        logging.info(f"Dataset created with {len(self.valid_indices)} valid samples")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        # Use cache if available
        if idx in self.cache:
            return self.cache[idx]
        
        try:
            # Get actual data index
            data_idx = self.valid_indices[idx]
            
            # Extract sequence
            sequence = self.data[data_idx:data_idx + self.sequence_length]
            sequence = self.validator.validate_sequence(sequence, f"sequence_{idx}")
            
            # Extract targets
            target_start = data_idx + self.sequence_length
            target_end = target_start + self.prediction_horizon
            targets = self.data[target_start:target_end]
            targets = self.validator.validate_sequence(targets, f"targets_{idx}")
            
            # Generate action label based on price movement
            if len(sequence) > 0 and len(targets) > 0:
                current_price = sequence[-1, 3]  # Last close price
                next_price = targets[0, 3]  # Next close price
                
                # Calculate percentage change
                price_change = (next_price - current_price) / (current_price + 1e-8)
                
                # Determine action with thresholds
                if price_change > 0.005:  # 0.5% threshold
                    action_label = 0  # Buy
                elif price_change < -0.005:
                    action_label = 2  # Sell
                else:
                    action_label = 1  # Hold
            else:
                action_label = 1  # Default to hold
            
            # Apply augmentation if enabled
            if self.augment and np.random.random() > 0.5:
                sequence = self._augment_sequence(sequence)
            
            # Create sample
            sample = {
                'input_ids': torch.FloatTensor(sequence),
                'labels': torch.FloatTensor(targets),
                'action_labels': torch.LongTensor([action_label]),
                'attention_mask': torch.ones(self.sequence_length),
                'idx': idx
            }
            
            # Cache if space available
            if len(self.cache) < self.cache_size:
                self.cache[idx] = sample
            
            return sample
            
        except Exception as e:
            logging.error(f"Error processing sample {idx}: {e}")
            # Return a valid dummy sample to avoid breaking training
            return self._get_dummy_sample()
    
    def _augment_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """Apply data augmentation"""
        # Add small noise
        noise = np.random.normal(0, 0.001, sequence.shape)
        sequence = sequence + noise
        
        # Random scaling
        scale = np.random.uniform(0.98, 1.02)
        sequence = sequence * scale
        
        return sequence
    
    def _get_dummy_sample(self):
        """Get a dummy sample for error cases"""
        return {
            'input_ids': torch.zeros(self.sequence_length, self.data.shape[1]),
            'labels': torch.zeros(self.prediction_horizon, self.data.shape[1]),
            'action_labels': torch.LongTensor([1]),  # Hold
            'attention_mask': torch.ones(self.sequence_length),
            'idx': -1
        }


class RobustCollator:
    """Robust data collator with error handling"""
    
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch: List[Dict]) -> Dict:
        """Collate batch with validation"""
        # Filter out invalid samples
        valid_batch = [sample for sample in batch if sample['idx'] != -1]
        
        if not valid_batch:
            logging.warning("No valid samples in batch, using dummy batch")
            valid_batch = [batch[0]]  # Use at least one sample
        
        # Stack tensors
        try:
            batch_dict = {
                'input_ids': torch.stack([s['input_ids'] for s in valid_batch]),
                'labels': torch.stack([s['labels'] for s in valid_batch]),
                'action_labels': torch.stack([s['action_labels'] for s in valid_batch]),
                'attention_mask': torch.stack([s['attention_mask'] for s in valid_batch])
            }
            
            # Validate batch shapes
            batch_size = batch_dict['input_ids'].size(0)
            assert batch_dict['action_labels'].size(0) == batch_size, "Action labels batch size mismatch"
            
            return batch_dict
            
        except Exception as e:
            logging.error(f"Error in collator: {e}")
            # Return minimal valid batch
            return {
                'input_ids': valid_batch[0]['input_ids'].unsqueeze(0),
                'labels': valid_batch[0]['labels'].unsqueeze(0),
                'action_labels': valid_batch[0]['action_labels'].unsqueeze(0),
                'attention_mask': valid_batch[0]['attention_mask'].unsqueeze(0)
            }


class AdvancedDataProcessor:
    """Advanced data processor with technical indicators"""
    
    def __init__(self):
        self.scaler = RobustScaler()
        self.feature_names = []
    
    def process_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """Process dataframe with technical indicators"""
        df = df.copy()
        
        # Basic features
        features = ['open', 'high', 'low', 'close', 'volume']
        
        # Price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['price_range'] = (df['high'] - df['low']) / df['close']
        df['close_to_open'] = (df['close'] - df['open']) / df['open']
        
        features.extend(['returns', 'log_returns', 'price_range', 'close_to_open'])
        
        # Volume features
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        features.extend(['volume_sma', 'volume_ratio'])
        
        # Technical indicators using ta library
        try:
            # Trend
            df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
            df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)
            df['macd'] = ta.trend.macd(df['close'])
            df['macd_signal'] = ta.trend.macd_signal(df['close'])
            
            # Momentum
            df['rsi'] = ta.momentum.rsi(df['close'])
            df['stoch'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
            
            # Volatility
            df['bb_high'] = ta.volatility.bollinger_hband(df['close'])
            df['bb_low'] = ta.volatility.bollinger_lband(df['close'])
            df['bb_width'] = df['bb_high'] - df['bb_low']
            df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
            
            features.extend([
                'sma_20', 'ema_20', 'macd', 'macd_signal',
                'rsi', 'stoch', 'bb_high', 'bb_low', 'bb_width', 'atr'
            ])
        except:
            logging.warning("Could not compute all technical indicators")
        
        # Select features that exist
        available_features = [f for f in features if f in df.columns]
        self.feature_names = available_features
        
        # Extract and clean data
        data = df[available_features].values
        
        # Handle NaN and Inf
        data = np.nan_to_num(data, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Normalize
        data = self.scaler.fit_transform(data)
        
        return data


def create_robust_dataloader(
    data: np.ndarray,
    batch_size: int = 32,
    sequence_length: int = 60,
    prediction_horizon: int = 5,
    shuffle: bool = True,
    num_workers: int = 4,
    augment: bool = False
) -> DataLoader:
    """Create a robust dataloader with error handling"""
    
    # Create dataset
    dataset = EnhancedStockDataset(
        data=data,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        augment=augment
    )
    
    # Create collator
    collator = RobustCollator()
    
    # Create dataloader with error handling
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        drop_last=True,  # Drop incomplete batches
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    return dataloader


def download_and_process_stocks(
    symbols: List[str],
    start_date: str = '2018-01-01',
    end_date: str = None
) -> Tuple[np.ndarray, List[str]]:
    """Load local CSVs for symbols and process with indicators.

    This function no longer downloads via yfinance. It expects CSVs under
    the trainingdata/ directory (or a configured data dir) and will raise if
    none are found.
    """
    
    processor = AdvancedDataProcessor()
    validator = DataValidator()
    all_data = []
    
    # Load local CSVs
    local = load_local_stock_data(symbols, data_dir="trainingdata")
    if not local:
        raise ValueError("No local CSVs found for provided symbols under trainingdata/")

    for symbol, df in local.items():
        try:
            logging.info(f"Processing {symbol}...")

            if len(df) < 100:
                logging.warning(f"Insufficient data for {symbol}, skipping")
                continue
            
            # Clean column names and ensure lowercase
            df = df.copy()
            df.columns = df.columns.str.lower()
            if 'date' in df.columns:
                try:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date')
                except Exception:
                    pass
            
            # Validate data
            df = validator.validate_dataframe(df)
            
            # Process with technical indicators
            processed_data = processor.process_dataframe(df)
            
            all_data.append(processed_data)
            logging.info(f"Processed {symbol}: {processed_data.shape}")
            
        except Exception as e:
            logging.error(f"Failed to process {symbol}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No valid stock data could be processed")
    
    # Combine all data
    combined_data = np.vstack(all_data)
    logging.info(f"Combined data shape: {combined_data.shape}")
    
    return combined_data, processor.feature_names


def test_data_pipeline():
    """Test the robust data pipeline"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Robust Data Pipeline...")
    
    # Test with synthetic data
    np.random.seed(42)
    test_data = np.random.randn(1000, 10)
    
    # Create dataloader
    dataloader = create_robust_dataloader(
        data=test_data,
        batch_size=16,
        sequence_length=30,
        prediction_horizon=5,
        shuffle=True,
        num_workers=0,
        augment=True
    )
    
    # Test iteration
    print(f"Dataloader created with {len(dataloader)} batches")
    
    for i, batch in enumerate(dataloader):
        print(f"Batch {i}: input_ids={batch['input_ids'].shape}, "
              f"action_labels={batch['action_labels'].shape}")
        
        # Validate batch
        assert batch['input_ids'].size(0) == batch['action_labels'].size(0), "Batch size mismatch!"
        
        if i >= 5:  # Test first 5 batches
            break
    
    print("✅ Data pipeline test passed!")
    
    # Test with real stock data
    print("\nTesting with real stock data...")
    try:
        data, features = download_and_process_stocks(['AAPL'], start_date='2023-01-01')
        print(f"Downloaded data shape: {data.shape}")
        print(f"Features: {features[:5]}...")
        
        # Create dataloader with real data
        real_dataloader = create_robust_dataloader(
            data=data,
            batch_size=8,
            sequence_length=60,
            prediction_horizon=5
        )
        
        # Test one batch
        batch = next(iter(real_dataloader))
        print(f"Real data batch: input_ids={batch['input_ids'].shape}")
        print("✅ Real data test passed!")
        
    except Exception as e:
        print(f"Real data test failed (may need internet): {e}")
    
    return True


if __name__ == "__main__":
    test_data_pipeline()
