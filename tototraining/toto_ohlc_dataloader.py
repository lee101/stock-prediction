#!/usr/bin/env python3
"""
Comprehensive OHLC DataLoader for Toto Model Training

This module provides a robust dataloader system for training the Toto transformer model
on OHLC stock data with proper preprocessing, normalization, and batching.
"""

import os
import sys
import json
import logging
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, NamedTuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import random

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

# Add the toto directory to sys.path
toto_path = Path(__file__).parent.parent / "toto"
sys.path.insert(0, str(toto_path))

try:
    from toto.data.util.dataset import MaskedTimeseries, pad_array, pad_id_mask, replace_extreme_values
except ImportError:
    # Create minimal fallback implementations for testing
    from typing import NamedTuple
    try:
        from jaxtyping import Bool, Float, Int
    except ImportError:
        # Fallback type aliases if jaxtyping not available
        Bool = torch.Tensor
        Float = torch.Tensor
        Int = torch.Tensor
    import torch
    
    class MaskedTimeseries(NamedTuple):
        series: torch.Tensor
        padding_mask: torch.Tensor
        id_mask: torch.Tensor
        timestamp_seconds: torch.Tensor
        time_interval_seconds: torch.Tensor
        
        def to(self, device: torch.device) -> "MaskedTimeseries":
            return MaskedTimeseries(
                series=self.series.to(device),
                padding_mask=self.padding_mask.to(device),
                id_mask=self.id_mask.to(device),
                timestamp_seconds=self.timestamp_seconds.to(device),
                time_interval_seconds=self.time_interval_seconds.to(device),
            )
    
    def replace_extreme_values(t: torch.Tensor, replacement: float = 0.0) -> torch.Tensor:
        """Replace extreme values with replacement value"""
        is_extreme = torch.logical_or(
            torch.logical_or(torch.isinf(t), torch.isnan(t)),
            t.abs() >= 1e10
        )
        return torch.where(is_extreme, torch.tensor(replacement, dtype=t.dtype, device=t.device), t)


@dataclass
class DataLoaderConfig:
    """Configuration for OHLC DataLoader"""
    # Data paths
    train_data_path: str = "trainingdata/train"
    test_data_path: str = "trainingdata/test"
    
    # Model parameters
    patch_size: int = 12
    stride: int = 6
    sequence_length: int = 96  # Number of time steps to use as input
    prediction_length: int = 24  # Number of time steps to predict
    
    # Data preprocessing
    normalization_method: str = "robust"  # "standard", "minmax", "robust"
    handle_missing: str = "interpolate"  # "drop", "interpolate", "zero"
    outlier_threshold: float = 3.0  # Standard deviations for outlier detection
    
    # Training parameters
    batch_size: int = 32
    validation_split: float = 0.2  # Fraction for validation
    test_split_days: int = 30  # Last N days for test set
    
    # Cross-validation
    cv_folds: int = 5
    cv_gap: int = 24  # Gap between train/val in CV (hours)
    
    # Data filtering
    min_sequence_length: int = 100  # Minimum length for a valid sequence
    max_symbols: Optional[int] = None  # Maximum number of symbols to load
    
    # Features to use
    ohlc_features: List[str] = None
    additional_features: List[str] = None
    target_feature: str = "Close"
    
    # Technical indicators
    add_technical_indicators: bool = True
    rsi_period: int = 14
    ma_periods: List[int] = None
    
    # Data loading
    num_workers: int = 2
    pin_memory: bool = True
    drop_last: bool = True
    
    # Random seed
    random_seed: int = 42
    
    def __post_init__(self):
        if self.ohlc_features is None:
            self.ohlc_features = ["Open", "High", "Low", "Close"]
        if self.additional_features is None:
            self.additional_features = ["Volume"]
        if self.ma_periods is None:
            self.ma_periods = [5, 10, 20]
    
    def save(self, path: str):
        """Save configuration to JSON file"""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


class OHLCPreprocessor:
    """Handles OHLC data preprocessing and feature engineering"""
    
    def __init__(self, config: DataLoaderConfig):
        self.config = config
        self.scalers = {}
        self.fitted = False
        
        # Initialize scalers
        if config.normalization_method == "standard":
            self.scaler_class = StandardScaler
        elif config.normalization_method == "minmax":
            self.scaler_class = MinMaxScaler
        else:  # robust
            self.scaler_class = RobustScaler
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataframe"""
        if not self.config.add_technical_indicators:
            return df
        
        df = df.copy()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.config.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config.rsi_period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Moving averages
        for period in self.config.ma_periods:
            df[f'MA_{period}'] = df['Close'].rolling(window=period).mean()
            df[f'MA_{period}_ratio'] = df['Close'] / df[f'MA_{period}']
        
        # Price momentum
        df['price_momentum_1'] = df['Close'].pct_change(1)
        df['price_momentum_5'] = df['Close'].pct_change(5)
        
        # Volatility (rolling standard deviation)
        df['volatility'] = df['Close'].rolling(window=20).std()
        
        # OHLC ratios
        df['hl_ratio'] = (df['High'] - df['Low']) / df['Close']
        df['oc_ratio'] = (df['Close'] - df['Open']) / df['Open']
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values according to configuration"""
        if self.config.handle_missing == "drop":
            return df.dropna()
        elif self.config.handle_missing == "interpolate":
            return df.interpolate(method='linear', limit_direction='both')
        else:  # zero
            return df.fillna(0)
    
    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers based on z-score threshold"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col not in ['timestamp']:  # Skip timestamp column
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df = df[z_scores < self.config.outlier_threshold]
        
        return df
    
    def fit_scalers(self, data: Dict[str, pd.DataFrame]):
        """Fit scalers on training data"""
        # Combine all training data for fitting scalers
        all_data = pd.concat(list(data.values()), ignore_index=True)
        
        # Get feature columns (exclude timestamp)
        feature_cols = [col for col in all_data.columns if col != 'timestamp']
        
        for col in feature_cols:
            if all_data[col].dtype in [np.float32, np.float64, np.int32, np.int64]:
                scaler = self.scaler_class()
                valid_data = all_data[col].dropna()
                if len(valid_data) > 0:
                    scaler.fit(valid_data.values.reshape(-1, 1))
                    self.scalers[col] = scaler
        
        self.fitted = True
    
    def transform(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """Apply preprocessing transformations"""
        if not self.fitted:
            raise ValueError("Scalers must be fitted before transformation")
        
        df = df.copy()

        # Ensure numeric columns are float32 for compatibility with scalers
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].astype(np.float32, copy=False)
        
        # Add technical indicators
        df = self.add_technical_indicators(df)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Remove outliers
        df = self.remove_outliers(df)
        
        # Apply normalization
        for col, scaler in self.scalers.items():
            if col in df.columns:
                valid_mask = ~df[col].isna()
                if valid_mask.any():
                    df.loc[valid_mask, col] = scaler.transform(
                        df.loc[valid_mask, col].values.reshape(-1, 1)
                    ).flatten()
        
        # Replace extreme values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'timestamp':
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                df[col] = df[col].fillna(0)
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare feature array for model input"""
        feature_cols = (self.config.ohlc_features + 
                       self.config.additional_features)
        
        # Add technical indicator columns if enabled
        if self.config.add_technical_indicators:
            tech_cols = ['RSI', 'volatility', 'hl_ratio', 'oc_ratio', 
                        'price_momentum_1', 'price_momentum_5']
            tech_cols += [f'MA_{p}_ratio' for p in self.config.ma_periods]
            feature_cols.extend(tech_cols)
        
        # Filter existing columns
        available_cols = [col for col in feature_cols if col in df.columns]
        
        if not available_cols:
            raise ValueError(f"No valid feature columns found in data")
        
        return df[available_cols].values.astype(np.float32)


class OHLCDataset(Dataset):
    """PyTorch Dataset for OHLC data compatible with Toto model"""
    
    def __init__(self, 
                 data: Dict[str, pd.DataFrame], 
                 config: DataLoaderConfig,
                 preprocessor: OHLCPreprocessor,
                 mode: str = 'train'):
        
        self.config = config
        self.preprocessor = preprocessor
        self.mode = mode
        self.sequences = []
        self.symbol_mapping = {}
        
        # Process and prepare sequences
        self._prepare_sequences(data)
        
        # Set random seed
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
    
    def _prepare_sequences(self, data: Dict[str, pd.DataFrame]):
        """Prepare sequences from raw data"""
        symbol_id = 0
        
        for symbol, df in data.items():
            if len(df) < self.config.min_sequence_length:
                continue
            
            # Transform data using preprocessor
            try:
                processed_df = self.preprocessor.transform(df, symbol)
                features = self.preprocessor.prepare_features(processed_df)
                
                if len(features) < self.config.sequence_length + self.config.prediction_length:
                    continue
                
                # Create time intervals (assume regular intervals)
                if 'timestamp' in processed_df.columns:
                    timestamps = pd.to_datetime(processed_df['timestamp']).astype(np.int64) // 10**9
                    timestamps = timestamps.values  # Convert to numpy array
                    time_intervals = np.diff(timestamps)
                    avg_interval = int(np.median(time_intervals)) if len(time_intervals) > 0 else 3600
                else:
                    avg_interval = 3600  # Default 1 hour
                    timestamps = np.arange(len(features), dtype=np.int64) * avg_interval
                
                # Store symbol mapping
                self.symbol_mapping[symbol] = symbol_id
                
                # Create sequences with sliding window
                max_start_idx = len(features) - self.config.sequence_length - self.config.prediction_length
                
                for start_idx in range(0, max_start_idx + 1, self.config.stride):
                    end_idx = start_idx + self.config.sequence_length
                    pred_end_idx = end_idx + self.config.prediction_length
                    
                    if pred_end_idx <= len(features):
                        sequence_data = {
                            'features': features[start_idx:end_idx],
                            'target': features[end_idx:pred_end_idx, 
                                            self._get_target_column_index(processed_df)],
                            'symbol_id': symbol_id,
                            'symbol_name': symbol,
                            'timestamps': timestamps[start_idx:end_idx],
                            'time_interval': avg_interval,
                            'start_idx': start_idx
                        }
                        self.sequences.append(sequence_data)
                
                symbol_id += 1
                
            except Exception as e:
                logging.warning(f"Error processing symbol {symbol}: {e}")
                continue
    
    def _get_target_column_index(self, df: pd.DataFrame) -> int:
        """Get the index of target column"""
        feature_cols = (self.config.ohlc_features + 
                       self.config.additional_features)
        
        if self.config.add_technical_indicators:
            tech_cols = ['RSI', 'volatility', 'hl_ratio', 'oc_ratio', 
                        'price_momentum_1', 'price_momentum_5']
            tech_cols += [f'MA_{p}_ratio' for p in self.config.ma_periods]
            feature_cols.extend(tech_cols)
        
        available_cols = [col for col in feature_cols if col in df.columns]
        
        if self.config.target_feature in available_cols:
            return available_cols.index(self.config.target_feature)
        else:
            return 0  # Default to first column
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> MaskedTimeseries:
        """Return a MaskedTimeseries object compatible with Toto model"""
        seq = self.sequences[idx]
        
        # Prepare tensor data
        series = torch.from_numpy(seq['features'].T).float()  # Shape: (features, time)
        n_features, seq_len = series.shape
        
        # Create padding mask (all True since we don't have padding here)
        padding_mask = torch.ones(n_features, seq_len, dtype=torch.bool)

        # Create ID mask (same ID for all features of same symbol)
        id_mask = torch.full((n_features, seq_len), seq['symbol_id'], dtype=torch.long)

        # Create timestamps
        timestamps = torch.from_numpy(seq['timestamps']).long()
        timestamps = timestamps.unsqueeze(0).repeat(n_features, 1)
        
        # Time intervals
        time_intervals = torch.full((n_features,), seq['time_interval'], dtype=torch.long)
        
        # Handle extreme values
        series = replace_extreme_values(series, replacement=0.0)
        
        masked = MaskedTimeseries(
            series=series,
            padding_mask=padding_mask,
            id_mask=id_mask,
            timestamp_seconds=timestamps,
            time_interval_seconds=time_intervals
        )
        
        target = torch.from_numpy(seq['target']).float()
        
        return masked, target
    
    def get_targets(self) -> torch.Tensor:
        """Get all targets for this dataset"""
        targets = []
        for seq in self.sequences:
            targets.append(torch.from_numpy(seq['target']).float())
        return torch.stack(targets) if targets else torch.empty(0)


class TotoOHLCDataLoader:
    """Comprehensive DataLoader for Toto OHLC training"""
    
    def __init__(self, config: DataLoaderConfig):
        self.config = config
        self.preprocessor = OHLCPreprocessor(config)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Data storage
        self.train_data = {}
        self.val_data = {}
        self.test_data = {}
        
        # Set random seeds
        self._set_random_seeds()
    
    def _set_random_seeds(self):
        """Set random seeds for reproducibility"""
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.random_seed)
    
    def load_data(self) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """Load and split OHLC data from train/test directories"""
        train_data = {}
        test_data = {}
        
        # Load training data
        train_path = self._resolve_path(self.config.train_data_path)
        if train_path.exists():
            train_data = self._load_data_from_directory(train_path, "train")
        else:
            self.logger.warning(f"Training data path does not exist: {train_path}")
        
        # Load test data
        test_path = self._resolve_path(self.config.test_data_path)
        if test_path.exists():
            test_data = self._load_data_from_directory(test_path, "test")
        elif self.config.test_data_path:
            self.logger.warning(f"Test data path does not exist: {test_path}")
        
        # If no separate test data, use time-based split
        if not test_data and train_data:
            train_data, test_data = self._time_split_data(train_data)
        
        # Create validation split from training data
        train_data, val_data = self._validation_split(train_data)
        
        self.logger.info(f"Loaded {len(train_data)} training symbols, "
                        f"{len(val_data)} validation symbols, "
                        f"{len(test_data)} test symbols")
        
        return train_data, val_data, test_data

    def _resolve_path(self, path_str: str) -> Path:
        """Resolve relative paths against the tototraining directory"""
        if not path_str:
            return Path(__file__).parent
        path = Path(path_str)
        if path.is_absolute():
            return path
        
        cwd_candidate = (Path.cwd() / path).resolve()
        if cwd_candidate.exists():
            return cwd_candidate
        
        return (Path(__file__).parent / path).resolve()
    
    def _load_data_from_directory(self, directory: Path, split_name: str) -> Dict[str, pd.DataFrame]:
        """Load CSV files from directory"""
        data = {}
        csv_files = list(directory.glob("*.csv"))
        
        # Limit number of symbols if specified
        if self.config.max_symbols and len(csv_files) > self.config.max_symbols:
            csv_files = csv_files[:self.config.max_symbols]
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)

                # Normalize column casing for OHLCV schema
                column_renames = {}
                for col in df.columns:
                    col_lower = col.lower()
                    if col_lower == "open":
                        column_renames[col] = "Open"
                    elif col_lower == "high":
                        column_renames[col] = "High"
                    elif col_lower == "low":
                        column_renames[col] = "Low"
                    elif col_lower == "close":
                        column_renames[col] = "Close"
                    elif col_lower == "volume":
                        column_renames[col] = "Volume"
                    elif col_lower == "timestamp":
                        column_renames[col] = "timestamp"
                if column_renames:
                    df = df.rename(columns=column_renames)
                
                # Basic validation
                required_cols = set(self.config.ohlc_features)
                if not required_cols.issubset(set(df.columns)):
                    self.logger.warning(f"Missing required columns in {csv_file}")
                    continue
                
                # Parse timestamp if exists
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.sort_values('timestamp').reset_index(drop=True)
                
                # Filter minimum length
                if len(df) >= self.config.min_sequence_length:
                    symbol = csv_file.stem
                    data[symbol] = df
                    
            except Exception as e:
                self.logger.warning(f"Error loading {csv_file}: {e}")
                continue
        
        self.logger.info(f"Loaded {len(data)} files from {directory}")
        return data
    
    def _time_split_data(self, data: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """Split data based on time (last N days for test)"""
        train_data = {}
        test_data = {}
        
        for symbol, df in data.items():
            if 'timestamp' in df.columns and len(df) > self.config.min_sequence_length:
                # Calculate split point
                last_date = df['timestamp'].max()
                split_date = last_date - timedelta(days=self.config.test_split_days)
                
                train_df = df[df['timestamp'] <= split_date].copy()
                test_df = df[df['timestamp'] > split_date].copy()
                
                if len(train_df) >= self.config.min_sequence_length:
                    train_data[symbol] = train_df
                if len(test_df) >= self.config.min_sequence_length:
                    test_data[symbol] = test_df
            else:
                # Fallback to simple split
                split_idx = int(len(df) * 0.8)
                train_data[symbol] = df.iloc[:split_idx].copy()
                if len(df) - split_idx >= self.config.min_sequence_length:
                    test_data[symbol] = df.iloc[split_idx:].copy()
        
        return train_data, test_data
    
    def _validation_split(self, train_data: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """Create validation split from training data"""
        if self.config.validation_split <= 0:
            return train_data, {}
        
        symbols = list(train_data.keys())
        random.shuffle(symbols)
        
        split_idx = int(len(symbols) * (1 - self.config.validation_split))
        train_symbols = symbols[:split_idx]
        val_symbols = symbols[split_idx:]
        
        new_train_data = {s: train_data[s] for s in train_symbols}
        val_data = {s: train_data[s] for s in val_symbols}
        
        return new_train_data, val_data
    
    def prepare_dataloaders(self) -> Dict[str, DataLoader]:
        """Prepare PyTorch DataLoaders for training"""
        # Load data
        train_data, val_data, test_data = self.load_data()
        
        if not train_data:
            raise ValueError("No training data found!")
        
        # Fit preprocessor on training data
        self.preprocessor.fit_scalers(train_data)
        
        # Create datasets
        datasets = {}
        dataloaders = {}
        
        if train_data:
            datasets['train'] = OHLCDataset(train_data, self.config, self.preprocessor, 'train')
            dataloaders['train'] = DataLoader(
                datasets['train'],
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                drop_last=self.config.drop_last
            )
        
        if val_data:
            datasets['val'] = OHLCDataset(val_data, self.config, self.preprocessor, 'val')
            dataloaders['val'] = DataLoader(
                datasets['val'],
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                drop_last=self.config.drop_last
            )
        
        if test_data:
            datasets['test'] = OHLCDataset(test_data, self.config, self.preprocessor, 'test')
            dataloaders['test'] = DataLoader(
                datasets['test'],
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                drop_last=False
            )
        
        self.logger.info(f"Created dataloaders: {list(dataloaders.keys())}")
        for name, loader in dataloaders.items():
            self.logger.info(f"{name}: {len(loader.dataset)} samples, {len(loader)} batches")
        
        # Store references
        self.train_data = train_data
        self.val_data = val_data  
        self.test_data = test_data
        
        return dataloaders
    
    def get_cross_validation_splits(self, n_splits: int = None) -> List[Tuple[DataLoader, DataLoader]]:
        """Generate cross-validation splits for time series data"""
        if n_splits is None:
            n_splits = self.config.cv_folds
        
        if not self.train_data:
            raise ValueError("No training data loaded!")
        
        cv_splits = []
        # Note: gap parameter may not be available in older sklearn versions
        try:
            tscv = TimeSeriesSplit(n_splits=n_splits, gap=self.config.cv_gap)
        except TypeError:
            # Fallback for older sklearn versions
            tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Combine all data for CV splitting
        all_symbols = list(self.train_data.keys())
        
        for fold, (train_indices, val_indices) in enumerate(tscv.split(all_symbols)):
            train_symbols = [all_symbols[i] for i in train_indices]
            val_symbols = [all_symbols[i] for i in val_indices]
            
            fold_train_data = {s: self.train_data[s] for s in train_symbols}
            fold_val_data = {s: self.train_data[s] for s in val_symbols}
            
            # Create datasets for this fold
            train_dataset = OHLCDataset(fold_train_data, self.config, self.preprocessor, 'train')
            val_dataset = OHLCDataset(fold_val_data, self.config, self.preprocessor, 'val')
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                drop_last=self.config.drop_last
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                drop_last=self.config.drop_last
            )
            
            cv_splits.append((train_loader, val_loader))
            self.logger.info(f"CV Fold {fold + 1}: {len(train_loader.dataset)} train, "
                           f"{len(val_loader.dataset)} val samples")
        
        return cv_splits
    
    def get_feature_info(self) -> Dict:
        """Get information about features used"""
        feature_cols = (self.config.ohlc_features + 
                       self.config.additional_features)
        
        if self.config.add_technical_indicators:
            tech_cols = ['RSI', 'volatility', 'hl_ratio', 'oc_ratio', 
                        'price_momentum_1', 'price_momentum_5']
            tech_cols += [f'MA_{p}_ratio' for p in self.config.ma_periods]
            feature_cols.extend(tech_cols)
        
        return {
            'feature_columns': feature_cols,
            'n_features': len(feature_cols),
            'target_feature': self.config.target_feature,
            'sequence_length': self.config.sequence_length,
            'prediction_length': self.config.prediction_length,
            'patch_size': self.config.patch_size,
            'stride': self.config.stride
        }
    
    def save_preprocessor(self, path: str):
        """Save fitted preprocessor"""
        torch.save({
            'scalers': self.preprocessor.scalers,
            'config': asdict(self.config),
            'fitted': self.preprocessor.fitted
        }, path)
    
    def load_preprocessor(self, path: str):
        """Load fitted preprocessor"""
        checkpoint = torch.load(path)
        self.preprocessor.scalers = checkpoint['scalers']
        self.preprocessor.fitted = checkpoint['fitted']
        self.config = DataLoaderConfig(**checkpoint['config'])


def main():
    """Example usage of TotoOHLCDataLoader"""
    print("🚀 Toto OHLC DataLoader Example")
    
    # Create configuration
    config = DataLoaderConfig(
        train_data_path="trainingdata/train",
        test_data_path="trainingdata/test",
        batch_size=16,
        sequence_length=96,
        prediction_length=24,
        patch_size=12,
        stride=6,
        validation_split=0.2,
        add_technical_indicators=True,
        normalization_method="robust",
        max_symbols=10  # Limit for testing
    )
    
    # Initialize dataloader
    dataloader = TotoOHLCDataLoader(config)
    
    try:
        # Prepare dataloaders
        dataloaders = dataloader.prepare_dataloaders()
        
        print(f"✅ Created dataloaders: {list(dataloaders.keys())}")
        
        # Print feature information
        feature_info = dataloader.get_feature_info()
        print(f"📊 Features: {feature_info['n_features']} columns")
        print(f"🎯 Target: {feature_info['target_feature']}")
        print(f"📏 Sequence length: {feature_info['sequence_length']}")
        
        # Test data loading
        if 'train' in dataloaders:
            train_loader = dataloaders['train']
            print(f"🔄 Training samples: {len(train_loader.dataset)}")
            
            # Test one batch
            for batch in train_loader:
                print(f"✅ Successfully loaded batch:")
                print(f"   - Series shape: {batch.series.shape}")
                print(f"   - Padding mask shape: {batch.padding_mask.shape}")
                print(f"   - ID mask shape: {batch.id_mask.shape}")
                print(f"   - Timestamps shape: {batch.timestamp_seconds.shape}")
                break
        
        # Test cross-validation
        if config.cv_folds > 1:
            cv_splits = dataloader.get_cross_validation_splits(2)  # Test with 2 folds
            print(f"🔀 Cross-validation: {len(cv_splits)} folds prepared")
        
        print("✅ DataLoader test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
