#!/usr/bin/env python3
"""Unit tests for hftraining data utilities."""

import pytest
import numpy as np
import pandas as pd
import torch
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from pathlib import Path

# Add hftraining to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../hftraining'))

from hftraining.data_utils import (
    StockDataProcessor,
    download_stock_data,
    create_sequences,
    split_data,
    augment_data,
    load_training_data,
    generate_synthetic_data,
    DataCollator
)


class TestStockDataProcessor:
    """Test StockDataProcessor functionality."""
    
    def test_init_default(self):
        """Test default initialization."""
        processor = StockDataProcessor()
        assert processor.sequence_length == 60
        assert processor.prediction_horizon == 5
        assert 'close' in processor.features
        assert len(processor.scalers) == 0
        assert len(processor.feature_names) == 0
    
    def test_init_custom(self):
        """Test custom initialization."""
        features = ['open', 'high', 'low', 'close']
        processor = StockDataProcessor(
            sequence_length=30,
            prediction_horizon=10,
            features=features
        )
        assert processor.sequence_length == 30
        assert processor.prediction_horizon == 10
        assert processor.features == features
    
    def test_add_technical_indicators(self):
        """Test technical indicator calculation."""
        processor = StockDataProcessor()
        
        # Create sample data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'open': np.random.uniform(95, 105, 100),
            'high': np.random.uniform(100, 110, 100),
            'low': np.random.uniform(90, 100, 100),
            'close': np.random.uniform(95, 105, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        })
        
        # Make prices somewhat realistic (trending)
        df['close'] = 100 + np.cumsum(np.random.normal(0, 0.5, 100))
        
        result = processor.add_technical_indicators(df)
        
        # Check that indicators were added
        expected_indicators = [
            'ma_5', 'ma_10', 'ma_20', 'ma_50',
            'ema_5', 'ema_10', 'ema_20', 'ema_50',
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_lower', 'bb_width', 'bb_position',
            'price_change', 'price_change_2', 'price_change_5',
            'high_low_ratio', 'close_open_ratio',
            'volume_ma', 'volume_ratio',
            'volatility', 'volatility_ratio',
            'resistance', 'support', 'resistance_distance', 'support_distance'
        ]
        
        for indicator in expected_indicators:
            assert indicator in result.columns, f"Missing indicator: {indicator}"
        
        # Check RSI is bounded
        rsi_values = result['rsi'].dropna()
        assert all(rsi_values >= 0) and all(rsi_values <= 100)
        
        # Check ratios are positive
        assert all(result['high_low_ratio'].dropna() >= 1.0)
    
    def test_prepare_features(self):
        """Test feature preparation."""
        processor = StockDataProcessor()
        
        # Create sample data
        df = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [105, 106, 107, 108, 109],
            'low': [95, 96, 97, 98, 99],
            'close': [102, 103, 104, 105, 106],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        features = processor.prepare_features(df)
        
        # Check output shape
        assert features.shape[0] == 5  # Same number of rows
        assert features.shape[1] > 5   # More features than input
        assert len(processor.feature_names) == features.shape[1]
        
        # Check no NaN values in output
        assert not np.any(np.isnan(features))
    
    def test_fit_and_transform_scalers(self):
        """Test scaler fitting and transformation."""
        processor = StockDataProcessor()
        
        # Create sample data
        data = np.random.randn(100, 10)
        
        # Fit scalers
        processor.fit_scalers(data)
        
        # Check scalers were created
        assert 'standard' in processor.scalers
        assert 'minmax' in processor.scalers
        
        # Transform data
        transformed = processor.transform(data)
        
        # Check transformation properties
        assert transformed.shape == data.shape
        assert abs(np.mean(transformed)) < 0.1  # Close to zero mean
        assert abs(np.std(transformed) - 1.0) < 0.1  # Close to unit std
    
    def test_save_and_load_scalers(self):
        """Test saving and loading scalers."""
        processor = StockDataProcessor()
        
        # Fit scalers on sample data
        data = np.random.randn(50, 5)
        processor.fit_scalers(data)
        processor.feature_names = ['f1', 'f2', 'f3', 'f4', 'f5']
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            try:
                # Save scalers
                processor.save_scalers(tmp.name)
                
                # Create new processor and load
                new_processor = StockDataProcessor()
                new_processor.load_scalers(tmp.name)
                
                # Check loaded attributes
                assert new_processor.feature_names == processor.feature_names
                assert new_processor.sequence_length == processor.sequence_length
                assert 'standard' in new_processor.scalers
                
                # Check transformation consistency
                transformed1 = processor.transform(data)
                transformed2 = new_processor.transform(data)
                np.testing.assert_array_almost_equal(transformed1, transformed2)
                
            finally:
                os.unlink(tmp.name)


class TestDataFunctions:
    """Test standalone data functions."""
    
    @patch('hftraining.data_utils.yf.Ticker')
    def test_download_stock_data(self, mock_ticker):
        """Test stock data downloading."""
        # Mock yfinance response
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [102, 103, 104],
            'Volume': [1000, 1100, 1200]
        })
        mock_data.index = pd.date_range('2020-01-01', periods=3)
        
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_data
        mock_ticker.return_value = mock_ticker_instance
        
        # Test single symbol
        result = download_stock_data('AAPL')
        assert 'AAPL' in result
        assert 'close' in result['AAPL'].columns
        
        # Test multiple symbols
        result = download_stock_data(['AAPL', 'GOOGL'])
        assert 'AAPL' in result
        assert 'GOOGL' in result
    
    def test_create_sequences(self):
        """Test sequence creation."""
        # Create sample data
        data = np.random.randn(100, 5)
        sequence_length = 20
        prediction_horizon = 5
        
        sequences, targets, actions = create_sequences(
            data, sequence_length, prediction_horizon
        )
        
        # Check shapes
        expected_num_sequences = 100 - sequence_length - prediction_horizon + 1
        assert sequences.shape == (expected_num_sequences, sequence_length, 5)
        assert targets.shape == (expected_num_sequences, prediction_horizon, 5)
        assert actions.shape == (expected_num_sequences,)
        
        # Check action labels are valid (0, 1, 2)
        assert all(action in [0, 1, 2] for action in actions)
    
    def test_create_sequences_insufficient_data(self):
        """Test sequence creation with insufficient data."""
        data = np.random.randn(10, 5)  # Too short
        
        with pytest.raises(ValueError, match="Data too short"):
            create_sequences(data, sequence_length=20, prediction_horizon=5)
    
    def test_split_data(self):
        """Test data splitting."""
        data = np.random.randn(1000, 10)
        
        train, val, test = split_data(data, 0.7, 0.2, 0.1)
        
        # Check sizes
        assert len(train) == 700
        assert len(val) == 200
        assert len(test) == 100
        
        # Check no overlap
        assert len(train) + len(val) + len(test) == len(data)
    
    def test_split_data_invalid_ratios(self):
        """Test data splitting with invalid ratios."""
        data = np.random.randn(100, 5)
        
        with pytest.raises(AssertionError, match="Ratios must sum to 1"):
            split_data(data, 0.8, 0.3, 0.2)  # Sums to 1.3
    
    def test_augment_data(self):
        """Test data augmentation."""
        original_data = np.ones((100, 10))  # All ones for easy testing
        
        augmented = augment_data(original_data, noise_factor=0.1, scaling_factor=0.05)
        
        # Check shape preserved
        assert augmented.shape == original_data.shape
        
        # Check data was modified
        assert not np.array_equal(original_data, augmented)
        
        # Check augmentation is reasonable (not too different)
        diff = np.abs(augmented - original_data)
        assert np.mean(diff) < 0.5  # Should be close to original
    
    def test_generate_synthetic_data(self):
        """Test synthetic data generation."""
        length = 1000
        n_features = 25
        
        data = generate_synthetic_data(length, n_features)
        
        # Check shape
        assert data.shape == (length, n_features)
        
        # Check no NaN or infinite values
        assert np.all(np.isfinite(data))
        
        # Check prices are positive (first 5 features are OHLCV)
        assert np.all(data[:, :5] > 0)
        
        # Check volume is positive
        assert np.all(data[:, 4] > 0)
    
    def test_load_training_data_synthetic_fallback(self):
        """Test loading training data falls back to synthetic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test with non-existent directory
            data = load_training_data(data_dir=tmpdir, symbols=None)
            
            # Should return synthetic data
            assert isinstance(data, np.ndarray)
            assert data.shape[0] > 0
            assert data.shape[1] > 0


class TestDataCollator:
    """Test DataCollator functionality."""
    
    def test_collate_batch(self):
        """Test batch collation."""
        collator = DataCollator()
        
        # Create mock examples with different sequence lengths
        examples = [
            {
                'input_ids': torch.randn(30, 10),
                'labels': torch.randn(5, 10),
                'action_labels': torch.tensor(1)
            },
            {
                'input_ids': torch.randn(25, 10),
                'labels': torch.randn(5, 10),
                'action_labels': torch.tensor(0)
            },
            {
                'input_ids': torch.randn(35, 10),
                'labels': torch.randn(5, 10),
                'action_labels': torch.tensor(2)
            }
        ]
        
        batch = collator(examples)
        
        # Check output structure
        assert 'input_ids' in batch
        assert 'attention_mask' in batch
        assert 'labels' in batch
        assert 'action_labels' in batch
        
        # Check shapes - should be padded to max length (35)
        assert batch['input_ids'].shape == (3, 35, 10)
        assert batch['attention_mask'].shape == (3, 35)
        assert batch['labels'].shape == (3, 5, 10)
        assert batch['action_labels'].shape == (3,)
        
        # Check attention masks are correct
        assert torch.sum(batch['attention_mask'][0]) == 30  # First example length
        assert torch.sum(batch['attention_mask'][1]) == 25  # Second example length
        assert torch.sum(batch['attention_mask'][2]) == 35  # Third example length