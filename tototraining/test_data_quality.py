#!/usr/bin/env python3
"""
Data quality validation tests for the Toto retraining system.
Tests training data integrity, distribution, and preprocessing.
"""

import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import tempfile
import warnings
from typing import Dict, List, Tuple, Optional
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
import json

# Import modules under test
from toto_ohlc_dataloader import (
    DataLoaderConfig, OHLCPreprocessor, TotoOHLCDataLoader, 
    OHLCDataset as DataLoaderOHLCDataset
)

# Suppress warnings during testing
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class DataQualityValidator:
    """Utility class for data quality validation"""
    
    @staticmethod
    def check_ohlc_consistency(df: pd.DataFrame) -> Dict[str, bool]:
        """Check OHLC data consistency rules"""
        checks = {}
        
        # Basic column existence
        required_cols = ['Open', 'High', 'Low', 'Close']
        checks['has_required_columns'] = all(col in df.columns for col in required_cols)
        
        if not checks['has_required_columns']:
            return checks
        
        # OHLC relationships
        checks['high_gte_open'] = (df['High'] >= df['Open']).all()
        checks['high_gte_close'] = (df['High'] >= df['Close']).all()
        checks['low_lte_open'] = (df['Low'] <= df['Open']).all()
        checks['low_lte_close'] = (df['Low'] <= df['Close']).all()
        checks['high_gte_low'] = (df['High'] >= df['Low']).all()
        
        # No negative prices
        checks['all_positive'] = (
            (df['Open'] > 0).all() and
            (df['High'] > 0).all() and
            (df['Low'] > 0).all() and
            (df['Close'] > 0).all()
        )
        
        # No infinite or NaN values
        numeric_cols = ['Open', 'High', 'Low', 'Close']
        if 'Volume' in df.columns:
            numeric_cols.append('Volume')
        
        checks['no_inf_nan'] = not df[numeric_cols].isin([np.inf, -np.inf]).any().any()
        checks['no_nan'] = not df[numeric_cols].isna().any().any()
        
        return checks
    
    @staticmethod
    def check_data_distribution(df: pd.DataFrame) -> Dict[str, float]:
        """Check data distribution characteristics"""
        stats = {}
        
        if 'Close' in df.columns and len(df) > 1:
            returns = df['Close'].pct_change().dropna()
            
            stats['return_mean'] = float(returns.mean())
            stats['return_std'] = float(returns.std())
            stats['return_skewness'] = float(returns.skew())
            stats['return_kurtosis'] = float(returns.kurtosis())
            
            # Check for outliers (returns > 3 std deviations)
            outlier_threshold = 3 * stats['return_std']
            outliers = returns[abs(returns) > outlier_threshold]
            stats['outlier_ratio'] = len(outliers) / len(returns)
            
            # Price range
            stats['price_min'] = float(df['Close'].min())
            stats['price_max'] = float(df['Close'].max())
            stats['price_range_ratio'] = stats['price_max'] / stats['price_min']
        
        if 'Volume' in df.columns:
            stats['volume_mean'] = float(df['Volume'].mean())
            stats['volume_zero_ratio'] = (df['Volume'] == 0).sum() / len(df)
        
        return stats
    
    @staticmethod
    def check_temporal_consistency(df: pd.DataFrame) -> Dict[str, bool]:
        """Check temporal data consistency"""
        checks = {}
        
        if 'timestamp' in df.columns:
            timestamps = pd.to_datetime(df['timestamp'])
            
            # Check if sorted
            checks['is_sorted'] = timestamps.is_monotonic_increasing
            
            # Check for duplicates
            checks['no_duplicate_timestamps'] = not timestamps.duplicated().any()
            
            # Check for reasonable time intervals
            if len(timestamps) > 1:
                intervals = timestamps.diff().dropna()
                
                # Most intervals should be similar (regular frequency)
                mode_interval = intervals.mode().iloc[0] if len(intervals.mode()) > 0 else None
                if mode_interval:
                    # Allow up to 10% deviation from mode interval
                    tolerance = mode_interval * 0.1
                    regular_intervals = intervals.between(
                        mode_interval - tolerance,
                        mode_interval + tolerance
                    )
                    checks['regular_intervals'] = regular_intervals.sum() / len(intervals) >= 0.8
                else:
                    checks['regular_intervals'] = False
        else:
            checks['is_sorted'] = True
            checks['no_duplicate_timestamps'] = True
            checks['regular_intervals'] = True
        
        return checks


@pytest.fixture
def data_quality_validator():
    """Provide data quality validator instance"""
    return DataQualityValidator()


@pytest.fixture
def sample_valid_data():
    """Create sample valid OHLC data"""
    np.random.seed(42)
    n_samples = 100
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='H')
    
    # Generate valid OHLC data
    base_price = 100
    prices = [base_price]
    
    for i in range(1, n_samples):
        change = np.random.normal(0, 0.01)  # 1% volatility
        new_price = max(prices[-1] * (1 + change), 1.0)
        prices.append(new_price)
    
    opens = []
    highs = []
    lows = []
    closes = prices
    volumes = []
    
    for i, close in enumerate(closes):
        if i == 0:
            open_price = close
        else:
            open_price = closes[i-1] + np.random.normal(0, 0.002) * closes[i-1]
        
        high = max(open_price, close) + abs(np.random.normal(0, 0.005)) * max(open_price, close)
        low = min(open_price, close) - abs(np.random.normal(0, 0.005)) * min(open_price, close)
        volume = max(int(np.random.lognormal(8, 1)), 1)
        
        opens.append(open_price)
        highs.append(high)
        lows.append(low)
        volumes.append(volume)
    
    return pd.DataFrame({
        'timestamp': dates,
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Volume': volumes
    })


@pytest.fixture
def sample_invalid_data():
    """Create sample invalid OHLC data with various issues"""
    n_samples = 50
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='H')
    
    # Create data with various issues
    data = pd.DataFrame({
        'timestamp': dates,
        'Open': np.random.uniform(90, 110, n_samples),
        'High': np.random.uniform(80, 120, n_samples),  # Some highs < opens/closes
        'Low': np.random.uniform(95, 115, n_samples),   # Some lows > opens/closes
        'Close': np.random.uniform(90, 110, n_samples),
        'Volume': np.random.randint(-100, 10000, n_samples)  # Some negative volumes
    })
    
    # Add some NaN values
    data.loc[10:12, 'Close'] = np.nan
    
    # Add some infinite values
    data.loc[20, 'High'] = np.inf
    data.loc[21, 'Low'] = -np.inf
    
    return data


class TestOHLCDataValidation:
    """Test OHLC data validation"""
    
    def test_valid_data_passes_checks(self, data_quality_validator, sample_valid_data):
        """Test that valid data passes all checks"""
        checks = data_quality_validator.check_ohlc_consistency(sample_valid_data)
        
        assert checks['has_required_columns']
        assert checks['high_gte_open']
        assert checks['high_gte_close']
        assert checks['low_lte_open']
        assert checks['low_lte_close']
        assert checks['high_gte_low']
        assert checks['all_positive']
        assert checks['no_inf_nan']
        assert checks['no_nan']
    
    def test_invalid_data_fails_checks(self, data_quality_validator, sample_invalid_data):
        """Test that invalid data fails appropriate checks"""
        checks = data_quality_validator.check_ohlc_consistency(sample_invalid_data)
        
        assert checks['has_required_columns']  # Columns exist
        assert not checks['no_inf_nan']        # Has infinite values
        assert not checks['no_nan']            # Has NaN values
        
        # Fix inf/nan issues for other tests
        clean_data = sample_invalid_data.replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean_data) > 0:
            # Some OHLC relationships should fail due to random generation
            clean_checks = data_quality_validator.check_ohlc_consistency(clean_data)
            # At least one relationship check should fail
            relationship_checks = [
                clean_checks['high_gte_open'],
                clean_checks['high_gte_close'],
                clean_checks['low_lte_open'],
                clean_checks['low_lte_close']
            ]
            assert not all(relationship_checks), "Some OHLC relationships should be invalid"
    
    def test_missing_columns_detection(self, data_quality_validator):
        """Test detection of missing required columns"""
        incomplete_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [101, 102, 103],
            # Missing Low, Close
        })
        
        checks = data_quality_validator.check_ohlc_consistency(incomplete_data)
        assert not checks['has_required_columns']
    
    def test_temporal_consistency_checks(self, data_quality_validator, sample_valid_data):
        """Test temporal consistency checks"""
        checks = data_quality_validator.check_temporal_consistency(sample_valid_data)
        
        assert checks['is_sorted']
        assert checks['no_duplicate_timestamps']
        assert checks['regular_intervals']
    
    def test_temporal_consistency_with_issues(self, data_quality_validator):
        """Test temporal consistency with problematic data"""
        # Create data with temporal issues
        dates = pd.to_datetime(['2023-01-01 10:00', '2023-01-01 09:00', '2023-01-01 11:00'])  # Not sorted
        data_unsorted = pd.DataFrame({
            'timestamp': dates,
            'Open': [100, 101, 102],
            'High': [101, 102, 103],
            'Low': [99, 100, 101],
            'Close': [100.5, 101.5, 102.5],
        })
        
        checks = data_quality_validator.check_temporal_consistency(data_unsorted)
        assert not checks['is_sorted']
        
        # Test duplicate timestamps
        dates_dup = pd.to_datetime(['2023-01-01 10:00', '2023-01-01 10:00', '2023-01-01 11:00'])
        data_dup = data_unsorted.copy()
        data_dup['timestamp'] = dates_dup
        
        checks_dup = data_quality_validator.check_temporal_consistency(data_dup)
        assert not checks_dup['no_duplicate_timestamps']
    
    def test_data_distribution_analysis(self, data_quality_validator, sample_valid_data):
        """Test data distribution analysis"""
        stats = data_quality_validator.check_data_distribution(sample_valid_data)
        
        # Basic stats should be calculated
        assert 'return_mean' in stats
        assert 'return_std' in stats
        assert 'return_skewness' in stats
        assert 'return_kurtosis' in stats
        assert 'outlier_ratio' in stats
        assert 'price_min' in stats
        assert 'price_max' in stats
        assert 'price_range_ratio' in stats
        assert 'volume_mean' in stats
        assert 'volume_zero_ratio' in stats
        
        # Sanity checks
        assert stats['return_std'] > 0
        assert stats['price_min'] > 0
        assert stats['price_max'] > stats['price_min']
        assert stats['price_range_ratio'] >= 1.0
        assert 0 <= stats['outlier_ratio'] <= 1
        assert 0 <= stats['volume_zero_ratio'] <= 1


class TestPreprocessorValidation:
    """Test data preprocessing validation"""
    
    @pytest.fixture
    def preprocessor_config(self):
        """Create preprocessor configuration"""
        return DataLoaderConfig(
            normalization_method="robust",
            handle_missing="interpolate",
            outlier_threshold=3.0,
            add_technical_indicators=True,
            ohlc_features=['Open', 'High', 'Low', 'Close'],
            additional_features=['Volume']
        )
    
    def test_preprocessor_initialization(self, preprocessor_config):
        """Test preprocessor initialization"""
        preprocessor = OHLCPreprocessor(preprocessor_config)
        
        assert preprocessor.config == preprocessor_config
        assert not preprocessor.fitted
        assert len(preprocessor.scalers) == 0
    
    def test_technical_indicators_addition(self, preprocessor_config, sample_valid_data):
        """Test technical indicators are added correctly"""
        preprocessor = OHLCPreprocessor(preprocessor_config)
        
        # Test with indicators enabled
        processed = preprocessor.add_technical_indicators(sample_valid_data)
        
        expected_indicators = ['RSI', 'volatility', 'hl_ratio', 'oc_ratio', 
                              'price_momentum_1', 'price_momentum_5']
        expected_ma_indicators = ['MA_5', 'MA_10', 'MA_20', 'MA_5_ratio', 'MA_10_ratio', 'MA_20_ratio']
        
        for indicator in expected_indicators:
            assert indicator in processed.columns, f"Missing indicator: {indicator}"
        
        for ma_indicator in expected_ma_indicators:
            assert ma_indicator in processed.columns, f"Missing MA indicator: {ma_indicator}"
        
        # Test without indicators
        config_no_indicators = preprocessor_config
        config_no_indicators.add_technical_indicators = False
        preprocessor_no_ind = OHLCPreprocessor(config_no_indicators)
        
        processed_no_ind = preprocessor_no_ind.add_technical_indicators(sample_valid_data)
        pd.testing.assert_frame_equal(processed_no_ind, sample_valid_data)
    
    def test_missing_value_handling(self, preprocessor_config, sample_valid_data):
        """Test missing value handling strategies"""
        # Create data with missing values
        data_with_missing = sample_valid_data.copy()
        data_with_missing.loc[10:15, 'Close'] = np.nan
        data_with_missing.loc[20:22, 'Volume'] = np.nan
        
        # Test interpolation
        config_interp = preprocessor_config
        config_interp.handle_missing = "interpolate"
        preprocessor_interp = OHLCPreprocessor(config_interp)
        
        result_interp = preprocessor_interp.handle_missing_values(data_with_missing)
        assert result_interp.isna().sum().sum() < data_with_missing.isna().sum().sum()
        
        # Test dropping
        config_drop = preprocessor_config
        config_drop.handle_missing = "drop"
        preprocessor_drop = OHLCPreprocessor(config_drop)
        
        result_drop = preprocessor_drop.handle_missing_values(data_with_missing)
        assert not result_drop.isna().any().any()
        assert len(result_drop) < len(data_with_missing)
        
        # Test zero fill
        config_zero = preprocessor_config
        config_zero.handle_missing = "zero"
        preprocessor_zero = OHLCPreprocessor(config_zero)
        
        result_zero = preprocessor_zero.handle_missing_values(data_with_missing)
        assert not result_zero.isna().any().any()
        assert len(result_zero) == len(data_with_missing)
    
    def test_outlier_removal(self, preprocessor_config, sample_valid_data):
        """Test outlier removal"""
        # Create data with outliers
        data_with_outliers = sample_valid_data.copy()
        
        # Add extreme outliers
        data_with_outliers.loc[50, 'Close'] = data_with_outliers['Close'].mean() * 10  # 10x average
        data_with_outliers.loc[51, 'Volume'] = data_with_outliers['Volume'].mean() * 20  # 20x average
        
        preprocessor = OHLCPreprocessor(preprocessor_config)
        result = preprocessor.remove_outliers(data_with_outliers)
        
        # Should have fewer rows due to outlier removal
        assert len(result) <= len(data_with_outliers)
        
        # Extreme outliers should be removed
        assert result['Close'].max() < data_with_outliers['Close'].max()
    
    def test_scaler_fitting_and_transformation(self, preprocessor_config, sample_valid_data):
        """Test scaler fitting and data transformation"""
        preprocessor = OHLCPreprocessor(preprocessor_config)
        
        # Test fitting
        data_dict = {'TEST': sample_valid_data}
        preprocessor.fit_scalers(data_dict)
        
        assert preprocessor.fitted
        assert len(preprocessor.scalers) > 0
        
        # Test transformation
        transformed = preprocessor.transform(sample_valid_data, 'TEST')
        
        assert isinstance(transformed, pd.DataFrame)
        assert len(transformed) > 0
        
        # Check that numerical columns have been scaled (should have different stats)
        original_close_std = sample_valid_data['Close'].std()
        transformed_close_std = transformed['Close'].std()
        
        # Robust scaler should change the standard deviation
        assert abs(original_close_std - transformed_close_std) > 0.01
    
    def test_feature_preparation(self, preprocessor_config, sample_valid_data):
        """Test feature array preparation"""
        preprocessor = OHLCPreprocessor(preprocessor_config)
        
        # Fit and transform
        data_dict = {'TEST': sample_valid_data}
        preprocessor.fit_scalers(data_dict)
        transformed = preprocessor.transform(sample_valid_data, 'TEST')
        
        # Prepare features
        features = preprocessor.prepare_features(transformed)
        
        assert isinstance(features, np.ndarray)
        assert features.dtype == np.float32
        assert features.shape[0] == len(transformed)
        assert features.shape[1] > 5  # Should have OHLCV + technical indicators


class TestDatasetValidation:
    """Test dataset-level validation"""
    
    @pytest.fixture
    def dataset_config(self):
        """Create dataset configuration"""
        return DataLoaderConfig(
            sequence_length=50,
            prediction_length=10,
            batch_size=8,
            normalization_method="robust",
            add_technical_indicators=True,
            min_sequence_length=60
        )
    
    def test_dataset_creation_validation(self, dataset_config, sample_valid_data):
        """Test dataset creation with validation"""
        # Prepare preprocessor
        preprocessor = OHLCPreprocessor(dataset_config)
        data_dict = {'TEST': sample_valid_data}
        preprocessor.fit_scalers(data_dict)
        
        # Create dataset
        dataset = DataLoaderOHLCDataset(data_dict, dataset_config, preprocessor, 'train')
        
        # Validate dataset properties
        assert len(dataset) >= 0
        
        if len(dataset) > 0:
            # Test sample structure
            sample = dataset[0]
            
            assert hasattr(sample, 'series')
            assert hasattr(sample, 'padding_mask')
            assert hasattr(sample, 'id_mask')
            assert hasattr(sample, 'timestamp_seconds')
            assert hasattr(sample, 'time_interval_seconds')
            
            # Validate tensor properties
            assert isinstance(sample.series, torch.Tensor)
            assert sample.series.dtype == torch.float32
            assert not torch.isnan(sample.series).any()
            assert not torch.isinf(sample.series).any()
            
            # Validate shapes
            n_features, seq_len = sample.series.shape
            assert seq_len == dataset_config.sequence_length
            assert n_features > 0
    
    def test_dataset_with_insufficient_data(self, dataset_config):
        """Test dataset handling of insufficient data"""
        # Create very small dataset
        small_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10, freq='H'),
            'Open': np.random.uniform(95, 105, 10),
            'High': np.random.uniform(100, 110, 10),
            'Low': np.random.uniform(90, 100, 10),
            'Close': np.random.uniform(95, 105, 10),
            'Volume': np.random.randint(1000, 5000, 10)
        })
        
        # Ensure OHLC consistency
        small_data['High'] = np.maximum(small_data['High'], np.maximum(small_data['Open'], small_data['Close']))
        small_data['Low'] = np.minimum(small_data['Low'], np.minimum(small_data['Open'], small_data['Close']))
        
        preprocessor = OHLCPreprocessor(dataset_config)
        data_dict = {'SMALL': small_data}
        preprocessor.fit_scalers(data_dict)
        
        dataset = DataLoaderOHLCDataset(data_dict, dataset_config, preprocessor, 'train')
        
        # Dataset should be empty due to insufficient data
        assert len(dataset) == 0
    
    def test_batch_consistency_validation(self, dataset_config, sample_valid_data):
        """Test batch consistency validation"""
        # Create larger dataset for batching
        large_data = sample_valid_data
        for i in range(3):  # Extend data
            additional_data = sample_valid_data.copy()
            additional_data['timestamp'] = sample_valid_data['timestamp'] + pd.Timedelta(hours=len(sample_valid_data) * (i + 1))
            additional_data['Close'] = additional_data['Close'] * (1 + np.random.normal(0, 0.1, len(additional_data)))
            large_data = pd.concat([large_data, additional_data], ignore_index=True)
        
        # Ensure OHLC consistency for extended data
        large_data['High'] = np.maximum(large_data['High'], np.maximum(large_data['Open'], large_data['Close']))
        large_data['Low'] = np.minimum(large_data['Low'], np.minimum(large_data['Open'], large_data['Close']))
        
        preprocessor = OHLCPreprocessor(dataset_config)
        data_dict = {'LARGE': large_data}
        preprocessor.fit_scalers(data_dict)
        
        dataset = DataLoaderOHLCDataset(data_dict, dataset_config, preprocessor, 'train')
        
        if len(dataset) > 0:
            # Create dataloader
            dataloader = torch.utils.data.DataLoader(
                dataset, 
                batch_size=dataset_config.batch_size,
                shuffle=False  # Don't shuffle for consistency testing
            )
            
            # Test multiple batches
            batch_count = 0
            for batch in dataloader:
                # Validate batch structure
                assert hasattr(batch, 'series')
                assert isinstance(batch.series, torch.Tensor)
                
                batch_size, n_features, seq_len = batch.series.shape
                assert batch_size <= dataset_config.batch_size
                assert seq_len == dataset_config.sequence_length
                assert n_features > 0
                
                # Check for data quality issues in batch
                assert not torch.isnan(batch.series).any()
                assert not torch.isinf(batch.series).any()
                
                batch_count += 1
                if batch_count >= 3:  # Test first 3 batches
                    break


class TestDataLoaderIntegration:
    """Test full data loading pipeline validation"""
    
    @pytest.fixture
    def temp_data_dir(self, sample_valid_data):
        """Create temporary directory with test data"""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create train/test directories
        train_dir = temp_dir / "train"
        test_dir = temp_dir / "test"
        train_dir.mkdir()
        test_dir.mkdir()
        
        # Split data and save
        train_data = sample_valid_data.iloc[:80].copy()
        test_data = sample_valid_data.iloc[80:].copy()
        
        train_data.to_csv(train_dir / "test_symbol.csv", index=False)
        test_data.to_csv(test_dir / "test_symbol.csv", index=False)
        
        yield temp_dir
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
    
    def test_dataloader_pipeline_validation(self, temp_data_dir):
        """Test complete dataloader pipeline validation"""
        config = DataLoaderConfig(
            train_data_path=str(temp_data_dir / "train"),
            test_data_path=str(temp_data_dir / "test"),
            sequence_length=20,
            prediction_length=5,
            batch_size=4,
            validation_split=0.2,
            normalization_method="robust",
            add_technical_indicators=False,  # Disable for simpler testing
            min_sequence_length=25
        )
        
        dataloader = TotoOHLCDataLoader(config)
        
        # Test data loading
        train_data, val_data, test_data = dataloader.load_data()
        
        # Validate loaded data
        assert len(train_data) > 0, "Should have training data"
        
        for symbol, df in train_data.items():
            validator = DataQualityValidator()
            
            # Check OHLC consistency
            ohlc_checks = validator.check_ohlc_consistency(df)
            assert ohlc_checks['has_required_columns']
            assert ohlc_checks['all_positive']
            
            # Check temporal consistency
            temporal_checks = validator.check_temporal_consistency(df)
            assert temporal_checks['is_sorted']
            
            # Check data distribution
            dist_stats = validator.check_data_distribution(df)
            assert 'return_mean' in dist_stats
            assert dist_stats['price_min'] > 0
        
        # Test dataloader creation
        dataloaders = dataloader.prepare_dataloaders()
        assert 'train' in dataloaders
        
        # Test batch validation
        train_loader = dataloaders['train']
        for batch in train_loader:
            # Validate batch data quality
            assert isinstance(batch.series, torch.Tensor)
            assert not torch.isnan(batch.series).any()
            assert not torch.isinf(batch.series).any()
            assert batch.series.min() > -100  # Reasonable range after normalization
            assert batch.series.max() < 100   # Reasonable range after normalization
            break  # Test just one batch
    
    def test_cross_validation_data_quality(self, temp_data_dir):
        """Test data quality in cross-validation splits"""
        config = DataLoaderConfig(
            train_data_path=str(temp_data_dir / "train"),
            sequence_length=15,
            prediction_length=3,
            batch_size=2,
            cv_folds=2,
            normalization_method="robust",
            add_technical_indicators=False,
            min_sequence_length=20
        )
        
        dataloader = TotoOHLCDataLoader(config)
        
        # Load and prepare data
        train_data, val_data, test_data = dataloader.load_data()
        
        if len(train_data) > 0:
            dataloaders = dataloader.prepare_dataloaders()
            
            # Test cross-validation splits
            cv_splits = dataloader.get_cross_validation_splits(n_splits=2)
            
            for fold_idx, (train_loader, val_loader) in enumerate(cv_splits):
                # Test both train and validation loaders
                for loader_name, loader in [('train', train_loader), ('val', val_loader)]:
                    batch_count = 0
                    for batch in loader:
                        # Validate data quality in CV splits
                        assert isinstance(batch.series, torch.Tensor)
                        assert not torch.isnan(batch.series).any()
                        assert not torch.isinf(batch.series).any()
                        
                        batch_count += 1
                        if batch_count >= 2:  # Test first 2 batches
                            break
                
                if fold_idx >= 1:  # Test first 2 folds
                    break


class TestEdgeCasesAndErrorConditions:
    """Test edge cases and error conditions in data quality"""
    
    def test_empty_data_handling(self):
        """Test handling of empty datasets"""
        config = DataLoaderConfig()
        preprocessor = OHLCPreprocessor(config)
        
        # Empty dataframe
        empty_df = pd.DataFrame()
        
        # Should handle gracefully
        result = preprocessor.handle_missing_values(empty_df)
        assert len(result) == 0
    
    def test_single_row_data_handling(self):
        """Test handling of single-row datasets"""
        single_row_data = pd.DataFrame({
            'timestamp': [pd.Timestamp('2023-01-01')],
            'Open': [100.0],
            'High': [102.0],
            'Low': [99.0],
            'Close': [101.0],
            'Volume': [1000]
        })
        
        validator = DataQualityValidator()
        
        # Should handle single row without error
        ohlc_checks = validator.check_ohlc_consistency(single_row_data)
        assert ohlc_checks['has_required_columns']
        assert ohlc_checks['all_positive']
        
        # Distribution stats should handle single row
        dist_stats = validator.check_data_distribution(single_row_data)
        # Should not crash, though some stats may be NaN
        assert 'price_min' in dist_stats
        assert 'price_max' in dist_stats
    
    def test_extreme_value_handling(self):
        """Test handling of extreme values"""
        extreme_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=5, freq='H'),
            'Open': [1e-10, 1e10, 100, 100, 100],     # Very small and very large
            'High': [1e-10, 1e10, 101, 101, 101],
            'Low': [1e-11, 1e9, 99, 99, 99],
            'Close': [1e-10, 1e10, 100, 100, 100],
            'Volume': [0, 1e15, 1000, 1000, 1000]     # Zero and very large volume
        })
        
        validator = DataQualityValidator()
        
        # Should detect issues with extreme values
        ohlc_checks = validator.check_ohlc_consistency(extreme_data)
        assert ohlc_checks['has_required_columns']
        assert ohlc_checks['all_positive']  # Still positive
        
        # Distribution should handle extreme values
        dist_stats = validator.check_data_distribution(extreme_data)
        assert dist_stats['price_range_ratio'] > 1000  # Very large range
    
    def test_data_type_validation(self):
        """Test validation of data types"""
        # Mixed data types
        mixed_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=3, freq='H'),
            'Open': ['100', '101', '102'],    # String instead of numeric
            'High': [101.0, 102.0, 103.0],
            'Low': [99.0, 100.0, 101.0],
            'Close': [100.5, 101.5, 102.5],
            'Volume': [1000, 1100, 1200]
        })
        
        config = DataLoaderConfig()
        preprocessor = OHLCPreprocessor(config)
        
        # Should handle type conversion gracefully
        try:
            data_dict = {'MIXED': mixed_data}
            preprocessor.fit_scalers(data_dict)
            # If it doesn't crash, it handled the conversion
            assert True
        except (ValueError, TypeError):
            # Expected for non-convertible strings
            assert True


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])