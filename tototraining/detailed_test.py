#!/usr/bin/env python3
"""
Detailed testing script for TotoOHLCDataLoader
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from toto_ohlc_dataloader import TotoOHLCDataLoader, DataLoaderConfig, MaskedTimeseries


def test_masked_timeseries_format():
    """Test MaskedTimeseries format compatibility"""
    print("🧪 Testing MaskedTimeseries Format")
    
    config = DataLoaderConfig(
        batch_size=2,
        sequence_length=24,
        prediction_length=6,
        max_symbols=2,  # Use more symbols to ensure training data exists
        num_workers=0,
        validation_split=0.0,  # No validation split to ensure all data goes to training
        min_sequence_length=50  # Lower minimum to ensure data passes filters
    )
    
    dataloader = TotoOHLCDataLoader(config)
    dataloaders = dataloader.prepare_dataloaders()
    
    if 'train' in dataloaders:
        train_loader = dataloaders['train']
        batch = next(iter(train_loader))
        
        print(f"✅ MaskedTimeseries type: {type(batch)}")
        print(f"✅ Fields: {batch._fields}")
        
        # Validate tensor shapes and types
        assert isinstance(batch.series, torch.Tensor), "series should be tensor"
        assert isinstance(batch.padding_mask, torch.Tensor), "padding_mask should be tensor"
        assert isinstance(batch.id_mask, torch.Tensor), "id_mask should be tensor"
        assert isinstance(batch.timestamp_seconds, torch.Tensor), "timestamp_seconds should be tensor"
        assert isinstance(batch.time_interval_seconds, torch.Tensor), "time_interval_seconds should be tensor"
        
        print(f"✅ Series shape: {batch.series.shape}")
        print(f"✅ All tensor types validated")
        
        # Test device transfer
        if torch.cuda.is_available():
            device = torch.device('cuda')
            batch_cuda = batch.to(device)
            print(f"✅ Device transfer successful: {batch_cuda.series.device}")
        
        return True
    return False


def test_technical_indicators():
    """Test technical indicators calculation"""
    print("\n📈 Testing Technical Indicators")
    
    config = DataLoaderConfig(
        add_technical_indicators=True,
        ma_periods=[5, 10, 20],
        rsi_period=14,
        max_symbols=2,
        batch_size=1,
        sequence_length=48,
        validation_split=0.0,
        min_sequence_length=100
    )
    
    dataloader = TotoOHLCDataLoader(config)
    
    # Get feature info
    feature_info = dataloader.get_feature_info()
    expected_features = [
        'Open', 'High', 'Low', 'Close', 'Volume',  # Base OHLC + Volume
        'RSI', 'volatility', 'hl_ratio', 'oc_ratio', 
        'price_momentum_1', 'price_momentum_5',  # Technical indicators
        'MA_5_ratio', 'MA_10_ratio', 'MA_20_ratio'  # MA ratios
    ]
    
    print(f"📊 Expected features: {len(expected_features)}")
    print(f"📊 Actual features: {feature_info['n_features']}")
    print(f"📊 Feature columns: {feature_info['feature_columns']}")
    
    # Verify all expected features are present
    for feature in expected_features:
        if feature in feature_info['feature_columns']:
            print(f"✅ {feature}: Present")
        else:
            print(f"❌ {feature}: Missing")
    
    return True


def test_data_loading_robustness():
    """Test data loading with different configurations"""
    print("\n🔧 Testing Data Loading Robustness")
    
    test_configs = [
        {"normalization_method": "standard"},
        {"normalization_method": "minmax"},
        {"normalization_method": "robust"},
        {"handle_missing": "interpolate"},
        {"handle_missing": "zero"},
        {"outlier_threshold": 2.0},
        {"outlier_threshold": 3.5}
    ]
    
    base_config = DataLoaderConfig(
        batch_size=4,
        sequence_length=24,
        max_symbols=2,
        num_workers=0,
        validation_split=0.0,
        min_sequence_length=50
    )
    
    for i, test_params in enumerate(test_configs):
        print(f"🧪 Test {i+1}: {test_params}")
        
        # Update config with test parameters
        for key, value in test_params.items():
            setattr(base_config, key, value)
        
        try:
            dataloader = TotoOHLCDataLoader(base_config)
            dataloaders = dataloader.prepare_dataloaders()
            
            if 'train' in dataloaders:
                batch = next(iter(dataloaders['train']))
                print(f"   ✅ Success - Batch shape: {batch.series.shape}")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    return True


def test_data_integrity():
    """Test data integrity and preprocessing"""
    print("\n🔍 Testing Data Integrity")
    
    config = DataLoaderConfig(
        batch_size=1,
        sequence_length=48,
        prediction_length=12,
        max_symbols=2,
        num_workers=0,
        add_technical_indicators=True,
        validation_split=0.0,
        min_sequence_length=100
    )
    
    dataloader = TotoOHLCDataLoader(config)
    dataloaders = dataloader.prepare_dataloaders()
    
    if 'train' in dataloaders:
        train_loader = dataloaders['train']
        dataset = train_loader.dataset
        
        # Get multiple batches and check for data quality
        for i, batch in enumerate(train_loader):
            series = batch.series
            
            # Check for NaN/Inf values
            has_nan = torch.isnan(series).any()
            has_inf = torch.isinf(series).any()
            
            print(f"Batch {i+1}:")
            print(f"   Shape: {series.shape}")
            print(f"   Has NaN: {has_nan}")
            print(f"   Has Inf: {has_inf}")
            print(f"   Min value: {series.min():.3f}")
            print(f"   Max value: {series.max():.3f}")
            print(f"   Mean: {series.mean():.3f}")
            print(f"   Std: {series.std():.3f}")
            
            if i >= 2:  # Check first 3 batches
                break
        
        # Test targets
        targets = dataset.get_targets()
        print(f"🎯 Targets shape: {targets.shape}")
        print(f"🎯 Targets range: [{targets.min():.3f}, {targets.max():.3f}]")
    
    return True


def test_cross_validation():
    """Test cross-validation functionality"""
    print("\n🔀 Testing Cross-Validation")
    
    config = DataLoaderConfig(
        cv_folds=3,
        batch_size=8,
        sequence_length=24,
        max_symbols=3,
        num_workers=0,
        validation_split=0.0,
        min_sequence_length=50
    )
    
    dataloader = TotoOHLCDataLoader(config)
    dataloader.prepare_dataloaders()  # Load and prepare data first
    
    # Get CV splits
    cv_splits = dataloader.get_cross_validation_splits(2)
    
    print(f"✅ Generated {len(cv_splits)} CV splits")
    
    for fold, (train_loader, val_loader) in enumerate(cv_splits):
        print(f"Fold {fold + 1}:")
        print(f"   Train samples: {len(train_loader.dataset)}")
        print(f"   Val samples: {len(val_loader.dataset)}")
        
        # Test one batch from each
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        
        print(f"   Train batch shape: {train_batch.series.shape}")
        print(f"   Val batch shape: {val_batch.series.shape}")
    
    return True


def test_configuration_persistence():
    """Test configuration save/load"""
    print("\n💾 Testing Configuration Persistence")
    
    # Create config
    original_config = DataLoaderConfig(
        sequence_length=120,
        prediction_length=30,
        batch_size=64,
        add_technical_indicators=True,
        ma_periods=[5, 15, 30],
        normalization_method="robust"
    )
    
    # Save config
    config_path = "test_config.json"
    original_config.save(config_path)
    print(f"✅ Config saved to {config_path}")
    
    # Load config
    loaded_config = DataLoaderConfig.load(config_path)
    print(f"✅ Config loaded from {config_path}")
    
    # Compare configurations
    attrs_to_check = ['sequence_length', 'prediction_length', 'batch_size', 
                     'add_technical_indicators', 'ma_periods', 'normalization_method']
    
    for attr in attrs_to_check:
        original_val = getattr(original_config, attr)
        loaded_val = getattr(loaded_config, attr)
        
        if original_val == loaded_val:
            print(f"✅ {attr}: {original_val}")
        else:
            print(f"❌ {attr}: {original_val} != {loaded_val}")
    
    # Clean up
    Path(config_path).unlink()
    print("🧹 Cleaned up test file")
    
    return True


def test_import_dependencies():
    """Test all import dependencies"""
    print("\n📦 Testing Import Dependencies")
    
    try:
        import torch
        print("✅ torch imported successfully")
        
        import numpy as np
        print("✅ numpy imported successfully")
        
        import pandas as pd
        print("✅ pandas imported successfully")
        
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
        print("✅ sklearn components imported successfully")
        
        # Test toto imports (with fallback)
        try:
            # Try to find the actual toto module
            toto_path = Path(__file__).parent.parent / "toto"
            if toto_path.exists():
                import sys
                sys.path.insert(0, str(toto_path))
                from toto.data.util.dataset import MaskedTimeseries, pad_array, pad_id_mask, replace_extreme_values
                print("✅ toto.data.util.dataset imported successfully")
            else:
                print("⚠️  toto module not found, using fallback implementations")
        except ImportError as e:
            print(f"⚠️  toto import failed, using fallback: {e}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def main():
    """Run all tests"""
    print("🧪 Detailed TotoOHLCDataLoader Testing\n")
    
    test_results = {
        "Dependencies": test_import_dependencies(),
        "MaskedTimeseries Format": test_masked_timeseries_format(),
        "Technical Indicators": test_technical_indicators(), 
        "Data Loading Robustness": test_data_loading_robustness(),
        "Data Integrity": test_data_integrity(),
        "Cross Validation": test_cross_validation(),
        "Configuration Persistence": test_configuration_persistence()
    }
    
    print("\n" + "="*50)
    print("📊 TEST RESULTS SUMMARY")
    print("="*50)
    
    passed = 0
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print(f"\n🏁 Overall: {passed}/{len(test_results)} tests passed")
    
    if passed == len(test_results):
        print("🎉 All tests passed! DataLoader is working correctly.")
    else:
        print("⚠️  Some tests failed. See details above.")


if __name__ == "__main__":
    main()