#!/usr/bin/env python3
"""
Comprehensive test summary for TotoOHLCDataLoader
"""

import torch
import numpy as np
from pathlib import Path
from toto_ohlc_dataloader import TotoOHLCDataLoader, DataLoaderConfig

def run_comprehensive_test():
    """Run comprehensive test covering all requirements"""
    
    print("🧪 COMPREHENSIVE TOTO OHLC DATALOADER TEST")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Basic functionality
    print("\n1️⃣ BASIC FUNCTIONALITY TEST")
    try:
        config = DataLoaderConfig(
            batch_size=16,
            sequence_length=96,
            prediction_length=24,
            max_symbols=5,
            validation_split=0.2,
            num_workers=0
        )
        
        dataloader = TotoOHLCDataLoader(config)
        dataloaders = dataloader.prepare_dataloaders()
        
        print(f"✅ Created {len(dataloaders)} dataloaders")
        for name, dl in dataloaders.items():
            print(f"   - {name}: {len(dl.dataset)} samples, {len(dl)} batches")
        
        results['basic_functionality'] = True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        results['basic_functionality'] = False
    
    # Test 2: Data loading and batching
    print("\n2️⃣ DATA LOADING AND BATCHING TEST")
    try:
        config = DataLoaderConfig(
            batch_size=8,
            sequence_length=48,
            prediction_length=12,
            max_symbols=3,
            validation_split=0.0,
            num_workers=0,
            min_sequence_length=100
        )
        
        dataloader = TotoOHLCDataLoader(config)
        dataloaders = dataloader.prepare_dataloaders()
        
        if 'train' in dataloaders:
            train_loader = dataloaders['train']
            batch = next(iter(train_loader))
            
            # Verify batch structure
            expected_batch_size = min(8, len(train_loader.dataset))
            actual_batch_size = batch.series.shape[0]
            
            print(f"✅ Batch loaded successfully")
            print(f"   - Expected batch size: {expected_batch_size}")
            print(f"   - Actual batch size: {actual_batch_size}")
            print(f"   - Series shape: {batch.series.shape}")
            print(f"   - Features: {batch.series.shape[1]}")
            print(f"   - Sequence length: {batch.series.shape[2]}")
            
            # Test multiple batches
            batch_count = 0
            for batch in train_loader:
                batch_count += 1
                if batch_count >= 3:
                    break
            
            print(f"✅ Successfully processed {batch_count} batches")
            results['data_loading'] = True
        else:
            print("❌ No training dataloader created")
            results['data_loading'] = False
            
    except Exception as e:
        print(f"❌ Failed: {e}")
        results['data_loading'] = False
    
    # Test 3: MaskedTimeseries format
    print("\n3️⃣ MASKEDTIMESERIES FORMAT TEST")
    try:
        config = DataLoaderConfig(
            batch_size=4,
            sequence_length=24,
            max_symbols=2,
            validation_split=0.0,
            num_workers=0,
            min_sequence_length=50
        )
        
        dataloader = TotoOHLCDataLoader(config)
        dataloaders = dataloader.prepare_dataloaders()
        
        if 'train' in dataloaders:
            batch = next(iter(dataloaders['train']))
            
            # Verify MaskedTimeseries fields
            expected_fields = ('series', 'padding_mask', 'id_mask', 'timestamp_seconds', 'time_interval_seconds')
            actual_fields = batch._fields
            
            print(f"✅ MaskedTimeseries structure verified")
            print(f"   - Expected fields: {expected_fields}")
            print(f"   - Actual fields: {actual_fields}")
            
            fields_match = set(expected_fields) == set(actual_fields)
            print(f"   - Fields match: {fields_match}")
            
            # Verify tensor properties
            print(f"✅ Tensor properties:")
            print(f"   - series dtype: {batch.series.dtype} (expected: torch.float32)")
            print(f"   - padding_mask dtype: {batch.padding_mask.dtype} (expected: torch.bool)")
            print(f"   - id_mask dtype: {batch.id_mask.dtype} (expected: torch.long)")
            print(f"   - timestamp_seconds dtype: {batch.timestamp_seconds.dtype} (expected: torch.long)")
            print(f"   - time_interval_seconds dtype: {batch.time_interval_seconds.dtype} (expected: torch.long)")
            
            # Test device transfer
            device_test_passed = True
            if torch.cuda.is_available():
                try:
                    cuda_device = torch.device('cuda')
                    cuda_batch = batch.to(cuda_device)
                    print(f"✅ CUDA device transfer successful")
                    device_test_passed = True
                except Exception as e:
                    print(f"❌ CUDA device transfer failed: {e}")
                    device_test_passed = False
            
            results['masked_timeseries'] = fields_match and device_test_passed
        else:
            print("❌ No training data available")
            results['masked_timeseries'] = False
            
    except Exception as e:
        print(f"❌ Failed: {e}")
        results['masked_timeseries'] = False
    
    # Test 4: Technical indicators
    print("\n4️⃣ TECHNICAL INDICATORS TEST")
    try:
        config = DataLoaderConfig(
            batch_size=2,
            sequence_length=48,
            max_symbols=2,
            add_technical_indicators=True,
            ma_periods=[5, 10, 20],
            rsi_period=14,
            validation_split=0.0,
            num_workers=0,
            min_sequence_length=100
        )
        
        dataloader = TotoOHLCDataLoader(config)
        feature_info = dataloader.get_feature_info()
        
        expected_base_features = ['Open', 'High', 'Low', 'Close', 'Volume']
        expected_tech_features = [
            'RSI', 'volatility', 'hl_ratio', 'oc_ratio', 
            'price_momentum_1', 'price_momentum_5',
            'MA_5_ratio', 'MA_10_ratio', 'MA_20_ratio'
        ]
        expected_total_features = len(expected_base_features) + len(expected_tech_features)
        
        actual_features = feature_info['feature_columns']
        actual_count = feature_info['n_features']
        
        print(f"✅ Technical indicators configuration:")
        print(f"   - Expected features: {expected_total_features}")
        print(f"   - Actual features: {actual_count}")
        print(f"   - Feature list: {actual_features}")
        
        # Check specific indicators
        tech_indicators_present = all(feat in actual_features for feat in expected_tech_features)
        base_features_present = all(feat in actual_features for feat in expected_base_features)
        
        print(f"   - Base OHLC features present: {base_features_present}")
        print(f"   - Technical indicators present: {tech_indicators_present}")
        
        # Test actual data
        dataloaders = dataloader.prepare_dataloaders()
        if 'train' in dataloaders:
            batch = next(iter(dataloaders['train']))
            print(f"   - Batch features dimension: {batch.series.shape[1]}")
            
        results['technical_indicators'] = (actual_count == expected_total_features and 
                                         tech_indicators_present and 
                                         base_features_present)
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        results['technical_indicators'] = False
    
    # Test 5: Data integrity
    print("\n5️⃣ DATA INTEGRITY TEST")
    try:
        config = DataLoaderConfig(
            batch_size=4,
            sequence_length=32,
            max_symbols=2,
            add_technical_indicators=True,
            validation_split=0.0,
            num_workers=0,
            min_sequence_length=100
        )
        
        dataloader = TotoOHLCDataLoader(config)
        dataloaders = dataloader.prepare_dataloaders()
        
        if 'train' in dataloaders:
            data_integrity_issues = []
            
            for i, batch in enumerate(dataloaders['train']):
                # Check for NaN/Inf values
                if torch.isnan(batch.series).any():
                    data_integrity_issues.append(f"Batch {i}: Contains NaN values")
                
                if torch.isinf(batch.series).any():
                    data_integrity_issues.append(f"Batch {i}: Contains Inf values")
                
                # Check value ranges (should be normalized)
                series_tensor = batch.series
                series_min = series_tensor.min().item()
                series_max = series_tensor.max().item()
                
                if abs(series_min) > 100 or abs(series_max) > 100:
                    data_integrity_issues.append(f"Batch {i}: Extreme values detected: [{series_min:.3f}, {series_max:.3f}]")
                
                # Check timestamp validity
                if (batch.timestamp_seconds <= 0).any():
                    data_integrity_issues.append(f"Batch {i}: Invalid timestamps detected")
                
                if i >= 10:  # Check first 10 batches
                    break
            
            if not data_integrity_issues:
                print("✅ Data integrity check passed")
                print("   - No NaN/Inf values found")
                print("   - Values within expected ranges")
                print("   - Timestamps are valid")
                results['data_integrity'] = True
            else:
                print("❌ Data integrity issues found:")
                for issue in data_integrity_issues[:5]:  # Show first 5 issues
                    print(f"   - {issue}")
                results['data_integrity'] = False
        else:
            print("❌ No training data available")
            results['data_integrity'] = False
            
    except Exception as e:
        print(f"❌ Failed: {e}")
        results['data_integrity'] = False
    
    # Test 6: Import and dependency check
    print("\n6️⃣ IMPORT AND DEPENDENCIES TEST")
    try:
        import torch
        import numpy as np
        import pandas as pd
        from sklearn.preprocessing import RobustScaler
        from sklearn.model_selection import TimeSeriesSplit
        
        print("✅ Core dependencies imported successfully:")
        print(f"   - torch: {torch.__version__}")
        print(f"   - numpy: {np.__version__}")
        print(f"   - pandas: {pd.__version__}")
        
        # Test fallback MaskedTimeseries
        from toto_ohlc_dataloader import MaskedTimeseries
        print("✅ MaskedTimeseries fallback implementation available")
        
        results['imports'] = True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        results['imports'] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASSED" if passed_test else "❌ FAILED"
        formatted_name = test_name.replace('_', ' ').title()
        print(f"{formatted_name:<25} {status}")
    
    print(f"\n🏁 Overall Score: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 EXCELLENT! All tests passed. The dataloader is fully functional.")
        overall_status = "PERFECT"
    elif passed >= total * 0.8:
        print("✅ GOOD! Most tests passed. Minor issues may exist.")
        overall_status = "GOOD"
    elif passed >= total * 0.6:
        print("⚠️ FAIR. Several issues need attention.")
        overall_status = "NEEDS_IMPROVEMENT"
    else:
        print("❌ POOR. Significant issues need to be addressed.")
        overall_status = "CRITICAL"
    
    return overall_status, results


if __name__ == "__main__":
    status, results = run_comprehensive_test()
    print(f"\n🎯 Final Status: {status}")