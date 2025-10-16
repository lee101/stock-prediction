#!/usr/bin/env python3
"""
Test Toto model integration with the OHLC DataLoader
"""

import sys
import torch
from pathlib import Path

# Add toto to path
toto_path = Path(__file__).parent.parent / "toto"
sys.path.insert(0, str(toto_path))

from toto_ohlc_dataloader import TotoOHLCDataLoader, DataLoaderConfig, MaskedTimeseries as DataLoaderMaskedTimeseries

try:
    from toto.data.util.dataset import MaskedTimeseries as TotoMaskedTimeseries, replace_extreme_values
    TOTO_AVAILABLE = True
    print("✅ Successfully imported actual Toto MaskedTimeseries")
except ImportError as e:
    print(f"❌ Could not import Toto MaskedTimeseries: {e}")
    TOTO_AVAILABLE = False
    # Use fallback from dataloader
    replace_extreme_values = None


def test_maskedtimeseries_compatibility():
    """Test that our MaskedTimeseries is compatible with Toto's"""
    if not TOTO_AVAILABLE:
        print("⚠️ Skipping compatibility test - Toto not available")
        return False
    
    print("\n🔧 Testing MaskedTimeseries Compatibility")
    
    # Compare field names
    toto_fields = TotoMaskedTimeseries._fields
    dataloader_fields = DataLoaderMaskedTimeseries._fields
    
    print(f"Toto fields: {toto_fields}")
    print(f"DataLoader fields: {dataloader_fields}")
    
    if toto_fields == dataloader_fields:
        print("✅ Field names match perfectly")
    else:
        print("❌ Field names don't match")
        return False
    
    # Test creating instances
    config = DataLoaderConfig(
        batch_size=2,
        sequence_length=12,
        prediction_length=3,
        max_symbols=1,
        num_workers=0,
        validation_split=0.0,
        min_sequence_length=20
    )
    
    dataloader = TotoOHLCDataLoader(config)
    dataloaders = dataloader.prepare_dataloaders()
    
    if 'train' in dataloaders:
        batch = next(iter(dataloaders['train']))
        
        print(f"✅ Batch type: {type(batch)}")
        print(f"✅ Batch fields: {batch._fields}")
        print(f"✅ Series shape: {batch.series.shape}")
        print(f"✅ Series dtype: {batch.series.dtype}")
        
        # Test device transfer (both should work the same way)
        if torch.cuda.is_available():
            device = torch.device('cuda')
            batch_cuda = batch.to(device)
            print(f"✅ Device transfer works: {batch_cuda.series.device}")
        
        return True
    
    return False


def test_with_actual_toto_functions():
    """Test using actual Toto utility functions"""
    if not TOTO_AVAILABLE:
        print("⚠️ Skipping Toto functions test - Toto not available")
        return False
    
    print("\n🧪 Testing with Actual Toto Functions")
    
    config = DataLoaderConfig(
        batch_size=1,
        sequence_length=24,
        prediction_length=6,
        max_symbols=1,
        num_workers=0,
        validation_split=0.0,
        min_sequence_length=50
    )
    
    dataloader = TotoOHLCDataLoader(config)
    dataloaders = dataloader.prepare_dataloaders()
    
    if 'train' in dataloaders:
        batch = next(iter(dataloaders['train']))
        
        # Test replace_extreme_values with actual Toto function
        original_series = batch.series.clone()
        
        # Add some extreme values for testing
        test_tensor = original_series.clone()
        test_tensor[0, 0, 0] = float('inf')
        test_tensor[0, 1, 5] = float('-inf')
        test_tensor[0, 2, 10] = float('nan')
        
        cleaned_tensor = replace_extreme_values(test_tensor, replacement=0.0)
        
        print(f"✅ Original had inf/nan: {torch.isinf(test_tensor).any() or torch.isnan(test_tensor).any()}")
        print(f"✅ Cleaned has inf/nan: {torch.isinf(cleaned_tensor).any() or torch.isnan(cleaned_tensor).any()}")
        
        # Should have no extreme values after cleaning
        assert not torch.isinf(cleaned_tensor).any(), "Should not have inf values"
        assert not torch.isnan(cleaned_tensor).any(), "Should not have nan values"
        
        print("✅ replace_extreme_values works correctly")
        
        return True
    
    return False


def test_batch_format_details():
    """Test detailed batch format compatibility"""
    print("\n📊 Testing Detailed Batch Format")
    
    config = DataLoaderConfig(
        batch_size=2,
        sequence_length=48,
        prediction_length=12,
        max_symbols=2,
        num_workers=0,
        validation_split=0.0,
        add_technical_indicators=True,
        min_sequence_length=100
    )
    
    dataloader = TotoOHLCDataLoader(config)
    dataloaders = dataloader.prepare_dataloaders()
    
    if 'train' in dataloaders:
        batch = next(iter(dataloaders['train']))
        
        # Detailed shape analysis
        print(f"Batch shape analysis:")
        print(f"  series: {batch.series.shape} (batch_size, n_features, seq_len)")
        print(f"  padding_mask: {batch.padding_mask.shape}")
        print(f"  id_mask: {batch.id_mask.shape}")
        print(f"  timestamp_seconds: {batch.timestamp_seconds.shape}")
        print(f"  time_interval_seconds: {batch.time_interval_seconds.shape}")
        
        # Verify expected shapes
        batch_size, n_features, seq_len = batch.series.shape
        
        assert batch_size == config.batch_size, f"Expected batch size {config.batch_size}, got {batch_size}"
        assert seq_len == config.sequence_length, f"Expected sequence length {config.sequence_length}, got {seq_len}"
        
        # Check data types
        assert batch.series.dtype == torch.float32, f"Expected float32, got {batch.series.dtype}"
        assert batch.padding_mask.dtype == torch.bool, f"Expected bool, got {batch.padding_mask.dtype}"
        assert batch.id_mask.dtype == torch.long, f"Expected long, got {batch.id_mask.dtype}"
        assert batch.timestamp_seconds.dtype == torch.long, f"Expected long, got {batch.timestamp_seconds.dtype}"
        assert batch.time_interval_seconds.dtype == torch.long, f"Expected long, got {batch.time_interval_seconds.dtype}"
        
        print("✅ All shape and type checks passed")
        
        # Check data ranges and validity
        print(f"Data ranges:")
        print(f"  series: [{batch.series.min():.3f}, {batch.series.max():.3f}]")
        print(f"  timestamps: [{batch.timestamp_seconds.min()}, {batch.timestamp_seconds.max()}]")
        print(f"  time_intervals: {torch.unique(batch.time_interval_seconds).tolist()}")
        print(f"  id_mask unique: {torch.unique(batch.id_mask).tolist()}")
        
        # Verify no extreme values
        assert not torch.isinf(batch.series).any(), "Series should not contain inf"
        assert not torch.isnan(batch.series).any(), "Series should not contain nan"
        
        print("✅ Data validity checks passed")
        
        return True
    
    return False


def main():
    """Run all Toto integration tests"""
    print("🧪 Toto Integration Tests\n")
    
    test_results = {
        "MaskedTimeseries Compatibility": test_maskedtimeseries_compatibility(),
        "Toto Functions Integration": test_with_actual_toto_functions(), 
        "Batch Format Details": test_batch_format_details()
    }
    
    print("\n" + "="*50)
    print("📊 TOTO INTEGRATION TEST RESULTS")
    print("="*50)
    
    passed = 0
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<30} {status}")
        if result:
            passed += 1
    
    print(f"\n🏁 Overall: {passed}/{len(test_results)} tests passed")
    
    if passed == len(test_results):
        print("🎉 Perfect Toto integration! DataLoader is fully compatible.")
        return True
    else:
        print("⚠️ Some integration issues found.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)