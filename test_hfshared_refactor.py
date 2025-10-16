#!/usr/bin/env python3
"""Test script to verify hfshared refactoring works correctly."""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import shared utilities
import hfshared

def test_shared_utilities():
    """Test that shared utilities work correctly."""
    
    print("Testing hfshared utilities...")
    
    # Create sample data
    np.random.seed(42)
    data = pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=100),
        'Open': 100 + np.random.randn(100) * 2,
        'High': 102 + np.random.randn(100) * 2,
        'Low': 98 + np.random.randn(100) * 2,
        'Close': 100 + np.random.randn(100) * 2,
        'Volume': 1000000 + np.random.randn(100) * 100000
    })
    
    # Test 1: Compute training style features
    print("\n1. Testing compute_training_style_features...")
    features_df = hfshared.compute_training_style_features(data)
    assert isinstance(features_df, pd.DataFrame)
    assert len(features_df) == len(data)
    print(f"   ✓ Generated {len(features_df.columns)} features")
    
    # Test 2: Get canonical feature list
    print("\n2. Testing training_feature_columns_list...")
    feature_list = hfshared.training_feature_columns_list()
    assert isinstance(feature_list, list)
    assert 'close' in feature_list
    print(f"   ✓ Got {len(feature_list)} canonical features")
    
    # Test 3: Compute compact features
    print("\n3. Testing compute_compact_features...")
    compact_feats = hfshared.compute_compact_features(data, feature_mode='ohlcv')
    assert isinstance(compact_feats, np.ndarray)
    assert compact_feats.shape[0] == len(data)
    assert compact_feats.shape[1] == 5  # OHLCV
    print(f"   ✓ Generated compact features shape: {compact_feats.shape}")
    
    # Test 4: Z-score normalization
    print("\n4. Testing zscore_per_window...")
    normalized = hfshared.zscore_per_window(compact_feats)
    assert normalized.shape == compact_feats.shape
    assert np.abs(normalized.mean()) < 0.1  # Should be close to 0
    assert np.abs(normalized.std() - 1.0) < 0.1  # Should be close to 1
    print(f"   ✓ Z-score normalized: mean={normalized.mean():.3f}, std={normalized.std():.3f}")
    
    # Test 5: Input dimension inference (mock state dict)
    print("\n5. Testing infer_input_dim_from_state...")
    mock_state = {
        'input_projection.weight': np.zeros((512, 30)),
        'other_layer.weight': np.zeros((256, 512))
    }
    input_dim = hfshared.infer_input_dim_from_state(mock_state)
    assert input_dim == 30
    print(f"   ✓ Inferred input dimension: {input_dim}")
    
    print("\n✅ All hfshared utility tests passed!")

def test_inference_engines():
    """Test that refactored inference engines can import and initialize."""
    
    print("\n\nTesting inference engines...")
    
    try:
        # Test HF Trading Engine import
        print("\n1. Testing hf_trading_engine import...")
        from hfinference.hf_trading_engine import HFTradingEngine, DataProcessor
        print("   ✓ HFTradingEngine imported successfully")
        
        # Test DataProcessor initialization
        config = {'sequence_length': 60}
        processor = DataProcessor(config)
        print("   ✓ DataProcessor initialized")
        
        # Test Production Engine import
        print("\n2. Testing production_engine import...")
        from hfinference.production_engine import ProductionTradingEngine
        print("   ✓ ProductionTradingEngine imported successfully")
        
        print("\n✅ All inference engine imports successful!")
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return False
    
    return True

def main():
    """Main test function."""
    print("=" * 60)
    print("HFSHARED REFACTORING TEST")
    print("=" * 60)
    
    # Test shared utilities
    test_shared_utilities()
    
    # Test inference engines
    success = test_inference_engines()
    
    print("\n" + "=" * 60)
    if success:
        print("ALL TESTS PASSED! ✅")
        print("The hfshared refactoring is working correctly.")
    else:
        print("SOME TESTS FAILED ❌")
        print("Please check the errors above.")
    print("=" * 60)

if __name__ == "__main__":
    main()