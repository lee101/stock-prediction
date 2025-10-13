#!/usr/bin/env python3
"""
Example usage of the TotoOHLCDataLoader with different configurations
"""

import torch
from pathlib import Path
from toto_ohlc_dataloader import TotoOHLCDataLoader, DataLoaderConfig

def example_basic_usage():
    """Basic usage example"""
    print("🚀 Basic DataLoader Usage")
    
    config = DataLoaderConfig(
        batch_size=8,
        sequence_length=48,
        prediction_length=12,
        max_symbols=3,  # Limit for quick testing
        validation_split=0.3
    )
    
    dataloader = TotoOHLCDataLoader(config)
    dataloaders = dataloader.prepare_dataloaders()
    
    print(f"✅ Created {len(dataloaders)} dataloaders")
    for name, dl in dataloaders.items():
        print(f"   {name}: {len(dl.dataset)} samples")
    
    return dataloaders

def example_advanced_features():
    """Advanced features example"""
    print("\n📈 Advanced Features Example")
    
    config = DataLoaderConfig(
        batch_size=16,
        sequence_length=96,
        prediction_length=24,
        
        # Advanced preprocessing
        normalization_method="robust",
        add_technical_indicators=True,
        ma_periods=[5, 20, 50],
        
        # Data filtering
        outlier_threshold=2.5,
        min_sequence_length=200,
        
        # Cross-validation
        cv_folds=3,
        
        max_symbols=5
    )
    
    dataloader = TotoOHLCDataLoader(config)
    dataloaders = dataloader.prepare_dataloaders()
    
    # Get feature information
    feature_info = dataloader.get_feature_info()
    print(f"📊 Features: {feature_info['n_features']}")
    print(f"🎯 Target: {feature_info['target_feature']}")
    
    # Test cross-validation
    cv_splits = dataloader.get_cross_validation_splits(2)
    print(f"🔀 Cross-validation splits: {len(cv_splits)}")
    
    return dataloaders, cv_splits

def example_config_management():
    """Configuration management example"""
    print("\n⚙️ Configuration Management Example")
    
    # Create and save config
    config = DataLoaderConfig(
        sequence_length=120,
        prediction_length=30,
        batch_size=32,
        add_technical_indicators=True,
        normalization_method="standard"
    )
    
    config_path = "example_config.json"
    config.save(config_path)
    print(f"💾 Saved config to {config_path}")
    
    # Load config
    loaded_config = DataLoaderConfig.load(config_path)
    print(f"📂 Loaded config: sequence_length={loaded_config.sequence_length}")
    
    # Clean up
    Path(config_path).unlink()

def example_data_inspection():
    """Data inspection example"""
    print("\n🔍 Data Inspection Example")
    
    config = DataLoaderConfig(
        batch_size=4,
        sequence_length=24,
        prediction_length=6,
        max_symbols=2,
        num_workers=0  # Disable multiprocessing for debugging
    )
    
    dataloader = TotoOHLCDataLoader(config)
    dataloaders = dataloader.prepare_dataloaders()
    
    if 'train' in dataloaders:
        train_loader = dataloaders['train']
        
        # Inspect first batch
        for i, batch in enumerate(train_loader):
            print(f"Batch {i + 1}:")
            print(f"  Series shape: {batch.series.shape}")
            print(f"  Series dtype: {batch.series.dtype}")
            print(f"  Series range: [{batch.series.min():.3f}, {batch.series.max():.3f}]")
            print(f"  Padding mask: {batch.padding_mask.sum().item()} valid elements")
            print(f"  ID mask unique values: {torch.unique(batch.id_mask).tolist()}")
            print(f"  Timestamps range: [{batch.timestamp_seconds.min()}, {batch.timestamp_seconds.max()}]")
            
            if i >= 1:  # Just show first 2 batches
                break
    
    # Check targets
    if 'train' in dataloaders:
        train_dataset = dataloaders['train'].dataset
        targets = train_dataset.get_targets()
        if len(targets) > 0:
            print(f"🎯 Targets shape: {targets.shape}")
            print(f"   Targets range: [{targets.min():.3f}, {targets.max():.3f}]")

def main():
    """Run all examples"""
    print("🧪 Toto OHLC DataLoader Examples\n")
    
    try:
        # Basic usage
        basic_dataloaders = example_basic_usage()
        
        # Advanced features
        advanced_dataloaders, cv_splits = example_advanced_features()
        
        # Configuration management
        example_config_management()
        
        # Data inspection
        example_data_inspection()
        
        print("\n✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()