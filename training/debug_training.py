#!/usr/bin/env python3
"""
Debug script to test data generation and initial setup
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from train_full_model import generate_synthetic_data
from trading_env import DailyTradingEnv
from trading_config import get_trading_costs


def test_data_generation():
    """Test the data generation process"""
    print("\n🧪 Testing data generation...")
    
    # Test basic generation
    try:
        data = generate_synthetic_data(n_days=100)
        print(f"✅ Basic generation: {data.shape}")
        print(f"   Columns: {list(data.columns)}")
        print(f"   Date range: {data.index[0]} to {data.index[-1]}")
        
        # Check for NaN values
        nan_count = data.isnull().sum().sum()
        print(f"   NaN values: {nan_count}")
        
        return True
    except Exception as e:
        print(f"❌ Data generation failed: {e}")
        return False


def test_environment_creation():
    """Test environment creation"""
    print("\n🧪 Testing environment creation...")
    
    try:
        # Generate test data
        data = generate_synthetic_data(n_days=200)
        
        # Get costs
        costs = get_trading_costs('stock', 'alpaca')
        
        # Define features
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns']
        available_features = [f for f in features if f in data.columns]
        
        print(f"   Available features: {available_features}")
        
        # Create environment
        env = DailyTradingEnv(
            data,
            window_size=30,
            initial_balance=100000,
            transaction_cost=costs.commission,
            spread_pct=costs.spread_pct,
            slippage_pct=costs.slippage_pct,
            features=available_features
        )
        
        # Test reset
        state = env.reset()
        print(f"✅ Environment created: state shape {state.shape}")
        
        # Test step
        action = [0.5]  # Test action
        next_state, reward, done, info = env.step(action)
        print(f"   Step test: reward={reward:.4f}, done={done}")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment creation failed: {e}")
        return False


def test_model_creation():
    """Test modern transformer model creation"""
    print("\n🧪 Testing model creation...")
    
    try:
        from modern_transformer_trainer import (
            ModernTransformerConfig, 
            ModernTrainingConfig,
            ModernTransformerTradingAgent
        )
        
        # Create configs
        model_config = ModernTransformerConfig(
            d_model=64,      # Smaller for testing
            n_heads=2,       
            n_layers=1,      
            input_dim=10,    # Test input dim
            dropout=0.1
        )
        
        # Create model
        model = ModernTransformerTradingAgent(model_config)
        print(f"✅ Model created: {model.get_num_parameters():,} parameters")
        
        # Test forward pass
        batch_size = 2
        seq_len = 30
        features = 10
        
        test_input = torch.randn(batch_size, seq_len, features)
        
        with torch.no_grad():
            action, value, attention = model(test_input)
            print(f"   Forward pass: action {action.shape}, value {value.shape}")
            
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def quick_training_test():
    """Quick test of training loop setup"""
    print("\n🧪 Testing training setup...")
    
    try:
        from modern_transformer_trainer import (
            ModernTransformerConfig,
            ModernTrainingConfig, 
            ModernPPOTrainer
        )
        
        # Small configs for testing
        model_config = ModernTransformerConfig(
            d_model=32,
            n_heads=2,
            n_layers=1,
            input_dim=8,
            dropout=0.1
        )
        
        training_config = ModernTrainingConfig(
            model_config=model_config,
            learning_rate=1e-4,
            batch_size=16,
            gradient_accumulation_steps=2,
            num_episodes=10,  # Very small for testing
            eval_interval=5
        )
        
        # Create trainer
        trainer = ModernPPOTrainer(training_config, device='cpu')  # Use CPU for testing
        print(f"✅ Trainer created")
        
        # Create test environment  
        data = generate_synthetic_data(n_days=100)
        costs = get_trading_costs('stock', 'alpaca')
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_features = [f for f in features if f in data.columns]
        
        env = DailyTradingEnv(
            data,
            window_size=10,  # Smaller window
            initial_balance=100000,
            transaction_cost=costs.commission,
            spread_pct=costs.spread_pct,
            slippage_pct=costs.slippage_pct,
            features=available_features
        )
        
        # Test single episode
        reward, steps = trainer.train_episode(env)
        print(f"✅ Training episode: reward={reward:.4f}, steps={steps}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🔧 DEBUGGING MODERN TRAINING SETUP")
    print("="*60)
    
    tests = [
        ("Data Generation", test_data_generation),
        ("Environment Creation", test_environment_creation),
        ("Model Creation", test_model_creation),
        ("Training Setup", quick_training_test)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"🧪 Running: {test_name}")
        print('='*60)
        
        results[test_name] = test_func()
    
    print(f"\n{'='*60}")
    print("📊 SUMMARY")
    print('='*60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:20} {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print(f"\n🎉 All tests passed! Ready for full training.")
    else:
        print(f"\n⚠️  Some tests failed. Fix issues before full training.")