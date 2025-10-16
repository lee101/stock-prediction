#!/usr/bin/env python3
"""
Quick training demo to show the logging in action
"""

import sys
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# Import our modern trainer and existing infrastructure
from modern_transformer_trainer import (
    ModernTransformerConfig,
    ModernTrainingConfig, 
    ModernPPOTrainer
)
from trading_env import DailyTradingEnv
from trading_config import get_trading_costs
from train_full_model import generate_synthetic_data


def quick_demo():
    """Quick training demo with immediate stdout output"""
    print("\n" + "="*80)
    print("🚀 QUICK MODERN TRAINING DEMO")
    print("="*80)
    
    # Small configuration for quick demo
    model_config = ModernTransformerConfig(
        d_model=64,           # Small for demo
        n_heads=4,            
        n_layers=2,           
        d_ff=128,             
        dropout=0.3,          
        input_dim=6,          # Will be updated
        weight_decay=0.01,
        gradient_checkpointing=False  # Disable for demo
    )
    
    training_config = ModernTrainingConfig(
        model_config=model_config,
        learning_rate=1e-4,
        batch_size=16,
        gradient_accumulation_steps=4,
        num_episodes=200,     # Short demo
        eval_interval=20,     # Frequent evaluation
        save_interval=100,
        patience=100,
        train_data_size=1000, # Small dataset for demo
        use_mixup=False       # Disable for simplicity
    )
    
    print("⚙️  Quick configuration:")
    print(f"   Model: {model_config.d_model} dim, {model_config.n_layers} layers")
    print(f"   Learning rate: {training_config.learning_rate}")
    print(f"   Episodes: {training_config.num_episodes}")
    print(f"   Eval interval: {training_config.eval_interval}")
    
    # Generate small dataset
    print(f"\n📊 Generating demo dataset...")
    train_data = generate_synthetic_data(n_days=600)
    val_data = generate_synthetic_data(n_days=200)
    
    print(f"   Train data: {len(train_data):,} samples")
    print(f"   Val data: {len(val_data):,} samples")
    
    # Create environments
    costs = get_trading_costs('stock', 'alpaca')
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns']
    available_features = [f for f in features if f in train_data.columns]
    
    print(f"   Features: {available_features}")
    
    train_env = DailyTradingEnv(
        train_data,
        window_size=20,      # Smaller window for demo
        initial_balance=100000,
        transaction_cost=costs.commission,
        spread_pct=costs.spread_pct,
        slippage_pct=costs.slippage_pct,
        features=available_features
    )
    
    val_env = DailyTradingEnv(
        val_data,
        window_size=20,
        initial_balance=100000,
        transaction_cost=costs.commission,
        spread_pct=costs.spread_pct,
        slippage_pct=costs.slippage_pct,
        features=available_features
    )
    
    # Update input dimension
    state = train_env.reset()
    print(f"   State shape: {state.shape}")
    
    # State is (window_size, features) - we need features per timestep
    if len(state.shape) == 2:
        input_dim_per_step = state.shape[1]  # Features per timestep
    else:
        input_dim_per_step = state.shape[-1]  # Last dimension
    
    training_config.model_config.input_dim = input_dim_per_step
    print(f"   Input dimension per timestep: {input_dim_per_step}")
    
    # Create trainer
    print(f"\n🤖 Creating trainer...")
    device = 'cpu'  # Use CPU for demo to avoid GPU memory issues
    trainer = ModernPPOTrainer(training_config, device=device)
    
    print(f"   Device: {device}")
    print(f"   Model parameters: {trainer.model.get_num_parameters():,}")
    
    # Start training with enhanced logging
    print(f"\n🏋️  Starting demo training...")
    print("\n" + "="*100)
    print(f"{'Episode':>7} {'Reward':>8} {'Steps':>6} {'Loss':>8} {'LR':>10} {'ValRwd':>8} {'Profit':>8} {'Sharpe':>7} {'Drwdn':>7} {'Status'}")
    print("="*100)
    
    try:
        # Run training
        metrics = trainer.train(
            train_env, 
            val_env,
            num_episodes=training_config.num_episodes
        )
        
        print(f"\n✅ Demo training completed!")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        trainer.close()


if __name__ == '__main__':
    quick_demo()