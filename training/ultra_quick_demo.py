#!/usr/bin/env python3
"""
Ultra quick training demo for immediate feedback
"""

import sys
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

from modern_transformer_trainer import (
    ModernTransformerConfig,
    ModernTrainingConfig, 
    ModernPPOTrainer
)
from trading_env import DailyTradingEnv
from trading_config import get_trading_costs
from train_full_model import generate_synthetic_data


def ultra_quick_demo():
    """Ultra quick demo with minimal complexity"""
    print("\n" + "="*80)
    print("🚀 ULTRA QUICK TRAINING DEMO (20 episodes)")
    print("="*80)
    
    # Minimal configuration
    model_config = ModernTransformerConfig(
        d_model=32,           # Very small
        n_heads=2,            
        n_layers=1,           # Just 1 layer
        d_ff=64,              
        dropout=0.1,          
        input_dim=9,          # Will be updated
        gradient_checkpointing=False  
    )
    
    training_config = ModernTrainingConfig(
        model_config=model_config,
        learning_rate=1e-3,   # Higher LR for faster learning
        batch_size=8,
        gradient_accumulation_steps=2,
        num_episodes=20,      # Very short
        eval_interval=5,      # Frequent evaluation
        patience=50
    )
    
    print("⚙️  Ultra-quick config:")
    print(f"   Model: {model_config.d_model} dim, {model_config.n_layers} layer")
    print(f"   Learning rate: {training_config.learning_rate}")
    print(f"   Episodes: {training_config.num_episodes}")
    
    # Minimal dataset
    print(f"\n📊 Creating minimal dataset...")
    train_data = generate_synthetic_data(n_days=100)  # Very small
    val_data = generate_synthetic_data(n_days=50)
    
    # Simple features
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns']
    available_features = [f for f in features if f in train_data.columns]
    
    print(f"   Train: {len(train_data)} samples, Val: {len(val_data)} samples")
    print(f"   Features: {available_features}")
    
    # Create environments
    costs = get_trading_costs('stock', 'alpaca')
    
    train_env = DailyTradingEnv(
        train_data,
        window_size=10,       # Small window
        initial_balance=100000,
        transaction_cost=costs.commission,
        spread_pct=costs.spread_pct,
        slippage_pct=costs.slippage_pct,
        features=available_features
    )
    
    val_env = DailyTradingEnv(
        val_data,
        window_size=10,
        initial_balance=100000,
        transaction_cost=costs.commission,
        spread_pct=costs.spread_pct,
        slippage_pct=costs.slippage_pct,
        features=available_features
    )
    
    # Get actual input dimension
    state = train_env.reset()
    input_dim = state.shape[1]  # Features per timestep
    training_config.model_config.input_dim = input_dim
    
    print(f"   State shape: {state.shape}")
    print(f"   Input dim per timestep: {input_dim}")
    
    # Create trainer
    print(f"\n🤖 Creating trainer...")
    trainer = ModernPPOTrainer(training_config, device='cpu')
    
    print(f"   Parameters: {trainer.model.get_num_parameters():,}")
    
    # Training header
    print(f"\n🏋️  Training with detailed logging...")
    print("=" * 100)
    print(f"{'Ep':>3} {'Reward':>8} {'Steps':>5} {'Loss':>8} {'LR':>10} {'VRew':>8} {'Profit':>8} {'Sharpe':>6} {'Drwdn':>7} {'Status'}")
    print("=" * 100)
    
    try:
        # Manual training loop for better control and logging
        best_reward = -float('inf')
        
        for episode in range(training_config.num_episodes):
            # Train episode
            reward, steps = trainer.train_episode(train_env)
            
            # Get loss if available
            loss = trainer.training_metrics['actor_losses'][-1] if trainer.training_metrics['actor_losses'] else 0.0
            lr = trainer.scheduler.get_last_lr()[0] if hasattr(trainer.scheduler, 'get_last_lr') else training_config.learning_rate
            
            # Evaluation every few episodes
            val_reward = reward  # Default to train reward
            profit = 0.0
            sharpe = 0.0
            drawdown = 0.0
            status = "Train"
            
            if (episode + 1) % training_config.eval_interval == 0:
                # Quick validation
                val_reward, _ = trainer.evaluate(val_env, num_episodes=1)
                
                # Get metrics
                val_env.reset()
                state = val_env.reset()
                done = False
                while not done:
                    action, _ = trainer.select_action(state, deterministic=True)
                    state, _, done, _ = val_env.step([action])
                
                val_metrics = val_env.get_metrics()
                profit = val_metrics.get('total_return', 0)
                sharpe = val_metrics.get('sharpe_ratio', 0)
                drawdown = val_metrics.get('max_drawdown', 0)
                
                status = "🔥BEST" if val_reward > best_reward else "Eval"
                if val_reward > best_reward:
                    best_reward = val_reward
            
            # Print progress
            print(f"{episode+1:3d} "
                  f"{reward:8.4f} "
                  f"{steps:5d} "
                  f"{loss:8.4f} "
                  f"{lr:10.6f} "
                  f"{val_reward:8.4f} "
                  f"{profit:8.2%} "
                  f"{sharpe:6.2f} "
                  f"{drawdown:7.2%} "
                  f"{status}")
        
        print("=" * 100)
        print(f"🏁 Ultra-quick demo complete!")
        print(f"   Best validation reward: {best_reward:.4f}")
        
        # Analysis
        print(f"\n📊 ANALYSIS:")
        rewards = trainer.training_metrics['episode_rewards']
        losses = trainer.training_metrics['actor_losses']
        
        if rewards:
            print(f"   Reward trend: {rewards[0]:.4f} → {rewards[-1]:.4f} (change: {rewards[-1] - rewards[0]:+.4f})")
        if losses:
            print(f"   Loss trend:   {losses[0]:.4f} → {losses[-1]:.4f} (change: {losses[-1] - losses[0]:+.4f})")
        
        # Simple trend analysis
        if len(rewards) >= 10:
            early_avg = np.mean(rewards[:5])
            late_avg = np.mean(rewards[-5:])
            improvement = late_avg - early_avg
            
            print(f"\n🔍 TREND ANALYSIS:")
            print(f"   Early episodes avg: {early_avg:.4f}")
            print(f"   Late episodes avg:  {late_avg:.4f}")
            print(f"   Improvement:        {improvement:+.4f}")
            
            if improvement > 0.01:
                print("   ✅ Learning trend: POSITIVE (model improving)")
            elif improvement > -0.01:
                print("   ⚠️  Learning trend: STABLE (no significant change)")
            else:
                print("   ❌ Learning trend: NEGATIVE (model degrading)")
        
        # Loss analysis
        if len(losses) >= 10:
            if losses[-1] < losses[0]:
                print("   ✅ Loss trend: DECREASING (good optimization)")
            else:
                print("   ⚠️  Loss trend: INCREASING (potential overfitting)")
        
        print(f"\n💡 QUICK RECOMMENDATIONS:")
        if len(rewards) < 5:
            print("   • Run more episodes for better analysis")
        else:
            avg_reward = np.mean(rewards)
            if avg_reward < 0:
                print("   • Negative rewards suggest poor policy - consider higher LR or different architecture")
            elif avg_reward < 0.1:
                print("   • Low rewards - may need more exploration (higher entropy) or different reward shaping")
            else:
                print("   • Reasonable rewards - continue training with current settings")
        
        return True
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Demo interrupted")
        return False
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        trainer.close()


if __name__ == '__main__':
    ultra_quick_demo()