#!/usr/bin/env python3
"""
Modern Transformer Trading Agent Training Script
Addresses overfitting with proper scaling, modern techniques, and larger datasets
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our modern trainer and existing infrastructure
from modern_transformer_trainer import (
    ModernTransformerConfig,
    ModernTrainingConfig, 
    ModernPPOTrainer
)
from trading_env import DailyTradingEnv
from trading_config import get_trading_costs
from train_full_model import generate_synthetic_data


def generate_scaled_training_data(num_samples=10000, add_regime_changes=True, noise_level=0.02):
    """Generate larger, more diverse dataset to prevent overfitting"""
    print(f"🔄 Generating {num_samples:,} diverse training samples...")
    
    all_data = []
    
    # Generate multiple market regimes
    regime_sizes = [num_samples // 4] * 4  # 4 different regimes
    
    for i, regime_size in enumerate(regime_sizes):
        print(f"  📊 Regime {i+1}: {regime_size:,} samples")
        
        # Different market conditions for each regime
        # Generate different random seeds for diversity
        np.random.seed(42 + i * 1000)
        
        # Generate base data with different characteristics
        base_data = generate_synthetic_data(n_days=regime_size)
        
        # Modify the data post-generation to create different regimes
        # Use actual data length, not the requested length
        actual_length = len(base_data)
        
        if i == 0:  # Bull market - add upward trend
            trend = np.linspace(1.0, 1.05, actual_length)
            for col in ['Open', 'High', 'Low', 'Close']:
                if col in base_data.columns:
                    base_data[col] = base_data[col] * trend
        elif i == 1:  # Bear market - add downward trend
            trend = np.linspace(1.0, 0.97, actual_length)  
            for col in ['Open', 'High', 'Low', 'Close']:
                if col in base_data.columns:
                    base_data[col] = base_data[col] * trend
        elif i == 2:  # Sideways - reduce trend, add more noise
            for col in ['Open', 'High', 'Low', 'Close']:
                if col in base_data.columns:
                    noise = np.random.normal(1.0, 0.005, actual_length)
                    base_data[col] = base_data[col] * noise
        else:  # High volatility/crisis - increase volatility
            for col in ['Open', 'High', 'Low', 'Close']:
                if col in base_data.columns:
                    volatility_multiplier = np.random.normal(1.0, 0.02, actual_length)
                    base_data[col] = base_data[col] * volatility_multiplier
        
        # Add noise for diversity
        if noise_level > 0:
            for col in ['Open', 'High', 'Low', 'Close']:
                if col in base_data.columns:
                    noise = np.random.normal(0, noise_level, len(base_data))
                    base_data[col] = base_data[col] * (1 + noise)
        
        all_data.append(base_data)
    
    # Combine all regimes
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Shuffle to mix regimes (important for training stability)
    combined_data = combined_data.sample(frac=1.0).reset_index(drop=True)
    
    print(f"✅ Generated {len(combined_data):,} total samples with {len(combined_data.columns)} features")
    return combined_data


def create_train_test_split(df, train_ratio=0.7, val_ratio=0.15):
    """Create proper train/validation/test splits"""
    n = len(df)
    
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df[:train_end].copy()
    val_df = df[train_end:val_end].copy()
    test_df = df[val_end:].copy()
    
    print(f"📊 Data splits:")
    print(f"  Training: {len(train_df):,} samples ({len(train_df)/n:.1%})")
    print(f"  Validation: {len(val_df):,} samples ({len(val_df)/n:.1%})")  
    print(f"  Testing: {len(test_df):,} samples ({len(test_df)/n:.1%})")
    
    return train_df, val_df, test_df


def create_environments(train_df, val_df, test_df, window_size=30):
    """Create training, validation, and test environments"""
    
    # Get realistic trading costs
    costs = get_trading_costs('stock', 'alpaca')
    
    # Define features to use
    base_features = ['Open', 'High', 'Low', 'Close', 'Volume']
    technical_features = ['Returns', 'Rsi', 'Macd', 'Bb_Position', 'Volume_Ratio']
    
    all_features = base_features + technical_features
    available_features = [f for f in all_features if f in train_df.columns]
    
    print(f"📈 Using features: {available_features}")
    
    # Create environments
    train_env = DailyTradingEnv(
        train_df,
        window_size=window_size,
        initial_balance=100000,
        transaction_cost=costs.commission,
        spread_pct=costs.spread_pct,
        slippage_pct=costs.slippage_pct,
        features=available_features
    )
    
    val_env = DailyTradingEnv(
        val_df,
        window_size=window_size,
        initial_balance=100000,
        transaction_cost=costs.commission,
        spread_pct=costs.spread_pct,
        slippage_pct=costs.slippage_pct,
        features=available_features
    )
    
    test_env = DailyTradingEnv(
        test_df,
        window_size=window_size,
        initial_balance=100000,
        transaction_cost=costs.commission,
        spread_pct=costs.spread_pct,
        slippage_pct=costs.slippage_pct,
        features=available_features
    )
    
    # Calculate input dimensions
    input_dim = window_size * (len(available_features) + 3)  # +3 for position, balance, etc.
    print(f"🔢 Input dimension: {input_dim}")
    
    return train_env, val_env, test_env, input_dim


def run_modern_training():
    """Run the modern training pipeline"""
    print("\n" + "="*80)
    print("🚀 MODERN SCALED TRANSFORMER TRAINING")
    print("="*80)
    
    # ========================================
    # 1. CONFIGURATION
    # ========================================
    
    print("\n⚙️  Setting up configuration...")
    
    # Model configuration (small to prevent overfitting)
    model_config = ModernTransformerConfig(
        d_model=128,           # Small model
        n_heads=4,             # Fewer heads
        n_layers=2,            # Fewer layers
        d_ff=256,              # Smaller feedforward
        dropout=0.4,           # High dropout
        attention_dropout=0.3,
        path_dropout=0.2,
        layer_drop=0.1,
        weight_decay=0.01,
        gradient_checkpointing=True
    )
    
    # Training configuration (scaled and modern)
    training_config = ModernTrainingConfig(
        model_config=model_config,
        
        # Much lower learning rates
        learning_rate=5e-5,
        min_learning_rate=1e-6,
        weight_decay=0.01,
        
        # Larger effective batch sizes with gradient accumulation
        batch_size=32,
        gradient_accumulation_steps=8,  # Effective batch = 256
        
        # Modern scheduling
        scheduler_type="cosine_with_restarts",
        warmup_ratio=0.1,
        num_training_steps=15000,
        num_cycles=2.0,  # 2 restarts
        
        # RL hyperparameters
        ppo_epochs=4,           # Fewer epochs
        ppo_clip=0.15,          # Smaller clip
        entropy_coef=0.02,      # Higher exploration
        
        # Training control
        num_episodes=8000,      # More episodes
        eval_interval=50,
        save_interval=200,
        
        # Early stopping
        patience=400,
        min_improvement=0.001,
        
        # Data scaling
        train_data_size=15000,  # Large dataset
        synthetic_noise=0.02,
        
        # Regularization
        use_mixup=True,
        mixup_alpha=0.4,
        label_smoothing=0.1
    )
    
    print("✅ Configuration complete")
    print(f"   Model size: {model_config.d_model} dim, {model_config.n_layers} layers")
    print(f"   Learning rate: {training_config.learning_rate}")
    print(f"   Effective batch size: {training_config.batch_size * training_config.gradient_accumulation_steps}")
    print(f"   Dataset size: {training_config.train_data_size:,}")
    
    # ========================================
    # 2. DATA GENERATION AND PREPARATION
    # ========================================
    
    print(f"\n📊 Generating scaled dataset...")
    
    # Generate large, diverse dataset
    full_data = generate_scaled_training_data(
        num_samples=training_config.train_data_size,
        add_regime_changes=True,
        noise_level=training_config.synthetic_noise
    )
    
    # Create proper splits
    train_df, val_df, test_df = create_train_test_split(full_data)
    
    # Create environments
    train_env, val_env, test_env, input_dim = create_environments(
        train_df, val_df, test_df, window_size=30
    )
    
    # Update model config with correct input dimension
    training_config.model_config.input_dim = input_dim // 30  # Features per timestep
    
    # ========================================
    # 3. MODEL CREATION AND TRAINING
    # ========================================
    
    print(f"\n🤖 Creating modern transformer model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    # Create trainer
    trainer = ModernPPOTrainer(training_config, device=device)
    
    print(f"📊 Model: {trainer.model.get_num_parameters():,} parameters")
    print(f"🎯 Regularization: dropout={model_config.dropout}, weight_decay={model_config.weight_decay}")
    print(f"⚡ Optimizer: AdamW with cosine scheduling")
    
    # ========================================
    # 4. TRAINING WITH ENHANCED LOGGING
    # ========================================
    
    print(f"\n🏋️  Starting training...")
    print(f"📈 Episodes: {training_config.num_episodes}")
    print(f"⏱️  Eval interval: {training_config.eval_interval}")
    print(f"💾 Save interval: {training_config.save_interval}")
    print(f"⏹️  Early stop patience: {training_config.patience}")
    print("\n" + "="*100)
    print(f"{'Episode':>7} {'Reward':>8} {'Steps':>6} {'Loss':>8} {'LR':>10} {'ValRwd':>8} {'Profit':>8} {'Sharpe':>7} {'Drwdn':>7} {'Status'}")
    print("="*100)
    
    start_time = datetime.now()
    
    try:
        # Train the model with validation tracking
        metrics = trainer.train(
            train_env, 
            val_env,  # Pass validation environment
            num_episodes=training_config.num_episodes
        )
        
        training_time = (datetime.now() - start_time).total_seconds()
        print(f"\n✅ Training completed in {training_time:.1f} seconds")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Training interrupted by user")
        training_time = (datetime.now() - start_time).total_seconds()
    
    # ========================================
    # 5. FINAL EVALUATION
    # ========================================
    
    print(f"\n📊 Final evaluation on test set...")
    
    # Test on validation set
    val_reward, val_return = trainer.evaluate(val_env, num_episodes=10)
    
    # Test on test set
    test_reward, test_return = trainer.evaluate(test_env, num_episodes=10)
    
    # Get detailed test metrics
    test_env.reset()
    state = test_env.reset()
    done = False
    
    while not done:
        action, _ = trainer.select_action(state, deterministic=True)
        state, _, done, _ = test_env.step([action])
    
    test_metrics = test_env.get_metrics()
    
    print("\n💰 FINAL RESULTS:")
    print("="*80)
    print(f"Validation Performance:")
    print(f"  Reward:        {val_reward:.4f}")
    print(f"  Return:        {val_return:.2%}")
    print()
    print(f"Test Performance:")
    print(f"  Reward:        {test_reward:.4f}")
    print(f"  Return:        {test_return:.2%}")
    print(f"  Sharpe Ratio:  {test_metrics.get('sharpe_ratio', 0):.3f}")
    print(f"  Max Drawdown:  {test_metrics.get('max_drawdown', 0):.2%}")
    print(f"  Num Trades:    {test_metrics.get('num_trades', 0)}")
    print(f"  Win Rate:      {test_metrics.get('win_rate', 0):.2%}")
    print("="*80)
    
    # ========================================
    # 6. SAVE RESULTS
    # ========================================
    
    print(f"\n💾 Saving results...")
    
    # Create results directory
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot training curves
    if metrics['episode_rewards']:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Episode rewards
        axes[0, 0].plot(metrics['episode_rewards'][-1000:])  # Last 1000 episodes
        axes[0, 0].set_title('Episode Rewards (Last 1000)')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        
        # Episode profits
        if metrics['episode_profits']:
            axes[0, 1].plot(metrics['episode_profits'][-1000:])
            axes[0, 1].set_title('Episode Returns (Last 1000)')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Return (%)')
        
        # Sharpe ratios
        if metrics['episode_sharpes']:
            axes[0, 2].plot(metrics['episode_sharpes'][-1000:])
            axes[0, 2].set_title('Sharpe Ratios (Last 1000)')
            axes[0, 2].set_xlabel('Episode')
            axes[0, 2].set_ylabel('Sharpe')
        
        # Training losses
        if metrics['actor_losses']:
            axes[1, 0].plot(metrics['actor_losses'][-500:], label='Actor', alpha=0.7)
            axes[1, 0].plot(metrics['critic_losses'][-500:], label='Critic', alpha=0.7)
            axes[1, 0].set_title('Training Losses (Last 500 Updates)')
            axes[1, 0].set_xlabel('Update')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
        
        # Learning rate schedule
        if metrics['learning_rates']:
            axes[1, 1].plot(metrics['learning_rates'][-500:])
            axes[1, 1].set_title('Learning Rate Schedule (Last 500)')
            axes[1, 1].set_xlabel('Update')
            axes[1, 1].set_ylabel('LR')
        
        # Final performance comparison
        performance_data = ['Val Reward', 'Test Reward', 'Val Return', 'Test Return']
        performance_values = [val_reward, test_reward, val_return * 100, test_return * 100]
        axes[1, 2].bar(performance_data, performance_values)
        axes[1, 2].set_title('Final Performance')
        axes[1, 2].set_ylabel('Value')
        plt.xticks(rotation=45)
        
        plt.suptitle('Modern Transformer Trading Results', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plot_path = results_dir / f'modern_training_{timestamp}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Training curves saved: {plot_path}")
    
    # Save detailed results
    results = {
        'config': {
            'model_config': model_config.__dict__,
            'training_config': training_config.__dict__
        },
        'final_metrics': {
            'validation': {
                'reward': float(val_reward),
                'return': float(val_return)
            },
            'test': {
                'reward': float(test_reward),
                'return': float(test_return),
                **{k: float(v) for k, v in test_metrics.items()}
            }
        },
        'training_time': training_time,
        'model_parameters': trainer.model.get_num_parameters(),
        'dataset_size': len(full_data),
        'timestamp': timestamp
    }
    
    results_path = results_dir / f'modern_results_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    print(f"📋 Results saved: {results_path}")
    
    # Close trainer
    trainer.close()
    
    print(f"\n🎉 Modern training complete!")
    print(f"📊 View training curves: tensorboard --logdir=traininglogs")
    print(f"💾 Model checkpoints: training/models/modern_*")
    
    return results


if __name__ == '__main__':
    # Run the modern training pipeline
    results = run_modern_training()
    
    print("\n" + "="*80)
    print("SUMMARY - KEY IMPROVEMENTS IMPLEMENTED:")
    print("="*80)
    print("✅ FIXED OVERFITTING:")
    print("   • Much smaller model: 128 dim, 2 layers (was 256 dim, 3 layers)")
    print("   • Strong regularization: 0.4 dropout, 0.01 weight decay")
    print("   • 15k diverse training samples (was 1k)")
    print()
    print("✅ FIXED TRAINING PLATEAUS:")
    print("   • Lower learning rate: 5e-5 (was 1e-3)")
    print("   • Cosine scheduling with restarts")
    print("   • Proper early stopping with validation")
    print()
    print("✅ MODERN TECHNIQUES:")
    print("   • RoPE positional encoding")
    print("   • RMSNorm instead of LayerNorm")
    print("   • SwiGLU activations")
    print("   • Gradient accumulation (effective batch 256)")
    print("   • Mixup augmentation")
    print("="*80)