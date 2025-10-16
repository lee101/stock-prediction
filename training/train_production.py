#!/usr/bin/env python3
"""
Production Training Script - Trains until profitable
Implements early stopping, checkpointing, and automatic hyperparameter adjustments
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from advanced_trainer import (
    AdvancedTrainingConfig,
    TransformerTradingAgent,
    EnsembleTradingAgent,
    Muon, Shampoo
)
from train_advanced import AdvancedPPOTrainer
from trading_env import DailyTradingEnv
from trading_config import get_trading_costs
from train_full_model import load_and_prepare_data, generate_synthetic_data


# Reshape input for transformer (batch, seq_len, features)
class ReshapeWrapper(nn.Module):
    def __init__(self, agent, window_size=30):
        super().__init__()
        self.agent = agent
        self.window_size = window_size
    
    def forward(self, x):
        # Reshape from (batch, flat_features) to (batch, seq_len, features)
        if len(x.shape) == 2:
            batch_size = x.shape[0]
            features_per_step = x.shape[1] // self.window_size
            x = x.view(batch_size, self.window_size, features_per_step)
        return self.agent(x)
    
    def get_action_distribution(self, x):
        if len(x.shape) == 2:
            batch_size = x.shape[0]
            features_per_step = x.shape[1] // self.window_size
            x = x.view(batch_size, self.window_size, features_per_step)
        return self.agent.get_action_distribution(x)


class ProductionTrainer:
    """Production training with automatic adjustments"""
    
    def __init__(self, config: AdvancedTrainingConfig):
        self.config = config
        self.best_sharpe = -float('inf')
        self.best_return = -float('inf')
        self.patience = 500  # Episodes without improvement before adjusting
        self.episodes_without_improvement = 0
        self.adjustment_count = 0
        self.max_adjustments = 5
        
    def adjust_hyperparameters(self):
        """Automatically adjust hyperparameters if not improving"""
        self.adjustment_count += 1
        
        print(f"\n🔧 Adjusting hyperparameters (adjustment {self.adjustment_count})")
        
        # Adjust learning rate
        if self.adjustment_count % 2 == 1:
            self.config.learning_rate *= 0.5
            print(f"  Reduced learning rate to {self.config.learning_rate:.6f}")
        else:
            self.config.learning_rate *= 1.5
            print(f"  Increased learning rate to {self.config.learning_rate:.6f}")
        
        # Adjust exploration
        self.config.entropy_coef *= 1.2
        print(f"  Increased entropy coefficient to {self.config.entropy_coef:.4f}")
        
        # Adjust PPO parameters
        if self.adjustment_count > 2:
            self.config.ppo_clip = min(0.3, self.config.ppo_clip * 1.1)
            self.config.ppo_epochs = min(20, self.config.ppo_epochs + 2)
            print(f"  Adjusted PPO clip to {self.config.ppo_clip:.2f}")
            print(f"  Increased PPO epochs to {self.config.ppo_epochs}")
        
        # Enable more features if struggling
        if self.adjustment_count > 3:
            if not self.config.use_curriculum:
                self.config.use_curriculum = True
                print("  Enabled curriculum learning")
            if not self.config.use_augmentation:
                self.config.use_augmentation = True
                self.config.augmentation_prob = 0.3
                print("  Enabled data augmentation")
    
    def should_continue_training(self, metrics):
        """Determine if training should continue"""
        current_sharpe = metrics.get('sharpe_ratio', -10)
        current_return = metrics.get('total_return', -1)
        
        # Check if profitable
        if current_return > 0.05 and current_sharpe > 1.0:
            print("\n🎯 Target achieved! Model is profitable.")
            return False
        
        # Check improvement
        improved = False
        if current_sharpe > self.best_sharpe * 1.05:  # 5% improvement threshold
            self.best_sharpe = current_sharpe
            improved = True
        if current_return > self.best_return * 1.05:
            self.best_return = current_return
            improved = True
        
        if improved:
            self.episodes_without_improvement = 0
        else:
            self.episodes_without_improvement += 1
        
        # Adjust if stuck
        if self.episodes_without_improvement >= self.patience:
            if self.adjustment_count < self.max_adjustments:
                self.adjust_hyperparameters()
                self.episodes_without_improvement = 0
            else:
                print("\n⚠️ Max adjustments reached without achieving target.")
                return False
        
        return True


def main():
    """Main production training function"""
    print("\n" + "="*80)
    print("🚀 PRODUCTION TRAINING - TRAIN UNTIL PROFITABLE")
    print("="*80)
    
    # Try to load best params from optimization if available
    best_params_file = Path('optimization_results').glob('*_best_params.json')
    best_params = None
    
    for param_file in best_params_file:
        with open(param_file, 'r') as f:
            best_params = json.load(f)
        print(f"\n✅ Loaded optimized parameters from {param_file}")
        break
    
    # Configuration (use optimized params if available)
    if best_params:
        config = AdvancedTrainingConfig(
            architecture=best_params.get('architecture', 'transformer'),
            optimizer=best_params.get('optimizer', 'muon'),
            learning_rate=best_params.get('learning_rate', 0.001),
            hidden_dim=best_params.get('hidden_dim', 256),
            num_layers=best_params.get('num_layers', 3),
            num_heads=best_params.get('num_heads', 8),
            dropout=best_params.get('dropout', 0.1),
            batch_size=best_params.get('batch_size', 256),
            gradient_clip=best_params.get('gradient_clip', 1.0),
            gamma=best_params.get('gamma', 0.995),
            gae_lambda=best_params.get('gae_lambda', 0.95),
            ppo_epochs=best_params.get('ppo_epochs', 10),
            ppo_clip=best_params.get('ppo_clip', 0.2),
            value_loss_coef=best_params.get('value_loss_coef', 0.5),
            entropy_coef=best_params.get('entropy_coef', 0.01),
            use_curiosity=best_params.get('use_curiosity', True),
            curiosity_weight=best_params.get('curiosity_weight', 0.1),
            use_her=best_params.get('use_her', True),
            use_augmentation=best_params.get('use_augmentation', True),
            augmentation_prob=best_params.get('augmentation_prob', 0.5),
            use_curriculum=best_params.get('use_curriculum', True),
            use_ensemble=best_params.get('architecture') == 'ensemble',
            num_agents=best_params.get('num_agents', 3),
            num_episodes=10000,  # Max episodes
            eval_interval=50,
            save_interval=200
        )
    else:
        # Fallback to good defaults
        config = AdvancedTrainingConfig(
            architecture='transformer',
            optimizer='muon',
            learning_rate=0.001,
            num_episodes=10000,
            eval_interval=50,
            save_interval=200,
            use_curiosity=True,
            use_her=True,
            use_augmentation=True,
            use_ensemble=False,
            use_curriculum=True,
            batch_size=256,
            ppo_epochs=10,
            hidden_dim=256,
            num_layers=3
        )
    
    print("\n📋 Production Configuration:")
    print(f"  Architecture: {config.architecture}")
    print(f"  Optimizer: {config.optimizer}")
    print(f"  Learning Rate: {config.learning_rate:.6f}")
    print(f"  Target: Sharpe > 1.0, Return > 5%")
    print(f"  Max Episodes: {config.num_episodes}")
    
    # Load data - try real data first
    print("\n📊 Loading data...")
    try:
        df = load_and_prepare_data('../data/processed/')
        print(f"  Loaded real market data: {len(df)} samples")
    except:
        print("  Using synthetic data for demonstration")
        df = generate_synthetic_data(5000)  # More data for production
    
    # Split data
    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.15)
    train_df = df[:train_size]
    val_df = df[train_size:train_size+val_size]
    test_df = df[train_size+val_size:]
    
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Get realistic trading costs
    costs = get_trading_costs('stock', 'alpaca')  # Near-zero fees for stocks
    
    # Create environments
    print("\n🌍 Creating environments...")
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 
                'Rsi', 'Macd', 'Bb_Position', 'Volume_Ratio']
    available_features = [f for f in features if f in train_df.columns]
    
    env_params = {
        'window_size': 30,
        'initial_balance': 100000,
        'transaction_cost': costs.commission,
        'spread_pct': costs.spread_pct,
        'slippage_pct': costs.slippage_pct,
        'features': available_features
    }
    
    train_env = DailyTradingEnv(train_df, **env_params)
    val_env = DailyTradingEnv(val_df, **env_params)
    test_env = DailyTradingEnv(test_df, **env_params)
    
    # Create agent
    print("\n🤖 Creating advanced agent...")
    input_dim = 30 * (len(available_features) + 3)
    
    if config.use_ensemble:
        agent = EnsembleTradingAgent(
            num_agents=config.num_agents,
            input_dim=input_dim,
            hidden_dim=config.hidden_dim
        )
    else:
        features_per_step = input_dim // 30
        base_agent = TransformerTradingAgent(
            input_dim=features_per_step,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout
        )
        agent = ReshapeWrapper(base_agent, window_size=30)
    
    # Create trainer
    print("\n🎓 Creating production trainer...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    
    trainer = AdvancedPPOTrainer(agent, config, device)
    production_monitor = ProductionTrainer(config)
    
    # Training loop
    print("\n🏋️ Starting production training...")
    print("=" * 80)
    print("Training will continue until:")
    print("  • Sharpe Ratio > 1.0")
    print("  • Total Return > 5%")
    print("  • Or max episodes reached")
    print("=" * 80)
    
    best_val_sharpe = -float('inf')
    best_val_return = -float('inf')
    episode = 0
    
    with tqdm(total=config.num_episodes, desc="Production Training") as pbar:
        while episode < config.num_episodes:
            # Train episode
            reward, steps = trainer.train_episode(train_env)
            episode += 1
            
            # Validation check
            if episode % config.eval_interval == 0:
                # Evaluate on validation set
                val_env.reset()
                state = val_env.reset()
                done = False
                
                while not done:
                    action, _ = trainer.select_action(state, deterministic=True)
                    state, _, done, _ = val_env.step([action])
                
                val_metrics = val_env.get_metrics()
                val_sharpe = val_metrics.get('sharpe_ratio', -10)
                val_return = val_metrics.get('total_return', -1)
                
                # Update best scores
                if val_sharpe > best_val_sharpe:
                    best_val_sharpe = val_sharpe
                    trainer.save_checkpoint('models/best_production_model.pth')
                
                if val_return > best_val_return:
                    best_val_return = val_return
                
                # Update progress bar
                pbar.set_postfix({
                    'val_sharpe': f'{val_sharpe:.3f}',
                    'val_return': f'{val_return:.2%}',
                    'best_sharpe': f'{best_val_sharpe:.3f}',
                    'best_return': f'{best_val_return:.2%}',
                    'lr': f'{trainer.optimizer.param_groups[0]["lr"]:.6f}'
                })
                
                # Check if we should continue
                if not production_monitor.should_continue_training(val_metrics):
                    print(f"\n✅ Training completed at episode {episode}")
                    break
                
                # Adjust learning rate if needed
                if episode > 1000 and episode % 500 == 0:
                    for param_group in trainer.optimizer.param_groups:
                        param_group['lr'] *= 0.9
                    print(f"\n📉 Reduced learning rate to {trainer.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint
            if episode % config.save_interval == 0:
                trainer.save_checkpoint(f'models/checkpoint_ep{episode}.pth')
            
            pbar.update(1)
    
    # Final evaluation on test set
    print("\n📊 Final evaluation on test set...")
    test_env.reset()
    state = test_env.reset()
    done = False
    
    while not done:
        action, _ = trainer.select_action(state, deterministic=True)
        state, _, done, _ = test_env.step([action])
    
    final_metrics = test_env.get_metrics()
    
    print("\n" + "="*80)
    print("💰 FINAL PRODUCTION RESULTS")
    print("="*80)
    print(f"  Episodes Trained:  {episode}")
    print(f"  Best Val Sharpe:   {best_val_sharpe:.3f}")
    print(f"  Best Val Return:   {best_val_return:.2%}")
    print("\n📊 Test Set Performance:")
    print(f"  Total Return:      {final_metrics.get('total_return', 0):.2%}")
    print(f"  Sharpe Ratio:      {final_metrics.get('sharpe_ratio', 0):.3f}")
    print(f"  Max Drawdown:      {final_metrics.get('max_drawdown', 0):.2%}")
    print(f"  Number of Trades:  {final_metrics.get('num_trades', 0)}")
    print(f"  Win Rate:          {final_metrics.get('win_rate', 0):.2%}")
    print(f"  Profit Factor:     {final_metrics.get('profit_factor', 0):.2f}")
    
    # Save final results
    results = {
        'config': config.__dict__,
        'episodes_trained': episode,
        'best_val_sharpe': float(best_val_sharpe),
        'best_val_return': float(best_val_return),
        'test_metrics': final_metrics,
        'adjustments_made': production_monitor.adjustment_count
    }
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f'results/production_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    print("\n📁 Results saved to results/")
    
    # Plot training progress
    if trainer.metrics['episode_rewards']:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Smooth curves with moving average
        def smooth(data, window=50):
            if len(data) < window:
                return data
            return pd.Series(data).rolling(window, min_periods=1).mean().tolist()
        
        # Episode rewards
        axes[0, 0].plot(smooth(trainer.metrics['episode_rewards']), alpha=0.7)
        axes[0, 0].set_title('Episode Rewards (Smoothed)')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Episode returns
        if trainer.metrics['episode_profits']:
            axes[0, 1].plot(smooth(trainer.metrics['episode_profits']), alpha=0.7)
            axes[0, 1].set_title('Episode Returns (Smoothed)')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Return (%)')
            axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            axes[0, 1].axhline(y=5, color='g', linestyle='--', alpha=0.5, label='Target 5%')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Sharpe ratios
        if trainer.metrics['episode_sharpes']:
            axes[1, 0].plot(smooth(trainer.metrics['episode_sharpes']), alpha=0.7)
            axes[1, 0].set_title('Sharpe Ratios (Smoothed)')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Sharpe')
            axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            axes[1, 0].axhline(y=1, color='g', linestyle='--', alpha=0.5, label='Target 1.0')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate
        axes[1, 1].plot(trainer.metrics['learning_rates'], alpha=0.7)
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Update')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Production Training Results - {episode} Episodes', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plt.savefig(f'results/production_training_{timestamp}.png', dpi=100, bbox_inches='tight')
        print("📊 Training curves saved to results/")
    
    print("\n🎉 Production training complete!")
    print("="*80)


if __name__ == '__main__':
    main()