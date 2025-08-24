#!/usr/bin/env python3
"""
Quick experiment runner to test key hyperparameters
Focus on what really matters: learning rate, model size, and regularization
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Any

from modern_transformer_trainer import (
    ModernTransformerConfig,
    ModernTrainingConfig, 
    ModernPPOTrainer
)
from trading_env import DailyTradingEnv
from trading_config import get_trading_costs
from train_full_model import generate_synthetic_data


def run_quick_experiment(name: str, config_overrides: Dict, episodes: int = 100) -> Dict[str, Any]:
    """Run a single quick experiment"""
    
    print(f"\n{'='*60}")
    print(f"🧪 Experiment: {name}")
    print(f"   Config: {config_overrides}")
    
    # Base configuration (small for speed)
    model_config = ModernTransformerConfig(
        d_model=64,
        n_heads=4,
        n_layers=1,
        d_ff=128,
        dropout=config_overrides.get('dropout', 0.3),
        weight_decay=config_overrides.get('weight_decay', 0.01),
        gradient_checkpointing=False
    )
    
    training_config = ModernTrainingConfig(
        model_config=model_config,
        learning_rate=config_overrides.get('learning_rate', 1e-4),
        min_learning_rate=config_overrides.get('min_learning_rate', 1e-6),
        scheduler_type=config_overrides.get('scheduler_type', 'cosine_with_restarts'),
        num_cycles=config_overrides.get('num_cycles', 2.0),
        ppo_clip=config_overrides.get('ppo_clip', 0.2),
        ppo_epochs=config_overrides.get('ppo_epochs', 4),
        num_episodes=episodes,
        eval_interval=20,
        batch_size=32,
        gradient_accumulation_steps=2
    )
    
    # Update model size if specified
    if 'd_model' in config_overrides:
        model_config.d_model = config_overrides['d_model']
        model_config.d_ff = config_overrides['d_model'] * 2
    if 'n_layers' in config_overrides:
        model_config.n_layers = config_overrides['n_layers']
    
    # Generate small dataset
    train_data = generate_synthetic_data(n_days=200)
    val_data = generate_synthetic_data(n_days=100)
    
    # Create environments
    costs = get_trading_costs('stock', 'alpaca')
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns']
    available_features = [f for f in features if f in train_data.columns]
    
    train_env = DailyTradingEnv(
        train_data,
        window_size=15,
        initial_balance=100000,
        transaction_cost=costs.commission,
        spread_pct=costs.spread_pct,
        slippage_pct=costs.slippage_pct,
        features=available_features
    )
    
    val_env = DailyTradingEnv(
        val_data,
        window_size=15,
        initial_balance=100000,
        transaction_cost=costs.commission,
        spread_pct=costs.spread_pct,
        slippage_pct=costs.slippage_pct,
        features=available_features
    )
    
    # Update input dimension
    state = train_env.reset()
    training_config.model_config.input_dim = state.shape[1]
    
    # Create trainer
    trainer = ModernPPOTrainer(training_config, device='cpu')
    
    print(f"   Model params: {trainer.model.get_num_parameters():,}")
    
    # Train
    start_time = datetime.now()
    
    best_reward = -float('inf')
    best_return = -float('inf')
    rewards = []
    losses = []
    
    for episode in range(episodes):
        # Train episode
        reward, steps = trainer.train_episode(train_env)
        rewards.append(reward)
        
        if trainer.training_metrics['actor_losses']:
            losses.append(trainer.training_metrics['actor_losses'][-1])
        
        # Quick evaluation
        if (episode + 1) % 20 == 0:
            val_reward, val_return = trainer.evaluate(val_env, num_episodes=2)
            best_reward = max(best_reward, val_reward)
            best_return = max(best_return, val_return)
            
            print(f"   Ep {episode+1:3d}: Train={reward:.3f}, Val={val_reward:.3f}, Return={val_return:.1%}")
    
    training_time = (datetime.now() - start_time).total_seconds()
    
    # Final evaluation
    final_reward, final_return = trainer.evaluate(val_env, num_episodes=5)
    
    # Get metrics
    val_env.reset()
    state = val_env.reset()
    done = False
    while not done:
        action, _ = trainer.select_action(state, deterministic=True)
        state, _, done, _ = val_env.step([action])
    
    final_metrics = val_env.get_metrics()
    
    # Calculate improvement
    early_avg = np.mean(rewards[:10]) if len(rewards) >= 10 else rewards[0] if rewards else 0
    late_avg = np.mean(rewards[-10:]) if len(rewards) >= 10 else rewards[-1] if rewards else 0
    improvement = late_avg - early_avg
    
    results = {
        'name': name,
        'config': config_overrides,
        'model_params': trainer.model.get_num_parameters(),
        'training_time': training_time,
        'final_reward': final_reward,
        'final_return': final_return,
        'final_sharpe': final_metrics.get('sharpe_ratio', 0),
        'best_reward': best_reward,
        'best_return': best_return,
        'reward_improvement': improvement,
        'final_loss': losses[-1] if losses else 0
    }
    
    trainer.close()
    
    print(f"   ✅ Complete: Reward={final_reward:.3f}, Return={final_return:.1%}, Sharpe={results['final_sharpe']:.2f}")
    
    return results


def main():
    """Run quick experiments and analyze results"""
    
    print("\n" + "="*80)
    print("🚀 QUICK HYPERPARAMETER EXPERIMENTS")
    print("="*80)
    
    experiments = [
        # Learning rate experiments (most important)
        ("LR_1e-5", {"learning_rate": 1e-5}),
        ("LR_5e-5", {"learning_rate": 5e-5}),
        ("LR_1e-4", {"learning_rate": 1e-4}),
        ("LR_5e-4", {"learning_rate": 5e-4}),
        ("LR_1e-3", {"learning_rate": 1e-3}),
        
        # Regularization experiments
        ("Dropout_0.0", {"dropout": 0.0}),
        ("Dropout_0.2", {"dropout": 0.2}),
        ("Dropout_0.4", {"dropout": 0.4}),
        ("Dropout_0.6", {"dropout": 0.6}),
        
        # Model size experiments
        ("Model_32", {"d_model": 32}),
        ("Model_64", {"d_model": 64}),
        ("Model_128", {"d_model": 128}),
        
        # Best combinations
        ("Best_Small", {"learning_rate": 1e-4, "dropout": 0.3, "d_model": 64}),
        ("Best_Medium", {"learning_rate": 5e-5, "dropout": 0.4, "d_model": 128}),
        ("Best_LowReg", {"learning_rate": 1e-4, "dropout": 0.1, "d_model": 64}),
    ]
    
    results = []
    
    print(f"\n📊 Running {len(experiments)} experiments with 100 episodes each...")
    
    for name, config in experiments:
        try:
            result = run_quick_experiment(name, config, episodes=100)
            results.append(result)
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append({
                'name': name,
                'config': config,
                'error': str(e),
                'final_reward': -999,
                'final_return': -999,
                'final_sharpe': -999
            })
    
    # Analyze results
    print("\n" + "="*80)
    print("📊 RESULTS ANALYSIS")
    print("="*80)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    df_valid = df[df['final_reward'] != -999].copy()
    
    if len(df_valid) == 0:
        print("❌ No experiments completed successfully")
        return
    
    # Sort by different metrics
    print("\n🏆 TOP 5 BY REWARD:")
    top_reward = df_valid.nlargest(5, 'final_reward')[['name', 'final_reward', 'final_return', 'final_sharpe']]
    print(top_reward.to_string(index=False))
    
    print("\n💰 TOP 5 BY RETURN:")
    top_return = df_valid.nlargest(5, 'final_return')[['name', 'final_reward', 'final_return', 'final_sharpe']]
    print(top_return.to_string(index=False))
    
    print("\n📈 TOP 5 BY SHARPE:")
    top_sharpe = df_valid.nlargest(5, 'final_sharpe')[['name', 'final_reward', 'final_return', 'final_sharpe']]
    print(top_sharpe.to_string(index=False))
    
    print("\n🔄 TOP 5 BY IMPROVEMENT:")
    top_improve = df_valid.nlargest(5, 'reward_improvement')[['name', 'reward_improvement', 'final_reward', 'final_return']]
    print(top_improve.to_string(index=False))
    
    # Analyze by experiment type
    print("\n📊 ANALYSIS BY EXPERIMENT TYPE:")
    
    # Learning rate analysis
    lr_experiments = df_valid[df_valid['name'].str.startswith('LR_')]
    if not lr_experiments.empty:
        print("\n🎯 Learning Rate Analysis:")
        for _, row in lr_experiments.iterrows():
            lr = row['config'].get('learning_rate', 0)
            print(f"   LR={lr:.1e}: Reward={row['final_reward']:.3f}, Return={row['final_return']:.1%}, Sharpe={row['final_sharpe']:.2f}")
        
        best_lr_idx = lr_experiments['final_sharpe'].idxmax()
        best_lr = df_valid.loc[best_lr_idx]
        print(f"   ✅ Best LR: {best_lr['config'].get('learning_rate'):.1e}")
    
    # Dropout analysis
    dropout_experiments = df_valid[df_valid['name'].str.startswith('Dropout_')]
    if not dropout_experiments.empty:
        print("\n💧 Dropout Analysis:")
        for _, row in dropout_experiments.iterrows():
            dropout = row['config'].get('dropout', 0)
            print(f"   Dropout={dropout:.1f}: Reward={row['final_reward']:.3f}, Return={row['final_return']:.1%}, Sharpe={row['final_sharpe']:.2f}")
        
        best_dropout_idx = dropout_experiments['final_sharpe'].idxmax()
        best_dropout = df_valid.loc[best_dropout_idx]
        print(f"   ✅ Best Dropout: {best_dropout['config'].get('dropout'):.1f}")
    
    # Model size analysis
    model_experiments = df_valid[df_valid['name'].str.startswith('Model_')]
    if not model_experiments.empty:
        print("\n📏 Model Size Analysis:")
        for _, row in model_experiments.iterrows():
            d_model = row['config'].get('d_model', 0)
            print(f"   Size={d_model}: Params={row['model_params']:,}, Reward={row['final_reward']:.3f}, Return={row['final_return']:.1%}")
        
        best_model_idx = model_experiments['final_sharpe'].idxmax()
        best_model = df_valid.loc[best_model_idx]
        print(f"   ✅ Best Size: {best_model['config'].get('d_model')}")
    
    # Overall best
    print("\n🌟 OVERALL BEST CONFIGURATION:")
    best_overall = df_valid.loc[df_valid['final_sharpe'].idxmax()]
    print(f"   Name: {best_overall['name']}")
    print(f"   Config: {best_overall['config']}")
    print(f"   Final Reward: {best_overall['final_reward']:.3f}")
    print(f"   Final Return: {best_overall['final_return']:.1%}")
    print(f"   Final Sharpe: {best_overall['final_sharpe']:.2f}")
    print(f"   Improvement: {best_overall['reward_improvement']:.3f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Learning rate vs performance
    if not lr_experiments.empty:
        ax = axes[0, 0]
        lrs = [row['config'].get('learning_rate', 0) for _, row in lr_experiments.iterrows()]
        sharpes = lr_experiments['final_sharpe'].values
        ax.semilogx(lrs, sharpes, 'o-')
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Sharpe Ratio')
        ax.set_title('Learning Rate vs Performance')
        ax.grid(True)
    
    # Dropout vs performance
    if not dropout_experiments.empty:
        ax = axes[0, 1]
        dropouts = [row['config'].get('dropout', 0) for _, row in dropout_experiments.iterrows()]
        sharpes = dropout_experiments['final_sharpe'].values
        ax.plot(dropouts, sharpes, 'o-')
        ax.set_xlabel('Dropout Rate')
        ax.set_ylabel('Sharpe Ratio')
        ax.set_title('Dropout vs Performance')
        ax.grid(True)
    
    # Model size vs performance
    if not model_experiments.empty:
        ax = axes[1, 0]
        sizes = [row['config'].get('d_model', 0) for _, row in model_experiments.iterrows()]
        sharpes = model_experiments['final_sharpe'].values
        ax.plot(sizes, sharpes, 'o-')
        ax.set_xlabel('Model Size (d_model)')
        ax.set_ylabel('Sharpe Ratio')
        ax.set_title('Model Size vs Performance')
        ax.grid(True)
    
    # Overall comparison
    ax = axes[1, 1]
    names = df_valid.nlargest(10, 'final_sharpe')['name'].values
    sharpes = df_valid.nlargest(10, 'final_sharpe')['final_sharpe'].values
    y_pos = np.arange(len(names))
    ax.barh(y_pos, sharpes)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel('Sharpe Ratio')
    ax.set_title('Top 10 Configurations')
    
    plt.suptitle('Hyperparameter Experiment Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save results
    Path('results').mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    plt.savefig(f'results/quick_experiments_{timestamp}.png', dpi=150, bbox_inches='tight')
    df_valid.to_csv(f'results/quick_experiments_{timestamp}.csv', index=False)
    
    # Save best config
    best_config = {
        'name': best_overall['name'],
        'config': best_overall['config'],
        'performance': {
            'final_reward': float(best_overall['final_reward']),
            'final_return': float(best_overall['final_return']),
            'final_sharpe': float(best_overall['final_sharpe'])
        }
    }
    
    with open(f'results/best_config_{timestamp}.json', 'w') as f:
        json.dump(best_config, f, indent=2)
    
    print(f"\n💾 Results saved:")
    print(f"   Plot: results/quick_experiments_{timestamp}.png")
    print(f"   Data: results/quick_experiments_{timestamp}.csv")
    print(f"   Best: results/best_config_{timestamp}.json")
    
    print("\n" + "="*80)
    print("✅ EXPERIMENTS COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()