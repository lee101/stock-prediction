#!/usr/bin/env python3
"""
Multi-Experiment Runner for Testing Different Hyperparameters
Runs multiple training experiments in parallel/sequence to find optimal settings
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

from modern_transformer_trainer import (
    ModernTransformerConfig,
    ModernTrainingConfig, 
    ModernPPOTrainer
)
from trading_env import DailyTradingEnv
from trading_config import get_trading_costs
from train_full_model import generate_synthetic_data


class ExperimentConfig:
    """Configuration for a single experiment"""
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.config = kwargs
        self.results = {}
        
    def __repr__(self):
        return f"Experiment({self.name})"


def create_experiment_configs() -> List[ExperimentConfig]:
    """Create different experiment configurations to test"""
    
    experiments = []
    
    # ========================================
    # EXPERIMENT 1: Learning Rate Tests
    # ========================================
    lr_tests = [
        ("LR_VeryLow", {"learning_rate": 1e-5, "min_learning_rate": 1e-7}),
        ("LR_Low", {"learning_rate": 5e-5, "min_learning_rate": 1e-6}),
        ("LR_Medium", {"learning_rate": 1e-4, "min_learning_rate": 5e-6}),
        ("LR_High", {"learning_rate": 5e-4, "min_learning_rate": 1e-5}),
        ("LR_VeryHigh", {"learning_rate": 1e-3, "min_learning_rate": 5e-5}),
    ]
    
    for name, config in lr_tests:
        experiments.append(ExperimentConfig(name, **config))
    
    # ========================================
    # EXPERIMENT 2: Model Size Tests
    # ========================================
    model_size_tests = [
        ("Model_Tiny", {"d_model": 32, "n_heads": 2, "n_layers": 1}),
        ("Model_Small", {"d_model": 64, "n_heads": 4, "n_layers": 1}),
        ("Model_Medium", {"d_model": 128, "n_heads": 4, "n_layers": 2}),
        ("Model_Large", {"d_model": 256, "n_heads": 8, "n_layers": 2}),
    ]
    
    for name, config in model_size_tests:
        experiments.append(ExperimentConfig(name, **config))
    
    # ========================================
    # EXPERIMENT 3: Regularization Tests
    # ========================================
    regularization_tests = [
        ("Reg_None", {"dropout": 0.0, "weight_decay": 0.0}),
        ("Reg_Light", {"dropout": 0.1, "weight_decay": 0.001}),
        ("Reg_Medium", {"dropout": 0.3, "weight_decay": 0.01}),
        ("Reg_Heavy", {"dropout": 0.5, "weight_decay": 0.05}),
    ]
    
    for name, config in regularization_tests:
        experiments.append(ExperimentConfig(name, **config))
    
    # ========================================
    # EXPERIMENT 4: Scheduler Tests
    # ========================================
    scheduler_tests = [
        ("Sched_Linear", {"scheduler_type": "linear_warmup", "warmup_ratio": 0.1}),
        ("Sched_Cosine1", {"scheduler_type": "cosine_with_restarts", "num_cycles": 1.0}),
        ("Sched_Cosine3", {"scheduler_type": "cosine_with_restarts", "num_cycles": 3.0}),
        ("Sched_Cosine5", {"scheduler_type": "cosine_with_restarts", "num_cycles": 5.0}),
    ]
    
    for name, config in scheduler_tests:
        experiments.append(ExperimentConfig(name, **config))
    
    # ========================================
    # EXPERIMENT 5: PPO Hyperparameters
    # ========================================
    ppo_tests = [
        ("PPO_Conservative", {"ppo_clip": 0.1, "ppo_epochs": 3}),
        ("PPO_Standard", {"ppo_clip": 0.2, "ppo_epochs": 4}),
        ("PPO_Aggressive", {"ppo_clip": 0.3, "ppo_epochs": 10}),
    ]
    
    for name, config in ppo_tests:
        experiments.append(ExperimentConfig(name, **config))
    
    # ========================================
    # EXPERIMENT 6: Best Combined Settings
    # ========================================
    combined_tests = [
        ("Best_Conservative", {
            "learning_rate": 5e-5,
            "min_learning_rate": 1e-6,
            "d_model": 64,
            "n_heads": 4,
            "n_layers": 1,
            "dropout": 0.3,
            "weight_decay": 0.01,
            "scheduler_type": "cosine_with_restarts",
            "num_cycles": 3.0,
            "ppo_clip": 0.15,
            "ppo_epochs": 4
        }),
        ("Best_Balanced", {
            "learning_rate": 1e-4,
            "min_learning_rate": 5e-6,
            "d_model": 128,
            "n_heads": 4,
            "n_layers": 2,
            "dropout": 0.4,
            "weight_decay": 0.01,
            "scheduler_type": "cosine_with_restarts",
            "num_cycles": 2.0,
            "ppo_clip": 0.2,
            "ppo_epochs": 5
        }),
        ("Best_Aggressive", {
            "learning_rate": 5e-4,
            "min_learning_rate": 1e-5,
            "d_model": 128,
            "n_heads": 8,
            "n_layers": 2,
            "dropout": 0.2,
            "weight_decay": 0.005,
            "scheduler_type": "cosine_with_restarts",
            "num_cycles": 5.0,
            "ppo_clip": 0.25,
            "ppo_epochs": 8
        })
    ]
    
    for name, config in combined_tests:
        experiments.append(ExperimentConfig(name, **config))
    
    return experiments


def run_single_experiment(exp_config: ExperimentConfig, episodes: int = 500, device: str = 'cuda') -> Dict[str, Any]:
    """Run a single experiment with given configuration"""
    
    print(f"\n{'='*60}")
    print(f"🧪 Running Experiment: {exp_config.name}")
    print(f"{'='*60}")
    print(f"Config: {json.dumps(exp_config.config, indent=2)}")
    
    try:
        # Create model configuration
        model_config = ModernTransformerConfig(
            d_model=exp_config.config.get('d_model', 64),
            n_heads=exp_config.config.get('n_heads', 4),
            n_layers=exp_config.config.get('n_layers', 1),
            d_ff=exp_config.config.get('d_model', 64) * 2,
            dropout=exp_config.config.get('dropout', 0.3),
            weight_decay=exp_config.config.get('weight_decay', 0.01),
            gradient_checkpointing=False
        )
        
        # Create training configuration
        training_config = ModernTrainingConfig(
            model_config=model_config,
            learning_rate=exp_config.config.get('learning_rate', 1e-4),
            min_learning_rate=exp_config.config.get('min_learning_rate', 1e-6),
            weight_decay=exp_config.config.get('weight_decay', 0.01),
            scheduler_type=exp_config.config.get('scheduler_type', 'cosine_with_restarts'),
            num_cycles=exp_config.config.get('num_cycles', 2.0),
            warmup_ratio=exp_config.config.get('warmup_ratio', 0.1),
            ppo_clip=exp_config.config.get('ppo_clip', 0.2),
            ppo_epochs=exp_config.config.get('ppo_epochs', 4),
            num_episodes=episodes,
            eval_interval=50,
            batch_size=32,
            gradient_accumulation_steps=4
        )
        
        # Generate data
        train_data = generate_synthetic_data(n_days=500)
        val_data = generate_synthetic_data(n_days=200)
        
        # Create environments
        costs = get_trading_costs('stock', 'alpaca')
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns']
        available_features = [f for f in features if f in train_data.columns]
        
        train_env = DailyTradingEnv(
            train_data,
            window_size=20,
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
        training_config.model_config.input_dim = state.shape[1]
        
        # Create trainer
        trainer = ModernPPOTrainer(training_config, device=device)
        
        print(f"📊 Model: {trainer.model.get_num_parameters():,} parameters")
        
        # Train
        start_time = datetime.now()
        metrics = trainer.train(train_env, val_env, num_episodes=episodes)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Final evaluation
        final_reward, final_return = trainer.evaluate(val_env, num_episodes=5)
        
        # Get detailed metrics
        val_env.reset()
        state = val_env.reset()
        done = False
        while not done:
            action, _ = trainer.select_action(state, deterministic=True)
            state, _, done, _ = val_env.step([action])
        
        final_metrics = val_env.get_metrics()
        
        # Compile results
        results = {
            'name': exp_config.name,
            'config': exp_config.config,
            'model_params': trainer.model.get_num_parameters(),
            'training_time': training_time,
            'final_reward': final_reward,
            'final_return': final_return,
            'final_sharpe': final_metrics.get('sharpe_ratio', 0),
            'final_drawdown': final_metrics.get('max_drawdown', 0),
            'final_trades': final_metrics.get('num_trades', 0),
            'final_win_rate': final_metrics.get('win_rate', 0),
            'episode_rewards': metrics['episode_rewards'][-100:] if metrics['episode_rewards'] else [],
            'actor_losses': metrics['actor_losses'][-100:] if metrics['actor_losses'] else [],
            'learning_rates': metrics['learning_rates'][-100:] if metrics['learning_rates'] else []
        }
        
        # Close trainer
        trainer.close()
        
        print(f"✅ Experiment complete: Reward={final_reward:.4f}, Return={final_return:.2%}, Sharpe={results['final_sharpe']:.3f}")
        
        return results
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'name': exp_config.name,
            'config': exp_config.config,
            'error': str(e),
            'final_reward': -999,
            'final_return': -999,
            'final_sharpe': -999
        }


def run_experiments_parallel(experiments: List[ExperimentConfig], episodes: int = 500, max_workers: int = 2):
    """Run experiments in parallel"""
    
    print(f"\n{'='*80}")
    print(f"🚀 RUNNING {len(experiments)} EXPERIMENTS")
    print(f"{'='*80}")
    print(f"Episodes per experiment: {episodes}")
    print(f"Parallel workers: {max_workers}")
    
    results = []
    
    # Use CPU for parallel experiments to avoid GPU memory issues
    device = 'cpu'
    
    # Run experiments in batches
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for exp in experiments:
            future = executor.submit(run_single_experiment, exp, episodes, device)
            futures.append((exp.name, future))
        
        # Collect results
        for name, future in futures:
            try:
                result = future.result(timeout=600)  # 10 minute timeout
                results.append(result)
            except Exception as e:
                print(f"❌ {name} failed: {e}")
                results.append({
                    'name': name,
                    'error': str(e),
                    'final_reward': -999,
                    'final_return': -999,
                    'final_sharpe': -999
                })
    
    return results


def analyze_results(results: List[Dict[str, Any]]):
    """Analyze and visualize experiment results"""
    
    print(f"\n{'='*80}")
    print(f"📊 EXPERIMENT RESULTS ANALYSIS")
    print(f"{'='*80}")
    
    # Convert to DataFrame for easier analysis
    df_results = pd.DataFrame(results)
    
    # Remove failed experiments
    df_valid = df_results[df_results['final_reward'] != -999].copy()
    
    print(f"\nCompleted experiments: {len(df_valid)}/{len(results)}")
    
    if len(df_valid) == 0:
        print("❌ No experiments completed successfully")
        return
    
    # Sort by different metrics
    print("\n🏆 TOP 5 BY REWARD:")
    print(df_valid.nlargest(5, 'final_reward')[['name', 'final_reward', 'final_return', 'final_sharpe']])
    
    print("\n💰 TOP 5 BY RETURN:")
    print(df_valid.nlargest(5, 'final_return')[['name', 'final_reward', 'final_return', 'final_sharpe']])
    
    print("\n📈 TOP 5 BY SHARPE:")
    print(df_valid.nlargest(5, 'final_sharpe')[['name', 'final_reward', 'final_return', 'final_sharpe']])
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Bar plot of rewards
    ax = axes[0, 0]
    top_rewards = df_valid.nlargest(10, 'final_reward')
    ax.bar(range(len(top_rewards)), top_rewards['final_reward'])
    ax.set_xticks(range(len(top_rewards)))
    ax.set_xticklabels(top_rewards['name'], rotation=45, ha='right')
    ax.set_title('Top 10 by Reward')
    ax.set_ylabel('Final Reward')
    
    # Bar plot of returns
    ax = axes[0, 1]
    top_returns = df_valid.nlargest(10, 'final_return')
    ax.bar(range(len(top_returns)), top_returns['final_return'] * 100)
    ax.set_xticks(range(len(top_returns)))
    ax.set_xticklabels(top_returns['name'], rotation=45, ha='right')
    ax.set_title('Top 10 by Return (%)')
    ax.set_ylabel('Final Return (%)')
    
    # Bar plot of Sharpe ratios
    ax = axes[0, 2]
    top_sharpe = df_valid.nlargest(10, 'final_sharpe')
    ax.bar(range(len(top_sharpe)), top_sharpe['final_sharpe'])
    ax.set_xticks(range(len(top_sharpe)))
    ax.set_xticklabels(top_sharpe['name'], rotation=45, ha='right')
    ax.set_title('Top 10 by Sharpe Ratio')
    ax.set_ylabel('Sharpe Ratio')
    
    # Scatter plot: Return vs Sharpe
    ax = axes[1, 0]
    ax.scatter(df_valid['final_return'] * 100, df_valid['final_sharpe'])
    ax.set_xlabel('Return (%)')
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('Return vs Sharpe Ratio')
    for i, row in df_valid.iterrows():
        if row['final_sharpe'] > df_valid['final_sharpe'].quantile(0.9):
            ax.annotate(row['name'], (row['final_return'] * 100, row['final_sharpe']), fontsize=8)
    
    # Scatter plot: Reward vs Drawdown
    ax = axes[1, 1]
    ax.scatter(df_valid['final_reward'], df_valid['final_drawdown'] * 100)
    ax.set_xlabel('Final Reward')
    ax.set_ylabel('Max Drawdown (%)')
    ax.set_title('Reward vs Drawdown')
    
    # Win rate distribution
    ax = axes[1, 2]
    ax.hist(df_valid['final_win_rate'] * 100, bins=20, edgecolor='black')
    ax.set_xlabel('Win Rate (%)')
    ax.set_ylabel('Count')
    ax.set_title('Win Rate Distribution')
    
    plt.suptitle('Experiment Results Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save plot
    plt.savefig(f'results/experiments_{timestamp}.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Plot saved: results/experiments_{timestamp}.png")
    
    # Save detailed results
    df_valid.to_csv(f'results/experiments_{timestamp}.csv', index=False)
    print(f"📋 Results saved: results/experiments_{timestamp}.csv")
    
    # Save best configurations
    best_overall = df_valid.nlargest(1, 'final_sharpe').iloc[0]
    best_config = {
        'name': best_overall['name'],
        'config': best_overall['config'],
        'final_reward': float(best_overall['final_reward']),
        'final_return': float(best_overall['final_return']),
        'final_sharpe': float(best_overall['final_sharpe'])
    }
    
    with open(f'results/best_config_{timestamp}.json', 'w') as f:
        json.dump(best_config, f, indent=2)
    
    print(f"🏆 Best config saved: results/best_config_{timestamp}.json")
    
    return df_valid


def main():
    """Main experiment runner"""
    
    print("\n" + "="*80)
    print("🧪 HYPERPARAMETER EXPERIMENT RUNNER")
    print("="*80)
    
    # Create experiment configurations
    experiments = create_experiment_configs()
    
    print(f"\n📊 Configured {len(experiments)} experiments:")
    for exp in experiments[:10]:  # Show first 10
        print(f"   • {exp.name}")
    if len(experiments) > 10:
        print(f"   ... and {len(experiments) - 10} more")
    
    # Select subset for quick testing
    quick_test = True
    if quick_test:
        print("\n⚡ Quick test mode - running subset of experiments")
        # Run a diverse subset
        selected_experiments = [
            exp for exp in experiments 
            if any(x in exp.name for x in ['LR_Low', 'LR_Medium', 'LR_High', 
                                           'Model_Small', 'Model_Medium',
                                           'Reg_Light', 'Reg_Medium',
                                           'Best_Conservative', 'Best_Balanced'])
        ]
        experiments = selected_experiments[:8]  # Limit to 8 for speed
        episodes = 200  # Fewer episodes for quick test
    else:
        episodes = 500
    
    print(f"\n🚀 Running {len(experiments)} experiments with {episodes} episodes each")
    
    # Run experiments
    results = run_experiments_parallel(experiments, episodes=episodes, max_workers=2)
    
    # Analyze results
    Path('results').mkdir(exist_ok=True)
    df_results = analyze_results(results)
    
    print("\n" + "="*80)
    print("✅ EXPERIMENT RUNNER COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()