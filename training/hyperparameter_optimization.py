#!/usr/bin/env python3
"""
Comprehensive Hyperparameter Optimization for Trading System
Uses Optuna for Bayesian optimization
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from advanced_trainer import (
    AdvancedTrainingConfig,
    TransformerTradingAgent,
    EnsembleTradingAgent,
    Muon, Shampoo,
    create_advanced_agent,
    create_optimizer
)
from train_advanced import AdvancedPPOTrainer, ReshapeWrapper
from trading_env import DailyTradingEnv
from trading_config import get_trading_costs
from train_full_model import generate_synthetic_data


def objective(trial):
    """Objective function for hyperparameter optimization"""
    
    # Hyperparameters to optimize
    config = AdvancedTrainingConfig(
        # Architecture
        architecture=trial.suggest_categorical('architecture', ['transformer', 'ensemble']),
        hidden_dim=trial.suggest_int('hidden_dim', 128, 512, step=64),
        num_layers=trial.suggest_int('num_layers', 2, 5),
        num_heads=trial.suggest_int('num_heads', 4, 8),
        dropout=trial.suggest_float('dropout', 0.0, 0.3, step=0.05),
        
        # Optimization
        optimizer=trial.suggest_categorical('optimizer', ['adam', 'adamw', 'muon']),
        learning_rate=trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        batch_size=trial.suggest_int('batch_size', 64, 512, step=64),
        gradient_clip=trial.suggest_float('gradient_clip', 0.5, 2.0, step=0.25),
        
        # RL
        gamma=trial.suggest_float('gamma', 0.95, 0.999, step=0.005),
        gae_lambda=trial.suggest_float('gae_lambda', 0.9, 0.99, step=0.01),
        ppo_epochs=trial.suggest_int('ppo_epochs', 5, 20, step=5),
        ppo_clip=trial.suggest_float('ppo_clip', 0.1, 0.3, step=0.05),
        value_loss_coef=trial.suggest_float('value_loss_coef', 0.25, 1.0, step=0.25),
        entropy_coef=trial.suggest_float('entropy_coef', 0.001, 0.1, log=True),
        
        # Advanced features
        use_curiosity=trial.suggest_categorical('use_curiosity', [True, False]),
        curiosity_weight=trial.suggest_float('curiosity_weight', 0.01, 0.5, log=True) 
            if trial.params.get('use_curiosity', False) else 0.0,
        use_her=trial.suggest_categorical('use_her', [True, False]),
        use_augmentation=trial.suggest_categorical('use_augmentation', [True, False]),
        augmentation_prob=trial.suggest_float('augmentation_prob', 0.1, 0.7, step=0.1)
            if trial.params.get('use_augmentation', False) else 0.0,
        use_curriculum=trial.suggest_categorical('use_curriculum', [True, False]),
        
        # Training
        num_episodes=200,  # Shorter for optimization
        eval_interval=50,
        save_interval=100,
        
        # Ensemble
        use_ensemble=False,  # Set based on architecture
        num_agents=trial.suggest_int('num_agents', 3, 7) 
            if trial.params.get('architecture') == 'ensemble' else 3
    )
    
    # Update ensemble flag
    config.use_ensemble = (config.architecture == 'ensemble')
    
    # Generate data
    df = generate_synthetic_data(1000)
    train_size = int(len(df) * 0.8)
    train_df = df[:train_size]
    test_df = df[train_size:]
    
    # Get realistic trading costs
    costs = get_trading_costs('stock', 'alpaca')
    
    # Create environments
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 
                'Rsi', 'Macd', 'Bb_Position', 'Volume_Ratio']
    available_features = [f for f in features if f in train_df.columns]
    
    train_env = DailyTradingEnv(
        train_df,
        window_size=30,
        initial_balance=100000,
        transaction_cost=costs.commission,
        spread_pct=costs.spread_pct,
        slippage_pct=costs.slippage_pct,
        features=available_features
    )
    
    test_env = DailyTradingEnv(
        test_df,
        window_size=30,
        initial_balance=100000,
        transaction_cost=costs.commission,
        spread_pct=costs.spread_pct,
        slippage_pct=costs.slippage_pct,
        features=available_features
    )
    
    # Create agent
    input_dim = 30 * (len(available_features) + 3)
    
    try:
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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        trainer = AdvancedPPOTrainer(agent, config, device)
        
        # Train
        metrics = trainer.train(train_env, num_episodes=config.num_episodes)
        
        # Evaluate
        test_reward = trainer.evaluate(test_env, num_episodes=10)
        
        # Get final metrics
        test_env.reset()
        state = test_env.reset()
        done = False
        
        while not done:
            action, _ = trainer.select_action(state, deterministic=True)
            state, _, done, _ = test_env.step([action])
        
        final_metrics = test_env.get_metrics()
        
        # Compute objective value (maximize Sharpe ratio and return)
        sharpe = final_metrics.get('sharpe_ratio', -10)
        total_return = final_metrics.get('total_return', -1)
        
        # Weighted objective
        objective_value = 0.7 * sharpe + 0.3 * (total_return * 10)
        
        # Report intermediate values
        trial.report(objective_value, config.num_episodes)
        
        # Handle pruning
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        return objective_value
        
    except Exception as e:
        print(f"Trial failed with error: {e}")
        return -100  # Return bad score for failed trials


def main():
    """Main optimization function"""
    print("\n" + "="*80)
    print("🔬 HYPERPARAMETER OPTIMIZATION FOR ADVANCED TRADING SYSTEM")
    print("="*80)
    
    # Create study
    study_name = f"trading_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=50
        ),
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # Optimize
    print("\n🏃 Starting optimization...")
    print("-" * 40)
    
    n_trials = 50  # Number of trials
    
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=1,  # Set to >1 for parallel optimization if you have multiple GPUs
        show_progress_bar=True
    )
    
    # Print results
    print("\n" + "="*80)
    print("📊 OPTIMIZATION RESULTS")
    print("="*80)
    
    print("\n🏆 Best trial:")
    best_trial = study.best_trial
    print(f"  Objective Value: {best_trial.value:.4f}")
    print(f"  Trial Number: {best_trial.number}")
    
    print("\n📈 Best parameters:")
    for key, value in best_trial.params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    # Save results
    Path('optimization_results').mkdir(exist_ok=True)
    
    # Save study
    study_df = study.trials_dataframe()
    study_df.to_csv(f'optimization_results/{study_name}.csv', index=False)
    
    # Save best params
    with open(f'optimization_results/{study_name}_best_params.json', 'w') as f:
        json.dump(best_trial.params, f, indent=2)
    
    # Create visualization plots
    try:
        # Optimization history
        fig = plot_optimization_history(study)
        fig.write_html(f'optimization_results/{study_name}_history.html')
        
        # Parameter importance
        fig = plot_param_importances(study)
        fig.write_html(f'optimization_results/{study_name}_importance.html')
        
        print(f"\n📊 Visualizations saved to optimization_results/")
    except Exception as e:
        print(f"Could not create visualizations: {e}")
    
    # Print top 5 trials
    print("\n🥇 Top 5 trials:")
    print("-" * 40)
    
    trials_df = study.trials_dataframe().sort_values('value', ascending=False).head(5)
    for idx, row in trials_df.iterrows():
        print(f"\nTrial {int(row['number'])}:")
        print(f"  Value: {row['value']:.4f}")
        print(f"  Architecture: {row['params_architecture']}")
        print(f"  Optimizer: {row['params_optimizer']}")
        print(f"  Learning Rate: {row['params_learning_rate']:.6f}")
        print(f"  Hidden Dim: {int(row['params_hidden_dim'])}")
    
    # Configuration recommendation
    print("\n" + "="*80)
    print("💡 CONFIGURATION RECOMMENDATION")
    print("="*80)
    
    print("\nBased on optimization results, here's the recommended configuration:")
    print("\n```python")
    print("config = AdvancedTrainingConfig(")
    for key, value in best_trial.params.items():
        if isinstance(value, float):
            print(f"    {key}={value:.6f},")
        elif isinstance(value, str):
            print(f"    {key}='{value}',")
        else:
            print(f"    {key}={value},")
    print("    num_episodes=1000,  # Increase for production")
    print("    eval_interval=50,")
    print("    save_interval=200")
    print(")")
    print("```")
    
    print("\n✅ Optimization complete!")
    print("="*80)


if __name__ == '__main__':
    main()