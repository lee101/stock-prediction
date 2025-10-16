#!/usr/bin/env python3
"""
Smart Hyperparameter Optimization with Early Stopping
Uses curve fitting to predict final performance and stops unpromising runs early
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
from scipy.optimize import curve_fit
from torch.utils.tensorboard import SummaryWriter

from advanced_trainer import (
    AdvancedTrainingConfig,
    TransformerTradingAgent,
    EnsembleTradingAgent,
    Muon, Shampoo,
    create_advanced_agent,
    create_optimizer
)
from train_advanced import AdvancedPPOTrainer
from trading_env import DailyTradingEnv
from trading_config import get_trading_costs
from train_full_model import generate_synthetic_data


# Reshape wrapper for transformer
class ReshapeWrapper(nn.Module):
    def __init__(self, agent, window_size=30):
        super().__init__()
        self.agent = agent
        self.window_size = window_size
    
    def forward(self, x):
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


class SmartEarlyStopping:
    """Smart early stopping based on curve fitting"""
    
    def __init__(self, patience=20, min_episodes=30):
        self.patience = patience
        self.min_episodes = min_episodes
        self.val_losses = []
        self.val_sharpes = []
        self.val_returns = []
        
    def should_stop(self, episode, val_loss, val_sharpe, val_return):
        """Determine if training should stop based on curve fitting"""
        
        self.val_losses.append(val_loss)
        self.val_sharpes.append(val_sharpe)
        self.val_returns.append(val_return)
        
        # Need minimum episodes before evaluating
        if episode < self.min_episodes:
            return False, "Collecting initial data"
        
        # Fit curves to predict final performance
        x = np.arange(len(self.val_sharpes))
        
        try:
            # Fit exponential decay for loss: loss(t) = a * exp(-b * t) + c
            def exp_decay(t, a, b, c):
                return a * np.exp(-b * t) + c
            
            # Fit logarithmic growth for Sharpe: sharpe(t) = a * log(b * t + 1) + c
            def log_growth(t, a, b, c):
                return a * np.log(b * t + 1) + c
            
            # Fit loss curve
            if len(self.val_losses) > 10:
                try:
                    loss_params, _ = curve_fit(exp_decay, x, self.val_losses, 
                                              bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf]))
                    predicted_final_loss = exp_decay(len(x) * 3, *loss_params)  # Predict 3x further
                except:
                    predicted_final_loss = np.mean(self.val_losses[-5:])
            else:
                predicted_final_loss = np.mean(self.val_losses[-5:])
            
            # Fit Sharpe curve
            if len(self.val_sharpes) > 10:
                try:
                    sharpe_params, _ = curve_fit(log_growth, x, self.val_sharpes,
                                                bounds=([-np.inf, 0, -np.inf], [np.inf, np.inf, np.inf]))
                    predicted_final_sharpe = log_growth(len(x) * 3, *sharpe_params)
                except:
                    # Linear extrapolation if curve fit fails
                    recent_slope = (self.val_sharpes[-1] - self.val_sharpes[-10]) / 10
                    predicted_final_sharpe = self.val_sharpes[-1] + recent_slope * len(x)
            else:
                predicted_final_sharpe = np.mean(self.val_sharpes[-5:])
            
            # Check if we're trending badly
            recent_sharpes = self.val_sharpes[-self.patience:]
            sharpe_improving = np.mean(recent_sharpes) > np.mean(self.val_sharpes[-2*self.patience:-self.patience]) if len(self.val_sharpes) > 2*self.patience else True
            
            recent_returns = self.val_returns[-self.patience:]
            return_improving = np.mean(recent_returns) > np.mean(self.val_returns[-2*self.patience:-self.patience]) if len(self.val_returns) > 2*self.patience else True
            
            # Early stop if:
            # 1. Predicted final Sharpe is very bad (< 0.5)
            # 2. Not improving for patience episodes
            # 3. Returns are consistently negative
            
            if predicted_final_sharpe < 0.5 and not sharpe_improving:
                return True, f"Poor predicted Sharpe: {predicted_final_sharpe:.3f}"
            
            if np.mean(recent_returns) < -0.1 and not return_improving:
                return True, f"Consistently negative returns: {np.mean(recent_returns):.3%}"
            
            if episode > 100 and predicted_final_sharpe < 1.0 and predicted_final_loss > 0.1:
                return True, f"Unlikely to achieve target (Sharpe: {predicted_final_sharpe:.3f})"
                
        except Exception as e:
            # If curve fitting fails, use simple heuristics
            if episode > 50:
                if np.mean(self.val_sharpes[-10:]) < 0 and np.mean(self.val_returns[-10:]) < -0.05:
                    return True, "Poor recent performance"
        
        return False, f"Continuing (Sharpe: {val_sharpe:.3f}, Return: {val_return:.3%})"


def objective_with_smart_stopping(trial):
    """Objective function with smart early stopping"""
    
    # Create TensorBoard writer for this trial
    writer = SummaryWriter(f'traininglogs/optuna_trial_{trial.number}')
    
    # Hyperparameters to optimize
    config = AdvancedTrainingConfig(
        architecture=trial.suggest_categorical('architecture', ['transformer']),
        hidden_dim=trial.suggest_int('hidden_dim', 128, 512, step=64),
        num_layers=trial.suggest_int('num_layers', 2, 4),
        num_heads=trial.suggest_int('num_heads', 4, 8),
        dropout=trial.suggest_float('dropout', 0.0, 0.2, step=0.05),
        
        optimizer=trial.suggest_categorical('optimizer', ['adam', 'adamw', 'muon']),
        learning_rate=trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        batch_size=trial.suggest_int('batch_size', 128, 512, step=128),
        gradient_clip=trial.suggest_float('gradient_clip', 0.5, 2.0, step=0.5),
        
        gamma=trial.suggest_float('gamma', 0.98, 0.999, step=0.005),
        gae_lambda=trial.suggest_float('gae_lambda', 0.92, 0.98, step=0.02),
        ppo_epochs=trial.suggest_int('ppo_epochs', 5, 15, step=5),
        ppo_clip=trial.suggest_float('ppo_clip', 0.1, 0.3, step=0.05),
        value_loss_coef=trial.suggest_float('value_loss_coef', 0.25, 0.75, step=0.25),
        entropy_coef=trial.suggest_float('entropy_coef', 0.001, 0.05, log=True),
        
        use_curiosity=trial.suggest_categorical('use_curiosity', [True, False]),
        use_her=trial.suggest_categorical('use_her', [True, False]),
        use_augmentation=trial.suggest_categorical('use_augmentation', [True, False]),
        augmentation_prob=0.3 if trial.params.get('use_augmentation', False) else 0.0,
        use_curriculum=trial.suggest_categorical('use_curriculum', [True, False]),
        
        num_episodes=300,  # Max episodes per trial
        eval_interval=10,  # Frequent evaluation for early stopping
        save_interval=100,
        use_ensemble=False
    )
    
    # Log hyperparameters to TensorBoard
    writer.add_text('Hyperparameters', json.dumps(trial.params, indent=2), 0)
    
    # Generate data
    df = generate_synthetic_data(2000)
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
        
        # Smart early stopping
        early_stopper = SmartEarlyStopping(patience=20, min_episodes=30)
        
        best_sharpe = -float('inf')
        best_return = -float('inf')
        
        # Training loop with early stopping
        for episode in range(config.num_episodes):
            # Train episode
            reward, steps = trainer.train_episode(train_env)
            
            # Evaluate every eval_interval
            if (episode + 1) % config.eval_interval == 0:
                # Evaluate on test set
                test_reward = trainer.evaluate(test_env, num_episodes=3)
                
                # Get metrics
                test_env.reset()
                state = test_env.reset()
                done = False
                
                while not done:
                    action, _ = trainer.select_action(state, deterministic=True)
                    state, _, done, _ = test_env.step([action])
                
                test_metrics = test_env.get_metrics()
                sharpe = test_metrics.get('sharpe_ratio', -10)
                total_return = test_metrics.get('total_return', -1)
                
                # Update best scores
                best_sharpe = max(best_sharpe, sharpe)
                best_return = max(best_return, total_return)
                
                # Log to TensorBoard
                writer.add_scalar('Evaluation/Sharpe', sharpe, episode)
                writer.add_scalar('Evaluation/Return', total_return, episode)
                writer.add_scalar('Evaluation/Reward', test_reward, episode)
                writer.add_scalar('Evaluation/BestSharpe', best_sharpe, episode)
                writer.add_scalar('Evaluation/BestReturn', best_return, episode)
                
                # Check early stopping
                should_stop, reason = early_stopper.should_stop(
                    episode, 
                    -test_reward,  # Use negative reward as "loss"
                    sharpe,
                    total_return
                )
                
                if should_stop:
                    print(f"Trial {trial.number} stopped early at episode {episode}: {reason}")
                    writer.add_text('EarlyStopping', f"Stopped at episode {episode}: {reason}", episode)
                    break
                
                # Report to Optuna
                trial.report(sharpe, episode)
                
                # Optuna pruning
                if trial.should_prune():
                    writer.add_text('Pruning', f"Pruned by Optuna at episode {episode}", episode)
                    raise optuna.TrialPruned()
        
        # Final objective value
        objective_value = 0.7 * best_sharpe + 0.3 * (best_return * 10)
        
        writer.add_scalar('Final/ObjectiveValue', objective_value, 0)
        writer.add_scalar('Final/BestSharpe', best_sharpe, 0)
        writer.add_scalar('Final/BestReturn', best_return, 0)
        writer.close()
        
        return objective_value
        
    except optuna.TrialPruned:
        writer.close()
        raise
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        writer.add_text('Error', str(e), 0)
        writer.close()
        return -100


def main():
    """Main optimization function with smart early stopping"""
    print("\n" + "="*80)
    print("🔬 SMART HYPERPARAMETER OPTIMIZATION")
    print("="*80)
    print("\n📊 Features:")
    print("  • Curve fitting to predict final performance")
    print("  • Early stopping for unpromising runs")
    print("  • TensorBoard logging for each trial")
    print("  • Continues training hard on promising models")
    
    # Create study
    study_name = f"smart_trading_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=30
        ),
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # Optimize
    print("\n🏃 Starting smart optimization...")
    print(f"📊 TensorBoard: tensorboard --logdir=traininglogs")
    print("-" * 40)
    
    n_trials = 30
    
    study.optimize(
        objective_with_smart_stopping,
        n_trials=n_trials,
        n_jobs=1,
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
    
    print(f"\n📊 Results saved to optimization_results/")
    print(f"📊 View all trials: tensorboard --logdir=traininglogs")
    
    print("\n✅ Smart optimization complete!")
    print("="*80)


if __name__ == '__main__':
    main()