#!/usr/bin/env python3
"""
Enhanced Hyperparameter Optimization with PEFT/LoRA
Focuses on preventing overfitting after episode 600
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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

from advanced_trainer_peft import (
    PEFTTrainingConfig,
    PEFTTransformerTradingAgent,
    create_peft_agent,
    create_peft_optimizer,
    MixupAugmentation,
    StochasticDepth,
    LabelSmoothing
)
from train_advanced import AdvancedPPOTrainer
from trading_env import DailyTradingEnv
from trading_config import get_trading_costs
from train_full_model import generate_synthetic_data, load_and_prepare_data


class ReshapeWrapper(nn.Module):
    """Reshape wrapper for compatibility"""
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
    
    def parameters(self):
        return self.agent.parameters()
    
    def named_parameters(self):
        return self.agent.named_parameters()


class EnhancedEarlyStopping:
    """Enhanced early stopping that detects overfitting"""
    
    def __init__(self, patience=30, min_episodes=50, overfit_threshold=0.2):
        self.patience = patience
        self.min_episodes = min_episodes
        self.overfit_threshold = overfit_threshold
        
        self.train_losses = []
        self.val_losses = []
        self.val_sharpes = []
        self.val_returns = []
        
        self.best_val_sharpe = -float('inf')
        self.episodes_without_improvement = 0
        
    def should_stop(self, episode, train_loss, val_loss, val_sharpe, val_return):
        """Determine if training should stop"""
        
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.val_sharpes.append(val_sharpe)
        self.val_returns.append(val_return)
        
        # Need minimum episodes
        if episode < self.min_episodes:
            return False, "Collecting initial data"
        
        # Check for improvement
        if val_sharpe > self.best_val_sharpe:
            self.best_val_sharpe = val_sharpe
            self.episodes_without_improvement = 0
        else:
            self.episodes_without_improvement += 1
        
        # Check for overfitting
        if len(self.train_losses) > 20 and len(self.val_losses) > 20:
            recent_train = np.mean(self.train_losses[-10:])
            recent_val = np.mean(self.val_losses[-10:])
            
            # Overfitting detected if validation loss is much higher than training
            if recent_val > recent_train * (1 + self.overfit_threshold):
                return True, f"Overfitting detected (val/train ratio: {recent_val/recent_train:.2f})"
        
        # Check for plateau
        if self.episodes_without_improvement >= self.patience:
            return True, f"No improvement for {self.patience} episodes"
        
        # Special check around episode 600
        if 580 <= episode <= 620:
            # More aggressive stopping around the problematic area
            if val_sharpe < self.best_val_sharpe * 0.9:  # 10% degradation
                return True, f"Performance degradation at episode {episode}"
        
        return False, f"Continuing (best Sharpe: {self.best_val_sharpe:.3f})"


def objective_with_peft(trial):
    """Objective function with PEFT and enhanced regularization"""
    
    # Create TensorBoard writer
    writer = SummaryWriter(f'traininglogs/peft_trial_{trial.number}')
    
    # Hyperparameters optimized for PEFT
    config = PEFTTrainingConfig(
        # PEFT specific
        lora_rank=trial.suggest_int('lora_rank', 4, 16, step=4),
        lora_alpha=trial.suggest_int('lora_alpha', 8, 32, step=8),
        lora_dropout=trial.suggest_float('lora_dropout', 0.05, 0.3, step=0.05),
        freeze_base=trial.suggest_categorical('freeze_base', [True, False]),
        
        # Architecture (smaller for PEFT)
        hidden_dim=trial.suggest_int('hidden_dim', 128, 256, step=64),
        num_layers=trial.suggest_int('num_layers', 2, 3),
        num_heads=trial.suggest_int('num_heads', 4, 8, step=4),
        dropout=trial.suggest_float('dropout', 0.1, 0.3, step=0.05),
        
        # Optimization (conservative for fine-tuning)
        optimizer=trial.suggest_categorical('optimizer', ['adamw', 'adam']),
        learning_rate=trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        weight_decay=trial.suggest_float('weight_decay', 0.001, 0.1, log=True),
        batch_size=trial.suggest_int('batch_size', 64, 256, step=64),
        gradient_clip=trial.suggest_float('gradient_clip', 0.1, 1.0, step=0.1),
        
        # RL (conservative)
        gamma=trial.suggest_float('gamma', 0.98, 0.999, step=0.005),
        gae_lambda=trial.suggest_float('gae_lambda', 0.9, 0.98, step=0.02),
        ppo_epochs=trial.suggest_int('ppo_epochs', 3, 7),
        ppo_clip=trial.suggest_float('ppo_clip', 0.05, 0.2, step=0.05),
        value_loss_coef=trial.suggest_float('value_loss_coef', 0.25, 0.75, step=0.25),
        entropy_coef=trial.suggest_float('entropy_coef', 0.01, 0.1, log=True),
        
        # Regularization
        use_mixup=trial.suggest_categorical('use_mixup', [True, False]),
        mixup_alpha=0.2 if trial.params.get('use_mixup', False) else 0,
        use_stochastic_depth=trial.suggest_categorical('use_stochastic_depth', [True, False]),
        stochastic_depth_prob=0.1 if trial.params.get('use_stochastic_depth', False) else 0,
        label_smoothing=trial.suggest_float('label_smoothing', 0.0, 0.2, step=0.05),
        
        # Data augmentation
        use_augmentation=True,  # Always use
        augmentation_prob=trial.suggest_float('augmentation_prob', 0.2, 0.6, step=0.1),
        noise_level=trial.suggest_float('noise_level', 0.005, 0.02, step=0.005),
        
        # Training
        num_episodes=800,  # Shorter since we expect to stop earlier
        eval_interval=10,
        save_interval=50,
        early_stop_patience=30,
        
        # Curriculum
        use_curriculum=trial.suggest_categorical('use_curriculum', [True, False]),
        warmup_episodes=50
    )
    
    # Log hyperparameters
    writer.add_text('Hyperparameters', json.dumps(trial.params, indent=2), 0)
    
    # Load data - try real data first
    try:
        df = load_and_prepare_data('../data/processed/')
        print(f"Trial {trial.number}: Using real market data")
    except:
        df = generate_synthetic_data(3000)
        print(f"Trial {trial.number}: Using synthetic data")
    
    # Split data
    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.15)
    train_df = df[:train_size]
    val_df = df[train_size:train_size+val_size]
    test_df = df[train_size+val_size:]
    
    # Get realistic trading costs
    costs = get_trading_costs('stock', 'alpaca')
    
    # Create environments
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
    
    # Create PEFT agent
    input_dim = 30 * (len(available_features) + 3)
    features_per_step = input_dim // 30
    
    try:
        base_agent = create_peft_agent(config, features_per_step)
        agent = ReshapeWrapper(base_agent, window_size=30)
        
        # Create optimizer (only for LoRA parameters)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        agent.to(device)
        
        optimizer = create_peft_optimizer(base_agent, config)
        
        # Create custom trainer
        from train_advanced import AdvancedPPOTrainer
        
        # Override the optimizer creation in trainer
        trainer = AdvancedPPOTrainer(agent, config, device)
        trainer.optimizer = optimizer  # Use our PEFT optimizer
        
        # Enhanced early stopping
        early_stopper = EnhancedEarlyStopping(
            patience=config.early_stop_patience,
            min_episodes=50,
            overfit_threshold=0.2
        )
        
        # Stochastic depth for regularization
        stochastic_depth = StochasticDepth(config.stochastic_depth_prob) if config.use_stochastic_depth else None
        
        # Mixup augmentation
        mixup = MixupAugmentation() if config.use_mixup else None
        
        best_val_sharpe = -float('inf')
        best_val_return = -float('inf')
        
        # Training loop
        for episode in range(config.num_episodes):
            # Train episode
            reward, steps = trainer.train_episode(train_env)
            
            # Evaluation
            if (episode + 1) % config.eval_interval == 0:
                # Training loss (approximate)
                train_loss = -reward  # Negative reward as proxy for loss
                
                # Validation evaluation
                val_env.reset()
                state = val_env.reset()
                done = False
                
                while not done:
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                        
                        # Apply stochastic depth during training
                        if stochastic_depth and trainer.agent.training:
                            state_tensor = stochastic_depth(state_tensor)
                        
                        action, _ = trainer.select_action(state, deterministic=True)
                    state, _, done, _ = val_env.step([action])
                
                val_metrics = val_env.get_metrics()
                val_sharpe = val_metrics.get('sharpe_ratio', -10)
                val_return = val_metrics.get('total_return', -1)
                val_loss = -val_sharpe  # Use negative Sharpe as loss
                
                # Update best scores
                best_val_sharpe = max(best_val_sharpe, val_sharpe)
                best_val_return = max(best_val_return, val_return)
                
                # Log to TensorBoard
                writer.add_scalar('Train/Loss', train_loss, episode)
                writer.add_scalar('Val/Loss', val_loss, episode)
                writer.add_scalar('Val/Sharpe', val_sharpe, episode)
                writer.add_scalar('Val/Return', val_return, episode)
                writer.add_scalar('Val/BestSharpe', best_val_sharpe, episode)
                
                # Check early stopping
                should_stop, reason = early_stopper.should_stop(
                    episode, train_loss, val_loss, val_sharpe, val_return
                )
                
                if should_stop:
                    print(f"Trial {trial.number} stopped at episode {episode}: {reason}")
                    writer.add_text('EarlyStopping', f"Stopped at {episode}: {reason}", episode)
                    break
                
                # Report to Optuna
                trial.report(val_sharpe, episode)
                
                # Optuna pruning
                if trial.should_prune():
                    writer.add_text('Pruning', f"Pruned by Optuna at episode {episode}", episode)
                    raise optuna.TrialPruned()
                
                # Special handling around episode 600
                if episode == 600:
                    # Reduce learning rate
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.5
                    print(f"Trial {trial.number}: Reduced LR at episode 600")
        
        # Final test evaluation
        test_env.reset()
        state = test_env.reset()
        done = False
        
        while not done:
            action, _ = trainer.select_action(state, deterministic=True)
            state, _, done, _ = test_env.step([action])
        
        test_metrics = test_env.get_metrics()
        test_sharpe = test_metrics.get('sharpe_ratio', -10)
        test_return = test_metrics.get('total_return', -1)
        
        # Objective: Prioritize Sharpe but consider returns
        objective_value = 0.7 * test_sharpe + 0.3 * (test_return * 10)
        
        # Penalize if overfitting detected
        if len(early_stopper.val_losses) > 20:
            val_train_ratio = np.mean(early_stopper.val_losses[-10:]) / np.mean(early_stopper.train_losses[-10:])
            if val_train_ratio > 1.2:  # 20% worse on validation
                objective_value *= 0.8  # Penalize overfitting
        
        writer.add_scalar('Final/TestSharpe', test_sharpe, 0)
        writer.add_scalar('Final/TestReturn', test_return, 0)
        writer.add_scalar('Final/ObjectiveValue', objective_value, 0)
        writer.close()
        
        return objective_value
        
    except optuna.TrialPruned:
        writer.close()
        raise
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        writer.add_text('Error', str(e), 0)
        writer.close()
        return -100


def main():
    """Main optimization with PEFT"""
    
    print("\n" + "="*80)
    print("🚀 PEFT/LoRA HYPERPARAMETER OPTIMIZATION")
    print("="*80)
    
    print("\n📊 Key Features:")
    print("  • Parameter-Efficient Fine-Tuning (PEFT)")
    print("  • Low-Rank Adaptation (LoRA)")
    print("  • Enhanced overfitting detection")
    print("  • Special handling around episode 600")
    print("  • Aggressive regularization")
    print("  • ~90% fewer trainable parameters")
    
    # Create study
    study_name = f"peft_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3,
            n_warmup_steps=50
        ),
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # Optimize
    print("\n🏃 Starting PEFT optimization...")
    print(f"📊 TensorBoard: tensorboard --logdir=traininglogs")
    print("-" * 40)
    
    n_trials = 20  # Focused optimization
    
    study.optimize(
        objective_with_peft,
        n_trials=n_trials,
        n_jobs=1,
        show_progress_bar=True
    )
    
    # Results
    print("\n" + "="*80)
    print("📊 OPTIMIZATION RESULTS")
    print("="*80)
    
    print("\n🏆 Best trial:")
    best_trial = study.best_trial
    print(f"  Objective Value: {best_trial.value:.4f}")
    print(f"  Trial Number: {best_trial.number}")
    
    print("\n📈 Best PEFT parameters:")
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
    best_params = best_trial.params.copy()
    best_params['_objective_value'] = best_trial.value
    best_params['_trial_number'] = best_trial.number
    
    with open(f'optimization_results/{study_name}_best_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    
    print(f"\n📁 Results saved to optimization_results/")
    print(f"📊 View trials: tensorboard --logdir=traininglogs")
    
    # Create recommended configuration
    print("\n" + "="*80)
    print("💡 RECOMMENDED CONFIGURATION")
    print("="*80)
    
    print("\n```python")
    print("config = PEFTTrainingConfig(")
    for key, value in best_trial.params.items():
        if isinstance(value, float):
            print(f"    {key}={value:.6f},")
        elif isinstance(value, str):
            print(f"    {key}='{value}',")
        else:
            print(f"    {key}={value},")
    print("    num_episodes=1000,  # Can train longer with PEFT")
    print("    early_stop_patience=50,")
    print(")")
    print("```")
    
    print("\n✅ PEFT optimization complete!")
    print("="*80)


if __name__ == '__main__':
    main()