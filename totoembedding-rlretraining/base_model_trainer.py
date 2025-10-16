#!/usr/bin/env python3
"""
Base Model Trainer - Foundation model approach for universal trading patterns
Train once on all assets, then fine-tune for specific strategies
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random

from hf_rl_trainer import HFRLConfig, TotoTransformerRL, PPOTrainer
from multi_asset_env import MultiAssetTradingEnv
from launch_hf_training import HFRLLauncher

# Import for cross-validation
from sklearn.model_selection import KFold


@dataclass
class BaseModelConfig:
    """Configuration for base model training"""
    
    # Base model parameters
    name: str = "universal_base_model"
    description: str = "Foundation model for all trading patterns"
    
    # Training strategy
    validation_split: float = 0.2
    cross_validation_folds: int = 5
    generalization_test: bool = True
    
    # Data augmentation
    time_shift: bool = True
    noise_injection: float = 0.01
    market_regime_mixing: bool = True
    
    # Profit tracking
    profit_tracking_enabled: bool = True
    profit_log_interval: int = 500
    
    # Fine-tuning
    fine_tune_enabled: bool = True
    freeze_base_layers: int = 6
    task_specific_heads: bool = True


class ProfitTracker:
    """Track trading profit during training"""
    
    def __init__(
        self,
        initial_capital: float = 100000,
        commission: float = 0.001,
        slippage: float = 0.0005,
        max_position_size: float = 0.5,
        stop_loss: float = 0.02,
        take_profit: float = 0.05
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.max_position_size = max_position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        
        self.reset()
    
    def reset(self):
        """Reset profit tracking"""
        self.current_capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.daily_returns = []
        self.peak_capital = self.initial_capital
        self.max_drawdown = 0.0
        
    def simulate_trade(self, symbol: str, action: float, price: float, prediction: float):
        """Simulate a trade based on model prediction"""
        
        # Convert action to position size
        position_size = np.clip(action, -self.max_position_size, self.max_position_size)
        
        if abs(position_size) < 0.01:  # Too small, skip
            return
        
        # Calculate trade value
        trade_value = abs(position_size) * self.current_capital
        
        # Apply costs
        costs = trade_value * (self.commission + self.slippage)
        
        # Record trade
        trade = {
            'symbol': symbol,
            'position_size': position_size,
            'price': price,
            'value': trade_value,
            'costs': costs,
            'prediction': prediction,
            'timestamp': datetime.now()
        }
        
        self.trades.append(trade)
        self.current_capital -= costs
        
        # Update positions
        if symbol not in self.positions:
            self.positions[symbol] = {'size': 0, 'entry_price': 0}
        
        # Close existing position if direction changed
        if (self.positions[symbol]['size'] > 0 and position_size < 0) or \
           (self.positions[symbol]['size'] < 0 and position_size > 0):
            self._close_position(symbol, price)
        
        # Open new position
        self.positions[symbol] = {
            'size': position_size,
            'entry_price': price
        }
    
    def _close_position(self, symbol: str, exit_price: float):
        """Close a position and realize P&L"""
        if symbol not in self.positions or self.positions[symbol]['size'] == 0:
            return
        
        position = self.positions[symbol]
        entry_price = position['entry_price']
        size = position['size']
        
        # Calculate P&L
        if size > 0:  # Long position
            pnl = (exit_price - entry_price) / entry_price * size * self.current_capital
        else:  # Short position
            pnl = (entry_price - exit_price) / entry_price * abs(size) * self.current_capital
        
        self.current_capital += pnl
        self.positions[symbol] = {'size': 0, 'entry_price': 0}
    
    def update_capital(self, price_changes: Dict[str, float]):
        """Update capital based on price changes"""
        total_pnl = 0
        
        for symbol, position in self.positions.items():
            if position['size'] != 0 and symbol in price_changes:
                price_change = price_changes[symbol]
                if position['size'] > 0:  # Long
                    pnl = price_change * position['size'] * self.current_capital
                else:  # Short
                    pnl = -price_change * abs(position['size']) * self.current_capital
                total_pnl += pnl
        
        self.current_capital += total_pnl
        
        # Update drawdown
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Record daily return
        daily_return = total_pnl / self.initial_capital
        self.daily_returns.append(daily_return)
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current profit metrics"""
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital
        
        if len(self.daily_returns) > 20:
            returns_array = np.array(self.daily_returns)
            sharpe = np.mean(returns_array) / (np.std(returns_array) + 1e-8) * np.sqrt(252)
            volatility = np.std(returns_array) * np.sqrt(252)
        else:
            sharpe = 0
            volatility = 0
        
        winning_trades = sum(1 for t in self.trades if t.get('profit', 0) > 0)
        win_rate = winning_trades / len(self.trades) if self.trades else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': self.max_drawdown,
            'volatility': volatility,
            'win_rate': win_rate,
            'num_trades': len(self.trades),
            'current_capital': self.current_capital
        }


class BaseModelTrainer:
    """
    Trainer for universal base model that learns general trading patterns
    """
    
    def __init__(self, config_path: str = "config/base_model_config.json"):
        # Load configuration
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Store config dict and create HFRLConfig
        self.config_dict = config_dict
        self.config = self._dict_to_config(config_dict)
        self.base_config = BaseModelConfig()
        
        # Setup profit tracking
        self.profit_tracker = ProfitTracker(**config_dict.get('evaluation', {}).get('profit_tracking', {}))
        
        # Setup paths
        self.output_dir = Path(config_dict['output']['output_dir'])
        self.logging_dir = Path(config_dict['output']['logging_dir'])
        self.checkpoint_dir = Path(config_dict['output']['checkpoint_dir'])
        
        for path in [self.output_dir, self.logging_dir, self.checkpoint_dir]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.best_model_path = None
        self.training_metrics = []
        self.validation_metrics = []
        
        print(f"BaseModelTrainer initialized")
        print(f"Output directory: {self.output_dir}")
        print(f"Training on {len(config_dict['data']['symbols'])} symbols")
    
    def _dict_to_config(self, config_dict: Dict) -> HFRLConfig:
        """Convert dictionary to HFRLConfig"""
        config = HFRLConfig()
        
        # Update config with dictionary values
        for section, values in config_dict.items():
            if hasattr(config, section):
                if isinstance(values, dict):
                    for key, value in values.items():
                        if hasattr(getattr(config, section), key):
                            setattr(getattr(config, section), key, value)
                        else:
                            setattr(config, key, value)
                else:
                    setattr(config, section, values)
            else:
                # Try to set individual attributes
                if isinstance(values, dict):
                    for key, value in values.items():
                        if hasattr(config, key):
                            setattr(config, key, value)
        
        return config
    
    def create_cross_validation_splits(self) -> List[Dict[str, List[str]]]:
        """Create cross-validation splits across assets"""
        symbols = self.config_dict['data']['symbols'].copy()
        random.shuffle(symbols)
        
        kfold = KFold(n_splits=self.base_config.cross_validation_folds, shuffle=True)
        splits = []
        
        for train_idx, val_idx in kfold.split(symbols):
            train_symbols = [symbols[i] for i in train_idx]
            val_symbols = [symbols[i] for i in val_idx]
            
            splits.append({
                'train': train_symbols,
                'val': val_symbols
            })
        
        return splits
    
    def train_base_model(self) -> str:
        """Train the universal base model"""
        print("\n" + "="*60)
        print("TRAINING UNIVERSAL BASE MODEL")
        print("="*60)
        
        if self.base_config.generalization_test:
            return self._train_with_cross_validation()
        else:
            return self._train_single_model()
    
    def _train_with_cross_validation(self) -> str:
        """Train with cross-validation for generalization"""
        splits = self.create_cross_validation_splits()
        fold_results = []
        
        for fold, split in enumerate(splits):
            print(f"\n--- Cross-Validation Fold {fold + 1}/{len(splits)} ---")
            print(f"Training symbols: {split['train'][:5]}... ({len(split['train'])} total)")
            print(f"Validation symbols: {split['val']}")
            
            # Create environments for this fold
            train_env = MultiAssetTradingEnv(
                data_dir=self.config_dict['data']['train_dir'],
                symbols=split['train'],
                **self.config_dict['environment']
            )
            
            val_env = MultiAssetTradingEnv(
                data_dir=self.config_dict['data']['test_dir'],
                symbols=split['val'],
                **self.config_dict['environment']
            )
            
            # Create model
            obs_dim = train_env.observation_space.shape[0]
            action_dim = train_env.action_space.shape[0]
            model = TotoTransformerRL(self.config, obs_dim, action_dim)
            
            # Create trainer
            trainer = PPOTrainer(
                config=self.config,
                model=model,
                env=train_env,
                eval_env=val_env
            )
            
            # Add profit tracking
            self._add_profit_tracking(trainer)
            
            # Train this fold
            fold_metrics = trainer.train()
            fold_results.append(fold_metrics)
            
            # Save fold model
            fold_path = self.checkpoint_dir / f"fold_{fold}_model.pth"
            trainer.save_model(str(fold_path))
            
            print(f"Fold {fold + 1} completed. Model saved to {fold_path}")
        
        # Select best fold and ensemble
        best_fold = self._select_best_fold(fold_results)
        ensemble_path = self._create_ensemble_model(splits, best_fold)
        
        return ensemble_path
    
    def _train_single_model(self) -> str:
        """Train single model on all data"""
        print("Training single base model on all assets...")
        
        # Create environments
        train_env = MultiAssetTradingEnv(
            data_dir=self.config_dict['data']['train_dir'],
            symbols=self.config_dict['data']['symbols'],
            **self.config_dict['environment']
        )
        
        val_env = MultiAssetTradingEnv(
            data_dir=self.config_dict['data']['test_dir'],
            symbols=self.config_dict['data']['symbols'],
            **self.config_dict['environment']
        )
        
        # Create model
        obs_dim = train_env.observation_space.shape[0]
        action_dim = train_env.action_space.shape[0]
        model = TotoTransformerRL(self.config, obs_dim, action_dim)
        
        # Create trainer
        trainer = PPOTrainer(
            config=self.config,
            model=model,
            env=train_env,
            eval_env=val_env
        )
        
        # Add profit tracking
        self._add_profit_tracking(trainer)
        
        # Train
        final_metrics = trainer.train()
        
        # Save base model
        base_path = self.output_dir / "base_model.pth"
        trainer.save_model(str(base_path))
        
        self.best_model_path = str(base_path)
        return str(base_path)
    
    def _add_profit_tracking(self, trainer: PPOTrainer):
        """Add profit tracking to trainer"""
        if not self.base_config.profit_tracking_enabled:
            return
        
        original_train_epoch = trainer.train_epoch
        
        def train_epoch_with_profit():
            # Original training
            original_train_epoch()
            
            # Profit tracking every N steps
            if trainer.global_step % self.base_config.profit_log_interval == 0:
                self._log_profit_metrics(trainer)
        
        trainer.train_epoch = train_epoch_with_profit
    
    def _log_profit_metrics(self, trainer: PPOTrainer):
        """Log profit metrics during training"""
        try:
            # Simulate trading with current model
            obs = trainer.env.reset()
            for _ in range(100):  # Simulate 100 steps
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(trainer.device)
                    outputs = trainer.model(obs_tensor)
                    action = outputs['actions'].cpu().numpy()[0]
                
                next_obs, reward, done, info = trainer.env.step(action)
                
                # Track profit
                if 'current_price' in info:
                    # Simplified profit tracking
                    price_change = reward  # Assuming reward correlates with profit
                    self.profit_tracker.update_capital({'current': price_change})
                
                obs = next_obs
                if done:
                    obs = trainer.env.reset()
            
            # Log metrics
            metrics = self.profit_tracker.get_metrics()
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    trainer.writer.add_scalar(f'Profit/{key}', value, trainer.global_step)
            
            # Console logging
            if trainer.global_step % (self.base_config.profit_log_interval * 2) == 0:
                print(f"\n--- Profit Metrics (Step {trainer.global_step}) ---")
                print(f"Total Return: {metrics['total_return']:.2%}")
                print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
                print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
                print(f"Win Rate: {metrics['win_rate']:.2%}")
                print(f"Current Capital: ${metrics['current_capital']:,.2f}")
        
        except Exception as e:
            print(f"Error in profit tracking: {e}")
    
    def _select_best_fold(self, fold_results: List[Dict]) -> int:
        """Select best performing fold"""
        best_fold = 0
        best_score = -np.inf
        
        for i, metrics in enumerate(fold_results):
            # Combine multiple metrics for scoring
            score = (
                metrics.get('eval_return', 0) * 0.4 +
                metrics.get('eval_sharpe', 0) * 0.3 +
                (1 - abs(metrics.get('eval_drawdown', 0))) * 0.3
            )
            
            if score > best_score:
                best_score = score
                best_fold = i
        
        print(f"Best fold: {best_fold + 1} with score: {best_score:.4f}")
        return best_fold
    
    def _create_ensemble_model(self, splits: List[Dict], best_fold: int) -> str:
        """Create ensemble model from best performers"""
        # For now, just return the best fold model
        best_model_path = self.checkpoint_dir / f"fold_{best_fold}_model.pth"
        ensemble_path = self.output_dir / "base_model_ensemble.pth"
        
        # Copy best model as ensemble (can enhance this later)
        import shutil
        shutil.copy(best_model_path, ensemble_path)
        
        self.best_model_path = str(ensemble_path)
        return str(ensemble_path)
    
    def fine_tune_for_strategy(
        self,
        base_model_path: str,
        target_symbols: List[str] = None,
        strategy_name: str = "custom",
        num_epochs: int = 50
    ) -> str:
        """Fine-tune base model for specific strategy or symbols"""
        print(f"\n--- Fine-tuning for {strategy_name} ---")
        
        if target_symbols is None:
            target_symbols = self.config_dict['data']['symbols'][:5]  # Use first 5 symbols
        
        print(f"Target symbols: {target_symbols}")
        
        # Create fine-tuning environment
        finetune_env = MultiAssetTradingEnv(
            data_dir=self.config_dict['data']['train_dir'],
            symbols=target_symbols,
            **self.config_dict['environment']
        )
        
        # Load base model
        base_checkpoint = torch.load(base_model_path, map_location='cpu', weights_only=False)
        
        obs_dim = finetune_env.observation_space.shape[0]
        action_dim = finetune_env.action_space.shape[0]
        model = TotoTransformerRL(self.config, obs_dim, action_dim)
        
        # Load base weights
        model.load_state_dict(base_checkpoint['model_state_dict'], strict=False)
        
        # Freeze base layers if specified
        if self.base_config.freeze_base_layers > 0:
            self._freeze_base_layers(model, self.base_config.freeze_base_layers)
        
        # Create fine-tuning config
        finetune_config = self.config
        finetune_config.num_train_epochs = num_epochs
        finetune_config.learning_rate = finetune_config.learning_rate * 0.1  # Lower LR for fine-tuning
        
        # Create trainer
        trainer = PPOTrainer(
            config=finetune_config,
            model=model,
            env=finetune_env,
            eval_env=finetune_env
        )
        
        # Fine-tune
        final_metrics = trainer.train()
        
        # Save fine-tuned model
        finetune_path = self.output_dir / f"finetuned_{strategy_name}.pth"
        trainer.save_model(str(finetune_path))
        
        print(f"Fine-tuned model saved to {finetune_path}")
        return str(finetune_path)
    
    def _freeze_base_layers(self, model: nn.Module, num_layers: int):
        """Freeze first N transformer layers"""
        print(f"Freezing first {num_layers} transformer layers")
        
        layer_count = 0
        for name, param in model.named_parameters():
            if 'transformer' in name and layer_count < num_layers:
                param.requires_grad = False
                if 'layers.' in name:
                    layer_num = int(name.split('layers.')[1].split('.')[0])
                    if layer_num >= num_layers:
                        break
            layer_count += 1
    
    def evaluate_generalization(self, model_path: str) -> Dict[str, float]:
        """Evaluate model generalization across different assets"""
        print("Evaluating model generalization...")
        
        results = {}
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Test on different asset categories
        asset_categories = {
            'tech_stocks': ['AAPL', 'GOOG', 'MSFT', 'NVDA'],
            'crypto': ['BTCUSD', 'ETHUSD', 'LTCUSD'],
            'growth_stocks': ['TSLA', 'NFLX', 'ADBE'],
            'all_assets': self.config_dict['data']['symbols']
        }
        
        for category, symbols in asset_categories.items():
            print(f"Testing on {category}: {symbols}")
            
            # Create test environment
            test_env = MultiAssetTradingEnv(
                data_dir=self.config_dict['data']['test_dir'],
                symbols=symbols,
                **self.config_dict['environment']
            )
            
            # Create model
            obs_dim = test_env.observation_space.shape[0]
            action_dim = test_env.action_space.shape[0]
            model = TotoTransformerRL(self.config, obs_dim, action_dim)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Run evaluation
            category_metrics = self._run_evaluation(model, test_env, num_episodes=10)
            results[category] = category_metrics
        
        # Save generalization results
        results_path = self.output_dir / "generalization_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results
    
    def _run_evaluation(self, model: nn.Module, env: MultiAssetTradingEnv, num_episodes: int = 10) -> Dict[str, float]:
        """Run evaluation on environment"""
        episode_returns = []
        episode_sharpes = []
        
        for episode in range(num_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    outputs = model(obs_tensor)
                    action = outputs['actions'].cpu().numpy()[0]
                
                obs, reward, done, info = env.step(action)
                episode_reward += reward
            
            episode_returns.append(episode_reward)
            
            # Get portfolio metrics
            metrics = env.get_portfolio_metrics()
            if metrics:
                episode_sharpes.append(metrics.get('sharpe_ratio', 0))
        
        return {
            'mean_return': np.mean(episode_returns),
            'std_return': np.std(episode_returns),
            'mean_sharpe': np.mean(episode_sharpes) if episode_sharpes else 0,
            'consistency': 1.0 - (np.std(episode_returns) / (abs(np.mean(episode_returns)) + 1e-8))
        }


def main():
    """Run base model training pipeline"""
    print("Starting Base Model Training Pipeline")
    
    # Initialize trainer
    trainer = BaseModelTrainer("config/base_model_config.json")
    
    # Train base model
    base_model_path = trainer.train_base_model()
    
    # Evaluate generalization
    generalization_results = trainer.evaluate_generalization(base_model_path)
    
    # Fine-tune for different strategies
    strategies = [
        {'name': 'tech_focus', 'symbols': ['AAPL', 'GOOG', 'MSFT', 'NVDA']},
        {'name': 'crypto_focus', 'symbols': ['BTCUSD', 'ETHUSD', 'LTCUSD']},
        {'name': 'balanced', 'symbols': ['AAPL', 'BTCUSD', 'TSLA', 'MSFT', 'ETHUSD']}
    ]
    
    finetuned_models = {}
    for strategy in strategies:
        model_path = trainer.fine_tune_for_strategy(
            base_model_path=base_model_path,
            target_symbols=strategy['symbols'],
            strategy_name=strategy['name']
        )
        finetuned_models[strategy['name']] = model_path
    
    print("\n" + "="*60)
    print("BASE MODEL TRAINING COMPLETED")
    print("="*60)
    print(f"Base Model: {base_model_path}")
    print("Fine-tuned Models:")
    for name, path in finetuned_models.items():
        print(f"  {name}: {path}")
    print(f"Generalization Results: {trainer.output_dir}/generalization_results.json")


if __name__ == "__main__":
    main()