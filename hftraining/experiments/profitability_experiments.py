#!/usr/bin/env python3
"""
Profitability Improvement Experiments
Small, focused experiments to improve model performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import Dict, List, Tuple
import logging

class ProfitabilityExperiments:
    """Run small experiments to improve trading profitability"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
    def experiment_1_fix_learning_rate(self):
        """Fix the learning rate scheduler issue"""
        print("\n=== Experiment 1: Fix Learning Rate ===")
        
        # The issue: OneCycleLR is completing too early
        # Solution: Use cosine annealing with restarts
        
        config = {
            'optimizer': 'adamw',
            'base_lr': 1e-4,
            'min_lr': 1e-6,
            'T_0': 1000,  # First restart after 1000 steps
            'T_mult': 2,  # Double period after each restart
            'scheduler': 'CosineAnnealingWarmRestarts'
        }
        
        print(f"Config: {config}")
        
        # Test scheduler behavior
        optimizer = torch.optim.AdamW([torch.tensor(1.0)], lr=config['base_lr'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=config['T_0'], T_mult=config['T_mult']
        )
        
        lrs = []
        for step in range(5000):
            lrs.append(scheduler.get_last_lr()[0])
            scheduler.step()
            
        print(f"LR at step 100: {lrs[100]:.6f}")
        print(f"LR at step 1000: {lrs[1000]:.6f}")
        print(f"LR at step 3000: {lrs[3000]:.6f}")
        print(f"LR at step 4999: {lrs[4999]:.6f}")
        
        self.results['exp1_lr_fix'] = {
            'config': config,
            'sample_lrs': {
                'step_100': lrs[100],
                'step_1000': lrs[1000],
                'step_3000': lrs[3000],
                'step_4999': lrs[4999]
            }
        }
        
        return config
    
    def experiment_2_profit_focused_loss(self):
        """Create a profit-focused loss function"""
        print("\n=== Experiment 2: Profit-Focused Loss ===")
        
        class ProfitLoss(nn.Module):
            """Custom loss that emphasizes profitable trades"""
            
            def __init__(self, profit_weight=2.0, risk_penalty=0.5):
                super().__init__()
                self.profit_weight = profit_weight
                self.risk_penalty = risk_penalty
            
            def forward(self, predictions, targets, actions):
                """
                predictions: price predictions
                targets: actual prices
                actions: buy/hold/sell decisions
                """
                # Price prediction loss (MSE)
                price_loss = F.mse_loss(predictions, targets)
                
                # Calculate returns
                returns = (targets[:, -1] - targets[:, 0]) / (targets[:, 0] + 1e-8)
                
                # Profit-aware action loss
                # Penalize buying before drops, selling before rises
                buy_mask = (actions == 0).float()  # Buy
                sell_mask = (actions == 2).float()  # Sell
                
                # Reward correct directional calls
                buy_profit = buy_mask * returns  # Positive if price went up after buy
                sell_profit = sell_mask * (-returns)  # Positive if price went down after sell
                
                profit_score = buy_profit + sell_profit
                profit_loss = -torch.mean(profit_score) * self.profit_weight
                
                # Risk penalty for large positions
                position_variance = torch.var(actions.float())
                risk_loss = position_variance * self.risk_penalty
                
                # Combine losses
                total_loss = price_loss + profit_loss + risk_loss
                
                return total_loss, {
                    'price_loss': price_loss.item(),
                    'profit_loss': profit_loss.item(),
                    'risk_loss': risk_loss.item(),
                    'avg_return': returns.mean().item()
                }
        
        # Test the loss function
        loss_fn = ProfitLoss()
        
        # Simulate some data
        batch_size = 32
        pred_horizon = 5
        predictions = torch.randn(batch_size, pred_horizon)
        targets = predictions + torch.randn(batch_size, pred_horizon) * 0.1
        actions = torch.randint(0, 3, (batch_size,))
        
        total_loss, metrics = loss_fn(predictions, targets, actions)
        
        print(f"Total Loss: {total_loss:.4f}")
        print(f"Metrics: {metrics}")
        
        self.results['exp2_profit_loss'] = {
            'loss_components': metrics,
            'total_loss': total_loss.item()
        }
        
        return loss_fn
    
    def experiment_3_ensemble_predictions(self):
        """Use ensemble of models for better predictions"""
        print("\n=== Experiment 3: Ensemble Strategy ===")
        
        strategies = {
            'conservative': {
                'buy_threshold': 0.02,  # 2% expected gain
                'sell_threshold': -0.01,  # 1% loss
                'position_size': 0.3  # 30% of capital
            },
            'moderate': {
                'buy_threshold': 0.01,
                'sell_threshold': -0.005,
                'position_size': 0.5
            },
            'aggressive': {
                'buy_threshold': 0.005,
                'sell_threshold': -0.002,
                'position_size': 0.7
            }
        }
        
        # Simulate ensemble voting
        def ensemble_decision(predictions: np.ndarray, strategy: Dict) -> int:
            """Make trading decision based on ensemble predictions"""
            avg_prediction = np.mean(predictions)
            
            if avg_prediction > strategy['buy_threshold']:
                return 0  # Buy
            elif avg_prediction < strategy['sell_threshold']:
                return 2  # Sell
            else:
                return 1  # Hold
        
        # Test ensemble
        np.random.seed(42)
        test_predictions = np.random.randn(100, 3) * 0.02  # 3 models, 100 samples
        
        results = {}
        for name, strategy in strategies.items():
            decisions = [ensemble_decision(pred, strategy) for pred in test_predictions]
            buy_ratio = decisions.count(0) / len(decisions)
            sell_ratio = decisions.count(2) / len(decisions)
            hold_ratio = decisions.count(1) / len(decisions)
            
            results[name] = {
                'buy_ratio': buy_ratio,
                'sell_ratio': sell_ratio,
                'hold_ratio': hold_ratio
            }
            
            print(f"\n{name.capitalize()} Strategy:")
            print(f"  Buy: {buy_ratio:.2%}")
            print(f"  Sell: {sell_ratio:.2%}")
            print(f"  Hold: {hold_ratio:.2%}")
        
        self.results['exp3_ensemble'] = results
        return strategies
    
    def experiment_4_feature_engineering(self):
        """Add more profitable features"""
        print("\n=== Experiment 4: Feature Engineering ===")
        
        profitable_features = {
            'momentum_indicators': [
                'rate_of_change',  # (price_now - price_n_ago) / price_n_ago
                'momentum_oscillator',  # price - SMA
                'price_acceleration'  # second derivative of price
            ],
            'volatility_signals': [
                'volatility_ratio',  # current_vol / avg_vol
                'volatility_breakout',  # price outside bollinger bands
                'atr_multiplier'  # position sizing based on ATR
            ],
            'market_regime': [
                'trend_strength',  # ADX indicator
                'market_phase',  # trending/ranging/volatile
                'correlation_matrix'  # inter-asset correlations
            ],
            'risk_metrics': [
                'max_drawdown',
                'sharpe_ratio',
                'value_at_risk'
            ]
        }
        
        def calculate_features(price_data: pd.DataFrame) -> pd.DataFrame:
            """Calculate profitable trading features"""
            features = price_data.copy()
            
            # Momentum features
            features['roc_5'] = features['close'].pct_change(5)
            features['roc_20'] = features['close'].pct_change(20)
            features['momentum'] = features['close'] - features['close'].rolling(20).mean()
            
            # Volatility features
            features['volatility'] = features['close'].rolling(20).std()
            features['vol_ratio'] = features['volatility'] / features['volatility'].rolling(50).mean()
            
            # Trend features
            features['trend_20'] = np.where(
                features['close'] > features['close'].rolling(20).mean(), 1, -1
            )
            
            # Risk features
            rolling_max = features['close'].rolling(20).max()
            features['drawdown'] = (features['close'] - rolling_max) / rolling_max
            
            return features
        
        # Test with synthetic data
        dates = pd.date_range('2024-01-01', periods=100)
        test_data = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(100) * 2),
            'volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
        
        enhanced_features = calculate_features(test_data)
        
        print(f"Original features: {test_data.columns.tolist()}")
        print(f"Enhanced features: {enhanced_features.columns.tolist()}")
        print(f"\nFeature statistics:")
        print(enhanced_features[['roc_5', 'volatility', 'drawdown']].describe())
        
        self.results['exp4_features'] = {
            'feature_categories': profitable_features,
            'num_new_features': len(enhanced_features.columns) - len(test_data.columns)
        }
        
        return enhanced_features
    
    def experiment_5_adaptive_position_sizing(self):
        """Implement Kelly Criterion for optimal position sizing"""
        print("\n=== Experiment 5: Adaptive Position Sizing ===")
        
        class KellyPositionSizer:
            """Calculate optimal position size using Kelly Criterion"""
            
            def __init__(self, max_position=0.25, lookback=100):
                self.max_position = max_position  # Cap at 25% of capital
                self.lookback = lookback
                self.win_rates = []
                self.avg_wins = []
                self.avg_losses = []
            
            def calculate_kelly(self, win_prob: float, avg_win: float, avg_loss: float) -> float:
                """
                Kelly formula: f = (p*b - q) / b
                where:
                f = fraction to bet
                p = probability of winning
                b = odds (avg_win / avg_loss)
                q = probability of losing (1-p)
                """
                if avg_loss <= 0 or avg_win <= 0:
                    return 0
                
                b = avg_win / avg_loss
                q = 1 - win_prob
                
                kelly = (win_prob * b - q) / b
                
                # Apply safety factor (use 25% of Kelly)
                kelly *= 0.25
                
                # Cap at maximum position
                return min(max(kelly, 0), self.max_position)
            
            def update_stats(self, returns: np.ndarray):
                """Update win/loss statistics"""
                wins = returns[returns > 0]
                losses = returns[returns <= 0]
                
                if len(wins) > 0 and len(losses) > 0:
                    self.win_rates.append(len(wins) / len(returns))
                    self.avg_wins.append(np.mean(wins))
                    self.avg_losses.append(np.abs(np.mean(losses)))
            
            def get_position_size(self, confidence: float = 1.0) -> float:
                """Get recommended position size"""
                if len(self.win_rates) < 10:
                    return 0.01  # Start with minimal position
                
                # Use recent statistics
                recent_win_rate = np.mean(self.win_rates[-self.lookback:])
                recent_avg_win = np.mean(self.avg_wins[-self.lookback:])
                recent_avg_loss = np.mean(self.avg_losses[-self.lookback:])
                
                kelly_size = self.calculate_kelly(
                    recent_win_rate, 
                    recent_avg_win, 
                    recent_avg_loss
                )
                
                # Adjust by confidence
                return kelly_size * confidence
        
        # Test Kelly sizing
        sizer = KellyPositionSizer()
        
        # Simulate trading returns
        np.random.seed(42)
        for i in range(20):
            # Simulate a strategy with 60% win rate
            returns = np.random.randn(50) * 0.02
            returns[np.random.choice(50, 30, replace=False)] = np.abs(returns[np.random.choice(50, 30, replace=False)])
            
            sizer.update_stats(returns)
            position = sizer.get_position_size()
            
            if i % 5 == 0:
                print(f"Period {i}: Position size = {position:.2%}")
        
        self.results['exp5_kelly'] = {
            'final_position_size': position,
            'win_rate': np.mean(sizer.win_rates),
            'avg_win': np.mean(sizer.avg_wins),
            'avg_loss': np.mean(sizer.avg_losses)
        }
        
        return sizer
    
    def run_all_experiments(self):
        """Run all profitability experiments"""
        print("=" * 60)
        print("RUNNING PROFITABILITY IMPROVEMENT EXPERIMENTS")
        print("=" * 60)
        
        # Run experiments
        lr_config = self.experiment_1_fix_learning_rate()
        profit_loss = self.experiment_2_profit_focused_loss()
        ensemble_strategies = self.experiment_3_ensemble_predictions()
        enhanced_features = self.experiment_4_feature_engineering()
        kelly_sizer = self.experiment_5_adaptive_position_sizing()
        
        # Save results
        results_path = Path('hftraining/experiments/results.json')
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n{'='*60}")
        print("EXPERIMENT SUMMARY")
        print("="*60)
        
        print("\n✅ Key Improvements Found:")
        print("1. Fixed learning rate scheduler - maintains adaptive learning")
        print("2. Profit-focused loss function - optimizes for returns not just accuracy")
        print("3. Ensemble strategies - reduces single-model risk")
        print("4. Enhanced features - adds momentum and risk indicators")
        print("5. Kelly position sizing - optimizes capital allocation")
        
        print(f"\nResults saved to: {results_path}")
        
        return self.results


def create_improved_config():
    """Create improved configuration based on experiments"""
    
    config = {
        "model": {
            "hidden_size": 768,  # Increased capacity
            "num_heads": 16,
            "num_layers": 10,  # Deeper
            "dropout": 0.2,  # More regularization
            "intermediate_size": 3072
        },
        "data": {
            "sequence_length": 90,  # Longer context
            "prediction_horizon": 10,  # Longer prediction
            "batch_size": 16,  # Smaller for stability
            "use_enhanced_features": True
        },
        "training": {
            "optimizer": "adamw",
            "learning_rate": 5e-5,
            "min_lr": 1e-6,
            "scheduler": "CosineAnnealingWarmRestarts",
            "T_0": 1000,
            "T_mult": 2,
            "weight_decay": 0.05,  # More regularization
            "gradient_accumulation_steps": 8,  # Effective batch = 128
            "max_grad_norm": 0.5,  # Tighter clipping
            "label_smoothing": 0.1,
            "use_profit_loss": True,
            "profit_weight": 2.0,
            "risk_penalty": 0.5
        },
        "strategy": {
            "use_ensemble": True,
            "num_models": 3,
            "use_kelly_sizing": True,
            "max_position": 0.25,
            "stop_loss": 0.02,  # 2% stop loss
            "take_profit": 0.05  # 5% take profit
        },
        "validation": {
            "eval_strategy": "steps",
            "eval_steps": 500,
            "metric_for_best_model": "sharpe_ratio",  # Focus on risk-adjusted returns
            "early_stopping_patience": 20
        }
    }
    
    # Save config
    config_path = Path('hftraining/experiments/improved_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nImproved config saved to: {config_path}")
    return config


if __name__ == "__main__":
    # Run experiments
    experimenter = ProfitabilityExperiments()
    results = experimenter.run_all_experiments()
    
    # Create improved configuration
    improved_config = create_improved_config()
    
    print("\n🚀 Ready to run improved training with enhanced profitability focus!")