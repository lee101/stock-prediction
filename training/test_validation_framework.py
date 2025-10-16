#!/usr/bin/env python3
"""
Test-Driven Validation Framework for Stock Trading Models
Comprehensive testing suite to validate model performance and profitability.
"""

import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

sys.path.append('..')

from trading_agent import TradingAgent
from trading_env import DailyTradingEnv
from trading_config import get_trading_costs
from train_per_stock import PerStockTrainer, StockTrainingConfig

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ValidationMetrics:
    """Container for validation metrics"""
    symbol: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    final_portfolio_value: float
    volatility: float
    calmar_ratio: float


class ModelValidator:
    """Comprehensive model validation framework"""
    
    def __init__(self):
        self.training_data_dir = Path('../trainingdata')
        self.models_dir = Path('models/per_stock')
        self.validation_dir = Path('validation_results')
        self.validation_dir.mkdir(parents=True, exist_ok=True)
        
        # Trading configuration
        self.initial_balance = 10000.0
        self.window_size = 30
        self.transaction_cost = 0.001
    
    def load_model(self, symbol: str, model_type: str = 'best') -> Optional[TradingAgent]:
        """Load a trained model for validation"""
        model_file = self.models_dir / f'{symbol}_{model_type}.pth'
        
        if not model_file.exists():
            logger.warning(f"Model not found: {model_file}")
            return None
        
        try:
            # Load test data to get dimensions
            test_data = self.load_test_data(symbol)
            if test_data is None:
                return None
            
            # Create environment to get observation dimensions
            env = DailyTradingEnv(
                df=test_data,
                window_size=self.window_size,
                initial_balance=self.initial_balance,
                transaction_cost=self.transaction_cost
            )
            
            obs_dim = env.observation_space.shape
            action_dim = env.action_space.shape[0]
            
            # Create and load agent
            agent = TradingAgent(obs_dim=obs_dim, action_dim=action_dim)
            agent.load_state_dict(torch.load(model_file, map_location='cpu'))
            agent.eval()
            
            logger.info(f"Loaded model for {symbol}")
            return agent
            
        except Exception as e:
            logger.error(f"Failed to load model for {symbol}: {e}")
            return None
    
    def load_test_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load test data for a symbol"""
        test_file = self.training_data_dir / 'test' / f'{symbol}.csv'
        
        if not test_file.exists():
            logger.warning(f"Test data not found for {symbol}")
            return None
        
        try:
            df = pd.read_csv(test_file)
            
            # Standardize columns
            df.columns = [col.lower() for col in df.columns]
            
            # Ensure required columns
            required = ['open', 'high', 'low', 'close', 'volume']
            for col in required:
                if col not in df.columns:
                    if col == 'volume':
                        df[col] = 1000000
                    elif col in ['high', 'low']:
                        df[col] = df['close']
            
            # Add technical indicators (using same logic as training)
            from train_full_model import add_technical_indicators
            df = add_technical_indicators(df)
            
            # Capitalize columns
            df.columns = [col.title() for col in df.columns]
            df = df.dropna()
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load test data for {symbol}: {e}")
            return None
    
    def validate_single_model(self, symbol: str, model_type: str = 'best') -> Optional[ValidationMetrics]:
        """Validate a single model and return comprehensive metrics"""
        logger.info(f"Validating {symbol} model...")
        
        # Load model and data
        agent = self.load_model(symbol, model_type)
        test_data = self.load_test_data(symbol)
        
        if agent is None or test_data is None:
            return None
        
        # Create test environment
        env = DailyTradingEnv(
            df=test_data,
            window_size=self.window_size,
            initial_balance=self.initial_balance,
            transaction_cost=self.transaction_cost
        )
        
        # Run validation episode
        obs, _ = env.reset()
        done = False
        
        portfolio_values = [self.initial_balance]
        actions_taken = []
        rewards = []
        positions = []
        
        while not done:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action, _, _ = agent(obs_tensor)
                action = action.cpu().numpy().flatten()
            
            obs, reward, done, truncated, info = env.step(action)
            
            portfolio_values.append(info['portfolio_value'])
            actions_taken.append(action[0])
            rewards.append(reward)
            positions.append(info.get('position', 0))
            
            done = done or truncated
        
        # Calculate comprehensive metrics
        metrics = self.calculate_metrics(
            symbol=symbol,
            portfolio_values=portfolio_values,
            actions=actions_taken,
            positions=positions,
            initial_balance=self.initial_balance
        )
        
        # Save detailed results
        self.save_validation_details(symbol, metrics, portfolio_values, actions_taken, positions)
        
        return metrics
    
    def calculate_metrics(self, symbol: str, portfolio_values: List[float], 
                         actions: List[float], positions: List[float], 
                         initial_balance: float) -> ValidationMetrics:
        """Calculate comprehensive trading metrics"""
        
        portfolio_values = np.array(portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Basic metrics
        total_return = (portfolio_values[-1] - initial_balance) / initial_balance
        final_portfolio_value = portfolio_values[-1]
        
        # Risk metrics
        volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        max_drawdown = self.calculate_max_drawdown(portfolio_values)
        calmar_ratio = total_return / (abs(max_drawdown) + 1e-8)
        
        # Trading metrics
        win_rate, profit_factor, total_trades = self.calculate_trading_metrics(
            portfolio_values, actions, positions
        )
        
        return ValidationMetrics(
            symbol=symbol,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            final_portfolio_value=final_portfolio_value,
            volatility=volatility,
            calmar_ratio=calmar_ratio
        )
    
    def calculate_max_drawdown(self, portfolio_values: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        return float(np.min(drawdown))
    
    def calculate_trading_metrics(self, portfolio_values: np.ndarray, 
                                actions: List[float], positions: List[float]) -> Tuple[float, float, int]:
        """Calculate trading-specific metrics"""
        
        # Identify trades (position changes)
        position_changes = np.diff(np.array([0] + positions))
        trades = np.where(np.abs(position_changes) > 0.01)[0]  # Significant position changes
        
        if len(trades) == 0:
            return 0.0, 1.0, 0
        
        # Calculate trade returns
        trade_returns = []
        for i in range(len(trades) - 1):
            start_idx = trades[i]
            end_idx = trades[i + 1]
            if start_idx < len(portfolio_values) - 1 and end_idx < len(portfolio_values):
                trade_return = (portfolio_values[end_idx] - portfolio_values[start_idx]) / portfolio_values[start_idx]
                trade_returns.append(trade_return)
        
        if not trade_returns:
            return 0.0, 1.0, 0
        
        # Win rate
        winning_trades = [r for r in trade_returns if r > 0]
        losing_trades = [r for r in trade_returns if r < 0]
        win_rate = len(winning_trades) / len(trade_returns) if trade_returns else 0
        
        # Profit factor
        gross_profit = sum(winning_trades) if winning_trades else 0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 1e-8
        profit_factor = gross_profit / gross_loss
        
        return win_rate, profit_factor, len(trade_returns)
    
    def save_validation_details(self, symbol: str, metrics: ValidationMetrics, 
                              portfolio_values: List[float], actions: List[float], 
                              positions: List[float]):
        """Save detailed validation results"""
        
        # Create results dictionary
        results = {
            'symbol': symbol,
            'metrics': {
                'total_return': metrics.total_return,
                'sharpe_ratio': metrics.sharpe_ratio,
                'max_drawdown': metrics.max_drawdown,
                'win_rate': metrics.win_rate,
                'profit_factor': metrics.profit_factor,
                'total_trades': metrics.total_trades,
                'final_portfolio_value': metrics.final_portfolio_value,
                'volatility': metrics.volatility,
                'calmar_ratio': metrics.calmar_ratio
            },
            'time_series': {
                'portfolio_values': portfolio_values,
                'actions': actions,
                'positions': positions
            },
            'validation_date': datetime.now().isoformat()
        }
        
        # Save to file
        results_file = self.validation_dir / f'{symbol}_validation.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create visualization
        self.create_validation_plots(symbol, portfolio_values, actions, positions)
    
    def create_validation_plots(self, symbol: str, portfolio_values: List[float], 
                               actions: List[float], positions: List[float]):
        """Create validation visualization plots"""
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Portfolio value over time
        axes[0].plot(portfolio_values, label='Portfolio Value', linewidth=2)
        axes[0].axhline(y=self.initial_balance, color='r', linestyle='--', alpha=0.7, label='Initial Balance')
        axes[0].set_title(f'{symbol} - Portfolio Performance')
        axes[0].set_ylabel('Portfolio Value ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Actions over time
        axes[1].plot(actions, label='Actions', alpha=0.7)
        axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.5)
        axes[1].set_title('Trading Actions')
        axes[1].set_ylabel('Action Value')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Positions over time
        axes[2].plot(positions, label='Position', alpha=0.7)
        axes[2].axhline(y=0, color='k', linestyle='-', alpha=0.5)
        axes[2].set_title('Position Size')
        axes[2].set_ylabel('Position')
        axes[2].set_xlabel('Time Steps')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.validation_dir / f'{symbol}_validation.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def validate_all_models(self, symbols: Optional[List[str]] = None) -> Dict:
        """Validate all available models"""
        
        if symbols is None:
            # Get all available models
            model_files = list(self.models_dir.glob('*_best.pth'))
            symbols = [f.stem.replace('_best', '') for f in model_files]
        
        logger.info(f"Validating {len(symbols)} models...")
        
        validation_results = []
        for symbol in symbols:
            metrics = self.validate_single_model(symbol)
            if metrics:
                validation_results.append(metrics)
        
        # Create summary report
        summary = self.create_summary_report(validation_results)
        
        return {
            'validation_timestamp': datetime.now().isoformat(),
            'total_models': len(symbols),
            'successful_validations': len(validation_results),
            'summary': summary,
            'detailed_results': [vars(m) for m in validation_results]
        }
    
    def create_summary_report(self, results: List[ValidationMetrics]) -> Dict:
        """Create summary validation report"""
        
        if not results:
            return {}
        
        # Calculate aggregate metrics
        total_returns = [r.total_return for r in results]
        sharpe_ratios = [r.sharpe_ratio for r in results if not np.isnan(r.sharpe_ratio)]
        max_drawdowns = [r.max_drawdown for r in results]
        win_rates = [r.win_rate for r in results]
        
        # Profitable models
        profitable_models = [r for r in results if r.total_return > 0]
        high_sharpe_models = [r for r in results if r.sharpe_ratio > 1.0]
        
        summary = {
            'total_models_validated': len(results),
            'profitable_models': len(profitable_models),
            'high_sharpe_models': len(high_sharpe_models),
            'avg_return': np.mean(total_returns),
            'median_return': np.median(total_returns),
            'std_return': np.std(total_returns),
            'avg_sharpe_ratio': np.mean(sharpe_ratios) if sharpe_ratios else 0,
            'avg_max_drawdown': np.mean(max_drawdowns),
            'best_performing_model': max(results, key=lambda x: x.total_return).symbol,
            'best_sharpe_model': max(results, key=lambda x: x.sharpe_ratio).symbol if sharpe_ratios else None,
            'profitability_rate': len(profitable_models) / len(results)
        }
        
        # Save summary
        summary_file = self.validation_dir / 'validation_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        logger.info("📊 Validation Summary:")
        logger.info(f"  Models validated: {summary['total_models_validated']}")
        logger.info(f"  Profitable models: {summary['profitable_models']}")
        logger.info(f"  Profitability rate: {summary['profitability_rate']:.1%}")
        logger.info(f"  Average return: {summary['avg_return']:.2%}")
        logger.info(f"  Best performing: {summary['best_performing_model']}")
        
        return summary


def main():
    parser = argparse.ArgumentParser(description='Validate trained trading models')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to validate')
    parser.add_argument('--model_type', default='best', help='Model type to validate')
    
    args = parser.parse_args()
    
    # Create validator
    validator = ModelValidator()
    
    # Run validation
    results = validator.validate_all_models(symbols=args.symbols)
    
    logger.info(f"🎉 Validation completed! Results saved to {validator.validation_dir}")


if __name__ == "__main__":
    main()