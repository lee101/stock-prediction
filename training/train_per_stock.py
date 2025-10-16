#!/usr/bin/env python3
"""
Per-Stock Training System with Test-Driven Validation
Trains separate models for each stock pair and validates on unseen test data.
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
from tqdm import tqdm
import multiprocessing as mp
from typing import Dict, List, Tuple, Optional
import logging

sys.path.append('..')

from trading_agent import TradingAgent
from trading_env import DailyTradingEnv
from ppo_trainer import PPOTrainer
from trading_config import get_trading_costs
from train_full_model import add_technical_indicators

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StockTrainingConfig:
    """Configuration for per-stock training"""
    def __init__(self):
        self.episodes = 1000
        self.window_size = 30
        self.initial_balance = 10000.0
        self.transaction_cost = 0.001
        self.learning_rate = 3e-4
        self.batch_size = 64
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_ratio = 0.2
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        self.max_grad_norm = 0.5
        self.ppo_epochs = 10
        self.save_interval = 100
        self.validation_interval = 50


class PerStockTrainer:
    """Trains and validates models for individual stock pairs"""
    
    def __init__(self, config: StockTrainingConfig):
        self.config = config
        self.training_data_dir = Path('../trainingdata')
        self.models_dir = Path('models/per_stock')
        self.results_dir = Path('results/per_stock')
        self.logs_dir = Path('traininglogs/per_stock')
        
        # Create directories
        for dir_path in [self.models_dir, self.results_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def load_stock_data(self, symbol: str, split: str = 'train') -> pd.DataFrame:
        """Load training or test data for a specific stock"""
        data_file = self.training_data_dir / split / f'{symbol}.csv'
        if not data_file.exists():
            raise FileNotFoundError(f"No {split} data found for {symbol}")
        
        df = pd.read_csv(data_file)
        
        # Standardize column names
        df.columns = [col.lower() for col in df.columns]
        
        # Ensure required columns exist
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                if 'adj close' in df.columns and col == 'close':
                    df[col] = df['adj close']
                elif col == 'volume' and col not in df.columns:
                    df[col] = 1000000  # Default volume
                elif col in ['high', 'low'] and col not in df.columns:
                    df[col] = df['close']
        
        # Add date column if missing
        if 'date' not in df.columns:
            df['date'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
        
        # Add technical indicators
        df = add_technical_indicators(df)
        
        # Capitalize columns
        df.columns = [col.title() for col in df.columns]
        
        # Remove NaN values
        df = df.dropna()
        
        logger.info(f"Loaded {len(df)} rows of {split} data for {symbol}")
        return df
    
    def train_single_stock(self, symbol: str) -> Dict:
        """Train a model for a single stock and return results"""
        logger.info(f"🚀 Starting training for {symbol}")
        
        try:
            # Load training data
            train_df = self.load_stock_data(symbol, 'train')
            
            # Create environment
            env = DailyTradingEnv(
                df=train_df,
                window_size=self.config.window_size,
                initial_balance=self.config.initial_balance,
                transaction_cost=self.config.transaction_cost
            )
            
            # Create agent
            obs_dim = env.observation_space.shape
            action_dim = env.action_space.shape[0]
            
            agent = TradingAgent(
                obs_dim=obs_dim,
                action_dim=action_dim,
                lr=self.config.learning_rate
            )
            
            # Create trainer
            trainer = PPOTrainer(
                agent=agent,
                env=env,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
                clip_ratio=self.config.clip_ratio,
                entropy_coef=self.config.entropy_coef,
                value_coef=self.config.value_coef,
                max_grad_norm=self.config.max_grad_norm,
                ppo_epochs=self.config.ppo_epochs,
                batch_size=self.config.batch_size
            )
            
            # Training metrics
            training_rewards = []
            validation_results = []
            best_validation_return = -float('inf')
            
            # Training loop
            for episode in tqdm(range(self.config.episodes), desc=f"Training {symbol}"):
                reward = trainer.train_episode()
                training_rewards.append(reward)
                
                # Validation check
                if episode % self.config.validation_interval == 0 and episode > 0:
                    val_result = self.validate_model(agent, symbol)
                    validation_results.append({
                        'episode': episode,
                        'validation_return': val_result['total_return'],
                        'sharpe_ratio': val_result['sharpe_ratio'],
                        'max_drawdown': val_result['max_drawdown']
                    })
                    
                    # Save best model
                    if val_result['total_return'] > best_validation_return:
                        best_validation_return = val_result['total_return']
                        model_path = self.models_dir / f'{symbol}_best.pth'
                        torch.save(agent.state_dict(), model_path)
                        logger.info(f"New best model for {symbol}: {best_validation_return:.2%}")
                
                # Regular save
                if episode % self.config.save_interval == 0 and episode > 0:
                    model_path = self.models_dir / f'{symbol}_ep{episode}.pth'
                    torch.save(agent.state_dict(), model_path)
            
            # Final validation
            final_validation = self.validate_model(agent, symbol)
            
            # Compile results
            results = {
                'symbol': symbol,
                'training_episodes': self.config.episodes,
                'final_training_reward': np.mean(training_rewards[-100:]) if training_rewards else 0,
                'best_validation_return': best_validation_return,
                'final_validation': final_validation,
                'validation_history': validation_results,
                'training_rewards': training_rewards
            }
            
            # Save results
            results_file = self.results_dir / f'{symbol}_results.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"✅ Completed training for {symbol}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to train {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}
    
    def validate_model(self, agent: TradingAgent, symbol: str) -> Dict:
        """Validate model on test data"""
        try:
            # Load test data
            test_df = self.load_stock_data(symbol, 'test')
            
            # Create test environment
            test_env = DailyTradingEnv(
                df=test_df,
                window_size=self.config.window_size,
                initial_balance=self.config.initial_balance,
                transaction_cost=self.config.transaction_cost
            )
            
            # Run validation episode
            agent.eval()
            obs, _ = test_env.reset()
            done = False
            total_reward = 0
            portfolio_values = []
            
            while not done:
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    action, _, _ = agent(obs_tensor)
                    action = action.cpu().numpy().flatten()
                
                obs, reward, done, truncated, info = test_env.step(action)
                total_reward += reward
                portfolio_values.append(info['portfolio_value'])
                done = done or truncated
            
            # Calculate metrics
            portfolio_values = np.array(portfolio_values)
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            
            total_return = (portfolio_values[-1] - self.config.initial_balance) / self.config.initial_balance
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
            max_drawdown = self.calculate_max_drawdown(portfolio_values)
            
            agent.train()
            
            return {
                'total_return': total_return,
                'final_portfolio_value': portfolio_values[-1],
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'total_reward': total_reward,
                'num_days': len(portfolio_values)
            }
            
        except Exception as e:
            logger.error(f"Validation failed for {symbol}: {e}")
            return {'error': str(e)}
    
    def calculate_max_drawdown(self, portfolio_values: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        return float(np.min(drawdown))
    
    def train_all_stocks(self, symbols: Optional[List[str]] = None, parallel: bool = True) -> Dict:
        """Train models for all available stocks"""
        
        if symbols is None:
            # Get all available symbols
            train_dir = self.training_data_dir / 'train'
            symbols = [f.stem for f in train_dir.glob('*.csv')]
        
        logger.info(f"Training models for {len(symbols)} stocks: {symbols}")
        
        if parallel and len(symbols) > 1:
            # Parallel training
            with mp.Pool(processes=min(len(symbols), mp.cpu_count())) as pool:
                results = pool.map(self.train_single_stock, symbols)
        else:
            # Sequential training
            results = [self.train_single_stock(symbol) for symbol in symbols]
        
        # Compile overall results
        successful_results = [r for r in results if 'error' not in r]
        failed_results = [r for r in results if 'error' in r]
        
        overall_results = {
            'timestamp': datetime.now().isoformat(),
            'total_symbols': len(symbols),
            'successful_trainings': len(successful_results),
            'failed_trainings': len(failed_results),
            'results': results,
            'config': vars(self.config)
        }
        
        # Save overall results
        overall_file = self.results_dir / f'overall_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(overall_file, 'w') as f:
            json.dump(overall_results, f, indent=2)
        
        # Generate summary report
        self.generate_summary_report(overall_results)
        
        return overall_results
    
    def generate_summary_report(self, results: Dict):
        """Generate a summary report of all training results"""
        successful = [r for r in results['results'] if 'error' not in r]
        
        if not successful:
            logger.warning("No successful trainings to report")
            return
        
        # Extract metrics
        validation_returns = [r['best_validation_return'] for r in successful if r['best_validation_return'] != -float('inf')]
        final_validations = [r['final_validation'] for r in successful if 'final_validation' in r and 'error' not in r['final_validation']]
        
        # Create summary
        summary = {
            'successful_symbols': len(successful),
            'avg_validation_return': np.mean(validation_returns) if validation_returns else 0,
            'std_validation_return': np.std(validation_returns) if validation_returns else 0,
            'best_performing_symbol': max(successful, key=lambda x: x.get('best_validation_return', -float('inf')))['symbol'] if successful else None,
            'profitable_models': len([r for r in validation_returns if r > 0]),
            'avg_sharpe_ratio': np.mean([v['sharpe_ratio'] for v in final_validations if 'sharpe_ratio' in v]) if final_validations else 0
        }
        
        # Save summary
        summary_file = self.results_dir / 'training_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        logger.info("📊 Training Summary:")
        logger.info(f"  Successful models: {summary['successful_symbols']}")
        logger.info(f"  Average validation return: {summary['avg_validation_return']:.2%}")
        logger.info(f"  Profitable models: {summary['profitable_models']}")
        logger.info(f"  Best performing: {summary['best_performing_symbol']}")


def main():
    parser = argparse.ArgumentParser(description='Train per-stock trading models')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to train')
    parser.add_argument('--episodes', type=int, default=1000, help='Training episodes')
    parser.add_argument('--parallel', action='store_true', help='Enable parallel training')
    parser.add_argument('--config', help='Config file path')
    
    args = parser.parse_args()
    
    # Create config
    config = StockTrainingConfig()
    if args.episodes:
        config.episodes = args.episodes
    
    # Create trainer
    trainer = PerStockTrainer(config)
    
    # Run training
    results = trainer.train_all_stocks(
        symbols=args.symbols,
        parallel=args.parallel
    )
    
    logger.info(f"🎉 Training completed! Results saved to {trainer.results_dir}")


if __name__ == "__main__":
    main()