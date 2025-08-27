#!/usr/bin/env python3
"""
Quick Training Monitor - Train for ~2 minutes and show profit metrics
Supports incremental checkpointing and rapid feedback on training progress.
"""

import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import time
import argparse
from typing import Dict, List, Tuple, Optional
import json

sys.path.append('..')

from trading_agent import TradingAgent
from trading_env import DailyTradingEnv
from ppo_trainer import PPOTrainer
from trading_config import get_trading_costs
from train_full_model import add_technical_indicators

class QuickTrainingMonitor:
    """Quick training monitor with profit tracking and incremental checkpointing"""
    
    def __init__(self, symbol: str, training_time_minutes: float = 2.0):
        self.symbol = symbol
        self.training_time_seconds = training_time_minutes * 60
        self.training_data_dir = Path('../trainingdata')
        self.models_dir = Path('models/per_stock')
        self.checkpoints_dir = Path('models/checkpoints')
        self.quick_results_dir = Path('quick_training_results')
        
        # Create directories
        for dir_path in [self.models_dir, self.checkpoints_dir, self.quick_results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Training config
        self.config = {
            'window_size': 30,
            'initial_balance': 10000.0,
            'transaction_cost': 0.001,
            'learning_rate': 3e-4,
            'batch_size': 64,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_ratio': 0.2,
            'entropy_coef': 0.01,
            'value_coef': 0.5,
            'max_grad_norm': 0.5,
            'ppo_epochs': 4,  # Reduced for faster iterations
        }
        
        # Metrics tracking
        self.metrics_history = []
        self.start_time = None
        self.last_checkpoint_episode = 0
    
    def load_stock_data(self, split: str = 'train') -> pd.DataFrame:
        """Load training or test data for the symbol"""
        data_file = self.training_data_dir / split / f'{self.symbol}.csv'
        if not data_file.exists():
            raise FileNotFoundError(f"No {split} data found for {self.symbol}")
        
        df = pd.read_csv(data_file)
        
        # Standardize column names
        df.columns = [col.lower() for col in df.columns]
        
        # Ensure required columns exist
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                if 'adj close' in df.columns and col == 'close':
                    df[col] = df['adj close']
                elif col == 'volume':
                    df[col] = 1000000
                elif col in ['high', 'low']:
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
        
        return df
    
    def find_latest_checkpoint(self) -> Optional[Tuple[str, int]]:
        """Find the latest checkpoint for this symbol"""
        checkpoint_pattern = f'{self.symbol}_ep*.pth'
        checkpoint_files = list(self.checkpoints_dir.glob(checkpoint_pattern))
        
        if not checkpoint_files:
            return None
        
        # Extract episode numbers and find latest
        latest_episode = 0
        latest_file = None
        
        for file_path in checkpoint_files:
            try:
                # Extract episode number from filename
                episode_str = file_path.stem.split('_ep')[1]
                episode_num = int(episode_str)
                
                if episode_num > latest_episode:
                    latest_episode = episode_num
                    latest_file = file_path
            except (IndexError, ValueError):
                continue
        
        return (str(latest_file), latest_episode) if latest_file else None
    
    def create_agent(self, train_df: pd.DataFrame) -> TradingAgent:
        """Create trading agent and load checkpoint if available"""
        # Create environment to get dimensions
        env = DailyTradingEnv(
            df=train_df,
            window_size=self.config['window_size'],
            initial_balance=self.config['initial_balance'],
            transaction_cost=self.config['transaction_cost']
        )
        
        obs_dim = env.observation_space.shape
        input_dim = np.prod(obs_dim)  # Flatten the observation space
        
        # Create a simple backbone that handles the actual input dimensions
        backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Create agent
        agent = TradingAgent(
            backbone_model=backbone,
            hidden_dim=128
        )
        
        # Try to load latest checkpoint
        checkpoint_info = self.find_latest_checkpoint()
        if checkpoint_info:
            checkpoint_file, episode_num = checkpoint_info
            try:
                agent.load_state_dict(torch.load(checkpoint_file, map_location='cpu'))
                self.last_checkpoint_episode = episode_num
                print(f"📁 Loaded checkpoint from episode {episode_num}")
            except Exception as e:
                print(f"⚠️  Failed to load checkpoint: {e}")
                self.last_checkpoint_episode = 0
        else:
            print(f"🆕 Starting fresh training for {self.symbol}")
            self.last_checkpoint_episode = 0
        
        return agent
    
    def validate_agent_quickly(self, agent: TradingAgent) -> Dict:
        """Quick validation on test data"""
        try:
            test_df = self.load_stock_data('test')
            
            test_env = DailyTradingEnv(
                df=test_df,
                window_size=self.config['window_size'],
                initial_balance=self.config['initial_balance'],
                transaction_cost=self.config['transaction_cost']
            )
            
            # Run validation episode
            agent.eval()
            obs = test_env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            done = False
            total_reward = 0
            portfolio_values = [self.config['initial_balance']]
            
            while not done:
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    action, _, _ = agent.act(obs_tensor, deterministic=True)
                    action = action.cpu().numpy().flatten()
                
                step_result = test_env.step(action)
                if len(step_result) == 4:
                    obs, reward, done, info = step_result
                    truncated = False
                else:
                    obs, reward, done, truncated, info = step_result
                
                total_reward += reward
                portfolio_values.append(info.get('portfolio_value', portfolio_values[-1]))
                done = done or truncated
            
            # Calculate metrics
            portfolio_values = np.array(portfolio_values)
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            
            total_return = (portfolio_values[-1] - self.config['initial_balance']) / self.config['initial_balance']
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
            
            # Max drawdown
            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (portfolio_values - peak) / peak
            max_drawdown = float(np.min(drawdown))
            
            agent.train()
            
            return {
                'total_return': total_return,
                'final_portfolio_value': portfolio_values[-1],
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'total_reward': total_reward,
                'profit_loss': portfolio_values[-1] - self.config['initial_balance']
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def print_metrics(self, episode: int, training_reward: float, validation_metrics: Dict, 
                     loss_info: Dict, elapsed_time: float):
        """Print comprehensive metrics in a nice format"""
        
        print(f"\n{'='*70}")
        print(f"🚀 {self.symbol} - Episode {episode} ({elapsed_time:.1f}s elapsed)")
        print(f"{'='*70}")
        
        # Training metrics
        def safe_float(val):
            """Safely convert to float, handling tuples/arrays"""
            if isinstance(val, (tuple, list, np.ndarray)):
                return float(val[0]) if len(val) > 0 else 0.0
            return float(val) if val is not None else 0.0
        
        training_reward = safe_float(training_reward)
        avg_reward = np.mean(self.metrics_history[-10:]) if len(self.metrics_history) >= 10 else training_reward
        
        print(f"📈 TRAINING:")
        print(f"   Episode Reward: {training_reward:+.2f}")
        print(f"   Avg Reward (last 10): {avg_reward:+.2f}")
        
        # Loss information
        if loss_info:
            print(f"📉 LOSSES:")
            for key, value in loss_info.items():
                if isinstance(value, (int, float)):
                    print(f"   {key}: {value:.6f}")
        
        # Validation metrics
        if 'error' not in validation_metrics:
            profit_loss = safe_float(validation_metrics['profit_loss'])
            total_return = safe_float(validation_metrics['total_return'])
            sharpe = safe_float(validation_metrics['sharpe_ratio'])
            drawdown = safe_float(validation_metrics['max_drawdown'])
            final_value = safe_float(validation_metrics['final_portfolio_value'])
            
            print(f"💰 VALIDATION (30-day test data):")
            print(f"   Profit/Loss: ${profit_loss:+,.2f}")
            print(f"   Total Return: {total_return:+.2%}")
            print(f"   Final Portfolio: ${final_value:,.2f}")
            print(f"   Sharpe Ratio: {sharpe:.3f}")
            print(f"   Max Drawdown: {drawdown:.2%}")
            
            # Profit status
            if profit_loss > 0:
                status = "🟢 PROFITABLE" if total_return > 0.05 else "🟡 MARGINAL PROFIT"
            else:
                status = "🔴 LOSING MONEY"
            print(f"   Status: {status}")
        else:
            print(f"❌ VALIDATION ERROR: {validation_metrics['error']}")
        
        print(f"{'='*70}")
    
    def save_checkpoint(self, agent: TradingAgent, episode: int, metrics: Dict):
        """Save checkpoint with metadata"""
        # Save model
        checkpoint_file = self.checkpoints_dir / f'{self.symbol}_ep{episode}.pth'
        torch.save(agent.state_dict(), checkpoint_file)
        
        # Save metadata
        metadata = {
            'symbol': self.symbol,
            'episode': episode,
            'timestamp': datetime.now().isoformat(),
            'training_time_minutes': (time.time() - self.start_time) / 60,
            'validation_metrics': metrics,
            'config': self.config
        }
        
        metadata_file = self.checkpoints_dir / f'{self.symbol}_ep{episode}_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Saved checkpoint: {checkpoint_file.name}")
    
    def train_quick_session(self) -> Dict:
        """Run a quick training session with live monitoring"""
        
        print(f"\n🎯 Starting {self.training_time_seconds/60:.1f}-minute training session for {self.symbol}")
        print(f"🔍 Looking for existing checkpoints...")
        
        # Load data
        try:
            train_df = self.load_stock_data('train')
            print(f"📊 Loaded {len(train_df)} training samples")
        except Exception as e:
            print(f"❌ Failed to load data: {e}")
            return {'error': str(e)}
        
        # Create agent and environment
        agent = self.create_agent(train_df)
        
        env = DailyTradingEnv(
            df=train_df,
            window_size=self.config['window_size'],
            initial_balance=self.config['initial_balance'],
            transaction_cost=self.config['transaction_cost']
        )
        
        # Create trainer
        trainer = PPOTrainer(
            agent=agent,
            gamma=self.config['gamma'],
            gae_lambda=self.config['gae_lambda'],
            eps_clip=self.config['clip_ratio'],
            k_epochs=self.config['ppo_epochs'],
            entropy_coef=self.config['entropy_coef'],
            value_loss_coef=self.config['value_coef']
        )
        
        # Training loop with time limit
        self.start_time = time.time()
        episode = self.last_checkpoint_episode
        
        # Initial validation
        initial_metrics = self.validate_agent_quickly(agent)
        
        print(f"\n🎬 Starting training from episode {episode}")
        if 'error' not in initial_metrics:
            print(f"📊 Initial validation profit: ${initial_metrics['profit_loss']:+,.2f}")
        
        try:
            while True:
                episode_start = time.time()
                
                # Train one episode
                training_reward = trainer.train_episode(env)
                self.metrics_history.append(training_reward)
                
                # Get loss info from trainer
                loss_info = getattr(trainer, 'last_losses', {})
                
                episode += 1
                elapsed_time = time.time() - self.start_time
                
                # Validate periodically or if near time limit
                should_validate = (episode % 10 == 0) or (elapsed_time > self.training_time_seconds - 30)
                
                if should_validate:
                    validation_metrics = self.validate_agent_quickly(agent)
                    
                    # Print metrics
                    self.print_metrics(episode, training_reward, validation_metrics, loss_info, elapsed_time)
                    
                    # Save checkpoint
                    self.save_checkpoint(agent, episode, validation_metrics)
                else:
                    # Quick progress update
                    print(f"📈 Episode {episode}: reward={training_reward:+.2f}, time={elapsed_time:.1f}s")
                
                # Check time limit
                if elapsed_time >= self.training_time_seconds:
                    break
                
        except KeyboardInterrupt:
            print(f"\n⏹️  Training interrupted by user")
        except Exception as e:
            print(f"❌ Training error: {e}")
            return {'error': str(e)}
        
        # Final validation
        print(f"\n🏁 Training session complete!")
        final_metrics = self.validate_agent_quickly(agent)
        
        # Save final checkpoint
        self.save_checkpoint(agent, episode, final_metrics)
        
        # Summary
        total_time = time.time() - self.start_time
        episodes_trained = episode - self.last_checkpoint_episode
        
        summary = {
            'symbol': self.symbol,
            'episodes_trained': episodes_trained,
            'total_episodes': episode,
            'training_time_minutes': total_time / 60,
            'episodes_per_minute': episodes_trained / (total_time / 60),
            'initial_metrics': initial_metrics,
            'final_metrics': final_metrics,
            'improvement': {}
        }
        
        # Calculate improvement
        if 'error' not in initial_metrics and 'error' not in final_metrics:
            summary['improvement'] = {
                'profit_change': final_metrics['profit_loss'] - initial_metrics['profit_loss'],
                'return_change': final_metrics['total_return'] - initial_metrics['total_return'],
                'sharpe_change': final_metrics['sharpe_ratio'] - initial_metrics['sharpe_ratio']
            }
        
        # Print final summary
        self.print_final_summary(summary)
        
        # Save session results
        results_file = self.quick_results_dir / f'{self.symbol}_session_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def print_final_summary(self, summary: Dict):
        """Print final session summary"""
        print(f"\n{'🎉 TRAINING SESSION SUMMARY 🎉':^70}")
        print(f"{'='*70}")
        print(f"Symbol: {summary['symbol']}")
        print(f"Episodes Trained: {summary['episodes_trained']}")
        print(f"Total Episodes: {summary['total_episodes']}")
        print(f"Training Time: {summary['training_time_minutes']:.1f} minutes")
        print(f"Speed: {summary['episodes_per_minute']:.1f} episodes/minute")
        
        if summary.get('improvement'):
            imp = summary['improvement']
            print(f"\n📊 IMPROVEMENT:")
            print(f"   Profit Change: ${imp['profit_change']:+,.2f}")
            print(f"   Return Change: {imp['return_change']:+.2%}")
            print(f"   Sharpe Change: {imp['sharpe_change']:+.3f}")
            
            # Overall assessment
            if imp['profit_change'] > 0:
                print(f"   Assessment: 🟢 IMPROVING")
            elif imp['profit_change'] > -100:
                print(f"   Assessment: 🟡 STABLE")
            else:
                print(f"   Assessment: 🔴 DECLINING")
        
        print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description='Quick training monitor')
    parser.add_argument('symbol', help='Stock symbol to train')
    parser.add_argument('--time', type=float, default=2.0, help='Training time in minutes')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device)
    torch.cuda.empty_cache() if device.type == 'cuda' else None
    
    print(f"🖥️  Using device: {device}")
    
    # Check if symbol data exists
    training_data_dir = Path('../trainingdata')
    train_file = training_data_dir / 'train' / f'{args.symbol}.csv'
    test_file = training_data_dir / 'test' / f'{args.symbol}.csv'
    
    if not train_file.exists():
        print(f"❌ No training data found for {args.symbol}")
        available_symbols = [f.stem for f in (training_data_dir / 'train').glob('*.csv')][:10]
        print(f"Available symbols: {', '.join(available_symbols)}")
        return
    
    if not test_file.exists():
        print(f"⚠️  No test data found for {args.symbol} - validation will be limited")
    
    # Run quick training session
    monitor = QuickTrainingMonitor(args.symbol, args.time)
    results = monitor.train_quick_session()
    
    if 'error' in results:
        print(f"❌ Training failed: {results['error']}")
        exit(1)
    else:
        print(f"✅ Training session completed successfully!")


if __name__ == "__main__":
    main()