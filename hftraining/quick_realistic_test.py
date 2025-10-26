#!/usr/bin/env python3
"""Quick test of realistic backtesting RL"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import logging
from realistic_backtest_rl import (
    RealisticTradingConfig, RealisticTradingEnvironment, 
    RealisticRLModel, train_realistic_rl
)
from data_utils import StockDataProcessor, split_data
import pandas as pd
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quick_test():
    """Quick test with minimal episodes"""
    
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Configuration
    config = RealisticTradingConfig()
    config.sequence_length = 30  # Shorter sequence
    
    # Load minimal local data
    from pathlib import Path
    data_dir = Path('trainingdata')
    candidates = list(data_dir.glob('SPY.csv')) or [p for p in data_dir.glob('*.csv') if 'spy' in p.stem.lower()]
    if not candidates:
        logger.error("No SPY CSV found under trainingdata/")
        return None, None
    df = pd.read_csv(candidates[0])
    df.columns = df.columns.str.lower()
    logger.info(f"Loaded {len(df)} records")
    
    # Process data
    processor = StockDataProcessor()
    features = processor.prepare_features(df)
    processor.fit_scalers(features)
    normalized_data = processor.transform(features)
    
    # Split data
    train_data, val_data, _ = split_data(normalized_data, 0.7, 0.15, 0.15)
    logger.info(f"Train size: {len(train_data)}, Val size: {len(val_data)}")
    
    # Create environments
    train_env = RealisticTradingEnvironment(train_data, config)
    val_env = RealisticTradingEnvironment(val_data, config)
    
    # Create model
    input_dim = normalized_data.shape[1]
    model = RealisticRLModel(config, input_dim)
    device = torch.device('cpu')
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Quick training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    best_sharpe = -float('inf')
    
    for episode in range(10):  # Just 10 episodes
        logger.info(f"Episode {episode + 1}/10")
        
        # Reset environment
        state = train_env.reset()
        total_reward = 0
        steps = 0
        
        while steps < 100:  # Limit steps per episode
            steps += 1
            
            # Get state
            market_data, portfolio_state = state
            market_tensor = torch.FloatTensor(market_data).unsqueeze(0)
            portfolio_tensor = torch.FloatTensor(portfolio_state).unsqueeze(0)
            
            # Get action
            with torch.no_grad():
                outputs = model(market_tensor, portfolio_tensor)
            
            # Simple epsilon-greedy
            if np.random.random() < 0.3:
                action = {
                    'trade': np.random.choice([0, 0, 0, 1, 2]),  # Bias toward holding
                    'position_size': 0.1,
                    'stop_loss': 0.02,
                    'take_profit': 0.05
                }
            else:
                action = {
                    'trade': torch.argmax(outputs['trade_logits']).item(),
                    'position_size': min(outputs['position_size'].item(), 0.2),
                    'stop_loss': outputs['stop_loss'].item(),
                    'take_profit': outputs['take_profit'].item()
                }
            
            # Step
            next_state, reward, done, metrics = train_env.step(action)
            total_reward += reward
            
            # Simple update
            if steps % 10 == 0:
                outputs = model(market_tensor, portfolio_tensor)
                loss = -outputs['value'].mean()  # Simple loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if done:
                break
            
            state = next_state
        
        # Log results
        logger.info(f"Episode {episode + 1} - Steps: {steps}, Reward: {total_reward:.4f}")
        logger.info(f"Metrics - Return: {metrics.total_return:.2%}, Sharpe: {metrics.sharpe_ratio:.3f}, "
                   f"Trades: {metrics.total_trades}, Win Rate: {metrics.win_rate:.1%}")
        
        if metrics.sharpe_ratio > best_sharpe:
            best_sharpe = metrics.sharpe_ratio
            logger.info(f"New best Sharpe: {best_sharpe:.3f}")
    
    # Final validation
    logger.info("\nRunning final validation...")
    val_state = val_env.reset()
    val_steps = 0
    
    while val_steps < 200 and val_state is not None:
        val_steps += 1
        
        market_data, portfolio_state = val_state
        market_tensor = torch.FloatTensor(market_data).unsqueeze(0)
        portfolio_tensor = torch.FloatTensor(portfolio_state).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(market_tensor, portfolio_tensor)
        
        action = {
            'trade': torch.argmax(outputs['trade_logits']).item(),
            'position_size': min(outputs['position_size'].item(), 0.15),
            'stop_loss': max(outputs['stop_loss'].item(), 0.01),
            'take_profit': min(outputs['take_profit'].item(), 0.1)
        }
        
        val_state, _, done, val_metrics = val_env.step(action)
        
        if done:
            break
    
    logger.info(f"\nFinal Validation Results:")
    logger.info(f"Return: {val_metrics.total_return:.2%}")
    logger.info(f"Sharpe Ratio: {val_metrics.sharpe_ratio:.3f}")
    logger.info(f"Sortino Ratio: {val_metrics.sortino_ratio:.3f}")
    logger.info(f"Max Drawdown: {val_metrics.max_drawdown:.2%}")
    logger.info(f"Total Trades: {val_metrics.total_trades}")
    logger.info(f"Win Rate: {val_metrics.win_rate:.1%}")
    logger.info(f"Profit Factor: {val_metrics.profit_factor:.2f}")
    logger.info(f"Commission: ${val_metrics.total_commission:.2f}")
    logger.info(f"Slippage: ${val_metrics.total_slippage:.2f}")
    
    return model, val_metrics, val_env.equity_curve

if __name__ == "__main__":
    model, metrics, equity_curve = quick_test()
    
    if metrics:
        print("\n" + "="*50)
        print("REALISTIC BACKTESTING COMPLETE")
        print(f"Final Sharpe: {metrics.sharpe_ratio:.3f}")
        print(f"Total Return: {metrics.total_return:.2%}")
        print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
        print("="*50)
