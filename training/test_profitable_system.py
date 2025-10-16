#!/usr/bin/env python3
"""
Quick test of the profitable trading system
"""

import torch
import numpy as np
import pandas as pd
import sys
sys.path.append('/media/lee/crucial2/code/stock/training')

from realistic_trading_env import RealisticTradingEnvironment, TradingConfig, create_market_data_generator
from differentiable_trainer import DifferentiableTradingModel

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_trading_system():
    """Test the trading system with a simple strategy"""
    
    logger.info("Testing Profitable Trading System")
    
    # Create environment with relaxed constraints for testing
    config = TradingConfig(
        initial_capital=100000,
        max_position_size=0.2,  # Allow larger positions
        commission_rate=0.0005,  # Lower commission
        slippage_factor=0.0002,  # Lower slippage
        stop_loss_pct=0.03,  # 3% stop loss
        take_profit_pct=0.06,  # 6% take profit
        min_trade_size=50.0  # Lower minimum
    )
    
    env = RealisticTradingEnvironment(config)
    
    # Generate test data
    market_data = create_market_data_generator(n_samples=1000, volatility=0.015)
    
    # Simple momentum strategy for testing
    logger.info("Running simple momentum strategy...")
    
    for i in range(100, 500):
        # Get market state
        current_price = market_data.iloc[i]['close']
        prev_price = market_data.iloc[i-1]['close']
        
        market_state = {
            'price': current_price,
            'timestamp': i
        }
        
        # Simple momentum signal
        price_change = (current_price - prev_price) / prev_price
        
        # Calculate moving averages
        sma_5 = market_data.iloc[i-5:i]['close'].mean()
        sma_20 = market_data.iloc[i-20:i]['close'].mean()
        
        # Generate signal
        if current_price > sma_5 > sma_20 and price_change > 0.001:
            signal = 0.8  # Strong buy
            confidence = min(1.0, abs(price_change) * 100)
        elif current_price < sma_5 < sma_20 and price_change < -0.001:
            signal = -0.8  # Strong sell
            confidence = min(1.0, abs(price_change) * 100)
        else:
            signal = 0.0  # Hold
            confidence = 0.5
        
        action = {
            'signal': torch.tensor(signal),
            'confidence': torch.tensor(confidence)
        }
        
        # Execute step
        metrics = env.step(action, market_state)
        
        # Log progress
        if i % 50 == 0:
            logger.info(f"Step {i}: Capital=${env.capital:,.2f}, "
                       f"Positions={len(env.positions)}, "
                       f"Trades={len(env.trades)}, "
                       f"Unrealized PnL=${metrics['unrealized_pnl']:.2f}")
    
    # Get final performance
    performance = env.get_performance_summary()
    
    logger.info("\n" + "="*60)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("="*60)
    
    # Display key metrics
    metrics_to_show = [
        ('Total Return', performance['total_return'], '.2%'),
        ('Sharpe Ratio', performance['sharpe_ratio'], '.3f'),
        ('Max Drawdown', performance['max_drawdown'], '.2%'),
        ('Win Rate', performance['win_rate'], '.1%'),
        ('Profit Factor', performance['profit_factor'], '.2f'),
        ('Total Trades', performance['total_trades'], 'd'),
        ('Final Capital', performance['current_capital'], ',.2f')
    ]
    
    for name, value, fmt in metrics_to_show:
        if 'f' in fmt or 'd' in fmt:
            logger.info(f"{name}: {value:{fmt}}")
        elif '%' in fmt:
            logger.info(f"{name}: {value:{fmt}}")
    
    # Check profitability
    is_profitable = performance['total_return'] > 0 and performance['sharpe_ratio'] > 0
    
    if is_profitable:
        logger.info("\n✅ SYSTEM IS PROFITABLE!")
    else:
        logger.info("\n❌ System needs more training")
    
    # Save performance plot
    env.plot_performance('training/test_performance.png')
    
    return performance, is_profitable


def test_with_model():
    """Test with trained model"""
    
    logger.info("\nTesting with Neural Model")
    
    # Create model
    model = DifferentiableTradingModel(
        input_dim=6,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        dropout=0.1
    )
    
    # Create environment
    config = TradingConfig(
        initial_capital=100000,
        max_position_size=0.15,
        commission_rate=0.0007,
        slippage_factor=0.0003
    )
    
    env = RealisticTradingEnvironment(config)
    
    # Generate test data
    market_data = create_market_data_generator(n_samples=2000, volatility=0.018)
    
    # Prepare features
    market_data['sma_5'] = market_data['close'].rolling(5).mean()
    market_data['sma_20'] = market_data['close'].rolling(20).mean()
    market_data['rsi'] = calculate_rsi(market_data['close'])
    market_data['volatility'] = market_data['returns'].rolling(20).std()
    market_data = market_data.dropna()
    
    model.eval()
    seq_len = 20
    
    with torch.no_grad():
        for i in range(seq_len, min(500, len(market_data)-1)):
            # Prepare input sequence
            seq_data = market_data.iloc[i-seq_len:i]
            features = ['close', 'volume', 'sma_5', 'sma_20', 'rsi', 'volatility']
            X = seq_data[features].values
            X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
            X_tensor = torch.FloatTensor(X).unsqueeze(0)
            
            # Get model prediction
            outputs = model(X_tensor)
            
            # Convert to action
            action_probs = torch.softmax(outputs['actions'], dim=-1).squeeze()
            position_size = outputs['position_sizes'].squeeze().item()
            confidence = outputs['confidences'].squeeze().item()
            
            # Generate trading signal
            if action_probs[0] > 0.5:  # Buy
                signal = abs(position_size)
            elif action_probs[2] > 0.5:  # Sell
                signal = -abs(position_size)
            else:
                signal = 0.0
            
            # Execute trade
            market_state = {
                'price': market_data.iloc[i]['close'],
                'timestamp': i
            }
            
            action = {
                'signal': torch.tensor(signal),
                'confidence': torch.tensor(confidence)
            }
            
            metrics = env.step(action, market_state)
            
            if i % 100 == 0:
                logger.info(f"Step {i}: Sharpe={metrics['sharpe_ratio']:.3f}, "
                           f"Return={metrics['reward']:.4f}")
    
    performance = env.get_performance_summary()
    
    logger.info("\nModel-Based Trading Results:")
    logger.info(f"Total Return: {performance['total_return']:.2%}")
    logger.info(f"Sharpe Ratio: {performance['sharpe_ratio']:.3f}")
    logger.info(f"Win Rate: {performance['win_rate']:.1%}")
    
    return performance


def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi


if __name__ == "__main__":
    # Test simple strategy
    simple_performance, is_profitable = test_trading_system()
    
    # Test with model
    model_performance = test_with_model()
    
    logger.info("\n" + "="*60)
    logger.info("FINAL COMPARISON")
    logger.info("="*60)
    logger.info(f"Simple Strategy Return: {simple_performance['total_return']:.2%}")
    logger.info(f"Model Strategy Return: {model_performance['total_return']:.2%}")
    
    if model_performance['total_return'] > simple_performance['total_return']:
        logger.info("✅ Model outperforms simple strategy!")
    else:
        logger.info("📊 Simple strategy still better - more training needed")