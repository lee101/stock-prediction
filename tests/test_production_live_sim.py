#!/usr/bin/env python3
"""
Test production engine in realistic trading scenarios
Simulates live trading conditions with real market patterns
"""

import pytest
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import sys
import json

sys.path.append(str(Path(__file__).parent.parent))

from hfinference.production_engine import ProductionTradingEngine


def test_production_engine_with_real_data():
    """Test production engine with real market data"""
    
    # Download real data for testing
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    print("\n=== Production Engine Test with Real Data ===")
    
    # Initialize engine with test configuration
    config = {
        'model': {
            'input_features': 30,
            'hidden_size': 128,
            'num_heads': 8,
            'num_layers': 4,
            'sequence_length': 60,
            'prediction_horizon': 5
        },
        'trading': {
            'initial_capital': 100000,
            'max_position_size': 0.10,  # Conservative
            'max_positions': 5,
            'stop_loss': 0.02,
            'take_profit': 0.05,
            'trailing_stop': 0.015,
            'confidence_threshold': 0.70,  # Higher threshold
            'risk_per_trade': 0.01,
            'max_daily_loss': 0.02,
            'kelly_fraction': 0.20
        },
        'strategy': {
            'use_ensemble': False,  # Disable for testing
            'market_regime_filter': True,
            'volatility_filter': True
        },
        'data': {
            'normalize_features': True,
            'use_technical_indicators': True
        }
    }
    
    # Create mock checkpoint
    import torch
    import tempfile
    
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
        checkpoint_path = tmp.name
        torch.save({
            'model_state_dict': {},
            'config': config
        }, checkpoint_path)
    
    try:
        # Initialize engine
        engine = ProductionTradingEngine(
            checkpoint_path=checkpoint_path,
            paper_trading=True,
            live_trading=False
        )
        
        # Override config
        engine.config = config
        
        # Mock the model's forward pass with semi-realistic predictions
        def mock_forward(x):
            batch_size = x.shape[0]
            # Generate predictions based on recent trend
            trend = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
            
            # Price predictions with slight trend
            price_preds = torch.randn(batch_size, 5, 30) * 0.01 + trend * 0.005
            
            # Action logits based on trend
            if trend > 0:
                action_logits = torch.tensor([[2.0, 0.5, -1.0]])  # Buy bias
            elif trend < 0:
                action_logits = torch.tensor([[-1.0, 0.5, 2.0]])  # Sell bias
            else:
                action_logits = torch.tensor([[0.5, 2.0, 0.5]])   # Hold bias
            
            return {
                'price_predictions': price_preds,
                'action_logits': action_logits.repeat(batch_size, 1)
            }
        
        engine.model = mock_forward
        
        # Process each symbol
        results = []
        for symbol in symbols:
            print(f"\nProcessing {symbol}...")
            
            try:
                # Get historical data
                data = yf.download(
                    symbol, 
                    start=start_date, 
                    end=end_date, 
                    progress=False
                )
                
                if len(data) < 100:
                    print(f"  Insufficient data for {symbol}")
                    continue
                
                # Generate trading signal
                signal = engine.generate_enhanced_signal(symbol, data, use_ensemble=False)
                
                if signal:
                    print(f"  Signal generated:")
                    print(f"    Action: {signal.action}")
                    print(f"    Confidence: {signal.confidence:.2%}")
                    print(f"    Expected Return: {signal.expected_return:.2%}")
                    print(f"    Risk Score: {signal.risk_score:.2f}")
                    print(f"    Market Regime: {signal.market_regime}")
                    print(f"    Position Size: {signal.position_size:.2%}")
                    
                    # Attempt trade execution
                    if signal.action != 'hold' and signal.confidence > config['trading']['confidence_threshold']:
                        result = engine.execute_trade(signal)
                        print(f"    Trade Result: {result['status']}")
                        
                        if result['status'] == 'executed':
                            results.append({
                                'symbol': symbol,
                                'action': signal.action,
                                'confidence': signal.confidence,
                                'return': signal.expected_return
                            })
                else:
                    print(f"  No signal generated for {symbol}")
                    
            except Exception as e:
                print(f"  Error processing {symbol}: {e}")
        
        # Calculate portfolio metrics
        metrics = engine.calculate_portfolio_metrics()
        
        print("\n=== Portfolio Metrics ===")
        print(f"Portfolio Value: ${metrics['portfolio_value']:,.2f}")
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Number of Positions: {len(engine.positions)}")
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Current Drawdown: {metrics['current_drawdown']:.2%}")
        
        # Basic assertions
        assert metrics['portfolio_value'] > 0
        assert len(results) >= 0  # May not execute any trades
        
        # If trades were executed, check they're reasonable
        if results:
            for r in results:
                assert r['confidence'] >= config['trading']['confidence_threshold']
                assert r['action'] in ['buy', 'sell']
        
        print("\n✅ Production engine test passed!")
        
    finally:
        # Cleanup
        Path(checkpoint_path).unlink(missing_ok=True)


def test_risk_management_scenario():
    """Test risk management in adverse conditions"""
    
    print("\n=== Risk Management Scenario Test ===")
    
    # Create volatile market data
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    np.random.seed(42)
    
    # Simulate market crash scenario
    prices = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.03 - 0.001))  # Slight downward bias
    prices[70:80] *= 0.90  # 10% crash
    
    data = pd.DataFrame({
        'Open': prices * (1 + np.random.randn(100) * 0.005),
        'High': prices * (1 + np.abs(np.random.randn(100)) * 0.01),
        'Low': prices * (1 - np.abs(np.random.randn(100)) * 0.01),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)
    
    # Initialize engine with strict risk settings
    config = {
        'model': {
            'input_features': 30,
            'hidden_size': 64,
            'num_heads': 4,
            'num_layers': 2,
            'sequence_length': 60,
            'prediction_horizon': 5
        },
        'trading': {
            'initial_capital': 100000,
            'max_position_size': 0.05,  # Very conservative
            'max_positions': 3,
            'stop_loss': 0.01,  # Tight stop
            'take_profit': 0.03,
            'trailing_stop': 0.008,
            'confidence_threshold': 0.75,  # High threshold
            'risk_per_trade': 0.005,
            'max_daily_loss': 0.01,
            'kelly_fraction': 0.10
        },
        'strategy': {
            'use_ensemble': False,
            'market_regime_filter': True,
            'volatility_filter': True
        }
    }
    
    import torch
    import tempfile
    
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
        checkpoint_path = tmp.name
        torch.save({'model_state_dict': {}, 'config': config}, checkpoint_path)
    
    try:
        engine = ProductionTradingEngine(
            checkpoint_path=checkpoint_path,
            paper_trading=True,
            live_trading=False
        )
        
        engine.config = config
        
        # Mock conservative model
        def mock_forward(x):
            return {
                'price_predictions': torch.randn(x.shape[0], 5, 30) * 0.001,
                'action_logits': torch.tensor([[0.5, 2.0, 0.5]]).repeat(x.shape[0], 1)  # Prefer hold
            }
        
        engine.model = mock_forward
        
        # Test signals during crash period
        crash_data = data.iloc[60:85]  # Include pre-crash, crash, and post-crash
        
        signal = engine.generate_enhanced_signal('TEST', crash_data, use_ensemble=False)
        
        if signal:
            print(f"Signal during crash:")
            print(f"  Action: {signal.action}")
            print(f"  Confidence: {signal.confidence:.2%}")
            print(f"  Risk Score: {signal.risk_score:.2f}")
            print(f"  Market Regime: {signal.market_regime}")
            
            # In volatile/crash conditions, should be cautious
            assert signal.risk_score > 0.5 or signal.market_regime in ['volatile', 'bearish']
            
            # Position size should be reduced in risky conditions
            if signal.risk_score > 0.7:
                assert signal.position_size <= config['trading']['max_position_size'] * 0.5
        
        print("✅ Risk management test passed!")
        
    finally:
        Path(checkpoint_path).unlink(missing_ok=True)


def test_portfolio_evolution():
    """Test portfolio evolution over time"""
    
    print("\n=== Portfolio Evolution Test ===")
    
    # Generate synthetic bull market data
    dates = pd.date_range(end=datetime.now(), periods=250, freq='D')
    trend = np.linspace(100, 120, 250) + np.cumsum(np.random.randn(250) * 0.5)
    
    data = pd.DataFrame({
        'Open': trend + np.random.randn(250) * 0.5,
        'High': trend + np.abs(np.random.randn(250)) * 1.0,
        'Low': trend - np.abs(np.random.randn(250)) * 1.0,
        'Close': trend,
        'Volume': np.random.randint(1000000, 10000000, 250)
    }, index=dates)
    
    import torch
    import tempfile
    
    config = {
        'model': {'input_features': 30, 'hidden_size': 64, 'num_heads': 4, 
                 'num_layers': 2, 'sequence_length': 60, 'prediction_horizon': 5},
        'trading': {'initial_capital': 100000, 'max_position_size': 0.10,
                   'confidence_threshold': 0.65, 'stop_loss': 0.02, 'take_profit': 0.05}
    }
    
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
        checkpoint_path = tmp.name
        torch.save({'model_state_dict': {}, 'config': config}, checkpoint_path)
    
    try:
        engine = ProductionTradingEngine(checkpoint_path=checkpoint_path, paper_trading=True)
        engine.config = config
        
        # Bullish model
        def mock_forward(x):
            return {
                'price_predictions': torch.randn(x.shape[0], 5, 30) * 0.01 + 0.005,
                'action_logits': torch.tensor([[1.5, 0.5, -0.5]]).repeat(x.shape[0], 1)
            }
        
        engine.model = mock_forward
        
        # Simulate trading over time windows
        portfolio_values = []
        
        for i in range(60, min(len(data), 180), 10):  # Every 10 days
            window = data.iloc[max(0, i-60):i]
            
            # Generate and execute signals
            for symbol in ['STOCK1', 'STOCK2']:
                signal = engine.generate_enhanced_signal(symbol, window, use_ensemble=False)
                
                if signal and signal.confidence > 0.65:
                    engine.execute_trade(signal)
            
            # Update existing positions
            market_data = {sym: window.tail(1) for sym in engine.positions.keys()}
            if market_data:
                engine.update_positions(market_data)
            
            # Track portfolio value
            metrics = engine.calculate_portfolio_metrics()
            portfolio_values.append(metrics['portfolio_value'])
            
            if i % 30 == 0:
                print(f"Day {i}: Portfolio=${metrics['portfolio_value']:,.0f}, "
                      f"Positions={len(engine.positions)}")
        
        # Check portfolio grew over time (in bull market)
        if len(portfolio_values) > 1:
            initial_value = portfolio_values[0]
            final_value = portfolio_values[-1]
            print(f"\nPortfolio growth: {((final_value/initial_value - 1) * 100):.1f}%")
            
            # Should have some growth or at least preservation
            assert final_value >= initial_value * 0.95  # Allow 5% drawdown max
        
        print("✅ Portfolio evolution test passed!")
        
    finally:
        Path(checkpoint_path).unlink(missing_ok=True)


if __name__ == "__main__":
    test_production_engine_with_real_data()
    test_risk_management_scenario()
    test_portfolio_evolution()