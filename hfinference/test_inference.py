#!/usr/bin/env python3
"""
Test HuggingFace Inference System
Verify the system works with trained models
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
import sys
from datetime import datetime, timedelta

sys.path.append(str(Path(__file__).parent.parent))

from hfinference.hf_trading_engine import (
    HFTradingEngine, 
    DataProcessor,
    TradingSignal
)


def test_model_loading():
    """Test loading a model checkpoint"""
    
    print("\n" + "="*60)
    print("TEST 1: Model Loading")
    print("="*60)
    
    # Check for available checkpoints
    checkpoint_dir = Path('hftraining/checkpoints/production')
    
    if not checkpoint_dir.exists():
        print("❌ No checkpoints found")
        return False
    
    checkpoints = list(checkpoint_dir.glob('*.pt'))
    
    if not checkpoints:
        print("❌ No checkpoint files found")
        return False
    
    # Use the latest checkpoint
    checkpoint_path = sorted(checkpoints)[-1]
    print(f"✅ Found checkpoint: {checkpoint_path}")
    
    # Try loading
    try:
        config_path = 'hfinference/configs/default_config.json'
        engine = HFTradingEngine(str(checkpoint_path), config_path)
        print("✅ Model loaded successfully")
        
        # Check model architecture
        total_params = sum(p.numel() for p in engine.model.parameters())
        print(f"✅ Model has {total_params:,} parameters")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False


def test_inference():
    """Test model inference"""
    
    print("\n" + "="*60)
    print("TEST 2: Model Inference")
    print("="*60)
    
    # Find checkpoint
    checkpoint_path = Path('hftraining/checkpoints/production/final.pt')
    
    if not checkpoint_path.exists():
        # Try any checkpoint
        checkpoint_dir = Path('hftraining/checkpoints/production')
        checkpoints = list(checkpoint_dir.glob('*.pt'))
        
        if not checkpoints:
            print("❌ No checkpoints available for testing")
            return False
        
        checkpoint_path = checkpoints[0]
    
    try:
        # Load model
        config_path = 'hfinference/configs/default_config.json'
        engine = HFTradingEngine(str(checkpoint_path), config_path)
        
        # Create synthetic data for testing
        dates = pd.date_range(end=datetime.now(), periods=100)
        
        # Generate realistic OHLCV data
        np.random.seed(42)
        close_prices = 100 + np.cumsum(np.random.randn(100) * 2)
        
        test_data = pd.DataFrame({
            'Open': close_prices + np.random.randn(100) * 0.5,
            'High': close_prices + np.abs(np.random.randn(100)) * 2,
            'Low': close_prices - np.abs(np.random.randn(100)) * 2,
            'Close': close_prices,
            'Volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
        
        # Generate signal
        signal = engine.generate_signal('TEST', test_data)
        
        if signal:
            print("✅ Inference successful")
            print(f"\nGenerated Signal:")
            print(f"  Action: {signal.action}")
            print(f"  Confidence: {signal.confidence:.2%}")
            print(f"  Current Price: ${signal.current_price:.2f}")
            print(f"  Predicted Price: ${signal.predicted_price:.2f}")
            print(f"  Expected Return: {signal.expected_return:.2%}")
            print(f"  Position Size: {signal.position_size:.2%}")
            
            return True
        else:
            print("❌ Failed to generate signal")
            return False
            
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_trading_simulation():
    """Test a simple trading simulation"""
    
    print("\n" + "="*60)
    print("TEST 3: Trading Simulation")
    print("="*60)
    
    # Find checkpoint
    checkpoint_path = Path('hftraining/checkpoints/production/final.pt')
    
    if not checkpoint_path.exists():
        checkpoint_dir = Path('hftraining/checkpoints/production')
        checkpoints = list(checkpoint_dir.glob('*.pt'))
        
        if not checkpoints:
            print("❌ No checkpoints available")
            return False
        
        checkpoint_path = checkpoints[0]
    
    try:
        # Load engine
        config_path = 'hfinference/configs/default_config.json'
        engine = HFTradingEngine(str(checkpoint_path), config_path)
        
        # Set initial capital
        engine.current_capital = 10000
        
        # Generate test data
        dates = pd.date_range(end=datetime.now(), periods=200)
        np.random.seed(42)
        
        # Create trending data
        trend = np.linspace(100, 120, 200)
        noise = np.random.randn(200) * 2
        close_prices = trend + noise
        
        test_data = pd.DataFrame({
            'Open': close_prices + np.random.randn(200) * 0.5,
            'High': close_prices + np.abs(np.random.randn(200)) * 2,
            'Low': close_prices - np.abs(np.random.randn(200)) * 2,
            'Close': close_prices,
            'Volume': np.random.randint(1000000, 10000000, 200)
        }, index=dates)
        
        # Simulate trading
        trades_executed = 0
        
        for i in range(60, len(test_data)):
            window = test_data.iloc[i-60:i]
            
            # Generate signal
            signal = engine.generate_signal('TEST', window)
            
            if signal and signal.confidence > 0.6:
                # Execute trade
                result = engine.execute_trade(signal)
                
                if result['status'] == 'executed':
                    trades_executed += 1
                    print(f"Trade {trades_executed}: {signal.action.upper()} @ ${signal.current_price:.2f}")
        
        # Calculate final value
        final_value = engine.calculate_portfolio_value()
        profit = final_value - 10000
        
        print(f"\n✅ Simulation completed")
        print(f"  Trades Executed: {trades_executed}")
        print(f"  Final Portfolio Value: ${final_value:.2f}")
        print(f"  Profit/Loss: ${profit:.2f} ({profit/100:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        return False


def test_backtest_integration():
    """Test backtest with real data"""
    
    print("\n" + "="*60)
    print("TEST 4: Backtest Integration")
    print("="*60)
    
    # Find checkpoint
    checkpoint_path = Path('hftraining/checkpoints/production/final.pt')
    
    if not checkpoint_path.exists():
        checkpoint_dir = Path('hftraining/checkpoints/production')
        checkpoints = list(checkpoint_dir.glob('*.pt'))
        
        if not checkpoints:
            print("❌ No checkpoints available")
            return False
        
        checkpoint_path = checkpoints[0]
    
    try:
        # Import run_trading module
        from hfinference.run_trading import HFTrader
        
        # Initialize trader
        trader = HFTrader(
            checkpoint_path=str(checkpoint_path),
            config_path='hfinference/configs/default_config.json',
            mode='backtest'
        )
        
        # Run short backtest
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        results = trader.run_backtest(
            symbols=['AAPL'],
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            initial_capital=10000
        )
        
        if results and 'metrics' in results:
            print("✅ Backtest completed successfully")
            return True
        else:
            print("❌ Backtest failed to produce results")
            return False
            
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    
    print("="*60)
    print("HFINFERENCE SYSTEM TEST SUITE")
    print("="*60)
    
    tests = [
        ("Model Loading", test_model_loading),
        ("Inference", test_inference),
        ("Trading Simulation", test_trading_simulation),
        ("Backtest Integration", test_backtest_integration)
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"Test {name} crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name:<25} {status}")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed! System is ready for trading.")
    else:
        print("\n⚠️  Some tests failed. Check the logs above.")
    
    return passed_count == total_count


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)