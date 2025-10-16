#!/usr/bin/env python3
"""
Quick simulation for testing strategies without GPU-heavy forecasting.
Uses simplified mock data to test position sizing strategies rapidly.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

# Add project root to path  
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtests.simulate_trading_strategies import TradingSimulator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuickSimulator(TradingSimulator):
    """Quick simulator that uses mock forecasts instead of real GPU predictions."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Don't load the actual Toto pipeline for quick testing
        self.pipeline = "mock_pipeline"
    
    def generate_forecasts_for_symbol(self, symbol: str, csv_file: Path) -> dict:
        """Generate mock forecasts for quick testing."""
        logger.info(f"Generating MOCK forecasts for {symbol}...")
        
        # Load basic data to get realistic price ranges
        try:
            data = self.load_and_preprocess_data(csv_file)
            if data is None:
                return None
                
            last_close = data['Close'].iloc[-1]
            
            # Generate realistic mock predictions based on symbol characteristics
            np.random.seed(hash(symbol) % 2**32)  # Deterministic per symbol
            
            # Different symbols get different prediction profiles
            if symbol in ['NVDA', 'TSLA', 'QUBT']:  # High volatility stocks
                base_return = np.random.uniform(-0.1, 0.15)  # -10% to +15%
                volatility = 0.8
            elif symbol in ['AAPL', 'MSFT', 'GOOGL', 'META']:  # Large cap tech
                base_return = np.random.uniform(-0.05, 0.08)  # -5% to +8%
                volatility = 0.5
            elif 'USD' in symbol:  # Crypto
                base_return = np.random.uniform(-0.15, 0.2)  # -15% to +20%
                volatility = 1.2
            else:  # Other stocks
                base_return = np.random.uniform(-0.08, 0.1)  # -8% to +10%
                volatility = 0.6
            
            # Generate predictions for close, high, low
            close_change = base_return + np.random.normal(0, 0.02)
            high_change = close_change + abs(np.random.normal(0.02, 0.01)) * volatility
            low_change = close_change - abs(np.random.normal(0.02, 0.01)) * volatility
            
            # Create realistic prediction structure
            predictions = []
            for i in range(7):  # 7 day predictions
                daily_change = close_change / 7 + np.random.normal(0, 0.005)
                predictions.append(daily_change)
            
            results = {
                'symbol': symbol,
                'close_last_price': last_close,
                'close_predictions': predictions,
                'close_predicted_changes': predictions,
                'close_total_predicted_change': sum(predictions),
                'close_predicted_price_value': last_close * (1 + sum(predictions)),
                
                'high_last_price': data['High'].iloc[-1],
                'high_total_predicted_change': high_change,
                'high_predicted_price_value': data['High'].iloc[-1] * (1 + high_change),
                
                'low_last_price': data['Low'].iloc[-1], 
                'low_total_predicted_change': low_change,
                'low_predicted_price_value': data['Low'].iloc[-1] * (1 + low_change),
                
                'forecast_generated_at': datetime.now().isoformat()
            }
            
            logger.info(f"{symbol}: {close_change:.4f} total predicted change")
            return results
            
        except Exception as e:
            logger.error(f"Error generating mock forecast for {symbol}: {e}")
            return None


def analyze_strategy_performance(results: dict):
    """Analyze and compare strategy performance."""
    print("\n" + "="*80)
    print("STRATEGY PERFORMANCE ANALYSIS")
    print("="*80)
    
    if 'strategies' not in results:
        print("No strategy results to analyze")
        return
    
    strategies = results['strategies']
    valid_strategies = {k: v for k, v in strategies.items() if 'error' not in v}
    
    if not valid_strategies:
        print("No valid strategies found")
        return
    
    print(f"\nAnalyzing {len(valid_strategies)} strategies...")
    
    # Sort strategies by simulated return
    sorted_strategies = sorted(
        valid_strategies.items(),
        key=lambda x: x[1].get('performance', {}).get('simulated_actual_return', 0),
        reverse=True
    )
    
    print("\nSTRATEGY RANKINGS (by simulated return):")
    print("-" * 60)
    
    for i, (name, data) in enumerate(sorted_strategies, 1):
        perf = data.get('performance', {})
        expected = data.get('expected_return', 0)
        simulated = perf.get('simulated_actual_return', 0)
        profit = perf.get('profit_loss', 0)
        positions = data.get('num_positions', len(data.get('allocation', {})))
        risk = data.get('risk_level', 'Unknown')
        
        print(f"{i:2d}. {name.replace('_', ' ').title():25s}")
        print(f"    Expected Return: {expected:7.3f} ({expected*100:5.1f}%)")
        print(f"    Simulated Return: {simulated:6.3f} ({simulated*100:5.1f}%)")
        print(f"    Profit/Loss: ${profit:10,.2f}")
        print(f"    Positions: {positions:2d}    Risk Level: {risk}")
        
        # Show top allocations
        allocation = data.get('allocation', {})
        if allocation:
            top_allocations = sorted(allocation.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"    Top Allocations: {', '.join([f'{symbol}({weight:.1%})' for symbol, weight in top_allocations])}")
        print()
    
    # Find best strategies by different metrics
    print("BEST STRATEGIES BY METRIC:")
    print("-" * 40)
    
    # Best by return
    best_return = max(valid_strategies.items(), key=lambda x: x[1].get('performance', {}).get('simulated_actual_return', 0))
    print(f"Best Return: {best_return[0].replace('_', ' ').title()} ({best_return[1].get('performance', {}).get('simulated_actual_return', 0)*100:.1f}%)")
    
    # Best by profit
    best_profit = max(valid_strategies.items(), key=lambda x: x[1].get('performance', {}).get('profit_loss', 0))
    print(f"Best Profit: {best_profit[0].replace('_', ' ').title()} (${best_profit[1].get('performance', {}).get('profit_loss', 0):,.2f})")
    
    # Most diversified (most positions)
    most_diversified = max(valid_strategies.items(), key=lambda x: x[1].get('num_positions', 0))
    print(f"Most Diversified: {most_diversified[0].replace('_', ' ').title()} ({most_diversified[1].get('num_positions', 0)} positions)")
    
    # Analyze forecast quality
    forecasts = results.get('forecasts', {})
    if forecasts:
        print(f"\nFORECAST ANALYSIS:")
        print("-" * 30)
        
        predicted_returns = []
        positive_predictions = 0
        
        for symbol, data in forecasts.items():
            if 'close_total_predicted_change' in data:
                ret = data['close_total_predicted_change']
                predicted_returns.append(ret)
                if ret > 0:
                    positive_predictions += 1
        
        if predicted_returns:
            print(f"Total Symbols: {len(predicted_returns)}")
            print(f"Positive Predictions: {positive_predictions} ({positive_predictions/len(predicted_returns)*100:.1f}%)")
            print(f"Mean Predicted Return: {np.mean(predicted_returns)*100:.2f}%")
            print(f"Std Predicted Return: {np.std(predicted_returns)*100:.2f}%")
            print(f"Best Predicted: {max(predicted_returns)*100:.2f}%")
            print(f"Worst Predicted: {min(predicted_returns)*100:.2f}%")
            
            # Show top 5 predictions
            forecast_items = [(symbol, data['close_total_predicted_change']) 
                             for symbol, data in forecasts.items() 
                             if 'close_total_predicted_change' in data]
            top_forecasts = sorted(forecast_items, key=lambda x: x[1], reverse=True)[:5]
            
            print(f"\nTOP 5 PREDICTED PERFORMERS:")
            for symbol, ret in top_forecasts:
                print(f"  {symbol}: {ret*100:+5.2f}%")


def main():
    """Run quick simulation for strategy testing."""
    print("Starting QUICK trading strategy simulation (with mock forecasts)...")
    
    # Create quick simulator
    simulator = QuickSimulator(
        backtestdata_dir="backtestdata",
        forecast_days=7,
        initial_capital=100000,
        output_dir="backtests/quick_results"
    )
    
    try:
        # Run simulation
        results = simulator.run_comprehensive_strategy_test()
        
        if not results:
            logger.error("No results generated")
            return
        
        # Analyze performance
        analyze_strategy_performance(results)
        
        # Save results  
        csv_file, forecasts_csv = simulator.save_results("quick_simulation_results")
        
        # Create visualizations (skip for quick test to avoid matplotlib issues)
        try:
            logger.info("Creating visualizations...")
            viz_files = simulator.viz_logger.create_all_visualizations(results)
            print(f"\nVisualizations created:")
            for viz_file in viz_files:
                print(f"  - {viz_file}")
        except Exception as e:
            logger.warning(f"Visualization creation failed (this is OK for quick test): {e}")
        
        print(f"\n" + "="*80)
        print(f"Results saved to: {csv_file} and {forecasts_csv}")
        print(f"TensorBoard logs: {simulator.viz_logger.tb_writer.log_dir}")
        print("="*80)
        
        # Close visualization logger
        simulator.viz_logger.close()
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise


if __name__ == "__main__":
    main()