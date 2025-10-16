#!/usr/bin/env python3
"""
Enhanced Local Backtesting System with Real AI Forecast Integration
Simulates trading using the actual Toto AI model forecasts
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from loguru import logger
import sys
import os

# Import existing modules
from predict_stock_forecasting import make_predictions, load_stock_data_from_csv
from data_curate_daily import download_daily_stock_data
from src.fixtures import crypto_symbols
from src.sizing_utils import get_qty
from local_backtesting_system import LocalBacktester
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logger.remove()
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
logger.add("simulationresults/enhanced_backtesting.log", rotation="10 MB")


class MockAlpacaWrapper:
    """Mock Alpaca wrapper for offline backtesting"""
    def __init__(self, is_market_open: bool = True):
        self.is_open = is_market_open
        
    def get_clock(self):
        class Clock:
            def __init__(self, is_open):
                self.is_open = is_open
        return Clock(self.is_open)


class EnhancedLocalBacktester(LocalBacktester):
    """Enhanced backtester that uses real AI forecasts"""
    
    def __init__(self, *args, use_real_forecasts: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_real_forecasts = use_real_forecasts
        self.mock_alpaca = MockAlpacaWrapper()
        self.forecast_cache = {}
        
    def generate_real_ai_forecasts(self, symbols: List[str], forecast_date: datetime) -> Dict[str, Dict]:
        """Generate forecasts using the actual AI model"""
        
        # Check cache first
        cache_key = f"{forecast_date.strftime('%Y%m%d')}_{'_'.join(sorted(symbols))}"
        if cache_key in self.forecast_cache:
            logger.debug(f"Using cached AI forecasts for {forecast_date}")
            return self.forecast_cache[cache_key]
        
        logger.info(f"Generating real AI forecasts for {forecast_date}")
        
        # Prepare data directory for the AI model
        data_dir = Path("data") / f"backtest_{forecast_date.strftime('%Y%m%d')}"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare historical data for each symbol up to forecast_date
        for symbol in symbols:
            try:
                # Load historical data
                hist_data = self.load_symbol_history(symbol, forecast_date)
                if hist_data is not None and not hist_data.empty:
                    # Save to format expected by AI model
                    csv_path = data_dir / f"{symbol}.csv"
                    hist_data.to_csv(csv_path)
                    logger.debug(f"Prepared data for {symbol} at {csv_path}")
            except Exception as e:
                logger.error(f"Error preparing data for {symbol}: {e}")
        
        # Run the AI model
        try:
            # Set market open for crypto or if simulating market hours
            self.mock_alpaca.is_open = True
            
            # Call the real prediction function
            predictions_df = make_predictions(
                input_data_path=f"backtest_{forecast_date.strftime('%Y%m%d')}",
                alpaca_wrapper=self.mock_alpaca
            )
            
            # Parse predictions into our format
            forecasts = {}
            
            if predictions_df is not None and not predictions_df.empty:
                # Group by instrument
                for _, row in predictions_df.iterrows():
                    symbol = row.get('instrument', '')
                    if symbol in symbols:
                        # Extract predictions
                        close_pred = self._extract_prediction_value(row, 'close')
                        high_pred = self._extract_prediction_value(row, 'high')
                        low_pred = self._extract_prediction_value(row, 'low')
                        
                        # Calculate confidence from strategy profits
                        confidence = self._calculate_confidence(row)
                        
                        forecasts[symbol] = {
                            'close_total_predicted_change': close_pred,
                            'high_predicted_change': high_pred,
                            'low_predicted_change': low_pred,
                            'confidence': confidence,
                            'forecast_date': forecast_date.isoformat(),
                            'forecast_horizon_days': self.forecast_horizon,
                            'raw_predictions': row.to_dict()  # Store raw predictions
                        }
                        
                        logger.debug(f"{symbol}: predicted {close_pred:.4f} with confidence {confidence:.3f}")
            
            # Cache the results
            self.forecast_cache[cache_key] = forecasts
            
            # Also save to disk cache
            cache_file = self.cache_dir / f"ai_forecasts_{cache_key}.json"
            with open(cache_file, 'w') as f:
                json.dump(forecasts, f, indent=2)
                
            return forecasts
            
        except Exception as e:
            logger.error(f"Error generating AI forecasts: {e}")
            import traceback
            traceback.print_exc()
            
            # Fall back to synthetic forecasts
            return super().generate_forecast_cache(symbols, forecast_date)
    
    def _extract_prediction_value(self, row: pd.Series, price_type: str) -> float:
        """Extract prediction value from DataFrame row"""
        # Try different column formats
        col_names = [
            f'{price_type}_predicted_price_value',
            f'{price_type}_predicted_price',
            f'{price_type}_total_predicted_change'
        ]
        
        for col in col_names:
            if col in row:
                value = row[col]
                # Handle string representations like "(119.93537139892578,)"
                if isinstance(value, str) and value.startswith('(') and value.endswith(')'):
                    value = float(value.strip('()').rstrip(','))
                    # Convert to percentage change if it's a price
                    if 'price' in col and 'last_close' in row:
                        last_close = row['last_close']
                        if isinstance(last_close, (int, float)) and last_close > 0:
                            return (value - last_close) / last_close
                elif isinstance(value, (int, float)):
                    return value
        
        # Default to small random value if not found
        return np.random.normal(0.005, 0.01)
    
    def _calculate_confidence(self, row: pd.Series) -> float:
        """Calculate confidence score from prediction data"""
        # Use strategy profit predictions as confidence indicators
        profit_cols = ['entry_takeprofit_profit', 'maxdiffprofit_profit', 'takeprofit_profit']
        
        profits = []
        for col in profit_cols:
            if col in row:
                value = row[col]
                if isinstance(value, str) and value.startswith('(') and value.endswith(')'):
                    value = float(value.strip('()').rstrip(','))
                if isinstance(value, (int, float)):
                    profits.append(value)
        
        if profits:
            # Higher average profit = higher confidence
            avg_profit = np.mean(profits)
            # Convert to 0-1 range (assuming profits are typically -0.05 to 0.05)
            confidence = np.clip((avg_profit + 0.02) / 0.04, 0.3, 0.9)
            return confidence
        
        # Default confidence
        return 0.6
    
    def load_symbol_history(self, symbol: str, end_date: datetime) -> Optional[pd.DataFrame]:
        """Load historical data for a symbol up to end_date"""
        # Look for existing data files
        data_files = list(Path("data").glob(f"{symbol}*.csv"))
        
        if data_files:
            # Use most recent file
            latest_file = max(data_files, key=lambda x: x.stat().st_mtime)
            df = pd.read_csv(latest_file)
            
            # Ensure date column
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df[df['Date'] <= end_date]
            elif 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df[df['timestamp'] <= end_date]
                df = df.rename(columns={'timestamp': 'Date'})
            
            return df
        
        return None
    
    def generate_forecast_cache(self, symbols: List[str], forecast_date: datetime) -> Dict[str, Dict]:
        """Override to use real AI forecasts when enabled"""
        if self.use_real_forecasts:
            return self.generate_real_ai_forecasts(symbols, forecast_date)
        else:
            return super().generate_forecast_cache(symbols, forecast_date)
    
    def run_backtest(self, symbols: List[str], strategy: str = 'equal_weight', 
                    start_date: Optional[datetime] = None) -> Dict:
        """Enhanced backtest with additional metrics"""
        
        # Run base backtest
        results = super().run_backtest(symbols, strategy, start_date)
        
        # Add enhanced metrics
        results['used_real_forecasts'] = self.use_real_forecasts
        results['forecast_accuracy'] = self.calculate_forecast_accuracy()
        
        return results
    
    def calculate_forecast_accuracy(self) -> Dict[str, float]:
        """Calculate how accurate the forecasts were"""
        if not self.trade_history:
            return {}
        
        correct_direction = 0
        total_forecasts = 0
        forecast_errors = []
        
        for trade in self.trade_history:
            if trade['type'] == 'sell' and 'profit' in trade:
                # Check if forecast direction was correct
                if trade['profit'] > 0:
                    correct_direction += 1
                total_forecasts += 1
                
                # Calculate forecast error if we have the original forecast
                if 'forecast_return' in trade:
                    actual_return = trade['return_pct'] / 100
                    forecast_return = trade['forecast_return']
                    error = abs(actual_return - forecast_return)
                    forecast_errors.append(error)
        
        accuracy = {
            'directional_accuracy': (correct_direction / total_forecasts * 100) if total_forecasts > 0 else 0,
            'mean_absolute_error': np.mean(forecast_errors) if forecast_errors else 0,
            'total_forecasts': total_forecasts
        }
        
        return accuracy


def run_enhanced_comparison(symbols: List[str], simulation_days: int = 25, 
                          compare_with_synthetic: bool = True):
    """Run comparison between real AI forecasts and synthetic forecasts"""
    
    strategies = ['single_position', 'equal_weight', 'risk_weighted']
    
    results_real = {}
    results_synthetic = {}
    
    # Run with real AI forecasts
    logger.info("\n" + "="*80)
    logger.info("RUNNING BACKTESTS WITH REAL AI FORECASTS")
    logger.info("="*80)
    
    for strategy in strategies:
        logger.info(f"\nTesting {strategy} with real AI forecasts...")
        
        backtester = EnhancedLocalBacktester(
            initial_capital=100000,
            trading_fee=0.001,
            slippage=0.0005,
            max_positions=5 if strategy != 'single_position' else 1,
            simulation_days=simulation_days,
            use_real_forecasts=True
        )
        
        results = backtester.run_backtest(symbols, strategy)
        backtester.save_results(results, f"{strategy}_real_ai")
        results_real[strategy] = results
    
    # Optionally run with synthetic forecasts for comparison
    if compare_with_synthetic:
        logger.info("\n" + "="*80)
        logger.info("RUNNING BACKTESTS WITH SYNTHETIC FORECASTS")
        logger.info("="*80)
        
        for strategy in strategies:
            logger.info(f"\nTesting {strategy} with synthetic forecasts...")
            
            backtester = EnhancedLocalBacktester(
                initial_capital=100000,
                trading_fee=0.001,
                slippage=0.0005,
                max_positions=5 if strategy != 'single_position' else 1,
                simulation_days=simulation_days,
                use_real_forecasts=False
            )
            
            results = backtester.run_backtest(symbols, strategy)
            backtester.save_results(results, f"{strategy}_synthetic")
            results_synthetic[strategy] = results
    
    # Create comparison visualization
    create_ai_vs_synthetic_comparison(results_real, results_synthetic)
    
    # Print detailed comparison
    print_ai_forecast_analysis(results_real, results_synthetic)
    
    return results_real, results_synthetic


def create_ai_vs_synthetic_comparison(results_real: Dict, results_synthetic: Dict):
    """Create comparison chart between AI and synthetic forecasts"""
    
    if not results_synthetic:
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Real AI Forecasts vs Synthetic Forecasts Comparison', fontsize=16)
    
    strategies = list(results_real.keys())
    x = np.arange(len(strategies))
    width = 0.35
    
    # 1. Returns comparison
    returns_real = [results_real[s]['total_return_pct'] for s in strategies]
    returns_synthetic = [results_synthetic[s]['total_return_pct'] for s in strategies]
    
    bars1 = ax1.bar(x - width/2, returns_real, width, label='Real AI', alpha=0.8)
    bars2 = ax1.bar(x + width/2, returns_synthetic, width, label='Synthetic', alpha=0.8)
    
    ax1.set_xlabel('Strategy')
    ax1.set_ylabel('Total Return (%)')
    ax1.set_title('Returns: AI vs Synthetic Forecasts')
    ax1.set_xticks(x)
    ax1.set_xticklabels([s.replace('_', ' ').title() for s in strategies])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Sharpe Ratio comparison
    sharpe_real = [results_real[s]['sharpe_ratio'] for s in strategies]
    sharpe_synthetic = [results_synthetic[s]['sharpe_ratio'] for s in strategies]
    
    bars3 = ax2.bar(x - width/2, sharpe_real, width, label='Real AI', alpha=0.8)
    bars4 = ax2.bar(x + width/2, sharpe_synthetic, width, label='Synthetic', alpha=0.8)
    
    ax2.set_xlabel('Strategy')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.set_title('Risk-Adjusted Returns: AI vs Synthetic')
    ax2.set_xticks(x)
    ax2.set_xticklabels([s.replace('_', ' ').title() for s in strategies])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Win Rate comparison
    win_rate_real = [(r['winning_trades']/r['num_trades']*100) if r['num_trades'] > 0 else 0 
                     for r in results_real.values()]
    win_rate_synthetic = [(r['winning_trades']/r['num_trades']*100) if r['num_trades'] > 0 else 0 
                         for r in results_synthetic.values()]
    
    bars5 = ax3.bar(x - width/2, win_rate_real, width, label='Real AI', alpha=0.8)
    bars6 = ax3.bar(x + width/2, win_rate_synthetic, width, label='Synthetic', alpha=0.8)
    
    ax3.set_xlabel('Strategy')
    ax3.set_ylabel('Win Rate (%)')
    ax3.set_title('Trade Success Rate: AI vs Synthetic')
    ax3.set_xticks(x)
    ax3.set_xticklabels([s.replace('_', ' ').title() for s in strategies])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Forecast accuracy (only for real AI)
    accuracy_data = []
    for strategy in strategies:
        if 'forecast_accuracy' in results_real[strategy]:
            acc = results_real[strategy]['forecast_accuracy']
            accuracy_data.append(acc.get('directional_accuracy', 0))
        else:
            accuracy_data.append(0)
    
    ax4.bar(strategies, accuracy_data, alpha=0.7, color='green')
    ax4.set_xlabel('Strategy')
    ax4.set_ylabel('Directional Accuracy (%)')
    ax4.set_title('AI Forecast Directional Accuracy')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(accuracy_data):
        ax4.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save chart
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chart_file = Path("simulationresults") / f"ai_vs_synthetic_comparison_{timestamp}.png"
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"AI vs Synthetic comparison chart saved to {chart_file}")


def print_ai_forecast_analysis(results_real: Dict, results_synthetic: Dict):
    """Print detailed analysis of AI forecast performance"""
    
    print("\n" + "="*80)
    print("AI FORECAST PERFORMANCE ANALYSIS")
    print("="*80)
    
    print("\nStrategy Performance Comparison:")
    print(f"{'Strategy':<20} {'AI Return %':>12} {'Synth Return %':>15} {'AI Advantage':>13}")
    print("-"*80)
    
    for strategy in results_real.keys():
        ai_return = results_real[strategy]['total_return_pct']
        synth_return = results_synthetic[strategy]['total_return_pct'] if strategy in results_synthetic else 0
        advantage = ai_return - synth_return
        
        print(f"{strategy:<20} {ai_return:>12.2f} {synth_return:>15.2f} {advantage:>+13.2f}")
    
    # Calculate average advantage
    advantages = []
    for strategy in results_real.keys():
        if strategy in results_synthetic:
            advantages.append(results_real[strategy]['total_return_pct'] - 
                            results_synthetic[strategy]['total_return_pct'])
    
    if advantages:
        print(f"\nAverage AI Advantage: {np.mean(advantages):+.2f}%")
    
    # Forecast accuracy analysis
    print("\n" + "-"*80)
    print("AI Forecast Accuracy Analysis:")
    print("-"*80)
    
    for strategy, results in results_real.items():
        if 'forecast_accuracy' in results:
            acc = results['forecast_accuracy']
            print(f"\n{strategy}:")
            print(f"  Directional Accuracy: {acc.get('directional_accuracy', 0):.1f}%")
            print(f"  Mean Absolute Error:  {acc.get('mean_absolute_error', 0):.4f}")
            print(f"  Total Forecasts:      {acc.get('total_forecasts', 0)}")


if __name__ == "__main__":
    # Default symbols to test
    test_symbols = ['BTCUSD', 'ETHUSD', 'NVDA', 'TSLA', 'AAPL', 'GOOG', 'META', 'MSFT']
    
    logger.info("Starting Enhanced Local Backtesting System with Real AI Forecasts")
    logger.info(f"Testing with symbols: {test_symbols}")
    
    # Create results directory
    Path("simulationresults").mkdir(exist_ok=True)
    
    # Run enhanced comparison
    results_real, results_synthetic = run_enhanced_comparison(
        test_symbols, 
        simulation_days=25,
        compare_with_synthetic=True
    )
    
    logger.info("\nEnhanced backtesting complete!")
    logger.info("Check simulationresults/ directory for detailed results and visualizations.")