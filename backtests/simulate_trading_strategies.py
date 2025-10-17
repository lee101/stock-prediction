#!/usr/bin/env python3
"""
Simulate actual trading strategies using all backtestdata CSV files.
Tests different portfolio allocation strategies based on Toto model forecasts.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import csv
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project root to path  
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Import visualization logger
from backtests.visualization_logger import VisualizationLogger

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TradingSimulator:
    """Simulates trading strategies across all available stock data."""
    
    def __init__(self, 
                 backtestdata_dir: str = "backtestdata",
                 forecast_days: int = 5,
                 initial_capital: float = 100000,
                 output_dir: str = "backtests/results"):
        self.backtestdata_dir = Path(backtestdata_dir)
        self.forecast_days = forecast_days
        self.initial_capital = initial_capital
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load all CSV files
        self.csv_files = list(self.backtestdata_dir.glob("*.csv"))
        self.symbols = [f.stem.split('-')[0] for f in self.csv_files]
        
        logger.info(f"Found {len(self.csv_files)} data files for symbols: {self.symbols}")
        
        # Initialize prediction infrastructure
        self.pipeline = None
        self._load_prediction_pipeline()
        
        # Initialize visualization logger
        self.viz_logger = VisualizationLogger(
            output_dir=str(self.output_dir),
            tb_log_dir=f"./logs/trading_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Results storage
        self.results = {}
        self.forecast_data = {}
        
    def _load_prediction_pipeline(self):
        """Load the Toto prediction pipeline."""
        try:
            from src.models.toto_wrapper import TotoPipeline
            if self.pipeline is None:
                logger.info("Loading Toto pipeline...")
                self.pipeline = TotoPipeline.from_pretrained(
                    "Datadog/Toto-Open-Base-1.0",
                    device_map="cuda",
                )
                logger.info("Toto pipeline loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Toto pipeline: {e}")
            self.pipeline = None
    
    def load_and_preprocess_data(self, csv_file: Path) -> pd.DataFrame:
        """Load and preprocess stock data from CSV file."""
        try:
            df = pd.read_csv(csv_file)
            df.columns = [col.title() for col in df.columns]
            
            # Ensure we have required columns
            required_cols = ['Close', 'High', 'Low', 'Open']
            for col in required_cols:
                if col not in df.columns:
                    logger.error(f"Missing required column {col} in {csv_file}")
                    return None
            
            # Remove any NaN values
            df = df.dropna()
            
            if df.empty:
                logger.warning(f"Empty data after cleaning for {csv_file}")
                return None
                
            return df
            
        except Exception as e:
            logger.error(f"Error loading {csv_file}: {e}")
            return None
    
    def preprocess_for_prediction(self, data: pd.DataFrame, key_to_predict: str) -> pd.DataFrame:
        """Preprocess data for Toto model prediction."""
        from loss_utils import percent_movements_augment
        
        newdata = data.copy(deep=True)
        newdata[key_to_predict] = percent_movements_augment(
            newdata[key_to_predict].values.reshape(-1, 1)
        )
        return newdata
    
    def generate_forecasts_for_symbol(self, symbol: str, csv_file: Path) -> Optional[Dict]:
        """Generate forecasts for a single symbol using the real predict_stock_forecasting.py logic."""
        logger.info(f"Generating forecasts for {symbol}...")
        
        # Use the real prediction logic from predict_stock_forecasting.py
        try:
            from predict_stock_forecasting import load_pipeline, load_stock_data_from_csv, pre_process_data
            from loss_utils import percent_movements_augment
            import torch
            
            # Load pipeline if not already loaded
            if self.pipeline is None:
                load_pipeline()
                from predict_stock_forecasting import pipeline
                self.pipeline = pipeline
            
            if self.pipeline is None:
                logger.error("Failed to load Toto pipeline")
                return None
            
            # Load and preprocess data using the real functions
            stock_data = load_stock_data_from_csv(csv_file)
            if stock_data is None or stock_data.empty:
                logger.warning(f"No data loaded for {symbol}")
                return None
            
            results = {}
            results['symbol'] = symbol
            
            # Process each price type using the same logic as predict_stock_forecasting.py
            for key_to_predict in ['Close', 'High', 'Low']:
                try:
                    # Preprocess data exactly like predict_stock_forecasting.py
                    data = stock_data.copy()
                    data = pre_process_data(data, "High")
                    data = pre_process_data(data, "Low") 
                    data = pre_process_data(data, "Open")
                    data = pre_process_data(data, "Close")
                    
                    price = data[["Close", "High", "Low", "Open"]]
                    price["ds"] = pd.date_range(start="1949-01-01", periods=len(price), freq="D").values
                    price['y'] = price[key_to_predict].shift(-1)
                    price.drop(price.tail(1).index, inplace=True)  # drop last row
                    
                    # Remove NaN values
                    price = price.dropna()
                    
                    if len(price) < 7:
                        logger.warning(f"Insufficient data for {symbol} {key_to_predict}")
                        continue
                    
                    # Use last 7 days as validation (like in predict_stock_forecasting.py)
                    validation = price[-7:]
                    
                    predictions = []
                    # Make 7 predictions exactly like predict_stock_forecasting.py
                    for pred_idx in reversed(range(1, 8)):
                        current_context = price[:-pred_idx]
                        context = torch.tensor(current_context["y"].values, dtype=torch.float)
                        
                        prediction_length = 1
                        forecast = self.pipeline.predict(context, prediction_length)
                        low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)
                        predictions.append(median.item())
                    
                    # Store results in the same format as predict_stock_forecasting.py
                    last_price = stock_data[key_to_predict].iloc[-1]
                    
                    results[key_to_predict.lower() + "_last_price"] = last_price
                    results[key_to_predict.lower() + "_predictions"] = predictions
                    results[key_to_predict.lower() + "_predicted_changes"] = predictions  # These are already percent changes
                    
                    # Calculate final predicted price
                    total_change = sum(predictions)
                    final_predicted_price = last_price * (1 + total_change)
                    results[key_to_predict.lower() + "_predicted_price_value"] = final_predicted_price
                    results[key_to_predict.lower() + "_total_predicted_change"] = total_change
                    
                    logger.info(f"{symbol} {key_to_predict}: {predictions[-1]:.4f} latest prediction")
                    
                except Exception as e:
                    logger.error(f"Error predicting {symbol} {key_to_predict}: {e}")
                    continue
            
            if len(results) > 1:  # More than just symbol
                results['forecast_generated_at'] = datetime.now().isoformat()
                return results
            
        except Exception as e:
            logger.error(f"Error in forecast generation for {symbol}: {e}")
        
        return None
    
    def generate_all_forecasts(self) -> Dict[str, Dict]:
        """Generate forecasts for all symbols."""
        logger.info(f"Generating forecasts for {len(self.csv_files)} symbols...")
        
        all_forecasts = {}
        
        for csv_file in self.csv_files:
            symbol = csv_file.stem.split('-')[0]
            forecast = self.generate_forecasts_for_symbol(symbol, csv_file)
            if forecast:
                all_forecasts[symbol] = forecast
        
        logger.info(f"Generated forecasts for {len(all_forecasts)} symbols")
        self.forecast_data = all_forecasts
        return all_forecasts
    
    def strategy_best_single_stock(self, forecasts: Dict) -> Dict:
        """Strategy 1: All-in on single best predicted stock."""
        logger.info("Testing strategy: All-in on single best predicted stock")
        
        best_stock = None
        best_predicted_return = float('-inf')
        
        for symbol, data in forecasts.items():
            if 'close_total_predicted_change' in data:
                predicted_return = data['close_total_predicted_change']
                if predicted_return > best_predicted_return:
                    best_predicted_return = predicted_return
                    best_stock = symbol
        
        if best_stock is None:
            return {'error': 'No valid predictions found'}
        
        allocation = {best_stock: 1.0}  # 100% allocation
        
        return {
            'strategy': 'best_single_stock',
            'allocation': allocation,
            'expected_return': best_predicted_return,
            'selected_stock': best_stock,
            'risk_level': 'High - Single asset concentration'
        }
    
    def strategy_best_two_stocks(self, forecasts: Dict) -> Dict:
        """Strategy 2: All-in on top 2 best predicted stocks (50/50 split)."""
        logger.info("Testing strategy: All-in on top 2 best predicted stocks")
        
        stock_returns = []
        for symbol, data in forecasts.items():
            if 'close_total_predicted_change' in data:
                predicted_return = data['close_total_predicted_change']
                stock_returns.append((symbol, predicted_return))
        
        # Sort by predicted return and take top 2
        stock_returns.sort(key=lambda x: x[1], reverse=True)
        top_2 = stock_returns[:2]
        
        if len(top_2) < 2:
            return {'error': 'Insufficient valid predictions for top 2 strategy'}
        
        allocation = {stock: 0.5 for stock, _ in top_2}  # 50/50 split
        expected_return = sum(ret for _, ret in top_2) * 0.5
        
        return {
            'strategy': 'best_two_stocks',
            'allocation': allocation,
            'expected_return': expected_return,
            'selected_stocks': [stock for stock, _ in top_2],
            'risk_level': 'Medium-High - Two asset concentration'
        }
    
    def strategy_weighted_portfolio(self, forecasts: Dict, top_n: int = 5) -> Dict:
        """Strategy 3: Weighted portfolio based on predicted gains (risk-weighted)."""
        logger.info(f"Testing strategy: Weighted portfolio top {top_n} picks")
        
        stock_returns = []
        for symbol, data in forecasts.items():
            if 'close_total_predicted_change' in data:
                predicted_return = data['close_total_predicted_change']
                if predicted_return > 0:  # Only positive predictions
                    stock_returns.append((symbol, predicted_return))
        
        if not stock_returns:
            return {'error': 'No positive predictions found for weighted portfolio'}
        
        # Sort by predicted return and take top N
        stock_returns.sort(key=lambda x: x[1], reverse=True)
        top_n_stocks = stock_returns[:min(top_n, len(stock_returns))]
        
        # Weight by predicted return (higher prediction = higher weight)
        total_predicted_return = sum(ret for _, ret in top_n_stocks)
        
        if total_predicted_return <= 0:
            return {'error': 'No positive total predicted return'}
        
        allocation = {}
        expected_return = 0
        
        for stock, predicted_return in top_n_stocks:
            weight = predicted_return / total_predicted_return
            allocation[stock] = weight
            expected_return += predicted_return * weight
        
        return {
            'strategy': 'weighted_portfolio',
            'allocation': allocation,
            'expected_return': expected_return,
            'num_positions': len(top_n_stocks),
            'risk_level': 'Medium - Diversified portfolio'
        }
    
    def strategy_risk_adjusted_portfolio(self, forecasts: Dict, top_n: int = 5) -> Dict:
        """Strategy 4: Risk-adjusted weighted portfolio with volatility consideration."""
        logger.info(f"Testing strategy: Risk-adjusted portfolio top {top_n} picks")
        
        stock_data = []
        for symbol, data in forecasts.items():
            if 'close_total_predicted_change' in data and 'high_total_predicted_change' in data and 'low_total_predicted_change' in data:
                predicted_return = data['close_total_predicted_change']
                if predicted_return > 0:
                    # Calculate predicted volatility as proxy for risk
                    high_change = data['high_total_predicted_change']
                    low_change = data['low_total_predicted_change']
                    volatility = abs(high_change - low_change)
                    
                    # Risk-adjusted return (return per unit of risk)
                    risk_adjusted_return = predicted_return / (volatility + 0.001)  # Small epsilon to avoid division by zero
                    
                    stock_data.append((symbol, predicted_return, volatility, risk_adjusted_return))
        
        if not stock_data:
            return {'error': 'Insufficient data for risk-adjusted portfolio'}
        
        # Sort by risk-adjusted return
        stock_data.sort(key=lambda x: x[3], reverse=True)
        top_stocks = stock_data[:min(top_n, len(stock_data))]
        
        # Weight by risk-adjusted return
        total_risk_adjusted = sum(risk_adj for _, _, _, risk_adj in top_stocks)
        
        if total_risk_adjusted <= 0:
            return {'error': 'No positive risk-adjusted returns'}
        
        allocation = {}
        expected_return = 0
        total_risk = 0
        
        for stock, ret, vol, risk_adj in top_stocks:
            weight = risk_adj / total_risk_adjusted
            allocation[stock] = weight
            expected_return += ret * weight
            total_risk += vol * weight
        
        return {
            'strategy': 'risk_adjusted_portfolio',
            'allocation': allocation,
            'expected_return': expected_return,
            'expected_volatility': total_risk,
            'sharpe_proxy': expected_return / (total_risk + 0.001),
            'num_positions': len(top_stocks),
            'risk_level': 'Medium-Low - Risk-adjusted diversification'
        }
    
    def simulate_portfolio_performance(self, strategy_result: Dict, days_ahead: int = 5) -> Dict:
        """Simulate portfolio performance (placeholder - would need actual future data)."""
        if 'allocation' not in strategy_result:
            return strategy_result
        
        # This is a simulation - in real implementation, you'd track actual performance
        # For now, we'll use the predicted returns as a proxy
        simulated_return = strategy_result.get('expected_return', 0)
        
        # Add some realistic noise/variance to the simulation
        np.random.seed(42)  # For reproducible results
        actual_return = simulated_return + np.random.normal(0, abs(simulated_return) * 0.3)
        
        performance = {
            'predicted_return': simulated_return,
            'simulated_actual_return': actual_return,
            'outperformance': actual_return - simulated_return,
            'capital_after': self.initial_capital * (1 + actual_return),
            'profit_loss': self.initial_capital * actual_return
        }
        
        strategy_result['performance'] = performance
        return strategy_result
    
    def run_comprehensive_strategy_test(self) -> Dict:
        """Run comprehensive test of all trading strategies."""
        logger.info("Running comprehensive trading strategy simulation...")
        
        # Generate forecasts for all symbols
        forecasts = self.generate_all_forecasts()
        
        if not forecasts:
            logger.error("No forecasts generated - cannot run strategies")
            return {}
        
        # Test all strategies
        strategies = {}
        
        # Strategy 1: Best single stock
        strategies['best_single'] = self.strategy_best_single_stock(forecasts)
        strategies['best_single'] = self.simulate_portfolio_performance(strategies['best_single'])
        
        # Strategy 2: Best two stocks
        strategies['best_two'] = self.strategy_best_two_stocks(forecasts)
        strategies['best_two'] = self.simulate_portfolio_performance(strategies['best_two'])
        
        # Strategy 3: Weighted portfolio
        strategies['weighted_top5'] = self.strategy_weighted_portfolio(forecasts, top_n=5)
        strategies['weighted_top5'] = self.simulate_portfolio_performance(strategies['weighted_top5'])
        
        # Strategy 4: Risk-adjusted portfolio
        strategies['risk_adjusted'] = self.strategy_risk_adjusted_portfolio(forecasts, top_n=5)
        strategies['risk_adjusted'] = self.simulate_portfolio_performance(strategies['risk_adjusted'])
        
        # Additional variations
        strategies['weighted_top3'] = self.strategy_weighted_portfolio(forecasts, top_n=3)
        strategies['weighted_top3'] = self.simulate_portfolio_performance(strategies['weighted_top3'])
        
        self.results = {
            'forecasts': forecasts,
            'strategies': strategies,
            'simulation_params': {
                'initial_capital': self.initial_capital,
                'forecast_days': self.forecast_days,
                'symbols_available': self.symbols,
                'simulation_date': datetime.now().isoformat()
            }
        }
        
        return self.results
    
    def save_results(self, filename: Optional[str] = None):
        """Save results to CSV and JSON files."""
        if not self.results:
            logger.error("No results to save")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if filename is None:
            base_filename = f"trading_simulation_{timestamp}"
        else:
            base_filename = filename
        
        # Save strategy comparison CSV
        strategies_data = []
        for strategy_name, strategy_data in self.results['strategies'].items():
            if 'error' not in strategy_data and 'allocation' in strategy_data:
                perf = strategy_data.get('performance', {})
                
                row = {
                    'strategy': strategy_name,
                    'expected_return': strategy_data.get('expected_return', 0),
                    'simulated_return': perf.get('simulated_actual_return', 0),
                    'profit_loss': perf.get('profit_loss', 0),
                    'risk_level': strategy_data.get('risk_level', 'Unknown'),
                    'num_positions': strategy_data.get('num_positions', len(strategy_data.get('allocation', {}))),
                    'top_allocation': max(strategy_data.get('allocation', {}).values()) if strategy_data.get('allocation') else 0
                }
                
                # Add individual allocations
                for symbol, weight in strategy_data.get('allocation', {}).items():
                    row[f'allocation_{symbol}'] = weight
                
                strategies_data.append(row)
        
        strategies_df = pd.DataFrame(strategies_data)
        csv_file = f"{base_filename}_strategies.csv"
        strategies_df.to_csv(csv_file, index=False)
        logger.info(f"Strategy results saved to {csv_file}")
        
        # Save detailed forecasts CSV
        forecasts_data = []
        for symbol, forecast_data in self.results['forecasts'].items():
            if 'close_total_predicted_change' in forecast_data:
                row = {
                    'symbol': symbol,
                    'last_close_price': forecast_data.get('close_last_price', 0),
                    'predicted_change': forecast_data['close_total_predicted_change'],
                    'predicted_final_price': forecast_data.get('close_predicted_price_value', 0),
                }
                
                # Add daily predictions if available
                if 'close_predictions' in forecast_data:
                    for i, change in enumerate(forecast_data['close_predictions']):
                        row[f'day_{i+1}_change'] = change
                
                forecasts_data.append(row)
        
        forecasts_df = pd.DataFrame(forecasts_data)
        forecasts_csv = f"{base_filename}_forecasts.csv"
        forecasts_df.to_csv(forecasts_csv, index=False)
        logger.info(f"Forecast results saved to {forecasts_csv}")
        
        return csv_file, forecasts_csv
    
    def print_summary(self):
        """Print summary of strategy performance."""
        if not self.results:
            logger.error("No results to summarize")
            return
        
        print("\n" + "="*80)
        print("TRADING STRATEGY SIMULATION SUMMARY")
        print("="*80)
        
        print(f"\nSimulation Parameters:")
        params = self.results['simulation_params']
        print(f"  Initial Capital: ${params['initial_capital']:,.2f}")
        print(f"  Forecast Days: {params['forecast_days']}")
        print(f"  Symbols Available: {len(params['symbols_available'])}")
        
        print(f"\nForecasts Generated: {len(self.results['forecasts'])}")
        
        print("\nStrategy Performance:")
        print("-" * 80)
        
        for strategy_name, strategy_data in self.results['strategies'].items():
            if 'error' in strategy_data:
                print(f"{strategy_name:20} ERROR: {strategy_data['error']}")
                continue
            
            perf = strategy_data.get('performance', {})
            
            print(f"\n{strategy_name.upper().replace('_', ' ')}:")
            print(f"  Expected Return: {strategy_data.get('expected_return', 0):8.4f} ({strategy_data.get('expected_return', 0)*100:.2f}%)")
            if perf:
                print(f"  Simulated Return: {perf.get('simulated_actual_return', 0):7.4f} ({perf.get('simulated_actual_return', 0)*100:.2f}%)")
                print(f"  Profit/Loss: ${perf.get('profit_loss', 0):11,.2f}")
                print(f"  Final Capital: ${perf.get('capital_after', 0):9,.2f}")
            print(f"  Risk Level: {strategy_data.get('risk_level', 'Unknown')}")
            print(f"  Positions: {strategy_data.get('num_positions', 'N/A')}")
            
            # Show top allocations
            allocation = strategy_data.get('allocation', {})
            if allocation:
                sorted_allocation = sorted(allocation.items(), key=lambda x: x[1], reverse=True)
                print(f"  Top Allocations:")
                for symbol, weight in sorted_allocation[:3]:  # Show top 3
                    print(f"    {symbol}: {weight:.3f} ({weight*100:.1f}%)")


def main():
    """Main execution function."""
    logger.info("Starting trading strategy simulation...")
    
    # Create simulator
    simulator = TradingSimulator(
        backtestdata_dir="backtestdata",
        forecast_days=5,
        initial_capital=100000
    )
    
    try:
        # Run comprehensive test
        results = simulator.run_comprehensive_strategy_test()
        
        if not results:
            logger.error("No results generated")
            return
        
        # Print summary
        simulator.print_summary()
        
        # Save results
        csv_file, forecasts_csv = simulator.save_results()
        
        # Create visualizations
        logger.info("Creating comprehensive visualizations...")
        viz_files = simulator.viz_logger.create_all_visualizations(results)
        
        print(f"\n" + "="*80)
        print(f"Results saved to: {csv_file} and {forecasts_csv}")
        print(f"Visualizations created:")
        for viz_file in viz_files:
            print(f"  - {viz_file}")
        print(f"TensorBoard logs: {simulator.viz_logger.tb_writer.log_dir}")
        print("="*80)
        
        # Close visualization logger
        simulator.viz_logger.close()
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise


if __name__ == "__main__":
    main()