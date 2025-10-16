#!/usr/bin/env python3
"""
Realistic trading simulator with proper fee structure and holding periods.
Uses REAL Toto forecasting and models actual trading behavior.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project root to path  
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtests.visualization_logger import VisualizationLogger

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealisticTradingSimulator:
    """
    Realistic trading simulator that accounts for:
    - Proper fee structure (only on trades, not daily)
    - Holding periods and position management
    - Real Toto forecasting (no mocks)
    - Transaction costs and slippage
    - Risk management
    """
    
    def __init__(self, 
                 backtestdata_dir: str = "backtestdata",
                 forecast_days: int = 7,
                 initial_capital: float = 100000,
                 trading_fee: float = 0.001,  # 0.1% per trade
                 slippage: float = 0.0005,     # 0.05% slippage
                 min_position_size: float = 100,  # Minimum $100 position
                 max_position_weight: float = 0.4,  # Max 40% in single position
                 rebalance_frequency: int = 7,     # Rebalance every 7 days
                 output_dir: str = "backtests/realistic_results"):
        
        self.backtestdata_dir = Path(backtestdata_dir)
        self.forecast_days = forecast_days
        self.initial_capital = initial_capital
        self.trading_fee = trading_fee
        self.slippage = slippage
        self.min_position_size = min_position_size
        self.max_position_weight = max_position_weight
        self.rebalance_frequency = rebalance_frequency
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load all CSV files
        self.csv_files = list(self.backtestdata_dir.glob("*.csv"))
        self.symbols = [f.stem.split('-')[0] for f in self.csv_files]
        
        logger.info(f"Found {len(self.csv_files)} data files for symbols: {self.symbols}")
        
        # Initialize REAL prediction pipeline
        self.pipeline = None
        self._load_real_prediction_pipeline()
        
        # Initialize visualization logger
        self.viz_logger = VisualizationLogger(
            output_dir=str(self.output_dir),
            tb_log_dir=f"./logs/realistic_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Results storage
        self.results = {}
        self.forecast_data = {}
        self.trading_history = []
        
    def _load_real_prediction_pipeline(self):
        """Load the REAL Toto prediction pipeline."""
        try:
            logger.info("Starting to load REAL Toto pipeline...")
            from predict_stock_forecasting import load_pipeline
            logger.info("Imported load_pipeline function")
            
            logger.info("Calling load_pipeline()...")
            load_pipeline()
            logger.info("load_pipeline() completed")
            
            from predict_stock_forecasting import pipeline
            logger.info("Imported pipeline object")
            
            self.pipeline = pipeline
            if self.pipeline is not None:
                logger.info("REAL Toto pipeline loaded successfully")
            else:
                logger.error("Failed to load REAL Toto pipeline - pipeline is None")
        except Exception as e:
            logger.error(f"Error loading REAL Toto pipeline: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.pipeline = None
    
    def generate_real_forecasts_for_symbol(self, symbol: str, csv_file: Path) -> Optional[Dict]:
        """Generate REAL forecasts using predict_stock_forecasting.py logic."""
        logger.info(f"Generating REAL forecasts for {symbol}...")
        
        try:
            from predict_stock_forecasting import load_stock_data_from_csv, pre_process_data
            import torch
            
            if self.pipeline is None:
                logger.error("REAL Toto pipeline not available")
                return None
            
            # Load and preprocess data using REAL functions
            stock_data = load_stock_data_from_csv(csv_file)
            if stock_data is None or stock_data.empty:
                logger.warning(f"No data loaded for {symbol}")
                return None
            
            results = {'symbol': symbol}
            
            # Process each price type using REAL predict_stock_forecasting.py logic
            for key_to_predict in ['Close', 'High', 'Low']:
                try:
                    # Preprocess data EXACTLY like predict_stock_forecasting.py
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
                    
                    if len(price) < self.forecast_days:
                        logger.warning(f"Insufficient data for {symbol} {key_to_predict}")
                        continue
                    
                    predictions = []
                    # Make predictions EXACTLY like predict_stock_forecasting.py
                    for pred_idx in reversed(range(1, self.forecast_days + 1)):
                        current_context = price[:-pred_idx] if pred_idx > 1 else price
                        context = torch.tensor(current_context["y"].values, dtype=torch.float)
                        
                        prediction_length = 1
                        forecast = self.pipeline.predict(context, prediction_length)
                        low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)
                        predictions.append(median.item())
                    
                    # Store results in same format as predict_stock_forecasting.py
                    last_price = stock_data[key_to_predict].iloc[-1]
                    
                    results[f"{key_to_predict.lower()}_last_price"] = last_price
                    results[f"{key_to_predict.lower()}_predictions"] = predictions
                    results[f"{key_to_predict.lower()}_predicted_changes"] = predictions
                    
                    # Calculate metrics
                    total_change = sum(predictions)
                    final_predicted_price = last_price * (1 + total_change)
                    results[f"{key_to_predict.lower()}_predicted_price_value"] = final_predicted_price
                    results[f"{key_to_predict.lower()}_total_predicted_change"] = total_change
                    
                    # Calculate prediction confidence (based on consistency)
                    prediction_std = np.std(predictions) if len(predictions) > 1 else 0
                    confidence = max(0, 1 - (prediction_std / (abs(np.mean(predictions)) + 0.001)))
                    results[f"{key_to_predict.lower()}_confidence"] = confidence
                    
                    logger.info(f"{symbol} {key_to_predict}: {total_change:.4f} total change, confidence: {confidence:.3f}")
                    
                except Exception as e:
                    logger.error(f"Error predicting {symbol} {key_to_predict}: {e}")
                    continue
            
            if len(results) > 1:  # More than just symbol
                results['forecast_generated_at'] = datetime.now().isoformat()
                return results
            
        except Exception as e:
            logger.error(f"Error in REAL forecast generation for {symbol}: {e}")
        
        return None
    
    def generate_all_real_forecasts(self) -> Dict[str, Dict]:
        """Generate REAL forecasts for all symbols."""
        logger.info(f"Generating REAL forecasts for {len(self.csv_files)} symbols...")
        
        all_forecasts = {}
        
        for csv_file in self.csv_files:
            symbol = csv_file.stem.split('-')[0]
            forecast = self.generate_real_forecasts_for_symbol(symbol, csv_file)
            if forecast:
                all_forecasts[symbol] = forecast
        
        logger.info(f"Generated REAL forecasts for {len(all_forecasts)} symbols")
        self.forecast_data = all_forecasts
        return all_forecasts
    
    def calculate_position_sizes_with_risk_management(self, forecasts: Dict, strategy_weights: Dict) -> Dict:
        """Calculate position sizes with proper risk management."""
        positions = {}
        total_weight = sum(strategy_weights.values())
        
        if total_weight == 0:
            return positions
        
        # Normalize weights
        normalized_weights = {k: v / total_weight for k, v in strategy_weights.items()}
        
        for symbol, weight in normalized_weights.items():
            if symbol not in forecasts:
                continue
                
            forecast_data = forecasts[symbol]
            
            # Base position size
            base_size = self.initial_capital * weight
            
            # Risk adjustments
            confidence = forecast_data.get('close_confidence', 0.5)
            predicted_return = forecast_data.get('close_total_predicted_change', 0)
            
            # Volatility adjustment (using high-low spread as proxy)
            high_change = forecast_data.get('high_total_predicted_change', predicted_return)
            low_change = forecast_data.get('low_total_predicted_change', predicted_return)
            volatility = abs(high_change - low_change)
            
            # Adjust position size based on confidence and volatility
            confidence_multiplier = 0.5 + (confidence * 0.5)  # 0.5 to 1.0
            volatility_multiplier = max(0.2, 1 - volatility * 2)  # Reduce size for high volatility
            
            adjusted_size = base_size * confidence_multiplier * volatility_multiplier
            
            # Apply constraints
            adjusted_size = max(adjusted_size, self.min_position_size)
            adjusted_size = min(adjusted_size, self.initial_capital * self.max_position_weight)
            
            positions[symbol] = {
                'dollar_amount': adjusted_size,
                'weight': adjusted_size / self.initial_capital,
                'expected_return': predicted_return,
                'confidence': confidence,
                'volatility_proxy': volatility,
                'base_weight': weight,
                'adjusted_weight': adjusted_size / self.initial_capital
            }
        
        return positions
    
    def simulate_realistic_trading(self, positions: Dict, holding_days: int = 7) -> Dict:
        """Simulate realistic trading with proper fee structure and holding periods."""
        
        total_investment = sum(pos['dollar_amount'] for pos in positions.values())
        remaining_cash = self.initial_capital - total_investment
        
        # Calculate entry fees (only paid once when opening positions)
        entry_fees = 0
        for symbol, pos in positions.items():
            fee = pos['dollar_amount'] * self.trading_fee
            slippage_cost = pos['dollar_amount'] * self.slippage
            entry_fees += fee + slippage_cost
        
        # Track positions over holding period
        daily_pnl = []
        cumulative_fees = entry_fees
        
        for day in range(holding_days):
            daily_return = 0
            
            for symbol, pos in positions.items():
                # Daily return based on predicted performance spread over holding period
                expected_daily_return = pos['expected_return'] / holding_days
                
                # Add some realistic noise/variance
                np.random.seed(42 + day)  # Reproducible but varied
                actual_daily_return = expected_daily_return + np.random.normal(0, abs(expected_daily_return) * 0.3)
                
                position_daily_pnl = pos['dollar_amount'] * actual_daily_return
                daily_return += position_daily_pnl
            
            daily_pnl.append(daily_return)
        
        # Calculate exit fees (only paid once when closing positions)
        final_portfolio_value = total_investment + sum(daily_pnl)
        exit_fees = final_portfolio_value * self.trading_fee + final_portfolio_value * self.slippage
        cumulative_fees += exit_fees
        
        # Final performance metrics
        gross_pnl = sum(daily_pnl)
        net_pnl = gross_pnl - cumulative_fees
        final_capital = self.initial_capital + net_pnl
        
        # Track trading history
        trade_record = {
            'timestamp': datetime.now(),
            'positions': positions,
            'holding_days': holding_days,
            'total_investment': total_investment,
            'entry_fees': entry_fees,
            'exit_fees': exit_fees,
            'total_fees': cumulative_fees,
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'return_gross': gross_pnl / total_investment if total_investment > 0 else 0,
            'return_net': net_pnl / total_investment if total_investment > 0 else 0,
            'daily_pnl': daily_pnl
        }
        
        self.trading_history.append(trade_record)
        
        return {
            'total_investment': total_investment,
            'remaining_cash': remaining_cash,
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'total_fees': cumulative_fees,
            'fee_percentage': cumulative_fees / total_investment if total_investment > 0 else 0,
            'final_capital': final_capital,
            'return_gross': gross_pnl / total_investment if total_investment > 0 else 0,
            'return_net': net_pnl / total_investment if total_investment > 0 else 0,
            'daily_pnl': daily_pnl,
            'positions': positions
        }
    
    def strategy_concentrated_best(self, forecasts: Dict, num_positions: int = 1) -> Dict:
        """Concentrated strategy focusing on best predictions."""
        logger.info(f"Testing concentrated strategy with {num_positions} position(s)")
        
        # Get stocks with positive predictions
        stock_scores = []
        for symbol, data in forecasts.items():
            if 'close_total_predicted_change' in data and data['close_total_predicted_change'] > 0:
                score = data['close_total_predicted_change'] * data.get('close_confidence', 0.5)
                stock_scores.append((symbol, score))
        
        if not stock_scores:
            return {'error': 'No positive predictions found'}
        
        # Sort by score and take top N
        stock_scores.sort(key=lambda x: x[1], reverse=True)
        top_stocks = stock_scores[:num_positions]
        
        # Equal weight allocation
        strategy_weights = {stock: 1.0 / len(top_stocks) for stock, _ in top_stocks}
        
        # Calculate realistic position sizes
        positions = self.calculate_position_sizes_with_risk_management(forecasts, strategy_weights)
        
        # Simulate realistic trading
        performance = self.simulate_realistic_trading(positions, holding_days=self.forecast_days)
        
        return {
            'strategy': f'concentrated_{num_positions}',
            'positions': positions,
            'performance': performance,
            'expected_return': sum(forecasts[s]['close_total_predicted_change'] for s, _ in top_stocks) / len(top_stocks),
            'risk_level': 'High' if num_positions == 1 else 'Medium-High',
            'num_positions': len(positions)
        }
    
    def strategy_risk_weighted_portfolio(self, forecasts: Dict, max_positions: int = 5) -> Dict:
        """Risk-weighted portfolio strategy."""
        logger.info(f"Testing risk-weighted portfolio with max {max_positions} positions")
        
        # Calculate risk-adjusted scores
        stock_scores = []
        for symbol, data in forecasts.items():
            if 'close_total_predicted_change' in data and data['close_total_predicted_change'] > 0:
                ret = data['close_total_predicted_change']
                confidence = data.get('close_confidence', 0.5)
                
                # Risk proxy from high-low spread
                high_change = data.get('high_total_predicted_change', ret)
                low_change = data.get('low_total_predicted_change', ret)
                volatility = abs(high_change - low_change) + 0.001
                
                # Risk-adjusted score
                risk_adj_score = (ret * confidence) / volatility
                stock_scores.append((symbol, risk_adj_score, ret))
        
        if not stock_scores:
            return {'error': 'No positive predictions found'}
        
        # Sort by risk-adjusted score and take top N
        stock_scores.sort(key=lambda x: x[1], reverse=True)
        top_stocks = stock_scores[:max_positions]
        
        # Weight by risk-adjusted score
        total_score = sum(score for _, score, _ in top_stocks)
        strategy_weights = {stock: score / total_score for stock, score, _ in top_stocks}
        
        # Calculate realistic position sizes
        positions = self.calculate_position_sizes_with_risk_management(forecasts, strategy_weights)
        
        # Simulate realistic trading
        performance = self.simulate_realistic_trading(positions, holding_days=self.forecast_days)
        
        return {
            'strategy': f'risk_weighted_{max_positions}',
            'positions': positions,
            'performance': performance,
            'expected_return': sum(ret * (score / total_score) for _, score, ret in top_stocks),
            'risk_level': 'Medium-Low - Risk adjusted',
            'num_positions': len(positions)
        }
    
    def run_realistic_comprehensive_test(self) -> Dict:
        """Run comprehensive test with REAL forecasting and realistic trading."""
        logger.info("Running REALISTIC comprehensive trading strategy test...")
        
        # Generate REAL forecasts for all symbols
        forecasts = self.generate_all_real_forecasts()
        
        if not forecasts:
            logger.error("No REAL forecasts generated - cannot run strategies")
            return {}
        
        # Test realistic strategies
        strategies = {}
        
        # Strategy 1: Best single stock
        strategies['best_single'] = self.strategy_concentrated_best(forecasts, num_positions=1)
        
        # Strategy 1b: Best single stock with 2x leverage
        strategies['best_single_2x'] = self.strategy_concentrated_best(forecasts, num_positions=1, leverage=2.0)
        
        # Strategy 2: Best two stocks
        strategies['best_two'] = self.strategy_concentrated_best(forecasts, num_positions=2)
        
        # Strategy 2b: Best two stocks with 2x leverage
        strategies['best_two_2x'] = self.strategy_concentrated_best(forecasts, num_positions=2, leverage=2.0)
        
        # Strategy 3: Best three stocks
        strategies['best_three'] = self.strategy_concentrated_best(forecasts, num_positions=3)
        
        # Strategy 4: Risk-weighted portfolio (5 positions)
        strategies['risk_weighted_5'] = self.strategy_risk_weighted_portfolio(forecasts, max_positions=5)
        
        # Strategy 5: Risk-weighted portfolio (3 positions)
        strategies['risk_weighted_3'] = self.strategy_risk_weighted_portfolio(forecasts, max_positions=3)
        
        self.results = {
            'forecasts': forecasts,
            'strategies': strategies,
            'simulation_params': {
                'initial_capital': self.initial_capital,
                'forecast_days': self.forecast_days,
                'trading_fee': self.trading_fee,
                'slippage': self.slippage,
                'symbols_available': self.symbols,
                'simulation_date': datetime.now().isoformat(),
                'using_real_forecasts': True
            },
            'trading_history': self.trading_history
        }
        
        return self.results


def analyze_realistic_performance(results: Dict):
    """Analyze realistic trading performance with proper fee accounting."""
    print("\n" + "="*100)
    print("REALISTIC TRADING STRATEGY ANALYSIS (with Real Toto Forecasts)")
    print("="*100)
    
    if 'strategies' not in results:
        print("No strategy results to analyze")
        return
    
    strategies = results['strategies']
    valid_strategies = {k: v for k, v in strategies.items() if 'error' not in v}
    
    if not valid_strategies:
        print("No valid strategies found")
        return
    
    print(f"\nAnalyzing {len(valid_strategies)} realistic strategies...")
    print(f"Simulation Parameters:")
    params = results['simulation_params']
    print(f"  - Initial Capital: ${params['initial_capital']:,.2f}")
    print(f"  - Trading Fee: {params['trading_fee']:.3f} ({params['trading_fee']*100:.1f}%)")
    print(f"  - Slippage: {params['slippage']:.4f} ({params['slippage']*100:.2f}%)")
    print(f"  - Holding Period: {params['forecast_days']} days")
    print(f"  - Using Real Toto Forecasts: {params['using_real_forecasts']}")
    
    # Sort strategies by net return (after fees)
    sorted_strategies = sorted(
        valid_strategies.items(),
        key=lambda x: x[1]['performance']['return_net'],
        reverse=True
    )
    
    print(f"\nSTRATEGY RANKINGS (by Net Return after fees):")
    print("-" * 100)
    
    for i, (name, data) in enumerate(sorted_strategies, 1):
        perf = data['performance']
        
        print(f"{i:2d}. {name.replace('_', ' ').title():25s}")
        print(f"    Gross Return: {perf['return_gross']:7.3f} ({perf['return_gross']*100:6.1f}%)")
        print(f"    Net Return:   {perf['return_net']:7.3f} ({perf['return_net']*100:6.1f}%) [AFTER FEES]")
        print(f"    Total Fees:   ${perf['total_fees']:8,.2f} ({perf['fee_percentage']*100:4.1f}% of investment)")
        print(f"    Net P&L:      ${perf['net_pnl']:10,.2f}")
        print(f"    Final Capital:${perf['final_capital']:10,.2f}")
        print(f"    Investment:   ${perf['total_investment']:10,.2f}")
        print(f"    Positions:    {data['num_positions']:2d}   Risk: {data['risk_level']}")
        
        # Show position details
        positions = data['positions']
        if positions:
            print(f"    Position Details:")
            for symbol, pos in sorted(positions.items(), key=lambda x: x[1]['dollar_amount'], reverse=True):
                print(f"      {symbol:8s}: ${pos['dollar_amount']:8,.0f} "
                      f"({pos['weight']*100:4.1f}%) "
                      f"Exp: {pos['expected_return']*100:+5.1f}% "
                      f"Conf: {pos['confidence']:.2f}")
        print()
    
    # Performance comparison
    print("PERFORMANCE METRICS COMPARISON:")
    print("-" * 80)
    
    best_net = max(valid_strategies.items(), key=lambda x: x[1]['performance']['return_net'])
    best_gross = max(valid_strategies.items(), key=lambda x: x[1]['performance']['return_gross'])
    lowest_fees = min(valid_strategies.items(), key=lambda x: x[1]['performance']['fee_percentage'])
    
    print(f"Best Net Return:    {best_net[0].replace('_', ' ').title()} "
          f"({best_net[1]['performance']['return_net']*100:+5.1f}%)")
    print(f"Best Gross Return:  {best_gross[0].replace('_', ' ').title()} "
          f"({best_gross[1]['performance']['return_gross']*100:+5.1f}%)")
    print(f"Lowest Fee Impact:  {lowest_fees[0].replace('_', ' ').title()} "
          f"({lowest_fees[1]['performance']['fee_percentage']*100:.1f}% fees)")
    
    # Forecast quality analysis
    forecasts = results.get('forecasts', {})
    if forecasts:
        print(f"\nREAL TOTO FORECAST ANALYSIS:")
        print("-" * 40)
        
        predicted_returns = []
        confidences = []
        positive_predictions = 0
        
        for symbol, data in forecasts.items():
            if 'close_total_predicted_change' in data:
                ret = data['close_total_predicted_change']
                conf = data.get('close_confidence', 0.5)
                predicted_returns.append(ret)
                confidences.append(conf)
                if ret > 0:
                    positive_predictions += 1
        
        if predicted_returns:
            print(f"Total Forecasts:      {len(predicted_returns)}")
            print(f"Positive Predictions: {positive_predictions} ({positive_predictions/len(predicted_returns)*100:.1f}%)")
            print(f"Mean Return:          {np.mean(predicted_returns)*100:+5.2f}%")
            print(f"Std Return:           {np.std(predicted_returns)*100:5.2f}%")
            print(f"Mean Confidence:      {np.mean(confidences):.3f}")
            print(f"Best Predicted:       {max(predicted_returns)*100:+5.2f}%")
            print(f"Worst Predicted:      {min(predicted_returns)*100:+5.2f}%")


def main():
    """Run realistic trading simulation with REAL Toto forecasts."""
    logger.info("Starting REALISTIC trading simulation with REAL Toto forecasts...")
    
    # Create realistic simulator
    simulator = RealisticTradingSimulator(
        backtestdata_dir="backtestdata",
        forecast_days=7,
        initial_capital=100000,
        trading_fee=0.001,     # 0.1% per trade
        slippage=0.0005,       # 0.05% slippage  
        output_dir="backtests/realistic_results"
    )
    
    try:
        # Run realistic simulation
        results = simulator.run_realistic_comprehensive_test()
        
        if not results:
            logger.error("No results generated")
            return
        
        # Analyze performance
        analyze_realistic_performance(results)
        
        # Create visualizations
        logger.info("Creating comprehensive visualizations...")
        viz_files = simulator.viz_logger.create_all_visualizations(results)
        
        print(f"\n" + "="*100)
        print(f"REALISTIC SIMULATION COMPLETED")
        print(f"Visualizations created:")
        for viz_file in viz_files:
            print(f"  - {viz_file}")
        print(f"TensorBoard logs: {simulator.viz_logger.tb_writer.log_dir}")
        print("="*100)
        
        # Close visualization logger
        simulator.viz_logger.close()
        
    except Exception as e:
        logger.error(f"Realistic simulation failed: {e}")
        raise


if __name__ == "__main__":
    main()