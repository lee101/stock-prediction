#!/usr/bin/env python3
"""
Simplified Portfolio Simulation System

This system simulates different portfolio allocation strategies using existing
prediction CSV files to test which allocation approach works best over time.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import logging
from loguru import logger
import json
import glob
import re

@dataclass
class PortfolioState:
    """Represents the current state of a portfolio"""
    cash: float = 100000.0  # Starting cash
    positions: Dict[str, float] = field(default_factory=dict)  # symbol -> quantity
    position_values: Dict[str, float] = field(default_factory=dict)  # symbol -> current value
    daily_returns: List[float] = field(default_factory=list)
    total_trades: int = 0
    winning_trades: int = 0
    
    @property
    def total_value(self) -> float:
        return self.cash + sum(self.position_values.values())
    
    @property
    def win_rate(self) -> float:
        return self.winning_trades / max(self.total_trades, 1)

@dataclass 
class AllocationStrategy:
    """Defines a portfolio allocation strategy"""
    name: str
    max_positions: int
    max_position_size: float  # As fraction of portfolio
    rebalance_threshold: float = 0.1

class PortfolioSimulation:
    """Portfolio simulation using existing prediction data"""
    
    def __init__(self, initial_cash: float = 100000.0):
        self.initial_cash = initial_cash
        self.results_dir = Path("results")
        self.prediction_files = []
        self.load_prediction_files()
        
        # Define allocation strategies to test
        self.strategies = [
            AllocationStrategy("single_best", max_positions=1, max_position_size=0.95),
            AllocationStrategy("dual_best", max_positions=2, max_position_size=0.47),  
            AllocationStrategy("balanced_3", max_positions=3, max_position_size=0.32),
            AllocationStrategy("diversified_5", max_positions=5, max_position_size=0.19),
        ]
        
    def load_prediction_files(self):
        """Load all prediction CSV files sorted by date"""
        pattern = str(self.results_dir / "predictions-*.csv")
        files = glob.glob(pattern)
        
        # Extract dates and sort - only include 2024 files for better format consistency
        file_data = []
        for file in files:
            # Extract date from filename like "predictions-2024-06-22_10-00-18.csv"
            match = re.search(r'predictions-(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})\.csv', file)
            if match:
                date_str = match.group(1)
                time_str = match.group(2)
                
                # Only include 2024 files
                if not date_str.startswith('2024'):
                    continue
                    
                try:
                    date_obj = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H-%M-%S")
                    file_data.append((date_obj, file))
                except:
                    continue
                    
        # Sort by date
        file_data.sort(key=lambda x: x[0])
        self.prediction_files = [(date, file) for date, file in file_data]
        
        logger.info(f"Loaded {len(self.prediction_files)} prediction files")
        if self.prediction_files:
            logger.info(f"Date range: {self.prediction_files[0][0]} to {self.prediction_files[-1][0]}")
    
    def load_predictions(self, file_path: str) -> Dict[str, Dict]:
        """Load predictions from a CSV file"""
        try:
            df = pd.read_csv(file_path)
            predictions = {}
            
            for _, row in df.iterrows():
                symbol = row['instrument']
                
                # Extract key metrics for trading decisions
                predictions[symbol] = {
                    'close_last_price': float(row['close_last_price']),
                    'close_predicted_price_value': float(row['close_predicted_price_value']),
                    'high_predicted_price_value': float(row['high_predicted_price_value']),
                    'low_predicted_price_value': float(row['low_predicted_price_value']),
                    
                    # Trading strategy returns
                    'takeprofit_profit': float(row.get('takeprofit_profit', 0)),
                    'maxdiffprofit_profit': float(row.get('maxdiffprofit_profit', 0)),
                    'entry_takeprofit_profit': float(row.get('entry_takeprofit_profit', 0)),
                    
                    # Calculate expected return based on best strategy
                    'expected_return': max(
                        float(row.get('takeprofit_profit', 0)),
                        float(row.get('maxdiffprofit_profit', 0)),
                        float(row.get('entry_takeprofit_profit', 0))
                    ),
                    
                    'predicted_movement': (float(row['close_predicted_price_value']) - float(row['close_last_price'])) / float(row['close_last_price'])
                }
                
            return predictions
            
        except Exception as e:
            logger.warning(f"Could not load predictions from {file_path}: {e}")
            return {}
    
    def select_positions(self, predictions: Dict, strategy: AllocationStrategy) -> List[str]:
        """Select which positions to hold based on predictions and allocation strategy"""
        
        # Filter to symbols with positive expected return
        profitable_symbols = {k: v for k, v in predictions.items() 
                            if v['expected_return'] > 0}
        
        # Sort by expected return
        sorted_symbols = sorted(profitable_symbols.keys(), 
                              key=lambda s: profitable_symbols[s]['expected_return'], 
                              reverse=True)
        
        # Select top N based on strategy
        selected = sorted_symbols[:strategy.max_positions]
        
        logger.info(f"Selected positions for {strategy.name}: {selected}")
        return selected
    
    def simulate_day(self, 
                    portfolio: PortfolioState,
                    predictions: Dict,
                    strategy: AllocationStrategy) -> List[Dict]:
        """Simulate one day of trading"""
        trades = []
        
        # Update current position values with "current" prices (last prices)
        for symbol in list(portfolio.positions.keys()):
            if symbol in predictions and portfolio.positions[symbol] != 0:
                current_price = predictions[symbol]['close_last_price']
                portfolio.position_values[symbol] = portfolio.positions[symbol] * current_price
        
        # Select target positions
        target_positions = self.select_positions(predictions, strategy)
        
        # Close positions not in target
        for symbol in list(portfolio.positions.keys()):
            if symbol not in target_positions and portfolio.positions[symbol] != 0:
                if symbol in predictions:
                    current_price = predictions[symbol]['close_last_price']
                    # Sell position
                    sell_value = portfolio.positions[symbol] * current_price
                    portfolio.cash += sell_value
                    
                    trades.append({
                        "symbol": symbol,
                        "action": "sell",
                        "quantity": portfolio.positions[symbol],
                        "price": current_price,
                        "value": sell_value
                    })
                    
                    portfolio.positions[symbol] = 0
                    portfolio.position_values[symbol] = 0
                    portfolio.total_trades += 1
        
        # Open/adjust positions for targets
        if target_positions:
            position_allocation = portfolio.total_value * strategy.max_position_size
            
            for symbol in target_positions:
                if symbol in predictions:
                    current_price = predictions[symbol]['close_last_price']
                    target_quantity = position_allocation / current_price
                    current_quantity = portfolio.positions.get(symbol, 0)
                    quantity_diff = target_quantity - current_quantity
                    
                    if abs(quantity_diff * current_price) > 100:  # Minimum $100 trade
                        if quantity_diff > 0:
                            # Buy more
                            trade_value = quantity_diff * current_price
                            if portfolio.cash >= trade_value:
                                portfolio.cash -= trade_value
                                portfolio.positions[symbol] = target_quantity
                                portfolio.position_values[symbol] = target_quantity * current_price
                                
                                trades.append({
                                    "symbol": symbol,
                                    "action": "buy", 
                                    "quantity": quantity_diff,
                                    "price": current_price,
                                    "value": trade_value
                                })
                                
                                portfolio.total_trades += 1
                        else:
                            # Sell some
                            sell_quantity = abs(quantity_diff)
                            sell_value = sell_quantity * current_price
                            portfolio.cash += sell_value
                            portfolio.positions[symbol] = target_quantity
                            portfolio.position_values[symbol] = target_quantity * current_price
                            
                            trades.append({
                                "symbol": symbol,
                                "action": "sell",
                                "quantity": sell_quantity,
                                "price": current_price,
                                "value": sell_value
                            })
                            
                            portfolio.total_trades += 1
        
        return trades
    
    def calculate_actual_returns(self, 
                               portfolio: PortfolioState, 
                               predictions: Dict) -> float:
        """Calculate actual returns based on predicted price movements"""
        total_return = 0.0
        
        for symbol, quantity in portfolio.positions.items():
            if quantity != 0 and symbol in predictions:
                current_price = predictions[symbol]['close_last_price']
                predicted_price = predictions[symbol]['close_predicted_price_value']
                
                # Calculate return based on predicted movement
                predicted_return = (predicted_price - current_price) / current_price
                position_value = quantity * current_price
                total_return += position_value * predicted_return
        
        return total_return
    
    def simulate_strategy(self, strategy: AllocationStrategy, max_days: int = None) -> Dict:
        """Simulate a portfolio allocation strategy over available prediction data"""
        logger.info(f"Simulating strategy: {strategy.name}")
        
        portfolio = PortfolioState(cash=self.initial_cash)
        all_trades = []
        daily_portfolio_values = []
        
        # Use subset of files if max_days specified
        files_to_use = self.prediction_files[:max_days] if max_days else self.prediction_files
        
        for i, (date, file_path) in enumerate(files_to_use):
            try:
                # Load predictions for this day
                predictions = self.load_predictions(file_path)
                if not predictions:
                    continue
                
                # Simulate trading for this day (every 5th day to reduce frequency)
                if i % 5 == 0:  # Trade every 5th prediction file
                    trades = self.simulate_day(portfolio, predictions, strategy)
                    all_trades.extend(trades)
                
                # Calculate actual returns based on predictions
                if portfolio.positions:
                    actual_return = self.calculate_actual_returns(portfolio, predictions)
                    
                    # Update portfolio values with "actual" movements
                    for symbol, quantity in portfolio.positions.items():
                        if quantity != 0 and symbol in predictions:
                            current_price = predictions[symbol]['close_last_price']
                            predicted_price = predictions[symbol]['close_predicted_price_value']
                            portfolio.position_values[symbol] = quantity * predicted_price
                
                # Record daily portfolio value
                daily_portfolio_values.append({
                    "date": date.isoformat(),
                    "total_value": portfolio.total_value,
                    "cash": portfolio.cash,
                    "positions_value": sum(portfolio.position_values.values())
                })
                
            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")
                continue
        
        # Calculate performance metrics
        if not daily_portfolio_values:
            return {}
            
        initial_value = self.initial_cash
        final_value = portfolio.total_value
        total_return = (final_value - initial_value) / initial_value
        
        # Calculate Sharpe ratio (simplified)
        daily_values = [d["total_value"] for d in daily_portfolio_values]
        if len(daily_values) > 1:
            daily_returns = np.diff(daily_values) / daily_values[:-1]
            sharpe_ratio = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)
        else:
            sharpe_ratio = 0
            
        # Calculate max drawdown
        peak = initial_value
        max_drawdown = 0
        for value in daily_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return {
            "strategy": strategy.name,
            "initial_value": initial_value,
            "final_value": final_value,
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "total_trades": portfolio.total_trades,
            "days_simulated": len(daily_portfolio_values),
            "daily_values": daily_portfolio_values,
            "all_trades": all_trades,
            "final_positions": dict(portfolio.positions)
        }
    
    def run_simulation(self, max_days: int = 50) -> Dict:
        """Run simulation for all allocation strategies"""
        logger.info("Starting portfolio allocation simulation")
        
        if not self.prediction_files:
            logger.error("No prediction files found")
            return {}
        
        results = {}
        
        # Test each allocation strategy
        for strategy in self.strategies:
            try:
                result = self.simulate_strategy(strategy, max_days)
                if result:
                    results[strategy.name] = result
                    
                    logger.info(f"Strategy {strategy.name} completed:")
                    logger.info(f"  Days simulated: {result['days_simulated']}")
                    logger.info(f"  Total Return: {result['total_return']:.2%}")
                    logger.info(f"  Sharpe Ratio: {result['sharpe_ratio']:.3f}")
                    logger.info(f"  Max Drawdown: {result['max_drawdown']:.2%}")
                    logger.info(f"  Total Trades: {result['total_trades']}")
                
            except Exception as e:
                logger.error(f"Simulation failed for strategy {strategy.name}: {e}")
                continue
        
        # Save results
        self.save_results(results)
        
        return results
    
    def save_results(self, results: Dict):
        """Save simulation results to files"""
        output_dir = Path("portfolio_sim_results")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results as JSON
        results_file = output_dir / f"portfolio_sim_results_{timestamp}.json"
        
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
            
        # Save summary as CSV
        summary_data = []
        for strategy_name, result in results.items():
            summary_data.append({
                "Strategy": strategy_name,
                "Days Simulated": result['days_simulated'],
                "Total Return": f"{result['total_return']:.2%}",
                "Sharpe Ratio": f"{result['sharpe_ratio']:.3f}",
                "Max Drawdown": f"{result['max_drawdown']:.2%}", 
                "Total Trades": result['total_trades'],
                "Final Value": f"${result['final_value']:.2f}"
            })
            
        summary_df = pd.DataFrame(summary_data)
        summary_file = output_dir / f"portfolio_sim_summary_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)
        
        logger.info(f"Results saved to {results_file} and {summary_file}")
        
        # Print summary
        print("\n" + "="*80)
        print("PORTFOLIO ALLOCATION SIMULATION RESULTS")
        print("="*80)
        print(summary_df.to_string(index=False))
        print("="*80)

def main():
    """Run the portfolio simulation system"""
    
    # Create and run the simulation
    simulation = PortfolioSimulation(initial_cash=100000.0)
    
    # Run simulation with limited days for faster testing
    results = simulation.run_simulation(max_days=100)  # Use last 100 prediction files
    
    if results:
        # Find best performing strategy
        best_strategy = max(results.keys(), key=lambda k: results[k]["total_return"])
        best_return = results[best_strategy]["total_return"]
        
        print(f"\nBest performing strategy: {best_strategy}")
        print(f"Total return: {best_return:.2%}")
        
        # Show position comparison
        print("\nFinal positions by strategy:")
        for strategy_name, result in results.items():
            positions = result['final_positions']
            active_positions = {k: v for k, v in positions.items() if v != 0}
            print(f"{strategy_name}: {len(active_positions)} positions - {list(active_positions.keys())}")
    else:
        print("No results generated. Check logs for errors.")

if __name__ == "__main__":
    main()