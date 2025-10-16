#!/usr/bin/env python3
"""
End-to-End Testing System for Stock Prediction and Portfolio Allocation

This system simulates trading over multiple days using historical data to test:
1. Different portfolio allocation strategies (1 stock vs 2 vs balanced 3+)
2. Prediction accuracy and profitability
3. Risk management strategies
4. Overall portfolio performance

The system runs entirely in Python for efficient simulation.
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

from backtest_test3_inline import backtest_forecasts
from src.fixtures import crypto_symbols
from show_forecasts import show_forecasts


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
    rebalance_threshold: float = 0.1  # Rebalance if allocation drifts by this much


class E2ETestingSystem:
    """End-to-end testing system for stock prediction strategies"""
    
    def __init__(self, 
                 start_date: str = "2024-01-01",
                 end_date: str = "2024-12-31", 
                 initial_cash: float = 100000.0):
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.initial_cash = initial_cash
        self.symbols = crypto_symbols + ["GOOG", "NET", "TSLA", "NVDA", "AAPL"]  # Mix crypto + stocks
        
        # Define allocation strategies to test
        self.strategies = [
            AllocationStrategy("single_best", max_positions=1, max_position_size=0.95),
            AllocationStrategy("dual_best", max_positions=2, max_position_size=0.47),  
            AllocationStrategy("balanced_3", max_positions=3, max_position_size=0.32),
            AllocationStrategy("diversified_5", max_positions=5, max_position_size=0.19),
        ]
        
        self.results = {}
        self.historical_prices = {}
        
    def load_historical_data(self) -> bool:
        """Load historical price data for all symbols"""
        logger.info("Loading historical price data...")
        
        # Check for cached data files
        data_dir = Path("historical_data")
        data_dir.mkdir(exist_ok=True)
        
        for symbol in self.symbols:
            data_file = data_dir / f"{symbol}_daily.csv"
            if data_file.exists():
                try:
                    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
                    self.historical_prices[symbol] = df
                    logger.info(f"Loaded {len(df)} days of data for {symbol}")
                except Exception as e:
                    logger.warning(f"Could not load data for {symbol}: {e}")
            else:
                logger.warning(f"No historical data found for {symbol} at {data_file}")
                
        return len(self.historical_prices) > 0
    
    def get_price_at_date(self, symbol: str, date: datetime, price_type: str = "close") -> Optional[float]:
        """Get price for symbol at specific date"""
        if symbol not in self.historical_prices:
            return None
            
        df = self.historical_prices[symbol]
        date_str = date.strftime("%Y-%m-%d")
        
        # Find closest date if exact match not found
        try:
            if date_str in df.index:
                return df.loc[date_str, price_type]
            else:
                # Find nearest date within 7 days
                target_date = pd.to_datetime(date_str)
                df_dates = pd.to_datetime(df.index)
                date_diffs = abs(df_dates - target_date)
                closest_idx = date_diffs.idxmin()
                
                if date_diffs[closest_idx].days <= 7:  # Within a week
                    closest_date_str = df_dates[closest_idx].strftime("%Y-%m-%d")
                    return df.loc[closest_date_str, price_type]
                    
        except (KeyError, IndexError, AttributeError):
            pass
            
        return None
    
    def run_daily_analysis(self, date: datetime) -> Dict[str, Dict]:
        """Run prediction analysis for all symbols on a given date"""
        logger.info(f"Running analysis for {date.strftime('%Y-%m-%d')}")
        
        analysis_results = {}
        
        for symbol in self.symbols:
            try:
                # Run backtest to get predictions (simulate what would happen on this date)
                logger.info(f"Analyzing {symbol}")
                backtest_df = backtest_forecasts(symbol, num_simulations=30)  # Reduced for speed
                
                if len(backtest_df) > 0:
                    last_prediction = backtest_df.iloc[-1]
                    
                    # Calculate strategy returns
                    simple_return = backtest_df["simple_strategy_return"].mean()
                    all_signals_return = backtest_df["all_signals_strategy_return"].mean()
                    takeprofit_return = backtest_df["entry_takeprofit_return"].mean()
                    highlow_return = backtest_df["highlow_return"].mean()
                    
                    # Find best strategy
                    returns = {
                        "simple": simple_return,
                        "all_signals": all_signals_return,
                        "takeprofit": takeprofit_return,
                        "highlow": highlow_return
                    }
                    
                    best_strategy = max(returns.keys(), key=lambda k: returns[k])
                    best_return = returns[best_strategy]
                    
                    # Get current price
                    current_price = self.get_price_at_date(symbol, date)
                    if current_price is None:
                        continue
                    
                    analysis_results[symbol] = {
                        "best_strategy": best_strategy,
                        "expected_return": best_return,
                        "current_price": current_price,
                        "predicted_close": float(last_prediction.get("predicted_close", current_price)),
                        "predicted_high": float(last_prediction.get("predicted_high", current_price)),
                        "predicted_low": float(last_prediction.get("predicted_low", current_price)),
                        "strategy_returns": returns
                    }
                    
            except Exception as e:
                logger.warning(f"Analysis failed for {symbol}: {e}")
                continue
                
        return analysis_results
    
    def select_positions(self, analysis: Dict, strategy: AllocationStrategy) -> List[str]:
        """Select which positions to hold based on analysis and allocation strategy"""
        
        # Sort symbols by expected return
        sorted_symbols = sorted(analysis.keys(), 
                              key=lambda s: analysis[s]["expected_return"], 
                              reverse=True)
        
        # Filter to positive expected returns only
        profitable_symbols = [s for s in sorted_symbols 
                            if analysis[s]["expected_return"] > 0]
        
        # Select top N based on strategy
        selected = profitable_symbols[:strategy.max_positions]
        
        logger.info(f"Selected positions for {strategy.name}: {selected}")
        return selected
    
    def update_portfolio_values(self, portfolio: PortfolioState, date: datetime):
        """Update portfolio position values based on current market prices"""
        for symbol in list(portfolio.positions.keys()):
            if portfolio.positions[symbol] != 0:
                current_price = self.get_price_at_date(symbol, date)
                if current_price:
                    portfolio.position_values[symbol] = portfolio.positions[symbol] * current_price
                else:
                    # If no price data, assume position unchanged
                    pass
    
    def execute_trades(self, 
                      portfolio: PortfolioState, 
                      target_positions: List[str], 
                      analysis: Dict,
                      strategy: AllocationStrategy,
                      date: datetime) -> List[Dict]:
        """Execute trades to reach target portfolio allocation"""
        trades = []
        
        # Close positions not in target
        for symbol in list(portfolio.positions.keys()):
            if symbol not in target_positions and portfolio.positions[symbol] != 0:
                current_price = self.get_price_at_date(symbol, date)
                if current_price:
                    # Sell position
                    sell_value = portfolio.positions[symbol] * current_price
                    portfolio.cash += sell_value
                    
                    trades.append({
                        "symbol": symbol,
                        "action": "sell",
                        "quantity": portfolio.positions[symbol],
                        "price": current_price,
                        "value": sell_value,
                        "date": date
                    })
                    
                    portfolio.positions[symbol] = 0
                    portfolio.position_values[symbol] = 0
                    portfolio.total_trades += 1
        
        # Open/adjust positions for targets
        if target_positions:
            position_allocation = portfolio.total_value * strategy.max_position_size
            
            for symbol in target_positions:
                current_price = self.get_price_at_date(symbol, date)
                if not current_price:
                    continue
                    
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
                                "value": trade_value,
                                "date": date
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
                            "value": sell_value,
                            "date": date
                        })
                        
                        portfolio.total_trades += 1
        
        return trades
    
    def simulate_strategy(self, strategy: AllocationStrategy) -> Dict:
        """Simulate a portfolio allocation strategy over the test period"""
        logger.info(f"Simulating strategy: {strategy.name}")
        
        portfolio = PortfolioState(cash=self.initial_cash)
        all_trades = []
        daily_portfolio_values = []
        
        current_date = self.start_date
        
        while current_date <= self.end_date:
            # Skip weekends for stock trading 
            if current_date.weekday() < 5:  # Monday = 0, Friday = 4
                # Update portfolio values with current prices
                self.update_portfolio_values(portfolio, current_date)
                
                # Record daily portfolio value
                daily_portfolio_values.append({
                    "date": current_date,
                    "total_value": portfolio.total_value,
                    "cash": portfolio.cash,
                    "positions_value": sum(portfolio.position_values.values())
                })
                
                # Run analysis every 7 days (weekly rebalancing)
                if (current_date - self.start_date).days % 7 == 0:
                    try:
                        analysis = self.run_daily_analysis(current_date)
                        
                        if analysis:  # Only trade if we have analysis results
                            target_positions = self.select_positions(analysis, strategy)
                            trades = self.execute_trades(portfolio, target_positions, 
                                                       analysis, strategy, current_date)
                            all_trades.extend(trades)
                            
                    except Exception as e:
                        logger.warning(f"Analysis failed on {current_date}: {e}")
            
            current_date += timedelta(days=1)
        
        # Final portfolio update
        self.update_portfolio_values(portfolio, self.end_date)
        
        # Calculate performance metrics
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
            "win_rate": portfolio.win_rate,
            "daily_values": daily_portfolio_values,
            "all_trades": all_trades,
            "final_positions": dict(portfolio.positions)
        }
    
    def run_full_simulation(self) -> Dict:
        """Run simulation for all allocation strategies"""
        logger.info("Starting full E2E simulation")
        
        # Load historical data
        if not self.load_historical_data():
            logger.error("Failed to load historical data. Cannot run simulation.")
            return {}
        
        results = {}
        
        # Test each allocation strategy
        for strategy in self.strategies:
            try:
                result = self.simulate_strategy(strategy)
                results[strategy.name] = result
                
                logger.info(f"Strategy {strategy.name} completed:")
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
        output_dir = Path("e2e_results")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results as JSON
        results_file = output_dir / f"e2e_results_{timestamp}.json"
        
        # Convert datetime objects to strings for JSON serialization
        json_results = {}
        for strategy_name, result in results.items():
            json_result = result.copy()
            
            # Convert daily values
            if "daily_values" in json_result:
                for daily_val in json_result["daily_values"]:
                    daily_val["date"] = daily_val["date"].isoformat()
                    
            # Convert trades  
            if "all_trades" in json_result:
                for trade in json_result["all_trades"]:
                    trade["date"] = trade["date"].isoformat()
                    
            json_results[strategy_name] = json_result
        
        with open(results_file, "w") as f:
            json.dump(json_results, f, indent=2, default=str)
            
        # Save summary as CSV
        summary_data = []
        for strategy_name, result in results.items():
            summary_data.append({
                "Strategy": strategy_name,
                "Total Return": f"{result['total_return']:.2%}",
                "Sharpe Ratio": f"{result['sharpe_ratio']:.3f}",
                "Max Drawdown": f"{result['max_drawdown']:.2%}", 
                "Total Trades": result['total_trades'],
                "Final Value": f"${result['final_value']:.2f}"
            })
            
        summary_df = pd.DataFrame(summary_data)
        summary_file = output_dir / f"e2e_summary_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)
        
        logger.info(f"Results saved to {results_file} and {summary_file}")
        
        # Print summary
        print("\n" + "="*80)
        print("E2E SIMULATION RESULTS SUMMARY")
        print("="*80)
        print(summary_df.to_string(index=False))
        print("="*80)


def main():
    """Run the E2E testing system"""
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create and run the testing system
    # Use shorter date range for initial testing
    system = E2ETestingSystem(
        start_date="2024-10-01",  # Last 3 months for faster testing
        end_date="2024-12-31",
        initial_cash=100000.0
    )
    
    results = system.run_full_simulation()
    
    if results:
        # Find best performing strategy
        best_strategy = max(results.keys(), key=lambda k: results[k]["total_return"])
        best_return = results[best_strategy]["total_return"]
        
        print(f"\nBest performing strategy: {best_strategy}")
        print(f"Total return: {best_return:.2%}")
    else:
        print("No results generated. Check logs for errors.")


if __name__ == "__main__":
    main()