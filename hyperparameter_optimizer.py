#!/usr/bin/env python3
"""
Systematic Hyperparameter Optimization Framework for Stock Trading

This framework systematically tests different combinations of:
1. Portfolio allocation strategies
2. Rebalancing frequencies
3. Risk management parameters
4. Position sizing methods
5. Signal filtering techniques

Results are automatically written to findings.md after each experiment.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from itertools import product
import logging
from loguru import logger
import json
import glob
import re
from portfolio_simulation_system import PortfolioSimulation, AllocationStrategy, PortfolioState

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment"""
    name: str
    description: str
    max_positions: int
    max_position_size: float
    rebalance_frequency: int  # Trade every N days
    min_expected_return: float  # Minimum return threshold to enter position
    position_sizing_method: str  # 'equal_weight', 'return_weighted', 'risk_weighted'
    stop_loss: Optional[float] = None  # Stop loss percentage
    take_profit: Optional[float] = None  # Take profit percentage
    max_drawdown_stop: Optional[float] = None  # Stop trading if max drawdown exceeded
    volatility_filter: bool = False  # Filter high volatility periods
    correlation_filter: bool = False  # Avoid highly correlated positions

@dataclass
class ExperimentResult:
    """Results from a single experiment"""
    config: ExperimentConfig
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    total_trades: int
    win_rate: float
    avg_trade_return: float
    final_positions: Dict[str, float]
    daily_returns: List[float]
    execution_time: float

class HyperparameterOptimizer:
    """Systematic hyperparameter optimization for trading strategies"""
    
    def __init__(self, data_dir: str = "results", max_days: int = 50):
        self.data_dir = Path(data_dir)
        self.max_days = max_days
        self.results = []
        self.reports_dir = Path("optimization_reports")
        self.reports_dir.mkdir(exist_ok=True)
        
        # Load prediction files
        self.simulation = PortfolioSimulation()
        self.simulation.results_dir = self.data_dir
        self.simulation.load_prediction_files()
        
    def generate_experiment_configs(self) -> List[ExperimentConfig]:
        """Generate all experiment configurations to test"""
        
        # Define parameter ranges to test
        max_positions_range = [1, 2, 3, 5]
        max_position_sizes = [0.95, 0.47, 0.32, 0.19]  # Corresponding to max positions
        rebalance_frequencies = [1, 3, 5, 7]  # Daily, every 3 days, weekly, etc
        min_return_thresholds = [0.0, 0.01, 0.02, 0.05]  # 0%, 1%, 2%, 5%
        position_sizing_methods = ['equal_weight', 'return_weighted']
        
        # Risk management parameters
        stop_losses = [None, 0.05, 0.10]  # No stop, 5%, 10%
        take_profits = [None, 0.15, 0.25]  # No take profit, 15%, 25%
        max_drawdown_stops = [None, 0.10, 0.20]  # Stop at 10% or 20% drawdown
        
        configs = []
        
        # Generate base strategy variations
        for i, (max_pos, pos_size) in enumerate(zip(max_positions_range, max_position_sizes)):
            for rebal_freq in rebalance_frequencies:
                for min_ret in min_return_thresholds:
                    for sizing_method in position_sizing_methods:
                        config = ExperimentConfig(
                            name=f"base_{max_pos}pos_rebal{rebal_freq}_minret{int(min_ret*100)}_{sizing_method}",
                            description=f"Base strategy with {max_pos} positions, rebalance every {rebal_freq} days, min return {min_ret:.1%}",
                            max_positions=max_pos,
                            max_position_size=pos_size,
                            rebalance_frequency=rebal_freq,
                            min_expected_return=min_ret,
                            position_sizing_method=sizing_method
                        )
                        configs.append(config)
        
        # Generate risk management variations (for best performing base strategies)
        base_configs = configs[:8]  # Take first 8 base configs
        
        for base_config in base_configs:
            for stop_loss in stop_losses:
                for take_profit in take_profits:
                    for max_dd_stop in max_drawdown_stops:
                        if stop_loss is None and take_profit is None and max_dd_stop is None:
                            continue  # Skip no-risk-management case (already covered in base)
                            
                        config = ExperimentConfig(
                            name=f"risk_{base_config.name}_sl{stop_loss or 0}_tp{take_profit or 0}_dd{max_dd_stop or 0}",
                            description=f"Risk managed version of {base_config.description}",
                            max_positions=base_config.max_positions,
                            max_position_size=base_config.max_position_size,
                            rebalance_frequency=base_config.rebalance_frequency,
                            min_expected_return=base_config.min_expected_return,
                            position_sizing_method=base_config.position_sizing_method,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            max_drawdown_stop=max_dd_stop
                        )
                        configs.append(config)
        
        logger.info(f"Generated {len(configs)} experiment configurations")
        return configs
    
    def calculate_position_sizes(self, 
                               predictions: Dict, 
                               target_positions: List[str],
                               config: ExperimentConfig,
                               portfolio_value: float) -> Dict[str, float]:
        """Calculate position sizes based on the configured method"""
        
        if not target_positions:
            return {}
            
        position_allocations = {}
        
        if config.position_sizing_method == 'equal_weight':
            # Equal weight allocation
            allocation_per_position = portfolio_value * config.max_position_size
            for symbol in target_positions:
                if symbol in predictions:
                    current_price = predictions[symbol]['close_last_price']
                    position_allocations[symbol] = allocation_per_position / current_price
        
        elif config.position_sizing_method == 'return_weighted':
            # Weight by expected return
            returns = [predictions[symbol]['expected_return'] for symbol in target_positions if symbol in predictions]
            total_return = sum(max(r, 0.001) for r in returns)  # Avoid division by zero
            
            for symbol in target_positions:
                if symbol in predictions:
                    weight = max(predictions[symbol]['expected_return'], 0.001) / total_return
                    allocation = portfolio_value * config.max_position_size * weight
                    current_price = predictions[symbol]['close_last_price']
                    position_allocations[symbol] = allocation / current_price
        
        return position_allocations
    
    def apply_risk_management(self, 
                            portfolio: PortfolioState,
                            predictions: Dict,
                            config: ExperimentConfig) -> List[Dict]:
        """Apply risk management rules and return trades"""
        trades = []
        
        # Stop loss check
        if config.stop_loss:
            for symbol, quantity in list(portfolio.positions.items()):
                if quantity != 0 and symbol in predictions:
                    current_price = predictions[symbol]['close_last_price']
                    # Get original entry price (simplified - would need to track this properly)
                    position_value = quantity * current_price
                    if position_value < 0:  # Loss position
                        loss_pct = abs(position_value) / (quantity * current_price)  # Simplified
                        if loss_pct >= config.stop_loss:
                            # Trigger stop loss
                            portfolio.cash += quantity * current_price
                            trades.append({
                                "symbol": symbol,
                                "action": "stop_loss",
                                "quantity": quantity,
                                "price": current_price,
                                "value": quantity * current_price
                            })
                            portfolio.positions[symbol] = 0
                            portfolio.position_values[symbol] = 0
                            portfolio.total_trades += 1
        
        # Max drawdown stop
        if config.max_drawdown_stop and len(portfolio.daily_returns) > 0:
            peak_value = max(portfolio.daily_returns) if portfolio.daily_returns else portfolio.total_value
            current_drawdown = (peak_value - portfolio.total_value) / peak_value
            if current_drawdown >= config.max_drawdown_stop:
                # Liquidate all positions
                for symbol, quantity in list(portfolio.positions.items()):
                    if quantity != 0 and symbol in predictions:
                        current_price = predictions[symbol]['close_last_price']
                        portfolio.cash += quantity * current_price
                        trades.append({
                            "symbol": symbol,
                            "action": "drawdown_stop",
                            "quantity": quantity,
                            "price": current_price,
                            "value": quantity * current_price
                        })
                        portfolio.positions[symbol] = 0
                        portfolio.position_values[symbol] = 0
                        portfolio.total_trades += 1
        
        return trades
    
    def run_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """Run a single experiment with the given configuration"""
        logger.info(f"Running experiment: {config.name}")
        start_time = datetime.now()
        
        portfolio = PortfolioState(cash=100000.0)
        all_trades = []
        daily_returns = []
        
        files_to_use = self.simulation.prediction_files[:self.max_days]
        
        for i, (date, file_path) in enumerate(files_to_use):
            try:
                # Load predictions for this day
                predictions = self.simulation.load_predictions(file_path)
                if not predictions:
                    continue
                
                # Filter predictions by minimum expected return
                filtered_predictions = {
                    k: v for k, v in predictions.items() 
                    if v['expected_return'] >= config.min_expected_return
                }
                
                # Update portfolio values
                self.simulation.update_portfolio_values(portfolio, predictions, date)
                
                # Apply risk management
                risk_trades = self.apply_risk_management(portfolio, predictions, config)
                all_trades.extend(risk_trades)
                
                # Rebalance portfolio based on frequency
                if i % config.rebalance_frequency == 0:
                    # Select target positions
                    target_positions = self.simulation.select_positions(
                        filtered_predictions, 
                        AllocationStrategy(config.name, config.max_positions, config.max_position_size)
                    )
                    
                    # Calculate position sizes
                    target_allocations = self.calculate_position_sizes(
                        predictions, target_positions, config, portfolio.total_value
                    )
                    
                    # Execute trades (simplified version)
                    trades = self.execute_advanced_trades(portfolio, target_allocations, predictions, config)
                    all_trades.extend(trades)
                
                # Record daily return
                daily_returns.append(portfolio.total_value)
                
            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")
                continue
        
        # Calculate performance metrics
        execution_time = (datetime.now() - start_time).total_seconds()
        
        if not daily_returns:
            return ExperimentResult(
                config=config, total_return=0, sharpe_ratio=0, max_drawdown=0,
                volatility=0, total_trades=0, win_rate=0, avg_trade_return=0,
                final_positions={}, daily_returns=[], execution_time=execution_time
            )
        
        initial_value = 100000.0
        final_value = portfolio.total_value
        total_return = (final_value - initial_value) / initial_value
        
        # Calculate returns and risk metrics
        returns = np.diff(daily_returns) / daily_returns[:-1] if len(daily_returns) > 1 else [0]
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        volatility = np.std(returns) * np.sqrt(252)
        
        # Max drawdown
        peak = initial_value
        max_drawdown = 0
        for value in daily_returns:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Win rate and average trade return
        winning_trades = [t for t in all_trades if t.get('value', 0) > 0]
        win_rate = len(winning_trades) / max(len(all_trades), 1)
        avg_trade_return = np.mean([t.get('value', 0) for t in all_trades]) if all_trades else 0
        
        return ExperimentResult(
            config=config,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            volatility=volatility,
            total_trades=len(all_trades),
            win_rate=win_rate,
            avg_trade_return=avg_trade_return,
            final_positions=dict(portfolio.positions),
            daily_returns=returns.tolist(),
            execution_time=execution_time
        )
    
    def execute_advanced_trades(self, 
                               portfolio: PortfolioState,
                               target_allocations: Dict[str, float],
                               predictions: Dict,
                               config: ExperimentConfig) -> List[Dict]:
        """Execute trades with advanced logic"""
        trades = []
        
        # Close positions not in target
        for symbol in list(portfolio.positions.keys()):
            if symbol not in target_allocations and portfolio.positions[symbol] != 0:
                if symbol in predictions:
                    current_price = predictions[symbol]['close_last_price']
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
        for symbol, target_quantity in target_allocations.items():
            if symbol in predictions:
                current_price = predictions[symbol]['close_last_price']
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
    
    def update_portfolio_values(self, portfolio: PortfolioState, predictions: Dict, date: datetime):
        """Update portfolio position values - fixed version from simulation"""
        for symbol in list(portfolio.positions.keys()):
            if portfolio.positions[symbol] != 0:
                if symbol in predictions:
                    current_price = predictions[symbol]['close_last_price']
                    portfolio.position_values[symbol] = portfolio.positions[symbol] * current_price
                else:
                    # If no price data, keep previous value
                    pass
    
    def run_optimization_batch(self, batch_name: str = None) -> List[ExperimentResult]:
        """Run a batch of experiments and save results"""
        if batch_name is None:
            batch_name = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        configs = self.generate_experiment_configs()
        results = []
        
        logger.info(f"Starting optimization batch '{batch_name}' with {len(configs)} experiments")
        
        for i, config in enumerate(configs):
            try:
                result = self.run_experiment(config)
                results.append(result)
                
                logger.info(f"Experiment {i+1}/{len(configs)} completed: {config.name}")
                logger.info(f"  Return: {result.total_return:.2%}, Sharpe: {result.sharpe_ratio:.3f}")
                
                # Save intermediate results every 10 experiments
                if (i + 1) % 10 == 0:
                    self.save_batch_results(results, batch_name, partial=True)
                    
            except Exception as e:
                logger.error(f"Experiment {config.name} failed: {e}")
                continue
        
        # Save final results
        self.results.extend(results)
        self.save_batch_results(results, batch_name)
        self.generate_findings_report(results, batch_name)
        
        return results
    
    def save_batch_results(self, results: List[ExperimentResult], batch_name: str, partial: bool = False):
        """Save batch results to JSON file"""
        suffix = "_partial" if partial else ""
        results_file = self.reports_dir / f"{batch_name}_results{suffix}.json"
        
        # Convert results to JSON-serializable format
        json_results = []
        for result in results:
            json_result = {
                "config": {
                    "name": result.config.name,
                    "description": result.config.description,
                    "max_positions": result.config.max_positions,
                    "max_position_size": result.config.max_position_size,
                    "rebalance_frequency": result.config.rebalance_frequency,
                    "min_expected_return": result.config.min_expected_return,
                    "position_sizing_method": result.config.position_sizing_method,
                    "stop_loss": result.config.stop_loss,
                    "take_profit": result.config.take_profit,
                    "max_drawdown_stop": result.config.max_drawdown_stop
                },
                "total_return": result.total_return,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown": result.max_drawdown,
                "volatility": result.volatility,
                "total_trades": result.total_trades,
                "win_rate": result.win_rate,
                "avg_trade_return": result.avg_trade_return,
                "final_positions": result.final_positions,
                "execution_time": result.execution_time
            }
            json_results.append(json_result)
        
        with open(results_file, "w") as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Saved {'partial ' if partial else ''}results to {results_file}")
    
    def generate_findings_report(self, results: List[ExperimentResult], batch_name: str):
        """Generate a comprehensive findings report in Markdown"""
        report_file = Path("findings.md")
        
        # Sort results by total return
        sorted_results = sorted(results, key=lambda x: x.total_return, reverse=True)
        
        # Generate report content
        report_content = f"""# Stock Trading Optimization Findings

**Batch:** {batch_name}  
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Experiments:** {len(results)}  

## Executive Summary

### Top Performing Strategies

"""
        
        # Top 10 strategies
        for i, result in enumerate(sorted_results[:10]):
            report_content += f"""**#{i+1}: {result.config.name}**
- **Total Return:** {result.total_return:.2%}
- **Sharpe Ratio:** {result.sharpe_ratio:.3f}
- **Max Drawdown:** {result.max_drawdown:.2%}
- **Strategy:** {result.config.description}
- **Positions:** {result.config.max_positions} max positions
- **Rebalance:** Every {result.config.rebalance_frequency} days
- **Min Return:** {result.config.min_expected_return:.1%}
- **Sizing:** {result.config.position_sizing_method}

"""
        
        # Analysis by parameter
        report_content += """## Parameter Analysis

### Portfolio Size Impact
"""
        
        # Analyze by number of positions
        for pos_count in [1, 2, 3, 5]:
            pos_results = [r for r in results if r.config.max_positions == pos_count]
            if pos_results:
                avg_return = np.mean([r.total_return for r in pos_results])
                avg_sharpe = np.mean([r.sharpe_ratio for r in pos_results])
                best_result = max(pos_results, key=lambda x: x.total_return)
                
                report_content += f"""**{pos_count} Positions:**
- Average Return: {avg_return:.2%}
- Average Sharpe: {avg_sharpe:.3f}
- Best Return: {best_result.total_return:.2%} ({best_result.config.name})

"""
        
        # Analyze rebalancing frequency
        report_content += """### Rebalancing Frequency Impact
"""
        
        for freq in [1, 3, 5, 7]:
            freq_results = [r for r in results if r.config.rebalance_frequency == freq]
            if freq_results:
                avg_return = np.mean([r.total_return for r in freq_results])
                avg_sharpe = np.mean([r.sharpe_ratio for r in freq_results])
                best_result = max(freq_results, key=lambda x: x.total_return)
                
                report_content += f"""**Every {freq} days:**
- Average Return: {avg_return:.2%}
- Average Sharpe: {avg_sharpe:.3f}
- Best Return: {best_result.total_return:.2%} ({best_result.config.name})

"""
        
        # Risk analysis
        report_content += """### Risk Management Analysis
"""
        
        risk_results = [r for r in results if r.config.stop_loss or r.config.take_profit or r.config.max_drawdown_stop]
        no_risk_results = [r for r in results if not (r.config.stop_loss or r.config.take_profit or r.config.max_drawdown_stop)]
        
        if risk_results and no_risk_results:
            risk_avg_return = np.mean([r.total_return for r in risk_results])
            no_risk_avg_return = np.mean([r.total_return for r in no_risk_results])
            risk_avg_drawdown = np.mean([r.max_drawdown for r in risk_results])
            no_risk_avg_drawdown = np.mean([r.max_drawdown for r in no_risk_results])
            
            report_content += f"""**With Risk Management:**
- Average Return: {risk_avg_return:.2%}
- Average Max Drawdown: {risk_avg_drawdown:.2%}

**Without Risk Management:**
- Average Return: {no_risk_avg_return:.2%}
- Average Max Drawdown: {no_risk_avg_drawdown:.2%}

"""
        
        # Key insights
        report_content += """## Key Insights

"""
        
        best_strategy = sorted_results[0]
        worst_strategy = sorted_results[-1]
        
        insights = [
            f"Best performing strategy achieved {best_strategy.total_return:.2%} return with {best_strategy.sharpe_ratio:.3f} Sharpe ratio",
            f"Optimal portfolio size appears to be {best_strategy.config.max_positions} positions",
            f"Best rebalancing frequency is every {best_strategy.config.rebalance_frequency} days",
            f"Return range: {worst_strategy.total_return:.2%} to {best_strategy.total_return:.2%}",
            f"Average execution time per experiment: {np.mean([r.execution_time for r in results]):.1f} seconds"
        ]
        
        for insight in insights:
            report_content += f"- {insight}\n"
        
        report_content += f"""
## Next Experiments to Consider

Based on these findings, the next batch should focus on:

1. **Fine-tune optimal portfolio size** around {best_strategy.config.max_positions} positions
2. **Test intermediate rebalancing frequencies** around {best_strategy.config.rebalance_frequency} days  
3. **Explore position sizing variations** for the best performing strategies
4. **Test additional risk management parameters** for top strategies
5. **Investigate correlation-based position selection** to reduce portfolio correlation

## Raw Data

Full experimental results are available in: `{self.reports_dir}/{batch_name}_results.json`

---
*Report generated by HyperparameterOptimizer on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # Write report
        with open(report_file, "w") as f:
            f.write(report_content)
        
        logger.info(f"Generated findings report: {report_file}")
        
        # Print summary to console
        print("\n" + "="*80)
        print("OPTIMIZATION BATCH COMPLETED")
        print("="*80)
        print(f"Experiments completed: {len(results)}")
        print(f"Best strategy: {best_strategy.config.name}")
        print(f"Best return: {best_strategy.total_return:.2%}")
        print(f"Best Sharpe ratio: {best_strategy.sharpe_ratio:.3f}")
        print(f"Report saved to: findings.md")
        print("="*80)

def main():
    """Run hyperparameter optimization"""
    
    optimizer = HyperparameterOptimizer(
        data_dir="results",  # Use existing prediction files
        max_days=30  # Test with more data
    )
    
    # Run first optimization batch
    results = optimizer.run_optimization_batch("baseline_strategies")
    
    print(f"Completed {len(results)} experiments. Check findings.md for detailed analysis.")

if __name__ == "__main__":
    main()