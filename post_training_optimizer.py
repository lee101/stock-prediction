#!/usr/bin/env python3
"""
Post-Training Parameter Optimization for Portfolio Management

This script optimizes portfolio-level parameters after RL model training,
including number of simultaneous positions, exposure limits, and rebalancing frequency.
"""

import sys
import json
import itertools
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger

# Add paths
sys.path.extend(['.', './training', './src', './rlinference'])

from trade_stock_e2e_trained import TradeStockE2ETrained
from training.trading_config import get_trading_costs
import alpaca_wrapper


class PostTrainingOptimizer:
    """
    Optimize portfolio-level parameters after RL training.
    """
    
    def __init__(self, base_config_path: str = None):
        self.logger = logger
        self.logger.add(f"post_training_optimization_{datetime.now():%Y%m%d_%H%M%S}.log")
        
        # Base configuration
        self.base_config = self._load_base_config(base_config_path)
        
        # Optimization parameters to test
        self.param_grid = {
            'max_positions': [1, 2, 3, 4, 5],  # Number of simultaneous positions
            'max_exposure_per_symbol': [0.3, 0.4, 0.5, 0.6, 0.8],  # Max exposure per symbol
            'min_confidence': [0.2, 0.3, 0.4, 0.5, 0.6],  # Minimum RL confidence threshold
            'rebalance_frequency_minutes': [15, 30, 60, 120, 240],  # Rebalancing frequency
        }
        
        # Risk parameters to test
        self.risk_param_grid = {
            'max_daily_loss': [0.02, 0.03, 0.05, 0.07, 0.10],  # Max daily loss %
            'max_drawdown': [0.10, 0.15, 0.20, 0.25, 0.30],   # Max drawdown %
        }
        
        self.results = []
        
    def _load_base_config(self, config_path: str = None) -> Dict:
        """Load base configuration."""
        default_config = {
            'symbols': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMD', 'AMZN', 'META'],
            'initial_balance': 100000,
            'max_positions': 2,
            'max_exposure_per_symbol': 0.6,
            'min_confidence': 0.4,
            'rebalance_frequency_minutes': 30,
            'risk_management': {
                'max_daily_loss': 0.05,
                'max_drawdown': 0.15,
                'position_timeout_hours': 24
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def generate_parameter_combinations(self, sample_size: int = 50) -> List[Dict]:
        """Generate parameter combinations to test."""
        # Create all possible combinations
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        
        all_combinations = list(itertools.product(*param_values))
        
        # If too many combinations, sample randomly
        if len(all_combinations) > sample_size:
            import random
            selected_combinations = random.sample(all_combinations, sample_size)
        else:
            selected_combinations = all_combinations
        
        # Convert to list of dictionaries
        param_combinations = []
        for combo in selected_combinations:
            param_dict = dict(zip(param_names, combo))
            param_combinations.append(param_dict)
        
        self.logger.info(f"Generated {len(param_combinations)} parameter combinations to test")
        return param_combinations
    
    def create_test_config(self, params: Dict) -> Dict:
        """Create test configuration with specific parameters."""
        config = self.base_config.copy()
        
        # Update main parameters
        config.update(params)
        
        # Add risk management parameters if specified
        if 'max_daily_loss' in params or 'max_drawdown' in params:
            config['risk_management'] = config.get('risk_management', {})
            if 'max_daily_loss' in params:
                config['risk_management']['max_daily_loss'] = params['max_daily_loss']
            if 'max_drawdown' in params:
                config['risk_management']['max_drawdown'] = params['max_drawdown']
        
        return config
    
    def simulate_trading_performance(self, config: Dict, simulation_days: int = 5) -> Dict:
        """
        Simulate trading performance with given configuration.
        
        Note: This is a simplified simulation. In practice, you'd want to:
        - Use historical data backtesting
        - Monte Carlo simulation
        - Walk-forward analysis
        """
        try:
            # Create temporary config file
            temp_config_path = f"temp_config_{datetime.now():%Y%m%d_%H%M%S}.json"
            with open(temp_config_path, 'w') as f:
                json.dump(config, f)
            
            # Initialize trader with test config
            trader = TradeStockE2ETrained(config_path=temp_config_path, paper_trading=True)
            
            # Simulate multiple trading cycles
            results = []
            initial_equity = 100000  # Simulated starting equity
            
            for day in range(simulation_days):
                # Run trading cycle (dry run)
                cycle_result = trader.run_trading_cycle(dry_run=True)
                results.append(cycle_result)
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(results, config, initial_equity)
            
            # Clean up temp file
            Path(temp_config_path).unlink(missing_ok=True)
            
            return performance_metrics
            
        except Exception as e:
            self.logger.error(f"Error in simulation: {e}")
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'num_trades': 0,
                'error': str(e)
            }
    
    def _calculate_performance_metrics(self, results: List[Dict], config: Dict, initial_equity: float) -> Dict:
        """Calculate performance metrics from simulation results."""
        
        # Extract key metrics
        total_trades = sum(r.get('trades_executed', 0) for r in results)
        
        # Simulate returns based on number of positions and confidence
        # This is a simplified model - replace with actual backtesting data
        avg_positions = config.get('max_positions', 2)
        avg_confidence = config.get('min_confidence', 0.4)
        exposure_per_symbol = config.get('max_exposure_per_symbol', 0.6)
        
        # Simple performance model (replace with real backtesting)
        # Higher confidence and more focused positions tend to perform better
        base_return = 0.001  # 0.1% base daily return
        confidence_bonus = avg_confidence * 0.002  # Confidence bonus
        position_penalty = max(0, (avg_positions - 2) * 0.0005)  # Penalty for too many positions
        exposure_bonus = min(exposure_per_symbol * 0.001, 0.0005)  # Moderate exposure bonus
        
        daily_return = base_return + confidence_bonus - position_penalty + exposure_bonus
        
        # Add some randomness to simulate market conditions
        np.random.seed(42)  # For reproducibility
        daily_returns = np.random.normal(daily_return, 0.002, len(results))
        
        # Calculate cumulative metrics
        cumulative_returns = np.cumprod(1 + daily_returns) - 1
        total_return = cumulative_returns[-1] if len(cumulative_returns) > 0 else 0
        
        # Calculate Sharpe ratio
        if len(daily_returns) > 1:
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Calculate max drawdown
        cumulative_equity = (1 + cumulative_returns) * initial_equity
        peak_equity = np.maximum.accumulate(cumulative_equity)
        drawdowns = (cumulative_equity - peak_equity) / peak_equity
        max_drawdown = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0
        
        # Trading frequency metrics
        trading_frequency = total_trades / len(results) if results else 0
        
        # Calculate efficiency metrics
        efficiency_score = self._calculate_efficiency_score(config)
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': total_trades,
            'trading_frequency': trading_frequency,
            'efficiency_score': efficiency_score,
            'daily_returns': daily_returns.tolist()
        }
    
    def _calculate_efficiency_score(self, config: Dict) -> float:
        """Calculate efficiency score based on parameter balance."""
        # Reward balanced configurations
        max_positions = config.get('max_positions', 2)
        min_confidence = config.get('min_confidence', 0.4)
        exposure_per_symbol = config.get('max_exposure_per_symbol', 0.6)
        rebalance_freq = config.get('rebalance_frequency_minutes', 30)
        
        # Optimal ranges (these could be learned from data)
        position_score = 1.0 - abs(max_positions - 2) * 0.1  # Optimal around 2 positions
        confidence_score = 1.0 - abs(min_confidence - 0.4) * 0.5  # Optimal around 0.4
        exposure_score = 1.0 - abs(exposure_per_symbol - 0.5) * 0.3  # Optimal around 0.5
        freq_score = 1.0 - abs(rebalance_freq - 30) / 60  # Optimal around 30 minutes
        
        # Combine scores
        efficiency_score = np.mean([position_score, confidence_score, exposure_score, freq_score])
        return max(0, efficiency_score)
    
    def optimize_parameters(self, sample_size: int = 30, simulation_days: int = 5) -> Dict:
        """Run parameter optimization."""
        self.logger.info("Starting post-training parameter optimization")
        
        # Generate parameter combinations
        param_combinations = self.generate_parameter_combinations(sample_size)
        
        # Test each combination
        for i, params in enumerate(param_combinations):
            self.logger.info(f"Testing combination {i+1}/{len(param_combinations)}: {params}")
            
            # Create test configuration
            test_config = self.create_test_config(params)
            
            # Simulate performance
            performance = self.simulate_trading_performance(test_config, simulation_days)
            
            # Store results
            result = {
                'params': params,
                'performance': performance,
                'score': self._calculate_optimization_score(performance)
            }
            self.results.append(result)
            
            self.logger.info(f"  Performance: Return={performance['total_return']:.3%}, "
                           f"Sharpe={performance['sharpe_ratio']:.2f}, "
                           f"Score={result['score']:.3f}")
        
        # Find best parameters
        best_result = max(self.results, key=lambda x: x['score'])
        
        self.logger.info(f"Optimization completed. Best parameters: {best_result['params']}")
        self.logger.info(f"Best performance: {best_result['performance']}")
        
        return best_result
    
    def _calculate_optimization_score(self, performance: Dict) -> float:
        """Calculate overall optimization score."""
        # Weighted combination of metrics
        total_return = performance.get('total_return', 0)
        sharpe_ratio = performance.get('sharpe_ratio', 0)
        max_drawdown = performance.get('max_drawdown', 0)
        efficiency_score = performance.get('efficiency_score', 0)
        
        # Normalize and weight
        return_score = min(total_return * 10, 1.0)  # Cap at 1.0
        sharpe_score = min(max(sharpe_ratio / 2.0, 0), 1.0)  # Normalize to 0-1
        drawdown_score = max(1.0 - max_drawdown * 2, 0)  # Penalty for high drawdown
        
        # Weighted combination
        score = (0.4 * return_score + 
                0.3 * sharpe_score + 
                0.2 * drawdown_score + 
                0.1 * efficiency_score)
        
        return score
    
    def save_results(self, output_path: str = None):
        """Save optimization results."""
        if not output_path:
            output_path = f"portfolio_optimization_results_{datetime.now():%Y%m%d_%H%M%S}.json"
        
        # Prepare results for saving
        save_data = {
            'optimization_date': datetime.now().isoformat(),
            'base_config': self.base_config,
            'param_grid': self.param_grid,
            'results': self.results,
            'best_result': max(self.results, key=lambda x: x['score']) if self.results else None
        }
        
        with open(output_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        self.logger.info(f"Results saved to {output_path}")
        
        # Also save best config for easy use
        if self.results:
            best_result = max(self.results, key=lambda x: x['score'])
            best_config = self.create_test_config(best_result['params'])
            
            best_config_path = output_path.replace('.json', '_best_config.json')
            with open(best_config_path, 'w') as f:
                json.dump(best_config, f, indent=2)
            
            self.logger.info(f"Best configuration saved to {best_config_path}")
    
    def print_summary(self):
        """Print optimization summary."""
        if not self.results:
            print("No results to summarize")
            return
        
        print("\n" + "="*80)
        print("PORTFOLIO PARAMETER OPTIMIZATION SUMMARY")
        print("="*80)
        
        # Sort results by score
        sorted_results = sorted(self.results, key=lambda x: x['score'], reverse=True)
        
        print(f"\nTested {len(self.results)} parameter combinations")
        print(f"Optimization metric: Weighted score (return + sharpe + drawdown + efficiency)")
        
        print(f"\n🏆 TOP 5 CONFIGURATIONS:")
        print("-"*80)
        for i, result in enumerate(sorted_results[:5]):
            params = result['params']
            perf = result['performance']
            print(f"\n#{i+1} (Score: {result['score']:.3f})")
            print(f"  Max Positions: {params.get('max_positions', 2)}")
            print(f"  Max Exposure per Symbol: {params.get('max_exposure_per_symbol', 0.6):.1%}")
            print(f"  Min Confidence: {params.get('min_confidence', 0.4):.1%}")
            print(f"  Rebalance Frequency: {params.get('rebalance_frequency_minutes', 30)} min")
            print(f"  Performance: Return={perf['total_return']:.2%}, Sharpe={perf['sharpe_ratio']:.2f}, Drawdown={perf['max_drawdown']:.2%}")
        
        # Parameter sensitivity analysis
        print(f"\n📊 PARAMETER SENSITIVITY:")
        print("-"*40)
        
        for param in self.param_grid.keys():
            param_scores = {}
            for result in self.results:
                param_value = result['params'].get(param)
                if param_value not in param_scores:
                    param_scores[param_value] = []
                param_scores[param_value].append(result['score'])
            
            # Calculate average score for each parameter value
            avg_scores = {val: np.mean(scores) for val, scores in param_scores.items()}
            best_value = max(avg_scores.keys(), key=lambda x: avg_scores[x])
            
            print(f"  {param}: Best = {best_value} (avg score: {avg_scores[best_value]:.3f})")
        
        print("\n" + "="*80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Post-Training Portfolio Optimization")
    parser.add_argument('--config', type=str, help='Base configuration file')
    parser.add_argument('--sample-size', type=int, default=30, help='Number of parameter combinations to test')
    parser.add_argument('--simulation-days', type=int, default=5, help='Days to simulate for each test')
    parser.add_argument('--output', type=str, help='Output file path')
    
    args = parser.parse_args()
    
    # Run optimization
    optimizer = PostTrainingOptimizer(args.config)
    best_result = optimizer.optimize_parameters(args.sample_size, args.simulation_days)
    
    # Save and print results
    optimizer.save_results(args.output)
    optimizer.print_summary()
    
    print(f"\n✅ Optimization complete! Best configuration achieves score: {best_result['score']:.3f}")


if __name__ == "__main__":
    main()