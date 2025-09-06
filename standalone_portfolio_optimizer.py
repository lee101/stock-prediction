#!/usr/bin/env python3
"""
Standalone Portfolio Parameter Optimization

This version can run without the full trading infrastructure to optimize portfolio parameters.
"""

import json
import itertools
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger


class StandalonePortfolioOptimizer:
    """
    Standalone version for optimizing portfolio parameters without full trading setup.
    """
    
    def __init__(self, base_config_path: str = None):
        self.logger = logger
        log_file = f"portfolio_optimization_{datetime.now():%Y%m%d_%H%M%S}.log"
        self.logger.add(log_file)
        
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
        
        # Market simulation parameters
        self.market_volatility = 0.02  # Daily volatility
        self.market_trend = 0.001      # Daily trend
        self.confidence_alpha = 0.003  # Confidence impact on returns
        
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
    
    def simulate_rl_trading_performance(self, config: Dict, simulation_days: int = 10) -> Dict:
        """
        Simulate RL trading performance based on realistic market dynamics.
        """
        try:
            np.random.seed(42)  # For reproducibility
            random.seed(42)
            
            # Extract parameters
            max_positions = config.get('max_positions', 2)
            min_confidence = config.get('min_confidence', 0.4)
            max_exposure = config.get('max_exposure_per_symbol', 0.6)
            rebalance_freq = config.get('rebalance_frequency_minutes', 30)
            symbols = config.get('symbols', ['AAPL', 'MSFT', 'GOOGL'])
            
            # Simulation state
            initial_equity = config.get('initial_balance', 100000)
            current_equity = initial_equity
            positions = {}  # {symbol: {'qty': float, 'entry_price': float, 'confidence': float}}
            daily_returns = []
            trade_count = 0
            equity_curve = [current_equity]
            
            # Simulate each day
            for day in range(simulation_days):
                daily_start_equity = current_equity
                
                # Market movements for each symbol
                symbol_returns = {}
                for symbol in symbols:
                    # Base market return with trend and volatility
                    base_return = np.random.normal(self.market_trend, self.market_volatility)
                    symbol_returns[symbol] = base_return
                
                # Update existing positions
                for symbol, position in list(positions.items()):
                    market_return = symbol_returns[symbol]
                    
                    # RL model effectiveness: higher confidence -> better risk-adjusted returns
                    confidence_boost = (position['confidence'] - 0.5) * self.confidence_alpha
                    adjusted_return = market_return + confidence_boost
                    
                    # Update position value
                    old_value = position['qty'] * position['entry_price']
                    new_price = position['entry_price'] * (1 + adjusted_return)
                    new_value = position['qty'] * new_price
                    
                    # Update equity
                    current_equity += (new_value - old_value)
                    position['entry_price'] = new_price
                
                # Simulate RL trading decisions (rebalancing based on frequency)
                rebalances_per_day = max(1, int(1440 / rebalance_freq))  # 1440 minutes per day
                
                for rebalance in range(rebalances_per_day):
                    # Simulate RL model generating signals
                    for symbol in symbols:
                        # Simulate RL confidence score
                        rl_confidence = np.random.beta(2, 3)  # Skewed toward lower confidence
                        
                        # Only trade if above minimum confidence
                        if rl_confidence >= min_confidence:
                            
                            # Simulate RL position recommendation
                            if symbol in positions:
                                # Existing position - might adjust or close
                                if rl_confidence < min_confidence + 0.1:
                                    # Close position (low confidence)
                                    del positions[symbol]
                                    trade_count += 1
                            else:
                                # New position opportunity
                                if len(positions) < max_positions:
                                    # Calculate position size based on confidence and constraints
                                    confidence_size = min(rl_confidence, 1.0)
                                    max_position_value = current_equity * max_exposure
                                    position_value = max_position_value * confidence_size * 0.8  # Conservative sizing
                                    
                                    if position_value > 1000:  # Minimum position size
                                        current_price = 100 * (1 + np.random.uniform(-0.02, 0.02))  # Simulate price
                                        qty = position_value / current_price
                                        
                                        positions[symbol] = {
                                            'qty': qty,
                                            'entry_price': current_price,
                                            'confidence': rl_confidence
                                        }
                                        trade_count += 1
                
                # Record daily performance
                daily_return = (current_equity - daily_start_equity) / daily_start_equity
                daily_returns.append(daily_return)
                equity_curve.append(current_equity)
            
            # Calculate performance metrics
            total_return = (current_equity - initial_equity) / initial_equity
            
            if len(daily_returns) > 1:
                sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
            else:
                sharpe_ratio = 0
            
            # Calculate max drawdown
            equity_array = np.array(equity_curve)
            peak_equity = np.maximum.accumulate(equity_array)
            drawdowns = (equity_array - peak_equity) / peak_equity
            max_drawdown = abs(np.min(drawdowns))
            
            # Calculate other metrics
            win_rate = len([r for r in daily_returns if r > 0]) / len(daily_returns) if daily_returns else 0
            avg_daily_return = np.mean(daily_returns) if daily_returns else 0
            volatility = np.std(daily_returns) if len(daily_returns) > 1 else 0
            
            # Trading efficiency
            trades_per_day = trade_count / simulation_days
            
            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'num_trades': trade_count,
                'trades_per_day': trades_per_day,
                'win_rate': win_rate,
                'avg_daily_return': avg_daily_return,
                'volatility': volatility,
                'final_equity': current_equity,
                'daily_returns': daily_returns
            }
            
        except Exception as e:
            self.logger.error(f"Error in simulation: {e}")
            return {
                'total_return': -0.1,  # Penalty for failed simulations
                'sharpe_ratio': -1,
                'max_drawdown': 0.2,
                'num_trades': 0,
                'error': str(e)
            }
    
    def _calculate_optimization_score(self, performance: Dict) -> float:
        """Calculate overall optimization score with realistic weighting."""
        # Extract metrics
        total_return = performance.get('total_return', -0.1)
        sharpe_ratio = performance.get('sharpe_ratio', -1)
        max_drawdown = performance.get('max_drawdown', 0.2)
        win_rate = performance.get('win_rate', 0.4)
        trades_per_day = performance.get('trades_per_day', 0)
        
        # Normalize metrics to 0-1 range
        return_score = max(0, min(total_return + 0.5, 1.0))  # -50% to +50% -> 0 to 1
        sharpe_score = max(0, min((sharpe_ratio + 2) / 4, 1.0))  # -2 to +2 -> 0 to 1
        drawdown_score = max(0, 1 - max_drawdown * 2)  # 0% to 50% drawdown -> 1 to 0
        win_rate_score = win_rate  # Already 0-1
        
        # Trading frequency penalty/bonus
        optimal_trades_per_day = 0.5  # About 1 trade every 2 days
        trade_freq_score = max(0, 1 - abs(trades_per_day - optimal_trades_per_day) / optimal_trades_per_day)
        
        # Weighted combination
        score = (0.35 * return_score +     # Most important: returns
                0.25 * sharpe_score +      # Risk-adjusted returns
                0.20 * drawdown_score +    # Drawdown control
                0.10 * win_rate_score +    # Win rate
                0.10 * trade_freq_score)   # Trading efficiency
        
        return score
    
    def optimize_parameters(self, sample_size: int = 30, simulation_days: int = 10) -> Dict:
        """Run parameter optimization."""
        self.logger.info("Starting standalone portfolio parameter optimization")
        self.logger.info(f"Testing {sample_size} combinations over {simulation_days} simulation days")
        
        # Generate parameter combinations
        param_combinations = self.generate_parameter_combinations(sample_size)
        
        # Test each combination
        best_score = -1
        best_result = None
        
        for i, params in enumerate(param_combinations):
            self.logger.info(f"Testing combination {i+1}/{len(param_combinations)}: {params}")
            
            # Create test configuration
            test_config = self.base_config.copy()
            test_config.update(params)
            
            # Simulate performance
            performance = self.simulate_rl_trading_performance(test_config, simulation_days)
            
            # Calculate optimization score
            score = self._calculate_optimization_score(performance)
            
            # Store results
            result = {
                'params': params,
                'performance': performance,
                'score': score
            }
            self.results.append(result)
            
            # Track best result
            if score > best_score:
                best_score = score
                best_result = result
            
            self.logger.info(f"  Performance: Return={performance['total_return']:.2%}, "
                           f"Sharpe={performance['sharpe_ratio']:.2f}, "
                           f"Drawdown={performance['max_drawdown']:.2%}, "
                           f"Score={score:.3f}")
        
        self.logger.info(f"Optimization completed. Best score: {best_score:.3f}")
        return best_result
    
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
            'best_result': max(self.results, key=lambda x: x['score']) if self.results else None,
            'summary_stats': self._calculate_summary_stats()
        }
        
        with open(output_path, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {output_path}")
        
        # Save best config
        if self.results:
            best_result = max(self.results, key=lambda x: x['score'])
            best_config = self.base_config.copy()
            best_config.update(best_result['params'])
            
            best_config_path = output_path.replace('.json', '_best_config.json')
            with open(best_config_path, 'w') as f:
                json.dump(best_config, f, indent=2)
            
            self.logger.info(f"Best configuration saved to {best_config_path}")
        
        return output_path
    
    def _calculate_summary_stats(self) -> Dict:
        """Calculate summary statistics across all tests."""
        if not self.results:
            return {}
        
        scores = [r['score'] for r in self.results]
        returns = [r['performance']['total_return'] for r in self.results]
        sharpes = [r['performance']['sharpe_ratio'] for r in self.results]
        
        return {
            'num_tests': len(self.results),
            'score_mean': np.mean(scores),
            'score_std': np.std(scores),
            'score_min': np.min(scores),
            'score_max': np.max(scores),
            'return_mean': np.mean(returns),
            'return_std': np.std(returns),
            'sharpe_mean': np.mean(sharpes),
            'sharpe_std': np.std(sharpes)
        }
    
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
        print(f"Optimization metric: Weighted score (return + sharpe + drawdown + win_rate + trade_freq)")
        
        print(f"\n🏆 TOP 5 CONFIGURATIONS:")
        print("-"*80)
        for i, result in enumerate(sorted_results[:5]):
            params = result['params']
            perf = result['performance']
            print(f"\n#{i+1} (Score: {result['score']:.3f})")
            print(f"  Max Positions: {params.get('max_positions', 2)}")
            print(f"  Max Exposure per Symbol: {params.get('max_exposure_per_symbol', 0.6):.0%}")
            print(f"  Min Confidence: {params.get('min_confidence', 0.4):.0%}")
            print(f"  Rebalance Frequency: {params.get('rebalance_frequency_minutes', 30)} min")
            print(f"  Performance:")
            print(f"    Return: {perf['total_return']:.2%}")
            print(f"    Sharpe: {perf['sharpe_ratio']:.2f}")  
            print(f"    Max Drawdown: {perf['max_drawdown']:.2%}")
            print(f"    Win Rate: {perf.get('win_rate', 0):.1%}")
            print(f"    Trades/Day: {perf.get('trades_per_day', 0):.1f}")
        
        # Parameter sensitivity analysis
        print(f"\n📊 PARAMETER SENSITIVITY ANALYSIS:")
        print("-"*50)
        
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
            worst_value = min(avg_scores.keys(), key=lambda x: avg_scores[x])
            
            print(f"\n{param}:")
            print(f"  Best value: {best_value} (avg score: {avg_scores[best_value]:.3f})")
            print(f"  Worst value: {worst_value} (avg score: {avg_scores[worst_value]:.3f})")
            print(f"  Impact: {avg_scores[best_value] - avg_scores[worst_value]:.3f}")
        
        # Summary stats
        summary = self._calculate_summary_stats()
        print(f"\n📈 OVERALL STATISTICS:")
        print("-"*30)
        print(f"Average Score: {summary['score_mean']:.3f} ± {summary['score_std']:.3f}")
        print(f"Best Score: {summary['score_max']:.3f}")
        print(f"Average Return: {summary['return_mean']:.2%} ± {summary['return_std']:.2%}")
        print(f"Average Sharpe: {summary['sharpe_mean']:.2f} ± {summary['sharpe_std']:.2f}")
        
        print("\n" + "="*80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Standalone Portfolio Parameter Optimization")
    parser.add_argument('--config', type=str, help='Base configuration file')
    parser.add_argument('--sample-size', type=int, default=25, help='Number of parameter combinations to test')
    parser.add_argument('--simulation-days', type=int, default=10, help='Days to simulate for each test')
    parser.add_argument('--output', type=str, help='Output file path')
    
    args = parser.parse_args()
    
    # Run optimization
    optimizer = StandalonePortfolioOptimizer(args.config)
    best_result = optimizer.optimize_parameters(args.sample_size, args.simulation_days)
    
    # Save and print results
    output_path = optimizer.save_results(args.output)
    optimizer.print_summary()
    
    print(f"\n✅ Optimization complete!")
    print(f"📊 Best configuration achieves score: {best_result['score']:.3f}")
    print(f"💾 Results saved to: {output_path}")
    
    # Show best parameters
    best_params = best_result['params']
    print(f"\n🎯 OPTIMAL PARAMETERS:")
    print(f"   Max Positions: {best_params['max_positions']}")
    print(f"   Max Exposure per Symbol: {best_params['max_exposure_per_symbol']:.0%}")
    print(f"   Min Confidence Threshold: {best_params['min_confidence']:.0%}")
    print(f"   Rebalance Frequency: {best_params['rebalance_frequency_minutes']} minutes")


if __name__ == "__main__":
    main()