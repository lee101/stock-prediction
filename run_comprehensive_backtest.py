#!/usr/bin/env python3
"""
Script to run comprehensive backtest with real GPU forecasts using .venv environment.
"""

import sys
import os
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Set up environment
os.environ['PYTHONPATH'] = str(ROOT)

try:
    from comprehensive_backtest_real_gpu import ComprehensiveBacktester
    from src.advanced_position_sizing import get_all_advanced_strategies, get_dataframe_only_strategies
    import logging
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    def run_enhanced_backtest():
        """Run the enhanced backtest with advanced strategies."""
        
        # Define test symbols (start with a smaller set for testing)
        test_symbols = [
            "AAPL", "GOOGL", "TSLA", "NVDA", "MSFT"
        ]
        
        logger.info(f"Starting enhanced backtest with symbols: {test_symbols}")
        
        # Create enhanced backtester
        backtester = ComprehensiveBacktester(
            symbols=test_symbols,
            start_date="2023-01-01",
            end_date="2024-12-31"
        )
        
        # Override the test_position_sizing_strategies method to include advanced strategies
        original_test_method = backtester.test_position_sizing_strategies
        
        def enhanced_test_strategies(actual_df, predicted_df):
            """Enhanced version with advanced strategies."""
            logger.info("Testing enhanced position sizing strategies...")
            
            # Get original strategies
            results = original_test_method(actual_df, predicted_df)
            
            # Add advanced strategies
            advanced_strategies = get_all_advanced_strategies()
            dataframe_strategies = get_dataframe_only_strategies()
            
            # Test advanced strategies
            for name, strategy_func in advanced_strategies.items():
                logger.info(f"Testing advanced strategy: {name}")
                try:
                    sizes = strategy_func(predicted_df)
                    sizes = sizes.clip(-3, 3)  # Reasonable bounds
                    
                    from src.position_sizing_optimizer import backtest_position_sizing_series, sharpe_ratio
                    
                    pnl_series = backtest_position_sizing_series(
                        actual_df, 
                        predicted_df, 
                        lambda _: sizes,
                        trading_fee=0.001
                    )
                    
                    # Calculate metrics
                    total_return = pnl_series.sum()
                    sharpe = sharpe_ratio(pnl_series, risk_free_rate=0.02)
                    max_drawdown = backtester.calculate_max_drawdown(pnl_series.cumsum())
                    volatility = pnl_series.std() * (252**0.5)
                    
                    results[f"advanced_{name}"] = {
                        'pnl_series': pnl_series,
                        'cumulative_pnl': pnl_series.cumsum(),
                        'total_return': total_return,
                        'sharpe_ratio': sharpe,
                        'max_drawdown': max_drawdown,
                        'volatility': volatility,
                        'num_trades': len(pnl_series),
                        'win_rate': (pnl_series > 0).mean()
                    }
                    
                    logger.info(f"advanced_{name}: Return={total_return:.4f}, Sharpe={sharpe:.3f}")
                    
                except Exception as e:
                    logger.error(f"Error with advanced strategy {name}: {e}")
                    continue
            
            # Test DataFrame-only strategies
            for name, strategy_func in dataframe_strategies.items():
                logger.info(f"Testing DataFrame strategy: {name}")
                try:
                    sizes = strategy_func(predicted_df)
                    sizes = sizes.clip(-3, 3)
                    
                    pnl_series = backtest_position_sizing_series(
                        actual_df, 
                        predicted_df, 
                        lambda _: sizes,
                        trading_fee=0.001
                    )
                    
                    # Calculate metrics
                    total_return = pnl_series.sum()
                    sharpe = sharpe_ratio(pnl_series, risk_free_rate=0.02)
                    max_drawdown = backtester.calculate_max_drawdown(pnl_series.cumsum())
                    volatility = pnl_series.std() * (252**0.5)
                    
                    results[f"df_{name}"] = {
                        'pnl_series': pnl_series,
                        'cumulative_pnl': pnl_series.cumsum(),
                        'total_return': total_return,
                        'sharpe_ratio': sharpe,
                        'max_drawdown': max_drawdown,
                        'volatility': volatility,
                        'num_trades': len(pnl_series),
                        'win_rate': (pnl_series > 0).mean()
                    }
                    
                    logger.info(f"df_{name}: Return={total_return:.4f}, Sharpe={sharpe:.3f}")
                    
                except Exception as e:
                    logger.error(f"Error with DataFrame strategy {name}: {e}")
                    continue
            
            return results
        
        # Replace the method
        backtester.test_position_sizing_strategies = enhanced_test_strategies
        
        # Run the enhanced backtest
        logger.info("Running enhanced comprehensive backtest...")
        results, plot_file, csv_file = backtester.run_comprehensive_backtest("enhanced_backtest_results")
        
        logger.info("Enhanced backtest completed successfully!")
        return results, plot_file, csv_file
    
    def run_quick_test():
        """Run a quick test with limited data."""
        logger.info("Running quick test...")
        
        # Just test with AAPL for now
        backtester = ComprehensiveBacktester(
            symbols=["AAPL"],
            start_date="2024-01-01",
            end_date="2024-06-30"
        )
        
        try:
            results, plot_file, csv_file = backtester.run_comprehensive_backtest("quick_test_results")
            logger.info("Quick test completed successfully!")
            return results, plot_file, csv_file
        except Exception as e:
            logger.error(f"Quick test failed: {e}")
            return None, None, None
    
    if __name__ == "__main__":
        import argparse
        
        parser = argparse.ArgumentParser(description="Run comprehensive backtest")
        parser.add_argument("--quick", action="store_true", help="Run quick test with limited data")
        parser.add_argument("--enhanced", action="store_true", help="Run enhanced backtest with advanced strategies")
        args = parser.parse_args()
        
        if args.quick:
            run_quick_test()
        elif args.enhanced:
            run_enhanced_backtest()
        else:
            # Run standard backtest
            from comprehensive_backtest_real_gpu import main
            main()

except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory and .venv is activated")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)