#!/usr/bin/env python3
"""
Simple runner script for the backtesting system
Usage: python run_backtest.py [--days DAYS] [--real-ai] [--symbols SYMBOL1,SYMBOL2,...]
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from loguru import logger

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))



def main():
    parser = argparse.ArgumentParser(description='Run trading strategy backtests')
    parser.add_argument('--days', type=int, default=25, 
                       help='Number of days to simulate (default: 25)')
    parser.add_argument('--real-ai', action='store_true',
                       help='Use real AI forecasts instead of synthetic')
    parser.add_argument('--symbols', type=str, 
                       default='BTCUSD,ETHUSD,NVDA,TSLA,AAPL,GOOG,META,MSFT',
                       help='Comma-separated list of symbols to test')
    parser.add_argument('--compare', action='store_true',
                       help='Compare real AI vs synthetic forecasts')
    parser.add_argument('--torch', action='store_true',
                       help='Run differentiable PyTorch backtester with gradient-based optimisation')
    parser.add_argument('--torch-steps', type=int, default=200,
                       help='Gradient steps for the PyTorch backtester (default: 200)')
    parser.add_argument('--torch-lr', type=float, default=0.05,
                       help='Learning rate for the PyTorch backtester (default: 0.05)')
    parser.add_argument('--lookback', type=int, default=5,
                       help='Lookback window (days) for PyTorch forecast features (default: 5)')
    
    args = parser.parse_args()
    
    # Parse symbols
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    logger.info(f"Running backtest with following parameters:")
    logger.info(f"  Days: {args.days}")
    logger.info(f"  Use Real AI: {args.real_ai}")
    logger.info(f"  Symbols: {symbols}")
    logger.info(f"  Compare Mode: {args.compare}")
    
    # Create results directory
    Path("simulationresults").mkdir(exist_ok=True)
    
    # Differentiable PyTorch path
    if args.torch:
        logger.info("\nUsing PyTorch differentiable backtester...")
        from torch_backtester import run_torch_backtest

        torch_summary = run_torch_backtest(
            symbols,
            simulation_days=args.days,
            lookback=args.lookback,
            optimisation_steps=args.torch_steps,
            lr=args.torch_lr,
        )

        logger.info("\nPyTorch Backtest Summary:")
        logger.info(f"  Final equity: ${torch_summary['final_equity']:,.2f}")
        logger.info(f"  Total return: {torch_summary['total_return']*100:.2f}%")
        logger.info(f"  Sharpe ratio: {torch_summary['sharpe']:.2f}")
        logger.info(f"  Max drawdown: {torch_summary['max_drawdown']*100:.2f}%")
        logger.info(f"  Device used: {torch_summary['device']}")
        logger.info("  Policy parameters: {}", torch_summary['policy_state'])
        return

    # Run appropriate classical backtest
    if args.real_ai or args.compare:
        # Use enhanced backtester
        logger.info("\nUsing Enhanced Backtester with AI integration...")
        from enhanced_local_backtester import run_enhanced_comparison
        results = run_enhanced_comparison(
            symbols, 
            simulation_days=args.days,
            compare_with_synthetic=args.compare
        )
    else:
        # Use basic backtester
        logger.info("\nUsing Basic Backtester with synthetic forecasts...")
        from local_backtesting_system import run_strategy_comparison
        results = run_strategy_comparison(symbols, simulation_days=args.days)
    
    logger.info("\n" + "="*80)
    logger.info("BACKTEST COMPLETE!")
    logger.info("="*80)
    logger.info("Results saved to simulationresults/ directory")
    logger.info("Check the generated charts and JSON files for detailed analysis")
    
    # Print summary of best strategy
    if isinstance(results, dict) and results:
        best_strategy = None
        best_return = -float('inf')
        
        # Handle both return formats
        if isinstance(results, tuple):
            # Enhanced comparison returns (real, synthetic)
            results_to_check = results[0]  # Use real AI results
        else:
            results_to_check = results
            
        for strategy, data in results_to_check.items():
            if 'total_return_pct' in data and data['total_return_pct'] > best_return:
                best_return = data['total_return_pct']
                best_strategy = strategy
        
        if best_strategy:
            logger.info(f"\nBest performing strategy: {best_strategy}")
            logger.info(f"Expected return over {args.days} days: {best_return:.2f}%")
            logger.info(f"Annualized return (if maintained): {(best_return/args.days)*365:.1f}%")


if __name__ == "__main__":
    main()
