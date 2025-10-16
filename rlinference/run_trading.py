#!/usr/bin/env python3
"""
RL-based Trading System
Main entry point for running the trading engine with trained RL models.
"""

import argparse
import sys
import os
from pathlib import Path
from loguru import logger

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from rlinference.configs.trading_config import TradingConfig, AlpacaConfig
from rlinference.rl_trading_engine import RLTradingEngine


def main():
    parser = argparse.ArgumentParser(description='Run RL Trading System')
    
    # Trading configuration
    parser.add_argument('--symbols', nargs='+', default=['AAPL', 'NVDA', 'TSLA', 'SPY'],
                       help='Symbols to trade')
    parser.add_argument('--initial-balance', type=float, default=100000,
                       help='Initial account balance')
    parser.add_argument('--max-positions', type=int, default=2,
                       help='Maximum number of concurrent positions')
    parser.add_argument('--max-position-size', type=float, default=0.47,
                       help='Maximum position size as fraction of equity')
    
    # Model configuration
    parser.add_argument('--models-dir', type=str, default='models',
                       help='Directory containing trained models')
    parser.add_argument('--use-ensemble', action='store_true',
                       help='Use ensemble of top-k models')
    
    # Risk management
    parser.add_argument('--stop-loss', type=float, default=0.05,
                       help='Stop loss percentage (0.05 = 5%)')
    parser.add_argument('--take-profit', type=float, default=0.20,
                       help='Take profit percentage (0.20 = 20%)')
    parser.add_argument('--max-drawdown', type=float, default=0.10,
                       help='Maximum drawdown before stopping')
    parser.add_argument('--circuit-breaker', type=float, default=0.15,
                       help='Daily loss limit before stopping')
    
    # Trading mode
    parser.add_argument('--paper', action='store_true', default=True,
                       help='Use paper trading (default)')
    parser.add_argument('--live', action='store_true',
                       help='Use live trading (requires confirmation)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Dry run mode - no actual trades')
    
    # Execution
    parser.add_argument('--interval', type=int, default=300,
                       help='Trading interval in seconds (default: 5 minutes)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.live and not args.paper:
        confirm = input("WARNING: Live trading mode selected. Are you sure? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Exiting...")
            sys.exit(0)
    
    # Create configuration
    config = TradingConfig(
        symbols=args.symbols,
        initial_balance=args.initial_balance,
        max_positions=args.max_positions,
        max_position_size=args.max_position_size,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit,
        max_drawdown_stop=args.max_drawdown,
        circuit_breaker_loss=args.circuit_breaker,
        models_dir=Path(args.models_dir),
        use_top_k_models=args.use_ensemble,
        ensemble_predictions=args.use_ensemble,
        paper_trading=not args.live,
        dry_run=args.dry_run,
        log_level=args.log_level
    )
    
    # Create Alpaca configuration
    # Make sure to set environment variables:
    # - ALP_KEY_ID_PAPER / ALP_KEY_ID_PROD
    # - ALP_SECRET_KEY_PAPER / ALP_SECRET_KEY_PROD
    alpaca_config = AlpacaConfig.from_env(paper=config.paper_trading)
    
    # Validate API keys
    if not alpaca_config.api_key or not alpaca_config.secret_key:
        logger.error("Alpaca API keys not found in environment variables")
        logger.error("Please set ALP_KEY_ID_PAPER/PROD and ALP_SECRET_KEY_PAPER/PROD")
        sys.exit(1)
    
    # Log configuration
    logger.info("=" * 60)
    logger.info("RL Trading System Configuration")
    logger.info("=" * 60)
    logger.info(f"Symbols: {config.symbols}")
    logger.info(f"Mode: {'PAPER' if config.paper_trading else 'LIVE'}")
    logger.info(f"Dry Run: {config.dry_run}")
    logger.info(f"Initial Balance: ${config.initial_balance:,.2f}")
    logger.info(f"Max Positions: {config.max_positions}")
    logger.info(f"Max Position Size: {config.max_position_size:.1%}")
    logger.info(f"Stop Loss: {config.stop_loss:.1%}")
    logger.info(f"Take Profit: {config.take_profit:.1%}")
    logger.info(f"Models Directory: {config.models_dir}")
    logger.info(f"Use Ensemble: {config.ensemble_predictions}")
    logger.info(f"Trading Interval: {args.interval} seconds")
    logger.info("=" * 60)
    
    # Create and run trading engine
    try:
        engine = RLTradingEngine(config, alpaca_config)
        engine.run(interval_seconds=args.interval)
    except KeyboardInterrupt:
        logger.info("Trading stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()