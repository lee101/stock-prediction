#!/usr/bin/env python3
"""
Train RL models for individual stocks.
This script trains separate models for each stock symbol.
"""

import argparse
import sys
import os
from pathlib import Path
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from loguru import logger
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def train_single_stock(symbol: str, args):
    """Train a model for a single stock."""
    
    logger.info(f"Starting training for {symbol}")
    
    # Prepare command
    cmd = [
        sys.executable,
        "training/train_rl_agent.py",
        "--symbol", symbol,
        "--data_dir", args.data_dir,
        "--save_dir", f"models/{symbol}",
        "--num_episodes", str(args.num_episodes),
        "--window_size", str(args.window_size),
        "--initial_balance", str(args.initial_balance),
        "--lr_actor", str(args.lr_actor),
        "--lr_critic", str(args.lr_critic),
        "--top_k", str(args.top_k)
    ]
    
    # Create output directory
    output_dir = Path(f"models/{symbol}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run training
    log_file = output_dir / f"training_{datetime.now():%Y%m%d_%H%M%S}.log"
    
    try:
        with open(log_file, 'w') as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                check=True
            )
        
        # Copy best model to main models directory with symbol prefix
        best_model_src = output_dir / "best_model.pth"
        best_model_dst = Path("models") / f"{symbol}_best_model.pth"
        
        if best_model_src.exists():
            import shutil
            shutil.copy2(best_model_src, best_model_dst)
            logger.info(f"Model for {symbol} saved to {best_model_dst}")
        
        # Copy top-k summary
        top_k_src = output_dir / "top_k_summary.json"
        top_k_dst = Path("models") / f"{symbol}_top_k_summary.json"
        
        if top_k_src.exists():
            shutil.copy2(top_k_src, top_k_dst)
        
        return {
            'symbol': symbol,
            'status': 'success',
            'model_path': str(best_model_dst),
            'log_file': str(log_file)
        }
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed for {symbol}: {e}")
        return {
            'symbol': symbol,
            'status': 'failed',
            'error': str(e),
            'log_file': str(log_file)
        }
    except Exception as e:
        logger.error(f"Unexpected error training {symbol}: {e}")
        return {
            'symbol': symbol,
            'status': 'error',
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description='Train RL models for multiple stocks')
    
    # Stock selection
    parser.add_argument('--symbols', nargs='+', 
                       default=['AAPL', 'NVDA', 'TSLA', 'GOOGL', 'MSFT', 'AMZN', 'SPY', 'QQQ'],
                       help='Stock symbols to train models for')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory containing stock data')
    
    # Training parameters
    parser.add_argument('--num-episodes', type=int, default=500,
                       help='Number of training episodes per stock')
    parser.add_argument('--window-size', type=int, default=30,
                       help='Observation window size')
    parser.add_argument('--initial-balance', type=float, default=10000,
                       help='Initial balance for training environment')
    
    # Model parameters
    parser.add_argument('--lr-actor', type=float, default=3e-4,
                       help='Actor learning rate')
    parser.add_argument('--lr-critic', type=float, default=1e-3,
                       help='Critic learning rate')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of top profitable models to keep')
    
    # Execution
    parser.add_argument('--parallel', type=int, default=2,
                       help='Number of parallel training processes')
    parser.add_argument('--sequential', action='store_true',
                       help='Train models sequentially instead of in parallel')
    
    args = parser.parse_args()
    
    # Setup logging
    log_dir = Path("models/training_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        log_dir / f"multi_train_{datetime.now():%Y%m%d_%H%M%S}.log",
        level="INFO",
        format="{time} {level} {message}"
    )
    
    logger.info("=" * 60)
    logger.info("Multi-Stock RL Training")
    logger.info("=" * 60)
    logger.info(f"Symbols: {args.symbols}")
    logger.info(f"Episodes per stock: {args.num_episodes}")
    logger.info(f"Parallel processes: {args.parallel if not args.sequential else 1}")
    logger.info("=" * 60)
    
    # Train models
    results = []
    
    if args.sequential:
        # Sequential training
        for symbol in args.symbols:
            result = train_single_stock(symbol, args)
            results.append(result)
            logger.info(f"Completed {symbol}: {result['status']}")
    else:
        # Parallel training
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(train_single_stock, symbol, args): symbol
                for symbol in args.symbols
            }
            
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Completed {symbol}: {result['status']}")
                except Exception as e:
                    logger.error(f"Failed to get result for {symbol}: {e}")
                    results.append({
                        'symbol': symbol,
                        'status': 'error',
                        'error': str(e)
                    })
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'symbols': args.symbols,
        'parameters': vars(args),
        'results': results
    }
    
    summary_file = Path("models/training_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    logger.info("=" * 60)
    logger.info("Training Summary")
    logger.info("=" * 60)
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] != 'success']
    
    logger.info(f"Successful: {len(successful)}/{len(results)}")
    if successful:
        logger.info("Successfully trained models for:")
        for r in successful:
            logger.info(f"  - {r['symbol']}: {r.get('model_path', 'N/A')}")
    
    if failed:
        logger.warning(f"Failed: {len(failed)}")
        for r in failed:
            logger.warning(f"  - {r['symbol']}: {r.get('error', 'Unknown error')}")
    
    logger.info(f"Summary saved to: {summary_file}")
    
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())