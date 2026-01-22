#!/usr/bin/env python
"""Main entry point for ChronosPnL Trader.

Usage:
    # Train on a single symbol
    python -m chronospnltrader.run train --symbol AAPL

    # Train on multiple symbols
    python -m chronospnltrader.run train --multi --symbols AAPL,MSFT,GOOG

    # Train on all available symbols
    python -m chronospnltrader.run train --multi --all-symbols

    # Evaluate a trained model
    python -m chronospnltrader.run eval --checkpoint chronospnltrader/checkpoints/best.pt

    # Compare neural model vs simple algorithm
    python -m chronospnltrader.run compare --symbol AAPL

    # Run simple algorithm only
    python -m chronospnltrader.run simple --symbol AAPL
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import torch

from chronospnltrader.config import (
    DataConfig,
    ForecastConfig,
    SimpleAlgoConfig,
    SimulationConfig,
    TrainingConfig,
)
from chronospnltrader.data import ChronosPnLDataModule, get_all_stock_symbols
from chronospnltrader.forecaster import Chronos2Forecaster, create_forecaster
from chronospnltrader.model import create_model
from chronospnltrader.simple_algo import SimpleChronosAlgo
from chronospnltrader.simulator import run_simulation_30_days
from chronospnltrader.trainer import ChronosPnLTrainer, train_multi_symbol, train_single_symbol

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def train_command(args: argparse.Namespace) -> None:
    """Train the ChronosPnL model."""
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        run_name=args.run_name or "chronospnltrader",
    )

    if args.multi:
        if args.all_symbols:
            symbols = get_all_stock_symbols()
            logger.info(f"Training on all {len(symbols)} symbols")
        else:
            symbols = args.symbols.split(",") if args.symbols else None
            if symbols is None:
                symbols = get_all_stock_symbols()[:50]
            logger.info(f"Training on {len(symbols)} symbols: {symbols[:5]}...")

        metrics = train_multi_symbol(symbols, config)
    else:
        symbol = args.symbol or "AAPL"
        logger.info(f"Training on single symbol: {symbol}")
        metrics = train_single_symbol(symbol, config)

    logger.info(f"Training complete!")
    logger.info(f"Final metrics: Sortino={metrics.val_sortino:.2f}, PnL={metrics.val_pnl:.6f}")


def eval_command(args: argparse.Namespace) -> None:
    """Evaluate a trained model."""
    if not args.checkpoint or not Path(args.checkpoint).exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint.get("config", TrainingConfig())

    # Create data module
    symbol = args.symbol or "AAPL"
    data_config = DataConfig(symbols=(symbol,))
    data_module = ChronosPnLDataModule(data_config)

    # Create model
    input_dim = len(data_module.feature_columns)
    policy_config = config.get_policy_config(input_dim)
    model = create_model(policy_config)
    model.load_state_dict(checkpoint["model_state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Run 30-day simulation
    sim_config = config.get_simulation_config()
    results = run_simulation_30_days(
        data_module=data_module,
        model=model,
        config=sim_config,
        device=device,
        use_simple_algo=False,
    )

    logger.info("=" * 50)
    logger.info("30-Day Simulation Results (Neural Model)")
    logger.info("=" * 50)
    logger.info(f"Total PnL: {results['total_pnl']:.4f}")
    logger.info(f"Mean Return: {results['mean_return']:.6f}")
    logger.info(f"Sharpe Ratio: {results['sharpe']:.2f}")
    logger.info(f"Sortino Ratio: {results['sortino']:.2f}")
    logger.info(f"Win Rate: {results['win_rate']:.2%}")
    logger.info(f"Trade Count: {results['trade_count']}")


def simple_command(args: argparse.Namespace) -> None:
    """Run the simple Chronos2-based algorithm."""
    symbol = args.symbol or "AAPL"
    logger.info(f"Running simple algorithm on {symbol}")

    # Create components
    data_config = DataConfig(symbols=(symbol,))
    data_module = ChronosPnLDataModule(data_config)

    forecast_config = ForecastConfig()
    sim_config = SimulationConfig()
    algo_config = SimpleAlgoConfig()

    # Note: Forecaster initialization is lazy
    forecaster = create_forecaster(forecast_config)
    algo = SimpleChronosAlgo(algo_config, sim_config, forecaster)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Run simulation using simple algorithm
    results = run_simulation_30_days(
        data_module=data_module,
        model=None,  # Not used
        config=sim_config,
        device=device,
        use_simple_algo=True,
    )

    logger.info("=" * 50)
    logger.info("30-Day Simulation Results (Simple Algorithm)")
    logger.info("=" * 50)
    logger.info(f"Total PnL: {results['total_pnl']:.4f}")
    logger.info(f"Mean Return: {results['mean_return']:.6f}")
    logger.info(f"Sharpe Ratio: {results['sharpe']:.2f}")
    logger.info(f"Sortino Ratio: {results['sortino']:.2f}")
    logger.info(f"Win Rate: {results['win_rate']:.2%}")
    logger.info(f"Trade Count: {results['trade_count']}")


def compare_command(args: argparse.Namespace) -> None:
    """Compare neural model vs simple algorithm."""
    symbol = args.symbol or "AAPL"
    checkpoint_path = args.checkpoint or "chronospnltrader/checkpoints/best.pt"

    logger.info(f"Comparing algorithms on {symbol}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create data module
    data_config = DataConfig(symbols=(symbol,))
    data_module = ChronosPnLDataModule(data_config)

    # Simple algorithm results
    sim_config = SimulationConfig()
    simple_results = run_simulation_30_days(
        data_module=data_module,
        model=None,
        config=sim_config,
        device=device,
        use_simple_algo=True,
    )

    # Neural model results (if checkpoint exists)
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = checkpoint.get("config", TrainingConfig())

        input_dim = len(data_module.feature_columns)
        policy_config = config.get_policy_config(input_dim)
        model = create_model(policy_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)

        neural_results = run_simulation_30_days(
            data_module=data_module,
            model=model,
            config=sim_config,
            device=device,
            use_simple_algo=False,
        )
    else:
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        neural_results = None

    # Print comparison
    logger.info("=" * 60)
    logger.info("30-Day Simulation Comparison")
    logger.info("=" * 60)
    logger.info(f"{'Metric':<20} {'Simple':<15} {'Neural':<15} {'Winner':<10}")
    logger.info("-" * 60)

    metrics = ["total_pnl", "mean_return", "sharpe", "sortino", "win_rate"]

    for metric in metrics:
        simple_val = simple_results[metric]
        if neural_results:
            neural_val = neural_results[metric]
            winner = "Neural" if neural_val > simple_val else "Simple"
            logger.info(f"{metric:<20} {simple_val:<15.4f} {neural_val:<15.4f} {winner:<10}")
        else:
            logger.info(f"{metric:<20} {simple_val:<15.4f} {'N/A':<15} {'Simple':<10}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ChronosPnL Trader - Neural trading with Chronos2 PnL forecasting"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--symbol", type=str, help="Symbol to train on (single mode)")
    train_parser.add_argument("--multi", action="store_true", help="Train on multiple symbols")
    train_parser.add_argument("--symbols", type=str, help="Comma-separated symbols (multi mode)")
    train_parser.add_argument("--all-symbols", action="store_true", help="Train on all available symbols")
    train_parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    train_parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    train_parser.add_argument("--run-name", type=str, help="Name for this training run")

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained model")
    eval_parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    eval_parser.add_argument("--symbol", type=str, help="Symbol to evaluate on")

    # Simple command
    simple_parser = subparsers.add_parser("simple", help="Run simple algorithm")
    simple_parser.add_argument("--symbol", type=str, help="Symbol to run on")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare neural vs simple")
    compare_parser.add_argument("--symbol", type=str, help="Symbol to compare on")
    compare_parser.add_argument("--checkpoint", type=str, help="Path to neural model checkpoint")

    args = parser.parse_args()

    if args.command == "train":
        train_command(args)
    elif args.command == "eval":
        eval_command(args)
    elif args.command == "simple":
        simple_command(args)
    elif args.command == "compare":
        compare_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
