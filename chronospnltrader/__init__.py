"""ChronosPnL Trader - Neural trading with Chronos2 PnL forecasting.

Key innovation: Uses Chronos2 both for price forecasting AND as a "judge"
to predict whether the strategy's PnL will be profitable tomorrow.
The neural model is trained to maximize predicted next-day profitability.

Components:
- config.py: Configuration dataclasses
- data.py: Hourly data loading with market hours filtering
- forecaster.py: Chronos2 wrapper for price and PnL forecasting
- simple_algo.py: Simple baseline algorithm using Chronos2
- simulator.py: Differentiable market simulator with PnL tracking
- model.py: Neural transformer model for trade optimization
- trainer.py: Training loop with Chronos2 PnL judge
- run.py: CLI entry point

Usage:
    # Train on a single symbol
    python -m chronospnltrader.run train --symbol AAPL

    # Train on multiple symbols
    python -m chronospnltrader.run train --multi --symbols AAPL,MSFT,GOOG

    # Compare neural vs simple algorithm
    python -m chronospnltrader.run compare --symbol AAPL
"""

from chronospnltrader.config import (
    DataConfig,
    ForecastConfig,
    PolicyConfig,
    SimpleAlgoConfig,
    SimulationConfig,
    TrainingConfig,
)
from chronospnltrader.data import ChronosPnLDataModule, get_all_stock_symbols
from chronospnltrader.forecaster import Chronos2Forecaster, create_forecaster
from chronospnltrader.model import ChronosPnLPolicy, create_model
from chronospnltrader.simple_algo import SimpleChronosAlgo
from chronospnltrader.simulator import run_simulation_30_days, simulate_trade
from chronospnltrader.trainer import ChronosPnLTrainer, train_multi_symbol, train_single_symbol

__all__ = [
    # Config
    "DataConfig",
    "ForecastConfig",
    "PolicyConfig",
    "SimpleAlgoConfig",
    "SimulationConfig",
    "TrainingConfig",
    # Data
    "ChronosPnLDataModule",
    "get_all_stock_symbols",
    # Forecaster
    "Chronos2Forecaster",
    "create_forecaster",
    # Model
    "ChronosPnLPolicy",
    "create_model",
    # Simple algo
    "SimpleChronosAlgo",
    # Simulator
    "run_simulation_30_days",
    "simulate_trade",
    # Trainer
    "ChronosPnLTrainer",
    "train_multi_symbol",
    "train_single_symbol",
]

__version__ = "0.1.0"
