"""Neural Hourly Stocks V5 - Hourly stock trading with learned position timing.

Key features:
- Learned position length (0-24 hours): 0 = skip, 1-24 = hold hours
- Fee-aware price clamping with 2bps dead zone (stock maker fees)
- Market hours filtering (9:30 AM - 4:00 PM ET only)
- 168-bar sequences with 4-hour patching
- Multi-scale Chronos2 integration (1h, 2h, 4h)
- Differentiable simulation with temperature annealing
"""
from neuralhourlystocksv5.config import (
    DatasetConfigStocksV5,
    DefaultStockStrategyConfig,
    PolicyConfigStocksV5,
    SimulationConfigStocksV5,
    TemperatureScheduleV5,
    TrainingConfigStocksV5,
)
from neuralhourlystocksv5.model import HourlyStockPolicyV5
from neuralhourlystocksv5.data import (
    HourlyStockDataModuleV5,
    MultiSymbolStockDataModuleV5,
    HourlyStockDatasetV5,
    StockFeatureNormalizer,
    HOURLY_FEATURES_STOCKS_V5,
)
from neuralhourlystocksv5.simulation import (
    simulate_hourly_trade,
    compute_v5_loss,
    HourlyTradeResult,
)
from neuralhourlystocksv5.trainer import HourlyStockTrainerV5
from neuralhourlystocksv5.backtest import (
    HourlyStockMarketSimulatorV5,
    run_10day_validation,
    BacktestResult,
    TradeRecord,
)

__all__ = [
    # Config
    "DatasetConfigStocksV5",
    "DefaultStockStrategyConfig",
    "PolicyConfigStocksV5",
    "SimulationConfigStocksV5",
    "TemperatureScheduleV5",
    "TrainingConfigStocksV5",
    # Model
    "HourlyStockPolicyV5",
    # Data
    "HourlyStockDataModuleV5",
    "MultiSymbolStockDataModuleV5",
    "HourlyStockDatasetV5",
    "StockFeatureNormalizer",
    "HOURLY_FEATURES_STOCKS_V5",
    # Simulation
    "simulate_hourly_trade",
    "compute_v5_loss",
    "HourlyTradeResult",
    # Training
    "HourlyStockTrainerV5",
    # Backtest
    "HourlyStockMarketSimulatorV5",
    "run_10day_validation",
    "BacktestResult",
    "TradeRecord",
]
