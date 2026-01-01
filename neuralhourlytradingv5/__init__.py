"""Neural Hourly Trading V5 - Hourly crypto trading with learned position timing.

Key features:
- Learned position length (0-24 hours): 0 = skip, 1-24 = hold hours
- Fee-aware price clamping with 8bps dead zone
- 168-hour sequences with 4-hour patching
- Multi-scale Chronos2 integration (1h, 2h, 4h)
- Differentiable simulation with temperature annealing
"""
from neuralhourlytradingv5.config import (
    DatasetConfigV5,
    DefaultStrategyConfig,
    PolicyConfigV5,
    SimulationConfigV5,
    TemperatureScheduleV5,
    TrainingConfigV5,
)
from neuralhourlytradingv5.model import HourlyCryptoPolicyV5
from neuralhourlytradingv5.data import (
    HourlyDataModuleV5,
    MultiSymbolDataModuleV5,
    HourlyDatasetV5,
    FeatureNormalizer,
    HOURLY_FEATURES_V5,
)
from neuralhourlytradingv5.simulation import (
    simulate_hourly_trade,
    compute_v5_loss,
    HourlyTradeResult,
)
from neuralhourlytradingv5.trainer import HourlyCryptoTrainerV5
from neuralhourlytradingv5.backtest import (
    HourlyMarketSimulatorV5,
    run_10day_validation,
    BacktestResult,
    TradeRecord,
)

__all__ = [
    # Config
    "DatasetConfigV5",
    "DefaultStrategyConfig",
    "PolicyConfigV5",
    "SimulationConfigV5",
    "TemperatureScheduleV5",
    "TrainingConfigV5",
    # Model
    "HourlyCryptoPolicyV5",
    # Data
    "HourlyDataModuleV5",
    "MultiSymbolDataModuleV5",
    "HourlyDatasetV5",
    "FeatureNormalizer",
    "HOURLY_FEATURES_V5",
    # Simulation
    "simulate_hourly_trade",
    "compute_v5_loss",
    "HourlyTradeResult",
    # Training
    "HourlyCryptoTrainerV5",
    # Backtest
    "HourlyMarketSimulatorV5",
    "run_10day_validation",
    "BacktestResult",
    "TradeRecord",
]
