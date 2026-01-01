"""NeuralDaily V3 Timed - Neural trading with explicit exit timing.

Key V3 features:
- Model outputs exit_days (1-10) for maximum hold duration
- Trade episode simulation instead of continuous inventory
- TradingPlan includes exit_timestamp for forced exits
- Training matches inference: same exit logic everywhere
"""
from neuraldailyv3timed.config import (
    DailyTrainingConfigV3,
    DailyDatasetConfigV3,
    PolicyConfigV3,
    SimulationConfig,
    TemperatureSchedule,
)
from neuraldailyv3timed.model import MultiSymbolPolicyV3
from neuraldailyv3timed.runtime import DailyTradingRuntimeV3, TradingPlan, compute_exit_timestamp
from neuraldailyv3timed.simulation import (
    TradeEpisodeResult,
    simulate_trade_episode,
    compute_episode_loss,
)
from neuraldailyv3timed.data import (
    DailyDataModuleV3,
    MultiSymbolDatasetV3,
    SymbolFrameBuilderV3,
)

__all__ = [
    # Config
    "DailyTrainingConfigV3",
    "DailyDatasetConfigV3",
    "PolicyConfigV3",
    "SimulationConfig",
    "TemperatureSchedule",
    # Model
    "MultiSymbolPolicyV3",
    # Runtime
    "DailyTradingRuntimeV3",
    "TradingPlan",
    "compute_exit_timestamp",
    # Simulation
    "TradeEpisodeResult",
    "simulate_trade_episode",
    "compute_episode_loss",
    # Data
    "DailyDataModuleV3",
    "MultiSymbolDatasetV3",
    "SymbolFrameBuilderV3",
]
