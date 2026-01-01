"""Neural Daily V4 - Chronos-2 inspired architecture.

Key features:
- Patching: 5-day patches for faster processing
- Multi-window: Direct prediction of multiple future windows
- Quantile outputs: Price distribution instead of point estimates
- Trimmed mean: Robust aggregation across windows
- Learned position sizing: Dynamic sizing based on uncertainty
"""

from neuraldailyv4.config import (
    PolicyConfigV4,
    SimulationConfigV4,
    DailyTrainingConfigV4,
    DailyDatasetConfigV4,
)
from neuraldailyv4.model import MultiSymbolPolicyV4

__all__ = [
    "PolicyConfigV4",
    "SimulationConfigV4",
    "DailyTrainingConfigV4",
    "DailyDatasetConfigV4",
    "MultiSymbolPolicyV4",
]
