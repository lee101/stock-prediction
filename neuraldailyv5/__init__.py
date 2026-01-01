"""NeuralDailyV5: NEPA-style Portfolio Latent Prediction.

Key features:
- Sequence-of-latents: Autoregressively predict portfolio embeddings
- Portfolio weights: Target allocations per asset (10% ETH, 20% BTC, etc.)
- Ramp-into-position: Watcher matches target portfolio throughout the day
- NEPA loss: Cosine similarity regularization for coherent sequences
- Sortino optimization: Focus on downside risk
"""

from neuraldailyv5.config import (
    DailyDatasetConfigV5,
    DailyTrainingConfigV5,
    PolicyConfigV5,
    SimulationConfigV5,
)
from neuraldailyv5.model import PortfolioPolicyV5
from neuraldailyv5.simulation import PortfolioSimulatorV5

__all__ = [
    "DailyDatasetConfigV5",
    "DailyTrainingConfigV5",
    "PolicyConfigV5",
    "SimulationConfigV5",
    "PortfolioPolicyV5",
    "PortfolioSimulatorV5",
]
