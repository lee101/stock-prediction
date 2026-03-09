"""FastForecaster2: high-throughput, MAE-focused forecasting pipeline."""

from .config import FastForecaster2Config
from .data import DataBundle, ForecastWindowDataset, build_data_bundle
from .model import FastForecaster2Model
from .trainer import FastForecaster2Trainer

__all__ = [
    "DataBundle",
    "FastForecaster2Config",
    "FastForecaster2Model",
    "FastForecaster2Trainer",
    "ForecastWindowDataset",
    "build_data_bundle",
]
