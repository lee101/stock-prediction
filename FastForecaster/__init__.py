"""FastForecaster: high-throughput, MAE-focused forecasting pipeline."""

from .config import FastForecasterConfig
from .data import DataBundle, ForecastWindowDataset, build_data_bundle
from .model import FastForecasterModel
from .trainer import FastForecasterTrainer

__all__ = [
    "DataBundle",
    "FastForecasterConfig",
    "FastForecasterModel",
    "FastForecasterTrainer",
    "ForecastWindowDataset",
    "build_data_bundle",
]
