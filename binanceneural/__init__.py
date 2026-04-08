"""Binance-focused hourly neural trading policy and simulation utilities."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "BinanceHourlyPolicy",
    "BinanceHourlyPolicyNano",
    "DatasetConfig",
    "ForecastConfig",
    "PolicyConfig",
    "TRAINER_BACKENDS",
    "TrainingConfig",
    "TrainerBackend",
    "build_trainer",
    "build_policy",
]


_EXPORTS = {
    "BinanceHourlyPolicy": ("binanceneural.model", "BinanceHourlyPolicy"),
    "BinanceHourlyPolicyNano": ("binanceneural.model", "BinanceHourlyPolicyNano"),
    "PolicyConfig": ("binanceneural.model", "PolicyConfig"),
    "build_policy": ("binanceneural.model", "build_policy"),
    "DatasetConfig": ("binanceneural.config", "DatasetConfig"),
    "ForecastConfig": ("binanceneural.config", "ForecastConfig"),
    "TRAINER_BACKENDS": ("binanceneural.config", "TRAINER_BACKENDS"),
    "TrainerBackend": ("binanceneural.config", "TrainerBackend"),
    "TrainingConfig": ("binanceneural.config", "TrainingConfig"),
    "build_trainer": ("binanceneural.trainer_factory", "build_trainer"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
