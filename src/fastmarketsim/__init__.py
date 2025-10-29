"""
Fast market simulator bindings and Gym environment.

This package exposes a thin Python wrapper around the accelerated C++/LibTorch
market simulator as well as a Gym-compatible environment that mirrors the
behaviour of the Torch-first trading environment.
"""

from .config import build_sim_config
from .env import FastMarketEnv
from .module import load_extension

__all__ = [
    "build_sim_config",
    "FastMarketEnv",
    "load_extension",
]
