"""
pufferlibtraining2
==================

High-throughput reinforcement learning pipeline for the stock trading simulator.
This package wires the differentiable trading environment into PufferLib's
`PuffeRL` trainer, providing a composable configuration system, vectorised
environment builders, and production-grade logging hooks (TensorBoard + Weights
and Biases).
"""

from .config import TrainingPlan, load_plan


def train(*args, **kwargs):
    from .trainer import train as _train

    return _train(*args, **kwargs)


def run_with_config(*args, **kwargs):
    from .trainer import run_with_config as _run_with_config

    return _run_with_config(*args, **kwargs)


__all__ = ["TrainingPlan", "load_plan", "train", "run_with_config"]
