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
from .trainer import train, run_with_config

__all__ = ["TrainingPlan", "load_plan", "train", "run_with_config"]
