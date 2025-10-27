"""
Runtime inference utilities for deploying PufferLib portfolio allocators.

The module exposes a high-level ``PortfolioRLInferenceEngine`` that mirrors the
training data pipeline from ``pufferlibtraining`` and produces allocation
decisions, risk telemetry, and equity curves suitable for integration with the
production trading loop.
"""

from .config import InferenceDataConfig, PufferInferenceConfig
from .engine import AllocationDecision, InferenceResult, PortfolioRLInferenceEngine

__all__ = [
    "InferenceDataConfig",
    "PufferInferenceConfig",
    "AllocationDecision",
    "InferenceResult",
    "PortfolioRLInferenceEngine",
]
