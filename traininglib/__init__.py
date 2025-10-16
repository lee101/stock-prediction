"""
Utilities for experimenting with optimizer configurations in isolation before
integrating them into the wider training pipeline.

The package exposes a small registry that keeps advanced optimizers such as
Shampoo and Muon side-by-side with baseline choices like AdamW.  The registry
is intentionally lightweight so the same builders can be reused both inside
unit tests/benchmarks and within Hugging Face `Trainer` entry points.
"""

from .optimizers import OptimizerRegistry, optimizer_registry
from .optimizers import create_optimizer  # noqa: F401  (re-export for convenience)
from .benchmarking import RegressionBenchmark  # noqa: F401
from .hf_integration import build_hf_optimizers  # noqa: F401

__all__ = [
    "OptimizerRegistry",
    "optimizer_registry",
    "create_optimizer",
    "RegressionBenchmark",
    "build_hf_optimizers",
]
