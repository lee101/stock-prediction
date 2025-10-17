"""
Utilities for fine-tuning Kronos against the local training dataset.

This package wires the external Kronos foundation model into our project-
level workflows, providing custom data loaders, training loops, and
evaluation helpers tailored to the contents of ``trainingdata/``.
"""

from .config import KronosTrainingConfig  # noqa: F401
