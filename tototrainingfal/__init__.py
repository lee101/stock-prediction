"""Fal-friendly Toto training helpers with injectable heavy dependencies."""

from __future__ import annotations

from .runner import run_training, setup_training_imports

__all__ = ["run_training", "setup_training_imports"]
