"""Simple registry mapping strategy names to experiment classes."""

from __future__ import annotations

from typing import Dict, Type

from .base import StrategyExperiment

_REGISTRY: Dict[str, Type[StrategyExperiment]] = {}


def register(name: str):
    """Decorator used by strategy modules."""

    def _wrap(cls: Type[StrategyExperiment]) -> Type[StrategyExperiment]:
        if name in _REGISTRY:
            raise ValueError(f"Duplicate experiment registration for '{name}'")
        _REGISTRY[name] = cls
        return cls

    return _wrap


def get_experiment_class(name: str) -> Type[StrategyExperiment]:
    try:
        return _REGISTRY[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"Unknown experiment '{name}'. Registered: {list(_REGISTRY)}") from exc


def list_registered_strategies() -> Dict[str, str]:
    return {name: cls.__name__ for name, cls in _REGISTRY.items()}
