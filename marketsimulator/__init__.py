"""Market simulator package providing a self-contained mock trading stack."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from .environment import activate_simulation
from .runner import simulate_strategy


_LEGACY_MODULE: ModuleType | None = None


def _load_legacy_shared_cash_api() -> ModuleType:
    legacy_path = Path(__file__).resolve().parent.parent / "marketsimulator.py"
    spec = importlib.util.spec_from_file_location("_marketsimulator_legacy_module", legacy_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load legacy marketsimulator module from {legacy_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _legacy_module() -> ModuleType:
    global _LEGACY_MODULE
    if _LEGACY_MODULE is None:
        _LEGACY_MODULE = _load_legacy_shared_cash_api()
    return _LEGACY_MODULE


def __getattr__(name: str):
    if name in {"SimulationConfig", "run_shared_cash_simulation"}:
        module = _legacy_module()
        return getattr(module, name)
    raise AttributeError(name)


__all__ = [
    "SimulationConfig",
    "activate_simulation",
    "run_shared_cash_simulation",
    "simulate_strategy",
]
