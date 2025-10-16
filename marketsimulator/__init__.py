"""Market simulator package providing a self-contained mock trading stack."""

from .environment import activate_simulation
from .runner import simulate_strategy

__all__ = ["activate_simulation", "simulate_strategy"]
