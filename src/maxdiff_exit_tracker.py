from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ExitTargetTracker:
    """Utility that tracks partial exit progress for take-profit watchers."""

    target_qty: Optional[float]
    initial_qty: Optional[float] = None
    filled_qty: float = 0.0

    def __post_init__(self) -> None:
        self.target_qty = self._normalize(self.target_qty)

    @staticmethod
    def _normalize(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        try:
            qty = float(value)
        except (TypeError, ValueError):
            return None
        return qty if qty > 0 else None

    def is_partial(self) -> bool:
        return self.target_qty is not None

    def remaining(self) -> float:
        if self.target_qty is None:
            return 0.0
        return max(self.target_qty - self.filled_qty, 0.0)

    def plan_order(self, position_qty: float) -> Tuple[float, float, bool]:
        """Return (order_qty, remaining_qty, reached_target)."""
        qty = max(float(position_qty), 0.0)
        if self.target_qty is None:
            return qty, 0.0, False
        if self.initial_qty is None:
            if qty <= 0:
                return 0.0, self.remaining(), False
            self.initial_qty = qty
            self.target_qty = min(self.target_qty, qty)
        self.filled_qty = max(0.0, (self.initial_qty or 0.0) - qty)
        remaining = self.remaining()
        if remaining <= 1e-6:
            return 0.0, 0.0, True
        if qty <= 0:
            return 0.0, remaining, False
        return min(qty, remaining), remaining, False
