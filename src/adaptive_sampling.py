"""
Helpers for adaptively tuning Monte Carlo sampling bounds during backtesting.

These utilities are intentionally dependency-light so they can be exercised in
unit tests without pulling in heavyweight forecasting stacks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class AdaptiveSamplerBounds:
    """
    Track adaptive minimum sampling bounds with monotonic downwards adjustment.

    The class keeps the original (baseline) minima alongside the effective
    minima currently in force. Call ``reduce()`` when GPU memory pressure is
    detected to gradually relax the minima until a configured floor is reached.
    """

    base_min_num: int
    base_min_batch: int
    num_floor: int
    batch_floor: int

    def __post_init__(self) -> None:
        if self.base_min_num <= 0 or self.base_min_batch <= 0:
            raise ValueError("Baseline minima must be positive.")
        if self.base_min_batch > self.base_min_num:
            raise ValueError("Batch baseline cannot exceed num baseline.")
        if self.num_floor <= 0 or self.batch_floor <= 0:
            raise ValueError("Sampling floors must be positive.")
        if self.batch_floor > self.num_floor:
            raise ValueError("Batch floor cannot exceed num floor.")

        self._min_num = max(self.base_min_num, self.num_floor)
        self._min_batch = max(self.base_min_batch, self.batch_floor)
        if self._min_batch > self._min_num:
            self._min_batch = self._min_num

    @property
    def min_num(self) -> int:
        """Current minimum Monte Carlo sample count."""
        return self._min_num

    @property
    def min_batch(self) -> int:
        """Current minimum samples-per-batch."""
        return self._min_batch

    def minima(self) -> Tuple[int, int]:
        """Return the current (num_min, batch_min) pair."""
        return self._min_num, self._min_batch

    def clamp(self, num_samples: int, samples_per_batch: int, *, max_num: int, max_batch: int) -> Tuple[int, int]:
        """
        Constrain sampling parameters to lie within the adaptive minima and supplied maxima.
        Ensures ``num_samples`` is divisible by ``samples_per_batch`` by trimming excess.
        """
        if max_num <= 0 or max_batch <= 0:
            raise ValueError("Sampling maxima must be positive.")

        effective_num = max(self._min_num, min(max_num, num_samples))
        effective_batch = max(self._min_batch, min(max_batch, samples_per_batch, effective_num))
        if effective_batch <= 0:
            effective_batch = self._min_batch
        if effective_num < effective_batch:
            effective_num = effective_batch

        remainder = effective_num % effective_batch
        if remainder:
            effective_num -= remainder
            if effective_num < effective_batch:
                effective_num = effective_batch

        return effective_num, effective_batch

    def reduce(self) -> bool:
        """
        Halve the adaptive minima while respecting configured floors.

        Returns True when a reduction occurred, False when already at the floor.
        """
        next_min_num = max(self.num_floor, self._min_num // 2)
        next_min_batch = max(self.batch_floor, self._min_batch // 2)

        changed = False
        if next_min_num < self._min_num:
            self._min_num = max(next_min_num, self.batch_floor)
            changed = True

        if next_min_batch < self._min_batch:
            self._min_batch = max(next_min_batch, self.batch_floor)
            changed = True

        if self._min_batch > self._min_num:
            self._min_batch = self._min_num

        return changed

    def reset(self) -> None:
        """Restore the adaptive minima back to their baseline values."""
        self._min_num = max(self.base_min_num, self.num_floor)
        self._min_batch = max(self.base_min_batch, self.batch_floor)
        if self._min_batch > self._min_num:
            self._min_batch = self._min_num
