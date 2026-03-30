"""Helpers for sampling Cartesian-product hyperparameter grids."""
from __future__ import annotations

import itertools
import random
from typing import Sequence


def cartesian_product_size(value_sets: Sequence[Sequence[object]]) -> int:
    """Return the total number of combinations in a Cartesian product."""
    total = 1
    for values in value_sets:
        total *= len(values)
    return total


def combo_from_cartesian_index(
    value_sets: Sequence[Sequence[object]],
    index: int,
) -> tuple[object, ...]:
    """Map a flat Cartesian-product index to its corresponding combination."""
    total = cartesian_product_size(value_sets)
    if index < 0 or index >= total:
        raise IndexError(f"Cartesian index {index} out of range for total size {total}")

    combo: list[object] = []
    remaining = index
    for values in reversed(value_sets):
        size = len(values)
        if size == 0:
            raise ValueError("Grid values must be non-empty")
        remaining, value_index = divmod(remaining, size)
        combo.append(values[value_index])
    combo.reverse()
    return tuple(combo)


def sample_cartesian_product(
    value_sets: Sequence[Sequence[object]],
    *,
    max_trials: int | None = None,
    seed: int = 42,
) -> tuple[int, list[tuple[object, ...]]]:
    """Sample combinations without materializing the full product when possible."""
    total = cartesian_product_size(value_sets)
    if max_trials is None or max_trials >= total:
        return total, list(itertools.product(*value_sets))

    rng = random.Random(seed)
    sampled_indices = rng.sample(range(total), max_trials)
    combos = [combo_from_cartesian_index(value_sets, index) for index in sampled_indices]
    return total, combos
