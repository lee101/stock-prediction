from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np


def purged_kfold_indices(
    n_samples: int,
    *,
    n_splits: int = 5,
    embargo: int = 0,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate train/test indices for Purged K-Fold cross-validation with an
    optional embargo period.

    Based on López de Prado, *Advances in Financial Machine Learning* (2018).
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if n_samples <= n_splits:
        raise ValueError("Not enough samples to perform the requested splits")

    fold_sizes = (n_samples // n_splits) * np.ones(n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1

    indices = np.arange(n_samples)
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_idx = indices[start:stop]

        purge_start = start
        purge_stop = min(stop + embargo, n_samples)
        train_mask = (indices < purge_start) | (indices >= purge_stop)
        train_idx = indices[train_mask]

        yield train_idx, test_idx
        current = stop
