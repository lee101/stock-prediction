"""Lightweight Toto feature shims used in tests and local workflows.

The full Toto integration is optional in this repository. Several data-loading
paths only need the interfaces to exist so they can disable Toto features or
fall back to precomputed predictions. This module provides that minimal surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(slots=True)
class TotoOptions:
    use_toto: bool = False
    horizon: int = 8
    context_length: int = 60
    num_samples: int = 2048
    toto_model_id: str = "Datadog/Toto-Open-Base-1.0"
    toto_device: str = "cpu"


class TotoFeatureGenerator:
    """No-op fallback feature generator.

    The real Toto pipeline is expensive and optional. For tests that do not
    exercise Toto itself, returning an empty feature matrix is sufficient.
    """

    def __init__(self, options: TotoOptions | None = None) -> None:
        self.options = options or TotoOptions()

    def compute_features(
        self,
        price_matrix: np.ndarray,
        _price_columns: Sequence[str],
        *,
        symbol_prefix: str = "toto",
    ) -> tuple[np.ndarray, list[str]]:
        rows = int(len(price_matrix))
        return np.zeros((rows, 0), dtype=np.float32), []


def append_toto_columns(
    frame: pd.DataFrame,
    features: np.ndarray,
    *,
    column_names: Sequence[str],
) -> None:
    if features.size == 0 or not column_names:
        return
    for idx, name in enumerate(column_names):
        frame[str(name)] = features[:, idx]
