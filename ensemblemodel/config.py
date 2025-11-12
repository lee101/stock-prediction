"""Configuration helpers for ensemblemodel components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(slots=True)
class BackendSettings:
    """User-facing configuration for an ensemble backend."""

    name: str
    weight: float = 1.0
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AggregationSettings:
    """Settings that control the shared aggregation policy."""

    method: str = "trimmed_mean_10"
    clip_percentile: float = 0.10
    clip_std: float = 2.5
    weight_resolution: int = 8

