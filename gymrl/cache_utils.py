"""
Utilities for serialising and loading GymRL feature cubes on disk.

Feature caches make Toto/Chronos preprocessing a one-off cost; subsequent
training or evaluation scripts can skip the expensive sampling step.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .feature_pipeline import FeatureCube


def _meta_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".meta.json")


def save_feature_cache(
    cache_path: Path,
    cube: FeatureCube,
    *,
    extra_metadata: Optional[Dict[str, object]] = None,
) -> None:
    """
    Persist a feature cube to ``cache_path`` as a compressed NPZ plus metadata.
    """

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        cache_path,
        features=cube.features.astype(np.float32),
        realized_returns=cube.realized_returns.astype(np.float32),
        forecast_cvar=(cube.forecast_cvar.astype(np.float32) if cube.forecast_cvar is not None else np.empty((0,), dtype=np.float32)),
        forecast_uncertainty=(cube.forecast_uncertainty.astype(np.float32) if cube.forecast_uncertainty is not None else np.empty((0,), dtype=np.float32)),
    )

    meta: Dict[str, object] = {
        "feature_names": cube.feature_names,
        "symbols": cube.symbols,
        "timestamps": [ts.isoformat() for ts in cube.timestamps],
    }
    if extra_metadata:
        meta["extra_metadata"] = extra_metadata

    meta_path = _meta_path(cache_path)
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)


def load_feature_cache(cache_path: Path) -> Tuple[FeatureCube, Dict[str, object]]:
    """
    Load a feature cube previously stored with ``save_feature_cache``.
    """

    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(f"Feature cache not found: {cache_path}")

    archive = np.load(cache_path, allow_pickle=False)
    features = archive["features"]
    realized = archive["realized_returns"]

    forecast_cvar = archive.get("forecast_cvar")
    if forecast_cvar is not None and forecast_cvar.size == 0:
        forecast_cvar = None

    forecast_uncertainty = archive.get("forecast_uncertainty")
    if forecast_uncertainty is not None and forecast_uncertainty.size == 0:
        forecast_uncertainty = None

    meta_path = _meta_path(cache_path)
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata for feature cache: {meta_path}")
    with meta_path.open("r", encoding="utf-8") as fh:
        meta = json.load(fh)

    feature_names = list(meta["feature_names"])
    symbols = list(meta["symbols"])
    timestamps = [pd.Timestamp(ts) for ts in meta["timestamps"]]

    cube = FeatureCube(
        features=features,
        realized_returns=realized,
        feature_names=feature_names,
        symbols=symbols,
        timestamps=timestamps,
        forecast_cvar=forecast_cvar,
        forecast_uncertainty=forecast_uncertainty,
    )

    extra_metadata = meta.get("extra_metadata", {})
    return cube, extra_metadata  # type: ignore[return-value]


__all__ = ["save_feature_cache", "load_feature_cache"]

