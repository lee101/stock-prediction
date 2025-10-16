#!/usr/bin/env python3
"""Scaler I/O helpers for loading training processors at inference."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def load_processor(path: str) -> Dict[str, Any]:
    """Load a training processor dump created by StockDataProcessor.save_scalers.

    Returns a dict with keys:
        - scalers: mapping containing the fitted scalers (may be empty)
        - feature_names: list of feature column names used during training
        - sequence_length: model sequence length if stored
        - prediction_horizon: model prediction horizon if stored

    Falls back to an empty structure if the payload cannot be loaded.
    """
    try:
        import joblib  # local import to avoid hard dependency
    except Exception:
        return {"scalers": {}, "feature_names": [], "sequence_length": None, "prediction_horizon": None}

    try:
        data: Dict[str, Any] = joblib.load(path)
        scalers_obj = data.get('scalers', {})
        feature_names = list(data.get('feature_names', []) or [])
        return {
            "scalers": scalers_obj if isinstance(scalers_obj, dict) else {},
            "feature_names": feature_names,
            "sequence_length": data.get('sequence_length'),
            "prediction_horizon": data.get('prediction_horizon'),
        }
    except Exception:
        return {"scalers": {}, "feature_names": [], "sequence_length": None, "prediction_horizon": None}
