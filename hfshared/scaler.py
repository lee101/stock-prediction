#!/usr/bin/env python3
"""Scaler I/O helpers for loading training processors at inference."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def load_processor(path: str) -> Tuple[Optional[Any], List[str], Optional[int], Optional[int]]:
    """Load a training processor dump created by StockDataProcessor.save_scalers.

    Returns (standard_scaler, feature_names, sequence_length, prediction_horizon)
    where any of the first/last elements may be None if unavailable.
    """
    try:
        import joblib  # local import to avoid hard dependency
    except Exception:
        return None, [], None, None

    try:
        data: Dict[str, Any] = joblib.load(path)
        scalers = data.get('scalers', {})
        std = scalers.get('standard') if isinstance(scalers, dict) else None
        feats = list(data.get('feature_names', []) or [])
        seq_len = data.get('sequence_length')
        horizon = data.get('prediction_horizon')
        return std, feats, seq_len, horizon
    except Exception:
        return None, [], None, None

