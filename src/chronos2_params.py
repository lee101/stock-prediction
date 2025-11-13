from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

from hyperparamstore import HyperparamStore, load_best_config

logger = logging.getLogger(__name__)

DEFAULT_CHRONOS_PREDICTION_LENGTH = 7
_chronos2_params_cache: Dict[Tuple[str, str], Dict[str, object]] = {}


def _normalize_frequency(value: Optional[str]) -> str:
    if not value:
        return "daily"
    normalized = value.strip().lower()
    if normalized not in {"daily", "hourly"}:
        return "daily"
    return normalized


def resolve_chronos2_params(
    symbol: str,
    *,
    frequency: Optional[str] = None,
    default_prediction_length: int = DEFAULT_CHRONOS_PREDICTION_LENGTH,
) -> Dict[str, object]:
    """
    Resolve Chronos2 hyperparameters for the requested symbol/frequency.

    Args:
        symbol: Trading symbol (e.g., AAPL, BTCUSD)
        frequency: Optional cadence override (\"daily\" or \"hourly\")
        default_prediction_length: Fallback horizon when config omits it
    """

    freq = _normalize_frequency(frequency or os.getenv("CHRONOS2_FREQUENCY"))
    cache_key = (symbol.upper(), freq)
    cached = _chronos2_params_cache.get(cache_key)
    if cached is not None:
        return dict(cached)

    hyperparam_root = Path(os.getenv("HYPERPARAM_ROOT", "hyperparams"))
    base_model = "chronos2"
    record = load_best_config(base_model, symbol)
    config = record.config if record else {}
    config_path = hyperparam_root / base_model / f"{symbol.upper()}.json"

    if freq != "daily":
        variant_store = HyperparamStore(root=hyperparam_root / base_model)
        variant_record = load_best_config(freq, symbol, store=variant_store)
        if variant_record is not None:
            record = variant_record
            config = variant_record.config
            config_path = hyperparam_root / base_model / freq / f"{symbol.upper()}.json"
            logger.info("Loaded Chronos2 hyperparameters for %s (frequency=%s).", symbol, freq)
        else:
            logger.info(
                "Chronos2 %s hyperparameters for %s unavailable; falling back to daily config.",
                freq,
                symbol,
            )
    elif record is not None:
        logger.info("Loaded Chronos2 hyperparameters for %s (frequency=daily).", symbol)

    quantile_levels = config.get("quantile_levels", [0.1, 0.5, 0.9])
    try:
        quantile_tuple = tuple(float(level) for level in quantile_levels)
    except (TypeError, ValueError):
        quantile_tuple = (0.1, 0.5, 0.9)

    if record is not None:
        params = {
            "model_id": config.get("model_id", "amazon/chronos-2"),
            "device_map": config.get("device_map", "cuda"),
            "context_length": int(config.get("context_length", 512)),
            "prediction_length": max(2, int(config.get("prediction_length", default_prediction_length))),
            "quantile_levels": quantile_tuple,
            "batch_size": int(config.get("batch_size", 128)),
            "aggregation": str(config.get("aggregation", "median")),
            "sample_count": int(config.get("sample_count", 0)),
            "scaler": str(config.get("scaler", "none")),
            "predict_kwargs": dict(config.get("predict_kwargs") or {}),
            "_config_path": str(config_path) if config_path.exists() else None,
            "_config_name": str(config.get("name") or ""),
        }
    else:
        params = {
            "model_id": "amazon/chronos-2",
            "device_map": "cuda",
            "context_length": 512,
            "prediction_length": default_prediction_length,
            "quantile_levels": (0.1, 0.5, 0.9),
            "batch_size": 128,
            "aggregation": "median",
            "sample_count": 0,
            "scaler": "none",
            "predict_kwargs": {},
            "_config_path": str(config_path) if config_path.exists() else None,
            "_config_name": "",
        }
        logger.info("No stored Chronos2 hyperparameters for %s (frequency=%s); using defaults.", symbol, freq)

    _chronos2_params_cache[cache_key] = dict(params)
    return dict(params)


__all__ = ["DEFAULT_CHRONOS_PREDICTION_LENGTH", "resolve_chronos2_params"]
