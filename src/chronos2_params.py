from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from hyperparamstore import HyperparamStore, load_best_config

logger = logging.getLogger(__name__)

DEFAULT_CHRONOS_PREDICTION_LENGTH = 7
_chronos2_params_cache: Dict[Tuple[str, str], Dict[str, object]] = {}
_multivariate_config_cache: Optional[Dict[str, Dict[str, Any]]] = None

# Environment variable to enable multivariate OHLC forecasting
# When enabled, predicts OHLC together (80% MAE improvement for stocks)
CHRONOS2_USE_MULTIVARIATE = os.getenv("CHRONOS2_USE_MULTIVARIATE", "1").strip().lower() in {"1", "true", "yes", "on"}


def _normalize_frequency(value: Optional[str]) -> str:
    if not value:
        return "daily"
    normalized = value.strip().lower()
    if normalized not in {"daily", "hourly"}:
        return "daily"
    return normalized


def _load_multivariate_config() -> Dict[str, Dict[str, Any]]:
    """Load per-symbol multivariate configuration from tuning output."""
    global _multivariate_config_cache
    if _multivariate_config_cache is not None:
        return _multivariate_config_cache

    _multivariate_config_cache = {}
    config_paths = [
        Path("reports/multivariate_config.json"),
        Path("reports/multivariate_config_stocks.json"),
        Path("reports/multivariate_config_crypto.json"),
    ]

    for path in config_paths:
        if path.exists():
            try:
                data = json.loads(path.read_text())
                symbol_configs = data.get("symbol_configs", {})
                _multivariate_config_cache.update(symbol_configs)
                logger.debug("Loaded multivariate config from %s (%d symbols)", path, len(symbol_configs))
            except Exception as e:
                logger.warning("Failed to load multivariate config from %s: %s", path, e)

    return _multivariate_config_cache


def _get_symbol_multivariate_setting(symbol: str, default: bool) -> bool:
    """Get per-symbol multivariate setting, falling back to default."""
    config = _load_multivariate_config()
    symbol_cfg = config.get(symbol.upper())
    if symbol_cfg is not None:
        return bool(symbol_cfg.get("use_multivariate", default))
    return default


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

    # Check if symbol is crypto to determine multivariate default
    crypto_suffixes = ("USD", "BTC", "ETH", "USDT", "USDC")
    is_crypto = any(symbol.upper().endswith(suf) for suf in crypto_suffixes)

    # Multivariate helps stocks (~80% MAE improvement) but not crypto
    # Check per-symbol tuned config first, then fall back to global setting
    global_default = CHRONOS2_USE_MULTIVARIATE and not is_crypto
    default_multivariate = _get_symbol_multivariate_setting(symbol, global_default)

    env_cross_learning = os.getenv("CHRONOS2_CROSS_LEARNING")
    if record is not None:
        # Extract skip_rates - new multiscale parameter from hourly tuning
        raw_skip_rates = config.get("skip_rates", [1])
        try:
            skip_rates = tuple(int(r) for r in raw_skip_rates)
        except (TypeError, ValueError):
            skip_rates = (1,)

        # Determine if multiscale should be used based on skip_rates
        has_multiscale = len(skip_rates) > 1
        aggregation_method = str(config.get("aggregation_method", "single"))

        use_cross_learning = bool(config.get("use_cross_learning", False))
        if env_cross_learning is not None:
            use_cross_learning = env_cross_learning.strip().lower() in {"1", "true", "yes", "on"}

        predict_kwargs = dict(config.get("predict_kwargs") or {})
        if "cross_learning" in predict_kwargs and "predict_batches_jointly" not in predict_kwargs:
            predict_kwargs["predict_batches_jointly"] = bool(predict_kwargs.pop("cross_learning"))
        else:
            predict_kwargs.pop("cross_learning", None)
        if use_cross_learning:
            predict_kwargs.setdefault("predict_batches_jointly", True)

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
            "predict_kwargs": predict_kwargs,
            "_config_path": str(config_path) if config_path.exists() else None,
            "_config_name": str(config.get("name") or ""),
            # Multivariate forecasting parameters
            "use_multivariate": bool(config.get("use_multivariate", default_multivariate)),
            "use_cross_learning": use_cross_learning,
            # Multiscale skip-rate parameters (from hourly tuning)
            "skip_rates": skip_rates,
            "aggregation_method": aggregation_method,
            "use_multiscale": has_multiscale or bool(config.get("use_multiscale", False)),
            "multiscale_method": aggregation_method if has_multiscale else str(config.get("multiscale_method", "single")),
        }
    else:
        use_cross_learning = False
        if env_cross_learning is not None:
            use_cross_learning = env_cross_learning.strip().lower() in {"1", "true", "yes", "on"}
        predict_kwargs: Dict[str, object] = {}
        if use_cross_learning:
            predict_kwargs["predict_batches_jointly"] = True
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
            "predict_kwargs": predict_kwargs,
            "_config_path": str(config_path) if config_path.exists() else None,
            "_config_name": "",
            # Multivariate forecasting parameters
            "use_multivariate": default_multivariate,
            "use_cross_learning": use_cross_learning,
            # Multiscale skip-rate parameters (defaults)
            "skip_rates": (1,),
            "aggregation_method": "single",
            "use_multiscale": False,
            "multiscale_method": "single",
        }
        logger.info("No stored Chronos2 hyperparameters for %s (frequency=%s); using defaults.", symbol, freq)

    _chronos2_params_cache[cache_key] = dict(params)
    return dict(params)


__all__ = ["DEFAULT_CHRONOS_PREDICTION_LENGTH", "resolve_chronos2_params"]
