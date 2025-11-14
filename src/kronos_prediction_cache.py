"""Disk-based caching for Kronos predictions with comprehensive cache key generation."""

import functools
import hashlib
import json
import os
import pickle
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import torch


class KronosPredictionCache:
    """
    Disk-based cache for Kronos time series predictions.

    Cache keys include:
    - Input data hash (last N rows for context)
    - All prediction parameters (pred_len, lookback, temperature, top_p, top_k, sample_count)
    - Model settings from environment (KRONOS_DTYPE, compile flags, etc.)
    - Symbol and column being predicted

    This ensures predictions are reused when:
    1. Same symbol/column predicted with same recent data
    2. Same model parameters
    3. Within TTL window (default: 5 minutes for trading)
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        ttl_seconds: float = 300,  # 5 minutes default
        enabled: bool = True,
        max_cache_size_mb: int = 1000,  # 1GB default
        value_precision: Optional[int] = None,
    ):
        if cache_dir is None:
            cache_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                ".cache",
                "kronos_predictions",
            )

        self.cache_dir = Path(cache_dir)
        self.ttl_seconds = ttl_seconds
        self.enabled = enabled and os.environ.get("TESTING") != "True"
        self.max_cache_size_mb = max_cache_size_mb
        if value_precision is None:
            env_precision = os.environ.get("KRONOS_CACHE_DECIMALS")
            value_precision = int(env_precision) if env_precision not in (None, "") else 8
        self._value_precision = max(0, int(value_precision))

        # Stats tracking
        self.hits = 0
        self.misses = 0
        self.saves = 0

        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_data_hash(
        self, data: pd.DataFrame, columns: list, lookback: Optional[int] = None
    ) -> str:
        """
        Generate hash of input data.

        Only uses the relevant columns and last N rows (lookback window).
        This allows cache hits when new data is added but historical context unchanged.
        """
        # Use only relevant columns
        relevant_cols = ["timestamp"] + [
            col for col in columns if col in data.columns
        ]
        subset = data[relevant_cols].copy()

        # Use only last lookback rows (or all if lookback not specified)
        if lookback and len(subset) > lookback:
            subset = subset.tail(lookback)

        # Convert to bytes for hashing
        # Use a stable serialization: sort columns, convert timestamp to string
        subset = subset.sort_index(axis=1)  # Sort columns alphabetically
        if "timestamp" in subset.columns:
            subset["timestamp"] = subset["timestamp"].astype(str)

        float_cols = subset.select_dtypes(include=["float16", "float32", "float64"])
        if not float_cols.empty and self._value_precision is not None:
            subset[float_cols.columns] = float_cols.round(self._value_precision)

        # Create hash from data values
        data_bytes = subset.to_csv(index=False).encode("utf-8")
        return hashlib.sha256(data_bytes).hexdigest()[:16]

    def _get_env_settings_hash(self) -> str:
        """Hash of environment settings that affect model behavior."""
        env_settings = {
            "KRONOS_DTYPE": os.environ.get("KRONOS_DTYPE", ""),
            "KRONOS_COMPILE": os.environ.get("KRONOS_COMPILE", ""),
            "KRONOS_COMPILE_MODE": os.environ.get("KRONOS_COMPILE_MODE", ""),
            "KRONOS_COMPILE_BACKEND": os.environ.get("KRONOS_COMPILE_BACKEND", ""),
            "TOTO_DTYPE": os.environ.get("TOTO_DTYPE", ""),  # May affect mixed precision
        }
        settings_str = json.dumps(env_settings, sort_keys=True)
        return hashlib.md5(settings_str.encode()).hexdigest()[:8]

    def _build_cache_key(
        self,
        symbol: str,
        column: str,
        data: pd.DataFrame,
        pred_len: int,
        lookback: Optional[int],
        temperature: Optional[float],
        top_p: Optional[float],
        top_k: Optional[int],
        sample_count: Optional[int],
    ) -> str:
        """
        Build comprehensive cache key from all parameters that affect predictions.

        Format: {symbol}_{column}_{data_hash}_{params_hash}_{env_hash}
        """
        # Data hash (includes lookback context)
        data_hash = self._get_data_hash(data, [column], lookback)

        # Parameters that affect prediction
        params = {
            "pred_len": pred_len,
            "lookback": lookback,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "sample_count": sample_count,
        }
        params_str = json.dumps(params, sort_keys=True)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]

        # Environment settings
        env_hash = self._get_env_settings_hash()

        # Combine into cache key
        return f"{symbol}_{column}_{data_hash}_{params_hash}_{env_hash}"

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get path to cache file for given key."""
        return self.cache_dir / f"{cache_key}.pkl"

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cached file exists and is within TTL."""
        if not cache_path.exists():
            return False

        # Check TTL
        if self.ttl_seconds <= 0:
            return True  # No TTL expiration

        file_age = time.time() - cache_path.stat().st_mtime
        return file_age < self.ttl_seconds

    def get(
        self,
        symbol: str,
        column: str,
        data: pd.DataFrame,
        pred_len: int,
        lookback: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        sample_count: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached prediction if available and valid.

        Returns None if cache miss or expired.
        """
        if not self.enabled:
            return None

        cache_key = self._build_cache_key(
            symbol, column, data, pred_len, lookback, temperature, top_p, top_k, sample_count
        )
        cache_path = self._get_cache_path(cache_key)

        if not self._is_cache_valid(cache_path):
            self.misses += 1
            return None

        try:
            with open(cache_path, "rb") as f:
                cached_data = pickle.load(f)

            self.hits += 1
            return cached_data

        except Exception as e:
            # Cache corruption or unpickling error - treat as miss
            self.misses += 1
            if cache_path.exists():
                cache_path.unlink()  # Remove corrupted cache
            return None

    def set(
        self,
        symbol: str,
        column: str,
        data: pd.DataFrame,
        pred_len: int,
        result: Dict[str, Any],
        lookback: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        sample_count: Optional[int] = None,
    ):
        """Save prediction result to cache."""
        if not self.enabled:
            return

        cache_key = self._build_cache_key(
            symbol, column, data, pred_len, lookback, temperature, top_p, top_k, sample_count
        )
        cache_path = self._get_cache_path(cache_key)

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

            self.saves += 1

            # Cleanup old cache if size exceeds limit
            self._cleanup_if_needed()

        except Exception:
            # Fail silently - caching is optional optimization
            pass

    def _cleanup_if_needed(self):
        """Remove old cache entries if total size exceeds max_cache_size_mb."""
        try:
            # Calculate total cache size
            total_size_bytes = sum(
                f.stat().st_size for f in self.cache_dir.glob("*.pkl")
            )
            max_size_bytes = self.max_cache_size_mb * 1024 * 1024

            if total_size_bytes <= max_size_bytes:
                return

            # Sort files by modification time (oldest first)
            cache_files = sorted(
                self.cache_dir.glob("*.pkl"),
                key=lambda p: p.stat().st_mtime,
            )

            # Remove oldest files until under limit
            for cache_file in cache_files:
                if total_size_bytes <= max_size_bytes:
                    break

                file_size = cache_file.stat().st_size
                cache_file.unlink()
                total_size_bytes -= file_size

        except Exception:
            # Cleanup errors shouldn't break predictions
            pass

    def clear(self):
        """Clear all cached predictions."""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            time.sleep(0.1)
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Reset stats
        self.hits = 0
        self.misses = 0
        self.saves = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0.0

        cache_size_bytes = 0
        num_entries = 0
        if self.cache_dir.exists():
            cache_files = list(self.cache_dir.glob("*.pkl"))
            num_entries = len(cache_files)
            cache_size_bytes = sum(f.stat().st_size for f in cache_files)

        return {
            "enabled": self.enabled,
            "hits": self.hits,
            "misses": self.misses,
            "saves": self.saves,
            "hit_rate_percent": hit_rate,
            "num_entries": num_entries,
            "cache_size_mb": cache_size_bytes / (1024 * 1024),
            "ttl_seconds": self.ttl_seconds,
        }


# Global cache instance
_global_cache: Optional[KronosPredictionCache] = None


def get_prediction_cache() -> KronosPredictionCache:
    """Get or create global prediction cache instance."""
    global _global_cache

    if _global_cache is None:
        # Read configuration from environment
        ttl_seconds = float(os.environ.get("KRONOS_PREDICTION_CACHE_TTL", "300"))
        enabled = os.environ.get("KRONOS_PREDICTION_CACHE_ENABLED", "1") == "1"
        max_size_mb = int(os.environ.get("KRONOS_PREDICTION_CACHE_MAX_SIZE_MB", "1000"))
        precision = os.environ.get("KRONOS_CACHE_DECIMALS")
        precision_int = int(precision) if precision not in (None, "") else 8

        _global_cache = KronosPredictionCache(
            ttl_seconds=ttl_seconds,
            enabled=enabled,
            max_cache_size_mb=max_size_mb,
            value_precision=precision_int,
        )

    return _global_cache


def clear_prediction_cache():
    """Clear global prediction cache."""
    global _global_cache
    if _global_cache is not None:
        _global_cache.clear()
        _global_cache = None
