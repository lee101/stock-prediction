"""
Prediction cache for walk-forward backtesting.

Caches Toto/Kronos predictions to avoid redundant computation.
In walk-forward validation, earlier days appear in multiple simulations.

Example with 70 simulations:
- Day 0-130: Predicted 70 times (appears in all simulations)
- Day 150: Predicted 50 times
- Day 190: Predicted 10 times

With caching, each day is predicted once, then reused.
Potential speedup: ~3x total, ~58x on model inference alone.
"""

import hashlib
from typing import Optional, Tuple, Any
import os


class PredictionCache:
    """
    In-memory cache for model predictions during backtest.

    Cache key: (data_hash, key_to_predict, day_index)
    Value: (predictions, band, abs_predictions)
    """

    def __init__(self, enabled: bool = None):
        if enabled is None:
            enabled = os.getenv("MARKETSIM_CACHE_PREDICTIONS", "1") in {"1", "true", "yes", "on"}

        self.enabled = enabled
        self._cache = {}
        self._hits = 0
        self._misses = 0

    def _make_key(self, data_hash: str, key_to_predict: str, day_index: int) -> Tuple:
        """Create cache key from data, prediction target, and day index"""
        return (data_hash, key_to_predict, day_index)

    def get(self, data_hash: str, key_to_predict: str, day_index: int) -> Optional[Any]:
        """Get cached prediction if available"""
        if not self.enabled:
            return None

        key = self._make_key(data_hash, key_to_predict, day_index)
        result = self._cache.get(key)

        if result is not None:
            self._hits += 1
        else:
            self._misses += 1

        return result

    def put(self, data_hash: str, key_to_predict: str, day_index: int, value: Any):
        """Store prediction in cache"""
        if not self.enabled:
            return

        key = self._make_key(data_hash, key_to_predict, day_index)
        self._cache[key] = value

    def clear(self):
        """Clear cache and stats"""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> dict:
        """Return cache statistics"""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0

        return {
            'enabled': self.enabled,
            'hits': self._hits,
            'misses': self._misses,
            'total_requests': total,
            'hit_rate': hit_rate,
            'cache_size': len(self._cache),
        }

    def __len__(self):
        return len(self._cache)


# Global cache instance for backtest runs
_global_cache = None


def get_cache() -> PredictionCache:
    """Get or create global cache instance"""
    global _global_cache
    if _global_cache is None:
        _global_cache = PredictionCache()
    return _global_cache


def reset_cache():
    """Reset global cache (call at start of backtest)"""
    global _global_cache
    if _global_cache is not None:
        _global_cache.clear()
    else:
        _global_cache = PredictionCache()


def hash_dataframe(df) -> str:
    """
    Create hash of dataframe for cache key.
    Uses shape and a sample of values for speed.
    """
    # Use shape + first/last few rows for fast hash
    parts = [
        str(df.shape),
        str(df.columns.tolist()),
    ]

    # Sample first and last rows
    if len(df) > 0:
        parts.append(str(df.iloc[0].values.tobytes()) if hasattr(df.iloc[0].values, 'tobytes') else str(df.iloc[0].values))
    if len(df) > 1:
        parts.append(str(df.iloc[-1].values.tobytes()) if hasattr(df.iloc[-1].values, 'tobytes') else str(df.iloc[-1].values))

    # Middle sample
    if len(df) > 10:
        mid = len(df) // 2
        parts.append(str(df.iloc[mid].values.tobytes()) if hasattr(df.iloc[mid].values, 'tobytes') else str(df.iloc[mid].values))

    combined = "|".join(parts)
    return hashlib.md5(combined.encode()).hexdigest()[:16]


# Example usage:
#
# At start of backtest_forecasts:
#   reset_cache()
#
# In run_single_simulation, before model prediction:
#   cache = get_cache()
#   cache_key = hash_dataframe(simulation_data)
#   cached = cache.get(cache_key, key_to_predict, len(simulation_data))
#   if cached is not None:
#       toto_predictions, toto_band, toto_abs = cached
#   else:
#       # Compute prediction
#       toto_predictions, toto_band, toto_abs = _compute_toto_forecast(...)
#       cache.put(cache_key, key_to_predict, len(simulation_data), (toto_predictions, toto_band, toto_abs))
#
# At end of backtest_forecasts:
#   stats = cache.stats()
#   logger.info(f"Cache stats: {stats['hit_rate']:.1f}% hit rate, {stats['hits']} hits, {stats['misses']} misses")
