"""Multi-scale Chronos forecasting configuration.

This module provides per-symbol multi-scale configuration for Chronos2 forecasting.
Multi-scale forecasting runs predictions at different time granularities (skip rates)
and aggregates them, which can improve accuracy for certain symbols.

Based on experiments showing:
- ~50% of symbols benefit from multi-scale
- Average MAE improvement of 8% when multi-scale helps
- Low volatility stocks often benefit more
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MultiscaleChoice:
    """Multi-scale configuration for a symbol."""

    symbol: str
    method: str  # 'single', 'trimmed', 'median', 'weighted'
    skip_rates: Tuple[int, ...]
    mae_improvement: float
    volatility: float
    source_path: Path

    @property
    def use_multiscale(self) -> bool:
        """Whether to use multi-scale (vs single-scale baseline)."""
        return self.method != "single" and len(self.skip_rates) > 1


class MultiscaleSelector:
    """Load and select per-symbol multi-scale configurations."""

    DEFAULT_CONFIG_PATHS = (
        Path("preaugstrategies/multiscale/config.json"),
        Path("reports/multiscale_config.json"),
    )

    def __init__(
        self,
        config_paths: Optional[Sequence[str | Path]] = None,
        default_method: str = "single",
        default_skip_rates: Tuple[int, ...] = (1,),
    ) -> None:
        """Initialize multi-scale selector.

        Args:
            config_paths: Paths to check for multi-scale config JSON files.
            default_method: Default method when no config found.
            default_skip_rates: Default skip rates when no config found.
        """
        self._config_paths = tuple(
            Path(p) for p in (config_paths or self.DEFAULT_CONFIG_PATHS)
        )
        self._default_method = default_method
        self._default_skip_rates = default_skip_rates
        self._cache: Dict[str, Optional[MultiscaleChoice]] = {}
        self._config: Optional[Dict[str, Dict[str, Any]]] = None
        self._config_path: Optional[Path] = None

    def _load_config(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """Load config from first available path."""
        if self._config is not None:
            return self._config

        for path in self._config_paths:
            if path.exists():
                try:
                    data = json.loads(path.read_text())
                    self._config = data.get("symbol_configs", data)
                    self._config_path = path
                    logger.info("Loaded multi-scale config from %s", path)
                    return self._config
                except Exception as e:
                    logger.warning("Failed to load multi-scale config from %s: %s", path, e)

        return None

    def get_choice(self, symbol: str) -> Optional[MultiscaleChoice]:
        """Get multi-scale configuration for a symbol."""
        symbol_key = symbol.upper()
        if symbol_key in self._cache:
            return self._cache[symbol_key]

        config = self._load_config()
        if config is None:
            self._cache[symbol_key] = None
            return None

        symbol_config = config.get(symbol_key)
        if symbol_config is None:
            self._cache[symbol_key] = None
            return None

        try:
            choice = MultiscaleChoice(
                symbol=symbol_key,
                method=symbol_config.get("method", self._default_method),
                skip_rates=tuple(symbol_config.get("skip_rates", self._default_skip_rates)),
                mae_improvement=float(symbol_config.get("mae_improvement", 0.0)),
                volatility=float(symbol_config.get("volatility", 0.0)),
                source_path=self._config_path or Path("."),
            )
            self._cache[symbol_key] = choice
            return choice
        except Exception as e:
            logger.warning("Failed to parse multi-scale config for %s: %s", symbol_key, e)
            self._cache[symbol_key] = None
            return None

    def get_or_default(self, symbol: str) -> MultiscaleChoice:
        """Get multi-scale config or return default."""
        choice = self.get_choice(symbol)
        if choice is not None:
            return choice
        return MultiscaleChoice(
            symbol=symbol.upper(),
            method=self._default_method,
            skip_rates=self._default_skip_rates,
            mae_improvement=0.0,
            volatility=0.0,
            source_path=Path("."),
        )


def aggregate_forecasts(
    forecasts: Dict[int, Dict[float, Any]],
    method: str,
    trim_pct: float = 0.1,
    base_weight: float = 2.0,
) -> Dict[float, Any]:
    """Aggregate multi-scale forecasts using specified method.

    Args:
        forecasts: Dict mapping skip_rate -> quantile_frames
        method: Aggregation method ('trimmed', 'median', 'weighted')
        trim_pct: Trim percentage for trimmed mean
        base_weight: Base weight for weighted aggregation

    Returns:
        Aggregated quantile frames
    """
    if not forecasts:
        raise ValueError("No forecasts to aggregate")

    if method == "single" or len(forecasts) == 1:
        # Just return the first (or only) forecast
        return next(iter(forecasts.values()))

    from scipy import stats

    # Get all quantile levels present
    all_quantiles = set()
    for qf in forecasts.values():
        all_quantiles.update(qf.keys())

    result = {}
    for q_level in all_quantiles:
        frames = {sr: qf[q_level] for sr, qf in forecasts.items() if q_level in qf}
        if not frames:
            continue

        ref_df = next(iter(frames.values()))
        agg_df = ref_df.copy()

        for col in ref_df.columns:
            values = []
            weights = []
            for skip_rate, df in frames.items():
                if col in df.columns and len(df) > 0:
                    values.append(df[col].iloc[-1])
                    weights.append(base_weight / skip_rate)

            if not values:
                continue

            if method == "trimmed":
                agg_value = stats.trim_mean(values, proportiontocut=trim_pct)
            elif method == "median":
                agg_value = np.median(values)
            elif method == "weighted":
                weights = np.array(weights)
                weights = weights / weights.sum()
                agg_value = np.average(values, weights=weights)
            else:
                agg_value = values[0]  # fallback to first

            # Cast to match dtype
            agg_df.loc[agg_df.index[-1], col] = agg_df[col].dtype.type(agg_value)

        result[q_level] = agg_df

    return result


__all__ = ["MultiscaleChoice", "MultiscaleSelector", "aggregate_forecasts"]
