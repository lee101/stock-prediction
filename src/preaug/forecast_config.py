"""Forecasting configuration system for Chronos2.

This module provides a tag-based configuration system that selects optimal
forecasting settings based on what's being predicted:
- Symbol type (stock vs crypto)
- Target columns (OHLC together vs close only)
- Multi-symbol batching (joint vs independent)

Based on experiments showing:
- Multi-target OHLC: ~80% MAE improvement for stocks
- Cross-learning (predict_batches_jointly): +9-11% for crypto, -20% for diverse stocks
- Multi-scale: +10% average when applicable
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.symbol_utils import is_crypto_symbol

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ForecastTag:
    """Tag identifying a forecasting configuration."""

    symbols: Tuple[str, ...]  # e.g., ("AAPL",) or ("BTCUSD", "ETHUSD")
    targets: Tuple[str, ...]  # e.g., ("open", "high", "low", "close") or ("close",)
    asset_type: str  # "stock", "crypto", "mixed"

    @classmethod
    def from_symbols_and_targets(
        cls,
        symbols: Sequence[str],
        targets: Sequence[str] = ("open", "high", "low", "close"),
    ) -> "ForecastTag":
        """Create a tag from symbols and target columns."""
        symbol_list = [s.upper() for s in symbols]

        # Use robust crypto detection to avoid false positives like MU/LULU/BIDU (stock tickers ending in "U").
        n_crypto = sum(1 for s in symbol_list if is_crypto_symbol(s))
        n_stock = len(symbol_list) - n_crypto

        if n_crypto == 0:
            asset_type = "stock"
        elif n_stock == 0:
            asset_type = "crypto"
        else:
            asset_type = "mixed"

        return cls(
            symbols=tuple(sorted(symbol_list)),
            targets=tuple(targets),
            asset_type=asset_type,
        )

    @property
    def is_multi_target(self) -> bool:
        """Whether this uses multiple target columns (OHLC together)."""
        return len(self.targets) > 1

    @property
    def is_multi_symbol(self) -> bool:
        """Whether this includes multiple symbols."""
        return len(self.symbols) > 1

    @property
    def tag_key(self) -> str:
        """Unique key for this tag configuration."""
        return f"{self.asset_type}:{','.join(self.symbols)}:{','.join(self.targets)}"


@dataclass
class ForecastConfig:
    """Optimal forecasting configuration for a specific tag."""

    tag: ForecastTag
    use_multivariate: bool = True  # Predict OHLC together as multivariate
    use_cross_learning: bool = False  # predict_batches_jointly
    use_multiscale: bool = False  # Multi-scale aggregation
    multiscale_method: str = "single"  # 'single', 'trimmed', 'median', 'weighted'
    multiscale_skip_rates: Tuple[int, ...] = (1,)
    batch_size: int = 100
    context_length: int = 512
    mae_improvement: float = 0.0  # Expected improvement over baseline
    source: str = "default"  # Where this config came from

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbols": list(self.tag.symbols),
            "targets": list(self.tag.targets),
            "asset_type": self.tag.asset_type,
            "use_multivariate": self.use_multivariate,
            "use_cross_learning": self.use_cross_learning,
            "use_multiscale": self.use_multiscale,
            "multiscale_method": self.multiscale_method,
            "multiscale_skip_rates": list(self.multiscale_skip_rates),
            "batch_size": self.batch_size,
            "context_length": self.context_length,
            "mae_improvement": self.mae_improvement,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ForecastConfig":
        """Create from dictionary."""
        tag = ForecastTag(
            symbols=tuple(data.get("symbols", [])),
            targets=tuple(data.get("targets", ["open", "high", "low", "close"])),
            asset_type=data.get("asset_type", "stock"),
        )
        return cls(
            tag=tag,
            use_multivariate=data.get("use_multivariate", True),
            use_cross_learning=data.get("use_cross_learning", False),
            use_multiscale=data.get("use_multiscale", False),
            multiscale_method=data.get("multiscale_method", "single"),
            multiscale_skip_rates=tuple(data.get("multiscale_skip_rates", [1])),
            batch_size=data.get("batch_size", 100),
            context_length=data.get("context_length", 512),
            mae_improvement=data.get("mae_improvement", 0.0),
            source=data.get("source", "loaded"),
        )


class ForecastConfigSelector:
    """Select optimal forecasting configuration based on input tag."""

    DEFAULT_CONFIG_PATHS = (
        Path("preaugstrategies/forecast/config.json"),
        Path("reports/forecast_config.json"),
    )

    # Default configurations based on experiment findings
    DEFAULT_CONFIGS = {
        "stock_single_ohlc": ForecastConfig(
            tag=ForecastTag(symbols=("*",), targets=("open", "high", "low", "close"), asset_type="stock"),
            use_multivariate=True,
            use_cross_learning=False,  # Hurts for diverse stocks
            use_multiscale=True,  # Check per-symbol config
            mae_improvement=80.0,  # ~80% from experiment
            source="default",
        ),
        "crypto_single_ohlc": ForecastConfig(
            tag=ForecastTag(symbols=("*",), targets=("open", "high", "low", "close"), asset_type="crypto"),
            use_multivariate=False,  # Didn't help for crypto
            use_cross_learning=False,  # Only helps for multi-symbol
            use_multiscale=False,  # Crypto prefers single-scale
            mae_improvement=0.0,
            source="default",
        ),
        "crypto_multi_ohlc": ForecastConfig(
            tag=ForecastTag(symbols=("*", "*"), targets=("open", "high", "low", "close"), asset_type="crypto"),
            use_multivariate=False,
            use_cross_learning=True,  # +9-11% for homogeneous crypto
            use_multiscale=False,
            mae_improvement=10.0,
            source="default",
        ),
        "stock_close_only": ForecastConfig(
            tag=ForecastTag(symbols=("*",), targets=("close",), asset_type="stock"),
            use_multivariate=False,  # Single target
            use_cross_learning=False,
            use_multiscale=True,
            mae_improvement=10.0,
            source="default",
        ),
    }

    def __init__(
        self,
        config_paths: Optional[Sequence[str | Path]] = None,
        multiscale_config_paths: Optional[Sequence[str | Path]] = None,
    ) -> None:
        """Initialize forecast config selector.

        Args:
            config_paths: Paths to check for forecast config JSON files.
            multiscale_config_paths: Paths for per-symbol multiscale config.
        """
        self._config_paths = tuple(
            Path(p) for p in (config_paths or self.DEFAULT_CONFIG_PATHS)
        )
        self._multiscale_paths = multiscale_config_paths
        self._cache: Dict[str, ForecastConfig] = {}
        self._loaded_configs: Dict[str, ForecastConfig] = {}
        self._multiscale_configs: Optional[Dict[str, Dict[str, Any]]] = None

    def _load_configs(self) -> None:
        """Load configurations from files."""
        if self._loaded_configs:
            return

        for path in self._config_paths:
            if path.exists():
                try:
                    data = json.loads(path.read_text())
                    for key, config_data in data.get("configs", {}).items():
                        self._loaded_configs[key] = ForecastConfig.from_dict(config_data)
                    logger.info("Loaded forecast config from %s", path)
                except Exception as e:
                    logger.warning("Failed to load forecast config from %s: %s", path, e)

    def _load_multiscale_config(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Load multiscale config for a specific symbol."""
        if self._multiscale_configs is None:
            self._multiscale_configs = {}
            default_paths = [
                Path("reports/multiscale_config_stocks.json"),
                Path("reports/multiscale_config_crypto.json"),
                Path("reports/multiscale_config.json"),
            ]
            paths = list(self._multiscale_paths) if self._multiscale_paths else default_paths
            for path in paths:
                if Path(path).exists():
                    try:
                        data = json.loads(Path(path).read_text())
                        symbol_configs = data.get("symbol_configs", data)
                        self._multiscale_configs.update(symbol_configs)
                    except Exception as e:
                        logger.warning("Failed to load multiscale config from %s: %s", path, e)

        return self._multiscale_configs.get(symbol.upper())

    def get_config(self, tag: ForecastTag) -> ForecastConfig:
        """Get optimal configuration for a forecast tag."""
        cache_key = tag.tag_key
        if cache_key in self._cache:
            return self._cache[cache_key]

        self._load_configs()

        # Check for exact match in loaded configs
        if cache_key in self._loaded_configs:
            config = self._loaded_configs[cache_key]
            self._cache[cache_key] = config
            return config

        # Build config based on defaults and per-symbol multiscale
        config = self._build_default_config(tag)
        self._cache[cache_key] = config
        return config

    def _build_default_config(self, tag: ForecastTag) -> ForecastConfig:
        """Build a default configuration based on tag properties."""
        is_ohlc = set(tag.targets) == {"open", "high", "low", "close"}
        is_multi_symbol = len(tag.symbols) > 1

        # Start with base defaults
        use_multivariate = is_ohlc and tag.asset_type == "stock"
        use_cross_learning = is_multi_symbol and tag.asset_type == "crypto"
        use_multiscale = False
        multiscale_method = "single"
        multiscale_skip_rates = (1,)

        # Check per-symbol multiscale config
        if len(tag.symbols) == 1:
            symbol = tag.symbols[0]
            ms_config = self._load_multiscale_config(symbol)
            if ms_config:
                method = ms_config.get("method", "single")
                if method != "single":
                    use_multiscale = True
                    multiscale_method = method
                    multiscale_skip_rates = tuple(ms_config.get("skip_rates", [1, 2, 3]))

        # Calculate expected improvement
        mae_improvement = 0.0
        if use_multivariate:
            mae_improvement += 70.0  # ~70-80% from multi-target
        if use_cross_learning and tag.asset_type == "crypto":
            mae_improvement += 10.0  # ~10% from cross-learning
        if use_multiscale:
            mae_improvement += 10.0  # ~10% from multi-scale

        return ForecastConfig(
            tag=tag,
            use_multivariate=use_multivariate,
            use_cross_learning=use_cross_learning,
            use_multiscale=use_multiscale,
            multiscale_method=multiscale_method,
            multiscale_skip_rates=multiscale_skip_rates,
            mae_improvement=mae_improvement,
            source="default_built",
        )

    def get_config_for_symbols(
        self,
        symbols: Sequence[str],
        targets: Sequence[str] = ("open", "high", "low", "close"),
    ) -> ForecastConfig:
        """Convenience method to get config from symbols and targets."""
        tag = ForecastTag.from_symbols_and_targets(symbols, targets)
        return self.get_config(tag)


def save_forecast_configs(
    configs: Dict[str, ForecastConfig],
    path: str | Path,
) -> None:
    """Save forecast configurations to a JSON file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "configs": {key: config.to_dict() for key, config in configs.items()},
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info("Saved %d forecast configs to %s", len(configs), output_path)


__all__ = [
    "ForecastTag",
    "ForecastConfig",
    "ForecastConfigSelector",
    "save_forecast_configs",
]
