"""Tuned per-symbol Chronos2 forecasting for long-term market simulation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import DataConfigLong, ForecastConfigLong
from .data import DailyDataLoader, is_crypto_symbol
from .forecaster import SymbolForecast, DailyForecasts

logger = logging.getLogger(__name__)


@dataclass
class SymbolConfig:
    """Per-symbol tuned configuration."""

    symbol: str
    context_length: int = 512
    use_multivariate: bool = True
    preaug_strategy: str = "baseline"
    skip_rates: Tuple[int, ...] = (1,)
    aggregation_method: str = "single"
    mae_pct: float = float("inf")
    directional_accuracy: float = 0.0


def load_all_symbol_configs(config_dir: Path) -> Dict[str, SymbolConfig]:
    """Load all per-symbol configs from directory."""
    configs = {}

    for config_path in config_dir.glob("*.json"):
        try:
            with open(config_path) as f:
                data = json.load(f)

            configs[data["symbol"]] = SymbolConfig(
                symbol=data["symbol"],
                context_length=data.get("context_length", 512),
                use_multivariate=data.get("use_multivariate", True),
                preaug_strategy=data.get("preaug_strategy", "baseline"),
                skip_rates=tuple(data.get("skip_rates", [1])),
                aggregation_method=data.get("aggregation_method", "single"),
                mae_pct=data.get("mae_pct", float("inf")),
                directional_accuracy=data.get("directional_accuracy", 0.0),
            )
        except Exception as e:
            logger.warning("Failed to load config %s: %s", config_path, e)

    return configs


class TunedChronos2Forecaster:
    """Generates Chronos2 forecasts using per-symbol tuned configs."""

    def __init__(
        self,
        data_loader: DailyDataLoader,
        forecast_config: ForecastConfigLong,
        symbol_configs_dir: Optional[Path] = None,
    ) -> None:
        self.data_loader = data_loader
        self.config = forecast_config
        self._wrapper = None
        self._initialized = False

        # Load per-symbol configs
        if symbol_configs_dir is None:
            symbol_configs_dir = Path("hyperparams/chronos2_long/daily")

        self._symbol_configs = load_all_symbol_configs(symbol_configs_dir)
        logger.info("Loaded %d per-symbol configs", len(self._symbol_configs))

    def _ensure_initialized(self) -> None:
        """Lazy initialization of Chronos2 wrapper."""
        if self._initialized:
            return

        try:
            from src.models.chronos2_wrapper import Chronos2OHLCWrapper

            self._wrapper = Chronos2OHLCWrapper.from_pretrained(
                model_id=self.config.model_id,
                device_map=self.config.device_map,
                default_context_length=self.config.context_length,
                quantile_levels=list(self.config.quantile_levels),
                default_batch_size=self.config.batch_size,
            )
            self._initialized = True
            logger.info("Initialized Chronos2 forecaster with model %s", self.config.model_id)
        except Exception as e:
            logger.error("Failed to initialize Chronos2: %s", e)
            raise

    def _get_symbol_config(self, symbol: str) -> SymbolConfig:
        """Get config for a symbol, with defaults if not found."""
        if symbol in self._symbol_configs:
            return self._symbol_configs[symbol]

        # Default config
        return SymbolConfig(
            symbol=symbol,
            context_length=512,
            use_multivariate=not is_crypto_symbol(symbol),
            preaug_strategy="baseline",
            skip_rates=(1,),
            aggregation_method="single",
        )

    def _predict_with_skip_rate(
        self,
        context_df: pd.DataFrame,
        symbol: str,
        skip_rate: int,
        use_multivariate: bool,
    ) -> Optional[Dict[str, float]]:
        """Generate prediction with a specific skip rate."""
        # Subsample context if skip_rate > 1
        if skip_rate > 1:
            subsampled = context_df.iloc[::skip_rate].copy().reset_index(drop=True)
        else:
            subsampled = context_df

        if len(subsampled) < 20:
            return None

        try:
            if use_multivariate and hasattr(self._wrapper, "predict_ohlc_multivariate"):
                batch = self._wrapper.predict_ohlc_multivariate(
                    subsampled,
                    symbol=symbol,
                    prediction_length=self.config.prediction_length,
                    context_length=len(subsampled),
                    quantile_levels=list(self.config.quantile_levels),
                )
            else:
                batch = self._wrapper.predict_ohlc(
                    subsampled,
                    symbol=symbol,
                    prediction_length=self.config.prediction_length,
                    context_length=len(subsampled),
                    quantile_levels=list(self.config.quantile_levels),
                )

            q50 = batch.quantile_frames.get(0.5)
            q10 = batch.quantile_frames.get(0.1)
            q90 = batch.quantile_frames.get(0.9)

            if q50 is None or q50.empty:
                return None

            result = {
                "close": float(q50.iloc[0].get("close", 0)),
                "high": float(q50.iloc[0].get("high", 0)),
                "low": float(q50.iloc[0].get("low", 0)),
            }

            if q10 is not None and not q10.empty:
                result["close_p10"] = float(q10.iloc[0].get("close", result["close"]))
            if q90 is not None and not q90.empty:
                result["close_p90"] = float(q90.iloc[0].get("close", result["close"]))

            return result

        except Exception as e:
            logger.debug("Prediction failed for %s with skip_rate=%d: %s", symbol, skip_rate, e)
            return None

    def _aggregate_predictions(
        self,
        predictions: Dict[int, Dict[str, float]],
        aggregation_method: str,
    ) -> Dict[str, float]:
        """Aggregate predictions from multiple skip rates."""
        if not predictions:
            return {}

        if len(predictions) == 1:
            return list(predictions.values())[0]

        # Aggregate each field
        result = {}
        fields = ["close", "high", "low", "close_p10", "close_p90"]

        for field in fields:
            values = [p[field] for p in predictions.values() if field in p]
            if not values:
                continue

            if aggregation_method == "trimmed" and len(values) >= 3:
                from scipy import stats
                result[field] = stats.trim_mean(values, proportiontocut=0.1)
            elif aggregation_method == "median":
                result[field] = np.median(values)
            else:
                result[field] = np.mean(values)

        return result

    def forecast_symbol(
        self,
        symbol: str,
        target_date: date,
    ) -> Optional[SymbolForecast]:
        """Generate forecast for a single symbol using tuned config.

        Args:
            symbol: Trading symbol
            target_date: Date to forecast

        Returns:
            SymbolForecast or None if forecast failed
        """
        self._ensure_initialized()

        # Get per-symbol tuned config
        sym_config = self._get_symbol_config(symbol)

        # Get historical context
        context_df = self.data_loader.get_context_for_date(
            symbol,
            target_date,
            context_days=sym_config.context_length,
        )

        if context_df.empty or len(context_df) < 10:
            logger.warning("Insufficient context for %s on %s", symbol, target_date)
            return None

        # Get current close (last available close before target_date)
        current_close = float(context_df.iloc[-1]["close"])

        # Determine whether to use multivariate
        should_multivariate = sym_config.use_multivariate

        try:
            # Generate predictions for each skip rate
            predictions = {}
            for skip_rate in sym_config.skip_rates:
                pred = self._predict_with_skip_rate(
                    context_df,
                    symbol,
                    skip_rate,
                    should_multivariate,
                )
                if pred is not None:
                    predictions[skip_rate] = pred

            if not predictions:
                logger.warning("No valid predictions for %s on %s", symbol, target_date)
                return None

            # Aggregate predictions
            agg_pred = self._aggregate_predictions(
                predictions,
                sym_config.aggregation_method,
            )

            pred_close = agg_pred.get("close", current_close)
            pred_high = agg_pred.get("high", pred_close)
            pred_low = agg_pred.get("low", pred_close)
            pred_close_p10 = agg_pred.get("close_p10", pred_close * 0.95)
            pred_close_p90 = agg_pred.get("close_p90", pred_close * 1.05)

            return SymbolForecast(
                symbol=symbol,
                forecast_date=target_date,
                current_close=current_close,
                predicted_close=pred_close,
                predicted_high=pred_high,
                predicted_low=pred_low,
                predicted_close_p10=pred_close_p10,
                predicted_close_p90=pred_close_p90,
            )

        except Exception as e:
            logger.error("Forecast failed for %s on %s: %s", symbol, target_date, e)
            return None

    def forecast_all_symbols(
        self,
        target_date: date,
        symbols: Optional[List[str]] = None,
    ) -> DailyForecasts:
        """Generate forecasts for all symbols on a target date.

        Args:
            target_date: Date to forecast
            symbols: Optional list of symbols (defaults to all tradable)

        Returns:
            DailyForecasts containing all symbol forecasts
        """
        self._ensure_initialized()

        if symbols is None:
            symbols = self.data_loader.get_tradable_symbols_on_date(target_date)

        forecasts = {}
        for symbol in symbols:
            forecast = self.forecast_symbol(symbol, target_date)
            if forecast is not None:
                forecasts[symbol] = forecast

        logger.info(
            "Generated %d tuned forecasts for %s (out of %d symbols)",
            len(forecasts),
            target_date,
            len(symbols),
        )

        return DailyForecasts(
            forecast_date=target_date,
            forecasts=forecasts,
        )

    def unload(self) -> None:
        """Release GPU memory."""
        if self._wrapper is not None:
            self._wrapper.unload()
            self._wrapper = None
            self._initialized = False
            logger.info("Tuned Chronos2 forecaster unloaded")


def create_tuned_forecaster(
    data_config: DataConfigLong,
    forecast_config: ForecastConfigLong,
    symbol_configs_dir: Optional[Path] = None,
) -> TunedChronos2Forecaster:
    """Factory function to create a tuned forecaster.

    Args:
        data_config: Data configuration
        forecast_config: Forecast configuration
        symbol_configs_dir: Directory with per-symbol configs

    Returns:
        Initialized TunedChronos2Forecaster
    """
    data_loader = DailyDataLoader(data_config)
    data_loader.load_all_symbols()
    return TunedChronos2Forecaster(data_loader, forecast_config, symbol_configs_dir)
