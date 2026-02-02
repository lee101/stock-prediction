"""Chronos2 forecasting for hourly market simulation."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import DataConfigHourly, ForecastConfigHourly
from .data import HourlyDataLoader, is_crypto_symbol

logger = logging.getLogger(__name__)


@dataclass
class SymbolForecast:
    symbol: str
    forecast_time: pd.Timestamp
    current_close: float
    predicted_close: float
    predicted_high: float
    predicted_low: float
    predicted_close_p10: float
    predicted_close_p90: float

    @property
    def predicted_return(self) -> float:
        if self.current_close <= 0:
            return 0.0
        return (self.predicted_close - self.current_close) / self.current_close

    @property
    def predicted_upside(self) -> float:
        if self.current_close <= 0:
            return 0.0
        return (self.predicted_close_p90 - self.current_close) / self.current_close

    @property
    def predicted_downside(self) -> float:
        if self.current_close <= 0:
            return 0.0
        return (self.predicted_close_p10 - self.current_close) / self.current_close

    @property
    def confidence_spread(self) -> float:
        if self.current_close <= 0:
            return float("inf")
        return (self.predicted_close_p90 - self.predicted_close_p10) / self.current_close


@dataclass
class HourlyForecasts:
    forecast_time: pd.Timestamp
    forecasts: Dict[str, SymbolForecast]

    def get_ranked_symbols(
        self,
        metric: str = "predicted_return",
        ascending: bool = False,
    ) -> List[Tuple[str, float]]:
        ranked = []
        for symbol, forecast in self.forecasts.items():
            value = getattr(forecast, metric, 0.0)
            if np.isfinite(value):
                ranked.append((symbol, value))
        ranked.sort(key=lambda x: x[1], reverse=not ascending)
        return ranked

    def get_top_n_symbols(
        self,
        n: int = 1,
        metric: str = "predicted_return",
        min_return: float = 0.0,
    ) -> List[str]:
        ranked = self.get_ranked_symbols(metric=metric, ascending=False)
        filtered = [(sym, val) for sym, val in ranked if val >= min_return]
        return [sym for sym, _ in filtered[:n]]


class Chronos2HourlyForecaster:
    """Generates Chronos2 forecasts for hourly data."""

    def __init__(
        self,
        data_loader: HourlyDataLoader,
        forecast_config: ForecastConfigHourly,
    ) -> None:
        self.data_loader = data_loader
        self.config = forecast_config
        self._wrapper = None
        self._initialized = False

    @staticmethod
    def _env_cross_learning_override() -> Optional[bool]:
        value = os.getenv("CHRONOS2_CROSS_LEARNING")
        if value is None:
            return None
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        return None

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return

        if self.config.frequency:
            os.environ.setdefault("CHRONOS2_FREQUENCY", self.config.frequency)

        try:
            from src.models.chronos2_wrapper import Chronos2OHLCWrapper

            self._wrapper = Chronos2OHLCWrapper.from_pretrained(
                model_id=self.config.model_id,
                device_map=self.config.device_map,
                default_context_length=self.config.context_length,
                quantile_levels=list(self.config.quantile_levels),
                default_batch_size=self.config.batch_size,
                preaugmentation_dirs=(
                    list(self.config.preaugmentation_dirs)
                    if self.config.use_preaugmentation
                    else None
                ),
            )
            self._initialized = True
            logger.info("Initialized Chronos2 hourly forecaster with model %s", self.config.model_id)
        except Exception as exc:
            logger.error("Failed to initialize Chronos2 hourly forecaster: %s", exc)
            raise

    def _forecast_from_batch(
        self,
        symbol: str,
        target_time: pd.Timestamp,
        current_close: float,
        batch: Any,
    ) -> Optional[SymbolForecast]:
        q50 = getattr(batch, "quantile_frames", {}).get(0.5)
        q10 = getattr(batch, "quantile_frames", {}).get(0.1)
        q90 = getattr(batch, "quantile_frames", {}).get(0.9)
        if q50 is None or q50.empty:
            logger.warning("Empty predictions for %s at %s", symbol, target_time)
            return None

        pred_close = float(q50.iloc[0].get("close", current_close))
        pred_high = float(q50.iloc[0].get("high", pred_close))
        pred_low = float(q50.iloc[0].get("low", pred_close))
        pred_close_p10 = float(q10.iloc[0].get("close", pred_close)) if q10 is not None else pred_close * 0.95
        pred_close_p90 = float(q90.iloc[0].get("close", pred_close)) if q90 is not None else pred_close * 1.05

        return SymbolForecast(
            symbol=symbol,
            forecast_time=target_time,
            current_close=current_close,
            predicted_close=pred_close,
            predicted_high=pred_high,
            predicted_low=pred_low,
            predicted_close_p10=pred_close_p10,
            predicted_close_p90=pred_close_p90,
        )

    @staticmethod
    def _extract_last_close(context_df: pd.DataFrame) -> float:
        close_col = context_df["close"]
        if isinstance(close_col, pd.DataFrame):
            close_series = close_col.iloc[:, 0]
        else:
            close_series = close_col
        return float(close_series.iloc[-1])

    def _forecast_from_context(
        self,
        symbol: str,
        target_time: pd.Timestamp,
        context_df: pd.DataFrame,
        current_close: float,
        *,
        use_multivariate: bool,
    ) -> Optional[SymbolForecast]:
        if self._wrapper is None:
            raise RuntimeError("Chronos2 wrapper not initialized.")

        try:
            if use_multivariate and hasattr(self._wrapper, "predict_ohlc_multivariate"):
                batch = self._wrapper.predict_ohlc_multivariate(
                    context_df,
                    symbol=symbol,
                    prediction_length=self.config.prediction_length,
                    context_length=len(context_df),
                    quantile_levels=list(self.config.quantile_levels),
                )
            else:
                batch = self._wrapper.predict_ohlc(
                    context_df,
                    symbol=symbol,
                    prediction_length=self.config.prediction_length,
                    context_length=len(context_df),
                    quantile_levels=list(self.config.quantile_levels),
                )
        except Exception as exc:
            logger.error("Forecast failed for %s at %s: %s", symbol, target_time, exc)
            return None

        return self._forecast_from_batch(symbol, target_time, current_close, batch)

    def forecast_all_symbols(
        self,
        target_time: pd.Timestamp,
        symbols: Optional[List[str]] = None,
    ) -> HourlyForecasts:
        self._ensure_initialized()

        if symbols is None:
            symbols = self.data_loader.get_tradable_symbols_at(target_time)

        forecasts: Dict[str, SymbolForecast] = {}
        use_cross_learning = self._env_cross_learning_override()
        if use_cross_learning is None:
            use_cross_learning = bool(self.config.use_cross_learning)

        entries: List[Dict[str, Any]] = []
        for symbol in symbols:
            context_df = self.data_loader.get_context_for_timestamp(
                symbol,
                target_time,
                context_hours=self.config.context_length,
            )
            if context_df.empty or len(context_df) < 10:
                logger.warning("Insufficient context for %s at %s", symbol, target_time)
                continue
            current_close = self._extract_last_close(context_df)
            entries.append(
                {
                    "symbol": symbol,
                    "context": context_df,
                    "current_close": current_close,
                    "asset_type": "crypto" if is_crypto_symbol(symbol) else "stock",
                    "use_multivariate": self.config.use_multivariate and not is_crypto_symbol(symbol),
                }
            )

        if not entries:
            return HourlyForecasts(forecast_time=target_time, forecasts={})

        if use_cross_learning:
            min_batch = max(2, int(self.config.cross_learning_min_batch))
            if self.config.cross_learning_group_by_asset_type:
                grouped: Dict[str, List[Dict[str, Any]]] = {"stock": [], "crypto": []}
                for entry in entries:
                    grouped[entry["asset_type"]].append(entry)
                groups = [group for group in grouped.values() if group]
            else:
                groups = [entries]

            for group in groups:
                chunk_size = self.config.cross_learning_chunk_size
                if chunk_size is None or chunk_size <= 0:
                    chunk_size = len(group)

                for start_idx in range(0, len(group), chunk_size):
                    chunk = group[start_idx:start_idx + chunk_size]
                    if len(chunk) < min_batch:
                        for entry in chunk:
                            forecast = self._forecast_from_context(
                                entry["symbol"],
                                target_time,
                                entry["context"],
                                entry["current_close"],
                                use_multivariate=bool(entry["use_multivariate"]),
                            )
                            if forecast is not None:
                                forecasts[entry["symbol"]] = forecast
                        continue

                    contexts = [entry["context"] for entry in chunk]
                    symbols_batch = [entry["symbol"] for entry in chunk]
                    predict_kwargs = {"predict_batches_jointly": True}
                    try:
                        batches = self._wrapper.predict_ohlc_batch(
                            contexts,
                            symbols=symbols_batch,
                            prediction_length=self.config.prediction_length,
                            context_length=self.config.context_length,
                            quantile_levels=list(self.config.quantile_levels),
                            batch_size=self.config.batch_size,
                            predict_kwargs=predict_kwargs,
                        )
                    except Exception as exc:
                        logger.error(
                            "Batch forecast failed (%d symbols) at %s: %s",
                            len(chunk),
                            target_time,
                            exc,
                        )
                        for entry in chunk:
                            forecast = self._forecast_from_context(
                                entry["symbol"],
                                target_time,
                                entry["context"],
                                entry["current_close"],
                                use_multivariate=bool(entry["use_multivariate"]),
                            )
                            if forecast is not None:
                                forecasts[entry["symbol"]] = forecast
                        continue

                    for entry, batch in zip(chunk, batches):
                        forecast = self._forecast_from_batch(
                            entry["symbol"],
                            target_time,
                            entry["current_close"],
                            batch,
                        )
                        if forecast is not None:
                            forecasts[entry["symbol"]] = forecast
        else:
            for entry in entries:
                forecast = self._forecast_from_context(
                    entry["symbol"],
                    target_time,
                    entry["context"],
                    entry["current_close"],
                    use_multivariate=bool(entry["use_multivariate"]),
                )
                if forecast is not None:
                    forecasts[entry["symbol"]] = forecast

        return HourlyForecasts(forecast_time=target_time, forecasts=forecasts)

    def unload(self) -> None:
        if self._wrapper is not None:
            self._wrapper.unload()
            self._wrapper = None
            self._initialized = False
            logger.info("Chronos2 hourly forecaster unloaded")


def create_forecaster(
    data_config: DataConfigHourly,
    forecast_config: ForecastConfigHourly,
) -> Chronos2HourlyForecaster:
    data_loader = HourlyDataLoader(data_config)
    data_loader.load_all_symbols()
    return Chronos2HourlyForecaster(data_loader, forecast_config)
