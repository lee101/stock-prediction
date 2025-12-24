"""Chronos2 forecasting wrapper for stockagent3 with preaugmentation + hyperparams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import pandas as pd
from loguru import logger

from stockagent.agentsimulator.market_data import MarketDataBundle
from src.chronos2_params import resolve_chronos2_params
from src.models.chronos2_wrapper import Chronos2OHLCWrapper


@dataclass
class Chronos2Forecast:
    symbol: str
    predicted_open: float
    predicted_high: float
    predicted_low: float
    predicted_close: float
    low_close: float
    high_close: float
    last_close: float
    expected_return_pct: float
    volatility_range_pct: float
    applied_preaug_strategy: Optional[str] = None
    applied_preaug_source: Optional[str] = None
    config_path: Optional[str] = None


class Chronos2Forecaster:
    """Generate Chronos2 forecasts with per-symbol hyperparameters and preaug strategies."""

    def __init__(self, *, device_map: str = "cuda") -> None:
        self._device_map = device_map
        self._wrapper_cache: Dict[Tuple, Chronos2OHLCWrapper] = {}

    def _cache_key(self, params: Mapping[str, Any]) -> Tuple:
        quantiles = tuple(params.get("quantile_levels") or ())
        return (
            params.get("model_id"),
            params.get("context_length"),
            params.get("batch_size"),
            quantiles,
            params.get("device_map") or self._device_map,
            bool(params.get("use_multiscale")),
            bool(params.get("use_multivariate")),
        )

    def _get_wrapper(self, params: Mapping[str, Any]) -> Chronos2OHLCWrapper:
        key = self._cache_key(params)
        cached = self._wrapper_cache.get(key)
        if cached is not None:
            return cached

        wrapper = Chronos2OHLCWrapper.from_pretrained(
            model_id=str(params.get("model_id") or "amazon/chronos-2"),
            device_map=params.get("device_map") or self._device_map,
            default_context_length=int(params.get("context_length", 512)),
            default_batch_size=int(params.get("batch_size", 128)),
            quantile_levels=tuple(params.get("quantile_levels", (0.1, 0.5, 0.9))),
            multiscale_enabled=bool(params.get("use_multiscale", False)),
        )
        self._wrapper_cache[key] = wrapper
        return wrapper

    def _prepare_context(self, frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
        context_df = frame.reset_index().copy()
        if "timestamp" not in context_df.columns and "index" in context_df.columns:
            context_df = context_df.rename(columns={"index": "timestamp"})
        if "timestamp" not in context_df.columns:
            context_df["timestamp"] = pd.date_range(
                end=pd.Timestamp.now(tz="UTC"),
                periods=len(context_df),
                freq="D",
            )
        context_df.columns = [col.lower() for col in context_df.columns]
        if "symbol" not in context_df.columns:
            context_df["symbol"] = symbol
        return context_df

    def forecast_symbol(
        self,
        symbol: str,
        frame: pd.DataFrame,
        *,
        frequency: Optional[str] = None,
    ) -> Optional[Chronos2Forecast]:
        if frame.empty:
            return None

        params = resolve_chronos2_params(symbol, frequency=frequency)
        wrapper = self._get_wrapper(params)
        context_df = self._prepare_context(frame, symbol)

        prediction_length = int(params.get("prediction_length", 1))
        quantiles = tuple(params.get("quantile_levels", (0.1, 0.5, 0.9)))
        context_length = int(params.get("context_length", 512))
        batch_size = int(params.get("batch_size", 128))
        predict_kwargs = dict(params.get("predict_kwargs") or {})
        use_multivariate = bool(params.get("use_multivariate", False))

        try:
            if use_multivariate:
                result = wrapper.predict_ohlc_multivariate(
                    context_df=context_df,
                    symbol=symbol,
                    prediction_length=prediction_length,
                    context_length=context_length,
                    quantile_levels=quantiles,
                    batch_size=batch_size,
                )
            else:
                result = wrapper.predict_ohlc(
                    context_df=context_df,
                    symbol=symbol,
                    prediction_length=prediction_length,
                    context_length=context_length,
                    batch_size=batch_size,
                    quantile_levels=quantiles,
                    predict_kwargs=predict_kwargs or None,
                )
        except Exception as exc:
            logger.warning("Chronos2 forecast failed for %s: %s", symbol, exc)
            return None

        median = result.quantile_frames.get(0.5)
        low_q = result.quantile_frames.get(0.1)
        high_q = result.quantile_frames.get(0.9)
        if median is None or median.empty:
            return None

        def _get_val(df: Optional[pd.DataFrame], column: str, default: float) -> float:
            if df is None or df.empty or column not in df.columns:
                return default
            return float(df[column].iloc[-1])

        last_close = float(frame["close"].iloc[-1])
        pred_close = _get_val(median, "close", last_close)
        low_close = _get_val(low_q, "close", pred_close)
        high_close = _get_val(high_q, "close", pred_close)

        expected_return = (pred_close - last_close) / last_close if last_close else 0.0
        volatility_range = (high_close - low_close) / last_close if last_close else 0.0

        applied_aug = getattr(result, "applied_augmentation", None)
        applied_choice = getattr(result, "applied_choice", None)
        source_path = None
        if applied_choice is not None:
            source_path = getattr(applied_choice, "source_path", None)

        return Chronos2Forecast(
            symbol=symbol,
            predicted_open=_get_val(median, "open", last_close),
            predicted_high=_get_val(median, "high", last_close),
            predicted_low=_get_val(median, "low", last_close),
            predicted_close=pred_close,
            low_close=low_close,
            high_close=high_close,
            last_close=last_close,
            expected_return_pct=expected_return,
            volatility_range_pct=volatility_range,
            applied_preaug_strategy=applied_aug,
            applied_preaug_source=str(source_path) if source_path else None,
            config_path=params.get("_config_path") if isinstance(params, dict) else None,
        )

    def forecast_all(
        self,
        *,
        market_data: MarketDataBundle | Mapping[str, pd.DataFrame],
        symbols: Sequence[str],
        frequency: Optional[str] = None,
    ) -> Dict[str, Chronos2Forecast]:
        if isinstance(market_data, MarketDataBundle):
            frames = market_data.bars
        else:
            frames = dict(market_data)

        forecasts: Dict[str, Chronos2Forecast] = {}
        for symbol in symbols:
            frame = frames.get(symbol.upper())
            if frame is None or frame.empty:
                continue
            forecast = self.forecast_symbol(symbol, frame, frequency=frequency)
            if forecast is None:
                continue
            forecasts[symbol] = forecast
        return forecasts

