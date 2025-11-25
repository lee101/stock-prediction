"""Chronos2 forecasting integration for Claude Opus trading agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import pandas as pd
from loguru import logger

from stockagent.agentsimulator.market_data import MarketDataBundle


@dataclass
class Chronos2Forecast:
    """Container for Chronos2 price forecasts with quantiles."""

    symbol: str
    # Median (point) forecasts for OHLC
    predicted_open: float
    predicted_high: float
    predicted_low: float
    predicted_close: float
    # 10th percentile (pessimistic)
    low_open: float
    low_high: float
    low_low: float
    low_close: float
    # 90th percentile (optimistic)
    high_open: float
    high_high: float
    high_low: float
    high_close: float
    # Context info
    last_close: float
    expected_return_pct: float
    volatility_range_pct: float


def generate_chronos2_forecasts(
    *,
    market_data: MarketDataBundle,
    symbols: Sequence[str],
    prediction_length: int = 1,
    context_length: int = 512,
    device_map: str = "cuda",
) -> Mapping[str, Chronos2Forecast]:
    """Generate Chronos2 forecasts for given symbols."""
    try:
        from src.models.chronos2_wrapper import Chronos2OHLCWrapper
    except ImportError as e:
        logger.warning(f"Chronos2 not available: {e}")
        return {}

    # Load Chronos2 model
    try:
        wrapper = Chronos2OHLCWrapper.from_pretrained(
            model_id="amazon/chronos-2",
            device_map=device_map,
            default_context_length=context_length,
        )
    except Exception as e:
        logger.error(f"Failed to load Chronos2 model: {e}")
        return {}

    forecasts: dict[str, Chronos2Forecast] = {}

    for symbol in symbols:
        try:
            frame = market_data.get_symbol_bars(symbol)
            if frame.empty:
                logger.warning(f"No market data for {symbol}")
                continue

            # Prepare context dataframe for Chronos2
            context_df = frame.reset_index().copy()
            if "timestamp" not in context_df.columns and "index" in context_df.columns:
                context_df = context_df.rename(columns={"index": "timestamp"})
            if "timestamp" not in context_df.columns:
                context_df["timestamp"] = pd.date_range(
                    end=pd.Timestamp.now(tz="UTC"),
                    periods=len(context_df),
                    freq="D"
                )
            context_df["symbol"] = symbol

            # Get prediction
            batch = wrapper.predict_ohlc(
                context_df,
                symbol=symbol,
                prediction_length=prediction_length,
                context_length=min(context_length, len(context_df)),
            )

            # Extract quantile forecasts
            median = batch.quantile_frames.get(0.5)
            low_q = batch.quantile_frames.get(0.1)
            high_q = batch.quantile_frames.get(0.9)

            if median is None:
                logger.warning(f"No median forecast for {symbol}")
                continue

            # Get last row of each quantile
            def get_val(df: Optional[pd.DataFrame], col: str, default: float) -> float:
                if df is None or df.empty or col not in df.columns:
                    return default
                return float(df[col].iloc[-1])

            last_close = float(frame["close"].iloc[-1])
            pred_close = get_val(median, "close", last_close)

            expected_return = (pred_close - last_close) / last_close if last_close else 0.0
            low_close_val = get_val(low_q, "close", pred_close)
            high_close_val = get_val(high_q, "close", pred_close)
            volatility_range = (high_close_val - low_close_val) / last_close if last_close else 0.0

            forecasts[symbol] = Chronos2Forecast(
                symbol=symbol,
                # Median forecasts
                predicted_open=get_val(median, "open", last_close),
                predicted_high=get_val(median, "high", last_close),
                predicted_low=get_val(median, "low", last_close),
                predicted_close=pred_close,
                # Low quantile
                low_open=get_val(low_q, "open", last_close * 0.98),
                low_high=get_val(low_q, "high", last_close * 0.98),
                low_low=get_val(low_q, "low", last_close * 0.98),
                low_close=low_close_val,
                # High quantile
                high_open=get_val(high_q, "open", last_close * 1.02),
                high_high=get_val(high_q, "high", last_close * 1.02),
                high_low=get_val(high_q, "low", last_close * 1.02),
                high_close=high_close_val,
                # Context
                last_close=last_close,
                expected_return_pct=expected_return,
                volatility_range_pct=volatility_range,
            )
            logger.info(
                f"Chronos2 forecast for {symbol}: "
                f"close ${pred_close:.2f} (exp return {expected_return:.2%})"
            )

        except Exception as e:
            logger.warning(f"Failed to generate Chronos2 forecast for {symbol}: {e}")
            continue

    # Unload model to free memory
    try:
        wrapper.unload()
    except Exception:
        pass

    return forecasts


__all__ = ["Chronos2Forecast", "generate_chronos2_forecasts"]
