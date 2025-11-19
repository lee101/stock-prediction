from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from strategytraining.collect_strategy_pnl_dataset import StrategyPnLCollector
from src.chronos2_params import resolve_chronos2_params

try:  # pragma: no cover - optional dependency in tests
    from src.models.chronos2_wrapper import (
        Chronos2OHLCWrapper,
        DEFAULT_QUANTILE_LEVELS,
        DEFAULT_TARGET_COLUMNS,
    )
except Exception as exc:  # pragma: no cover - surfaced when chronos deps missing
    Chronos2OHLCWrapper = None  # type: ignore
    _CHRONOS_IMPORT_ERROR = exc
    DEFAULT_TARGET_COLUMNS = ("open", "high", "low", "close")  # type: ignore
    DEFAULT_QUANTILE_LEVELS = (0.1, 0.5, 0.9)  # type: ignore
else:  # pragma: no cover - executed only when import succeeds
    _CHRONOS_IMPORT_ERROR = None

logger = logging.getLogger(__name__)

ForecastRow = Dict[str, object]
WrapperFactory = Callable[[], object]


REQUIRED_QUANTILES: Tuple[float, ...] = (0.1, 0.5, 0.9)


@dataclass
class ForecastGenerationConfig:
    context_length: int = 512
    prediction_length: int = 1
    quantile_levels: Sequence[float] = DEFAULT_QUANTILE_LEVELS
    batch_size: int = 256
    frequency: str = "daily"
    use_symbol_hyperparams: bool = True

    def __post_init__(self) -> None:  # pragma: no cover - trivial normalization
        quantiles = tuple(float(level) for level in self.quantile_levels)
        deduped = sorted({round(level, 8) for level in (*quantiles, *REQUIRED_QUANTILES)})
        self.quantile_levels = tuple(deduped)
        freq = (self.frequency or "daily").strip().lower()
        if freq not in {"daily", "hourly"}:
            freq = "daily"
        self.frequency = freq


@dataclass(frozen=True)
class SymbolChronosSpec:
    symbol: str
    context_length: int
    prediction_length: int
    quantile_levels: Tuple[float, ...]
    batch_size: int
    predict_kwargs: Mapping[str, Any]
    config_name: str = ""
    config_path: Optional[str] = None


class ForecastCache:
    """Simple on-disk cache storing per-symbol forecast parquet files."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _symbol_path(self, symbol: str) -> Path:
        safe = symbol.replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{safe}.parquet"

    def load(self, symbol: str) -> pd.DataFrame:
        path = self._symbol_path(symbol)
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def write(self, symbol: str, frame: pd.DataFrame) -> None:
        path = self._symbol_path(symbol)
        frame.sort_values("timestamp", inplace=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)


class ChronosForecastGenerator:
    """Generate next-day Chronos2 forecasts and persist them via ForecastCache."""

    def __init__(
        self,
        *,
        data_dir: Path,
        cache_dir: Path,
        config: ForecastGenerationConfig,
        wrapper_kwargs: Optional[Mapping[str, object]] = None,
        wrapper_factory: Optional[WrapperFactory] = None,
    ) -> None:
        self.collector = StrategyPnLCollector(data_dir=str(data_dir))
        self.cache = ForecastCache(cache_dir)
        self.config = config
        self.wrapper_kwargs = dict(wrapper_kwargs or {})
        self._wrapper_factory = wrapper_factory
        self._wrapper: Optional[object] = None
        self._base_quantile_levels = tuple(float(level) for level in config.quantile_levels)
        self._symbol_specs: Dict[str, SymbolChronosSpec] = {}

    def _ensure_wrapper(self) -> object:
        if self._wrapper is not None:
            return self._wrapper
        if self._wrapper_factory is not None:
            self._wrapper = self._wrapper_factory()
            return self._wrapper
        if Chronos2OHLCWrapper is None:
            raise RuntimeError("Chronos2OHLCWrapper unavailable; install chronos-forecasting>=2.0.") from _CHRONOS_IMPORT_ERROR
        kwargs = dict(self.wrapper_kwargs)
        kwargs.setdefault("model_id", "amazon/chronos-2")
        kwargs.setdefault("device_map", "cuda")
        kwargs.setdefault("default_context_length", self.config.context_length)
        kwargs.setdefault("default_batch_size", self.config.batch_size)
        kwargs.setdefault("quantile_levels", self._base_quantile_levels)
        self._wrapper = Chronos2OHLCWrapper.from_pretrained(**kwargs)
        return self._wrapper

    def _normalize_quantiles(self, levels: Optional[Sequence[float]]) -> Tuple[float, ...]:
        raw = tuple(float(level) for level in (levels or self._base_quantile_levels))
        merged = (*raw, *REQUIRED_QUANTILES)
        deduped = sorted({round(level, 8) for level in merged})
        return tuple(deduped)

    def _spec_from_config(self, symbol: str) -> SymbolChronosSpec:
        return SymbolChronosSpec(
            symbol=symbol,
            context_length=int(self.config.context_length),
            prediction_length=int(self.config.prediction_length),
            quantile_levels=self._normalize_quantiles(self._base_quantile_levels),
            batch_size=int(self.config.batch_size),
            predict_kwargs={},
        )

    def _resolve_symbol_spec(self, symbol: str) -> SymbolChronosSpec:
        key = symbol.upper()
        cached = self._symbol_specs.get(key)
        if cached is not None:
            return cached

        if not self.config.use_symbol_hyperparams:
            spec = self._spec_from_config(key)
            self._symbol_specs[key] = spec
            return spec

        try:
            params = resolve_chronos2_params(
                key,
                frequency=self.config.frequency,
                default_prediction_length=self.config.prediction_length,
            )
        except Exception as exc:
            logger.warning(
                "Chronos2 hyperparameter lookup failed for %s (%s); falling back to defaults.",
                key,
                exc,
            )
            spec = self._spec_from_config(key)
        else:
            quantiles = self._normalize_quantiles(params.get("quantile_levels"))
            spec = SymbolChronosSpec(
                symbol=key,
                context_length=int(params.get("context_length", self.config.context_length)),
                prediction_length=int(params.get("prediction_length", self.config.prediction_length)),
                quantile_levels=quantiles,
                batch_size=int(params.get("batch_size", self.config.batch_size)),
                predict_kwargs=dict(params.get("predict_kwargs") or {}),
                config_name=str(params.get("_config_name") or ""),
                config_path=params.get("_config_path"),
            )
            if spec.config_path:
                logger.info(
                    "Using Chronos2 hyperparams for %s from %s (ctx=%d, batch=%d, q=%s).",
                    key,
                    spec.config_path,
                    spec.context_length,
                    spec.batch_size,
                    ",".join(f"{level:.2f}" for level in spec.quantile_levels),
                )

        self._symbol_specs[key] = spec
        return spec

    def _load_symbol_history(self, symbol: str) -> pd.DataFrame:
        df = self.collector.load_symbol_data(symbol)
        if df is None or df.empty:
            raise ValueError(f"No historical data found for symbol {symbol}.")
        renamed = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
                "timestamp": "timestamp",
            }
        ).copy()
        renamed["timestamp"] = pd.to_datetime(renamed["timestamp"], utc=True)
        renamed["symbol"] = symbol
        keep_cols = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        available = [col for col in keep_cols if col in renamed.columns]
        history = renamed[available].sort_values("timestamp").reset_index(drop=True)
        return history

    @staticmethod
    def _to_timestamp(value: Optional[str]) -> Optional[pd.Timestamp]:
        if not value:
            return None
        ts = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(ts):
            return None
        return ts

    def generate(
        self,
        symbols: Sequence[str],
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> None:
        start_ts = self._to_timestamp(start_date)
        end_ts = self._to_timestamp(end_date)
        wrapper = None

        for symbol in symbols:
            try:
                history = self._load_symbol_history(symbol)
            except Exception as exc:
                logger.warning("Skipping %s due to history load failure: %s", symbol, exc)
                continue
            if len(history) < 2:
                logger.info("Skipping %s – not enough rows for forecast generation.", symbol)
                continue

            spec = self._resolve_symbol_spec(symbol)

            existing = self.cache.load(symbol)
            existing_timestamps = set()
            if not existing.empty and "timestamp" in existing.columns:
                existing_timestamps = set(pd.to_datetime(existing["timestamp"], utc=True))

            new_rows: List[ForecastRow] = []
            last_index = len(history) - 2
            for idx in range(0, last_index + 1):
                target_idx = idx + 1
                target_ts = history.iloc[target_idx]["timestamp"]
                if start_ts and target_ts < start_ts:
                    continue
                if end_ts and target_ts > end_ts:
                    break
                if target_ts in existing_timestamps:
                    continue
                start_idx = max(0, target_idx - spec.context_length)
                context_slice = history.iloc[start_idx:target_idx].copy()
                if context_slice.empty:
                    continue
                context_slice = context_slice.rename(columns=str.lower)
                # Ensure required columns exist
                missing = [col for col in DEFAULT_TARGET_COLUMNS if col not in context_slice.columns]
                if missing:
                    logger.debug("Skipping %s at %s due to missing columns %s", symbol, target_ts, missing)
                    continue
                wrapper = wrapper or self._ensure_wrapper()
                try:
                    row = self._predict_row(
                        wrapper,
                        symbol,
                        context_slice,
                        target_ts,
                        quantile_levels=spec.quantile_levels,
                        prediction_length=spec.prediction_length,
                        batch_size=spec.batch_size,
                        context_length=spec.context_length,
                        predict_kwargs=spec.predict_kwargs,
                    )
                except Exception as exc:
                    logger.warning("Chronos forecast failed for %s at %s: %s", symbol, target_ts, exc)
                    continue
                new_rows.append(row)

            if not new_rows:
                logger.info("No new forecasts generated for %s.", symbol)
                continue

            new_df = pd.DataFrame(new_rows)
            combined = (
                pd.concat([existing, new_df], ignore_index=True)
                .drop_duplicates(subset=["timestamp", "symbol"], keep="last")
                .sort_values("timestamp")
                .reset_index(drop=True)
            )
            self.cache.write(symbol, combined)
            logger.info("Generated %d forecasts for %s (total cached: %d).", len(new_rows), symbol, len(combined))

    def _predict_row(
        self,
        wrapper: object,
        symbol: str,
        context_slice: pd.DataFrame,
        target_ts: pd.Timestamp,
        *,
        quantile_levels: Sequence[float],
        prediction_length: int,
        batch_size: int,
        context_length: int,
        predict_kwargs: Mapping[str, Any],
    ) -> ForecastRow:
        prediction = wrapper.predict_ohlc(
            context_slice,
            symbol=symbol,
            prediction_length=prediction_length,
            context_length=context_length,
            batch_size=batch_size,
            quantile_levels=quantile_levels,
            predict_kwargs=predict_kwargs,
        )
        quantile_frames: Mapping[float, pd.DataFrame] = prediction.quantile_frames

        def _extract(level: float, column: str) -> Optional[float]:
            frame = quantile_frames.get(level)
            if frame is None:
                return None
            try:
                lookup_ts = target_ts if target_ts in frame.index else frame.index[0]
                value = frame.loc[lookup_ts, column]
            except KeyError:
                return None
            if pd.isna(value):
                return None
            return float(value)

        last_close = float(context_slice["close"].iloc[-1])
        q10_close = _extract(0.1, "close")
        q50_close = _extract(0.5, "close")
        q90_close = _extract(0.9, "close")
        q50_high = _extract(0.5, "high")
        q50_low = _extract(0.5, "low")

        move_pct = 0.0
        if q50_close is not None and last_close:
            move_pct = (q50_close - last_close) / last_close

        volatility_pct = 0.0
        if q10_close is not None and q90_close is not None and last_close:
            volatility_pct = (q90_close - q10_close) / abs(last_close)

        row: ForecastRow = {
            "symbol": symbol,
            "timestamp": pd.to_datetime(target_ts, utc=True),
            "predicted_close": q50_close or 0.0,
            "predicted_close_p10": q10_close or 0.0,
            "predicted_close_p90": q90_close or 0.0,
            "predicted_high": q50_high or 0.0,
            "predicted_low": q50_low or 0.0,
            "forecast_move_pct": move_pct,
            "forecast_volatility_pct": volatility_pct,
            "context_close": last_close,
            "quantile_levels": json.dumps(tuple(float(q) for q in quantile_levels)),
        }
        return row


__all__ = [
    "ChronosForecastGenerator",
    "ForecastCache",
    "ForecastGenerationConfig",
]
