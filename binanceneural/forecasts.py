from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd

from src.chronos2_params import resolve_chronos2_params

try:  # pragma: no cover - optional heavy dependency
    from src.models.chronos2_wrapper import Chronos2OHLCWrapper, Chronos2PredictionBatch
except Exception as exc:  # pragma: no cover
    Chronos2OHLCWrapper = None  # type: ignore
    Chronos2PredictionBatch = Any  # type: ignore
    _CHRONOS_IMPORT_ERROR = exc
else:  # pragma: no cover
    _CHRONOS_IMPORT_ERROR = None

from .config import ForecastConfig

logger = logging.getLogger(__name__)


class ForecastCache:
    """Parquet-backed cache for Chronos forecasts."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, symbol: str) -> Path:
        safe = symbol.replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{safe.upper()}.parquet"

    def load(self, symbol: str) -> pd.DataFrame:
        path = self._path(symbol)
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def write(self, symbol: str, frame: pd.DataFrame) -> None:
        path = self._path(symbol)
        frame = frame.sort_values("timestamp").reset_index(drop=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)


@dataclass
class ChronosForecastManager:
    """Generate and cache Chronos2 forecasts for a given horizon."""

    config: ForecastConfig
    wrapper_factory: Optional[Callable[[], object]] = None

    def __post_init__(self) -> None:
        self.cache = ForecastCache(self.config.cache_dir)
        self._wrapper: Optional[object] = None
        self._predict_kwargs: Dict[str, object] = {}
        self._warned_missing = False

    def ensure_latest(
        self,
        *,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        cache_only: bool = False,
    ) -> pd.DataFrame:
        """Ensure forecasts exist for the requested range."""

        history = self._load_history()
        if history.empty:
            raise RuntimeError(f"No hourly history found for {self.config.symbol} at {self.config.data_root}")
        start_ts = self._coerce_timestamp(start) or history["timestamp"].min()
        end_ts = self._coerce_timestamp(end) or history["timestamp"].max()
        existing = self.cache.load(self.config.symbol)
        existing_ts = set(pd.to_datetime(existing["timestamp"], utc=True)) if not existing.empty else set()

        min_idx = max(1, int(self.config.context_hours))
        missing_indices: List[int] = []
        for idx in range(min_idx, len(history)):
            target_ts = history.iloc[idx]["timestamp"]
            if target_ts <= start_ts or target_ts > end_ts:
                continue
            if target_ts in existing_ts:
                continue
            missing_indices.append(idx)

        if not missing_indices:
            return existing
        if cache_only:
            logger.info(
                "Cache-only mode: skipping %d missing forecasts for %s (horizon=%dh)",
                len(missing_indices),
                self.config.symbol,
                self.config.prediction_horizon_hours,
            )
            return existing

        new_rows: List[pd.DataFrame] = []
        batch_size = max(1, int(self.config.batch_size))
        for start_idx in range(0, len(missing_indices), batch_size):
            chunk = missing_indices[start_idx : start_idx + batch_size]
            frame = self._generate_forecast_chunk(history, chunk)
            if frame is None or frame.empty:
                continue
            new_rows.append(frame)
            existing_ts.update(pd.to_datetime(frame["timestamp"], utc=True))

        if not new_rows:
            return existing
        combined = (
            pd.concat([existing, *new_rows], ignore_index=True)
            .drop_duplicates(subset=["timestamp", "symbol"], keep="last")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        self.cache.write(self.config.symbol, combined)
        logger.info(
            "Chronos cache updated for %s (horizon=%dh): +%d rows (%d total)",
            self.config.symbol,
            self.config.prediction_horizon_hours,
            sum(len(chunk) for chunk in new_rows),
            len(combined),
        )
        return combined

    # ------------------------------------------------------------------
    def _load_history(self) -> pd.DataFrame:
        path = Path(self.config.data_root) / f"{self.config.symbol.upper()}.csv"
        if not path.exists():
            return pd.DataFrame()
        frame = pd.read_csv(path)
        frame.columns = [col.lower() for col in frame.columns]
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.sort_values("timestamp").reset_index(drop=True)
        return frame

    def _generate_forecast_chunk(self, history: pd.DataFrame, target_indices: Sequence[int]) -> Optional[pd.DataFrame]:
        if not target_indices:
            return pd.DataFrame()
        min_context = max(16, int(self.config.context_hours // 8) or 16)
        records: List[Dict[str, object]] = []
        for ordinal, target_idx in enumerate(target_indices):
            context_end_idx = target_idx
            context_start_idx = max(0, context_end_idx - int(self.config.context_hours))
            context = history.iloc[context_start_idx:context_end_idx].copy()
            if context.empty or len(context) < min_context:
                continue
            target_ts = history.iloc[target_idx]["timestamp"]
            issued_at = context["timestamp"].max()
            if pd.isna(target_ts) or pd.isna(issued_at):
                continue
            records.append(
                {
                    "context": context,
                    "target_ts": target_ts,
                    "issued_at": issued_at,
                    "batch_symbol": self._build_batch_symbol(target_ts, ordinal),
                }
            )
        if not records:
            return pd.DataFrame()

        wrapper = self._ensure_wrapper()
        if wrapper is None:
            rows = [self._heuristic_forecast(rec["context"], rec["target_ts"]) for rec in records]
            return pd.DataFrame(rows)

        contexts = [rec["context"] for rec in records]
        symbols = [rec["batch_symbol"] for rec in records]
        try:
            batches = wrapper.predict_ohlc_batch(  # type: ignore[attr-defined]
                contexts,
                symbols=symbols,
                prediction_length=self.config.prediction_horizon_hours,
                context_length=self.config.context_hours,
                batch_size=self.config.batch_size,
                predict_kwargs=self._predict_kwargs,
            )
        except Exception as exc:
            logger.warning("Chronos forecast failed for %s: %s", self.config.symbol, exc)
            rows = [self._heuristic_forecast(rec["context"], rec["target_ts"]) for rec in records]
            return pd.DataFrame(rows)

        rows: List[Dict[str, object]] = []
        for rec, batch in zip(records, batches):
            row = self._forecast_row_from_batch(batch, rec["target_ts"], rec["issued_at"])
            if row is not None:
                rows.append(row)
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).dropna(subset=["predicted_close_p50", "predicted_high_p50", "predicted_low_p50"])

    def _forecast_row_from_batch(
        self,
        batch: Chronos2PredictionBatch,
        target_ts: pd.Timestamp,
        issued_at: pd.Timestamp,
    ) -> Optional[Dict[str, object]]:
        quantiles: Mapping[float, pd.DataFrame] = batch.quantile_frames
        if not quantiles:
            return None
        return {
            "timestamp": target_ts,
            "symbol": self.config.symbol,
            "issued_at": issued_at,
            "predicted_close_p50": self._extract_quantile(quantiles, 0.5, "close", target_ts),
            "predicted_close_p10": self._extract_quantile(quantiles, 0.1, "close", target_ts),
            "predicted_close_p90": self._extract_quantile(quantiles, 0.9, "close", target_ts),
            "predicted_high_p50": self._extract_quantile(quantiles, 0.5, "high", target_ts),
            "predicted_low_p50": self._extract_quantile(quantiles, 0.5, "low", target_ts),
        }

    def _build_batch_symbol(self, target_ts: pd.Timestamp, ordinal: int) -> str:
        suffix = pd.Timestamp(target_ts).strftime("%Y%m%dT%H%M%SZ")
        return f"{self.config.symbol.upper()}__{suffix}__{ordinal}"

    def _ensure_wrapper(self) -> Optional[object]:
        if self._wrapper is not None:
            return self._wrapper
        if self.wrapper_factory is not None:
            self._wrapper = self.wrapper_factory()
            return self._wrapper
        if Chronos2OHLCWrapper is None:
            if not self._warned_missing:
                logger.warning(
                    "Chronos2 unavailable for %s; using heuristic forecasts. (%s)",
                    self.config.symbol,
                    _CHRONOS_IMPORT_ERROR,
                )
                self._warned_missing = True
            return None
        params = resolve_chronos2_params(self.config.symbol, frequency="hourly")
        quantiles = tuple(params.get("quantile_levels") or self.config.quantile_levels)
        self._predict_kwargs = dict(params.get("predict_kwargs") or {})

        self._wrapper = Chronos2OHLCWrapper.from_pretrained(  # type: ignore[assignment]
            model_id=params.get("model_id", "amazon/chronos-2"),
            device_map=params.get("device_map", "cuda"),
            default_context_length=int(params.get("context_length", self.config.context_hours)),
            default_batch_size=int(params.get("batch_size", self.config.batch_size)),
            quantile_levels=quantiles,
        )
        return self._wrapper

    def _heuristic_forecast(self, context: pd.DataFrame, target_ts: pd.Timestamp) -> Dict[str, object]:
        recent = context.iloc[-1]
        close = float(recent["close"])
        high = float(recent.get("high", close))
        low = float(recent.get("low", close))
        move = float(context["close"].pct_change().dropna().tail(8).mean() or 0.0)
        predicted_close = close * (1 + move)
        return {
            "timestamp": target_ts,
            "symbol": self.config.symbol,
            "issued_at": recent["timestamp"],
            "predicted_close_p50": predicted_close,
            "predicted_close_p10": min(predicted_close, low),
            "predicted_close_p90": max(predicted_close, high),
            "predicted_high_p50": max(high, predicted_close),
            "predicted_low_p50": min(low, predicted_close),
        }

    @staticmethod
    def _extract_quantile(
        quantiles: Mapping[float, pd.DataFrame],
        level: float,
        column: str,
        timestamp: pd.Timestamp,
    ) -> Optional[float]:
        frame = quantiles.get(level)
        if frame is None or column not in frame.columns:
            return None
        if timestamp not in frame.index:
            return None
        value = frame.loc[timestamp, column]
        if pd.isna(value):
            return None
        return float(value)

    @staticmethod
    def _coerce_timestamp(value: Optional[pd.Timestamp | str]) -> Optional[pd.Timestamp]:
        if value is None:
            return None
        ts = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(ts):
            return None
        return ts


def build_forecast_bundle(
    *,
    symbol: str,
    data_root: Path,
    cache_root: Path,
    horizons: Sequence[int],
    context_hours: int,
    quantile_levels: Sequence[float],
    batch_size: int,
    cache_only: bool = False,
) -> pd.DataFrame:
    """Build merged Chronos forecast frame with horizon-specific suffixes."""

    merged: Optional[pd.DataFrame] = None
    for horizon in horizons:
        horizon_dir = cache_root / f"h{int(horizon)}"
        cfg = ForecastConfig(
            symbol=symbol,
            data_root=data_root,
            context_hours=context_hours,
            prediction_horizon_hours=int(horizon),
            quantile_levels=tuple(quantile_levels),
            batch_size=batch_size,
            cache_dir=horizon_dir,
        )
        manager = ChronosForecastManager(cfg)
        forecast = manager.ensure_latest(cache_only=cache_only)
        if forecast.empty:
            continue
        suffix = f"_h{int(horizon)}"
        horizon_frame = forecast.rename(
            columns={
                "predicted_close_p50": f"predicted_close_p50{suffix}",
                "predicted_close_p10": f"predicted_close_p10{suffix}",
                "predicted_close_p90": f"predicted_close_p90{suffix}",
                "predicted_high_p50": f"predicted_high_p50{suffix}",
                "predicted_low_p50": f"predicted_low_p50{suffix}",
            }
        )
        horizon_frame = horizon_frame[[
            "timestamp",
            "symbol",
            f"predicted_close_p50{suffix}",
            f"predicted_high_p50{suffix}",
            f"predicted_low_p50{suffix}",
        ]]
        if merged is None:
            merged = horizon_frame
        else:
            merged = merged.merge(horizon_frame, on=["timestamp", "symbol"], how="inner")
    if merged is None:
        raise RuntimeError(
            f"No Chronos forecasts available for {symbol}. Generate caches first or disable cache_only."
        )
    return merged.sort_values("timestamp").reset_index(drop=True)


__all__ = ["ChronosForecastManager", "ForecastCache", "build_forecast_bundle"]
