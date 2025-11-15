from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional

import pandas as pd

from src.chronos2_params import resolve_chronos2_params

try:  # pragma: no cover - optional heavyweight dependency
    from src.models.chronos2_wrapper import Chronos2OHLCWrapper
except Exception as exc:  # pragma: no cover - surfaced in tests when chronos missing
    Chronos2OHLCWrapper = None  # type: ignore
    _CHRONOS_IMPORT_ERROR = exc
else:  # pragma: no cover
    _CHRONOS_IMPORT_ERROR = None

from .config import ForecastConfig

logger = logging.getLogger(__name__)


class ForecastCache:
    """Simple parquet-backed cache for hourly Chronos forecasts."""

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
class DailyChronosForecastManager:
    """Generate and cache 1-day-ahead hourly Chronos forecasts for LINKUSD."""

    config: ForecastConfig
    wrapper_factory: Optional[Callable[[], object]] = None

    def __post_init__(self) -> None:
        self.cache = ForecastCache(self.config.cache_dir)
        self._wrapper: Optional[object] = None
        self._predict_kwargs: Dict[str, object] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def ensure_latest(
        self,
        *,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Ensure hourly 1-step-ahead forecasts exist for the requested range."""

        history = self._load_history()
        if history.empty:
            raise RuntimeError(f"No hourly history found for {self.config.symbol} at {self.config.data_root}")
        start_ts = self._coerce_timestamp(start) or history["timestamp"].min()
        end_ts = self._coerce_timestamp(end) or history["timestamp"].max()
        existing = self.cache.load(self.config.symbol)
        existing_ts = set(pd.to_datetime(existing["timestamp"], utc=True)) if not existing.empty else set()
        new_rows: List[pd.DataFrame] = []
        min_idx = self.config.context_hours
        for idx in range(min_idx, len(history)):
            target_ts = history.iloc[idx]["timestamp"]
            if target_ts <= start_ts or target_ts > end_ts:
                continue
            if target_ts in existing_ts:
                continue
            row = self._generate_hour_forecast(history, idx)
            if row is None or row.empty:
                continue
            new_rows.append(row)
            existing_ts.update(pd.to_datetime(row["timestamp"], utc=True))
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
            "Chronos hourly forecast cache updated for %s: +%d rows (%d total)",
            self.config.symbol,
            sum(len(chunk) for chunk in new_rows),
            len(combined),
        )
        return combined

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _load_history(self) -> pd.DataFrame:
        data_root = getattr(self.config, "data_root", Path("trainingdatahourly") / "crypto")
        path = Path(data_root) / f"{self.config.symbol.upper()}.csv"
        if not path.exists():
            return pd.DataFrame()
        frame = pd.read_csv(path)
        frame.columns = [col.lower() for col in frame.columns]
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.sort_values("timestamp").reset_index(drop=True)
        return frame

    def _generate_hour_forecast(self, history: pd.DataFrame, target_idx: int) -> Optional[pd.DataFrame]:
        context_end_idx = target_idx
        context_start_idx = max(0, context_end_idx - self.config.context_hours)
        context = history.iloc[context_start_idx:context_end_idx].copy()
        if context.empty or len(context) < max(16, self.config.context_hours // 8):
            return None
        target_ts = history.iloc[target_idx]["timestamp"]
        wrapper = self._ensure_wrapper()
        if wrapper is None:
            heur = self._heuristic_forecast(context, target_ts)
            return pd.DataFrame([heur])
        try:
            batch = wrapper.predict_ohlc(  # type: ignore[attr-defined]
                context,
                symbol=self.config.symbol,
                prediction_length=self.config.prediction_horizon_hours,
                context_length=min(len(context), self.config.context_hours),
                batch_size=self.config.batch_size,
                predict_kwargs=self._predict_kwargs,
            )
        except Exception as exc:
            logger.warning("Chronos forecast failed for %s @ %s: %s", self.config.symbol, target_ts, exc)
            heur = self._heuristic_forecast(context, target_ts)
            return pd.DataFrame([heur])

        issued_at = context["timestamp"].max()
        quantiles: Mapping[float, pd.DataFrame] = batch.quantile_frames  # type: ignore[attr-defined]
        return pd.DataFrame([
            {
                "timestamp": target_ts,
                "symbol": self.config.symbol,
                "issued_at": issued_at,
                "predicted_close_p50": self._extract_quantile(quantiles, 0.5, "close", target_ts),
                "predicted_close_p10": self._extract_quantile(quantiles, 0.1, "close", target_ts),
                "predicted_close_p90": self._extract_quantile(quantiles, 0.9, "close", target_ts),
                "predicted_high_p50": self._extract_quantile(quantiles, 0.5, "high", target_ts),
                "predicted_low_p50": self._extract_quantile(quantiles, 0.5, "low", target_ts),
            }
        ]).dropna(subset=["predicted_close_p50", "predicted_high_p50", "predicted_low_p50"])

    def _ensure_wrapper(self) -> Optional[object]:
        if self._wrapper is not None:
            return self._wrapper
        if self.wrapper_factory is not None:
            self._wrapper = self.wrapper_factory()
            return self._wrapper
        if Chronos2OHLCWrapper is None:
            raise RuntimeError("Chronos2 dependencies missing; install chronos>=2 or provide wrapper_factory") from _CHRONOS_IMPORT_ERROR
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
        ts = timestamp
        if ts not in frame.index:
            return None
        value = frame.loc[ts, column]
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
