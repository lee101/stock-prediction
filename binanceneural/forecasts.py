from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.chronos2_params import resolve_chronos2_params
from src.date_utils import is_nyse_open_on_date
from src.models.chronos2_postprocessing import repair_forecast_ohlc

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
NEW_YORK = ZoneInfo("America/New_York")
_TIME_COVARIATE_COLUMNS = (
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_market_open",
    "session_progress",
)


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
        frame = pd.read_parquet(path)
        if frame.empty:
            return frame
        if "timestamp" in frame.columns:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        if "issued_at" in frame.columns:
            frame["issued_at"] = pd.to_datetime(frame["issued_at"], utc=True, errors="coerce")
        return frame

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
        self._use_multivariate: bool = False
        self._use_cross_learning: bool = False
        self._use_time_covariates: bool = self._resolve_use_time_covariates()
        self._known_future_covariates: tuple[str, ...] = (
            _TIME_COVARIATE_COLUMNS if self._use_time_covariates else ()
        )
        self._resolved_params: Dict[str, object] = {}
        self._warned_missing = False
        self._load_inference_params()

    def ensure_latest(
        self,
        *,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        cache_only: bool = False,
        force_rebuild: bool = False,
    ) -> pd.DataFrame:
        """Ensure forecasts exist for the requested range."""

        history = self._load_history()
        if history.empty:
            raise RuntimeError(f"No hourly history found for {self.config.symbol} at {self.config.data_root}")
        start_ts = self._coerce_timestamp(start) or history["timestamp"].min()
        end_ts = self._coerce_timestamp(end) or history["timestamp"].max()
        existing = self.cache.load(self.config.symbol)
        if force_rebuild and cache_only:
            raise ValueError("force_rebuild=True requires cache_only=False.")
        if force_rebuild and not existing.empty:
            existing = existing.copy()
            if "timestamp" in existing.columns:
                existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True, errors="coerce")
                # Rebuild the same interval that the missing-index loop targets: (start_ts, end_ts].
                keep_mask = (existing["timestamp"] <= start_ts) | (existing["timestamp"] > end_ts)
                existing = existing.loc[keep_mask].reset_index(drop=True)
        existing_ts = set(pd.to_datetime(existing["timestamp"], utc=True)) if not existing.empty else set()

        # Allow short-history symbols (e.g., new Binance markets like BTCU) to produce forecasts.
        # The wrapper trims context_length to the available history, so we only require a smaller
        # minimum context window for generation rather than context_hours full length.
        min_context = max(16, int(self.config.context_hours // 8) or 16)
        min_idx = max(1, min_context)
        if len(history) <= min_idx:
            if existing.empty:
                raise RuntimeError(
                    f"Insufficient hourly history for {self.config.symbol}: "
                    f"need >{min_idx} rows for forecasting but found {len(history)}."
                )
            return existing
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

    @staticmethod
    def _is_invalid_price(value: object) -> bool:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return True
        return (not math.isfinite(out)) or out <= 0.0

    @classmethod
    def _row_is_plausible(
        cls,
        row: Mapping[str, object],
        *,
        last_close: float,
        min_ratio: float = 0.01,
        max_ratio: float = 100.0,
    ) -> bool:
        """Return True when extracted forecast values look numerically safe.

        Chronos2 (especially with pre-augmentation + multi-step prediction) can occasionally
        emit extreme or negative prices. These values are unusable for feature construction
        and can poison backtests/trading. We treat them as invalid and fall back to a
        deterministic heuristic forecast.
        """
        close = row.get("predicted_close_p50")
        high = row.get("predicted_high_p50")
        low = row.get("predicted_low_p50")
        if cls._is_invalid_price(close) or cls._is_invalid_price(high) or cls._is_invalid_price(low):
            return False

        if last_close > 0.0 and math.isfinite(last_close):
            ratio_bounds = (float(min_ratio), float(max_ratio))
            for key in ("predicted_close_p50", "predicted_high_p50", "predicted_low_p50"):
                value = float(row[key])
                ratio = value / last_close
                if ratio < ratio_bounds[0] or ratio > ratio_bounds[1]:
                    return False

        # Sanity: ensure high/low ordering (leave stricter enforcement to consumers).
        if float(row["predicted_low_p50"]) > float(row["predicted_high_p50"]):
            return False
        return True

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
            model_context = context
            future_covariates = None
            if self._use_time_covariates:
                model_context = self._augment_context_with_time_covariates(context)
                future_covariates = self._build_future_covariates(issued_at)
            records.append(
                {
                    "context": context,
                    "model_context": model_context,
                    "target_ts": target_ts,
                    "issued_at": issued_at,
                    "batch_symbol": self._build_batch_symbol(target_ts, ordinal),
                    "future_covariates": future_covariates,
                }
            )
        if not records:
            return pd.DataFrame()

        wrapper = self._ensure_wrapper()
        if wrapper is None:
            rows = [self._heuristic_forecast(rec["context"], rec["target_ts"]) for rec in records]
            return pd.DataFrame(rows)

        contexts = [rec["model_context"] for rec in records]
        symbols = [rec["batch_symbol"] for rec in records]
        future_covariates = [rec["future_covariates"] for rec in records]
        try:
            batches = self._predict_batches(
                wrapper,
                contexts,
                symbols,
                future_covariates=future_covariates,
            )
        except Exception as exc:
            logger.warning("Chronos forecast failed for %s: %s", self.config.symbol, exc)
            rows = [self._heuristic_forecast(rec["context"], rec["target_ts"]) for rec in records]
            return pd.DataFrame(rows)

        rows: List[Dict[str, object]] = []
        invalid_rows = 0
        for rec, batch in zip(records, batches):
            row = self._forecast_row_from_batch(batch, rec["target_ts"], rec["issued_at"])
            context = rec["context"]
            if row is None:
                invalid_rows += 1
                rows.append(self._heuristic_forecast(context, rec["target_ts"]))
                continue

            # Fail safe: reject non-finite / non-positive / wildly out-of-range predictions.
            try:
                last_close = float(context.iloc[-1]["close"])
            except Exception:
                last_close = 0.0

            if not self._row_is_plausible(row, last_close=last_close):
                invalid_rows += 1
                rows.append(self._heuristic_forecast(context, rec["target_ts"]))
                continue

            repaired = repair_forecast_ohlc(
                last_close=last_close,
                close_p50=row.get("predicted_close_p50"),
                close_p10=row.get("predicted_close_p10"),
                close_p90=row.get("predicted_close_p90"),
                high_p50=row.get("predicted_high_p50"),
                low_p50=row.get("predicted_low_p50"),
            )
            row["predicted_close_p10"] = repaired.close_p10
            row["predicted_close_p50"] = repaired.close_p50
            row["predicted_close_p90"] = repaired.close_p90
            row["predicted_high_p50"] = repaired.high_p50
            row["predicted_low_p50"] = repaired.low_p50
            rows.append(row)
        if not rows:
            return pd.DataFrame()
        if invalid_rows:
            logger.warning(
                "Used heuristic fallback for %d/%d forecasts for %s (horizon=%dh) due to invalid Chronos outputs.",
                invalid_rows,
                len(records),
                self.config.symbol,
                int(self.config.prediction_horizon_hours),
            )
        frame = pd.DataFrame(rows)
        return frame.dropna(subset=["predicted_close_p50", "predicted_high_p50", "predicted_low_p50"])

    def _forecast_row_from_batch(
        self,
        batch: Chronos2PredictionBatch,
        target_ts: pd.Timestamp,
        issued_at: pd.Timestamp,
    ) -> Optional[Dict[str, object]]:
        quantiles: Mapping[float, pd.DataFrame] = batch.quantile_frames
        if not quantiles:
            return None
        horizon = max(1, int(self.config.prediction_horizon_hours))
        extract_ts = target_ts + pd.Timedelta(hours=horizon - 1)
        return {
            "timestamp": target_ts,
            "symbol": self.config.symbol,
            "issued_at": issued_at,
            # For horizon>1, shift extraction forward so "_hN" represents N hours ahead from `issued_at`.
            "target_timestamp": extract_ts,
            "horizon_hours": horizon,
            "predicted_close_p50": self._extract_quantile(quantiles, 0.5, "close", extract_ts),
            "predicted_close_p10": self._extract_quantile(quantiles, 0.1, "close", extract_ts),
            "predicted_close_p90": self._extract_quantile(quantiles, 0.9, "close", extract_ts),
            "predicted_high_p50": self._extract_quantile(quantiles, 0.5, "high", extract_ts),
            "predicted_low_p50": self._extract_quantile(quantiles, 0.5, "low", extract_ts),
        }

    def _build_batch_symbol(self, target_ts: pd.Timestamp, ordinal: int) -> str:
        suffix = pd.Timestamp(target_ts).strftime("%Y%m%dT%H%M%SZ")
        return f"{self.config.symbol.upper()}__{suffix}__{ordinal}"

    def _load_inference_params(self) -> None:
        """Load symbol-specific inference defaults from chronos2 params."""
        try:
            params = resolve_chronos2_params(self.config.symbol, frequency="hourly")
        except Exception as exc:
            logger.debug("Failed to resolve Chronos2 params for %s: %s", self.config.symbol, exc)
            return

        self._resolved_params = dict(params)
        resolved_predict_kwargs = dict(params.get("predict_kwargs") or {})
        if resolved_predict_kwargs:
            # Let explicit runtime overrides win over defaults from hyperparams.
            self._predict_kwargs = {**resolved_predict_kwargs, **self._predict_kwargs}
        self._use_multivariate = bool(params.get("use_multivariate", False))
        self._use_cross_learning = bool(params.get("use_cross_learning", False))
        self._apply_runtime_overrides()

    def _resolve_use_time_covariates(self) -> bool:
        if self.config.use_time_covariates is None:
            return False
        return bool(self.config.use_time_covariates)

    @staticmethod
    def _build_time_covariate_frame(timestamps: Sequence[pd.Timestamp]) -> pd.DataFrame:
        ts_index = pd.DatetimeIndex(pd.to_datetime(list(timestamps), utc=True), name="timestamp")
        local = ts_index.tz_convert(NEW_YORK)
        hour_float = local.hour.to_numpy(dtype=float) + (local.minute.to_numpy(dtype=float) / 60.0)
        dow = local.dayofweek.to_numpy(dtype=float)
        hour_angle = 2.0 * np.pi * (hour_float / 24.0)
        dow_angle = 2.0 * np.pi * (dow / 7.0)

        is_open = []
        progress = []
        open_hour = 9.5
        close_hour = 16.0
        session_length = close_hour - open_hour
        for ts_local, hour in zip(local, hour_float):
            open_today = is_nyse_open_on_date(pd.Timestamp(ts_local.date()))
            open_now = bool(open_today and open_hour <= hour <= close_hour)
            is_open.append(1.0 if open_now else 0.0)
            if open_now:
                progress.append(float((hour - open_hour) / max(session_length, 1e-8)))
            else:
                progress.append(0.0)

        return pd.DataFrame(
            {
                "timestamp": ts_index,
                "hour_sin": np.sin(hour_angle),
                "hour_cos": np.cos(hour_angle),
                "dow_sin": np.sin(dow_angle),
                "dow_cos": np.cos(dow_angle),
                "is_market_open": np.asarray(is_open, dtype=float),
                "session_progress": np.asarray(progress, dtype=float),
            }
        )

    def _augment_context_with_time_covariates(self, context: pd.DataFrame) -> pd.DataFrame:
        augmented = context.copy()
        covariates = self._build_time_covariate_frame(context["timestamp"].tolist())
        return augmented.merge(covariates, on="timestamp", how="left")

    def _build_future_covariates(self, issued_at: pd.Timestamp) -> pd.DataFrame:
        horizon = max(1, int(self.config.prediction_horizon_hours))
        future_timestamps = [
            pd.to_datetime(issued_at, utc=True) + pd.Timedelta(hours=step)
            for step in range(1, horizon + 1)
        ]
        return self._build_time_covariate_frame(future_timestamps)

    def _apply_runtime_overrides(self) -> None:
        if self.config.force_multivariate is not None:
            self._use_multivariate = bool(self.config.force_multivariate)
        if self.config.force_cross_learning is not None:
            self._use_cross_learning = bool(self.config.force_cross_learning)
            if self._use_cross_learning:
                self._use_multivariate = True
                self._predict_kwargs["predict_batches_jointly"] = True
            else:
                self._predict_kwargs.pop("predict_batches_jointly", None)
        elif self._use_cross_learning:
            self._predict_kwargs.setdefault("predict_batches_jointly", True)

    def _predict_batches(
        self,
        wrapper: object,
        contexts: Sequence[pd.DataFrame],
        symbols: Sequence[str],
        future_covariates: Sequence[Optional[pd.DataFrame]] | None = None,
    ) -> List[object]:
        """Run forecast inference with configured policy, falling back safely."""
        prediction_length = int(self.config.prediction_horizon_hours)
        context_length = int(self.config.context_hours)
        batch_size = int(self.config.batch_size)
        quantile_levels = list(self.config.quantile_levels)

        if self._use_time_covariates:
            return wrapper.predict_ohlc_batch(  # type: ignore[attr-defined]
                contexts,
                symbols=list(symbols),
                prediction_length=prediction_length,
                context_length=context_length,
                batch_size=batch_size,
                predict_kwargs=self._predict_kwargs,
                future_covariates=list(future_covariates or []),
                known_future_covariates=list(self._known_future_covariates),
            )

        if self._use_multivariate:
            if self._use_cross_learning and len(contexts) > 1 and hasattr(wrapper, "predict_ohlc_joint"):
                try:
                    joint_batches = wrapper.predict_ohlc_joint(  # type: ignore[attr-defined]
                        contexts,
                        symbols=list(symbols),
                        prediction_length=prediction_length,
                        context_length=context_length,
                        quantile_levels=quantile_levels,
                        predict_batches_jointly=True,
                        batch_size=batch_size,
                    )
                    if len(joint_batches) == len(contexts):
                        return list(joint_batches)
                    logger.warning(
                        "Chronos joint prediction for %s returned %d/%d batches; falling back.",
                        self.config.symbol,
                        len(joint_batches),
                        len(contexts),
                    )
                except Exception as exc:
                    logger.warning(
                        "Chronos joint prediction failed for %s, falling back: %s",
                        self.config.symbol,
                        exc,
                    )

            if hasattr(wrapper, "predict_ohlc_multivariate"):
                multi_batches: List[object] = []
                for context, symbol in zip(contexts, symbols):
                    batch = wrapper.predict_ohlc_multivariate(  # type: ignore[attr-defined]
                        context,
                        symbol=symbol,
                        prediction_length=prediction_length,
                        context_length=context_length,
                        quantile_levels=quantile_levels,
                        batch_size=batch_size,
                    )
                    multi_batches.append(batch)
                if len(multi_batches) == len(contexts):
                    return multi_batches

        return wrapper.predict_ohlc_batch(  # type: ignore[attr-defined]
            contexts,
            symbols=list(symbols),
            prediction_length=prediction_length,
            context_length=context_length,
            batch_size=batch_size,
            predict_kwargs=self._predict_kwargs,
        )

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
        params = self._resolved_params or resolve_chronos2_params(self.config.symbol, frequency="hourly")
        quantiles = tuple(params.get("quantile_levels") or self.config.quantile_levels)
        resolved_predict_kwargs = dict(params.get("predict_kwargs") or {})
        if resolved_predict_kwargs:
            self._predict_kwargs = {**resolved_predict_kwargs, **self._predict_kwargs}
        self._use_multivariate = bool(params.get("use_multivariate", self._use_multivariate))
        self._use_cross_learning = bool(params.get("use_cross_learning", self._use_cross_learning))
        self._apply_runtime_overrides()

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
        horizon = max(1, int(self.config.prediction_horizon_hours))
        move = float(context["close"].pct_change().dropna().tail(8).mean() or 0.0)
        # Approximate multi-step horizon by compounding the recent 1h drift.
        step = max(1e-6, 1.0 + move)
        predicted_close = close * (step ** horizon)
        return {
            "timestamp": target_ts,
            "symbol": self.config.symbol,
            "issued_at": recent["timestamp"],
            "target_timestamp": target_ts + pd.Timedelta(hours=horizon - 1),
            "horizon_hours": horizon,
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
    start: Optional[pd.Timestamp | str] = None,
    end: Optional[pd.Timestamp | str] = None,
    force_multivariate: bool | None = None,
    force_cross_learning: bool | None = None,
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
            force_multivariate=force_multivariate,
            force_cross_learning=force_cross_learning,
        )
        manager = ChronosForecastManager(cfg)
        forecast = manager.ensure_latest(start=start, end=end, cache_only=cache_only)
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
        keep_cols = [
            "timestamp",
            "symbol",
            f"predicted_close_p50{suffix}",
            f"predicted_high_p50{suffix}",
            f"predicted_low_p50{suffix}",
        ]
        # Include quantile spread columns when available (for confidence features).
        for extra in (f"predicted_close_p10{suffix}", f"predicted_close_p90{suffix}"):
            if extra in horizon_frame.columns:
                keep_cols.append(extra)
        horizon_frame = horizon_frame[keep_cols]
        if merged is None:
            merged = horizon_frame
        else:
            # Preserve the full union of forecast timestamps so sparse long-horizon
            # caches do not collapse shorter-horizon history.
            merged = merged.merge(horizon_frame, on=["timestamp", "symbol"], how="outer")
    if merged is None:
        raise RuntimeError(
            f"No Chronos forecasts available for {symbol}. Generate caches first or disable cache_only."
        )
    if start is not None or end is not None:
        start_ts = ChronosForecastManager._coerce_timestamp(start) if start is not None else None
        end_ts = ChronosForecastManager._coerce_timestamp(end) if end is not None else None
        merged = merged.copy()
        merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True, errors="coerce")
        merged = merged.dropna(subset=["timestamp"])
        if start_ts is not None:
            merged = merged[merged["timestamp"] > start_ts]
        if end_ts is not None:
            merged = merged[merged["timestamp"] <= end_ts]
    return merged.sort_values("timestamp").reset_index(drop=True)


__all__ = ["ChronosForecastManager", "ForecastCache", "build_forecast_bundle"]
