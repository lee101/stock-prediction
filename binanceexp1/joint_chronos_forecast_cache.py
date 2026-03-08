from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.chronos2_params import resolve_chronos2_params
from src.data_loading_utils import read_csv_tail

try:  # pragma: no cover - optional heavy dependency
    from src.models.chronos2_wrapper import Chronos2OHLCWrapper, Chronos2PredictionBatch
except Exception as exc:  # pragma: no cover
    Chronos2OHLCWrapper = None  # type: ignore[assignment]
    Chronos2PredictionBatch = Any  # type: ignore[assignment]
    _CHRONOS_IMPORT_ERROR = exc
else:  # pragma: no cover
    _CHRONOS_IMPORT_ERROR = None

logger = logging.getLogger(__name__)

_TIME_COVARIATE_COLUMNS = (
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_market_open",
    "session_progress",
)


@dataclass(frozen=True)
class ForecastJob:
    symbol: str
    batch_symbol: str
    target_ts: pd.Timestamp
    issued_at: pd.Timestamp
    horizon: int
    context: pd.DataFrame
    model_context: pd.DataFrame
    future_covariates: pd.DataFrame | None


def _cache_path(cache_root: Path, horizon: int, symbol: str) -> Path:
    safe = str(symbol).upper().replace("/", "_").replace("\\", "_")
    return cache_root / f"h{int(horizon)}" / f"{safe}.parquet"


def _load_cache(cache_root: Path, horizon: int, symbol: str) -> pd.DataFrame:
    path = _cache_path(cache_root, horizon, symbol)
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    if frame.empty:
        return frame
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if "issued_at" in frame.columns:
        frame["issued_at"] = pd.to_datetime(frame["issued_at"], utc=True, errors="coerce")
    if "target_timestamp" in frame.columns:
        frame["target_timestamp"] = pd.to_datetime(frame["target_timestamp"], utc=True, errors="coerce")
    return frame


def _write_cache(cache_root: Path, horizon: int, symbol: str, frame: pd.DataFrame) -> None:
    path = _cache_path(cache_root, horizon, symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.sort_values("timestamp").reset_index(drop=True).to_parquet(path, index=False)


def _load_history(data_root: Path, symbol: str, *, max_history_hours: int | None = None) -> pd.DataFrame:
    path = Path(data_root) / f"{symbol.upper()}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing hourly dataset {path}")
    if max_history_hours is not None and int(max_history_hours) > 0:
        max_rows = int(max_history_hours)
        frame = read_csv_tail(
            path,
            max_rows=max_rows,
            chunksize=200_000,
            low_memory=False,
        )
    else:
        frame = pd.read_csv(path, low_memory=False)
    frame.columns = [col.lower() for col in frame.columns]
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if "symbol" not in frame.columns:
        frame["symbol"] = symbol.upper()
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    if max_history_hours is not None and int(max_history_hours) > 0 and len(frame) > int(max_history_hours):
        frame = frame.iloc[-int(max_history_hours) :].reset_index(drop=True)
    return frame[["timestamp", "symbol", "open", "high", "low", "close", "volume"]]


def _time_covariate_frame(timestamps: Sequence[pd.Timestamp]) -> pd.DataFrame:
    ts_index = pd.DatetimeIndex(pd.to_datetime(list(timestamps), utc=True, errors="coerce"))
    if ts_index.empty:
        return pd.DataFrame(columns=["timestamp", *_TIME_COVARIATE_COLUMNS])
    hour_fraction = ts_index.hour.astype(np.float32) + (ts_index.minute.astype(np.float32) / 60.0)
    hour_angle = 2.0 * math.pi * (hour_fraction / 24.0)
    dow_fraction = ts_index.dayofweek.astype(np.float32)
    dow_angle = 2.0 * math.pi * (dow_fraction / 7.0)
    return pd.DataFrame(
        {
            "timestamp": ts_index,
            "hour_sin": np.sin(hour_angle).astype(np.float32),
            "hour_cos": np.cos(hour_angle).astype(np.float32),
            "dow_sin": np.sin(dow_angle).astype(np.float32),
            "dow_cos": np.cos(dow_angle).astype(np.float32),
            "is_market_open": np.ones(len(ts_index), dtype=np.float32),
            "session_progress": (hour_fraction / 24.0).astype(np.float32),
        }
    )


def _augment_context_with_time_covariates(context: pd.DataFrame) -> pd.DataFrame:
    merged = context.copy()
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True, errors="coerce")
    return merged.merge(_time_covariate_frame(tuple(merged["timestamp"])), on="timestamp", how="left")


def _build_future_covariates(issued_at: pd.Timestamp, horizon: int) -> pd.DataFrame:
    future_index = pd.date_range(
        start=pd.to_datetime(issued_at, utc=True, errors="coerce") + pd.Timedelta(hours=1),
        periods=max(1, int(horizon)),
        freq="h",
        tz="UTC",
    )
    return _time_covariate_frame(tuple(future_index))


def _extract_quantile(
    quantiles: dict[float, pd.DataFrame],
    level: float,
    column: str,
    timestamp: pd.Timestamp,
) -> float | None:
    frame = quantiles.get(level)
    if frame is None or column not in frame.columns or timestamp not in frame.index:
        return None
    value = frame.loc[timestamp, column]
    if pd.isna(value):
        return None
    return float(value)


def _forecast_row_from_batch(
    batch: Chronos2PredictionBatch,
    *,
    symbol: str,
    target_ts: pd.Timestamp,
    issued_at: pd.Timestamp,
    horizon: int,
) -> dict[str, object] | None:
    quantiles = batch.quantile_frames
    if not quantiles:
        return None
    extract_ts = pd.to_datetime(target_ts, utc=True) + pd.Timedelta(hours=int(horizon) - 1)
    return {
        "timestamp": pd.to_datetime(target_ts, utc=True),
        "symbol": str(symbol).upper(),
        "issued_at": pd.to_datetime(issued_at, utc=True),
        "target_timestamp": extract_ts,
        "horizon_hours": int(horizon),
        "predicted_close_p50": _extract_quantile(quantiles, 0.5, "close", extract_ts),
        "predicted_close_p10": _extract_quantile(quantiles, 0.1, "close", extract_ts),
        "predicted_close_p90": _extract_quantile(quantiles, 0.9, "close", extract_ts),
        "predicted_high_p50": _extract_quantile(quantiles, 0.5, "high", extract_ts),
        "predicted_low_p50": _extract_quantile(quantiles, 0.5, "low", extract_ts),
    }


def _is_invalid_price(value: object) -> bool:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return True
    return (not math.isfinite(out)) or out <= 0.0


def _row_is_plausible(row: dict[str, object], *, last_close: float) -> bool:
    close = row.get("predicted_close_p50")
    high = row.get("predicted_high_p50")
    low = row.get("predicted_low_p50")
    if _is_invalid_price(close) or _is_invalid_price(high) or _is_invalid_price(low):
        return False
    if last_close > 0.0 and math.isfinite(last_close):
        for key in ("predicted_close_p50", "predicted_high_p50", "predicted_low_p50"):
            value = float(row[key])
            ratio = value / last_close
            if ratio < 0.01 or ratio > 100.0:
                return False
    if float(row["predicted_low_p50"]) > float(row["predicted_high_p50"]):
        return False
    return True


def _heuristic_forecast(context: pd.DataFrame, *, symbol: str, target_ts: pd.Timestamp, horizon: int) -> dict[str, object]:
    recent = context.iloc[-1]
    close = float(recent["close"])
    high = float(recent.get("high", close))
    low = float(recent.get("low", close))
    move = float(context["close"].pct_change().dropna().tail(8).mean() or 0.0)
    step = max(1e-6, 1.0 + move)
    predicted_close = close * (step ** max(1, int(horizon)))
    return {
        "timestamp": pd.to_datetime(target_ts, utc=True),
        "symbol": str(symbol).upper(),
        "issued_at": pd.to_datetime(recent["timestamp"], utc=True),
        "target_timestamp": pd.to_datetime(target_ts, utc=True) + pd.Timedelta(hours=int(horizon) - 1),
        "horizon_hours": int(horizon),
        "predicted_close_p50": predicted_close,
        "predicted_close_p10": min(predicted_close, low),
        "predicted_close_p90": max(predicted_close, high),
        "predicted_high_p50": max(high, predicted_close),
        "predicted_low_p50": min(low, predicted_close),
    }


def _build_jobs_for_symbol(
    *,
    symbol: str,
    history: pd.DataFrame,
    horizon: int,
    context_hours: int,
    existing_timestamps: set[pd.Timestamp],
    use_time_covariates: bool,
) -> list[ForecastJob]:
    jobs: list[ForecastJob] = []
    min_context = max(16, int(context_hours // 8) or 16)
    for idx in range(min_context, len(history)):
        target_ts = pd.to_datetime(history.iloc[idx]["timestamp"], utc=True)
        if target_ts in existing_timestamps:
            continue
        context_start_idx = max(0, idx - int(context_hours))
        context = history.iloc[context_start_idx:idx].copy()
        if context.empty or len(context) < min_context:
            continue
        issued_at = pd.to_datetime(context["timestamp"].max(), utc=True)
        batch_symbol = f"{symbol.upper()}__{target_ts.strftime('%Y%m%dT%H%M%SZ')}__{idx}"
        if use_time_covariates:
            model_context = _augment_context_with_time_covariates(context)
            future_covariates = _build_future_covariates(issued_at, horizon)
        else:
            model_context = context
            future_covariates = None
        jobs.append(
            ForecastJob(
                symbol=symbol.upper(),
                batch_symbol=batch_symbol,
                target_ts=target_ts,
                issued_at=issued_at,
                horizon=int(horizon),
                context=context,
                model_context=model_context,
                future_covariates=future_covariates,
            )
        )
    return jobs


def _build_wrapper(*, anchor_symbol: str, context_hours: int, batch_size: int) -> Chronos2OHLCWrapper:
    if Chronos2OHLCWrapper is None:  # pragma: no cover - optional dependency
        raise RuntimeError(f"Chronos2 unavailable: {_CHRONOS_IMPORT_ERROR}")
    params = resolve_chronos2_params(anchor_symbol, frequency="hourly")
    quantiles = tuple(params.get("quantile_levels") or (0.1, 0.5, 0.9))
    return Chronos2OHLCWrapper.from_pretrained(
        model_id=str(params.get("model_id", "amazon/chronos-2")),
        device_map=params.get("device_map", "cuda"),
        default_context_length=int(context_hours),
        default_batch_size=int(batch_size),
        quantile_levels=quantiles,
    )


def build_joint_forecast_cache(
    *,
    symbols: Sequence[str],
    data_root: Path,
    cache_root: Path,
    horizons: Sequence[int],
    context_hours: int,
    batch_size: int = 128,
    max_history_hours: int | None = None,
    use_cross_learning: bool = True,
    use_time_covariates: bool = False,
    force_rebuild: bool = False,
) -> dict[str, dict[str, int]]:
    cleaned_symbols = [str(symbol).upper() for symbol in symbols if str(symbol).strip()]
    if not cleaned_symbols:
        raise ValueError("At least one symbol is required.")
    if not horizons:
        raise ValueError("At least one horizon is required.")

    histories = {
        symbol: _load_history(Path(data_root), symbol, max_history_hours=max_history_hours)
        for symbol in cleaned_symbols
    }
    wrapper = _build_wrapper(
        anchor_symbol=cleaned_symbols[0],
        context_hours=int(context_hours),
        batch_size=int(batch_size),
    )
    summary: dict[str, dict[str, int]] = {}
    known_future_covariates = list(_TIME_COVARIATE_COLUMNS) if use_time_covariates else None
    predict_kwargs = {"predict_batches_jointly": bool(use_cross_learning)}

    for horizon in [int(value) for value in horizons]:
        jobs: list[ForecastJob] = []
        existing_by_symbol: dict[str, pd.DataFrame] = {}
        for symbol in cleaned_symbols:
            existing = pd.DataFrame() if force_rebuild else _load_cache(cache_root, horizon, symbol)
            existing_by_symbol[symbol] = existing
            existing_timestamps = set(pd.to_datetime(existing["timestamp"], utc=True)) if not existing.empty else set()
            jobs.extend(
                _build_jobs_for_symbol(
                    symbol=symbol,
                    history=histories[symbol],
                    horizon=horizon,
                    context_hours=int(context_hours),
                    existing_timestamps=existing_timestamps,
                    use_time_covariates=use_time_covariates,
                )
            )

        jobs.sort(key=lambda item: (item.target_ts, item.symbol, item.issued_at))
        generated_rows: dict[str, list[dict[str, object]]] = {symbol: [] for symbol in cleaned_symbols}

        for start_idx in range(0, len(jobs), max(1, int(batch_size))):
            chunk = jobs[start_idx : start_idx + max(1, int(batch_size))]
            contexts = [job.model_context for job in chunk]
            batch_symbols = [job.batch_symbol for job in chunk]
            future_covariates = [job.future_covariates for job in chunk] if use_time_covariates else None
            try:
                batches = wrapper.predict_ohlc_batch(
                    contexts,
                    symbols=batch_symbols,
                    prediction_length=int(horizon),
                    context_length=int(context_hours),
                    known_future_covariates=known_future_covariates,
                    future_covariates=future_covariates,
                    batch_size=max(1, int(batch_size)),
                    predict_kwargs=predict_kwargs,
                )
            except Exception as exc:
                logger.warning(
                    "Joint Chronos batch failed for horizon=%dh (%d jobs); falling back to per-job inference: %s",
                    horizon,
                    len(chunk),
                    exc,
                )
                batches = []
                for job in chunk:
                    try:
                        batch = wrapper.predict_ohlc(
                            job.model_context,
                            symbol=job.batch_symbol,
                            prediction_length=int(job.horizon),
                            context_length=int(context_hours),
                            known_future_covariates=known_future_covariates,
                            future_covariates=job.future_covariates,
                            batch_size=max(1, int(batch_size)),
                            predict_kwargs={},
                        )
                    except Exception:
                        batch = None
                    batches.append(batch)

            for job, batch in zip(chunk, batches):
                row = None
                if batch is not None:
                    row = _forecast_row_from_batch(
                        batch,
                        symbol=job.symbol,
                        target_ts=job.target_ts,
                        issued_at=job.issued_at,
                        horizon=job.horizon,
                    )
                if row is None or not _row_is_plausible(row, last_close=float(job.context["close"].iloc[-1])):
                    row = _heuristic_forecast(job.context, symbol=job.symbol, target_ts=job.target_ts, horizon=job.horizon)
                close = float(row["predicted_close_p50"])
                high = float(row["predicted_high_p50"])
                low = float(row["predicted_low_p50"])
                row["predicted_high_p50"] = max(high, close, low)
                row["predicted_low_p50"] = min(low, close, high)
                generated_rows[job.symbol].append(row)

        horizon_summary: dict[str, int] = {}
        for symbol in cleaned_symbols:
            existing = existing_by_symbol[symbol]
            new_frame = pd.DataFrame(generated_rows[symbol])
            if new_frame.empty:
                combined = existing
            elif existing.empty:
                combined = new_frame
            else:
                combined = (
                    pd.concat([existing, new_frame], ignore_index=True)
                    .drop_duplicates(subset=["timestamp", "symbol"], keep="last")
                    .sort_values("timestamp")
                    .reset_index(drop=True)
                )
            if not combined.empty:
                _write_cache(cache_root, horizon, symbol, combined)
            horizon_summary[symbol] = int(len(new_frame))
        summary[f"h{horizon}"] = horizon_summary

    return summary


__all__ = ["build_joint_forecast_cache"]
