from __future__ import annotations

import importlib.util
import os
import time
from contextlib import suppress
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .logging_utils import logger
from .state import get_state
from src.chronos2_params import resolve_chronos2_params
from src.models.chronos2_wrapper import _default_preaug_dirs
from src.preaug import PreAugmentationChoice, PreAugmentationSelector

_REAL_BACKTEST_MODULE = None
_REAL_BACKTEST_ERROR: Optional[Exception] = None
_DEFAULT_NUM_SIMULATIONS = int(os.getenv("MARKETSIM_NUM_SIMULATIONS", "20"))
_PCTDIFF_MAX_RETURN = float(os.getenv("PCTDIFF_MAX_DAILY_RETURN", "0.10"))
_SKIP_REAL_IMPORT = os.getenv("MARKETSIM_SKIP_REAL_IMPORT", "0").lower() in {"1", "true", "yes", "on"}
_LOOKAHEAD_ENV_KEY = "MARKETSIM_FORECAST_LOOKAHEAD"

_PREAUG_SELECTOR: Optional[PreAugmentationSelector] = None
_PREAUG_DIRS: Tuple[Path, ...] = ()
_PREAUG_CACHE: Dict[str, Optional[PreAugmentationChoice]] = {}

_REAL_BACKTEST_PATH = Path(__file__).resolve().parent.parent / "backtest_test3_inline.py"
if _REAL_BACKTEST_PATH.exists() and not _SKIP_REAL_IMPORT:
    try:  # pragma: no cover - integration with heavy forecasting stack
        spec = importlib.util.spec_from_file_location(
            "_marketsim_real_backtest", str(_REAL_BACKTEST_PATH)
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore[arg-type]
            _REAL_BACKTEST_MODULE = module
    except Exception as exc:  # pragma: no cover - exercised when dependencies missing
        _REAL_BACKTEST_ERROR = exc
        logger.warning(
            "[sim] Failed to load real backtest_test3_inline module (%s); "
            "falling back to lightweight simulator analytics.",
            exc,
        )
elif _SKIP_REAL_IMPORT:
    logger.info("[sim] Skipping real backtest_test3_inline import (mock analytics enabled).")


def _reset_preaug_cache() -> None:
    global _PREAUG_SELECTOR, _PREAUG_DIRS, _PREAUG_CACHE
    _PREAUG_SELECTOR = None
    _PREAUG_DIRS = ()
    _PREAUG_CACHE.clear()


def _preaug_dirs() -> Tuple[Path, ...]:
    try:
        freq = os.getenv("CHRONOS2_FREQUENCY")
        return tuple(_default_preaug_dirs(freq))
    except Exception as exc:  # pragma: no cover - defensive guard for path issues
        logger.debug("[sim] Unable to resolve pre-augmentation directories: %s", exc)
        return ()


def _get_preaug_selector() -> Optional[PreAugmentationSelector]:
    global _PREAUG_SELECTOR, _PREAUG_DIRS
    dirs = _preaug_dirs()
    if _PREAUG_SELECTOR is not None and dirs == _PREAUG_DIRS:
        return _PREAUG_SELECTOR
    if not dirs:
        _reset_preaug_cache()
        return None
    try:
        selector = PreAugmentationSelector(dirs)
    except Exception as exc:  # pragma: no cover - invalid configs shouldn't break simulator
        logger.debug("[sim] Failed to build pre-augmentation selector: %s", exc)
        _reset_preaug_cache()
        return None
    _PREAUG_SELECTOR = selector
    _PREAUG_DIRS = dirs
    _PREAUG_CACHE.clear()
    return _PREAUG_SELECTOR


def _resolve_preaug_choice(symbol: str) -> Optional[PreAugmentationChoice]:
    key = symbol.upper()
    if key in _PREAUG_CACHE:
        return _PREAUG_CACHE[key]
    selector = _get_preaug_selector()
    if selector is None:
        _PREAUG_CACHE[key] = None
        return None
    choice = selector.get_choice(key)
    _PREAUG_CACHE[key] = choice
    return choice


def _assign_metadata_column(result: pd.DataFrame, column: str, value) -> None:
    if result.empty or value is None:
        return
    length = len(result)
    if isinstance(value, dict):
        result[column] = [dict(value) for _ in range(length)]
    elif isinstance(value, (list, tuple)):
        seq = list(value)
        result[column] = [seq.copy() for _ in range(length)]
    else:
        result[column] = value


def _annotate_chronos2_metadata(result: pd.DataFrame, symbol: str) -> None:
    """Attach Chronos2 pre-aug and hyperparameter metadata to fallback outputs."""

    if result.empty:
        return

    choice = _resolve_preaug_choice(symbol)
    if choice is not None:
        result["chronos2_preaug_strategy"] = choice.strategy
        source_path = getattr(choice, "source_path", None)
        if source_path:
            result["chronos2_preaug_source"] = str(source_path)

    try:
        params = resolve_chronos2_params(symbol, frequency=os.getenv("CHRONOS2_FREQUENCY"))
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.debug("[sim] Failed to resolve Chronos2 params for %s: %s", symbol, exc)
        return

    if not params:
        return

    _assign_metadata_column(result, "chronos2_context_length", params.get("context_length"))
    _assign_metadata_column(result, "chronos2_batch_size", params.get("batch_size"))
    _assign_metadata_column(result, "chronos2_prediction_length", params.get("prediction_length"))
    quantiles = params.get("quantile_levels") or (0.1, 0.5, 0.9)
    _assign_metadata_column(result, "chronos2_quantile_levels", list(quantiles))
    predict_kwargs = dict(params.get("predict_kwargs") or {})
    _assign_metadata_column(result, "chronos2_predict_kwargs", predict_kwargs)
    config_path = params.get("_config_path")
    if config_path:
        _assign_metadata_column(result, "chronos2_hparams_config_path", config_path)
    model_id = params.get("model_id")
    if model_id:
        _assign_metadata_column(result, "chronos2_model_id", model_id)


def _reset_chronos2_metadata_cache() -> None:
    """Expose cache reset for unit tests."""

    _reset_preaug_cache()


def _resolve_lookahead(default: int = 1) -> int:
    raw = os.getenv(_LOOKAHEAD_ENV_KEY)
    if not raw:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return max(1, value)


def _window_from_state(symbol: str, num_simulations: int) -> Optional[pd.DataFrame]:
    state = None
    with suppress(RuntimeError):
        state = get_state()
    if state is None:
        return None

    series = state.prices.get(symbol)
    if series is None:
        return None

    frame = series.frame
    end_idx = min(series.cursor + num_simulations, len(frame))
    start_idx = max(0, end_idx - num_simulations)
    window = frame.iloc[start_idx:end_idx].copy()
    if window.empty:
        return None
    return window


def _load_live_price_history(symbol: str, num_simulations: int) -> Optional[pd.DataFrame]:
    try:
        from alpaca.data import StockHistoricalDataClient
        from data_curate_daily import (
            download_exchange_historical_data,
            download_exchange_latest_data,
        )
        from env_real import ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD
        from src.stock_utils import remap_symbols
    except Exception as exc:  # pragma: no cover - import guards runtime environments
        logger.warning(
            "[sim] Unable to import live data interfaces for fallback analytics (%s).",
            exc,
        )
        return None

    client = StockHistoricalDataClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD)

    try:
        history = download_exchange_historical_data(client, symbol)
    except Exception as exc:  # pragma: no cover - network/API issues
        logger.warning("[sim] Failed fetching historical data for %s: %s", symbol, exc)
        return None

    if history is None or history.empty:
        logger.warning("[sim] Historical data empty for %s", symbol)
        return None

    history_frame = history.reset_index()
    candidates = {symbol}
    with suppress(Exception):
        mapped = remap_symbols(symbol)
        candidates.add(mapped)
    if "symbol" in history_frame.columns:
        filtered = history_frame[history_frame["symbol"].isin(candidates)]
        if not filtered.empty:
            history_frame = filtered
    if "symbol" in history_frame.columns:
        history_frame = history_frame.drop(columns=["symbol"])

    rename_map = {}
    for column in history_frame.columns:
        lowered = column.lower()
        if lowered in {"open", "high", "low", "close"}:
            rename_map[column] = lowered.capitalize()
        elif lowered == "timestamp":
            rename_map[column] = "timestamp"
    history_frame = history_frame.rename(columns=rename_map)
    if "timestamp" not in history_frame.columns:
        if history_frame.index.name == "timestamp":
            history_frame = history_frame.reset_index()
        else:
            logger.warning("[sim] Historical frame for %s missing timestamp column", symbol)
            return None

    history_frame["timestamp"] = pd.to_datetime(
        history_frame["timestamp"], utc=True, errors="coerce"
    )
    history_frame = history_frame.dropna(subset=["timestamp"])

    latest_frame = None
    try:
        latest = download_exchange_latest_data(client, symbol)
        if latest is not None and not latest.empty:
            latest_frame = latest.reset_index()
    except Exception as exc:  # pragma: no cover - treat missing latest data as non-fatal
        logger.debug("[sim] Failed fetching latest bars for %s: %s", symbol, exc)

    if latest_frame is not None:
        if "symbol" in latest_frame.columns:
            latest_filtered = latest_frame[latest_frame["symbol"].isin(candidates)]
            if not latest_filtered.empty:
                latest_frame = latest_filtered
        if "symbol" in latest_frame.columns:
            latest_frame = latest_frame.drop(columns=["symbol"])
        rename_map = {}
        for column in latest_frame.columns:
            lowered = column.lower()
            if lowered in {"open", "high", "low", "close"}:
                rename_map[column] = lowered.capitalize()
            elif lowered == "timestamp":
                rename_map[column] = "timestamp"
        latest_frame = latest_frame.rename(columns=rename_map)
        latest_frame["timestamp"] = pd.to_datetime(
            latest_frame["timestamp"], utc=True, errors="coerce"
        )
        latest_frame = latest_frame.dropna(subset=["timestamp"])

        combined = pd.concat([history_frame, latest_frame], ignore_index=True)
    else:
        combined = history_frame

    combined = combined.sort_values("timestamp")
    combined = combined.drop_duplicates(subset=["timestamp"], keep="last")

    if combined.empty:
        logger.warning("[sim] Combined price history empty for %s", symbol)
        return None

    window = combined.tail(num_simulations).copy()
    if window.empty:
        return None

    window.reset_index(drop=True, inplace=True)
    return window


def _fallback_backtest(symbol: str, num_simulations: int | None = None) -> pd.DataFrame:
    num_simulations = num_simulations or _DEFAULT_NUM_SIMULATIONS
    window = _window_from_state(symbol, num_simulations)
    if window is None:
        window = _load_live_price_history(symbol, num_simulations)

    if window is None or window.empty:
        raise ValueError(f"No data available for fallback analytics on {symbol}")

    window = window.copy()
    timestamp_series = None
    price_groups = {"open": [], "high": [], "low": [], "close": []}
    other_columns: Dict[str, pd.Series] = {}

    for idx, column in enumerate(window.columns):
        key = str(column).strip().lower()
        series = window.iloc[:, idx]
        if key == "timestamp":
            timestamp_series = series if timestamp_series is None else timestamp_series.combine_first(series)
        elif key in price_groups:
            price_groups[key].append(series)
        elif key == "volume":
            other_columns["Volume"] = series

    consolidated = {}
    if timestamp_series is not None:
        consolidated["timestamp"] = timestamp_series

    for key, series_list in price_groups.items():
        if not series_list:
            continue
        combined = series_list[0].copy()
        for extra in series_list[1:]:
            combined = combined.combine_first(extra)
        consolidated[key.capitalize()] = combined

    consolidated.update(other_columns)

    if consolidated:
        window = pd.DataFrame(consolidated)
    else:
        rename_map = {}
        for column in window.columns:
            key = str(column).strip().lower()
            if key == "timestamp":
                rename_map[column] = "timestamp"
            elif key == "open":
                rename_map[column] = "Open"
            elif key == "high":
                rename_map[column] = "High"
            elif key == "low":
                rename_map[column] = "Low"
            elif key == "close":
                rename_map[column] = "Close"
        if rename_map:
            window.rename(columns=rename_map, inplace=True)
        if window.columns.duplicated().any():
            window = window.loc[:, ~window.columns.duplicated(keep="first")]

    required_cols = {"Close", "High", "Low"}
    missing = required_cols.difference(window.columns)
    if missing:
        raise ValueError(
            f"Fallback data for {symbol} missing required columns: {sorted(missing)}"
        )

    # Ensure numeric dtypes for calculations
    for column in required_cols:
        window[column] = pd.to_numeric(window[column], errors="coerce")
    window = window.dropna(subset=list(required_cols))
    if window.empty:
        raise ValueError(f"Fallback price window contained no numeric data for {symbol}")

    if len(window) < num_simulations:
        logger.warning(
            "[sim] Only %d rows of price history available for %s (requested %d).",
            len(window),
            symbol,
            num_simulations,
        )

    window["close_return"] = window["Close"].pct_change().fillna(0.0)
    window["high_return"] = window["High"].pct_change().fillna(0.0)
    window["low_return"] = window["Low"].pct_change().fillna(0.0)

    lookahead = _resolve_lookahead()

    def _future_last(series: pd.Series, steps: int) -> pd.Series:
        values = series.to_numpy(copy=False)
        n = len(values)
        result = np.empty(n, dtype=float)
        for idx in range(n):
            future_idx = min(n - 1, idx + steps)
            result[idx] = float(values[future_idx])
        return pd.Series(result, index=series.index, dtype=float)

    def _future_extreme(series: pd.Series, steps: int, op) -> pd.Series:
        values = series.to_numpy(copy=False)
        n = len(values)
        result = np.empty(n, dtype=float)
        for idx in range(n):
            start = min(n, idx + 1)
            end = min(n, idx + steps + 1)
            if start >= end:
                fallback_val = float(values[min(n - 1, idx)])
                result[idx] = fallback_val
                continue
            window_vals = values[start:end]
            if window_vals.size == 0:
                result[idx] = float(values[min(n - 1, idx)])
                continue
            agg = op(window_vals)
            if np.isnan(agg):
                result[idx] = float(values[min(n - 1, idx)])
            else:
                result[idx] = float(agg)
        return pd.Series(result, index=series.index, dtype=float)

    predicted_close = _future_last(window["Close"], lookahead)
    predicted_high = _future_extreme(window["High"], lookahead, np.nanmax)
    predicted_low = _future_extreme(window["Low"], lookahead, np.nanmin)

    simple = predicted_close.pct_change().fillna(0.0)
    all_signals = (predicted_close + predicted_high + predicted_low) / 3.0
    all_signals = all_signals.pct_change().fillna(0.0)
    takeprofit = (predicted_high - window["Close"]) / window["Close"]
    highlow = (predicted_high - predicted_low) / window["Close"]
    up_edge = (predicted_high - window["Close"]) / window["Close"]
    down_edge = (window["Close"] - predicted_low) / window["Close"]
    maxdiff = up_edge.where(up_edge >= down_edge, -down_edge)

    logger.debug(
        "[sim] Using fallback backtest for %s with %d rows (real module unavailable).",
        symbol,
        len(window),
    )

    def _sharpe(series: pd.Series) -> float:
        if series.empty:
            return 0.0
        std = float(series.std(ddof=1))
        if not np.isfinite(std) or std == 0.0:
            return 0.0
        mean = float(series.mean())
        if not np.isfinite(mean):
            return 0.0
        return mean / std

    def _rev(series: pd.Series) -> pd.Series:
        return series.iloc[::-1].reset_index(drop=True)

    close_series = window["Close"].astype(float)
    high_series = window["High"].astype(float)
    low_series = window["Low"].astype(float)
    if "Volume" in window.columns:
        volume_series = pd.to_numeric(window["Volume"], errors="coerce")
    else:
        volume_series = pd.Series(np.nan, index=window.index, dtype=float)
    if volume_series.isna().all():
        volume_series = pd.Series(1_000.0, index=window.index, dtype=float)
    else:
        volume_series = volume_series.fillna(volume_series.median())

    raw_expected_move_pct = (predicted_close - close_series) / close_series.replace(0.0, np.nan)
    adjusted_move_pct = raw_expected_move_pct.fillna(0.0)
    default_move = window["close_return"].fillna(0.0)
    adjusted_move_pct = adjusted_move_pct.where(adjusted_move_pct.abs() > 1e-6, default_move)
    kronos_expected = adjusted_move_pct.ewm(span=5, adjust=False, min_periods=1).mean()

    returns = close_series.pct_change().fillna(0.0)
    realized_vol = returns.rolling(window=20, min_periods=5).std().fillna(returns.std())
    dollar_vol = (volume_series * close_series).rolling(window=20, min_periods=1).mean()
    atr_pct = (high_series - low_series).rolling(window=14, min_periods=1).mean() / close_series.replace(0.0, np.nan)
    spread_bps = (
        (high_series - low_series) / close_series.replace(0.0, np.nan)
    ).rolling(window=5, min_periods=1).mean() * 10_000

    unprofit_series = -simple.abs()

    simple_sharpe_val = _sharpe(simple)
    all_signals_sharpe_val = _sharpe(all_signals)
    takeprofit_sharpe_val = _sharpe(takeprofit)
    highlow_sharpe_val = _sharpe(highlow)
    maxdiff_sharpe_val = _sharpe(maxdiff)
    buy_hold_sharpe_val = _sharpe(returns)
    unprofit_sharpe_val = _sharpe(unprofit_series)

    result = pd.DataFrame(
        {
            "close": close_series,
            "predicted_close": predicted_close,
            "predicted_high": predicted_high,
            "predicted_low": predicted_low,
            "simple_strategy_return": simple,
            "all_signals_strategy_return": all_signals,
            "entry_takeprofit_return": takeprofit,
            "highlow_return": highlow,
            "maxdiff_return": maxdiff,
        }
    )
    if "timestamp" in window.columns:
        result["timestamp"] = window["timestamp"]

    result = result.iloc[::-1].reset_index(drop=True)

    result["toto_expected_move_pct"] = _rev(adjusted_move_pct).fillna(0.0)
    result["kronos_expected_move_pct"] = _rev(kronos_expected).fillna(0.0)
    result["raw_expected_move_pct"] = _rev(raw_expected_move_pct).fillna(0.0)
    result["calibrated_expected_move_pct"] = result["kronos_expected_move_pct"]
    result["calibration_slope"] = 1.0
    result["calibration_intercept"] = 0.0
    result["close_prediction_source"] = "SIM_FALLBACK"

    result["realized_volatility_pct"] = _rev(realized_vol).abs().fillna(0.0)
    result["dollar_vol_20d"] = _rev(dollar_vol).fillna(0.0)
    result["atr_pct_14"] = _rev(atr_pct).fillna(0.0)
    result["spread_bps_estimate"] = _rev(spread_bps).fillna(20.0)

    result["buy_hold_return"] = _rev(returns).fillna(0.0)
    result["buy_hold_sharpe"] = buy_hold_sharpe_val
    result["buy_hold_finalday"] = float(returns.iloc[-1]) if not returns.empty else 0.0

    result["simple_strategy_sharpe"] = simple_sharpe_val
    result["simple_strategy_finalday"] = float(simple.iloc[-1]) if not simple.empty else 0.0
    result["all_signals_strategy_sharpe"] = all_signals_sharpe_val
    result["all_signals_strategy_finalday"] = float(all_signals.iloc[-1]) if not all_signals.empty else 0.0

    result["entry_takeprofit_sharpe"] = takeprofit_sharpe_val
    result["entry_takeprofit_finalday"] = float(takeprofit.iloc[-1]) if not takeprofit.empty else 0.0
    result["entry_takeprofit_turnover"] = float(takeprofit.abs().mean()) if not takeprofit.empty else 0.0

    result["highlow_sharpe"] = highlow_sharpe_val
    result["highlow_finalday_return"] = float(highlow.iloc[-1]) if not highlow.empty else 0.0
    result["highlow_turnover"] = float(highlow.abs().mean()) if not highlow.empty else 0.0

    result["maxdiff_sharpe"] = maxdiff_sharpe_val
    result["maxdiff_finalday_return"] = float(maxdiff.iloc[-1]) if not maxdiff.empty else 0.0
    result["maxdiff_turnover"] = float(maxdiff.abs().mean()) if not maxdiff.empty else 0.0

    close_safe = window["Close"].replace(0.0, np.nan)
    result["maxdiffprofit_high_price"] = predicted_high
    result["maxdiffprofit_low_price"] = predicted_low
    result["maxdiffprofit_profit_high_multiplier"] = (predicted_high / close_safe - 1.0).fillna(0.0)
    result["maxdiffprofit_profit_low_multiplier"] = (predicted_low / close_safe - 1.0).fillna(0.0)
    result["maxdiffprofit_profit"] = result["maxdiff_return"]
    result["maxdiffprofit_profit_values"] = result["maxdiff_return"]

    unprofit_rev = _rev(unprofit_series).fillna(0.0)
    result["unprofit_shutdown_return"] = unprofit_rev
    result["unprofit_shutdown_sharpe"] = unprofit_sharpe_val
    result["unprofit_shutdown_finalday"] = (
        float(unprofit_series.iloc[-1]) if not unprofit_series.empty else 0.0
    )

    result["walk_forward_oos_sharpe"] = simple_sharpe_val
    result["walk_forward_turnover"] = float(simple.abs().mean()) if not simple.empty else 0.0
    result["walk_forward_highlow_sharpe"] = highlow_sharpe_val
    result["walk_forward_takeprofit_sharpe"] = takeprofit_sharpe_val
    result["walk_forward_maxdiff_sharpe"] = maxdiff_sharpe_val

    result["close_val_loss"] = _rev((predicted_close - close_series).abs()).fillna(0.0)
    result["high_val_loss"] = _rev((predicted_high - high_series).abs()).fillna(0.0)
    result["low_val_loss"] = _rev((predicted_low - low_series).abs()).fillna(0.0)

    result["simulated_backtest"] = True

    _annotate_chronos2_metadata(result, symbol)

    latency_env = os.getenv("MARKETSIM_FALLBACK_INFERENCE_LATENCY")
    if latency_env is None:
        latency = 0.15
    else:
        try:
            latency = max(0.0, float(latency_env))
        except ValueError:
            latency = 0.0
    if latency > 0:
        time.sleep(min(latency, 5.0))

    return result


def backtest_forecasts(symbol: str, num_simulations: int | None = None) -> pd.DataFrame:
    num_simulations = num_simulations or _DEFAULT_NUM_SIMULATIONS
    if _REAL_BACKTEST_MODULE and hasattr(_REAL_BACKTEST_MODULE, "backtest_forecasts"):
        try:
            return _REAL_BACKTEST_MODULE.backtest_forecasts(symbol, num_simulations=num_simulations)  # type: ignore[return-value]
        except Exception as exc:  # pragma: no cover - mirrors behaviour if real stack fails at runtime
            import traceback

            logger.warning(
                f"[sim] Real backtest_forecasts failed for {symbol} ({exc}); using fallback simulator analytics.\n{traceback.format_exc()}"
            )
    elif _REAL_BACKTEST_ERROR:
        logger.debug(
            "[sim] Real backtest module previously failed to load (%s); continuing with fallback.",
            _REAL_BACKTEST_ERROR,
        )
    return _fallback_backtest(symbol, num_simulations)


def validate_pctdiff(
    symbol: str,
    *,
    num_simulations: Optional[int] = None,
    max_return: Optional[float] = None,
) -> float:
    """Validate pctdiff return magnitudes for the given symbol."""

    frame = backtest_forecasts(symbol, num_simulations=num_simulations)
    if frame is None or frame.empty:
        raise ValueError(f"No backtest analytics available for {symbol}")
    if "pctdiff_return" not in frame.columns:
        raise ValueError("pctdiff_return column missing from backtest output")

    returns = frame["pctdiff_return"].astype(float).to_numpy()
    if returns.size == 0:
        raise ValueError("pctdiff return series empty")
    max_abs = float(np.max(np.abs(returns)))
    if np.isnan(max_abs):
        raise ValueError("pctdiff returns contain NaN")

    limit = _PCTDIFF_MAX_RETURN if max_return is None else max_return
    if limit is not None and limit > 0 and max_abs > limit:
        raise ValueError(
            f"pctdiff returns {max_abs:.4f} exceed limit {limit:.4f} for {symbol} (num_simulations={num_simulations})"
        )

    logger.info(
        "[sim] pctdiff validation for %s succeeded (max |return|=%.4f, limit %.4f)",
        symbol,
        max_abs,
        limit if limit is not None else max_abs,
    )
    return max_abs


def release_model_resources() -> None:
    """Match production API surface even when using simulator fallback."""
    if _REAL_BACKTEST_MODULE and hasattr(_REAL_BACKTEST_MODULE, "release_model_resources"):
        try:
            _REAL_BACKTEST_MODULE.release_model_resources()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug("[sim] Ignored error releasing real backtest resources: %s", exc)
