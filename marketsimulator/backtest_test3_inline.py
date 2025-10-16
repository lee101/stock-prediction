from __future__ import annotations

import importlib.util
import os
from contextlib import suppress
from pathlib import Path
from typing import Optional

import pandas as pd

from .logging_utils import logger
from .state import get_state

_REAL_BACKTEST_MODULE = None
_REAL_BACKTEST_ERROR: Optional[Exception] = None
_DEFAULT_NUM_SIMULATIONS = int(os.getenv("MARKETSIM_NUM_SIMULATIONS", "20"))
_SKIP_REAL_IMPORT = os.getenv("MARKETSIM_SKIP_REAL_IMPORT", "0").lower() in {"1", "true", "yes", "on"}

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

    predicted_close = window["Close"].shift(-1).fillna(window["Close"])
    predicted_high = window["High"].shift(-1).fillna(window["High"])
    predicted_low = window["Low"].shift(-1).fillna(window["Low"])

    simple = predicted_close.pct_change().fillna(0.0)
    all_signals = (predicted_close + predicted_high + predicted_low) / 3.0
    all_signals = all_signals.pct_change().fillna(0.0)
    takeprofit = (predicted_high - window["Close"]) / window["Close"]
    highlow = (predicted_high - predicted_low) / window["Close"]

    logger.debug(
        "[sim] Using fallback backtest for %s with %d rows (real module unavailable).",
        symbol,
        len(window),
    )

    result = pd.DataFrame(
        {
            "close": window["Close"],
            "predicted_close": predicted_close,
            "predicted_high": predicted_high,
            "predicted_low": predicted_low,
            "simple_strategy_return": simple,
            "all_signals_strategy_return": all_signals,
            "entry_takeprofit_return": takeprofit,
            "highlow_return": highlow,
        }
    )
    if "timestamp" in window.columns:
        result["timestamp"] = window["timestamp"]
    return result


def backtest_forecasts(symbol: str, num_simulations: int | None = None) -> pd.DataFrame:
    num_simulations = num_simulations or _DEFAULT_NUM_SIMULATIONS
    if _REAL_BACKTEST_MODULE and hasattr(_REAL_BACKTEST_MODULE, "backtest_forecasts"):
        try:
            return _REAL_BACKTEST_MODULE.backtest_forecasts(symbol, num_simulations=num_simulations)  # type: ignore[return-value]
        except Exception as exc:  # pragma: no cover - mirrors behaviour if real stack fails at runtime
            logger.warning(
                "[sim] Real backtest_forecasts failed for %s (%s); using fallback simulator analytics.",
                symbol,
                exc,
            )
    elif _REAL_BACKTEST_ERROR:
        logger.debug(
            "[sim] Real backtest module previously failed to load (%s); continuing with fallback.",
            _REAL_BACKTEST_ERROR,
        )
    return _fallback_backtest(symbol, num_simulations)


def release_model_resources() -> None:
    """Match production API surface even when using simulator fallback."""
    if _REAL_BACKTEST_MODULE and hasattr(_REAL_BACKTEST_MODULE, "release_model_resources"):
        try:
            _REAL_BACKTEST_MODULE.release_model_resources()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug("[sim] Ignored error releasing real backtest resources: %s", exc)
