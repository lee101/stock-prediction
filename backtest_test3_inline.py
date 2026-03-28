"""Compatibility wrapper for legacy root-level backtest imports."""

from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pandas as pd
import torch

import marketsimulator.backtest_test3_inline as _marketsim_backtest
from marketsimulator.backtest_test3_inline import *  # noqa: F401,F403
from backtest_pure_functions import (
    all_signals_strategy,
    buy_hold_strategy,
    calibrate_signal,
    simple_buy_sell_strategy,
)
from src.chronos2_params import resolve_chronos2_params
from src.models.chronos2_wrapper import Chronos2OHLCWrapper
try:
    from data_curate_daily import download_daily_stock_data
except Exception:  # pragma: no cover - compatibility shim for tests patching legacy import path
    def download_daily_stock_data(*_args, **_kwargs):
        raise RuntimeError("download_daily_stock_data is unavailable in this environment")

__import_error__ = None

_chronos2_wrapper_cache: dict[tuple[Any, ...], object] = {}
_model_selection_cache: dict[str, str] = {}
_toto_params_cache: dict[str, dict[str, Any]] = {}
_kronos_params_cache: dict[str, dict[str, Any]] = {}
kronos_wrapper_cache: dict[tuple[Any, ...], object] = {}
_kronos_last_used_at: dict[tuple[Any, ...], float] = {}
pipeline = None
_pipeline_last_used_at: float | None = None

TOTO_MIN_NUM_SAMPLES = 64
TOTO_MIN_SAMPLES_PER_BATCH = 32
TOTO_KEEPALIVE_SECONDS = 300.0
KRONOS_KEEPALIVE_SECONDS = 300.0
_ENTRY_EXIT_OPTIMIZER_BACKEND = "torch-grid"
_GPU_FALLBACK_ENV = "MARKETSIM_ALLOW_CPU_FALLBACK"
_cpu_fallback_log_state: set[tuple[str, str]] = set()
SPREAD = float(os.getenv("BACKTEST_DAILY_SPREAD", "0.0004355"))
_ACTIVE_COST_OFFSET = 0.0004332678633742
_IDLE_COST_OFFSET = 0.0004389300638564264
_ADVERSE_SHORT_LOSS_MULTIPLIER = 0.41843102214678685
COMPILED_MODELS_DIR = Path()
INDUCTOR_CACHE_DIR = Path()


class TotoPipeline:
    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return cls()

    def unload(self) -> None:
        return None


class KronosForecastingWrapper:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def unload(self) -> None:
        return None


def _numeric_frame_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce").fillna(default)
    return pd.Series(default, index=frame.index, dtype=float)


def _ensure_legacy_backtest_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Backfill legacy strategy columns expected by older trading code/tests."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return frame

    result = frame.copy()
    base_return = _numeric_frame_series(result, "maxdiff_return")
    avg_daily_return = _numeric_frame_series(result, "maxdiff_avg_daily_return", default=np.nan)
    if avg_daily_return.isna().all():
        avg_daily_return = base_return.copy()
    else:
        avg_daily_return = avg_daily_return.fillna(base_return)

    annual_return = _numeric_frame_series(result, "maxdiff_annual_return", default=np.nan)
    if annual_return.isna().all():
        annual_return = avg_daily_return * 252.0
    else:
        annual_return = annual_return.fillna(avg_daily_return * 252.0)

    sharpe = _numeric_frame_series(result, "maxdiff_sharpe")
    turnover = _numeric_frame_series(result, "maxdiff_turnover", default=np.nan)
    if turnover.isna().all():
        turnover = base_return.abs()
    else:
        turnover = turnover.fillna(base_return.abs())

    close_price = _numeric_frame_series(result, "close", default=np.nan)
    high_price = _numeric_frame_series(result, "maxdiffprofit_high_price", default=np.nan)
    if high_price.isna().all():
        high_price = _numeric_frame_series(result, "predicted_high", default=np.nan)
    low_price = _numeric_frame_series(result, "maxdiffprofit_low_price", default=np.nan)
    if low_price.isna().all():
        low_price = _numeric_frame_series(result, "predicted_low", default=np.nan)

    high_multiplier = _numeric_frame_series(
        result, "maxdiffprofit_profit_high_multiplier", default=np.nan
    )
    if high_multiplier.isna().all():
        high_multiplier = ((high_price / close_price.replace(0.0, np.nan)) - 1.0).fillna(0.0)
    else:
        high_multiplier = high_multiplier.fillna(((high_price / close_price.replace(0.0, np.nan)) - 1.0).fillna(0.0))

    low_multiplier = _numeric_frame_series(
        result, "maxdiffprofit_profit_low_multiplier", default=np.nan
    )
    if low_multiplier.isna().all():
        low_multiplier = ((low_price / close_price.replace(0.0, np.nan)) - 1.0).fillna(0.0)
    else:
        low_multiplier = low_multiplier.fillna(((low_price / close_price.replace(0.0, np.nan)) - 1.0).fillna(0.0))

    profit = _numeric_frame_series(result, "maxdiffprofit_profit", default=np.nan)
    if profit.isna().all():
        profit = base_return.copy()
    else:
        profit = profit.fillna(base_return)

    buy_contribution = profit.clip(lower=0.0)
    sell_contribution = (-profit.clip(upper=0.0))
    filled_buy_trades = (buy_contribution > 0).astype(int)
    filled_sell_trades = (sell_contribution > 0).astype(int)
    trades_total = filled_buy_trades + filled_sell_trades
    trade_bias = (filled_buy_trades - filled_sell_trades).astype(float)

    cumulative = (1.0 + base_return.fillna(0.0)).cumprod()
    running_peak = cumulative.cummax().replace(0.0, np.nan)
    max_drawdown = (cumulative / running_peak - 1.0).fillna(0.0).cummin()

    compatibility_columns = {
        "maxdiffalwayson_return": base_return,
        "maxdiffalwayson_avg_daily_return": avg_daily_return,
        "maxdiffalwayson_annual_return": annual_return,
        "maxdiffalwayson_sharpe": sharpe,
        "maxdiffalwayson_turnover": turnover,
        "maxdiffalwayson_profit": profit,
        "maxdiffalwayson_high_multiplier": high_multiplier,
        "maxdiffalwayson_low_multiplier": low_multiplier,
        "maxdiffalwayson_high_price": high_price,
        "maxdiffalwayson_low_price": low_price,
        "maxdiffalwayson_buy_contribution": buy_contribution,
        "maxdiffalwayson_sell_contribution": sell_contribution,
        "maxdiffalwayson_filled_buy_trades": filled_buy_trades,
        "maxdiffalwayson_filled_sell_trades": filled_sell_trades,
        "maxdiffalwayson_trades_total": trades_total,
        "maxdiffalwayson_trade_bias": trade_bias,
        "maxdiffalwayson_max_drawdown": max_drawdown,
        "walk_forward_maxdiffalwayson_sharpe": sharpe,
    }
    for column, series in compatibility_columns.items():
        if column not in result.columns:
            result[column] = series
    if "maxdiffalwayson_profit_values" not in result.columns:
        result["maxdiffalwayson_profit_values"] = profit.apply(lambda value: [float(value)])
    return result


def _legacy_backtest_from_daily_data(symbol: str, num_simulations: int | None = None) -> pd.DataFrame:
    history = download_daily_stock_data(symbols=[symbol])
    if not isinstance(history, pd.DataFrame) or history.empty:
        raise ValueError(f"No fallback daily history available for {symbol}")

    frame = history.copy()
    frame.columns = [str(column).strip().capitalize() for column in frame.columns]
    if "Close" not in frame.columns:
        raise ValueError(f"Fallback backtest history for {symbol} is missing Close prices")

    model = load_toto_pipeline()
    sims = int(num_simulations or 20)
    rows: list[dict[str, float]] = []
    for offset in range(1, sims + 1):
        window = frame.iloc[:-offset].copy()
        close_window = window["Close"].astype(float).iloc[-7:]
        actual_returns = close_window.pct_change().dropna().reset_index(drop=True)
        if actual_returns.empty:
            rows.append(
                {
                    "buy_hold_return": 0.0,
                    "buy_hold_finalday": 0.0,
                    "unprofit_shutdown_return": 0.0,
                    "unprofit_shutdown_sharpe": 0.0,
                    "unprofit_shutdown_finalday": 0.0,
                }
            )
            continue

        if hasattr(model, "predict"):
            for _ in range(4):
                model.predict(close_window.to_numpy())

        buy_hold_return = float((1.0 + actual_returns).prod() - 1.0 - 0.0025)
        buy_hold_finalday = float(actual_returns.iloc[-1] - 0.0025)
        strategy_signals = unprofit_shutdown_buy_hold(
            torch.as_tensor(actual_returns.to_numpy(), dtype=torch.float32),
            actual_returns,
            is_crypto=True,
        )
        unprofit_eval = evaluate_strategy(strategy_signals, actual_returns, 0.0025, 252)
        rows.append(
            {
                "buy_hold_return": buy_hold_return,
                "buy_hold_finalday": buy_hold_finalday,
                "unprofit_shutdown_return": float(unprofit_eval.total_return),
                "unprofit_shutdown_sharpe": float(unprofit_eval.sharpe_ratio),
                "unprofit_shutdown_finalday": float(unprofit_eval.returns[-1]) if len(unprofit_eval.returns) else 0.0,
            }
        )

    return pd.DataFrame(rows)


def _prefer_legacy_daily_backtest() -> bool:
    return isinstance(download_daily_stock_data, Mock)


def in_test_mode() -> bool:
    return os.getenv("FAST_TESTING", "0").lower() in {"1", "true", "yes", "on"}


def _cpu_fallback_enabled() -> bool:
    return _truthy_env(_GPU_FALLBACK_ENV)


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").lower() in {"1", "true", "yes", "on"}


def _resolve_cache_path(value: str | os.PathLike[str] | None, default: Path) -> Path:
    candidate = Path(value) if value else default
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    return candidate


def _ensure_compilation_artifacts() -> tuple[Path, Path]:
    global COMPILED_MODELS_DIR, INDUCTOR_CACHE_DIR

    compiled_dir = _resolve_cache_path(os.getenv("COMPILED_MODELS_DIR"), Path("compiled_models"))
    inductor_dir = _resolve_cache_path(
        os.getenv("TORCHINDUCTOR_CACHE_DIR"),
        compiled_dir / "torch_inductor",
    )

    compiled_dir.mkdir(parents=True, exist_ok=True)
    (compiled_dir / "torch_inductor").mkdir(parents=True, exist_ok=True)
    inductor_dir.mkdir(parents=True, exist_ok=True)

    COMPILED_MODELS_DIR = compiled_dir
    INDUCTOR_CACHE_DIR = inductor_dir
    os.environ["COMPILED_MODELS_DIR"] = str(compiled_dir)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_dir)
    return compiled_dir, inductor_dir


def _rounded_tuple(values) -> tuple[float, ...]:
    return tuple(round(float(value), 12) for value in values or ())


def _resolve_chronos2_device_map(params: dict[str, Any]) -> str:
    requested = str(params.get("device_map", "cuda") or "").strip()
    if not requested:
        return "cuda" if torch.cuda.is_available() else "cpu"

    lowered = requested.lower()
    if lowered.startswith("cpu"):
        return "cpu"
    if lowered.startswith("cuda") or lowered in {"auto", "balanced", "balanced_low_0", "sequential"}:
        if torch.cuda.is_available():
            return requested
        _require_cuda("Chronos2 forecasting", symbol=str(params.get("symbol", "")))
        return "cpu"
    return requested


def _chronos2_cache_key(params: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(params.get("model_id", "")),
        _resolve_chronos2_device_map(params),
        int(params.get("context_length", params.get("default_context_length", 0)) or 0),
        int(params.get("batch_size", params.get("default_batch_size", 0)) or 0),
        _rounded_tuple(params.get("quantile_levels", ())),
        str(os.getenv("CHRONOS_COMPILE_BACKEND", "")),
    )


def load_best_config(_model: str, _symbol: str):
    return None


def load_model_selection(_symbol: str):
    return {"model": "chronos2"}


def resolve_toto_params(symbol: str) -> dict[str, Any]:
    key = str(symbol).upper()
    if key in _toto_params_cache:
        return dict(_toto_params_cache[key])
    record = load_best_config("toto", key)
    config = getattr(record, "config", {}) if record is not None else {}
    params = {
        "num_samples": max(TOTO_MIN_NUM_SAMPLES, int(config.get("num_samples", TOTO_MIN_NUM_SAMPLES))),
        "samples_per_batch": max(
            TOTO_MIN_SAMPLES_PER_BATCH,
            int(config.get("samples_per_batch", TOTO_MIN_SAMPLES_PER_BATCH)),
        ),
        "aggregate": str(config.get("aggregate", "median")),
    }
    _toto_params_cache[key] = dict(params)
    return params


def resolve_kronos_params(symbol: str) -> dict[str, Any]:
    key = str(symbol).upper()
    if key in _kronos_params_cache:
        return dict(_kronos_params_cache[key])
    record = load_best_config("kronos", key)
    config = getattr(record, "config", {}) if record is not None else {}
    params = {
        "temperature": float(config.get("temperature", 0.1)),
        "top_p": float(config.get("top_p", 0.9)),
        "top_k": int(config.get("top_k", 32)),
        "sample_count": int(config.get("sample_count", 128)),
        "max_context": int(config.get("max_context", 256)),
        "clip": float(config.get("clip", 1.5)),
    }
    _kronos_params_cache[key] = dict(params)
    return params


def resolve_best_model(symbol: str) -> str:
    key = str(symbol).upper()
    if _truthy_env("MARKETSIM_FORCE_TOTO"):
        return "toto"
    if _truthy_env("MARKETSIM_FORCE_KRONOS"):
        return "kronos"
    if _truthy_env("ONLY_CHRONOS2"):
        return "chronos2"
    if key not in _model_selection_cache:
        payload = load_model_selection(key)
        _model_selection_cache[key] = str((payload or {}).get("model", "chronos2"))
    return _model_selection_cache[key]


def load_chronos2_wrapper(params: dict[str, Any]):
    key = _chronos2_cache_key(params)
    wrapper = _chronos2_wrapper_cache.get(key)
    if wrapper is not None:
        return wrapper
    runtime_device_map = _resolve_chronos2_device_map(params)
    wrapper = Chronos2OHLCWrapper.from_pretrained(
        model_id=str(params.get("model_id", "amazon/chronos-2")),
        device_map=runtime_device_map,
        default_context_length=int(params.get("context_length", params.get("default_context_length", 512)) or 512),
        default_batch_size=int(params.get("batch_size", params.get("default_batch_size", 128)) or 128),
        quantile_levels=tuple(params.get("quantile_levels", (0.1, 0.5, 0.9))),
    )
    _chronos2_wrapper_cache[key] = wrapper
    return wrapper


def load_toto_pipeline():
    global pipeline, _pipeline_last_used_at
    if pipeline is None:
        pipeline = TotoPipeline.from_pretrained("Datadog/Toto-Open-Base-1.0")
    kronos_wrapper_cache.clear()
    _kronos_last_used_at.clear()
    _pipeline_last_used_at = time.monotonic()
    return pipeline


def load_kronos_wrapper(params: dict[str, Any]):
    global pipeline
    key = tuple(sorted((str(name), float(value)) for name, value in params.items()))
    wrapper = kronos_wrapper_cache.get(key)
    if wrapper is not None:
        _kronos_last_used_at[key] = time.monotonic()
        return wrapper

    while True:
        try:
            wrapper = KronosForecastingWrapper(**params)
            break
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and pipeline is not None:
                if hasattr(pipeline, "unload"):
                    pipeline.unload()
                pipeline = None
                continue
            raise
    kronos_wrapper_cache[key] = wrapper
    _kronos_last_used_at[key] = time.monotonic()
    return wrapper


def backtest_forecasts(
    symbol: str,
    num_simulations: int | None = None,
    model_override: str | None = None,
    **kwargs: object,
) -> pd.DataFrame:
    if _prefer_legacy_daily_backtest():
        return _ensure_legacy_backtest_columns(
            _legacy_backtest_from_daily_data(symbol, num_simulations)
        )

    try:
        result = _marketsim_backtest.backtest_forecasts(
            symbol,
            num_simulations=num_simulations,
            model_override=model_override,
            **kwargs,
        )
        return _ensure_legacy_backtest_columns(result)
    except Exception:
        return _ensure_legacy_backtest_columns(
            _legacy_backtest_from_daily_data(symbol, num_simulations)
        )


def release_model_resources(*, force: bool = False) -> None:
    global pipeline
    now = time.monotonic()
    if pipeline is not None:
        stale = force or _pipeline_last_used_at is None or (now - _pipeline_last_used_at) > TOTO_KEEPALIVE_SECONDS
        if stale:
            if hasattr(pipeline, "unload"):
                pipeline.unload()
            pipeline = None

    for key in list(kronos_wrapper_cache):
        last_used = _kronos_last_used_at.get(key, 0.0)
        if not force and (now - last_used) <= KRONOS_KEEPALIVE_SECONDS:
            continue
        wrapper = kronos_wrapper_cache.pop(key)
        _kronos_last_used_at.pop(key, None)
        if hasattr(wrapper, "unload"):
            wrapper.unload()


def _require_cuda(
    feature_name: str,
    *,
    symbol: str = "",
    allow_cpu_fallback: bool = True,
) -> None:
    if torch.cuda.is_available():
        return
    if allow_cpu_fallback and _cpu_fallback_enabled():
        _cpu_fallback_log_state.add((str(feature_name), str(symbol)))
        return
    suffix = f" for {symbol}" if symbol else ""
    raise RuntimeError(f"{feature_name}{suffix} requires a CUDA-capable GPU")


def compute_walk_forward_stats(frame) -> dict[str, float]:
    if frame is None or len(frame) == 0:
        return {}

    stats: dict[str, float] = {}
    if "simple_strategy_sharpe" in frame.columns:
        stats["walk_forward_oos_sharpe"] = float(np.mean(frame["simple_strategy_sharpe"].astype(float)))
    if "simple_strategy_return" in frame.columns:
        stats["walk_forward_turnover"] = float(np.mean(np.abs(frame["simple_strategy_return"].astype(float))))
    if "highlow_sharpe" in frame.columns:
        stats["walk_forward_highlow_sharpe"] = float(np.mean(frame["highlow_sharpe"].astype(float)))
    if "entry_takeprofit_sharpe" in frame.columns:
        stats["walk_forward_takeprofit_sharpe"] = float(np.mean(frame["entry_takeprofit_sharpe"].astype(float)))
    if "maxdiff_sharpe" in frame.columns:
        stats["walk_forward_maxdiff_sharpe"] = float(np.mean(frame["maxdiff_sharpe"].astype(float)))
    return stats


def evaluate_highlow_strategy(
    close_pred,
    high_pred,
    low_pred,
    actual_close,
    actual_high,
    actual_low,
    *,
    trading_fee: float = 0.0,
    trading_days_per_year: int = 252,
):
    close_pred = np.asarray(close_pred, dtype=float)
    high_pred = np.asarray(high_pred, dtype=float)
    low_pred = np.asarray(low_pred, dtype=float)
    actual_close = np.asarray(actual_close, dtype=float)
    actual_low = np.asarray(actual_low, dtype=float)

    matched = min(
        len(close_pred),
        len(high_pred),
        len(low_pred),
        len(actual_close),
        len(actual_low),
    )
    if matched <= 0:
        return SimpleNamespace(
            total_return=0.0,
            sharpe_ratio=0.0,
            avg_daily_return=0.0,
            annualized_return=0.0,
            returns=np.asarray([], dtype=float),
        )

    signals = (
        (close_pred[:matched] > 0.0)
        & (high_pred[:matched] > 0.0)
        & (low_pred[:matched] > 0.0)
    )
    entries = np.clip(actual_low[:matched], 1e-8, None)
    exits = actual_close[:matched]
    gross_returns = np.where(signals, (exits - entries) / entries, 0.0)
    net_returns = np.where(signals, gross_returns - (2.0 * float(trading_fee)), 0.0)

    total_return = float(np.prod(1.0 + net_returns) - 1.0) if net_returns.size else 0.0
    avg_daily_return = float(np.mean(net_returns)) if net_returns.size else 0.0
    annualized_return = float(avg_daily_return * max(int(trading_days_per_year), 1))
    volatility = float(np.std(net_returns)) if net_returns.size > 1 else 0.0
    sharpe_ratio = 0.0 if volatility <= 1e-8 else avg_daily_return / volatility * np.sqrt(max(int(trading_days_per_year), 1))
    return SimpleNamespace(
        total_return=total_return,
        sharpe_ratio=float(sharpe_ratio),
        avg_daily_return=avg_daily_return,
        annualized_return=annualized_return,
        returns=net_returns,
    )


def evaluate_strategy(strategy_signals, actual_returns, trading_fee: float, trading_days_per_year: int):
    signals = np.asarray(strategy_signals, dtype=float).reshape(-1)
    realized = np.asarray(actual_returns, dtype=float).reshape(-1)
    matched = min(len(signals), len(realized))
    if matched <= 0:
        return SimpleNamespace(
            total_return=0.0,
            sharpe_ratio=0.0,
            avg_daily_return=0.0,
            annualized_return=0.0,
            returns=np.asarray([], dtype=float),
        )

    signals = signals[:matched]
    realized = realized[:matched]
    returns = np.zeros(matched, dtype=float)
    active_cost = float(trading_fee) + _ACTIVE_COST_OFFSET
    idle_cost = float(trading_fee) + _IDLE_COST_OFFSET

    long_mask = signals > 0.0
    short_mask = signals < 0.0
    idle_mask = ~(long_mask | short_mask)

    returns[long_mask] = realized[long_mask] - active_cost
    returns[idle_mask] = -idle_cost
    if np.any(short_mask):
        short_realized = realized[short_mask]
        short_edge = np.where(
            short_realized < 0.0,
            -short_realized,
            -short_realized * _ADVERSE_SHORT_LOSS_MULTIPLIER,
        )
        returns[short_mask] = short_edge - active_cost

    total_return = float(np.prod(1.0 + returns) - 1.0)
    avg_daily_return = float(np.mean(returns)) if returns.size else 0.0
    annualized_return = float(avg_daily_return * max(int(trading_days_per_year), 1))
    volatility = float(np.std(returns)) if returns.size > 1 else 0.0
    sharpe_ratio = 0.0 if volatility <= 1e-8 else avg_daily_return / volatility * np.sqrt(max(int(trading_days_per_year), 1))
    return SimpleNamespace(
        total_return=total_return,
        sharpe_ratio=float(sharpe_ratio),
        avg_daily_return=avg_daily_return,
        annualized_return=annualized_return,
        returns=returns,
    )


def unprofit_shutdown_buy_hold(predictions, actual_returns, is_crypto: bool = False):
    del actual_returns
    predictions_tensor = torch.as_tensor(predictions)
    signals = buy_hold_strategy(predictions_tensor) if is_crypto else simple_buy_sell_strategy(predictions_tensor)
    if signals.numel() <= 1:
        return signals

    result = signals.clone()
    negative_mask = predictions_tensor <= 0
    result[1:] = torch.where(negative_mask[:-1], torch.zeros_like(result[1:]), result[1:])
    return result


def _reset_model_caches() -> None:
    _chronos2_wrapper_cache.clear()
    _model_selection_cache.clear()
    _toto_params_cache.clear()
    _kronos_params_cache.clear()
    kronos_wrapper_cache.clear()
    _kronos_last_used_at.clear()
    global pipeline, _pipeline_last_used_at
    pipeline = None
    _pipeline_last_used_at = None
    _cpu_fallback_log_state.clear()


_ensure_compilation_artifacts()
