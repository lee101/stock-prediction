from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pytest
import torch

from faltrain.forecasting import create_kronos_wrapper, create_toto_pipeline
from faltrain.hyperparams import HyperparamResolver, HyperparamResult
from src.dependency_injection import setup_imports as setup_src_imports
from src.models.toto_aggregation import aggregate_with_spec


DATA_DIR = Path("trainingdata")
BEST_DIR = Path("hyperparams/best")
MAX_EVAL_STEPS = 5
MAX_SYMBOLS = 2


@dataclass
class ForecastMetrics:
    price_mae: float
    pct_return_mae: float
    avg_latency_s: float
    predictions: List[float]
    actuals: List[float]


class _StaticResolver:
    """Resolver shim that always returns the provided hyperparameter result."""

    def __init__(self, result: HyperparamResult) -> None:
        self._result = result

    def load(self, *_: object, **__: object) -> HyperparamResult:
        return self._result


def _load_series(symbol: str) -> pd.DataFrame:
    path = DATA_DIR / f"{symbol}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset for symbol '{symbol}' at {path}.")
    df = pd.read_csv(path)
    if "timestamp" not in df.columns or "close" not in df.columns:
        raise KeyError(f"{path} requires 'timestamp' and 'close' columns for evaluation.")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _mean_absolute_error(actual: Sequence[float], predicted: Sequence[float]) -> float:
    if not actual or not predicted:
        raise ValueError("MAE requires at least one value.")
    actual_arr = np.asarray(actual, dtype=np.float64)
    predicted_arr = np.asarray(predicted, dtype=np.float64)
    if actual_arr.shape != predicted_arr.shape:
        raise ValueError("Actual and predicted sequences must share the same shape.")
    return float(np.mean(np.abs(actual_arr - predicted_arr)))


def _extract_window(result: Optional[HyperparamResult], key: str, default: int) -> int:
    if result is None:
        return int(default)
    windows = result.payload.get("windows", {})
    value = windows.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _extract_horizon(result: Optional[HyperparamResult], default: int = 1) -> int:
    if result is None:
        return int(default)
    windows = result.payload.get("windows", {})
    horizon = windows.get("forecast_horizon", default)
    try:
        return int(horizon)
    except (TypeError, ValueError):
        return int(default)


def _build_eval_indices(length: int, *, window: int, horizon: int) -> range:
    if length <= horizon:
        return range(0, 0)
    start = max(horizon, length - window)
    start = max(start, 2)  # need at least two prices for returns
    end = length - horizon + 1
    if start >= end:
        return range(0, 0)
    return range(start, end)


def _compute_return(current_price: float, previous_price: float) -> float:
    if previous_price == 0.0:
        return 0.0
    return (current_price - previous_price) / previous_price


def _evaluate_kronos(
    df: pd.DataFrame,
    *,
    bundle,
    indices: Iterable[int],
    horizon: int,
) -> ForecastMetrics:
    wrapper = bundle.wrapper
    predictions: List[float] = []
    actuals: List[float] = []
    pred_returns: List[float] = []
    actual_returns: List[float] = []
    latencies: List[float] = []

    close_values = df["close"].to_numpy(dtype=np.float64)

    for step_idx, idx in enumerate(indices):
        if step_idx >= MAX_EVAL_STEPS:
            break
        history = df.iloc[:idx].copy()
        if history.shape[0] < 2:
            continue
        start_time = time.perf_counter()
        result = wrapper.predict_series(
            data=history,
            timestamp_col="timestamp",
            columns=["close"],
            pred_len=horizon,
            lookback=bundle.max_context,
            temperature=bundle.temperature,
            top_p=bundle.top_p,
            top_k=bundle.top_k,
            sample_count=bundle.sample_count,
        )
        latencies.append(time.perf_counter() - start_time)

        kronos_close = result.get("close")
        if kronos_close is None or kronos_close.absolute.size < horizon:
            raise RuntimeError("Kronos forecast did not return expected horizon.")

        price_pred = float(kronos_close.absolute[0])
        predictions.append(price_pred)

        actual_price = float(close_values[idx])
        actuals.append(actual_price)

        prev_price = float(close_values[idx - 1])
        pred_returns.append(_compute_return(price_pred, prev_price))
        actual_returns.append(_compute_return(actual_price, prev_price))

    if not predictions:
        raise RuntimeError("Kronos evaluation produced no forecasts.")

    price_mae = _mean_absolute_error(actuals, predictions)
    pct_return_mae = _mean_absolute_error(actual_returns, pred_returns)
    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    return ForecastMetrics(price_mae, pct_return_mae, avg_latency, predictions, actuals)


def _evaluate_toto(
    df: pd.DataFrame,
    *,
    pipeline,
    config: Dict[str, object],
    indices: Iterable[int],
    horizon: int,
) -> ForecastMetrics:
    close_values = df["close"].to_numpy(dtype=np.float64)

    num_samples = int(config.get("num_samples", 4096))
    samples_per_batch = int(config.get("samples_per_batch", min(512, num_samples)))
    samples_per_batch = max(1, min(samples_per_batch, num_samples))
    aggregate_spec = str(config.get("aggregate", "mean")).strip() or "mean"

    predictions: List[float] = []
    actuals: List[float] = []
    pred_returns: List[float] = []
    actual_returns: List[float] = []
    latencies: List[float] = []

    for step_idx, idx in enumerate(indices):
        if step_idx >= MAX_EVAL_STEPS:
            break
        context = close_values[:idx].astype(np.float32)
        if context.size < 2:
            continue

        start_time = time.perf_counter()
        forecasts = pipeline.predict(
            context=context,
            prediction_length=horizon,
            num_samples=num_samples,
            samples_per_batch=samples_per_batch,
        )
        latencies.append(time.perf_counter() - start_time)

        if not forecasts:
            raise RuntimeError("Toto pipeline returned no forecasts.")
        aggregated = aggregate_with_spec(forecasts[0].samples, aggregate_spec)
        if aggregated.size < horizon:
            raise RuntimeError("Aggregated Toto forecast shorter than requested horizon.")

        price_pred = float(np.asarray(aggregated, dtype=np.float64)[0])
        predictions.append(price_pred)

        actual_price = float(close_values[idx])
        actuals.append(actual_price)

        prev_price = float(close_values[idx - 1])
        pred_returns.append(_compute_return(price_pred, prev_price))
        actual_returns.append(_compute_return(actual_price, prev_price))

    if not predictions:
        raise RuntimeError("Toto evaluation produced no forecasts.")

    price_mae = _mean_absolute_error(actuals, predictions)
    pct_return_mae = _mean_absolute_error(actual_returns, pred_returns)
    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    return ForecastMetrics(price_mae, pct_return_mae, avg_latency, predictions, actuals)


def _load_best_payload(symbol: str) -> Optional[Dict[str, object]]:
    path = BEST_DIR / f"{symbol}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


@pytest.mark.cuda_required
@pytest.mark.integration
def test_kronos_toto_line_eval() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for Kronos/Toto line evaluation.")

    setup_src_imports(torch=torch, numpy=np, pandas=pd)

    resolver = HyperparamResolver()

    kronos_paths = {path.stem for path in (Path("hyperparams/kronos")).glob("*.json")}
    toto_paths = {path.stem for path in (Path("hyperparams/toto")).glob("*.json")}
    data_paths = {path.stem for path in DATA_DIR.glob("*.csv")}

    symbols = sorted(kronos_paths & toto_paths & data_paths)
    if not symbols:
        pytest.skip("No overlapping symbols across hyperparams and trading data.")

    summaries: List[str] = []

    toto_pipeline = None

    try:
        for idx_symbol, symbol in enumerate(symbols):
            if idx_symbol >= MAX_SYMBOLS:
                break
            kronos_result = resolver.load(symbol, "kronos", prefer_best=True, allow_remote=False)
            toto_result = resolver.load(symbol, "toto", prefer_best=True, allow_remote=False)

            if kronos_result is None and toto_result is None:
                continue

            df = _load_series(symbol)

            kronos_window = _extract_window(kronos_result, "test_window", 20)
            toto_window = _extract_window(toto_result, "test_window", 20)
            eval_window = max(kronos_window, toto_window, 20)

            kronos_horizon = _extract_horizon(kronos_result)
            toto_horizon = _extract_horizon(toto_result)
            horizon = max(kronos_horizon, toto_horizon, 1)
            if horizon != 1:
                pytest.skip(f"Forecast horizon {horizon} currently unsupported for symbol {symbol}.")

            indices = _build_eval_indices(len(df), window=eval_window, horizon=horizon)
            if not indices:
                pytest.skip(f"Insufficient data to evaluate symbol {symbol} with window {eval_window}.")

            kronos_metrics: Optional[ForecastMetrics] = None
            kronos_config_name: Optional[str] = None

            if kronos_result is not None:
                kronos_bundle = create_kronos_wrapper(
                    symbol,
                    resolver=_StaticResolver(kronos_result),
                    device="cuda:0",
                    prefer_best=False,
                )
                try:
                    kronos_metrics = _evaluate_kronos(
                        df,
                        bundle=kronos_bundle,
                        indices=indices,
                        horizon=horizon,
                    )
                finally:
                    kronos_bundle.wrapper.unload()
                kronos_config_name = kronos_result.config.get("name") or "unknown"

            toto_metrics: Optional[ForecastMetrics] = None
            toto_config_name: Optional[str] = None

            if toto_result is not None:
                if toto_pipeline is None:
                    bundle = create_toto_pipeline(
                        symbol,
                        resolver=_StaticResolver(toto_result),
                        device_map="cuda",
                        prefer_best=False,
                    )
                    toto_pipeline = bundle.pipeline
                toto_metrics = _evaluate_toto(
                    df,
                    pipeline=toto_pipeline,
                    config=toto_result.config,
                    indices=indices,
                    horizon=horizon,
                )
                toto_config_name = toto_result.config.get("name") or "unknown"

            best_payload = _load_best_payload(symbol)
            best_model = best_payload.get("model") if best_payload else None
            best_name = None
            if best_payload:
                best_config = best_payload.get("config") or {}
                if isinstance(best_config, dict):
                    best_name = best_config.get("name")

            summary_parts = [f"{symbol}"]
            if best_model:
                summary_parts.append(f"best={best_model}/{best_name or 'n/a'}")
            if kronos_metrics:
                summary_parts.append(
                    (
                        f"Kronos[{kronos_config_name}] "
                        f"price_mae={kronos_metrics.price_mae:.4f} "
                        f"pct_mae={kronos_metrics.pct_return_mae:.5f} "
                        f"avg_latency_s={kronos_metrics.avg_latency_s:.3f}"
                    )
                )
            if toto_metrics:
                summary_parts.append(
                    (
                        f"Toto[{toto_config_name}] "
                        f"price_mae={toto_metrics.price_mae:.4f} "
                        f"pct_mae={toto_metrics.pct_return_mae:.5f} "
                        f"avg_latency_s={toto_metrics.avg_latency_s:.3f}"
                    )
                )
            summaries.append(" | ".join(summary_parts))
    finally:
        if toto_pipeline is not None:
            try:
                toto_pipeline.unload()
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    if not summaries:
        pytest.skip("No symbols produced evaluation summaries.")

    print("Kronos/Toto line evaluation results:")
    for line in summaries:
        print(line)

    assert summaries, "Expected at least one evaluation summary."
