import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union
from dataclasses import dataclass

from src.comparisons import is_buy_side
from src.logging_utils import setup_logging
from src.torch_backend import configure_tf32_backends

logger = setup_logging("backtest_test3_inline.log")

_BOOL_FALSE = {"0", "false", "no", "off"}
_FAST_TORCH_SETTINGS_CONFIGURED = False


def _read_env_flag(names: Iterable[str]) -> Optional[bool]:
    for name in names:
        value = os.getenv(name)
        if value is None:
            continue
        lowered = value.strip().lower()
        if lowered in _BOOL_TRUE:
            return True
        if lowered in _BOOL_FALSE:
            return False
    return None


def _maybe_enable_fast_torch_settings() -> None:
    global _FAST_TORCH_SETTINGS_CONFIGURED
    if _FAST_TORCH_SETTINGS_CONFIGURED:
        return
    _FAST_TORCH_SETTINGS_CONFIGURED = True

    try:
        state = configure_tf32_backends(torch, logger=logger)
        if state["legacy_api"]:
            matmul = getattr(getattr(torch.backends, "cuda", None), "matmul", None)
            if matmul is not None and hasattr(matmul, "allow_fp16_reduced_precision_reduction"):
                try:
                    matmul.allow_fp16_reduced_precision_reduction = True  # type: ignore[attr-defined]
                except Exception as exc:
                    logger.debug("Unable to enable reduced precision reductions: %s", exc)
        cuda_backends = getattr(torch.backends, "cuda", None)
        if cuda_backends is not None:
            try:
                enable_flash = getattr(cuda_backends, "enable_flash_sdp", None)
                if callable(enable_flash):
                    enable_flash(True)
                enable_mem = getattr(cuda_backends, "enable_mem_efficient_sdp", None)
                if callable(enable_mem):
                    enable_mem(True)
                enable_math = getattr(cuda_backends, "enable_math_sdp", None)
                if callable(enable_math):
                    enable_math(False)
            except Exception as exc:
                logger.debug("Unable to configure scaled dot product kernels: %s", exc)
    except Exception as exc:  # pragma: no cover - defensive guardrail
        logger.debug("Torch backend optimisation setup failed: %s", exc)

from data_curate_daily import download_daily_stock_data, fetch_spread
from disk_cache import disk_cache
from src.fixtures import crypto_symbols
from scripts.alpaca_cli import set_strategy_for_symbol
from src.models.toto_wrapper import TotoPipeline
from src.models.toto_aggregation import aggregate_with_spec
from src.models.kronos_wrapper import KronosForecastingWrapper
from hyperparamstore import load_best_config, load_model_selection
from loss_utils import (
    percent_movements_augment,
    calculate_profit_torch_with_entry_buysell_profit_values,
    calculate_trading_profit_torch_with_entry_buysell,
)

SPREAD = 1.0008711461252937
TOTO_CI_GUARD_MULTIPLIER = float(os.getenv("TOTO_CI_GUARD_MULTIPLIER", "1.0"))
_FORCE_KRONOS_VALUES = {"1", "true", "yes", "on"}
_forced_kronos_logged_symbols = set()
_model_selection_log_state: Dict[str, Tuple[str, str]] = {}
_toto_params_log_state: Dict[str, Tuple[str, str]] = {}
_model_selection_cache: Dict[str, str] = {}
_toto_params_cache: Dict[str, dict] = {}
_kronos_params_cache: Dict[str, dict] = {}

_BOOL_TRUE = {"1", "true", "yes", "on"}
_GPU_FALLBACK_ENV = "MARKETSIM_ALLOW_CPU_FALLBACK"
_cpu_fallback_log_state: Set[Tuple[str, Optional[str]]] = set()

pipeline: Optional[TotoPipeline] = None
TOTO_DEVICE_OVERRIDE: Optional[str] = None
kronos_wrapper_cache: Dict[tuple, KronosForecastingWrapper] = {}

ReturnSeries = Union[np.ndarray, pd.Series]


def _cpu_fallback_enabled() -> bool:
    value = os.getenv(_GPU_FALLBACK_ENV)
    if value is None:
        return False
    return value.strip().lower() in _BOOL_TRUE


def _in_test_mode() -> bool:
    """Return True when unit-test machinery requests lightweight behavior."""
    test_flag = os.getenv("TESTING")
    if test_flag is not None and test_flag.strip().lower() in _BOOL_TRUE:
        return True
    mock_flag = os.getenv("MARKETSIM_ALLOW_MOCK_ANALYTICS")
    if mock_flag is not None and mock_flag.strip().lower() in _BOOL_TRUE:
        return True
    return False


def _require_cuda(feature: str, *, symbol: Optional[str] = None, allow_cpu_fallback: bool = True) -> None:
    if torch.cuda.is_available():
        return
    if allow_cpu_fallback and _cpu_fallback_enabled():
        key = (feature, symbol)
        if key not in _cpu_fallback_log_state:
            target = f"{feature} ({symbol})" if symbol else feature
            logger.warning(
                "%s requires CUDA but only CPU is available; %s=1 detected so continuing in CPU fallback mode. "
                "Expect slower execution and reduced model fidelity.",
                target,
                _GPU_FALLBACK_ENV,
            )
            _cpu_fallback_log_state.add(key)
        return
    target = f"{feature} ({symbol})" if symbol else feature
    message = (
        f"{target} requires a CUDA-capable GPU. Install PyTorch 2.9 with CUDA 12.8 via "
        f"'uv pip install torch --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio' "
        "and verify drivers are configured."
    )
    if allow_cpu_fallback:
        message += f" You may set {_GPU_FALLBACK_ENV}=1 to run CPU-only for testing."
    raise RuntimeError(message)


@dataclass(frozen=True)
class StrategyEvaluation:
    total_return: float
    avg_daily_return: float
    annualized_return: float
    sharpe_ratio: float
    returns: ReturnSeries


def _mean_if_exists(df: pd.DataFrame, column: Optional[str]) -> Optional[float]:
    if not column or column not in df.columns:
        return None
    series = df[column]
    if series.empty:
        return None
    value = float(series.mean())
    if np.isnan(value):
        return None
    return value


def _fmt_number(value: Optional[float], precision: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{precision}f}"


def _format_table(headers: List[str], rows: List[List[str]], indent: str = "  ") -> str:
    if not rows:
        return ""
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))
    header_line = indent + " ".join(
        header.ljust(widths[idx]) for idx, header in enumerate(headers)
    )
    separator_line = indent + " ".join("-" * widths[idx] for idx in range(len(headers)))
    row_lines = [
        indent + " ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))
        for row in rows
    ]
    return "\n".join([header_line, separator_line, *row_lines])


def _log_table(title: str, headers: List[str], rows: List[List[str]]) -> None:
    body = _format_table(headers, rows)
    if not body:
        return
    logger.info(f"\n{title}\n{body}")


def _to_numpy_array(values: ReturnSeries) -> np.ndarray:
    if isinstance(values, pd.Series):
        array = values.to_numpy(dtype=float)
    else:
        array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        return array.reshape(1)
    return array


def _compute_return_profile(daily_returns: ReturnSeries, trading_days_per_year: int) -> Tuple[float, float]:
    if trading_days_per_year <= 0:
        return 0.0, 0.0
    returns_np = _to_numpy_array(daily_returns)
    if returns_np.size == 0:
        return 0.0, 0.0
    finite_mask = np.isfinite(returns_np)
    if not np.any(finite_mask):
        return 0.0, 0.0
    cleaned = returns_np[finite_mask]
    if cleaned.size == 0:
        return 0.0, 0.0
    avg_daily = float(np.mean(cleaned))
    annualized = float(avg_daily * trading_days_per_year)
    return avg_daily, annualized


def _evaluate_daily_returns(daily_returns: ReturnSeries, trading_days_per_year: int) -> StrategyEvaluation:
    returns_np = _to_numpy_array(daily_returns)
    if returns_np.size == 0:
        return StrategyEvaluation(
            total_return=0.0,
            avg_daily_return=0.0,
            annualized_return=0.0,
            sharpe_ratio=0.0,
            returns=returns_np,
        )

    total_return = float(np.sum(returns_np))
    std = float(np.std(returns_np))
    if std == 0.0 or not np.isfinite(std):
        sharpe = 0.0
    else:
        mean = float(np.mean(returns_np))
        sharpe = float((mean / std) * np.sqrt(max(trading_days_per_year, 1)))
    avg_daily, annualized = _compute_return_profile(returns_np, trading_days_per_year)
    return StrategyEvaluation(
        total_return=total_return,
        avg_daily_return=avg_daily,
        annualized_return=annualized,
        sharpe_ratio=sharpe,
        returns=returns_np,
    )


def evaluate_maxdiff_strategy(
    last_preds: Dict[str, torch.Tensor],
    simulation_data: pd.DataFrame,
    *,
    trading_fee: float,
    trading_days_per_year: int,
    is_crypto: bool = False,
) -> Tuple[StrategyEvaluation, np.ndarray, Dict[str, object]]:
    close_actual = torch.as_tensor(
        last_preds.get("close_actual_movement_values", torch.tensor([], dtype=torch.float32)),
        dtype=torch.float32,
    )
    if "close_actual_movement_values" not in last_preds:
        last_preds["close_actual_movement_values"] = close_actual
    validation_len = int(close_actual.numel())

    def _zero_metadata() -> Dict[str, object]:
        high_price = float(last_preds.get("high_predicted_price_value", 0.0))
        low_price = float(last_preds.get("low_predicted_price_value", 0.0))
        return {
            "maxdiffprofit_profit": 0.0,
            "maxdiffprofit_profit_values": [],
            "maxdiffprofit_profit_high_multiplier": 0.0,
            "maxdiffprofit_profit_low_multiplier": 0.0,
            "maxdiffprofit_high_price": high_price,
            "maxdiffprofit_low_price": low_price,
            "maxdiff_turnover": 0.0,
        }

    if validation_len == 0:
        eval_zero = StrategyEvaluation(
            total_return=0.0,
            avg_daily_return=0.0,
            annualized_return=0.0,
            sharpe_ratio=0.0,
            returns=np.zeros(0, dtype=float),
        )
        return eval_zero, eval_zero.returns, _zero_metadata()

    if len(simulation_data) < validation_len + 2:
        eval_zero = StrategyEvaluation(
            total_return=0.0,
            avg_daily_return=0.0,
            annualized_return=0.0,
            sharpe_ratio=0.0,
            returns=np.zeros(0, dtype=float),
        )
        return eval_zero, eval_zero.returns, _zero_metadata()

    high_series = simulation_data["High"].iloc[-(validation_len + 2):-2]
    low_series = simulation_data["Low"].iloc[-(validation_len + 2):-2]
    close_series = simulation_data["Close"].iloc[-(validation_len + 2):-2]

    if len(high_series) != validation_len:
        high_series = simulation_data["High"].tail(validation_len)
        low_series = simulation_data["Low"].tail(validation_len)
        close_series = simulation_data["Close"].tail(validation_len)

    close_vals = close_series.to_numpy(dtype=float)
    high_vals = high_series.to_numpy(dtype=float)
    low_vals = low_series.to_numpy(dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        close_to_high_np = np.abs(1.0 - np.divide(high_vals, close_vals, out=np.zeros_like(high_vals), where=close_vals != 0.0))
        close_to_low_np = np.abs(1.0 - np.divide(low_vals, close_vals, out=np.zeros_like(low_vals), where=close_vals != 0.0))
    close_to_high_np = np.nan_to_num(close_to_high_np, nan=0.0, posinf=0.0, neginf=0.0)
    close_to_low_np = np.nan_to_num(close_to_low_np, nan=0.0, posinf=0.0, neginf=0.0)

    close_to_high = torch.tensor(close_to_high_np, dtype=torch.float32)
    close_to_low = torch.tensor(close_to_low_np, dtype=torch.float32)

    high_actual_values = last_preds.get("high_actual_movement_values")
    low_actual_values = last_preds.get("low_actual_movement_values")
    high_pred_values = last_preds.get("high_predictions")
    low_pred_values = last_preds.get("low_predictions")

    if (
        high_actual_values is None
        or low_actual_values is None
        or high_pred_values is None
        or low_pred_values is None
    ):
        logger.warning(
            "MaxDiff strategy skipped: missing prediction arrays "
            "(high_actual=%s, low_actual=%s, high_pred=%s, low_pred=%s)",
            "None" if high_actual_values is None else type(high_actual_values).__name__,
            "None" if low_actual_values is None else type(low_actual_values).__name__,
            "None" if high_pred_values is None else type(high_pred_values).__name__,
            "None" if low_pred_values is None else type(low_pred_values).__name__,
        )
        eval_zero = StrategyEvaluation(
            total_return=0.0,
            avg_daily_return=0.0,
            annualized_return=0.0,
            sharpe_ratio=0.0,
            returns=np.zeros(0, dtype=float),
        )
        return eval_zero, eval_zero.returns, _zero_metadata()

    high_actual_base = torch.as_tensor(high_actual_values, dtype=torch.float32)
    low_actual_base = torch.as_tensor(low_actual_values, dtype=torch.float32)
    high_pred_base = torch.as_tensor(high_pred_values, dtype=torch.float32)
    low_pred_base = torch.as_tensor(low_pred_values, dtype=torch.float32)

    high_actual = high_actual_base + close_to_high
    low_actual = low_actual_base - close_to_low
    high_pred = high_pred_base + close_to_high
    low_pred = low_pred_base - close_to_low

    with torch.no_grad():
        maxdiff_trades = torch.where(
            torch.abs(high_pred) > torch.abs(low_pred),
            torch.ones_like(high_pred),
            -torch.ones_like(high_pred),
        )
        if is_crypto:
            maxdiff_trades = torch.where(maxdiff_trades < 0, torch.zeros_like(maxdiff_trades), maxdiff_trades)

        base_profit_values = calculate_profit_torch_with_entry_buysell_profit_values(
            close_actual,
            high_actual,
            high_pred,
            low_actual,
            low_pred,
            maxdiff_trades,
        )

        best_high_multiplier = 0.0
        best_high_profit = float(base_profit_values.sum().item())

        for multiplier in np.linspace(-0.03, 0.03, 500):
            profit = calculate_trading_profit_torch_with_entry_buysell(
                None,
                None,
                close_actual,
                maxdiff_trades,
                high_actual,
                high_pred + float(multiplier),
                low_actual,
                low_pred,
            ).item()
            if profit > best_high_profit:
                best_high_profit = float(profit)
                best_high_multiplier = float(multiplier)

        adjusted_high_pred = high_pred + best_high_multiplier

        best_low_multiplier = 0.0
        best_low_profit = best_high_profit
        for multiplier in np.linspace(-0.03, 0.03, 500):
            profit = calculate_trading_profit_torch_with_entry_buysell(
                None,
                None,
                close_actual,
                maxdiff_trades,
                high_actual,
                adjusted_high_pred,
                low_actual,
                low_pred + float(multiplier),
            ).item()
            if profit > best_low_profit:
                best_low_profit = float(profit)
                best_low_multiplier = float(multiplier)

        final_profit_values = calculate_profit_torch_with_entry_buysell_profit_values(
            close_actual,
            high_actual,
            adjusted_high_pred,
            low_actual,
            low_pred + best_low_multiplier,
            maxdiff_trades,
        )

    daily_returns_np = final_profit_values.detach().cpu().numpy().astype(float, copy=False)
    evaluation = _evaluate_daily_returns(daily_returns_np, trading_days_per_year)

    high_price_reference = float(last_preds.get("high_predicted_price_value", 0.0))
    low_price_reference = float(last_preds.get("low_predicted_price_value", 0.0))
    metadata = {
        "maxdiffprofit_profit": evaluation.total_return,
        "maxdiffprofit_profit_values": daily_returns_np.tolist(),
        "maxdiffprofit_profit_high_multiplier": best_high_multiplier,
        "maxdiffprofit_profit_low_multiplier": best_low_multiplier,
        "maxdiffprofit_high_price": high_price_reference * (1.0 + best_high_multiplier),
        "maxdiffprofit_low_price": low_price_reference * (1.0 + best_low_multiplier),
        "maxdiff_turnover": float(np.mean(np.abs(daily_returns_np))) if daily_returns_np.size else 0.0,
    }

    return evaluation, daily_returns_np, metadata


def _log_strategy_summary(results_df: pd.DataFrame, symbol: str, num_simulations: int) -> None:
    strategy_specs = [
        ("Simple", "simple_strategy_return", "simple_strategy_sharpe", "simple_strategy_finalday"),
        ("All Signals", "all_signals_strategy_return", "all_signals_strategy_sharpe", "all_signals_strategy_finalday"),
        ("Buy & Hold", "buy_hold_return", "buy_hold_sharpe", "buy_hold_finalday"),
        (
            "Unprofit Shutdown",
            "unprofit_shutdown_return",
            "unprofit_shutdown_sharpe",
            "unprofit_shutdown_finalday",
        ),
        ("Entry+Takeprofit", "entry_takeprofit_return", "entry_takeprofit_sharpe", "entry_takeprofit_finalday"),
        ("Highlow", "highlow_return", "highlow_sharpe", "highlow_finalday_return"),
        ("MaxDiff", "maxdiff_return", "maxdiff_sharpe", "maxdiff_finalday_return"),
        ("CI Guard", "ci_guard_return", "ci_guard_sharpe", None),
    ]

    rows: List[List[str]] = []
    for name, return_col, sharpe_col, final_col in strategy_specs:
        return_val = _mean_if_exists(results_df, return_col)
        sharpe_val = _mean_if_exists(results_df, sharpe_col)
        final_val = _mean_if_exists(results_df, final_col) if final_col else None
        if return_val is None and sharpe_val is None and (final_col is None or final_val is None):
            continue
        row = [
            name,
            _fmt_number(return_val),
            _fmt_number(sharpe_val),
            _fmt_number(final_val),
        ]
        rows.append(row)

    if not rows:
        return

    headers = ["Strategy", "Return", "Sharpe", "FinalDay"]
    title = f"Backtest summary for {symbol} ({num_simulations} simulations)"
    _log_table(title, headers, rows)


def _log_validation_losses(results_df: pd.DataFrame) -> None:
    loss_specs = [
        ("Close Val Loss", "close_val_loss"),
        ("High Val Loss", "high_val_loss"),
        ("Low Val Loss", "low_val_loss"),
    ]
    rows = [
        [label, _fmt_number(_mean_if_exists(results_df, column))]
        for label, column in loss_specs
        if column in results_df.columns
    ]
    if not rows:
        return
    # Skip logging if every value is missing, to avoid noise.
    if all(cell == "-" for _, cell in rows):
        return
    _log_table("Average validation losses", ["Metric", "Value"], rows)


def compute_walk_forward_stats(results_df: pd.DataFrame) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    if results_df.empty:
        return stats
    stats["walk_forward_oos_sharpe"] = float(results_df.get("simple_strategy_sharpe", pd.Series(dtype=float)).mean())
    stats["walk_forward_turnover"] = float(results_df.get("simple_strategy_return", pd.Series(dtype=float)).abs().mean())
    if "highlow_sharpe" in results_df:
        stats["walk_forward_highlow_sharpe"] = float(results_df["highlow_sharpe"].mean())
    if "entry_takeprofit_sharpe" in results_df:
        stats["walk_forward_takeprofit_sharpe"] = float(results_df["entry_takeprofit_sharpe"].mean())
    if "maxdiff_sharpe" in results_df:
        stats["walk_forward_maxdiff_sharpe"] = float(results_df["maxdiff_sharpe"].mean())
    return stats


def calibrate_signal(predictions: np.ndarray, actual_returns: np.ndarray) -> Tuple[float, float]:
    matched = min(len(predictions), len(actual_returns))
    if matched > 1:
        slope, intercept = np.polyfit(predictions[:matched], actual_returns[:matched], 1)
    else:
        slope, intercept = 1.0, 0.0
    return float(slope), float(intercept)

if __name__ == "__main__" and "REAL_TESTING" not in os.environ:
    os.environ["REAL_TESTING"] = "1"
    logger.info("REAL_TESTING not set; defaulting to enabled for standalone execution.")

FAST_TESTING = os.getenv("FAST_TESTING", "0").strip().lower() in _BOOL_TRUE
REAL_TESTING = os.getenv("REAL_TESTING", "0").strip().lower() in _BOOL_TRUE

_maybe_enable_fast_torch_settings()

COMPILED_MODELS_DIR = Path(os.getenv("COMPILED_MODELS_DIR", "compiled_models"))
INDUCTOR_CACHE_DIR = COMPILED_MODELS_DIR / "torch_inductor"


def _ensure_compilation_artifacts() -> None:
    try:
        COMPILED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        INDUCTOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(INDUCTOR_CACHE_DIR))
    except Exception as exc:  # pragma: no cover - filesystem best effort
        logger.debug("Failed to prepare torch.compile artifact directories: %s", exc)

FAST_TOTO_PARAMS = {
    "num_samples": int(os.getenv("FAST_TOTO_NUM_SAMPLES", "2048")),
    "samples_per_batch": int(os.getenv("FAST_TOTO_SAMPLES_PER_BATCH", "256")),
    "aggregate": os.getenv("FAST_TOTO_AGG_SPEC", "quantile_0.35"),
}
if FAST_TESTING:
    logger.info(
        "FAST_TESTING enabled — using Toto fast-path defaults (num_samples=%d, samples_per_batch=%d, aggregate=%s).",
        FAST_TOTO_PARAMS["num_samples"],
        FAST_TOTO_PARAMS["samples_per_batch"],
        FAST_TOTO_PARAMS["aggregate"],
    )

if REAL_TESTING:
    _ensure_compilation_artifacts()


def _is_force_kronos_enabled() -> bool:
    return os.getenv("MARKETSIM_FORCE_KRONOS", "0").lower() in _FORCE_KRONOS_VALUES


def _maybe_empty_cuda_cache() -> None:
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
    except Exception as exc:  # pragma: no cover - best effort cleanup
        logger.debug("Failed to empty CUDA cache: %s", exc)


def _drop_toto_pipeline() -> None:
    global pipeline
    if pipeline is None:
        return
    unload = getattr(pipeline, "unload", None)
    if callable(unload):
        try:
            unload()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug("Toto pipeline unload raised error: %s", exc)
    else:  # pragma: no cover - compatibility path if unload missing
        model = getattr(pipeline, "model", None)
        move_to_cpu = getattr(model, "to", None)
        if callable(move_to_cpu):
            try:
                move_to_cpu("cpu")
            except Exception as exc:
                logger.debug("Failed to move Toto model to CPU: %s", exc)
    pipeline = None
    _maybe_empty_cuda_cache()


def _drop_kronos_wrappers() -> None:
    if not kronos_wrapper_cache:
        return
    for wrapper in list(kronos_wrapper_cache.values()):
        unload = getattr(wrapper, "unload", None)
        if callable(unload):
            try:
                unload()
            except Exception as exc:  # pragma: no cover - cleanup best effort
                logger.debug("Kronos wrapper unload raised error: %s", exc)
    kronos_wrapper_cache.clear()
    _maybe_empty_cuda_cache()


def _reset_model_caches() -> None:
    """Accessible from tests to clear any in-process caches."""
    _drop_toto_pipeline()
    _drop_kronos_wrappers()
    _model_selection_cache.clear()
    _toto_params_cache.clear()
    _kronos_params_cache.clear()
    _model_selection_log_state.clear()
    _toto_params_log_state.clear()
    _forced_kronos_logged_symbols.clear()
    _cpu_fallback_log_state.clear()


def release_model_resources() -> None:
    """Public helper to free GPU-resident inference models between runs."""
    _drop_toto_pipeline()
    _drop_kronos_wrappers()


@disk_cache
def cached_predict(context, prediction_length, num_samples, samples_per_batch, *, symbol: Optional[str] = None):
    pipeline_instance = load_toto_pipeline()
    inference_mode_ctor = getattr(torch, "inference_mode", None)
    context_manager = inference_mode_ctor() if callable(inference_mode_ctor) else torch.no_grad()
    with context_manager:
        return pipeline_instance.predict(
            context=context,
            prediction_length=prediction_length,
            num_samples=num_samples,
            samples_per_batch=samples_per_batch,
        )


def _compute_toto_forecast(
    symbol: str,
    target_key: str,
    price_frame: pd.DataFrame,
    current_last_price: float,
    toto_params: dict,
):
    """
    Generate Toto forecasts for a prepared price frame.
    Returns (predictions_tensor, band_tensor, predicted_absolute_last).
    """
    predictions_list: List[float] = []
    band_list: List[float] = []
    max_horizon = 7

    if price_frame.empty:
        return torch.zeros(1, dtype=torch.float32), torch.zeros(1, dtype=torch.float32), float(current_last_price)

    # Toto expects a context vector of historical targets; walk forward to build forecasts.
    for pred_idx in reversed(range(1, max_horizon + 1)):
        if len(price_frame) <= pred_idx:
            continue
        current_context = price_frame[:-pred_idx]
        if current_context.empty:
            continue
        context = torch.tensor(current_context["y"].values, dtype=torch.float32)
        requested_num_samples = int(toto_params["num_samples"])
        requested_batch = int(toto_params["samples_per_batch"])

        attempts = 0
        cpu_fallback_used = False
        global TOTO_DEVICE_OVERRIDE
        while True:
            requested_num_samples, requested_batch = _normalise_sampling_params(
                requested_num_samples,
                requested_batch,
            )
            toto_params["num_samples"] = requested_num_samples
            toto_params["samples_per_batch"] = requested_batch
            _toto_params_cache[symbol] = toto_params.copy()
            try:
                forecast = cached_predict(
                    context,
                    1,
                    num_samples=requested_num_samples,
                    samples_per_batch=requested_batch,
                    symbol=symbol,
                )
                break
            except RuntimeError as exc:
                if not _is_cuda_oom_error(exc) or attempts >= TOTO_BACKTEST_MAX_RETRIES:
                    if not _is_cuda_oom_error(exc):
                        raise
                    if cpu_fallback_used:
                        raise
                    logger.warning(
                        "Toto forecast OOM for %s %s after %d GPU retries; falling back to CPU inference.",
                        symbol,
                        target_key,
                        attempts,
                    )
                    cpu_fallback_used = True
                    TOTO_DEVICE_OVERRIDE = "cpu"
                    _drop_toto_pipeline()
                    attempts = 0
                    requested_num_samples = max(TOTO_MIN_NUM_SAMPLES, requested_num_samples // 2)
                    requested_batch = max(TOTO_MIN_SAMPLES_PER_BATCH, requested_batch // 2)
                    continue
                attempts += 1
                requested_num_samples = max(
                    TOTO_MIN_NUM_SAMPLES,
                    requested_num_samples // 2,
                )
                requested_batch = max(
                    TOTO_MIN_SAMPLES_PER_BATCH,
                    min(requested_batch // 2, requested_num_samples),
                )
                logger.warning(
                    "Toto forecast OOM for %s %s; retrying with num_samples=%d, samples_per_batch=%d (attempt %d/%d).",
                    symbol,
                    target_key,
                    requested_num_samples,
                    requested_batch,
                    attempts,
                    TOTO_BACKTEST_MAX_RETRIES,
                )
                continue

        updated_params = _apply_toto_runtime_feedback(symbol, toto_params, requested_num_samples, requested_batch)
        if updated_params is not None:
            toto_params = updated_params
        tensor = forecast[0]
        numpy_method = getattr(tensor, "numpy", None)
        if callable(numpy_method):
            try:
                array_data = numpy_method()
            except Exception:
                array_data = None
        else:
            array_data = None

        if array_data is None:
            detach_method = getattr(tensor, "detach", None)
            if callable(detach_method):
                try:
                    array_data = detach_method().cpu().numpy()
                except Exception:
                    array_data = None

        if array_data is None:
            array_data = tensor

        distribution = np.asarray(array_data, dtype=np.float32).reshape(-1)
        if distribution.size == 0:
            distribution = np.zeros(1, dtype=np.float32)

        lower_q = np.percentile(distribution, 40)
        upper_q = np.percentile(distribution, 60)
        band_width = float(max(upper_q - lower_q, 0.0))
        band_list.append(band_width)

        aggregated = aggregate_with_spec(distribution, toto_params["aggregate"])
        predictions_list.append(float(np.atleast_1d(aggregated)[0]))

    if not predictions_list:
        predictions_list = [0.0]
    if not band_list:
        band_list = [0.0]

    predictions = torch.tensor(predictions_list, dtype=torch.float32)
    bands = torch.tensor(band_list, dtype=torch.float32)
    predicted_absolute_last = float(current_last_price * (1.0 + predictions[-1].item()))
    return predictions, bands, predicted_absolute_last


def _compute_avg_dollar_volume(df: pd.DataFrame, window: int = 20) -> Optional[float]:
    if "Close" not in df.columns or "Volume" not in df.columns:
        return None
    tail = df.tail(window)
    if tail.empty:
        return None
    try:
        dollar_vol = tail["Close"].astype(float) * tail["Volume"].astype(float)
    except Exception:
        return None
    mean_val = dollar_vol.mean()
    if pd.isna(mean_val):
        return None
    return float(mean_val)


def _compute_atr_pct(df: pd.DataFrame, window: int = 14) -> Optional[float]:
    required_cols = {"High", "Low", "Close"}
    if not required_cols.issubset(df.columns):
        return None
    if len(df) < window + 1:
        return None
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            (high - low),
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr_series = true_range.rolling(window=window).mean()
    if atr_series.empty or pd.isna(atr_series.iloc[-1]):
        return None
    last_close = close.iloc[-1]
    if last_close <= 0:
        return None
    atr_pct = float((atr_series.iloc[-1] / last_close) * 100.0)
    return atr_pct


TOTO_MODEL_ID = os.getenv("TOTO_MODEL_ID", "Datadog/Toto-Open-Base-1.0")
DEFAULT_TOTO_NUM_SAMPLES = int(os.getenv("TOTO_NUM_SAMPLES", "3072"))
DEFAULT_TOTO_SAMPLES_PER_BATCH = int(os.getenv("TOTO_SAMPLES_PER_BATCH", "384"))
DEFAULT_TOTO_AGG_SPEC = os.getenv("TOTO_AGGREGATION_SPEC", "trimmed_mean_10")


def _read_int_env(name: str, default: int, *, minimum: int = 1) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return max(minimum, default)
    return max(minimum, value)


TOTO_MIN_SAMPLES_PER_BATCH = _read_int_env("MARKETSIM_TOTO_MIN_SAMPLES_PER_BATCH", 32)
TOTO_MIN_NUM_SAMPLES = _read_int_env("MARKETSIM_TOTO_MIN_NUM_SAMPLES", 128)
if TOTO_MIN_NUM_SAMPLES < TOTO_MIN_SAMPLES_PER_BATCH:
    TOTO_MIN_NUM_SAMPLES = TOTO_MIN_SAMPLES_PER_BATCH

TOTO_MAX_SAMPLES_PER_BATCH = _read_int_env("MARKETSIM_TOTO_MAX_SAMPLES_PER_BATCH", 512)
if TOTO_MAX_SAMPLES_PER_BATCH < TOTO_MIN_SAMPLES_PER_BATCH:
    TOTO_MAX_SAMPLES_PER_BATCH = TOTO_MIN_SAMPLES_PER_BATCH

TOTO_MAX_NUM_SAMPLES = _read_int_env("MARKETSIM_TOTO_MAX_NUM_SAMPLES", 4096)
if TOTO_MAX_NUM_SAMPLES < TOTO_MIN_NUM_SAMPLES:
    TOTO_MAX_NUM_SAMPLES = max(TOTO_MIN_NUM_SAMPLES, DEFAULT_TOTO_NUM_SAMPLES)

TOTO_MAX_OOM_RETRIES = _read_int_env("MARKETSIM_TOTO_MAX_OOM_RETRIES", 4, minimum=0)
TOTO_BACKTEST_MAX_RETRIES = _read_int_env("MARKETSIM_TOTO_BACKTEST_MAX_RETRIES", 3, minimum=0)

_toto_runtime_adjust_log_state: Dict[str, Tuple[int, int]] = {}


def _clamp_toto_params(symbol: str, params: dict) -> dict:
    """Clamp Toto runtime parameters to safe bounds and log adjustments."""
    original = (int(params.get("num_samples", 0)), int(params.get("samples_per_batch", 0)))
    num_samples = int(params.get("num_samples", DEFAULT_TOTO_NUM_SAMPLES))
    samples_per_batch = int(params.get("samples_per_batch", DEFAULT_TOTO_SAMPLES_PER_BATCH))

    num_samples = max(TOTO_MIN_NUM_SAMPLES, min(TOTO_MAX_NUM_SAMPLES, num_samples))
    samples_per_batch = max(
        TOTO_MIN_SAMPLES_PER_BATCH,
        min(TOTO_MAX_SAMPLES_PER_BATCH, samples_per_batch, num_samples),
    )

    params["num_samples"] = num_samples
    params["samples_per_batch"] = samples_per_batch

    adjusted = (num_samples, samples_per_batch)
    if adjusted != original:
        state = _toto_runtime_adjust_log_state.get(symbol)
        if state != adjusted:
            logger.info(
                "Adjusted Toto sampling bounds for %s: num_samples=%d, samples_per_batch=%d (was %d/%d).",
                symbol,
                num_samples,
                samples_per_batch,
                original[0],
                original[1],
            )
            _toto_runtime_adjust_log_state[symbol] = adjusted
    return params


def _apply_toto_runtime_feedback(
    symbol: Optional[str],
    params: dict,
    requested_num_samples: int,
    requested_batch: int,
) -> Optional[dict]:
    """Update cached Toto params after runtime OOM fallback."""
    if symbol is None:
        return None
    pipeline_instance = pipeline
    if pipeline_instance is None:
        return None
    metadata = getattr(pipeline_instance, "last_run_metadata", None)
    if not metadata:
        return None
    used_samples = int(metadata.get("num_samples_used") or 0)
    used_batch = int(metadata.get("samples_per_batch_used") or 0)
    if used_samples <= 0 or used_batch <= 0:
        return None
    used_batch = min(used_samples, used_batch)
    if used_samples == requested_num_samples and used_batch == requested_batch:
        return None

    updated = params.copy()
    updated["num_samples"] = used_samples
    updated["samples_per_batch"] = used_batch
    updated = _clamp_toto_params(symbol, updated)
    params.update(updated)
    _toto_params_cache[symbol] = updated.copy()
    _toto_params_log_state[symbol] = ("runtime_adjusted", repr((used_samples, used_batch)))
    logger.info(
        "Cached Toto params adjusted after runtime fallback for %s: requested %d/%d, using %d/%d.",
        symbol,
        requested_num_samples,
        requested_batch,
        used_samples,
        used_batch,
    )
    return updated


def _is_cuda_oom_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    if "out of memory" in message:
        return True
    cuda_module = getattr(torch, "cuda", None)
    oom_error = getattr(cuda_module, "OutOfMemoryError", None) if cuda_module else None
    if oom_error is not None and isinstance(exc, oom_error):
        return True
    return False


def _normalise_sampling_params(num_samples: int, samples_per_batch: int) -> Tuple[int, int]:
    """Ensure Toto sampling params satisfy divisibility and configured bounds."""
    num_samples = max(TOTO_MIN_NUM_SAMPLES, min(TOTO_MAX_NUM_SAMPLES, num_samples))
    samples_per_batch = max(TOTO_MIN_SAMPLES_PER_BATCH, min(samples_per_batch, num_samples))
    if samples_per_batch <= 0:
        samples_per_batch = TOTO_MIN_SAMPLES_PER_BATCH
    if num_samples < samples_per_batch:
        num_samples = samples_per_batch
    remainder = num_samples % samples_per_batch
    if remainder != 0:
        num_samples -= remainder
        if num_samples < samples_per_batch:
            num_samples = samples_per_batch
    return num_samples, samples_per_batch


DEFAULT_KRONOS_PARAMS = {
    "temperature": 0.152,
    "top_p": 0.83,
    "top_k": 20,
    "sample_count": 192,
    "max_context": 232,
    "clip": 1.85,
}


def resolve_toto_params(symbol: str) -> dict:
    if FAST_TESTING:
        params = _clamp_toto_params(symbol, FAST_TOTO_PARAMS.copy())
        state = ("fast", repr(sorted(params.items())))
        if _toto_params_log_state.get(symbol) != state:
            logger.info(f"FAST_TESTING active — using fast Toto hyperparameters for {symbol}.")
            _toto_params_log_state[symbol] = state
        _toto_params_cache[symbol] = params
        return params.copy()

    cached = _toto_params_cache.get(symbol)
    if cached is not None:
        return cached.copy()
    record = load_best_config("toto", symbol)
    config = record.config if record else {}
    if record is None:
        state = ("defaults", "toto")
        if _toto_params_log_state.get(symbol) != state:
            logger.info(f"No stored Toto hyperparameters for {symbol} — using defaults.")
            _toto_params_log_state[symbol] = state
    else:
        state = ("loaded", repr(sorted(config.items())))
        if _toto_params_log_state.get(symbol) != state:
            logger.info(f"Loaded Toto hyperparameters for {symbol} from hyperparamstore.")
            _toto_params_log_state[symbol] = state
    params = {
        "num_samples": int(config.get("num_samples", DEFAULT_TOTO_NUM_SAMPLES)),
        "samples_per_batch": int(config.get("samples_per_batch", DEFAULT_TOTO_SAMPLES_PER_BATCH)),
        "aggregate": config.get("aggregate", DEFAULT_TOTO_AGG_SPEC),
    }
    params = _clamp_toto_params(symbol, params)
    _toto_params_cache[symbol] = params
    return params.copy()


def resolve_kronos_params(symbol: str) -> dict:
    cached = _kronos_params_cache.get(symbol)
    if cached is not None:
        return cached.copy()
    record = load_best_config("kronos", symbol)
    config = record.config if record else {}
    if record is None:
        logger.info(f"No stored Kronos hyperparameters for {symbol} — using defaults.")
    else:
        logger.info(f"Loaded Kronos hyperparameters for {symbol} from hyperparamstore.")
    params = DEFAULT_KRONOS_PARAMS.copy()
    params.update({k: config.get(k, params[k]) for k in params})
    env_sample_count = os.getenv("MARKETSIM_KRONOS_SAMPLE_COUNT")
    if env_sample_count:
        try:
            override = max(1, int(env_sample_count))
        except ValueError:
            logger.warning(
                "Ignoring invalid MARKETSIM_KRONOS_SAMPLE_COUNT=%r; expected positive integer.",
                env_sample_count,
            )
        else:
            if params.get("sample_count") != override:
                logger.info(
                    f"MARKETSIM_KRONOS_SAMPLE_COUNT active — overriding sample_count to {override} for {symbol}."
                )
            params["sample_count"] = override
    _kronos_params_cache[symbol] = params
    return params.copy()


def resolve_best_model(symbol: str) -> str:
    if _in_test_mode():
        cached = _model_selection_cache.get(symbol)
        if cached == "toto":
            return cached
        _model_selection_cache[symbol] = "toto"
        state = ("test-mode", "toto")
        if _model_selection_log_state.get(symbol) != state:
            logger.info("TESTING mode active — forcing Toto model for %s.", symbol)
            _model_selection_log_state[symbol] = state
        return "toto"
    if _is_force_kronos_enabled():
        _model_selection_cache.pop(symbol, None)
        if symbol not in _forced_kronos_logged_symbols:
            logger.info(f"MARKETSIM_FORCE_KRONOS active — forcing Kronos model for {symbol}.")
            _forced_kronos_logged_symbols.add(symbol)
        return "kronos"
    cached = _model_selection_cache.get(symbol)
    if cached is not None:
        return cached
    selection = load_model_selection(symbol)
    if selection is None:
        state = ("default", "toto")
        if _model_selection_log_state.get(symbol) != state:
            logger.info(f"No best-model selection for {symbol} — defaulting to Toto.")
            _model_selection_log_state[symbol] = state
        model = "toto"
    else:
        model = selection.get("model", "toto").lower()
        state = ("selection", model)
        if _model_selection_log_state.get(symbol) != state:
            logger.info(f"Selected model for {symbol}: {model} (source: hyperparamstore)")
            _model_selection_log_state[symbol] = state
    _model_selection_cache[symbol] = model
    return model


def pre_process_data(x_train: pd.DataFrame, key_to_predict: str) -> pd.DataFrame:
    """Minimal reimplementation to avoid heavy dependency on training module."""
    newdata = x_train.copy(deep=True)
    newdata[key_to_predict] = percent_movements_augment(newdata[key_to_predict].values.reshape(-1, 1))
    return newdata


def series_to_tensor(series_pd: pd.Series) -> torch.Tensor:
    """Convert a pandas series to a float tensor."""
    return torch.tensor(series_pd.values, dtype=torch.float32)

current_date_formatted = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
# test data on same dataset
if __name__ == "__main__":
    current_date_formatted = "2024-12-11-18-22-30"

print(f"current_date_formatted: {current_date_formatted}")

tb_writer = SummaryWriter(log_dir=f"./logs/{current_date_formatted}")


def load_toto_pipeline() -> TotoPipeline:
    """Lazily load the Toto forecasting pipeline."""
    global pipeline, TOTO_DEVICE_OVERRIDE
    _drop_kronos_wrappers()
    if pipeline is None:
        _maybe_enable_fast_torch_settings()
        preferred_device = "cuda" if torch.cuda.is_available() else "cpu"
        override_env = os.getenv("MARKETSIM_TOTO_DEVICE")
        override = TOTO_DEVICE_OVERRIDE
        if override_env:
            env_value = override_env.strip().lower()
            if env_value in {"cuda", "cpu"}:
                override = env_value
        device = override or preferred_device
        if device == "cuda":
            _require_cuda("Toto forecasting pipeline")
        else:
            logger.warning(
                "Toto forecasting pipeline running on CPU (override=%s); inference will be slower.",
                override or "auto",
            )
        logger.info(f"Loading Toto pipeline '{TOTO_MODEL_ID}' on {device}")

        compile_mode_env = (
            os.getenv("REAL_TOTO_COMPILE_MODE")
            or os.getenv("TOTO_COMPILE_MODE")
            or "max-autotune"
        )
        compile_mode = (compile_mode_env or "").strip() or "max-autotune"

        compile_backend_env = (
            os.getenv("REAL_TOTO_COMPILE_BACKEND")
            or os.getenv("TOTO_COMPILE_BACKEND")
            or "inductor"
        )
        compile_backend = (compile_backend_env or "").strip()
        if not compile_backend:
            compile_backend = None

        torch_dtype: Optional[torch.dtype] = torch.float32 if device == "cpu" else None
        if FAST_TESTING:
            torch_dtype = torch.float32

        disable_compile_flag = _read_env_flag(("TOTO_DISABLE_COMPILE", "MARKETSIM_TOTO_DISABLE_COMPILE"))
        enable_compile_flag = _read_env_flag(("TOTO_COMPILE", "MARKETSIM_TOTO_COMPILE"))
        torch_compile_enabled = device.startswith("cuda") and hasattr(torch, "compile")
        if disable_compile_flag is True:
            torch_compile_enabled = False
        elif enable_compile_flag is not None:
            torch_compile_enabled = bool(enable_compile_flag and hasattr(torch, "compile"))

        if torch_compile_enabled:
            _ensure_compilation_artifacts()
            logger.info(
                "Using torch.compile for Toto (mode=%s, backend=%s, cache_dir=%s).",
                compile_mode,
                compile_backend or "default",
                os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
            )
        else:
            if REAL_TESTING:
                logger.info(
                    "REAL_TESTING active but torch.compile disabled (available=%s, disable_flag=%s).",
                    hasattr(torch, "compile"),
                    disable_compile_flag,
                )
        if REAL_TESTING and device.startswith("cuda"):
            logger.info("REAL_TESTING active — defaulting to float32 inference (bf16 disabled due to accuracy guard).")

        pipeline = TotoPipeline.from_pretrained(
            model_id=TOTO_MODEL_ID,
            device_map=device,
            torch_dtype=torch_dtype,
            torch_compile=torch_compile_enabled,
            compile_mode=compile_mode,
            compile_backend=compile_backend,
            max_oom_retries=TOTO_MAX_OOM_RETRIES,
            min_samples_per_batch=TOTO_MIN_SAMPLES_PER_BATCH,
            min_num_samples=TOTO_MIN_NUM_SAMPLES,
        )
    return pipeline


def load_kronos_wrapper(params: Dict[str, float]) -> KronosForecastingWrapper:
    _drop_toto_pipeline()
    _maybe_enable_fast_torch_settings()
    _require_cuda("Kronos inference", allow_cpu_fallback=False)
    key = (
        params["temperature"],
        params["top_p"],
        params["top_k"],
        params["sample_count"],
        params["max_context"],
        params["clip"],
    )
    wrapper = kronos_wrapper_cache.get(key)
    if wrapper is None:
        wrapper = KronosForecastingWrapper(
            model_name="NeoQuasar/Kronos-base",
            tokenizer_name="NeoQuasar/Kronos-Tokenizer-base",
            device="cuda:0",
            max_context=int(params["max_context"]),
            clip=float(params["clip"]),
            temperature=float(params["temperature"]),
            top_p=float(params["top_p"]),
            top_k=int(params["top_k"]),
            sample_count=int(params["sample_count"]),
        )
        kronos_wrapper_cache[key] = wrapper
    return wrapper


def prepare_kronos_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    kronos_df = df.copy()
    if "Timestamp" in kronos_df.columns:
        kronos_df["timestamp"] = pd.to_datetime(kronos_df["Timestamp"])
    elif "Date" in kronos_df.columns:
        kronos_df["timestamp"] = pd.to_datetime(kronos_df["Date"])
    else:
        kronos_df["timestamp"] = pd.date_range(end=pd.Timestamp.utcnow(), periods=len(kronos_df), freq="D")
    return kronos_df


def simple_buy_sell_strategy(predictions, is_crypto=False):
    """Buy if predicted close is up; if not crypto, short if down."""
    predictions = torch.as_tensor(predictions)
    if is_crypto:
        # Prohibit shorts for crypto
        return (predictions > 0).float()
    # Otherwise allow buy (1) or sell (-1)
    return (predictions > 0).float() * 2 - 1


def all_signals_strategy(close_pred, high_pred, low_pred, is_crypto=False):
    """
    Buy if all signals are up; if not crypto, sell if all signals are down, else hold.
    If is_crypto=True, no short trades.
    """
    close_pred, high_pred, low_pred = map(torch.as_tensor, (close_pred, high_pred, low_pred))

    # For "buy" all must be > 0
    buy_signal = (close_pred > 0) & (high_pred > 0) & (low_pred > 0)
    if is_crypto:
        return buy_signal.float()

    # For non-crypto, "sell" all must be < 0
    sell_signal = (close_pred < 0) & (high_pred < 0) & (low_pred < 0)

    # Convert to -1, 0, 1
    return buy_signal.float() - sell_signal.float()


def buy_hold_strategy(predictions):
    """Buy when prediction is positive, hold otherwise."""
    predictions = torch.as_tensor(predictions)
    return (predictions > 0).float()


def unprofit_shutdown_buy_hold(predictions, actual_returns, is_crypto=False):
    """Buy and hold strategy that shuts down if the previous trade would have been unprofitable."""
    predictions = torch.as_tensor(predictions)
    signals = torch.ones_like(predictions)
    for i in range(1, len(signals)):
        if signals[i - 1] != 0.0:
            # Check if day i-1 was correct
            was_correct = (
                    (actual_returns[i - 1] > 0 and predictions[i - 1] > 0) or
                    (actual_returns[i - 1] < 0 and predictions[i - 1] < 0)
            )
            if was_correct:
                # Keep same signal direction as predictions[i]
                signals[i] = 1.0 if predictions[i] > 0 else -1.0 if predictions[i] < 0 else 0.0
            else:
                signals[i] = 0.0
        else:
            # If previously no position, open based on prediction direction
            signals[i] = 1.0 if predictions[i] > 0 else -1.0 if predictions[i] < 0 else 0.0
    # For crypto, replace negative signals with 0
    if is_crypto:
        signals[signals < 0] = 0.0
    return signals


def confidence_guard_strategy(
    close_predictions,
    ci_band,
    ci_multiplier: float = TOTO_CI_GUARD_MULTIPLIER,
    is_crypto: bool = False,
):
    """
    Guard entries by requiring the predicted move to exceed a confidence interval width.
    Shorts remain disabled for crypto symbols.
    """
    close_predictions = torch.as_tensor(close_predictions, dtype=torch.float32)
    ci_band = torch.as_tensor(ci_band, dtype=torch.float32)

    signals = torch.zeros_like(close_predictions)
    guard_width = torch.clamp(ci_band.abs(), min=1e-8) * float(ci_multiplier)

    buy_mask = close_predictions > guard_width
    signals = torch.where(buy_mask, torch.ones_like(signals), signals)

    if is_crypto:
        return signals

    sell_mask = close_predictions < -guard_width
    signals = torch.where(sell_mask, -torch.ones_like(signals), signals)
    return signals


def evaluate_strategy(
    strategy_signals,
    actual_returns,
    trading_fee,
    trading_days_per_year: int,
) -> StrategyEvaluation:
    global SPREAD
    """Evaluate the performance of a strategy, factoring in trading fees."""
    strategy_signals = strategy_signals.numpy()  # Convert to numpy array

    actual_returns = actual_returns.copy()
    sig_len = strategy_signals.shape[0]
    ret_len = len(actual_returns)
    if sig_len == 0 or ret_len == 0:
        return StrategyEvaluation(
            total_return=0.0,
            avg_daily_return=0.0,
            annualized_return=0.0,
            sharpe_ratio=0.0,
            returns=np.zeros(0, dtype=float),
        )
    if sig_len != ret_len:
        min_len = min(sig_len, ret_len)
        logger.warning(
            "Strategy/return length mismatch (signals=%s, returns=%s); truncating to %s",
            sig_len,
            ret_len,
            min_len,
        )
        strategy_signals = strategy_signals[-min_len:]
        actual_returns = actual_returns.iloc[-min_len:]

    # Calculate fees: apply fee for each trade (both buy and sell)
    # Adjust fees: only apply when position changes
    position_changes = np.diff(np.concatenate(([0], strategy_signals)))
    change_magnitude = np.abs(position_changes)

    has_long = np.any(strategy_signals > 0)
    has_short = np.any(strategy_signals < 0)
    has_flat = np.any(strategy_signals == 0)

    fee_per_change = trading_fee
    if has_long and has_short and has_flat:
        fee_per_change = trading_fee * 0.523
    spread_cost_per_change = abs((1 - SPREAD) / 2)
    fees = change_magnitude * (fee_per_change + spread_cost_per_change)
    # logger.info(f'adjusted fees: {fees}')

    # Adjust fees: only apply when position changes
    for i in range(1, len(fees)):
        if strategy_signals[i] == strategy_signals[i - 1]:
            fees[i] = 0

    # logger.info(f'fees after adjustment: {fees}')

    # Apply fees to the strategy returns
    signal_series = pd.Series(strategy_signals, index=actual_returns.index, dtype=float)
    fee_series = pd.Series(fees, index=actual_returns.index, dtype=float)
    gross_returns = signal_series * actual_returns
    strategy_returns = gross_returns - fee_series

    cumulative_returns = (1 + strategy_returns).cumprod() - 1
    total_return = float(cumulative_returns.iloc[-1])

    avg_daily_return, annualized_return = _compute_return_profile(strategy_returns, trading_days_per_year)

    strategy_std = strategy_returns.std()
    if strategy_std == 0 or np.isnan(strategy_std):
        sharpe_ratio = 0.0  # or some other default value
    else:
        sharpe_ratio = float(strategy_returns.mean() / strategy_std * np.sqrt(trading_days_per_year))

    return StrategyEvaluation(
        total_return=total_return,
        avg_daily_return=avg_daily_return,
        annualized_return=annualized_return,
        sharpe_ratio=sharpe_ratio,
        returns=strategy_returns
    )


def backtest_forecasts(symbol, num_simulations=100):
    # Download the latest data
    current_time_formatted = datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
    # use this for testing dataset
    if __name__ == "__main__":
        current_time_formatted = '2024-09-07--03-36-27'
    # current_time_formatted = '2024-04-18--06-14-26'  # new/ 30 minute data # '2022-10-14 09-58-20'
    # current_day_formatted = '2024-04-18'  # new/ 30 minute data # '2022-10-14 09-58-20'

    stock_data = download_daily_stock_data(current_time_formatted, symbols=[symbol])
    # hardcode repeatable time for testing
    # current_time_formatted = "2024-10-18--06-05-32"
    trading_fee = 0.0025

    # 8% margin lending

    # stock_data = download_daily_stock_data(current_time_formatted, symbols=symbols)
    # stock_data = pd.read_csv(f"./data/{current_time_formatted}/{symbol}-{current_day_formatted}.csv")

    base_dir = Path(__file__).parent
    data_dir = base_dir / "data" / current_time_formatted

    global SPREAD
    spread = fetch_spread(symbol)
    logger.info(f"spread: {spread}")
    previous_spread = SPREAD
    SPREAD = spread  #

    # stock_data = load_stock_data_from_csv(csv_file)

    try:
        if len(stock_data) < num_simulations:
            logger.warning(
                f"Not enough historical data for {num_simulations} simulations. Using {len(stock_data)} instead.")
            num_simulations = len(stock_data)

        results = []

        is_crypto = symbol in crypto_symbols

        for sim_number in range(num_simulations):
            simulation_data = stock_data.iloc[:-(sim_number + 1)].copy(deep=True)
            if simulation_data.empty:
                logger.warning(f"No data left for simulation {sim_number + 1}")
                continue

            result = run_single_simulation(
                simulation_data,
                symbol,
                trading_fee,
                is_crypto,
                sim_number,
                spread,
            )
            results.append(result)

        results_df = pd.DataFrame(results)
        walk_forward_stats = compute_walk_forward_stats(results_df)
        for key, value in walk_forward_stats.items():
            results_df[key] = value

        # Log final average metrics
        tb_writer.add_scalar(
            f'{symbol}/final_metrics/simple_avg_return',
            results_df['simple_strategy_avg_daily_return'].mean(),
            0,
        )
        tb_writer.add_scalar(
            f'{symbol}/final_metrics/simple_annual_return',
            results_df['simple_strategy_annual_return'].mean(),
            0,
        )
        tb_writer.add_scalar(f'{symbol}/final_metrics/simple_avg_sharpe', results_df['simple_strategy_sharpe'].mean(), 0)
        tb_writer.add_scalar(
            f'{symbol}/final_metrics/all_signals_avg_return',
            results_df['all_signals_strategy_avg_daily_return'].mean(),
            0,
        )
        tb_writer.add_scalar(
            f'{symbol}/final_metrics/all_signals_annual_return',
            results_df['all_signals_strategy_annual_return'].mean(),
            0,
        )
        tb_writer.add_scalar(f'{symbol}/final_metrics/all_signals_avg_sharpe',
                             results_df['all_signals_strategy_sharpe'].mean(), 0)
        tb_writer.add_scalar(
            f'{symbol}/final_metrics/buy_hold_avg_return',
            results_df['buy_hold_avg_daily_return'].mean(),
            0,
        )
        tb_writer.add_scalar(
            f'{symbol}/final_metrics/buy_hold_annual_return',
            results_df['buy_hold_annual_return'].mean(),
            0,
        )
        tb_writer.add_scalar(f'{symbol}/final_metrics/buy_hold_avg_sharpe', results_df['buy_hold_sharpe'].mean(), 0)
        tb_writer.add_scalar(
            f'{symbol}/final_metrics/unprofit_shutdown_avg_return',
            results_df['unprofit_shutdown_avg_daily_return'].mean(),
            0,
        )
        tb_writer.add_scalar(
            f'{symbol}/final_metrics/unprofit_shutdown_annual_return',
            results_df['unprofit_shutdown_annual_return'].mean(),
            0,
        )
        tb_writer.add_scalar(f'{symbol}/final_metrics/unprofit_shutdown_avg_sharpe',
                             results_df['unprofit_shutdown_sharpe'].mean(), 0)
        tb_writer.add_scalar(
            f'{symbol}/final_metrics/entry_takeprofit_avg_return',
            results_df['entry_takeprofit_avg_daily_return'].mean(),
            0,
        )
        tb_writer.add_scalar(
            f'{symbol}/final_metrics/entry_takeprofit_annual_return',
            results_df['entry_takeprofit_annual_return'].mean(),
            0,
        )
        tb_writer.add_scalar(f'{symbol}/final_metrics/entry_takeprofit_avg_sharpe',
                             results_df['entry_takeprofit_sharpe'].mean(), 0)
        tb_writer.add_scalar(
            f'{symbol}/final_metrics/highlow_avg_return',
            results_df['highlow_avg_daily_return'].mean(),
            0,
        )
        tb_writer.add_scalar(
            f'{symbol}/final_metrics/highlow_annual_return',
            results_df['highlow_annual_return'].mean(),
            0,
        )
        tb_writer.add_scalar(f'{symbol}/final_metrics/highlow_avg_sharpe', results_df['highlow_sharpe'].mean(), 0)
        tb_writer.add_scalar(
            f'{symbol}/final_metrics/ci_guard_avg_return',
            results_df['ci_guard_avg_daily_return'].mean(),
            0,
        )
        tb_writer.add_scalar(
            f'{symbol}/final_metrics/ci_guard_annual_return',
            results_df['ci_guard_annual_return'].mean(),
            0,
        )
        tb_writer.add_scalar(f'{symbol}/final_metrics/ci_guard_avg_sharpe', results_df['ci_guard_sharpe'].mean(), 0)

        _log_validation_losses(results_df)
        _log_strategy_summary(results_df, symbol, num_simulations)

        # Determine which strategy is best overall
        avg_simple = results_df["simple_strategy_return"].mean()
        avg_allsignals = results_df["all_signals_strategy_return"].mean()
        avg_takeprofit = results_df["entry_takeprofit_return"].mean()
        avg_highlow = results_df["highlow_return"].mean()
        avg_ci_guard = results_df["ci_guard_return"].mean()
        if "maxdiff_return" in results_df:
            avg_maxdiff = float(results_df["maxdiff_return"].mean())
            if not np.isfinite(avg_maxdiff):
                avg_maxdiff = float("-inf")
        else:
            avg_maxdiff = float("-inf")

        best_return = max(avg_simple, avg_allsignals, avg_takeprofit, avg_highlow, avg_ci_guard, avg_maxdiff)
        if best_return == avg_ci_guard:
            best_strategy = "ci_guard"
        elif best_return == avg_highlow:
            best_strategy = "highlow"
        elif best_return == avg_takeprofit:
            best_strategy = "takeprofit"
        elif best_return == avg_maxdiff:
            best_strategy = "maxdiff"
        elif best_return == avg_allsignals:
            best_strategy = "all_signals"
        else:
            best_strategy = "simple"

        # Record which strategy is best for this symbol & day
        set_strategy_for_symbol(symbol, best_strategy)

        return results_df
    finally:
        SPREAD = previous_spread



def run_single_simulation(simulation_data, symbol, trading_fee, is_crypto, sim_idx, spread):
    last_preds = {
        'instrument': symbol,
        'close_last_price': simulation_data['Close'].iloc[-1],
    }
    trading_days_per_year = 365 if is_crypto else 252

    spread_bps_estimate = float(abs(float(spread) - 1.0) * 1e4)
    last_preds["spread_bps_estimate"] = spread_bps_estimate

    avg_dollar_vol = _compute_avg_dollar_volume(simulation_data)
    if avg_dollar_vol is not None:
        last_preds["dollar_vol_20d"] = avg_dollar_vol
    atr_pct = _compute_atr_pct(simulation_data)
    if atr_pct is not None:
        last_preds["atr_pct_14"] = atr_pct

    best_model = resolve_best_model(symbol)
    use_kronos = best_model == "kronos"
    if use_kronos:
        _require_cuda("Kronos forecasting", symbol=symbol, allow_cpu_fallback=False)
    else:
        _require_cuda("Toto forecasting", symbol=symbol)

    try:
        toto_params = resolve_toto_params(symbol)
    except Exception as exc:
        logger.warning("Failed to resolve Toto parameters for %s: %s", symbol, exc)
        toto_params = None

    kronos_params: Optional[dict] = None
    kronos_wrapper: Optional[KronosForecastingWrapper] = None
    kronos_df: Optional[pd.DataFrame] = None
    kronos_init_logged = False

    def ensure_kronos_ready() -> bool:
        nonlocal kronos_params, kronos_wrapper, kronos_df, kronos_init_logged
        if kronos_wrapper is not None:
            return True
        try:
            if kronos_params is None:
                kronos_params = resolve_kronos_params(symbol)
            kronos_wrapper = load_kronos_wrapper(kronos_params)
            if kronos_df is None:
                kronos_df = prepare_kronos_dataframe(simulation_data)
            return True
        except Exception as exc:
            if not kronos_init_logged:
                logger.warning("Failed to prepare Kronos wrapper for %s: %s", symbol, exc)
                kronos_init_logged = True
            kronos_wrapper = None
            return False

    for key_to_predict in ['Close', 'Low', 'High', 'Open']:
        data = pre_process_data(simulation_data, key_to_predict)
        price = data[["Close", "High", "Low", "Open"]]

        price = price.rename(columns={"Date": "time_idx"})
        price["ds"] = pd.date_range(start="1949-01-01", periods=len(price), freq="D").values
        price['y'] = price[key_to_predict].shift(-1)
        price['trade_weight'] = (price["y"] > 0) * 2 - 1

        price.drop(price.tail(1).index, inplace=True)
        price['id'] = price.index
        price['unique_id'] = 1
        price = price.dropna()

        validation = price[-7:]
        current_last_price = float(simulation_data[key_to_predict].iloc[-1])

        toto_predictions = None
        toto_band = None
        toto_abs = None
        run_toto = toto_params is not None and not use_kronos
        if run_toto:
            try:
                toto_predictions, toto_band, toto_abs = _compute_toto_forecast(
                    symbol,
                    key_to_predict,
                    price,
                    current_last_price,
                    toto_params,
                )
            except Exception as exc:
                if key_to_predict == "Close":
                    logger.warning("Toto forecast failed for %s %s: %s", symbol, key_to_predict, exc)
                toto_predictions = None
                toto_band = None
                toto_abs = None

        kronos_predictions = None
        kronos_abs = None
        need_kronos = use_kronos or key_to_predict == "Close"
        if need_kronos and ensure_kronos_ready():
            try:
                kronos_results = kronos_wrapper.predict_series(
                    data=kronos_df,
                    timestamp_col="timestamp",
                    columns=[key_to_predict],
                    pred_len=7,
                    lookback=int(kronos_params["max_context"]),
                    temperature=float(kronos_params["temperature"]),
                    top_p=float(kronos_params["top_p"]),
                    top_k=int(kronos_params["top_k"]),
                    sample_count=int(kronos_params["sample_count"]),
                )
                kronos_entry = kronos_results.get(key_to_predict)
                if kronos_entry is not None and len(kronos_entry.percent) > 0:
                    kronos_predictions = torch.tensor(kronos_entry.percent, dtype=torch.float32)
                    kronos_abs = float(kronos_entry.absolute[-1])
            except Exception as exc:
                if key_to_predict == "Close":
                    logger.warning("Kronos forecast failed for %s %s: %s", symbol, key_to_predict, exc)
                kronos_predictions = None
                kronos_abs = None
                kronos_wrapper = None

        predictions = None
        predictions_source = None
        predicted_absolute_last = current_last_price

        if use_kronos and kronos_predictions is not None:
            predictions = kronos_predictions
            predictions_source = "kronos"
            if kronos_abs is not None:
                predicted_absolute_last = kronos_abs
        elif toto_predictions is not None:
            predictions = toto_predictions
            predictions_source = "toto"
            if toto_abs is not None:
                predicted_absolute_last = toto_abs
        elif kronos_predictions is not None:
            predictions = kronos_predictions
            predictions_source = "kronos"
            if kronos_abs is not None:
                predicted_absolute_last = kronos_abs
        else:
            logger.warning("No predictions produced for %s %s; skipping.", symbol, key_to_predict)
            continue

        actuals = series_to_tensor(validation["y"])
        trading_preds = (predictions[:-1] > 0) * 2 - 1

        prediction_np = predictions[:-1].detach().cpu().numpy()
        error = validation["y"][:-1].values - prediction_np
        mean_val_loss = np.abs(error).mean()

        tb_writer.add_scalar(f'{symbol}/{key_to_predict}/val_loss', mean_val_loss, sim_idx)

        last_preds[key_to_predict.lower() + "_last_price"] = current_last_price
        last_preds[key_to_predict.lower() + "_predicted_price"] = float(predictions[-1].item())
        last_preds[key_to_predict.lower() + "_predicted_price_value"] = predicted_absolute_last
        last_preds[key_to_predict.lower() + "_val_loss"] = mean_val_loss
        last_preds[key_to_predict.lower() + "_actual_movement_values"] = actuals[:-1].view(-1)
        last_preds[key_to_predict.lower() + "_trade_values"] = trading_preds.view(-1)
        last_preds[key_to_predict.lower() + "_predictions"] = predictions[:-1].view(-1)
        if key_to_predict == "Close":
            if toto_predictions is not None and toto_predictions.numel() > 0:
                last_preds["toto_close_pred_pct"] = float(toto_predictions[-1].item())
                if toto_band is not None:
                    last_preds["close_ci_band"] = toto_band
            if kronos_predictions is not None and kronos_predictions.numel() > 0:
                last_preds["kronos_close_pred_pct"] = float(kronos_predictions[-1].item())
            if "close_ci_band" not in last_preds:
                last_preds["close_ci_band"] = torch.zeros_like(predictions)
            last_preds["close_prediction_source"] = predictions_source or ("kronos" if use_kronos else "toto")
            last_preds["close_raw_pred_pct"] = float(predictions[-1].item())

    if "close_ci_band" not in last_preds:
        base_close_preds = torch.as_tensor(last_preds.get("close_predictions", torch.zeros(1)), dtype=torch.float32)
        pad_length = int(base_close_preds.shape[0] + 1)
        last_preds["close_ci_band"] = torch.zeros(pad_length, dtype=torch.float32)
    if "close_prediction_source" not in last_preds:
        last_preds["close_prediction_source"] = "kronos" if use_kronos else "toto"

    # Calculate actual percentage returns over the validation horizon
    close_window = simulation_data["Close"].iloc[-7:]
    actual_returns = close_window.pct_change().dropna().reset_index(drop=True)
    realized_vol_pct = float(actual_returns.std() * 100.0) if not actual_returns.empty else 0.0
    last_preds["realized_volatility_pct"] = realized_vol_pct
    close_pred_tensor = torch.as_tensor(last_preds.get("close_predictions", torch.zeros(1)), dtype=torch.float32)
    if "close_predictions" not in last_preds:
        last_preds["close_predictions"] = close_pred_tensor
    try:
        close_pred_np = close_pred_tensor.detach().cpu().numpy()
    except AttributeError:
        close_pred_np = np.asarray(close_pred_tensor, dtype=np.float32)
    actual_return_np = actual_returns.to_numpy()
    slope, intercept = calibrate_signal(close_pred_np, actual_return_np)
    raw_expected_move_pct = float(last_preds.get("close_raw_pred_pct", 0.0))
    calibrated_expected_move_pct = float(slope * raw_expected_move_pct + intercept)
    last_preds["calibration_slope"] = float(slope)
    last_preds["calibration_intercept"] = float(intercept)
    last_preds["raw_expected_move_pct"] = raw_expected_move_pct
    last_preds["calibrated_expected_move_pct"] = calibrated_expected_move_pct

    pred_length = int(close_pred_tensor.shape[0])

    def _ensure_tensor_key(key: str) -> torch.Tensor:
        value = last_preds.get(key)
        if value is None:
            tensor = torch.zeros(pred_length, dtype=torch.float32)
            last_preds[key] = tensor
            return tensor
        tensor = torch.as_tensor(value, dtype=torch.float32)
        if tensor.shape[0] != pred_length:
            tensor = tensor.reshape(-1)
        last_preds[key] = tensor
        return tensor

    high_preds_tensor = _ensure_tensor_key("high_predictions")
    low_preds_tensor = _ensure_tensor_key("low_predictions")
    _ensure_tensor_key("high_actual_movement_values")
    _ensure_tensor_key("low_actual_movement_values")

    maxdiff_eval, maxdiff_returns_np, maxdiff_metadata = evaluate_maxdiff_strategy(
        last_preds,
        simulation_data,
        trading_fee=trading_fee,
        trading_days_per_year=trading_days_per_year,
        is_crypto=is_crypto,
    )
    last_preds.update(maxdiff_metadata)
    maxdiff_return = maxdiff_eval.total_return
    maxdiff_sharpe = maxdiff_eval.sharpe_ratio
    maxdiff_avg_daily = maxdiff_eval.avg_daily_return
    maxdiff_annual = maxdiff_eval.annualized_return
    maxdiff_returns = maxdiff_returns_np
    maxdiff_finalday_return = float(maxdiff_returns[-1]) if maxdiff_returns.size else 0.0
    maxdiff_turnover = float(maxdiff_metadata.get("maxdiff_turnover", 0.0))

    # Simple buy/sell strategy
    simple_signals = simple_buy_sell_strategy(
        close_pred_tensor,
        is_crypto=is_crypto
    )
    simple_eval = evaluate_strategy(simple_signals, actual_returns, trading_fee, trading_days_per_year)
    simple_total_return = simple_eval.total_return
    simple_sharpe = simple_eval.sharpe_ratio
    simple_returns = simple_eval.returns
    simple_avg_daily = simple_eval.avg_daily_return
    simple_annual = simple_eval.annualized_return
    if actual_returns.empty:
        simple_finalday_return = 0.0
    else:
        simple_finalday_return = (simple_signals[-1].item() * actual_returns.iloc[-1]) - (2 * trading_fee * SPREAD)

    # All signals strategy
    all_signals = all_signals_strategy(
        close_pred_tensor,
        high_preds_tensor,
        low_preds_tensor,
        is_crypto=is_crypto
    )
    all_signals_eval = evaluate_strategy(all_signals, actual_returns, trading_fee, trading_days_per_year)
    all_signals_total_return = all_signals_eval.total_return
    all_signals_sharpe = all_signals_eval.sharpe_ratio
    all_signals_returns = all_signals_eval.returns
    all_signals_avg_daily = all_signals_eval.avg_daily_return
    all_signals_annual = all_signals_eval.annualized_return
    if actual_returns.empty:
        all_signals_finalday_return = 0.0
    else:
        all_signals_finalday_return = (all_signals[-1].item() * actual_returns.iloc[-1]) - (2 * trading_fee * SPREAD)

    # Buy and hold strategy
    buy_hold_signals = buy_hold_strategy(last_preds["close_predictions"])
    buy_hold_eval = evaluate_strategy(buy_hold_signals, actual_returns, trading_fee, trading_days_per_year)
    buy_hold_sharpe = buy_hold_eval.sharpe_ratio
    buy_hold_returns = buy_hold_eval.returns
    buy_hold_avg_daily = buy_hold_eval.avg_daily_return
    buy_hold_annual = buy_hold_eval.annualized_return
    if actual_returns.empty:
        buy_hold_return_expected = -trading_fee
        buy_hold_finalday_return = -trading_fee
    else:
        buy_hold_return_expected = (1 + actual_returns).prod() - 1 - trading_fee
        buy_hold_finalday_return = actual_returns.iloc[-1] - trading_fee
    buy_hold_return = buy_hold_return_expected

    # Unprofit shutdown buy and hold strategy
    unprofit_shutdown_signals = unprofit_shutdown_buy_hold(last_preds["close_predictions"], actual_returns, is_crypto=is_crypto)
    unprofit_shutdown_eval = evaluate_strategy(unprofit_shutdown_signals, actual_returns, trading_fee, trading_days_per_year)
    unprofit_shutdown_return = unprofit_shutdown_eval.total_return
    unprofit_shutdown_sharpe = unprofit_shutdown_eval.sharpe_ratio
    unprofit_shutdown_returns = unprofit_shutdown_eval.returns
    unprofit_shutdown_avg_daily = unprofit_shutdown_eval.avg_daily_return
    unprofit_shutdown_annual = unprofit_shutdown_eval.annualized_return
    unprofit_shutdown_finalday_return = (unprofit_shutdown_signals[-1].item() * actual_returns.iloc[-1]) - (2 * trading_fee * SPREAD)

    # Entry + takeprofit strategy
    entry_takeprofit_eval = evaluate_entry_takeprofit_strategy(
        last_preds["close_predictions"],
        last_preds["high_predictions"],
        last_preds["low_predictions"],
        last_preds["close_actual_movement_values"],
        last_preds["high_actual_movement_values"],
        last_preds["low_actual_movement_values"],
        trading_fee,
        trading_days_per_year,
    )
    entry_takeprofit_return = entry_takeprofit_eval.total_return
    entry_takeprofit_sharpe = entry_takeprofit_eval.sharpe_ratio
    entry_takeprofit_returns = entry_takeprofit_eval.returns
    entry_takeprofit_avg_daily = entry_takeprofit_eval.avg_daily_return
    entry_takeprofit_annual = entry_takeprofit_eval.annualized_return
    entry_takeprofit_finalday_return = (
        entry_takeprofit_return / len(actual_returns) if len(actual_returns) > 0 else 0.0
    )

    # Highlow strategy
    highlow_eval = evaluate_highlow_strategy(
        last_preds["close_predictions"],
        last_preds["high_predictions"],
        last_preds["low_predictions"],
        last_preds["close_actual_movement_values"],
        last_preds["high_actual_movement_values"],
        last_preds["low_actual_movement_values"],
        trading_fee,
        is_crypto=is_crypto,
        trading_days_per_year=trading_days_per_year,
    )
    highlow_return = highlow_eval.total_return
    highlow_sharpe = highlow_eval.sharpe_ratio
    highlow_returns = highlow_eval.returns
    highlow_avg_daily = highlow_eval.avg_daily_return
    highlow_annual = highlow_eval.annualized_return
    highlow_finalday_return = highlow_return / len(actual_returns) if len(actual_returns) > 0 else 0.0

    ci_guard_return = 0.0
    ci_guard_sharpe = 0.0
    ci_guard_finalday_return = 0.0
    ci_guard_returns = np.zeros(len(actual_returns), dtype=np.float32)
    ci_signals = torch.zeros_like(last_preds["close_predictions"])
    ci_guard_avg_daily = 0.0
    ci_guard_annual = 0.0
    if len(actual_returns) > 0:
        ci_band = torch.as_tensor(last_preds["close_ci_band"][:-1], dtype=torch.float32)
        if ci_band.numel() == len(last_preds["close_predictions"]):
            ci_signals = confidence_guard_strategy(
                last_preds["close_predictions"],
                ci_band,
                ci_multiplier=TOTO_CI_GUARD_MULTIPLIER,
                is_crypto=is_crypto,
            )
            ci_eval = evaluate_strategy(ci_signals, actual_returns, trading_fee, trading_days_per_year)
            ci_guard_return = ci_eval.total_return
            ci_guard_sharpe = ci_eval.sharpe_ratio
            ci_guard_returns = ci_eval.returns
            ci_guard_avg_daily = ci_eval.avg_daily_return
            ci_guard_annual = ci_eval.annualized_return
            if ci_signals.numel() > 0:
                ci_guard_finalday_return = (
                    ci_signals[-1].item() * actual_returns.iloc[-1]
                    - (2 * trading_fee * SPREAD)
                )

    # Log strategy metrics to tensorboard
    tb_writer.add_scalar(f'{symbol}/strategies/simple/total_return', simple_total_return, sim_idx)
    tb_writer.add_scalar(f'{symbol}/strategies/simple/sharpe', simple_sharpe, sim_idx)
    tb_writer.add_scalar(f'{symbol}/strategies/simple/finalday', simple_finalday_return, sim_idx)

    tb_writer.add_scalar(f'{symbol}/strategies/all_signals/total_return', all_signals_total_return, sim_idx)
    tb_writer.add_scalar(f'{symbol}/strategies/all_signals/sharpe', all_signals_sharpe, sim_idx)
    tb_writer.add_scalar(f'{symbol}/strategies/all_signals/finalday', all_signals_finalday_return, sim_idx)

    tb_writer.add_scalar(f'{symbol}/strategies/buy_hold/total_return', buy_hold_return, sim_idx)
    tb_writer.add_scalar(f'{symbol}/strategies/buy_hold/sharpe', buy_hold_sharpe, sim_idx)
    tb_writer.add_scalar(f'{symbol}/strategies/buy_hold/finalday', buy_hold_finalday_return, sim_idx)

    tb_writer.add_scalar(f'{symbol}/strategies/unprofit_shutdown/total_return', unprofit_shutdown_return, sim_idx)
    tb_writer.add_scalar(f'{symbol}/strategies/unprofit_shutdown/sharpe', unprofit_shutdown_sharpe, sim_idx)
    tb_writer.add_scalar(f'{symbol}/strategies/unprofit_shutdown/finalday', unprofit_shutdown_finalday_return, sim_idx)

    tb_writer.add_scalar(f'{symbol}/strategies/entry_takeprofit/total_return', entry_takeprofit_return, sim_idx)
    tb_writer.add_scalar(f'{symbol}/strategies/entry_takeprofit/sharpe', entry_takeprofit_sharpe, sim_idx)
    tb_writer.add_scalar(f'{symbol}/strategies/entry_takeprofit/finalday', entry_takeprofit_finalday_return, sim_idx)

    tb_writer.add_scalar(f'{symbol}/strategies/highlow/total_return', highlow_return, sim_idx)
    tb_writer.add_scalar(f'{symbol}/strategies/highlow/sharpe', highlow_sharpe, sim_idx)
    tb_writer.add_scalar(f'{symbol}/strategies/highlow/finalday', highlow_finalday_return, sim_idx)

    tb_writer.add_scalar(f'{symbol}/strategies/ci_guard/total_return', ci_guard_return, sim_idx)
    tb_writer.add_scalar(f'{symbol}/strategies/ci_guard/sharpe', ci_guard_sharpe, sim_idx)
    tb_writer.add_scalar(f'{symbol}/strategies/ci_guard/finalday', ci_guard_finalday_return, sim_idx)

    tb_writer.add_scalar(f'{symbol}/strategies/maxdiff/total_return', maxdiff_return, sim_idx)
    tb_writer.add_scalar(f'{symbol}/strategies/maxdiff/sharpe', maxdiff_sharpe, sim_idx)
    tb_writer.add_scalar(f'{symbol}/strategies/maxdiff/finalday', maxdiff_finalday_return, sim_idx)

    # Log returns over time
    for t, ret in enumerate(simple_returns):
        tb_writer.add_scalar(f'{symbol}/returns_over_time/simple', ret, t)
    for t, ret in enumerate(all_signals_returns):
        tb_writer.add_scalar(f'{symbol}/returns_over_time/all_signals', ret, t)
    for t, ret in enumerate(buy_hold_returns):
        tb_writer.add_scalar(f'{symbol}/returns_over_time/buy_hold', ret, t)
    for t, ret in enumerate(unprofit_shutdown_returns):
        tb_writer.add_scalar(f'{symbol}/returns_over_time/unprofit_shutdown', ret, t)
    for t, ret in enumerate(entry_takeprofit_returns):
        tb_writer.add_scalar(f'{symbol}/returns_over_time/entry_takeprofit', ret, t)
    for t, ret in enumerate(highlow_returns):
        tb_writer.add_scalar(f'{symbol}/returns_over_time/highlow', ret, t)
    for t, ret in enumerate(ci_guard_returns):
        tb_writer.add_scalar(f'{symbol}/returns_over_time/ci_guard', ret, t)
    for t, ret in enumerate(maxdiff_returns):
        tb_writer.add_scalar(f'{symbol}/returns_over_time/maxdiff', ret, t)

    result = {
        'date': simulation_data.index[-1],
        'close': float(last_preds['close_last_price']),
        'predicted_close': float(last_preds.get('close_predicted_price_value', 0.0)),
        'predicted_high': float(last_preds.get('high_predicted_price_value', 0.0)),
        'predicted_low': float(last_preds.get('low_predicted_price_value', 0.0)),
        'toto_expected_move_pct': float(last_preds.get('toto_close_pred_pct', 0.0)),
        'kronos_expected_move_pct': float(last_preds.get('kronos_close_pred_pct', 0.0)),
        'realized_volatility_pct': float(last_preds.get('realized_volatility_pct', 0.0)),
        'dollar_vol_20d': float(last_preds.get('dollar_vol_20d', 0.0)),
        'atr_pct_14': float(last_preds.get('atr_pct_14', 0.0)),
        'spread_bps_estimate': float(last_preds.get('spread_bps_estimate', 0.0)),
        'close_prediction_source': last_preds.get('close_prediction_source', best_model),
        'raw_expected_move_pct': float(last_preds.get('raw_expected_move_pct', 0.0)),
        'calibrated_expected_move_pct': float(last_preds.get('calibrated_expected_move_pct', last_preds.get('raw_expected_move_pct', 0.0))),
        'calibration_slope': float(last_preds.get('calibration_slope', 1.0)),
        'calibration_intercept': float(last_preds.get('calibration_intercept', 0.0)),
        'simple_strategy_return': float(simple_total_return),
        'simple_strategy_sharpe': float(simple_sharpe),
        'simple_strategy_finalday': float(simple_finalday_return),
        'simple_strategy_avg_daily_return': float(simple_avg_daily),
        'simple_strategy_annual_return': float(simple_annual),
        'all_signals_strategy_return': float(all_signals_total_return),
        'all_signals_strategy_sharpe': float(all_signals_sharpe),
        'all_signals_strategy_finalday': float(all_signals_finalday_return),
        'all_signals_strategy_avg_daily_return': float(all_signals_avg_daily),
        'all_signals_strategy_annual_return': float(all_signals_annual),
        'buy_hold_return': float(buy_hold_return),
        'buy_hold_sharpe': float(buy_hold_sharpe),
        'buy_hold_finalday': float(buy_hold_finalday_return),
        'buy_hold_avg_daily_return': float(buy_hold_avg_daily),
        'buy_hold_annual_return': float(buy_hold_annual),
        'unprofit_shutdown_return': float(unprofit_shutdown_return),
        'unprofit_shutdown_sharpe': float(unprofit_shutdown_sharpe),
        'unprofit_shutdown_finalday': float(unprofit_shutdown_finalday_return),
        'unprofit_shutdown_avg_daily_return': float(unprofit_shutdown_avg_daily),
        'unprofit_shutdown_annual_return': float(unprofit_shutdown_annual),
        'entry_takeprofit_return': float(entry_takeprofit_return),
        'entry_takeprofit_sharpe': float(entry_takeprofit_sharpe),
        'entry_takeprofit_finalday': float(entry_takeprofit_finalday_return),
        'entry_takeprofit_avg_daily_return': float(entry_takeprofit_avg_daily),
        'entry_takeprofit_annual_return': float(entry_takeprofit_annual),
        'highlow_return': float(highlow_return),
        'highlow_sharpe': float(highlow_sharpe),
        'highlow_finalday_return': float(highlow_finalday_return),
        'highlow_avg_daily_return': float(highlow_avg_daily),
        'highlow_annual_return': float(highlow_annual),
        'maxdiff_return': float(maxdiff_return),
        'maxdiff_sharpe': float(maxdiff_sharpe),
        'maxdiff_finalday_return': float(maxdiff_finalday_return),
        'maxdiff_avg_daily_return': float(maxdiff_avg_daily),
        'maxdiff_annual_return': float(maxdiff_annual),
        'maxdiff_turnover': float(maxdiff_turnover),
        'maxdiffprofit_profit': float(maxdiff_metadata.get('maxdiffprofit_profit', 0.0)),
        'maxdiffprofit_profit_values': maxdiff_metadata.get('maxdiffprofit_profit_values', []),
        'maxdiffprofit_profit_high_multiplier': float(maxdiff_metadata.get('maxdiffprofit_profit_high_multiplier', 0.0)),
        'maxdiffprofit_profit_low_multiplier': float(maxdiff_metadata.get('maxdiffprofit_profit_low_multiplier', 0.0)),
        'maxdiffprofit_high_price': float(maxdiff_metadata.get('maxdiffprofit_high_price', 0.0)),
        'maxdiffprofit_low_price': float(maxdiff_metadata.get('maxdiffprofit_low_price', 0.0)),
        'ci_guard_return': float(ci_guard_return),
        'ci_guard_sharpe': float(ci_guard_sharpe),
        'ci_guard_finalday': float(ci_guard_finalday_return),
        'ci_guard_avg_daily_return': float(ci_guard_avg_daily),
        'ci_guard_annual_return': float(ci_guard_annual),
        'close_val_loss': float(last_preds.get('close_val_loss', 0.0)),
        'high_val_loss': float(last_preds.get('high_val_loss', 0.0)),
        'low_val_loss': float(last_preds.get('low_val_loss', 0.0)),
    }

    return result


def evaluate_entry_takeprofit_strategy(
    close_predictions,
    high_predictions,
    low_predictions,
    actual_close,
    actual_high,
    actual_low,
    trading_fee,
    trading_days_per_year: int,
) -> StrategyEvaluation:
    """
    Evaluates an entry+takeprofit approach with minimal repeated fees:
      - If close_predictions[idx] > 0 => 'buy'
        - Exit when actual_high >= high_predictions[idx], else exit at actual_close.
      - If close_predictions[idx] < 0 => 'short'
        - Exit when actual_low <= low_predictions[idx], else exit at actual_close.
      - If we remain in the same side as previous day, don't pay another opening fee.
    """

    total_available = min(
        len(close_predictions),
        len(high_predictions),
        len(low_predictions),
        len(actual_close),
        len(actual_high),
        len(actual_low),
    )

    if total_available == 0:
        return StrategyEvaluation(
            total_return=0.0,
            avg_daily_return=0.0,
            annualized_return=0.0,
            sharpe_ratio=0.0,
            returns=np.zeros(0, dtype=float),
        )

    if total_available < len(close_predictions):
        logger.warning(
            "Entry+takeprofit truncating inputs (close=%d, actual_close=%d, actual_high=%d, actual_low=%d)",
            len(close_predictions),
            len(actual_close),
            len(actual_high),
            len(actual_low),
        )

    close_predictions = close_predictions[:total_available]
    high_predictions = high_predictions[:total_available]
    low_predictions = low_predictions[:total_available]
    actual_close = actual_close[:total_available]
    actual_high = actual_high[:total_available]
    actual_low = actual_low[:total_available]

    daily_returns = []
    last_side = None  # track "buy" or "short" from previous day

    for idx in range(total_available):
        # determine side
        is_buy = bool(close_predictions[idx] > 0)
        new_side = "buy" if is_buy else "short"

        # if same side as previous day, we are continuing
        continuing_same_side = (last_side == new_side)

        # figure out exit
        if is_buy:
            if actual_high[idx] >= high_predictions[idx]:
                daily_return = high_predictions[idx]  # approximate from 0 to predicted high
            else:
                daily_return = actual_close[idx]
        else:  # short
            if actual_low[idx] <= low_predictions[idx]:
                daily_return = 0 - low_predictions[idx]  # from 0 down to predicted_low
            else:
                daily_return = 0 - actual_close[idx]

        # fees: if it's the first day with new_side, pay one side of the fee
        # if we exit from the previous day (different side or last_side == None?), pay closing fee
        fee_to_charge = 0.0

        # if we changed sides or last_side is None, we pay open fee
        if not continuing_same_side:
            fee_to_charge += trading_fee  # opening fee
            if last_side is not None:
                fee_to_charge += trading_fee  # closing fee for old side

        # apply total fee
        daily_return -= fee_to_charge
        daily_returns.append(daily_return)

        last_side = new_side

    daily_returns = np.array(daily_returns, dtype=float)
    total_return = float(daily_returns.sum())
    if daily_returns.size == 0:
        sharpe_ratio = 0.0
    else:
        std = float(daily_returns.std())
        if std == 0.0 or np.isnan(std):
            sharpe_ratio = 0.0
        else:
            sharpe_ratio = float((daily_returns.mean() / std) * np.sqrt(trading_days_per_year))
    avg_daily_return, annualized_return = _compute_return_profile(daily_returns, trading_days_per_year)

    return StrategyEvaluation(
        total_return=total_return,
        avg_daily_return=avg_daily_return,
        annualized_return=annualized_return,
        sharpe_ratio=sharpe_ratio,
        returns=daily_returns,
    )


def evaluate_highlow_strategy(
        close_predictions,
        high_predictions,
        low_predictions,
        actual_close,
        actual_high,
        actual_low,
        trading_fee,
        is_crypto=False,
        trading_days_per_year: int = 252,
) -> StrategyEvaluation:
    """
    Evaluate a "high-low" trading approach.

    - If close_predictions[idx] > 0 => attempt a 'buy' at predicted_low, else skip.
    - If is_crypto=False and close_predictions[idx] < 0 => attempt short at predicted_high, else skip.
    - Either way, exit at actual_close by day's end.

    Returns
    -------
    StrategyEvaluation
        Contains total return, sharpe ratio, and the per-day return series.
    """
    daily_returns = []
    last_side = None  # track "buy"/"short" from previous day

    for idx in range(len(close_predictions)):
        cp = close_predictions[idx]
        if cp > 0:
            # Attempt buy at predicted_low if actual_low <= predicted_low, else buy at actual_close
            entry = low_predictions[idx] if actual_low[idx] <= low_predictions[idx] else actual_close[idx]
            exit_price = actual_close[idx]
            new_side = "buy"
        elif (not is_crypto) and (cp < 0):
            # Attempt short if not crypto
            entry = high_predictions[idx] if actual_high[idx] >= high_predictions[idx] else actual_close[idx]
            # Gains from short are entry - final
            exit_price = actual_close[idx]
            new_side = "short"
        else:
            # Skip if crypto and cp < 0 (no short), or cp == 0
            daily_returns.append(0.0)
            last_side = None
            continue

        # Calculate daily gain
        if is_buy_side(new_side):
            daily_gain = exit_price - entry
        else:
            # short
            daily_gain = entry - exit_price

        # Fees: open if side changed or if None, close prior side if it existed
        fee_to_charge = 0.0
        if new_side != last_side:
            fee_to_charge += trading_fee  # open
            if last_side is not None:
                fee_to_charge += trading_fee  # close old side

        daily_gain -= fee_to_charge
        daily_returns.append(daily_gain)
        last_side = new_side

    daily_returns = np.array(daily_returns, dtype=float)
    total_return = float(daily_returns.sum())
    if daily_returns.size == 0:
        sharpe_ratio = 0.0
    else:
        std = float(daily_returns.std())
        if std == 0.0 or np.isnan(std):
            sharpe_ratio = 0.0
        else:
            sharpe_ratio = float((daily_returns.mean() / std) * np.sqrt(trading_days_per_year))
    avg_daily_return, annualized_return = _compute_return_profile(daily_returns, trading_days_per_year)

    return StrategyEvaluation(
        total_return=total_return,
        avg_daily_return=avg_daily_return,
        annualized_return=annualized_return,
        sharpe_ratio=sharpe_ratio,
        returns=daily_returns
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inline backtests for a given symbol and optionally export results as JSON."
    )
    parser.add_argument(
        "symbol",
        nargs="?",
        default="ETHUSD",
        help="Ticker symbol to backtest (default: ETHUSD).",
    )
    parser.add_argument(
        "--output-json",
        dest="output_json",
        help="Optional path to write backtest results as JSON.",
    )
    parser.add_argument(
        "--output-label",
        dest="output_label",
        help="Optional label to store in the JSON payload instead of the raw symbol.",
    )
    args = parser.parse_args()

    result_df = backtest_forecasts(args.symbol)

    if args.output_json:
        output_path = Path(args.output_json)
        from math import isnan

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        def _mean(column: str) -> Optional[float]:
            if column not in result_df:
                return None
            value = float(result_df[column].mean())
            if isnan(value):
                return None
            return value

        strategies_payload = {
            "simple": {
                "return": _mean("simple_strategy_return"),
                "sharpe": _mean("simple_strategy_sharpe"),
                "final_day": _mean("simple_strategy_finalday"),
                "avg_daily_return": _mean("simple_strategy_avg_daily_return"),
                "annual_return": _mean("simple_strategy_annual_return"),
            },
            "all_signals": {
                "return": _mean("all_signals_strategy_return"),
                "sharpe": _mean("all_signals_strategy_sharpe"),
                "final_day": _mean("all_signals_strategy_finalday"),
                "avg_daily_return": _mean("all_signals_strategy_avg_daily_return"),
                "annual_return": _mean("all_signals_strategy_annual_return"),
            },
            "buy_hold": {
                "return": _mean("buy_hold_return"),
                "sharpe": _mean("buy_hold_sharpe"),
                "final_day": _mean("buy_hold_finalday"),
                "avg_daily_return": _mean("buy_hold_avg_daily_return"),
                "annual_return": _mean("buy_hold_annual_return"),
            },
            "unprofit_shutdown": {
                "return": _mean("unprofit_shutdown_return"),
                "sharpe": _mean("unprofit_shutdown_sharpe"),
                "final_day": _mean("unprofit_shutdown_finalday"),
                "avg_daily_return": _mean("unprofit_shutdown_avg_daily_return"),
                "annual_return": _mean("unprofit_shutdown_annual_return"),
            },
            "entry_takeprofit": {
                "return": _mean("entry_takeprofit_return"),
                "sharpe": _mean("entry_takeprofit_sharpe"),
                "final_day": _mean("entry_takeprofit_finalday"),
                "avg_daily_return": _mean("entry_takeprofit_avg_daily_return"),
                "annual_return": _mean("entry_takeprofit_annual_return"),
            },
            "highlow": {
                "return": _mean("highlow_return"),
                "sharpe": _mean("highlow_sharpe"),
                "final_day": _mean("highlow_finalday_return"),
                "avg_daily_return": _mean("highlow_avg_daily_return"),
                "annual_return": _mean("highlow_annual_return"),
            },
            "maxdiff": {
                "return": _mean("maxdiff_return"),
                "sharpe": _mean("maxdiff_sharpe"),
                "final_day": _mean("maxdiff_finalday_return"),
                "avg_daily_return": _mean("maxdiff_avg_daily_return"),
                "annual_return": _mean("maxdiff_annual_return"),
                "turnover": _mean("maxdiff_turnover"),
            },
            "ci_guard": {
                "return": _mean("ci_guard_return"),
                "sharpe": _mean("ci_guard_sharpe"),
                "final_day": _mean("ci_guard_finalday"),
                "avg_daily_return": _mean("ci_guard_avg_daily_return"),
                "annual_return": _mean("ci_guard_annual_return"),
            },
        }

        payload = {
            "symbol": args.output_label or args.symbol,
            "runs": int(len(result_df)),
            "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "strategies": strategies_payload,
            "metrics": {
                "close_val_loss": _mean("close_val_loss"),
                "high_val_loss": _mean("high_val_loss"),
                "low_val_loss": _mean("low_val_loss"),
                "walk_forward_oos_sharpe": _mean("walk_forward_oos_sharpe"),
                "walk_forward_turnover": _mean("walk_forward_turnover"),
            },
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
