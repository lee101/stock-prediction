import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass

from src.comparisons import is_buy_side
from src.logging_utils import setup_logging

logger = setup_logging("backtest_test3_inline.log")

from data_curate_daily import download_daily_stock_data, fetch_spread
from disk_cache import disk_cache
from src.fixtures import crypto_symbols
from scripts.alpaca_cli import set_strategy_for_symbol
from src.models.toto_wrapper import TotoPipeline
from src.models.toto_aggregation import aggregate_with_spec
from src.models.kronos_wrapper import KronosForecastingWrapper
from hyperparamstore import load_best_config, load_model_selection
from loss_utils import percent_movements_augment

SPREAD = 1.0008711461252937
TOTO_CI_GUARD_MULTIPLIER = float(os.getenv("TOTO_CI_GUARD_MULTIPLIER", "1.0"))
_FORCE_KRONOS_VALUES = {"1", "true", "yes", "on"}
_forced_kronos_logged_symbols = set()
_model_selection_log_state: Dict[str, Tuple[str, str]] = {}
_toto_params_log_state: Dict[str, Tuple[str, str]] = {}
_model_selection_cache: Dict[str, str] = {}
_toto_params_cache: Dict[str, dict] = {}
_kronos_params_cache: Dict[str, dict] = {}

pipeline: Optional[TotoPipeline] = None
kronos_wrapper_cache: Dict[tuple, KronosForecastingWrapper] = {}

ReturnSeries = Union[np.ndarray, pd.Series]


@dataclass(frozen=True)
class StrategyEvaluation:
    total_return: float
    sharpe_ratio: float
    returns: ReturnSeries

_BOOL_TRUE = {"1", "true", "yes", "on"}

if __name__ == "__main__" and "REAL_TESTING" not in os.environ:
    os.environ["REAL_TESTING"] = "1"
    logger.info("REAL_TESTING not set; defaulting to enabled for standalone execution.")

FAST_TESTING = os.getenv("FAST_TESTING", "0").strip().lower() in _BOOL_TRUE
REAL_TESTING = os.getenv("REAL_TESTING", "0").strip().lower() in _BOOL_TRUE

COMPILED_MODELS_DIR = Path(os.getenv("COMPILED_MODELS_DIR", "compiled_models"))
INDUCTOR_CACHE_DIR = COMPILED_MODELS_DIR / "torch_inductor"

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
    COMPILED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    INDUCTOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(INDUCTOR_CACHE_DIR))


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


def release_model_resources() -> None:
    """Public helper to free GPU-resident inference models between runs."""
    _drop_toto_pipeline()
    _drop_kronos_wrappers()


@disk_cache
def cached_predict(context, prediction_length, num_samples, samples_per_batch):
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


TOTO_MODEL_ID = os.getenv("TOTO_MODEL_ID", "Datadog/Toto-Open-Base-1.0")
DEFAULT_TOTO_NUM_SAMPLES = int(os.getenv("TOTO_NUM_SAMPLES", "3072"))
DEFAULT_TOTO_SAMPLES_PER_BATCH = int(os.getenv("TOTO_SAMPLES_PER_BATCH", "384"))
DEFAULT_TOTO_AGG_SPEC = os.getenv("TOTO_AGGREGATION_SPEC", "trimmed_mean_10")

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
        params = FAST_TOTO_PARAMS.copy()
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
                    "MARKETSIM_KRONOS_SAMPLE_COUNT active — overriding sample_count to %d for %s.",
                    override,
                    symbol,
                )
            params["sample_count"] = override
    _kronos_params_cache[symbol] = params
    return params.copy()


def resolve_best_model(symbol: str) -> str:
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
    global pipeline
    _drop_kronos_wrappers()
    if pipeline is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Toto pipeline '{TOTO_MODEL_ID}' on {device}")

        torch_dtype: Optional[torch.dtype] = None
        torch_compile_enabled = False
        compile_mode: Optional[str] = None
        compile_backend: Optional[str] = None

        if REAL_TESTING:
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            if torch.cuda.is_available() and hasattr(torch, "compile"):
                torch_compile_enabled = True
                compile_mode = os.getenv("REAL_TOTO_COMPILE_MODE") or os.getenv("TOTO_COMPILE_MODE") or "reduce-overhead"
                compile_mode = (compile_mode or "").strip() or None
                compile_backend = os.getenv("REAL_TOTO_COMPILE_BACKEND") or os.getenv("TOTO_COMPILE_BACKEND")
                if compile_backend is not None and not compile_backend.strip():
                    compile_backend = None
                logger.info(
                    "REAL_TESTING enabled Toto torch.compile "
                    "(dtype=%s, mode=%s, backend=%s, cache_dir=%s).",
                    torch_dtype,
                    compile_mode,
                    compile_backend,
                    os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
                )
            else:
                logger.info("REAL_TESTING requested but torch.compile not available; falling back to eager mode.")
        elif FAST_TESTING:
            torch_dtype = torch.float32
        else:
            torch_dtype = torch.float32 if device == "cpu" else None

        pipeline = TotoPipeline.from_pretrained(
            model_id=TOTO_MODEL_ID,
            device_map=device,
            torch_dtype=torch_dtype,
            torch_compile=torch_compile_enabled,
            compile_mode=compile_mode,
            compile_backend=compile_backend,
        )
    return pipeline


def load_kronos_wrapper(params: Dict[str, float]) -> KronosForecastingWrapper:
    _drop_toto_pipeline()
    if not torch.cuda.is_available():
        raise RuntimeError("Kronos inference requires a CUDA-capable GPU.")
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


def evaluate_strategy(strategy_signals, actual_returns, trading_fee) -> StrategyEvaluation:
    global SPREAD
    """Evaluate the performance of a strategy, factoring in trading fees."""
    strategy_signals = strategy_signals.numpy()  # Convert to numpy array

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
    strategy_returns = strategy_signals * actual_returns - fees

    cumulative_returns = (1 + strategy_returns).cumprod() - 1
    total_return = float(cumulative_returns.iloc[-1])

    strategy_std = strategy_returns.std()
    if strategy_std == 0 or np.isnan(strategy_std):
        sharpe_ratio = 0.0  # or some other default value
    else:
        sharpe_ratio = float(strategy_returns.mean() / strategy_std * np.sqrt(252))

    return StrategyEvaluation(
        total_return=total_return,
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

    spread = fetch_spread(symbol)
    logger.info(f"spread: {spread}")
    SPREAD = spread  #

    # stock_data = load_stock_data_from_csv(csv_file)

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

        result = run_single_simulation(simulation_data, symbol, trading_fee, is_crypto, sim_number)
        results.append(result)

    results_df = pd.DataFrame(results)

    # Log final average metrics
    tb_writer.add_scalar(f'{symbol}/final_metrics/simple_avg_return', results_df['simple_strategy_return'].mean(), 0)
    tb_writer.add_scalar(f'{symbol}/final_metrics/simple_avg_sharpe', results_df['simple_strategy_sharpe'].mean(), 0)
    tb_writer.add_scalar(f'{symbol}/final_metrics/all_signals_avg_return',
                         results_df['all_signals_strategy_return'].mean(), 0)
    tb_writer.add_scalar(f'{symbol}/final_metrics/all_signals_avg_sharpe',
                         results_df['all_signals_strategy_sharpe'].mean(), 0)
    tb_writer.add_scalar(f'{symbol}/final_metrics/buy_hold_avg_return', results_df['buy_hold_return'].mean(), 0)
    tb_writer.add_scalar(f'{symbol}/final_metrics/buy_hold_avg_sharpe', results_df['buy_hold_sharpe'].mean(), 0)
    tb_writer.add_scalar(f'{symbol}/final_metrics/unprofit_shutdown_avg_return',
                         results_df['unprofit_shutdown_return'].mean(), 0)
    tb_writer.add_scalar(f'{symbol}/final_metrics/unprofit_shutdown_avg_sharpe',
                         results_df['unprofit_shutdown_sharpe'].mean(), 0)
    tb_writer.add_scalar(f'{symbol}/final_metrics/entry_takeprofit_avg_return',
                         results_df['entry_takeprofit_return'].mean(), 0)
    tb_writer.add_scalar(f'{symbol}/final_metrics/entry_takeprofit_avg_sharpe',
                         results_df['entry_takeprofit_sharpe'].mean(), 0)
    tb_writer.add_scalar(f'{symbol}/final_metrics/highlow_avg_return', results_df['highlow_return'].mean(), 0)
    tb_writer.add_scalar(f'{symbol}/final_metrics/highlow_avg_sharpe', results_df['highlow_sharpe'].mean(), 0)
    tb_writer.add_scalar(f'{symbol}/final_metrics/ci_guard_avg_return', results_df['ci_guard_return'].mean(), 0)
    tb_writer.add_scalar(f'{symbol}/final_metrics/ci_guard_avg_sharpe', results_df['ci_guard_sharpe'].mean(), 0)

    logger.info(f"\nAverage Validation Losses:")
    logger.info(f"Close Val Loss: {results_df['close_val_loss'].mean():.4f}")
    logger.info(f"High Val Loss: {results_df['high_val_loss'].mean():.4f}") 
    logger.info(f"Low Val Loss: {results_df['low_val_loss'].mean():.4f}")

    logger.info(f"\nBacktest results for {symbol} over {num_simulations} simulations:")
    logger.info(f"Average Simple Strategy Return: {results_df['simple_strategy_return'].mean():.4f}")
    logger.info(f"Average Simple Strategy Sharpe: {results_df['simple_strategy_sharpe'].mean():.4f}")
    logger.info(f"Average Simple Strategy Final Day Return: {results_df['simple_strategy_finalday'].mean():.4f}")
    logger.info(f"Average All Signals Strategy Return: {results_df['all_signals_strategy_return'].mean():.4f}")
    logger.info(f"Average All Signals Strategy Sharpe: {results_df['all_signals_strategy_sharpe'].mean():.4f}")
    logger.info(
        f"Average All Signals Strategy Final Day Return: {results_df['all_signals_strategy_finalday'].mean():.4f}")
    logger.info(f"Average Buy and Hold Return: {results_df['buy_hold_return'].mean():.4f}")
    logger.info(f"Average Buy and Hold Sharpe: {results_df['buy_hold_sharpe'].mean():.4f}")
    logger.info(f"Average Buy and Hold Final Day Return: {results_df['buy_hold_finalday'].mean():.4f}")
    logger.info(f"Average Unprofit Shutdown Buy and Hold Return: {results_df['unprofit_shutdown_return'].mean():.4f}")
    logger.info(f"Average Unprofit Shutdown Buy and Hold Sharpe: {results_df['unprofit_shutdown_sharpe'].mean():.4f}")
    logger.info(
        f"Average Unprofit Shutdown Buy and Hold Final Day Return: {results_df['unprofit_shutdown_finalday'].mean():.4f}")
    logger.info(f"Average Entry+Takeprofit Return: {results_df['entry_takeprofit_return'].mean():.4f}")
    logger.info(f"Average Entry+Takeprofit Sharpe: {results_df['entry_takeprofit_sharpe'].mean():.4f}")
    logger.info(
        f"Average Entry+Takeprofit Final Day Return: {results_df['entry_takeprofit_finalday'].mean():.4f}")
    logger.info(f"Average Highlow Return: {results_df['highlow_return'].mean():.4f}")
    logger.info(f"Average Highlow Sharpe: {results_df['highlow_sharpe'].mean():.4f}")
    logger.info(f"Average Highlow Final Day Return: {results_df['highlow_finalday_return'].mean():.4f}")
    logger.info(f"Average CI Guard Return: {results_df['ci_guard_return'].mean():.4f}")
    logger.info(f"Average CI Guard Sharpe: {results_df['ci_guard_sharpe'].mean():.4f}")

    # Determine which strategy is best overall
    avg_simple = results_df["simple_strategy_return"].mean()
    avg_allsignals = results_df["all_signals_strategy_return"].mean()
    avg_takeprofit = results_df["entry_takeprofit_return"].mean()
    avg_highlow = results_df["highlow_return"].mean()
    avg_ci_guard = results_df["ci_guard_return"].mean()

    best_return = max(avg_simple, avg_allsignals, avg_takeprofit, avg_highlow, avg_ci_guard)
    if best_return == avg_ci_guard:
        best_strategy = "ci_guard"
    elif best_return == avg_highlow:
        best_strategy = "highlow"
    elif best_return == avg_takeprofit:
        best_strategy = "takeprofit"
    elif best_return == avg_allsignals:
        best_strategy = "all_signals"
    else:
        best_strategy = "simple"

    # Record which strategy is best for this symbol & day
    set_strategy_for_symbol(symbol, best_strategy)

    return results_df



def run_single_simulation(simulation_data, symbol, trading_fee, is_crypto, sim_idx):
    last_preds = {
        'instrument': symbol,
        'close_last_price': simulation_data['Close'].iloc[-1],
    }

    best_model = resolve_best_model(symbol)
    use_kronos = best_model == "kronos"
    toto_params = None
    kronos_params = None
    kronos_wrapper = None
    kronos_df = None

    if use_kronos:
        kronos_params = resolve_kronos_params(symbol)
        kronos_wrapper = load_kronos_wrapper(kronos_params)
        kronos_df = prepare_kronos_dataframe(simulation_data)
    else:
        toto_params = resolve_toto_params(symbol)
        load_toto_pipeline()

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

        if use_kronos:
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
            if kronos_entry is None or len(kronos_entry.percent) == 0:
                logger.warning("Kronos produced no forecast for %s", key_to_predict)
                continue
            predictions = torch.tensor(kronos_entry.percent, dtype=torch.float32)
            predicted_absolute_last = float(kronos_entry.absolute[-1])
        else:
            predictions_list = []
            band_list = []
            for pred_idx in reversed(range(1, 8)):
                current_context = price[:-pred_idx]
                context = torch.tensor(current_context["y"].values, dtype=torch.float32)

                forecast = cached_predict(
                    context,
                    1,
                    num_samples=toto_params["num_samples"],
                    samples_per_batch=toto_params["samples_per_batch"],
                )
                tensor = forecast[0]

                array_data = None
                numpy_method = getattr(tensor, "numpy", None)
                if callable(numpy_method):
                    try:
                        array_data = numpy_method()
                    except Exception:
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

            predictions = torch.tensor(predictions_list, dtype=torch.float32)
            predicted_absolute_last = float(current_last_price + (current_last_price * predictions[-1].item()))

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
            if use_kronos:
                close_ci_band = torch.zeros_like(predictions)
            else:
                close_ci_band = torch.tensor(band_list, dtype=torch.float32)
            last_preds["close_ci_band"] = close_ci_band

    if "close_ci_band" not in last_preds:
        base_close_preds = torch.as_tensor(last_preds.get("close_predictions", torch.zeros(1)), dtype=torch.float32)
        pad_length = int(base_close_preds.shape[0] + 1)
        last_preds["close_ci_band"] = torch.zeros(pad_length, dtype=torch.float32)

    # Calculate actual percentage returns over the validation horizon
    close_window = simulation_data["Close"].iloc[-7:]
    actual_returns = close_window.pct_change().dropna().reset_index(drop=True)

    # Simple buy/sell strategy
    simple_signals = simple_buy_sell_strategy(
        last_preds["close_predictions"],
        is_crypto=is_crypto
    )
    simple_eval = evaluate_strategy(simple_signals, actual_returns, trading_fee)
    simple_total_return = simple_eval.total_return
    simple_sharpe = simple_eval.sharpe_ratio
    simple_returns = simple_eval.returns
    simple_finalday_return = (simple_signals[-1].item() * actual_returns.iloc[-1]) - (2 * trading_fee * SPREAD)

    # All signals strategy
    all_signals = all_signals_strategy(
        last_preds["close_predictions"],
        last_preds["high_predictions"],
        last_preds["low_predictions"],
        is_crypto=is_crypto
    )
    all_signals_eval = evaluate_strategy(all_signals, actual_returns, trading_fee)
    all_signals_total_return = all_signals_eval.total_return
    all_signals_sharpe = all_signals_eval.sharpe_ratio
    all_signals_returns = all_signals_eval.returns
    all_signals_finalday_return = (all_signals[-1].item() * actual_returns.iloc[-1]) - (2 * trading_fee * SPREAD)

    # Buy and hold strategy
    buy_hold_signals = buy_hold_strategy(last_preds["close_predictions"])
    buy_hold_eval = evaluate_strategy(buy_hold_signals, actual_returns, trading_fee)
    buy_hold_sharpe = buy_hold_eval.sharpe_ratio
    buy_hold_returns = buy_hold_eval.returns
    if actual_returns.empty:
        buy_hold_return_expected = -trading_fee
        buy_hold_finalday_return = -trading_fee
    else:
        buy_hold_return_expected = (1 + actual_returns).prod() - 1 - trading_fee
        buy_hold_finalday_return = actual_returns.iloc[-1] - trading_fee
    buy_hold_return = buy_hold_return_expected

    # Unprofit shutdown buy and hold strategy
    unprofit_shutdown_signals = unprofit_shutdown_buy_hold(last_preds["close_predictions"], actual_returns, is_crypto=is_crypto)
    unprofit_shutdown_eval = evaluate_strategy(unprofit_shutdown_signals, actual_returns, trading_fee)
    unprofit_shutdown_return = unprofit_shutdown_eval.total_return
    unprofit_shutdown_sharpe = unprofit_shutdown_eval.sharpe_ratio
    unprofit_shutdown_returns = unprofit_shutdown_eval.returns
    unprofit_shutdown_finalday_return = (unprofit_shutdown_signals[-1].item() * actual_returns.iloc[-1]) - (2 * trading_fee * SPREAD)

    # Entry + takeprofit strategy
    entry_takeprofit_return, entry_takeprofit_sharpe, entry_takeprofit_returns = evaluate_entry_takeprofit_strategy(
        last_preds["close_predictions"],
        last_preds["high_predictions"],
        last_preds["low_predictions"],
        last_preds["close_actual_movement_values"],
        last_preds["high_actual_movement_values"],
        last_preds["low_actual_movement_values"],
        trading_fee
    )
    entry_takeprofit_finalday_return = entry_takeprofit_return / len(actual_returns)

    # Highlow strategy
    highlow_eval = evaluate_highlow_strategy(
        last_preds["close_predictions"],
        last_preds["high_predictions"],
        last_preds["low_predictions"],
        last_preds["close_actual_movement_values"],
        last_preds["high_actual_movement_values"],
        last_preds["low_actual_movement_values"],
        trading_fee,
        is_crypto=is_crypto
    )
    highlow_return = highlow_eval.total_return
    highlow_sharpe = highlow_eval.sharpe_ratio
    highlow_returns = highlow_eval.returns
    highlow_finalday_return = highlow_return / len(actual_returns)

    ci_guard_return = 0.0
    ci_guard_sharpe = 0.0
    ci_guard_finalday_return = 0.0
    ci_guard_returns = np.zeros(len(actual_returns), dtype=np.float32)
    ci_signals = torch.zeros_like(last_preds["close_predictions"])
    if len(actual_returns) > 0:
        ci_band = torch.as_tensor(last_preds["close_ci_band"][:-1], dtype=torch.float32)
        if ci_band.numel() == len(last_preds["close_predictions"]):
            ci_signals = confidence_guard_strategy(
                last_preds["close_predictions"],
                ci_band,
                ci_multiplier=TOTO_CI_GUARD_MULTIPLIER,
                is_crypto=is_crypto,
            )
            ci_eval = evaluate_strategy(ci_signals, actual_returns, trading_fee)
            ci_guard_return = ci_eval.total_return
            ci_guard_sharpe = ci_eval.sharpe_ratio
            ci_guard_returns = ci_eval.returns
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

    result = {
        'date': simulation_data.index[-1],
        'close': float(last_preds['close_last_price']),
        'predicted_close': float(last_preds['close_predicted_price_value']),
        'predicted_high': float(last_preds['high_predicted_price_value']),
        'predicted_low': float(last_preds['low_predicted_price_value']),
        'simple_strategy_return': float(simple_total_return),
        'simple_strategy_sharpe': float(simple_sharpe),
        'simple_strategy_finalday': float(simple_finalday_return),
        'all_signals_strategy_return': float(all_signals_total_return),
        'all_signals_strategy_sharpe': float(all_signals_sharpe),
        'all_signals_strategy_finalday': float(all_signals_finalday_return),
        'buy_hold_return': float(buy_hold_return),
        'buy_hold_sharpe': float(buy_hold_sharpe),
        'buy_hold_finalday': float(buy_hold_finalday_return),
        'unprofit_shutdown_return': float(unprofit_shutdown_return),
        'unprofit_shutdown_sharpe': float(unprofit_shutdown_sharpe),
        'unprofit_shutdown_finalday': float(unprofit_shutdown_finalday_return),
        'entry_takeprofit_return': float(entry_takeprofit_return),
        'entry_takeprofit_sharpe': float(entry_takeprofit_sharpe),
        'entry_takeprofit_finalday': float(entry_takeprofit_finalday_return),
        'highlow_return': float(highlow_return),
        'highlow_sharpe': float(highlow_sharpe),
        'highlow_finalday_return': float(highlow_finalday_return),
        'ci_guard_return': float(ci_guard_return),
        'ci_guard_sharpe': float(ci_guard_sharpe),
        'ci_guard_finalday': float(ci_guard_finalday_return),
        'close_val_loss': float(last_preds['close_val_loss']),
        'high_val_loss': float(last_preds['high_val_loss']),
        'low_val_loss': float(last_preds['low_val_loss']),
    }

    return result


def evaluate_entry_takeprofit_strategy(
        close_predictions, high_predictions, low_predictions,
        actual_close, actual_high, actual_low,
        trading_fee
):
    """
    Evaluates an entry+takeprofit approach with minimal repeated fees:
      - If close_predictions[idx] > 0 => 'buy'
        - Exit when actual_high >= high_predictions[idx], else exit at actual_close.
      - If close_predictions[idx] < 0 => 'short'
        - Exit when actual_low <= low_predictions[idx], else exit at actual_close.
      - If we remain in the same side as previous day, don't pay another opening fee.
    """

    daily_returns = []
    last_side = None  # track "buy" or "short" from previous day

    for idx in range(len(close_predictions)):
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
    if daily_returns.std() == 0:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = float(daily_returns.mean() / daily_returns.std() * np.sqrt(252))

    return total_return, sharpe_ratio, daily_returns


def evaluate_highlow_strategy(
        close_predictions,
        high_predictions,
        low_predictions,
        actual_close,
        actual_high,
        actual_low,
        trading_fee,
        is_crypto=False
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
    if daily_returns.std() == 0:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = float((daily_returns.mean() / daily_returns.std()) * np.sqrt(252))

    return StrategyEvaluation(
        total_return=total_return,
        sharpe_ratio=sharpe_ratio,
        returns=daily_returns
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        symbol = "ETHUSD"
        print("Usage: python backtest_test.py <symbol> defaultint to eth")
    else:
        symbol = sys.argv[1]

    # backtest_forecasts("NVDA")
    backtest_forecasts(symbol)
    # backtest_forecasts("UNIUSD")
    # backtest_forecasts("AAPL")
    # backtest_forecasts("GOOG")
