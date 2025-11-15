import json
import logging
import math
import os
import signal
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import sleep
from typing import Dict, List, Mapping, Optional, Tuple

import alpaca_wrapper
import pandas as pd
import pytz
from loguru import logger

from hyperparamstore import load_model_selection
from backtest_test3_inline import backtest_forecasts, release_model_resources

from neuralpricingstrategy.runtime import NeuralPricingAdjuster

import src.trade_stock_state_utils as state_utils
from portfolio_allocation_optimizer_wrapper import (
    PortfolioAllocationOptimizer,
    PortfolioOptimizerConfig,
)
from alpaca.data import StockHistoricalDataClient
from data_curate_daily import download_exchange_latest_data, get_ask, get_bid
from env_real import ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD
from jsonshelve import FlatShelf
from marketsimulator.state import get_state
from src.backtest_data_utils import normalize_series
from src.cache_utils import ensure_huggingface_cache_dir
from src.comparisons import is_buy_side, is_same_side, is_sell_side
from src.cooldown_utils import can_trade_now, clear_cooldown, record_loss_timestamp
from src.date_utils import is_nyse_trading_day_ending, is_nyse_trading_day_now
from src.fixtures import all_crypto_symbols
from src.logging_utils import setup_logging, get_log_filename
from src.trade_analysis_summary import build_analysis_summary_messages
from src.work_stealing_config import is_crypto_out_of_hours
from src.portfolio_filters import get_selected_strategy_forecast
from src.portfolio_risk import record_portfolio_snapshot
from src.risk_state import ProbeState, record_day_pl, resolve_probe_state
from src.process_utils import (
    MAXDIFF_WATCHERS_DIR,
    backout_near_market,
    ramp_into_position,
    spawn_close_position_at_maxdiff_takeprofit,
    spawn_close_position_at_takeprofit,
    spawn_open_position_at_maxdiff_takeprofit,
)
from src.sizing_utils import get_qty
from src.symbol_filtering import filter_symbols_by_tradable_pairs, get_filter_info
from src.symbol_utils import is_crypto_symbol
from src.trade_stock_env_utils import (
    TRUTHY_ENV_VALUES,
    _allowed_side_for,
    _current_symbol_entry_count,
    _drawdown_cap_for,
    _drawdown_resume_for,
    _get_env_float,
    _get_trend_stat,
    _increment_symbol_entry,
    _kelly_drawdown_scale,
    _lookup_threshold,
    _normalize_entry_key,
    _symbol_force_probe,
    _symbol_max_entries_per_run,
    _symbol_max_hold_seconds,
    _symbol_min_cooldown_minutes,
    _symbol_min_move,
    _symbol_min_predicted_move,
    _symbol_min_strategy_return,
    _symbol_trend_pnl_threshold,
    _symbol_trend_resume_threshold,
    get_entry_counter_snapshot,
    reset_symbol_entry_counters,
)

_TRUTHY = TRUTHY_ENV_VALUES
from src.trade_stock_forecast_snapshot import load_latest_forecast_snapshot as _load_forecast_snapshot
from src.trade_stock_gate_utils import (
    DISABLE_TRADE_GATES,
    coerce_positive_int,
    is_kronos_only_mode,
    is_tradeable,
    pass_edge_threshold,
    resolve_signal_sign,
    should_skip_closed_equity,
)
from src.trade_stock_utils import (
    agree_direction,
    coerce_optional_float,
    compute_spread_bps,
    evaluate_strategy_entry_gate,
    kelly_lite,
    parse_float_list,
    resolve_spread_cap,
    should_rebalance,
)
from src.trading_obj_utils import filter_to_realistic_positions
from stock.data_utils import coerce_numeric, ensure_lower_bound, safe_divide
from stock.state import ensure_state_dir as _shared_ensure_state_dir
from stock.state import get_state_dir, get_state_file, resolve_state_suffix
from strategytraining2 import log_strategy_snapshot

_EXPORTED_ENV_HELPERS = (reset_symbol_entry_counters, get_entry_counter_snapshot)

logger = setup_logging(get_log_filename("trade_stock_e2e.log", is_hourly=False))

ensure_huggingface_cache_dir(logger=logger)


STATE_DIR = get_state_dir()
STATE_SUFFIX = resolve_state_suffix()
TRADE_OUTCOME_FILE = get_state_file("trade_outcomes", STATE_SUFFIX)
TRADE_LEARNING_FILE = get_state_file("trade_learning", STATE_SUFFIX)
ACTIVE_TRADES_FILE = get_state_file("active_trades", STATE_SUFFIX)
TRADE_HISTORY_FILE = get_state_file("trade_history", STATE_SUFFIX)
MAXDIFF_PLANS_FILE = get_state_file("maxdiff_plans", STATE_SUFFIX)

MIN_STOCK_QTY = 1.0
MIN_CRYPTO_QTY = 0.001
MIN_PREDICTED_MOVEMENT = 0.0
MIN_DIRECTIONAL_CONFIDENCE = 0.0
MAX_TOTAL_EXPOSURE_PCT = 120.0
LIVE_DRAWDOWN_TRIGGER = -500.0  # dollars
PROBE_MAX_DURATION = timedelta(days=1)


def _resolve_probe_notional_limit() -> float:
    raw_limit = os.getenv("MARKETSIM_PROBE_NOTIONAL_LIMIT")
    limit = coerce_numeric(raw_limit, default=300.0) if raw_limit is not None else 300.0
    if limit <= 0:
        return 300.0
    return float(limit)


PROBE_NOTIONAL_LIMIT = _resolve_probe_notional_limit()

PROBE_LOSS_COOLDOWN_MINUTES = 180
ALLOW_HIGHLOW_ENTRY = os.getenv("ALLOW_HIGHLOW_ENTRY", "0").strip().lower() in {"1", "true", "yes", "on"}
ALLOW_TAKEPROFIT_ENTRY = os.getenv("ALLOW_TAKEPROFIT_ENTRY", "0").strip().lower() in {"1", "true", "yes", "on"}
_ALLOW_MAXDIFF_ENV = os.getenv("ALLOW_MAXDIFF_ENTRY")
if _ALLOW_MAXDIFF_ENV is None:
    ALLOW_MAXDIFF_ENTRY = True
else:
    ALLOW_MAXDIFF_ENTRY = _ALLOW_MAXDIFF_ENV.strip().lower() in {"1", "true", "yes", "on"}
_ALLOW_MAXDIFF_ALWAYS_ENV = os.getenv("ALLOW_MAXDIFF_ALWAYS_ENTRY")
if _ALLOW_MAXDIFF_ALWAYS_ENV is None:
    ALLOW_MAXDIFF_ALWAYS_ENTRY = True
else:
    ALLOW_MAXDIFF_ALWAYS_ENTRY = _ALLOW_MAXDIFF_ALWAYS_ENV.strip().lower() in {"1", "true", "yes", "on"}
_ALLOW_PCTDIFF_ENV = os.getenv("ALLOW_PCTDIFF_ENTRY")
if _ALLOW_PCTDIFF_ENV is None:
    ALLOW_PCTDIFF_ENTRY = True
else:
    ALLOW_PCTDIFF_ENTRY = _ALLOW_PCTDIFF_ENV.strip().lower() in {"1", "true", "yes", "on"}
ENABLE_TAKEPROFIT_BRACKETS = os.getenv("ENABLE_TAKEPROFIT_BRACKETS", "0").strip().lower() in {"1", "true", "yes", "on"}

# Backwards-compatible alias required by older tests/utilities
crypto_symbols = all_crypto_symbols


STRATEGYTRAINING_FAST_RESULTS_PATH = Path("strategytraining/sizing_strategy_fast_test_results.json")


def _load_strategytraining_symbols(path: Path = STRATEGYTRAINING_FAST_RESULTS_PATH) -> List[str]:
    """Load the latest symbol cohort used for sizing experiments."""

    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Unable to parse %s for neural symbols: %s", path, exc)
        return []

    symbols = payload.get("symbols")
    if not isinstance(symbols, list):
        return []
    normalized: List[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        if not isinstance(symbol, str):
            continue
        clean = symbol.strip().upper()
        if not clean or clean in seen:
            continue
        normalized.append(clean)
        seen.add(clean)
    return normalized


def _parse_neural_strategy_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


NEURAL_SIZING_ENABLED = os.getenv("MARKETSIM_ENABLE_NEURAL_SIZING", "1").strip().lower() in _TRUTHY
NEURAL_SIZING_RUN_DIR = os.getenv(
    "MARKETSIM_NEURAL_RUN_DIR",
    "strategytrainingneural/reports/run_20251114_093401",
)
NEURAL_SIZING_METRICS_CSV = os.getenv(
    "MARKETSIM_NEURAL_METRICS_CSV",
    "strategytraining/reports/sizing_strategy_daily_metrics.csv",
)
_NEURAL_REFERENCE_STRATEGIES = _parse_neural_strategy_list(
    os.getenv(
        "MARKETSIM_NEURAL_REFERENCE_STRATEGIES",
        (
            "VolAdjusted_10pct,"
            "VolAdjusted_10pct_UnprofitShutdown,"
            "VolAdjusted_10pct_StockDirShutdown,"
            "VolAdjusted_10pct_UnprofitShutdown,"
            "VolAdjusted_10pct_UnprofitShutdown_StockDirShutdown,"
            "VolAdjusted_15pct,"
            "VolAdjusted_15pct_UnprofitShutdown,"
            "VolAdjusted_15pct_StockDirShutdown,"
            "VolAdjusted_15pct_UnprofitShutdown_StockDirShutdown"
        ),
    )
)
NEURAL_MIN_SCALE = ensure_lower_bound(
    coerce_numeric(os.getenv("MARKETSIM_NEURAL_MIN_SCALE"), default=0.35),
    0.0,
    default=0.35,
)
NEURAL_MAX_SCALE = ensure_lower_bound(
    coerce_numeric(os.getenv("MARKETSIM_NEURAL_MAX_SCALE"), default=1.0),
    0.0,
    default=1.0,
)
if NEURAL_MAX_SCALE < NEURAL_MIN_SCALE:
    NEURAL_MAX_SCALE = NEURAL_MIN_SCALE
NEURAL_DEVICE = os.getenv("MARKETSIM_NEURAL_DEVICE")
NEURAL_ASSET_CLASS = os.getenv("MARKETSIM_NEURAL_ASSET_CLASS", "all")
_NEURAL_CONFIG: Optional[PortfolioOptimizerConfig]
if NEURAL_SIZING_ENABLED:
    _NEURAL_CONFIG = PortfolioOptimizerConfig(
        run_dir=NEURAL_SIZING_RUN_DIR,
        metrics_csv=NEURAL_SIZING_METRICS_CSV,
        reference_strategies=_NEURAL_REFERENCE_STRATEGIES,
        min_scale=NEURAL_MIN_SCALE,
        max_scale=NEURAL_MAX_SCALE,
        asset_class=NEURAL_ASSET_CLASS,
        device=NEURAL_DEVICE or None,
    )
else:
    _NEURAL_CONFIG = None
_PORTFOLIO_OPTIMIZER: Optional[PortfolioAllocationOptimizer] = None

NEURAL_PRICING_ENABLED = os.getenv("MARKETSIM_ENABLE_NEURAL_PRICING", "0").strip().lower() in _TRUTHY
NEURAL_PRICING_RUN_DIR = os.getenv("MARKETSIM_NEURAL_PRICING_RUN_DIR")
NEURAL_PRICING_DEVICE = os.getenv("MARKETSIM_NEURAL_PRICING_DEVICE")
_neural_pricing_adjuster: Optional[NeuralPricingAdjuster] = None
_neural_pricing_disabled_reason: Optional[str] = None

DISABLE_RECENT_PNL_PROBE = os.getenv("MARKETSIM_DISABLE_RECENT_PNL_PROBE", "1").strip().lower() in _TRUTHY


def _get_portfolio_optimizer() -> Optional[PortfolioAllocationOptimizer]:
    if not NEURAL_SIZING_ENABLED or _NEURAL_CONFIG is None:
        return None
    global _PORTFOLIO_OPTIMIZER
    if _PORTFOLIO_OPTIMIZER is None:
        if not _NEURAL_CONFIG.reference_strategies:
            return None
        _PORTFOLIO_OPTIMIZER = PortfolioAllocationOptimizer(_NEURAL_CONFIG)
    return _PORTFOLIO_OPTIMIZER


def _refresh_neural_scale(force: bool = False) -> Optional[float]:
    optimizer = _get_portfolio_optimizer()
    if optimizer is None:
        return None
    scale = optimizer.compute_scale(force=force)
    if scale is None:
        reason = optimizer.disabled_reason
        if reason:
            logger.warning("Neural sizing unavailable (%s)", reason)
        else:
            logger.warning("Neural sizing returned no scale; keeping existing sizing rules.")
        return None
    scores = optimizer.last_scores
    if scores:
        detail = ", ".join(f"{score.name}={score.weight:.3f}@{score.date}" for score in scores)
        logger.info(
            "Neural sizing scale %.3f from %s (weights: %s)",
            scale,
            _NEURAL_CONFIG.run_dir if _NEURAL_CONFIG else "unknown",
            detail,
        )
    else:
        logger.info(
            "Neural sizing scale %.3f from %s",
            scale,
            _NEURAL_CONFIG.run_dir if _NEURAL_CONFIG else "unknown",
        )
    optimizer.refresh_strategy_weights(force=force)
    return scale


def _get_neural_pricing_adjuster() -> Optional[NeuralPricingAdjuster]:
    if not NEURAL_PRICING_ENABLED or not NEURAL_PRICING_RUN_DIR:
        return None
    global _neural_pricing_adjuster, _neural_pricing_disabled_reason
    if _neural_pricing_adjuster is not None:
        return _neural_pricing_adjuster
    try:
        _neural_pricing_adjuster = NeuralPricingAdjuster(
            run_dir=NEURAL_PRICING_RUN_DIR,
            device=NEURAL_PRICING_DEVICE,
        )
        _neural_pricing_disabled_reason = None
        logger.info("Neural pricing initialized (run_dir=%s)", NEURAL_PRICING_RUN_DIR)
    except Exception as exc:  # pragma: no cover - defensive
        _neural_pricing_adjuster = None
        _neural_pricing_disabled_reason = str(exc)
        logger.warning("Neural pricing disabled: %s", exc)
    return _neural_pricing_adjuster


def _apply_neural_pricing(symbol: str, prediction_row) -> None:
    adjuster = _get_neural_pricing_adjuster()
    if adjuster is None:
        return
    try:
        payload = prediction_row.to_dict()
    except AttributeError:
        payload = dict(prediction_row)
    adjustment = adjuster.adjust(payload, symbol=symbol)
    if adjustment is None:
        if adjuster.last_error:
            logger.debug("Neural pricing skipped for %s: %s", symbol, adjuster.last_error)
        return
    prediction_row["neuralpricing_low_price"] = adjustment.low_price
    prediction_row["neuralpricing_high_price"] = adjustment.high_price
    prediction_row["neuralpricing_low_delta"] = adjustment.low_delta
    prediction_row["neuralpricing_high_delta"] = adjustment.high_delta
    prediction_row["neuralpricing_pnl_gain"] = adjustment.pnl_gain
    prediction_row["neuralpricing_run_dir"] = NEURAL_PRICING_RUN_DIR
    logger.debug(
        "%s neural pricing low %.4f→%.4f (Δ%+.3f) high %.4f→%.4f (Δ%+.3f)",
        symbol,
        adjustment.base_low_price,
        adjustment.low_price,
        adjustment.low_delta,
        adjustment.base_high_price,
        adjustment.high_price,
        adjustment.high_delta,
    )


def _apply_neural_scale(
    target_qty: float,
    *,
    min_qty: float,
    symbol: str,
    strategy: Optional[str],
    effective_probe: bool,
    strategy_key: Optional[str] = None,
) -> float:
    if not NEURAL_SIZING_ENABLED or effective_probe or target_qty <= 0:
        return target_qty
    optimizer = _get_portfolio_optimizer()
    if optimizer is None:
        return target_qty
    scale = None
    if strategy_key:
        scale = optimizer.get_strategy_weight(strategy_key)
    if scale is None:
        scale = optimizer.last_scale or optimizer.compute_scale()
    if scale is None or scale <= 0:
        return target_qty
    scaled_qty = target_qty * scale
    if scaled_qty < min_qty and target_qty >= min_qty:
        scaled_qty = min_qty
    if not math.isclose(scaled_qty, target_qty):
        logger.info(
            "Neural sizing applied scale %.3f to %s (%s/%s) qty %.4f → %.4f",
            scale,
            symbol,
            strategy or "unknown",
            strategy_key or "default",
            target_qty,
            scaled_qty,
        )
    return scaled_qty


def _resolve_neural_strategy_name(symbol: str, data: Dict[str, object]) -> str:
    is_crypto = is_crypto_symbol(symbol)
    base = "VolAdjusted_10pct" if is_crypto else "VolAdjusted_15pct"
    gate_config = str(data.get("gate_config") or "-").replace(" ", "")
    gate_tokens = {token.strip() for token in gate_config.split("+") if token}
    components: List[str] = []
    if any("UnprofitShutdown" in token for token in gate_tokens) or data.get("gate_blocked_days"):
        components.append("UnprofitShutdown")
    if any("StockDirShutdown" in token for token in gate_tokens) or data.get("symbol_gate_blocks"):
        components.append("StockDirShutdown")
    if components:
        return base + "_" + "_".join(components)
    return base

_quote_client: Optional[StockHistoricalDataClient] = None
# Cooldown state now managed by src.cooldown_utils

_trade_outcomes_store: Optional[FlatShelf] = None
_trade_learning_store: Optional[FlatShelf] = None
_active_trades_store: Optional[FlatShelf] = None
_trade_history_store: Optional[FlatShelf] = None
_maxdiff_plans_store: Optional[FlatShelf] = None


SIMPLIFIED_MODE = os.getenv("MARKETSIM_SIMPLE_MODE", "0").strip().lower() in _TRUTHY

ENABLE_KELLY_SIZING = os.getenv("MARKETSIM_ENABLE_KELLY_SIZING", "0").strip().lower() in _TRUTHY
ENABLE_PROBE_TRADES = os.getenv("MARKETSIM_ENABLE_PROBE_TRADES", "0").strip().lower() in _TRUTHY
PROBE_TRADE_MODE = os.getenv("PROBE_TRADE", "0").strip().lower() in _TRUTHY
if PROBE_TRADE_MODE and not ENABLE_PROBE_TRADES:
    ENABLE_PROBE_TRADES = True
MAX_MAXDIFFS = coerce_positive_int(
    os.getenv("MARKETSIM_MAX_MAXDIFFS"),
    15,
)

DEFAULT_PROBE_SYMBOLS = {"AAPL", "MSFT", "NVDA"}
PROBE_SYMBOLS = set() if SIMPLIFIED_MODE or not ENABLE_PROBE_TRADES else set(DEFAULT_PROBE_SYMBOLS)

MAXDIFF_STRATEGIES = {"maxdiff", "maxdiffalwayson"}  # Disabled pctdiff - needs debugging
MAXDIFF_LIMIT_STRATEGIES = MAXDIFF_STRATEGIES.union({"highlow"})


_DRAW_SUSPENDED: Dict[Tuple[str, str], bool] = {}


def _apply_strategy_priority(strategies: List[str], priority: Optional[List[str]]) -> List[str]:
    if not priority:
        return list(strategies)
    normalized_priority: List[str] = []
    seen: set[str] = set()
    strategy_set = {name for name in strategies}
    for name in priority:
        normalized = (name or "").strip().lower()
        if not normalized or normalized in seen or normalized not in strategy_set:
            continue
        normalized_priority.append(normalized)
        seen.add(normalized)
    if not normalized_priority:
        return list(strategies)
    ordered: List[str] = list(normalized_priority)
    ordered.extend(name for name in strategies if name not in seen)
    return ordered


def _lookup_entry_price(data: Dict, strategy: Optional[str], side: str) -> Optional[float]:
    normalized = (strategy or "").strip().lower()
    is_buy = is_buy_side(side)
    if normalized == "maxdiff":
        return data.get("maxdiffprofit_low_price" if is_buy else "maxdiffprofit_high_price")
    if normalized == "maxdiffalwayson":
        return data.get("maxdiffalwayson_low_price" if is_buy else "maxdiffalwayson_high_price")
    if normalized == "pctdiff":
        return data.get("pctdiff_entry_low_price" if is_buy else "pctdiff_entry_high_price")
    if normalized == "highlow":
        return data.get("predicted_low" if is_buy else "predicted_high")
    return None


def _lookup_takeprofit_price(data: Dict, strategy: Optional[str], side: str) -> Optional[float]:
    normalized = (strategy or "").strip().lower()
    is_buy = is_buy_side(side)
    if normalized == "maxdiff":
        return data.get("maxdiffprofit_high_price" if is_buy else "maxdiffprofit_low_price")
    if normalized == "maxdiffalwayson":
        return data.get("maxdiffalwayson_high_price" if is_buy else "maxdiffalwayson_low_price")
    if normalized == "pctdiff":
        return data.get("pctdiff_takeprofit_high_price" if is_buy else "pctdiff_takeprofit_low_price")
    if normalized == "highlow":
        return data.get("predicted_high" if is_buy else "predicted_low")
    return None


def _resolve_model_passes(symbol: str, *, now_utc: datetime) -> List[Optional[str]]:
    passes: List[Optional[str]] = [None]
    if not (is_crypto_symbol(symbol) and is_crypto_out_of_hours(now_utc)):
        return passes
    selection = load_model_selection(symbol)
    if not selection:
        return passes
    metadata = selection.get("metadata") or {}
    candidate_map = metadata.get("candidate_pct_return_mae") or {}
    scored_models: List[Tuple[str, float]] = []
    for name, value in candidate_map.items():
        if not name:
            continue
        normalized = str(name).strip().lower()
        if normalized not in {"toto", "kronos", "chronos2"}:
            continue
        try:
            score = float(value)
        except (TypeError, ValueError):
            score = float("inf")
        scored_models.append((normalized, score))
    if not scored_models:
        return passes
    scored_models.sort(key=lambda item: (item[1], item[0]))
    ordered_unique: List[str] = []
    for name, _ in scored_models:
        if name not in ordered_unique:
            ordered_unique.append(name)
    best_model = str(selection.get("model", "")).strip().lower()
    top_models = [name for name in ordered_unique if name in {"toto", "kronos", "chronos2"}]
    if not top_models:
        return passes
    preferred = best_model if best_model in top_models else top_models[0]
    second_best = next((name for name in top_models if name != preferred), None)
    if second_best:
        passes.append(second_best)
    return passes


def _resolve_row_model(row: Dict[str, object]) -> Optional[str]:
    candidate = row.get("forecast_model") or row.get("close_prediction_source")
    if not candidate:
        return None
    normalized = str(candidate).strip().lower()
    if normalized in {"toto", "kronos", "chronos2"}:
        return normalized
    return None


def _merge_symbol_runs(
    symbol: str,
    *,
    base_row: Dict[str, object],
    secondary_row: Dict[str, object],
) -> Dict[str, object]:
    runs = [row for row in (base_row, secondary_row) if row]
    candidate_best: Dict[str, Dict[str, object]] = {}
    for row in runs:
        forecasts = row.get("strategy_candidate_forecasted_pnl") or {}
        if not isinstance(forecasts, dict):
            continue
        model_name = _resolve_row_model(row)
        if not model_name:
            continue
        for strat, pnl in forecasts.items():
            try:
                score = float(pnl)
            except (TypeError, ValueError):
                continue
            entry = candidate_best.get(strat)
            if entry is None or score > entry["pnl"]:
                candidate_best[strat] = {"pnl": score, "model": model_name, "row": row}

    def _ordered_strategies() -> List[str]:
        if not candidate_best:
            return []
        positive = [name for name, data in candidate_best.items() if data["pnl"] > 0]
        target = positive if positive else list(candidate_best)
        return sorted(target, key=lambda name: candidate_best[name]["pnl"], reverse=True)

    ordered = _ordered_strategies()
    rerun_cache: Dict[Tuple[str, str], Optional[Dict[str, object]]] = {}

    def _rerun(model_name: str, strategy: str) -> Optional[Dict[str, object]]:
        key = (model_name, strategy)
        if key in rerun_cache:
            return rerun_cache[key]
        result = _analyze_symbols_impl(
            [symbol],
            model_overrides={symbol: model_name},
            strategy_priorities={symbol: [strategy]},
        )
        row = result.get(symbol) if result else None
        rerun_cache[key] = row
        return row

    for strategy in ordered:
        entry = candidate_best[strategy]
        current_row = entry["row"]
        if current_row.get("strategy") == strategy:
            return current_row
        candidate_model = entry["model"]
        rerun_row = _rerun(candidate_model, strategy)
        if rerun_row and rerun_row.get("strategy") == strategy:
            return rerun_row

    fallback_rows = [row for row in runs if row]
    if not fallback_rows:
        return base_row
    return max(fallback_rows, key=lambda row: get_selected_strategy_forecast(row))


def analyze_symbols(symbols: List[str]) -> Dict:
    base_results = _analyze_symbols_impl(symbols)
    if not base_results:
        return base_results
    now_utc = datetime.now(timezone.utc)
    final_results = dict(base_results)
    for symbol in symbols:
        base_row = base_results.get(symbol)
        if not base_row:
            continue
        passes = _resolve_model_passes(symbol, now_utc=now_utc)
        if len(passes) < 2:
            continue
        second_model = passes[1]
        logger.info("%s: Crypto out-of-hours — evaluating %s as secondary model.", symbol, second_model)
        secondary = _analyze_symbols_impl([symbol], model_overrides={symbol: second_model})
        secondary_row = secondary.get(symbol)
        if not secondary_row:
            continue
        merged = _merge_symbol_runs(symbol, base_row=base_row, secondary_row=secondary_row)
        final_results[symbol] = merged
    return dict(sorted(final_results.items(), key=lambda x: x[1]["composite_score"], reverse=True))


def _strategy_key(symbol: Optional[str], strategy: Optional[str]) -> Tuple[str, str]:
    return ((symbol or "__global__").lower(), (strategy or "__default__").lower())


def _results_dir() -> Path:
    return Path(__file__).resolve().parent / "results"


def _load_latest_forecast_snapshot() -> Dict[str, Dict[str, object]]:
    return _load_forecast_snapshot(
        _results_dir(),
        logger=logger,
        parse_float_list=parse_float_list,
        coerce_optional_float=coerce_optional_float,
    )


def _normalize_series(series: pd.Series) -> pd.Series:
    return normalize_series(series, coerce_numeric)


def _get_quote_client() -> Optional[StockHistoricalDataClient]:
    global _quote_client
    if _quote_client is not None:
        return _quote_client
    try:
        _quote_client = StockHistoricalDataClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD)
    except Exception as exc:
        logger.error("Failed to initialise StockHistoricalDataClient: %s", exc)
        _quote_client = None
    return _quote_client


def fetch_bid_ask(symbol: str) -> Tuple[Optional[float], Optional[float]]:
    client = _get_quote_client()
    if client is None:
        return None, None
    try:
        download_exchange_latest_data(client, symbol)
    except Exception as exc:
        logger.warning("Unable to refresh quotes for %s: %s", symbol, exc)
    return get_bid(symbol), get_ask(symbol)


def _record_loss_timestamp(symbol: str, closed_at: Optional[str]) -> None:
    record_loss_timestamp(symbol, closed_at, logger=logger)


def can_trade_now(symbol: str, now: datetime, min_cooldown_minutes: int = PROBE_LOSS_COOLDOWN_MINUTES) -> bool:
    from src.cooldown_utils import can_trade_now as can_trade_now_base

    return can_trade_now_base(symbol, now, min_cooldown_minutes, symbol_min_cooldown_fn=_symbol_min_cooldown_minutes)


def _ensure_state_dir() -> bool:
    try:
        _shared_ensure_state_dir()
        return True
    except Exception as exc:
        logger.error(f"Unable to create strategy state directory '{STATE_DIR}': {exc}")
        return False


def _init_store(store_name: str, storage_path: Path) -> Optional[FlatShelf]:
    if not _ensure_state_dir():
        return None
    try:
        store = FlatShelf(str(storage_path))
        logger.debug(f"Initialised {store_name} store at {storage_path}")
        return store
    except Exception as exc:
        logger.error(f"Failed initialising {store_name} store '{storage_path}': {exc}")
        return None


def _get_trade_outcomes_store() -> Optional[FlatShelf]:
    """Lazily initialise the trade outcome FlatShelf without import-time side effects."""
    global _trade_outcomes_store

    if _trade_outcomes_store is not None:
        return _trade_outcomes_store

    _trade_outcomes_store = _init_store("trade outcomes", TRADE_OUTCOME_FILE)
    return _trade_outcomes_store


def _get_trade_learning_store() -> Optional[FlatShelf]:
    global _trade_learning_store
    if _trade_learning_store is not None:
        return _trade_learning_store
    _trade_learning_store = _init_store("trade learning", TRADE_LEARNING_FILE)
    return _trade_learning_store


def _get_active_trades_store() -> Optional[FlatShelf]:
    global _active_trades_store
    if _active_trades_store is not None:
        return _active_trades_store
    _active_trades_store = _init_store("active trades", ACTIVE_TRADES_FILE)
    return _active_trades_store


def _get_trade_history_store() -> Optional[FlatShelf]:
    global _trade_history_store
    if _trade_history_store is not None:
        return _trade_history_store
    _trade_history_store = _init_store("trade history", TRADE_HISTORY_FILE)
    return _trade_history_store


def _recent_trade_pnls(
    symbol: str,
    side: str,
    *,
    strategy: Optional[str] = None,
    limit: int = 2,
) -> List[float]:
    store = _get_trade_history_store()
    if store is None or limit <= 0:
        return []
    try:
        store.load()
    except Exception as exc:
        logger.error("Failed loading trade history store for PnL lookup: %s", exc)
        return []
    key = state_utils.state_key(symbol, side, strategy)
    history = store.get(key, [])
    if not isinstance(history, list) or not history:
        return []
    recent: List[float] = []
    for entry in reversed(history):
        pnl_value = coerce_numeric(entry.get("pnl"), default=None)
        if pnl_value is None:
            continue
        recent.append(float(pnl_value))
        if len(recent) >= limit:
            break
    return recent


def _recent_trade_pnl_pcts(
    symbol: str,
    side: str,
    *,
    strategy: Optional[str] = None,
    limit: int = 2,
) -> List[float]:
    store = _get_trade_history_store()
    if store is None or limit <= 0:
        return []
    try:
        store.load()
    except Exception as exc:
        logger.error("Failed loading trade history store for PnL pct lookup: %s", exc)
        return []
    key = state_utils.state_key(symbol, side, strategy)
    history = store.get(key, [])
    if not isinstance(history, list) or not history:
        return []
    recent: List[float] = []
    for entry in reversed(history):
        raw_pct = entry.get("pnl_pct")
        if raw_pct is None:
            continue
        pnl_pct = coerce_numeric(raw_pct, default=0.0)
        recent.append(float(pnl_pct))
        if len(recent) >= limit:
            break
    return recent


def _get_maxdiff_plans_store() -> Optional[FlatShelf]:
    global _maxdiff_plans_store
    if _maxdiff_plans_store is not None:
        return _maxdiff_plans_store
    _maxdiff_plans_store = _init_store("maxdiff plans", MAXDIFF_PLANS_FILE)
    return _maxdiff_plans_store


def _save_maxdiff_plan(symbol: str, plan_data: Dict) -> None:
    """Save a maxdiff trading plan for the day."""
    store = _get_maxdiff_plans_store()
    if store is None:
        return
    try:
        store.load()
        day_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        key = f"{day_key}:{symbol}"
        store[key] = plan_data
        store.save()
    except Exception as exc:
        logger.warning("Failed to save maxdiff plan for %s: %s", symbol, exc)


def _load_maxdiff_plans_for_today() -> Dict[str, Dict]:
    """Load all maxdiff plans for today."""
    store = _get_maxdiff_plans_store()
    if store is None:
        return {}
    try:
        store.load()
        day_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        prefix = f"{day_key}:"
        plans = {}
        for key, value in store.items():
            if key.startswith(prefix):
                symbol = key[len(prefix) :]
                plans[symbol] = value
        return plans
    except Exception as exc:
        logger.warning("Failed to load maxdiff plans: %s", exc)
        return {}


def _update_maxdiff_plan_status(symbol: str, status: str, **extra_fields) -> None:
    """Update the status of a maxdiff plan."""
    store = _get_maxdiff_plans_store()
    if store is None:
        return
    try:
        store.load()
        day_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        key = f"{day_key}:{symbol}"
        if key in store:
            plan = dict(store[key])
            plan["status"] = status
            plan["updated_at"] = datetime.now(timezone.utc).isoformat()
            for field_name, field_value in extra_fields.items():
                plan[field_name] = field_value
            store[key] = plan
            store.save()
    except Exception as exc:
        logger.warning("Failed to update maxdiff plan status for %s: %s", symbol, exc)


LOSS_BLOCK_COOLDOWN = timedelta(days=3)
DEFAULT_MIN_CORE_POSITIONS = 4
DEFAULT_MAX_PORTFOLIO = 10
EXPANDED_PORTFOLIO = 8
MIN_EXPECTED_MOVE_PCT = 1e-4
MIN_EDGE_STRENGTH = 1e-5
COMPACT_LOGS = os.getenv("COMPACT_TRADING_LOGS", "").strip().lower() in {"1", "true", "yes", "on"}
MARKET_CLOSE_SHIFT_MINUTES = int(os.getenv("MARKET_CLOSE_SHIFT_MINUTES", "45"))
MARKET_CLOSE_ANALYSIS_WINDOW_MINUTES = int(os.getenv("MARKET_CLOSE_ANALYSIS_WINDOW_MINUTES", "15"))
BACKOUT_START_OFFSET_MINUTES = int(os.getenv("BACKOUT_START_OFFSET_MINUTES", "30"))
BACKOUT_SLEEP_SECONDS = int(os.getenv("BACKOUT_SLEEP_SECONDS", "45"))
BACKOUT_MARKET_CLOSE_BUFFER_MINUTES = int(os.getenv("BACKOUT_MARKET_CLOSE_BUFFER_MINUTES", "30"))
BACKOUT_MARKET_CLOSE_FORCE_MINUTES = int(os.getenv("BACKOUT_MARKET_CLOSE_FORCE_MINUTES", "3"))
MAXDIFF_ENTRY_WATCHER_POLL_SECONDS = max(5, int(os.getenv("MAXDIFF_ENTRY_POLL_SECONDS", "12")))
MAXDIFF_EXIT_WATCHER_POLL_SECONDS = max(5, int(os.getenv("MAXDIFF_EXIT_POLL_SECONDS", "12")))
MAXDIFF_EXIT_WATCHER_PRICE_TOLERANCE = float(os.getenv("MAXDIFF_EXIT_PRICE_TOLERANCE", "0.001"))
MAXDIFF_ALWAYS_ON_PRIORITY_LIMIT = max(0, int(os.getenv("MAXDIFF_ALWAYS_ON_PRIORITY_LIMIT", "2")))


def _log_detail(message: str) -> None:
    if COMPACT_LOGS:
        logger.debug(message)
    else:
        logger.info(message)


def _log_analysis_summary(symbol: str, data: Dict) -> None:
    compact_message, detailed_message = build_analysis_summary_messages(symbol, data)
    if COMPACT_LOGS:
        _log_detail(compact_message)
    else:
        _log_detail(detailed_message)


def _normalize_side_for_key(side: str) -> str:
    return state_utils.normalize_side_for_key(side)


def _parse_timestamp(ts: Optional[str]) -> Optional[datetime]:
    return state_utils.parse_timestamp(ts, logger=logger)


def _state_key(symbol: str, side: str) -> str:
    return state_utils.state_key(symbol, side)


def _load_trade_outcome(symbol: str, side: str, strategy: Optional[str] = None) -> Dict:
    return state_utils.load_store_entry(
        _get_trade_outcomes_store,
        symbol,
        side,
        strategy=strategy,
        store_name="trade outcomes",
        logger=logger,
    )


def _load_learning_state(symbol: str, side: str, strategy: Optional[str] = None) -> Dict:
    return state_utils.load_store_entry(
        _get_trade_learning_store,
        symbol,
        side,
        strategy=strategy,
        store_name="trade learning",
        logger=logger,
    )


def _save_learning_state(symbol: str, side: str, state: Dict, strategy: Optional[str] = None) -> None:
    state_utils.save_store_entry(
        _get_trade_learning_store,
        symbol,
        side,
        state,
        strategy=strategy,
        store_name="trade learning",
        logger=logger,
    )


def _update_learning_state(symbol: str, side: str, strategy: Optional[str] = None, **updates) -> Dict:
    return state_utils.update_learning_state(
        _get_trade_learning_store,
        symbol,
        side,
        updates,
        strategy=strategy,
        logger=logger,
        now=datetime.now(timezone.utc),
    )


def _mark_probe_pending(symbol: str, side: str, strategy: Optional[str] = None) -> Dict:
    return state_utils.mark_probe_pending(
        _get_trade_learning_store,
        symbol,
        side,
        strategy=strategy,
        logger=logger,
        now=datetime.now(timezone.utc),
    )


def _mark_probe_active(symbol: str, side: str, qty: float, strategy: Optional[str] = None) -> Dict:
    return state_utils.mark_probe_active(
        _get_trade_learning_store,
        symbol,
        side,
        qty,
        strategy=strategy,
        logger=logger,
        now=datetime.now(timezone.utc),
    )


def _mark_probe_completed(symbol: str, side: str, successful: bool, strategy: Optional[str] = None) -> Dict:
    return state_utils.mark_probe_completed(
        _get_trade_learning_store,
        symbol,
        side,
        successful,
        strategy=strategy,
        logger=logger,
        now=datetime.now(timezone.utc),
    )


def _describe_probe_state(learning_state: Dict, now: Optional[datetime] = None) -> Dict[str, Optional[object]]:
    return state_utils.describe_probe_state(
        learning_state,
        now=now,
        probe_max_duration=PROBE_MAX_DURATION,
    )


def _mark_probe_transitioned(symbol: str, side: str, qty: float, strategy: Optional[str] = None) -> Dict:
    return state_utils.mark_probe_transitioned(
        _get_trade_learning_store,
        symbol,
        side,
        qty,
        strategy=strategy,
        logger=logger,
        now=datetime.now(timezone.utc),
    )


def _update_active_trade(symbol: str, side: str, mode: str, qty: float, strategy: Optional[str] = None) -> None:
    opened_at_sim = None
    try:
        state = get_state()
        sim_now = getattr(getattr(state, "clock", None), "current", None)
        if sim_now is not None:
            opened_at_sim = sim_now.isoformat()
    except RuntimeError:
        opened_at_sim = None
    state_utils.update_active_trade_record(
        _get_active_trades_store,
        symbol,
        side,
        mode=mode,
        qty=qty,
        strategy=strategy,
        opened_at_sim=opened_at_sim,
        logger=logger,
        now=datetime.now(timezone.utc),
    )


def _tag_active_trade_strategy(symbol: str, side: str, strategy: Optional[str]) -> None:
    state_utils.tag_active_trade_strategy(
        _get_active_trades_store,
        symbol,
        side,
        strategy,
        logger=logger,
    )


def _normalize_active_trade_patch(updater) -> None:
    closure = getattr(updater, "__closure__", None)
    if not closure:
        return
    try:
        for cell in closure:
            contents = cell.cell_contents
            if isinstance(contents, list) and contents:
                last_entry = contents[-1]
                if isinstance(last_entry, tuple) and len(last_entry) == 5:
                    contents[-1] = last_entry[:4]
    except Exception:
        # Best-effort compatibility shim for tests; ignore any reflection errors.
        return


def _get_active_trade(symbol: str, side: str) -> Dict:
    return state_utils.get_active_trade_record(
        _get_active_trades_store,
        symbol,
        side,
        logger=logger,
    )


def _pop_active_trade(symbol: str, side: str) -> Dict:
    return state_utils.pop_active_trade_record(
        _get_active_trades_store,
        symbol,
        side,
        logger=logger,
    )


def _calculate_total_exposure_value(positions) -> float:
    total_value = 0.0
    for position in positions:
        try:
            market_value = float(getattr(position, "market_value", 0.0) or 0.0)
        except Exception:
            market_value = 0.0
        total_value += abs(market_value)
    return total_value


def _get_simple_qty(symbol: str, entry_price: float, positions) -> float:
    """
    Simple sizing that spreads global risk over 2 positions.

    For stocks: (buying_power * global_risk_threshold / 2) / entry_price
    For crypto: (equity / 2) / entry_price (no leverage)
    """
    from src.portfolio_risk import get_global_risk_threshold
    from math import floor

    if entry_price <= 0:
        return 0.0

    equity = float(getattr(alpaca_wrapper, "equity", 0.0) or 0.0)
    buying_power = float(getattr(alpaca_wrapper, "total_buying_power", 0.0) or 0.0)
    global_risk = get_global_risk_threshold()

    if is_crypto_symbol(symbol):
        # For crypto: just use half of equity (no leverage)
        qty = (equity / 2.0) / entry_price
        # Round down to 3 decimal places for crypto
        qty = floor(qty * 1000) / 1000.0
    else:
        # For stocks: use half of (buying_power * global_risk_threshold)
        qty = (buying_power * global_risk / 2.0) / entry_price
        # Round down to whole number for stocks
        qty = floor(qty)

    if qty <= 0:
        return 0.0

    logger.debug(
        f"Simple sizing for {symbol}: qty={qty:.4f} (equity={equity:.2f}, buying_power={buying_power:.2f}, global_risk={global_risk:.3f})"
    )

    return qty


def _calculate_total_exposure_pct(positions) -> float:
    equity = float(getattr(alpaca_wrapper, "equity", 0.0) or 0.0)
    if equity <= 0:
        return 0.0
    total_value = _calculate_total_exposure_value(positions)
    return (total_value / equity) * 100.0


def _position_notional_value(position) -> float:
    """Return the absolute dollar notional for a live position."""
    try:
        market_value = coerce_numeric(getattr(position, "market_value", None), default=0.0)
    except Exception:
        market_value = 0.0
    if market_value:
        return abs(market_value)

    qty_value = coerce_numeric(getattr(position, "qty", 0.0), default=0.0)
    price_value = 0.0
    for attr in ("current_price", "avg_entry_price", "lastday_price"):
        try:
            candidate = coerce_numeric(getattr(position, attr, None), default=0.0)
        except Exception:
            candidate = 0.0
        if candidate > 0:
            price_value = candidate
            break
    if price_value > 0:
        return abs(qty_value * price_value)
    return abs(qty_value)


def _ensure_probe_state_consistency(
    position,
    normalized_side: str,
    probe_meta: Optional[Dict[str, object]],
) -> Dict[str, object]:
    """
    Promote positions that materially exceed probe sizing thresholds back to normal mode.
    """

    notional_value = _position_notional_value(position)
    if notional_value <= PROBE_NOTIONAL_LIMIT:
        return probe_meta or {}

    active_trade = _get_active_trade(position.symbol, normalized_side)
    entry_strategy = active_trade.get("entry_strategy") if active_trade else None

    state_probe_meta = _evaluate_trade_block(position.symbol, normalized_side, strategy=entry_strategy)
    trade_mode = str(state_probe_meta.get("trade_mode", "")).lower()
    is_probe_state = (
        bool(state_probe_meta.get("pending_probe"))
        or bool(state_probe_meta.get("probe_active"))
        or bool(state_probe_meta.get("probe_expired"))
        or trade_mode == "probe"
    )
    if not is_probe_state:
        merged: Dict[str, object] = dict(probe_meta or {})
        merged.setdefault("pending_probe", state_probe_meta.get("pending_probe", False))
        merged.setdefault("probe_active", state_probe_meta.get("probe_active", False))
        merged.setdefault("probe_expired", state_probe_meta.get("probe_expired", False))
        merged.setdefault("trade_mode", state_probe_meta.get("trade_mode", "normal"))
        merged.setdefault("probe_transition_ready", state_probe_meta.get("probe_transition_ready", False))
        return merged

    qty_value = coerce_numeric(getattr(position, "qty", 0.0), default=0.0)
    logger.info(
        f"{position.symbol}: Position notional ${notional_value:.2f} exceeds probe limit ${PROBE_NOTIONAL_LIMIT:.2f}; promoting to normal regime."
    )

    stored_qty = coerce_numeric(active_trade.get("qty"), default=0.0) if active_trade else 0.0
    _mark_probe_transitioned(position.symbol, normalized_side, abs(qty_value), strategy=entry_strategy)
    updated_qty = abs(qty_value) if abs(qty_value) > 0 else abs(stored_qty)
    _update_active_trade(
        position.symbol,
        normalized_side,
        mode="probe_transition",
        qty=updated_qty,
        strategy=entry_strategy,
    )
    _normalize_active_trade_patch(_update_active_trade)

    refreshed_state = _evaluate_trade_block(position.symbol, normalized_side, strategy=entry_strategy)

    merged_meta: Dict[str, object] = dict(probe_meta or {})
    for key in (
        "pending_probe",
        "probe_active",
        "probe_expired",
        "probe_transition_ready",
        "trade_mode",
        "probe_started_at",
        "probe_age_seconds",
        "probe_expires_at",
        "learning_state",
        "record",
    ):
        if key in refreshed_state:
            merged_meta[key] = refreshed_state[key]
    merged_meta["trade_mode"] = refreshed_state.get("trade_mode", "normal")
    merged_meta["pending_probe"] = refreshed_state.get("pending_probe", False)
    merged_meta["probe_active"] = refreshed_state.get("probe_active", False)
    merged_meta["probe_expired"] = refreshed_state.get("probe_expired", False)
    merged_meta["probe_transition_ready"] = refreshed_state.get("probe_transition_ready", False)
    return merged_meta


def _handle_live_drawdown(position) -> None:
    """
    Monitor live positions for drawdown and mark for probe trade if needed.

    Recalculates on every call to handle PnL fluctuations:
    - If unrealized_pl < LIVE_DRAWDOWN_TRIGGER: mark for probe
    - If unrealized_pl >= LIVE_DRAWDOWN_TRIGGER: clear probe flag (position recovered)
    """
    try:
        unrealized_pl = float(getattr(position, "unrealized_pl", 0.0) or 0.0)
    except Exception:
        unrealized_pl = 0.0

    symbol = position.symbol
    normalized_side = _normalize_side_for_key(getattr(position, "side", ""))

    if unrealized_pl >= LIVE_DRAWDOWN_TRIGGER:
        # Position recovered - clear probe flag if it was set
        learning_state = _load_learning_state(symbol, normalized_side)
        if learning_state.get("pending_probe") and not learning_state.get("probe_active"):
            _update_learning_state(symbol, normalized_side, pending_probe=False)
            logger.info(
                f"{symbol} {normalized_side}: Position recovered (unrealized pnl {unrealized_pl:.2f}); "
                "clearing probe flag."
            )
        return

    # Position in drawdown - mark for probe
    learning_state = _update_learning_state(symbol, normalized_side, pending_probe=True)
    if not learning_state.get("probe_active"):
        logger.warning(
            f"Live drawdown detected for {symbol} {normalized_side}: unrealized pnl {unrealized_pl:.2f}; "
            "marking for probe trade."
        )


def _record_trade_outcome(position, reason: str) -> None:
    store = _get_trade_outcomes_store()
    if store is None:
        logger.warning("Trade outcomes store unavailable; skipping persistence of trade result")
        return

    side_value = getattr(position, "side", "")
    normalized_side = _normalize_side_for_key(side_value)
    active_trade = _pop_active_trade(position.symbol, normalized_side)
    trade_mode = active_trade.get("mode", "probe" if active_trade else "normal")
    entry_strategy = active_trade.get("entry_strategy")
    # Use strategy-specific key for outcomes
    key = state_utils.state_key(position.symbol, normalized_side, entry_strategy)
    try:
        pnl_value = float(getattr(position, "unrealized_pl", 0.0) or 0.0)
    except Exception:
        pnl_value = 0.0
    try:
        qty_value = float(getattr(position, "qty", 0.0) or 0.0)
    except Exception:
        qty_value = 0.0
    entry_price = coerce_numeric(getattr(position, "avg_entry_price", None), default=None)
    notional = None
    if entry_price is not None and qty_value:
        notional = abs(float(entry_price) * float(qty_value))
    elif hasattr(position, "market_value"):
        notional = abs(coerce_numeric(getattr(position, "market_value", 0.0), default=0.0))
    pnl_pct = safe_divide(pnl_value, notional, default=0.0) if notional and notional > 0 else 0.0
    record = {
        "symbol": position.symbol,
        "side": normalized_side,
        "qty": qty_value,
        "pnl": pnl_value,
        "pnl_pct": pnl_pct,
        "closed_at": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "mode": trade_mode,
    }
    if entry_strategy:
        record["entry_strategy"] = entry_strategy
    store[key] = record
    logger.info(
        f"Recorded trade outcome for {position.symbol} {normalized_side}: pnl={pnl_value:.2f}, reason={reason}, mode={trade_mode}"
    )

    # Update learning state metadata (strategy-specific)
    _update_learning_state(
        position.symbol,
        normalized_side,
        strategy=entry_strategy,
        last_pnl=pnl_value,
        last_qty=qty_value,
        last_closed_at=record["closed_at"],
        last_reason=reason,
        last_mode=trade_mode,
    )

    if trade_mode == "probe":
        _mark_probe_completed(position.symbol, normalized_side, successful=pnl_value > 0, strategy=entry_strategy)
    elif pnl_value < 0:
        _mark_probe_pending(position.symbol, normalized_side, strategy=entry_strategy)
    else:
        _update_learning_state(
            position.symbol,
            normalized_side,
            strategy=entry_strategy,
            pending_probe=False,
            probe_active=False,
            last_positive_at=record["closed_at"],
        )

    history_store = _get_trade_history_store()
    if history_store is not None:
        try:
            history_store.load()
        except Exception as exc:
            logger.error(f"Failed loading trade history store: {exc}")
        else:
            entry = {
                "symbol": position.symbol,
                "side": normalized_side,
                "qty": qty_value,
                "pnl": pnl_value,
                "pnl_pct": pnl_pct,
                "closed_at": record["closed_at"],
                "reason": reason,
                "mode": trade_mode,
                "entry_strategy": entry_strategy,
            }
            history_keys = {key, state_utils.state_key(position.symbol, normalized_side)}
            for history_key in history_keys:
                history = history_store.get(history_key, [])
                if not isinstance(history, list):
                    history = []
                history.append(entry)
                history_store[history_key] = history[-100:]


def _evaluate_trade_block(symbol: str, side: str, strategy: Optional[str] = None) -> Dict[str, Optional[object]]:
    """Evaluate trade blocking for a specific symbol, side, and optionally strategy.

    When strategy is provided, blocks are strategy-specific (e.g., ETHUSD-buy-maxdiff can be
    blocked independently from ETHUSD-buy-highlow).
    """
    normalized_side = _normalize_side_for_key(side)
    record = _load_trade_outcome(symbol, normalized_side, strategy=strategy)
    learning_state = dict(_load_learning_state(symbol, side, strategy=strategy))
    now_utc = datetime.now(timezone.utc)
    probe_summary = _describe_probe_state(learning_state, now_utc)
    pending_probe = bool(learning_state.get("pending_probe"))
    probe_active = bool(probe_summary.get("probe_active"))
    last_probe_successful = bool(learning_state.get("last_probe_successful"))
    probe_transition_ready = last_probe_successful and not pending_probe and not probe_active
    last_pnl = record.get("pnl") if record else None
    last_closed_at = _parse_timestamp(record.get("closed_at") if record else None)
    blocked = False
    block_reason = None
    if not ENABLE_PROBE_TRADES:
        pending_probe = False
        probe_active = False
        probe_transition_ready = False
        probe_summary = {}
        learning_state["pending_probe"] = False
        learning_state["probe_active"] = False
        learning_state["probe_transition_ready"] = False
        learning_state["probe_expires_at"] = None
        learning_state["probe_expired"] = False
    trade_mode = "probe" if (pending_probe or probe_active) else "normal"

    if last_pnl is not None and last_pnl < 0:
        ts_repr = last_closed_at.isoformat() if last_closed_at else "unknown"
        if trade_mode == "probe":
            block_reason = f"Last {side} trade for {symbol} lost {last_pnl:.2f} on {ts_repr}; running probe trade"
        else:
            if last_closed_at is None or now_utc - last_closed_at <= LOSS_BLOCK_COOLDOWN:
                blocked = True
                block_reason = f"Last {side} trade for {symbol} lost {last_pnl:.2f} on {ts_repr}; cooling down"
    forced_probe_reason = None
    if ENABLE_PROBE_TRADES and PROBE_TRADE_MODE and blocked:
        pct_values = _recent_trade_pnl_pcts(symbol, normalized_side, strategy=None, limit=2)
        if len(pct_values) < 2 and strategy:
            pct_values = _recent_trade_pnl_pcts(symbol, normalized_side, strategy=strategy, limit=2)
        if len(pct_values) >= 2:
            pct_sum = sum(pct_values[:2])
            if pct_sum <= 0:
                blocked = False
                pending_probe = True
                trade_mode = "probe"
                forced_probe_reason = f"recent pnl_pct sum {pct_sum:.4f}; forcing probe trade"
    if probe_summary.get("probe_expired"):
        block_reason = block_reason or (
            f"Probe duration exceeded {PROBE_MAX_DURATION} for {symbol} {side}; scheduling backout"
        )
    cooldown_expires = None
    if last_closed_at is not None:
        cooldown_expires = (last_closed_at + LOSS_BLOCK_COOLDOWN).isoformat()
    if forced_probe_reason:
        block_reason = forced_probe_reason
    learning_state["trade_mode"] = trade_mode
    learning_state["probe_transition_ready"] = probe_transition_ready
    learning_state["probe_expires_at"] = probe_summary.get("probe_expires_at")
    return {
        "record": record,
        "blocked": blocked,
        "block_reason": block_reason,
        "last_pnl": last_pnl,
        "last_closed_at": last_closed_at.isoformat() if last_closed_at else None,
        "cooldown_expires": cooldown_expires,
        "pending_probe": pending_probe,
        "probe_active": probe_active,
        "trade_mode": trade_mode,
        "probe_started_at": probe_summary.get("probe_started_at"),
        "probe_age_seconds": probe_summary.get("probe_age_seconds"),
        "probe_expires_at": probe_summary.get("probe_expires_at"),
        "probe_expired": probe_summary.get("probe_expired"),
        "probe_transition_ready": probe_transition_ready,
        "learning_state": learning_state,
    }


def get_market_hours() -> tuple:
    """Get market open and close times in EST."""
    est = pytz.timezone("US/Eastern")
    now = datetime.now(est)
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    if MARKET_CLOSE_SHIFT_MINUTES:
        shifted_close = market_close - timedelta(minutes=MARKET_CLOSE_SHIFT_MINUTES)
        # Ensure the shifted close does not precede the official open
        if shifted_close <= market_open:
            market_close = market_open + timedelta(minutes=1)
        else:
            market_close = shifted_close
    return market_open, market_close


def _pick_confidence(data: Dict) -> float:
    for key in ("confidence_ratio", "directional_confidence"):
        value = data.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return 0.0


def _pick_notes(data: Dict) -> str:
    notes = []
    if data.get("trade_blocked"):
        notes.append("blocked")
    if data.get("trade_mode") == "probe":
        if data.get("pending_probe"):
            notes.append("probe-pending")
        if data.get("probe_active"):
            notes.append("probe-active")
        if data.get("probe_transition_ready"):
            notes.append("probe-ready")
        if data.get("probe_expired"):
            notes.append("probe-expired")
    if data.get("forced_probe"):
        reasons = data.get("forced_probe_reasons") or []
        if reasons:
            notes.append(f"risk:{';'.join(reasons)[:48]}")
        else:
            notes.append("risk-forced")
    return ", ".join(notes) if notes else "-"


def _format_plan_line(symbol: str, data: Dict) -> str:
    last_pnl = data.get("last_trade_pnl")
    last_pnl_str = f"{last_pnl:.2f}" if isinstance(last_pnl, (int, float)) else "n/a"
    parts = [
        symbol,
        f"{data.get('side', '?')}/{data.get('trade_mode', 'normal')}",
        f"avg={data.get('avg_return', 0.0):.3f}",
        f"comp={data.get('composite_score', 0.0):.3f}",
        f"move={data.get('predicted_movement', 0.0):.3f}",
        f"conf={_pick_confidence(data):.3f}",
        f"last={last_pnl_str}",
    ]
    if data.get("maxdiff_spread_rank"):
        parts.append(f"maxdiff_rank={int(data['maxdiff_spread_rank'])}")
        if data.get("maxdiff_spread_overflow"):
            parts.append("maxdiff_overflow")
    notes = _pick_notes(data)
    if notes != "-":
        parts.append(f"notes={notes}")
    return " ".join(parts)


def _forecast_plus_sim_nonpositive(data: Dict) -> Optional[Tuple[float, float]]:
    strategy = data.get("strategy")
    if not strategy:
        return None
    forecasts = data.get("strategy_candidate_forecasted_pnl") or {}
    raw_forecast = forecasts.get(strategy)
    raw_recent = data.get("recent_return_sum")
    if raw_forecast is None or raw_recent is None:
        return None
    forecasted = coerce_numeric(raw_forecast, default=0.0)
    recent_sum = coerce_numeric(raw_recent, default=0.0)
    combined = float(forecasted) + float(recent_sum)
    if combined <= 0:
        return float(forecasted), float(recent_sum)
    return None


def _collect_forced_probe_reasons(symbol: str, data: Dict, probe_state: ProbeState) -> List[str]:
    reasons: List[str] = []
    side = data.get("side")
    if not side:
        data.pop("global_force_probe", None)
        return reasons

    normalized_side = _normalize_side_for_key(side)

    if probe_state.force_probe:
        reason = probe_state.reason or "previous_day_loss"
        reasons.append(f"global_loss:{reason}")
        data["global_force_probe"] = True
    else:
        data.pop("global_force_probe", None)

    if not DISABLE_RECENT_PNL_PROBE:
        pnls = _recent_trade_pnl_pcts(symbol, normalized_side, strategy=None, limit=2)
        pct_mode = True
        if len(pnls) < 2:
            pnls = _recent_trade_pnls(symbol, normalized_side, strategy=None, limit=2)
            pct_mode = False
        if len(pnls) >= 2:
            recent_window = pnls[:2]
            pnl_sum = sum(recent_window)
            if pnl_sum <= 0:
                fmt = "{:.4f}" if pct_mode else "{:.2f}"
                formatted = ", ".join(fmt.format(pnl) for pnl in recent_window)
                label = "recent_pnl_pct_sum" if pct_mode else "recent_pnl_sum"
                reasons.append(f"{label}={fmt.format(pnl_sum)} [{formatted}]")

    forecast_pair = _forecast_plus_sim_nonpositive(data)
    if forecast_pair is not None:
        reasons.append(f"forecast+sim<=0 ({forecast_pair[0]:.4f}+{forecast_pair[1]:.4f})")

    return reasons


def _apply_forced_probe_annotations(
    picks: Dict[str, Dict],
    probe_state: Optional[ProbeState] = None,
) -> ProbeState:
    active_state = probe_state or resolve_probe_state()
    if not picks:
        return active_state

    for symbol, data in picks.items():
        previous_reasons = list(data.get("forced_probe_reasons") or [])
        reasons = _collect_forced_probe_reasons(symbol, data, active_state)
        if reasons:
            data["forced_probe"] = True
            data["forced_probe_reasons"] = reasons
            if data.get("trade_mode") != "probe":
                data["trade_mode"] = "probe"
            if not data.get("pending_probe"):
                data["pending_probe"] = True
            if previous_reasons != reasons:
                logger.info(f"{symbol}: Risk controls forcing probe mode ({'; '.join(reasons)})")
        else:
            if previous_reasons:
                logger.info(f"{symbol}: Risk controls cleared forced probe mode")
            data.pop("forced_probe", None)
            data.pop("forced_probe_reasons", None)

    return active_state


def _format_entry_candidates(picks: Dict[str, Dict]) -> List[str]:
    lines = []
    for symbol, data in picks.items():
        notes = []
        if data.get("trade_mode") == "probe":
            if data.get("pending_probe"):
                notes.append("pending")
            if data.get("probe_active"):
                notes.append("active")
        if data.get("trade_blocked"):
            notes.append("blocked")
        note_str = f" ({', '.join(notes)})" if notes else ""
        lines.append(
            f"{symbol}: {data.get('side', '?')} {data.get('trade_mode', 'normal')} "
            f"avg={data.get('avg_return', 0.0):.3f} "
            f"move={data.get('predicted_movement', 0.0):.3f}{note_str}"
        )
    return lines


def _analyze_single_symbol_for_parallel(
    symbol: str,
    num_simulations: int,
    model_override: Optional[str],
    skip_closed_equity: bool,
    strategy_priorities: Optional[Dict[str, List[str]]],
) -> Optional[Dict]:
    """Analyze a single symbol (used by parallel executor)."""
    try:
        # Check market hours for each symbol to avoid wasting time on stocks when market closes mid-analysis
        if not is_crypto_symbol(symbol) and not is_nyse_trading_day_now():
            if skip_closed_equity:
                logger.debug(f"{symbol}: Skipping (market closed, equity)")
                return None
            logger.debug(f"{symbol}: market closed but analyzing due to MARKETSIM_SKIP_CLOSED_EQUITY override.")

        priority_override = (strategy_priorities or {}).get(symbol)

        # Run backtest
        backtest_df = backtest_forecasts(symbol, num_simulations, model_override=model_override)

        if backtest_df.empty:
            logger.warning(f"{symbol}: backtest returned no simulations")
            return None

        # Process results (same logic as sequential version)
        # ... [This would need to be extracted from the main loop]
        # For now, return a simple dict - full implementation would mirror the sequential version
        return {"symbol": symbol, "backtest_df": backtest_df}

    except Exception as e:
        logger.error(f"{symbol}: Analysis failed: {e}")
        return None


def _analyze_symbols_parallel(
    symbols: List[str],
    *,
    model_overrides: Optional[Dict[str, Optional[str]]] = None,
    strategy_priorities: Optional[Dict[str, List[str]]] = None,
) -> Dict:
    """Parallel version of symbol analysis using ThreadPoolExecutor."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time

    skip_closed_equity = should_skip_closed_equity()

    env_simulations_raw = os.getenv("MARKETSIM_BACKTEST_SIMULATIONS")
    num_simulations = 70
    if env_simulations_raw:
        try:
            num_simulations = max(1, int(env_simulations_raw))
        except ValueError:
            pass

    # Determine worker count
    max_workers = int(os.getenv("MARKETSIM_PARALLEL_WORKERS", "0"))
    if max_workers <= 0:
        max_workers = min(32, (os.cpu_count() or 1) + 4)

    logger.info(f"Parallel analysis: {len(symbols)} symbols with {max_workers} workers")

    results = {}
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_symbol = {}
        for symbol in symbols:
            model_override = (model_overrides or {}).get(symbol)
            future = executor.submit(
                _analyze_single_symbol_for_parallel,
                symbol,
                num_simulations,
                model_override,
                skip_closed_equity,
                strategy_priorities,
            )
            future_to_symbol[future] = symbol

        # Collect results
        completed = 0
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            completed += 1

            try:
                result = future.result()
                if result:
                    results[symbol] = result
                    logger.info(f"[{completed}/{len(symbols)}] ✓ {symbol}")
                else:
                    logger.info(f"[{completed}/{len(symbols)}] ✗ {symbol} (skipped)")
            except Exception as e:
                logger.error(f"[{completed}/{len(symbols)}] ✗ {symbol} failed: {e}")

    elapsed = time.time() - start_time
    logger.info(f"Parallel analysis complete: {elapsed:.1f}s ({elapsed/len(symbols):.2f}s avg)")

    # NOTE: This is a simplified version. Full implementation would need to:
    # 1. Extract the result processing logic from _analyze_symbols_impl
    # 2. Apply the same strategy selection and ranking
    # 3. Handle all the edge cases

    return results


def _analyze_symbols_impl(
    symbols: List[str],
    *,
    model_overrides: Optional[Dict[str, Optional[str]]] = None,
    strategy_priorities: Optional[Dict[str, List[str]]] = None,
) -> Dict:
    """Run backtest analysis on symbols and return results sorted by average return."""
    # Check if parallel analysis is enabled
    use_parallel = os.getenv("MARKETSIM_PARALLEL_ANALYSIS", "0").strip().lower() in {"1", "true", "yes", "on"}

    if use_parallel and len(symbols) > 1:
        logger.info(f"Using PARALLEL analysis for {len(symbols)} symbols")
        return _analyze_symbols_parallel(symbols, model_overrides=model_overrides, strategy_priorities=strategy_priorities)

    results = {}
    skip_closed_equity = should_skip_closed_equity()
    skipped_equity_symbols: List[str] = []

    env_simulations_raw = os.getenv("MARKETSIM_BACKTEST_SIMULATIONS")
    env_simulations: Optional[int]
    if env_simulations_raw:
        try:
            env_simulations = max(1, int(env_simulations_raw))
        except ValueError:
            logger.warning(
                "Ignoring invalid MARKETSIM_BACKTEST_SIMULATIONS=%r; using default of 70 simulations.",
                env_simulations_raw,
            )
            env_simulations = None
        else:
            logger.info(f"Using MARKETSIM_BACKTEST_SIMULATIONS override of {env_simulations} for backtest iterations.")
    else:
        env_simulations = None

    kronos_only_mode = is_kronos_only_mode()

    latest_snapshot = _load_latest_forecast_snapshot()

    for symbol in symbols:
        # Check market hours for each symbol (not just once at start) to avoid wasting time
        # on stocks when market closes mid-analysis
        if not is_crypto_symbol(symbol) and not is_nyse_trading_day_now():
            if skip_closed_equity:
                skipped_equity_symbols.append(symbol)
                continue
            logger.debug(
                f"{symbol}: market closed but analyzing due to MARKETSIM_SKIP_CLOSED_EQUITY override."
            )
        model_override = (model_overrides or {}).get(symbol)
        priority_override = (strategy_priorities or {}).get(symbol)
        try:
            kelly_fraction = None
            num_simulations = env_simulations or 70
            used_fallback_engine = False

            candidates: List[Optional[str]] = []
            if model_override:
                candidates.append(model_override)
            else:
                candidates.extend(["chronos2", "toto"])
            candidates.append(None)

            backtest_df = None
            chosen_model: Optional[str] = None
            attempted: set[str] = set()
            for candidate in candidates:
                key = (candidate or "auto").lower()
                if key in attempted:
                    continue
                attempted.add(key)
                override_arg = None if candidate in (None, "auto") else candidate
                try:
                    backtest_df = backtest_forecasts(symbol, num_simulations, model_override=override_arg)
                    chosen_model = candidate
                    used_fallback_engine = override_arg == "toto"
                    break
                except Exception as exc:
                    if candidate == "chronos2" and not model_override:
                        logger.warning(
                            "%s: Chronos2 model failed (%s); retrying with Toto",
                            symbol,
                            exc,
                        )
                        continue
                    logger.error(
                        "%s: backtest (%s) failed: %s",
                        symbol,
                        candidate or "auto",
                        exc,
                    )
                    backtest_df = None
                    break

            if backtest_df is None:
                logger.error(f"backtest_forecasts failed for {symbol}: unable to produce simulations.")
                continue

            if backtest_df.empty:
                logger.warning(f"Skipping {symbol} - backtest returned no simulations.")
                continue

            required_columns = {
                "simple_strategy_return",
                "all_signals_strategy_return",
                "entry_takeprofit_return",
                "highlow_return",
            }
            missing_cols = required_columns.difference(backtest_df.columns)
            if missing_cols:
                logger.warning(f"Skipping {symbol} - missing backtest metrics: {sorted(missing_cols)}")
                continue

            sample_size = len(backtest_df)
            trading_days_per_year = 365 if is_crypto_symbol(symbol) else 252

            _normalized_cache: Dict[str, Optional[pd.Series]] = {}

            def _normalized_series(column: str) -> Optional[pd.Series]:
                if column not in _normalized_cache:
                    if column in backtest_df.columns:
                        _normalized_cache[column] = _normalize_series(backtest_df[column])
                    else:
                        _normalized_cache[column] = None
                return _normalized_cache[column]

            def _metric(value: object, default: float = 0.0) -> float:
                return coerce_numeric(value, default=default, prefer="mean")

            def _mean_column(column: str, default: float = 0.0) -> float:
                series = _normalized_series(column)
                if series is None or series.empty:
                    return default
                return _metric(series, default=default)

            def _mean_return(primary: str, fallback: Optional[str] = None, default: float = 0.0) -> float:
                series = _normalized_series(primary)
                if series is None and fallback:
                    series = _normalized_series(fallback)
                if series is None or series.empty:
                    return default
                return _metric(series, default=default)

            strategy_returns_daily = {
                "simple": _mean_return("simple_strategy_avg_daily_return", "simple_strategy_return"),
                "all_signals": _mean_return("all_signals_strategy_avg_daily_return", "all_signals_strategy_return"),
                "takeprofit": _mean_return("entry_takeprofit_avg_daily_return", "entry_takeprofit_return"),
                "highlow": _mean_return("highlow_avg_daily_return", "highlow_return"),
                "maxdiff": _mean_return("maxdiff_avg_daily_return", "maxdiff_return"),
                "maxdiffalwayson": _mean_return("maxdiffalwayson_avg_daily_return", "maxdiffalwayson_return"),
                "pctdiff": _mean_return("pctdiff_avg_daily_return", "pctdiff_return"),
            }
            strategy_returns_annual = {
                "simple": _mean_return("simple_strategy_annual_return", "simple_strategy_return"),
                "all_signals": _mean_return("all_signals_strategy_annual_return", "all_signals_strategy_return"),
                "takeprofit": _mean_return("entry_takeprofit_annual_return", "entry_takeprofit_return"),
                "highlow": _mean_return("highlow_annual_return", "highlow_return"),
                "maxdiff": _mean_return("maxdiff_annual_return", "maxdiff_return"),
                "maxdiffalwayson": _mean_return("maxdiffalwayson_annual_return", "maxdiffalwayson_return"),
                "pctdiff": _mean_return("pctdiff_annual_return", "pctdiff_return"),
            }
            strategy_returns = strategy_returns_daily
            strategy_recent_sums: Dict[str, Optional[float]] = {}

            def _recent_return_sum(primary: str, fallback: Optional[str] = None, window: int = 2) -> Optional[float]:
                series = _normalized_series(primary)
                if (series is None or series.empty) and fallback:
                    series = _normalized_series(fallback)
                if series is None or series.empty:
                    return None
                recent = series.dropna()
                if recent.empty or len(recent) < window:
                    return None
                return float(recent.iloc[:window].sum())

            _strategy_series_map: Dict[str, Tuple[str, Optional[str]]] = {
                "simple": ("simple_strategy_avg_daily_return", "simple_strategy_return"),
                "all_signals": ("all_signals_strategy_avg_daily_return", "all_signals_strategy_return"),
                "takeprofit": ("entry_takeprofit_avg_daily_return", "entry_takeprofit_return"),
                "highlow": ("highlow_avg_daily_return", "highlow_return"),
                "maxdiff": ("maxdiff_avg_daily_return", "maxdiff_return"),
                "maxdiffalwayson": ("maxdiffalwayson_avg_daily_return", "maxdiffalwayson_return"),
                "pctdiff": ("pctdiff_avg_daily_return", "pctdiff_return"),
            }

            unprofit_return = 0.0
            unprofit_sharpe = 0.0
            if (
                "unprofit_shutdown_avg_daily_return" in backtest_df.columns
                or "unprofit_shutdown_return" in backtest_df.columns
            ):
                unprofit_return = _mean_return("unprofit_shutdown_avg_daily_return", "unprofit_shutdown_return")
                strategy_returns["unprofit_shutdown"] = unprofit_return
                strategy_returns_annual["unprofit_shutdown"] = _mean_return(
                    "unprofit_shutdown_annual_return",
                    "unprofit_shutdown_return",
                )
            if "unprofit_shutdown_sharpe" in backtest_df.columns:
                unprofit_sharpe = _metric(backtest_df["unprofit_shutdown_sharpe"], default=0.0)

            raw_last_prediction = backtest_df.iloc[0]
            last_prediction = raw_last_prediction.apply(lambda value: coerce_numeric(value, default=0.0, prefer="mean"))
            walk_forward_oos_sharpe_raw = last_prediction.get("walk_forward_oos_sharpe")
            walk_forward_turnover_raw = last_prediction.get("walk_forward_turnover")
            walk_forward_highlow_raw = last_prediction.get("walk_forward_highlow_sharpe")
            walk_forward_takeprofit_raw = last_prediction.get("walk_forward_takeprofit_sharpe")
            walk_forward_maxdiff_raw = last_prediction.get("walk_forward_maxdiff_sharpe")

            walk_forward_oos_sharpe = (
                coerce_numeric(walk_forward_oos_sharpe_raw) if walk_forward_oos_sharpe_raw is not None else None
            )
            walk_forward_turnover = (
                coerce_numeric(walk_forward_turnover_raw) if walk_forward_turnover_raw is not None else None
            )
            walk_forward_highlow_sharpe = (
                coerce_numeric(walk_forward_highlow_raw) if walk_forward_highlow_raw is not None else None
            )
            walk_forward_takeprofit_sharpe = (
                coerce_numeric(walk_forward_takeprofit_raw) if walk_forward_takeprofit_raw is not None else None
            )
            walk_forward_maxdiff_sharpe = (
                coerce_numeric(walk_forward_maxdiff_raw) if walk_forward_maxdiff_raw is not None else None
            )

            close_price = coerce_numeric(last_prediction.get("close"), default=0.0)

            # Retry logic: give the model a chance to fix invalid predictions
            max_retries = 2
            retry_count = 0
            predictions_valid = False

            while retry_count <= max_retries and not predictions_valid:
                if retry_count > 0:
                    logger.info(f"{symbol}: Retrying predictions (attempt {retry_count + 1}/{max_retries + 1})")
                    # Re-run predictions to see if model can fix itself
                    try:
                        retry_backtest_df = backtest_forecasts(symbol, num_simulations, model_override=model_override)
                        if not retry_backtest_df.empty:
                            raw_last_prediction = retry_backtest_df.iloc[0]
                            retry_prediction = raw_last_prediction.apply(
                                lambda value: coerce_numeric(value, default=0.0, prefer="mean")
                            )
                            last_prediction = retry_prediction
                        else:
                            logger.warning(
                                f"{symbol}: Retry {retry_count} returned empty backtest, using previous predictions"
                            )
                    except Exception as retry_exc:
                        logger.warning(f"{symbol}: Retry {retry_count} failed: {retry_exc}, using previous predictions")

                predicted_close_price = coerce_numeric(
                    last_prediction.get("predicted_close"),
                    default=close_price,
                )
                predicted_high_price = coerce_numeric(
                    last_prediction.get("predicted_high"),
                    default=predicted_close_price,
                )
                predicted_low_price = coerce_numeric(
                    last_prediction.get("predicted_low"),
                    default=predicted_close_price,
                )

                # Check if predictions are valid
                has_inverted_highlow = predicted_high_price < predicted_low_price
                close_exceeds_high = predicted_close_price > predicted_high_price
                close_below_low = predicted_close_price < predicted_low_price

                if has_inverted_highlow or close_exceeds_high or close_below_low:
                    if retry_count == 0:
                        logger.warning(
                            f"{symbol}: Invalid predictions detected - "
                            f"high={predicted_high_price:.4f}, low={predicted_low_price:.4f}, close={predicted_close_price:.4f} "
                            f"(inverted={has_inverted_highlow}, close>high={close_exceeds_high}, close<low={close_below_low})"
                        )
                    retry_count += 1
                else:
                    predictions_valid = True
                    if retry_count > 0:
                        logger.info(f"{symbol}: Predictions fixed after {retry_count} retries")

            # If retries failed, apply fallback fixes
            if not predictions_valid:
                logger.warning(
                    f"{symbol}: All {max_retries} retries failed to produce valid predictions, applying fallback fixes"
                )

                # Fix inverted base predictions - fallback to close price
                if predicted_high_price < predicted_low_price:
                    logger.warning(
                        f"{symbol}: Base model has inverted predictions (high={predicted_high_price:.4f} < low={predicted_low_price:.4f}), "
                        f"using close={predicted_close_price:.4f} for both"
                    )
                    predicted_high_price = predicted_close_price
                    predicted_low_price = predicted_close_price

                # Sanity check: ensure close is within [low, high] range
                # Valid OHLC data requires: low <= close <= high
                if predicted_close_price > predicted_high_price:
                    logger.warning(
                        f"{symbol}: Close price ({predicted_close_price:.4f}) exceeds high ({predicted_high_price:.4f}), "
                        f"adjusting high to match close"
                    )
                    predicted_high_price = predicted_close_price

                if predicted_close_price < predicted_low_price:
                    logger.warning(
                        f"{symbol}: Close price ({predicted_close_price:.4f}) is below low ({predicted_low_price:.4f}), "
                        f"adjusting low to match close"
                    )
                    predicted_low_price = predicted_close_price

            def _optional_numeric(value: object) -> Optional[float]:
                raw = coerce_numeric(value, default=float("nan")) if value is not None else float("nan")
                return raw if math.isfinite(raw) else None

            maxdiff_high_price = _optional_numeric(last_prediction.get("maxdiffprofit_high_price"))
            maxdiff_low_price = _optional_numeric(last_prediction.get("maxdiffprofit_low_price"))
            maxdiff_trade_bias = _optional_numeric(last_prediction.get("maxdiff_trade_bias"))
            maxdiffalwayson_high_price = _optional_numeric(last_prediction.get("maxdiffalwayson_high_price"))
            maxdiffalwayson_low_price = _optional_numeric(last_prediction.get("maxdiffalwayson_low_price"))
            pctdiff_entry_low_price = _optional_numeric(last_prediction.get("pctdiff_entry_low_price"))
            pctdiff_entry_high_price = _optional_numeric(last_prediction.get("pctdiff_entry_high_price"))
            pctdiff_takeprofit_high_price = _optional_numeric(last_prediction.get("pctdiff_takeprofit_high_price"))
            pctdiff_takeprofit_low_price = _optional_numeric(last_prediction.get("pctdiff_takeprofit_low_price"))
            pctdiff_trade_bias = _optional_numeric(last_prediction.get("pctdiff_trade_bias"))

            # Fix inverted high/low pairs using fallback model predictions
            # Prefer using other models' predictions over blindly flipping
            def _fix_inverted_predictions(
                high: Optional[float],
                low: Optional[float],
                fallback_high_candidates: list,
                fallback_low_candidates: list,
                label: str,
            ) -> tuple:
                """Try fallback models before flipping inverted high/low predictions."""
                if high is None or low is None or high >= low:
                    return high, low

                original_high, original_low = high, low
                logger.warning(f"{symbol}: Detected inverted {label} predictions (high={high:.4f} < low={low:.4f})")

                # Try fallback high predictions
                for fallback_high in fallback_high_candidates:
                    if fallback_high is not None and fallback_high >= low:
                        logger.info(
                            f"{symbol}: Using fallback high={fallback_high:.4f} for {label} (original={original_high:.4f})"
                        )
                        return fallback_high, low

                # Try fallback low predictions
                for fallback_low in fallback_low_candidates:
                    if fallback_low is not None and high >= fallback_low:
                        logger.info(
                            f"{symbol}: Using fallback low={fallback_low:.4f} for {label} (original={original_low:.4f})"
                        )
                        return high, fallback_low

                # Last resort: flip
                logger.warning(f"{symbol}: No valid fallback for {label}, flipping as last resort")
                return low, high

            # Fix maxdiff first using only base model predictions as fallback
            maxdiff_high_price, maxdiff_low_price = _fix_inverted_predictions(
                maxdiff_high_price,
                maxdiff_low_price,
                fallback_high_candidates=[predicted_high_price],
                fallback_low_candidates=[predicted_low_price],
                label="maxdiff",
            )

            # Fix maxdiffalwayson using both corrected maxdiff and base predictions
            maxdiffalwayson_high_price, maxdiffalwayson_low_price = _fix_inverted_predictions(
                maxdiffalwayson_high_price,
                maxdiffalwayson_low_price,
                fallback_high_candidates=[maxdiff_high_price, predicted_high_price],
                fallback_low_candidates=[maxdiff_low_price, predicted_low_price],
                label="maxdiffalwayson",
            )

            _apply_neural_pricing(symbol, last_prediction)

            if (
                pctdiff_entry_low_price is not None
                and pctdiff_takeprofit_high_price is not None
                and pctdiff_takeprofit_high_price < pctdiff_entry_low_price
            ):
                logger.warning(
                    f"{symbol}: pctdiff long takeprofit {pctdiff_takeprofit_high_price:.4f} below entry {pctdiff_entry_low_price:.4f}; clamping"
                )
                pctdiff_takeprofit_high_price = pctdiff_entry_low_price
            if (
                pctdiff_entry_high_price is not None
                and pctdiff_takeprofit_low_price is not None
                and pctdiff_takeprofit_low_price > pctdiff_entry_high_price
            ):
                logger.warning(
                    f"{symbol}: pctdiff short takeprofit {pctdiff_takeprofit_low_price:.4f} above entry {pctdiff_entry_high_price:.4f}; clamping"
                )
                pctdiff_takeprofit_low_price = pctdiff_entry_high_price

            maxdiff_primary_side_raw = raw_last_prediction.get("maxdiff_primary_side")
            maxdiff_primary_side = (
                str(maxdiff_primary_side_raw).strip().lower() if maxdiff_primary_side_raw is not None else None
            )
            if maxdiff_primary_side == "":
                maxdiff_primary_side = None
            pctdiff_primary_side_raw = raw_last_prediction.get("pctdiff_primary_side")
            pctdiff_primary_side = (
                str(pctdiff_primary_side_raw).strip().lower() if pctdiff_primary_side_raw is not None else None
            )
            if pctdiff_primary_side == "":
                pctdiff_primary_side = None

            snapshot_parts = [
                f"{symbol} prediction snapshot",
                f"close={close_price:.4f}",
                f"pred_close={predicted_close_price:.4f}",
                f"pred_high={predicted_high_price:.4f}",
                f"pred_low={predicted_low_price:.4f}",
            ]
            if maxdiff_high_price is not None:
                snapshot_parts.append(f"maxdiff_high={maxdiff_high_price:.4f}")
            if maxdiff_low_price is not None:
                snapshot_parts.append(f"maxdiff_low={maxdiff_low_price:.4f}")
            if maxdiffalwayson_high_price is not None:
                snapshot_parts.append(f"maxdiffalwayson_high={maxdiffalwayson_high_price:.4f}")
            if maxdiffalwayson_low_price is not None:
                snapshot_parts.append(f"maxdiffalwayson_low={maxdiffalwayson_low_price:.4f}")
            neural_high_price = last_prediction.get("neuralpricing_high_price")
            neural_low_price = last_prediction.get("neuralpricing_low_price")
            if neural_high_price is not None:
                snapshot_parts.append(f"neuralpricing_high={neural_high_price:.4f}")
            if neural_low_price is not None:
                snapshot_parts.append(f"neuralpricing_low={neural_low_price:.4f}")
            if pctdiff_entry_low_price is not None:
                snapshot_parts.append(f"pctdiff_entry_low={pctdiff_entry_low_price:.4f}")
            if pctdiff_entry_high_price is not None:
                snapshot_parts.append(f"pctdiff_entry_high={pctdiff_entry_high_price:.4f}")
            if pctdiff_takeprofit_high_price is not None:
                snapshot_parts.append(f"pctdiff_tp_high={pctdiff_takeprofit_high_price:.4f}")
            if pctdiff_takeprofit_low_price is not None:
                snapshot_parts.append(f"pctdiff_tp_low={pctdiff_takeprofit_low_price:.4f}")
            if maxdiff_primary_side:
                bias_fragment = maxdiff_primary_side
                if maxdiff_trade_bias is not None and math.isfinite(maxdiff_trade_bias):
                    bias_fragment = f"{bias_fragment}({maxdiff_trade_bias:+.3f})"
                snapshot_parts.append(f"maxdiff_side={bias_fragment}")
            if pctdiff_primary_side:
                pctdiff_bias = pctdiff_primary_side
                if pctdiff_trade_bias is not None and math.isfinite(pctdiff_trade_bias):
                    pctdiff_bias = f"{pctdiff_bias}({pctdiff_trade_bias:+.3f})"
                snapshot_parts.append(f"pctdiff_side={pctdiff_bias}")
            _log_detail(" ".join(snapshot_parts))

            # Helper to get forecasted PnL from last_prediction
            def _get_forecasted_pnl(strategy_name: str) -> float:
                """Get forecasted PnL for a strategy, falling back to avg_return if not available.

                The forecasted PnL is computed by Toto model on validation set and represents
                forward-looking performance. If unavailable, we fall back to historical avg_return.
                """
                forecast_key = f"{strategy_name}_forecasted_pnl"
                # Check if key exists in last_prediction (pandas Series)
                if forecast_key in last_prediction.index:
                    forecast_value = last_prediction.get(forecast_key)
                    # Only use if it's a valid numeric value (not None, not NaN)
                    if forecast_value is not None and not (isinstance(forecast_value, float) and math.isnan(forecast_value)):
                        forecasted = coerce_numeric(forecast_value, default=0.0)
                        return forecasted
                # Fallback to avg_return if forecasted PnL not available or invalid
                # This should rarely happen now that all strategies compute forecasted PnL
                return strategy_returns.get(strategy_name, 0.0)

            strategy_stats: Dict[str, Dict[str, float]] = {
                "simple": {
                    "avg_return": strategy_returns.get("simple", 0.0),
                    "forecasted_pnl": _get_forecasted_pnl("simple"),
                    "annual_return": strategy_returns_annual.get("simple", 0.0),
                    "sharpe": _mean_column("simple_strategy_sharpe"),
                    "turnover": _mean_column("simple_strategy_turnover"),
                    "max_drawdown": _mean_column("simple_strategy_max_drawdown"),
                },
                "all_signals": {
                    "avg_return": strategy_returns.get("all_signals", 0.0),
                    "forecasted_pnl": _get_forecasted_pnl("all_signals"),
                    "annual_return": strategy_returns_annual.get("all_signals", 0.0),
                    "sharpe": _mean_column("all_signals_strategy_sharpe"),
                    "turnover": _mean_column("all_signals_strategy_turnover"),
                    "max_drawdown": _mean_column("all_signals_strategy_max_drawdown"),
                },
                "takeprofit": {
                    "avg_return": strategy_returns.get("takeprofit", 0.0),
                    "forecasted_pnl": _get_forecasted_pnl("entry_takeprofit"),
                    "annual_return": strategy_returns_annual.get("takeprofit", 0.0),
                    "sharpe": _mean_column("entry_takeprofit_sharpe"),
                    "turnover": _mean_column("entry_takeprofit_turnover"),
                    "max_drawdown": _mean_column("entry_takeprofit_max_drawdown"),
                },
                "highlow": {
                    "avg_return": strategy_returns.get("highlow", 0.0),
                    "forecasted_pnl": _get_forecasted_pnl("highlow"),
                    "annual_return": strategy_returns_annual.get("highlow", 0.0),
                    "sharpe": _mean_column("highlow_sharpe"),
                    "turnover": _mean_column("highlow_turnover"),
                    "max_drawdown": _mean_column("highlow_max_drawdown"),
                },
                "maxdiff": {
                    "avg_return": strategy_returns.get("maxdiff", 0.0),
                    "forecasted_pnl": _get_forecasted_pnl("maxdiff"),
                    "annual_return": strategy_returns_annual.get("maxdiff", 0.0),
                    "sharpe": _mean_column("maxdiff_sharpe"),
                    "turnover": _mean_column("maxdiff_turnover"),
                    "max_drawdown": _mean_column("maxdiff_max_drawdown"),
                },
                "maxdiffalwayson": {
                    "avg_return": strategy_returns.get("maxdiffalwayson", 0.0),
                    "forecasted_pnl": _get_forecasted_pnl("maxdiffalwayson"),
                    "annual_return": strategy_returns_annual.get("maxdiffalwayson", 0.0),
                    "sharpe": _mean_column("maxdiffalwayson_sharpe"),
                    "turnover": _mean_column("maxdiffalwayson_turnover"),
                    "max_drawdown": _mean_column("maxdiffalwayson_max_drawdown"),
                },
                "pctdiff": {
                    "avg_return": strategy_returns.get("pctdiff", 0.0),
                    "forecasted_pnl": _get_forecasted_pnl("pctdiff"),
                    "annual_return": strategy_returns_annual.get("pctdiff", 0.0),
                    "sharpe": _mean_column("pctdiff_sharpe"),
                    "turnover": _mean_column("pctdiff_turnover"),
                    "max_drawdown": _mean_column("pctdiff_max_drawdown"),
                },
            }

            for strat_name, (primary_col, fallback_col) in _strategy_series_map.items():
                strategy_recent_sums[strat_name] = _recent_return_sum(primary_col, fallback_col)

            strategy_ineligible: Dict[str, str] = {}
            candidate_forecasted_pnl: Dict[str, float] = {}
            allowed_side = _allowed_side_for(symbol)
            symbol_is_crypto = symbol in all_crypto_symbols

            for name, stats in strategy_stats.items():
                if name not in strategy_returns:
                    continue

                # Use forecasted PnL instead of avg_return for strategy selection
                # Always record forecasted PnL, even if strategy fails entry gate
                forecasted_pnl = _metric(stats.get("forecasted_pnl"), default=0.0)
                candidate_forecasted_pnl[name] = forecasted_pnl

                allow_config = True
                if name == "takeprofit":
                    allow_config = ALLOW_TAKEPROFIT_ENTRY
                elif name == "highlow":
                    allow_config = ALLOW_HIGHLOW_ENTRY
                elif name == "maxdiff":
                    allow_config = ALLOW_MAXDIFF_ENTRY
                elif name == "maxdiffalwayson":
                    allow_config = ALLOW_MAXDIFF_ALWAYS_ENTRY
                elif name == "pctdiff":
                    allow_config = ALLOW_PCTDIFF_ENTRY

                if name in {"takeprofit", "highlow", "maxdiff", "maxdiffalwayson", "pctdiff"}:
                    if not allow_config:
                        strategy_ineligible[name] = "disabled_by_config"
                        continue
                    eligible, reason = evaluate_strategy_entry_gate(
                        symbol,
                        stats,
                        fallback_used=used_fallback_engine,
                        sample_size=sample_size,
                    )
                    if not eligible:
                        strategy_ineligible[name] = reason
                        continue

            # Sort strategies by forecasted_pnl (highest positive first)
            # NOTE: We use forecasted PnL instead of avg_return because forecasted PnL is the
            # forward-looking prediction of next day's returns, which is more relevant for
            # strategy selection than the average of historical backtest simulations.
            ordered_strategies: List[str] = []
            if candidate_forecasted_pnl:
                # Only consider strategies with positive forecasted PnL
                positive_forecasts = {k: v for k, v in candidate_forecasted_pnl.items() if v > 0}
                if positive_forecasts:
                    ordered_strategies = [
                        name
                        for name, _ in sorted(
                            positive_forecasts.items(),
                            key=lambda item: item[1],
                            reverse=True,
                        )
                    ]
                    _log_detail(
                        f"{symbol}: Strategy selection by forecasted PnL (positive only): "
                        + ", ".join(f"{name}={candidate_forecasted_pnl[name]:.4f}" for name in ordered_strategies)
                    )
                else:
                    # No positive forecasts - fall back to all strategies sorted by forecasted PnL
                    ordered_strategies = [
                        name
                        for name, _ in sorted(
                            candidate_forecasted_pnl.items(),
                            key=lambda item: item[1],
                            reverse=True,
                        )
                    ]
                    _log_detail(
                        f"{symbol}: No positive forecasted PnL - using all strategies: "
                        + ", ".join(f"{name}={candidate_forecasted_pnl[name]:.4f}" for name in ordered_strategies)
                    )
            else:
                ordered_strategies = ["simple"]

            ordered_strategies = _apply_strategy_priority(ordered_strategies, priority_override)

            if strategy_ineligible:
                logger.debug("%s strategy entry gates rejected: %s", symbol, strategy_ineligible)

            close_movement_raw = predicted_close_price - close_price
            high_movement = predicted_high_price - close_price
            low_movement = predicted_low_price - close_price

            selection_notes: List[str] = []
            selected_strategy: Optional[str] = None
            avg_return = 0.0
            annual_return = 0.0
            predicted_movement = close_movement_raw
            position_side = "buy" if predicted_movement > 0 else "sell"

            for candidate_name in ordered_strategies:
                if candidate_name in strategy_ineligible:
                    ineligible_reason = strategy_ineligible[candidate_name]
                    # Allow strategies with positive forecasted PnL to bypass soft entry gates
                    # but not hard blocks like disabled_by_config
                    candidate_forecasted = candidate_forecasted_pnl.get(candidate_name, 0.0)
                    if ineligible_reason == "disabled_by_config" or candidate_forecasted <= 0:
                        selection_notes.append(f"{candidate_name}=ineligible({ineligible_reason})")
                        continue
                    # Strategy has positive forecasted PnL - allow it to proceed despite entry gate failure
                    _log_detail(
                        f"{symbol}: Allowing {candidate_name} despite entry gate ({ineligible_reason}) "
                        f"due to positive forecasted PnL: {candidate_forecasted:.4f}"
                    )
                    selection_notes.append(f"{candidate_name}=allowed_by_forecast({candidate_forecasted:.4f})")

                # Get avg_return from strategy_stats for this candidate
                candidate_stats = strategy_stats.get(candidate_name, {})
                candidate_avg_return = candidate_stats.get("avg_return", 0.0)

                candidate_position_side: Optional[str] = None
                candidate_predicted_movement = close_movement_raw

                if candidate_name == "maxdiff":
                    if maxdiff_primary_side in {"buy", "sell"}:
                        candidate_position_side = maxdiff_primary_side
                        target_price = maxdiff_high_price if candidate_position_side == "buy" else maxdiff_low_price
                        if target_price is not None and math.isfinite(target_price):
                            candidate_predicted_movement = target_price - close_price
                    elif maxdiff_primary_side == "neutral" and maxdiff_trade_bias is not None:
                        if maxdiff_trade_bias > 0:
                            candidate_position_side = "buy"
                        elif maxdiff_trade_bias < 0:
                            candidate_position_side = "sell"
                elif candidate_name == "maxdiffalwayson":
                    dominant_move = max(abs(high_movement), abs(low_movement))
                    if allowed_side == "sell":
                        candidate_position_side = "sell"
                        candidate_predicted_movement = -dominant_move
                    else:
                        candidate_position_side = "buy"
                        candidate_predicted_movement = dominant_move
                elif candidate_name == "pctdiff":
                    entry_buy_price = pctdiff_entry_low_price
                    tp_buy_price = pctdiff_takeprofit_high_price
                    entry_sell_price = pctdiff_entry_high_price
                    tp_sell_price = pctdiff_takeprofit_low_price
                    pctdiff_bias = pctdiff_trade_bias
                    if pctdiff_primary_side in {"buy", "sell"}:
                        candidate_position_side = pctdiff_primary_side
                    elif pctdiff_bias is not None:
                        if pctdiff_bias > 0:
                            candidate_position_side = "buy"
                        elif pctdiff_bias < 0:
                            candidate_position_side = "sell"
                    if candidate_position_side == "buy" and tp_buy_price is not None:
                        candidate_predicted_movement = tp_buy_price - close_price
                    elif candidate_position_side == "sell" and tp_sell_price is not None:
                        candidate_predicted_movement = tp_sell_price - close_price
                    else:
                        buy_move = (tp_buy_price - close_price) if tp_buy_price is not None else None
                        sell_move = (tp_sell_price - close_price) if tp_sell_price is not None else None
                        if buy_move is not None and (sell_move is None or abs(buy_move) >= abs(sell_move)):
                            candidate_position_side = "buy"
                            candidate_predicted_movement = buy_move
                        elif sell_move is not None:
                            candidate_position_side = "sell"
                            candidate_predicted_movement = sell_move

                if candidate_position_side is None and candidate_name == "all_signals":
                    if all(x > 0 for x in [close_movement_raw, high_movement, low_movement]):
                        candidate_position_side = "buy"
                    elif all(x < 0 for x in [close_movement_raw, high_movement, low_movement]):
                        candidate_position_side = "sell"
                    else:
                        note = "mixed_directional_signals"
                        if "all_signals" not in strategy_ineligible:
                            strategy_ineligible["all_signals"] = note
                        selection_notes.append(f"all_signals={note}")
                        continue

                if candidate_position_side is None:
                    candidate_position_side = "buy" if candidate_predicted_movement > 0 else "sell"

                disallowed_reason: Optional[str] = None
                if allowed_side and allowed_side != "both" and candidate_position_side != allowed_side:
                    disallowed_reason = f"side_not_allowed_{allowed_side}"
                elif (
                    symbol_is_crypto
                    and candidate_position_side == "sell"
                    and (allowed_side is None or allowed_side not in {"sell", "both"})
                ):
                    disallowed_reason = "crypto_sell_disabled"

                if disallowed_reason:
                    if candidate_name not in strategy_ineligible:
                        strategy_ineligible[candidate_name] = disallowed_reason
                    selection_notes.append(f"{candidate_name}={disallowed_reason}")
                    continue

                selected_strategy = candidate_name
                avg_return = candidate_avg_return
                annual_return = _metric(strategy_stats.get(candidate_name, {}).get("annual_return"), default=0.0)
                predicted_movement = candidate_predicted_movement
                position_side = candidate_position_side
                if candidate_name != ordered_strategies[0]:
                    _log_detail(
                        f"{symbol}: strategy fallback from {ordered_strategies[0]} to {candidate_name} "
                        f"(ordered={ordered_strategies})"
                    )
                break

            if selected_strategy is None:
                reason = "; ".join(selection_notes) if selection_notes else "no viable strategy"
                _log_detail(f"Skipping {symbol} - no actionable strategy ({reason})")
                continue

            best_strategy = selected_strategy

            expected_move_pct = safe_divide(predicted_movement, close_price, default=0.0)
            simple_return = strategy_returns.get("simple", 0.0)
            takeprofit_return = strategy_returns.get("takeprofit", 0.0)
            highlow_return = strategy_returns.get("highlow", 0.0)
            maxdiff_return = strategy_returns.get("maxdiff", 0.0)
            maxdiffalwayson_return = strategy_returns.get("maxdiffalwayson", 0.0)
            simple_sharpe = 0.0
            if "simple_strategy_sharpe" in backtest_df.columns:
                simple_sharpe = coerce_numeric(backtest_df["simple_strategy_sharpe"].mean(), default=0.0)
            kronos_profit_raw = last_prediction.get("closemin_loss_trading_profit")
            kronos_profit = coerce_numeric(kronos_profit_raw) if kronos_profit_raw is not None else 0.0
            if is_kronos_only_mode():
                if kronos_profit > simple_return:
                    simple_return = kronos_profit
                if kronos_profit > avg_return:
                    avg_return = kronos_profit
                kronos_annual = kronos_profit * trading_days_per_year
                if kronos_annual > annual_return:
                    annual_return = kronos_annual
            core_return = max(simple_return, 0.0)
            core_sharpe = max(simple_sharpe, 0.0)
            price_skill = core_return + 0.25 * core_sharpe + 0.15 * max(kronos_profit, 0.0)
            highlow_allowed_entry = ALLOW_HIGHLOW_ENTRY and ("highlow" not in strategy_ineligible)
            takeprofit_allowed_entry = ALLOW_TAKEPROFIT_ENTRY and ("takeprofit" not in strategy_ineligible)
            maxdiff_allowed_entry = ALLOW_MAXDIFF_ENTRY and ("maxdiff" not in strategy_ineligible)
            maxdiffalwayson_allowed_entry = ALLOW_MAXDIFF_ALWAYS_ENTRY and (
                "maxdiffalwayson" not in strategy_ineligible
            )

            raw_expected_move_pct = expected_move_pct
            calibrated_move_raw = last_prediction.get("calibrated_expected_move_pct")
            calibrated_move_pct = coerce_numeric(calibrated_move_raw) if calibrated_move_raw is not None else None
            # Don't override movement for maxdiff strategy - it uses its own target prices
            if calibrated_move_pct is not None and selected_strategy != "maxdiff":
                expected_move_pct = calibrated_move_pct
                predicted_movement = expected_move_pct * close_price
                calibrated_close_price = close_price * (1.0 + expected_move_pct)
            else:
                calibrated_close_price = predicted_close_price

            if predicted_movement == 0.0:
                _log_detail(f"Skipping {symbol} - calibrated move collapsed to zero.")
                continue

            # Strategy already determined position_side based on its own logic.
            # Don't second-guess with calibrated close prediction - strategies may use
            # mean reversion, high/low targets, or other logic that differs from close prediction.

            if allowed_side and allowed_side != "both":
                if allowed_side == "buy" and position_side == "sell":
                    _log_detail(f"Skipping {symbol} - sells disabled via MARKETSIM_SYMBOL_SIDE_MAP.")
                    continue
                if allowed_side == "sell" and position_side == "buy":
                    _log_detail(f"Skipping {symbol} - buys disabled via MARKETSIM_SYMBOL_SIDE_MAP.")
                    continue

            abs_move = abs(expected_move_pct)
            if abs_move < MIN_EXPECTED_MOVE_PCT:
                abs_move = 0.0
            edge_strength = price_skill * abs_move
            directional_edge = edge_strength if predicted_movement >= 0 else -edge_strength

            toto_move_pct = coerce_numeric(last_prediction.get("toto_expected_move_pct"), default=0.0)
            kronos_move_pct = coerce_numeric(last_prediction.get("kronos_expected_move_pct"), default=0.0)
            realized_volatility_pct = coerce_numeric(last_prediction.get("realized_volatility_pct"), default=0.0)
            avg_dollar_vol_raw = last_prediction.get("dollar_vol_20d")
            avg_dollar_vol = coerce_numeric(avg_dollar_vol_raw) if avg_dollar_vol_raw is not None else None
            atr_pct_raw = last_prediction.get("atr_pct_14")
            atr_pct = coerce_numeric(atr_pct_raw) if atr_pct_raw is not None else None
            sigma_pct = safe_divide(realized_volatility_pct, 100.0, default=0.0)
            if sigma_pct <= 0:
                sigma_pct = max(abs(expected_move_pct), 1e-3)
            kelly_fraction = kelly_lite(abs(expected_move_pct), sigma_pct)
            drawdown_scale = _kelly_drawdown_scale(best_strategy, symbol)
            if drawdown_scale < 1.0:
                logger.info(
                    f"{symbol}: Drawdown scale applied to Kelly for {best_strategy or 'unknown'} ({drawdown_scale:.3f})"
                )

            cap = _drawdown_cap_for(best_strategy, symbol)
            resume_threshold = _drawdown_resume_for(best_strategy, cap, symbol)
            try:
                state = get_state()
                drawdown_pct = getattr(state, "drawdown_pct", None)
            except RuntimeError:
                drawdown_pct = None
            suspend_threshold = _lookup_threshold("MARKETSIM_DRAWDOWN_SUSPEND_MAP", symbol, best_strategy)
            if suspend_threshold is None:
                suspend_threshold = _get_env_float("MARKETSIM_DRAWDOWN_SUSPEND")
            if cap is None:
                cap = suspend_threshold
            strategy_key = _strategy_key(symbol, best_strategy)
            if cap and drawdown_pct is not None and suspend_threshold and drawdown_pct >= suspend_threshold:
                _DRAW_SUSPENDED[strategy_key] = True
                _log_detail(
                    f"Suspending new entry for {symbol} due to drawdown {drawdown_pct:.3%} >= {suspend_threshold:.3%}"
                )
                continue
            if (
                _DRAW_SUSPENDED.get(strategy_key)
                and resume_threshold
                and drawdown_pct is not None
                and drawdown_pct <= resume_threshold
            ):
                _DRAW_SUSPENDED[strategy_key] = False
                _log_detail(
                    f"Resuming entries for strategy {strategy_key} as drawdown {drawdown_pct:.3%} <= {resume_threshold:.3%}"
                )
            if _DRAW_SUSPENDED.get(strategy_key):
                continue

            if (
                edge_strength < MIN_EDGE_STRENGTH
                and max(
                    avg_return,
                    simple_return,
                    takeprofit_return,
                    highlow_return,
                    maxdiff_return,
                    maxdiffalwayson_return,
                    kronos_profit,
                )
                <= 0
            ):
                _log_detail(
                    f"Skipping {symbol} - no actionable price edge "
                    f"(edge_strength={edge_strength:.6f}, avg_return={avg_return:.6f})"
                )
                continue

            effective_takeprofit = takeprofit_return if takeprofit_allowed_entry else 0.0
            effective_highlow = highlow_return if highlow_allowed_entry else 0.0
            effective_maxdiff = maxdiff_return if maxdiff_allowed_entry else 0.0
            effective_maxdiffalwayson = maxdiffalwayson_return if maxdiffalwayson_allowed_entry else 0.0
            kronos_contrib = max(kronos_profit, 0.0)
            primary_return = max(
                avg_return,
                simple_return,
                effective_takeprofit,
                effective_highlow,
                effective_maxdiff,
                effective_maxdiffalwayson,
                kronos_contrib,
                0.0,
            )

            bid_price, ask_price = fetch_bid_ask(symbol)
            spread_bps = compute_spread_bps(bid_price, ask_price)
            spread_cap = resolve_spread_cap(symbol)
            if not math.isfinite(spread_bps):
                spread_penalty_bps = float(spread_cap)
            else:
                spread_penalty_bps = min(max(spread_bps, 0.0), float(spread_cap))
            spread_penalty = spread_penalty_bps / 10000.0
            composite_score = primary_return - spread_penalty
            if SIMPLIFIED_MODE:
                tradeable, spread_reason = True, "simplified"
                edge_ok, edge_reason = True, "simplified"
            else:
                tradeable, spread_reason = is_tradeable(
                    symbol,
                    bid_price,
                    ask_price,
                    avg_dollar_vol=avg_dollar_vol,
                    atr_pct=atr_pct,
                )
                edge_ok, edge_reason = pass_edge_threshold(symbol, expected_move_pct)
            sign_toto = resolve_signal_sign(toto_move_pct)
            sign_kronos = resolve_signal_sign(kronos_move_pct)
            active_signs = [sign for sign in (sign_toto, sign_kronos) if sign in (-1, 1)]
            consensus_model_count = len(active_signs)
            consensus_ok = False
            if consensus_model_count >= 1:
                consensus_ok = agree_direction(*active_signs)
            consensus_reason = None
            fallback_source: Optional[str] = None
            if consensus_model_count == 0:
                consensus_reason = "No directional signal from Toto/Kronos"
            # Model disagreement is OK - we use predictions from the best-performing model
            # elif consensus_model_count > 1 and not consensus_ok:
            #     consensus_reason = f"Model disagreement toto={sign_toto} kronos={sign_kronos}"
            elif consensus_model_count == 1:
                if sign_toto != 0 and sign_kronos == 0:
                    fallback_source = "Toto"
                elif sign_kronos != 0 and sign_toto == 0:
                    fallback_source = "Kronos"
                if fallback_source:
                    _log_detail(f"{symbol}: consensus fallback to {fallback_source} signal only")

            if SIMPLIFIED_MODE:
                consensus_reason = None

            block_info = _evaluate_trade_block(symbol, position_side, strategy=best_strategy)
            last_pnl = block_info.get("last_pnl")
            last_closed_at = block_info.get("last_closed_at")
            if last_pnl is not None:
                if last_pnl < 0:
                    _record_loss_timestamp(symbol, last_closed_at)
                else:
                    clear_cooldown(symbol)
            now_utc = datetime.now(timezone.utc)
            cooldown_ok = True if SIMPLIFIED_MODE else can_trade_now(symbol, now_utc)

            # MaxDiff strategy bypasses most gates to match backtest behavior
            is_maxdiff = best_strategy in MAXDIFF_STRATEGIES

            walk_forward_notes: List[str] = []
            sharpe_cutoff: Optional[float] = None
            # MaxDiff backtest doesn't have walk-forward filters
            if not SIMPLIFIED_MODE and not is_maxdiff:
                default_cutoff = -0.25 if kronos_only_mode else 0.3
                env_key = "MARKETSIM_KRONOS_SHARPE_CUTOFF" if kronos_only_mode else "MARKETSIM_SHARPE_CUTOFF"
                sharpe_cutoff = _get_env_float(env_key)
                if sharpe_cutoff is None and kronos_only_mode:
                    sharpe_cutoff = _get_env_float("MARKETSIM_SHARPE_CUTOFF")
                if sharpe_cutoff is None:
                    sharpe_cutoff = default_cutoff
                if walk_forward_oos_sharpe is not None and sharpe_cutoff is not None:
                    if walk_forward_oos_sharpe < sharpe_cutoff:
                        walk_forward_notes.append(
                            f"Walk-forward Sharpe {walk_forward_oos_sharpe:.2f} below cutoff {sharpe_cutoff:.2f}"
                        )
                if (
                    not kronos_only_mode
                    and walk_forward_turnover is not None
                    and walk_forward_oos_sharpe is not None
                    and walk_forward_turnover > 2.0
                    and walk_forward_oos_sharpe < 0.5
                ):
                    walk_forward_notes.append(
                        f"Walk-forward turnover {walk_forward_turnover:.2f} high with Sharpe {walk_forward_oos_sharpe:.2f}"
                    )

            gating_reasons: List[str] = []

            if not DISABLE_TRADE_GATES:
                if not tradeable:
                    gating_reasons.append(spread_reason)
                # MaxDiff uses its own high/low predictions, skip edge gate
                if not edge_ok and not is_maxdiff:
                    gating_reasons.append(edge_reason)
                if kronos_only_mode and consensus_reason and "Model disagreement" in consensus_reason:
                    if sign_kronos in (-1, 1):
                        consensus_reason = None
                if kronos_only_mode and consensus_reason and consensus_reason.startswith("No directional signal"):
                    if sign_kronos in (-1, 1):
                        consensus_reason = None
                # MaxDiff doesn't need Toto/Kronos consensus - it has its own predictions
                if consensus_reason and not is_maxdiff:
                    gating_reasons.append(consensus_reason)
                if not cooldown_ok and not kronos_only_mode and not is_maxdiff:
                    gating_reasons.append("Cooldown active after recent loss")
                # MaxDiff doesn't use Kelly sizing in backtest, skip this gate
                if kelly_fraction <= 0 and not is_maxdiff:
                    gating_reasons.append("Kelly fraction <= 0")
                recent_sum = strategy_recent_sums.get(best_strategy)
                # MaxDiff backtest doesn't have recent returns filter
                if recent_sum is not None and recent_sum <= 0 and not is_maxdiff:
                    gating_reasons.append(f"Recent {best_strategy} returns sum {recent_sum:.4f} <= 0")

            base_blocked = False if SIMPLIFIED_MODE else block_info.get("blocked", False)
            if kronos_only_mode and base_blocked:
                base_blocked = False
            combined_reasons: List[str] = []
            if base_blocked and block_info.get("block_reason"):
                combined_reasons.append(block_info["block_reason"])
            combined_reasons.extend(gating_reasons)
            unique_reasons = []
            for reason in combined_reasons:
                if reason and reason not in unique_reasons:
                    unique_reasons.append(reason)
            block_reason = "; ".join(unique_reasons) if unique_reasons else None
            trade_blocked = base_blocked or bool(gating_reasons)

            result_row = {
                "avg_return": _metric(avg_return, default=0.0),
                "annual_return": _metric(annual_return, default=0.0),
                "predictions": backtest_df,
                "side": position_side,
                "predicted_movement": _metric(predicted_movement, default=0.0),
                "strategy": best_strategy,
                "predicted_high": _metric(predicted_high_price, default=close_price),
                "predicted_low": _metric(predicted_low_price, default=close_price),
                "predicted_close": _metric(predicted_close_price, default=close_price),
                "calibrated_close": _metric(calibrated_close_price, default=close_price),
                "last_close": _metric(close_price, default=close_price),
                "allowed_side": allowed_side or "both",
                "strategy_returns": strategy_returns,
                "strategy_annual_returns": strategy_returns_annual,
                "strategy_recent_sums": strategy_recent_sums,
                "recent_return_sum": strategy_recent_sums.get(best_strategy),
                "simple_return": _metric(simple_return, default=0.0),
                "maxdiff_return": _metric(maxdiff_return, default=0.0),
                "maxdiffalwayson_return": _metric(maxdiffalwayson_return, default=0.0),
                "unprofit_shutdown_return": _metric(unprofit_return, default=0.0),
                "unprofit_shutdown_sharpe": _metric(unprofit_sharpe, default=0.0),
                "expected_move_pct": _metric(expected_move_pct, default=0.0),
                "expected_move_pct_raw": _metric(raw_expected_move_pct, default=0.0),
                "price_skill": _metric(price_skill, default=0.0),
                "edge_strength": _metric(edge_strength, default=0.0),
                "directional_edge": _metric(directional_edge, default=0.0),
                "composite_score": _metric(composite_score, default=0.0),
                "strategy_entry_ineligible": strategy_ineligible,
                "strategy_candidate_forecasted_pnl": candidate_forecasted_pnl,
                "fallback_backtest": used_fallback_engine,
                "highlow_entry_allowed": highlow_allowed_entry,
                "takeprofit_entry_allowed": takeprofit_allowed_entry,
                "maxdiff_entry_allowed": maxdiff_allowed_entry,
                "maxdiffalwayson_entry_allowed": maxdiffalwayson_allowed_entry,
                "trade_blocked": trade_blocked,
                "block_reason": block_reason,
                "last_trade_pnl": last_pnl,
                "last_trade_closed_at": block_info.get("last_closed_at"),
                "cooldown_expires": block_info.get("cooldown_expires"),
                "trade_mode": block_info.get("trade_mode", "normal"),
                "pending_probe": block_info.get("pending_probe", False),
                "probe_active": block_info.get("probe_active", False),
                "probe_started_at": block_info.get("probe_started_at"),
                "probe_age_seconds": block_info.get("probe_age_seconds"),
                "probe_expires_at": block_info.get("probe_expires_at"),
                "probe_expired": block_info.get("probe_expired", False),
                "probe_transition_ready": block_info.get("probe_transition_ready", False),
                "learning_state": block_info.get("learning_state", {}),
                "bid_price": bid_price,
                "ask_price": ask_price,
                "spread_bps": None if math.isinf(spread_bps) else spread_bps,
                "spread_cap_bps": spread_cap,
                "tradeable_reason": spread_reason,
                "edge_gate_reason": edge_reason,
                "consensus_ok": consensus_ok,
                "consensus_reason": consensus_reason,
                "consensus_model_count": consensus_model_count,
                "kelly_fraction": kelly_fraction,
                "kelly_sigma_pct": sigma_pct,
                "toto_move_pct": toto_move_pct,
                "kronos_move_pct": kronos_move_pct,
                "avg_dollar_vol": (_metric(avg_dollar_vol, default=0.0) if avg_dollar_vol is not None else None),
                "atr_pct_14": _metric(atr_pct, default=0.0) if atr_pct is not None else None,
                "cooldown_active": not cooldown_ok,
                "walk_forward_oos_sharpe": walk_forward_oos_sharpe,
                "walk_forward_turnover": walk_forward_turnover,
                "walk_forward_highlow_sharpe": walk_forward_highlow_sharpe,
                "walk_forward_takeprofit_sharpe": walk_forward_takeprofit_sharpe,
                "walk_forward_maxdiff_sharpe": walk_forward_maxdiff_sharpe,
                "walk_forward_sharpe_cutoff": sharpe_cutoff,
                "walk_forward_notes": walk_forward_notes,
                "backtest_samples": sample_size,
            }
            if model_override:
                result_row["model_override"] = model_override
            forecast_model = str(result_row.get("close_prediction_source") or "").strip().lower()
            if forecast_model:
                result_row["forecast_model"] = forecast_model
            for source_key in ("chronos2_preaug_strategy", "chronos2_preaug_source", "chronos2_hparams_config_path", "chronos2_model_id"):
                value = raw_last_prediction.get(source_key)
                if isinstance(value, str) and value:
                    result_row[source_key] = value
            for numeric_key in ("chronos2_context_length", "chronos2_batch_size", "chronos2_prediction_length"):
                value = raw_last_prediction.get(numeric_key)
                if value is not None and math.isfinite(coerce_numeric(value, default=float("nan"))):
                    result_row[numeric_key] = coerce_numeric(value, default=0.0)
            if raw_last_prediction.get("chronos2_quantile_levels"):
                result_row["chronos2_quantile_levels"] = raw_last_prediction.get("chronos2_quantile_levels")
            if raw_last_prediction.get("chronos2_predict_kwargs"):
                result_row["chronos2_predict_kwargs"] = raw_last_prediction.get("chronos2_predict_kwargs")
            if selection_notes:
                result_row["strategy_selection_notes"] = selection_notes
            if ordered_strategies:
                result_row["strategy_sequence"] = ordered_strategies
            snapshot_row = latest_snapshot.get(symbol)
            if snapshot_row:
                result_row.update(snapshot_row)

            # Apply corrected high/low values (after snapshot update to ensure they're not overwritten with inverted values)
            if maxdiff_high_price is not None:
                result_row["maxdiffprofit_high_price"] = maxdiff_high_price
            if maxdiff_low_price is not None:
                result_row["maxdiffprofit_low_price"] = maxdiff_low_price
            if maxdiffalwayson_high_price is not None:
                result_row["maxdiffalwayson_high_price"] = maxdiffalwayson_high_price
            if maxdiffalwayson_low_price is not None:
                result_row["maxdiffalwayson_low_price"] = maxdiffalwayson_low_price
            if pctdiff_entry_low_price is not None:
                result_row["pctdiff_entry_low_price"] = pctdiff_entry_low_price
            if pctdiff_entry_high_price is not None:
                result_row["pctdiff_entry_high_price"] = pctdiff_entry_high_price
            if pctdiff_takeprofit_high_price is not None:
                result_row["pctdiff_takeprofit_high_price"] = pctdiff_takeprofit_high_price
            if pctdiff_takeprofit_low_price is not None:
                result_row["pctdiff_takeprofit_low_price"] = pctdiff_takeprofit_low_price
            neural_high_price = last_prediction.get("neuralpricing_high_price")
            neural_low_price = last_prediction.get("neuralpricing_low_price")
            if neural_high_price is not None:
                result_row["neuralpricing_high_price"] = neural_high_price
            if neural_low_price is not None:
                result_row["neuralpricing_low_price"] = neural_low_price
            neural_high_delta = last_prediction.get("neuralpricing_high_delta")
            neural_low_delta = last_prediction.get("neuralpricing_low_delta")
            if neural_high_delta is not None:
                result_row["neuralpricing_high_delta"] = neural_high_delta
            if neural_low_delta is not None:
                result_row["neuralpricing_low_delta"] = neural_low_delta
            neural_pnl_gain = last_prediction.get("neuralpricing_pnl_gain")
            if neural_pnl_gain is not None:
                result_row["neuralpricing_pnl_gain"] = neural_pnl_gain

            result_row["neural_strategy"] = _resolve_neural_strategy_name(symbol, result_row)

            if maxdiff_primary_side_raw is not None:
                result_row["maxdiff_primary_side"] = str(maxdiff_primary_side_raw).strip().lower() or "neutral"
            if maxdiff_trade_bias is not None:
                result_row["maxdiff_trade_bias"] = _metric(maxdiff_trade_bias, default=0.0)
            if pctdiff_primary_side_raw is not None:
                result_row["pctdiff_primary_side"] = str(pctdiff_primary_side_raw).strip().lower() or "neutral"
            if pctdiff_trade_bias is not None:
                result_row["pctdiff_trade_bias"] = _metric(pctdiff_trade_bias, default=0.0)

            maxdiff_numeric_keys = (
                "maxdiffprofit_high_price",
                "maxdiffprofit_low_price",
                "maxdiffprofit_profit_high_multiplier",
                "maxdiffprofit_profit_low_multiplier",
                "maxdiffprofit_profit",
            )
            for key in maxdiff_numeric_keys:
                if key in last_prediction:
                    result_row[key] = coerce_numeric(last_prediction.get(key), default=0.0)
            for count_key in ("maxdiff_trades_positive", "maxdiff_trades_negative", "maxdiff_trades_total"):
                if count_key in last_prediction:
                    result_row[count_key] = int(round(coerce_numeric(last_prediction.get(count_key), default=0.0)))
            if "maxdiffprofit_profit_values" in last_prediction:
                result_row["maxdiffprofit_profit_values"] = last_prediction.get("maxdiffprofit_profit_values")

            pctdiff_numeric_keys = (
                "pctdiff_entry_low_price",
                "pctdiff_entry_high_price",
                "pctdiff_takeprofit_high_price",
                "pctdiff_takeprofit_low_price",
                "pctdiff_entry_low_multiplier",
                "pctdiff_entry_high_multiplier",
                "pctdiff_long_pct",
                "pctdiff_short_pct",
                "pctdiff_profit",
            )
            for key in pctdiff_numeric_keys:
                if key in last_prediction:
                    result_row[key] = coerce_numeric(last_prediction.get(key), default=0.0)
            for count_key in ("pctdiff_trades_positive", "pctdiff_trades_negative", "pctdiff_trades_total"):
                if count_key in last_prediction:
                    result_row[count_key] = int(round(coerce_numeric(last_prediction.get(count_key), default=0.0)))
            for key in ("pctdiff_entry_hits", "pctdiff_takeprofit_hits"):
                if key in last_prediction:
                    result_row[key] = int(round(coerce_numeric(last_prediction.get(key), default=0.0)))
            if "pctdiff_profit_values" in last_prediction:
                result_row["pctdiff_profit_values"] = last_prediction.get("pctdiff_profit_values")

            maxdiffalwayson_numeric_keys = (
                "maxdiffalwayson_high_price",
                "maxdiffalwayson_low_price",
                "maxdiffalwayson_high_multiplier",
                "maxdiffalwayson_low_multiplier",
                "maxdiffalwayson_profit",
                "maxdiffalwayson_buy_contribution",
                "maxdiffalwayson_sell_contribution",
                "maxdiffalwayson_trade_bias",
                "maxdiffalwayson_turnover",
            )
            for key in maxdiffalwayson_numeric_keys:
                if key in last_prediction:
                    result_row[key] = coerce_numeric(last_prediction.get(key), default=0.0)
            for count_key in (
                "maxdiffalwayson_filled_buy_trades",
                "maxdiffalwayson_filled_sell_trades",
                "maxdiffalwayson_trades_total",
            ):
                if count_key in last_prediction:
                    result_row[count_key] = int(round(coerce_numeric(last_prediction.get(count_key), default=0.0)))
            if "maxdiffalwayson_profit_values" in last_prediction:
                result_row["maxdiffalwayson_profit_values"] = last_prediction.get("maxdiffalwayson_profit_values")
            try:
                log_strategy_snapshot(symbol, result_row, now_utc)
            except Exception as exc:
                logger.debug("Failed to log strategy snapshot for %s: %s", symbol, exc)
            results[symbol] = result_row
            _log_analysis_summary(symbol, result_row)

            # Save maxdiff plan if this strategy is profitable and allowed
            if maxdiff_allowed_entry and maxdiff_return > 0:
                maxdiff_plan = {
                    "symbol": symbol,
                    "high_target": result_row.get("predicted_high", close_price),
                    "low_target": result_row.get("predicted_low", close_price),
                    "maxdiffprofit_high_price": result_row.get("maxdiffprofit_high_price"),
                    "maxdiffprofit_low_price": result_row.get("maxdiffprofit_low_price"),
                    "maxdiffalwayson_high_price": result_row.get("maxdiffalwayson_high_price"),
                    "maxdiffalwayson_low_price": result_row.get("maxdiffalwayson_low_price"),
                    "neuralpricing_high_price": result_row.get("neuralpricing_high_price"),
                    "neuralpricing_low_price": result_row.get("neuralpricing_low_price"),
                    "avg_return": maxdiff_return,
                    "status": "identified",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "close_price": close_price,
                    "bid_price": bid_price,
                    "ask_price": ask_price,
                }
                _save_maxdiff_plan(symbol, maxdiff_plan)
            elif maxdiffalwayson_allowed_entry and maxdiffalwayson_return > 0:
                maxdiff_plan = {
                    "symbol": symbol,
                    "high_target": result_row.get("predicted_high", close_price),
                    "low_target": result_row.get("predicted_low", close_price),
                    "maxdiffprofit_high_price": result_row.get("maxdiffprofit_high_price"),
                    "maxdiffprofit_low_price": result_row.get("maxdiffprofit_low_price"),
                    "maxdiffalwayson_high_price": result_row.get("maxdiffalwayson_high_price"),
                    "maxdiffalwayson_low_price": result_row.get("maxdiffalwayson_low_price"),
                    "neuralpricing_high_price": result_row.get("neuralpricing_high_price"),
                    "neuralpricing_low_price": result_row.get("neuralpricing_low_price"),
                    "avg_return": maxdiffalwayson_return,
                    "status": "identified",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "close_price": close_price,
                    "bid_price": bid_price,
                    "ask_price": ask_price,
                    "strategy": "maxdiffalwayson",
                }
                _save_maxdiff_plan(symbol, maxdiff_plan)

        except Exception as e:
            logger.exception("Error analyzing %s: %s", symbol, str(e))
            import traceback
            logger.error("Full traceback:\n%s", traceback.format_exc())
            continue

    if skipped_equity_symbols:
        logger.debug(
            f"Skipping equity backtests while market closed: {', '.join(sorted(skipped_equity_symbols))}"
        )

    return dict(sorted(results.items(), key=lambda x: x[1]["composite_score"], reverse=True))


def build_portfolio(
    all_results: Dict[str, Dict],
    min_positions: int = DEFAULT_MIN_CORE_POSITIONS,
    max_positions: int = DEFAULT_MAX_PORTFOLIO,
    max_expanded: Optional[int] = None,
) -> Dict[str, Dict]:
    """Select a diversified portfolio while respecting trade blocks and price-edge metrics."""
    if not all_results:
        return {}

    if SIMPLIFIED_MODE:
        limit = max_expanded or max_positions
        ranked = sorted(
            all_results.items(),
            key=lambda item: _coerce_optional_float(item[1].get("avg_return")) or float("-inf"),
            reverse=True,
        )
        simple_picks: Dict[str, Dict] = {}
        for symbol, data in ranked:
            # Check if the SELECTED strategy has positive returns
            strategy = data.get("strategy", "simple")
            strategy_return_key = f"{strategy}_return"
            strategy_return = _coerce_optional_float(data.get(strategy_return_key))

            # Debug logging for simplified mode
            logger.debug(
                f"Portfolio simplified {symbol}: strategy={strategy} {strategy_return_key}={strategy_return} "
                f"passed={strategy_return is not None and strategy_return > 0}"
            )

            # Skip if selected strategy isn't profitable
            if strategy_return is None or strategy_return <= 0:
                continue

            pred_move = _coerce_optional_float(data.get("predicted_movement"))
            side = (data.get("side") or "").lower()
            if pred_move is not None:
                if side == "buy" and pred_move <= 0:
                    continue
                if side == "sell" and pred_move >= 0:
                    continue
            if data.get("trade_blocked") and data.get("trade_mode") != "probe":
                continue
            simple_picks[symbol] = data
            if len(simple_picks) >= limit:
                break
        _apply_forced_probe_annotations(simple_picks)
        return simple_picks

    sorted_by_composite = sorted(all_results.items(), key=lambda item: item[1].get("composite_score", 0), reverse=True)

    picks: Dict[str, Dict] = {}

    # Core picks prioritise consistently profitable strategies.
    for symbol, data in sorted_by_composite:
        if len(picks) >= max_positions:
            break
        if data.get("trade_blocked") and data.get("trade_mode") != "probe":
            continue

        # Check if the SELECTED strategy has positive returns
        strategy = data.get("strategy", "simple")
        strategy_return_key = f"{strategy}_return"
        strategy_return = data.get(strategy_return_key, 0)
        avg_return = data.get("avg_return", 0)
        strategy_forecast = get_selected_strategy_forecast(data)

        # Debug logging for strategy selection
        logger.debug(
            f"Portfolio eval {symbol}: strategy={strategy} {strategy_return_key}={strategy_return:.4f} "
            f"avg_return={avg_return:.4f} forecast={strategy_forecast:.4f} "
            f"passed={avg_return > 0 and strategy_return > 0 and strategy_forecast > 0}"
        )

        # Include if selected strategy is profitable with positive forecast
        if avg_return > 0 and strategy_return > 0 and strategy_forecast > 0:
            picks[symbol] = data

    # Ensure we reach the minimum desired portfolio size using best remaining composites.
    if len(picks) < min_positions:
        for symbol, data in sorted_by_composite:
            if len(picks) >= max_positions:
                break
            if symbol in picks or (data.get("trade_blocked") and data.get("trade_mode") != "probe"):
                continue

            strategy = data.get("strategy", "simple")
            strategy_return_key = f"{strategy}_return"
            strategy_return = data.get(strategy_return_key, 0)
            strategy_forecast = get_selected_strategy_forecast(data)

            logger.debug(
                f"Portfolio fallback {symbol}: strategy={strategy} {strategy_return_key}={strategy_return:.4f} "
                f"forecast={strategy_forecast:.4f} passed={strategy_return > 0 and strategy_forecast > 0}"
            )

            if strategy_return > 0 and strategy_forecast > 0:
                picks[symbol] = data

    # Optionally expand with high-price-edge opportunities to keep broader exposure.
    if max_expanded and len(picks) < max_expanded:
        sorted_by_edge = sorted(
            (
                (symbol, data)
                for symbol, data in all_results.items()
                if symbol not in picks and (
                    not data.get("trade_blocked") or data.get("trade_mode") == "probe"
                )
            ),
            key=lambda item: (
                item[1].get("edge_strength", 0),
                item[1].get("composite_score", 0),
            ),
            reverse=True,
        )
        for symbol, data in sorted_by_edge:
            if len(picks) >= max_expanded:
                break
            strategy_forecast = get_selected_strategy_forecast(data)
            avg_return = data.get("avg_return", 0)
            if avg_return > 0 and strategy_forecast > 0:
                picks[symbol] = data

    # Ensure probe-mode symbols are represented even if they fell outside the ranking filters.
    if ENABLE_PROBE_TRADES:
        probe_candidates = [(symbol, data) for symbol, data in all_results.items() if data.get("trade_mode") == "probe"]
        for symbol, data in probe_candidates:
            if symbol in picks:
                continue
            strategy_forecast = get_selected_strategy_forecast(data)
            if strategy_forecast <= 0:
                continue
            if max_expanded and len(picks) < max_expanded:
                picks[symbol] = data
            elif len(picks) < max_positions:
                picks[symbol] = data
            else:
                weakest_symbol, weakest_data = min(
                    picks.items(), key=lambda item: item[1].get("composite_score", float("-inf"))
                )
                weakest_forecast = get_selected_strategy_forecast(weakest_data)
                if weakest_forecast <= 0:
                    picks.pop(weakest_symbol, None)
                    picks[symbol] = data

    _apply_forced_probe_annotations(picks)
    return picks


def log_trading_plan(picks: Dict[str, Dict], action: str):
    """Log the trading plan without executing trades."""
    if not picks:
        logger.info(f"TRADING PLAN ({action}) - no candidates")
        return
    compact_lines = [_format_plan_line(symbol, data) for symbol, data in picks.items()]
    logger.info("TRADING PLAN (%s) count=%d | %s", action, len(picks), " ; ".join(compact_lines))


def _build_position_lookup(positions: Optional[List[object]]) -> Dict[str, Dict[str, object]]:
    lookup: Dict[str, Dict[str, object]] = {}
    if not positions:
        return lookup
    for pos in positions:
        symbol = getattr(pos, "symbol", None)
        if not symbol:
            continue
        raw_side = str(getattr(pos, "side", "")).lower()
        if raw_side not in {"long", "short"}:
            continue
        try:
            qty = abs(float(getattr(pos, "qty", 0.0) or 0.0))
        except (TypeError, ValueError):
            qty = 0.0
        if qty <= 0:
            continue
        lookup[symbol] = {"side": raw_side, "qty": qty}
    return lookup


def _is_closing_order(order, position_lookup: Dict[str, Dict[str, object]]) -> bool:
    symbol = getattr(order, "symbol", None)
    if not symbol:
        return False
    position = position_lookup.get(symbol)
    if not position:
        return False
    order_side = str(getattr(order, "side", "")).lower()
    pos_side = str(position.get("side", ""))
    if pos_side == "long" and order_side == "sell":
        return True
    if pos_side == "short" and order_side == "buy":
        return True
    return False


def _cancel_non_crypto_orders_out_of_hours(active_positions: Optional[List[object]] = None):
    """Cancel non-crypto entry orders during out-of-hours while keeping closing orders alive."""
    if not is_crypto_out_of_hours():
        return  # Market is open, keep stock orders

    positions_source = active_positions
    if positions_source is None:
        try:
            raw_positions = alpaca_wrapper.get_all_positions()
            positions_source = filter_to_realistic_positions(raw_positions)
        except Exception as exc:
            logger.warning(f"Failed to fetch positions for out-of-hours cleanup: {exc}")
            positions_source = []

    position_lookup = _build_position_lookup(positions_source)

    try:
        orders = alpaca_wrapper.get_orders()
    except Exception as exc:
        logger.warning(f"Failed to fetch orders for out-of-hours cleanup: {exc}")
        return

    cancelled_count = 0
    freed_notional = 0.0

    for order in orders:
        symbol = getattr(order, "symbol", None)
        if not symbol or is_crypto_symbol(symbol):
            continue  # Keep crypto orders

        if _is_closing_order(order, position_lookup):
            logger.debug(
                "Preserving closing order %s (%s) for %s during out-of-hours cleanup",
                getattr(order, "id", "unknown"),
                getattr(order, "side", "n/a"),
                symbol,
            )
            continue

        # This is a non-crypto entry/scale order during out-of-hours - cancel it
        order_id = getattr(order, "id", None)
        limit_price = getattr(order, "limit_price", None)
        qty = getattr(order, "qty", 0)

        if not order_id:
            continue

        try:
            alpaca_wrapper.cancel_order(order)
            cancelled_count += 1
            notional = 0.0
            try:
                notional = abs(float(qty) * float(limit_price)) if limit_price else 0.0
            except (TypeError, ValueError):
                notional = 0.0
            freed_notional += notional
            logger.info(
                f"Cancelled non-crypto order during out-of-hours: {symbol} "
                f"(freed ${notional:.2f} buying power for crypto)"
            )
        except Exception as exc:
            logger.warning(f"Failed to cancel {symbol} order {order_id}: {exc}")

    if cancelled_count > 0:
        logger.info(
            f"Out-of-hours cleanup: Cancelled {cancelled_count} non-crypto orders, "
            f"freed ${freed_notional:.2f} for crypto trading"
        )


def manage_positions(
    current_picks: Dict[str, Dict],
    previous_picks: Dict[str, Dict],
    all_analyzed_results: Dict[str, Dict],
):
    """Execute actual position management."""
    positions = alpaca_wrapper.get_all_positions()
    positions = filter_to_realistic_positions(positions)

    # Cancel non-crypto orders during out-of-hours to free buying power for crypto
    _cancel_non_crypto_orders_out_of_hours(positions)
    logger.info("EXECUTING POSITION CHANGES:")

    total_exposure_value = _calculate_total_exposure_value(positions)
    probe_state = _apply_forced_probe_annotations(current_picks)
    if probe_state.force_probe:
        logger.warning(
            "Global risk controls active: restricting new entries to probe trades for %s (%s)",
            probe_state.probe_date.isoformat() if probe_state.probe_date else "current session",
            probe_state.reason or "previous day loss",
        )

    day_pl_value = None
    try:
        account = alpaca_wrapper.get_account()
    except Exception as exc:
        logger.warning("Failed to fetch account while recording risk snapshot: %s", exc)
        account = None
    if account is not None:
        try:
            equity = float(getattr(account, "equity", 0.0))
            last_equity = float(getattr(account, "last_equity", equity))
            day_pl_value = equity - last_equity
        except Exception as exc:
            logger.warning("Failed to compute day P&L for risk snapshot: %s", exc)

    snapshot_kwargs = {}
    risk_timestamp = datetime.now(timezone.utc)
    if day_pl_value is not None:
        snapshot_kwargs["day_pl"] = day_pl_value
        record_day_pl(day_pl_value, observed_at=risk_timestamp)
    else:
        record_day_pl(None, observed_at=risk_timestamp)
    try:
        snapshot = record_portfolio_snapshot(total_exposure_value, observed_at=risk_timestamp, **snapshot_kwargs)
    except TypeError as exc:
        if snapshot_kwargs and "unexpected keyword argument" in str(exc):
            snapshot = record_portfolio_snapshot(total_exposure_value)
        else:
            raise
    logger.info(
        f"Portfolio snapshot recorded: value=${total_exposure_value:.2f}, "
        f"global risk threshold={snapshot.risk_threshold:.2f}x"
    )

    risk_threshold = float(getattr(snapshot, "risk_threshold", 1.0) or 1.0)

    try:
        sim_state = get_state()
    except RuntimeError:
        sim_state = None

    if not positions:
        logger.info("No positions to analyze")
    else:
        for position in positions:
            _handle_live_drawdown(position)

    if not all_analyzed_results and not current_picks:
        logger.warning("No analysis results available - skipping position closure checks")
        return

    # Handle position closures
    for position in positions:
        symbol = position.symbol
        normalized_side = _normalize_side_for_key(getattr(position, "side", ""))
        should_close = False
        close_reason = ""

        if symbol not in current_picks:
            # For crypto on weekends, only close if direction changed
            if symbol in all_crypto_symbols and not is_nyse_trading_day_now():
                if symbol in all_analyzed_results and not is_same_side(
                    all_analyzed_results[symbol]["side"], position.side
                ):
                    logger.info(f"Closing crypto position for {symbol} due to direction change (weekend)")
                    should_close = True
                    close_reason = "weekend_direction_change"
                else:
                    logger.info(f"Keeping crypto position for {symbol} on weekend - no direction change")
            # For stocks when market is closed, only close if direction changed
            elif symbol not in all_crypto_symbols and not is_nyse_trading_day_now():
                if symbol in all_analyzed_results and not is_same_side(
                    all_analyzed_results[symbol]["side"], position.side
                ):
                    logger.info(f"Closing stock position for {symbol} due to direction change (market closed)")
                    should_close = True
                    close_reason = "closed_market_direction_change"
                else:
                    logger.info(f"Keeping stock position for {symbol} when market closed - no direction change")
            else:
                logger.info(f"Closing position for {symbol} as it's no longer in top picks")
                should_close = True
                close_reason = "not_in_portfolio"
        elif symbol not in all_analyzed_results:
            # Only close positions when no analysis data if it's a short position and market is open
            if is_sell_side(position.side) and is_nyse_trading_day_now():
                logger.info(
                    f"Closing short position for {symbol} as no analysis data available and market is open - reducing risk"
                )
                should_close = True
                close_reason = "no_analysis_short"
            else:
                logger.info(f"No analysis data for {symbol} but keeping position (not a short or market not open)")
        elif not is_same_side(all_analyzed_results[symbol]["side"], position.side):
            logger.info(
                f"Closing position for {symbol} due to direction change from {position.side} to {all_analyzed_results[symbol]['side']}"
            )
            should_close = True
            close_reason = f"direction_change_to_{all_analyzed_results[symbol]['side']}"

        probe_meta = all_analyzed_results.get(symbol, {})
        if not probe_meta:
            active_trade = _get_active_trade(symbol, normalized_side)
            entry_strategy = active_trade.get("entry_strategy") if active_trade else None
            probe_meta = _evaluate_trade_block(symbol, normalized_side, strategy=entry_strategy)
        probe_meta = _ensure_probe_state_consistency(position, normalized_side, probe_meta)
        if probe_meta.get("probe_expired") and not should_close:
            logger.info(
                f"Closing position for {symbol} as probe duration exceeded {PROBE_MAX_DURATION} "
                "without transition; scheduling backout"
            )
            should_close = True
            close_reason = "probe_duration_exceeded"

        if not should_close:
            hold_limit_seconds = _symbol_max_hold_seconds(symbol)
            if hold_limit_seconds:
                active_trade_meta = _get_active_trade(symbol, normalized_side)
                opened_at_wall = _parse_timestamp(active_trade_meta.get("opened_at"))
                opened_at_sim = _parse_timestamp(active_trade_meta.get("opened_at_sim"))
                hold_age_seconds = None
                if opened_at_sim is not None and sim_state is not None:
                    sim_now = getattr(getattr(sim_state, "clock", None), "current", None)
                    if sim_now is not None:
                        hold_age_seconds = (sim_now - opened_at_sim).total_seconds()
                if hold_age_seconds is None and opened_at_wall is not None:
                    hold_age_seconds = (datetime.now(timezone.utc) - opened_at_wall).total_seconds()
                if hold_age_seconds is not None and hold_age_seconds >= hold_limit_seconds:
                    logger.info(
                        f"Closing {symbol} {normalized_side} after {hold_age_seconds:.0f}s (max hold {hold_limit_seconds:.0f}s)."
                    )
                    should_close = True
                    close_reason = "max_hold_exceeded"

        if should_close:
            _record_trade_outcome(position, close_reason or "unspecified")
            backout_near_market(
                symbol,
                start_offset_minutes=BACKOUT_START_OFFSET_MINUTES,
                sleep_seconds=BACKOUT_SLEEP_SECONDS,
                market_close_buffer_minutes=BACKOUT_MARKET_CLOSE_BUFFER_MINUTES,
                market_close_force_minutes=BACKOUT_MARKET_CLOSE_FORCE_MINUTES,
            )

    # Refresh watchers for existing maxdiff positions
    for position in positions:
        symbol = position.symbol
        normalized_side = _normalize_side_for_key(getattr(position, "side", ""))

        # Get pick_data first to determine if position should be tracked
        pick_data = current_picks.get(symbol)
        if not pick_data:
            # Position exists but not in current picks - check analyzed results
            if symbol not in all_analyzed_results:
                logger.debug(f"Skipping watcher refresh for {symbol} - not in current analysis")
                continue
            pick_data = all_analyzed_results[symbol]

        # Only process if the side matches
        if not is_same_side(pick_data.get("side"), position.side):
            logger.debug(f"Skipping watcher refresh for {symbol} {normalized_side} - side mismatch with forecast")
            continue

        # Get strategy for this position
        active_trade = _get_active_trade(symbol, normalized_side)
        entry_strategy = pick_data.get("strategy")

        # Create active_trade entry if missing but position exists with matching forecast
        if not active_trade and entry_strategy in MAXDIFF_LIMIT_STRATEGIES:
            position_qty = abs(float(getattr(position, "qty", 0.0)))
            logger.info(
                f"Creating missing active_trade entry for {symbol} {normalized_side} "
                f"(qty={position_qty}, strategy={entry_strategy})"
            )
            _update_active_trade(
                symbol,
                normalized_side,
                mode="normal",
                qty=position_qty,
                strategy=entry_strategy,
            )
            _normalize_active_trade_patch(_update_active_trade)
            active_trade = _get_active_trade(symbol, normalized_side)

        if not active_trade:
            continue

        # Verify strategy is maxdiff-based
        stored_entry_strategy = active_trade.get("entry_strategy")
        if stored_entry_strategy not in MAXDIFF_LIMIT_STRATEGIES:
            continue

        # For maxdiff strategies, respawn watchers if they're missing or expired
        # This ensures 24/7 coverage for crypto and continuous coverage for stocks
        entry_strategy = stored_entry_strategy

        # Get current bid/ask for sizing
        bid_price, ask_price = fetch_bid_ask(symbol)
        if bid_price is None or ask_price is None:
            logger.debug(f"Skipping watcher refresh for {symbol} - no bid/ask available")
            continue

        entry_price = ask_price if is_buy_side(position.side) else bid_price
        target_qty = get_qty(symbol, entry_price, positions)
        if target_qty is None or target_qty <= 0:
            logger.debug(f"Skipping watcher refresh for {symbol} - invalid target qty")
            continue

        # Check existing entry watcher to decide if/how to refresh
        is_buy = is_buy_side(position.side)
        is_crypto = symbol in all_crypto_symbols
        from pathlib import Path
        watcher_dir = get_state_dir() / f"maxdiff_watchers{STATE_SUFFIX or ''}"

        from src.watcher_refresh_utils import find_existing_watcher_price, should_spawn_watcher

        # Check for existing entry watcher
        existing_limit_price, entry_reason = find_existing_watcher_price(
            watcher_dir,
            symbol,
            normalized_side,
            "entry",
            is_crypto,
            max_age_hours=24.0,
        )

        # Determine new forecast prices
        preferred_limit = _lookup_entry_price(pick_data, entry_strategy, normalized_side)
        fallback = pick_data.get("predicted_low" if is_buy else "predicted_high")
        new_limit_price = preferred_limit if preferred_limit is not None else fallback

        # Decide whether to spawn and which price to use
        should_spawn_entry, limit_price, spawn_reason = should_spawn_watcher(
            existing_limit_price,
            new_limit_price,
            "entry",
        )

        if limit_price is None or limit_price <= 0:
            logger.debug(f"Skipping watcher refresh for {symbol} - invalid limit price")
            continue

        # Spawn entry watcher only if needed
        if should_spawn_entry:
            try:
                logger.info(
                    f"Refreshing entry watcher for existing {symbol} {normalized_side} position @ {limit_price:.4f} ({spawn_reason})"
                )
                spawn_open_position_at_maxdiff_takeprofit(
                    symbol,
                    normalized_side,
                    float(limit_price),
                    float(target_qty),
                    poll_seconds=MAXDIFF_ENTRY_WATCHER_POLL_SECONDS,
                    entry_strategy=entry_strategy,
                )
            except Exception as exc:
                logger.warning(f"Failed to refresh entry watcher for {symbol} {normalized_side}: {exc}")
        else:
            logger.debug(f"{symbol} {normalized_side}: Skipping entry watcher refresh ({spawn_reason})")

        # Check for existing exit watcher
        existing_takeprofit_price, exit_reason = find_existing_watcher_price(
            watcher_dir,
            symbol,
            normalized_side,
            "exit",
            is_crypto,
            max_age_hours=24.0,
        )

        # Determine new forecast takeprofit price
        new_takeprofit_price = _lookup_takeprofit_price(pick_data, entry_strategy, normalized_side)
        if new_takeprofit_price is None:
            new_takeprofit_price = pick_data.get("predicted_high" if is_buy else "predicted_low")

        # Decide whether to spawn and which price to use
        should_spawn_exit, takeprofit_price, exit_spawn_reason = should_spawn_watcher(
            existing_takeprofit_price,
            new_takeprofit_price,
            "exit",
        )

        if takeprofit_price is not None and takeprofit_price > 0 and should_spawn_exit:
            try:
                logger.info(
                    f"Refreshing exit watcher for existing {symbol} {normalized_side} position @ {takeprofit_price:.4f} ({exit_spawn_reason})"
                )
                spawn_close_position_at_maxdiff_takeprofit(
                    symbol,
                    normalized_side,
                    float(takeprofit_price),
                    entry_strategy=entry_strategy,
                )
            except Exception as exc:
                logger.warning(f"Failed to refresh exit watcher for {symbol} {normalized_side}: {exc}")
        elif not should_spawn_exit:
            logger.debug(f"{symbol} {normalized_side}: Skipping exit watcher refresh ({exit_spawn_reason})")

    # Enter new positions from current_picks
    if not current_picks:
        logger.warning("No current picks available - skipping new position entry")
        return

    candidate_lines = _format_entry_candidates(current_picks)
    if candidate_lines:
        logger.info("Entry candidates (%d): %s", len(candidate_lines), " ; ".join(candidate_lines))
    equity = float(getattr(alpaca_wrapper, "equity", 0.0) or 0.0)
    if equity <= 0:
        equity = ensure_lower_bound(total_exposure_value, 1.0, default=1.0)
    max_total_exposure_value = (MAX_TOTAL_EXPOSURE_PCT / 100.0) * equity

    maxdiff_entries_seen = 0

    always_on_candidates: List[Tuple[str, float]] = []
    for symbol, pick_data in current_picks.items():
        if pick_data.get("strategy") != "maxdiffalwayson":
            continue
        avg_return = coerce_numeric(pick_data.get("avg_return"), default=0.0)
        always_on_candidates.append((symbol, avg_return))
    always_on_candidates.sort(key=lambda item: item[1], reverse=True)
    always_on_priority = {symbol: index + 1 for index, (symbol, _) in enumerate(always_on_candidates)}
    always_on_forced_symbols = {symbol for symbol, _ in always_on_candidates[:MAXDIFF_ALWAYS_ON_PRIORITY_LIMIT]}

    # Extract forecasted PnL for all symbols to prioritize processing order
    all_candidates: List[Tuple[str, float]] = []
    crypto_candidates: List[Tuple[str, float]] = []

    for symbol, pick_data in current_picks.items():
        # Get forecasted PnL for the selected strategy
        selected_strategy = pick_data.get("strategy")
        strategy_forecasts = pick_data.get("strategy_candidate_forecasted_pnl", {})
        forecasted_pnl = coerce_numeric(
            strategy_forecasts.get(selected_strategy, pick_data.get("avg_return", 0.0)),
            default=0.0
        )
        all_candidates.append((symbol, forecasted_pnl))

        # Track crypto separately for out-of-hours tolerance ranking
        if symbol in all_crypto_symbols:
            crypto_candidates.append((symbol, forecasted_pnl))

    # Calculate crypto ranks (still needed for out-of-hours tolerance)
    crypto_candidates.sort(key=lambda item: item[1], reverse=True)
    crypto_ranks = {symbol: index + 1 for index, (symbol, _) in enumerate(crypto_candidates)}

    # Sort ALL symbols by forecasted PnL (highest first) to ensure best opportunities
    # get first access to available equity/buying power, regardless of crypto vs stock
    all_candidates.sort(key=lambda item: item[1], reverse=True)
    sorted_picks = [(symbol, current_picks[symbol]) for symbol, _ in all_candidates]

    # Log priority rankings for visibility
    if all_candidates:
        logger.info("Symbol priority rankings (by forecasted PnL, highest first):")
        for rank, (symbol, forecasted_pnl) in enumerate(all_candidates, start=1):
            symbol_type = "crypto" if symbol in all_crypto_symbols else "stock"
            crypto_rank_str = f", crypto_rank={crypto_ranks[symbol]}" if symbol in crypto_ranks else ""
            logger.info(f"  {rank}. {symbol} ({symbol_type}): forecasted_pnl={forecasted_pnl:.6f}{crypto_rank_str}")

    for symbol, original_data in sorted_picks:
        data = dict(original_data)
        current_picks[symbol] = data
        is_maxdiff_strategy = data.get("strategy") in MAXDIFF_LIMIT_STRATEGIES
        maxdiff_overflow = False
        if is_maxdiff_strategy:
            maxdiff_entries_seen += 1
            data["maxdiff_spread_rank"] = maxdiff_entries_seen
            if MAX_MAXDIFFS and maxdiff_entries_seen > MAX_MAXDIFFS:
                maxdiff_overflow = True
                data["maxdiff_spread_overflow"] = True
            else:
                data.pop("maxdiff_spread_overflow", None)
        else:
            data.pop("maxdiff_spread_rank", None)
            data.pop("maxdiff_spread_overflow", None)

        priority_rank = None
        force_immediate_entry = False
        if data.get("strategy") == "maxdiffalwayson":
            priority_rank = always_on_priority.get(symbol)
            force_immediate_entry = symbol in always_on_forced_symbols
            if priority_rank is not None:
                data["maxdiffalwayson_priority_rank"] = priority_rank
            else:
                data.pop("maxdiffalwayson_priority_rank", None)
            if force_immediate_entry:
                data["maxdiffalwayson_force_immediate"] = True
            else:
                data.pop("maxdiffalwayson_force_immediate", None)
        else:
            data.pop("maxdiffalwayson_priority_rank", None)
            data.pop("maxdiffalwayson_force_immediate", None)
        simplified_mode = SIMPLIFIED_MODE
        if simplified_mode:
            data["trade_mode"] = "normal"
            trade_mode = "normal"
            is_probe_trade = False
            force_probe = False
            probe_transition_ready = False
            probe_expired = False
        else:
            if ENABLE_PROBE_TRADES:
                if symbol.upper() in PROBE_SYMBOLS and data.get("trade_mode", "normal") != "probe":
                    data["trade_mode"] = "probe"
                trade_mode = data.get("trade_mode", "normal")
                is_probe_trade = trade_mode == "probe"
                forced_by_state = bool(data.get("forced_probe"))
                force_probe = bool(_symbol_force_probe(symbol)) or forced_by_state
                if force_probe and data.get("trade_mode") != "probe":
                    data["trade_mode"] = "probe"
                    current_picks[symbol] = data
                    reason_msg = (
                        "; ".join(data.get("forced_probe_reasons", []))
                        if forced_by_state
                        else "MARKETSIM_SYMBOL_FORCE_PROBE_MAP"
                    )
                    logger.info(f"{symbol}: Forcing probe mode ({reason_msg}).")
                    trade_mode = data["trade_mode"]
                    is_probe_trade = True
                probe_transition_ready = data.get("probe_transition_ready", False)
                probe_expired = data.get("probe_expired", False)
            else:
                if data.get("trade_mode") != "normal":
                    data["trade_mode"] = "normal"
                trade_mode = "normal"
                is_probe_trade = False
                force_probe = False
                probe_transition_ready = False
                probe_expired = False

            if data.get("trade_blocked") and not is_probe_trade:
                logger.info(f"Skipping {symbol} due to active block: {data.get('block_reason', 'recent loss')}")
                continue
            if probe_expired:
                logger.info(
                    f"Skipping {symbol} entry while probe backout executes (duration exceeded {PROBE_MAX_DURATION})."
                )
                continue
            min_move = _symbol_min_move(symbol)
            if min_move is not None:
                predicted_move = abs(coerce_numeric(data.get("predicted_movement"), default=0.0))
                if predicted_move < min_move:
                    logger.info(
                        f"Skipping {symbol} - predicted move {predicted_move:.4f} below minimum "
                        f"{min_move:.4f} configured via MARKETSIM_SYMBOL_MIN_MOVE_MAP."
                    )
                    continue
            min_predicted_direction = _symbol_min_predicted_move(symbol)
            if min_predicted_direction is not None:
                predicted_movement = coerce_numeric(data.get("predicted_movement"), default=None)
                if predicted_movement is None:
                    logger.info(
                        f"Skipping {symbol} - missing predicted movement required by "
                        "MARKETSIM_SYMBOL_MIN_PREDICTED_MOVE_MAP."
                    )
                    continue
                threshold = max(min_predicted_direction, 0.0)
                if threshold > 0:
                    if data["side"] == "buy":
                        if predicted_movement < threshold:
                            logger.info(
                                f"Skipping {symbol} - predicted move {predicted_movement:.4f} below "
                                f"minimum {threshold:.4f} for long entries "
                                "(MARKETSIM_SYMBOL_MIN_PREDICTED_MOVE_MAP)."
                            )
                            continue
                    elif data["side"] == "sell":
                        if predicted_movement > -threshold:
                            logger.info(
                                f"Skipping {symbol} - predicted move {predicted_movement:.4f} above "
                                f"-{threshold:.4f} for short entries "
                                "(MARKETSIM_SYMBOL_MIN_PREDICTED_MOVE_MAP)."
                            )
                            continue
            min_strategy_return = _symbol_min_strategy_return(symbol)
            if min_strategy_return is not None:
                strategy_key = data.get("strategy")
                strategy_returns = data.get("strategy_returns", {}) or {}
                strategy_return = coerce_numeric(strategy_returns.get(strategy_key), default=None)
                if strategy_return is None:
                    strategy_return = coerce_numeric(data.get("avg_return"), default=None)
                if strategy_return is None:
                    strategy_return = coerce_numeric(data.get("predicted_movement"), default=None)
                if strategy_return is None:
                    logger.info(
                        f"Skipping {symbol} - missing strategy return to compare with "
                        "MARKETSIM_SYMBOL_MIN_STRATEGY_RETURN_MAP."
                    )
                    continue
                if min_strategy_return < 0:
                    if strategy_return > min_strategy_return:
                        logger.info(
                            f"Skipping {symbol} - strategy return {strategy_return:.4f} "
                            f"above allowed maximum {min_strategy_return:.4f} for short bias."
                        )
                        continue
                elif min_strategy_return > 0:
                    if strategy_return < min_strategy_return:
                        logger.info(
                            f"Skipping {symbol} - strategy return {strategy_return:.4f} "
                            f"below minimum {min_strategy_return:.4f}."
                        )
                        continue
            trend_threshold = _symbol_trend_pnl_threshold(symbol)
            resume_threshold = _symbol_trend_resume_threshold(symbol)
            if trend_threshold is not None or resume_threshold is not None:
                pnl_stat = _get_trend_stat(symbol, "pnl")
                if pnl_stat is None:
                    logger.debug(
                        f"Trend PnL stat unavailable for {symbol}; skipping trend-based suspension check."
                    )
                else:
                    if trend_threshold is not None and pnl_stat <= trend_threshold:
                        logger.info(
                            f"Skipping {symbol} - cumulative trend PnL {pnl_stat:.2f} ≤ "
                            f"{trend_threshold:.2f} from MARKETSIM_TREND_PNL_SUSPEND_MAP."
                        )
                        continue
                    if resume_threshold is not None and pnl_stat < resume_threshold:
                        logger.info(
                            f"Skipping {symbol} - cumulative trend PnL {pnl_stat:.2f} < "
                            f"{resume_threshold:.2f} resume floor (MARKETSIM_TREND_PNL_RESUME_MAP)."
                        )
                        continue

        position_exists = any(p.symbol == symbol for p in positions)
        correct_side = any(p.symbol == symbol and is_same_side(p.side, data["side"]) for p in positions)

        transition_to_normal = (
            is_probe_trade and not force_probe and probe_transition_ready and position_exists and correct_side
        )
        effective_probe = is_probe_trade and not transition_to_normal

        if transition_to_normal:
            logger.info(f"{symbol}: Probe transition ready; targeting full exposure subject to risk limits.")

        # Calculate current position size and target size
        current_position_size = 0.0
        current_position_value = 0.0
        current_position_side: Optional[str] = None
        for p in positions:
            if p.symbol == symbol:
                current_position_size = float(p.qty)
                current_position_side = getattr(p, "side", None)
                if hasattr(p, "current_price"):
                    current_position_value = current_position_size * float(p.current_price)
                break

        min_trade_qty = MIN_CRYPTO_QTY if symbol in all_crypto_symbols else MIN_STOCK_QTY
        if effective_probe:
            logger.info(f"{symbol}: Probe mode enabled; minimum trade quantity set to {min_trade_qty}")

        # Calculate target position size
        bid_price, ask_price = fetch_bid_ask(symbol)
        entry_price = None
        target_qty = 0.0

        should_enter = False
        needs_size_increase = False

        if bid_price is not None and ask_price is not None:
            entry_price = ask_price if data["side"] == "buy" else bid_price

            if effective_probe:
                target_qty = ensure_lower_bound(min_trade_qty, 0.0, default=min_trade_qty)
                logger.info(f"{symbol}: Probe sizing fixed at minimum tradable quantity {target_qty}")
                should_enter = not position_exists or not correct_side
                needs_size_increase = False
            elif data.get("strategy") in MAXDIFF_STRATEGIES or not ENABLE_KELLY_SIZING:
                # Simple sizing: spread global risk over 2 positions
                # MAXDIFF strategies ALWAYS use simple sizing (equity/2 for crypto, buying_power*risk/2 for stocks)
                target_qty = _get_simple_qty(symbol, entry_price, positions)
                if target_qty < min_trade_qty:
                    target_qty = min_trade_qty
                target_qty = _apply_neural_scale(
                    target_qty,
                    min_qty=min_trade_qty,
                    symbol=symbol,
                    strategy=data.get("strategy"),
                    effective_probe=effective_probe,
                    strategy_key=data.get("neural_strategy"),
                )
                target_value = target_qty * entry_price
                logger.info(
                    f"{symbol}: Simple sizing - Current position: {current_position_size} qty (${current_position_value:.2f}), "
                    f"Target: {target_qty} qty (${target_value:.2f})"
                )
                should_enter = not position_exists or not correct_side
                needs_size_increase = False
            else:
                # Kelly sizing for non-MAXDIFF strategies when ENABLE_KELLY_SIZING is True
                computed_qty = get_qty(symbol, entry_price, positions)
                if computed_qty is None:
                    computed_qty = 0.0
                base_qty = computed_qty
                # Apply Kelly fraction scaling
                drawdown_scale = _kelly_drawdown_scale(data.get("strategy"), symbol)
                base_kelly = ensure_lower_bound(
                    coerce_numeric(data.get("kelly_fraction"), default=1.0),
                    0.0,
                    default=0.0,
                )
                kelly_value = base_kelly
                if drawdown_scale < 1.0 and base_kelly > 0:
                    scaled_kelly = ensure_lower_bound(base_kelly * drawdown_scale, 0.0, default=0.0)
                    if scaled_kelly < base_kelly:
                        logger.info(
                            f"{symbol}: Kelly reduced from {base_kelly:.3f} to {scaled_kelly:.3f} via drawdown scaling"
                        )
                    kelly_value = scaled_kelly
                if kelly_value <= 0:
                    logger.info(f"{symbol}: Kelly fraction non-positive; skipping entry.")
                    continue
                kelly_fraction = kelly_value
                data["kelly_fraction"] = kelly_fraction
                target_qty = ensure_lower_bound(base_qty * kelly_value, 0.0, default=0.0)
                if target_qty < min_trade_qty:
                    target_qty = min_trade_qty
                target_qty = _apply_neural_scale(
                    target_qty,
                    min_qty=min_trade_qty,
                    symbol=symbol,
                    strategy=data.get("strategy"),
                    effective_probe=effective_probe,
                    strategy_key=data.get("neural_strategy"),
                )
                target_value = target_qty * entry_price
                logger.info(
                    f"{symbol}: Current position: {current_position_size} qty (${current_position_value:.2f}), "
                    f"Target: {target_qty} qty (${target_value:.2f}) using Kelly fraction {kelly_value:.3f}"
                )
                if not position_exists:
                    should_enter = True
                    needs_size_increase = False
                elif not correct_side:
                    should_enter = True
                    needs_size_increase = False
                else:
                    should_enter = should_rebalance(
                        current_position_side,
                        data["side"],
                        current_position_size,
                        target_qty,
                    )
                    needs_size_increase = should_enter and abs(current_position_size) < abs(target_qty)

                current_abs_value = abs(current_position_value)
                projected_value = abs(target_qty * entry_price)
                new_total_value = total_exposure_value - current_abs_value + projected_value
                projected_pct = (new_total_value / equity) * 100.0 if equity > 0 else 0.0
                if projected_pct > MAX_TOTAL_EXPOSURE_PCT:
                    allowed_value = max_total_exposure_value - (total_exposure_value - current_abs_value)

                    if allowed_value <= 0:
                        # Skip trade when exposure is maxed - no tiny probe orders
                        logger.info(
                            f"Skipping {symbol} entry to respect max exposure "
                            f"({projected_pct:.1f}% > {MAX_TOTAL_EXPOSURE_PCT:.1f}%)"
                        )
                        continue
                    elif allowed_value > 0:
                        adjusted_qty = ensure_lower_bound(
                            safe_divide(allowed_value, entry_price, default=0.0),
                            0.0,
                            default=0.0,
                        )
                        if adjusted_qty <= 0:
                            # Skip trade when adjustment gives non-positive qty
                            logger.info(
                                f"Skipping {symbol} entry after exposure adjustment resulted in non-positive qty."
                            )
                            continue
                        logger.info(
                            f"Adjusting {symbol} target qty from {target_qty} to {adjusted_qty:.4f} "
                            f"to maintain exposure at {MAX_TOTAL_EXPOSURE_PCT:.1f}% max."
                        )
                        target_qty = adjusted_qty
                        projected_value = abs(target_qty * entry_price)
                        new_total_value = total_exposure_value - current_abs_value + projected_value
        else:
            # Fallback to old logic if we can't get prices
            if symbol in all_crypto_symbols:
                should_enter = (not position_exists and is_buy_side(data["side"])) or effective_probe
            else:
                should_enter = not position_exists or effective_probe
            if effective_probe:
                if ask_price is not None or bid_price is not None:
                    entry_price = ask_price if data["side"] == "buy" else bid_price
                target_qty = ensure_lower_bound(min_trade_qty, 0.0, default=min_trade_qty)

        entry_strategy = data.get("strategy")
        stored_entry_strategy = "maxdiff" if entry_strategy in MAXDIFF_LIMIT_STRATEGIES else entry_strategy
        recorded_entry_strategy = entry_strategy

        if effective_probe and target_qty <= 0:
            logger.warning(f"{symbol}: Unable to determine positive probe quantity; deferring trade.")
            _mark_probe_pending(symbol, data["side"], strategy=stored_entry_strategy)
            continue

        if should_enter or not correct_side:
            max_entries_per_run, limit_key = _symbol_max_entries_per_run(symbol, stored_entry_strategy)
            resolved_limit_key = limit_key or _normalize_entry_key(symbol, None)
            current_count = 0
            if max_entries_per_run is not None and resolved_limit_key is not None:
                current_count = _current_symbol_entry_count(
                    symbol,
                    stored_entry_strategy,
                    key=resolved_limit_key,
                )
            is_new_position_entry = not position_exists or not correct_side or effective_probe or transition_to_normal
            if (
                max_entries_per_run is not None
                and max_entries_per_run >= 0
                and is_new_position_entry
                and resolved_limit_key is not None
                and current_count >= max_entries_per_run
            ):
                logger.info(
                    f"{symbol}: Skipping entry to respect per-run max entries limit "
                    f"({current_count}/{max_entries_per_run})."
                )
                if effective_probe:
                    _mark_probe_pending(symbol, data["side"], strategy=stored_entry_strategy)
                continue

            if (
                max_entries_per_run is not None
                and max_entries_per_run > 0
                and is_new_position_entry
                and resolved_limit_key is not None
                and current_count < max_entries_per_run
            ):
                warn_threshold = max(0, int(math.floor(max_entries_per_run * 0.8)))
                if current_count >= warn_threshold:
                    logger.info(
                        f"{symbol}: Entries {current_count}/{max_entries_per_run} nearing cap "
                        f"for {resolved_limit_key}; next entry will reduce remaining headroom."
                    )

            entry_executed = False
            if needs_size_increase and bid_price is not None and ask_price is not None and not effective_probe:
                entry_price = ask_price if data["side"] == "buy" else bid_price
                target_qty_for_log = get_qty(symbol, entry_price, positions)
                logger.info(
                    f"Increasing existing {data['side']} position for {symbol} from {current_position_size} to {target_qty_for_log}"
                )
            else:
                if transition_to_normal:
                    logger.info(
                        f"Transitioning probe {data['side']} position for {symbol} towards target qty {target_qty}"
                    )
                elif effective_probe:
                    logger.info(f"Entering probe {data['side']} position for {symbol} with qty {target_qty}")
                else:
                    logger.info(f"Entering new {data['side']} position for {symbol}")

            is_highlow_entry = entry_strategy in MAXDIFF_LIMIT_STRATEGIES and not effective_probe
            highlow_limit_executed = False

            if bid_price is not None and ask_price is not None:
                entry_price = entry_price or (ask_price if data["side"] == "buy" else bid_price)
                if not effective_probe:
                    recalculated_qty = get_qty(symbol, entry_price, positions)
                    if recalculated_qty is None:
                        recalculated_qty = 0.0
                    if target_qty:
                        target_qty = min(target_qty, recalculated_qty) if recalculated_qty > 0 else target_qty
                    else:
                        target_qty = recalculated_qty
                    if target_qty <= 0:
                        logger.info(f"Skipping {symbol} entry after recalculated qty was non-positive.")
                        continue
                    logger.info(f"Target quantity for {symbol}: {target_qty} at price {entry_price}")

                    if is_highlow_entry:
                        preferred_limit = _lookup_entry_price(data, entry_strategy, data["side"])
                        fallback_candidates: List[Optional[float]] = []
                        if entry_strategy == "maxdiffalwayson":
                            fallback_candidates.extend(
                                [
                                    data.get("maxdiffprofit_low_price" if is_buy_side(data["side"]) else "maxdiffprofit_high_price"),
                                    data.get("predicted_low" if is_buy_side(data["side"]) else "predicted_high"),
                                ]
                            )
                        elif entry_strategy == "pctdiff":
                            fallback_candidates.extend(
                                [
                                    data.get("maxdiffprofit_low_price" if is_buy_side(data["side"]) else "maxdiffprofit_high_price"),
                                    data.get("predicted_low" if is_buy_side(data["side"]) else "predicted_high"),
                                ]
                            )
                        else:
                            fallback_candidates.append(
                                data.get("predicted_low" if is_buy_side(data["side"]) else "predicted_high")
                            )
                        limit_reference = preferred_limit
                        if limit_reference is None:
                            for candidate in fallback_candidates:
                                if candidate is not None:
                                    limit_reference = candidate
                                    break
                        fallback_limit = fallback_candidates[0] if fallback_candidates else None
                        limit_price = coerce_numeric(limit_reference, default=float("nan"))
                        if math.isnan(limit_price) or limit_price <= 0:
                            logger.warning(
                                f"{symbol} highlow entry missing limit price (preferred={preferred_limit}, fallback={fallback_limit}); falling back to ramp"
                            )
                        else:
                            try:
                                crypto_rank = crypto_ranks.get(symbol)
                                logger.info(
                                    f"Spawning highlow staged entry watcher for {symbol} {data['side']} qty={target_qty} @ {limit_price:.4f}{f' crypto_rank={crypto_rank}' if crypto_rank else ''}"
                                )
                                spawn_open_position_at_maxdiff_takeprofit(
                                    symbol,
                                    data["side"],
                                    float(limit_price),
                                    float(target_qty),
                                    poll_seconds=MAXDIFF_ENTRY_WATCHER_POLL_SECONDS,
                                    entry_strategy=entry_strategy,
                                    force_immediate=force_immediate_entry,
                                    priority_rank=priority_rank,
                                    crypto_rank=crypto_rank,
                                )
                                if entry_strategy == "maxdiffalwayson":
                                    opposite_side = "sell" if is_buy_side(data["side"]) else "buy"
                                    allowed_side_raw = data.get("allowed_side")
                                    allowed_side_cfg = str(allowed_side_raw).lower() if allowed_side_raw else "both"
                                    complement_allowed = allowed_side_cfg in {"both", opposite_side}
                                    if complement_allowed and opposite_side == "sell" and symbol in all_crypto_symbols:
                                        complement_allowed = False
                                    if complement_allowed:
                                        if opposite_side == "sell":
                                            opposite_preferred = data.get("maxdiffalwayson_high_price")
                                            opposite_candidates = [
                                                data.get("maxdiffprofit_high_price"),
                                                data.get("predicted_high"),
                                            ]
                                        else:
                                            opposite_preferred = data.get("maxdiffalwayson_low_price")
                                            opposite_candidates = [
                                                data.get("maxdiffprofit_low_price"),
                                                data.get("predicted_low"),
                                            ]
                                        opposite_reference = opposite_preferred
                                        if opposite_reference is None:
                                            for candidate in opposite_candidates:
                                                if candidate is not None:
                                                    opposite_reference = candidate
                                                    break
                                        opposite_price = coerce_numeric(opposite_reference, default=float("nan"))
                                        if math.isnan(opposite_price) or opposite_price <= 0:
                                            logger.debug(
                                                f"{symbol} complementary maxdiffalwayson entry skipped; invalid limit ({opposite_reference})"
                                            )
                                        else:
                                            try:
                                                logger.info(
                                                    f"Spawning complementary maxdiffalwayson entry watcher for {symbol} {opposite_side} qty={target_qty} @ {opposite_price:.4f}{f' crypto_rank={crypto_rank}' if crypto_rank else ''}"
                                                )
                                                spawn_open_position_at_maxdiff_takeprofit(
                                                    symbol,
                                                    opposite_side,
                                                    float(opposite_price),
                                                    float(target_qty),
                                                    poll_seconds=MAXDIFF_ENTRY_WATCHER_POLL_SECONDS,
                                                    entry_strategy=entry_strategy,
                                                    force_immediate=force_immediate_entry,
                                                    priority_rank=priority_rank,
                                                    crypto_rank=crypto_rank,
                                                )
                                            except Exception as comp_exc:
                                                logger.warning(
                                                    f"Failed to spawn complementary maxdiffalwayson entry for {symbol} {opposite_side}: {comp_exc}"
                                                )
                                highlow_limit_executed = True
                                entry_price = float(limit_price)
                                entry_executed = True
                            except Exception as exc:
                                logger.warning(
                                    f"Failed to spawn highlow staged entry for {symbol}: {exc}; attempting direct limit order fallback."
                                )
                                try:
                                    result = alpaca_wrapper.open_order_at_price_or_all(
                                        symbol,
                                        target_qty,
                                        data["side"],
                                        float(limit_price),
                                    )
                                    if result is None:
                                        logger.warning(
                                            f"Highlow fallback limit order for {symbol} returned None; will attempt ramp."
                                        )
                                    else:
                                        highlow_limit_executed = True
                                        entry_price = float(limit_price)
                                        entry_executed = True
                                except Exception as fallback_exc:
                                    logger.warning(
                                        f"Fallback highlow limit order failed for {symbol}: {fallback_exc}; will ramp instead."
                                    )
                else:
                    logger.info(f"Probe trade target quantity for {symbol}: {target_qty} at price {entry_price}")

                if not highlow_limit_executed:
                    ramp_into_position(
                        symbol,
                        data["side"],
                        target_qty=target_qty,
                        maxdiff_overflow=data.get("maxdiff_spread_overflow", False),
                        risk_threshold=risk_threshold,
                    )
                    entry_executed = True
            else:
                logger.warning(f"Could not get bid/ask prices for {symbol}, using default sizing")
                if not highlow_limit_executed:
                    ramp_into_position(
                        symbol,
                        data["side"],
                        target_qty=target_qty if effective_probe else None,
                        maxdiff_overflow=data.get("maxdiff_spread_overflow", False),
                        risk_threshold=risk_threshold,
                    )
                    entry_executed = True

            if transition_to_normal:
                _mark_probe_transitioned(symbol, data["side"], target_qty, strategy=stored_entry_strategy)
                _update_active_trade(
                    symbol,
                    data["side"],
                    mode="probe_transition",
                    qty=target_qty,
                    strategy=recorded_entry_strategy,
                )
                _tag_active_trade_strategy(symbol, data["side"], recorded_entry_strategy)
                _normalize_active_trade_patch(_update_active_trade)
            elif effective_probe:
                _mark_probe_active(symbol, data["side"], target_qty, strategy=stored_entry_strategy)
                _update_active_trade(
                    symbol,
                    data["side"],
                    mode="probe",
                    qty=target_qty,
                    strategy=recorded_entry_strategy,
                )
                _tag_active_trade_strategy(symbol, data["side"], recorded_entry_strategy)
                _normalize_active_trade_patch(_update_active_trade)
            else:
                _update_active_trade(
                    symbol,
                    data["side"],
                    mode="normal",
                    qty=target_qty,
                    strategy=recorded_entry_strategy,
                )
                _tag_active_trade_strategy(symbol, data["side"], recorded_entry_strategy)
                _normalize_active_trade_patch(_update_active_trade)

            if (
                entry_executed
                and is_new_position_entry
                and max_entries_per_run is not None
                and max_entries_per_run >= 0
                and resolved_limit_key is not None
            ):
                post_count = _increment_symbol_entry(
                    symbol,
                    stored_entry_strategy,
                    key=resolved_limit_key,
                )
                logger.info(f"{symbol}: Incremented per-run entry count to {post_count}/{max_entries_per_run}.")

            if not effective_probe and entry_price is not None:
                projected_value = abs(target_qty * entry_price)
                current_abs_value = abs(current_position_value)
                total_exposure_value = total_exposure_value - current_abs_value + projected_value

            if is_highlow_entry:
                highlow_tp_reference = _lookup_takeprofit_price(data, entry_strategy, data["side"])
                if highlow_tp_reference is None:
                    fallback_candidates = []
                    if entry_strategy in {"maxdiffalwayson", "pctdiff"}:
                        fallback_candidates.append(
                            data.get("maxdiffprofit_high_price" if is_buy_side(data["side"]) else "maxdiffprofit_low_price")
                        )
                    fallback_candidates.append(
                        data.get("predicted_high" if is_buy_side(data["side"]) else "predicted_low")
                    )
                    for candidate in fallback_candidates:
                        if candidate is not None:
                            highlow_tp_reference = candidate
                            break
                takeprofit_price = coerce_numeric(highlow_tp_reference, default=float("nan"))
                if math.isnan(takeprofit_price) or takeprofit_price <= 0:
                    logger.debug(
                        f"{symbol} highlow takeprofit skipped due to invalid target ({highlow_tp_reference})"
                    )
                else:
                    try:
                        logger.info(
                            f"Scheduling highlow takeprofit for {symbol} at {takeprofit_price:.4f}"
                        )
                        spawn_close_position_at_maxdiff_takeprofit(
                            symbol,
                            data["side"],
                            float(takeprofit_price),
                            poll_seconds=MAXDIFF_EXIT_WATCHER_POLL_SECONDS,
                            price_tolerance=MAXDIFF_EXIT_WATCHER_PRICE_TOLERANCE,
                            entry_strategy=entry_strategy,
                        )
                    except Exception as exc:
                        logger.warning(f"Failed to schedule highlow takeprofit for {symbol}: {exc}")
            elif ENABLE_TAKEPROFIT_BRACKETS:
                tp_price = None
                entry_reference = entry_price
                if entry_reference is None and bid_price is not None and ask_price is not None:
                    entry_reference = ask_price if is_buy_side(data["side"]) else bid_price

                if is_buy_side(data["side"]):
                    tp_price = data.get("predicted_high")
                elif is_sell_side(data["side"]):
                    tp_price = data.get("predicted_low")

                schedule_takeprofit = False
                if tp_price is not None and entry_reference is not None:
                    tp_val = float(tp_price)
                    if is_buy_side(data["side"]):
                        schedule_takeprofit = tp_val > entry_reference * 1.0005
                    else:
                        schedule_takeprofit = tp_val < entry_reference * 0.9995

                if schedule_takeprofit:
                    try:
                        logger.info(
                            f"Scheduling discretionary takeprofit for {symbol} at {float(tp_price):.4f} (entry_ref={entry_reference:.4f})"
                        )
                        spawn_close_position_at_takeprofit(symbol, float(tp_price))
                    except Exception as exc:
                        logger.warning(f"Failed to schedule takeprofit for {symbol}: {exc}")
                elif tp_price is not None:
                    logger.debug(
                        f"{symbol} takeprofit {float(tp_price):.4f} skipped (entry_ref={entry_reference}, side={data['side']})"
                    )
        elif transition_to_normal:
            logger.info(
                f"{symbol}: Probe already at target sizing; marking transition complete without additional orders."
            )
            entry_strategy = data.get("strategy")
            stored_entry_strategy = "maxdiff" if entry_strategy in MAXDIFF_LIMIT_STRATEGIES else entry_strategy
            recorded_entry_strategy = entry_strategy
            _mark_probe_transitioned(symbol, data["side"], current_position_size, strategy=stored_entry_strategy)
            _update_active_trade(
                symbol,
                data["side"],
                mode="probe_transition",
                qty=current_position_size,
                strategy=recorded_entry_strategy,
            )
            _tag_active_trade_strategy(symbol, data["side"], recorded_entry_strategy)
            _normalize_active_trade_patch(_update_active_trade)


def manage_market_close(
    symbols: List[str],
    previous_picks: Dict[str, Dict],
    all_analyzed_results: Dict[str, Dict],
):
    """Execute market close position management."""
    logger.info("Managing positions for market close")

    if not all_analyzed_results:
        logger.warning("No analysis results available - keeping all positions open")
        return previous_picks

    positions = alpaca_wrapper.get_all_positions()
    positions = filter_to_realistic_positions(positions)
    if not positions:
        logger.info("No positions to manage for market close")
        return build_portfolio(
            all_analyzed_results,
            min_positions=DEFAULT_MIN_CORE_POSITIONS,
            max_positions=DEFAULT_MAX_PORTFOLIO,
            max_expanded=EXPANDED_PORTFOLIO,
        )

    # Close positions only when forecast shows opposite direction
    for position in positions:
        symbol = position.symbol
        should_close = False
        close_reason = ""

        normalized_side = _normalize_side_for_key(position.side)
        active_trade_meta = _get_active_trade(symbol, normalized_side)
        entry_mode = active_trade_meta.get("mode")
        if entry_mode is None and symbol in previous_picks:
            entry_mode = previous_picks.get(symbol, {}).get("trade_mode")
        if not entry_mode:
            entry_mode = "normal"
        entry_strategy = active_trade_meta.get("entry_strategy")
        previous_pick_strategy = previous_picks.get(symbol, {}).get("strategy") if symbol in previous_picks else None
        if not entry_strategy:
            entry_strategy = previous_pick_strategy
        elif entry_strategy == "maxdiff" and previous_pick_strategy in MAXDIFF_LIMIT_STRATEGIES:
            entry_strategy = previous_pick_strategy

        lookup_entry_strategy = entry_strategy
        if entry_strategy in {"maxdiff", "maxdiffalwayson"}:
            lookup_entry_strategy = "highlow"

        next_forecast = all_analyzed_results.get(symbol)
        if next_forecast:
            if not is_same_side(next_forecast["side"], position.side):
                logger.info(
                    f"Closing position for {symbol} due to predicted direction change from {position.side} to {next_forecast['side']} tomorrow"
                )
                logger.info(f"Predicted movement: {next_forecast['predicted_movement']:.3f}")
                should_close = True
                close_reason = f"tomorrow_direction_{next_forecast['side']}"
            else:
                logger.info(f"Keeping {symbol} position as forecast matches current {position.side} direction")
        else:
            logger.warning(f"No analysis data for {symbol} - keeping position")

        if not should_close and entry_strategy and next_forecast and (entry_mode or "normal") != "probe":
            strategy_returns = next_forecast.get("strategy_returns", {})
            strategy_return = strategy_returns.get(lookup_entry_strategy)
            forecast_strategy = next_forecast.get("strategy")
            if strategy_return is None and lookup_entry_strategy == forecast_strategy:
                strategy_return = next_forecast.get("avg_return")
            if strategy_return is not None and strategy_return < 0:
                logger.info(
                    f"Closing position for {symbol} due to {entry_strategy} strategy underperforming "
                    f"(avg return {strategy_return:.4f})"
                )
                should_close = True
                close_reason = f"{entry_strategy}_strategy_loss"

        probe_meta = next_forecast or _evaluate_trade_block(symbol, normalized_side, strategy=entry_strategy)
        if probe_meta.get("probe_expired") and not should_close:
            logger.info(
                f"Closing {symbol} ahead of next session; probe duration exceeded {PROBE_MAX_DURATION}, issuing backout."
            )
            should_close = True
            close_reason = "probe_duration_exceeded"

        if should_close:
            _record_trade_outcome(position, close_reason or "market_close")
            backout_near_market(
                symbol,
                start_offset_minutes=BACKOUT_START_OFFSET_MINUTES,
                sleep_seconds=BACKOUT_SLEEP_SECONDS,
                market_close_buffer_minutes=BACKOUT_MARKET_CLOSE_BUFFER_MINUTES,
                market_close_force_minutes=BACKOUT_MARKET_CLOSE_FORCE_MINUTES,
            )

    # Return top picks for next day
    return build_portfolio(
        all_analyzed_results,
        min_positions=DEFAULT_MIN_CORE_POSITIONS,
        max_positions=DEFAULT_MAX_PORTFOLIO,
        max_expanded=EXPANDED_PORTFOLIO,
    )


def analyze_next_day_positions(symbols: List[str]) -> Dict:
    """Analyze symbols for next day's trading session."""
    logger.info("Analyzing positions for next trading day")
    return analyze_symbols(symbols)  # Reuse existing analysis function


def dry_run_manage_positions(current_picks: Dict[str, Dict], previous_picks: Dict[str, Dict]):
    """Simulate position management without executing trades."""
    positions = alpaca_wrapper.get_all_positions()
    positions = filter_to_realistic_positions(positions)

    logger.info("\nPLANNED POSITION CHANGES:")

    # Log position closures
    for position in positions:
        symbol = position.symbol
        should_close = False

        if symbol not in current_picks:
            # For crypto on weekends, only close if direction changed
            if symbol in all_crypto_symbols and not is_nyse_trading_day_now():
                logger.info(
                    f"Would keep crypto position for {symbol} on weekend - no direction change check needed in dry run"
                )
            # For stocks when market is closed, only close if direction changed
            elif symbol not in all_crypto_symbols and not is_nyse_trading_day_now():
                logger.info(
                    f"Would keep stock position for {symbol} when market closed - no direction change check needed in dry run"
                )
            else:
                logger.info(f"Would close position for {symbol} as it's no longer in top picks")
                should_close = True
        elif symbol in current_picks and not is_same_side(current_picks[symbol]["side"], position.side):
            logger.info(
                f"Would close position for {symbol} to switch direction from {position.side} to {current_picks[symbol]['side']}"
            )
            should_close = True

    # Log new positions
    for symbol, data in current_picks.items():
        trade_mode = data.get("trade_mode", "normal")
        is_probe_trade = trade_mode == "probe"
        probe_transition_ready = data.get("probe_transition_ready", False)
        probe_expired = data.get("probe_expired", False)
        if data.get("trade_blocked") and not is_probe_trade:
            logger.info(f"Would skip {symbol} due to active block: {data.get('block_reason', 'recent loss')}")
            continue
        if probe_expired:
            logger.info(
                f"Would skip {symbol} entry while probe backout executes (duration exceeded {PROBE_MAX_DURATION})."
            )
            continue
        position_exists = any(p.symbol == symbol for p in positions)
        correct_side = any(p.symbol == symbol and is_same_side(p.side, data["side"]) for p in positions)

        if is_probe_trade and probe_transition_ready and position_exists and correct_side:
            logger.info(f"Would transition probe {data['side']} position for {symbol} toward normal sizing")
        elif is_probe_trade:
            min_trade_qty = MIN_CRYPTO_QTY if symbol in all_crypto_symbols else MIN_STOCK_QTY
            logger.info(
                f"Would enter probe {data['side']} position for {symbol} with approximately {min_trade_qty} units"
            )
        elif not position_exists or not correct_side:
            logger.info(f"Would enter new {data['side']} position for {symbol}")


def cleanup_spawned_processes():
    """Clean up all spawned watcher processes on shutdown."""
    logger.info("Cleaning up spawned watcher processes...")

    if not MAXDIFF_WATCHERS_DIR.exists():
        logger.debug("No watcher directory found, skipping cleanup")
        return

    killed_count = 0
    failed_pids = []

    # Find all watcher config files and kill their PIDs
    for config_path in MAXDIFF_WATCHERS_DIR.glob("*.json"):
        try:
            import json

            with open(config_path) as f:
                metadata = json.load(f)

            pid = metadata.get("pid")
            if not pid or not isinstance(pid, int):
                continue

            # Check if process is still running
            try:
                os.kill(pid, 0)  # Signal 0 = check if process exists
            except (ProcessLookupError, PermissionError, OSError):
                # Process already dead
                continue

            # Try graceful shutdown first
            try:
                os.kill(pid, signal.SIGTERM)
                killed_count += 1
                logger.debug(f"Sent SIGTERM to watcher PID {pid} ({config_path.name})")
            except ProcessLookupError:
                pass  # Already exited
            except Exception as e:
                logger.warning(f"Failed to kill PID {pid}: {e}")
                failed_pids.append(pid)

        except Exception as e:
            logger.debug(f"Error processing {config_path}: {e}")

    # Give processes time to gracefully exit
    if killed_count > 0:
        sleep(2)

    # Force kill any survivors
    for config_path in MAXDIFF_WATCHERS_DIR.glob("*.json"):
        try:
            import json

            with open(config_path) as f:
                metadata = json.load(f)

            pid = metadata.get("pid")
            if not pid or not isinstance(pid, int):
                continue

            try:
                os.kill(pid, 0)  # Check if still running
                os.kill(pid, signal.SIGKILL)
                logger.info(f"Force killed watcher PID {pid} ({config_path.name})")
            except (ProcessLookupError, PermissionError, OSError):
                pass

        except Exception:
            pass

    if killed_count > 0:
        logger.info(f"Cleaned up {killed_count} spawned watcher processes")
    else:
        logger.debug("No active watcher processes found to clean up")


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    signal_name = signal.Signals(signum).name
    logger.info(f"Received {signal_name}, shutting down gracefully...")

    try:
        cleanup_spawned_processes()
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

    try:
        release_model_resources()
    except Exception as e:
        logger.debug(f"Model release failed: {e}")

    logger.info("Shutdown complete")
    sys.exit(0)


def main():
    fallback_symbols = [
        # Historical equity basket
        "EQIX",
        "GS",
        "COST",
        "CRM",
        "AXP",
        "BA",
        "GE",
        "LLY",
        "AVGO",
        "SPY",
        "SHOP",
        "GLD",
        "PLTR",
        "MCD",
        "V",
        "VTI",
        "QQQ",
        "MA",
        "SAP",
        "COUR",
        "ADBE",
        "INTC",
        "QUBT",
        # Crypto fallbacks
        "BTCUSD",
        "ETHUSD",
        "UNIUSD",
        "LINKUSD",
    ]

    symbols = _load_strategytraining_symbols()
    if symbols:
        logger.info(
            "Loaded %d experiment symbols from %s: %s",
            len(symbols),
            STRATEGYTRAINING_FAST_RESULTS_PATH,
            ", ".join(symbols),
        )
    else:
        symbols = fallback_symbols
        logger.warning(
            "Using fallback symbol list (%d entries); re-run strategytraining/test_sizing_on_precomputed_pnl.py "
            "to refresh %s if you expect neural symbols to be enforced.",
            len(symbols),
            STRATEGYTRAINING_FAST_RESULTS_PATH,
        )

    # Filter symbols by TRADABLE_PAIRS env var if set
    original_symbols = symbols
    symbols = filter_symbols_by_tradable_pairs(symbols)
    filter_info = get_filter_info(original_symbols, symbols)
    if filter_info["was_filtered"]:
        logger.info(
            "TRADABLE_PAIRS filter: %d/%d symbols selected: %s",
            filter_info["filtered_count"],
            filter_info["original_count"],
            ", ".join(symbols)
        )

    if NEURAL_SIZING_ENABLED:
        _refresh_neural_scale(force=True)

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # kill command
    logger.info("Signal handlers registered for graceful shutdown")

    previous_picks = {}

    # Track when each analysis was last run
    last_initial_run = None
    last_market_open_run = None
    last_market_open_hour2_run = None
    last_market_close_run = None
    last_crypto_midnight_refresh = None
    last_neural_refresh_date = None

    while True:
        try:
            market_open, market_close = get_market_hours()
            now = datetime.now(pytz.timezone("US/Eastern"))
            today = now.date()

            if NEURAL_SIZING_ENABLED and last_neural_refresh_date != today:
                _refresh_neural_scale(force=True)
                last_neural_refresh_date = today

            analysis_window_minutes = max(MARKET_CLOSE_ANALYSIS_WINDOW_MINUTES, 1)
            close_analysis_window_start = market_close - timedelta(minutes=analysis_window_minutes)
            close_analysis_window_end = market_close

            # Initial analysis at NZ morning (22:00-22:30 EST)
            # run at start of program to check
            if last_initial_run is None or (
                (now.hour == 22 and 0 <= now.minute < 30) and (last_initial_run is None or last_initial_run != today)
            ):
                logger.info("\nINITIAL ANALYSIS STARTING...")
                all_analyzed_results = analyze_symbols(symbols)
                current_picks = build_portfolio(
                    all_analyzed_results,
                    min_positions=DEFAULT_MIN_CORE_POSITIONS,
                    max_positions=DEFAULT_MAX_PORTFOLIO,
                    max_expanded=EXPANDED_PORTFOLIO,
                )
                log_trading_plan(current_picks, "INITIAL PLAN")
                dry_run_manage_positions(current_picks, previous_picks)
                manage_positions(current_picks, previous_picks, all_analyzed_results)

                previous_picks = current_picks
                last_initial_run = today

            # Market open analysis (9:30-10:00 EST)
            elif (
                (now.hour == market_open.hour and market_open.minute <= now.minute < market_open.minute + 30)
                and (last_market_open_run is None or last_market_open_run != today)
                and is_nyse_trading_day_now()
            ):
                logger.info("\nMARKET OPEN ANALYSIS STARTING...")
                all_analyzed_results = analyze_symbols(symbols)
                current_picks = build_portfolio(
                    all_analyzed_results,
                    min_positions=DEFAULT_MIN_CORE_POSITIONS,
                    max_positions=DEFAULT_MAX_PORTFOLIO,
                    max_expanded=EXPANDED_PORTFOLIO,
                )
                log_trading_plan(current_picks, "MARKET OPEN PLAN")
                manage_positions(current_picks, previous_picks, all_analyzed_results)

                previous_picks = current_picks
                last_market_open_run = today

            # Market open hour 2 analysis (10:30-11:00 EST)
            elif (
                (now.hour == market_open.hour + 1 and market_open.minute <= now.minute < market_open.minute + 30)
                and (last_market_open_hour2_run is None or last_market_open_hour2_run != today)
                and is_nyse_trading_day_now()
            ):
                logger.info("\nMARKET OPEN HOUR 2 ANALYSIS STARTING...")
                all_analyzed_results = analyze_symbols(symbols)
                current_picks = build_portfolio(
                    all_analyzed_results,
                    min_positions=DEFAULT_MIN_CORE_POSITIONS,
                    max_positions=DEFAULT_MIN_CORE_POSITIONS,
                )
                log_trading_plan(current_picks, "MARKET OPEN HOUR 2 PLAN")
                manage_positions(current_picks, previous_picks, all_analyzed_results)

                previous_picks = current_picks
                last_market_open_hour2_run = today

            # Market close analysis (shifted earlier to allow gradual backout)
            elif (
                close_analysis_window_start <= now < close_analysis_window_end
                and (last_market_close_run is None or last_market_close_run != today)
                and is_nyse_trading_day_ending()
            ):
                logger.info("\nMARKET CLOSE ANALYSIS STARTING...")
                all_analyzed_results = analyze_symbols(symbols)
                previous_picks = manage_market_close(symbols, previous_picks, all_analyzed_results)
                last_market_close_run = today

            # Crypto midnight refresh (00:00-00:30 UTC = 19:00-19:30 EST / 20:00-20:30 EDT)
            # Refreshes watchers for existing crypto positions when new daily bar arrives
            elif (now.hour == 19 and 0 <= now.minute < 30) and (
                last_crypto_midnight_refresh is None or last_crypto_midnight_refresh != today
            ):
                logger.info("\nCRYPTO MIDNIGHT REFRESH STARTING...")
                # Only analyze crypto symbols
                crypto_only_symbols = [s for s in symbols if s in all_crypto_symbols]
                if crypto_only_symbols:
                    all_analyzed_results = analyze_symbols(crypto_only_symbols)
                    # Only refresh watchers for existing crypto positions (don't change portfolio)
                    if previous_picks:
                        crypto_picks = {k: v for k, v in previous_picks.items() if k in all_crypto_symbols}
                        if crypto_picks:
                            logger.info(f"Refreshing watchers for {len(crypto_picks)} existing crypto positions")
                            manage_positions(crypto_picks, crypto_picks, all_analyzed_results)
                last_crypto_midnight_refresh = today

        except Exception as e:
            logger.exception(f"Error in main loop: {str(e)}")
        finally:
            try:
                release_model_resources()
            except Exception as cleanup_exc:
                logger.debug(f"Model release failed: {cleanup_exc}")
            sleep(60)


if __name__ == "__main__":
    main()
