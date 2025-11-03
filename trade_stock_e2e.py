import logging
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import sleep
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pytz
from loguru import logger

import alpaca_wrapper
try:
    from backtest_test3_inline import backtest_forecasts, release_model_resources
except Exception as import_exc:  # pragma: no cover - exercised via tests with stubs
    logging.getLogger(__name__).warning(
        "Falling back to stubbed backtest resources due to import failure: %s", import_exc
    )
    captured_import_error = import_exc

    def backtest_forecasts(*args, **kwargs):
        raise RuntimeError(
            "backtest_forecasts is unavailable because backtest_test3_inline could not be imported."
        ) from captured_import_error

    def release_model_resources() -> None:
        return None
from data_curate_daily import get_bid, get_ask, download_exchange_latest_data
from env_real import ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD
from jsonshelve import FlatShelf
from marketsimulator.state import get_state
from src.cache_utils import ensure_huggingface_cache_dir
from src.comparisons import is_buy_side, is_same_side, is_sell_side
from src.date_utils import is_nyse_trading_day_now, is_nyse_trading_day_ending
from src.fixtures import crypto_symbols, all_crypto_symbols, active_crypto_symbols
# Note: Use all_crypto_symbols for identification checks (fees, trading hours, etc.)
# Use active_crypto_symbols for deciding what to actively trade
from src.logging_utils import setup_logging
from src.trading_obj_utils import filter_to_realistic_positions
from src.process_utils import (
    backout_near_market,
    ramp_into_position,
    spawn_close_position_at_maxdiff_takeprofit,
    spawn_close_position_at_takeprofit,
    spawn_open_position_at_maxdiff_takeprofit,
)
from src.portfolio_risk import record_portfolio_snapshot
from src.sizing_utils import get_qty
from src.trade_stock_env_utils import (
    TRUTHY_ENV_VALUES,
    _allowed_side_for,
    _current_symbol_entry_count,
    _drawdown_cap_for,
    _drawdown_resume_for,
    _get_env_float,
    _load_trend_summary,
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
from src.trade_stock_utils import (
    agree_direction,
    coerce_optional_float,
    compute_spread_bps,
    edge_threshold_bps,
    evaluate_strategy_entry_gate,
    expected_cost_bps,
    kelly_lite,
    parse_float_list,
    resolve_spread_cap,
    should_rebalance,
)
from alpaca.data import StockHistoricalDataClient
import src.trade_stock_state_utils as state_utils
from src.trading_obj_utils import filter_to_realistic_positions
from stock.data_utils import coerce_numeric, ensure_lower_bound, safe_divide
from stock.state import ensure_state_dir as _shared_ensure_state_dir
from stock.state import get_state_dir, get_state_file, resolve_state_suffix

# Keep frequently patched helpers accessible for external callers.
_EXPORTED_ENV_HELPERS = (reset_symbol_entry_counters, get_entry_counter_snapshot)

# Configure logging
logger = setup_logging("trade_stock_e2e.log")

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
ENABLE_TAKEPROFIT_BRACKETS = os.getenv("ENABLE_TAKEPROFIT_BRACKETS", "0").strip().lower() in {"1", "true", "yes", "on"}
CONSENSUS_MIN_MOVE_PCT = float(os.getenv("CONSENSUS_MIN_MOVE_PCT", "0.001"))

_quote_client: Optional[StockHistoricalDataClient] = None
_COOLDOWN_STATE: Dict[str, Dict[str, datetime]] = {}

_trade_outcomes_store: Optional[FlatShelf] = None
_trade_learning_store: Optional[FlatShelf] = None
_active_trades_store: Optional[FlatShelf] = None
_trade_history_store: Optional[FlatShelf] = None
_maxdiff_plans_store: Optional[FlatShelf] = None

_TRUTHY = TRUTHY_ENV_VALUES

SIMPLIFIED_MODE = os.getenv("MARKETSIM_SIMPLE_MODE", "0").strip().lower() in _TRUTHY


def _coerce_positive_int(raw_value: Optional[str], default: int) -> int:
    if raw_value is None:
        return default
    try:
        parsed = int(str(raw_value).strip())
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


ENABLE_PROBE_TRADES = (
    os.getenv("MARKETSIM_ENABLE_PROBE_TRADES", "1").strip().lower() in _TRUTHY
)
MAX_MAXDIFFS = _coerce_positive_int(
    os.getenv("MARKETSIM_MAX_MAXDIFFS"),
    15,
)

DEFAULT_PROBE_SYMBOLS = {"AAPL", "MSFT", "NVDA"}
PROBE_SYMBOLS = (
    set()
    if SIMPLIFIED_MODE or not ENABLE_PROBE_TRADES
    else set(DEFAULT_PROBE_SYMBOLS)
)

_LATEST_FORECAST_CACHE: Dict[str, Dict[str, object]] = {}
_LATEST_FORECAST_PATH: Optional[Path] = None
DISABLE_TRADE_GATES = os.getenv("MARKETSIM_DISABLE_GATES", "0").strip().lower() in _TRUTHY

_coerce_optional_float = coerce_optional_float
_parse_float_list = parse_float_list
_edge_threshold_bps = edge_threshold_bps
_evaluate_strategy_entry_gate = evaluate_strategy_entry_gate

MAXDIFF_STRATEGIES = {"maxdiff", "maxdiffalwayson"}
MAXDIFF_LIMIT_STRATEGIES = MAXDIFF_STRATEGIES.union({"highlow"})


def _should_skip_closed_equity() -> bool:
    env_value = os.getenv("MARKETSIM_SKIP_CLOSED_EQUITY")
    if env_value is not None:
        return env_value.strip().lower() in _TRUTHY
    return True


def _get_trend_stat(symbol: str, key: str) -> Optional[float]:
    """Look up a trend summary metric for the provided symbol."""
    summary = _load_trend_summary()
    if not summary:
        return None
    symbol_info = summary.get((symbol or "").upper())
    if not symbol_info:
        return None
    value = symbol_info.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


_DRAW_SUSPENDED: Dict[Tuple[str, str], bool] = {}


def _strategy_key(symbol: Optional[str], strategy: Optional[str]) -> Tuple[str, str]:
    return ((symbol or "__global__").lower(), (strategy or "__default__").lower())


def _results_dir() -> Path:
    return Path(__file__).resolve().parent / "results"


def _normalize_series(series: pd.Series) -> pd.Series:
    return series.apply(lambda value: coerce_numeric(value, default=0.0, prefer="mean"))


def _find_latest_prediction_file() -> Optional[Path]:
    results_path = _results_dir()
    if not results_path.exists():
        return None
    candidates = list(results_path.glob("predictions-*.csv"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _load_latest_forecast_snapshot() -> Dict[str, Dict[str, object]]:
    global _LATEST_FORECAST_CACHE, _LATEST_FORECAST_PATH

    latest_file = _find_latest_prediction_file()
    if latest_file is None:
        return {}
    if _LATEST_FORECAST_PATH == latest_file and _LATEST_FORECAST_CACHE:
        return _LATEST_FORECAST_CACHE

    desired_columns = {
        "maxdiffprofit_profit",
        "maxdiffprofit_high_price",
        "maxdiffprofit_low_price",
        "maxdiffprofit_profit_high_multiplier",
        "maxdiffprofit_profit_low_multiplier",
        "maxdiffprofit_profit_values",
        "entry_takeprofit_profit",
        "entry_takeprofit_high_price",
        "entry_takeprofit_low_price",
        "entry_takeprofit_profit_values",
        "takeprofit_profit",
        "takeprofit_high_price",
        "takeprofit_low_price",
    }

    try:
        df = pd.read_csv(
            latest_file,
            usecols=lambda column: column == "instrument" or column in desired_columns,
        )
    except Exception as exc:  # pragma: no cover - guarded against missing pandas/corrupt files
        logger.warning("Failed to load latest prediction snapshot %s: %s", latest_file, exc)
        _LATEST_FORECAST_CACHE = {}
        _LATEST_FORECAST_PATH = latest_file
        return _LATEST_FORECAST_CACHE

    snapshot: Dict[str, Dict[str, object]] = {}

    for row in df.to_dict("records"):
        instrument = row.get("instrument")
        if not instrument:
            continue
        entry: Dict[str, object] = {}
        for key in desired_columns:
            if key not in row:
                continue
            if key.endswith("_values"):
                parsed_values = _parse_float_list(row.get(key))
                if parsed_values is not None:
                    entry[key] = parsed_values
            else:
                parsed_float = _coerce_optional_float(row.get(key))
                if parsed_float is not None:
                    entry[key] = parsed_float
        if entry:
            snapshot[str(instrument)] = entry

    _LATEST_FORECAST_CACHE = snapshot
    _LATEST_FORECAST_PATH = latest_file
    return snapshot


def _is_kronos_only_mode() -> bool:
    return os.getenv("MARKETSIM_FORCE_KRONOS", "0").lower() in _TRUTHY


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


def is_tradeable(
    symbol: str,
    bid: Optional[float],
    ask: Optional[float],
    *,
    avg_dollar_vol: Optional[float] = None,
    atr_pct: Optional[float] = None,
) -> Tuple[bool, str]:
    spread_bps = compute_spread_bps(bid, ask)
    if DISABLE_TRADE_GATES:
        return True, f"Gates disabled (spread {spread_bps:.1f}bps)"
    if math.isinf(spread_bps):
        return False, "Missing bid/ask quote"
    # Volume check disabled - strategy-specific blocks are sufficient
    # kronos_only = _is_kronos_only_mode()
    # min_dollar_vol = 5_000_000 if not kronos_only else 0.0
    # if avg_dollar_vol is not None and avg_dollar_vol < min_dollar_vol:
    #     return False, f"Low dollar vol {avg_dollar_vol:,.0f}"
    atr_note = f", ATR {atr_pct:.2f}%" if atr_pct is not None else ""
    return True, f"Spread {spread_bps:.1f}bps OK (gates relaxed{atr_note})"


def pass_edge_threshold(symbol: str, expected_move_pct: float) -> Tuple[bool, str]:
    move_bps = abs(expected_move_pct) * 1e4
    if DISABLE_TRADE_GATES:
        return True, f"Edge gating disabled ({move_bps:.1f}bps)"
    kronos_only = _is_kronos_only_mode()
    base_min = 40.0 if symbol.endswith("USD") else 15.0
    if kronos_only:
        base_min *= 0.6
    min_abs_move_bps = base_min
    buffer = 10.0 if not kronos_only else 5.0
    need = max(expected_cost_bps(symbol) + buffer, min_abs_move_bps)
    if move_bps < need:
        return False, f"Edge {move_bps:.1f}bps < need {need:.1f}bps"
    return True, f"Edge {move_bps:.1f}bps ≥ need {need:.1f}bps"


def resolve_signal_sign(move_pct: float) -> int:
    threshold = CONSENSUS_MIN_MOVE_PCT
    if _is_kronos_only_mode():
        threshold *= 0.25
    if abs(move_pct) < threshold:
        return 0
    return 1 if move_pct > 0 else -1


def _record_loss_timestamp(symbol: str, closed_at: Optional[str]) -> None:
    if not closed_at:
        return
    ts = _parse_timestamp(closed_at)
    if ts:
        _COOLDOWN_STATE[symbol] = {"last_stop_time": ts}


def clear_cooldown(symbol: str) -> None:
    _COOLDOWN_STATE.pop(symbol, None)


def can_trade_now(symbol: str, now: datetime, min_cooldown_minutes: int = PROBE_LOSS_COOLDOWN_MINUTES) -> bool:
    override_minutes = _symbol_min_cooldown_minutes(symbol)
    if override_minutes is not None and override_minutes >= 0:
        min_cooldown_minutes = float(override_minutes)
    state = _COOLDOWN_STATE.get(symbol)
    if not state:
        return True
    last_stop = state.get("last_stop_time")
    if isinstance(last_stop, datetime):
        delta = now - last_stop
        if delta.total_seconds() < min_cooldown_minutes * 60:
            return False
    return True


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
        day_key = datetime.now(timezone.utc).strftime('%Y-%m-%d')
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
        day_key = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        prefix = f"{day_key}:"
        plans = {}
        for key, value in store.items():
            if key.startswith(prefix):
                symbol = key[len(prefix):]
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
        day_key = datetime.now(timezone.utc).strftime('%Y-%m-%d')
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
MAXDIFF_ALWAYS_ON_PRIORITY_LIMIT = max(
    0, int(os.getenv("MAXDIFF_ALWAYS_ON_PRIORITY_LIMIT", "2"))
)


def _log_detail(message: str) -> None:
    if COMPACT_LOGS:
        logger.debug(message)
    else:
        logger.info(message)


def _format_metric_parts(parts):
    formatted = []
    for name, value, digits in parts:
        if value is None:
            continue
        try:
            formatted.append(f"{name}={value:.{digits}f}")
        except (TypeError, ValueError):
            continue
    return " ".join(formatted)


def _log_analysis_summary(symbol: str, data: Dict) -> None:
    status_parts = [
        f"{symbol} analysis",
        f"strategy={data.get('strategy')}",
        f"side={data.get('side')}",
        f"mode={data.get('trade_mode', 'normal')}",
        f"blocked={data.get('trade_blocked', False)}",
    ]
    strategy_returns = data.get("strategy_returns", {})
    returns_metrics = _format_metric_parts(
        [
            ("avg", data.get("avg_return"), 3),
            ("annual", data.get("annual_return"), 3),
            ("simple", data.get("simple_return"), 3),
            ("all", strategy_returns.get("all_signals"), 3),
            ("takeprofit", strategy_returns.get("takeprofit"), 3),
            ("highlow", strategy_returns.get("highlow"), 3),
            ("maxdiff", strategy_returns.get("maxdiff"), 3),
            ("ci_guard", strategy_returns.get("ci_guard"), 3),
            ("unprofit", data.get("unprofit_shutdown_return"), 3),
            ("composite", data.get("composite_score"), 3),
        ]
    )
    edges_metrics = _format_metric_parts(
        [
            ("move", data.get("predicted_movement"), 3),
            ("expected_pct", data.get("expected_move_pct"), 5),
            ("price_skill", data.get("price_skill"), 5),
            ("edge_strength", data.get("edge_strength"), 5),
            ("directional", data.get("directional_edge"), 5),
        ]
    )
    prices_metrics = _format_metric_parts(
        [
            ("pred_close", data.get("predicted_close"), 3),
            ("pred_high", data.get("predicted_high"), 3),
            ("pred_low", data.get("predicted_low"), 3),
            ("last_close", data.get("last_close"), 3),
        ]
    )
    walk_forward_notes = data.get("walk_forward_notes")
    summary_parts = [
        " ".join(status_parts),
        f"returns[{returns_metrics or '-'}]",
        f"edges[{edges_metrics or '-'}]",
        f"prices[{prices_metrics or '-'}]",
    ]
    if data.get("trade_blocked") and data.get("block_reason"):
        summary_parts.append(f"block_reason={data['block_reason']}")
    if walk_forward_notes:
        summary_parts.append("walk_forward_notes=" + "; ".join(str(note) for note in walk_forward_notes))

    probe_summary = None
    if data.get("trade_mode") == "probe":
        probe_notes = []
        if data.get("pending_probe"):
            probe_notes.append("pending")
        if data.get("probe_active"):
            probe_notes.append("active")
        if data.get("probe_transition_ready"):
            probe_notes.append("transition-ready")
        if data.get("probe_expired"):
            probe_notes.append("expired")
        if data.get("probe_age_seconds") is not None:
            try:
                probe_notes.append(f"age={int(data['probe_age_seconds'])}s")
            except (TypeError, ValueError):
                probe_notes.append(f"age={data['probe_age_seconds']}")
        probe_time_info = []
        if data.get("probe_started_at"):
            probe_time_info.append(f"start={data['probe_started_at']}")
        if data.get("probe_expires_at"):
            probe_time_info.append(f"expires={data['probe_expires_at']}")
        if probe_time_info:
            probe_notes.extend(probe_time_info)
        if probe_notes:
            probe_summary = "probe=" + ",".join(str(note) for note in probe_notes)
            summary_parts.append(probe_summary)

    compact_message = " | ".join(summary_parts)
    if COMPACT_LOGS:
        _log_detail(compact_message)
        return

    detail_lines = [" ".join(status_parts)]
    detail_lines.append(f"  returns: {returns_metrics or '-'}")
    detail_lines.append(f"  edges: {edges_metrics or '-'}")
    detail_lines.append(f"  prices: {prices_metrics or '-'}")

    walk_forward_metrics = _format_metric_parts(
        [
            ("oos", data.get("walk_forward_oos_sharpe"), 2),
            ("turnover", data.get("walk_forward_turnover"), 2),
            ("highlow", data.get("walk_forward_highlow_sharpe"), 2),
            ("takeprofit", data.get("walk_forward_takeprofit_sharpe"), 2),
            ("maxdiff", data.get("walk_forward_maxdiff_sharpe"), 2),
        ]
    )
    if walk_forward_metrics:
        detail_lines.append(f"  walk_forward: {walk_forward_metrics}")

    block_reason = data.get("block_reason")
    if data.get("trade_blocked") and block_reason:
        detail_lines.append(f"  block_reason: {block_reason}")

    if walk_forward_notes:
        detail_lines.append("  walk_forward_notes: " + "; ".join(str(note) for note in walk_forward_notes))

    if probe_summary:
        detail_lines.append("  " + probe_summary.replace("=", ": ", 1))

    _log_detail("\n".join(detail_lines))


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
        "%s: Position notional $%.2f exceeds probe limit $%.2f; promoting to normal regime.",
        position.symbol,
        notional_value,
        PROBE_NOTIONAL_LIMIT,
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
    try:
        unrealized_pl = float(getattr(position, "unrealized_pl", 0.0) or 0.0)
    except Exception:
        unrealized_pl = 0.0

    if unrealized_pl >= LIVE_DRAWDOWN_TRIGGER:
        return

    symbol = position.symbol
    normalized_side = _normalize_side_for_key(getattr(position, "side", ""))
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
    record = {
        "symbol": position.symbol,
        "side": normalized_side,
        "qty": qty_value,
        "pnl": pnl_value,
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
            history_key = key
            history = history_store.get(history_key, [])
            history.append(
                {
                    "symbol": position.symbol,
                    "side": normalized_side,
                    "qty": qty_value,
                    "pnl": pnl_value,
                    "closed_at": record["closed_at"],
                    "reason": reason,
                    "mode": trade_mode,
                    "entry_strategy": entry_strategy,
                }
            )
            history_store[history_key] = history[-100:]


def _evaluate_trade_block(symbol: str, side: str, strategy: Optional[str] = None) -> Dict[str, Optional[object]]:
    """Evaluate trade blocking for a specific symbol, side, and optionally strategy.

    When strategy is provided, blocks are strategy-specific (e.g., ETHUSD-buy-maxdiff can be
    blocked independently from ETHUSD-buy-highlow).
    """
    record = _load_trade_outcome(symbol, side, strategy=strategy)
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
    if probe_summary.get("probe_expired"):
        block_reason = block_reason or (
            f"Probe duration exceeded {PROBE_MAX_DURATION} for {symbol} {side}; scheduling backout"
        )
    cooldown_expires = None
    if last_closed_at is not None:
        cooldown_expires = (last_closed_at + LOSS_BLOCK_COOLDOWN).isoformat()
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


def analyze_symbols(symbols: List[str]) -> Dict:
    """Run backtest analysis on symbols and return results sorted by average return."""
    results = {}
    equities_tradable_now = is_nyse_trading_day_now()
    skip_closed_equity = _should_skip_closed_equity()
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

    kronos_only_mode = _is_kronos_only_mode()

    latest_snapshot = _load_latest_forecast_snapshot()

    for symbol in symbols:
        if symbol not in all_crypto_symbols and not equities_tradable_now:
            if skip_closed_equity:
                skipped_equity_symbols.append(symbol)
                continue
            logger.debug(
                "%s: market closed but analyzing due to MARKETSIM_SKIP_CLOSED_EQUITY override.",
                symbol,
            )
        try:
            kelly_fraction = None
            # not many because we need to adapt strats? eg the wierd spikes in uniusd are a big opportunity to trade w high/low
            # but then i bumped up because its not going to say buy crypto when its down, if its most recent based?
            num_simulations = env_simulations or 70
            used_fallback_engine = False

            try:
                backtest_df = backtest_forecasts(symbol, num_simulations)
            except Exception as exc:
                logger.warning(
                    f"Primary backtest_forecasts failed for {symbol}: {exc}. Attempting simulator fallback analytics."
                )
                try:
                    from marketsimulator import backtest_test3_inline as sim_backtest  # type: ignore

                    backtest_df = sim_backtest.backtest_forecasts(symbol, num_simulations)
                except Exception as fallback_exc:
                    logger.error(f"Fallback backtest also failed for {symbol}: {fallback_exc}. Skipping symbol.")
                    continue
                used_fallback_engine = True

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
            trading_days_per_year = 365 if symbol in all_crypto_symbols else 252

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
            }
            strategy_returns_annual = {
                "simple": _mean_return("simple_strategy_annual_return", "simple_strategy_return"),
                "all_signals": _mean_return("all_signals_strategy_annual_return", "all_signals_strategy_return"),
                "takeprofit": _mean_return("entry_takeprofit_annual_return", "entry_takeprofit_return"),
                "highlow": _mean_return("highlow_annual_return", "highlow_return"),
                "maxdiff": _mean_return("maxdiff_annual_return", "maxdiff_return"),
                "maxdiffalwayson": _mean_return("maxdiffalwayson_annual_return", "maxdiffalwayson_return"),
            }
            if "ci_guard_return" in backtest_df.columns:
                strategy_returns_daily["ci_guard"] = _mean_return(
                    "ci_guard_avg_daily_return",
                    "ci_guard_return",
                )
                strategy_returns_annual["ci_guard"] = _mean_return(
                    "ci_guard_annual_return",
                    "ci_guard_return",
                )
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
            }
            if "ci_guard" in strategy_returns:
                _strategy_series_map["ci_guard"] = ("ci_guard_avg_daily_return", "ci_guard_return")

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
            last_prediction = raw_last_prediction.apply(
                lambda value: coerce_numeric(value, default=0.0, prefer="mean")
            )
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

            def _optional_numeric(value: object) -> Optional[float]:
                raw = coerce_numeric(value, default=float("nan")) if value is not None else float("nan")
                return raw if math.isfinite(raw) else None

            maxdiff_high_price = _optional_numeric(last_prediction.get("maxdiffprofit_high_price"))
            maxdiff_low_price = _optional_numeric(last_prediction.get("maxdiffprofit_low_price"))
            maxdiff_trade_bias = _optional_numeric(last_prediction.get("maxdiff_trade_bias"))
            maxdiffalwayson_high_price = _optional_numeric(last_prediction.get("maxdiffalwayson_high_price"))
            maxdiffalwayson_low_price = _optional_numeric(last_prediction.get("maxdiffalwayson_low_price"))
            maxdiff_primary_side_raw = raw_last_prediction.get("maxdiff_primary_side")
            maxdiff_primary_side = (
                str(maxdiff_primary_side_raw).strip().lower()
                if maxdiff_primary_side_raw is not None
                else None
            )
            if maxdiff_primary_side == "":
                maxdiff_primary_side = None

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
            if maxdiff_primary_side:
                bias_fragment = maxdiff_primary_side
                if maxdiff_trade_bias is not None and math.isfinite(maxdiff_trade_bias):
                    bias_fragment = f"{bias_fragment}({maxdiff_trade_bias:+.3f})"
                snapshot_parts.append(f"maxdiff_side={bias_fragment}")
            _log_detail(" ".join(snapshot_parts))

            strategy_stats: Dict[str, Dict[str, float]] = {
                "simple": {
                    "avg_return": strategy_returns.get("simple", 0.0),
                    "annual_return": strategy_returns_annual.get("simple", 0.0),
                    "sharpe": _mean_column("simple_strategy_sharpe"),
                    "turnover": _mean_column("simple_strategy_turnover"),
                    "max_drawdown": _mean_column("simple_strategy_max_drawdown"),
                },
                "all_signals": {
                    "avg_return": strategy_returns.get("all_signals", 0.0),
                    "annual_return": strategy_returns_annual.get("all_signals", 0.0),
                    "sharpe": _mean_column("all_signals_strategy_sharpe"),
                    "turnover": _mean_column("all_signals_strategy_turnover"),
                    "max_drawdown": _mean_column("all_signals_strategy_max_drawdown"),
                },
                "takeprofit": {
                    "avg_return": strategy_returns.get("takeprofit", 0.0),
                    "annual_return": strategy_returns_annual.get("takeprofit", 0.0),
                    "sharpe": _mean_column("entry_takeprofit_sharpe"),
                    "turnover": _mean_column("entry_takeprofit_turnover"),
                    "max_drawdown": _mean_column("entry_takeprofit_max_drawdown"),
                },
                "highlow": {
                    "avg_return": strategy_returns.get("highlow", 0.0),
                    "annual_return": strategy_returns_annual.get("highlow", 0.0),
                    "sharpe": _mean_column("highlow_sharpe"),
                    "turnover": _mean_column("highlow_turnover"),
                    "max_drawdown": _mean_column("highlow_max_drawdown"),
                },
                "maxdiff": {
                    "avg_return": strategy_returns.get("maxdiff", 0.0),
                    "annual_return": strategy_returns_annual.get("maxdiff", 0.0),
                    "sharpe": _mean_column("maxdiff_sharpe"),
                    "turnover": _mean_column("maxdiff_turnover"),
                    "max_drawdown": _mean_column("maxdiff_max_drawdown"),
                },
                "maxdiffalwayson": {
                    "avg_return": strategy_returns.get("maxdiffalwayson", 0.0),
                    "annual_return": strategy_returns_annual.get("maxdiffalwayson", 0.0),
                    "sharpe": _mean_column("maxdiffalwayson_sharpe"),
                    "turnover": _mean_column("maxdiffalwayson_turnover"),
                    "max_drawdown": _mean_column("maxdiffalwayson_max_drawdown"),
                },
            }
            if "ci_guard" in strategy_returns:
                strategy_stats["ci_guard"] = {
                    "avg_return": strategy_returns.get("ci_guard", 0.0),
                    "annual_return": strategy_returns_annual.get("ci_guard", 0.0),
                    "sharpe": _mean_column("ci_guard_sharpe"),
                    "turnover": _mean_column("ci_guard_turnover"),
                    "max_drawdown": _mean_column("ci_guard_max_drawdown"),
                }

            for strat_name, (primary_col, fallback_col) in _strategy_series_map.items():
                strategy_recent_sums[strat_name] = _recent_return_sum(primary_col, fallback_col)

            strategy_ineligible: Dict[str, str] = {}
            candidate_avg_returns: Dict[str, float] = {}
            allowed_side = _allowed_side_for(symbol)
            symbol_is_crypto = symbol in all_crypto_symbols

            for name, stats in strategy_stats.items():
                if name not in strategy_returns:
                    continue
                allow_config = True
                if name == "takeprofit":
                    allow_config = ALLOW_TAKEPROFIT_ENTRY
                elif name == "highlow":
                    allow_config = ALLOW_HIGHLOW_ENTRY
                elif name == "maxdiff":
                    allow_config = ALLOW_MAXDIFF_ENTRY
                elif name == "maxdiffalwayson":
                    allow_config = ALLOW_MAXDIFF_ALWAYS_ENTRY

                if name in {"takeprofit", "highlow", "maxdiff", "maxdiffalwayson"}:
                    if not allow_config:
                        strategy_ineligible[name] = "disabled_by_config"
                        continue
                    eligible, reason = _evaluate_strategy_entry_gate(
                        symbol,
                        stats,
                        fallback_used=used_fallback_engine,
                        sample_size=sample_size,
                    )
                    if not eligible:
                        strategy_ineligible[name] = reason
                        continue

                avg_metric = _metric(stats.get("avg_return"), default=0.0)
                candidate_avg_returns[name] = avg_metric

            # Sort strategies by avg_return (simplified)
            ordered_strategies: List[str] = []
            if candidate_avg_returns:
                ordered_strategies = [
                    name for name, _ in sorted(
                        candidate_avg_returns.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                ]
            else:
                ordered_strategies = ["simple"]

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
                    selection_notes.append(f"{candidate_name}=ineligible({strategy_ineligible[candidate_name]})")
                    continue

                candidate_avg_return = candidate_avg_returns.get(candidate_name, 0.0)

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
            ci_guard_return = strategy_returns.get("ci_guard", 0.0)
            takeprofit_return = strategy_returns.get("takeprofit", 0.0)
            highlow_return = strategy_returns.get("highlow", 0.0)
            maxdiff_return = strategy_returns.get("maxdiff", 0.0)
            maxdiffalwayson_return = strategy_returns.get("maxdiffalwayson", 0.0)
            simple_sharpe = 0.0
            if "simple_strategy_sharpe" in backtest_df.columns:
                simple_sharpe = coerce_numeric(backtest_df["simple_strategy_sharpe"].mean(), default=0.0)
            ci_guard_sharpe = 0.0
            if "ci_guard_sharpe" in backtest_df.columns:
                ci_guard_sharpe = coerce_numeric(backtest_df["ci_guard_sharpe"].mean(), default=0.0)
            kronos_profit_raw = last_prediction.get("closemin_loss_trading_profit")
            kronos_profit = coerce_numeric(kronos_profit_raw) if kronos_profit_raw is not None else 0.0
            if _is_kronos_only_mode():
                if kronos_profit > simple_return:
                    simple_return = kronos_profit
                if kronos_profit > avg_return:
                    avg_return = kronos_profit
                kronos_annual = kronos_profit * trading_days_per_year
                if kronos_annual > annual_return:
                    annual_return = kronos_annual
            core_return = max(simple_return, ci_guard_return, 0.0)
            core_sharpe = max(simple_sharpe, ci_guard_sharpe, 0.0)
            price_skill = core_return + 0.25 * core_sharpe + 0.15 * max(kronos_profit, 0.0)
            highlow_allowed_entry = ALLOW_HIGHLOW_ENTRY and ("highlow" not in strategy_ineligible)
            takeprofit_allowed_entry = ALLOW_TAKEPROFIT_ENTRY and ("takeprofit" not in strategy_ineligible)
            maxdiff_allowed_entry = ALLOW_MAXDIFF_ENTRY and ("maxdiff" not in strategy_ineligible)
            maxdiffalwayson_allowed_entry = ALLOW_MAXDIFF_ALWAYS_ENTRY and ("maxdiffalwayson" not in strategy_ineligible)

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
                ci_guard_return,
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
            is_maxdiff = (best_strategy in MAXDIFF_STRATEGIES)

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
                    gating_reasons.append(
                        f"Recent {best_strategy} returns sum {recent_sum:.4f} <= 0"
                    )

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
                "ci_guard_return": _metric(ci_guard_return, default=0.0),
                "ci_guard_sharpe": _metric(ci_guard_sharpe, default=0.0),
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
                "strategy_candidate_avg_returns": candidate_avg_returns,
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
            if selection_notes:
                result_row["strategy_selection_notes"] = selection_notes
            if ordered_strategies:
                result_row["strategy_sequence"] = ordered_strategies
            snapshot_row = latest_snapshot.get(symbol)
            if snapshot_row:
                result_row.update(snapshot_row)

            if maxdiff_primary_side_raw is not None:
                result_row["maxdiff_primary_side"] = str(maxdiff_primary_side_raw).strip().lower() or "neutral"
            if maxdiff_trade_bias is not None:
                result_row["maxdiff_trade_bias"] = _metric(maxdiff_trade_bias, default=0.0)

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
                    result_row[count_key] = int(
                        round(coerce_numeric(last_prediction.get(count_key), default=0.0))
                    )
            if "maxdiffprofit_profit_values" in last_prediction:
                result_row["maxdiffprofit_profit_values"] = last_prediction.get("maxdiffprofit_profit_values")

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
                    result_row[count_key] = int(
                        round(coerce_numeric(last_prediction.get(count_key), default=0.0))
                    )
            if "maxdiffalwayson_profit_values" in last_prediction:
                result_row["maxdiffalwayson_profit_values"] = last_prediction.get("maxdiffalwayson_profit_values")
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
                    "avg_return": maxdiffalwayson_return,
                    "status": "identified",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "close_price": close_price,
                    "bid_price": bid_price,
                    "ask_price": ask_price,
                    "strategy": "maxdiffalwayson",
                }
                _save_maxdiff_plan(symbol, maxdiff_plan)

        except Exception:
            logger.exception("Error analyzing %s", symbol)
            continue

    if skipped_equity_symbols:
        logger.debug(
            "Skipping equity backtests while market closed: %s",
            ", ".join(sorted(skipped_equity_symbols)),
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
            avg_val = _coerce_optional_float(data.get("avg_return"))
            if avg_val is None or avg_val <= 0:
                continue
            pred_move = _coerce_optional_float(data.get("predicted_movement"))
            side = (data.get("side") or "").lower()
            if pred_move is not None:
                if side == "buy" and pred_move <= 0:
                    continue
                if side == "sell" and pred_move >= 0:
                    continue
            if data.get("trade_blocked"):
                continue
            simple_picks[symbol] = data
            if len(simple_picks) >= limit:
                break
        return simple_picks

    sorted_by_composite = sorted(all_results.items(), key=lambda item: item[1].get("composite_score", 0), reverse=True)

    picks: Dict[str, Dict] = {}

    # Core picks prioritise consistently profitable strategies.
    for symbol, data in sorted_by_composite:
        if len(picks) >= max_positions:
            break
        if data.get("trade_blocked"):
            continue
        if (
            data.get("avg_return", 0) > 0
            and data.get("unprofit_shutdown_return", 0) > 0
            and data.get("simple_return", 0) > 0
        ):
            picks[symbol] = data

    # Ensure we reach the minimum desired portfolio size using best remaining composites.
    if len(picks) < min_positions:
        for symbol, data in sorted_by_composite:
            if len(picks) >= max_positions:
                break
            if symbol in picks or data.get("trade_blocked"):
                continue
            if data.get("simple_return", 0) > 0 or data.get("composite_score", 0) > 0:
                picks[symbol] = data

    # Optionally expand with high-price-edge opportunities to keep broader exposure.
    if max_expanded and len(picks) < max_expanded:
        sorted_by_edge = sorted(
            (
                (symbol, data)
                for symbol, data in all_results.items()
                if symbol not in picks and not data.get("trade_blocked")
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
            picks[symbol] = data

    # Ensure probe-mode symbols are represented even if they fell outside the ranking filters.
    if ENABLE_PROBE_TRADES:
        probe_candidates = [
            (symbol, data)
            for symbol, data in all_results.items()
            if data.get("trade_mode") == "probe"
        ]
        for symbol, data in probe_candidates:
            if symbol in picks:
                continue
            if max_expanded and len(picks) < max_expanded:
                picks[symbol] = data
            elif len(picks) < max_positions:
                picks[symbol] = data
            else:
                # Replace the weakest pick to guarantee probe follow-up.
                weakest_symbol, _ = min(
                    picks.items(), key=lambda item: item[1].get("composite_score", float("-inf"))
                )
                picks.pop(weakest_symbol, None)
                picks[symbol] = data

    return picks


def log_trading_plan(picks: Dict[str, Dict], action: str):
    """Log the trading plan without executing trades."""
    if not picks:
        logger.info(f"TRADING PLAN ({action}) - no candidates")
        return
    compact_lines = [_format_plan_line(symbol, data) for symbol, data in picks.items()]
    logger.info("TRADING PLAN (%s) count=%d | %s", action, len(picks), " ; ".join(compact_lines))


def manage_positions(
    current_picks: Dict[str, Dict],
    previous_picks: Dict[str, Dict],
    all_analyzed_results: Dict[str, Dict],
):
    """Execute actual position management."""
    positions = alpaca_wrapper.get_all_positions()
    positions = filter_to_realistic_positions(positions)
    logger.info("EXECUTING POSITION CHANGES:")

    total_exposure_value = _calculate_total_exposure_value(positions)

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
    if day_pl_value is not None:
        snapshot_kwargs["day_pl"] = day_pl_value
    try:
        snapshot = record_portfolio_snapshot(total_exposure_value, **snapshot_kwargs)
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
    always_on_forced_symbols = {
        symbol for symbol, _ in always_on_candidates[:MAXDIFF_ALWAYS_ON_PRIORITY_LIMIT]
    }

    for symbol, original_data in current_picks.items():
        data = dict(original_data)
        current_picks[symbol] = data
        is_maxdiff_strategy = (data.get("strategy") in MAXDIFF_LIMIT_STRATEGIES)
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
                force_probe = _symbol_force_probe(symbol)
                if force_probe and data.get("trade_mode") != "probe":
                    data["trade_mode"] = "probe"
                    current_picks[symbol] = data
                    logger.info(f"{symbol}: Forcing probe mode via MARKETSIM_SYMBOL_FORCE_PROBE_MAP.")
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
                        "Trend PnL stat unavailable for %s; skipping trend-based suspension check.",
                        symbol,
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
            computed_qty = get_qty(symbol, entry_price, positions)
            if computed_qty is None:
                computed_qty = 0.0
            if effective_probe:
                target_qty = ensure_lower_bound(min_trade_qty, 0.0, default=min_trade_qty)
                logger.info(f"{symbol}: Probe sizing fixed at minimum tradable quantity {target_qty}")
                should_enter = not position_exists or not correct_side
                needs_size_increase = False
            else:
                base_qty = computed_qty
                # MaxDiff-family strategies use full sizing like in backtest, not Kelly
                if data.get("strategy") in MAXDIFF_STRATEGIES:
                    kelly_value = 1.0
                    logger.info(
                        f"{symbol}: {data.get('strategy')} using full position size (no Kelly scaling)"
                    )
                else:
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

                    # For crypto and non-leveraged strategies, shrink position instead of skipping
                    is_crypto = symbol in all_crypto_symbols
                    is_non_leveraged_strategy = data.get("strategy") in MAXDIFF_STRATEGIES

                    if allowed_value <= 0:
                        if is_crypto or is_non_leveraged_strategy:
                            # Use minimum trade size to still participate
                            logger.info(
                                f"Shrinking {symbol} to minimum trade size due to max exposure "
                                f"({projected_pct:.1f}% > {MAX_TOTAL_EXPOSURE_PCT:.1f}%). "
                                f"Crypto/non-leveraged strategies can't leverage anyway."
                            )
                            adjusted_qty = min_trade_qty
                            target_qty = adjusted_qty
                            projected_value = abs(target_qty * entry_price)
                            new_total_value = total_exposure_value - current_abs_value + projected_value
                        else:
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
                            if is_crypto or is_non_leveraged_strategy:
                                # For crypto/non-leveraged, use minimum size even when calculation gives 0
                                logger.info(
                                    f"Exposure adjustment for {symbol} gave non-positive qty; "
                                    f"using minimum trade size instead (crypto/non-leveraged)."
                                )
                                adjusted_qty = min_trade_qty
                            else:
                                logger.info(f"Skipping {symbol} entry after exposure adjustment resulted in non-positive qty.")
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
                        fallback_candidates: List[Optional[float]]
                        if is_buy_side(data["side"]):
                            if entry_strategy == "maxdiffalwayson":
                                preferred_limit = data.get("maxdiffalwayson_low_price")
                                fallback_candidates = [
                                    data.get("maxdiffprofit_low_price"),
                                    data.get("predicted_low"),
                                ]
                            else:
                                preferred_limit = data.get("maxdiffprofit_low_price")
                                fallback_candidates = [data.get("predicted_low")]
                        else:
                            if entry_strategy == "maxdiffalwayson":
                                preferred_limit = data.get("maxdiffalwayson_high_price")
                                fallback_candidates = [
                                    data.get("maxdiffprofit_high_price"),
                                    data.get("predicted_high"),
                                ]
                            else:
                                preferred_limit = data.get("maxdiffprofit_high_price")
                                fallback_candidates = [data.get("predicted_high")]
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
                                "%s highlow entry missing limit price (preferred=%s, fallback=%s); falling back to ramp",
                                symbol,
                                preferred_limit,
                                fallback_limit,
                            )
                        else:
                            try:
                                logger.info(
                                    "Spawning highlow staged entry watcher for %s %s qty=%s @ %.4f",
                                    symbol,
                                    data["side"],
                                    target_qty,
                                    limit_price,
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
                                )
                                if entry_strategy == "maxdiffalwayson":
                                    opposite_side = "sell" if is_buy_side(data["side"]) else "buy"
                                    allowed_side_raw = data.get("allowed_side")
                                    allowed_side_cfg = (str(allowed_side_raw).lower() if allowed_side_raw else "both")
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
                                                "%s complementary maxdiffalwayson entry skipped; invalid limit (%s)",
                                                symbol,
                                                opposite_reference,
                                            )
                                        else:
                                            try:
                                                logger.info(
                                                    "Spawning complementary maxdiffalwayson entry watcher for %s %s qty=%s @ %.4f",
                                                    symbol,
                                                    opposite_side,
                                                    target_qty,
                                                    opposite_price,
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
                                                )
                                            except Exception as comp_exc:
                                                logger.warning(
                                                    "Failed to spawn complementary maxdiffalwayson entry for %s %s: %s",
                                                    symbol,
                                                    opposite_side,
                                                    comp_exc,
                                                )
                                highlow_limit_executed = True
                                entry_price = float(limit_price)
                                entry_executed = True
                            except Exception as exc:
                                logger.warning(
                                    "Failed to spawn highlow staged entry for %s: %s; attempting direct limit order fallback.",
                                    symbol,
                                    exc,
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
                                            "Highlow fallback limit order for %s returned None; will attempt ramp.",
                                            symbol,
                                        )
                                    else:
                                        highlow_limit_executed = True
                                        entry_price = float(limit_price)
                                        entry_executed = True
                                except Exception as fallback_exc:
                                    logger.warning(
                                        "Fallback highlow limit order failed for %s: %s; will ramp instead.",
                                        symbol,
                                        fallback_exc,
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
                    strategy=stored_entry_strategy,
                )
                _tag_active_trade_strategy(symbol, data["side"], stored_entry_strategy)
                _normalize_active_trade_patch(_update_active_trade)
            elif effective_probe:
                _mark_probe_active(symbol, data["side"], target_qty, strategy=stored_entry_strategy)
                _update_active_trade(
                    symbol,
                    data["side"],
                    mode="probe",
                    qty=target_qty,
                    strategy=stored_entry_strategy,
                )
                _tag_active_trade_strategy(symbol, data["side"], stored_entry_strategy)
                _normalize_active_trade_patch(_update_active_trade)
            else:
                _update_active_trade(
                    symbol,
                    data["side"],
                    mode="normal",
                    qty=target_qty,
                    strategy=stored_entry_strategy,
                )
                _tag_active_trade_strategy(symbol, data["side"], stored_entry_strategy)
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
                if is_buy_side(data["side"]):
                    highlow_tp_reference = (
                        data.get("maxdiffalwayson_high_price")
                        if entry_strategy == "maxdiffalwayson"
                        else data.get("maxdiffprofit_high_price")
                    )
                    if highlow_tp_reference is None:
                        highlow_tp_reference = data.get("maxdiffprofit_high_price") or data.get("predicted_high")
                else:
                    highlow_tp_reference = (
                        data.get("maxdiffalwayson_low_price")
                        if entry_strategy == "maxdiffalwayson"
                        else data.get("maxdiffprofit_low_price")
                    )
                    if highlow_tp_reference is None:
                        highlow_tp_reference = data.get("maxdiffprofit_low_price") or data.get("predicted_low")
                takeprofit_price = coerce_numeric(highlow_tp_reference, default=float("nan"))
                if math.isnan(takeprofit_price) or takeprofit_price <= 0:
                    logger.debug(
                        "%s highlow takeprofit skipped due to invalid target (%s)",
                        symbol,
                        highlow_tp_reference,
                    )
                else:
                    try:
                        logger.info(
                            "Scheduling highlow takeprofit for %s at %.4f",
                            symbol,
                            takeprofit_price,
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
                        logger.warning("Failed to schedule highlow takeprofit for %s: %s", symbol, exc)
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
                            "Scheduling discretionary takeprofit for %s at %.4f (entry_ref=%.4f)",
                            symbol,
                            float(tp_price),
                            entry_reference,
                        )
                        spawn_close_position_at_takeprofit(symbol, float(tp_price))
                    except Exception as exc:
                        logger.warning("Failed to schedule takeprofit for %s: %s", symbol, exc)
                elif tp_price is not None:
                    logger.debug(
                        "%s takeprofit %.4f skipped (entry_ref=%s, side=%s)",
                        symbol,
                        float(tp_price),
                        entry_reference,
                        data["side"],
                    )
        elif transition_to_normal:
            logger.info(
                f"{symbol}: Probe already at target sizing; marking transition complete without additional orders."
            )
            entry_strategy = data.get("strategy")
            stored_entry_strategy = "maxdiff" if entry_strategy in MAXDIFF_LIMIT_STRATEGIES else entry_strategy
            _mark_probe_transitioned(symbol, data["side"], current_position_size, strategy=stored_entry_strategy)
            _update_active_trade(
                symbol,
                data["side"],
                mode="probe_transition",
                qty=current_position_size,
                strategy=stored_entry_strategy,
            )
            _tag_active_trade_strategy(symbol, data["side"], stored_entry_strategy)
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
        if not entry_strategy and symbol in previous_picks:
            entry_strategy = previous_picks.get(symbol, {}).get("strategy")
        lookup_entry_strategy = "highlow" if entry_strategy in MAXDIFF_STRATEGIES else entry_strategy

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


def main():
    symbols = [
        "COUR",
        "GOOG",
        "TSLA",
        "NVDA",
        "AAPL",
        "U",
        "ADSK",
        "ADBE",
        "MSFT",
        "COIN",
        # "MSFT",
        # "NFLX",
        # adding more as we do quite well now with volatility
        "AMZN",
        "AMD",
        "INTC",
        "QUBT",
        "BTCUSD",
        "ETHUSD",
        "UNIUSD",
        "SOLUSD",
        "AVAXUSD",
        "LINKUSD",
    ]
    previous_picks = {}

    # Track when each analysis was last run
    last_initial_run = None
    last_market_open_run = None
    last_market_open_hour2_run = None
    last_market_close_run = None
    last_crypto_midnight_refresh = None

    while True:
        try:
            market_open, market_close = get_market_hours()
            now = datetime.now(pytz.timezone("US/Eastern"))
            today = now.date()
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
            elif (
                (now.hour == 19 and 0 <= now.minute < 30)
                and (last_crypto_midnight_refresh is None or last_crypto_midnight_refresh != today)
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
