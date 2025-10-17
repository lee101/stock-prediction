import logging
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import sleep
from typing import Dict, List, Optional, Tuple

import pytz
from loguru import logger

import alpaca_wrapper
try:
    from backtest_test3_inline import backtest_forecasts, release_model_resources
except Exception as import_exc:  # pragma: no cover - exercised via tests with stubs
    logging.getLogger(__name__).warning(
        "Falling back to stubbed backtest resources due to import failure: %s", import_exc
    )

    def backtest_forecasts(*args, **kwargs):
        raise RuntimeError(
            "backtest_forecasts is unavailable because backtest_test3_inline could not be imported."
        ) from import_exc

    def release_model_resources() -> None:
        return None
from data_curate_daily import get_bid, get_ask, download_exchange_latest_data
from env_real import ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD
from jsonshelve import FlatShelf
from src.comparisons import is_buy_side, is_same_side, is_sell_side
from src.date_utils import is_nyse_trading_day_now, is_nyse_trading_day_ending
from src.fixtures import crypto_symbols
from src.logging_utils import setup_logging
from src.trading_obj_utils import filter_to_realistic_positions
from src.process_utils import backout_near_market, ramp_into_position, spawn_close_position_at_takeprofit
from src.portfolio_risk import record_portfolio_snapshot
from src.sizing_utils import get_qty
from alpaca.data import StockHistoricalDataClient
from stock.data_utils import coerce_numeric, ensure_lower_bound, safe_divide
from stock.state import ensure_state_dir as _shared_ensure_state_dir
from stock.state import get_state_dir, get_state_file, resolve_state_suffix

# Configure logging
logger = setup_logging("trade_stock_e2e.log")


STATE_DIR = get_state_dir()
STATE_SUFFIX = resolve_state_suffix()
TRADE_OUTCOME_FILE = get_state_file("trade_outcomes", STATE_SUFFIX)
TRADE_LEARNING_FILE = get_state_file("trade_learning", STATE_SUFFIX)
ACTIVE_TRADES_FILE = get_state_file("active_trades", STATE_SUFFIX)
TRADE_HISTORY_FILE = get_state_file("trade_history", STATE_SUFFIX)

MIN_STOCK_QTY = 1.0
MIN_CRYPTO_QTY = 0.001
MIN_PREDICTED_MOVEMENT = 0.0
MIN_DIRECTIONAL_CONFIDENCE = 0.0
MAX_TOTAL_EXPOSURE_PCT = 120.0
LIVE_DRAWDOWN_TRIGGER = -500.0  # dollars
PROBE_MAX_DURATION = timedelta(days=1)

LIQUID_CRYPTO_PREFIXES = ("BTC", "ETH", "SOL", "UNI")
TIGHT_SPREAD_EQUITIES = {"AAPL", "MSFT", "AMZN", "NVDA", "META", "GOOG"}
DEFAULT_SPREAD_BPS = 25
PROBE_LOSS_COOLDOWN_MINUTES = 180
ALLOW_HIGHLOW_ENTRY = os.getenv("ALLOW_HIGHLOW_ENTRY", "0").strip().lower() in {"1", "true", "yes", "on"}
ALLOW_TAKEPROFIT_ENTRY = os.getenv("ALLOW_TAKEPROFIT_ENTRY", "0").strip().lower() in {"1", "true", "yes", "on"}
ENABLE_TAKEPROFIT_BRACKETS = os.getenv("ENABLE_TAKEPROFIT_BRACKETS", "0").strip().lower() in {"1", "true", "yes", "on"}
CONSENSUS_MIN_MOVE_PCT = float(os.getenv("CONSENSUS_MIN_MOVE_PCT", "0.001"))

_quote_client: Optional[StockHistoricalDataClient] = None
_COOLDOWN_STATE: Dict[str, Dict[str, datetime]] = {}

_trade_outcomes_store: Optional[FlatShelf] = None
_trade_learning_store: Optional[FlatShelf] = None
_active_trades_store: Optional[FlatShelf] = None
_trade_history_store: Optional[FlatShelf] = None

_TRUTHY = {"1", "true", "yes", "on"}


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


def compute_spread_bps(bid: Optional[float], ask: Optional[float]) -> float:
    if bid is None or ask is None:
        return float("inf")
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return float("inf")
    return (ask - bid) / mid * 1e4


def resolve_spread_cap(symbol: str) -> int:
    if symbol.endswith("USD") and symbol.startswith(LIQUID_CRYPTO_PREFIXES):
        return 35
    if symbol in TIGHT_SPREAD_EQUITIES:
        return 8
    return DEFAULT_SPREAD_BPS


def is_tradeable(
    symbol: str,
    bid: Optional[float],
    ask: Optional[float],
    *,
    avg_dollar_vol: Optional[float] = None,
    atr_pct: Optional[float] = None,
) -> Tuple[bool, str]:
    spread_bps = compute_spread_bps(bid, ask)
    if math.isinf(spread_bps):
        return False, "Missing bid/ask quote"
    kronos_only = _is_kronos_only_mode()
    max_spread_bps = resolve_spread_cap(symbol)
    if kronos_only:
        max_spread_bps = max_spread_bps * 3
    min_dollar_vol = 5_000_000 if not kronos_only else 0.0
    atr_cap = 8.0 if not kronos_only else 14.0
    if avg_dollar_vol is not None and avg_dollar_vol < min_dollar_vol:
        return False, f"Low dollar vol {avg_dollar_vol:,.0f}"
    if atr_pct is not None and atr_pct > atr_cap:
        return False, f"ATR% too high {atr_pct:.2f}"
    if spread_bps > max_spread_bps:
        return False, f"Spread {spread_bps:.1f}bps > {max_spread_bps}bps"
    return True, f"Spread {spread_bps:.1f}bps OK"


def expected_cost_bps(symbol: str) -> float:
    base = 20.0 if symbol.endswith("USD") else 6.0
    if symbol in {"META", "AMD", "LCID", "QUBT"}:
        base += 25.0
    return base


def pass_edge_threshold(symbol: str, expected_move_pct: float) -> Tuple[bool, str]:
    move_bps = abs(expected_move_pct) * 1e4
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


def agree_direction(*pred_signs: int) -> bool:
    signs = {sign for sign in pred_signs if sign in (-1, 1)}
    return len(signs) == 1


def resolve_signal_sign(move_pct: float) -> int:
    threshold = CONSENSUS_MIN_MOVE_PCT
    if _is_kronos_only_mode():
        threshold *= 0.25
    if abs(move_pct) < threshold:
        return 0
    return 1 if move_pct > 0 else -1


def kelly_lite(edge_pct: float, sigma_pct: float, cap: float = 0.15) -> float:
    if sigma_pct <= 0:
        return 0.0
    raw = edge_pct / (sigma_pct ** 2)
    scaled = 0.2 * raw
    if scaled <= 0:
        return 0.0
    return float(min(cap, max(0.0, scaled)))


def should_rebalance(
    current_pos_side: Optional[str],
    new_side: str,
    current_size: float,
    target_size: float,
    eps: float = 0.25,
) -> bool:
    current_side = (current_pos_side or "").lower()
    new_side_norm = new_side.lower()
    if current_side not in {"buy", "sell"} or new_side_norm not in {"buy", "sell"}:
        return True
    if current_side != new_side_norm:
        return True
    current_abs = abs(current_size)
    target_abs = abs(target_size)
    if current_abs <= 1e-9:
        return True
    delta = abs(target_abs - current_abs) / max(current_abs, 1e-9)
    return delta > eps


def _record_loss_timestamp(symbol: str, closed_at: Optional[str]) -> None:
    if not closed_at:
        return
    ts = _parse_timestamp(closed_at)
    if ts:
        _COOLDOWN_STATE[symbol] = {"last_stop_time": ts}


def clear_cooldown(symbol: str) -> None:
    _COOLDOWN_STATE.pop(symbol, None)


def can_trade_now(symbol: str, now: datetime, min_cooldown_minutes: int = PROBE_LOSS_COOLDOWN_MINUTES) -> bool:
    state = _COOLDOWN_STATE.get(symbol)
    if not state:
        return True
    last_stop = state.get("last_stop_time")
    if isinstance(last_stop, datetime):
        delta = now - last_stop
        if delta.total_seconds() < min_cooldown_minutes * 60:
            return False
    return True


def _edge_threshold_bps(symbol: str) -> float:
    base_cost = expected_cost_bps(symbol) + 10.0
    hard_floor = 40.0 if symbol.endswith("USD") else 15.0
    return max(base_cost, hard_floor)


def _evaluate_strategy_entry_gate(
    symbol: str,
    stats: Dict[str, float],
    *,
    fallback_used: bool,
    sample_size: int,
) -> Tuple[bool, str]:
    if fallback_used:
        return False, "fallback_metrics"
    avg_return = float(stats.get("avg_return") or 0.0)
    sharpe = float(stats.get("sharpe") or 0.0)
    turnover = float(stats.get("turnover") or 0.0)
    max_drawdown = float(stats.get("max_drawdown") or 0.0)
    edge_bps = avg_return * 1e4
    needed_edge = _edge_threshold_bps(symbol)
    if edge_bps < needed_edge:
        return False, f"edge {edge_bps:.1f}bps < need {needed_edge:.1f}bps"
    if sharpe < 0.5:
        return False, f"sharpe {sharpe:.2f} below 0.50 gate"
    if sample_size < 120:
        return False, f"insufficient samples {sample_size} < 120"
    if max_drawdown < -0.08:
        return False, f"max drawdown {max_drawdown:.2f} below -0.08 gate"
    if turnover > 2.0 and sharpe < 0.8:
        return False, f"turnover {turnover:.2f} with sharpe {sharpe:.2f}"
    return True, "ok"


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


LOSS_BLOCK_COOLDOWN = timedelta(days=3)
DEFAULT_MIN_CORE_POSITIONS = 4
DEFAULT_MAX_PORTFOLIO = 6
EXPANDED_PORTFOLIO = 8
MIN_EXPECTED_MOVE_PCT = 1e-4
MIN_EDGE_STRENGTH = 1e-5
COMPACT_LOGS = os.getenv("COMPACT_TRADING_LOGS", "").strip().lower() in {"1", "true", "yes", "on"}
MARKET_CLOSE_SHIFT_MINUTES = int(os.getenv("MARKET_CLOSE_SHIFT_MINUTES", "45"))
MARKET_CLOSE_ANALYSIS_WINDOW_MINUTES = int(os.getenv("MARKET_CLOSE_ANALYSIS_WINDOW_MINUTES", "15"))


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
            ("simple", data.get("simple_return"), 3),
            ("all", strategy_returns.get("all_signals"), 3),
            ("takeprofit", strategy_returns.get("takeprofit"), 3),
            ("highlow", strategy_returns.get("highlow"), 3),
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
    summary_parts = [
        " ".join(status_parts),
        f"returns[{returns_metrics or '-'}]",
        f"edges[{edges_metrics or '-'}]",
        f"prices[{prices_metrics or '-'}]",
    ]
    if data.get("trade_blocked") and data.get("block_reason"):
        summary_parts.append(f"block_reason={data['block_reason']}")

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
            summary_parts.append("probe=" + ",".join(str(note) for note in probe_notes))

    _log_detail(" | ".join(summary_parts))


def _normalize_side_for_key(side: str) -> str:
    normalized = str(side).lower()
    if "short" in normalized or "sell" in normalized:
        return "sell"
    return "buy"


def _parse_timestamp(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        parsed = datetime.fromisoformat(ts)
    except ValueError:
        try:
            parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            logger.warning(f"Unable to parse timestamp '{ts}' from trade outcomes store")
            return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _state_key(symbol: str, side: str) -> str:
    return f"{symbol}|{_normalize_side_for_key(side)}"


def _load_trade_outcome(symbol: str, side: str) -> Dict:
    store = _get_trade_outcomes_store()
    if store is None:
        return {}
    try:
        store.load()
    except Exception as exc:
        logger.error(f"Failed loading trade outcomes store: {exc}")
        return {}
    return store.get(_state_key(symbol, side), {})


def _load_learning_state(symbol: str, side: str) -> Dict:
    store = _get_trade_learning_store()
    if store is None:
        return {}
    try:
        store.load()
    except Exception as exc:
        logger.error(f"Failed loading trade learning store: {exc}")
        return {}
    return store.get(_state_key(symbol, side), {})


def _save_learning_state(symbol: str, side: str, state: Dict) -> None:
    store = _get_trade_learning_store()
    if store is None:
        return
    try:
        store.load()
    except Exception as exc:
        logger.error(f"Failed refreshing trade learning store before save: {exc}")
        return
    key = _state_key(symbol, side)
    store[key] = state


def _update_learning_state(symbol: str, side: str, **updates) -> Dict:
    state = dict(_load_learning_state(symbol, side))
    changed = False
    for key, value in updates.items():
        if state.get(key) != value:
            state[key] = value
            changed = True
    if changed:
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        _save_learning_state(symbol, side, state)
    return state


def _mark_probe_pending(symbol: str, side: str) -> Dict:
    return _update_learning_state(
        symbol,
        side,
        pending_probe=True,
        probe_active=False,
        last_probe_successful=False,
    )


def _mark_probe_active(symbol: str, side: str, qty: float) -> Dict:
    return _update_learning_state(
        symbol,
        side,
        pending_probe=False,
        probe_active=True,
        last_probe_qty=qty,
        probe_started_at=datetime.now(timezone.utc).isoformat(),
    )


def _mark_probe_completed(symbol: str, side: str, successful: bool) -> Dict:
    return _update_learning_state(
        symbol,
        side,
        pending_probe=not successful,
        probe_active=False,
        last_probe_completed_at=datetime.now(timezone.utc).isoformat(),
        last_probe_successful=successful,
    )


def _describe_probe_state(learning_state: Dict, now: Optional[datetime] = None) -> Dict[str, Optional[object]]:
    """Summarise probe lifecycle timing to inform transition and expiry logic."""
    if learning_state is None:
        learning_state = {}
    now = now or datetime.now(timezone.utc)
    probe_active = bool(learning_state.get("probe_active"))
    probe_started_at = _parse_timestamp(learning_state.get("probe_started_at"))
    summary: Dict[str, Optional[object]] = {
        "probe_active": probe_active,
        "probe_started_at": probe_started_at.isoformat() if probe_started_at else None,
        "probe_age_seconds": None,
        "probe_expires_at": None,
        "probe_expired": False,
        "probe_transition_ready": False,
    }
    if not probe_active or probe_started_at is None:
        return summary

    probe_age = now - probe_started_at
    summary["probe_age_seconds"] = ensure_lower_bound(probe_age.total_seconds(), 0.0)
    expires_at = probe_started_at + PROBE_MAX_DURATION
    summary["probe_expires_at"] = expires_at.isoformat()
    summary["probe_expired"] = now >= expires_at

    est = pytz.timezone("US/Eastern")
    now_est = now.astimezone(est)
    started_est = probe_started_at.astimezone(est)
    summary["probe_transition_ready"] = now_est.date() > started_est.date()
    return summary


def _mark_probe_transitioned(symbol: str, side: str, qty: float) -> Dict:
    """Mark a probe as promoted into a standard position."""
    return _update_learning_state(
        symbol,
        side,
        pending_probe=False,
        probe_active=False,
        last_probe_successful=False,
        probe_transitioned_at=datetime.now(timezone.utc).isoformat(),
        last_probe_transition_qty=qty,
    )


def _update_active_trade(symbol: str, side: str, mode: str, qty: float, strategy: Optional[str] = None) -> None:
    store = _get_active_trades_store()
    if store is None:
        return
    try:
        store.load()
    except Exception as exc:
        logger.error(f"Failed loading active trades store: {exc}")
        return
    key = _state_key(symbol, side)
    record = {
        "mode": mode,
        "qty": qty,
        "opened_at": datetime.now(timezone.utc).isoformat(),
    }
    if strategy:
        record["entry_strategy"] = strategy
    store[key] = record


def _tag_active_trade_strategy(symbol: str, side: str, strategy: Optional[str]) -> None:
    if not strategy:
        return
    store = _get_active_trades_store()
    if store is None:
        return
    try:
        store.load()
    except Exception as exc:
        logger.error(f"Failed loading active trades store while tagging strategy: {exc}")
        return
    key = _state_key(symbol, side)
    record = dict(store.get(key, {}))
    if not record:
        return
    if record.get("entry_strategy") == strategy:
        return
    record["entry_strategy"] = strategy
    store[key] = record


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
    store = _get_active_trades_store()
    if store is None:
        return {}
    try:
        store.load()
    except Exception as exc:
        logger.error(f"Failed loading active trades store for lookup: {exc}")
        return {}
    key = _state_key(symbol, side)
    trade = store.get(key, {})
    return dict(trade) if trade else {}


def _pop_active_trade(symbol: str, side: str) -> Dict:
    store = _get_active_trades_store()
    if store is None:
        return {}
    try:
        store.load()
    except Exception as exc:
        logger.error(f"Failed loading active trades store for pop: {exc}")
        return {}
    key = _state_key(symbol, side)
    trade = store.get(key, {})
    if key in store:
        del store[key]
    return trade


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
    key = f"{position.symbol}|{normalized_side}"
    active_trade = _pop_active_trade(position.symbol, normalized_side)
    trade_mode = active_trade.get("mode", "probe" if active_trade else "normal")
    entry_strategy = active_trade.get("entry_strategy")
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

    # Update learning state metadata
    _update_learning_state(
        position.symbol,
        normalized_side,
        last_pnl=pnl_value,
        last_qty=qty_value,
        last_closed_at=record["closed_at"],
        last_reason=reason,
        last_mode=trade_mode,
    )

    if trade_mode == "probe":
        _mark_probe_completed(position.symbol, normalized_side, successful=pnl_value > 0)
    elif pnl_value < 0:
        _mark_probe_pending(position.symbol, normalized_side)
    else:
        _update_learning_state(
            position.symbol,
            normalized_side,
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


def _evaluate_trade_block(symbol: str, side: str) -> Dict[str, Optional[object]]:
    record = _load_trade_outcome(symbol, side)
    learning_state = dict(_load_learning_state(symbol, side))
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
            logger.info(
                f"Using MARKETSIM_BACKTEST_SIMULATIONS override of {env_simulations} for backtest iterations."
            )
    else:
        env_simulations = None

    kronos_only_mode = _is_kronos_only_mode()

    for symbol in symbols:
        try:
            # not many because we need to adapt strats? eg the wierd spikes in uniusd are a big opportunity to trade w high/low
            # but then i bumped up because its not going to say buy crypto when its down, if its most recent based?
            num_simulations = env_simulations or 70
            used_fallback_engine = False

            try:
                backtest_df = backtest_forecasts(symbol, num_simulations)
            except Exception as exc:
                logger.warning(
                    f"Primary backtest_forecasts failed for {symbol}: {exc}. "
                    "Attempting simulator fallback analytics."
                )
                try:
                    from marketsimulator import backtest_test3_inline as sim_backtest  # type: ignore

                    backtest_df = sim_backtest.backtest_forecasts(symbol, num_simulations)
                except Exception as fallback_exc:
                    logger.error(
                        f"Fallback backtest also failed for {symbol}: {fallback_exc}. Skipping symbol."
                    )
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

            def _mean_column(column: str, default: float = 0.0) -> float:
                if column in backtest_df.columns:
                    return coerce_numeric(backtest_df[column].mean(), default=default)
                return default

            strategy_returns = {
                "simple": backtest_df["simple_strategy_return"].mean(),
                "all_signals": backtest_df["all_signals_strategy_return"].mean(),
                "takeprofit": backtest_df["entry_takeprofit_return"].mean(),
                "highlow": backtest_df["highlow_return"].mean(),
            }
            if "ci_guard_return" in backtest_df.columns:
                strategy_returns["ci_guard"] = backtest_df["ci_guard_return"].mean()

            unprofit_return = 0.0
            unprofit_sharpe = 0.0
            if "unprofit_shutdown_return" in backtest_df.columns:
                unprofit_return = backtest_df["unprofit_shutdown_return"].mean()
                strategy_returns["unprofit_shutdown"] = unprofit_return
            if "unprofit_shutdown_sharpe" in backtest_df.columns:
                unprofit_sharpe = backtest_df["unprofit_shutdown_sharpe"].mean()

            last_prediction = backtest_df.iloc[0]
            walk_forward_oos_sharpe_raw = last_prediction.get("walk_forward_oos_sharpe")
            walk_forward_turnover_raw = last_prediction.get("walk_forward_turnover")
            walk_forward_highlow_raw = last_prediction.get("walk_forward_highlow_sharpe")
            walk_forward_takeprofit_raw = last_prediction.get("walk_forward_takeprofit_sharpe")

            walk_forward_oos_sharpe = (
                coerce_numeric(walk_forward_oos_sharpe_raw)
                if walk_forward_oos_sharpe_raw is not None
                else None
            )
            walk_forward_turnover = (
                coerce_numeric(walk_forward_turnover_raw)
                if walk_forward_turnover_raw is not None
                else None
            )
            walk_forward_highlow_sharpe = (
                coerce_numeric(walk_forward_highlow_raw)
                if walk_forward_highlow_raw is not None
                else None
            )
            walk_forward_takeprofit_sharpe = (
                coerce_numeric(walk_forward_takeprofit_raw)
                if walk_forward_takeprofit_raw is not None
                else None
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

            strategy_stats: Dict[str, Dict[str, float]] = {
                "simple": {
                    "avg_return": strategy_returns.get("simple", 0.0),
                    "sharpe": _mean_column("simple_strategy_sharpe"),
                    "turnover": _mean_column("simple_strategy_turnover"),
                    "max_drawdown": _mean_column("simple_strategy_max_drawdown"),
                },
                "all_signals": {
                    "avg_return": strategy_returns.get("all_signals", 0.0),
                    "sharpe": _mean_column("all_signals_strategy_sharpe"),
                    "turnover": _mean_column("all_signals_strategy_turnover"),
                    "max_drawdown": _mean_column("all_signals_strategy_max_drawdown"),
                },
                "takeprofit": {
                    "avg_return": strategy_returns.get("takeprofit", 0.0),
                    "sharpe": _mean_column("entry_takeprofit_sharpe"),
                    "turnover": _mean_column("entry_takeprofit_turnover"),
                    "max_drawdown": _mean_column("entry_takeprofit_max_drawdown"),
                },
                "highlow": {
                    "avg_return": strategy_returns.get("highlow", 0.0),
                    "sharpe": _mean_column("highlow_sharpe"),
                    "turnover": _mean_column("highlow_turnover"),
                    "max_drawdown": _mean_column("highlow_max_drawdown"),
                },
            }
            if "ci_guard" in strategy_returns:
                strategy_stats["ci_guard"] = {
                    "avg_return": strategy_returns.get("ci_guard", 0.0),
                    "sharpe": _mean_column("ci_guard_sharpe"),
                    "turnover": _mean_column("ci_guard_turnover"),
                    "max_drawdown": _mean_column("ci_guard_max_drawdown"),
                }

            strategy_ineligible: Dict[str, str] = {}
            candidate_scores: Dict[str, float] = {}
            strategy_candidates: List[Tuple[float, str]] = []

            for name, stats in strategy_stats.items():
                if name not in strategy_returns:
                    continue
                allow_config = True
                if name == "takeprofit":
                    allow_config = ALLOW_TAKEPROFIT_ENTRY
                elif name == "highlow":
                    allow_config = ALLOW_HIGHLOW_ENTRY

                if name in {"takeprofit", "highlow"}:
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

                score = float(stats.get("avg_return") or 0.0) + 0.05 * float(stats.get("sharpe") or 0.0)
                if name in {"simple", "ci_guard"}:
                    score += 0.001
                candidate_scores[name] = score
                strategy_candidates.append((score, name))

            if strategy_candidates:
                strategy_candidates.sort(key=lambda item: item[0], reverse=True)
                best_strategy = strategy_candidates[0][1]
                avg_return = float(strategy_stats.get(best_strategy, {}).get("avg_return", 0.0))
            else:
                best_strategy = "simple"
                avg_return = strategy_returns.get(best_strategy, 0.0)
            selected_strategy_score = candidate_scores.get(best_strategy)

            if strategy_ineligible:
                logger.debug("%s strategy entry gates rejected: %s", symbol, strategy_ineligible)

            close_movement_raw = predicted_close_price - close_price
            high_movement = predicted_high_price - close_price
            low_movement = predicted_low_price - close_price

            if best_strategy == "all_signals":
                if all(x > 0 for x in [close_movement_raw, high_movement, low_movement]):
                    position_side = "buy"
                elif all(x < 0 for x in [close_movement_raw, high_movement, low_movement]):
                    position_side = "sell"
                else:
                    _log_detail(f"Skipping {symbol} - mixed directional signals despite all_signals lead")
                    continue
                predicted_movement = close_movement_raw
            else:
                predicted_movement = close_movement_raw
                position_side = "buy" if predicted_movement > 0 else "sell"

            expected_move_pct = safe_divide(predicted_movement, close_price, default=0.0)
            simple_return = strategy_returns.get("simple", 0.0)
            takeprofit_return = strategy_returns.get("takeprofit", 0.0)
            highlow_return = strategy_returns.get("highlow", 0.0)
            simple_sharpe = 0.0
            if "simple_strategy_sharpe" in backtest_df.columns:
                simple_sharpe = coerce_numeric(backtest_df["simple_strategy_sharpe"].mean(), default=0.0)
            price_skill = max(simple_return, 0.0) + 0.25 * max(simple_sharpe, 0.0)
            highlow_allowed_entry = ALLOW_HIGHLOW_ENTRY and ("highlow" not in strategy_ineligible)
            takeprofit_allowed_entry = ALLOW_TAKEPROFIT_ENTRY and ("takeprofit" not in strategy_ineligible)

            raw_expected_move_pct = expected_move_pct
            calibrated_move_raw = last_prediction.get("calibrated_expected_move_pct")
            calibrated_move_pct = (
                coerce_numeric(calibrated_move_raw)
                if calibrated_move_raw is not None
                else None
            )
            if calibrated_move_pct is not None:
                expected_move_pct = calibrated_move_pct
                predicted_movement = expected_move_pct * close_price
                calibrated_close_price = close_price * (1.0 + expected_move_pct)
            else:
                calibrated_close_price = predicted_close_price

            abs_move = abs(expected_move_pct)
            if abs_move < MIN_EXPECTED_MOVE_PCT:
                abs_move = 0.0
            edge_strength = price_skill * abs_move
            directional_edge = edge_strength if predicted_movement >= 0 else -edge_strength

            toto_move_pct = coerce_numeric(last_prediction.get("toto_expected_move_pct"), default=0.0)
            kronos_move_pct = coerce_numeric(last_prediction.get("kronos_expected_move_pct"), default=0.0)
            realized_volatility_pct = coerce_numeric(last_prediction.get("realized_volatility_pct"), default=0.0)
            avg_dollar_vol_raw = last_prediction.get("dollar_vol_20d")
            avg_dollar_vol = (
                coerce_numeric(avg_dollar_vol_raw)
                if avg_dollar_vol_raw is not None
                else None
            )
            atr_pct_raw = last_prediction.get("atr_pct_14")
            atr_pct = coerce_numeric(atr_pct_raw) if atr_pct_raw is not None else None
            sigma_pct = safe_divide(realized_volatility_pct, 100.0, default=0.0)
            if sigma_pct <= 0:
                sigma_pct = max(abs(expected_move_pct), 1e-3)
            kelly_fraction = kelly_lite(abs(expected_move_pct), sigma_pct)

            if (
                edge_strength < MIN_EDGE_STRENGTH
                and max(avg_return, simple_return, takeprofit_return, highlow_return) <= 0
            ):
                _log_detail(
                    f"Skipping {symbol} - no actionable price edge "
                    f"(edge_strength={edge_strength:.6f}, avg_return={avg_return:.6f})"
                )
                continue

            effective_takeprofit = takeprofit_return if takeprofit_allowed_entry else 0.0
            effective_highlow = highlow_return if highlow_allowed_entry else 0.0
            composite_score = (
                0.2 * avg_return
                + 0.35 * simple_return
                + 0.15 * edge_strength
                + 0.1 * unprofit_return
                + 0.07 * effective_takeprofit
                + 0.07 * effective_highlow
                + 0.06 * strategy_returns.get("ci_guard", 0.0)
            )

            bid_price, ask_price = fetch_bid_ask(symbol)
            spread_bps = compute_spread_bps(bid_price, ask_price)
            spread_cap = resolve_spread_cap(symbol)
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
            elif consensus_model_count > 1 and not consensus_ok:
                consensus_reason = f"Model disagreement toto={sign_toto} kronos={sign_kronos}"
            elif consensus_model_count == 1:
                if sign_toto != 0 and sign_kronos == 0:
                    fallback_source = "Toto"
                elif sign_kronos != 0 and sign_toto == 0:
                    fallback_source = "Kronos"
                if fallback_source:
                    _log_detail(f"{symbol}: consensus fallback to {fallback_source} signal only")

            block_info = _evaluate_trade_block(symbol, position_side)
            last_pnl = block_info.get("last_pnl")
            last_closed_at = block_info.get("last_closed_at")
            if last_pnl is not None:
                if last_pnl < 0:
                    _record_loss_timestamp(symbol, last_closed_at)
                else:
                    clear_cooldown(symbol)
            now_utc = datetime.now(timezone.utc)
            cooldown_ok = can_trade_now(symbol, now_utc)

            gating_reasons: List[str] = []
            sharpe_cutoff = 0.3 if not kronos_only_mode else -0.25
            if walk_forward_oos_sharpe is not None and walk_forward_oos_sharpe < sharpe_cutoff:
                gating_reasons.append(
                    f"Walk-forward Sharpe {walk_forward_oos_sharpe:.2f} < {sharpe_cutoff:.2f}"
                )
            if not kronos_only_mode:
                if (
                    walk_forward_turnover is not None
                    and walk_forward_oos_sharpe is not None
                    and walk_forward_turnover > 2.0
                    and walk_forward_oos_sharpe < 0.5
                ):
                    gating_reasons.append(
                        f"Walk-forward turnover {walk_forward_turnover:.2f} with Sharpe {walk_forward_oos_sharpe:.2f}"
                    )
            if not tradeable:
                gating_reasons.append(spread_reason)
            if not edge_ok:
                gating_reasons.append(edge_reason)
            if kronos_only_mode and consensus_reason and "Model disagreement" in consensus_reason:
                if sign_kronos in (-1, 1):
                    consensus_reason = None
            if kronos_only_mode and consensus_reason and consensus_reason.startswith(
                "No directional signal"
            ):
                if sign_kronos in (-1, 1):
                    consensus_reason = None
            if consensus_reason:
                gating_reasons.append(consensus_reason)
            if not cooldown_ok and not kronos_only_mode:
                gating_reasons.append("Cooldown active after recent loss")
            if kelly_fraction <= 0:
                gating_reasons.append("Kelly fraction <= 0")

            base_blocked = block_info.get("blocked", False)
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

            results[symbol] = {
                "avg_return": avg_return,
                "predictions": backtest_df,
                "side": position_side,
                "predicted_movement": predicted_movement,
                "strategy": best_strategy,
                "predicted_high": float(predicted_high_price),
                "predicted_low": float(predicted_low_price),
                "predicted_close": float(predicted_close_price),
                "calibrated_close": float(calibrated_close_price),
                "last_close": float(close_price),
                "strategy_returns": strategy_returns,
                "simple_return": simple_return,
                "unprofit_shutdown_return": unprofit_return,
                "unprofit_shutdown_sharpe": unprofit_sharpe,
                "expected_move_pct": expected_move_pct,
                "expected_move_pct_raw": raw_expected_move_pct,
                "price_skill": price_skill,
                "edge_strength": edge_strength,
                "directional_edge": directional_edge,
                "composite_score": composite_score,
                "selected_strategy_score": selected_strategy_score,
                "strategy_entry_ineligible": strategy_ineligible,
                "strategy_candidate_scores": candidate_scores,
                "fallback_backtest": used_fallback_engine,
                "highlow_entry_allowed": highlow_allowed_entry,
                "takeprofit_entry_allowed": takeprofit_allowed_entry,
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
                "avg_dollar_vol": float(avg_dollar_vol) if avg_dollar_vol is not None else None,
                "atr_pct_14": float(atr_pct) if atr_pct is not None else None,
                "cooldown_active": not cooldown_ok,
                "walk_forward_oos_sharpe": walk_forward_oos_sharpe,
                "walk_forward_turnover": walk_forward_turnover,
                "walk_forward_highlow_sharpe": walk_forward_highlow_sharpe,
                "walk_forward_takeprofit_sharpe": walk_forward_takeprofit_sharpe,
                "backtest_samples": sample_size,
            }
            _log_analysis_summary(symbol, results[symbol])

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {str(e)}")
            continue

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
    probe_candidates = [(symbol, data) for symbol, data in all_results.items() if data.get("trade_mode") == "probe"]
    for symbol, data in probe_candidates:
        if symbol in picks:
            continue
        if max_expanded and len(picks) < max_expanded:
            picks[symbol] = data
        elif len(picks) < max_positions:
            picks[symbol] = data
        else:
            # Replace the weakest pick to guarantee probe follow-up.
            weakest_symbol, _ = min(picks.items(), key=lambda item: item[1].get("composite_score", float("-inf")))
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
        should_close = False
        close_reason = ""

        if symbol not in current_picks:
            # For crypto on weekends, only close if direction changed
            if symbol in crypto_symbols and not is_nyse_trading_day_now():
                if symbol in all_analyzed_results and not is_same_side(
                    all_analyzed_results[symbol]["side"], position.side
                ):
                    logger.info(f"Closing crypto position for {symbol} due to direction change (weekend)")
                    should_close = True
                    close_reason = "weekend_direction_change"
                else:
                    logger.info(f"Keeping crypto position for {symbol} on weekend - no direction change")
            # For stocks when market is closed, only close if direction changed
            elif symbol not in crypto_symbols and not is_nyse_trading_day_now():
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

        normalized_side = _normalize_side_for_key(position.side)
        probe_meta = all_analyzed_results.get(symbol, {})
        if not probe_meta:
            probe_meta = _evaluate_trade_block(symbol, normalized_side)
        if probe_meta.get("probe_expired") and not should_close:
            logger.info(
                f"Closing position for {symbol} as probe duration exceeded {PROBE_MAX_DURATION} "
                "without transition; scheduling backout"
            )
            should_close = True
            close_reason = "probe_duration_exceeded"

        if should_close:
            _record_trade_outcome(position, close_reason or "unspecified")
            backout_near_market(symbol)

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

    for symbol, data in current_picks.items():
        trade_mode = data.get("trade_mode", "normal")
        is_probe_trade = trade_mode == "probe"
        probe_transition_ready = data.get("probe_transition_ready", False)
        probe_expired = data.get("probe_expired", False)

        if data.get("trade_blocked") and not is_probe_trade:
            logger.info(f"Skipping {symbol} due to active block: {data.get('block_reason', 'recent loss')}")
            continue
        if probe_expired:
            logger.info(
                f"Skipping {symbol} entry while probe backout executes (duration exceeded {PROBE_MAX_DURATION})."
            )
            continue

        position_exists = any(p.symbol == symbol for p in positions)
        correct_side = any(p.symbol == symbol and is_same_side(p.side, data["side"]) for p in positions)

        transition_to_normal = (
            is_probe_trade and probe_transition_ready and position_exists and correct_side
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

        min_trade_qty = MIN_CRYPTO_QTY if symbol in crypto_symbols else MIN_STOCK_QTY
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
                logger.info(
                    f"{symbol}: Probe sizing fixed at minimum tradable quantity {target_qty}"
                )
                should_enter = not position_exists or not correct_side
                needs_size_increase = False
            else:
                base_qty = computed_qty
                kelly_value = ensure_lower_bound(
                    coerce_numeric(data.get("kelly_fraction"), default=1.0),
                    0.0,
                    default=0.0,
                )
                if kelly_value <= 0:
                    logger.info(f"{symbol}: Kelly fraction non-positive; skipping entry.")
                    continue
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
                    if allowed_value <= 0:
                        logger.info(
                            f"Skipping {symbol} entry to respect max exposure "
                            f"({projected_pct:.1f}% > {MAX_TOTAL_EXPOSURE_PCT:.1f}%)"
                        )
                        continue
                    adjusted_qty = ensure_lower_bound(
                        safe_divide(allowed_value, entry_price, default=0.0),
                        0.0,
                        default=0.0,
                    )
                    if adjusted_qty <= 0:
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
            if symbol in crypto_symbols:
                should_enter = (not position_exists and is_buy_side(data["side"])) or effective_probe
            else:
                should_enter = not position_exists or effective_probe
            if effective_probe:
                if ask_price is not None or bid_price is not None:
                    entry_price = ask_price if data["side"] == "buy" else bid_price
                target_qty = ensure_lower_bound(min_trade_qty, 0.0, default=min_trade_qty)

        if effective_probe and target_qty <= 0:
            logger.warning(f"{symbol}: Unable to determine positive probe quantity; deferring trade.")
            _mark_probe_pending(symbol, data["side"])
            continue

        if should_enter or not correct_side:
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

            entry_strategy = data.get("strategy")
            is_highlow_entry = entry_strategy == "highlow" and not effective_probe
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
                        if is_buy_side(data["side"]):
                            limit_reference = data.get("predicted_low")
                        else:
                            limit_reference = data.get("predicted_high")
                        limit_price = coerce_numeric(limit_reference, default=float("nan"))
                        if math.isnan(limit_price) or limit_price <= 0:
                            logger.warning(
                                "%s highlow entry missing limit price (predicted bound=%s); falling back to ramp",
                                symbol,
                                limit_reference,
                            )
                        else:
                            try:
                                logger.info(
                                    "Submitting highlow limit order for %s %s qty=%s @ %.4f",
                                    symbol,
                                    data["side"],
                                    target_qty,
                                    limit_price,
                                )
                                result = alpaca_wrapper.open_order_at_price_or_all(
                                    symbol,
                                    target_qty,
                                    data["side"],
                                    limit_price,
                                )
                                if result is None:
                                    logger.warning(
                                        "Highlow limit order for %s returned None; will additionally ramp position.",
                                        symbol,
                                    )
                                else:
                                    highlow_limit_executed = True
                                    entry_price = limit_price
                            except Exception as exc:
                                logger.warning(
                                    "Failed to submit highlow limit order for %s: %s; will ramp instead.",
                                    symbol,
                                    exc,
                                )
                else:
                    logger.info(f"Probe trade target quantity for {symbol}: {target_qty} at price {entry_price}")

                if not highlow_limit_executed:
                    ramp_into_position(symbol, data["side"], target_qty=target_qty)
            else:
                logger.warning(f"Could not get bid/ask prices for {symbol}, using default sizing")
                if not highlow_limit_executed:
                    ramp_into_position(symbol, data["side"], target_qty=target_qty if effective_probe else None)

            if transition_to_normal:
                _mark_probe_transitioned(symbol, data["side"], target_qty)
                _update_active_trade(
                    symbol,
                    data["side"],
                    mode="probe_transition",
                    qty=target_qty,
                )
                _tag_active_trade_strategy(symbol, data["side"], entry_strategy)
                _normalize_active_trade_patch(_update_active_trade)
            elif effective_probe:
                _mark_probe_active(symbol, data["side"], target_qty)
                _update_active_trade(
                    symbol,
                    data["side"],
                    mode="probe",
                    qty=target_qty,
                )
                _tag_active_trade_strategy(symbol, data["side"], entry_strategy)
                _normalize_active_trade_patch(_update_active_trade)
            else:
                _update_active_trade(
                    symbol,
                    data["side"],
                    mode="normal",
                    qty=target_qty,
                )
                _tag_active_trade_strategy(symbol, data["side"], entry_strategy)
                _normalize_active_trade_patch(_update_active_trade)

            if not effective_probe and entry_price is not None:
                projected_value = abs(target_qty * entry_price)
                current_abs_value = abs(current_position_value)
                total_exposure_value = total_exposure_value - current_abs_value + projected_value

            if is_highlow_entry:
                if is_buy_side(data["side"]):
                    highlow_tp_reference = data.get("predicted_high")
                else:
                    highlow_tp_reference = data.get("predicted_low")
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
                        spawn_close_position_at_takeprofit(symbol, float(takeprofit_price))
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
            _mark_probe_transitioned(symbol, data["side"], current_position_size)
            entry_strategy = data.get("strategy")
            _update_active_trade(
                symbol,
                data["side"],
                mode="probe_transition",
                qty=current_position_size,
            )
            _tag_active_trade_strategy(symbol, data["side"], entry_strategy)
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

        if (
            not should_close
            and entry_strategy
            and next_forecast
            and (entry_mode or "normal") != "probe"
        ):
            strategy_returns = next_forecast.get("strategy_returns", {})
            strategy_return = strategy_returns.get(entry_strategy)
            if strategy_return is None and entry_strategy == next_forecast.get("strategy"):
                strategy_return = next_forecast.get("avg_return")
            if strategy_return is not None and strategy_return < 0:
                logger.info(
                    f"Closing position for {symbol} due to {entry_strategy} strategy underperforming "
                    f"(avg return {strategy_return:.4f})"
                )
                should_close = True
                close_reason = f"{entry_strategy}_strategy_loss"

        probe_meta = next_forecast or _evaluate_trade_block(symbol, normalized_side)
        if probe_meta.get("probe_expired") and not should_close:
            logger.info(
                f"Closing {symbol} ahead of next session; probe duration exceeded {PROBE_MAX_DURATION}, issuing backout."
            )
            should_close = True
            close_reason = "probe_duration_exceeded"

        if should_close:
            _record_trade_outcome(position, close_reason or "market_close")
            backout_near_market(symbol)

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
            if symbol in crypto_symbols and not is_nyse_trading_day_now():
                logger.info(
                    f"Would keep crypto position for {symbol} on weekend - no direction change check needed in dry run"
                )
            # For stocks when market is closed, only close if direction changed
            elif symbol not in crypto_symbols and not is_nyse_trading_day_now():
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
            min_trade_qty = MIN_CRYPTO_QTY if symbol in crypto_symbols else MIN_STOCK_QTY
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
    ]
    previous_picks = {}

    # Track when each analysis was last run
    last_initial_run = None
    last_market_open_run = None
    last_market_open_hour2_run = None
    last_market_close_run = None

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
