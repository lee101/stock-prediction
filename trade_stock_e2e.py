import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import sleep
from typing import List, Dict, Optional

import pytz
from loguru import logger

import alpaca_wrapper
from backtest_test3_inline import backtest_forecasts
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

_trade_outcomes_store: Optional[FlatShelf] = None
_trade_learning_store: Optional[FlatShelf] = None
_active_trades_store: Optional[FlatShelf] = None
_trade_history_store: Optional[FlatShelf] = None


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


def _log_detail(message: str) -> None:
    if COMPACT_LOGS:
        logger.debug(message)
    else:
        logger.info(message)


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
    return _update_learning_state(symbol, side, pending_probe=True, probe_active=False)


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
        _mark_probe_completed(position.symbol, normalized_side, successful=pnl_value >= 0)
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
    learning_state["probe_transition_ready"] = probe_summary.get("probe_transition_ready")
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
        "probe_transition_ready": probe_summary.get("probe_transition_ready"),
        "learning_state": learning_state,
    }


def get_market_hours() -> tuple:
    """Get market open and close times in EST."""
    est = pytz.timezone("US/Eastern")
    now = datetime.now(est)
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
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
    last_pnl_str = f"{last_pnl:9.2f}" if isinstance(last_pnl, (int, float)) else "   n/a  "
    return (
        f"{symbol:<6} "
        f"{data.get('side', '?'):<4} "
        f"{data.get('trade_mode', 'normal'):<6} "
        f"{data.get('avg_return', 0.0):6.3f} "
        f"{data.get('composite_score', 0.0):6.3f} "
        f"{data.get('predicted_movement', 0.0):7.3f} "
        f"{_pick_confidence(data):5.3f} "
        f"{last_pnl_str} "
        f"{_pick_notes(data)}"
    )


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

    for symbol in symbols:
        try:
            _log_detail(f"Analyzing {symbol}")
            # not many because we need to adapt strats? eg the wierd spikes in uniusd are a big opportunity to trade w high/low
            # but then i bumped up because its not going to say buy crypto when its down, if its most recent based?
            num_simulations = 70

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

            strategy_returns = {
                "simple": backtest_df["simple_strategy_return"].mean(),
                "all_signals": backtest_df["all_signals_strategy_return"].mean(),
                "takeprofit": backtest_df["entry_takeprofit_return"].mean(),
                "highlow": backtest_df["highlow_return"].mean(),
            }

            unprofit_return = 0.0
            unprofit_sharpe = 0.0
            if "unprofit_shutdown_return" in backtest_df.columns:
                unprofit_return = backtest_df["unprofit_shutdown_return"].mean()
                strategy_returns["unprofit_shutdown"] = unprofit_return
            if "unprofit_shutdown_sharpe" in backtest_df.columns:
                unprofit_sharpe = backtest_df["unprofit_shutdown_sharpe"].mean()

            last_prediction = backtest_df.iloc[0]
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

            close_movement = predicted_close_price - close_price
            high_movement = predicted_high_price - close_price
            low_movement = predicted_low_price - close_price

            best_strategy = max(strategy_returns, key=strategy_returns.get)
            avg_return = strategy_returns[best_strategy]

            if best_strategy == "all_signals":
                if all(x > 0 for x in [close_movement, high_movement, low_movement]):
                    position_side = "buy"
                elif all(x < 0 for x in [close_movement, high_movement, low_movement]):
                    position_side = "sell"
                else:
                    _log_detail(f"Skipping {symbol} - mixed directional signals despite all_signals lead")
                    continue
                predicted_movement = close_movement
            else:
                predicted_movement = close_movement
                position_side = "buy" if predicted_movement > 0 else "sell"

            expected_move_pct = safe_divide(predicted_movement, close_price, default=0.0)
            simple_return = strategy_returns.get("simple", 0.0)
            takeprofit_return = strategy_returns.get("takeprofit", 0.0)
            highlow_return = strategy_returns.get("highlow", 0.0)
            simple_sharpe = 0.0
            if "simple_strategy_sharpe" in backtest_df.columns:
                simple_sharpe = coerce_numeric(backtest_df["simple_strategy_sharpe"].mean(), default=0.0)
            price_skill = max(simple_return, 0.0) + 0.25 * max(simple_sharpe, 0.0)
            abs_move = abs(expected_move_pct)
            if abs_move < MIN_EXPECTED_MOVE_PCT:
                abs_move = 0.0
            edge_strength = price_skill * abs_move
            directional_edge = edge_strength if predicted_movement >= 0 else -edge_strength

            if (
                edge_strength < MIN_EDGE_STRENGTH
                and max(avg_return, simple_return, takeprofit_return, highlow_return) <= 0
            ):
                _log_detail(
                    f"Skipping {symbol} - no actionable price edge "
                    f"(edge_strength={edge_strength:.6f}, avg_return={avg_return:.6f})"
                )
                continue

            composite_score = (
                0.2 * avg_return
                + 0.4 * simple_return
                + 0.15 * edge_strength
                + 0.1 * unprofit_return
                + 0.075 * takeprofit_return
                + 0.075 * highlow_return
            )

            block_info = _evaluate_trade_block(symbol, position_side)
            last_pnl = block_info.get("last_pnl")
            trade_blocked = block_info.get("blocked", False)

            results[symbol] = {
                "avg_return": avg_return,
                "predictions": backtest_df,
                "side": position_side,
                "predicted_movement": predicted_movement,
                "strategy": best_strategy,
                "predicted_high": float(last_prediction["predicted_high"]),
                "predicted_low": float(last_prediction["predicted_low"]),
                "strategy_returns": strategy_returns,
                "simple_return": simple_return,
                "unprofit_shutdown_return": unprofit_return,
                "unprofit_shutdown_sharpe": unprofit_sharpe,
                "expected_move_pct": expected_move_pct,
                "price_skill": price_skill,
                "edge_strength": edge_strength,
                "directional_edge": directional_edge,
                "composite_score": composite_score,
                "trade_blocked": trade_blocked,
                "block_reason": block_info.get("block_reason"),
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
            }
            _log_detail(
                f"Analysis complete for {symbol}: strategy={best_strategy}, avg_return={avg_return:.3f}, "
                f"simple={simple_return:.3f}, edge_strength={edge_strength:.5f}, composite={composite_score:.3f}, "
                f"side={position_side}"
            )
            _log_detail(
                f"Predicted movement: {predicted_movement:.3f}, expected_move_pct={expected_move_pct:.5f}, "
                f"price_skill={price_skill:.5f}, directional_edge={directional_edge:.5f}"
            )
            _log_detail(
                f"Predicted High: {last_prediction['predicted_high']:.3f}, "
                f"Predicted Low: {last_prediction['predicted_low']:.3f}, "
                f"Current Close: {last_prediction['close']:.3f}"
            )
            _log_detail(f"Predicted Close: {last_prediction['predicted_close']:.3f}")
            if trade_blocked and block_info.get("block_reason"):
                _log_detail(f"Trade blocked for {symbol}: {block_info['block_reason']}")
            if block_info.get("trade_mode") == "probe":
                _log_detail(f"Probe trade scheduled for {symbol} ({position_side}) due to recent loss")
                if block_info.get("probe_transition_ready"):
                    _log_detail(f"{symbol} probe eligible for transition to full sizing based on next-day signal")
                if block_info.get("probe_expired"):
                    _log_detail(f"{symbol} probe exceeded max duration; will trigger backout if still open")

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

    banner = "=" * 72
    logger.info(banner)
    logger.info(f"TRADING PLAN ({action})")
    logger.info("Symbol Dir  Mode   Avg    Comp    Move   Conf  LastPnL  Notes")
    for symbol, data in picks.items():
        logger.info(_format_plan_line(symbol, data))
    logger.info(banner)


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

    snapshot = record_portfolio_snapshot(total_exposure_value, day_pl=day_pl_value)
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
        logger.info("Entry candidates:\n  " + "\n  ".join(candidate_lines))
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
        current_position_size = 0
        current_position_value = 0
        for p in positions:
            if p.symbol == symbol:
                current_position_size = float(p.qty)
                if hasattr(p, "current_price"):
                    current_position_value = current_position_size * float(p.current_price)
                break

        min_trade_qty = MIN_CRYPTO_QTY if symbol in crypto_symbols else MIN_STOCK_QTY
        if effective_probe:
            logger.info(f"{symbol}: Probe mode enabled; minimum trade quantity set to {min_trade_qty}")

        # Calculate target position size
        client = StockHistoricalDataClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD)
        download_exchange_latest_data(client, symbol)
        bid_price = get_bid(symbol)
        ask_price = get_ask(symbol)
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
                if computed_qty <= 0:
                    target_qty = ensure_lower_bound(min_trade_qty, 0.0, default=min_trade_qty)
                    logger.info(
                        f"{symbol}: Exposure limits returned {computed_qty}; overriding to probe minimum qty {target_qty}"
                    )
                else:
                    target_qty = min(computed_qty, min_trade_qty)
                should_enter = not position_exists or not correct_side
                needs_size_increase = False
            else:
                target_qty = computed_qty
                target_value = target_qty * entry_price
                logger.info(
                    f"{symbol}: Current position: {current_position_size} qty (${current_position_value:.2f}), "
                    f"Target: {target_qty} qty (${target_value:.2f})"
                )
                if symbol in crypto_symbols:
                    should_enter = (not position_exists and is_buy_side(data["side"])) or (
                        current_position_size < target_qty * 0.95
                    )  # 5% tolerance
                else:
                    should_enter = not position_exists or (current_position_size < target_qty * 0.95)

                needs_size_increase = current_position_size > 0 and current_position_size < target_qty * 0.95

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
                    ramp_into_position(symbol, data["side"], target_qty=target_qty)
                else:
                    logger.info(f"Probe trade target quantity for {symbol}: {target_qty} at price {entry_price}")
                    ramp_into_position(symbol, data["side"], target_qty=target_qty)
            else:
                logger.warning(f"Could not get bid/ask prices for {symbol}, using default sizing")
                ramp_into_position(symbol, data["side"], target_qty=target_qty if effective_probe else None)

            if transition_to_normal:
                _mark_probe_transitioned(symbol, data["side"], target_qty)
                _update_active_trade(
                    symbol,
                    data["side"],
                    mode="probe_transition",
                    qty=target_qty,
                    strategy=data.get("strategy"),
                )
            elif effective_probe:
                _mark_probe_active(symbol, data["side"], target_qty)
                _update_active_trade(
                    symbol,
                    data["side"],
                    mode="probe",
                    qty=target_qty,
                    strategy=data.get("strategy"),
                )
            else:
                _update_active_trade(
                    symbol,
                    data["side"],
                    mode="normal",
                    qty=target_qty,
                    strategy=data.get("strategy"),
                )

            if not effective_probe and entry_price is not None:
                projected_value = abs(target_qty * entry_price)
                current_abs_value = abs(current_position_value)
                total_exposure_value = total_exposure_value - current_abs_value + projected_value

            # If strategy is 'takeprofit', place a takeprofit limit later
            if data["strategy"] == "takeprofit" and is_buy_side(data["side"]):
                # e.g. call close_position_at_takeprofit with predicted_high
                tp_price = data["predicted_high"]
                logger.info(f"Scheduling a takeprofit at {tp_price:.3f} for {symbol}")
                # call the new function from alpaca_cli
                spawn_close_position_at_takeprofit(symbol, tp_price)
            elif data["strategy"] == "takeprofit" and is_sell_side(data["side"]):
                # If short, we might want to place a limit buy at predicted_low
                # (though you'd need to store predicted_low similarly)
                # For example:
                predicted_low = data["predictions"].iloc[-1]["predicted_low"]
                logger.info(f"Scheduling a takeprofit at {predicted_low:.3f} for short {symbol}")
                spawn_close_position_at_takeprofit(symbol, predicted_low)

            # If strategy is 'highlow', place a limit order at predicted_low (for buys)
            # or predicted_high (for shorts), and then schedule a takeprofit at the opposite predicted price.
            elif data["strategy"] == "highlow":
                if data["side"] == "buy":
                    entry_price = data["predicted_low"]
                    logger.info(f"(Highlow) Placing limit BUY order for {symbol} at predicted_low={entry_price:.2f}")
                    qty = get_qty(symbol, entry_price, positions)
                    alpaca_wrapper.open_order_at_price_or_all(symbol, qty=qty, side="buy", price=entry_price)

                    tp_price = data["predicted_high"]
                    logger.info(f"(Highlow) Scheduling takeprofit at predicted_high={tp_price:.3f} for {symbol}")
                    spawn_close_position_at_takeprofit(symbol, tp_price)
                else:
                    entry_price = data["predicted_high"]
                    logger.info(
                        f"(Highlow) Placing limit SELL/short order for {symbol} at predicted_high={entry_price:.2f}"
                    )
                    qty = get_qty(symbol, entry_price, positions)
                    alpaca_wrapper.open_order_at_price_or_all(symbol, qty=qty, side="sell", price=entry_price)

                    tp_price = data["predicted_low"]
                    logger.info(f"(Highlow) Scheduling takeprofit at predicted_low={tp_price:.3f} for short {symbol}")
                    spawn_close_position_at_takeprofit(symbol, tp_price)
        elif transition_to_normal:
            logger.info(
                f"{symbol}: Probe already at target sizing; marking transition complete without additional orders."
            )
            _mark_probe_transitioned(symbol, data["side"], current_position_size)
            _update_active_trade(
                symbol,
                data["side"],
                mode="probe_transition",
                qty=current_position_size,
                strategy=data.get("strategy"),
            )


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
        "CRWD",
        "ADBE",
        "NET",
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

            # Market close analysis (15:45-16:00 EST)
            elif (
                (now.hour == market_close.hour - 1 and now.minute >= 45)
                and (last_market_close_run is None or last_market_close_run != today)
                and is_nyse_trading_day_ending()
            ):
                logger.info("\nMARKET CLOSE ANALYSIS STARTING...")
                all_analyzed_results = analyze_symbols(symbols)
                previous_picks = manage_market_close(symbols, previous_picks, all_analyzed_results)
                last_market_close_run = today

            sleep(60)

        except Exception as e:
            logger.exception(f"Error in main loop: {str(e)}")
            sleep(60)


if __name__ == "__main__":
    main()
