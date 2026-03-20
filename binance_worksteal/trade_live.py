#!/usr/bin/env python3
"""Live work-stealing daily trading bot for Binance margin.

Runs daily at UTC midnight:
1. Fetch latest daily bars for all symbols
2. Compute dip targets and proximity scores
3. Place limit orders for best candidates
4. Manage exits (profit target, stop loss, trailing stop, max hold)
5. Handle FDUSD<->USDT swaps for BTC/ETH execution
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from loguru import logger

from binance_worksteal.strategy import (
    WorkStealConfig,
    build_entry_candidates,
    compute_market_breadth_skip,
    get_fee,
    load_daily_bars,
    resolve_entry_config,
    FDUSD_SYMBOLS,
)

# Binance API
try:
    from binance.client import Client as BinanceClient
    from binance.enums import *
except ImportError:
    logger.warning("binance package not installed, using mock mode")
    BinanceClient = None

STATE_FILE = Path("binance_worksteal/live_state.json")
LOG_FILE = Path("binance_worksteal/trade_log.jsonl")
EVENTS_FILE = Path("binance_worksteal/events.jsonl")


# Default config (best from 30-symbol sweep)
DEFAULT_CONFIG = WorkStealConfig(
    dip_pct=0.20,
    proximity_pct=0.02,
    profit_target_pct=0.15,
    stop_loss_pct=0.10,
    max_positions=5,
    max_hold_days=14,
    lookback_days=20,
    ref_price_method="high",
    sma_filter_period=20,
    trailing_stop_pct=0.03,
    max_drawdown_exit=0.25,
    enable_shorts=False,
    max_leverage=1.0,
    maker_fee=0.001,
    fdusd_fee=0.0,
    initial_cash=10000.0,
    entry_proximity_bps=3000.0,
    risk_off_ref_price_method="high",
    risk_off_market_breadth_filter=0.70,
    risk_off_trigger_sma_period=30,
    risk_off_trigger_momentum_period=7,
    rebalance_seeded_positions=True,
)

# Symbol -> Binance trading pair mapping
SYMBOL_PAIRS = {
    "BTCUSD": {"fdusd": "BTCFDUSD", "usdt": "BTCUSDT"},
    "ETHUSD": {"fdusd": "ETHFDUSD", "usdt": "ETHUSDT"},
    "SOLUSD": {"fdusd": "SOLFDUSD", "usdt": "SOLUSDT"},
    "BNBUSD": {"fdusd": "BNBFDUSD", "usdt": "BNBUSDT"},
}
# All other symbols use USDT pairs
DEFAULT_QUOTE = "usdt"
PENDING_ENTRY_TTL = timedelta(days=1)


def _normalize_strategy_symbol(symbol: str) -> str:
    value = str(symbol or "").strip().upper()
    if not value:
        return value
    if value.endswith("FDUSD"):
        return f"{value[:-5]}USD"
    if value.endswith("USDT"):
        return f"{value[:-4]}USD"
    if value.endswith("USD"):
        return value
    return f"{value}USD"


def _relative_bps_distance(reference_price: float, candidate_price: float) -> float:
    ref = float(reference_price or 0.0)
    if ref <= 0.0:
        return float("inf")
    return abs(float(candidate_price) - ref) / ref * 10_000.0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--daemon", action="store_true")
    parser.add_argument("--dip-pct", type=float, default=DEFAULT_CONFIG.dip_pct)
    parser.add_argument("--proximity-pct", type=float, default=DEFAULT_CONFIG.proximity_pct)
    parser.add_argument("--profit-target", type=float, default=DEFAULT_CONFIG.profit_target_pct)
    parser.add_argument("--stop-loss", type=float, default=DEFAULT_CONFIG.stop_loss_pct)
    parser.add_argument("--max-positions", type=int, default=DEFAULT_CONFIG.max_positions)
    parser.add_argument("--max-position-pct", type=float, default=DEFAULT_CONFIG.max_position_pct)
    parser.add_argument("--max-hold-days", type=int, default=DEFAULT_CONFIG.max_hold_days)
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_CONFIG.lookback_days)
    parser.add_argument("--ref-method", default=DEFAULT_CONFIG.ref_price_method)
    parser.add_argument("--sma-filter", type=int, default=DEFAULT_CONFIG.sma_filter_period)
    parser.add_argument("--market-breadth-filter", type=float, default=DEFAULT_CONFIG.market_breadth_filter)
    parser.add_argument("--trailing-stop", type=float, default=DEFAULT_CONFIG.trailing_stop_pct)
    parser.add_argument("--entry-proximity-bps", type=float, default=DEFAULT_CONFIG.entry_proximity_bps)
    parser.add_argument("--risk-off-ref-method", default=DEFAULT_CONFIG.risk_off_ref_price_method)
    parser.add_argument(
        "--risk-off-market-breadth-filter",
        type=float,
        default=DEFAULT_CONFIG.risk_off_market_breadth_filter,
    )
    parser.add_argument(
        "--risk-off-trigger-sma-period",
        type=int,
        default=DEFAULT_CONFIG.risk_off_trigger_sma_period,
    )
    parser.add_argument(
        "--risk-off-trigger-momentum-period",
        type=int,
        default=DEFAULT_CONFIG.risk_off_trigger_momentum_period,
    )
    parser.add_argument(
        "--rebalance-seeded-positions",
        dest="rebalance_seeded_positions",
        action="store_true",
        default=DEFAULT_CONFIG.rebalance_seeded_positions,
    )
    parser.add_argument(
        "--no-rebalance-seeded-positions",
        dest="rebalance_seeded_positions",
        action="store_false",
    )
    parser.add_argument("--run-on-start", dest="run_on_start", action="store_true", default=True)
    parser.add_argument("--no-run-on-start", dest="run_on_start", action="store_false")
    parser.add_argument("--startup-preview-only", dest="startup_preview_only", action="store_true", default=True)
    parser.add_argument("--startup-live-cycle", dest="startup_preview_only", action="store_false")
    parser.add_argument("--gemini", action="store_true", help="Enable Gemini LLM overlay")
    parser.add_argument("--gemini-model", default="gemini-2.5-flash")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--entry-poll-hours", type=int, default=4)
    parser.add_argument("--health-report-hours", type=int, default=6)
    return parser


def build_runtime_config(args: argparse.Namespace) -> WorkStealConfig:
    return WorkStealConfig(
        dip_pct=float(args.dip_pct),
        proximity_pct=float(args.proximity_pct),
        profit_target_pct=float(args.profit_target),
        stop_loss_pct=float(args.stop_loss),
        max_positions=int(args.max_positions),
        max_position_pct=float(args.max_position_pct),
        max_hold_days=int(args.max_hold_days),
        lookback_days=int(args.lookback_days),
        ref_price_method=str(args.ref_method),
        sma_filter_period=int(args.sma_filter),
        market_breadth_filter=float(args.market_breadth_filter),
        trailing_stop_pct=float(args.trailing_stop),
        max_drawdown_exit=DEFAULT_CONFIG.max_drawdown_exit,
        enable_shorts=DEFAULT_CONFIG.enable_shorts,
        max_leverage=DEFAULT_CONFIG.max_leverage,
        maker_fee=DEFAULT_CONFIG.maker_fee,
        fdusd_fee=DEFAULT_CONFIG.fdusd_fee,
        initial_cash=DEFAULT_CONFIG.initial_cash,
        entry_proximity_bps=float(args.entry_proximity_bps),
        risk_off_ref_price_method=str(args.risk_off_ref_method),
        risk_off_market_breadth_filter=float(args.risk_off_market_breadth_filter),
        risk_off_trigger_sma_period=int(args.risk_off_trigger_sma_period),
        risk_off_trigger_momentum_period=int(args.risk_off_trigger_momentum_period),
        rebalance_seeded_positions=bool(args.rebalance_seeded_positions),
    )


def normalize_live_positions(raw_positions: dict, config: WorkStealConfig) -> dict:
    normalized = {}
    for raw_symbol, raw_position in (raw_positions or {}).items():
        if not isinstance(raw_position, dict):
            continue
        symbol = _normalize_strategy_symbol(raw_symbol)
        entry_price = float(raw_position.get("entry_price", 0.0) or 0.0)
        quantity = float(raw_position.get("quantity", 0.0) or 0.0)
        if not symbol or entry_price <= 0.0 or quantity <= 0.0:
            continue
        normalized[symbol] = {
            "entry_price": entry_price,
            "entry_date": str(raw_position.get("entry_date") or datetime.now(timezone.utc).isoformat()),
            "quantity": quantity,
            "peak_price": float(raw_position.get("peak_price", entry_price) or entry_price),
            "target_sell": float(
                raw_position.get("target_sell", entry_price * (1.0 + config.profit_target_pct))
                or entry_price * (1.0 + config.profit_target_pct)
            ),
            "stop_price": float(
                raw_position.get("stop_price", entry_price * (1.0 - config.stop_loss_pct))
                or entry_price * (1.0 - config.stop_loss_pct)
            ),
            "source": str(raw_position.get("source") or "legacy"),
        }
    return normalized


def _normalize_pending_entries(raw_pending: dict) -> dict:
    normalized = {}
    for raw_symbol, raw_entry in (raw_pending or {}).items():
        if not isinstance(raw_entry, dict):
            continue
        symbol = _normalize_strategy_symbol(raw_symbol)
        if not symbol:
            continue
        normalized[symbol] = {
            "buy_price": float(raw_entry.get("buy_price", 0.0) or 0.0),
            "quantity": float(raw_entry.get("quantity", 0.0) or 0.0),
            "target_sell": float(raw_entry.get("target_sell", 0.0) or 0.0),
            "stop_price": float(raw_entry.get("stop_price", 0.0) or 0.0),
            "placed_at": str(raw_entry.get("placed_at") or datetime.now(timezone.utc).isoformat()),
            "expires_at": str(raw_entry.get("expires_at") or (datetime.now(timezone.utc) + PENDING_ENTRY_TTL).isoformat()),
            "order_id": raw_entry.get("order_id"),
            "confidence": float(raw_entry.get("confidence", 1.0) or 1.0),
            "source": str(raw_entry.get("source") or "rule"),
            "status": str(raw_entry.get("status") or "staged"),
        }
    return normalized


def plan_legacy_rebalance_exits(
    *,
    now: datetime,
    positions: dict,
    current_bars: Dict[str, pd.Series],
    history: Dict[str, pd.DataFrame],
    last_exit: dict,
    config: WorkStealConfig,
) -> tuple[list[tuple[str, float, str, dict]], set[str]]:
    legacy_config = replace(config, sma_filter_period=0)
    entry_config = resolve_entry_config(current_bars=current_bars, history=history, config=legacy_config)
    if compute_market_breadth_skip(current_bars, history, entry_config):
        return [], set()

    candidates = build_entry_candidates(
        date=pd.Timestamp(now) if not isinstance(now, pd.Timestamp) else now,
        current_bars=current_bars,
        history=history,
        positions={},
        last_exit={_normalize_strategy_symbol(sym): pd.Timestamp(ts) for sym, ts in (last_exit or {}).items()},
        config=entry_config,
        base_symbol=None,
    )
    rebalance_symbols = {sym for sym, direction, *_rest in candidates if direction == "long"}
    exits = []
    for sym, position in positions.items():
        if str(position.get("source", "legacy")) != "legacy":
            continue
        if sym in rebalance_symbols:
            position["source"] = "strategy"
            continue
        close_price = float(current_bars[sym]["close"])
        exits.append((sym, close_price, "legacy_rebalance", position))
    return exits, {sym for sym, position in positions.items() if position.get("source") == "legacy"}


def get_binance_pair(symbol: str, prefer_fdusd: bool = True) -> str:
    base = symbol.replace("USD", "")
    if prefer_fdusd and symbol in FDUSD_SYMBOLS and symbol in SYMBOL_PAIRS:
        return SYMBOL_PAIRS[symbol]["fdusd"]
    return f"{base}USDT"


def fetch_daily_bars(client, symbol: str, lookback_days: int = 30) -> pd.DataFrame:
    if client is None:
        for data_dir in ("trainingdatadailybinance", "trainingdata/train"):
            local = load_daily_bars(data_dir, [symbol]).get(symbol)
            if local is not None and not local.empty:
                return local.tail(lookback_days + 5).copy()
        return pd.DataFrame()

    pair = get_binance_pair(symbol, prefer_fdusd=False)  # always use USDT for data
    try:
        klines = client.get_klines(
            symbol=pair,
            interval="1d",
            limit=lookback_days + 5,
        )
    except Exception as e:
        logger.error(f"Failed to fetch klines for {pair}: {e}")
        return pd.DataFrame()

    rows = []
    for k in klines:
        rows.append({
            "timestamp": pd.Timestamp(k[0], unit="ms", tz="UTC"),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
            "symbol": symbol,
        })
    return pd.DataFrame(rows)


def load_state() -> dict:
    if STATE_FILE.exists():
        payload = json.loads(STATE_FILE.read_text())
    else:
        payload = {}
    payload.setdefault("positions", {})
    payload.setdefault("pending_entries", {})
    payload.setdefault("last_exit", {})
    payload.setdefault("recent_trades", [])
    payload.setdefault("peak_equity", 0.0)
    return payload


def save_state(state: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def log_trade(trade: dict):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(trade, default=str) + "\n")


def log_event(event: dict):
    event.setdefault("ts", datetime.now(timezone.utc).isoformat())
    EVENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(EVENTS_FILE, "a") as f:
        f.write(json.dumps(event, default=str) + "\n")


def _fetch_all_bars(client, symbols: List[str], lookback_days: int) -> Dict[str, pd.DataFrame]:
    all_bars = {}
    for sym in symbols:
        bars = fetch_daily_bars(client, sym, lookback_days + 10)
        if not bars.empty and len(bars) > lookback_days:
            all_bars[sym] = bars
    return all_bars


def get_account_equity(client) -> float:
    try:
        info = client.get_margin_account()
        return float(info["totalNetAssetOfBtc"]) * float(
            client.get_symbol_ticker(symbol="BTCUSDT")["price"]
        )
    except Exception as e:
        logger.error(f"Failed to get equity: {e}")
        return 0


def swap_fdusd_to_usdt(client, amount: float):
    """Swap FDUSD to USDT (1:1) if needed for margin operations."""
    try:
        client.create_order(
            symbol="FDUSDUSDT", side="SELL",
            type="MARKET", quantity=f"{amount:.2f}",
        )
        logger.info(f"Swapped {amount:.2f} FDUSD -> USDT")
    except Exception as e:
        logger.warning(f"FDUSD->USDT swap failed: {e}")


def swap_usdt_to_fdusd(client, amount: float):
    """Swap USDT to FDUSD (1:1) for 0% fee trading."""
    try:
        client.create_order(
            symbol="FDUSDUSDT", side="BUY",
            type="MARKET", quantity=f"{amount:.2f}",
        )
        logger.info(f"Swapped {amount:.2f} USDT -> FDUSD")
    except Exception as e:
        logger.warning(f"USDT->FDUSD swap failed: {e}")


def place_limit_buy(client, symbol: str, price: float, quantity: float, config: WorkStealConfig):
    pair = get_binance_pair(symbol, prefer_fdusd=True)
    logger.info(f"Placing limit buy: {pair} qty={quantity:.6f} price={price:.2f}")
    try:
        order = client.create_margin_order(
            symbol=pair, side="BUY", type="LIMIT",
            timeInForce="GTC",
            quantity=f"{quantity:.6f}",
            price=f"{price:.2f}",
        )
        return order
    except Exception as e:
        logger.error(f"Limit buy failed for {pair}: {e}")
        return None


def place_limit_sell(client, symbol: str, price: float, quantity: float):
    pair = get_binance_pair(symbol, prefer_fdusd=True)
    logger.info(f"Placing limit sell: {pair} qty={quantity:.6f} price={price:.2f}")
    try:
        order = client.create_margin_order(
            symbol=pair, side="SELL", type="LIMIT",
            timeInForce="GTC",
            quantity=f"{quantity:.6f}",
            price=f"{price:.2f}",
        )
        return order
    except Exception as e:
        logger.error(f"Limit sell failed for {pair}: {e}")
        return None


def _cancel_pending_entry(client, symbol: str, entry: dict):
    order_id = entry.get("order_id")
    if client is None or order_id is None:
        return
    pair = get_binance_pair(symbol, prefer_fdusd=True)
    try:
        client.cancel_margin_order(symbol=pair, orderId=order_id)
        logger.info(f"Cancelled pending entry: {pair} orderId={order_id}")
    except Exception as exc:
        logger.warning(f"Failed to cancel pending entry for {pair}: {exc}")


def _pending_entry_to_position(entry: dict, *, entry_time: datetime) -> dict:
    price = float(entry.get("buy_price", 0.0) or 0.0)
    quantity = float(entry.get("quantity", 0.0) or 0.0)
    return {
        "entry_price": price,
        "entry_date": entry_time.isoformat(),
        "quantity": quantity,
        "peak_price": price,
        "target_sell": float(entry.get("target_sell", price)),
        "stop_price": float(entry.get("stop_price", price)),
        "source": str(entry.get("source") or "strategy"),
    }


def reconcile_pending_entries(
    *,
    client,
    pending_entries: dict,
    positions: dict,
    now: datetime,
    dry_run: bool,
) -> list[dict]:
    recent_trades: list[dict] = []
    for sym, entry in list(pending_entries.items()):
        expires_at = pd.Timestamp(entry.get("expires_at", now.isoformat()))
        if expires_at.tzinfo is None:
            expires_at = expires_at.tz_localize("UTC")
        else:
            expires_at = expires_at.tz_convert("UTC")
        if pd.Timestamp(now) >= expires_at:
            _cancel_pending_entry(client, sym, entry)
            del pending_entries[sym]
            continue

        if dry_run or client is None or entry.get("order_id") is None:
            continue

        pair = get_binance_pair(sym, prefer_fdusd=True)
        try:
            order = client.get_margin_order(symbol=pair, orderId=entry["order_id"])
        except Exception as exc:
            logger.warning(f"Failed to refresh pending entry {pair}: {exc}")
            continue

        status = str(order.get("status", "")).upper()
        if status == "FILLED":
            fill_qty = float(order.get("executedQty") or entry.get("quantity") or 0.0)
            fill_price = float(order.get("price") or entry.get("buy_price") or 0.0)
            if fill_qty <= 0.0 or fill_price <= 0.0:
                del pending_entries[sym]
                continue
            realized = dict(entry)
            realized["quantity"] = fill_qty
            realized["buy_price"] = fill_price
            positions[sym] = _pending_entry_to_position(realized, entry_time=now)
            trade = {
                "timestamp": now.isoformat(),
                "symbol": sym,
                "side": "buy",
                "price": fill_price,
                "quantity": fill_qty,
                "pnl": 0.0,
                "reason": f"pending_fill({entry.get('source', 'rule')})",
            }
            log_trade(trade)
            recent_trades.append(trade)
            del pending_entries[sym]
        elif status in {"CANCELED", "EXPIRED", "REJECTED"}:
            del pending_entries[sym]

    return recent_trades


def _stage_entry_candidates(
    *,
    client,
    candidates: list,
    all_bars: Dict[str, pd.DataFrame],
    staged_symbols: set,
    pending_entries: dict,
    recent_trades: list,
    entry_config: WorkStealConfig,
    config: WorkStealConfig,
    equity: float,
    now: datetime,
    dry_run: bool,
    slots: int,
    gemini_enabled: bool = False,
    gemini_model: str = "gemini-2.5-flash",
) -> dict:
    n_staged = 0
    n_already_held = 0
    n_proximity_skip = 0
    n_gemini_skip = 0

    for sym, direction, score, fill_price, bar in candidates:
        if n_staged >= slots:
            break
        if direction != "long" or sym in staged_symbols:
            n_already_held += 1
            continue
        close = float(bar["close"])
        buy_price = fill_price
        sell_target = fill_price * (1 + entry_config.profit_target_pct)
        stop = fill_price * (1 - entry_config.stop_loss_pct)
        confidence = 1.0
        source = "rule"

        if gemini_enabled:
            try:
                from binance_worksteal.gemini_overlay import (
                    build_daily_prompt, call_gemini_daily, load_forecast_daily,
                )
                fc = load_forecast_daily(sym)
                rule_signal = {"buy_target": fill_price, "dip_score": score, "ref_price": 0, "sma_ok": True}
                recent = [{"timestamp": t.get("timestamp",""), "side": t.get("side",""),
                           "symbol": t.get("symbol",""), "price": t.get("price",0),
                           "pnl": t.get("pnl",0), "reason": t.get("reason","")}
                          for t in recent_trades[-5:]]
                prompt = build_daily_prompt(
                    symbol=sym, bars=all_bars[sym], current_price=close,
                    rule_signal=rule_signal, recent_trades=recent,
                    forecast_24h=fc,
                    fee_bps=0 if sym in FDUSD_SYMBOLS else 10,
                    entry_proximity_bps=entry_config.entry_proximity_bps,
                )
                plan = call_gemini_daily(prompt, model=gemini_model)
                if plan:
                    if plan.action == "hold" and plan.confidence > 0.5:
                        logger.info(f"GEMINI SKIP {sym}: {plan.reasoning}")
                        n_gemini_skip += 1
                        continue
                    if plan.action in ("buy", "adjust") and plan.confidence > 0.3:
                        if plan.buy_price > 0:
                            buy_price = plan.buy_price
                        if plan.sell_price > 0:
                            sell_target = plan.sell_price
                        if plan.stop_price > 0:
                            stop = plan.stop_price
                        confidence = plan.confidence
                        source = f"gemini(conf={confidence:.2f})"
                        logger.info(f"GEMINI {sym}: {plan.action} buy=${buy_price:.2f} "
                                    f"tp=${sell_target:.2f} sl=${stop:.2f} conf={confidence:.2f} "
                                    f"reason={plan.reasoning}")
            except Exception as e:
                logger.warning(f"Gemini call failed for {sym}: {e}")

        dist_bps = _relative_bps_distance(close, buy_price)
        if dist_bps > float(entry_config.entry_proximity_bps):
            logger.info(f"SKIP {sym}: entry {buy_price:.4f} is {dist_bps:.0f}bps from close {close:.4f}")
            n_proximity_skip += 1
            continue

        fee_rate = get_fee(sym, config)
        alloc = equity * entry_config.max_position_pct
        quantity = alloc / (buy_price * (1 + fee_rate)) * min(confidence, 1.0)
        if quantity <= 0:
            continue

        logger.info(
            f"STAGE {sym}: buy limit at {buy_price:.2f} "
            f"(close={close:.2f}, score={score:.4f}, qty={quantity:.6f}, {source})"
        )

        order = None
        if not dry_run:
            order = place_limit_buy(client, sym, buy_price, quantity, entry_config)

        pending_entries[sym] = {
            "buy_price": buy_price,
            "placed_at": now.isoformat(),
            "expires_at": (now + PENDING_ENTRY_TTL).isoformat(),
            "quantity": quantity,
            "target_sell": sell_target,
            "stop_price": stop,
            "confidence": confidence,
            "source": source,
            "order_id": None if order is None else order.get("orderId"),
            "status": "preview" if dry_run else "open",
        }
        staged_symbols.add(sym)
        trade = {
            "timestamp": now.isoformat(), "symbol": sym, "side": "staged_buy",
            "price": buy_price, "quantity": quantity,
            "reason": f"dip_buy({source})",
            "dry_run": dry_run,
        }
        log_trade(trade)
        recent_trades.append(trade)
        n_staged += 1

    return {
        "n_staged": n_staged,
        "n_already_held": n_already_held,
        "n_proximity_skip": n_proximity_skip,
        "n_gemini_skip": n_gemini_skip,
    }


def run_health_report(client, symbols: List[str], config: WorkStealConfig, dry_run: bool = True):
    state = load_state()
    positions = normalize_live_positions(state.get("positions", {}), config)
    pending_entries = _normalize_pending_entries(state.get("pending_entries", {}))
    recent_trades = list(state.get("recent_trades", []))
    now = datetime.now(timezone.utc)
    equity = get_account_equity(client) if not dry_run else config.initial_cash

    last_trade_ts = None
    for t in reversed(recent_trades):
        ts_str = t.get("timestamp")
        if ts_str:
            last_trade_ts = pd.Timestamp(ts_str)
            break
    days_since_trade = (now - last_trade_ts).days if last_trade_ts else -1

    all_bars = _fetch_all_bars(client, symbols, config.lookback_days)
    current_bars = {sym: bars.iloc[-1] for sym, bars in all_bars.items()}
    entry_config = resolve_entry_config(current_bars=current_bars, history=all_bars, config=config)
    risk_off = compute_market_breadth_skip(current_bars, all_bars, entry_config)

    from binance_worksteal.strategy import compute_ref_price
    nearest_dip_bps = float("inf")
    nearest_dip_sym = ""
    for sym, bars in all_bars.items():
        if sym in positions:
            continue
        close = float(bars.iloc[-1]["close"])
        ref_high = compute_ref_price(bars, config.ref_price_method, config.lookback_days)
        buy_target = ref_high * (1 - config.dip_pct)
        dist_bps = (close - buy_target) / close * 10_000.0 if close > 0 else float("inf")
        if dist_bps < nearest_dip_bps:
            nearest_dip_bps = dist_bps
            nearest_dip_sym = sym

    logger.info(
        f"HEALTH: equity=${equity:.0f} positions={len(positions)} pending={len(pending_entries)} "
        f"regime={'risk-off' if risk_off else 'risk-on'} "
        f"nearest_dip={nearest_dip_sym}@{nearest_dip_bps:.0f}bps "
        f"days_since_trade={days_since_trade}"
    )
    log_event({
        "type": "health_report",
        "equity": equity,
        "n_positions": len(positions),
        "n_pending": len(pending_entries),
        "risk_off": risk_off,
        "nearest_dip_sym": nearest_dip_sym,
        "nearest_dip_bps": nearest_dip_bps,
        "days_since_trade": days_since_trade,
        "n_symbols_with_data": len(all_bars),
    })


def run_entry_scan(client, symbols: List[str], config: WorkStealConfig,
                   dry_run: bool = True, gemini_enabled: bool = False,
                   gemini_model: str = "gemini-2.5-flash"):
    state = load_state()
    positions = normalize_live_positions(state.get("positions", {}), config)
    pending_entries = _normalize_pending_entries(state.get("pending_entries", {}))
    last_exit = {_normalize_strategy_symbol(sym): ts for sym, ts in state.get("last_exit", {}).items()}
    recent_trades = list(state.get("recent_trades", []))
    now = datetime.now(timezone.utc)

    slots = config.max_positions - len(positions) - len(pending_entries)
    if slots <= 0:
        logger.info(f"ENTRY SCAN: skipped, {len(positions)} positions + {len(pending_entries)} pending >= {config.max_positions} max")
        return

    all_bars = _fetch_all_bars(client, symbols, config.lookback_days)
    equity = get_account_equity(client) if not dry_run else config.initial_cash
    current_bars = {sym: bars.iloc[-1] for sym, bars in all_bars.items()}
    entry_config = resolve_entry_config(current_bars=current_bars, history=all_bars, config=config)

    if compute_market_breadth_skip(current_bars, all_bars, entry_config):
        logger.info("ENTRY SCAN: market breadth risk-off, no entries")
        log_event({"type": "entry_scan", "n_checked": len(all_bars), "n_candidates": 0, "n_staged": 0, "skip_reason": "market_breadth_risk_off"})
        return

    staged_symbols = set(positions) | set(pending_entries)
    candidates = build_entry_candidates(
        date=pd.Timestamp(now),
        current_bars=current_bars,
        history=all_bars,
        positions={},
        last_exit={sym: pd.Timestamp(ts) for sym, ts in last_exit.items()},
        config=entry_config,
        base_symbol=None,
    )
    logger.info(f"ENTRY SCAN: {len(all_bars)} symbols checked, {len(candidates)} candidates found")

    counts = _stage_entry_candidates(
        client=client, candidates=candidates, all_bars=all_bars,
        staged_symbols=staged_symbols, pending_entries=pending_entries,
        recent_trades=recent_trades, entry_config=entry_config, config=config,
        equity=equity, now=now, dry_run=dry_run, slots=slots,
        gemini_enabled=gemini_enabled, gemini_model=gemini_model,
    )

    logger.info(
        f"ENTRY SCAN SUMMARY: candidates={len(candidates)} staged={counts['n_staged']} "
        f"already_held={counts['n_already_held']} proximity_skip={counts['n_proximity_skip']} "
        f"gemini_skip={counts['n_gemini_skip']}"
    )
    log_event({
        "type": "entry_scan",
        "n_checked": len(all_bars),
        "n_candidates": len(candidates),
        "slots_available": slots,
        **counts,
    })

    if counts["n_staged"] > 0:
        state["pending_entries"] = pending_entries
        state["recent_trades"] = recent_trades[-50:]
        save_state(state)


def run_daily_cycle(client, symbols: List[str], config: WorkStealConfig,
                    dry_run: bool = True, gemini_enabled: bool = False,
                    gemini_model: str = "gemini-2.5-flash"):
    state = load_state()
    positions = normalize_live_positions(state.get("positions", {}), config)
    pending_entries = _normalize_pending_entries(state.get("pending_entries", {}))
    last_exit = {_normalize_strategy_symbol(sym): ts for sym, ts in state.get("last_exit", {}).items()}
    recent_trades = list(state.get("recent_trades", []))

    now = datetime.now(timezone.utc)
    logger.info(
        f"Daily cycle at {now.isoformat()}, {len(positions)} open positions, "
        f"{len(pending_entries)} pending entries"
    )

    all_bars = _fetch_all_bars(client, symbols, config.lookback_days)
    logger.info(f"Fetched bars for {len(all_bars)}/{len(symbols)} symbols")

    equity = get_account_equity(client) if not dry_run else config.initial_cash
    recent_trades.extend(
        reconcile_pending_entries(
            client=client,
            pending_entries=pending_entries,
            positions=positions,
            now=now,
            dry_run=dry_run,
        )
    )

    if len(positions) >= config.max_positions and pending_entries:
        for sym, entry in list(pending_entries.items()):
            _cancel_pending_entry(client, sym, entry)
            del pending_entries[sym]

    # Check exits
    exits_to_process = []
    current_bars = {sym: bars.iloc[-1] for sym, bars in all_bars.items()}
    legacy_exits, _legacy_rebalance = plan_legacy_rebalance_exits(
        now=now,
        positions=positions,
        current_bars=current_bars,
        history=all_bars,
        last_exit=last_exit,
        config=config,
    )
    exits_to_process.extend(legacy_exits)

    for sym, pos in list(positions.items()):
        if sym not in all_bars:
            continue
        bars = all_bars[sym]
        close = float(bars.iloc[-1]["close"])
        high = float(bars.iloc[-1]["high"])
        low = float(bars.iloc[-1]["low"])

        entry_price = pos["entry_price"]
        entry_date = pd.Timestamp(pos["entry_date"])
        peak = max(pos.get("peak_price", entry_price), high)
        pos["peak_price"] = peak

        exit_price = None
        exit_reason = ""

        # Profit target
        target = entry_price * (1 + config.profit_target_pct)
        if high >= target:
            exit_price = target
            exit_reason = "profit_target"
        # Stop loss
        elif low <= entry_price * (1 - config.stop_loss_pct):
            exit_price = entry_price * (1 - config.stop_loss_pct)
            exit_reason = "stop_loss"
        # Trailing stop
        elif config.trailing_stop_pct > 0:
            trail = peak * (1 - config.trailing_stop_pct)
            if low <= trail:
                exit_price = trail
                exit_reason = "trailing_stop"
        # Max hold
        if exit_price is None and config.max_hold_days > 0:
            held = (now - entry_date).days
            if held >= config.max_hold_days:
                exit_price = close
                exit_reason = "max_hold"

        if exit_price is not None:
            exits_to_process.append((sym, exit_price, exit_reason, pos))

    seen_exits = set()
    for sym, exit_price, reason, pos in exits_to_process:
        if sym in seen_exits:
            continue
        seen_exits.add(sym)
        logger.info(f"EXIT {sym}: {reason} at {exit_price:.2f} (entry {pos['entry_price']:.2f})")
        if not dry_run:
            place_limit_sell(client, sym, exit_price, pos["quantity"])
        trade = {
            "timestamp": now.isoformat(), "symbol": sym, "side": "sell",
            "price": exit_price, "quantity": pos["quantity"],
            "reason": reason, "pnl": (exit_price - pos["entry_price"]) * pos["quantity"],
            "dry_run": dry_run,
        }
        log_trade(trade)
        recent_trades.append(trade)
        last_exit[sym] = now.isoformat()
        del positions[sym]

    # Stage new entries
    entry_config = resolve_entry_config(current_bars=current_bars, history=all_bars, config=config)
    skip_entries = compute_market_breadth_skip(current_bars, all_bars, entry_config)
    counts = {"n_staged": 0, "n_proximity_skip": 0, "n_gemini_skip": 0, "n_already_held": 0}
    n_candidates = 0

    if len(positions) >= config.max_positions:
        logger.info(f"ENTRY SCAN: skipped, max positions ({config.max_positions}) reached")
    elif skip_entries:
        logger.info("ENTRY SCAN: skipped, market breadth risk-off")
    else:
        staged_symbols = set(positions) | set(pending_entries)
        candidates = build_entry_candidates(
            date=pd.Timestamp(now),
            current_bars=current_bars,
            history=all_bars,
            positions={},
            last_exit={sym: pd.Timestamp(ts) for sym, ts in last_exit.items()},
            config=entry_config,
            base_symbol=None,
        )
        n_candidates = len(candidates)
        slots = config.max_positions - len(positions) - len(pending_entries)
        logger.info(f"ENTRY SCAN: {len(all_bars)} symbols checked, {n_candidates} candidates found")

        counts = _stage_entry_candidates(
            client=client, candidates=candidates, all_bars=all_bars,
            staged_symbols=staged_symbols, pending_entries=pending_entries,
            recent_trades=recent_trades, entry_config=entry_config, config=config,
            equity=equity, now=now, dry_run=dry_run, slots=slots,
            gemini_enabled=gemini_enabled, gemini_model=gemini_model,
        )

    logger.info(
        f"ENTRY SUMMARY: candidates={n_candidates} staged={counts['n_staged']} "
        f"proximity_skip={counts['n_proximity_skip']} gemini_skip={counts['n_gemini_skip']} "
        f"already_held={counts['n_already_held']} risk_off={skip_entries}"
    )
    log_event({
        "type": "entry_scan",
        "n_checked": len(all_bars),
        "n_candidates": n_candidates,
        "risk_off": skip_entries,
        "n_positions": len(positions),
        "n_pending": len(pending_entries),
        "equity": equity,
        **counts,
    })

    # Save state
    state["positions"] = positions
    state["pending_entries"] = pending_entries
    state["last_exit"] = last_exit
    state["recent_trades"] = recent_trades[-50:]
    state["peak_equity"] = max(state.get("peak_equity", 0), equity)
    save_state(state)

    logger.info(
        f"Cycle complete: {len(positions)} positions, {len(pending_entries)} pending, equity=${equity:.0f}"
    )
    for sym, pos in positions.items():
        logger.info(f"  {sym}: entry={pos['entry_price']:.2f} "
                    f"target={pos['target_sell']:.2f} stop={pos['stop_price']:.2f}")


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.live:
        args.dry_run = False

    from binance_worksteal.backtest import FULL_UNIVERSE
    symbols = args.symbols or FULL_UNIVERSE

    config = build_runtime_config(args)

    # Initialize Binance client
    if not args.dry_run and BinanceClient:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from env_real import BINANCE_API_KEY, BINANCE_SECRET
            client = BinanceClient(BINANCE_API_KEY, BINANCE_SECRET)
        except ImportError:
            logger.error("env_real.py not found - need BINANCE_API_KEY and BINANCE_SECRET")
            return 1
    else:
        client = None
        logger.info("Running in DRY RUN mode")

    gemini_on = getattr(args, "gemini", False)
    g_model = getattr(args, "gemini_model", "gemini-2.5-flash")
    if gemini_on:
        logger.info(f"Gemini overlay enabled (model={g_model})")

    if args.daemon:
        entry_poll_h = int(args.entry_poll_hours)
        health_h = int(args.health_report_hours)
        logger.info(f"Starting daemon mode: daily cycle at UTC 00:00, entry scan every {entry_poll_h}h, health every {health_h}h")
        last_cycle_date = None
        last_entry_scan_hour = None
        last_health_hour = None
        if args.run_on_start:
            startup_dry_run = args.dry_run or args.startup_preview_only
            run_daily_cycle(
                client,
                symbols,
                config,
                dry_run=startup_dry_run,
                gemini_enabled=gemini_on,
                gemini_model=g_model,
            )
            if not startup_dry_run:
                last_cycle_date = datetime.now(timezone.utc).date().isoformat()
            last_entry_scan_hour = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H")
            last_health_hour = last_entry_scan_hour
        while True:
            now = datetime.now(timezone.utc)
            next_run = (now + timedelta(days=1)).replace(hour=0, minute=5, second=0, microsecond=0)

            if now.hour == 0 and now.minute < 10 and last_cycle_date != now.date().isoformat():
                run_daily_cycle(
                    client,
                    symbols,
                    config,
                    dry_run=args.dry_run,
                    gemini_enabled=gemini_on,
                    gemini_model=g_model,
                )
                last_cycle_date = now.date().isoformat()
                last_entry_scan_hour = now.strftime("%Y-%m-%dT%H")
                last_health_hour = last_entry_scan_hour
                next_run = (now + timedelta(days=1)).replace(hour=0, minute=5, second=0)
            else:
                state = load_state()
                positions = normalize_live_positions(state.get("positions", {}), config)
                pending_entries = _normalize_pending_entries(state.get("pending_entries", {}))
                before_positions = len(positions)
                before_pending = len(pending_entries)
                refreshed = reconcile_pending_entries(
                    client=client,
                    pending_entries=pending_entries,
                    positions=positions,
                    now=now,
                    dry_run=args.dry_run,
                )
                if (
                    refreshed
                    or len(positions) != before_positions
                    or len(pending_entries) != before_pending
                ):
                    state["positions"] = positions
                    state["pending_entries"] = pending_entries
                    state["recent_trades"] = list(state.get("recent_trades", []))[-50:] + refreshed
                    save_state(state)

                current_hour = now.strftime("%Y-%m-%dT%H")
                if entry_poll_h > 0 and now.hour % entry_poll_h == 0 and current_hour != last_entry_scan_hour:
                    logger.info(f"Intermediate entry scan at {now.isoformat()}")
                    run_entry_scan(
                        client, symbols, config,
                        dry_run=args.dry_run,
                        gemini_enabled=gemini_on,
                        gemini_model=g_model,
                    )
                    last_entry_scan_hour = current_hour

                if health_h > 0 and now.hour % health_h == 0 and current_hour != last_health_hour:
                    run_health_report(client, symbols, config, dry_run=args.dry_run)
                    last_health_hour = current_hour

            sleep_secs = (next_run - datetime.now(timezone.utc)).total_seconds()
            logger.info(f"Next run in {sleep_secs/3600:.1f}h at {next_run}")
            time.sleep(max(60, min(float(args.poll_seconds), sleep_secs)))
    else:
        run_daily_cycle(client, symbols, config, dry_run=args.dry_run,
                        gemini_enabled=gemini_on, gemini_model=g_model)


if __name__ == "__main__":
    sys.exit(main() or 0)
