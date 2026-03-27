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
import concurrent.futures
import json
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from loguru import logger

from binance_worksteal.strategy import (
    WorkStealConfig,
    SymbolDiagnostic,
    build_entry_candidates,
    build_tiered_entry_candidates,
    compute_breadth_ratio,
    compute_market_breadth_skip,
    compute_ref_price,
    compute_sma,
    get_fee,
    load_daily_bars,
    passes_sma_filter,
    resolve_entry_config,
    FDUSD_SYMBOLS,
    _risk_off_triggered,
)
from binance_worksteal.data import compute_features, FEATURE_NAMES
from binance_worksteal.model import DailyWorkStealPolicy, PerSymbolWorkStealPolicy

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
MIN_TRACKED_POSITION_VALUE_USD = 5.0
TERMINAL_MARGIN_ORDER_STATUSES = {"FILLED", "CANCELED", "EXPIRED", "REJECTED"}


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


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _candidate_margin_pairs(symbol: str) -> List[str]:
    normalized = _normalize_strategy_symbol(symbol)
    if not normalized:
        return []
    pairs: List[str] = []
    mapping = SYMBOL_PAIRS.get(normalized, {})
    for key in ("fdusd", "usdt"):
        pair = str(mapping.get(key) or "").upper().strip()
        if pair and pair not in pairs:
            pairs.append(pair)
    fallback_pair = f"{normalized[:-3]}USDT"
    if fallback_pair not in pairs:
        pairs.append(fallback_pair)
    return pairs


def _order_status(order: dict) -> str:
    return str(order.get("status") or "").upper().strip()


def _order_pair(order: dict) -> str:
    return str(order.get("symbol") or "").upper().strip()


def _order_avg_price(order: dict, fallback: float = 0.0) -> float:
    price = _safe_float(order.get("price"), default=0.0)
    if price > 0.0:
        return price
    executed_qty = _safe_float(order.get("executedQty"), default=0.0)
    cumulative_quote = _safe_float(order.get("cummulativeQuoteQty"), default=0.0)
    if executed_qty > 0.0 and cumulative_quote > 0.0:
        return cumulative_quote / executed_qty
    return float(fallback or 0.0)


def _order_timestamp_iso(order: dict, *, fallback: datetime) -> str:
    raw_ts = order.get("updateTime") or order.get("time")
    try:
        return pd.Timestamp(int(raw_ts), unit="ms", tz="UTC").isoformat()
    except (TypeError, ValueError):
        return fallback.isoformat()


def _latest_filled_order(orders: List[dict], *, side: str) -> Optional[dict]:
    filtered = [
        order
        for order in orders
        if _order_status(order) == "FILLED"
        and str(order.get("side") or "").upper().strip() == side.upper()
        and _safe_float(order.get("executedQty"), default=0.0) > 0.0
    ]
    if not filtered:
        return None
    filtered.sort(key=lambda row: int(row.get("updateTime") or row.get("time") or 0), reverse=True)
    return filtered[0]


def _margin_position_quantity(balance_row: dict) -> float:
    borrowed_qty = _safe_float(balance_row.get("borrowed"), default=0.0)
    if borrowed_qty > 1e-8:
        return 0.0
    free_qty = _safe_float(balance_row.get("free"), default=0.0)
    net_qty = _safe_float(balance_row.get("netAsset"), default=0.0)
    return max(free_qty, net_qty, 0.0)


def _clear_pending_exit(position: dict) -> None:
    for key in ("exit_order_id", "exit_order_symbol", "exit_order_status", "exit_price", "exit_reason"):
        position.pop(key, None)


def _fetch_recent_margin_orders(client, symbol: str, *, preferred_pair: str = "", limit: int = 20) -> Tuple[List[dict], str]:
    candidates = []
    if preferred_pair:
        candidates.append(str(preferred_pair).upper().strip())
    candidates.extend(_candidate_margin_pairs(symbol))

    seen = set()
    for pair in candidates:
        if not pair or pair in seen:
            continue
        seen.add(pair)
        try:
            orders = client.get_all_margin_orders(symbol=pair, isIsolated="FALSE", limit=limit)
        except Exception as exc:
            logger.warning(f"Failed to fetch recent margin orders for {pair}: {exc}")
            continue
        if isinstance(orders, list) and orders:
            return [order for order in orders if isinstance(order, dict)], pair
    return [], candidates[0] if candidates else ""


def load_neural_model(checkpoint_path: str, device: str = "cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model_type = cfg.get("model_type", "persymbol")
    n_features = cfg.get("n_features", len(FEATURE_NAMES))
    n_symbols = cfg.get("n_symbols", 30)
    hidden_dim = cfg.get("hidden_dim", 128)
    num_layers = cfg.get("num_layers", 4)
    num_heads = cfg.get("num_heads", 4)
    seq_len = cfg.get("seq_len", 30)
    dropout = cfg.get("dropout", 0.0)

    if model_type == "persymbol":
        num_temporal = max(1, num_layers // 2)
        num_cross = max(1, num_layers - num_temporal)
        model = PerSymbolWorkStealPolicy(
            n_features=n_features, n_symbols=n_symbols,
            hidden_dim=hidden_dim,
            num_temporal_layers=num_temporal, num_cross_layers=num_cross,
            num_heads=num_heads, seq_len=seq_len, dropout=dropout,
        )
    else:
        model = DailyWorkStealPolicy(
            n_features=n_features, n_symbols=n_symbols,
            hidden_dim=hidden_dim, num_layers=num_layers,
            num_heads=num_heads, seq_len=seq_len, dropout=dropout,
        )

    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    model.to(device)

    symbols = cfg.get("symbols", [])
    return model, symbols, cfg


def prepare_neural_features(
    all_bars: Dict[str, pd.DataFrame],
    model_symbols: List[str],
    seq_len: int = 30,
) -> Optional[torch.Tensor]:
    sym_to_idx = {sym: i for i, sym in enumerate(model_symbols)}
    n_symbols = len(model_symbols)
    n_features = len(FEATURE_NAMES)

    all_dates = set()
    sym_feats = {}
    for sym in model_symbols:
        strategy_sym = _normalize_strategy_symbol(sym)
        bars = all_bars.get(strategy_sym)
        if bars is None:
            bars = all_bars.get(sym)
        if bars is None or len(bars) < seq_len:
            continue
        feats = compute_features(bars)
        sym_feats[sym] = (bars, feats)
        for ts in bars["timestamp"].tolist():
            all_dates.add(ts)

    if not sym_feats:
        return None

    sorted_dates = sorted(all_dates)
    if len(sorted_dates) < seq_len:
        return None

    use_dates = sorted_dates[-seq_len:]
    date_to_idx = {d: i for i, d in enumerate(use_dates)}

    feature_array = np.zeros((seq_len, n_symbols, n_features), dtype=np.float32)
    for sym, (bars, feats) in sym_feats.items():
        si = sym_to_idx[sym]
        for row_idx in range(len(bars)):
            ts = bars["timestamp"].iloc[row_idx]
            if ts in date_to_idx:
                di = date_to_idx[ts]
                vals = feats.iloc[row_idx].values[:n_features]
                feature_array[di, si, :len(vals)] = vals

    tensor = torch.from_numpy(feature_array).unsqueeze(0)  # [1, seq_len, n_symbols, n_features]
    return tensor


@torch.no_grad()
def run_neural_inference(
    model,
    all_bars: Dict[str, pd.DataFrame],
    model_symbols: List[str],
    seq_len: int = 30,
) -> Optional[Dict[str, Dict[str, float]]]:
    features = prepare_neural_features(all_bars, model_symbols, seq_len)
    if features is None:
        return None

    actions = model(features)
    buy_offset = actions["buy_offset"][0].cpu().numpy()
    sell_offset = actions["sell_offset"][0].cpu().numpy()
    intensity = actions["intensity"][0].cpu().numpy()

    if np.isnan(buy_offset).any() or np.isnan(sell_offset).any() or np.isnan(intensity).any():
        logger.warning("Neural model produced NaN outputs, falling back to rules")
        return None

    predictions = {}
    for i, sym in enumerate(model_symbols):
        strategy_sym = _normalize_strategy_symbol(sym)
        predictions[strategy_sym] = {
            "buy_offset": float(buy_offset[i]),
            "sell_offset": float(sell_offset[i]),
            "intensity": float(intensity[i]),
        }
    return predictions


def _try_neural_inference(neural_model, all_bars, neural_model_symbols, neural_seq_len):
    try:
        predictions = run_neural_inference(
            neural_model, all_bars, neural_model_symbols, neural_seq_len,
        )
        if predictions:
            log_event({"type": "neural_inference", "predictions": predictions})
            logger.info(f"Neural inference: {len(predictions)} symbol predictions")
        return predictions
    except Exception as e:
        logger.warning(f"Neural inference failed, falling back to rules: {e}")
        return None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--universe-file", default=None, help="YAML file with symbol universe")
    parser.add_argument("--max-symbols", type=int, default=100, help="Safety cap on number of symbols")
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
    parser.add_argument("--neural-model", default=None, help="Path to neural model checkpoint (.pt)")
    parser.add_argument("--neural-symbols", nargs="+", default=None, help="Symbols the model was trained on")
    parser.add_argument("--diagnose", action="store_true", help="Run one diagnostic cycle showing why each symbol is filtered")
    parser.add_argument("--min-dip-pct", type=float, default=0.10, help="Floor for adaptive dip reduction (default 0.10)")
    parser.add_argument("--adaptive-dip-cycles", type=int, default=3, help="Zero-candidate cycles before reducing dip_pct")
    parser.add_argument("--dip-pct-fallback", nargs="+", type=float, default=None,
                        help="Descending dip thresholds for tiered entry (e.g. 0.20 0.15 0.12)")
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
        exit_order_id = raw_position.get("exit_order_id")
        if exit_order_id is not None:
            normalized[symbol]["exit_order_id"] = exit_order_id
            normalized[symbol]["exit_order_symbol"] = str(raw_position.get("exit_order_symbol") or "")
            normalized[symbol]["exit_order_status"] = str(raw_position.get("exit_order_status") or "")
            normalized[symbol]["exit_price"] = float(raw_position.get("exit_price", 0.0) or 0.0)
            normalized[symbol]["exit_reason"] = str(raw_position.get("exit_reason") or "")
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
            "order_symbol": str(raw_entry.get("order_symbol") or ""),
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
        if sym not in current_bars:
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


def load_universe_file(path: str) -> List[str]:
    """Load symbols from a YAML universe file."""
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "symbols" not in data:
        raise ValueError(f"Universe file must have a 'symbols' key: {path}")
    symbols = []
    for entry in data["symbols"]:
        if isinstance(entry, str):
            symbols.append(_normalize_strategy_symbol(entry))
        elif isinstance(entry, dict) and "symbol" in entry:
            symbols.append(_normalize_strategy_symbol(entry["symbol"]))
    return symbols


def _fetch_all_bars(client, symbols: List[str], lookback_days: int,
                    max_workers: int = 10) -> Dict[str, pd.DataFrame]:
    t0 = time.monotonic()

    def _fetch_one(sym: str) -> Tuple[str, Optional[pd.DataFrame]]:
        try:
            bars = fetch_daily_bars(client, sym, lookback_days + 10)
            if not bars.empty and len(bars) > lookback_days:
                return sym, bars
        except Exception as e:
            logger.warning(f"Failed to fetch bars for {sym}: {e}")
        return sym, None

    all_bars = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        for sym, bars in pool.map(_fetch_one, symbols):
            if bars is not None:
                all_bars[sym] = bars

    elapsed = time.monotonic() - t0
    logger.info(f"Fetched bars for {len(all_bars)}/{len(symbols)} symbols in {elapsed:.1f}s")
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


def _format_price(price: float) -> str:
    if price >= 1.0:
        return f"{price:.2f}"
    elif price >= 0.01:
        return f"{price:.4f}"
    elif price >= 0.0001:
        return f"{price:.6f}"
    else:
        return f"{price:.8f}"


def _format_quantity(qty: float) -> str:
    if qty >= 1.0:
        return f"{qty:.4f}"
    elif qty >= 0.01:
        return f"{qty:.6f}"
    else:
        return f"{qty:.8f}"


def place_limit_buy(client, symbol: str, price: float, quantity: float, config: WorkStealConfig):
    pair = get_binance_pair(symbol, prefer_fdusd=True)
    price_str = _format_price(price)
    qty_str = _format_quantity(quantity)
    logger.info(f"Placing limit buy: {pair} qty={qty_str} price={price_str}")
    try:
        order = client.create_margin_order(
            symbol=pair, side="BUY", type="LIMIT",
            timeInForce="GTC",
            quantity=qty_str,
            price=price_str,
        )
        return order
    except Exception as e:
        logger.error(f"Limit buy failed for {pair}: {e}")
        return None


def place_limit_sell(client, symbol: str, price: float, quantity: float):
    pair = get_binance_pair(symbol, prefer_fdusd=True)
    price_str = _format_price(price)
    qty_str = _format_quantity(quantity)
    logger.info(f"Placing limit sell: {pair} qty={qty_str} price={price_str}")
    try:
        order = client.create_margin_order(
            symbol=pair, side="SELL", type="LIMIT",
            timeInForce="GTC",
            quantity=qty_str,
            price=price_str,
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

        if entry.get("status") == "preview" and not dry_run:
            logger.info(f"Cleaning up preview entry {sym} (no order placed)")
            del pending_entries[sym]
            continue

        if dry_run or client is None or entry.get("order_id") is None:
            continue

        pair = str(entry.get("order_symbol") or get_binance_pair(sym, prefer_fdusd=True)).upper().strip()
        try:
            order = client.get_margin_order(symbol=pair, orderId=entry["order_id"])
        except Exception as exc:
            logger.warning(f"Failed to refresh pending entry {pair}: {exc}")
            continue

        status = _order_status(order)
        if status == "FILLED":
            fill_qty = float(order.get("executedQty") or entry.get("quantity") or 0.0)
            fill_price = _order_avg_price(order, fallback=float(entry.get("buy_price") or 0.0))
            if fill_qty <= 0.0 or fill_price <= 0.0:
                del pending_entries[sym]
                continue
            realized = dict(entry)
            realized["quantity"] = fill_qty
            realized["buy_price"] = fill_price
            positions[sym] = _pending_entry_to_position(realized, entry_time=now)
            trade = {
                "timestamp": _order_timestamp_iso(order, fallback=now),
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
        elif status in TERMINAL_MARGIN_ORDER_STATUSES - {"FILLED"}:
            del pending_entries[sym]

    return recent_trades


def reconcile_exit_orders(
    *,
    client,
    positions: dict,
    last_exit: dict,
    now: datetime,
) -> list[dict]:
    if client is None:
        return []

    recent_trades: list[dict] = []
    for sym, position in list(positions.items()):
        order_id = position.get("exit_order_id")
        if order_id is None:
            continue

        pair = str(position.get("exit_order_symbol") or get_binance_pair(sym, prefer_fdusd=True)).upper().strip()
        try:
            order = client.get_margin_order(symbol=pair, orderId=order_id)
        except Exception as exc:
            logger.warning(f"Failed to refresh exit order {pair}#{order_id}: {exc}")
            continue

        status = _order_status(order)
        position["exit_order_status"] = status
        if status == "FILLED":
            fill_qty = _safe_float(order.get("executedQty"), default=position.get("quantity", 0.0))
            fill_price = _order_avg_price(
                order,
                fallback=float(position.get("exit_price") or position.get("target_sell") or position.get("entry_price") or 0.0),
            )
            if fill_qty > 0.0 and fill_price > 0.0:
                trade = {
                    "timestamp": _order_timestamp_iso(order, fallback=now),
                    "symbol": sym,
                    "side": "sell",
                    "price": fill_price,
                    "quantity": fill_qty,
                    "reason": str(position.get("exit_reason") or "pending_exit_fill"),
                    "pnl": (fill_price - float(position.get("entry_price") or 0.0)) * fill_qty,
                    "dry_run": False,
                }
                log_trade(trade)
                recent_trades.append(trade)
            last_exit[sym] = _order_timestamp_iso(order, fallback=now)
            del positions[sym]
        elif status in TERMINAL_MARGIN_ORDER_STATUSES - {"FILLED"}:
            _clear_pending_exit(position)

    return recent_trades


def synchronize_positions_from_exchange(
    *,
    client,
    symbols: List[str],
    positions: dict,
    current_bars: Dict[str, pd.Series],
    config: WorkStealConfig,
    now: datetime,
) -> list[dict]:
    if client is None:
        return []

    try:
        account = client.get_margin_account()
    except Exception as exc:
        logger.warning(f"Failed to fetch margin account for position sync: {exc}")
        return []

    try:
        open_orders = client.get_open_margin_orders(isIsolated="FALSE")
    except Exception as exc:
        logger.warning(f"Failed to fetch open margin orders for position sync: {exc}")
        open_orders = []

    tracked_symbols = {_normalize_strategy_symbol(symbol) for symbol in symbols}
    sell_orders_by_symbol: dict[str, dict] = {}
    for order in open_orders if isinstance(open_orders, list) else []:
        if not isinstance(order, dict):
            continue
        symbol = _normalize_strategy_symbol(order.get("symbol", ""))
        if symbol not in tracked_symbols:
            continue
        if str(order.get("side") or "").upper().strip() != "SELL":
            continue
        existing = sell_orders_by_symbol.get(symbol)
        existing_ts = int(existing.get("updateTime") or existing.get("time") or 0) if existing else -1
        candidate_ts = int(order.get("updateTime") or order.get("time") or 0)
        if existing is None or candidate_ts >= existing_ts:
            sell_orders_by_symbol[symbol] = order

    sync_events: list[dict] = []
    balance_rows = account.get("userAssets", []) if isinstance(account, dict) else []
    for row in balance_rows if isinstance(balance_rows, list) else []:
        if not isinstance(row, dict):
            continue
        asset = str(row.get("asset") or "").upper().strip()
        symbol = _normalize_strategy_symbol(asset)
        if symbol not in tracked_symbols:
            continue

        quantity = _margin_position_quantity(row)
        if quantity <= 0.0:
            continue

        close_price = _safe_float(current_bars.get(symbol, {}).get("close") if symbol in current_bars else 0.0, default=0.0)
        est_value = quantity * close_price
        if est_value < MIN_TRACKED_POSITION_VALUE_USD and symbol not in positions and symbol not in sell_orders_by_symbol:
            continue

        position = positions.get(symbol)
        if position is None:
            open_sell = sell_orders_by_symbol.get(symbol)
            recent_orders, _resolved_pair = _fetch_recent_margin_orders(
                client,
                symbol,
                preferred_pair=_order_pair(open_sell) if open_sell else "",
            )
            last_buy = _latest_filled_order(recent_orders, side="BUY")
            entry_price = _order_avg_price(last_buy or {}, fallback=close_price)
            if entry_price <= 0.0:
                continue
            position = {
                "entry_price": entry_price,
                "entry_date": _order_timestamp_iso(last_buy or {}, fallback=now),
                "quantity": quantity,
                "peak_price": max(close_price, entry_price),
                "target_sell": entry_price * (1.0 + config.profit_target_pct),
                "stop_price": entry_price * (1.0 - config.stop_loss_pct),
                "source": "exchange_sync",
            }
            positions[symbol] = position
            sync_events.append(
                {
                    "type": "exchange_position_sync",
                    "symbol": symbol,
                    "quantity": quantity,
                    "entry_price": entry_price,
                    "source": "exchange_sync",
                }
            )

        position["quantity"] = quantity
        if symbol in current_bars:
            position["peak_price"] = max(
                float(position.get("peak_price", position.get("entry_price", 0.0)) or 0.0),
                _safe_float(current_bars[symbol].get("high"), default=0.0),
            )

        open_sell = sell_orders_by_symbol.get(symbol)
        if open_sell is not None:
            position["exit_order_id"] = open_sell.get("orderId")
            position["exit_order_symbol"] = _order_pair(open_sell)
            position["exit_order_status"] = _order_status(open_sell)
            position["exit_price"] = _order_avg_price(open_sell, fallback=float(position.get("target_sell") or 0.0))
            position["exit_reason"] = str(position.get("exit_reason") or "exchange_open_sell")
            if position["exit_price"] > 0.0:
                position["target_sell"] = position["exit_price"]

    for event in sync_events:
        log_event(event)

    return sync_events


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
    neural_predictions: Optional[Dict[str, Dict[str, float]]] = None,
) -> dict:
    n_staged = 0
    n_already_held = 0
    n_proximity_skip = 0
    n_gemini_skip = 0
    n_neural_skip = 0
    n_order_fail = 0

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
        neural_override = None

        if neural_predictions and sym in neural_predictions:
            pred = neural_predictions[sym]
            intensity = pred["intensity"]
            if intensity < 0.1:
                logger.info(f"NEURAL SKIP {sym}: intensity={intensity:.3f} < 0.1")
                n_neural_skip += 1
                continue
            neural_buy_offset = pred["buy_offset"]
            neural_sell_offset = pred["sell_offset"]
            buy_price = close * (1.0 - neural_buy_offset)
            sell_target = close * (1.0 + neural_sell_offset)
            stop = buy_price * (1 - entry_config.stop_loss_pct)
            if intensity > 0.5:
                confidence = min(1.0 + (intensity - 0.5), 1.5)
            source = f"neural(int={intensity:.2f})"
            neural_override = {
                "buy_offset": neural_buy_offset,
                "sell_offset": neural_sell_offset,
                "intensity": intensity,
                "original_fill_price": fill_price,
            }
            logger.info(
                f"NEURAL {sym}: buy_offset={neural_buy_offset:.4f} sell_offset={neural_sell_offset:.4f} "
                f"intensity={intensity:.3f} buy=${buy_price:.2f} tp=${sell_target:.2f}"
            )

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
            order_id = None if order is None else order.get("orderId")
            if order_id is None:
                logger.warning(f"ENTRY ORDER FAILED {sym}: no live order placed at {buy_price:.4f}")
                log_event({
                    "type": "entry_order_failed",
                    "symbol": sym,
                    "price": buy_price,
                    "quantity": quantity,
                    "reason": source,
                })
                n_order_fail += 1
                continue

        entry = {
            "buy_price": buy_price,
            "placed_at": now.isoformat(),
            "expires_at": (now + PENDING_ENTRY_TTL).isoformat(),
            "quantity": quantity,
            "target_sell": sell_target,
            "stop_price": stop,
            "confidence": confidence,
            "source": source,
            "order_id": None if order is None else order.get("orderId"),
            "order_symbol": "" if order is None else str(order.get("symbol") or ""),
            "status": "preview" if dry_run else "open",
        }
        if neural_override:
            entry["neural_override"] = neural_override
        pending_entries[sym] = entry
        staged_symbols.add(sym)
        trade = {
            "timestamp": now.isoformat(), "symbol": sym, "side": "staged_buy",
            "price": buy_price, "quantity": quantity,
            "reason": f"dip_buy({source})",
            "dry_run": dry_run,
        }
        if neural_override:
            trade["neural_override"] = neural_override
        log_trade(trade)
        recent_trades.append(trade)
        n_staged += 1

    return {
        "n_staged": n_staged,
        "n_already_held": n_already_held,
        "n_proximity_skip": n_proximity_skip,
        "n_gemini_skip": n_gemini_skip,
        "n_neural_skip": n_neural_skip,
        "n_order_fail": n_order_fail,
    }


def _count_sma_pass_fail(all_bars: Dict[str, pd.DataFrame], config: WorkStealConfig) -> Tuple[int, int]:
    n_pass = 0
    n_fail = 0
    for bars in all_bars.values():
        close = float(bars.iloc[-1]["close"])
        if passes_sma_filter(bars, config, close):
            n_pass += 1
        else:
            n_fail += 1
    return n_pass, n_fail


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
    if not dry_run:
        synchronize_positions_from_exchange(
            client=client,
            symbols=symbols,
            positions=positions,
            current_bars=current_bars,
            config=config,
            now=now,
        )
    entry_config = resolve_entry_config(current_bars=current_bars, history=all_bars, config=config)
    risk_off = compute_market_breadth_skip(current_bars, all_bars, entry_config)

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
                   gemini_model: str = "gemini-2.5-flash",
                   neural_model=None, neural_model_symbols: Optional[List[str]] = None,
                   neural_seq_len: int = 30,
                   dip_pct_fallback: Optional[List[float]] = None):
    state = load_state()
    positions = normalize_live_positions(state.get("positions", {}), config)
    pending_entries = _normalize_pending_entries(state.get("pending_entries", {}))
    last_exit = {_normalize_strategy_symbol(sym): ts for sym, ts in state.get("last_exit", {}).items()}
    recent_trades = list(state.get("recent_trades", []))
    now = datetime.now(timezone.utc)

    all_bars = _fetch_all_bars(client, symbols, config.lookback_days)
    equity = get_account_equity(client) if not dry_run else config.initial_cash
    current_bars = {sym: bars.iloc[-1] for sym, bars in all_bars.items()}
    if not dry_run:
        synchronize_positions_from_exchange(
            client=client,
            symbols=symbols,
            positions=positions,
            current_bars=current_bars,
            config=config,
            now=now,
        )
        state["positions"] = positions
        save_state(state)

    slots = config.max_positions - len(positions) - len(pending_entries)
    if slots <= 0:
        logger.info(f"ENTRY SCAN: skipped, {len(positions)} positions + {len(pending_entries)} pending >= {config.max_positions} max")
        return

    entry_config = resolve_entry_config(current_bars=current_bars, history=all_bars, config=config)

    if compute_market_breadth_skip(current_bars, all_bars, entry_config):
        logger.info("ENTRY SCAN: market breadth risk-off, no entries")
        log_event({"type": "entry_scan", "n_checked": len(all_bars), "n_candidates": 0, "n_staged": 0, "skip_reason": "market_breadth_risk_off"})
        return

    neural_predictions = None
    if neural_model is not None and neural_model_symbols:
        neural_predictions = _try_neural_inference(
            neural_model, all_bars, neural_model_symbols, neural_seq_len,
        )

    staged_symbols = set(positions) | set(pending_entries)
    dip_tiers = dip_pct_fallback if dip_pct_fallback else None
    last_exit_ts = {sym: pd.Timestamp(ts) for sym, ts in last_exit.items()}

    if dip_tiers and len(dip_tiers) > 1:
        logger.info(f"TIERED DIP (entry_scan): tiers {[f'{t:.0%}' for t in dip_tiers]}")
        candidates, tier_map = _build_tiered_candidates(
            dip_tiers=dip_tiers,
            current_bars=current_bars,
            history=all_bars,
            positions=positions,
            last_exit=last_exit_ts,
            date=pd.Timestamp(now),
            entry_config=entry_config,
            max_positions=config.max_positions,
            pending_entries=pending_entries,
        )
    else:
        candidates = build_entry_candidates(
            date=pd.Timestamp(now),
            current_bars=current_bars,
            history=all_bars,
            positions={},
            last_exit=last_exit_ts,
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
        neural_predictions=neural_predictions,
    )

    logger.info(
        f"ENTRY SCAN SUMMARY: candidates={len(candidates)} staged={counts['n_staged']} "
        f"already_held={counts['n_already_held']} proximity_skip={counts['n_proximity_skip']} "
        f"gemini_skip={counts['n_gemini_skip']} neural_skip={counts['n_neural_skip']}"
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


def _build_tiered_candidates(
    *,
    dip_tiers: List[float],
    current_bars: Dict[str, pd.Series],
    history: Dict[str, pd.DataFrame],
    positions: dict,
    last_exit: Dict[str, pd.Timestamp],
    date: pd.Timestamp,
    entry_config: WorkStealConfig,
    max_positions: int,
    pending_entries: dict,
    diagnostics: Optional[List[SymbolDiagnostic]] = None,
) -> Tuple[list, Dict[str, int]]:
    """Run tiered dip passes, filling slots from deepest dip to shallowest."""
    slots = max(0, max_positions - len(set(positions) | set(pending_entries)))
    candidates, tier_map = build_tiered_entry_candidates(
        date=date,
        current_bars=current_bars,
        history=history,
        positions={sym: True for sym in (set(positions) | set(pending_entries))},
        last_exit=last_exit,
        config=replace(entry_config, dip_pct_fallback=tuple(dip_tiers)),
        base_symbol=None,
        max_candidates=slots,
        diagnostics=diagnostics,
    )
    for sym, tier_idx in tier_map.items():
        tier_dip = dip_tiers[tier_idx]
        logger.info(f"TIER {tier_idx} (dip={tier_dip:.0%}): {sym}")
    return candidates, tier_map


def run_daily_cycle(client, symbols: List[str], config: WorkStealConfig,
                    dry_run: bool = True, gemini_enabled: bool = False,
                    gemini_model: str = "gemini-2.5-flash",
                    neural_model=None, neural_model_symbols: Optional[List[str]] = None,
                    neural_seq_len: int = 30,
                    min_dip_pct: float = 0.10,
                    adaptive_dip_cycles: int = 3,
                    dip_pct_fallback: Optional[List[float]] = None):
    state = load_state()
    positions = normalize_live_positions(state.get("positions", {}), config)
    pending_entries = _normalize_pending_entries(state.get("pending_entries", {}))
    last_exit = {_normalize_strategy_symbol(sym): ts for sym, ts in state.get("last_exit", {}).items()}
    recent_trades = list(state.get("recent_trades", []))

    # Adaptive dip: track consecutive zero-candidate cycles
    zero_candidate_cycles = int(state.get("zero_candidate_cycles", 0))
    original_dip_pct = config.dip_pct
    if adaptive_dip_cycles > 0 and zero_candidate_cycles >= adaptive_dip_cycles and config.dip_pct > min_dip_pct:
        steps = (zero_candidate_cycles - adaptive_dip_cycles) + 1
        reduction = 0.02 * steps
        new_dip = max(min_dip_pct, config.dip_pct - reduction)
        logger.info(
            f"ADAPTIVE DIP: {zero_candidate_cycles} zero-candidate cycles, "
            f"reducing dip_pct {config.dip_pct:.2f} -> {new_dip:.2f} (floor={min_dip_pct:.2f})"
        )
        config = replace(config, dip_pct=new_dip)

    now = datetime.now(timezone.utc)
    logger.info(
        f"Daily cycle at {now.isoformat()}, {len(positions)} open positions, "
        f"{len(pending_entries)} pending entries, dip_pct={config.dip_pct:.2f}"
    )

    all_bars = _fetch_all_bars(client, symbols, config.lookback_days)
    current_bars = {sym: bars.iloc[-1] for sym, bars in all_bars.items()}

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
    recent_trades.extend(
        reconcile_exit_orders(
            client=client,
            positions=positions,
            last_exit=last_exit,
            now=now,
        )
    )
    if not dry_run:
        synchronize_positions_from_exchange(
            client=client,
            symbols=symbols,
            positions=positions,
            current_bars=current_bars,
            config=config,
            now=now,
        )

    if len(positions) >= config.max_positions and pending_entries:
        for sym, entry in list(pending_entries.items()):
            _cancel_pending_entry(client, sym, entry)
            del pending_entries[sym]

    # Check exits
    exits_to_process = []
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
        if pos.get("exit_order_id") is not None:
            continue
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
        if dry_run:
            trade = {
                "timestamp": now.isoformat(), "symbol": sym, "side": "sell",
                "price": exit_price, "quantity": pos["quantity"],
                "reason": reason, "pnl": (exit_price - pos["entry_price"]) * pos["quantity"],
                "dry_run": True,
            }
            log_trade(trade)
            recent_trades.append(trade)
            last_exit[sym] = now.isoformat()
            del positions[sym]
            continue

        order = place_limit_sell(client, sym, exit_price, pos["quantity"])
        if order is None:
            continue

        pos["exit_order_id"] = order.get("orderId")
        pos["exit_order_symbol"] = str(order.get("symbol") or get_binance_pair(sym, prefer_fdusd=True))
        pos["exit_order_status"] = _order_status(order) or "NEW"
        pos["exit_price"] = exit_price
        pos["exit_reason"] = reason
        pos["target_sell"] = exit_price
        trade = {
            "timestamp": now.isoformat(), "symbol": sym, "side": "staged_sell",
            "price": exit_price, "quantity": pos["quantity"],
            "reason": reason, "pnl": (exit_price - pos["entry_price"]) * pos["quantity"],
            "dry_run": False,
            "order_id": order.get("orderId"),
        }
        log_trade(trade)
        recent_trades.append(trade)

    # Stage new entries
    entry_config = resolve_entry_config(current_bars=current_bars, history=all_bars, config=config)
    skip_entries = compute_market_breadth_skip(current_bars, all_bars, entry_config)
    risk_off = _risk_off_triggered(current_bars, all_bars, entry_config)
    counts = {"n_staged": 0, "n_proximity_skip": 0, "n_gemini_skip": 0, "n_already_held": 0, "n_neural_skip": 0}
    n_candidates = 0

    # Log market breadth details
    breadth_ratio, n_breadth_dipping, n_breadth_total = compute_breadth_ratio(current_bars, all_bars)
    logger.info(
        f"MARKET STATE: breadth={n_breadth_dipping}/{n_breadth_total} dipping ({breadth_ratio:.1%}) "
        f"threshold={entry_config.market_breadth_filter:.0%} breadth_skip={skip_entries} risk_off={risk_off}"
    )

    # Log SMA pass/fail
    sma_pass, sma_fail = _count_sma_pass_fail(all_bars, config)
    logger.info(f"SMA FILTER: {sma_pass} pass, {sma_fail} fail (period={config.sma_filter_period}, method={config.sma_check_method})")

    neural_predictions = None
    if neural_model is not None and neural_model_symbols and not skip_entries:
        neural_predictions = _try_neural_inference(
            neural_model, all_bars, neural_model_symbols, neural_seq_len,
        )

    tier_map: Dict[str, int] = {}

    if len(positions) >= config.max_positions:
        logger.info(f"ENTRY SCAN: skipped, max positions ({config.max_positions}) reached")
    elif skip_entries:
        logger.info("ENTRY SCAN: skipped, market breadth risk-off")
    else:
        staged_symbols = set(positions) | set(pending_entries)
        diagnostics: List[SymbolDiagnostic] = []
        last_exit_ts = {sym: pd.Timestamp(ts) for sym, ts in last_exit.items()}

        if dip_pct_fallback and len(dip_pct_fallback) > 1:
            logger.info(f"TIERED DIP: using tiers {[f'{t:.0%}' for t in dip_pct_fallback]}")
            candidates, tier_map = _build_tiered_candidates(
                dip_tiers=dip_pct_fallback,
                current_bars=current_bars,
                history=all_bars,
                positions=positions,
                last_exit=last_exit_ts,
                date=pd.Timestamp(now),
                entry_config=entry_config,
                max_positions=config.max_positions,
                pending_entries=pending_entries,
                diagnostics=diagnostics,
            )
        else:
            candidates = build_entry_candidates(
                date=pd.Timestamp(now),
                current_bars=current_bars,
                history=all_bars,
                positions={},
                last_exit=last_exit_ts,
                config=entry_config,
                base_symbol=None,
                diagnostics=diagnostics,
            )

        n_candidates = len(candidates)
        slots = config.max_positions - len(positions) - len(pending_entries)
        logger.info(f"ENTRY SCAN: {len(all_bars)} symbols checked, {n_candidates} candidates found")

        if tier_map:
            for sym, tier_idx in tier_map.items():
                tier_dip = dip_pct_fallback[tier_idx] if dip_pct_fallback else config.dip_pct
                logger.info(f"  {sym}: tier {tier_idx} (dip={tier_dip:.0%})")

        # Log top 5 closest to entry
        prox_diags = [d for d in diagnostics if d.dist_pct > 0]
        prox_diags.sort(key=lambda d: d.dist_pct)
        if prox_diags:
            logger.info("TOP 5 CLOSEST TO ENTRY:")
            for d in prox_diags[:5]:
                logger.info(
                    f"  {d.symbol}: close=${d.close:.4f} target=${d.buy_target:.4f} "
                    f"dist={d.dist_pct:.4f} ({d.dist_pct*100:.1f}% away) "
                    f"{'BLOCKED:' + d.filter_reason if d.filter_reason else 'CANDIDATE'}"
                )

        # Log filter reason summary
        reason_counts: Dict[str, int] = {}
        for d in diagnostics:
            if d.filter_reason:
                key = d.filter_reason.split("(")[0]
                reason_counts[key] = reason_counts.get(key, 0) + 1
        if reason_counts:
            parts = [f"{r}={c}" for r, c in sorted(reason_counts.items(), key=lambda x: -x[1])]
            logger.info(f"FILTER REASONS: {', '.join(parts)}")

        counts = _stage_entry_candidates(
            client=client, candidates=candidates, all_bars=all_bars,
            staged_symbols=staged_symbols, pending_entries=pending_entries,
            recent_trades=recent_trades, entry_config=entry_config, config=config,
            equity=equity, now=now, dry_run=dry_run, slots=slots,
            gemini_enabled=gemini_enabled, gemini_model=gemini_model,
            neural_predictions=neural_predictions,
        )

    logger.info(
        f"ENTRY SUMMARY: candidates={n_candidates} staged={counts['n_staged']} "
        f"proximity_skip={counts['n_proximity_skip']} gemini_skip={counts['n_gemini_skip']} "
        f"neural_skip={counts['n_neural_skip']} "
        f"already_held={counts['n_already_held']} risk_off={skip_entries}"
    )
    event_data = {
        "type": "entry_scan",
        "n_checked": len(all_bars),
        "n_candidates": n_candidates,
        "risk_off": skip_entries,
        "n_positions": len(positions),
        "n_pending": len(pending_entries),
        "equity": equity,
        **counts,
    }
    if dip_pct_fallback and len(dip_pct_fallback) > 1:
        event_data["dip_tiers"] = dip_pct_fallback
        event_data["tier_map"] = {sym: idx for sym, idx in tier_map.items()} if tier_map else {}
    log_event(event_data)

    # Track consecutive zero-candidate cycles for adaptive dip
    sufficient_data = len(all_bars) >= len(symbols) * 0.5
    if n_candidates == 0 and not skip_entries and sufficient_data:
        state["zero_candidate_cycles"] = zero_candidate_cycles + 1
        logger.info(f"ADAPTIVE: zero_candidate_cycles={zero_candidate_cycles + 1}")
    elif n_candidates == 0 and not sufficient_data:
        logger.warning(f"ADAPTIVE: skipped counter increment, only {len(all_bars)}/{len(symbols)} symbols had data")
    elif n_candidates > 0:
        state["zero_candidate_cycles"] = 0

    # Save state
    state["positions"] = positions
    state["pending_entries"] = pending_entries
    state["last_exit"] = last_exit
    state["recent_trades"] = recent_trades[-50:]
    state["peak_equity"] = max(state.get("peak_equity", 0), equity)
    if config.dip_pct != original_dip_pct:
        state["effective_dip_pct"] = config.dip_pct
    save_state(state)

    logger.info(
        f"Cycle complete: {len(positions)} positions, {len(pending_entries)} pending, equity=${equity:.0f}"
    )
    for sym, pos in positions.items():
        logger.info(f"  {sym}: entry={pos['entry_price']:.2f} "
                    f"target={pos['target_sell']:.2f} stop={pos['stop_price']:.2f}")


def run_diagnose(client, symbols: List[str], config: WorkStealConfig):
    """One-shot diagnostic: show exactly why each symbol passes/fails entry filters."""
    all_bars = _fetch_all_bars(client, symbols, config.lookback_days)
    if not all_bars:
        logger.error("No bar data fetched for any symbol")
        return
    current_bars = {sym: bars.iloc[-1] for sym, bars in all_bars.items()}
    now = datetime.now(timezone.utc)

    # Market breadth
    breadth_ratio, n_dipping, n_total = compute_breadth_ratio(current_bars, all_bars)
    entry_config = resolve_entry_config(current_bars=current_bars, history=all_bars, config=config)
    breadth_skip = compute_market_breadth_skip(current_bars, all_bars, entry_config)
    risk_off = _risk_off_triggered(current_bars, all_bars, entry_config)

    print(f"\n{'='*70}")
    print(f"WORKSTEAL DIAGNOSTIC  {now.isoformat()}")
    print(f"{'='*70}")
    print(f"Symbols with data:  {len(all_bars)}/{len(symbols)}")
    print(f"Market breadth:     {n_dipping}/{n_total} dipping ({breadth_ratio:.1%})")
    print(f"Breadth threshold:  {entry_config.market_breadth_filter:.0%}")
    print(f"Breadth skip:       {breadth_skip}")
    print(f"Risk-off triggered: {risk_off}")
    print(f"Config dip_pct:     {config.dip_pct:.0%}")
    print(f"Config proximity:   {config.proximity_pct:.0%}")
    print(f"SMA filter period:  {config.sma_filter_period}")
    print(f"SMA check method:   {config.sma_check_method}")

    # SMA pass/fail summary
    sma_pass_count, sma_fail_count = _count_sma_pass_fail(all_bars, config)
    print(f"SMA filter:         {sma_pass_count} pass, {sma_fail_count} fail")

    # Run with diagnostics
    diagnostics: List[SymbolDiagnostic] = []
    state = load_state()
    positions = normalize_live_positions(state.get("positions", {}), config)
    last_exit = {_normalize_strategy_symbol(sym): ts for sym, ts in state.get("last_exit", {}).items()}

    candidates = build_entry_candidates(
        date=pd.Timestamp(now),
        current_bars=current_bars,
        history=all_bars,
        positions={},
        last_exit={sym: pd.Timestamp(ts) for sym, ts in last_exit.items()} if last_exit else {},
        config=entry_config,
        base_symbol=None,
        diagnostics=diagnostics,
    )

    print(f"\nCandidates found:   {len(candidates)}")

    # Show candidates
    if candidates:
        print(f"\n--- CANDIDATES (would enter) ---")
        for sym, direction, score, fill_price, bar in candidates:
            close = float(bar["close"])
            print(f"  {sym:12s} dir={direction} score={score:.4f} fill=${fill_price:.4f} close=${close:.4f}")

    # Show filtered symbols sorted by proximity to entry
    filtered = [d for d in diagnostics if not d.is_candidate and d.filter_reason]
    proximity_diags = [d for d in diagnostics if d.dist_pct > 0]
    proximity_diags.sort(key=lambda d: d.dist_pct)

    print(f"\n--- TOP 10 CLOSEST TO ENTRY (by distance to buy target) ---")
    for d in proximity_diags[:10]:
        dip_needed = d.dist_pct * 100
        print(
            f"  {d.symbol:12s} close=${d.close:<12.4f} ref_high=${d.ref_high:<12.4f} "
            f"buy_target=${d.buy_target:<12.4f} dist={d.dist_pct:.4f} ({dip_needed:.1f}% more dip needed) "
            f"{'** BLOCKED: ' + d.filter_reason if d.filter_reason else 'CANDIDATE'}"
        )

    # Group filter reasons
    reason_counts: Dict[str, int] = {}
    for d in filtered:
        key = d.filter_reason.split("(")[0]
        reason_counts[key] = reason_counts.get(key, 0) + 1

    print(f"\n--- FILTER SUMMARY ---")
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"  {reason:30s} {count} symbols")

    # Show all per-symbol details
    print(f"\n--- ALL SYMBOL DETAILS ---")
    for d in sorted(diagnostics, key=lambda x: x.symbol):
        status = "CANDIDATE" if d.is_candidate else f"FILTERED: {d.filter_reason}"
        parts = [f"  {d.symbol:12s}"]
        if d.close > 0:
            parts.append(f"close=${d.close:<10.4f}")
        if d.ref_high > 0:
            parts.append(f"ref=${d.ref_high:<10.4f}")
        if d.buy_target > 0:
            parts.append(f"tgt=${d.buy_target:<10.4f}")
        if d.dist_pct != 0:
            parts.append(f"dist={d.dist_pct:.4f}")
        if not d.sma_pass:
            parts.append(f"sma={d.sma_value:.4f}")
        parts.append(status)
        print(" ".join(parts))

    # Suggest adaptive thresholds
    if not candidates:
        min_dist = min((d.dist_pct for d in proximity_diags), default=999)
        if min_dist < 999:
            print(f"\n--- SUGGESTIONS ---")
            print(f"  Smallest distance to entry: {min_dist:.4f} ({min_dist*100:.1f}%)")
            print(f"  To capture nearest, increase proximity_pct to ~{min_dist + 0.01:.3f}")
            print(f"  Or reduce dip_pct to ~{max(0.05, config.dip_pct - min_dist):.3f}")


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.live:
        args.dry_run = False

    from binance_worksteal.backtest import FULL_UNIVERSE
    if args.symbols:
        symbols = args.symbols
    elif args.universe_file:
        symbols = load_universe_file(args.universe_file)
        logger.info(f"Loaded {len(symbols)} symbols from {args.universe_file}")
    else:
        symbols = FULL_UNIVERSE
    symbols = symbols[:args.max_symbols]

    config = build_runtime_config(args)

    # Initialize Binance client
    client = None
    if BinanceClient:
        try:
            from env_real import BINANCE_API_KEY, BINANCE_SECRET
            client = BinanceClient(BINANCE_API_KEY, BINANCE_SECRET)
        except Exception as e:
            logger.warning(f"Binance client unavailable: {e}")

    if args.diagnose:
        run_diagnose(client, symbols, config)
        return 0

    if client is None and not args.dry_run:
        logger.error("Binance client required for live mode but unavailable")
        return 1
    elif client is None:
        logger.info("Running in DRY RUN mode")

    gemini_on = getattr(args, "gemini", False)
    g_model = getattr(args, "gemini_model", "gemini-2.5-flash")
    if gemini_on:
        logger.info(f"Gemini overlay enabled (model={g_model})")

    nn_model = None
    nn_symbols = None
    nn_seq_len = 30
    if args.neural_model:
        try:
            nn_model, nn_symbols_loaded, nn_cfg = load_neural_model(args.neural_model)
            nn_seq_len = nn_cfg.get("seq_len", 30)
            if args.neural_symbols:
                nn_symbols = args.neural_symbols
            else:
                nn_symbols = nn_symbols_loaded
            logger.info(f"Neural model loaded: {args.neural_model} ({len(nn_symbols)} symbols, seq_len={nn_seq_len})")
        except Exception as e:
            logger.error(f"Failed to load neural model: {e}")
            nn_model = None

    if args.daemon:
        entry_poll_h = int(args.entry_poll_hours)
        health_h = int(args.health_report_hours)
        logger.info(f"Starting daemon mode: daily cycle at UTC 00:00, entry scan every {entry_poll_h}h, health every {health_h}h")
        last_cycle_date = None
        last_entry_scan_hour = None
        last_health_hour = None
        _min_dip = float(args.min_dip_pct)
        _adap_cycles = int(args.adaptive_dip_cycles)
        _dip_fallback = args.dip_pct_fallback
        if _dip_fallback:
            logger.info(f"Tiered dip fallback enabled: {_dip_fallback}")
        last_heartbeat_hour = None
        heartbeat_interval_h = 1
        if args.run_on_start:
            startup_dry_run = args.dry_run or args.startup_preview_only
            run_daily_cycle(
                client,
                symbols,
                config,
                dry_run=startup_dry_run,
                gemini_enabled=gemini_on,
                gemini_model=g_model,
                neural_model=nn_model,
                neural_model_symbols=nn_symbols,
                neural_seq_len=nn_seq_len,
                min_dip_pct=_min_dip,
                adaptive_dip_cycles=_adap_cycles,
                dip_pct_fallback=_dip_fallback,
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
                    neural_model=nn_model,
                    neural_model_symbols=nn_symbols,
                    neural_seq_len=nn_seq_len,
                    min_dip_pct=_min_dip,
                    adaptive_dip_cycles=_adap_cycles,
                    dip_pct_fallback=_dip_fallback,
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
                refreshed.extend(
                    reconcile_exit_orders(
                        client=client,
                        positions=positions,
                        last_exit=state.get("last_exit", {}),
                        now=now,
                    )
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
                        neural_model=nn_model,
                        neural_model_symbols=nn_symbols,
                        neural_seq_len=nn_seq_len,
                        dip_pct_fallback=_dip_fallback,
                    )
                    last_entry_scan_hour = current_hour

                if health_h > 0 and now.hour % health_h == 0 and current_hour != last_health_hour:
                    run_health_report(client, symbols, config, dry_run=args.dry_run)
                    last_health_hour = current_hour

            sleep_secs = (next_run - datetime.now(timezone.utc)).total_seconds()
            now_hb = datetime.now(timezone.utc)
            hb_hour = now_hb.strftime("%Y-%m-%dT%H")
            if hb_hour != last_heartbeat_hour:
                state_hb = load_state()
                n_pos = len(normalize_live_positions(state_hb.get("positions", {}), config))
                n_pend = len(_normalize_pending_entries(state_hb.get("pending_entries", {})))
                logger.info(
                    f"HEARTBEAT {now_hb.isoformat()} | positions={n_pos} pending={n_pend} "
                    f"next_daily={sleep_secs/3600:.1f}h poll={args.poll_seconds}s"
                )
                last_heartbeat_hour = hb_hour
            time.sleep(max(60, min(float(args.poll_seconds), sleep_secs)))
    else:
        run_daily_cycle(client, symbols, config, dry_run=args.dry_run,
                        gemini_enabled=gemini_on, gemini_model=g_model,
                        neural_model=nn_model, neural_model_symbols=nn_symbols,
                        neural_seq_len=nn_seq_len,
                        min_dip_pct=float(args.min_dip_pct),
                        adaptive_dip_cycles=int(args.adaptive_dip_cycles),
                        dip_pct_fallback=args.dip_pct_fallback)


if __name__ == "__main__":
    sys.exit(main() or 0)
