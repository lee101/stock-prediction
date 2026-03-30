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
import math
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from loguru import logger

from binance_worksteal.cli import (
    add_symbol_selection_args,
    print_resolved_symbols,
    resolve_cli_symbols_with_error,
)
from binance_worksteal.config_io import (
    add_config_file_arg,
    add_explain_config_arg,
    add_print_config_arg,
    build_worksteal_config_explanation,
    build_worksteal_config_from_args,
    maybe_handle_worksteal_config_output,
)
from binance_worksteal.reporting import (
    add_preview_run_arg,
    add_summary_json_arg,
    build_cli_error_summary,
    build_preview_run_summary,
    build_symbol_listing_summary,
    print_run_preview,
    run_with_optional_summary,
)
from binance_worksteal.strategy import (
    WorkStealConfig,
    SymbolDiagnostic,
    _build_symbol_metric_cache,
    build_entry_candidates,
    build_tiered_entry_candidates,
    compute_breadth_ratio,
    compute_ref_price,
    get_fee,
    load_daily_bars,
    passes_sma_filter,
    resolve_entry_regime,
    FDUSD_SYMBOLS,
)
from binance_worksteal.data import compute_features, FEATURE_NAMES
from binance_worksteal.model import DailyWorkStealPolicy, PerSymbolWorkStealPolicy
from binance_worksteal.io_utils import write_text_atomic
from binance_worksteal.universe import get_symbols as get_universe_symbols, load_universe

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
_ORDER_RULES_CACHE: dict[str, dict] = {}

LIVE_CONFIG_FLAG_TO_FIELD = {
    "--dip-pct": "dip_pct",
    "--proximity-pct": "proximity_pct",
    "--profit-target": "profit_target_pct",
    "--stop-loss": "stop_loss_pct",
    "--max-positions": "max_positions",
    "--max-position-pct": "max_position_pct",
    "--max-hold-days": "max_hold_days",
    "--lookback-days": "lookback_days",
    "--ref-method": ("ref_price_method", "ref_method"),
    "--sma-filter": "sma_filter_period",
    "--market-breadth-filter": "market_breadth_filter",
    "--trailing-stop": "trailing_stop_pct",
    "--entry-proximity-bps": "entry_proximity_bps",
    "--risk-off-ref-method": ("risk_off_ref_price_method", "risk_off_ref_method"),
    "--risk-off-market-breadth-filter": "risk_off_market_breadth_filter",
    "--risk-off-trigger-sma-period": "risk_off_trigger_sma_period",
    "--risk-off-trigger-momentum-period": "risk_off_trigger_momentum_period",
    "--rebalance-seeded-positions": "rebalance_seeded_positions",
    "--no-rebalance-seeded-positions": "rebalance_seeded_positions",
    "--dip-pct-fallback": "dip_pct_fallback",
}


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


def _normalize_strategy_symbols(symbols: List[str] | None) -> List[str]:
    normalized: List[str] = []
    seen: set[str] = set()
    for raw_symbol in symbols or []:
        symbol = _normalize_strategy_symbol(raw_symbol)
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        normalized.append(symbol)
    return normalized


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


def _safe_finite_float(value, default: float = 0.0) -> float:
    numeric = _safe_float(value, default=default)
    if math.isfinite(numeric):
        return numeric
    default_numeric = _safe_float(default, default=0.0)
    if math.isfinite(default_numeric):
        return default_numeric
    return 0.0


def _coerce_positive_finite_float(value, *, fallback: float = 0.0) -> float:
    numeric = _safe_float(value, default=fallback)
    if math.isfinite(numeric) and numeric > 0.0:
        return numeric
    fallback_numeric = _safe_float(fallback, default=0.0)
    if math.isfinite(fallback_numeric) and fallback_numeric > 0.0:
        return fallback_numeric
    return 0.0


def _finite_float_or_warn(value, *, context: str) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        logger.warning(f"Invalid {context}: expected a finite numeric value, got {value!r}")
        return None
    if not math.isfinite(numeric):
        logger.warning(f"Invalid {context}: expected a finite numeric value, got {value!r}")
        return None
    return numeric


def _mapping_response_or_warn(response, *, context: str) -> dict | None:
    if isinstance(response, dict):
        return response
    logger.warning(f"Invalid {context}: expected dict, got {type(response).__name__}")
    return None


def _list_response_or_warn(response, *, context: str) -> list | None:
    if isinstance(response, list):
        return response
    logger.warning(f"Invalid {context}: expected a list, got {type(response).__name__}")
    return None


def _submitted_margin_order_or_none(raw_order, *, context: str) -> tuple[dict | None, object | None]:
    order = None if raw_order is None else _mapping_response_or_warn(raw_order, context=context)
    if order is None:
        return None, None
    order_id = _coerce_order_id(order.get("orderId"))
    if order_id is None:
        return order, None
    return order, order_id


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _merge_peak_equity(existing, observed) -> float:
    existing_value = _safe_finite_float(existing, default=0.0)
    observed_value = _safe_finite_float(observed, default=0.0)
    return max(existing_value, observed_value, 0.0)


def _coerce_order_id(value) -> int | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            return None
    try:
        order_id = int(value)
    except (TypeError, ValueError):
        return None
    if order_id <= 0:
        return None
    return order_id


def _coerce_utc_timestamp(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        raise ValueError("timestamp is NaT")
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _safe_utc_timestamp(value, *, context: str, default: pd.Timestamp | None = None) -> pd.Timestamp | None:
    if value in (None, ""):
        return default
    try:
        return _coerce_utc_timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        logger.warning(f"Invalid {context}: {value!r} ({exc})")
        return default


def _safe_utc_timestamp_iso(value, *, context: str, default: datetime) -> str:
    timestamp = _safe_utc_timestamp(
        value,
        context=context,
        default=_coerce_utc_timestamp(default),
    )
    return (timestamp or _coerce_utc_timestamp(default)).isoformat()


def _extract_rule_float(filters: dict, filter_type: str, *keys: str) -> float:
    entry = filters.get(filter_type)
    if not isinstance(entry, dict):
        return 0.0
    for key in keys:
        value = _safe_float(entry.get(key), default=0.0)
        if value > 0.0:
            return value
    return 0.0


def _step_decimals(step: float) -> int:
    if step <= 0.0 or not math.isfinite(step):
        return 0
    text = f"{step:.10f}".rstrip("0")
    if "." not in text:
        return 0
    return len(text.split(".", 1)[1])


def _quantize_down(value: float, step: float) -> float:
    numeric = _safe_float(value, default=0.0)
    if numeric <= 0.0 or step <= 0.0 or not math.isfinite(step):
        return max(numeric, 0.0)
    return round(math.floor(numeric / step) * step, _step_decimals(step))


def _quantize_up(value: float, step: float) -> float:
    numeric = _safe_float(value, default=0.0)
    if numeric <= 0.0 or step <= 0.0 or not math.isfinite(step):
        return max(numeric, 0.0)
    return round(math.ceil(numeric / step) * step, _step_decimals(step))


def _format_decimal(value: float) -> str:
    return f"{float(value):.8f}".rstrip("0").rstrip(".") or "0"


def _order_rules_for_pair(client, pair: str) -> dict | None:
    normalized_pair = str(pair or "").upper().strip()
    cached = _ORDER_RULES_CACHE.get(normalized_pair)
    if cached is not None:
        return cached
    if client is None or not hasattr(client, "get_symbol_info"):
        return None
    try:
        info = client.get_symbol_info(normalized_pair)
    except Exception as exc:
        logger.warning(f"Failed to fetch symbol info for {normalized_pair}: {exc}")
        return None
    if not isinstance(info, dict):
        logger.warning(f"Invalid symbol info for {normalized_pair}: expected dict")
        return None
    raw_filters = info.get("filters")
    if not isinstance(raw_filters, list):
        logger.warning(f"Missing symbol filters for {normalized_pair}")
        return None
    parsed: dict = {}
    for entry in raw_filters:
        if not isinstance(entry, dict):
            continue
        filter_type = str(entry.get("filterType") or "").strip()
        if filter_type:
            parsed[filter_type] = entry
    if not parsed:
        logger.warning(f"No usable symbol filters for {normalized_pair}")
        return None
    _ORDER_RULES_CACHE[normalized_pair] = parsed
    return parsed


def _prepare_limit_order(client, pair: str, side: str, price: float, quantity: float) -> tuple[float, float, str | None]:
    adj_price = _safe_float(price, default=0.0)
    adj_qty = _safe_float(quantity, default=0.0)
    if adj_price <= 0.0:
        return 0.0, 0.0, "invalid_price"
    if adj_qty <= 0.0:
        return 0.0, 0.0, "invalid_quantity"

    filters = _order_rules_for_pair(client, pair)
    if filters is None:
        return 0.0, 0.0, "missing_symbol_rules"
    tick_size = _extract_rule_float(filters, "PRICE_FILTER", "tickSize")
    step_size = _extract_rule_float(filters, "LOT_SIZE", "stepSize")
    min_qty = _extract_rule_float(filters, "LOT_SIZE", "minQty")
    min_price = _extract_rule_float(filters, "PRICE_FILTER", "minPrice")
    min_notional = max(
        _extract_rule_float(filters, "MIN_NOTIONAL", "minNotional"),
        _extract_rule_float(filters, "NOTIONAL", "minNotional", "notional"),
    )

    side_norm = str(side or "").upper().strip()
    if tick_size > 0.0:
        if side_norm == "BUY":
            adj_price = _quantize_down(adj_price, tick_size)
        else:
            adj_price = _quantize_up(adj_price, tick_size)
    if step_size > 0.0:
        adj_qty = _quantize_down(adj_qty, step_size)

    if adj_price <= 0.0:
        return 0.0, 0.0, "price_quantized_to_zero"
    if adj_qty <= 0.0:
        return 0.0, 0.0, "qty_quantized_to_zero"
    if min_price > 0.0 and adj_price < min_price:
        return 0.0, 0.0, "price_below_min_price"
    if min_qty > 0.0 and adj_qty < min_qty:
        return 0.0, 0.0, "qty_below_min_qty"
    if min_notional > 0.0 and (adj_price * adj_qty) < min_notional:
        return 0.0, 0.0, "notional_below_min_notional"

    return float(adj_price), float(adj_qty), None


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
    executed_qty = _coerce_positive_finite_float(order.get("executedQty"), fallback=0.0)
    cumulative_quote = _coerce_positive_finite_float(order.get("cummulativeQuoteQty"), fallback=0.0)
    if executed_qty > 0.0 and cumulative_quote > 0.0:
        return cumulative_quote / executed_qty
    avg_price = _coerce_positive_finite_float(order.get("avgPrice"), fallback=0.0)
    if avg_price > 0.0:
        return avg_price
    price = _coerce_positive_finite_float(order.get("price"), fallback=0.0)
    if price > 0.0:
        return price
    return _coerce_positive_finite_float(fallback, fallback=0.0)


def _order_timestamp_iso(order: dict, *, fallback: datetime) -> str:
    raw_ts = order.get("updateTime") or order.get("time")
    try:
        timestamp = pd.Timestamp(int(raw_ts), unit="ms", tz="UTC")
    except (TypeError, ValueError, OverflowError):
        return fallback.isoformat()
    return timestamp.isoformat()


def _order_update_time_key(order: dict) -> int:
    return _safe_int(order.get("updateTime") or order.get("time") or 0, default=0)



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
    filtered.sort(key=_order_update_time_key, reverse=True)
    return filtered[0]


def _margin_position_quantity(balance_row: dict) -> float:
    asset = str(balance_row.get("asset") or "?").upper().strip() or "?"
    borrowed_qty = _finite_float_or_warn(
        balance_row.get("borrowed"),
        context=f"margin account borrowed quantity for {asset}",
    )
    if borrowed_qty is None or borrowed_qty > 1e-8:
        return 0.0
    free_qty = _finite_float_or_warn(
        balance_row.get("free"),
        context=f"margin account free quantity for {asset}",
    )
    net_qty = _finite_float_or_warn(
        balance_row.get("netAsset"),
        context=f"margin account netAsset quantity for {asset}",
    )
    return max(free_qty or 0.0, net_qty or 0.0, 0.0)


def _clear_pending_exit(position: dict) -> None:
    for key in ("exit_order_id", "exit_order_symbol", "exit_order_status", "exit_price", "exit_reason"):
        position.pop(key, None)


def _refresh_margin_order(client, *, pair: str, order_id, purpose: str) -> dict | None:
    try:
        order = client.get_margin_order(symbol=pair, orderId=order_id)
    except Exception as exc:
        logger.warning(f"Failed to refresh {purpose} {pair}#{order_id}: {exc}")
        return None
    return _mapping_response_or_warn(order, context=f"margin order response for {pair}#{order_id}")


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
        orders = _list_response_or_warn(orders, context=f"recent margin orders for {pair}")
        if orders:
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
    parser = argparse.ArgumentParser(
        description="Live Binance worksteal bot with dry-run, daemon, and diagnostics modes.",
    )
    add_symbol_selection_args(parser)
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
    add_config_file_arg(parser)
    add_print_config_arg(parser)
    add_explain_config_arg(parser)
    add_preview_run_arg(parser)
    add_summary_json_arg(parser)
    return parser


def build_runtime_cli_default_config(args: argparse.Namespace) -> WorkStealConfig:
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
        dip_pct_fallback=tuple(float(x) for x in (args.dip_pct_fallback or [])),
    )


def build_runtime_config(
    args: argparse.Namespace,
    raw_argv: list[str] | None = None,
) -> WorkStealConfig:
    return build_worksteal_config_from_args(
        base_config=build_runtime_cli_default_config(args),
        config_file=args.config_file,
        args=args,
        raw_argv=raw_argv,
        flag_to_field=LIVE_CONFIG_FLAG_TO_FIELD,
    )


def _format_preview_override_value(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (list, tuple, set)):
        rendered = [str(item) for item in value]
        return ", ".join(rendered) if rendered else "-"
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)



def _build_config_override_context(
    args: argparse.Namespace,
    raw_argv: list[str] | None,
) -> dict[str, object]:
    explanation = build_worksteal_config_explanation(
        base_config=build_runtime_cli_default_config(args),
        config_file=args.config_file,
        args=args,
        raw_argv=raw_argv,
        flag_to_field=LIVE_CONFIG_FLAG_TO_FIELD,
    )
    changed_fields = explanation.get("changed_fields") or {}
    config_file_overrides = explanation.get("config_file_overrides") or {}
    cli_overrides = explanation.get("cli_overrides") or {}
    override_details = [
        {
            "field": field_name,
            "source": str(change.get("source") or "default"),
            "value": change.get("value"),
        }
        for field_name, change in changed_fields.items()
    ]
    return {
        "config_override_count": len(override_details),
        "config_override_fields": [item["field"] for item in override_details],
        "config_file_override_fields": list(config_file_overrides),
        "cli_override_fields": list(cli_overrides),
        "config_override_preview": [
            f"{item['field']}={_format_preview_override_value(item['value'])} ({item['source']})"
            for item in override_details
        ],
        "config_changed_fields": changed_fields,
    }


def normalize_live_positions(raw_positions: dict, config: WorkStealConfig) -> dict:
    normalized = {}
    now = datetime.now(timezone.utc)
    for raw_symbol, raw_position in (raw_positions or {}).items():
        if not isinstance(raw_position, dict):
            continue
        symbol = _normalize_strategy_symbol(raw_symbol)
        entry_price = _safe_finite_float(raw_position.get("entry_price", 0.0), default=0.0)
        quantity = _safe_finite_float(raw_position.get("quantity", 0.0), default=0.0)
        if not symbol or entry_price <= 0.0 or quantity <= 0.0:
            continue
        default_target = entry_price * (1.0 + config.profit_target_pct)
        default_stop = entry_price * (1.0 - config.stop_loss_pct)
        raw_target = _safe_finite_float(raw_position.get("target_sell", default_target), default=default_target)
        raw_stop = _safe_finite_float(raw_position.get("stop_price", default_stop), default=default_stop)
        target_sell = raw_target if raw_target > entry_price else default_target
        stop_price = raw_stop if 0.0 < raw_stop < entry_price else default_stop
        peak_price = max(_safe_finite_float(raw_position.get("peak_price", entry_price), default=entry_price), entry_price)
        normalized[symbol] = {
            "entry_price": entry_price,
            "entry_date": _safe_utc_timestamp_iso(
                raw_position.get("entry_date"),
                context=f"positions[{symbol}].entry_date",
                default=now,
            ),
            "quantity": quantity,
            "peak_price": peak_price,
            "target_sell": target_sell,
            "stop_price": stop_price,
            "source": str(raw_position.get("source") or "legacy"),
        }
        exit_order_id = _coerce_order_id(raw_position.get("exit_order_id"))
        if exit_order_id is not None:
            normalized[symbol]["exit_order_id"] = exit_order_id
            normalized[symbol]["exit_order_symbol"] = str(raw_position.get("exit_order_symbol") or "")
            normalized[symbol]["exit_order_status"] = str(raw_position.get("exit_order_status") or "")
            exit_price = _safe_finite_float(raw_position.get("exit_price", target_sell), default=target_sell)
            normalized[symbol]["exit_price"] = exit_price if exit_price > 0.0 else target_sell
            normalized[symbol]["exit_reason"] = str(raw_position.get("exit_reason") or "")
    return normalized


def _prune_last_exit_for_open_positions(last_exit: dict, positions: dict) -> dict:
    held = {_normalize_strategy_symbol(symbol) for symbol in (positions or {})}
    pruned: dict = {}
    for raw_symbol, ts in (last_exit or {}).items():
        symbol = _normalize_strategy_symbol(raw_symbol)
        if not symbol or symbol in held:
            continue
        pruned[symbol] = ts
    return pruned


def _normalize_last_exit_state(raw_last_exit: dict) -> dict:
    normalized: dict[str, str] = {}
    for raw_symbol, raw_timestamp in (raw_last_exit or {}).items():
        symbol = _normalize_strategy_symbol(raw_symbol)
        if not symbol:
            continue
        timestamp = _safe_utc_timestamp(
            raw_timestamp,
            context=f"last_exit[{symbol}]",
            default=None,
        )
        if timestamp is None:
            continue
        normalized[symbol] = timestamp.isoformat()
    return normalized


def _last_exit_timestamps(last_exit: dict) -> dict[str, pd.Timestamp]:
    return {
        symbol: timestamp
        for symbol, raw_timestamp in (last_exit or {}).items()
        if (timestamp := _safe_utc_timestamp(raw_timestamp, context=f"last_exit[{symbol}]", default=None)) is not None
    }


def _normalize_pending_entries(raw_pending: dict) -> dict:
    normalized = {}
    now = datetime.now(timezone.utc)
    for raw_symbol, raw_entry in (raw_pending or {}).items():
        if not isinstance(raw_entry, dict):
            continue
        symbol = _normalize_strategy_symbol(raw_symbol)
        if not symbol:
            continue
        placed_at_raw = raw_entry.get("placed_at")
        expires_at_raw = raw_entry.get("expires_at")
        normalized[symbol] = {
            "buy_price": _safe_finite_float(raw_entry.get("buy_price", 0.0), default=0.0),
            "quantity": _safe_finite_float(raw_entry.get("quantity", 0.0), default=0.0),
            "target_sell": _safe_finite_float(raw_entry.get("target_sell", 0.0), default=0.0),
            "stop_price": _safe_finite_float(raw_entry.get("stop_price", 0.0), default=0.0),
            "placed_at": _safe_utc_timestamp_iso(
                placed_at_raw,
                context=f"pending_entries[{symbol}].placed_at",
                default=now,
            ),
            "expires_at": _safe_utc_timestamp_iso(
                expires_at_raw,
                context=f"pending_entries[{symbol}].expires_at",
                default=now + (PENDING_ENTRY_TTL if expires_at_raw in (None, "") else timedelta(0)),
            ),
            "order_id": _coerce_order_id(raw_entry.get("order_id")),
            "confidence": _safe_finite_float(raw_entry.get("confidence", 1.0), default=1.0),
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
    entry_regime = resolve_entry_regime(current_bars=current_bars, history=history, config=legacy_config)
    entry_config = entry_regime.config
    if entry_regime.skip_entries:
        return [], set()

    candidates = build_entry_candidates(
        date=pd.Timestamp(now) if not isinstance(now, pd.Timestamp) else now,
        current_bars=current_bars,
        history=history,
        positions={},
        last_exit=_last_exit_timestamps(last_exit),
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


def _build_pair_routing(symbols: List[str]) -> list[dict[str, str]]:
    routing: list[dict[str, str]] = []
    for symbol in symbols:
        routing.append(
            {
                "symbol": symbol,
                "data_pair": get_binance_pair(symbol, prefer_fdusd=False),
                "order_pair": get_binance_pair(symbol, prefer_fdusd=True),
            }
        )
    return routing



def _pair_quote_asset(pair: str) -> str:
    value = str(pair or "").upper().strip()
    for quote in ("FDUSD", "USDT"):
        if value.endswith(quote):
            return quote
    return ""



def _build_pair_routing_summary(routes: list[dict[str, str]]) -> dict[str, object]:
    cross_quote_symbols = [
        str(route.get("symbol") or "")
        for route in routes
        if str(route.get("data_pair") or "") != str(route.get("order_pair") or "")
    ]
    data_quote_counts: dict[str, int] = {}
    order_quote_counts: dict[str, int] = {}
    for route in routes:
        data_quote = _pair_quote_asset(str(route.get("data_pair") or ""))
        if data_quote:
            data_quote_counts[data_quote] = data_quote_counts.get(data_quote, 0) + 1
        order_quote = _pair_quote_asset(str(route.get("order_pair") or ""))
        if order_quote:
            order_quote_counts[order_quote] = order_quote_counts.get(order_quote, 0) + 1

    def _ordered_quote_counts(counts: dict[str, int]) -> dict[str, int]:
        ordered = {
            quote: counts[quote]
            for quote in ("FDUSD", "USDT")
            if quote in counts
        }
        for quote in sorted(counts):
            if quote not in ordered:
                ordered[quote] = counts[quote]
        return ordered

    ordered_data_quote_counts = _ordered_quote_counts(data_quote_counts)
    ordered_order_quote_counts = _ordered_quote_counts(order_quote_counts)
    return {
        "route_count": len(routes),
        "same_pair_count": len(routes) - len(cross_quote_symbols),
        "cross_quote_count": len(cross_quote_symbols),
        "cross_quote_symbols": cross_quote_symbols,
        "data_quote_counts": ordered_data_quote_counts,
        "order_quote_counts": ordered_order_quote_counts,
        "required_order_quotes": list(ordered_order_quote_counts),
    }



def _print_pair_routing(routes: list[dict[str, str]]) -> None:
    if not routes:
        return
    print("Pair routing:")
    for route in routes:
        print(
            f"  {route['symbol']}: data={route['data_pair']} order={route['order_pair']}"
        )
    summary = _build_pair_routing_summary(routes)
    print("Routing summary:")
    print(f"  total routes: {summary['route_count']}")
    print(f"  same-pair routes: {summary['same_pair_count']}")
    print(f"  cross-quote routes: {summary['cross_quote_count']}")
    if summary["cross_quote_symbols"]:
        print(f"  cross-quote symbols: {', '.join(summary['cross_quote_symbols'])}")
    if summary["data_quote_counts"]:
        mix = ", ".join(
            f"{quote}={count}" for quote, count in summary["data_quote_counts"].items()
        )
        print(f"  data quote mix: {mix}")
    if summary["order_quote_counts"]:
        mix = ", ".join(
            f"{quote}={count}" for quote, count in summary["order_quote_counts"].items()
        )
        print(f"  order quote mix: {mix}")
    if summary["required_order_quotes"]:
        print(f"  required order quotes: {', '.join(summary['required_order_quotes'])}")


def _entry_regime_breadth_snapshot(
    current_bars: Dict[str, pd.Series],
    history: Dict[str, pd.DataFrame],
    entry_regime,
) -> tuple[float, int, int]:
    if entry_regime.market_breadth_total_count > 0:
        return (
            float(entry_regime.market_breadth_ratio),
            int(entry_regime.market_breadth_dipping_count),
            int(entry_regime.market_breadth_total_count),
        )
    ratio, dipping, total = compute_breadth_ratio(current_bars, history)
    return float(ratio), int(dipping), int(total)



def _print_omitted_symbols(symbols: list[str]) -> None:
    if not symbols:
        return
    print("Excluded by --max-symbols:")
    for symbol in symbols:
        print(f"  {symbol}")


def _resolve_symbol_selection_context(
    symbols: List[str],
    symbol_source: str,
    *,
    max_symbols: int,
) -> dict[str, object]:
    requested_symbols = list(symbols)
    resolved_symbols = list(requested_symbols)
    was_capped = len(resolved_symbols) > max_symbols
    omitted_symbols: list[str] = []
    if was_capped:
        omitted_symbols = resolved_symbols[max_symbols:]
        resolved_symbols = resolved_symbols[:max_symbols]
    display_source = symbol_source
    if was_capped:
        display_source = f"{symbol_source} (capped to --max-symbols={max_symbols})"
    pair_routing = _build_pair_routing(resolved_symbols)
    return {
        "symbol_source": display_source,
        "symbols": resolved_symbols,
        "requested_symbol_count": len(requested_symbols),
        "symbol_count": len(resolved_symbols),
        "was_capped": was_capped,
        "omitted_symbol_count": len(omitted_symbols),
        "omitted_symbols": omitted_symbols,
        "pair_routing": pair_routing,
        "pair_routing_summary": _build_pair_routing_summary(pair_routing),
    }


def get_binance_pair(symbol: str, prefer_fdusd: bool = True) -> str:
    base = symbol.replace("USD", "")
    if prefer_fdusd and symbol in FDUSD_SYMBOLS and symbol in SYMBOL_PAIRS:
        return SYMBOL_PAIRS[symbol]["fdusd"]
    return f"{base}USDT"


def _load_local_daily_bars(
    symbols: List[str],
    *,
    min_rows_exclusive: int,
    tail_rows: int,
) -> Dict[str, pd.DataFrame]:
    remaining = list(dict.fromkeys(symbols))
    all_bars: Dict[str, pd.DataFrame] = {}
    for data_dir in ("trainingdatadailybinance", "trainingdata/train"):
        if not remaining:
            break
        try:
            loaded = load_daily_bars(data_dir, remaining)
        except Exception as e:
            logger.warning(f"Failed to load local bars from {data_dir}: {e}")
            loaded = {}
        if not isinstance(loaded, dict):
            logger.warning(
                f"Invalid local bars response from {data_dir}: expected dict, got {type(loaded).__name__}"
            )
            loaded = {}
        next_remaining: List[str] = []
        for sym in remaining:
            bars = loaded.get(sym)
            if bars is not None and not isinstance(bars, pd.DataFrame):
                logger.warning(
                    f"Invalid local bars for {sym} from {data_dir}: expected DataFrame, got {type(bars).__name__}"
                )
                bars = None
            if bars is not None and not bars.empty and len(bars) > min_rows_exclusive:
                all_bars[sym] = bars.tail(tail_rows).copy()
            else:
                next_remaining.append(sym)
        remaining = next_remaining
    return all_bars



def fetch_daily_bars(client, symbol: str, lookback_days: int = 30) -> pd.DataFrame:
    if client is None:
        return _load_local_daily_bars(
            [symbol],
            min_rows_exclusive=0,
            tail_rows=lookback_days + 5,
        ).get(symbol, pd.DataFrame())

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

    if not isinstance(klines, (list, tuple)):
        logger.warning(f"Invalid kline response for {pair}: expected a list, got {type(klines).__name__}")
        return pd.DataFrame()

    rows = []
    for idx, k in enumerate(klines):
        if not isinstance(k, (list, tuple)) or len(k) < 6:
            logger.warning(f"Skipping malformed kline row {idx} for {pair}: expected at least 6 fields")
            continue
        try:
            timestamp = pd.Timestamp(k[0], unit="ms", tz="UTC")
            open_ = float(k[1])
            high = float(k[2])
            low = float(k[3])
            close = float(k[4])
            volume = float(k[5])
        except (TypeError, ValueError) as exc:
            logger.warning(f"Skipping malformed kline row {idx} for {pair}: {exc}")
            continue
        if pd.isna(timestamp) or not all(math.isfinite(v) for v in (open_, high, low, close, volume)):
            logger.warning(f"Skipping malformed kline row {idx} for {pair}: non-finite values")
            continue
        rows.append({
            "timestamp": timestamp,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "symbol": symbol,
        })
    return pd.DataFrame(rows)


def _default_live_state() -> dict:
    return {
        "positions": {},
        "pending_entries": {},
        "last_exit": {},
        "recent_trades": [],
        "peak_equity": 0.0,
    }


def _normalize_live_state_payload(payload: object, path: Path) -> dict:
    defaults = _default_live_state()
    state = dict(payload)
    invalid_fields: List[str] = []

    for key in ("positions", "pending_entries", "last_exit"):
        value = state.get(key)
        if value is None and key not in state:
            state[key] = {}
            continue
        if isinstance(value, dict):
            continue
        state[key] = {}
        invalid_fields.append(key)

    recent_trades = state.get("recent_trades", defaults["recent_trades"])
    if isinstance(recent_trades, list):
        filtered_recent_trades = [trade for trade in recent_trades if isinstance(trade, dict)]
        state["recent_trades"] = filtered_recent_trades
        if len(filtered_recent_trades) != len(recent_trades):
            invalid_fields.append("recent_trades_items")
    else:
        state["recent_trades"] = []
        invalid_fields.append("recent_trades")

    peak_equity = state.get("peak_equity", defaults["peak_equity"])
    try:
        normalized_peak_equity = float(peak_equity or 0.0)
    except (TypeError, ValueError):
        normalized_peak_equity = float("nan")
    if math.isfinite(normalized_peak_equity):
        state["peak_equity"] = normalized_peak_equity
    else:
        state["peak_equity"] = 0.0
        invalid_fields.append("peak_equity")

    if invalid_fields:
        joined = ", ".join(sorted(set(invalid_fields)))
        logger.warning(f"Reset invalid live state fields in {path}: {joined}")
    return state


def _write_json_atomic(path: Path, payload: object):
    serialized = json.dumps(payload, indent=2, default=str)
    try:
        write_text_atomic(path, serialized, encoding="utf-8")
    except OSError as exc:
        logger.error(f"Failed to write {path}: {exc}")
        raise


def _quarantine_invalid_state_file(path: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = path.with_name(f"{path.stem}.corrupt.{stamp}{path.suffix}")
    counter = 1
    while backup_path.exists():
        backup_path = path.with_name(f"{path.stem}.corrupt.{stamp}.{counter}{path.suffix}")
        counter += 1
    try:
        path.replace(backup_path)
    except OSError as exc:
        logger.warning(f"Failed to quarantine invalid live state {path}: {exc}")
        return None
    return backup_path


def load_state() -> dict:
    if not STATE_FILE.exists():
        return _default_live_state()

    try:
        payload = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        backup_path = None
        if isinstance(exc, (UnicodeDecodeError, json.JSONDecodeError)):
            backup_path = _quarantine_invalid_state_file(STATE_FILE)
        if backup_path is not None:
            logger.warning(
                f"Failed to load live state from {STATE_FILE}: {exc}. "
                f"Moved corrupt state to {backup_path} and starting with empty state."
            )
        else:
            logger.warning(
                f"Failed to load live state from {STATE_FILE}: {exc}. Starting with empty state."
            )
        return _default_live_state()
    if not isinstance(payload, dict):
        backup_path = _quarantine_invalid_state_file(STATE_FILE)
        if backup_path is not None:
            logger.warning(
                f"Invalid live state in {STATE_FILE}: expected a JSON object, got {type(payload).__name__}. "
                f"Moved corrupt state to {backup_path} and starting with empty state."
            )
        else:
            logger.warning(
                f"Invalid live state in {STATE_FILE}: expected a JSON object, got {type(payload).__name__}. "
                "Starting with empty state."
            )
        return _default_live_state()
    return _normalize_live_state_payload(payload, STATE_FILE)


def save_state(state: dict):
    payload = _normalize_live_state_payload(state, STATE_FILE)
    _write_json_atomic(STATE_FILE, payload)


def _append_jsonl(path: Path, payload: object, *, kind: str):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")
    except OSError as exc:
        logger.warning(f"Failed to append {kind} to {path}: {exc}")


def _build_market_data_gap_event(*, event_type: str, symbols: List[str], positions: dict, pending_entries: dict, equity: float) -> dict:
    return {
        "type": event_type,
        "status": "no_market_data",
        "n_symbols_requested": len(symbols),
        "n_symbols_with_data": 0,
        "n_positions": len(positions),
        "n_pending": len(pending_entries),
        "equity": equity,
    }


def log_trade(trade: dict):
    _append_jsonl(LOG_FILE, trade, kind="trade log entry")


def log_event(event: dict):
    event.setdefault("ts", datetime.now(timezone.utc).isoformat())
    _append_jsonl(EVENTS_FILE, event, kind="event log entry")


def load_universe_file(path: str) -> List[str]:
    """Backward-compatible universe loader used by older tests and scripts."""
    return get_universe_symbols(load_universe(path))


def _fetch_all_bars(client, symbols: List[str], lookback_days: int,
                    max_workers: int = 10) -> Dict[str, pd.DataFrame]:
    t0 = time.monotonic()

    if client is None and fetch_daily_bars is _ORIGINAL_FETCH_DAILY_BARS:
        all_bars = _load_local_daily_bars(
            list(symbols),
            min_rows_exclusive=lookback_days,
            tail_rows=lookback_days + 15,
        )
        elapsed = time.monotonic() - t0
        logger.info(f"Fetched bars for {len(all_bars)}/{len(symbols)} symbols in {elapsed:.1f}s")
        return all_bars

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


_ORIGINAL_FETCH_DAILY_BARS = fetch_daily_bars


def get_account_equity(client) -> float:
    try:
        info = _mapping_response_or_warn(
            client.get_margin_account(),
            context="margin account response for equity",
        )
        if info is None:
            return 0.0
        ticker = _mapping_response_or_warn(
            client.get_symbol_ticker(symbol="BTCUSDT"),
            context="symbol ticker response for BTCUSDT",
        )
        if ticker is None:
            return 0.0
        total_net_asset_btc = _finite_float_or_warn(
            info.get("totalNetAssetOfBtc"),
            context="margin account totalNetAssetOfBtc",
        )
        if total_net_asset_btc is None:
            return 0.0
        btcusdt_price = _finite_float_or_warn(
            ticker.get("price"),
            context="BTCUSDT ticker price",
        )
        if btcusdt_price is None:
            return 0.0
        if btcusdt_price <= 0.0:
            logger.warning(
                f"Invalid BTCUSDT ticker price: expected a positive numeric value, got {ticker.get('price')!r}"
            )
            return 0.0
        equity = total_net_asset_btc * btcusdt_price
        if not math.isfinite(equity):
            logger.warning(
                f"Invalid computed equity from totalNetAssetOfBtc={info.get('totalNetAssetOfBtc')!r} "
                f"and BTCUSDT price={ticker.get('price')!r}"
            )
            return 0.0
        return equity
    except Exception as e:
        logger.error(f"Failed to get equity: {e}")
        return 0.0


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
    return _format_decimal(price)


def _format_quantity(qty: float) -> str:
    return _format_decimal(qty)


def _place_limit_order(client, symbol: str, side: str, price: float, quantity: float):
    pair = get_binance_pair(symbol, prefer_fdusd=True)
    adj_price, adj_qty, reason = _prepare_limit_order(client, pair, side, price, quantity)
    side_lower = side.lower()
    if reason is not None:
        logger.warning(
            f"Rejected limit {side_lower} before submit: {pair} price={price:.8f} qty={quantity:.8f} reason={reason}"
        )
        return None
    price_str = _format_price(adj_price)
    qty_str = _format_quantity(adj_qty)
    logger.info(f"Placing limit {side_lower}: {pair} qty={qty_str} price={price_str}")
    try:
        return client.create_margin_order(
            symbol=pair,
            side=side,
            type="LIMIT",
            timeInForce="GTC",
            quantity=qty_str,
            price=price_str,
        )
    except Exception as e:
        logger.error(f"Limit {side_lower} failed for {pair}: {e}")
        return None


def place_limit_buy(client, symbol: str, price: float, quantity: float, config: WorkStealConfig):
    return _place_limit_order(client, symbol, "BUY", price, quantity)


def place_limit_sell(client, symbol: str, price: float, quantity: float):
    return _place_limit_order(client, symbol, "SELL", price, quantity)


def _cancel_pending_entry(client, symbol: str, entry: dict) -> bool:
    order_id = entry.get("order_id")
    if client is None or order_id is None:
        return True
    pair = str(entry.get("order_symbol") or get_binance_pair(symbol, prefer_fdusd=True)).upper().strip()
    try:
        client.cancel_margin_order(symbol=pair, orderId=order_id)
        logger.info(f"Cancelled pending entry: {pair} orderId={order_id}")
        return True
    except Exception as exc:
        logger.warning(f"Failed to cancel pending entry for {pair}: {exc}")
        return False


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
        expires_at = _safe_utc_timestamp(
            entry.get("expires_at", now.isoformat()),
            context=f"pending_entries[{sym}].expires_at",
            default=_coerce_utc_timestamp(now),
        )
        if expires_at is None:
            expires_at = _coerce_utc_timestamp(now)
        if pd.Timestamp(now) >= expires_at:
            if _cancel_pending_entry(client, sym, entry):
                del pending_entries[sym]
            continue

        if entry.get("status") == "preview" and not dry_run:
            logger.info(f"Cleaning up preview entry {sym} (no order placed)")
            del pending_entries[sym]
            continue

        if dry_run or client is None or entry.get("order_id") is None:
            continue

        pair = str(entry.get("order_symbol") or get_binance_pair(sym, prefer_fdusd=True)).upper().strip()
        order = _refresh_margin_order(
            client,
            pair=pair,
            order_id=entry["order_id"],
            purpose="pending entry",
        )
        if order is None:
            continue

        status = _order_status(order)
        if status == "FILLED":
            fill_qty = _coerce_positive_finite_float(
                order.get("executedQty"),
                fallback=entry.get("quantity") or 0.0,
            )
            fill_price = _order_avg_price(order, fallback=float(entry.get("buy_price") or 0.0))
            if fill_qty <= 0.0 or fill_price <= 0.0:
                logger.warning(
                    f"Pending entry fill for {pair}#{entry['order_id']} missing execution details; keeping pending entry for retry"
                )
                entry["status"] = "fill_unconfirmed"
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
        order = _refresh_margin_order(
            client,
            pair=pair,
            order_id=order_id,
            purpose="exit order",
        )
        if order is None:
            continue

        status = _order_status(order)
        position["exit_order_status"] = status
        if status == "FILLED":
            fill_qty = _coerce_positive_finite_float(
                order.get("executedQty"),
                fallback=position.get("quantity", 0.0),
            )
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
            else:
                logger.warning(
                    f"Exit order fill for {pair}#{order_id} missing execution details; keeping position until execution can be confirmed"
                )
                continue
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
    account = _mapping_response_or_warn(account, context="margin account response for position sync")
    if account is None:
        return []

    try:
        open_orders = client.get_open_margin_orders(isIsolated="FALSE")
    except Exception as exc:
        logger.warning(f"Failed to fetch open margin orders for position sync: {exc}")
        open_orders = []
    else:
        open_orders = _list_response_or_warn(open_orders, context="open margin orders response for position sync") or []

    tracked_symbols = {_normalize_strategy_symbol(symbol) for symbol in symbols}
    sell_orders_by_symbol: dict[str, dict] = {}
    for order in open_orders:
        if not isinstance(order, dict):
            continue
        symbol = _normalize_strategy_symbol(order.get("symbol", ""))
        if symbol not in tracked_symbols:
            continue
        if str(order.get("side") or "").upper().strip() != "SELL":
            continue
        existing = sell_orders_by_symbol.get(symbol)
        existing_ts = _order_update_time_key(existing) if existing else -1
        candidate_ts = _order_update_time_key(order)
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
            exit_order_id = _coerce_order_id(open_sell.get("orderId"))
            if exit_order_id is None:
                logger.warning(
                    f"Invalid open sell order id for {symbol}: {open_sell.get('orderId')!r}; preserving existing exit state"
                )
            else:
                position["exit_order_id"] = exit_order_id
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
        order_id = None
        if not dry_run:
            raw_order = place_limit_buy(client, sym, buy_price, quantity, entry_config)
            order, order_id = _submitted_margin_order_or_none(
                raw_order,
                context=f"margin order submit response for entry {sym}",
            )
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
            "order_id": order_id,
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


def _count_sma_pass_fail(
    all_bars: Dict[str, pd.DataFrame],
    config: WorkStealConfig,
    *,
    symbol_metrics: Optional[dict] = None,
) -> Tuple[int, int]:
    n_pass = 0
    n_fail = 0
    for sym, bars in all_bars.items():
        metrics = symbol_metrics.get(sym) if symbol_metrics is not None else None
        close = metrics.close if metrics is not None else float(bars.iloc[-1]["close"])
        if passes_sma_filter(bars, config, close, metrics=metrics):
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
    for idx, t in enumerate(reversed(recent_trades)):
        if not isinstance(t, dict):
            continue
        ts_str = t.get("timestamp")
        if ts_str:
            context = f"recent_trades[{len(recent_trades) - idx - 1}].timestamp"
            last_trade_ts = _safe_utc_timestamp(ts_str, context=context, default=None)
            if last_trade_ts is not None:
                break
    days_since_trade = (now - last_trade_ts).days if last_trade_ts else -1

    all_bars = _fetch_all_bars(client, symbols, config.lookback_days)
    current_bars = {sym: bars.iloc[-1] for sym, bars in all_bars.items()}
    if not all_bars:
        logger.warning(f"HEALTH: no market data fetched for any of {len(symbols)} symbols")
        log_event({
            **_build_market_data_gap_event(
                event_type="health_report",
                symbols=symbols,
                positions=positions,
                pending_entries=pending_entries,
                equity=equity,
            ),
            "health_status": "no_market_data",
            "risk_off": False,
            "risk_off_triggered": False,
            "market_breadth_skip": False,
            "entry_skip": False,
            "nearest_dip_sym": "",
            "nearest_dip_bps": None,
            "days_since_trade": days_since_trade,
        })
        return
    if not dry_run:
        synchronize_positions_from_exchange(
            client=client,
            symbols=symbols,
            positions=positions,
            current_bars=current_bars,
            config=config,
            now=now,
        )
    symbol_metrics = _build_symbol_metric_cache(current_bars, all_bars)
    entry_regime = resolve_entry_regime(
        current_bars=current_bars,
        history=all_bars,
        config=config,
        symbol_metrics=symbol_metrics,
    )
    entry_config = entry_regime.config
    risk_off = entry_regime.risk_off
    market_breadth_skip = entry_regime.market_breadth_skip

    nearest_dip_bps = float("inf")
    nearest_dip_sym = ""
    for sym, bars in all_bars.items():
        if sym in positions:
            continue
        metrics = symbol_metrics.get(sym)
        close = metrics.close if metrics is not None else float(bars.iloc[-1]["close"])
        ref_high = (
            metrics.ref_high(entry_config.ref_price_method, config.lookback_days)
            if metrics is not None
            else compute_ref_price(bars, entry_config.ref_price_method, config.lookback_days)
        )
        buy_target = (
            metrics.buy_target(ref_high, entry_config)
            if metrics is not None
            else ref_high * (1 - entry_config.dip_pct)
        )
        dist_bps = (close - buy_target) / close * 10_000.0 if close > 0 else float("inf")
        if dist_bps < nearest_dip_bps:
            nearest_dip_bps = dist_bps
            nearest_dip_sym = sym

    logger.info(
        f"HEALTH: equity=${equity:.0f} positions={len(positions)} pending={len(pending_entries)} "
        f"regime={'risk-off' if entry_regime.skip_entries else 'risk-on'} "
        f"nearest_dip={nearest_dip_sym}@{nearest_dip_bps:.0f}bps "
        f"days_since_trade={days_since_trade}"
    )
    log_event({
        "type": "health_report",
        "equity": equity,
        "n_positions": len(positions),
        "n_pending": len(pending_entries),
        "risk_off": entry_regime.skip_entries,
        "risk_off_triggered": risk_off,
        "market_breadth_skip": market_breadth_skip,
        "entry_skip": entry_regime.skip_entries,
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
    last_exit = _prune_last_exit_for_open_positions(
        _normalize_last_exit_state(state.get("last_exit", {})),
        positions,
    )
    recent_trades = list(state.get("recent_trades", []))
    now = datetime.now(timezone.utc)

    all_bars = _fetch_all_bars(client, symbols, config.lookback_days)
    equity = get_account_equity(client) if not dry_run else config.initial_cash
    current_bars = {sym: bars.iloc[-1] for sym, bars in all_bars.items()}

    slots = config.max_positions - len(positions) - len(pending_entries)
    if slots <= 0:
        logger.info(f"ENTRY SCAN: skipped, {len(positions)} positions + {len(pending_entries)} pending >= {config.max_positions} max")
        return

    if not all_bars:
        logger.warning(f"ENTRY SCAN: skipped, no market data fetched for any of {len(symbols)} symbols")
        log_event({
            **_build_market_data_gap_event(
                event_type="entry_scan",
                symbols=symbols,
                positions=positions,
                pending_entries=pending_entries,
                equity=equity,
            ),
            "n_checked": 0,
            "n_candidates": 0,
            "n_staged": 0,
            "n_proximity_skip": 0,
            "n_gemini_skip": 0,
            "n_already_held": 0,
            "n_neural_skip": 0,
            "n_order_fail": 0,
            "slots_available": slots,
            "risk_off": False,
            "risk_off_triggered": False,
            "market_breadth_skip": False,
            "skip_reason": "no_market_data",
        })
        return

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
        state["last_exit"] = last_exit
        save_state(state)

    symbol_metrics = _build_symbol_metric_cache(current_bars, all_bars)
    entry_regime = resolve_entry_regime(
        current_bars=current_bars,
        history=all_bars,
        config=config,
        symbol_metrics=symbol_metrics,
    )
    entry_config = entry_regime.config

    if entry_regime.skip_entries:
        skip_reason = "risk_off" if entry_regime.risk_off and not entry_regime.market_breadth_skip else "market_breadth_risk_off"
        logger.info(f"ENTRY SCAN: skipped, {skip_reason}")
        log_event({
            "type": "entry_scan",
            "n_checked": len(all_bars),
            "n_candidates": 0,
            "n_staged": 0,
            "risk_off": entry_regime.skip_entries,
            "risk_off_triggered": entry_regime.risk_off,
            "market_breadth_skip": entry_regime.market_breadth_skip,
            "skip_reason": skip_reason,
        })
        return

    neural_predictions = None
    if neural_model is not None and neural_model_symbols:
        neural_predictions = _try_neural_inference(
            neural_model, all_bars, neural_model_symbols, neural_seq_len,
        )

    staged_symbols = set(positions) | set(pending_entries)
    dip_tiers = dip_pct_fallback if dip_pct_fallback else None
    last_exit_ts = _last_exit_timestamps(last_exit)

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
            symbol_metrics=symbol_metrics,
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
            symbol_metrics=symbol_metrics,
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
        "risk_off": entry_regime.skip_entries,
        "risk_off_triggered": entry_regime.risk_off,
        "market_breadth_skip": entry_regime.market_breadth_skip,
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
    symbol_metrics: Optional[dict] = None,
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
        symbol_metrics=symbol_metrics,
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
    last_exit = _prune_last_exit_for_open_positions(
        _normalize_last_exit_state(state.get("last_exit", {})),
        positions,
    )
    recent_trades = list(state.get("recent_trades", []))

    # Adaptive dip: track consecutive zero-candidate cycles
    zero_candidate_cycles = _safe_int(state.get("zero_candidate_cycles", 0), default=0)
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

    if len(positions) >= config.max_positions and pending_entries:
        for sym, entry in list(pending_entries.items()):
            if _cancel_pending_entry(client, sym, entry):
                del pending_entries[sym]

    if not all_bars:
        logger.warning(f"Daily cycle: no market data fetched for any of {len(symbols)} symbols; skipping exits and entries")
        log_event({
            **_build_market_data_gap_event(
                event_type="daily_cycle",
                symbols=symbols,
                positions=positions,
                pending_entries=pending_entries,
                equity=equity,
            ),
            "cycle_status": "no_market_data",
        })
        state["positions"] = positions
        state["pending_entries"] = pending_entries
        state["last_exit"] = last_exit
        state["recent_trades"] = recent_trades[-50:]
        state["peak_equity"] = _merge_peak_equity(state.get("peak_equity", 0), equity)
        if config.dip_pct != original_dip_pct:
            state["effective_dip_pct"] = config.dip_pct
        save_state(state)
        return

    if not dry_run:
        synchronize_positions_from_exchange(
            client=client,
            symbols=symbols,
            positions=positions,
            current_bars=current_bars,
            config=config,
            now=now,
        )

    symbol_metrics = _build_symbol_metric_cache(current_bars, all_bars)

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
        entry_date = _safe_utc_timestamp(
            pos.get("entry_date"),
            context=f"positions[{sym}].entry_date",
            default=_coerce_utc_timestamp(now),
        )
        if entry_date is None:
            entry_date = _coerce_utc_timestamp(now)
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

        raw_order = place_limit_sell(client, sym, exit_price, pos["quantity"])
        order, order_id = _submitted_margin_order_or_none(
            raw_order,
            context=f"margin order submit response for exit {sym}",
        )
        if order is None:
            continue
        if order_id is None:
            logger.warning(f"EXIT ORDER FAILED {sym}: no live order placed at {exit_price:.4f}")
            continue

        pos["exit_order_id"] = order_id
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
            "order_id": order_id,
        }
        log_trade(trade)
        recent_trades.append(trade)

    # Stage new entries
    entry_regime = resolve_entry_regime(
        current_bars=current_bars,
        history=all_bars,
        config=config,
        symbol_metrics=symbol_metrics,
    )
    entry_config = entry_regime.config
    skip_entries = entry_regime.skip_entries
    risk_off = entry_regime.risk_off
    counts = {"n_staged": 0, "n_proximity_skip": 0, "n_gemini_skip": 0, "n_already_held": 0, "n_neural_skip": 0}
    n_candidates = 0

    # Log market breadth details
    breadth_ratio, n_breadth_dipping, n_breadth_total = _entry_regime_breadth_snapshot(
        current_bars,
        all_bars,
        entry_regime,
    )
    logger.info(
        f"MARKET STATE: breadth={n_breadth_dipping}/{n_breadth_total} dipping ({breadth_ratio:.1%}) "
        f"threshold={entry_config.market_breadth_filter:.0%} breadth_skip={entry_regime.market_breadth_skip} "
        f"risk_off={risk_off} entry_skip={skip_entries}"
    )

    # Log SMA pass/fail
    sma_pass, sma_fail = _count_sma_pass_fail(all_bars, config, symbol_metrics=symbol_metrics)
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
        logger.info(
            "ENTRY SCAN: skipped, "
            + ("risk_off" if risk_off and not entry_regime.market_breadth_skip else "market_breadth_risk_off")
        )
    else:
        staged_symbols = set(positions) | set(pending_entries)
        diagnostics: List[SymbolDiagnostic] = []
        last_exit_ts = _last_exit_timestamps(last_exit)

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
                symbol_metrics=symbol_metrics,
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
                symbol_metrics=symbol_metrics,
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
        "risk_off_triggered": risk_off,
        "market_breadth_skip": entry_regime.market_breadth_skip,
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
    state["peak_equity"] = _merge_peak_equity(state.get("peak_equity", 0), equity)
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
    entry_regime = resolve_entry_regime(current_bars=current_bars, history=all_bars, config=config)
    breadth_ratio, n_dipping, n_total = _entry_regime_breadth_snapshot(
        current_bars,
        all_bars,
        entry_regime,
    )
    entry_config = entry_regime.config
    breadth_skip = entry_regime.market_breadth_skip
    risk_off = entry_regime.risk_off

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
    last_exit = _prune_last_exit_for_open_positions(
        _normalize_last_exit_state(state.get("last_exit", {})),
        positions,
    )

    candidates = build_entry_candidates(
        date=pd.Timestamp(now),
        current_bars=current_bars,
        history=all_bars,
        positions={},
        last_exit=_last_exit_timestamps(last_exit),
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


def main(argv: list[str] | None = None):
    parser = build_arg_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)

    if args.live:
        args.dry_run = False

    resolved_error_context: dict[str, object] = {}

    def return_main_error(
        *,
        error: str,
        error_type: str | None = None,
        logger_message: str | None = None,
    ) -> int:
        extra: dict[str, object] = {
            "max_symbols": args.max_symbols,
            "dry_run": args.dry_run,
            "live_mode": not args.dry_run,
        }
        extra.update(resolved_error_context)
        if args.list_symbols:
            extra["list_symbols_only"] = True
        if args.preview_run:
            extra["preview_only"] = True

        if args.summary_json:
            def error_run():
                print(error)
                return 1, build_cli_error_summary(
                    tool="trade_live",
                    error=error,
                    error_type=error_type,
                    config_file=args.config_file,
                    extra=extra,
                )

            return run_with_optional_summary(
                args.summary_json,
                error_run,
                module="binance_worksteal.trade_live",
                argv=raw_argv,
            )
        if logger_message is not None:
            logger.error(logger_message)
        else:
            print(error)
        return 1

    config_output_rc = maybe_handle_worksteal_config_output(
        args=args,
        build_config=lambda: build_runtime_config(args, raw_argv),
        base_config=build_runtime_cli_default_config(args),
        config_file=args.config_file,
        raw_argv=raw_argv,
        flag_to_field=LIVE_CONFIG_FLAG_TO_FIELD,
    )
    if config_output_rc is not None:
        return config_output_rc

    try:
        config = build_runtime_config(args, raw_argv)
    except (FileNotFoundError, OSError, ValueError) as exc:
        return return_main_error(
            error=f"ERROR: {exc}",
            error_type=exc.__class__.__name__,
            logger_message=f"Invalid config: {exc}",
        )

    config_override_context = _build_config_override_context(args, raw_argv)
    if int(config_override_context.get("config_override_count", 0)) > 0:
        resolved_error_context.update(config_override_context)

    from binance_worksteal.backtest import FULL_UNIVERSE
    resolved, symbol_error = resolve_cli_symbols_with_error(
        symbols_arg=args.symbols,
        universe_file=args.universe_file,
        default_symbols=FULL_UNIVERSE,
    )
    if symbol_error is not None:
        return return_main_error(
            error=symbol_error["error"],
            error_type=symbol_error["error_type"],
        )
    symbols, symbol_source = resolved
    raw_symbol_count = len(symbols)
    resolved_symbols = _resolve_symbol_selection_context(
        symbols,
        symbol_source,
        max_symbols=args.max_symbols,
    )
    if resolved_symbols["was_capped"]:
        logger.info(
            f"Truncated symbol universe from {raw_symbol_count} to {args.max_symbols} via --max-symbols"
        )
    symbols = resolved_symbols["symbols"]
    display_source = str(resolved_symbols["symbol_source"])
    was_capped = bool(resolved_symbols["was_capped"])
    omitted_symbol_count = int(resolved_symbols["omitted_symbol_count"])
    omitted_symbols = list(resolved_symbols["omitted_symbols"])
    pair_routing = resolved_symbols["pair_routing"]
    pair_routing_summary = resolved_symbols["pair_routing_summary"]
    resolved_error_context.update(resolved_symbols)
    symbol_summary_context = {
        "max_symbols": args.max_symbols,
        "was_capped": was_capped,
        "omitted_symbol_count": omitted_symbol_count,
        "omitted_symbols": omitted_symbols,
        "dry_run": args.dry_run,
        "live_mode": not args.dry_run,
        "pair_routing": pair_routing,
        "pair_routing_summary": pair_routing_summary,
    }
    if args.list_symbols:

        def list_symbols_run():
            print_resolved_symbols(symbols, display_source)
            _print_omitted_symbols(omitted_symbols)
            _print_pair_routing(pair_routing)
            return 0, build_symbol_listing_summary(
                tool="trade_live",
                data_dir=None,
                symbol_source=display_source,
                symbols=symbols,
                config_file=args.config_file,
                requested_symbol_count=int(resolved_symbols["requested_symbol_count"]),
                extra=symbol_summary_context,
            )

        return run_with_optional_summary(
            args.summary_json,
            list_symbols_run,
            module="binance_worksteal.trade_live",
            argv=raw_argv,
            announce_artifact_manifest_on_success=bool(args.summary_json),
        )
    if args.preview_run:
        preview_neural_symbols = _normalize_strategy_symbols(args.neural_symbols)
        preview_override_context = _build_config_override_context(args, raw_argv)
        preview_mode_context = {
            "dry_run": args.dry_run,
            "live_mode": not args.dry_run,
            "daemon": args.daemon,
            "diagnose": args.diagnose,
            "run_on_start": args.run_on_start,
            "startup_preview_only": args.startup_preview_only,
        }
        preview_runtime_context = {
            "poll_seconds": args.poll_seconds,
            "entry_poll_hours": args.entry_poll_hours,
            "health_report_hours": args.health_report_hours,
            "gemini_enabled": args.gemini,
            "gemini_model": args.gemini_model if args.gemini else None,
            "neural_model": args.neural_model,
            "neural_symbols": preview_neural_symbols,
        }
        preview_file_context = {
            "state_file": str(STATE_FILE),
            "trade_log": str(LOG_FILE),
            "events_log": str(EVENTS_FILE),
        }
        preview_sections = [
            (
                "Inputs",
                (
                    ("symbol_source", display_source),
                    ("requested_symbol_count", int(resolved_symbols["requested_symbol_count"])),
                    ("symbol_count", len(symbols)),
                    ("omitted_symbol_count", omitted_symbol_count),
                    ("omitted_symbols", omitted_symbols),
                    ("symbols", symbols),
                    ("max_symbols", args.max_symbols),
                ),
            ),
            ("Mode", tuple(preview_mode_context.items())),
            ("Runtime", tuple(preview_runtime_context.items())),
            (
                "Files",
                tuple(
                    {
                        "config_file": args.config_file,
                        "summary_json": args.summary_json,
                        **preview_file_context,
                    }.items()
                ),
            ),
            (
                "Config",
                (
                    ("dip_pct", config.dip_pct),
                    ("proximity_pct", config.proximity_pct),
                    ("profit_target_pct", config.profit_target_pct),
                    ("stop_loss_pct", config.stop_loss_pct),
                    ("max_positions", config.max_positions),
                    ("market_breadth_filter", config.market_breadth_filter),
                    ("risk_off_market_breadth_filter", config.risk_off_market_breadth_filter),
                    ("rebalance_seeded_positions", config.rebalance_seeded_positions),
                ),
            ),
        ]
        if preview_override_context["config_override_count"]:
            preview_sections.append(
                (
                    "Config overrides",
                    (
                        ("config_override_count", preview_override_context["config_override_count"]),
                        ("config_file_override_fields", preview_override_context["config_file_override_fields"]),
                        ("cli_override_fields", preview_override_context["cli_override_fields"]),
                        ("effective_overrides", preview_override_context["config_override_preview"]),
                    ),
                )
            )

        def preview_run():
            print_run_preview(
                tool="trade_live",
                sections=preview_sections,
            )
            _print_pair_routing(pair_routing)
            if not args.summary_json:
                return 0, None
            return 0, build_preview_run_summary(
                tool="trade_live",
                data_dir=None,
                symbol_source=display_source,
                symbols=symbols,
                config_file=args.config_file,
                requested_symbol_count=int(resolved_symbols["requested_symbol_count"]),
                config=asdict(config),
                extra={
                    **symbol_summary_context,
                    **preview_mode_context,
                    **preview_runtime_context,
                    **preview_override_context,
                    **preview_file_context,
                },
            )

        return run_with_optional_summary(
            args.summary_json,
            preview_run,
            module="binance_worksteal.trade_live",
            argv=raw_argv,
            announce_artifact_manifest_on_success=bool(args.summary_json),
        )
    logger.info(f"Using {len(symbols)} symbols from {symbol_source}")
    if args.config_file:
        logger.info(f"Loaded config overrides from {args.config_file}")

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
        return return_main_error(
            error="ERROR: Binance client required for live mode but unavailable",
            error_type="RuntimeError",
            logger_message="Binance client required for live mode but unavailable",
        )
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
                nn_symbols = _normalize_strategy_symbols(args.neural_symbols)
            else:
                nn_symbols = _normalize_strategy_symbols(nn_symbols_loaded)
            logger.info(f"Neural model loaded: {args.neural_model} ({len(nn_symbols)} symbols, seq_len={nn_seq_len})")
        except Exception as e:
            return return_main_error(
                error=f"ERROR: Failed to load neural model: {e}",
                error_type=e.__class__.__name__,
            )

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
                last_exit = _prune_last_exit_for_open_positions(
                    _normalize_last_exit_state(state.get("last_exit", {})),
                    positions,
                )
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
                        last_exit=last_exit,
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
                    state["last_exit"] = last_exit
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
