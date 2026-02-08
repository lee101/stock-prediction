from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from src.price_guard import enforce_gap
from src.process_utils import enforce_min_spread
from src.torch_load_utils import torch_load_compat
from src.binan import binance_wrapper
from src.fees import get_fee_for_symbol

from binanceneural.binance_watchers import WatcherPlan, spawn_watcher
from binanceneural.config import TrainingConfig
from binanceneural.execution import (
    compute_order_quantities,
    get_free_balances,
    resolve_symbol_rules,
    split_binance_symbol,
)
from binanceneural.inference import generate_latest_action
from binanceneural.model import build_policy, policy_config_from_payload
from binanceneural.trade_binance_hourly import _ensure_valid_levels, _parse_checkpoint_map

from .config import DatasetConfig
from .data import BinanceExp1DataModule, build_default_feature_columns
from .trade_binance_hourly import (
    _load_checkpoint_payload,
    _infer_input_dim,
    _resolve_dataset_config,
    _load_model_from_payload,
    _refresh_price_csv,
    _build_plan,
    _log_account_metrics,
    TradingPlan,
)

_USDT_FALLBACK = {"SOLUSD": "SOLUSDT", "BTCUSD": "BTCUSDT", "ETHUSD": "ETHUSDT", "LINKUSD": "LINKUSDT"}

STATE_FILE = Path("strategy_state/selector_state.json")


@dataclass
class SelectorState:
    open_symbol: Optional[str] = None
    open_ts: Optional[str] = None
    open_price: float = 0.0

    def hours_held(self) -> float:
        if not self.open_ts:
            return 0.0
        opened = datetime.fromisoformat(self.open_ts)
        if opened.tzinfo is None:
            opened = opened.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - opened).total_seconds() / 3600.0

    def save(self, path: Path = STATE_FILE) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "open_symbol": self.open_symbol,
            "open_ts": self.open_ts,
            "open_price": self.open_price,
        }))

    @classmethod
    def load(cls, path: Path = STATE_FILE) -> SelectorState:
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text())
            return cls(
                open_symbol=data.get("open_symbol"),
                open_ts=data.get("open_ts"),
                open_price=data.get("open_price", 0.0),
            )
        except Exception:
            return cls()


def _detect_holdings(symbols: List[str], min_notional: float = 5.0) -> List[Tuple[str, float, float]]:
    """Check Binance balances for all symbols. Returns [(symbol, qty, notional)]."""
    balances = binance_wrapper.get_account_balances()
    holdings = []
    for symbol in symbols:
        base, _ = split_binance_symbol(symbol)
        entry = binance_wrapper.get_asset_balance(base, balances=balances) or {}
        free = float(entry.get("free", 0))
        locked = float(entry.get("locked", 0))
        total = free + locked
        if total > 0:
            try:
                ticker = {"price": binance_wrapper.get_symbol_price(symbol)}
                price = float(ticker.get("price", 0))
                notional = total * price
                if notional >= min_notional:
                    holdings.append((symbol, total, notional))
            except Exception:
                if total > 1e-6:
                    holdings.append((symbol, total, 0.0))
    return holdings


def _compute_edge(
    action: dict,
    *,
    horizon: int,
    fee_rate: float,
    risk_weight: float,
) -> float:
    buy_price = float(action.get("buy_price", 0))
    if buy_price <= 0:
        return -999.0
    pred_high = float(action.get(f"predicted_high_p50_h{horizon}", 0))
    pred_low = float(action.get(f"predicted_low_p50_h{horizon}", 0))
    if pred_high <= 0 or pred_low <= 0:
        return -999.0
    buy_amount = float(action.get("buy_amount", 0))
    buy_intensity = max(0.0, min(1.0, buy_amount / 100.0))
    if buy_intensity <= 0:
        return -999.0
    upside = (pred_high - buy_price) / buy_price
    downside = max(0.0, (buy_price - pred_low) / buy_price)
    edge = upside - risk_weight * downside - 2.0 * fee_rate
    return edge * buy_intensity


def _generate_action_for_symbol(
    symbol: str,
    checkpoint_path: Path,
    *,
    horizon: int,
    sequence_length: int,
    data_root: Path,
    cache_only: bool,
    intensity_scale: float,
    price_offset_pct: float,
) -> Optional[dict]:
    try:
        payload = _load_checkpoint_payload(checkpoint_path)
        state_dict = payload.get("state_dict", payload)
        base_cfg = DatasetConfig(
            symbol=symbol,
            data_root=data_root,
            sequence_length=sequence_length,
            cache_only=cache_only,
        )
        fallback_dim = len(build_default_feature_columns(base_cfg))
        input_dim = _infer_input_dim(state_dict, fallback=fallback_dim)
        data_cfg = _resolve_dataset_config(base_cfg, input_dim=input_dim, horizon=horizon)
        data = BinanceExp1DataModule(data_cfg)
        model = _load_model_from_payload(payload, input_dim, TrainingConfig(sequence_length=sequence_length))
        action = generate_latest_action(
            model=model,
            frame=data.frame,
            feature_columns=data.feature_columns,
            normalizer=data.normalizer,
            sequence_length=sequence_length,
            horizon=horizon,
            require_gpu=True,
        )
        action["symbol"] = symbol
        last_row = data.frame.iloc[-1]
        for col in [f"predicted_high_p50_h{horizon}", f"predicted_low_p50_h{horizon}", f"predicted_close_p50_h{horizon}"]:
            if col in data.frame.columns:
                action[col] = float(last_row[col])
        return action
    except Exception as exc:
        print(f"Error generating action for {symbol}: {exc}")
        return None


def _run_selector_cycle(
    symbols: List[str],
    checkpoint_map: Dict[str, Path],
    *,
    horizon: int,
    sequence_length: int,
    intensity_scale: float,
    price_offset_map: Dict[str, float],
    default_offset: float,
    min_gap_pct: float,
    risk_weight: float,
    min_edge: float,
    max_hold_hours: Optional[int],
    poll_seconds: int,
    expiry_minutes: int,
    price_tolerance: float,
    data_root: Path,
    cache_only: bool,
    dry_run: bool,
    state_path: Path,
) -> None:
    for symbol in symbols:
        _refresh_price_csv(symbol, data_root)

    state = SelectorState.load(state_path)
    holdings = _detect_holdings(symbols)

    if len(holdings) > 1:
        holdings.sort(key=lambda x: x[2], reverse=True)
        primary = holdings[0]
        extras = holdings[1:]
        print(f"[selector] WARNING: holding {len(holdings)} assets, selling extras to consolidate")
        for sym, qty, notional in extras:
            print(f"[selector] selling extra {sym}: qty={qty:.6f} notional=${notional:.2f}")
            try:
                rules = resolve_symbol_rules(sym)
                _, base_free = get_free_balances(sym)
                ticker = {"price": binance_wrapper.get_symbol_price(sym)}
                market_price = float(ticker.get("price", 0))
                sell_price = market_price * 0.999
                validated = _ensure_valid_levels(sym, sell_price * 0.99, sell_price, min_gap_pct=min_gap_pct, rules=rules)
                if validated:
                    _, sell_price = validated
                    sizing = compute_order_quantities(
                        symbol=sym, buy_amount=0, sell_amount=100.0,
                        buy_price=sell_price, sell_price=sell_price,
                        quote_free=0, base_free=base_free, rules=rules,
                    )
                    if sizing.sell_qty > 0:
                        spawn_watcher(WatcherPlan(
                            symbol=sym, side="sell", mode="exit",
                            limit_price=sell_price, target_qty=sizing.sell_qty,
                            expiry_minutes=expiry_minutes, poll_seconds=poll_seconds,
                            price_tolerance=price_tolerance * 3, dry_run=dry_run,
                        ))
            except Exception as exc:
                print(f"[selector] failed to sell extra {sym}: {exc}")
        held_symbol, held_qty, _ = primary
        state.open_symbol = held_symbol
        if not state.open_ts:
            state.open_ts = datetime.now(timezone.utc).isoformat()
        state.save(state_path)
    elif len(holdings) == 1:
        held_symbol, held_qty, _ = holdings[0]
        if state.open_symbol != held_symbol:
            print(f"[selector] detected holding {held_symbol} ({held_qty:.6f}), syncing state")
            state.open_symbol = held_symbol
            if not state.open_ts:
                state.open_ts = datetime.now(timezone.utc).isoformat()
            state.save(state_path)
    elif state.open_symbol:
        print(f"[selector] state says {state.open_symbol} but no balance detected, clearing")
        state = SelectorState()
        state.save(state_path)

    actions: Dict[str, dict] = {}
    for symbol in symbols:
        ckpt = checkpoint_map.get(symbol)
        if not ckpt:
            continue
        offset = price_offset_map.get(symbol, default_offset)
        action = _generate_action_for_symbol(
            symbol, ckpt,
            horizon=horizon,
            sequence_length=sequence_length,
            data_root=data_root,
            cache_only=cache_only,
            intensity_scale=intensity_scale,
            price_offset_pct=offset,
        )
        if action:
            actions[symbol] = action

    if not actions:
        print("[selector] no actions generated")
        return

    if state.open_symbol:
        _handle_exit(
            state, actions, symbols,
            intensity_scale=intensity_scale,
            price_offset_map=price_offset_map,
            default_offset=default_offset,
            min_gap_pct=min_gap_pct,
            max_hold_hours=max_hold_hours,
            poll_seconds=poll_seconds,
            expiry_minutes=expiry_minutes,
            price_tolerance=price_tolerance,
            dry_run=dry_run,
            state_path=state_path,
        )
    else:
        _handle_entry(
            state, actions, symbols,
            horizon=horizon,
            intensity_scale=intensity_scale,
            price_offset_map=price_offset_map,
            default_offset=default_offset,
            min_gap_pct=min_gap_pct,
            risk_weight=risk_weight,
            min_edge=min_edge,
            poll_seconds=poll_seconds,
            expiry_minutes=expiry_minutes,
            price_tolerance=price_tolerance,
            dry_run=dry_run,
            state_path=state_path,
        )


def _handle_exit(
    state: SelectorState,
    actions: Dict[str, dict],
    symbols: List[str],
    *,
    intensity_scale: float,
    price_offset_map: Dict[str, float],
    default_offset: float,
    min_gap_pct: float,
    max_hold_hours: Optional[int],
    poll_seconds: int,
    expiry_minutes: int,
    price_tolerance: float,
    dry_run: bool,
    state_path: Path,
) -> None:
    symbol = state.open_symbol
    hours = state.hours_held()
    force_close = max_hold_hours is not None and hours >= max_hold_hours

    if force_close:
        print(f"[selector] FORCE CLOSE {symbol} after {hours:.1f}h (max={max_hold_hours}h)")
        try:
            _, base_free = get_free_balances(symbol)
        except Exception as exc:
            print(f"[selector] failed to get balances for {symbol}: {exc}")
            return
        if base_free <= 0:
            state.open_symbol = None
            state.open_ts = None
            state.save(state_path)
            return
        try:
            ticker = {"price": binance_wrapper.get_symbol_price(symbol)}
            market_price = float(ticker.get("price", 0))
        except Exception:
            market_price = 0
        if market_price <= 0:
            print(f"[selector] can't get market price for {symbol}")
            return
        rules = resolve_symbol_rules(symbol)
        sell_price = market_price * 0.999
        sell_price, _ = enforce_min_spread(sell_price, market_price * 1.001, min_spread_pct=min_gap_pct)
        validated = _ensure_valid_levels(symbol, sell_price * 0.99, sell_price, min_gap_pct=min_gap_pct, rules=rules)
        if validated is None:
            print(f"[selector] invalid force-close price for {symbol}")
            return
        _, sell_price = validated
        sizing = compute_order_quantities(
            symbol=symbol,
            buy_amount=0,
            sell_amount=100.0,
            buy_price=sell_price,
            sell_price=sell_price,
            quote_free=0,
            base_free=base_free,
            rules=rules,
        )
        if sizing.sell_qty > 0:
            spawn_watcher(WatcherPlan(
                symbol=symbol, side="sell", mode="exit",
                limit_price=sell_price, target_qty=sizing.sell_qty,
                expiry_minutes=expiry_minutes, poll_seconds=poll_seconds,
                price_tolerance=price_tolerance * 3, dry_run=dry_run,
            ))
            print(f"[selector] force-close {symbol} sell={sell_price:.4f} qty={sizing.sell_qty:.6f}")
        return

    action = actions.get(symbol)
    if not action:
        print(f"[selector] holding {symbol} ({hours:.1f}h) but no action, waiting")
        return

    plan = _build_plan(action, intensity_scale=intensity_scale)
    offset = price_offset_map.get(symbol, default_offset)
    sell_price = plan.sell_price * (1.0 + offset)

    try:
        _, base_free = get_free_balances(symbol)
    except Exception as exc:
        print(f"[selector] failed to get balances for {symbol}: {exc}")
        return

    if base_free <= 0:
        print(f"[selector] no {symbol} balance, clearing state")
        state.open_symbol = None
        state.open_ts = None
        state.save(state_path)
        return

    rules = resolve_symbol_rules(symbol)
    buy_price = plan.buy_price * (1.0 - offset)
    buy_price, sell_price = enforce_min_spread(buy_price, sell_price, min_spread_pct=min_gap_pct)
    buy_price, sell_price = enforce_gap(symbol, buy_price, sell_price, min_gap_pct=min_gap_pct)
    validated = _ensure_valid_levels(symbol, buy_price, sell_price, min_gap_pct=min_gap_pct, rules=rules)
    if validated is None:
        print(f"[selector] invalid exit prices for {symbol}")
        return
    buy_price, sell_price = validated
    sizing = compute_order_quantities(
        symbol=symbol, buy_amount=0, sell_amount=plan.sell_amount,
        buy_price=buy_price, sell_price=sell_price,
        quote_free=0, base_free=base_free, rules=rules,
    )
    if sizing.sell_qty > 0:
        spawn_watcher(WatcherPlan(
            symbol=symbol, side="sell", mode="exit",
            limit_price=sell_price, target_qty=sizing.sell_qty,
            expiry_minutes=expiry_minutes, poll_seconds=poll_seconds,
            price_tolerance=price_tolerance, dry_run=dry_run,
        ))
    print(
        f"[selector] hold {symbol} ({hours:.1f}h) sell={sell_price:.4f}({sizing.sell_qty:.6f}) "
        f"amt={plan.sell_amount:.2f}"
    )


def _handle_entry(
    state: SelectorState,
    actions: Dict[str, dict],
    symbols: List[str],
    *,
    horizon: int,
    intensity_scale: float,
    price_offset_map: Dict[str, float],
    default_offset: float,
    min_gap_pct: float,
    risk_weight: float,
    min_edge: float,
    poll_seconds: int,
    expiry_minutes: int,
    price_tolerance: float,
    dry_run: bool,
    state_path: Path,
) -> None:
    candidates: List[Tuple[float, str]] = []
    for symbol, action in actions.items():
        fee_rate = get_fee_for_symbol(symbol)
        edge = _compute_edge(action, horizon=horizon, fee_rate=fee_rate, risk_weight=risk_weight)
        if edge >= min_edge:
            candidates.append((edge, symbol))
            print(f"[selector] {symbol} edge={edge:.6f}")
        else:
            print(f"[selector] {symbol} edge={edge:.6f} (below min_edge={min_edge})")

    if not candidates:
        print("[selector] no candidates above min_edge, staying flat")
        return

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_edge, best_symbol = candidates[0]
    action = actions[best_symbol]
    plan = _build_plan(action, intensity_scale=intensity_scale)
    offset = price_offset_map.get(best_symbol, default_offset)
    buy_price = plan.buy_price * (1.0 - offset)
    sell_price = plan.sell_price * (1.0 + offset)
    buy_price, sell_price = enforce_min_spread(buy_price, sell_price, min_spread_pct=min_gap_pct)
    buy_price, sell_price = enforce_gap(best_symbol, buy_price, sell_price, min_gap_pct=min_gap_pct)

    if buy_price <= 0 or sell_price <= 0 or buy_price >= sell_price:
        print(f"[selector] invalid prices for {best_symbol}: buy={buy_price:.4f} sell={sell_price:.4f}")
        return

    try:
        quote_free, base_free = get_free_balances(best_symbol)
    except Exception as exc:
        print(f"[selector] failed to get balances for {best_symbol}: {exc}")
        return

    rules = resolve_symbol_rules(best_symbol)
    validated = _ensure_valid_levels(best_symbol, buy_price, sell_price, min_gap_pct=min_gap_pct, rules=rules)
    if validated is None:
        print(f"[selector] invalid quantized prices for {best_symbol}")
        return
    buy_price, sell_price = validated
    sizing = compute_order_quantities(
        symbol=best_symbol, buy_amount=plan.buy_amount, sell_amount=0,
        buy_price=buy_price, sell_price=sell_price,
        quote_free=quote_free, base_free=0, rules=rules,
    )
    if sizing.buy_qty > 0:
        spawn_watcher(WatcherPlan(
            symbol=best_symbol, side="buy", mode="entry",
            limit_price=buy_price, target_qty=sizing.buy_qty,
            expiry_minutes=expiry_minutes, poll_seconds=poll_seconds,
            price_tolerance=price_tolerance, dry_run=dry_run,
        ))
        state.open_symbol = best_symbol
        state.open_ts = datetime.now(timezone.utc).isoformat()
        state.open_price = buy_price
        state.save(state_path)
    print(
        f"[selector] ENTER {best_symbol} edge={best_edge:.6f} buy={buy_price:.4f}({sizing.buy_qty:.6f}) "
        f"amt={plan.buy_amount:.2f} quote_free={quote_free:.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-asset selector trading bot.")
    parser.add_argument("--symbols", default="BTCUSD,ETHUSD,SOLUSD")
    parser.add_argument("--checkpoints", required=True, help="SYMBOL=PATH checkpoint mapping")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=96)
    parser.add_argument("--intensity-scale", type=float, default=5.0)
    parser.add_argument("--default-offset", type=float, default=0.0)
    parser.add_argument("--offset-map", default=None, help="SYMBOL=VALUE overrides, e.g. ETHUSD=0.0003,SOLUSD=0.0005")
    parser.add_argument("--min-gap-pct", type=float, default=0.0003)
    parser.add_argument("--risk-weight", type=float, default=0.0)
    parser.add_argument("--min-edge", type=float, default=0.0)
    parser.add_argument("--max-hold-hours", type=int, default=6)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--expiry-minutes", type=int, default=90)
    parser.add_argument("--price-tolerance", type=float, default=0.0008)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--data-root", default=str(DatasetConfig().data_root))
    parser.add_argument("--cycle-minutes", type=int, default=5)
    parser.add_argument("--log-metrics", action="store_true")
    parser.add_argument("--metrics-log-path", default=None)
    parser.add_argument("--state-path", default=str(STATE_FILE))
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    checkpoint_map: Dict[str, Path] = {}
    for token in args.checkpoints.split(","):
        token = token.strip()
        if "=" in token:
            k, v = token.split("=", 1)
            checkpoint_map[k.strip().upper()] = Path(v.strip()).expanduser().resolve()

    offset_map: Dict[str, float] = {}
    if args.offset_map:
        for token in args.offset_map.split(","):
            token = token.strip()
            if "=" in token:
                k, v = token.split("=", 1)
                offset_map[k.strip().upper()] = float(v.strip())

    state_path = Path(args.state_path)
    metrics_log_path = Path(args.metrics_log_path) if args.metrics_log_path else None

    while True:
        try:
            _run_selector_cycle(
                symbols, checkpoint_map,
                horizon=args.horizon,
                sequence_length=args.sequence_length,
                intensity_scale=args.intensity_scale,
                price_offset_map=offset_map,
                default_offset=args.default_offset,
                min_gap_pct=args.min_gap_pct,
                risk_weight=args.risk_weight,
                min_edge=args.min_edge,
                max_hold_hours=args.max_hold_hours,
                poll_seconds=args.poll_seconds,
                expiry_minutes=args.expiry_minutes,
                price_tolerance=args.price_tolerance,
                data_root=Path(args.data_root),
                cache_only=args.cache_only,
                dry_run=args.dry_run,
                state_path=state_path,
            )
        except Exception as exc:
            print(f"[selector] cycle error: {exc}")

        if args.log_metrics and not args.dry_run and metrics_log_path:
            try:
                _log_account_metrics(symbols, log_path=metrics_log_path)
            except Exception as exc:
                print(f"[selector] metrics error: {exc}")

        if args.once:
            break
        sleep_seconds = args.cycle_minutes * 60
        print(f"[selector] sleeping {sleep_seconds}s...")
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
