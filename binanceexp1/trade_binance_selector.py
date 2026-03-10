from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.price_guard import enforce_gap
from src.process_utils import enforce_min_spread
from src.binan import binance_wrapper
from src.binan.binance_conversion import (
    build_stable_quote_conversion_plan,
    compute_spendable_quote,
    execute_stable_quote_conversion,
)
from src.binan.binance_margin import (
    get_margin_account,
    get_margin_asset_balance,
    get_margin_free_balance,
    get_margin_net_balance,
    get_max_borrowable,
)
from src.fees import get_fee_for_symbol
from src.margin_position_utils import choose_flat_entry_side, directional_signal, position_side_from_qty

from binanceneural.binance_watchers import WatcherPlan, spawn_watcher, cancel_entry_watchers
from binanceneural.execution import (
    compute_order_quantities,
    get_free_balances,
    get_total_balances,
    quantize_price,
    quantize_qty,
    resolve_symbol_rules,
    split_binance_symbol,
)
from binanceneural.inference import generate_latest_action
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
_SUPPORTED_CONVERSION_QUOTES = ("FDUSD", "USDT")

STATE_FILE = Path("strategy_state/selector_state.json")


@dataclass
class SelectorState:
    open_symbol: Optional[str] = None
    open_binance_symbol: Optional[str] = None
    open_ts: Optional[str] = None
    open_price: float = 0.0
    position_side: str = ""
    active_quote_asset: str = ""
    last_quote_conversion_ts: Optional[str] = None
    last_quote_conversion_from_asset: str = ""
    last_quote_conversion_to_asset: str = ""

    def hours_held(self) -> float:
        if not self.open_ts:
            return 0.0
        opened = _parse_iso_timestamp(self.open_ts)
        if opened is None:
            return 0.0
        return max(0.0, (datetime.now(timezone.utc) - opened).total_seconds() / 3600.0)

    def save(self, path: Path = STATE_FILE) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "open_symbol": self.open_symbol,
            "open_binance_symbol": self.open_binance_symbol,
            "open_ts": self.open_ts,
            "open_price": self.open_price,
            "position_side": self.position_side,
            "active_quote_asset": self.active_quote_asset,
            "last_quote_conversion_ts": self.last_quote_conversion_ts,
            "last_quote_conversion_from_asset": self.last_quote_conversion_from_asset,
            "last_quote_conversion_to_asset": self.last_quote_conversion_to_asset,
        }))

    @classmethod
    def load(cls, path: Path = STATE_FILE) -> SelectorState:
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text())
            return cls(
                open_symbol=data.get("open_symbol"),
                open_binance_symbol=data.get("open_binance_symbol"),
                open_ts=data.get("open_ts"),
                open_price=data.get("open_price", 0.0),
                position_side=str(data.get("position_side", "") or ""),
                active_quote_asset=str(data.get("active_quote_asset", "") or ""),
                last_quote_conversion_ts=data.get("last_quote_conversion_ts"),
                last_quote_conversion_from_asset=str(data.get("last_quote_conversion_from_asset", "") or ""),
                last_quote_conversion_to_asset=str(data.get("last_quote_conversion_to_asset", "") or ""),
            )
        except Exception:
            return cls()


@dataclass(frozen=True)
class DetectedHolding:
    symbol: str
    quantity: float
    notional: float
    position_side: str
    margin: bool = False


def _parse_iso_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed


def _normalize_position_side(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"long", "short"}:
        return normalized
    return ""


def _resolve_directional_leverages(
    *,
    max_account_leverage: float,
    max_long_leverage: Optional[float],
    max_short_leverage: Optional[float],
) -> tuple[float, float]:
    cap = max(1.0, float(max_account_leverage))
    long_lev = 1.0 if max_long_leverage is None else float(max_long_leverage)
    short_lev = 1.0 if max_short_leverage is None else float(max_short_leverage)
    if not np.isfinite(long_lev) or long_lev <= 0.0:
        raise ValueError(f"max_long_leverage must be > 0, got {max_long_leverage}.")
    if not np.isfinite(short_lev) or short_lev <= 0.0:
        raise ValueError(f"max_short_leverage must be > 0, got {max_short_leverage}.")
    if long_lev > cap or short_lev > cap:
        raise ValueError(
            f"Directional leverage exceeds max_account_leverage={cap}: long={long_lev}, short={short_lev}."
        )
    return long_lev, short_lev


def _quantized_fractional_qty(
    *,
    side: str,
    price: float,
    intensity_amount: float,
    notional_cap: float,
    rules,
) -> tuple[float, float]:
    intensity = max(0.0, min(1.0, float(intensity_amount) / 100.0))
    limit_price = quantize_price(float(price), tick_size=rules.tick_size, side=side)
    if intensity <= 0.0 or limit_price <= 0.0 or notional_cap <= 0.0:
        return 0.0, limit_price
    qty = quantize_qty((float(notional_cap) * intensity) / limit_price, step_size=rules.step_size)
    if rules.min_qty is not None and qty < rules.min_qty:
        return 0.0, limit_price
    if rules.min_notional is not None and qty * limit_price < rules.min_notional:
        return 0.0, limit_price
    return float(qty), float(limit_price)


def _detect_margin_context(symbol: str) -> dict:
    base, quote = split_binance_symbol(symbol)
    market_price = float(binance_wrapper.get_symbol_price(symbol) or 0.0)
    base_entry = get_margin_asset_balance(base) or {}
    base_free = float(base_entry.get("free", 0.0) or 0.0)
    base_locked = float(base_entry.get("locked", 0.0) or 0.0)
    base_net = float(get_margin_net_balance(base))
    quote_free = float(get_margin_free_balance(quote))
    equity = quote_free + base_net * market_price
    try:
        margin_account = get_margin_account()
        total_net_btc = float((margin_account or {}).get("totalNetAssetOfBtc", 0.0))
        btc_price = float(binance_wrapper.get_symbol_price("BTCUSDT") or 0.0)
        if total_net_btc and btc_price > 0.0:
            equity = total_net_btc * btc_price
    except Exception:
        pass
    return {
        "symbol": symbol,
        "base": base,
        "quote": quote,
        "market_price": market_price,
        "asset_free": base_free,
        "asset_locked": base_locked,
        "asset_net": base_net,
        "quote_free": quote_free,
        "equity": float(equity),
        "position_side": position_side_from_qty(base_net),
        "position_value": abs(base_net) * market_price,
    }


def _quote_conversion_due(state: SelectorState, *, cooldown_minutes: int) -> bool:
    cooldown = max(0, int(cooldown_minutes))
    if cooldown <= 0:
        return True
    last = _parse_iso_timestamp(state.last_quote_conversion_ts)
    if last is None:
        return True
    return datetime.now(timezone.utc) - last >= timedelta(minutes=cooldown)


def _maybe_convert_quote_asset(
    state: SelectorState,
    *,
    target_quote_asset: str,
    leave_quote_buffer: float,
    max_quote_conversion: Optional[float],
    conversion_cooldown_minutes: int,
    dry_run: bool,
) -> None:
    target_quote = str(target_quote_asset or "").strip().upper()
    if target_quote not in _SUPPORTED_CONVERSION_QUOTES:
        return
    current_free = float(binance_wrapper.get_asset_free_balance(target_quote))
    if current_free > max(0.0, float(leave_quote_buffer)):
        state.active_quote_asset = target_quote
        state.save()
        return
    if not _quote_conversion_due(state, cooldown_minutes=conversion_cooldown_minutes):
        return

    client = binance_wrapper.get_client()
    if client is None:
        return

    balances = binance_wrapper.get_account_balances(client=client)
    candidates: list[tuple[float, str]] = []
    for asset in _SUPPORTED_CONVERSION_QUOTES:
        if asset == target_quote:
            continue
        free_balance = float(binance_wrapper.get_asset_free_balance(asset, client=client))
        spendable = compute_spendable_quote(
            free_quote=free_balance,
            leave_quote=leave_quote_buffer,
            max_spend=max_quote_conversion,
        )
        if spendable > 0.0:
            candidates.append((spendable, asset))
    if not candidates:
        return

    for spendable, source_asset in sorted(candidates, reverse=True):
        direct_pairs = []
        for pair in (f"{target_quote}{source_asset}", f"{source_asset}{target_quote}"):
            info = client.get_symbol_info(pair)
            if isinstance(info, dict):
                direct_pairs.append(pair)
        plan = build_stable_quote_conversion_plan(
            from_asset=source_asset,
            to_asset=target_quote,
            amount=spendable,
            available_pairs=direct_pairs,
        )
        if plan is None:
            continue
        try:
            response = execute_stable_quote_conversion(plan, dry_run=dry_run, client=client)
        except Exception as exc:
            print(f"[selector] quote conversion {source_asset}->{target_quote} failed: {exc}")
            continue
        now = datetime.now(timezone.utc).isoformat()
        state.active_quote_asset = target_quote
        state.last_quote_conversion_ts = now
        state.last_quote_conversion_from_asset = source_asset
        state.last_quote_conversion_to_asset = target_quote
        state.save()
        print(
            f"[selector] converted {source_asset}->{target_quote} amount={spendable:.2f} "
            f"via {plan.symbol} side={plan.side} response={response}"
        )
        time.sleep(1)
        return


def _detect_holdings(
    symbols: List[str],
    min_notional: float = 5.0,
    *,
    margin: bool = False,
) -> List[DetectedHolding]:
    """Check Binance balances for all symbols."""
    holdings: List[DetectedHolding] = []
    balances = None if margin else binance_wrapper.get_account_balances()
    for symbol in symbols:
        base, _ = split_binance_symbol(symbol)
        if margin:
            qty = float(get_margin_net_balance(base))
        else:
            entry = binance_wrapper.get_asset_balance(base, balances=balances) or {}
            qty = float(entry.get("free", 0.0) or 0.0) + float(entry.get("locked", 0.0) or 0.0)
        if abs(qty) <= 0.0:
            continue
        try:
            price = float(binance_wrapper.get_symbol_price(symbol) or 0.0)
            notional = abs(qty) * price
        except Exception:
            price = 0.0
            notional = 0.0
        if notional >= min_notional or abs(qty) > 1e-6:
            holdings.append(
                DetectedHolding(
                    symbol=symbol,
                    quantity=float(qty),
                    notional=float(notional),
                    position_side=position_side_from_qty(qty),
                    margin=bool(margin),
                )
            )
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


def _compute_short_edge(
    action: dict,
    *,
    horizon: int,
    fee_rate: float,
    risk_weight: float,
) -> float:
    sell_price = float(action.get("sell_price", 0))
    if sell_price <= 0:
        return -999.0
    pred_high = float(action.get(f"predicted_high_p50_h{horizon}", 0))
    pred_low = float(action.get(f"predicted_low_p50_h{horizon}", 0))
    if pred_high <= 0 or pred_low <= 0:
        return -999.0
    sell_amount = float(action.get("sell_amount", 0))
    sell_intensity = max(0.0, min(1.0, sell_amount / 100.0))
    if sell_intensity <= 0:
        return -999.0
    upside = (sell_price - pred_low) / sell_price
    downside = max(0.0, (pred_high - sell_price) / sell_price)
    edge = upside - risk_weight * downside - 2.0 * fee_rate
    return edge * sell_intensity


def _spawn_exit_watcher(
    *,
    symbol: str,
    position_side: str,
    qty: float,
    limit_price: float,
    expiry_minutes: int,
    poll_seconds: int,
    price_tolerance: float,
    dry_run: bool,
    margin: bool,
) -> None:
    side = "buy" if position_side == "short" else "sell"
    side_effect = "AUTO_REPAY" if margin else "NO_SIDE_EFFECT"
    spawn_watcher(
        WatcherPlan(
            symbol=symbol,
            side=side,
            mode="exit",
            limit_price=limit_price,
            target_qty=qty,
            expiry_minutes=expiry_minutes,
            poll_seconds=poll_seconds,
            price_tolerance=price_tolerance,
            dry_run=dry_run,
            margin=margin,
            side_effect_type=side_effect,
        )
    )


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
        data_cfg = _resolve_dataset_config(
            base_cfg,
            input_dim=input_dim,
            horizon=horizon,
            saved_feature_columns=payload.get("feature_columns"),
        )
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
    allow_short: bool = False,
    margin: bool = False,
    max_account_leverage: float = 5.0,
    max_long_leverage: Optional[float] = None,
    max_short_leverage: Optional[float] = None,
    auto_convert_quotes: bool = True,
    quote_conversion_cooldown_minutes: int = 60,
    quote_leave_buffer: float = 10.0,
    max_quote_conversion: Optional[float] = None,
    work_steal: bool = False,
    work_steal_min_profit_pct: float = 0.001,
    work_steal_min_edge: float = 0.005,
    work_steal_edge_margin: float = 0.0,
) -> None:
    use_margin = bool(margin or allow_short)
    for symbol in symbols:
        _refresh_price_csv(symbol, data_root)

    state = SelectorState.load(state_path)
    holdings = _detect_holdings(symbols, margin=use_margin)

    if len(holdings) > 1:
        holdings.sort(key=lambda x: x.notional, reverse=True)
        primary = holdings[0]
        extras = holdings[1:]
        print(f"[selector] WARNING: holding {len(holdings)} assets, selling extras to consolidate")
        for holding in extras:
            sym = holding.symbol
            qty = abs(float(holding.quantity))
            notional = float(holding.notional)
            side_name = holding.position_side or "long"
            print(
                f"[selector] exiting extra {sym}/{side_name}: qty={qty:.6f} notional=${notional:.2f}"
            )
            try:
                rules = resolve_symbol_rules(sym)
                market_price = float(binance_wrapper.get_symbol_price(sym) or 0.0)
                if market_price <= 0:
                    continue
                if side_name == "short":
                    limit_price = quantize_price(market_price * 1.001, tick_size=rules.tick_size, side="buy")
                else:
                    limit_price = quantize_price(market_price * 0.999, tick_size=rules.tick_size, side="sell")
                exit_qty = quantize_qty(qty, step_size=rules.step_size)
                if exit_qty > 0:
                    _spawn_exit_watcher(
                        symbol=sym,
                        position_side=side_name,
                        qty=exit_qty,
                        limit_price=limit_price,
                        expiry_minutes=expiry_minutes,
                        poll_seconds=poll_seconds,
                        price_tolerance=price_tolerance * 3,
                        dry_run=dry_run,
                        margin=use_margin,
                    )
            except Exception as exc:
                print(f"[selector] failed to exit extra {sym}: {exc}")
        held_symbol = primary.symbol
        held_qty = primary.quantity
        state.open_symbol = held_symbol
        state.open_binance_symbol = held_symbol
        state.position_side = primary.position_side or "long"
        state.active_quote_asset = split_binance_symbol(held_symbol)[1]
        if not state.open_ts:
            state.open_ts = datetime.now(timezone.utc).isoformat()
        if state.open_price <= 0:
            try:
                state.open_price = float(binance_wrapper.get_symbol_price(held_symbol))
            except Exception:
                pass
        state.save(state_path)
    elif len(holdings) == 1:
        held_symbol = holdings[0].symbol
        held_qty = holdings[0].quantity
        if state.open_symbol != held_symbol:
            print(f"[selector] detected holding {held_symbol} ({held_qty:.6f}), syncing state")
            state.open_symbol = held_symbol
            state.open_binance_symbol = held_symbol
            state.position_side = holdings[0].position_side or "long"
            state.active_quote_asset = split_binance_symbol(held_symbol)[1]
            if not state.open_ts:
                state.open_ts = datetime.now(timezone.utc).isoformat()
            if state.open_price <= 0:
                try:
                    state.open_price = float(binance_wrapper.get_symbol_price(held_symbol))
                except Exception:
                    pass
            state.save(state_path)
            cancel_entry_watchers(exclude_symbol=held_symbol)
        elif state.open_price <= 0:
            try:
                state.open_price = float(binance_wrapper.get_symbol_price(held_symbol))
                state.save(state_path)
            except Exception:
                pass
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
        stolen = False
        if work_steal:
            stolen = _handle_work_steal(
                state, actions, symbols,
                horizon=horizon,
                risk_weight=risk_weight,
                allow_short=allow_short,
                margin=use_margin,
                work_steal_min_profit_pct=work_steal_min_profit_pct,
                work_steal_min_edge=work_steal_min_edge,
                work_steal_edge_margin=work_steal_edge_margin,
                min_gap_pct=min_gap_pct,
                poll_seconds=poll_seconds,
                expiry_minutes=expiry_minutes,
                price_tolerance=price_tolerance,
                dry_run=dry_run,
                state_path=state_path,
            )
        if not stolen:
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
                margin=use_margin,
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
            allow_short=allow_short,
            margin=use_margin,
            max_account_leverage=max_account_leverage,
            max_long_leverage=max_long_leverage,
            max_short_leverage=max_short_leverage,
            auto_convert_quotes=auto_convert_quotes,
            quote_conversion_cooldown_minutes=quote_conversion_cooldown_minutes,
            quote_leave_buffer=quote_leave_buffer,
            max_quote_conversion=max_quote_conversion,
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
    margin: bool = False,
) -> None:
    symbol = state.open_symbol
    position_side = _normalize_position_side(state.position_side) or "long"
    hours = state.hours_held()
    force_close = max_hold_hours is not None and hours >= max_hold_hours

    if margin:
        ctx = _detect_margin_context(symbol)
        net_qty = float(ctx.get("asset_net", 0.0))
        detected_side = _normalize_position_side(str(ctx.get("position_side", "")))
        if detected_side:
            position_side = detected_side
        if abs(net_qty) <= 0.0:
            state.open_symbol = None
            state.open_binance_symbol = None
            state.position_side = ""
            state.open_ts = None
            state.open_price = 0.0
            state.save(state_path)
            return

        rules = resolve_symbol_rules(symbol)
        market_price = float(ctx.get("market_price", 0.0))
        if force_close:
            print(f"[selector] FORCE CLOSE {symbol}/{position_side} after {hours:.1f}h (max={max_hold_hours}h)")
            if market_price <= 0.0:
                print(f"[selector] can't get market price for {symbol}")
                return
            if position_side == "short":
                limit_price = quantize_price(market_price * 1.001, tick_size=rules.tick_size, side="buy")
            else:
                limit_price = quantize_price(market_price * 0.999, tick_size=rules.tick_size, side="sell")
            qty = quantize_qty(abs(net_qty), step_size=rules.step_size)
            if qty > 0:
                _spawn_exit_watcher(
                    symbol=symbol,
                    position_side=position_side,
                    qty=qty,
                    limit_price=limit_price,
                    expiry_minutes=expiry_minutes,
                    poll_seconds=poll_seconds,
                    price_tolerance=price_tolerance * 3,
                    dry_run=dry_run,
                    margin=True,
                )
                print(f"[selector] force-close {symbol}/{position_side} px={limit_price:.4f} qty={qty:.6f}")
            return

        action = actions.get(symbol)
        if not action:
            print(f"[selector] holding {symbol}/{position_side} ({hours:.1f}h) but no action, waiting")
            return

        plan = _build_plan(action, intensity_scale=intensity_scale)
        offset = price_offset_map.get(symbol, default_offset)
        buy_price = plan.buy_price * (1.0 - offset)
        sell_price = plan.sell_price * (1.0 + offset)
        buy_price, sell_price = enforce_min_spread(buy_price, sell_price, min_spread_pct=min_gap_pct)
        buy_price, sell_price = enforce_gap(symbol, buy_price, sell_price, min_gap_pct=min_gap_pct)
        validated = _ensure_valid_levels(symbol, buy_price, sell_price, min_gap_pct=min_gap_pct, rules=rules)
        if validated is None:
            print(f"[selector] invalid exit prices for {symbol}")
            return
        buy_price, sell_price = validated

        if position_side == "short":
            qty = quantize_qty(abs(net_qty) * max(0.0, min(1.0, plan.buy_amount / 100.0)), step_size=rules.step_size)
            exit_price = quantize_price(buy_price, tick_size=rules.tick_size, side="buy")
        else:
            qty = quantize_qty(abs(net_qty) * max(0.0, min(1.0, plan.sell_amount / 100.0)), step_size=rules.step_size)
            exit_price = quantize_price(sell_price, tick_size=rules.tick_size, side="sell")
        if rules.min_qty is not None and qty < rules.min_qty:
            qty = 0.0
        if rules.min_notional is not None and qty * exit_price < rules.min_notional:
            qty = 0.0
        if qty > 0:
            _spawn_exit_watcher(
                symbol=symbol,
                position_side=position_side,
                qty=qty,
                limit_price=exit_price,
                expiry_minutes=expiry_minutes,
                poll_seconds=poll_seconds,
                price_tolerance=price_tolerance,
                dry_run=dry_run,
                margin=True,
            )
        print(
            f"[selector] hold {symbol}/{position_side} ({hours:.1f}h) px={exit_price:.4f}({qty:.6f}) "
            f"buy_amt={plan.buy_amount:.2f} sell_amt={plan.sell_amount:.2f}"
        )
        return

    if force_close:
        print(f"[selector] FORCE CLOSE {symbol} after {hours:.1f}h (max={max_hold_hours}h)")
        try:
            _, base_total = get_total_balances(symbol)
        except Exception as exc:
            print(f"[selector] failed to get balances for {symbol}: {exc}")
            return
        if base_total <= 0:
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
            base_free=base_total,
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
        print(f"[selector] holding {symbol}/{position_side} ({hours:.1f}h) but no action, waiting")
        return

    plan = _build_plan(action, intensity_scale=intensity_scale)
    offset = price_offset_map.get(symbol, default_offset)
    sell_price = plan.sell_price * (1.0 + offset)

    try:
        quote_free, base_total = get_total_balances(symbol)
    except Exception as exc:
        print(f"[selector] failed to get balances for {symbol}: {exc}")
        return

    if base_total <= 0:
        print(f"[selector] no {symbol} balance, clearing state")
        state.open_symbol = None
        state.open_binance_symbol = None
        state.position_side = ""
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
        quote_free=0, base_free=base_total, rules=rules,
    )
    if sizing.sell_qty > 0:
        spawn_watcher(WatcherPlan(
            symbol=symbol, side="sell", mode="exit",
            limit_price=sell_price, target_qty=sizing.sell_qty,
            expiry_minutes=expiry_minutes, poll_seconds=poll_seconds,
            price_tolerance=price_tolerance, dry_run=dry_run,
        ))
    print(
        f"[selector] hold {symbol}/{position_side} ({hours:.1f}h) sell={sell_price:.4f}({sizing.sell_qty:.6f}) "
        f"amt={plan.sell_amount:.2f}"
    )


def _handle_work_steal(
    state: SelectorState,
    actions: Dict[str, dict],
    symbols: List[str],
    *,
    horizon: int,
    risk_weight: float,
    allow_short: bool,
    margin: bool,
    work_steal_min_profit_pct: float,
    work_steal_min_edge: float,
    work_steal_edge_margin: float,
    min_gap_pct: float,
    poll_seconds: int,
    expiry_minutes: int,
    price_tolerance: float,
    dry_run: bool,
    state_path: Path,
) -> bool:
    symbol = state.open_symbol
    position_side = _normalize_position_side(state.position_side) or "long"
    if not symbol or state.open_price <= 0:
        return False
    if margin:
        try:
            market_price = float(_detect_margin_context(symbol).get("market_price", 0.0))
        except Exception:
            return False
    else:
        try:
            ticker = {"price": binance_wrapper.get_symbol_price(symbol)}
            market_price = float(ticker.get("price", 0))
        except Exception:
            return False
    if market_price <= 0:
        return False
    if position_side == "short":
        unrealized_pct = (state.open_price - market_price) / state.open_price
    else:
        unrealized_pct = (market_price - state.open_price) / state.open_price
    if unrealized_pct < work_steal_min_profit_pct:
        return False

    candidates: List[Tuple[float, str, str]] = []
    for sym, action in actions.items():
        if sym == symbol:
            continue
        fee_rate = get_fee_for_symbol(sym)
        edge = _compute_edge(action, horizon=horizon, fee_rate=fee_rate, risk_weight=risk_weight)
        if edge >= work_steal_min_edge and edge >= unrealized_pct + work_steal_edge_margin:
            candidates.append((edge, sym, "long"))
        if allow_short and margin:
            short_edge = _compute_short_edge(action, horizon=horizon, fee_rate=fee_rate, risk_weight=risk_weight)
            if short_edge >= work_steal_min_edge and short_edge >= unrealized_pct + work_steal_edge_margin:
                candidates.append((short_edge, sym, "short"))

    if not candidates:
        return False

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_edge, best_sym, best_side = candidates[0]
    print(
        f"[selector] WORK STEAL: exiting {symbol}/{position_side} (profit={unrealized_pct:.4f}) "
        f"to enter {best_sym}/{best_side} (edge={best_edge:.4f})"
    )

    rules = resolve_symbol_rules(symbol)
    if margin:
        ctx = _detect_margin_context(symbol)
        net_qty = abs(float(ctx.get("asset_net", 0.0)))
        if net_qty <= 0.0:
            return False
        if position_side == "short":
            exit_price = quantize_price(market_price * 1.001, tick_size=rules.tick_size, side="buy")
        else:
            exit_price = quantize_price(market_price * 0.999, tick_size=rules.tick_size, side="sell")
        qty = quantize_qty(net_qty, step_size=rules.step_size)
        if qty > 0:
            _spawn_exit_watcher(
                symbol=symbol,
                position_side=position_side,
                qty=qty,
                limit_price=exit_price,
                expiry_minutes=min(5, expiry_minutes),
                poll_seconds=poll_seconds,
                price_tolerance=price_tolerance * 3,
                dry_run=dry_run,
                margin=True,
            )
            print(f"[selector] work-steal exit {symbol}/{position_side} px={exit_price:.4f} qty={qty:.6f}")
    else:
        try:
            _, base_free = get_free_balances(symbol)
        except Exception as exc:
            print(f"[selector] work-steal failed to get balances for {symbol}: {exc}")
            return False
        if base_free <= 0:
            return False
        sell_price = market_price * 0.999
        validated = _ensure_valid_levels(symbol, sell_price * 0.99, sell_price, min_gap_pct=min_gap_pct, rules=rules)
        if validated is None:
            return False
        _, sell_price = validated
        sizing = compute_order_quantities(
            symbol=symbol, buy_amount=0, sell_amount=100.0,
            buy_price=sell_price, sell_price=sell_price,
            quote_free=0, base_free=base_free, rules=rules,
        )
        if sizing.sell_qty > 0:
            spawn_watcher(WatcherPlan(
                symbol=symbol, side="sell", mode="exit",
                limit_price=sell_price, target_qty=sizing.sell_qty,
                expiry_minutes=min(5, expiry_minutes), poll_seconds=poll_seconds,
                price_tolerance=price_tolerance * 3, dry_run=dry_run,
            ))
            print(f"[selector] work-steal exit {symbol} sell={sell_price:.4f} qty={sizing.sell_qty:.6f}")
    state.open_symbol = None
    state.open_binance_symbol = None
    state.open_ts = None
    state.open_price = 0.0
    state.position_side = ""
    state.save(state_path)
    return True


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
    allow_short: bool = False,
    margin: bool = False,
    max_account_leverage: float = 5.0,
    max_long_leverage: Optional[float] = None,
    max_short_leverage: Optional[float] = None,
    auto_convert_quotes: bool = True,
    quote_conversion_cooldown_minutes: int = 60,
    quote_leave_buffer: float = 10.0,
    max_quote_conversion: Optional[float] = None,
) -> None:
    long_lev, short_lev = _resolve_directional_leverages(
        max_account_leverage=max_account_leverage,
        max_long_leverage=max_long_leverage,
        max_short_leverage=max_short_leverage,
    )
    candidates: List[Tuple[float, str, str]] = []
    for symbol, action in actions.items():
        fee_rate = get_fee_for_symbol(symbol)
        edge = _compute_edge(action, horizon=horizon, fee_rate=fee_rate, risk_weight=risk_weight)
        if edge >= min_edge:
            candidates.append((edge, symbol, "long"))
            print(f"[selector] {symbol} long edge={edge:.6f}")
        else:
            print(f"[selector] {symbol} long edge={edge:.6f} (below min_edge={min_edge})")
        if allow_short and margin:
            short_edge = _compute_short_edge(action, horizon=horizon, fee_rate=fee_rate, risk_weight=risk_weight)
            if short_edge >= min_edge:
                candidates.append((short_edge, symbol, "short"))
                print(f"[selector] {symbol} short edge={short_edge:.6f}")
            else:
                print(f"[selector] {symbol} short edge={short_edge:.6f} (below min_edge={min_edge})")

    if not candidates:
        print("[selector] no candidates above min_edge, staying flat")
        return

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_edge, best_symbol, best_side = candidates[0]
    _, best_quote_asset = split_binance_symbol(best_symbol)
    selected_candidates = [
        candidate
        for candidate in candidates
        if split_binance_symbol(candidate[1])[1] == best_quote_asset and candidate[2] == best_side
    ]
    if not selected_candidates:
        print("[selector] no candidates in selected quote bucket, staying flat")
        return

    if auto_convert_quotes and not margin and best_side == "long":
        _maybe_convert_quote_asset(
            state,
            target_quote_asset=best_quote_asset,
            leave_quote_buffer=quote_leave_buffer,
            max_quote_conversion=max_quote_conversion,
            conversion_cooldown_minutes=quote_conversion_cooldown_minutes,
            dry_run=dry_run,
        )

    cancel_entry_watchers()
    time.sleep(1)

    quote_free_total = 0.0
    if not margin:
        try:
            quote_free, _ = get_free_balances(best_symbol)
        except Exception as exc:
            print(f"[selector] failed to get balances: {exc}")
            return
        quote_free_total = float(quote_free or 0.0)
        if quote_free_total <= 0:
            print(f"[selector] no {best_quote_asset} balance available, staying flat")
            return
    state.active_quote_asset = best_quote_asset
    state.save(state_path)

    best_entry_price = 0.0
    placed = 0

    per_candidate_quote = (
        quote_free_total / float(len(selected_candidates))
        if quote_free_total > 0 and selected_candidates
        else 0.0
    )

    for edge, symbol, direction in selected_candidates:
        action = actions.get(symbol)
        if not action:
            continue
        plan = _build_plan(action, intensity_scale=intensity_scale)
        offset = price_offset_map.get(symbol, default_offset)
        buy_price = plan.buy_price * (1.0 - offset)
        sell_price = plan.sell_price * (1.0 + offset)
        buy_price, sell_price = enforce_min_spread(buy_price, sell_price, min_spread_pct=min_gap_pct)
        buy_price, sell_price = enforce_gap(symbol, buy_price, sell_price, min_gap_pct=min_gap_pct)

        if buy_price <= 0 or sell_price <= 0 or buy_price >= sell_price:
            print(f"[selector] invalid prices for {symbol}")
            continue

        rules = resolve_symbol_rules(symbol)
        validated = _ensure_valid_levels(symbol, buy_price, sell_price, min_gap_pct=min_gap_pct, rules=rules)
        if validated is None:
            print(f"[selector] invalid levels for {symbol}")
            continue
        buy_price, sell_price = validated

        if direction == "short":
            if not margin:
                continue
            ctx = _detect_margin_context(symbol)
            base = str(ctx.get("base", ""))
            gross_power = max(0.0, float(ctx.get("equity", 0.0))) * short_lev
            per_candidate_power = gross_power / float(len(selected_candidates))
            max_borrowable_asset = get_max_borrowable(base) if short_lev > 0 else 0.0
            borrow_cap = (float(ctx.get("asset_free", 0.0)) + max_borrowable_asset) * sell_price
            qty, entry_price = _quantized_fractional_qty(
                side="sell",
                price=sell_price,
                intensity_amount=plan.sell_amount,
                notional_cap=min(per_candidate_power, borrow_cap),
                rules=rules,
            )
            if qty <= 0:
                continue
            spawn_watcher(
                WatcherPlan(
                    symbol=symbol,
                    side="sell",
                    mode="entry",
                    limit_price=entry_price,
                    target_qty=qty,
                    expiry_minutes=expiry_minutes,
                    poll_seconds=poll_seconds,
                    price_tolerance=price_tolerance,
                    dry_run=dry_run,
                    margin=True,
                    side_effect_type="AUTO_BORROW_REPAY",
                )
            )
            if symbol == best_symbol:
                best_entry_price = entry_price
        else:
            if margin:
                ctx = _detect_margin_context(symbol)
                quote_asset = str(ctx.get("quote", ""))
                gross_power = max(0.0, float(ctx.get("equity", 0.0))) * long_lev
                max_borrowable_quote = get_max_borrowable(quote_asset) if long_lev > 1.0 else 0.0
                available_quote = min(gross_power, float(ctx.get("quote_free", 0.0)) + max_borrowable_quote)
                sizing = compute_order_quantities(
                    symbol=symbol,
                    buy_amount=plan.buy_amount,
                    sell_amount=0,
                    buy_price=buy_price,
                    sell_price=sell_price,
                    quote_free=available_quote / float(len(selected_candidates)) if selected_candidates else 0.0,
                    base_free=0,
                    rules=rules,
                )
            else:
                sizing = compute_order_quantities(
                    symbol=symbol,
                    buy_amount=plan.buy_amount,
                    sell_amount=0,
                    buy_price=buy_price,
                    sell_price=sell_price,
                    quote_free=per_candidate_quote,
                    base_free=0,
                    rules=rules,
                )
            if sizing.buy_qty <= 0:
                continue
            spawn_watcher(
                WatcherPlan(
                    symbol=symbol,
                    side="buy",
                    mode="entry",
                    limit_price=buy_price,
                    target_qty=sizing.buy_qty,
                    expiry_minutes=expiry_minutes,
                    poll_seconds=poll_seconds,
                    price_tolerance=price_tolerance,
                    dry_run=dry_run,
                    margin=margin,
                    side_effect_type="MARGIN_BUY" if margin else "NO_SIDE_EFFECT",
                )
            )
            if symbol == best_symbol:
                best_entry_price = buy_price
        placed += 1
        print(
            f"[selector] ENTER {symbol}/{direction} edge={edge:.6f} "
            f"px={(sell_price if direction == 'short' else buy_price):.4f} "
            f"alloc_quote={(per_candidate_quote if not margin else 0.0):.2f}"
        )

    if placed <= 0:
        print("[selector] no valid entry orders placed")
        return

    if best_entry_price > 0:
        state.open_price = best_entry_price
        state.save(state_path)


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
    parser.add_argument("--work-steal", action="store_true")
    parser.add_argument("--work-steal-min-profit-pct", type=float, default=0.001)
    parser.add_argument("--work-steal-min-edge", type=float, default=0.005)
    parser.add_argument("--work-steal-edge-margin", type=float, default=0.0)
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
                work_steal=args.work_steal,
                work_steal_min_profit_pct=args.work_steal_min_profit_pct,
                work_steal_min_edge=args.work_steal_min_edge,
                work_steal_edge_margin=args.work_steal_edge_margin,
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
