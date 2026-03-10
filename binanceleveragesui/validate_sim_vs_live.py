#!/usr/bin/env python3
"""Validate market simulator against actual Binance production fills.

Architecture matching the live bot:
- Hourly signal generation (neural model on hourly bars)
- 5-minute execution granularity (check fills on 5m bars within each hour)
- Sell-first, no same-5m-bar roundtrips
- Lag=1 hour (signal from hour T-1 applied during hour T)
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import inspect
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import random
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timezone, timedelta

from src.forecast_horizon_utils import resolve_required_forecast_horizons
from src.margin_position_utils import remaining_entry_notional as shared_remaining_entry_notional
from binancechronossolexperiment.data import ChronosSolDataModule, SplitConfig
from binancechronossolexperiment.inference import load_policy_checkpoint
from binanceneural.inference import generate_latest_action
from binanceleveragesui.trade_margin_meta import (
    _has_effective_position as live_has_effective_position,
    _load_profit_gate_5m_bars,
    _signal_from_action,
)
from src.binan.binance_margin import get_margin_trades, get_all_margin_orders

REPO = Path(__file__).resolve().parents[1]


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_policy_checkpoint_compat(
    checkpoint_path: str,
    *,
    device: str,
    data_root: str | Path,
    forecast_cache_root: str | Path,
):
    kwargs = {"device": device}
    params = inspect.signature(load_policy_checkpoint).parameters
    if "data_root" in params:
        kwargs["data_root"] = Path(data_root)
    if "forecast_cache_root" in params:
        kwargs["forecast_cache_root"] = Path(forecast_cache_root)
    return load_policy_checkpoint(checkpoint_path, **kwargs)


def pull_prod_fills(symbol: str, start_ms: int, end_ms: int):
    MS_24H = 24 * 3600 * 1000
    raw_trades, raw_orders = [], []
    cursor = start_ms
    while cursor < end_ms:
        chunk_end = min(cursor + MS_24H, end_ms)
        raw_trades.extend(get_margin_trades(symbol, start_time=cursor, end_time=chunk_end, limit=1000))
        raw_orders.extend(get_all_margin_orders(symbol, start_time=cursor, end_time=chunk_end, limit=500))
        cursor = chunk_end

    trades = []
    for t in raw_trades:
        trades.append({
            "timestamp": pd.Timestamp(int(t["time"]), unit="ms", tz="UTC"),
            "side": "buy" if t.get("isBuyer") else "sell",
            "price": float(t["price"]),
            "qty": float(t["qty"]),
            "quote_qty": float(t.get("quoteQty", 0)),
            "commission": float(t.get("commission", 0)),
            "commission_asset": t.get("commissionAsset", ""),
            "order_id": t.get("orderId"),
        })
    if not trades:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    tdf = pd.DataFrame(trades)
    agg = tdf.groupby("order_id").agg(
        timestamp=("timestamp", "first"),
        side=("side", "first"),
        avg_price=("price", lambda x: np.average(x, weights=tdf.loc[x.index, "qty"])),
        total_qty=("qty", "sum"),
        total_quote=("quote_qty", "sum"),
        total_commission=("commission", "sum"),
        commission_asset=("commission_asset", "first"),
        n_fills=("qty", "count"),
    ).reset_index().sort_values("timestamp").reset_index(drop=True)
    agg = agg[agg["total_qty"] > 1e-12].reset_index(drop=True)

    orders = []
    for o in raw_orders:
        orders.append({
            "order_id": o.get("orderId"),
            "timestamp": pd.Timestamp(int(o["time"]), unit="ms", tz="UTC"),
            "side": o.get("side", "").lower(),
            "status": o.get("status", ""),
            "limit_price": float(o.get("price", 0)),
            "orig_qty": float(o.get("origQty", 0)),
            "executed_qty": float(o.get("executedQty", 0)),
            "cumm_quote": float(o.get("cummulativeQuoteQty", 0)),
        })
    odf = pd.DataFrame(orders) if orders else pd.DataFrame()
    return agg, tdf, odf


def load_5m_bars(
    symbol: str,
    start_ts,
    end_ts,
    *,
    data_root: str | Path | None = None,
    bars_5m_root: str | Path | None = None,
):
    loader_args = argparse.Namespace(
        data_root=data_root,
        profit_gate_5m_root=bars_5m_root,
    )
    df = _load_profit_gate_5m_bars(symbol, start_ts, end_ts, args=loader_args)
    if df.empty:
        raise FileNotFoundError(f"No 5m data for {symbol} between {start_ts} and {end_ts}")
    return df


def reconstruct_initial_state(symbol: str, start_ms: int, lookback_hours: int = 24):
    """Pull fills from before start to figure out initial position and entry time."""
    lb_ms = lookback_hours * 3600 * 1000
    pre_start = start_ms - lb_ms
    MS_24H = 24 * 3600 * 1000
    raw = []
    cursor = pre_start
    while cursor < start_ms:
        chunk_end = min(cursor + MS_24H, start_ms)
        raw.extend(get_margin_trades(symbol, start_time=cursor, end_time=chunk_end, limit=1000))
        cursor = chunk_end
    if not raw:
        return 0.0, None

    # replay fills to get net position at start
    pos = 0.0
    last_long_entry_ts = None
    last_short_entry_ts = None
    for t in sorted(raw, key=lambda x: int(x["time"])):
        qty = float(t["qty"])
        if t.get("isBuyer"):
            pos += qty
            last_long_entry_ts = pd.Timestamp(int(t["time"]), unit="ms", tz="UTC")
        else:
            pos -= qty
            last_short_entry_ts = pd.Timestamp(int(t["time"]), unit="ms", tz="UTC")

    if pos > 0:
        print(f"  Initial position: long {pos:.0f} {symbol} (entered ~{last_long_entry_ts})")
        return pos, last_long_entry_ts
    if pos < 0:
        print(f"  Initial position: short {abs(pos):.0f} {symbol} (entered ~{last_short_entry_ts})")
        return pos, last_short_entry_ts
    print(f"  Initial position: flat (raw={pos:.0f})")
    return 0.0, None


def resolve_initial_replay_state(args, start_ms: int):
    explicit_inv = getattr(args, "initial_inv", None)
    if explicit_inv is not None:
        inv = float(explicit_inv)
        entry_ts = None
        raw_entry_ts = getattr(args, "initial_entry_ts", None)
        if abs(inv) > 0.0 and raw_entry_ts:
            entry_ts = pd.Timestamp(raw_entry_ts)
            if entry_ts.tzinfo is None:
                entry_ts = entry_ts.tz_localize("UTC")
            else:
                entry_ts = entry_ts.tz_convert("UTC")
        return inv, entry_ts

    if bool(getattr(args, "skip_initial_reconstruction", False)):
        return 0.0, None

    lookback_hours = int(getattr(args, "initial_reconstruction_lookback_hours", 48))
    return reconstruct_initial_state(args.symbol, start_ms, lookback_hours=lookback_hours)


def _resolve_sim_directional_args(args):
    allow_short = bool(getattr(args, "allow_short", False))
    long_lev, short_lev = _resolve_directional_leverages(args)
    if not allow_short:
        short_lev = 0.0
    return allow_short, long_lev, short_lev


def _position_side(inv: float, *, eps: float = 1e-12) -> str:
    if inv > eps:
        return "long"
    if inv < -eps:
        return "short"
    return "flat"


@dataclass(frozen=True)
class DirectionalSignal:
    entry_price: float
    exit_price: float
    entry_amount: float
    exit_amount: float
    entry_side: str
    exit_side: str


def position_side_from_qty(qty: float, *, step_size: float = 0.0) -> str:
    eps = max(1e-12, float(step_size or 0.0) * 0.5)
    side = _position_side(float(qty), eps=eps)
    return "" if side == "flat" else side


def effective_position_side_from_qty(
    qty: float,
    *,
    market_price: float,
    step_size: float = 0.0,
    max_position_notional: float | None = None,
) -> str:
    side = position_side_from_qty(qty, step_size=step_size)
    if side == "":
        return ""
    if not live_has_effective_position(
        qty,
        abs(float(qty)) * max(0.0, float(market_price)),
        step_size=float(step_size or 0.0),
        max_position_notional=max_position_notional,
    ):
        return ""
    return side


def directional_signal(signal: dict, *, side: str) -> DirectionalSignal:
    if side == "short":
        return DirectionalSignal(
            entry_price=float(signal.get("sell_price", 0.0)),
            exit_price=float(signal.get("buy_price", 0.0)),
            entry_amount=float(signal.get("sell_amount", 0.0)),
            exit_amount=float(signal.get("buy_amount", 0.0)),
            entry_side="sell",
            exit_side="buy",
        )
    return DirectionalSignal(
        entry_price=float(signal.get("buy_price", 0.0)),
        exit_price=float(signal.get("sell_price", 0.0)),
        entry_amount=float(signal.get("buy_amount", 0.0)),
        exit_amount=float(signal.get("sell_amount", 0.0)),
        entry_side="buy",
        exit_side="sell",
    )


def choose_flat_entry_side(signal: dict, *, allow_short: bool) -> str:
    long_amount = max(0.0, float(signal.get("buy_amount", 0.0))) if float(signal.get("buy_price", 0.0)) > 0 else 0.0
    short_amount = (
        max(0.0, float(signal.get("sell_amount", 0.0)))
        if allow_short and float(signal.get("sell_price", 0.0)) > 0
        else 0.0
    )
    if long_amount <= 0.0 and short_amount <= 0.0:
        return ""
    return "long" if long_amount >= short_amount else "short"


def remaining_entry_notional(
    *,
    side: str,
    equity: float,
    current_qty: float,
    market_price: float,
    long_max_leverage: float,
    short_max_leverage: float,
) -> float:
    return shared_remaining_entry_notional(
        side=side,
        equity=equity,
        current_qty=current_qty,
        market_price=market_price,
        long_max_leverage=long_max_leverage,
        short_max_leverage=short_max_leverage,
    )


def _aligned_position_notional(
    *,
    side: str,
    current_qty: float,
    market_price: float,
) -> float:
    price = max(0.0, float(market_price))
    qty = float(current_qty)
    if str(side or "").strip().lower() == "short":
        return abs(min(qty, 0.0)) * price
    return max(qty, 0.0) * price


def _cap_directional_entry_notional(
    *,
    target_notional: float,
    side: str,
    current_qty: float,
    market_price: float,
    max_position_notional: float | None,
) -> float:
    target = max(0.0, float(target_notional))
    if max_position_notional is None:
        return target
    cap = max(0.0, float(max_position_notional))
    aligned = _aligned_position_notional(
        side=side,
        current_qty=current_qty,
        market_price=market_price,
    )
    current_notional = abs(float(current_qty)) * max(0.0, float(market_price))
    opposing = max(0.0, current_notional - aligned)
    remaining = max(0.0, cap - aligned + opposing)
    return min(target, remaining)


def _cap_position_notional(
    target_notional: float,
    current_asset_notional: float,
    max_position_notional: float | None,
) -> float:
    target = max(0.0, float(target_notional))
    if max_position_notional is None:
        return target
    cap = max(0.0, float(max_position_notional))
    remaining = max(0.0, cap - max(0.0, float(current_asset_notional)))
    return min(target, remaining)


def _resolve_directional_leverages(args) -> tuple[float, float]:
    long_max_leverage = float(
        getattr(
            args,
            "max_long_leverage",
            getattr(args, "long_max_leverage", getattr(args, "max_leverage", 1.0)),
        )
    )
    short_max_leverage = float(
        getattr(
            args,
            "max_short_leverage",
            getattr(args, "short_max_leverage", long_max_leverage),
        )
    )
    return max(0.0, long_max_leverage), max(0.0, short_max_leverage)


def generate_hourly_signals(args, frame, model, normalizer, feature_columns, meta):
    """Generate signals for each hourly bar."""
    seq_len = meta.get("sequence_length", args.sequence_length)
    start_ts = pd.Timestamp(args.start, tz="UTC") - pd.Timedelta(hours=2)
    start_idx_arr = frame.index[frame["timestamp"] >= start_ts]
    if len(start_idx_arr) == 0:
        return {}
    start_idx = start_idx_arr[0]

    signals = {}
    for bar_idx in range(start_idx, len(frame)):
        ts = frame.iloc[bar_idx]["timestamp"]
        sub_frame = frame.iloc[:bar_idx + 1].copy()
        set_seeds(42)
        action = generate_latest_action(
            model=model, frame=sub_frame, feature_columns=feature_columns,
            normalizer=normalizer, sequence_length=seq_len, horizon=args.horizon,
        )
        signal = _signal_from_action(
            action,
            frame.iloc[bar_idx],
            intensity_scale=float(args.intensity_scale),
            symbol=str(getattr(args, "symbol", frame.iloc[bar_idx].get("symbol", ""))),
        )
        signals[ts] = {
            "buy_price": float(signal["buy_price"]),
            "sell_price": float(signal["sell_price"]),
            "buy_amount": float(signal["buy_amount"]),
            "sell_amount": float(signal["sell_amount"]),
        }
    return signals


def _simulate_5m_legacy(args, hourly_signals, bars_5m, initial_inv=0.0, initial_entry_ts=None):
    fee = float(args.fee)
    fill_buf = float(args.fill_buffer_pct)
    cash = float(args.initial_cash)
    inv = float(initial_inv)
    margin_hourly_rate = float(getattr(args, "margin_hourly_rate", 0.0))
    sim_trades = []
    entry_ts = initial_entry_ts
    exec_start = pd.Timestamp(args.start, tz="UTC")
    verbose = bool(getattr(args, "verbose", False))

    realistic = bool(getattr(args, "realistic", False))
    expiry_minutes = float(getattr(args, "expiry_minutes", 90))
    max_fill_frac = float(getattr(args, "max_fill_fraction", 0.01))
    min_notional = float(getattr(args, "min_notional", 5.0))
    tick_size = float(getattr(args, "tick_size", 0.00001))
    step_size = float(getattr(args, "step_size", 1.0))
    max_position_notional = getattr(args, "max_position_notional", None)
    allow_short, long_max_leverage, short_max_leverage = _resolve_sim_directional_args(args)
    return_trace = bool(getattr(args, "return_trace", False))
    stop_after_cycle = bool(getattr(args, "stop_after_cycle", False))
    initial_mark_price = float(bars_5m.iloc[0]["close"]) if len(bars_5m) > 0 else 0.0
    cycle_started = (
        effective_position_side_from_qty(
            initial_inv,
            market_price=initial_mark_price,
            step_size=step_size if realistic else 0.0,
            max_position_notional=max_position_notional,
        )
        != ""
    )
    trace = []

    buy_order_ts = None
    sell_order_ts = None

    current_sig_hour = None
    bought_this_signal = False
    sold_this_signal = False
    orders_placed = 0
    orders_expired = 0
    orders_filled = 0
    margin_interest_total = 0.0
    prev_ts = None

    def _quantize_price(p, tick):
        if tick <= 0:
            return p
        return round(p / tick) * tick

    def _quantize_qty(q, step):
        if step <= 0:
            return q
        return int(q / step) * step

    def _passes_validation(price, qty):
        if not realistic:
            return True
        return (price * qty) >= min_notional

    def _can_fill(order_qty, bar_volume, bar_ts, order_ts):
        if not realistic:
            return True
        if order_ts is not None and (bar_ts - order_ts).total_seconds() > expiry_minutes * 60:
            return False
        if bar_volume > 0 and order_qty > bar_volume * max_fill_frac:
            return False
        return True

    def _position_side(market_price: float) -> str:
        return effective_position_side_from_qty(
            inv,
            market_price=market_price,
            step_size=step_size if realistic else 0.0,
            max_position_notional=max_position_notional,
        )

    def _in_position(market_price: float) -> bool:
        return _position_side(market_price) != ""

    def get_active_signal(ts_5m):
        bar_hour = ts_5m.floor("h")
        prev_hour = bar_hour - pd.Timedelta(hours=1)
        return hourly_signals.get(prev_hour), prev_hour

    for _, bar in bars_5m.iterrows():
        ts = bar["timestamp"]
        if ts < exec_start:
            continue
        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])
        volume = float(bar.get("volume", 0))

        if prev_ts is not None and margin_hourly_rate > 0:
            delta_hours = max(0.0, float((ts - prev_ts).total_seconds()) / 3600.0)
            if delta_hours > 0:
                if cash < 0:
                    interest = abs(cash) * margin_hourly_rate * delta_hours
                    cash -= interest
                    margin_interest_total += interest
                if inv < 0:
                    borrowed_value = abs(inv) * close
                    interest = borrowed_value * margin_hourly_rate * delta_hours
                    cash -= interest
                    margin_interest_total += interest
        prev_ts = ts

        sig, sig_hour = get_active_signal(ts)
        if sig is None:
            continue

        if sig_hour != current_sig_hour:
            if realistic:
                if buy_order_ts is not None:
                    orders_expired += 1
                if sell_order_ts is not None:
                    orders_expired += 1
            current_sig_hour = sig_hour
            bought_this_signal = False
            sold_this_signal = False
            buy_order_ts = ts
            sell_order_ts = ts
            if verbose:
                print(
                    f"  [{str(ts)[:16]}] new signal hour={str(sig_hour)[:13]} "
                    f"bp={sig['buy_price']:.5f} sp={sig['sell_price']:.5f} "
                    f"ba={sig['buy_amount']:.0f}% sa={sig['sell_amount']:.0f}% inv={inv:.4f} cash={cash:.2f}"
                )

        long_sig = directional_signal(sig, side="long")
        short_sig = directional_signal(sig, side="short")
        flat_entry_side = choose_flat_entry_side(sig, allow_short=allow_short)
        if realistic:
            long_sig = directional_signal(
                {
                    "buy_price": _quantize_price(long_sig.entry_price, tick_size),
                    "sell_price": _quantize_price(long_sig.exit_price, tick_size),
                    "buy_amount": long_sig.entry_amount,
                    "sell_amount": long_sig.exit_amount,
                },
                side="long",
            )
            short_sig = directional_signal(
                {
                    "buy_price": _quantize_price(short_sig.exit_price, tick_size),
                    "sell_price": _quantize_price(short_sig.entry_price, tick_size),
                    "buy_amount": short_sig.exit_amount,
                    "sell_amount": short_sig.entry_amount,
                },
                side="short",
            )

        if realistic:
            if buy_order_ts and (ts - buy_order_ts).total_seconds() > expiry_minutes * 60:
                bought_this_signal = True
                orders_expired += 1
                buy_order_ts = None
            if sell_order_ts and (ts - sell_order_ts).total_seconds() > expiry_minutes * 60:
                sold_this_signal = True
                orders_expired += 1
                sell_order_ts = None

        pos_side = _position_side(close)
        if args.max_hold_hours > 0 and pos_side and entry_ts is not None:
            hours_held = (ts - entry_ts).total_seconds() / 3600.0
            if hours_held >= args.max_hold_hours:
                if pos_side == "long":
                    force_price = close * 0.999
                    force_qty = abs(inv)
                    cash += force_qty * force_price * (1 - fee)
                    sim_trades.append({"ts": ts, "side": "force_sell", "price": force_price, "qty": force_qty})
                else:
                    force_price = close * 1.001
                    force_qty = abs(inv)
                    cash -= force_qty * force_price * (1 + fee)
                    sim_trades.append({"ts": ts, "side": "force_buy", "price": force_price, "qty": force_qty})
                inv = 0.0
                entry_ts = None
                bought_this_signal = False
                sold_this_signal = False
                buy_order_ts = None
                sell_order_ts = None
                orders_filled += 1
                if return_trace:
                    trace.append(
                        {
                            "timestamp": ts,
                            "equity": cash + inv * close,
                            "cash": cash,
                            "inv": inv,
                            "in_position": False,
                        }
                    )
                if stop_after_cycle and cycle_started:
                    break
                continue

        acted_this_bar = False

        if not sold_this_signal and short_sig.entry_price > 0:
            sell_target = short_sig.entry_price * (1 + fill_buf)
            if verbose and high >= short_sig.entry_price * 0.998:
                print(
                    f"    [{str(ts)[:16]}] sell check: H={high:.5f} target={sell_target:.5f} "
                    f"{'HIT' if high >= sell_target else 'MISS'}"
                )
            if high >= sell_target:
                sell_qty = 0.0
                if pos_side == "long" and long_sig.exit_amount > 0:
                    sell_qty = min((long_sig.exit_amount / 100.0) * abs(inv), abs(inv))
                elif allow_short and (pos_side == "short" or (pos_side == "" and flat_entry_side == "short")):
                    equity = cash + inv * close
                    max_sell_value = remaining_entry_notional(
                        side="short",
                        equity=equity,
                        current_qty=inv,
                        market_price=close,
                        long_max_leverage=long_max_leverage,
                        short_max_leverage=short_max_leverage,
                    )
                    max_sell_value = _cap_directional_entry_notional(
                        target_notional=max_sell_value,
                        side="short",
                        current_qty=inv,
                        market_price=close,
                        max_position_notional=max_position_notional,
                    )
                    if max_sell_value > 0:
                        amount_frac = short_sig.entry_amount / 100.0
                        sell_qty = (
                            max_sell_value / (short_sig.entry_price * (1 + fee))
                            if realistic
                            else amount_frac * max_sell_value / (short_sig.entry_price * (1 + fee))
                        )
                sell_qty = _quantize_qty(sell_qty, step_size) if realistic else sell_qty
                if sell_qty > 0 and _passes_validation(short_sig.entry_price, sell_qty) and _can_fill(sell_qty, volume, ts, sell_order_ts):
                    cash += sell_qty * short_sig.entry_price * (1 - fee)
                    inv -= sell_qty
                    cycle_started = True
                    sim_trades.append({"ts": ts, "side": "sell", "price": short_sig.entry_price, "qty": sell_qty})
                    acted_this_bar = True
                    sold_this_signal = True
                    bought_this_signal = False
                    sell_order_ts = None
                    orders_filled += 1
                    if verbose:
                        print(f"    [{str(ts)[:16]}] SELL {sell_qty:.4f} @ {short_sig.entry_price:.5f}")
                    current_side = _position_side(short_sig.entry_price)
                    if current_side == "":
                        inv = 0.0
                        entry_ts = None
                    elif pos_side == "" and entry_ts is None:
                        entry_ts = ts
                elif realistic and sell_qty > 0 and _passes_validation(short_sig.entry_price, sell_qty):
                    orders_placed += 1

        if not acted_this_bar and not bought_this_signal and long_sig.entry_price > 0 and low <= long_sig.entry_price * (1 - fill_buf):
            buy_qty = 0.0
            if pos_side == "short" and short_sig.exit_amount > 0:
                buy_qty = min((short_sig.exit_amount / 100.0) * abs(inv), abs(inv))
            elif pos_side == "long" or (pos_side == "" and flat_entry_side == "long"):
                equity = cash + inv * close
                max_buy_value = remaining_entry_notional(
                    side="long",
                    equity=equity,
                    current_qty=inv,
                    market_price=close,
                    long_max_leverage=long_max_leverage,
                    short_max_leverage=short_max_leverage,
                )
                max_buy_value = _cap_directional_entry_notional(
                    target_notional=max_buy_value,
                    side="long",
                    current_qty=inv,
                    market_price=close,
                    max_position_notional=max_position_notional,
                )
                if max_buy_value > 0:
                    amount_frac = long_sig.entry_amount / 100.0
                    buy_qty = (
                        max_buy_value / (long_sig.entry_price * (1 + fee))
                        if realistic
                        else amount_frac * max_buy_value / (long_sig.entry_price * (1 + fee))
                    )
            buy_qty = _quantize_qty(buy_qty, step_size) if realistic else buy_qty
            if buy_qty > 0 and _passes_validation(long_sig.entry_price, buy_qty) and _can_fill(buy_qty, volume, ts, buy_order_ts):
                cash -= buy_qty * long_sig.entry_price * (1 + fee)
                inv += buy_qty
                cycle_started = True
                sim_trades.append({"ts": ts, "side": "buy", "price": long_sig.entry_price, "qty": buy_qty})
                bought_this_signal = True
                sold_this_signal = False
                buy_order_ts = None
                orders_filled += 1
                if verbose:
                    print(f"    [{str(ts)[:16]}] BUY {buy_qty:.4f} @ {long_sig.entry_price:.5f} cash={cash:.2f}")
                current_side = _position_side(long_sig.entry_price)
                if current_side == "":
                    inv = 0.0
                    entry_ts = None
                elif pos_side == "" and entry_ts is None:
                    entry_ts = ts
            elif realistic and buy_qty > 0 and _passes_validation(long_sig.entry_price, buy_qty):
                orders_placed += 1

        if return_trace:
            trace.append(
                {
                    "timestamp": ts,
                    "equity": cash + inv * close,
                    "cash": cash,
                    "inv": inv,
                    "in_position": _in_position(close),
                }
            )

        if stop_after_cycle and cycle_started and not _in_position(close) and entry_ts is None:
            break

    last_close = float(bars_5m.iloc[-1]["close"]) if len(bars_5m) > 0 else 0
    final_eq = cash + inv * last_close

    if realistic:
        print(
            "  [realistic] "
            f"orders_filled={orders_filled} orders_expired={orders_expired} orders_unfilled={orders_placed} "
            f"margin_interest=${margin_interest_total:.4f}"
        )
    elif margin_interest_total > 0:
        print(f"  [sim] margin_interest=${margin_interest_total:.4f}")

    if return_trace:
        return sim_trades, final_eq, cash, inv, pd.DataFrame(trace)
    return sim_trades, final_eq, cash, inv


def _simulate_5m_live_like(args, hourly_signals, bars_5m, initial_inv=0.0, initial_entry_ts=None):
    """Live-aligned execution model with persistent orders and reprice threshold.

    This mirrors `trade_margin_meta` behavior more closely than the legacy
    per-hour reset/expiry model.
    """
    fee = float(args.fee)
    fill_buf = float(args.fill_buffer_pct)
    cash = float(args.initial_cash)
    inv = float(initial_inv)
    margin_hourly_rate = float(getattr(args, "margin_hourly_rate", 0.0))
    sim_trades = []
    entry_ts = initial_entry_ts
    exec_start = pd.Timestamp(args.start, tz="UTC")
    verbose = bool(getattr(args, "verbose", False))

    realistic = bool(getattr(args, "realistic", False))
    expiry_minutes = float(getattr(args, "expiry_minutes", 90))
    use_order_expiry = bool(getattr(args, "use_order_expiry", False))
    max_fill_frac = float(getattr(args, "max_fill_fraction", 0.01))
    min_notional = float(getattr(args, "min_notional", 5.0))
    tick_size = float(getattr(args, "tick_size", 0.00001))
    step_size = float(getattr(args, "step_size", 1.0))
    max_hold_hours = float(getattr(args, "max_hold_hours", 0.0))
    reprice_threshold = max(0.0, float(getattr(args, "reprice_threshold", 0.003)))
    max_position_notional = getattr(args, "max_position_notional", None)
    allow_short, long_max_leverage, short_max_leverage = _resolve_sim_directional_args(args)
    return_trace = bool(getattr(args, "return_trace", False))
    stop_after_cycle = bool(getattr(args, "stop_after_cycle", False))
    initial_mark_price = float(bars_5m.iloc[0]["close"]) if len(bars_5m) > 0 else 0.0
    cycle_started = (
        effective_position_side_from_qty(
            initial_inv,
            market_price=initial_mark_price,
            step_size=step_size if realistic else 0.0,
            max_position_notional=max_position_notional,
        )
        != ""
    )
    trace = []

    entry_order = None
    exit_order = None

    orders_placed = 0
    orders_repriced = 0
    orders_filled = 0
    orders_expired = 0
    orders_unfilled = 0
    margin_interest_total = 0.0
    prev_ts = None

    def _quantize_price(p, tick):
        if tick <= 0:
            return p
        return round(p / tick) * tick

    def _quantize_qty(q, step):
        if step <= 0:
            return q
        return int(q / step) * step

    def _passes_validation(price, qty):
        if not realistic:
            return True
        return (price * qty) >= min_notional

    def _can_fill(order_qty, bar_volume):
        if not realistic:
            return True
        if bar_volume > 0 and max_fill_frac > 0 and order_qty > bar_volume * max_fill_frac:
            return False
        return True

    def _order_age_exceeded(order, ts):
        if not use_order_expiry or order is None:
            return False
        placed_ts = order.get("ts")
        if placed_ts is None:
            return False
        return (ts - placed_ts).total_seconds() > expiry_minutes * 60.0

    def _position_side(market_price: float) -> str:
        return effective_position_side_from_qty(
            inv,
            market_price=market_price,
            step_size=step_size if realistic else 0.0,
            max_position_notional=max_position_notional,
        )

    def _in_position(market_price: float) -> bool:
        return _position_side(market_price) != ""

    def _is_maker_entry(side: str, price: float, close_price: float) -> bool:
        if side == "buy":
            return price < close_price
        if side == "sell":
            return price > close_price
        return False

    def _fill_reached(side: str, price: float, *, high: float, low: float) -> bool:
        if side == "buy":
            return low <= price * (1 - fill_buf)
        if side == "sell":
            return high >= price * (1 + fill_buf)
        return False

    def _apply_buy_fill(ts, price: float, qty: float, kind: str) -> None:
        nonlocal cash, inv, entry_ts, cycle_started
        prev_side = _position_side(price)
        cash -= qty * price * (1 + fee)
        inv += qty
        cycle_started = True
        side_name = "force_buy" if kind == "force" else "buy"
        sim_trades.append({"ts": ts, "side": side_name, "price": price, "qty": qty})
        current_side = _position_side(price)
        if current_side == "":
            inv = 0.0
            entry_ts = None
        elif prev_side == "" and entry_ts is None:
            entry_ts = ts

    def _apply_sell_fill(ts, price: float, qty: float, kind: str) -> None:
        nonlocal cash, inv, entry_ts, cycle_started
        prev_side = _position_side(price)
        cash += qty * price * (1 - fee)
        inv -= qty
        cycle_started = True
        side_name = "force_sell" if kind == "force" else "sell"
        sim_trades.append({"ts": ts, "side": side_name, "price": price, "qty": qty})
        current_side = _position_side(price)
        if current_side == "":
            inv = 0.0
            entry_ts = None
        elif prev_side == "" and entry_ts is None:
            entry_ts = ts

    def _maybe_set_order(order, *, side: str, price: float, qty: float, ts, kind: str, counter_name: str):
        nonlocal orders_placed, orders_repriced
        if qty <= 0 or not _passes_validation(price, qty):
            return order
        next_order = {"side": side, "price": price, "qty": qty, "ts": ts, "kind": kind}
        if order is None:
            orders_placed += 1
            return next_order
        prev_p = float(order["price"])
        prev_q = float(order["qty"])
        prev_side = str(order.get("side", ""))
        diff = abs(prev_p - price) / max(abs(prev_p), 1e-12)
        qty_changed = abs(prev_q - qty) >= max(step_size, 1e-12)
        if prev_side != side or order.get("kind") != kind or diff > reprice_threshold or qty_changed:
            orders_repriced += 1
            return next_order
        return order

    def get_active_signal(ts_5m):
        bar_hour = ts_5m.floor("h")
        prev_hour = bar_hour - pd.Timedelta(hours=1)
        return hourly_signals.get(prev_hour), prev_hour

    for _, bar in bars_5m.iterrows():
        ts = bar["timestamp"]
        if ts < exec_start:
            continue

        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])
        volume = float(bar.get("volume", 0))

        if prev_ts is not None and margin_hourly_rate > 0:
            delta_hours = max(0.0, float((ts - prev_ts).total_seconds()) / 3600.0)
            if delta_hours > 0:
                if cash < 0:
                    interest = abs(cash) * margin_hourly_rate * delta_hours
                    cash -= interest
                    margin_interest_total += interest
                if inv < 0:
                    borrowed_value = abs(inv) * close
                    interest = borrowed_value * margin_hourly_rate * delta_hours
                    cash -= interest
                    margin_interest_total += interest
        prev_ts = ts

        if _order_age_exceeded(entry_order, ts):
            entry_order = None
            orders_expired += 1
        if _order_age_exceeded(exit_order, ts):
            exit_order = None
            orders_expired += 1

        sig, _sig_hour = get_active_signal(ts)
        long_sig = directional_signal(sig or {}, side="long")
        short_sig = directional_signal(sig or {}, side="short")
        if realistic:
            long_sig = directional_signal(
                {
                    "buy_price": _quantize_price(long_sig.entry_price, tick_size),
                    "sell_price": _quantize_price(long_sig.exit_price, tick_size),
                    "buy_amount": long_sig.entry_amount,
                    "sell_amount": long_sig.exit_amount,
                },
                side="long",
            )
            short_sig = directional_signal(
                {
                    "buy_price": _quantize_price(short_sig.exit_price, tick_size),
                    "sell_price": _quantize_price(short_sig.entry_price, tick_size),
                    "buy_amount": short_sig.exit_amount,
                    "sell_amount": short_sig.entry_amount,
                },
                side="short",
            )

        pos_side = _position_side(close)
        if pos_side == "" and exit_order is not None:
            exit_order = None

        acted_this_bar = False

        if exit_order is not None:
            exit_side = str(exit_order.get("side", ""))
            exit_price = float(exit_order["price"])
            exit_qty = float(exit_order["qty"])
            exit_kind = str(exit_order.get("kind", ""))
            if _fill_reached(exit_side, exit_price, high=high, low=low) and exit_qty > 0 and _passes_validation(exit_price, exit_qty):
                if _can_fill(exit_qty, volume):
                    if exit_side == "buy":
                        _apply_buy_fill(ts, exit_price, exit_qty, exit_kind)
                    else:
                        _apply_sell_fill(ts, exit_price, exit_qty, exit_kind)
                    orders_filled += 1
                    acted_this_bar = True
                    exit_order = None
                    if _position_side(exit_price) == "":
                        entry_order = None
                    if verbose:
                        print(f"    [{str(ts)[:16]}] {exit_side.upper()} {exit_qty:.4f} @ {exit_price:.5f} ({exit_kind})")
                else:
                    orders_unfilled += 1

        if not acted_this_bar and entry_order is not None:
            entry_side = str(entry_order.get("side", ""))
            entry_price = float(entry_order["price"])
            entry_qty = float(entry_order["qty"])
            if _fill_reached(entry_side, entry_price, high=high, low=low) and entry_qty > 0 and _passes_validation(entry_price, entry_qty):
                if _can_fill(entry_qty, volume):
                    if entry_side == "buy":
                        _apply_buy_fill(ts, entry_price, entry_qty, str(entry_order.get("kind", "")))
                    else:
                        _apply_sell_fill(ts, entry_price, entry_qty, str(entry_order.get("kind", "")))
                    orders_filled += 1
                    acted_this_bar = True
                    if verbose:
                        print(f"    [{str(ts)[:16]}] {entry_side.upper()} {entry_qty:.4f} @ {entry_price:.5f}")
                    entry_order = None
                else:
                    orders_unfilled += 1

        pos_side = _position_side(close)

        if (
            stop_after_cycle
            and cycle_started
            and not _in_position(close)
            and entry_ts is None
            and entry_order is None
            and exit_order is None
        ):
            if return_trace:
                trace.append(
                    {
                        "timestamp": ts,
                        "equity": cash + inv * close,
                        "cash": cash,
                        "inv": inv,
                        "in_position": False,
                        "entry_order_open": False,
                        "exit_order_open": False,
                    }
                )
            break

        force_close = False
        if max_hold_hours > 0 and pos_side and entry_ts is not None:
            hours_held = (ts - entry_ts).total_seconds() / 3600.0
            force_close = hours_held >= max_hold_hours
        if force_close:
            if pos_side == "long":
                force_price = _quantize_price(close * 0.999, tick_size) if realistic else close * 0.999
                force_side = "sell"
            else:
                force_price = _quantize_price(close * 1.001, tick_size) if realistic else close * 1.001
                force_side = "buy"
            force_qty = _quantize_qty(abs(inv), step_size) if realistic else abs(inv)
            if force_qty > 0 and _passes_validation(force_price, force_qty):
                exit_order = _maybe_set_order(
                    exit_order,
                    side=force_side,
                    price=force_price,
                    qty=force_qty,
                    ts=ts,
                    kind="force",
                    counter_name="exit",
                )
            entry_order = None
            if return_trace:
                trace.append(
                    {
                        "timestamp": ts,
                        "equity": cash + inv * close,
                        "cash": cash,
                        "inv": inv,
                        "in_position": _in_position(close),
                        "entry_order_open": False,
                        "exit_order_open": exit_order is not None,
                    }
                )
            continue

        if sig is not None and pos_side == "long" and long_sig.exit_amount > 0 and long_sig.exit_price > 0:
            target_qty = min((long_sig.exit_amount / 100.0) * abs(inv), abs(inv))
            target_qty = _quantize_qty(target_qty, step_size) if realistic else target_qty
            if target_qty > 0 and _is_maker_entry("sell", long_sig.exit_price, close):
                exit_order = _maybe_set_order(
                    exit_order,
                    side="sell",
                    price=long_sig.exit_price,
                    qty=target_qty,
                    ts=ts,
                    kind="signal",
                    counter_name="exit",
                )

        if sig is not None and pos_side == "short" and short_sig.exit_amount > 0 and short_sig.exit_price > 0:
            target_qty = min((short_sig.exit_amount / 100.0) * abs(inv), abs(inv))
            target_qty = _quantize_qty(target_qty, step_size) if realistic else target_qty
            if target_qty > 0 and _is_maker_entry("buy", short_sig.exit_price, close):
                exit_order = _maybe_set_order(
                    exit_order,
                    side="buy",
                    price=short_sig.exit_price,
                    qty=target_qty,
                    ts=ts,
                    kind="signal",
                    counter_name="exit",
                )

        if sig is not None:
            candidate_side = ""
            candidate_sig = None
            leverage_side = ""
            kind = "entry"
            if pos_side == "long":
                candidate_side = "buy"
                candidate_sig = long_sig
                leverage_side = "long"
                kind = "add"
            elif pos_side == "short":
                candidate_side = "sell"
                candidate_sig = short_sig
                leverage_side = "short"
                kind = "add"
            else:
                flat_entry_side = choose_flat_entry_side(sig, allow_short=allow_short)
                if flat_entry_side == "long":
                    candidate_side = "buy"
                    candidate_sig = long_sig
                    leverage_side = "long"
                elif flat_entry_side == "short":
                    candidate_side = "sell"
                    candidate_sig = short_sig
                    leverage_side = "short"

            if candidate_side and candidate_sig is not None and candidate_sig.entry_price > 0 and candidate_sig.entry_amount > 0:
                equity = cash + inv * close
                max_entry_value = remaining_entry_notional(
                    side=leverage_side,
                    equity=equity,
                    current_qty=inv,
                    market_price=close,
                    long_max_leverage=long_max_leverage,
                    short_max_leverage=short_max_leverage,
                )
                max_entry_value = _cap_directional_entry_notional(
                    target_notional=max_entry_value,
                    side=leverage_side,
                    current_qty=inv,
                    market_price=close,
                    max_position_notional=max_position_notional,
                )
                if max_entry_value > 0:
                    amount_frac = candidate_sig.entry_amount / 100.0
                    target_qty = (
                        max_entry_value / (candidate_sig.entry_price * (1 + fee))
                        if realistic
                        else amount_frac * max_entry_value / (candidate_sig.entry_price * (1 + fee))
                    )
                    target_qty = _quantize_qty(target_qty, step_size) if realistic else target_qty
                    if target_qty > 0 and _is_maker_entry(candidate_side, candidate_sig.entry_price, close):
                        entry_order = _maybe_set_order(
                            entry_order,
                            side=candidate_side,
                            price=candidate_sig.entry_price,
                            qty=target_qty,
                            ts=ts,
                            kind=kind,
                            counter_name="entry",
                        )

        if return_trace:
            trace.append(
                {
                    "timestamp": ts,
                    "equity": cash + inv * close,
                    "cash": cash,
                    "inv": inv,
                    "in_position": _in_position(close),
                    "entry_order_open": entry_order is not None,
                    "exit_order_open": exit_order is not None,
                }
            )

        if (
            stop_after_cycle
            and cycle_started
            and not _in_position(close)
            and entry_ts is None
            and entry_order is None
            and exit_order is None
        ):
            break

    last_close = float(bars_5m.iloc[-1]["close"]) if len(bars_5m) > 0 else 0.0
    final_eq = cash + inv * last_close

    print(
        "  [live-like] "
        f"orders_filled={orders_filled} orders_placed={orders_placed} "
        f"orders_repriced={orders_repriced} orders_expired={orders_expired} "
        f"orders_unfilled={orders_unfilled} margin_interest=${margin_interest_total:.4f}"
    )

    if return_trace:
        return sim_trades, final_eq, cash, inv, pd.DataFrame(trace)
    return sim_trades, final_eq, cash, inv


def simulate_5m(args, hourly_signals, bars_5m, initial_inv=0.0, initial_entry_ts=None):
    if bool(getattr(args, "realistic", False)) and bool(getattr(args, "live_like", True)):
        return _simulate_5m_live_like(
            args,
            hourly_signals,
            bars_5m,
            initial_inv=initial_inv,
            initial_entry_ts=initial_entry_ts,
        )
    return _simulate_5m_legacy(
        args,
        hourly_signals,
        bars_5m,
        initial_inv=initial_inv,
        initial_entry_ts=initial_entry_ts,
    )


def simulate_5m_with_trace(
    args,
    hourly_signals,
    bars_5m,
    initial_inv=0.0,
    initial_entry_ts=None,
    *,
    stop_after_cycle: bool = False,
):
    run_args = argparse.Namespace(**vars(args))
    run_args.return_trace = True
    run_args.stop_after_cycle = bool(stop_after_cycle)
    return simulate_5m(
        run_args,
        hourly_signals,
        bars_5m,
        initial_inv=initial_inv,
        initial_entry_ts=initial_entry_ts,
    )


def match_trades(prod_fills, sim_trades):
    if prod_fills.empty:
        return [], pd.DataFrame(sim_trades) if sim_trades else pd.DataFrame()
    if not sim_trades:
        return [
            {
                "prod_ts": prod["timestamp"],
                "side": prod["side"],
                "prod_price": prod["avg_price"],
                "prod_qty": prod["total_qty"],
                "matched": False,
            }
            for _, prod in prod_fills.iterrows()
        ], pd.DataFrame()

    def _normalize_side(side: str) -> str:
        s = str(side).lower()
        if s == "force_sell":
            return "sell"
        if s == "force_buy":
            return "buy"
        return s

    sim_df = pd.DataFrame(sim_trades)
    sim_df["_match_side"] = sim_df["side"].map(_normalize_side)
    sim_df["_matched"] = False

    matches = []
    for _, prod in prod_fills.iterrows():
        pt = prod["timestamp"]
        prod_side = _normalize_side(prod["side"])
        # match within 30 min window, same direction
        cands = sim_df[
            (sim_df["_match_side"] == prod_side) &
            (~sim_df["_matched"]) &
            (abs((sim_df["ts"] - pt).dt.total_seconds()) <= 1800)
        ]
        if len(cands) > 0:
            # pick closest in time
            diffs = abs((cands["ts"] - pt).dt.total_seconds())
            best_idx = diffs.idxmin()
            best = sim_df.loc[best_idx]
            sim_df.at[best_idx, "_matched"] = True
            diff_bps = (best["price"] - prod["avg_price"]) / prod["avg_price"] * 10000
            matches.append({
                "prod_ts": prod["timestamp"], "sim_ts": best["ts"],
                "side": prod["side"],
                "prod_price": prod["avg_price"], "sim_price": best["price"],
                "diff_bps": diff_bps,
                "prod_qty": prod["total_qty"], "sim_qty": best["qty"],
                "matched": True,
            })
        else:
            matches.append({
                "prod_ts": prod["timestamp"], "side": prod["side"],
                "prod_price": prod["avg_price"], "prod_qty": prod["total_qty"],
                "matched": False,
            })

    unmatched_sim = sim_df[~sim_df["_matched"]]
    return matches, unmatched_sim


def slippage_analysis(prod_fills, orders_df, fill_buf):
    if prod_fills.empty or orders_df.empty:
        return
    merged = prod_fills.merge(orders_df[["order_id", "limit_price"]], on="order_id", how="left")
    merged["slippage_bps"] = (merged["avg_price"] - merged["limit_price"]) / merged["limit_price"] * 10000
    print("\n" + "=" * 70)
    print("SLIPPAGE ANALYSIS (fill_price vs limit_price)")
    print("=" * 70)
    for _, r in merged.iterrows():
        d = "+" if r["slippage_bps"] > 0 else ""
        print(f"  {str(r['timestamp'])[:16]} {r['side']:>4s} limit={r['limit_price']:.5f} "
              f"fill={r['avg_price']:.5f} slip={d}{r['slippage_bps']:.1f}bps")
    buys = merged[merged["side"] == "buy"]["slippage_bps"]
    sells = merged[merged["side"] == "sell"]["slippage_bps"]
    if len(buys) > 0:
        print(f"\n  Buy slippage:  mean={buys.mean():.1f}bps std={buys.std():.1f}bps")
    if len(sells) > 0:
        print(f"  Sell slippage: mean={sells.mean():.1f}bps std={sells.std():.1f}bps")
    all_slip = merged["slippage_bps"].abs()
    print(f"  Overall |slip|: mean={all_slip.mean():.1f}bps  (fill_buffer={fill_buf*10000:.0f}bps)")


def print_report(prod_fills, sim_trades, matches, unmatched_sim, prod_orders,
                 final_eq, initial_cash, fill_buf):
    print("=" * 70)
    print("SIM vs LIVE VALIDATION REPORT (5m execution)")
    print("=" * 70)

    print(f"\nPRODUCTION FILLS ({len(prod_fills)}):")
    if not prod_fills.empty:
        for _, r in prod_fills.iterrows():
            comm_str = f"comm={r['total_commission']:.4f} {r['commission_asset']}"
            print(f"  {str(r['timestamp'])[:16]} {r['side']:>4s} {r['total_qty']:>10.0f} "
                  f"@ {r['avg_price']:.5f}  ({comm_str})")

    print(f"\nSIMULATED TRADES ({len(sim_trades)}):")
    for t in sim_trades:
        print(f"  {str(t['ts'])[:16]} {t['side']:>4s} {t['qty']:>10.0f} @ {t['price']:.5f}")

    print(f"\nTRADE-BY-TRADE COMPARISON:")
    n_matched = sum(1 for m in matches if m["matched"])
    for m in matches:
        if m["matched"]:
            d = "+" if m["diff_bps"] > 0 else ""
            tdiff = abs((m["sim_ts"] - m["prod_ts"]).total_seconds() / 60)
            print(f"  {str(m['prod_ts'])[:16]} {m['side']:>4s} "
                  f"prod={m['prod_price']:.5f} sim={m['sim_price']:.5f} "
                  f"({d}{m['diff_bps']:.1f}bps, {tdiff:.0f}min apart)")
        else:
            print(f"  {str(m['prod_ts'])[:16]} {m['side']:>4s} "
                  f"prod={m['prod_price']:.5f} -> NO SIM MATCH")

    if len(unmatched_sim) > 0:
        print(f"\n  Sim-only trades ({len(unmatched_sim)}):")
        for _, s in unmatched_sim.iterrows():
            print(f"    {str(s['ts'])[:16]} {s['side']:>4s} {s['qty']:.0f} @ {s['price']:.5f}")

    print(f"\nACCURACY METRICS:")
    n_prod = len(prod_fills)
    match_rate = n_matched / max(n_prod, 1) * 100
    price_diffs = [m["diff_bps"] for m in matches if m["matched"]]
    avg_diff = np.mean(price_diffs) if price_diffs else 0
    print(f"  Fill direction match: {n_matched}/{n_prod} ({match_rate:.0f}%)")
    print(f"  Avg price diff:       {avg_diff:+.1f} bps")
    print(f"  Sim final equity:     ${final_eq:.2f} ({(final_eq/initial_cash - 1)*100:+.2f}%)")

    slippage_analysis(prod_fills, prod_orders, fill_buf)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="DOGEUSDT")
    p.add_argument("--data-symbol", default="DOGEUSD")
    p.add_argument("--checkpoint", default=str(REPO / "binanceleveragesui/checkpoints/DOGEUSD_rw30_ep4_full.pt"))
    p.add_argument("--start", default=(datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%d"))
    p.add_argument("--end", default=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"))
    p.add_argument("--intensity-scale", type=float, default=5.0)
    p.add_argument("--max-hold-hours", type=int, default=6)
    p.add_argument("--fill-buffer-pct", type=float, default=0.0)
    p.add_argument("--fee", type=float, default=0.001)
    p.add_argument("--max-leverage", type=float, default=1.0)
    p.add_argument("--initial-cash", type=float, default=3754.0)
    p.add_argument(
        "--initial-inv",
        type=float,
        default=None,
        help="Override reconstructed starting inventory for focused replay windows.",
    )
    p.add_argument(
        "--initial-entry-ts",
        default=None,
        help="Optional ISO timestamp for the overridden starting inventory.",
    )
    p.add_argument(
        "--skip-initial-reconstruction",
        action="store_true",
        help="Start the replay flat instead of reconstructing a pre-window live inventory.",
    )
    p.add_argument(
        "--initial-reconstruction-lookback-hours",
        type=int,
        default=48,
        help="Lookback horizon for pre-window live inventory reconstruction.",
    )
    p.add_argument("--debug", action="store_true")
    p.add_argument("--sequence-length", type=int, default=72)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--data-root", default=str(REPO / "trainingdatahourlybinance"))
    p.add_argument("--forecast-cache", default=str(REPO / "binanceneural/forecast_cache"))
    p.add_argument("--bars-5m-root", default=None)
    p.add_argument("--realistic", action="store_true", help="Enable realistic fill model (order expiry, volume filter, price validation)")
    p.add_argument("--expiry-minutes", type=int, default=90, help="Order expiry in minutes (realistic mode)")
    p.add_argument("--max-fill-fraction", type=float, default=0.01, help="Max fraction of bar volume fillable (realistic mode)")
    p.add_argument("--min-notional", type=float, default=5.0, help="Min order notional in USDT (realistic mode)")
    p.add_argument("--tick-size", type=float, default=0.00001, help="Price tick size (realistic mode)")
    p.add_argument("--step-size", type=float, default=1.0, help="Qty step size (realistic mode)")
    p.add_argument("--allow-short", action="store_true", help="Enable short-side replay to mirror live margin trading.")
    p.add_argument("--max-long-leverage", type=float, default=None, help="Override long leverage cap for replay.")
    p.add_argument("--max-short-leverage", type=float, default=None, help="Override short leverage cap for replay.")
    p.add_argument(
        "--max-position-notional",
        type=float,
        default=None,
        help="Hard cap on gross symbol notional to mirror live probe sizing.",
    )
    p.add_argument(
        "--live-like",
        dest="live_like",
        action="store_true",
        default=True,
        help="Use live-like persistent order model when --realistic is enabled (default).",
    )
    p.add_argument(
        "--legacy-realistic",
        dest="live_like",
        action="store_false",
        help="Use legacy per-hour reset model for --realistic mode.",
    )
    p.add_argument(
        "--reprice-threshold",
        type=float,
        default=0.003,
        help="Relative threshold for replacing working limit prices in live-like mode.",
    )
    p.add_argument(
        "--use-order-expiry",
        action="store_true",
        help="Enable explicit simulated order expiry in live-like mode.",
    )
    p.add_argument(
        "--margin-hourly-rate",
        type=float,
        default=0.0000025457,
        help="Hourly margin interest rate applied to borrowed exposure in simulator.",
    )
    p.add_argument("--verbose", action="store_true", help="Print bar-by-bar decisions")
    args = p.parse_args()

    torch.use_deterministic_algorithms(True)
    set_seeds(42)

    print(f"Loading checkpoint: {args.checkpoint}")
    model, normalizer, feature_columns, meta = load_policy_checkpoint_compat(
        args.checkpoint,
        device="cuda",
        data_root=args.data_root,
        forecast_cache_root=args.forecast_cache,
    )
    seq_len = meta.get("sequence_length", args.sequence_length)
    forecast_horizons = resolve_required_forecast_horizons(
        (int(args.horizon),),
        feature_columns=feature_columns,
        fallback_horizons=(int(args.horizon),),
    )
    if forecast_horizons != (int(args.horizon),):
        print(
            f"Expanded forecast horizons from {(int(args.horizon),)} to {forecast_horizons} "
            f"based on checkpoint features"
        )

    print(f"Loading hourly data for {args.data_symbol}...")
    dm = ChronosSolDataModule(
        symbol=args.data_symbol,
        data_root=Path(args.data_root),
        forecast_cache_root=Path(args.forecast_cache),
        forecast_horizons=forecast_horizons, context_hours=512,
        quantile_levels=(0.1, 0.5, 0.9),
        batch_size=32, model_id="amazon/chronos-t5-small",
        sequence_length=seq_len,
        split_config=SplitConfig(val_days=30, test_days=30),
        cache_only=True, max_history_days=365,
    )
    frame = dm.full_frame.copy()
    print(f"Hourly frame: {len(frame)} bars, {frame['timestamp'].min()} to {frame['timestamp'].max()}")

    start_ts = pd.Timestamp(args.start, tz="UTC")
    end_ts = pd.Timestamp(args.end, tz="UTC")
    start_ms = int(start_ts.timestamp() * 1000)
    end_ms = int(end_ts.timestamp() * 1000)

    print(f"\nLoading 5m bars for {args.symbol}...")
    bars_5m = load_5m_bars(
        args.symbol,
        start_ts - pd.Timedelta(hours=1),
        end_ts,
        data_root=args.data_root,
        bars_5m_root=args.bars_5m_root,
    )
    print(f"5m bars: {len(bars_5m)} ({bars_5m['timestamp'].min()} to {bars_5m['timestamp'].max()})")

    if args.initial_inv is not None:
        print(f"\nUsing explicit initial position override at {start_ts}...")
    elif args.skip_initial_reconstruction:
        print(f"\nSkipping initial position reconstruction at {start_ts} (starting flat)...")
    else:
        print(f"\nReconstructing initial position at {start_ts}...")
    initial_inv, initial_entry_ts = resolve_initial_replay_state(args, start_ms)

    print(f"\nPulling production fills: {args.symbol} {start_ts} to {end_ts}")
    prod_fills, raw_trades, prod_orders = pull_prod_fills(args.symbol, start_ms, end_ms)
    print(f"Got {len(prod_fills)} aggregated fills from {len(raw_trades)} raw trades")

    print(f"\nGenerating hourly signals (deterministic)...")
    hourly_signals = generate_hourly_signals(args, frame, model, normalizer, feature_columns, meta)
    print(f"Generated {len(hourly_signals)} hourly signals")

    if args.debug:
        print(f"\nHourly signals (active during next hour due to lag=1):")
        for ts in sorted(hourly_signals.keys()):
            s = hourly_signals[ts]
            if s["buy_price"] > 0 or s["sell_price"] > 0:
                print(f"  {str(ts)[:16]} -> applied {str(ts + pd.Timedelta(hours=1))[:16]}: "
                      f"bp={s['buy_price']:.5f} sp={s['sell_price']:.5f} "
                      f"ba={s['buy_amount']:.1f}% sa={s['sell_amount']:.1f}%")

    # Derive starting cash: total equity = cash + position_value
    if initial_inv > 0 and len(bars_5m) > 0:
        first_close = float(bars_5m[bars_5m["timestamp"] >= pd.Timestamp(args.start, tz="UTC")].iloc[0]["close"])
        starting_cash = args.initial_cash - initial_inv * first_close
        print(f"  Account equity=${args.initial_cash:.2f}, position={initial_inv:.0f}*{first_close:.5f}=${initial_inv*first_close:.2f}")
        print(f"  Derived cash: ${starting_cash:.2f}")
        args.initial_cash = starting_cash

    print(f"\nSimulating on 5m bars (hourly signals, 5m execution, lag=1)...")
    sim_trades, final_eq, cash, inv = simulate_5m(
        args, hourly_signals, bars_5m,
        initial_inv=initial_inv, initial_entry_ts=initial_entry_ts,
    )
    starting_eq = args.initial_cash + initial_inv * float(bars_5m.iloc[0]["close"]) if len(bars_5m) > 0 else args.initial_cash
    print(f"Sim: {len(sim_trades)} trades, equity=${final_eq:.2f}, position={inv:.0f}")
    print(f"Starting equity: ${starting_eq:.2f} (cash=${args.initial_cash:.2f} + {initial_inv:.0f} units)")

    matches, unmatched_sim = match_trades(prod_fills, sim_trades)
    print_report(prod_fills, sim_trades, matches, unmatched_sim, prod_orders,
                 final_eq, starting_eq, args.fill_buffer_pct)


if __name__ == "__main__":
    main()
