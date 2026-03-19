from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from src.margin_position_utils import (
    choose_flat_entry_side,
    directional_signal,
    position_side_from_qty,
    remaining_entry_notional,
)


@dataclass(frozen=True)
class CurrentPriceLockedConfig:
    name: str
    fee: float = 0.001
    spread_bps: float = 4.0
    slippage_bps: float = 2.0
    min_expected_edge_bps: float = 12.0
    min_profit_exit_bps: float = 4.0
    lock_minutes: int = 60
    cooldown_minutes_after_exit: int = 60
    max_hold_hours: float = 6.0
    allow_short: bool = False
    long_max_leverage: float = 1.0
    short_max_leverage: float = 0.0
    min_notional: float = 5.0
    step_size: float = 0.0
    max_position_notional: float | None = None


def _to_utc_timestamp(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _quantize_qty(qty: float, step_size: float) -> float:
    target = max(0.0, float(qty))
    step = max(0.0, float(step_size))
    if step <= 0.0:
        return target
    return int(target / step) * step


def _execution_price(market_price: float, *, action: str, config: CurrentPriceLockedConfig) -> float:
    price = max(0.0, float(market_price))
    if price <= 0.0:
        return 0.0
    half_spread = max(0.0, float(config.spread_bps)) / 20_000.0
    slippage = max(0.0, float(config.slippage_bps)) / 10_000.0
    if action == "buy":
        return price * (1.0 + half_spread + slippage)
    if action == "sell":
        return price * (1.0 - half_spread - slippage)
    raise ValueError(f"Unsupported action {action!r}; expected 'buy' or 'sell'.")


def _capped_entry_notional(
    target_notional: float,
    *,
    current_qty: float,
    market_price: float,
    max_position_notional: float | None,
) -> float:
    capped = max(0.0, float(target_notional))
    if max_position_notional is None:
        return capped
    remaining = max(0.0, float(max_position_notional) - abs(float(current_qty)) * max(0.0, float(market_price)))
    return min(capped, remaining)


def _long_entry_edge_bps(signal: Mapping[str, float], market_price: float, config: CurrentPriceLockedConfig) -> float:
    long_sig = directional_signal(signal, side="long")
    if long_sig.entry_amount <= 0.0 or long_sig.entry_price <= 0.0 or long_sig.exit_price <= 0.0:
        return float("-inf")
    buy_exec = _execution_price(market_price, action="buy", config=config)
    sell_exec = _execution_price(long_sig.exit_price, action="sell", config=config)
    if buy_exec <= 0.0 or sell_exec <= 0.0:
        return float("-inf")
    net_return = (sell_exec * (1.0 - config.fee)) / (buy_exec * (1.0 + config.fee)) - 1.0
    return net_return * 10_000.0


def _short_entry_edge_bps(signal: Mapping[str, float], market_price: float, config: CurrentPriceLockedConfig) -> float:
    short_sig = directional_signal(signal, side="short")
    if short_sig.entry_amount <= 0.0 or short_sig.entry_price <= 0.0 or short_sig.exit_price <= 0.0:
        return float("-inf")
    sell_exec = _execution_price(market_price, action="sell", config=config)
    buy_exec = _execution_price(short_sig.exit_price, action="buy", config=config)
    if sell_exec <= 0.0 or buy_exec <= 0.0:
        return float("-inf")
    net_return = (sell_exec * (1.0 - config.fee)) / (buy_exec * (1.0 + config.fee)) - 1.0
    return net_return * 10_000.0


def _exit_is_profitable(
    *,
    side: str,
    entry_exec_price: float,
    market_price: float,
    config: CurrentPriceLockedConfig,
) -> bool:
    min_profit = max(0.0, float(config.min_profit_exit_bps)) / 10_000.0
    if side == "long":
        sell_exec = _execution_price(market_price, action="sell", config=config)
        if sell_exec <= 0.0 or entry_exec_price <= 0.0:
            return False
        net_multiple = (sell_exec * (1.0 - config.fee)) / (entry_exec_price * (1.0 + config.fee))
        return net_multiple >= (1.0 + min_profit)
    if side == "short":
        buy_exec = _execution_price(market_price, action="buy", config=config)
        if buy_exec <= 0.0 or entry_exec_price <= 0.0:
            return False
        net_multiple = (entry_exec_price * (1.0 - config.fee)) / (buy_exec * (1.0 + config.fee))
        return net_multiple >= (1.0 + min_profit)
    raise ValueError(f"Unsupported side {side!r}; expected 'long' or 'short'.")


def simulate_current_price_locked(
    config: CurrentPriceLockedConfig,
    hourly_signals: Mapping[pd.Timestamp, Mapping[str, float]],
    bars_5m: pd.DataFrame,
    *,
    start_ts,
    initial_cash: float,
    initial_qty: float = 0.0,
    initial_entry_price: float | None = None,
    initial_entry_ts=None,
    signal_schedule=None,
):
    start_ts = _to_utc_timestamp(start_ts)
    frame = bars_5m.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if frame.empty:
        trace = pd.DataFrame(columns=["timestamp", "equity", "cash", "qty", "position_side"])
        return [], float(initial_cash), float(initial_cash), float(initial_qty), trace, {
            "blocked_loss_exit_count": 0,
            "blocked_reentry_count": 0,
        }

    cash = float(initial_cash)
    qty = float(initial_qty)
    first_price = max(0.0, float(frame.iloc[0]["close"]))
    entry_price = float(initial_entry_price if initial_entry_price is not None else first_price)
    entry_ts = None if initial_entry_ts is None else _to_utc_timestamp(initial_entry_ts)
    lock_until = (
        None
        if entry_ts is None or abs(qty) <= 1e-12
        else entry_ts + pd.Timedelta(minutes=int(config.lock_minutes))
    )
    next_entry_allowed_ts = (
        None
        if entry_ts is None or abs(qty) <= 1e-12
        else entry_ts + pd.Timedelta(minutes=int(config.cooldown_minutes_after_exit))
    )
    blocked_loss_exit_count = 0
    blocked_reentry_count = 0
    trades: list[dict] = []
    trace_rows: list[dict] = []
    schedule_rows: list[dict] = []
    schedule_idx = 0
    active_scheduled_signal: dict[str, float] | None = None

    if signal_schedule is not None:
        schedule_frame = signal_schedule.copy() if isinstance(signal_schedule, pd.DataFrame) else pd.DataFrame(signal_schedule)
        if not schedule_frame.empty:
            schedule_frame["effective_ts"] = pd.to_datetime(schedule_frame["effective_ts"], utc=True, errors="coerce")
            schedule_frame = schedule_frame.dropna(subset=["effective_ts"]).sort_values("effective_ts").reset_index(drop=True)
            schedule_rows = schedule_frame.to_dict(orient="records")

    for _, bar in frame.iterrows():
        ts = bar["timestamp"]
        if ts < start_ts:
            continue
        close = max(0.0, float(bar["close"]))
        if close <= 0.0:
            continue

        while schedule_idx < len(schedule_rows) and pd.Timestamp(schedule_rows[schedule_idx]["effective_ts"]) <= ts:
            active_scheduled_signal = schedule_rows[schedule_idx]
            schedule_idx += 1
        if active_scheduled_signal is not None:
            signal = active_scheduled_signal
        else:
            signal_hour = ts.floor("h") - pd.Timedelta(hours=1)
            signal = hourly_signals.get(signal_hour, {})
        long_sig = directional_signal(signal, side="long")
        short_sig = directional_signal(signal, side="short")
        position_side = position_side_from_qty(qty, step_size=float(config.step_size))
        lock_active = lock_until is not None and ts < lock_until
        closed_position_this_bar = False
        held_hours = 0.0
        if entry_ts is not None:
            held_hours = max(0.0, float((ts - entry_ts).total_seconds()) / 3600.0)

        if position_side == "long":
            signal_exit = long_sig.exit_price > 0.0 and close >= long_sig.exit_price
            timed_exit = config.max_hold_hours > 0.0 and held_hours >= float(config.max_hold_hours)
            if signal_exit or timed_exit:
                profitable = _exit_is_profitable(
                    side="long",
                    entry_exec_price=entry_price,
                    market_price=close,
                    config=config,
                )
                if profitable or not lock_active:
                    sell_exec = _execution_price(close, action="sell", config=config)
                    sell_qty = abs(qty)
                    cash += sell_qty * sell_exec * (1.0 - config.fee)
                    trades.append(
                        {
                            "ts": ts,
                            "side": "sell",
                            "qty": sell_qty,
                            "price": sell_exec,
                            "reason": "signal_exit" if signal_exit else "timed_exit",
                        }
                    )
                    qty = 0.0
                    entry_price = 0.0
                    entry_ts = None
                    lock_until = None
                    next_entry_allowed_ts = ts + pd.Timedelta(minutes=int(config.cooldown_minutes_after_exit))
                    closed_position_this_bar = True
                else:
                    blocked_loss_exit_count += 1

        elif position_side == "short":
            signal_exit = short_sig.exit_price > 0.0 and close <= short_sig.exit_price
            timed_exit = config.max_hold_hours > 0.0 and held_hours >= float(config.max_hold_hours)
            if signal_exit or timed_exit:
                profitable = _exit_is_profitable(
                    side="short",
                    entry_exec_price=entry_price,
                    market_price=close,
                    config=config,
                )
                if profitable or not lock_active:
                    buy_exec = _execution_price(close, action="buy", config=config)
                    cover_qty = abs(qty)
                    cash -= cover_qty * buy_exec * (1.0 + config.fee)
                    trades.append(
                        {
                            "ts": ts,
                            "side": "buy",
                            "qty": cover_qty,
                            "price": buy_exec,
                            "reason": "signal_exit" if signal_exit else "timed_exit",
                        }
                    )
                    qty = 0.0
                    entry_price = 0.0
                    entry_ts = None
                    lock_until = None
                    next_entry_allowed_ts = ts + pd.Timedelta(minutes=int(config.cooldown_minutes_after_exit))
                    closed_position_this_bar = True
                else:
                    blocked_loss_exit_count += 1

        position_side = position_side_from_qty(qty, step_size=float(config.step_size))
        if position_side == "" and not closed_position_this_bar:
            long_edge_bps = _long_entry_edge_bps(signal, close, config)
            short_edge_bps = _short_entry_edge_bps(signal, close, config) if config.allow_short else float("-inf")
            long_ready = (
                long_sig.entry_amount > 0.0
                and long_sig.entry_price > 0.0
                and close <= long_sig.entry_price
                and long_edge_bps >= float(config.min_expected_edge_bps)
            )
            short_ready = (
                config.allow_short
                and short_sig.entry_amount > 0.0
                and short_sig.entry_price > 0.0
                and close >= short_sig.entry_price
                and short_edge_bps >= float(config.min_expected_edge_bps)
            )
            desired_side = ""
            if long_ready and short_ready:
                desired_side = "long" if long_edge_bps >= short_edge_bps else "short"
            elif long_ready:
                desired_side = "long"
            elif short_ready:
                desired_side = "short"
            else:
                desired_side = choose_flat_entry_side(signal, allow_short=bool(config.allow_short))
                if desired_side == "long" and not long_ready:
                    desired_side = ""
                if desired_side == "short" and not short_ready:
                    desired_side = ""

            if desired_side:
                if next_entry_allowed_ts is not None and ts < next_entry_allowed_ts:
                    blocked_reentry_count += 1
                else:
                    equity = cash + qty * close
                    max_entry_notional = remaining_entry_notional(
                        side=desired_side,
                        equity=equity,
                        current_qty=qty,
                        market_price=close,
                        long_max_leverage=float(config.long_max_leverage),
                        short_max_leverage=float(config.short_max_leverage),
                    )
                    max_entry_notional = _capped_entry_notional(
                        max_entry_notional,
                        current_qty=qty,
                        market_price=close,
                        max_position_notional=config.max_position_notional,
                    )
                    active_signal = long_sig if desired_side == "long" else short_sig
                    target_notional = max_entry_notional * max(0.0, float(active_signal.entry_amount)) / 100.0
                    action = "buy" if desired_side == "long" else "sell"
                    exec_price = _execution_price(close, action=action, config=config)
                    if exec_price > 0.0:
                        target_qty = _quantize_qty(target_notional / exec_price, float(config.step_size))
                        if (target_qty * exec_price) >= max(0.0, float(config.min_notional)):
                            if desired_side == "long":
                                cash -= target_qty * exec_price * (1.0 + config.fee)
                                qty += target_qty
                            else:
                                cash += target_qty * exec_price * (1.0 - config.fee)
                                qty -= target_qty
                            entry_price = exec_price
                            entry_ts = ts
                            lock_until = ts + pd.Timedelta(minutes=int(config.lock_minutes))
                            next_entry_allowed_ts = None
                            trades.append(
                                {
                                    "ts": ts,
                                    "side": action,
                                    "qty": target_qty,
                                    "price": exec_price,
                                    "reason": f"{desired_side}_entry",
                                }
                            )

        trace_rows.append(
            {
                "timestamp": ts,
                "equity": cash + qty * close,
                "cash": cash,
                "qty": qty,
                "position_side": position_side_from_qty(qty, step_size=float(config.step_size)),
            }
        )

    last_close = max(0.0, float(frame.iloc[-1]["close"]))
    final_eq = cash + qty * last_close
    trace = pd.DataFrame(trace_rows, columns=["timestamp", "equity", "cash", "qty", "position_side"])
    return trades, float(final_eq), float(cash), float(qty), trace, {
        "blocked_loss_exit_count": int(blocked_loss_exit_count),
        "blocked_reentry_count": int(blocked_reentry_count),
    }


__all__ = [
    "CurrentPriceLockedConfig",
    "simulate_current_price_locked",
]
