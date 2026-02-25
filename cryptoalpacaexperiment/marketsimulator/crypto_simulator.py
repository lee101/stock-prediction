"""Crypto portfolio simulator -- trades only when US stock market is CLOSED.

Inverse of the stock simulator: skips NYSE open hours + buffer,
force-closes positions before market opens.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from typing import Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.date_utils import is_nyse_open_on_date
from src.fees import get_fee_for_symbol
from src.metrics_utils import annualized_sortino

from unified_hourly_experiment.marketsimulator.unified_selector import (
    _edge_score,
    UnifiedTradeRecord,
)

NEW_YORK = ZoneInfo("America/New_York")
BUFFER_OPEN = time(8, 30)
BUFFER_CLOSE = time(17, 0)
BACKOUT_START = time(8, 0)


def _is_crypto_allowed(ts: pd.Timestamp) -> bool:
    """True when crypto trading is allowed (US market CLOSED + buffer)."""
    try:
        ny = ts.tz_convert(NEW_YORK)
    except TypeError:
        ny = ts.tz_localize("UTC").tz_convert(NEW_YORK)
    if not is_nyse_open_on_date(ny):
        return True
    t = ny.time()
    return not (BUFFER_OPEN <= t < BUFFER_CLOSE)


def _is_backout_window(ts: pd.Timestamp) -> bool:
    """True during the 30min window before buffer (force-close time)."""
    try:
        ny = ts.tz_convert(NEW_YORK)
    except TypeError:
        ny = ts.tz_localize("UTC").tz_convert(NEW_YORK)
    if not is_nyse_open_on_date(ny):
        return False
    t = ny.time()
    return BACKOUT_START <= t < BUFFER_OPEN


@dataclass
class CryptoPortfolioConfig:
    initial_cash: float = 10_000.0
    max_positions: int = 2
    min_edge: float = 0.0
    edge_mode: str = "high_low"
    max_hold_hours: int = 6
    skip_during_market_hours: bool = True
    backout_before_market: bool = True
    symbols: Optional[Sequence[str]] = None
    trade_amount_scale: float = 100.0
    min_buy_amount: float = 0.0
    fee_by_symbol: Optional[Dict[str, float]] = None
    max_leverage: float = 1.0
    decision_lag_bars: int = 1
    bar_margin: float = 0.0005
    long_only_symbols: Optional[set] = None
    short_only_symbols: Optional[set] = None
    force_close_slippage: float = 0.001
    int_qty: bool = False


@dataclass
class Position:
    symbol: str
    qty: float
    entry_price: float
    entry_ts: pd.Timestamp
    sell_target: Optional[float] = None
    hours_held: int = 0
    direction: str = "long"


@dataclass
class CryptoPortfolioResult:
    equity_curve: pd.Series
    trades: List[UnifiedTradeRecord]
    metrics: Dict[str, float] = field(default_factory=dict)


def run_crypto_simulation(
    bars: pd.DataFrame,
    actions: pd.DataFrame,
    config: Optional[CryptoPortfolioConfig] = None,
    *,
    horizon: int = 1,
) -> CryptoPortfolioResult:
    cfg = config or CryptoPortfolioConfig()

    if cfg.decision_lag_bars > 0:
        lag = cfg.decision_lag_bars
        shifted_parts = []
        for sym in actions["symbol"].unique():
            sym_acts = actions[actions["symbol"] == sym].sort_values("timestamp").copy()
            sym_bars = bars[bars["symbol"] == sym].sort_values("timestamp")
            bar_ts = sym_bars["timestamp"].tolist()
            if len(sym_acts) > lag and len(bar_ts) > lag:
                sym_acts = sym_acts.iloc[:-lag].copy()
                sym_acts["timestamp"] = bar_ts[lag:lag + len(sym_acts)]
            shifted_parts.append(sym_acts)
        actions = pd.concat(shifted_parts, ignore_index=True)

    merged = bars.merge(actions, on=["timestamp", "symbol"], how="inner", suffixes=("", "_act"))
    if merged.empty:
        raise ValueError("No matching bars and actions")
    merged = merged.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    if cfg.symbols:
        merged = merged[merged["symbol"].isin(cfg.symbols)]

    symbols = merged["symbol"].unique().tolist()
    symbol_fee = {}
    for sym in symbols:
        fee = cfg.fee_by_symbol.get(sym) if cfg.fee_by_symbol else None
        if fee is None:
            fee = get_fee_for_symbol(sym)
        symbol_fee[sym] = fee

    long_only = cfg.long_only_symbols or set()
    short_only = cfg.short_only_symbols or set()

    def _direction(sym):
        if sym in short_only:
            return "short"
        if sym in long_only:
            return "long"
        return "long"

    high_col = f"predicted_high_p50_h{horizon}"
    low_col = f"predicted_low_p50_h{horizon}"
    close_col = f"predicted_close_p50_h{horizon}"

    cash = float(cfg.initial_cash)
    positions: Dict[str, Position] = {}
    last_close: Dict[str, float] = {}
    equity_values: List[Tuple[pd.Timestamp, float]] = []
    trades: List[UnifiedTradeRecord] = []

    def _mtm():
        total = 0.0
        for pos in positions.values():
            price = last_close.get(pos.symbol, pos.entry_price)
            if pos.direction == "short":
                total += pos.qty * (2 * pos.entry_price - price)
            else:
                total += pos.qty * price
        return total

    def _equity():
        return cash + _mtm()

    def _close_position(sym, pos, price, reason, ts_now):
        nonlocal cash
        fee = symbol_fee.get(sym, 0.001)
        slip = cfg.force_close_slippage if reason != "target" else 0.0
        if pos.direction == "short":
            exit_price = price * (1 + slip)
            cash += pos.qty * (2 * pos.entry_price - exit_price * (1 + fee))
            trades.append(UnifiedTradeRecord(ts_now, sym, "buy_cover", exit_price, pos.qty, cash, 0, reason))
        else:
            exit_price = price * (1 - slip)
            proceeds = pos.qty * exit_price * (1 - fee)
            cash += proceeds
            trades.append(UnifiedTradeRecord(ts_now, sym, "sell", exit_price, pos.qty, cash, 0, reason))

    groups = merged.groupby("timestamp", sort=True)

    for ts, group in groups:
        ts = pd.Timestamp(ts)

        for row in group.itertuples(index=False):
            last_close[row.symbol] = float(row.close)

        equity = _equity()
        equity_values.append((ts, equity))

        allowed = _is_crypto_allowed(ts) if cfg.skip_during_market_hours else True
        backout = _is_backout_window(ts) if cfg.backout_before_market else False

        # 1. Force close before market opens
        if backout:
            closed = list(positions.keys())
            for sym in closed:
                pos = positions[sym]
                price = last_close.get(sym, pos.entry_price)
                _close_position(sym, pos, price, "market_backout", ts)
            for sym in closed:
                del positions[sym]
            continue

        # 2. Check exit targets
        closed = []
        for sym, pos in positions.items():
            row_df = group[group["symbol"] == sym]
            if row_df.empty:
                continue
            row = row_df.iloc[0]

            if pos.sell_target:
                if pos.direction == "short":
                    target_hit = row["low"] <= pos.sell_target * (1 - cfg.bar_margin)
                else:
                    target_hit = row["high"] >= pos.sell_target * (1 + cfg.bar_margin)
                if target_hit and allowed:
                    fee = symbol_fee.get(sym, 0.001)
                    if pos.direction == "short":
                        cash += pos.qty * (2 * pos.entry_price - pos.sell_target * (1 + fee))
                        trades.append(UnifiedTradeRecord(ts, sym, "buy_cover", pos.sell_target, pos.qty, cash, 0, "target"))
                    else:
                        proceeds = pos.qty * pos.sell_target * (1 - fee)
                        cash += proceeds
                        trades.append(UnifiedTradeRecord(ts, sym, "sell", pos.sell_target, pos.qty, cash, 0, "target"))
                    closed.append(sym)
        for sym in closed:
            del positions[sym]

        # 3. Check hold timeout
        closed2 = []
        for sym, pos in positions.items():
            pos.hours_held += 1
            if cfg.max_hold_hours and pos.hours_held >= cfg.max_hold_hours:
                price = last_close.get(sym, pos.entry_price)
                _close_position(sym, pos, price, "timeout", ts)
                closed2.append(sym)
        for sym in closed2:
            del positions[sym]

        # 4. New entries (only when crypto trading allowed)
        if not allowed:
            continue

        open_slots = cfg.max_positions - len(positions)
        if open_slots <= 0:
            continue

        held_symbols = set(positions.keys())
        candidates = []
        for row in group.itertuples(index=False):
            sym = row.symbol
            if sym in held_symbols:
                continue

            buy_price = getattr(row, "buy_price", None)
            buy_int = getattr(row, "buy_amount", None) or getattr(row, "trade_amount", 0)
            sell_price = getattr(row, "sell_price", None)

            if not buy_price or not buy_int or buy_int <= 0:
                continue
            if cfg.min_buy_amount > 0 and buy_int < cfg.min_buy_amount:
                continue

            pred_high = getattr(row, high_col, None)
            pred_low = getattr(row, low_col, None)
            pred_close = getattr(row, close_col, None)

            if not (pred_high and pred_low and pred_close):
                continue

            direction = _direction(sym)
            is_long = direction == "long"
            fee = symbol_fee.get(sym, 0.001)

            if is_long:
                edge = _edge_score(pred_high, pred_low, pred_close, buy_price,
                                   is_long=True, edge_mode=cfg.edge_mode, fee_rate=fee)
                entry_price = buy_price
                exit_price = sell_price
            else:
                edge = (sell_price - pred_low) / sell_price - fee if sell_price > 0 else 0
                entry_price = sell_price
                exit_price = buy_price

            if edge is None or edge < cfg.min_edge:
                continue

            if is_long:
                fillable = row.low <= entry_price * (1 - cfg.bar_margin) if hasattr(row, "low") else True
            else:
                fillable = row.high >= entry_price * (1 + cfg.bar_margin) if hasattr(row, "high") else True
            if not fillable:
                continue

            candidates.append({
                "symbol": sym, "edge": edge,
                "entry_price": entry_price, "exit_price": exit_price,
                "buy_int": buy_int, "direction": direction,
            })

        candidates.sort(key=lambda x: x["edge"], reverse=True)

        per_position_alloc = (equity * cfg.max_leverage) / cfg.max_positions
        for cand in candidates[:open_slots]:
            sym = cand["symbol"]
            fee = symbol_fee.get(sym, 0.001)
            entry_price = cand["entry_price"]
            direction = cand["direction"]
            intensity_frac = min(cand["buy_int"] / cfg.trade_amount_scale, 1.0)
            alloc = per_position_alloc * intensity_frac
            qty = alloc / (entry_price * (1 + fee))
            if cfg.int_qty:
                qty = float(int(qty))
            if qty <= 0:
                continue

            cost = qty * entry_price * (1 + fee)
            cash -= cost
            positions[sym] = Position(
                symbol=sym, qty=qty, entry_price=entry_price,
                entry_ts=ts, sell_target=cand["exit_price"],
                direction=direction,
            )
            side = "short_sell" if direction == "short" else "buy"
            trades.append(UnifiedTradeRecord(ts, sym, side, entry_price, qty, cash, qty, "entry"))

    equity = _equity()
    if equity_values:
        eq_ts, eq_vals = zip(*equity_values)
        equity_curve = pd.Series(eq_vals, index=pd.DatetimeIndex(eq_ts))
    else:
        equity_curve = pd.Series(dtype=float)

    total_return = (equity / cfg.initial_cash - 1) if cfg.initial_cash > 0 else 0.0
    returns = equity_curve.pct_change().dropna()
    sortino = annualized_sortino(returns.values, periods_per_year=8760) if len(returns) > 1 else 0.0

    max_dd = 0.0
    if len(equity_curve) > 0:
        peak = equity_curve.expanding().max()
        dd = (equity_curve - peak) / peak
        max_dd = float(dd.min())

    n_buys = sum(1 for t in trades if t.side in ("buy", "short_sell"))
    n_exits = sum(1 for t in trades if t.side in ("sell", "buy_cover"))
    targets = sum(1 for t in trades if t.reason == "target")
    timeouts = sum(1 for t in trades if t.reason == "timeout")
    backouts = sum(1 for t in trades if t.reason == "market_backout")

    metrics = {
        "total_return": total_return,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "final_equity": equity,
        "num_entries": n_buys,
        "num_exits": n_exits,
        "target_exits": targets,
        "timeout_exits": timeouts,
        "market_backouts": backouts,
    }

    return CryptoPortfolioResult(equity_curve=equity_curve, trades=trades, metrics=metrics)
