"""Multi-position portfolio simulator for stock trading.

Supports N simultaneous positions with per-position equity allocation,
edge-based entry ranking, sell-target exits, hold-timeout exits, and EOD close.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.date_utils import is_nyse_open_on_date
from src.fees import get_fee_for_symbol
from src.metrics_utils import annualized_sortino
from src.symbol_utils import is_crypto_symbol

from .unified_selector import (
    _is_market_open,
    _next_market_open,
    _edge_score,
    _infer_periods_per_year,
    UnifiedTradeRecord,
)


LONG_ONLY_DEFAULT = {"NVDA", "MSFT", "META", "GOOG", "NET", "PLTR", "DBX", "TSLA", "AAPL"}
SHORT_ONLY_DEFAULT = {"YELP", "EBAY", "TRIP", "MTCH", "ANGI", "Z", "EXPE", "BKNG", "NWSA", "NYT"}


@dataclass
class PortfolioConfig:
    initial_cash: float = 10_000.0
    max_positions: int = 5
    min_edge: float = 0.0
    edge_mode: str = "high_low"
    max_hold_hours: int = 6
    enforce_market_hours: bool = True
    close_at_eod: bool = True
    symbols: Optional[Sequence[str]] = None
    trade_amount_scale: float = 100.0
    min_buy_amount: float = 0.0
    fee_by_symbol: Optional[Dict[str, float]] = None
    max_leverage: float = 2.0
    decision_lag_bars: int = 1
    market_order_entry: bool = False
    bar_margin: float = 0.0005
    long_only_symbols: Optional[set] = None
    short_only_symbols: Optional[set] = None
    force_close_slippage: float = 0.003
    int_qty: bool = True
    margin_annual_rate: float = 0.0


@dataclass
class Position:
    symbol: str
    qty: float
    entry_price: float
    entry_ts: pd.Timestamp
    sell_target: Optional[float] = None
    market_hours_held: int = 0
    direction: str = "long"


@dataclass
class PortfolioResult:
    equity_curve: pd.Series
    trades: List[UnifiedTradeRecord]
    metrics: Dict[str, float] = field(default_factory=dict)


def run_portfolio_simulation(
    bars: pd.DataFrame,
    actions: pd.DataFrame,
    config: Optional[PortfolioConfig] = None,
    *,
    horizon: int = 1,
) -> PortfolioResult:
    cfg = config or PortfolioConfig()

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

    long_only = cfg.long_only_symbols if cfg.long_only_symbols is not None else LONG_ONLY_DEFAULT
    short_only = cfg.short_only_symbols if cfg.short_only_symbols is not None else SHORT_ONLY_DEFAULT

    def _direction(sym):
        if sym in short_only:
            return "short"
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
        total = 0
        for pos in positions.values():
            price = last_close.get(pos.symbol, pos.entry_price)
            if pos.direction == "short":
                total += pos.qty * (2 * pos.entry_price - price)
            else:
                total += pos.qty * price
        return total

    def _equity():
        return cash + _mtm()

    groups = merged.groupby("timestamp", sort=True)

    for ts, group in groups:
        ts = pd.Timestamp(ts)

        for row in group.itertuples(index=False):
            last_close[row.symbol] = float(row.close)

        equity = _equity()
        equity_values.append((ts, equity))

        # Margin interest: charge hourly rate on borrowed amount
        if cfg.margin_annual_rate > 0:
            position_value = abs(_mtm())
            margin_used = max(0, position_value - max(0, equity))
            if margin_used > 0:
                cash -= margin_used * cfg.margin_annual_rate / 8760

        # 1. Check exit targets (sell for longs, buy-to-cover for shorts)
        closed = []
        for sym, pos in positions.items():
            row_df = group[group["symbol"] == sym]
            if row_df.empty:
                continue
            row = row_df.iloc[0]
            fee = symbol_fee.get(sym, 0.001)

            if pos.sell_target:
                if pos.direction == "short":
                    target_hit = row["low"] <= pos.sell_target * (1 - cfg.bar_margin)
                else:
                    target_hit = row["high"] >= pos.sell_target * (1 + cfg.bar_margin)

                if target_hit:
                    if cfg.enforce_market_hours and not is_crypto_symbol(sym) and not _is_market_open(ts, sym):
                        continue
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

        # 2. Check hold timeout (market hours)
        closed2 = []
        for sym, pos in positions.items():
            if _is_market_open(ts, sym) or is_crypto_symbol(sym):
                pos.market_hours_held += 1
            if cfg.max_hold_hours and pos.market_hours_held >= cfg.max_hold_hours:
                if cfg.enforce_market_hours and not is_crypto_symbol(sym) and not _is_market_open(ts, sym):
                    continue
                price = last_close.get(sym, pos.entry_price)
                fee = symbol_fee.get(sym, 0.001)
                slip = cfg.force_close_slippage
                if pos.direction == "short":
                    exit_price = price * (1 + slip)
                    cash += pos.qty * (2 * pos.entry_price - exit_price * (1 + fee))
                    trades.append(UnifiedTradeRecord(ts, sym, "buy_cover", exit_price, pos.qty, cash, 0, "timeout"))
                else:
                    exit_price = price * (1 - slip)
                    proceeds = pos.qty * exit_price * (1 - fee)
                    cash += proceeds
                    trades.append(UnifiedTradeRecord(ts, sym, "sell", exit_price, pos.qty, cash, 0, "timeout"))
                closed2.append(sym)
        for sym in closed2:
            del positions[sym]

        # 3. EOD close for stocks
        if cfg.close_at_eod:
            closed3 = []
            for sym, pos in positions.items():
                if is_crypto_symbol(sym):
                    continue
                if _is_market_open(ts, sym) and not _is_market_open(ts + pd.Timedelta(hours=1), sym):
                    price = last_close.get(sym, pos.entry_price)
                    fee = symbol_fee.get(sym, 0.001)
                    slip = cfg.force_close_slippage
                    if pos.direction == "short":
                        exit_price = price * (1 + slip)
                        cash += pos.qty * (2 * pos.entry_price - exit_price * (1 + fee))
                        trades.append(UnifiedTradeRecord(ts, sym, "buy_cover", exit_price, pos.qty, cash, 0, "eod"))
                    else:
                        exit_price = price * (1 - slip)
                        proceeds = pos.qty * exit_price * (1 - fee)
                        cash += proceeds
                        trades.append(UnifiedTradeRecord(ts, sym, "sell", exit_price, pos.qty, cash, 0, "eod"))
                    closed3.append(sym)
            for sym in closed3:
                del positions[sym]

        # 4. Find new entries if we have open slots
        open_slots = cfg.max_positions - len(positions)
        if open_slots <= 0:
            continue

        held_symbols = set(positions.keys())
        candidates = []
        for row in group.itertuples(index=False):
            sym = row.symbol
            if sym in held_symbols:
                continue
            if cfg.enforce_market_hours and not is_crypto_symbol(sym) and not _is_market_open(ts, sym):
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

            if cfg.market_order_entry:
                fillable = True
                actual_entry_price = row.open if hasattr(row, "open") else row.close
            else:
                if is_long:
                    fillable = row.low <= entry_price * (1 - cfg.bar_margin) if hasattr(row, "low") else True
                else:
                    fillable = row.high >= entry_price * (1 + cfg.bar_margin) if hasattr(row, "high") else True
                actual_entry_price = entry_price
            if not fillable:
                continue

            candidates.append({
                "symbol": sym,
                "edge": edge,
                "entry_price": actual_entry_price,
                "exit_price": exit_price,
                "buy_int": buy_int,
                "direction": direction,
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

    # Final equity
    equity = _equity()
    if equity_values:
        eq_ts, eq_vals = zip(*equity_values)
        equity_curve = pd.Series(eq_vals, index=pd.DatetimeIndex(eq_ts))
    else:
        equity_curve = pd.Series(dtype=float)

    total_return = (equity / cfg.initial_cash - 1) if cfg.initial_cash > 0 else 0.0
    returns = equity_curve.pct_change().dropna()
    avg_ppy = np.mean([_infer_periods_per_year(s) for s in symbols]) if symbols else 1764.0
    sortino = annualized_sortino(returns.values, periods_per_year=avg_ppy) if len(returns) > 1 else 0.0

    n_buys = sum(1 for t in trades if t.side in ("buy", "short_sell"))
    n_sells = sum(1 for t in trades if t.side in ("sell", "buy_cover"))
    wins = sum(1 for t in trades if t.side in ("sell", "buy_cover") and t.reason == "target")
    timeouts = sum(1 for t in trades if t.side in ("sell", "buy_cover") and t.reason == "timeout")
    eods = sum(1 for t in trades if t.side in ("sell", "buy_cover") and t.reason == "eod")

    max_drawdown = 0.0
    if len(equity_curve) > 0:
        running_max = equity_curve.cummax()
        drawdowns = (running_max - equity_curve) / running_max
        max_drawdown = float(drawdowns.max())

    metrics = {
        "total_return": total_return,
        "sortino": sortino,
        "final_equity": equity,
        "max_drawdown": max_drawdown,
        "num_buys": n_buys,
        "num_sells": n_sells,
        "target_exits": wins,
        "timeout_exits": timeouts,
        "eod_exits": eods,
    }

    return PortfolioResult(equity_curve=equity_curve, trades=trades, metrics=metrics)
