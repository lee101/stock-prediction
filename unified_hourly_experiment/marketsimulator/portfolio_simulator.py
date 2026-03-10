"""Multi-position portfolio simulator for stock trading.

Supports N simultaneous positions with per-position equity allocation,
edge-based entry ranking, sell-target exits, hold-timeout exits, and EOD close.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.date_utils import is_nyse_open_on_date
from src.fees import get_fee_for_symbol
from src.hourly_trader_utils import entry_intensity_fraction
from src.metrics_utils import annualized_sortino
from src.robust_trading_metrics import (
    compute_market_sim_goodness_score,
    compute_pnl_smoothness_from_equity,
    compute_trade_rate,
    compute_ulcer_index,
)
from src.symbol_utils import is_crypto_symbol
from src.trade_directions import DEFAULT_LONG_ONLY_STOCKS, DEFAULT_SHORT_ONLY_STOCKS
from src.tradinglib.metrics import annualized_return_from_returns

from .unified_selector import (
    _is_market_open,
    _next_market_open,
    _edge_score,
    _infer_periods_per_year,
    UnifiedTradeRecord,
)
from .portfolio_sim_native import load_portfolio_native_extension


LONG_ONLY_DEFAULT = set(DEFAULT_LONG_ONLY_STOCKS)
SHORT_ONLY_DEFAULT = set(DEFAULT_SHORT_ONLY_STOCKS)


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
    entry_intensity_power: float = 1.0
    entry_min_intensity_fraction: float = 0.0
    long_intensity_multiplier: float = 1.0
    short_intensity_multiplier: float = 1.0
    min_buy_amount: float = 0.0
    fee_by_symbol: Optional[Dict[str, float]] = None
    max_leverage: float = 2.0
    decision_lag_bars: int = 1
    market_order_entry: bool = False
    bar_margin: float = 0.0005
    entry_order_ttl_hours: int = 0  # 0 disables pending entry lifecycle
    entry_selection_mode: str = "edge_rank"  # edge_rank | first_trigger
    long_only_symbols: Optional[set] = None
    short_only_symbols: Optional[set] = None
    force_close_slippage: float = 0.003
    int_qty: bool = True
    margin_annual_rate: float = 0.0
    sim_backend: str = "python"  # python | native | auto


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


def _as_valid_positive(value: object) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v) or v <= 0:
        return None
    return v


_SIDE_CODE_TO_NAME = {0: "buy", 1: "sell", 2: "short_sell", 3: "buy_cover"}
_REASON_CODE_TO_NAME = {0: "entry", 1: "target", 2: "timeout", 3: "eod"}


def _build_portfolio_result(
    *,
    equity_index: pd.DatetimeIndex,
    equity_values: np.ndarray,
    trades: list[UnifiedTradeRecord],
    initial_cash: float,
    symbols: Sequence[str],
    final_equity: float,
) -> PortfolioResult:
    if len(equity_values) > 0:
        equity_curve = pd.Series(equity_values, index=equity_index)
    else:
        equity_curve = pd.Series(dtype=float)

    total_return = (final_equity / initial_cash - 1) if initial_cash > 0 else 0.0
    returns = equity_curve.pct_change().dropna()
    avg_ppy = np.mean([_infer_periods_per_year(s) for s in symbols]) if symbols else 1764.0
    annualized_return = (
        annualized_return_from_returns(returns.values, periods_per_year=avg_ppy)
        if len(returns) > 1
        else 0.0
    )
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
    pnl_smoothness = compute_pnl_smoothness_from_equity(equity_curve.values)
    pnl_smoothness_score = 1.0 / (1.0 + 250.0 * max(pnl_smoothness, 0.0))
    ulcer_index = compute_ulcer_index(equity_curve.values)
    trade_rate = compute_trade_rate(n_buys, len(equity_curve))
    goodness_score = compute_market_sim_goodness_score(
        total_return=total_return,
        sortino=sortino,
        max_drawdown=max_drawdown,
        pnl_smoothness=pnl_smoothness,
        ulcer_index=ulcer_index,
        trade_rate=trade_rate,
        period_count=len(equity_curve),
    )

    metrics = {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "sortino": sortino,
        "final_equity": final_equity,
        "max_drawdown": max_drawdown,
        "pnl_smoothness": pnl_smoothness,
        "pnl_smoothness_score": pnl_smoothness_score,
        "ulcer_index": ulcer_index,
        "trade_rate": trade_rate,
        "goodness_score": goodness_score,
        "num_buys": n_buys,
        "num_sells": n_sells,
        "target_exits": wins,
        "timeout_exits": timeouts,
        "eod_exits": eods,
    }
    return PortfolioResult(equity_curve=equity_curve, trades=trades, metrics=metrics)


def _normalize_backend(value: str | None) -> str:
    raw = (value or "python").strip().lower()
    if raw in {"cpp", "c++", "c"}:
        return "native"
    if raw in {"auto", "native", "python"}:
        return raw
    return "python"


def _get_numeric_column(frame: pd.DataFrame, name: str, *, default: float) -> np.ndarray:
    if name in frame.columns:
        return frame[name].to_numpy(dtype=np.float64, copy=False)
    return np.full(len(frame), default, dtype=np.float64)


def _lag_actions_to_bar_schedule(
    *,
    bars: pd.DataFrame,
    actions: pd.DataFrame,
    lag: int,
) -> pd.DataFrame:
    shifted_parts: list[pd.DataFrame] = []
    for sym in actions["symbol"].astype(str).unique():
        sym_acts = actions[actions["symbol"] == sym].sort_values("timestamp").copy()
        if sym_acts.empty:
            continue
        sym_bars = bars[bars["symbol"] == sym].sort_values("timestamp")
        bar_ts = pd.DatetimeIndex(pd.to_datetime(sym_bars["timestamp"], utc=True))
        if bar_ts.empty:
            shifted_parts.append(sym_acts.iloc[0:0].copy())
            continue

        action_ts = pd.DatetimeIndex(pd.to_datetime(sym_acts["timestamp"], utc=True))
        start_pos = bar_ts.searchsorted(action_ts, side="left")
        target_pos = start_pos + int(lag)
        valid = (start_pos < len(bar_ts)) & (target_pos < len(bar_ts))
        if not np.any(valid):
            shifted_parts.append(sym_acts.iloc[0:0].copy())
            continue

        shifted = sym_acts.loc[valid].copy()
        shifted["timestamp"] = bar_ts.take(target_pos[valid])
        shifted_parts.append(shifted)

    if not shifted_parts:
        return actions.iloc[0:0].copy()
    return pd.concat(shifted_parts, ignore_index=True)


def _run_portfolio_simulation_native(
    *,
    merged: pd.DataFrame,
    cfg: PortfolioConfig,
    symbols: list[str],
    symbol_fee: dict[str, float],
    direction_by_symbol: dict[str, str],
    horizon: int,
) -> PortfolioResult | None:
    ext = load_portfolio_native_extension(verbose=False)
    if ext is None:
        return None

    if merged.empty or not symbols:
        return None

    timestamps = pd.DatetimeIndex(pd.to_datetime(merged["timestamp"], utc=True)).unique().sort_values()
    t_count = len(timestamps)
    s_count = len(symbols)
    if t_count == 0 or s_count == 0:
        return None

    sym_to_idx = {sym: i for i, sym in enumerate(symbols)}
    ts_to_idx = {ts: i for i, ts in enumerate(timestamps)}

    row_t_idx = merged["timestamp"].map(ts_to_idx).to_numpy(dtype=np.int64, copy=False)
    row_s_idx = merged["symbol"].map(sym_to_idx).to_numpy(dtype=np.int64, copy=False)

    dense_shape = (t_count, s_count)
    present = np.zeros(dense_shape, dtype=np.uint8)
    open_arr = np.full(dense_shape, np.nan, dtype=np.float64)
    high_arr = np.full(dense_shape, np.nan, dtype=np.float64)
    low_arr = np.full(dense_shape, np.nan, dtype=np.float64)
    close_arr = np.full(dense_shape, np.nan, dtype=np.float64)
    buy_price_arr = np.full(dense_shape, np.nan, dtype=np.float64)
    sell_price_arr = np.full(dense_shape, np.nan, dtype=np.float64)
    buy_amount_arr = np.zeros(dense_shape, dtype=np.float64)
    sell_amount_arr = np.zeros(dense_shape, dtype=np.float64)
    trade_amount_arr = np.zeros(dense_shape, dtype=np.float64)
    pred_high_arr = np.full(dense_shape, np.nan, dtype=np.float64)
    pred_low_arr = np.full(dense_shape, np.nan, dtype=np.float64)
    pred_close_arr = np.full(dense_shape, np.nan, dtype=np.float64)

    present[row_t_idx, row_s_idx] = 1
    open_arr[row_t_idx, row_s_idx] = _get_numeric_column(merged, "open", default=np.nan)
    high_arr[row_t_idx, row_s_idx] = _get_numeric_column(merged, "high", default=np.nan)
    low_arr[row_t_idx, row_s_idx] = _get_numeric_column(merged, "low", default=np.nan)
    close_arr[row_t_idx, row_s_idx] = _get_numeric_column(merged, "close", default=np.nan)
    buy_price_arr[row_t_idx, row_s_idx] = _get_numeric_column(merged, "buy_price", default=np.nan)
    sell_price_arr[row_t_idx, row_s_idx] = _get_numeric_column(merged, "sell_price", default=np.nan)
    buy_amount_arr[row_t_idx, row_s_idx] = _get_numeric_column(merged, "buy_amount", default=0.0)
    sell_amount_arr[row_t_idx, row_s_idx] = _get_numeric_column(merged, "sell_amount", default=0.0)
    trade_amount_arr[row_t_idx, row_s_idx] = _get_numeric_column(merged, "trade_amount", default=0.0)

    high_col = f"predicted_high_p50_h{horizon}"
    low_col = f"predicted_low_p50_h{horizon}"
    close_col = f"predicted_close_p50_h{horizon}"
    pred_high_arr[row_t_idx, row_s_idx] = _get_numeric_column(merged, high_col, default=np.nan)
    pred_low_arr[row_t_idx, row_s_idx] = _get_numeric_column(merged, low_col, default=np.nan)
    pred_close_arr[row_t_idx, row_s_idx] = _get_numeric_column(merged, close_col, default=np.nan)

    market_open_now = np.zeros(dense_shape, dtype=np.uint8)
    market_open_next = np.zeros(dense_shape, dtype=np.uint8)
    for t_idx, ts in enumerate(timestamps):
        next_ts = ts + pd.Timedelta(hours=1)
        for s_idx, sym in enumerate(symbols):
            market_open_now[t_idx, s_idx] = 1 if _is_market_open(ts, sym) else 0
            market_open_next[t_idx, s_idx] = 1 if _is_market_open(next_ts, sym) else 0

    is_crypto = np.array([1 if is_crypto_symbol(sym) else 0 for sym in symbols], dtype=np.uint8)
    direction = np.array(
        [-1 if direction_by_symbol.get(sym, "long") == "short" else 1 for sym in symbols],
        dtype=np.int8,
    )
    fee = np.array([float(symbol_fee.get(sym, 0.001)) for sym in symbols], dtype=np.float64)
    timestamps_ns = timestamps.view("int64")

    out: dict[str, Any] = ext.simulate_portfolio_dense(
        open_arr,
        high_arr,
        low_arr,
        close_arr,
        buy_price_arr,
        sell_price_arr,
        buy_amount_arr,
        sell_amount_arr,
        trade_amount_arr,
        pred_high_arr,
        pred_low_arr,
        pred_close_arr,
        present,
        market_open_now,
        market_open_next,
        is_crypto,
        direction,
        fee,
        timestamps_ns,
        float(cfg.initial_cash),
        int(cfg.max_positions),
        float(cfg.min_edge),
        str(cfg.edge_mode),
        int(cfg.max_hold_hours),
        bool(cfg.enforce_market_hours),
        bool(cfg.close_at_eod),
        float(cfg.trade_amount_scale),
        float(cfg.entry_intensity_power),
        float(cfg.entry_min_intensity_fraction),
        float(cfg.long_intensity_multiplier),
        float(cfg.short_intensity_multiplier),
        float(cfg.min_buy_amount),
        float(cfg.max_leverage),
        bool(cfg.market_order_entry),
        float(cfg.bar_margin),
        str(cfg.entry_selection_mode),
        float(cfg.force_close_slippage),
        bool(cfg.int_qty),
        float(cfg.margin_annual_rate),
    )

    trade_t_idx = np.asarray(out["trade_t_idx"], dtype=np.int64)
    trade_sym_idx = np.asarray(out["trade_sym_idx"], dtype=np.int64)
    trade_side = np.asarray(out["trade_side"], dtype=np.int8)
    trade_price = np.asarray(out["trade_price"], dtype=np.float64)
    trade_qty = np.asarray(out["trade_qty"], dtype=np.float64)
    trade_cash_after = np.asarray(out["trade_cash_after"], dtype=np.float64)
    trade_inventory_after = np.asarray(out["trade_inventory_after"], dtype=np.float64)
    trade_reason = np.asarray(out["trade_reason"], dtype=np.int8)

    trades: list[UnifiedTradeRecord] = []
    for i in range(len(trade_t_idx)):
        t_idx = int(trade_t_idx[i])
        s_idx = int(trade_sym_idx[i])
        side_name = _SIDE_CODE_TO_NAME.get(int(trade_side[i]), "buy")
        reason_name = _REASON_CODE_TO_NAME.get(int(trade_reason[i]), "entry")
        trades.append(
            UnifiedTradeRecord(
                timestamp=timestamps[t_idx],
                symbol=symbols[s_idx],
                side=side_name,
                price=float(trade_price[i]),
                quantity=float(trade_qty[i]),
                cash_after=float(trade_cash_after[i]),
                inventory_after=float(trade_inventory_after[i]),
                reason=reason_name,
            )
        )

    equity_values = np.asarray(out["equity_values"], dtype=np.float64)
    final_equity = float(out["final_equity"])
    return _build_portfolio_result(
        equity_index=timestamps,
        equity_values=equity_values,
        trades=trades,
        initial_cash=float(cfg.initial_cash),
        symbols=symbols,
        final_equity=final_equity,
    )


def run_portfolio_simulation(
    bars: pd.DataFrame,
    actions: pd.DataFrame,
    config: Optional[PortfolioConfig] = None,
    *,
    horizon: int = 1,
) -> PortfolioResult:
    cfg = config or PortfolioConfig()

    required_keys = {"timestamp", "symbol"}
    if not required_keys.issubset(bars.columns):
        raise ValueError("Bars must contain timestamp and symbol columns")
    if not required_keys.issubset(actions.columns):
        raise ValueError("Actions must contain timestamp and symbol columns")

    if cfg.decision_lag_bars > 0:
        actions = _lag_actions_to_bar_schedule(
            bars=bars,
            actions=actions,
            lag=int(cfg.decision_lag_bars),
        )

    # Keep every market bar so state transitions (timeouts, EOD, pending entries)
    # advance even when no new signal row is present for that symbol/hour.
    merged = bars.merge(actions, on=["timestamp", "symbol"], how="left", suffixes=("", "_act"))
    if merged.empty:
        raise ValueError("No bars available for simulation")
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

    short_only = cfg.short_only_symbols if cfg.short_only_symbols is not None else SHORT_ONLY_DEFAULT

    def _direction(sym):
        if is_crypto_symbol(sym):
            return "long"
        if sym in short_only:
            return "short"
        return "long"

    direction_by_symbol = {sym: _direction(sym) for sym in symbols}

    backend = _normalize_backend(cfg.sim_backend)
    # Python-only feature: pending entry order lifecycle.
    native_allowed = int(cfg.entry_order_ttl_hours) <= 0
    if backend in {"native", "auto"} and native_allowed:
        native_result = _run_portfolio_simulation_native(
            merged=merged,
            cfg=cfg,
            symbols=symbols,
            symbol_fee=symbol_fee,
            direction_by_symbol=direction_by_symbol,
            horizon=horizon,
        )
        if native_result is not None:
            return native_result
        if backend == "native":
            raise RuntimeError("Native portfolio simulator backend requested but unavailable")

    high_col = f"predicted_high_p50_h{horizon}"
    low_col = f"predicted_low_p50_h{horizon}"
    close_col = f"predicted_close_p50_h{horizon}"

    cash = float(cfg.initial_cash)
    positions: Dict[str, Position] = {}
    pending_entries: Dict[str, Dict[str, Any]] = {}
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

    def _bar_margin_for_symbol(sym: str) -> float:
        del sym
        return float(cfg.bar_margin)

    def _required_move_to_fill(row, *, entry_price: float, is_long: bool) -> float:
        open_px = _as_valid_positive(getattr(row, "open", None))
        if open_px is None:
            return float("inf")
        if is_long:
            trigger = float(entry_price) * (1.0 - float(cfg.bar_margin))
            return max(0.0, (open_px - trigger) / open_px)
        trigger = float(entry_price) * (1.0 + float(cfg.bar_margin))
        return max(0.0, (trigger - open_px) / open_px)

    groups = merged.groupby("timestamp", sort=True)

    for ts, group in groups:
        ts = pd.Timestamp(ts)

        for row in group.itertuples(index=False):
            last_close[row.symbol] = float(row.close)

        equity_before = _equity()

        # Margin interest: charge hourly rate on borrowed amount
        if cfg.margin_annual_rate > 0:
            position_value = abs(_mtm())
            margin_used = max(0, position_value - max(0, equity_before))
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
            bar_margin = _bar_margin_for_symbol(sym)

            if pos.sell_target:
                if pos.direction == "short":
                    target_hit = row["low"] <= pos.sell_target * (1 - bar_margin)
                else:
                    target_hit = row["high"] >= pos.sell_target * (1 + bar_margin)

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
        closed3 = []
        if cfg.close_at_eod:
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

        # 3b. Pending entry orders: keep for configurable TTL and fill when touched.
        if pending_entries:
            filled_pending: list[str] = []
            expired_pending: list[str] = []
            for sym, pending in pending_entries.items():
                row_df = group[group["symbol"] == sym]
                if row_df.empty:
                    pending["bars_left"] = int(pending.get("bars_left", 0)) - 1
                    if pending["bars_left"] <= 0:
                        expired_pending.append(sym)
                    continue
                if sym in positions:
                    expired_pending.append(sym)
                    continue
                if len(positions) >= cfg.max_positions:
                    pending["bars_left"] = int(pending.get("bars_left", 0)) - 1
                    if pending["bars_left"] <= 0:
                        expired_pending.append(sym)
                    continue

                if cfg.enforce_market_hours and not is_crypto_symbol(sym) and not _is_market_open(ts, sym):
                    pending["bars_left"] = int(pending.get("bars_left", 0)) - 1
                    if pending["bars_left"] <= 0:
                        expired_pending.append(sym)
                    continue

                row = row_df.iloc[0]
                entry_price = float(pending["entry_price"])
                is_long = str(pending["direction"]) == "long"
                if is_long:
                    fillable = row["low"] <= entry_price * (1 - cfg.bar_margin)
                else:
                    fillable = row["high"] >= entry_price * (1 + cfg.bar_margin)
                if fillable:
                    qty = float(pending["qty"])
                    if qty > 0:
                        fee = symbol_fee.get(sym, 0.001)
                        cost = qty * entry_price * (1 + fee)
                        cash -= cost
                        positions[sym] = Position(
                            symbol=sym,
                            qty=qty,
                            entry_price=entry_price,
                            entry_ts=ts,
                            sell_target=pending["exit_price"],
                            direction=pending["direction"],
                        )
                        side = "short_sell" if pending["direction"] == "short" else "buy"
                        trades.append(UnifiedTradeRecord(ts, sym, side, entry_price, qty, cash, qty, "entry"))
                    filled_pending.append(sym)
                    continue

                pending["bars_left"] = int(pending.get("bars_left", 0)) - 1
                if pending["bars_left"] <= 0:
                    expired_pending.append(sym)

            for sym in filled_pending:
                pending_entries.pop(sym, None)
            for sym in expired_pending:
                pending_entries.pop(sym, None)

        # 4. Find new entries if we have open slots
        open_slots = cfg.max_positions - len(positions) - len(pending_entries)
        if open_slots <= 0:
            equity_values.append((ts, _equity()))
            continue

        held_symbols = set(positions.keys()) | set(pending_entries.keys())
        equity_for_entries = _equity()
        candidates = []
        for row in group.itertuples(index=False):
            sym = row.symbol
            if sym in held_symbols:
                continue
            if cfg.enforce_market_hours and not is_crypto_symbol(sym) and not _is_market_open(ts, sym):
                continue

            direction = direction_by_symbol.get(sym, "long")
            is_long = direction == "long"
            buy_price = _as_valid_positive(getattr(row, "buy_price", None))
            sell_price = _as_valid_positive(getattr(row, "sell_price", None))
            signal_amount, intensity_fraction = entry_intensity_fraction(
                row,
                is_short=not is_long,
                trade_amount_scale=cfg.trade_amount_scale,
                intensity_power=cfg.entry_intensity_power,
                min_intensity_fraction=cfg.entry_min_intensity_fraction,
                side_multiplier=(
                    cfg.short_intensity_multiplier if not is_long else cfg.long_intensity_multiplier
                ),
            )

            if buy_price is None or signal_amount <= 0:
                continue
            if cfg.min_buy_amount > 0 and signal_amount < cfg.min_buy_amount:
                continue

            fee = symbol_fee.get(sym, 0.001)
            pred_high = _as_valid_positive(getattr(row, high_col, None))
            pred_low = _as_valid_positive(getattr(row, low_col, None))
            pred_close = _as_valid_positive(getattr(row, close_col, None))

            if is_long:
                # Fall back to target-price edge when forecast columns are absent.
                if pred_high is not None and pred_low is not None and pred_close is not None:
                    edge = _edge_score(
                        pred_high,
                        pred_low,
                        pred_close,
                        buy_price,
                        is_long=True,
                        edge_mode=cfg.edge_mode,
                        fee_rate=fee,
                    )
                elif sell_price is not None:
                    edge = (sell_price - buy_price) / buy_price - fee
                else:
                    continue
                entry_price = buy_price
                exit_price = sell_price
            else:
                if sell_price is None:
                    continue
                if pred_low is not None:
                    edge = (sell_price - pred_low) / sell_price - fee
                else:
                    edge = (sell_price - buy_price) / sell_price - fee
                entry_price = sell_price
                exit_price = buy_price

            if edge is None or edge < cfg.min_edge:
                continue

            if cfg.market_order_entry:
                fillable = True
                actual_entry_price = row.open if hasattr(row, "open") else row.close
            else:
                bar_margin = _bar_margin_for_symbol(sym)
                if is_long:
                    fillable = row.low <= entry_price * (1 - bar_margin) if hasattr(row, "low") else True
                else:
                    fillable = row.high >= entry_price * (1 + bar_margin) if hasattr(row, "high") else True
                actual_entry_price = entry_price
            if not fillable:
                if int(cfg.entry_order_ttl_hours) <= 0:
                    continue

            candidates.append({
                "symbol": sym,
                "edge": edge,
                "entry_price": actual_entry_price,
                "exit_price": exit_price,
                "signal_amount": signal_amount,
                "intensity_fraction": intensity_fraction,
                "direction": direction,
                "fillable_now": bool(fillable),
                "required_move_frac": _required_move_to_fill(
                    row,
                    entry_price=actual_entry_price,
                    is_long=is_long,
                ),
            })

        if cfg.entry_selection_mode == "first_trigger":
            candidates.sort(key=lambda x: (x["required_move_frac"], -x["edge"]))
        else:
            candidates.sort(key=lambda x: x["edge"], reverse=True)

        for cand in candidates[:open_slots]:
            sym = cand["symbol"]
            sym_leverage = 1.0 if is_crypto_symbol(sym) else cfg.max_leverage
            per_position_alloc = (equity_for_entries * sym_leverage) / cfg.max_positions
            fee = symbol_fee.get(sym, 0.001)
            entry_price = cand["entry_price"]
            direction = cand["direction"]
            intensity_frac = float(cand["intensity_fraction"])
            alloc = per_position_alloc * intensity_frac
            qty = alloc / (entry_price * (1 + fee))
            if cfg.int_qty:
                qty = float(int(qty))
            if qty <= 0:
                continue

            if cand["fillable_now"]:
                cost = qty * entry_price * (1 + fee)
                cash -= cost
                positions[sym] = Position(
                    symbol=sym, qty=qty, entry_price=entry_price,
                    entry_ts=ts, sell_target=cand["exit_price"],
                    direction=direction,
                )
                side = "short_sell" if direction == "short" else "buy"
                trades.append(UnifiedTradeRecord(ts, sym, side, entry_price, qty, cash, qty, "entry"))
            else:
                pending_entries[sym] = {
                    "symbol": sym,
                    "direction": direction,
                    "entry_price": entry_price,
                    "exit_price": cand["exit_price"],
                    "qty": qty,
                    "bars_left": int(cfg.entry_order_ttl_hours),
                }

        equity_values.append((ts, _equity()))

    final_equity = _equity()
    if equity_values:
        eq_ts, eq_vals = zip(*equity_values)
        equity_index = pd.DatetimeIndex(eq_ts)
        eq_array = np.asarray(eq_vals, dtype=np.float64)
    else:
        equity_index = pd.DatetimeIndex([])
        eq_array = np.asarray([], dtype=np.float64)

    return _build_portfolio_result(
        equity_index=equity_index,
        equity_values=eq_array,
        trades=trades,
        initial_cash=float(cfg.initial_cash),
        symbols=symbols,
        final_equity=float(final_equity),
    )
