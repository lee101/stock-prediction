"""Unified stock+crypto selector with proper market hours handling.

Key features:
1. Stocks: only tradable 9:30 AM - 4:00 PM ET on NYSE trading days
2. Crypto: 24/7 tradable
3. Deferred orders: stock orders placed out-of-hours queue for market open
4. Racing orders: "first to hit level" across multiple symbols
5. Equity reservation: only reserve equity for executable orders
6. Annualization: stocks use 252 days, crypto uses 365
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
from src.symbol_utils import is_crypto_symbol

NEW_YORK = ZoneInfo("America/New_York")
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)


@dataclass
class UnifiedSelectionConfig:
    initial_cash: float = 10_000.0
    min_edge: float = 0.0
    risk_weight: float = 0.5
    edge_mode: str = "high_low"
    max_hold_hours: Optional[int] = None
    force_close_on_max_hold: bool = True
    symbols: Optional[Sequence[str]] = None
    allow_reentry_same_bar: bool = False
    enforce_market_hours: bool = True
    close_at_eod: bool = True
    fee_by_symbol: Optional[Dict[str, float]] = None
    allow_short: bool = False
    long_only_symbols: Optional[Sequence[str]] = None
    short_only_symbols: Optional[Sequence[str]] = None
    max_leverage_stock: float = 1.0
    max_leverage_crypto: float = 1.0
    decision_lag_bars: int = 1
    # Racing orders: buy first of N stocks to hit target level
    enable_racing_orders: bool = False
    racing_symbols: Optional[Sequence[str]] = None
    # Deferred orders: queue stock orders for market open
    enable_deferred_orders: bool = True


@dataclass
class DeferredOrder:
    symbol: str
    side: str  # "buy" or "sell"
    price: float
    intensity: float
    created_ts: pd.Timestamp
    expires_ts: Optional[pd.Timestamp] = None


@dataclass
class UnifiedTradeRecord:
    timestamp: pd.Timestamp
    symbol: str
    side: str
    price: float
    quantity: float
    cash_after: float
    inventory_after: float
    reason: Optional[str] = None


@dataclass
class UnifiedSelectorSimulationResult:
    equity_curve: pd.Series
    per_hour: pd.DataFrame
    trades: List[UnifiedTradeRecord]
    final_cash: float
    final_inventory: float
    open_symbol: Optional[str]
    metrics: Dict[str, float] = field(default_factory=dict)


def _is_market_open(ts: pd.Timestamp, symbol: str) -> bool:
    """Check if market is open for symbol at timestamp."""
    if is_crypto_symbol(symbol):
        return True
    try:
        ny_ts = ts.tz_convert(NEW_YORK) if ts.tzinfo else ts.tz_localize("UTC").tz_convert(NEW_YORK)
        if not is_nyse_open_on_date(ny_ts):
            return False
        t = ny_ts.time()
        return MARKET_OPEN <= t <= MARKET_CLOSE
    except Exception:
        return False


def _next_market_open(ts: pd.Timestamp) -> pd.Timestamp:
    """Find next market open time after ts."""
    ny_ts = ts.tz_convert(NEW_YORK) if ts.tzinfo else ts.tz_localize("UTC").tz_convert(NEW_YORK)
    candidate = ny_ts.replace(hour=9, minute=30, second=0, microsecond=0)
    if ny_ts.time() >= MARKET_OPEN:
        candidate += pd.Timedelta(days=1)
    for _ in range(10):
        if is_nyse_open_on_date(candidate):
            return candidate.tz_convert("UTC")
        candidate += pd.Timedelta(days=1)
    return candidate.tz_convert("UTC")


def _infer_periods_per_year(symbol: str, bars_per_day: float = 7.0) -> float:
    """Return annualization factor: crypto=8760, stocks=252*bars_per_day."""
    if is_crypto_symbol(symbol):
        return 24.0 * 365.0
    return bars_per_day * 252.0


def _edge_score(
    pred_high: float,
    pred_low: float,
    pred_close: float,
    entry_price: float,
    is_long: bool,
    edge_mode: str,
    fee_rate: float,
) -> Optional[float]:
    """Calculate edge score for a trade."""
    if not np.isfinite(entry_price) or entry_price <= 0:
        return None
    if is_long:
        if edge_mode == "high_low":
            edge = (pred_high - entry_price) / entry_price - fee_rate
        elif edge_mode == "high":
            edge = (pred_high - entry_price) / entry_price - fee_rate
        else:
            edge = (pred_close - entry_price) / entry_price - fee_rate
    else:
        if edge_mode == "high_low":
            edge = (entry_price - pred_low) / entry_price - fee_rate
        elif edge_mode == "high":
            edge = (entry_price - pred_low) / entry_price - fee_rate
        else:
            edge = (entry_price - pred_close) / entry_price - fee_rate
    return edge if np.isfinite(edge) else None


def run_unified_simulation(
    bars: pd.DataFrame,
    actions: pd.DataFrame,
    config: Optional[UnifiedSelectionConfig] = None,
    *,
    horizon: int = 1,
) -> UnifiedSelectorSimulationResult:
    """Run unified stock+crypto simulation with proper market hours."""
    cfg = config or UnifiedSelectionConfig()

    merged = bars.merge(actions, on=["timestamp", "symbol"], how="inner", suffixes=("", "_act"))
    if merged.empty:
        raise ValueError("No matching bars and actions")

    merged = merged.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    if cfg.symbols:
        merged = merged[merged["symbol"].isin(cfg.symbols)]

    symbols = merged["symbol"].unique().tolist()
    symbol_meta = {}
    for sym in symbols:
        fee = cfg.fee_by_symbol.get(sym) if cfg.fee_by_symbol else None
        if fee is None:
            fee = get_fee_for_symbol(sym)
        is_crypto = is_crypto_symbol(sym)
        symbol_meta[sym] = {
            "fee": fee,
            "is_crypto": is_crypto,
            "periods_per_year": _infer_periods_per_year(sym),
            "max_leverage": cfg.max_leverage_crypto if is_crypto else cfg.max_leverage_stock,
        }

    groups = merged.groupby("timestamp", sort=True)

    cash = float(cfg.initial_cash)
    inventory = 0.0
    open_symbol: Optional[str] = None
    open_ts: Optional[pd.Timestamp] = None
    open_price: float = 0.0
    last_close: Dict[str, float] = {}
    equity_values: List[Tuple[pd.Timestamp, float]] = []
    per_hour_rows: List[Dict] = []
    trades: List[UnifiedTradeRecord] = []
    deferred_orders: List[DeferredOrder] = []

    high_col = f"predicted_high_p50_h{horizon}"
    low_col = f"predicted_low_p50_h{horizon}"
    close_col = f"predicted_close_p50_h{horizon}"

    def _record_trade(ts, symbol, side, price, qty, reason=None):
        trades.append(UnifiedTradeRecord(
            timestamp=ts, symbol=symbol, side=side, price=price,
            quantity=qty, cash_after=cash, inventory_after=inventory, reason=reason,
        ))

    def _execute_buy(ts, symbol, qty, price, fee_rate, reason=None):
        nonlocal cash, inventory, open_symbol, open_ts, open_price
        if qty <= 0:
            return
        cost = qty * price * (1 + fee_rate)
        cash -= cost
        inventory += qty
        if open_symbol is None:
            open_ts = ts
            open_symbol = symbol
            open_price = price
        _record_trade(ts, symbol, "buy", price, qty, reason)

    def _execute_sell(ts, symbol, qty, price, fee_rate, reason=None):
        nonlocal cash, inventory, open_symbol, open_ts, open_price
        if qty <= 0:
            return
        proceeds = qty * price * (1 - fee_rate)
        cash += proceeds
        inventory -= qty
        if abs(inventory) < 1e-12:
            inventory = 0.0
            open_symbol = None
            open_ts = None
            open_price = 0.0
        _record_trade(ts, symbol, "sell", price, qty, reason)

    for ts, group in groups:
        ts = pd.Timestamp(ts)

        # Update last closes
        for row in group.itertuples(index=False):
            last_close[row.symbol] = float(row.close)

        # Calculate equity
        mtm = 0.0
        if open_symbol and inventory != 0:
            price = last_close.get(open_symbol, 0)
            mtm = inventory * price
        equity = cash + mtm
        equity_values.append((ts, equity))

        # Process deferred orders for stocks now that market is open
        if cfg.enable_deferred_orders and deferred_orders:
            executed = []
            for i, order in enumerate(deferred_orders):
                if not _is_market_open(ts, order.symbol):
                    continue
                row = group[group["symbol"] == order.symbol]
                if row.empty:
                    continue
                row = row.iloc[0]
                fee = symbol_meta[order.symbol]["fee"]
                if order.side == "buy":
                    if row.low <= order.price:
                        max_lev = symbol_meta[order.symbol]["max_leverage"]
                        max_notional = max(0, cash * max_lev)
                        intensity = min(1.0, max(0.0, order.intensity))
                        qty = intensity * max_notional / order.price
                        if qty > 0 and open_symbol is None:
                            _execute_buy(ts, order.symbol, qty, order.price, fee, "deferred")
                            executed.append(i)
                else:
                    if row.high >= order.price and open_symbol == order.symbol:
                        _execute_sell(ts, order.symbol, abs(inventory), order.price, fee, "deferred")
                        executed.append(i)
            for i in sorted(executed, reverse=True):
                deferred_orders.pop(i)

        # Close position at EOD for stocks
        if cfg.close_at_eod and open_symbol and not is_crypto_symbol(open_symbol):
            if not _is_market_open(ts + pd.Timedelta(hours=1), open_symbol):
                if _is_market_open(ts, open_symbol):
                    row = group[group["symbol"] == open_symbol]
                    if not row.empty:
                        price = float(row.iloc[0]["close"])
                        fee = symbol_meta[open_symbol]["fee"]
                        if inventory > 0:
                            _execute_sell(ts, open_symbol, inventory, price, fee, "eod")
                        elif inventory < 0:
                            _execute_buy(ts, open_symbol, abs(inventory), price, fee, "eod")

        # Skip if already holding position
        if open_symbol is not None:
            per_hour_rows.append({"timestamp": ts, "equity": equity, "action": "hold"})
            continue

        # Find best entry across all tradable symbols
        candidates = []
        for row in group.itertuples(index=False):
            sym = row.symbol
            meta = symbol_meta.get(sym)
            if not meta:
                continue

            is_open = _is_market_open(ts, sym)

            buy_price = getattr(row, "buy_price", None)
            buy_int = getattr(row, "buy_amount", None) or getattr(row, "trade_amount", 0)

            if buy_price and buy_int and buy_int > 0:
                pred_high = getattr(row, high_col, None)
                pred_low = getattr(row, low_col, None)
                pred_close = getattr(row, close_col, None)

                if pred_high and pred_low and pred_close:
                    edge = _edge_score(
                        pred_high, pred_low, pred_close,
                        buy_price, is_long=True,
                        edge_mode=cfg.edge_mode, fee_rate=meta["fee"],
                    )
                    if edge and edge >= cfg.min_edge:
                        fillable = row.low <= buy_price if hasattr(row, "low") else True
                        candidates.append({
                            "symbol": sym,
                            "edge": edge,
                            "buy_price": buy_price,
                            "buy_int": buy_int,
                            "is_open": is_open,
                            "fillable": fillable,
                            "row": row,
                        })

        if not candidates:
            per_hour_rows.append({"timestamp": ts, "equity": equity, "action": "no_candidates"})
            continue

        # Sort by edge, prefer currently tradable
        candidates.sort(key=lambda x: (x["is_open"], x["edge"]), reverse=True)

        # Find best executable candidate
        for cand in candidates:
            sym = cand["symbol"]
            meta = symbol_meta[sym]

            if cand["is_open"]:
                if cand["fillable"]:
                    max_lev = meta["max_leverage"]
                    max_notional = max(0, cash * max_lev)
                    buy_price = cand["buy_price"]
                    intensity = min(1.0, max(0.0, cand["buy_int"]))
                    qty = intensity * max_notional / buy_price
                    if qty > 0:
                        _execute_buy(ts, sym, qty, buy_price, meta["fee"])
                        per_hour_rows.append({"timestamp": ts, "equity": equity, "action": f"buy_{sym}"})
                        break
            elif cfg.enable_deferred_orders:
                deferred_orders.append(DeferredOrder(
                    symbol=sym,
                    side="buy",
                    price=cand["buy_price"],
                    intensity=cand["buy_int"],
                    created_ts=ts,
                    expires_ts=_next_market_open(ts) + pd.Timedelta(hours=1),
                ))
                per_hour_rows.append({"timestamp": ts, "equity": equity, "action": f"defer_{sym}"})
                break
        else:
            per_hour_rows.append({"timestamp": ts, "equity": equity, "action": "no_fill"})

    # Build equity curve
    if equity_values:
        eq_ts, eq_vals = zip(*equity_values)
        equity_curve = pd.Series(eq_vals, index=pd.DatetimeIndex(eq_ts))
    else:
        equity_curve = pd.Series(dtype=float)

    # Calculate metrics
    total_return = (equity_curve.iloc[-1] / cfg.initial_cash - 1) if len(equity_curve) > 0 else 0.0

    # Annualized sortino (use weighted average of periods)
    avg_periods = np.mean([m["periods_per_year"] for m in symbol_meta.values()]) if symbol_meta else 8760
    returns = equity_curve.pct_change().dropna()
    if len(returns) > 1:
        sortino = annualized_sortino(returns.values, periods_per_year=avg_periods)
    else:
        sortino = 0.0

    metrics = {
        "total_return": total_return,
        "sortino": sortino,
        "num_trades": len(trades),
        "final_equity": equity_curve.iloc[-1] if len(equity_curve) > 0 else cfg.initial_cash,
    }

    return UnifiedSelectorSimulationResult(
        equity_curve=equity_curve,
        per_hour=pd.DataFrame(per_hour_rows),
        trades=trades,
        final_cash=cash,
        final_inventory=inventory,
        open_symbol=open_symbol,
        metrics=metrics,
    )
