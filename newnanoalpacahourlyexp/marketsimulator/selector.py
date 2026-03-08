from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.fees import get_fee_for_symbol
from src.robust_trading_metrics import compute_pnl_smoothness_from_equity
from src.symbol_utils import is_crypto_symbol
from src.tradinglib.metrics import pnl_metrics
from src.trade_directions import resolve_trade_directions

from .simulator import _end_of_day_mask, _market_open_mask, _to_new_york, _infer_periods_per_year


@dataclass
class SelectionConfig:
    initial_cash: float = 10_000.0
    initial_inventory: float = 0.0
    initial_symbol: Optional[str] = None
    initial_open_price: Optional[float] = None
    initial_open_ts: Optional[pd.Timestamp] = None
    min_edge: float = 0.0
    risk_weight: float = 0.5
    edge_mode: str = "high_low"
    max_hold_hours: Optional[int] = None
    force_close_on_max_hold: bool = True
    symbols: Optional[Sequence[str]] = None
    allow_reentry_same_bar: bool = False
    enforce_market_hours: bool = True
    close_at_eod: bool = True
    # Optional realism control: cap fills to a fraction of bar volume (base units / shares).
    # When set, each bar tracks remaining fillable volume per symbol so multiple trades in the same
    # bar cannot exceed the cap.
    max_volume_fraction: Optional[float] = None
    # When True (legacy behaviour), the selector only considers entry candidates whose limit
    # prices would have filled on the current bar (e.g., low <= buy_price for longs).
    #
    # This is optimistic because it uses future bar high/low at decision time. Set to False
    # for a more live-like simulation: select by edge score, then simulate whether the chosen
    # order actually fills.
    select_fillable_only: bool = True
    fee_by_symbol: Optional[Dict[str, float]] = None
    periods_per_year_by_symbol: Optional[Dict[str, float]] = None
    allow_short: bool = False
    long_only_symbols: Optional[Sequence[str]] = None
    short_only_symbols: Optional[Sequence[str]] = None
    # Leverage/financing realism knobs (defaults preserve legacy behaviour).
    max_leverage_stock: float = 1.0
    max_leverage_crypto: float = 1.0
    long_max_leverage_stock: Optional[float] = None
    short_max_leverage_stock: Optional[float] = None
    long_max_leverage_crypto: Optional[float] = None
    short_max_leverage_crypto: Optional[float] = None
    margin_interest_annual: float = 0.0
    short_borrow_cost_annual: float = 0.0
    # Multi-position: 1 = legacy single-position, 2+ = hold up to N simultaneously.
    max_concurrent_positions: int = 1
    # Work-stealing: sell profitable position to enter better opportunity.
    work_steal_enabled: bool = False
    work_steal_min_profit_pct: float = 0.001
    work_steal_min_edge: float = 0.005
    work_steal_edge_margin: float = 0.0
    # Realism: when >0, shift decision inputs (actions + forecast columns) back by N bars per
    # symbol so a decision made after bar t closes is executed on bar t+N.
    #
    # This matches the live hourly loop which computes an action on the latest completed bar
    # then places orders for the next bar.
    decision_lag_bars: int = 0
    bar_margin: float = 0.0
    # Limit-order fill realism. "binary" preserves legacy touch/no-touch fills.
    # "penetration" scales fill size by how far the bar traded through the limit.
    limit_fill_model: str = "binary"
    # Optional minimum fill fraction when the bar only touches the limit exactly.
    # Only used when limit_fill_model="penetration".
    touch_fill_fraction: float = 0.0


@dataclass
class SelectorTradeRecord:
    timestamp: pd.Timestamp
    symbol: str
    side: str
    price: float
    quantity: float
    cash_after: float
    inventory_after: float
    reason: Optional[str] = None


@dataclass
class SelectorSimulationResult:
    equity_curve: pd.Series
    per_hour: pd.DataFrame
    trades: List[SelectorTradeRecord]
    final_cash: float
    final_inventory: float
    open_symbol: Optional[str]
    metrics: Dict[str, float] = field(default_factory=dict)


def run_best_trade_simulation(
    bars: pd.DataFrame,
    actions: pd.DataFrame,
    config: Optional[SelectionConfig] = None,
    *,
    horizon: int = 1,
) -> SelectorSimulationResult:
    cfg = config or SelectionConfig()
    merged = _prepare_frame(bars, actions, horizon=horizon, symbols=cfg.symbols)
    return run_best_trade_simulation_merged(merged, cfg, horizon=horizon)


def run_best_trade_simulation_merged(
    merged_df: pd.DataFrame,
    config: Optional[SelectionConfig] = None,
    *,
    horizon: int = 1,
) -> SelectorSimulationResult:
    """Run the selector simulation on a pre-merged bars/actions dataframe.

    The input is expected to already contain both bar columns (high/low/close/etc)
    and action columns (buy/sell prices and intensities) keyed by (timestamp, symbol),
    matching the output shape of :func:`_prepare_frame`.
    """
    cfg = config or SelectionConfig()
    merged = merged_df
    if "timestamp" not in merged.columns:
        raise ValueError("Merged dataframe must include a timestamp column.")
    if "symbol" not in merged.columns:
        merged = merged.assign(symbol="DEFAULT")
    if not pd.api.types.is_datetime64_any_dtype(merged["timestamp"]):
        merged = merged.assign(timestamp=pd.to_datetime(merged["timestamp"], utc=True))
    _validate_merged_frame(merged, horizon=horizon)

    if merged.empty:
        raise ValueError("Merged dataframe is empty; ensure actions cover the provided bars.")

    decision_lag_bars = int(getattr(cfg, "decision_lag_bars", 0) or 0)
    if decision_lag_bars < 0:
        raise ValueError(f"decision_lag_bars must be >= 0, got {cfg.decision_lag_bars}.")
    _validate_fill_config(cfg)
    if decision_lag_bars:
        # Shift decision-time inputs back by N bars per symbol so the fill/mark-to-market
        # uses the current bar while actions/forecasts reflect what was known N bars ago.
        merged = merged.copy()
        merged = merged.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
        horizon_int = int(horizon)
        forecast_cols = [
            f"predicted_high_p50_h{horizon_int}",
            f"predicted_low_p50_h{horizon_int}",
            f"predicted_close_p50_h{horizon_int}",
        ]
        action_cols = ["buy_price", "sell_price", "buy_amount", "sell_amount", "trade_amount"]
        shift_cols = [c for c in action_cols + forecast_cols if c in merged.columns]
        if shift_cols:
            shifted = merged.groupby("symbol", sort=False)[shift_cols].shift(decision_lag_bars)
            for col in shift_cols:
                merged[col] = shifted[col]
            merged = merged.dropna(subset=shift_cols).reset_index(drop=True)
        if merged.empty:
            raise ValueError(
                "Merged dataframe is empty after applying decision_lag_bars; "
                "ensure the evaluation window contains more bars."
            )
    max_volume_fraction: Optional[float]
    if cfg.max_volume_fraction is None:
        max_volume_fraction = None
    else:
        max_volume_fraction = float(cfg.max_volume_fraction)
        if not np.isfinite(max_volume_fraction) or max_volume_fraction <= 0.0 or max_volume_fraction > 1.0:
            raise ValueError(f"max_volume_fraction must be in (0, 1], got {cfg.max_volume_fraction}.")
        if "volume" not in merged.columns:
            raise ValueError("max_volume_fraction requires a 'volume' column in bars/actions merge.")

    if cfg.max_concurrent_positions > 1:
        return _run_multi_position_simulation(merged, cfg, horizon=horizon)
    return _run_best_trade_simulation_on_merged(merged, cfg, horizon=horizon)


# ------------------------------------------------------------------


def _validate_merged_frame(merged: pd.DataFrame, *, horizon: int) -> None:
    required = {"high", "low", "close", "buy_price", "sell_price"}
    missing = required - set(merged.columns)
    if missing:
        raise ValueError(f"Missing required columns in merged frame: {sorted(missing)}")

    high_col = f"predicted_high_p50_h{int(horizon)}"
    low_col = f"predicted_low_p50_h{int(horizon)}"
    close_col = f"predicted_close_p50_h{int(horizon)}"
    for col in (high_col, low_col, close_col):
        if col not in merged.columns:
            raise ValueError(f"Missing required forecast column {col}.")


def _run_best_trade_simulation_on_merged(
    merged: pd.DataFrame,
    cfg: SelectionConfig,
    *,
    horizon: int,
) -> SelectorSimulationResult:
    merged = merged.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    groups = merged.groupby("timestamp", sort=True)

    symbol_meta = _build_symbol_meta(merged, cfg)
    initial_position = _resolve_initial_position(merged, cfg)

    cash = float(cfg.initial_cash)
    inventory = float(initial_position["inventory"]) if initial_position is not None else 0.0  # signed: >0 long, <0 short
    open_symbol: Optional[str] = str(initial_position["symbol"]) if initial_position is not None else None
    open_ts: Optional[pd.Timestamp] = initial_position["open_ts"] if initial_position is not None else None
    open_price: float = float(initial_position["open_price"]) if initial_position is not None else 0.0
    last_close: Dict[str, float] = (
        {str(initial_position["symbol"]): float(initial_position["mark_price"])}
        if initial_position is not None
        else {}
    )
    financing_cost_paid = 0.0
    prev_ts: Optional[pd.Timestamp] = None
    equity_values: List[float] = []
    per_hour_rows: List[Dict[str, float | str]] = []
    trades: List[SelectorTradeRecord] = []

    max_hold_delta = None
    if cfg.max_hold_hours is not None and cfg.max_hold_hours > 0:
        max_hold_delta = pd.Timedelta(hours=int(cfg.max_hold_hours))

    def _record_trade(
        *,
        ts: pd.Timestamp,
        symbol: str,
        side: str,
        price: float,
        qty: float,
        reason: Optional[str] = None,
    ) -> None:
        trades.append(
            SelectorTradeRecord(
                timestamp=ts,
                symbol=symbol,
                side=side,
                price=float(price),
                quantity=float(qty),
                cash_after=float(cash),
                inventory_after=float(inventory),
                reason=reason,
            )
        )

    def _execute_buy(ts: pd.Timestamp, symbol: str, qty: float, price: float, fee_rate: float, reason: Optional[str] = None) -> None:
        nonlocal cash, inventory, open_symbol, open_ts, open_price
        if qty <= 0:
            return
        cost = qty * price * (1 + fee_rate)
        cash -= cost
        was_flat = (open_symbol is None) or abs(inventory) <= 1e-12
        inventory += qty
        if was_flat and abs(inventory) > 1e-12:
            open_ts = ts
            open_symbol = symbol
            open_price = price
        if abs(inventory) <= 1e-12:
            inventory = 0.0
            open_symbol = None
            open_ts = None
            open_price = 0.0
        _record_trade(ts=ts, symbol=symbol, side="buy", price=price, qty=qty, reason=reason)

    def _execute_sell(ts: pd.Timestamp, symbol: str, qty: float, price: float, fee_rate: float, reason: Optional[str] = None) -> None:
        nonlocal cash, inventory, open_symbol, open_ts, open_price
        if qty <= 0:
            return
        proceeds = qty * price * (1 - fee_rate)
        cash += proceeds
        was_flat = (open_symbol is None) or abs(inventory) <= 1e-12
        inventory -= qty
        if was_flat and abs(inventory) > 1e-12:
            open_ts = ts
            open_symbol = symbol
            open_price = price
        if abs(inventory) <= 1e-12:
            inventory = 0.0
            open_symbol = None
            open_ts = None
            open_price = 0.0
        _record_trade(ts=ts, symbol=symbol, side="sell", price=price, qty=qty, reason=reason)

    max_volume_fraction: Optional[float]
    if cfg.max_volume_fraction is None:
        max_volume_fraction = None
    else:
        max_volume_fraction = float(cfg.max_volume_fraction)

    for ts, group in groups:
        remaining_volume: Dict[str, float] = {}
        if max_volume_fraction is not None:
            # Remaining volume is tracked per symbol per bar. We cap each bar's fill volume in the
            # same units as quantity (base units / shares).
            for row in group.itertuples(index=False):
                symbol = str(row.symbol).upper()
                vol = _safe_float(getattr(row, "volume", None)) or 0.0
                if not np.isfinite(vol) or vol <= 0.0:
                    remaining_volume[symbol] = 0.0
                else:
                    remaining_volume[symbol] = float(vol) * float(max_volume_fraction)

        def _cap_qty(symbol: str, desired_qty: float) -> float:
            if desired_qty <= 0:
                return 0.0
            if max_volume_fraction is None:
                return float(desired_qty)
            avail = float(remaining_volume.get(symbol, 0.0))
            if not np.isfinite(avail) or avail <= 0.0:
                return 0.0
            filled = min(float(desired_qty), avail)
            remaining_volume[symbol] = avail - filled
            return float(filled)

        buy_filled = 0.0
        sell_filled = 0.0
        selected_symbol = ""
        selected_score = 0.0
        forced_close = False
        closed_this_step = False

        for row in group.itertuples(index=False):
            last_close[str(row.symbol)] = float(row.close)

        if prev_ts is not None:
            dt = ts - prev_ts
            delta_hours = float(getattr(dt, "total_seconds", lambda: 0.0)() / 3600.0)
            if np.isfinite(delta_hours) and delta_hours > 0.0:
                if cfg.margin_interest_annual:
                    debt = max(0.0, -float(cash))
                    if debt > 0.0:
                        rate_per_hour = float(cfg.margin_interest_annual) / (365.0 * 24.0)
                        cost = debt * rate_per_hour * delta_hours
                        if np.isfinite(cost) and cost > 0.0:
                            cash -= float(cost)
                            financing_cost_paid += float(cost)
                if cfg.short_borrow_cost_annual and open_symbol is not None and inventory < 0.0:
                    price = last_close.get(open_symbol)
                    if price is None:
                        price = last_close.get(str(open_symbol).upper())
                    if price is not None and price > 0.0:
                        borrow_notional = abs(float(inventory)) * float(price)
                        rate_per_hour = float(cfg.short_borrow_cost_annual) / (365.0 * 24.0)
                        cost = borrow_notional * rate_per_hour * delta_hours
                        if np.isfinite(cost) and cost > 0.0:
                            cash -= float(cost)
                            financing_cost_paid += float(cost)

        if open_symbol and max_hold_delta is not None and cfg.force_close_on_max_hold and open_ts is not None:
            if ts - open_ts >= max_hold_delta:
                row = _lookup_symbol_row(group, open_symbol)
                if row is not None:
                    fee_rate = symbol_meta[open_symbol]["fee"]
                    if inventory > 0:
                        qty = _cap_qty(open_symbol, inventory)
                        _execute_sell(ts, open_symbol, qty, float(row.close), fee_rate, reason="max_hold")
                        sell_filled = 1.0
                    elif inventory < 0:
                        qty = _cap_qty(open_symbol, abs(inventory))
                        _execute_buy(ts, open_symbol, qty, float(row.close), fee_rate, reason="max_hold")
                        buy_filled = 1.0
                    forced_close = True
                    closed_this_step = True

        work_stolen = False
        if (
            cfg.work_steal_enabled
            and not forced_close
            and open_symbol is not None
            and inventory > 0
            and open_price > 0
        ):
            cur_row = _lookup_symbol_row(group, open_symbol)
            if cur_row is not None:
                unrealized_pct = (float(cur_row.close) - open_price) / open_price
                if unrealized_pct >= cfg.work_steal_min_profit_pct:
                    steal_candidates: List[Tuple[float, str, object, float]] = []
                    for srow in group.itertuples(index=False):
                        ssym = str(srow.symbol)
                        if ssym == open_symbol:
                            continue
                        if not _is_tradable(symbol_meta, ssym, ts, cfg):
                            continue
                        sfee = symbol_meta[ssym]["fee"]
                        sdirs = symbol_meta[ssym].get("directions") or {}
                        if not bool(sdirs.get("can_long", True)):
                            continue
                        sbuy_int, _ = _extract_intensity(srow)
                        if sbuy_int <= 0:
                            continue
                        if cfg.select_fillable_only and _limit_fill_fraction(
                            srow,
                            side="buy",
                            limit_price=float(srow.buy_price),
                            cfg=cfg,
                        ) <= 0.0:
                            continue
                        score = _edge_score_long(srow, horizon=horizon, config=cfg, buy_intensity=sbuy_int, fee_rate=sfee)
                        if score is not None and score >= cfg.work_steal_min_edge:
                            steal_candidates.append((score, ssym, srow, sbuy_int))
                    if steal_candidates:
                        steal_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
                        best_score, best_sym, best_row, best_int = steal_candidates[0]
                        if best_score >= unrealized_pct + cfg.work_steal_edge_margin:
                            exit_fee = symbol_meta[open_symbol]["fee"]
                            qty_sell = _cap_qty(open_symbol, inventory)
                            _execute_sell(ts, open_symbol, qty_sell, float(cur_row.close), exit_fee, reason="work_steal_exit")
                            sell_filled = 1.0
                            entry_fee = symbol_meta[best_sym]["fee"]
                            max_lev = _entry_max_leverage(symbol_meta, best_sym, side="long")
                            max_notional = max(0.0, float(cash) * max_lev)
                            max_buy = max_notional / (best_row.buy_price * (1 + entry_fee)) if best_row.buy_price > 0 else 0.0
                            fill_fraction = _limit_fill_fraction(
                                best_row,
                                side="buy",
                                limit_price=float(best_row.buy_price),
                                cfg=cfg,
                            )
                            qty_buy = best_int * max_buy * fill_fraction
                            qty_buy = _cap_qty(best_sym, qty_buy)
                            if qty_buy > 0:
                                _execute_buy(ts, best_sym, qty_buy, float(best_row.buy_price), entry_fee, reason="work_steal_entry")
                                buy_filled = 1.0
                            work_stolen = True
                            closed_this_step = True

        if open_symbol and inventory != 0 and not work_stolen:
            row = _lookup_symbol_row(group, open_symbol)
            if row is not None and not forced_close:
                buy_intensity, sell_intensity = _extract_intensity(row)
                if not _is_tradable(symbol_meta, open_symbol, ts, cfg):
                    # Treat out-of-session bars as non-tradable for both sides so we do not
                    # accidentally cover shorts (buy) when market-hours enforcement is enabled.
                    buy_intensity = 0.0
                    sell_intensity = 0.0
                if inventory > 0:
                    fill_fraction = _limit_fill_fraction(
                        row,
                        side="sell",
                        limit_price=float(row.sell_price),
                        cfg=cfg,
                    )
                    if fill_fraction > 0.0 and sell_intensity > 0:
                        qty = sell_intensity * inventory * fill_fraction
                        qty = _cap_qty(open_symbol, qty)
                        fee_rate = symbol_meta[open_symbol]["fee"]
                        _execute_sell(ts, open_symbol, min(qty, inventory), float(row.sell_price), fee_rate)
                        sell_filled = 1.0
                        closed_this_step = True
                else:
                    fill_fraction = _limit_fill_fraction(
                        row,
                        side="buy",
                        limit_price=float(row.buy_price),
                        cfg=cfg,
                    )
                    if fill_fraction > 0.0 and buy_intensity > 0:
                        qty = buy_intensity * abs(inventory) * fill_fraction
                        qty = _cap_qty(open_symbol, qty)
                        fee_rate = symbol_meta[open_symbol]["fee"]
                        _execute_buy(ts, open_symbol, min(qty, abs(inventory)), float(row.buy_price), fee_rate)
                        buy_filled = 1.0
                        closed_this_step = True
            current_price = _resolve_close(open_symbol, row, last_close)
        else:
            current_price = None

        if open_symbol is None and not forced_close and (cfg.allow_reentry_same_bar or not closed_this_step):
            candidates: List[Tuple[float, str, str, object, float]] = []
            for row in group.itertuples(index=False):
                symbol = str(row.symbol)
                if not _is_tradable(symbol_meta, symbol, ts, cfg):
                    continue
                fee_rate = symbol_meta[symbol]["fee"]
                dirs = symbol_meta[symbol].get("directions") or {}
                can_long = bool(dirs.get("can_long", True))
                can_short = bool(dirs.get("can_short", False))

                buy_intensity, sell_intensity = _extract_intensity(row)

                if can_long and buy_intensity > 0:
                    if cfg.select_fillable_only and _limit_fill_fraction(
                        row,
                        side="buy",
                        limit_price=float(row.buy_price),
                        cfg=cfg,
                    ) <= 0.0:
                        pass
                    else:
                        score = _edge_score_long(
                            row,
                            horizon=horizon,
                            config=cfg,
                            buy_intensity=buy_intensity,
                            fee_rate=fee_rate,
                        )
                        if score is not None and score >= cfg.min_edge:
                            candidates.append((score, symbol, "long", row, buy_intensity))

                if can_short and sell_intensity > 0:
                    if cfg.select_fillable_only and _limit_fill_fraction(
                        row,
                        side="sell",
                        limit_price=float(row.sell_price),
                        cfg=cfg,
                    ) <= 0.0:
                        pass
                    else:
                        score = _edge_score_short(
                            row,
                            horizon=horizon,
                            config=cfg,
                            sell_intensity=sell_intensity,
                            fee_rate=fee_rate,
                        )
                        if score is not None and score >= cfg.min_edge:
                            candidates.append((score, symbol, "short", row, sell_intensity))

            if candidates:
                candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
                selected_score, selected_symbol, direction, row, intensity = candidates[0]
                fee_rate = symbol_meta[selected_symbol]["fee"]
                if direction == "long":
                    fill_fraction = _limit_fill_fraction(
                        row,
                        side="buy",
                        limit_price=float(row.buy_price),
                        cfg=cfg,
                    )
                    if fill_fraction > 0.0:
                        max_leverage = _entry_max_leverage(symbol_meta, selected_symbol, side="long")
                        max_buy_notional = max(0.0, float(cash) * max_leverage)
                        max_buy = (
                            max_buy_notional / (row.buy_price * (1 + fee_rate)) if row.buy_price > 0 else 0.0
                        )
                        qty = intensity * max_buy * fill_fraction
                        qty = _cap_qty(selected_symbol, qty)
                        if qty > 0:
                            _execute_buy(ts, selected_symbol, qty, float(row.buy_price), fee_rate)
                            buy_filled = 1.0
                            current_price = float(row.close)
                else:
                    fill_fraction = _limit_fill_fraction(
                        row,
                        side="sell",
                        limit_price=float(row.sell_price),
                        cfg=cfg,
                    )
                    if fill_fraction > 0.0:
                        max_leverage = _entry_max_leverage(symbol_meta, selected_symbol, side="short")
                        max_short_notional = max(0.0, float(cash) * max_leverage)
                        max_short = (
                            max_short_notional / (row.sell_price * (1 + fee_rate)) if row.sell_price > 0 else 0.0
                        )
                        qty = intensity * max(0.0, max_short) * fill_fraction
                        qty = _cap_qty(selected_symbol, qty)
                        if qty > 0:
                            _execute_sell(ts, selected_symbol, qty, float(row.sell_price), fee_rate)
                            sell_filled = 1.0
                            current_price = float(row.close)

        if open_symbol and current_price is None:
            current_price = _resolve_close(open_symbol, None, last_close)

        if (
            open_symbol
            and inventory > 0
            and cfg.close_at_eod
            and symbol_meta[open_symbol]["asset_class"] == "stock"
            and ts in symbol_meta[open_symbol]["eod_ts"]
        ):
            fee_rate = symbol_meta[open_symbol]["fee"]
            row = _lookup_symbol_row(group, open_symbol)
            if row is not None:
                qty = _cap_qty(open_symbol, inventory)
                _execute_sell(ts, open_symbol, qty, float(row.close), fee_rate, reason="eod")
                sell_filled = 1.0
                forced_close = True
                current_price = float(row.close)
        elif (
            open_symbol
            and inventory < 0
            and cfg.close_at_eod
            and symbol_meta[open_symbol]["asset_class"] == "stock"
            and ts in symbol_meta[open_symbol]["eod_ts"]
        ):
            fee_rate = symbol_meta[open_symbol]["fee"]
            row = _lookup_symbol_row(group, open_symbol)
            if row is not None:
                qty = _cap_qty(open_symbol, abs(inventory))
                _execute_buy(ts, open_symbol, qty, float(row.close), fee_rate, reason="eod")
                buy_filled = 1.0
                forced_close = True
                current_price = float(row.close)

        portfolio_value = cash + (inventory * current_price if current_price is not None else 0.0)
        equity_values.append(portfolio_value)
        per_hour_rows.append(
            {
                "timestamp": ts,
                "cash": cash,
                "inventory": inventory,
                "open_symbol": open_symbol or "",
                "portfolio_value": portfolio_value,
                "buy_filled": buy_filled,
                "sell_filled": sell_filled,
                "selected_symbol": selected_symbol,
                "selected_score": selected_score,
                "financing_cost_paid": float(financing_cost_paid),
            }
        )
        prev_ts = ts

    equity_curve = pd.Series(equity_values, index=[row["timestamp"] for row in per_hour_rows])
    periods_per_year = _weighted_periods_per_year(symbol_meta, merged)
    metrics = _compute_metrics(equity_curve, periods_per_year)
    metrics["financing_cost_paid"] = float(financing_cost_paid)
    metrics["trade_count"] = float(len(trades))
    metrics["work_steal_count"] = float(sum(1 for trade in trades if trade.reason == "work_steal_exit"))
    return SelectorSimulationResult(
        equity_curve=equity_curve,
        per_hour=pd.DataFrame(per_hour_rows),
        trades=trades,
        final_cash=cash,
        final_inventory=inventory,
        open_symbol=open_symbol,
        metrics=metrics,
    )


def _run_multi_position_simulation(
    merged: pd.DataFrame,
    cfg: SelectionConfig,
    *,
    horizon: int,
) -> SelectorSimulationResult:
    """Multi-position variant: hold up to ``cfg.max_concurrent_positions`` simultaneously."""
    merged = merged.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    groups = merged.groupby("timestamp", sort=True)
    symbol_meta = _build_symbol_meta(merged, cfg)
    initial_position = _resolve_initial_position(merged, cfg)

    max_pos = max(1, int(cfg.max_concurrent_positions))
    cash = float(cfg.initial_cash)
    positions: Dict[str, float] = (
        {str(initial_position["symbol"]): float(initial_position["inventory"])}
        if initial_position is not None
        else {}
    )  # symbol -> signed qty
    position_open_ts: Dict[str, pd.Timestamp] = (
        {str(initial_position["symbol"]): initial_position["open_ts"]}
        if initial_position is not None
        else {}
    )
    last_close: Dict[str, float] = (
        {str(initial_position["symbol"]): float(initial_position["mark_price"])}
        if initial_position is not None
        else {}
    )
    financing_cost_paid = 0.0
    prev_ts: Optional[pd.Timestamp] = None
    equity_values: List[float] = []
    per_hour_rows: List[Dict[str, float | str]] = []
    trades: List[SelectorTradeRecord] = []

    max_hold_delta = None
    if cfg.max_hold_hours is not None and cfg.max_hold_hours > 0:
        max_hold_delta = pd.Timedelta(hours=int(cfg.max_hold_hours))

    def _record(ts, symbol, side, price, qty, reason=None):
        inv_after = positions.get(symbol, 0.0)
        trades.append(SelectorTradeRecord(
            timestamp=ts, symbol=symbol, side=side, price=float(price),
            quantity=float(qty), cash_after=float(cash), inventory_after=float(inv_after),
            reason=reason,
        ))

    for ts, group in groups:
        for row in group.itertuples(index=False):
            last_close[str(row.symbol)] = float(row.close)

        # Financing costs.
        if prev_ts is not None:
            dt_hours = float((ts - prev_ts).total_seconds() / 3600.0)
            if np.isfinite(dt_hours) and dt_hours > 0.0:
                if cfg.margin_interest_annual:
                    debt = max(0.0, -float(cash))
                    if debt > 0.0:
                        cost = debt * float(cfg.margin_interest_annual) / (365.0 * 24.0) * dt_hours
                        if np.isfinite(cost) and cost > 0.0:
                            cash -= cost
                            financing_cost_paid += cost
                if cfg.short_borrow_cost_annual:
                    for sym, qty in positions.items():
                        if qty < 0:
                            price = last_close.get(sym, 0.0)
                            if price > 0:
                                notional = abs(qty) * price
                                cost = notional * float(cfg.short_borrow_cost_annual) / (365.0 * 24.0) * dt_hours
                                if np.isfinite(cost) and cost > 0.0:
                                    cash -= cost
                                    financing_cost_paid += cost

        # --- Process exits for open positions ---
        closed_symbols: List[str] = []
        for sym in list(positions.keys()):
            qty = positions[sym]
            if abs(qty) < 1e-12:
                closed_symbols.append(sym)
                continue
            row = _lookup_symbol_row(group, sym)
            if row is None:
                continue
            fee_rate = symbol_meta[sym]["fee"]

            # Max hold check.
            if max_hold_delta is not None and cfg.force_close_on_max_hold:
                open_at = position_open_ts.get(sym)
                if open_at is not None and ts - open_at >= max_hold_delta:
                    if qty > 0:
                        cash += qty * float(row.close) * (1 - fee_rate)
                        _record(ts, sym, "sell", float(row.close), qty, "max_hold")
                    else:
                        cash -= abs(qty) * float(row.close) * (1 + fee_rate)
                        _record(ts, sym, "buy", float(row.close), abs(qty), "max_hold")
                    closed_symbols.append(sym)
                    continue

            # EOD force-close for stocks.
            if (
                cfg.close_at_eod
                and symbol_meta[sym]["asset_class"] == "stock"
                and ts in symbol_meta[sym]["eod_ts"]
            ):
                if qty > 0:
                    cash += qty * float(row.close) * (1 - fee_rate)
                    _record(ts, sym, "sell", float(row.close), qty, "eod")
                else:
                    cash -= abs(qty) * float(row.close) * (1 + fee_rate)
                    _record(ts, sym, "buy", float(row.close), abs(qty), "eod")
                closed_symbols.append(sym)
                continue

            # Normal exit: check fill conditions.
            if not _is_tradable(symbol_meta, sym, ts, cfg):
                continue
            buy_intensity, sell_intensity = _extract_intensity(row)
            if qty > 0 and sell_intensity > 0:
                fill_fraction = _limit_fill_fraction(
                    row,
                    side="sell",
                    limit_price=float(row.sell_price),
                    cfg=cfg,
                )
                sell_qty = sell_intensity * qty * fill_fraction
                if sell_qty > 0:
                    cash += sell_qty * float(row.sell_price) * (1 - fee_rate)
                    positions[sym] = qty - sell_qty
                    _record(ts, sym, "sell", float(row.sell_price), sell_qty)
                    if abs(positions[sym]) < 1e-12:
                        closed_symbols.append(sym)
            elif qty < 0 and buy_intensity > 0:
                fill_fraction = _limit_fill_fraction(
                    row,
                    side="buy",
                    limit_price=float(row.buy_price),
                    cfg=cfg,
                )
                cover_qty = buy_intensity * abs(qty) * fill_fraction
                if cover_qty > 0:
                    cash -= cover_qty * float(row.buy_price) * (1 + fee_rate)
                    positions[sym] = qty + cover_qty
                    _record(ts, sym, "buy", float(row.buy_price), cover_qty)
                    if abs(positions[sym]) < 1e-12:
                        closed_symbols.append(sym)

        for sym in closed_symbols:
            positions.pop(sym, None)
            position_open_ts.pop(sym, None)

        # --- Open new positions if slots available ---
        n_slots = max_pos - len(positions)
        if n_slots > 0:
            candidates: List[Tuple[float, str, str, object, float]] = []
            for row in group.itertuples(index=False):
                symbol = str(row.symbol)
                if symbol in positions:
                    continue
                if not _is_tradable(symbol_meta, symbol, ts, cfg):
                    continue
                fee_rate = symbol_meta[symbol]["fee"]
                dirs = symbol_meta[symbol].get("directions") or {}

                buy_intensity, sell_intensity = _extract_intensity(row)

                if bool(dirs.get("can_long", True)) and buy_intensity > 0:
                    if cfg.select_fillable_only and _limit_fill_fraction(
                        row,
                        side="buy",
                        limit_price=float(row.buy_price),
                        cfg=cfg,
                    ) <= 0.0:
                        pass
                    else:
                        score = _edge_score_long(
                            row,
                            horizon=horizon,
                            config=cfg,
                            buy_intensity=buy_intensity,
                            fee_rate=fee_rate,
                        )
                        if score is not None and score >= cfg.min_edge:
                            candidates.append((score, symbol, "long", row, buy_intensity))

                if bool(dirs.get("can_short", False)) and sell_intensity > 0:
                    if cfg.select_fillable_only and _limit_fill_fraction(
                        row,
                        side="sell",
                        limit_price=float(row.sell_price),
                        cfg=cfg,
                    ) <= 0.0:
                        pass
                    else:
                        score = _edge_score_short(
                            row,
                            horizon=horizon,
                            config=cfg,
                            sell_intensity=sell_intensity,
                            fee_rate=fee_rate,
                        )
                        if score is not None and score >= cfg.min_edge:
                            candidates.append((score, symbol, "short", row, sell_intensity))

            candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
            for score, symbol, direction, row, intensity in candidates[:n_slots]:
                fee_rate = symbol_meta[symbol]["fee"]
                # Size each new position using an equal share of available cash.
                remaining_slots = max_pos - len(positions)
                if remaining_slots <= 0:
                    break
                slot_cash = max(0.0, float(cash)) / remaining_slots
                if direction == "long":
                    fill_fraction = _limit_fill_fraction(
                        row,
                        side="buy",
                        limit_price=float(row.buy_price),
                        cfg=cfg,
                    )
                    if fill_fraction <= 0.0:
                        continue
                    max_leverage = _entry_max_leverage(symbol_meta, symbol, side="long")
                    max_buy_notional = slot_cash * max_leverage
                    max_buy = max_buy_notional / (row.buy_price * (1 + fee_rate)) if row.buy_price > 0 else 0.0
                    qty = intensity * max_buy * fill_fraction
                    if qty > 0:
                        cost = qty * float(row.buy_price) * (1 + fee_rate)
                        cash -= cost
                        positions[symbol] = positions.get(symbol, 0.0) + qty
                        position_open_ts[symbol] = ts
                        _record(ts, symbol, "buy", float(row.buy_price), qty)
                else:
                    fill_fraction = _limit_fill_fraction(
                        row,
                        side="sell",
                        limit_price=float(row.sell_price),
                        cfg=cfg,
                    )
                    if fill_fraction <= 0.0:
                        continue
                    max_leverage = _entry_max_leverage(symbol_meta, symbol, side="short")
                    max_short_notional = slot_cash * max_leverage
                    max_short = max_short_notional / (row.sell_price * (1 + fee_rate)) if row.sell_price > 0 else 0.0
                    qty = intensity * max(0.0, max_short) * fill_fraction
                    if qty > 0:
                        proceeds = qty * float(row.sell_price) * (1 - fee_rate)
                        cash += proceeds
                        positions[symbol] = positions.get(symbol, 0.0) - qty
                        position_open_ts[symbol] = ts
                        _record(ts, symbol, "sell", float(row.sell_price), qty)

        # --- Compute equity ---
        portfolio_value = cash
        for sym, qty in positions.items():
            price = last_close.get(sym, 0.0)
            portfolio_value += qty * price
        equity_values.append(portfolio_value)

        open_syms = ",".join(sorted(positions.keys())) if positions else ""
        per_hour_rows.append({
            "timestamp": ts,
            "cash": cash,
            "inventory": sum(positions.values()),
            "open_symbol": open_syms,
            "portfolio_value": portfolio_value,
            "buy_filled": 0.0,
            "sell_filled": 0.0,
            "selected_symbol": "",
            "selected_score": 0.0,
            "financing_cost_paid": float(financing_cost_paid),
            "num_positions": len(positions),
        })
        prev_ts = ts

    equity_curve = pd.Series(equity_values, index=[row["timestamp"] for row in per_hour_rows])
    periods_per_year = _weighted_periods_per_year(symbol_meta, merged)
    metrics = _compute_metrics(equity_curve, periods_per_year)
    metrics["financing_cost_paid"] = float(financing_cost_paid)
    metrics["trade_count"] = float(len(trades))
    metrics["work_steal_count"] = float(sum(1 for trade in trades if trade.reason == "work_steal_exit"))

    total_inv = sum(positions.values())
    open_sym = ",".join(sorted(positions.keys())) if positions else None
    return SelectorSimulationResult(
        equity_curve=equity_curve,
        per_hour=pd.DataFrame(per_hour_rows),
        trades=trades,
        final_cash=cash,
        final_inventory=total_inv,
        open_symbol=open_sym,
        metrics=metrics,
    )


def _prepare_frame(
    bars: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    horizon: int,
    symbols: Optional[Sequence[str]],
) -> pd.DataFrame:
    bars = bars.copy()
    actions = actions.copy()
    if "timestamp" not in bars.columns or "timestamp" not in actions.columns:
        raise ValueError("Both bars and actions must include a timestamp column.")
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    actions["timestamp"] = pd.to_datetime(actions["timestamp"], utc=True)
    if "symbol" not in bars.columns:
        bars["symbol"] = "DEFAULT"
    if "symbol" not in actions.columns:
        actions["symbol"] = "DEFAULT"
    bars["symbol"] = bars["symbol"].astype(str).str.upper()
    actions["symbol"] = actions["symbol"].astype(str).str.upper()

    if symbols:
        allowed = {str(sym).upper() for sym in symbols}
        bars = bars[bars["symbol"].isin(allowed)]
        actions = actions[actions["symbol"].isin(allowed)]

    merged = bars.merge(actions, on=["timestamp", "symbol"], how="inner")
    _validate_merged_frame(merged, horizon=horizon)
    return merged


def _resolve_initial_position(merged: pd.DataFrame, cfg: SelectionConfig) -> Optional[Dict[str, object]]:
    inventory = float(getattr(cfg, "initial_inventory", 0.0) or 0.0)
    if abs(inventory) <= 1e-12:
        return None

    symbol_raw = getattr(cfg, "initial_symbol", None)
    symbol = str(symbol_raw or "").strip().upper()
    if not symbol:
        raise ValueError("initial_symbol is required when initial_inventory is non-zero.")
    if symbol not in set(merged["symbol"].astype(str).str.upper()):
        raise ValueError(f"initial_symbol '{symbol}' is not present in the merged frame.")

    symbol_rows = merged[merged["symbol"].astype(str).str.upper() == symbol]
    if symbol_rows.empty:
        raise ValueError(f"No rows available for initial_symbol '{symbol}'.")

    first_row = symbol_rows.iloc[0]
    mark_price = _safe_float(first_row.get("close"))
    if mark_price is None or mark_price <= 0.0:
        raise ValueError(f"Could not resolve a valid starting close for initial_symbol '{symbol}'.")

    open_price = _safe_float(getattr(cfg, "initial_open_price", None))
    if open_price is None or open_price <= 0.0:
        open_price = float(mark_price)

    open_ts_raw = getattr(cfg, "initial_open_ts", None)
    open_ts = pd.Timestamp(open_ts_raw) if open_ts_raw is not None else pd.Timestamp(first_row["timestamp"])
    if open_ts.tzinfo is None:
        open_ts = open_ts.tz_localize("UTC")
    else:
        open_ts = open_ts.tz_convert("UTC")

    return {
        "symbol": symbol,
        "inventory": float(inventory),
        "open_price": float(open_price),
        "open_ts": open_ts,
        "mark_price": float(mark_price),
    }


def _base_max_leverage(cfg: SelectionConfig, *, symbol: str, asset_class: str) -> float:
    attr = "max_leverage_crypto" if asset_class == "crypto" else "max_leverage_stock"
    value = float(getattr(cfg, attr))
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{attr} must be > 0 for {symbol} (asset_class={asset_class}), got {value}.")
    return value


def _directional_max_leverage(
    cfg: SelectionConfig,
    *,
    symbol: str,
    asset_class: str,
    side: str,
    fallback: float,
) -> float:
    attr = f"{side}_max_leverage_crypto" if asset_class == "crypto" else f"{side}_max_leverage_stock"
    raw_value = getattr(cfg, attr, None)
    if raw_value is None:
        return float(fallback)
    value = float(raw_value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{attr} must be >= 0 for {symbol} (asset_class={asset_class}), got {value}.")
    return value


def _entry_max_leverage(symbol_meta: Dict[str, Dict[str, object]], symbol: str, *, side: str) -> float:
    meta = symbol_meta[symbol]
    key = "long_max_leverage" if side == "long" else "short_max_leverage"
    return max(0.0, float(meta.get(key, meta.get("max_leverage", 1.0))))


def _build_symbol_meta(merged: pd.DataFrame, cfg: SelectionConfig) -> Dict[str, Dict[str, object]]:
    meta: Dict[str, Dict[str, object]] = {}
    fee_map = cfg.fee_by_symbol or {}
    periods_map = cfg.periods_per_year_by_symbol or {}
    for symbol, group in merged.groupby("symbol"):
        symbol = str(symbol).upper()
        asset_class = "crypto" if is_crypto_symbol(symbol) else "stock"
        fee_rate = float(fee_map.get(symbol, get_fee_for_symbol(symbol)))
        dirs = resolve_trade_directions(
            symbol,
            allow_short=bool(cfg.allow_short),
            long_only_symbols=cfg.long_only_symbols,
            short_only_symbols=cfg.short_only_symbols,
        )
        max_leverage = _base_max_leverage(cfg, symbol=symbol, asset_class=asset_class)
        long_max_leverage = _directional_max_leverage(
            cfg,
            symbol=symbol,
            asset_class=asset_class,
            side="long",
            fallback=max_leverage,
        )
        short_max_leverage = _directional_max_leverage(
            cfg,
            symbol=symbol,
            asset_class=asset_class,
            side="short",
            fallback=max_leverage,
        )
        if symbol in periods_map:
            periods_per_year = float(periods_map[symbol])
        else:
            periods_per_year = _infer_periods_per_year(group["timestamp"], asset_class)
        eod_ts = set()
        market_open_ts = set()
        if asset_class == "stock":
            ny = _to_new_york(group["timestamp"])
            eod_mask = _end_of_day_mask(ny)
            eod_ts = set(group.loc[eod_mask.values, "timestamp"])
            market_open_mask = _market_open_mask(ny)
            market_open_ts = set(group.loc[market_open_mask.values, "timestamp"])
        meta[symbol] = {
            "asset_class": asset_class,
            "fee": fee_rate,
            "periods_per_year": periods_per_year,
            "eod_ts": eod_ts,
            "market_open_ts": market_open_ts,
            "directions": {"can_long": bool(dirs.can_long), "can_short": bool(dirs.can_short)},
            "max_leverage": max_leverage,
            "long_max_leverage": long_max_leverage,
            "short_max_leverage": short_max_leverage,
        }
    return meta


def _is_tradable(symbol_meta: Dict[str, Dict[str, object]], symbol: str, ts: pd.Timestamp, cfg: SelectionConfig) -> bool:
    meta = symbol_meta.get(symbol)
    if not meta:
        return False
    if meta["asset_class"] == "crypto":
        return True
    if not cfg.enforce_market_hours:
        return True
    return ts in meta["market_open_ts"]


def _weighted_periods_per_year(symbol_meta: Dict[str, Dict[str, object]], merged: pd.DataFrame) -> float:
    if merged.empty:
        return 24 * 365
    weights = merged["symbol"].value_counts()
    total = float(weights.sum())
    if total <= 0:
        return 24 * 365
    weighted = 0.0
    for symbol, count in weights.items():
        meta = symbol_meta.get(str(symbol))
        if not meta:
            continue
        weighted += float(meta["periods_per_year"]) * float(count)
    return float(weighted / total) if total else 24 * 365


def _lookup_symbol_row(group: pd.DataFrame, symbol: str) -> Optional[object]:
    subset = group[group["symbol"] == symbol]
    if subset.empty:
        return None
    return next(subset.itertuples(index=False))


def _resolve_close(symbol: str, row: Optional[object], last_close: Dict[str, float]) -> Optional[float]:
    if row is not None:
        return float(row.close)
    if symbol in last_close:
        return float(last_close[symbol])
    return None


def _extract_intensity(row: object) -> Tuple[float, float]:
    buy_amount = getattr(row, "buy_amount", None)
    sell_amount = getattr(row, "sell_amount", None)
    trade_amount = getattr(row, "trade_amount", None)
    if buy_amount is None and trade_amount is None:
        raise ValueError("Actions must include buy_amount/sell_amount or trade_amount.")
    if buy_amount is None:
        buy_amount = trade_amount
    if sell_amount is None:
        sell_amount = trade_amount
    scale = 100.0 if max(float(buy_amount), float(sell_amount)) > 1.0 else 1.0
    buy_intensity = float(np.clip(float(buy_amount) / scale, 0.0, 1.0))
    sell_intensity = float(np.clip(float(sell_amount) / scale, 0.0, 1.0))
    return buy_intensity, sell_intensity


def _normalize_fill_model(cfg: SelectionConfig) -> str:
    model = str(getattr(cfg, "limit_fill_model", "binary") or "binary").strip().lower()
    if model not in {"binary", "penetration"}:
        raise ValueError(f"limit_fill_model must be one of ['binary', 'penetration'], got {cfg.limit_fill_model!r}.")
    return model


def _validate_fill_config(cfg: SelectionConfig) -> None:
    _normalize_fill_model(cfg)
    touch_fill_fraction = float(getattr(cfg, "touch_fill_fraction", 0.0) or 0.0)
    if not np.isfinite(touch_fill_fraction) or touch_fill_fraction < 0.0 or touch_fill_fraction > 1.0:
        raise ValueError(
            f"touch_fill_fraction must be in [0, 1], got {getattr(cfg, 'touch_fill_fraction', None)!r}."
        )


def _limit_fill_fraction(
    row: object,
    *,
    side: str,
    limit_price: float,
    cfg: SelectionConfig,
) -> float:
    if limit_price <= 0.0:
        return 0.0
    high = _safe_float(getattr(row, "high", None))
    low = _safe_float(getattr(row, "low", None))
    if high is None or low is None:
        return 0.0
    bar_high = max(float(high), float(low))
    bar_low = min(float(high), float(low))
    bar_margin = float(getattr(cfg, "bar_margin", 0.0) or 0.0)
    if side == "buy":
        threshold = float(limit_price) * (1.0 - bar_margin)
        if bar_low > threshold:
            return 0.0
        penetration = max(0.0, threshold - bar_low)
    elif side == "sell":
        threshold = float(limit_price) * (1.0 + bar_margin)
        if bar_high < threshold:
            return 0.0
        penetration = max(0.0, bar_high - threshold)
    else:
        raise ValueError(f"Unsupported fill side {side!r}.")

    if _normalize_fill_model(cfg) == "binary":
        return 1.0

    touch_fill_fraction = float(getattr(cfg, "touch_fill_fraction", 0.0) or 0.0)
    bar_range = max(0.0, bar_high - bar_low)
    denom = max(bar_range, abs(threshold) * 1e-6, 1e-12)
    fraction = penetration / denom if penetration > 0.0 else touch_fill_fraction
    if penetration > 0.0:
        fraction = max(touch_fill_fraction, fraction)
    return float(np.clip(fraction, 0.0, 1.0))


def _edge_score_long(
    row: object,
    *,
    horizon: int,
    config: SelectionConfig,
    buy_intensity: float,
    fee_rate: float,
) -> Optional[float]:
    buy_price = _safe_float(getattr(row, "buy_price", None))
    if buy_price is None or buy_price <= 0:
        return None
    pred_high = _safe_float(getattr(row, f"predicted_high_p50_h{int(horizon)}", None))
    pred_low = _safe_float(getattr(row, f"predicted_low_p50_h{int(horizon)}", None))
    pred_close = _safe_float(getattr(row, f"predicted_close_p50_h{int(horizon)}", None))
    if pred_high is None or pred_low is None or pred_close is None:
        return None

    mode = str(config.edge_mode or "high_low").lower()
    if mode == "close":
        upside = (pred_close - buy_price) / buy_price
        downside = 0.0
    elif mode == "high":
        upside = (pred_high - buy_price) / buy_price
        downside = 0.0
    elif mode == "high_low":
        upside = (pred_high - buy_price) / buy_price
        downside = max(0.0, (buy_price - pred_low) / buy_price)
    else:
        raise ValueError(f"Unsupported edge_mode '{config.edge_mode}'.")

    edge = upside - float(config.risk_weight) * downside - (2.0 * fee_rate)
    if not np.isfinite(edge):
        return None
    return float(edge * buy_intensity)


def _edge_score_short(
    row: object,
    *,
    horizon: int,
    config: SelectionConfig,
    sell_intensity: float,
    fee_rate: float,
) -> Optional[float]:
    sell_price = _safe_float(getattr(row, "sell_price", None))
    if sell_price is None or sell_price <= 0:
        return None
    pred_high = _safe_float(getattr(row, f"predicted_high_p50_h{int(horizon)}", None))
    pred_low = _safe_float(getattr(row, f"predicted_low_p50_h{int(horizon)}", None))
    pred_close = _safe_float(getattr(row, f"predicted_close_p50_h{int(horizon)}", None))
    if pred_high is None or pred_low is None or pred_close is None:
        return None

    mode = str(config.edge_mode or "high_low").lower()
    if mode == "close":
        upside = (sell_price - pred_close) / sell_price
        downside = 0.0
    elif mode == "high":
        upside = (sell_price - pred_low) / sell_price
        downside = 0.0
    elif mode == "high_low":
        upside = (sell_price - pred_low) / sell_price
        downside = max(0.0, (pred_high - sell_price) / sell_price)
    else:
        raise ValueError(f"Unsupported edge_mode '{config.edge_mode}'.")

    edge = upside - float(config.risk_weight) * downside - (2.0 * fee_rate)
    if not np.isfinite(edge):
        return None
    return float(edge * sell_intensity)


def _safe_float(value: object) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _compute_metrics(equity_curve: pd.Series, periods_per_year: float) -> Dict[str, float]:
    if equity_curve.empty:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "profit_factor": 0.0,
            "mean_hourly_return": 0.0,
            "volatility": 0.0,
            "pnl_smoothness": 0.0,
        }
    metrics = pnl_metrics(equity_curve=equity_curve.values, periods_per_year=periods_per_year)
    return {
        "total_return": float(metrics.total_return),
        "annualized_return": float(metrics.annualized_return),
        "sharpe": float(metrics.sharpe),
        "sortino": float(metrics.sortino),
        "max_drawdown": float(metrics.max_drawdown),
        "calmar": float(metrics.calmar),
        "profit_factor": float(metrics.profit_factor),
        "mean_hourly_return": float(metrics.avg_return),
        "volatility": float(metrics.volatility),
        "pnl_smoothness": float(compute_pnl_smoothness_from_equity(equity_curve.values)),
    }


__all__ = [
    "SelectionConfig",
    "SelectorSimulationResult",
    "SelectorTradeRecord",
    "run_best_trade_simulation",
    "run_best_trade_simulation_merged",
]
