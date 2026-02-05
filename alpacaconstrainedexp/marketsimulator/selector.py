from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.fees import get_fee_for_symbol
from src.metrics_utils import annualized_sortino, compute_step_returns
from src.symbol_utils import is_crypto_symbol

from newnanoalpacahourlyexp.marketsimulator.simulator import (
    _end_of_day_mask,
    _infer_periods_per_year,
    _market_open_mask,
    _to_new_york,
)


@dataclass
class SelectionConfig:
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
    periods_per_year_by_symbol: Optional[Dict[str, float]] = None
    long_symbols: Optional[Sequence[str]] = None
    short_symbols: Optional[Sequence[str]] = None
    allow_short: bool = True


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
    open_side: Optional[str]
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
    if merged.empty:
        raise ValueError("Merged dataframe is empty; ensure actions cover the provided bars.")

    merged = merged.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    groups = merged.groupby("timestamp", sort=True)

    symbol_meta = _build_symbol_meta(merged, cfg)
    long_set = _normalize_set(cfg.long_symbols)
    short_set = _normalize_set(cfg.short_symbols)

    cash = float(cfg.initial_cash)
    inventory = 0.0
    cost_basis = 0.0
    open_symbol: Optional[str] = None
    open_side: Optional[str] = None
    open_ts: Optional[pd.Timestamp] = None
    last_close: Dict[str, float] = {}
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
                inventory_after=float(_signed_inventory(open_side, inventory)),
                reason=reason,
            )
        )

    def _execute_buy(
        ts: pd.Timestamp,
        symbol: str,
        qty: float,
        price: float,
        fee_rate: float,
        *,
        side: str,
        reason: Optional[str] = None,
    ) -> None:
        nonlocal cash, inventory, cost_basis, open_symbol, open_side, open_ts
        if qty <= 0:
            return
        cost = qty * price * (1 + fee_rate)
        cash -= cost
        if inventory <= 0 or open_side != side:
            cost_basis = price * (1 + fee_rate)
            open_ts = ts
        else:
            cost_basis = (cost_basis * inventory + price * (1 + fee_rate) * qty) / (inventory + qty)
        inventory += qty
        open_symbol = symbol
        open_side = side
        _record_trade(ts=ts, symbol=symbol, side="buy", price=price, qty=qty, reason=reason)

    def _execute_sell(
        ts: pd.Timestamp,
        symbol: str,
        qty: float,
        price: float,
        fee_rate: float,
        *,
        reason: Optional[str] = None,
    ) -> None:
        nonlocal cash, inventory, cost_basis, open_symbol, open_side, open_ts
        if qty <= 0:
            return
        proceeds = qty * price * (1 - fee_rate)
        cash += proceeds
        inventory -= qty
        if inventory <= 1e-8:
            inventory = 0.0
            cost_basis = 0.0
            open_symbol = None
            open_side = None
            open_ts = None
        _record_trade(ts=ts, symbol=symbol, side="sell", price=price, qty=qty, reason=reason)

    def _execute_short(
        ts: pd.Timestamp,
        symbol: str,
        qty: float,
        price: float,
        fee_rate: float,
        *,
        reason: Optional[str] = None,
    ) -> None:
        nonlocal cash, inventory, cost_basis, open_symbol, open_side, open_ts
        if qty <= 0:
            return
        proceeds = qty * price * (1 - fee_rate)
        cash += proceeds
        if inventory <= 0 or open_side != "short":
            cost_basis = price * (1 - fee_rate)
            open_ts = ts
        else:
            cost_basis = (cost_basis * inventory + price * (1 - fee_rate) * qty) / (inventory + qty)
        inventory += qty
        open_symbol = symbol
        open_side = "short"
        _record_trade(ts=ts, symbol=symbol, side="sell_short", price=price, qty=qty, reason=reason)

    def _execute_cover(
        ts: pd.Timestamp,
        symbol: str,
        qty: float,
        price: float,
        fee_rate: float,
        *,
        reason: Optional[str] = None,
    ) -> None:
        nonlocal cash, inventory, cost_basis, open_symbol, open_side, open_ts
        if qty <= 0:
            return
        cost = qty * price * (1 + fee_rate)
        cash -= cost
        inventory -= qty
        if inventory <= 1e-8:
            inventory = 0.0
            cost_basis = 0.0
            open_symbol = None
            open_side = None
            open_ts = None
        _record_trade(ts=ts, symbol=symbol, side="buy_to_cover", price=price, qty=qty, reason=reason)

    for ts, group in groups:
        buy_filled = 0.0
        sell_filled = 0.0
        selected_symbol = ""
        selected_score = 0.0
        selected_side = ""
        forced_close = False
        sold_this_step = False

        for row in group.itertuples(index=False):
            last_close[str(row.symbol)] = float(row.close)

        if open_symbol and max_hold_delta is not None and cfg.force_close_on_max_hold and open_ts is not None:
            if ts - open_ts >= max_hold_delta:
                row = _lookup_symbol_row(group, open_symbol)
                if row is not None:
                    fee_rate = symbol_meta[open_symbol]["fee"]
                    if open_side == "short":
                        _execute_cover(ts, open_symbol, inventory, float(row.close), fee_rate, reason="max_hold")
                    else:
                        _execute_sell(ts, open_symbol, inventory, float(row.close), fee_rate, reason="max_hold")
                    sell_filled = 1.0
                    forced_close = True
                    sold_this_step = True

        current_price: Optional[float] = None
        if open_symbol and inventory > 0:
            row = _lookup_symbol_row(group, open_symbol)
            if row is not None and not forced_close:
                buy_intensity, sell_intensity = _extract_intensity(row)
                if open_side == "short":
                    if not _is_tradable(symbol_meta, open_symbol, ts, cfg) or not _can_short(
                        open_symbol, symbol_meta, short_set, cfg.allow_short
                    ):
                        buy_intensity = 0.0
                    buy_fill = bool(row.low <= row.buy_price and buy_intensity > 0)
                    if buy_fill:
                        qty = buy_intensity * inventory
                        fee_rate = symbol_meta[open_symbol]["fee"]
                        _execute_cover(ts, open_symbol, min(qty, inventory), float(row.buy_price), fee_rate)
                        sell_filled = 1.0
                        sold_this_step = True
                else:
                    if not _is_tradable(symbol_meta, open_symbol, ts, cfg) or not _can_long(
                        open_symbol, symbol_meta, long_set
                    ):
                        sell_intensity = 0.0
                    sell_fill = bool(row.high >= row.sell_price and sell_intensity > 0)
                    if sell_fill:
                        qty = sell_intensity * inventory
                        fee_rate = symbol_meta[open_symbol]["fee"]
                        _execute_sell(ts, open_symbol, min(qty, inventory), float(row.sell_price), fee_rate)
                        sell_filled = 1.0
                        sold_this_step = True
            current_price = _resolve_close(open_symbol, row, last_close)

        if open_symbol is None and not forced_close and (cfg.allow_reentry_same_bar or not sold_this_step):
            candidates: List[Tuple[float, str, str, object, float, float]] = []
            for row in group.itertuples(index=False):
                symbol = str(row.symbol)
                if not _is_tradable(symbol_meta, symbol, ts, cfg):
                    continue

                buy_intensity, sell_intensity = _extract_intensity(row)
                fee_rate = symbol_meta[symbol]["fee"]

                if buy_intensity > 0 and _can_long(symbol, symbol_meta, long_set):
                    if row.low <= row.buy_price:
                        score = _edge_score_long(
                            row,
                            horizon=horizon,
                            config=cfg,
                            buy_intensity=buy_intensity,
                            fee_rate=fee_rate,
                        )
                        if score is not None and score >= cfg.min_edge:
                            candidates.append((score, symbol, "long", row, buy_intensity, fee_rate))

                if sell_intensity > 0 and _can_short(symbol, symbol_meta, short_set, cfg.allow_short):
                    if row.high >= row.sell_price:
                        score = _edge_score_short(
                            row,
                            horizon=horizon,
                            config=cfg,
                            sell_intensity=sell_intensity,
                            fee_rate=fee_rate,
                        )
                        if score is not None and score >= cfg.min_edge:
                            candidates.append((score, symbol, "short", row, sell_intensity, fee_rate))

            if candidates:
                candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
                selected_score, selected_symbol, selected_side, row, intensity, fee_rate = candidates[0]
                if selected_side == "short":
                    max_short = cash / (row.sell_price * (1 + fee_rate)) if row.sell_price > 0 else 0.0
                    qty = float(intensity) * max_short
                    if qty > 0:
                        _execute_short(ts, selected_symbol, qty, float(row.sell_price), fee_rate)
                        buy_filled = 1.0
                        current_price = float(row.close)
                else:
                    max_buy = cash / (row.buy_price * (1 + fee_rate)) if row.buy_price > 0 else 0.0
                    qty = float(intensity) * max_buy
                    if qty > 0:
                        _execute_buy(ts, selected_symbol, qty, float(row.buy_price), fee_rate, side="long")
                        buy_filled = 1.0
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
                if open_side == "short":
                    _execute_cover(ts, open_symbol, inventory, float(row.close), fee_rate, reason="eod")
                else:
                    _execute_sell(ts, open_symbol, inventory, float(row.close), fee_rate, reason="eod")
                sell_filled = 1.0
                forced_close = True
                current_price = float(row.close)

        signed_inventory = _signed_inventory(open_side, inventory)
        portfolio_value = cash + (signed_inventory * current_price if current_price is not None else 0.0)
        equity_values.append(portfolio_value)
        per_hour_rows.append(
            {
                "timestamp": ts,
                "cash": cash,
                "inventory": signed_inventory,
                "open_symbol": open_symbol or "",
                "open_side": open_side or "",
                "portfolio_value": portfolio_value,
                "buy_filled": buy_filled,
                "sell_filled": sell_filled,
                "selected_symbol": selected_symbol,
                "selected_score": selected_score,
                "selected_side": selected_side,
            }
        )

    equity_curve = pd.Series(equity_values, index=[row["timestamp"] for row in per_hour_rows])
    periods_per_year = _weighted_periods_per_year(symbol_meta, merged)
    metrics = _compute_metrics(equity_curve, periods_per_year)
    return SelectorSimulationResult(
        equity_curve=equity_curve,
        per_hour=pd.DataFrame(per_hour_rows),
        trades=trades,
        final_cash=cash,
        final_inventory=_signed_inventory(open_side, inventory),
        open_symbol=open_symbol,
        open_side=open_side,
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
    return merged


def _build_symbol_meta(merged: pd.DataFrame, cfg: SelectionConfig) -> Dict[str, Dict[str, object]]:
    meta: Dict[str, Dict[str, object]] = {}
    fee_map = cfg.fee_by_symbol or {}
    periods_map = cfg.periods_per_year_by_symbol or {}
    for symbol, group in merged.groupby("symbol"):
        symbol = str(symbol).upper()
        asset_class = "crypto" if is_crypto_symbol(symbol) else "stock"
        fee_rate = float(fee_map.get(symbol, get_fee_for_symbol(symbol)))
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
        }
    return meta


def _normalize_set(symbols: Optional[Sequence[str]]) -> Optional[set[str]]:
    if symbols is None:
        return None
    cleaned = {str(sym).upper() for sym in symbols if str(sym).strip()}
    return cleaned


def _can_long(symbol: str, symbol_meta: Dict[str, Dict[str, object]], long_set: Optional[set[str]]) -> bool:
    if symbol not in symbol_meta:
        return False
    if long_set is None:
        return True
    return symbol in long_set


def _can_short(
    symbol: str,
    symbol_meta: Dict[str, Dict[str, object]],
    short_set: Optional[set[str]],
    allow_short: bool,
) -> bool:
    if not allow_short:
        return False
    meta = symbol_meta.get(symbol)
    if not meta:
        return False
    if meta.get("asset_class") == "crypto":
        return False
    if short_set is None:
        return True
    return symbol in short_set


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
        profit = (sell_price - pred_close) / sell_price
        risk = 0.0
    elif mode == "high":
        profit = (sell_price - pred_low) / sell_price
        risk = 0.0
    elif mode == "high_low":
        profit = (sell_price - pred_low) / sell_price
        risk = max(0.0, (pred_high - sell_price) / sell_price)
    else:
        raise ValueError(f"Unsupported edge_mode '{config.edge_mode}'.")

    edge = profit - float(config.risk_weight) * risk - (2.0 * fee_rate)
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


def _signed_inventory(side: Optional[str], qty: float) -> float:
    if not side or qty <= 0:
        return 0.0
    return -qty if side == "short" else qty


def _compute_metrics(equity_curve: pd.Series, periods_per_year: float) -> Dict[str, float]:
    if equity_curve.empty:
        return {"total_return": 0.0, "sortino": 0.0}
    returns = compute_step_returns(equity_curve.values)
    sortino = annualized_sortino(returns, periods_per_year=periods_per_year)
    total_return = (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]
    return {
        "total_return": float(total_return),
        "sortino": float(sortino),
        "mean_hourly_return": float(returns.mean() if returns.size else 0.0),
    }


__all__ = [
    "SelectionConfig",
    "SelectorSimulationResult",
    "SelectorTradeRecord",
    "run_best_trade_simulation",
]
