from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Dict, List, Optional, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.date_utils import is_nyse_open_on_date
from src.fees import get_fee_for_symbol
from src.metrics_utils import annualized_sortino, compute_step_returns
from src.symbol_utils import is_crypto_symbol

NEW_YORK = ZoneInfo("America/New_York")


@dataclass
class SimulationConfig:
    initial_cash: float = 10_000.0
    total_cash: Optional[float] = None
    symbols: Optional[Sequence[str]] = None
    enable_probe_mode: bool = False
    probe_notional: float = 1.0
    max_hold_hours: Optional[int] = None
    force_close_on_max_hold: bool = True
    enforce_market_hours: bool = True
    close_at_eod: bool = True
    fee_by_symbol: Optional[Dict[str, float]] = None
    periods_per_year_by_symbol: Optional[Dict[str, float]] = None
    allow_intrabar_round_trips: bool = False
    max_round_trips_per_bar: int = 1


@dataclass
class TradeRecord:
    timestamp: pd.Timestamp
    side: str
    price: float
    quantity: float
    cash_after: float
    inventory_after: float
    reason: Optional[str] = None


@dataclass
class SymbolSimulationResult:
    symbol: str
    equity_curve: pd.Series
    trades: List[TradeRecord]
    per_hour: pd.DataFrame
    final_cash: float
    final_inventory: float
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class MarketSimulationResult:
    per_symbol: Dict[str, SymbolSimulationResult]
    combined_equity: pd.Series
    metrics: Dict[str, float]


class AlpacaMarketSimulator:
    def __init__(self, config: Optional[SimulationConfig] = None) -> None:
        self.config = config or SimulationConfig()

    def run(self, bars: pd.DataFrame, actions: pd.DataFrame) -> MarketSimulationResult:
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

        symbols = sorted(set(bars["symbol"]).intersection(actions["symbol"]))
        if self.config.symbols:
            allowed = {s.upper() for s in self.config.symbols}
            symbols = [s for s in symbols if s in allowed]
        if not symbols:
            raise ValueError("No overlapping symbols between bars and actions.")

        cash_per_symbol = self._cash_per_symbol(len(symbols))
        results: Dict[str, SymbolSimulationResult] = {}
        combined_equity: Optional[pd.Series] = None
        combined_periods: List[float] = []

        for symbol in symbols:
            sym_bars = bars[bars["symbol"] == symbol]
            sym_actions = actions[actions["symbol"] == symbol]
            meta = self._symbol_meta(symbol, sym_bars)
            result = self._run_single(symbol, sym_bars, sym_actions, cash_per_symbol, meta)
            results[symbol] = result
            combined_periods.append(meta["periods_per_year"])
            if combined_equity is None:
                combined_equity = result.equity_curve
            else:
                combined_equity = combined_equity.add(result.equity_curve, fill_value=0.0)

        if combined_equity is None:
            raise RuntimeError("Failed to compute combined equity curve.")

        combined_periods_per_year = float(np.mean(combined_periods)) if combined_periods else 24 * 365
        metrics = self._compute_metrics(combined_equity, combined_periods_per_year)
        return MarketSimulationResult(per_symbol=results, combined_equity=combined_equity, metrics=metrics)

    def _cash_per_symbol(self, count: int) -> float:
        if self.config.total_cash is not None:
            return float(self.config.total_cash) / max(1, count)
        return float(self.config.initial_cash)

    def _symbol_meta(self, symbol: str, bars: pd.DataFrame) -> Dict[str, float | str]:
        symbol = symbol.upper()
        asset_class = "crypto" if is_crypto_symbol(symbol) else "stock"
        fee_map = self.config.fee_by_symbol or {}
        maker_fee = float(fee_map.get(symbol, get_fee_for_symbol(symbol)))
        periods_map = self.config.periods_per_year_by_symbol or {}
        if symbol in periods_map:
            periods_per_year = float(periods_map[symbol])
        else:
            periods_per_year = _infer_periods_per_year(bars["timestamp"], asset_class)
        return {
            "asset_class": asset_class,
            "maker_fee": maker_fee,
            "periods_per_year": periods_per_year,
        }

    def _run_single(
        self,
        symbol: str,
        bars: pd.DataFrame,
        actions: pd.DataFrame,
        initial_cash: float,
        meta: Dict[str, float | str],
    ) -> SymbolSimulationResult:
        frame = self._prepare_frame(bars, actions)
        cash = float(initial_cash)
        normal_inventory = 0.0
        probe_inventory = 0.0
        normal_cost_basis = 0.0
        probe_cost_basis = 0.0
        normal_open_ts: Optional[pd.Timestamp] = None
        probe_open_ts: Optional[pd.Timestamp] = None
        risk_mode = "normal"
        probe_realized_pnl = 0.0
        equity_values: List[float] = []
        per_hour_rows: List[Dict[str, float | str]] = []
        trades: List[TradeRecord] = []

        max_hold_delta = None
        if self.config.max_hold_hours is not None and self.config.max_hold_hours > 0:
            max_hold_delta = pd.Timedelta(hours=int(self.config.max_hold_hours))

        maker_fee = float(meta["maker_fee"])
        asset_class = str(meta["asset_class"])
        is_stock = asset_class == "stock"

        ny_timestamps = _to_new_york(frame["timestamp"]).reset_index(drop=True)
        eod_mask = _end_of_day_mask(ny_timestamps) if is_stock else pd.Series(False, index=frame.index)
        market_open_mask = (
            _market_open_mask(ny_timestamps) if (is_stock and self.config.enforce_market_hours) else None
        )

        def total_inventory() -> float:
            return normal_inventory + probe_inventory

        def _record_trade(side: str, price: float, quantity: float, reason: Optional[str] = None) -> None:
            trades.append(
                TradeRecord(
                    timestamp=row.timestamp,
                    side=side,
                    price=float(price),
                    quantity=float(quantity),
                    cash_after=cash,
                    inventory_after=total_inventory(),
                    reason=reason,
                )
            )

        def _execute_buy(quantity: float, price: float, *, is_probe: bool) -> None:
            nonlocal cash, normal_inventory, probe_inventory, normal_cost_basis, probe_cost_basis
            nonlocal normal_open_ts, probe_open_ts
            if quantity <= 0:
                return
            cost = quantity * price * (1 + maker_fee)
            cash -= cost
            if is_probe:
                if probe_inventory <= 0:
                    probe_open_ts = row.timestamp
                    probe_cost_basis = price * (1 + maker_fee)
                else:
                    probe_cost_basis = (probe_cost_basis * probe_inventory + price * (1 + maker_fee) * quantity) / (
                        probe_inventory + quantity
                    )
                probe_inventory += quantity
            else:
                if normal_inventory <= 0:
                    normal_open_ts = row.timestamp
                    normal_cost_basis = price * (1 + maker_fee)
                else:
                    normal_cost_basis = (normal_cost_basis * normal_inventory + price * (1 + maker_fee) * quantity) / (
                        normal_inventory + quantity
                    )
                normal_inventory += quantity
            _record_trade("buy", price, quantity)

        def _execute_sell(quantity: float, price: float, *, is_probe: bool, reason: Optional[str] = None) -> float:
            nonlocal cash, normal_inventory, probe_inventory, normal_cost_basis, probe_cost_basis
            nonlocal normal_open_ts, probe_open_ts, risk_mode, probe_realized_pnl
            if quantity <= 0:
                return 0.0
            proceeds = quantity * price * (1 - maker_fee)
            cash += proceeds
            realized = 0.0
            if is_probe:
                realized = (price * (1 - maker_fee) - probe_cost_basis) * quantity
                probe_inventory -= quantity
                if probe_inventory <= 0:
                    probe_inventory = 0.0
                    probe_cost_basis = 0.0
                    probe_open_ts = None
                    probe_realized_pnl += realized
                    if self.config.enable_probe_mode and risk_mode == "probe":
                        if probe_realized_pnl > 0:
                            risk_mode = "normal"
                        probe_realized_pnl = 0.0
                else:
                    probe_realized_pnl += realized
            else:
                realized = (price * (1 - maker_fee) - normal_cost_basis) * quantity
                normal_inventory -= quantity
                if normal_inventory <= 0:
                    normal_inventory = 0.0
                    normal_cost_basis = 0.0
                    normal_open_ts = None
                if self.config.enable_probe_mode and realized < 0:
                    risk_mode = "probe"
                    probe_realized_pnl = 0.0
            _record_trade("sell", price, quantity, reason=reason)
            return realized

        for idx, row in enumerate(frame.itertuples(index=False)):
            forced_close = False
            is_market_open = True
            if market_open_mask is not None:
                is_market_open = bool(market_open_mask.iloc[idx])

            if max_hold_delta is not None and self.config.force_close_on_max_hold:
                if normal_inventory > 0 and normal_open_ts is not None:
                    if row.timestamp - normal_open_ts >= max_hold_delta:
                        _execute_sell(normal_inventory, float(row.close), is_probe=False, reason="max_hold")
                        forced_close = True
                if probe_inventory > 0 and probe_open_ts is not None:
                    if row.timestamp - probe_open_ts >= max_hold_delta:
                        _execute_sell(probe_inventory, float(row.close), is_probe=True, reason="max_hold")
                        forced_close = True

            buy_intensity, sell_intensity = self._extract_intensity(row)
            if not is_market_open:
                buy_intensity = 0.0
                sell_intensity = 0.0
            buy_fill = bool(row.low <= row.buy_price and buy_intensity > 0)
            sell_fill = bool(row.high >= row.sell_price and sell_intensity > 0)
            executed_buy = 0.0
            executed_sell = 0.0
            cycle_count = 0
            cycle_qty = 0.0
            cycle_profit = 0.0

            if (
                self.config.allow_intrabar_round_trips
                and self.config.max_round_trips_per_bar > 1
                and buy_fill
                and sell_fill
                and row.buy_price > 0
                and row.sell_price > row.buy_price
            ):
                cycle_intensity = min(buy_intensity, sell_intensity)
                if cycle_intensity > 0:
                    max_buy = cash / (row.buy_price * (1 + maker_fee))
                    cycle_qty = cycle_intensity * max_buy
                    spread = float(row.sell_price - row.buy_price)
                    range_move = float(row.high - row.low)
                    if spread > 0 and range_move > 0 and cycle_qty > 0 and max_buy > 0:
                        bounce_count = max(1, int(range_move / spread))
                        cycle_count = min(int(self.config.max_round_trips_per_bar), bounce_count)
                        cycle_profit_per_unit = float(row.sell_price * (1 - maker_fee) - row.buy_price * (1 + maker_fee))
                        cycle_cost = cycle_qty * row.buy_price * (1 + maker_fee)
                        if cash < cycle_cost:
                            cycle_count = 0
                        else:
                            if cycle_profit_per_unit < 0:
                                loss_per_cycle = -cycle_profit_per_unit * cycle_qty
                                if loss_per_cycle > 0:
                                    max_cycles_cash = 1 + int(max(0.0, (cash - cycle_cost) / loss_per_cycle))
                                    cycle_count = min(cycle_count, max_cycles_cash)
                            if cycle_count > 0:
                                cycle_profit = cycle_count * cycle_qty * cycle_profit_per_unit
                                cash += cycle_profit
                                total_cycle_qty = cycle_qty * cycle_count
                                _record_trade("buy", float(row.buy_price), total_cycle_qty, reason="intrabar_cycle")
                                _record_trade("sell", float(row.sell_price), total_cycle_qty, reason="intrabar_cycle")
                    buy_intensity = max(0.0, buy_intensity - cycle_intensity)
                    sell_intensity = max(0.0, sell_intensity - cycle_intensity)
            if buy_fill:
                max_buy = cash / (row.buy_price * (1 + maker_fee)) if row.buy_price > 0 else 0.0
                executed_buy = buy_intensity * max_buy
                if self.config.enable_probe_mode and risk_mode == "probe":
                    probe_cap = 0.0
                    if self.config.probe_notional and row.buy_price > 0:
                        probe_cap = self.config.probe_notional / (row.buy_price * (1 + maker_fee))
                    executed_buy = min(executed_buy, probe_cap)
            if sell_fill:
                executed_sell = sell_intensity * max(0.0, total_inventory())

            if executed_buy > 0:
                _execute_buy(
                    executed_buy,
                    float(row.buy_price),
                    is_probe=self.config.enable_probe_mode and risk_mode == "probe",
                )
            if executed_sell > 0 and total_inventory() > 0:
                sell_qty = min(executed_sell, total_inventory())
                probe_sell = min(probe_inventory, sell_qty)
                normal_sell = sell_qty - probe_sell
                if probe_sell > 0:
                    _execute_sell(probe_sell, float(row.sell_price), is_probe=True)
                if normal_sell > 0:
                    _execute_sell(normal_sell, float(row.sell_price), is_probe=False)

            if is_stock and self.config.close_at_eod and bool(eod_mask.iloc[idx]) and total_inventory() > 0:
                _execute_sell(total_inventory(), float(row.close), is_probe=False, reason="eod")
                forced_close = True

            portfolio_value = cash + total_inventory() * row.close
            equity_values.append(portfolio_value)
            per_hour_rows.append(
                {
                    "timestamp": row.timestamp,
                    "portfolio_value": portfolio_value,
                    "cash": cash,
                    "inventory": total_inventory(),
                    "normal_inventory": normal_inventory,
                    "probe_inventory": probe_inventory,
                    "buy_filled": float(executed_buy > 0),
                    "sell_filled": float(executed_sell > 0),
                    "forced_close": float(forced_close),
                    "risk_mode": risk_mode,
                    "market_open": float(is_market_open),
                    "cycle_count": float(cycle_count),
                    "cycle_qty": float(cycle_qty),
                    "cycle_profit": float(cycle_profit),
                }
            )

        equity_curve = pd.Series(equity_values, index=frame["timestamp"].values)
        per_hour = pd.DataFrame(per_hour_rows)
        metrics = self._compute_metrics(equity_curve, float(meta["periods_per_year"]))
        return SymbolSimulationResult(
            symbol=symbol,
            equity_curve=equity_curve,
            trades=trades,
            per_hour=per_hour,
            final_cash=cash,
            final_inventory=total_inventory(),
            metrics=metrics,
        )

    def _prepare_frame(self, bars: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
        bars = bars.copy()
        actions = actions.copy()
        bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
        actions["timestamp"] = pd.to_datetime(actions["timestamp"], utc=True)
        merged = bars.merge(actions, on=["timestamp", "symbol"], how="inner")
        if merged.empty:
            raise ValueError("Merged dataframe is empty; ensure actions cover the provided bars.")
        required = {"high", "low", "close", "buy_price", "sell_price"}
        missing = required - set(merged.columns)
        if missing:
            raise ValueError(f"Missing required columns in merged frame: {sorted(missing)}")
        return merged.sort_values("timestamp").reset_index(drop=True)

    def _extract_intensity(self, row: object) -> tuple[float, float]:
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

    @staticmethod
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


# ------------------------------------------------------------------


def _to_new_york(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, utc=True, errors="coerce")
    if ts.empty:
        return pd.Series(dtype="datetime64[ns, America/New_York]")
    try:
        return ts.dt.tz_convert(NEW_YORK)
    except Exception:
        return ts


def _end_of_day_mask(ny_timestamps: pd.Series) -> pd.Series:
    if ny_timestamps.empty:
        return pd.Series(dtype=bool)
    dates = ny_timestamps.dt.date
    next_dates = dates.shift(-1)
    return dates.ne(next_dates).fillna(True)


def _market_open_mask(ny_timestamps: pd.Series) -> pd.Series:
    if ny_timestamps.empty:
        return pd.Series(dtype=bool)
    dates = ny_timestamps.dt.date
    unique_dates = sorted(set(dates))
    date_open: Dict[date, bool] = {}
    for d in unique_dates:
        midday = datetime.combine(d, time(12, 0), tzinfo=NEW_YORK)
        date_open[d] = bool(is_nyse_open_on_date(midday))
    market_open = time(9, 30)
    market_close = time(16, 0)
    mask = []
    for ts in ny_timestamps:
        if ts is pd.NaT:
            mask.append(False)
            continue
        if not date_open.get(ts.date(), False):
            mask.append(False)
            continue
        t = ts.timetz()
        mask.append(market_open <= t <= market_close)
    return pd.Series(mask, index=ny_timestamps.index)


def _infer_periods_per_year(timestamps: pd.Series, asset_class: str) -> float:
    if asset_class == "crypto":
        return float(24 * 365)
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce")
    ts = ts.dropna()
    if ts.empty:
        return float(252 * 7)
    try:
        ts_local = ts.dt.tz_convert(NEW_YORK)
    except Exception:
        ts_local = ts
    counts = ts_local.dt.date.value_counts()
    avg_bars = float(counts.mean()) if not counts.empty else 0.0
    if not np.isfinite(avg_bars) or avg_bars <= 0:
        avg_bars = 7.0
    avg_bars = min(max(avg_bars, 1.0), 24.0)
    return float(avg_bars * 252)


__all__ = [
    "AlpacaMarketSimulator",
    "MarketSimulationResult",
    "SimulationConfig",
    "SymbolSimulationResult",
    "TradeRecord",
]
