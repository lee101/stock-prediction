from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.tradinglib.metrics import PnlMetrics, pnl_metrics


@dataclass(frozen=True)
class DailyLevelSimulationConfig:
    """Configuration for simulating daily buy/sell level trading on intraday bars."""

    initial_cash: float = 10_000.0
    maker_fee: float = 0.0
    allocation_fraction: float = 1.0
    min_spread_pct: float = 0.0
    min_spread_mode: str = "skip"  # skip|widen|none
    close_at_eod: bool = True
    allow_reentry_same_bar: bool = False
    allow_roundtrip_same_bar: bool = False
    intrabar_assumption: str = "none"  # none|ohlc
    periods_per_year: float = 24.0 * 365.0
    record_per_bar: bool = True
    stop_loss_pct: float = 0.0
    stop_loss_lockout_until_next_day: bool = False


@dataclass(frozen=True)
class DailyLevelTrade:
    timestamp: pd.Timestamp
    side: str  # buy|sell
    price: float
    quantity: float
    cash_after: float
    base_after: float
    reason: str


@dataclass(frozen=True)
class DailyLevelSimulationResult:
    equity_curve: pd.Series
    per_bar: pd.DataFrame
    trades: List[DailyLevelTrade]
    metrics: PnlMetrics


def _coerce_ts(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, utc=True, errors="coerce")
    return ts


def _validate_daily_levels(levels: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "buy_price", "sell_price"}
    missing = required - set(levels.columns)
    if missing:
        raise ValueError(f"daily_levels is missing required columns: {sorted(missing)}")
    frame = levels.copy()
    frame["timestamp"] = _coerce_ts(frame["timestamp"])
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    frame["buy_price"] = pd.to_numeric(frame["buy_price"], errors="coerce")
    frame["sell_price"] = pd.to_numeric(frame["sell_price"], errors="coerce")
    frame = frame.dropna(subset=["buy_price", "sell_price"]).reset_index(drop=True)
    return frame


def _validate_hourly_bars(bars: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "high", "low", "close"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"bars is missing required columns: {sorted(missing)}")
    frame = bars.copy()
    frame["timestamp"] = _coerce_ts(frame["timestamp"])
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    for col in ("open", "high", "low", "close"):
        if col not in frame.columns:
            continue
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["high", "low", "close"]).reset_index(drop=True)
    return frame


def simulate_daily_levels_on_intraday_bars(
    bars: pd.DataFrame,
    daily_levels: pd.DataFrame,
    config: Optional[DailyLevelSimulationConfig] = None,
) -> DailyLevelSimulationResult:
    """Simulate trading a single asset using fixed daily buy/sell levels over intraday bars.

    Model:
    - Long-only spot style.
    - Flat -> buy when intraday low <= daily buy_price.
    - Long -> sell when intraday high >= daily sell_price.
    - Optionally close any open position at end-of-day (UTC) using the last bar close.
    """

    cfg = config or DailyLevelSimulationConfig()
    if cfg.initial_cash <= 0:
        raise ValueError("initial_cash must be positive")
    if cfg.allocation_fraction <= 0 or cfg.allocation_fraction > 1.0:
        raise ValueError("allocation_fraction must be in (0, 1]")
    if cfg.maker_fee < 0 or cfg.maker_fee >= 1.0:
        raise ValueError("maker_fee must be in [0, 1)")
    if cfg.min_spread_pct < 0:
        raise ValueError("min_spread_pct must be >= 0")
    min_spread_mode = str(cfg.min_spread_mode or "skip").strip().lower()
    if min_spread_mode not in {"skip", "widen", "none"}:
        raise ValueError("min_spread_mode must be one of: skip, widen, none")
    intrabar_assumption = str(cfg.intrabar_assumption or "none").strip().lower()
    if intrabar_assumption not in {"none", "ohlc"}:
        raise ValueError("intrabar_assumption must be one of: none, ohlc")
    if cfg.stop_loss_pct < 0 or cfg.stop_loss_pct >= 1.0:
        raise ValueError("stop_loss_pct must be in [0, 1)")
    if cfg.periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    bars_frame = _validate_hourly_bars(bars)
    levels_frame = _validate_daily_levels(daily_levels)
    if bars_frame.empty or levels_frame.empty:
        empty_equity = pd.Series([cfg.initial_cash], index=[pd.Timestamp.now(tz="UTC")])
        metrics = pnl_metrics(equity_curve=empty_equity.values, periods_per_year=float(cfg.periods_per_year))
        return DailyLevelSimulationResult(
            equity_curve=empty_equity,
            per_bar=pd.DataFrame(),
            trades=[],
            metrics=metrics,
        )

    # Map UTC day-start -> (buy, sell).
    levels_frame = levels_frame.assign(day=levels_frame["timestamp"].dt.floor("D"))
    levels_frame = levels_frame.drop_duplicates(subset=["day"], keep="last")
    level_map: Dict[pd.Timestamp, Tuple[float, float]] = {
        pd.Timestamp(row.day): (float(row.buy_price), float(row.sell_price)) for row in levels_frame.itertuples(index=False)
    }

    cash = float(cfg.initial_cash)
    base_qty = 0.0
    entry_price: Optional[float] = None
    entry_ts: Optional[pd.Timestamp] = None
    stop_lockout_day: Optional[pd.Timestamp] = None
    trades: List[DailyLevelTrade] = []

    record_per_bar = bool(cfg.record_per_bar)
    per_rows: List[Dict[str, object]] = []
    equity_values: List[float] = []
    if record_per_bar:
        equity_index: List[pd.Timestamp] = []
    else:
        # Reuse validated timestamps to avoid per-iteration list appends during large sweeps.
        equity_index = bars_frame["timestamp"].tolist()

    current_day: Optional[pd.Timestamp] = None
    last_bar_close: Optional[float] = None
    last_bar_ts: Optional[pd.Timestamp] = None

    def _record_trade(ts: pd.Timestamp, side: str, price: float, qty: float, reason: str) -> None:
        trades.append(
            DailyLevelTrade(
                timestamp=ts,
                side=str(side),
                price=float(price),
                quantity=float(qty),
                cash_after=float(cash),
                base_after=float(base_qty),
                reason=str(reason),
            )
        )

    def _close_position(ts: pd.Timestamp, price: float, reason: str) -> None:
        nonlocal cash, base_qty, entry_price, entry_ts
        if base_qty <= 0:
            return
        qty = float(base_qty)
        proceeds = qty * float(price) * (1.0 - float(cfg.maker_fee))
        cash += proceeds
        base_qty = 0.0
        entry_price = None
        entry_ts = None
        _record_trade(ts, "sell", float(price), qty, reason)

    for row in bars_frame.itertuples(index=False):
        ts = row.timestamp
        if not isinstance(ts, pd.Timestamp):
            ts = pd.Timestamp(ts)
        day = ts.floor("D")

        # Day boundary: optionally close at previous bar close.
        if current_day is not None and day != current_day:
            if cfg.close_at_eod and base_qty > 0 and last_bar_close is not None and last_bar_ts is not None:
                _close_position(last_bar_ts, float(last_bar_close), reason="eod")
        if current_day is None or day != current_day:
            current_day = day
            if stop_lockout_day is not None and stop_lockout_day != current_day:
                stop_lockout_day = None

        last_bar_close = float(row.close)
        last_bar_ts = ts

        opened = False
        closed = False

        # Risk management: stop-loss based on the last entry price (armed starting the next bar).
        if (
            base_qty > 0
            and cfg.stop_loss_pct > 0
            and entry_price is not None
            and np.isfinite(entry_price)
            and entry_price > 0
            and entry_ts is not None
            and ts > entry_ts
        ):
            stop_price = float(entry_price) * (1.0 - float(cfg.stop_loss_pct))
            if stop_price > 0 and float(row.low) <= stop_price:
                _close_position(ts, stop_price, reason="stop_loss")
                closed = True
                if cfg.stop_loss_lockout_until_next_day:
                    stop_lockout_day = day

        buy_price, sell_price = level_map.get(day, (np.nan, np.nan))
        tradable_today = np.isfinite(buy_price) and np.isfinite(sell_price) and buy_price > 0 and sell_price > 0
        if tradable_today and sell_price <= buy_price:
            if cfg.min_spread_pct > 0 and min_spread_mode == "widen":
                sell_price = float(buy_price) * (1.0 + float(cfg.min_spread_pct))
            else:
                tradable_today = False
        if tradable_today and cfg.min_spread_pct > 0 and min_spread_mode != "none":
            spread = (sell_price - buy_price) / buy_price
            if not np.isfinite(spread):
                tradable_today = False
            elif spread < float(cfg.min_spread_pct):
                if min_spread_mode == "widen":
                    sell_price = float(buy_price) * (1.0 + float(cfg.min_spread_pct))
                else:
                    tradable_today = False

        if tradable_today:
            had_position_at_start = base_qty > 0

            def _try_open() -> bool:
                nonlocal cash, base_qty, entry_price, entry_ts, stop_lockout_day
                if cfg.stop_loss_lockout_until_next_day and stop_lockout_day == day:
                    return False
                if base_qty > 0:
                    return False
                if float(row.low) > buy_price:
                    return False
                if closed and not cfg.allow_reentry_same_bar:
                    return False

                budget = float(cash) * float(cfg.allocation_fraction)
                cost_per_unit = buy_price * (1.0 + float(cfg.maker_fee))
                qty = budget / cost_per_unit if cost_per_unit > 0 else 0.0
                qty = max(0.0, float(qty))
                if qty > 0 and np.isfinite(qty):
                    cash -= qty * cost_per_unit
                    base_qty += qty
                    entry_price = float(buy_price)
                    entry_ts = ts
                    _record_trade(ts, "buy", float(buy_price), qty, reason="entry")
                    return True
                return False

            def _try_close() -> bool:
                if base_qty <= 0:
                    return False
                if float(row.high) < sell_price:
                    return False
                if not had_position_at_start and opened and not cfg.allow_roundtrip_same_bar:
                    return False
                _close_position(ts, float(sell_price), reason="take_profit")
                return True

            if intrabar_assumption == "ohlc":
                open_price = float(getattr(row, "open", np.nan))
                if not np.isfinite(open_price) or open_price <= 0:
                    open_price = float(row.close)
                # Common OHLC path assumption: if candle closes up, low happens before high; else high before low.
                order = ("low", "high") if float(row.close) >= open_price else ("high", "low")
                for event in order:
                    if event == "high":
                        if _try_close():
                            closed = True
                    else:
                        if _try_open():
                            opened = True
            else:
                # Conservative default: at most one action per bar; sell before buy.
                if _try_close():
                    closed = True
                if _try_open():
                    opened = True

        equity = cash + base_qty * float(row.close)
        equity_values.append(float(equity))
        if record_per_bar:
            equity_index.append(ts)
        if record_per_bar:
            per_rows.append(
                {
                    "timestamp": ts,
                    "cash": cash,
                    "base_qty": base_qty,
                    "equity": equity,
                    "day": day,
                    "buy_price": float(buy_price) if np.isfinite(buy_price) else np.nan,
                    "sell_price": float(sell_price) if np.isfinite(sell_price) else np.nan,
                    "opened": float(opened),
                    "closed": float(closed),
                }
            )

    # Close any remaining position at the end of the simulation (at last close).
    if cfg.close_at_eod and base_qty > 0 and last_bar_close is not None and last_bar_ts is not None:
        _close_position(last_bar_ts, float(last_bar_close), reason="final_close")

        # Update final equity point after the forced close.
        equity = cash
        equity_values.append(float(equity))
        if record_per_bar:
            equity_index.append(last_bar_ts + pd.Timedelta(seconds=1))
        else:
            equity_index.append(last_bar_ts + pd.Timedelta(seconds=1))
        if record_per_bar:
            per_rows.append(
                {
                    "timestamp": last_bar_ts + pd.Timedelta(seconds=1),
                    "cash": cash,
                    "base_qty": base_qty,
                    "equity": equity,
                    "day": (last_bar_ts + pd.Timedelta(seconds=1)).floor("D"),
                    "buy_price": np.nan,
                    "sell_price": np.nan,
                    "opened": 0.0,
                    "closed": 1.0,
                }
            )

    equity_curve = pd.Series(equity_values, index=pd.DatetimeIndex(equity_index, name="timestamp"))
    metrics = pnl_metrics(equity_curve=equity_curve.values, periods_per_year=float(cfg.periods_per_year))
    return DailyLevelSimulationResult(
        equity_curve=equity_curve,
        per_bar=pd.DataFrame(per_rows) if record_per_bar else pd.DataFrame(),
        trades=trades,
        metrics=metrics,
    )


__all__ = [
    "DailyLevelSimulationConfig",
    "DailyLevelSimulationResult",
    "DailyLevelTrade",
    "simulate_daily_levels_on_intraday_bars",
]
