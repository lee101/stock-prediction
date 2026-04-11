"""Intra-bar hourly fill simulator for daily-decision RL policies.

The existing `hourly_replay.replay_hourly_frozen_daily_actions` only executes
trades at one chosen "trade hour" per calendar day and only uses the close
price of that hour. That hides realism: a daily long entered at 10am with a
2% stop can get stopped out at 1pm and the position would simulate as flat
for the rest of the day, but the daily-only sim only sees the daily close.

This module walks every hourly bar and:

1. At the daily "trade hour" of a calendar day, executes the daily action.
   Limit-style entries use the bar's [low, high] to decide the actual fill
   price (or skip if not crossed).
2. At every hour while a position is open, checks stop-loss / take-profit /
   max-hold-hours against the bar's [low, high]. If triggered, exits at the
   trigger price (which is what a real exchange-side stop would do).
3. Records every fill as an `IntraBarFill` carrying the actual hourly index,
   so the eval video can show the markers at the exact hour the trade fired.

This is the slow-but-realistic eval path. Training still uses the C env on
daily bars; this is for promotion gating + the rendered videos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from pufferlib_market.hourly_replay import (
    INITIAL_CASH,
    P_CLOSE,
    P_OPEN,
    MktdData,
    Position,
    _close_position,
    _compute_equity,
    _is_tradable,
    _normalize_fill_buffer_bps,
    _open_long,
    _open_short,
    _resolve_limit_fill_price,
    _load_hourly_bars,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HourlyOHLC:
    """Aligned hourly OHLC for a list of symbols.

    `index` is the hourly UTC DatetimeIndex shared by all symbols.
    Each `open/high/low/close` dict entry is a float64 array of length len(index).
    `tradable[sym]` is True at hours where a real bar was present (not ffilled).
    """

    index: pd.DatetimeIndex
    symbols: list[str]
    open: dict[str, np.ndarray]
    high: dict[str, np.ndarray]
    low: dict[str, np.ndarray]
    close: dict[str, np.ndarray]
    tradable: dict[str, np.ndarray]


@dataclass
class IntraBarFill:
    hourly_idx: int
    timestamp: pd.Timestamp
    sym: int
    side: str  # "long_open" | "long_close" | "short_open" | "short_close"
    price: float
    qty: float
    cash_after: float
    equity_after: float
    kind: str = "entry"  # "entry" | "exit" | "stop" | "take_profit" | "max_hold"


@dataclass
class IntraBarReplayResult:
    equity_curve: np.ndarray  # float64 [H]
    timestamps: pd.DatetimeIndex
    fills: list[IntraBarFill]
    total_return: float
    sortino: float
    max_drawdown: float
    num_trades: int  # closes
    num_orders: int  # any fill
    win_rate: float
    initial_equity: float
    final_equity: float


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_hourly_ohlc(
    symbols: Sequence[str],
    hourly_data_root: str | Path,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> HourlyOHLC:
    """Load aligned hourly OHLC for the given symbol list.

    Reuses pufferlib_market.hourly_replay._load_hourly_bars to find each
    symbol's CSV under crypto/ or stocks/ subdirs and align to a uniform
    UTC hourly grid. Missing bars are forward-filled for the close (so
    mark-to-market still works) but the `tradable` mask is False for them
    so trade execution skips closed-market hours.
    """
    hourly_data_root = Path(hourly_data_root)
    start_ts = pd.to_datetime(start, utc=True)
    end_ts = pd.to_datetime(end, utc=True)
    if start_ts.tz is None:
        start_ts = start_ts.tz_localize("UTC")
    if end_ts.tz is None:
        end_ts = end_ts.tz_localize("UTC")
    if end_ts < start_ts:
        raise ValueError(f"end before start: {start_ts} > {end_ts}")

    index = pd.date_range(start_ts.floor("h"), end_ts.floor("h"), freq="h", tz="UTC")
    if len(index) < 2:
        raise ValueError("Hourly grid too small")

    sym_list = [s.upper() for s in symbols]
    o: dict[str, np.ndarray] = {}
    h: dict[str, np.ndarray] = {}
    l: dict[str, np.ndarray] = {}
    c: dict[str, np.ndarray] = {}
    tr: dict[str, np.ndarray] = {}

    for sym in sym_list:
        raw = _load_hourly_bars(sym, hourly_data_root)
        raw = raw.loc[(raw.index >= index[0]) & (raw.index <= index[-1])]

        # Required columns: open, high, low, close. Tolerate missing volume.
        for col in ("open", "high", "low", "close"):
            if col not in raw.columns:
                raise ValueError(f"{sym}: hourly bars missing column '{col}'")

        # tradable = bar present pre-reindex
        present = pd.Series(True, index=raw.index, dtype=bool)
        present_aligned = present.reindex(index, fill_value=False)

        # forward/back fill OHLC so mark-to-market keeps working in market-closed hours
        close_aligned = raw["close"].astype(float).reindex(index).ffill().bfill().fillna(0.0)
        open_aligned = raw["open"].astype(float).reindex(index).ffill().bfill().fillna(0.0)
        high_aligned = raw["high"].astype(float).reindex(index).ffill().bfill().fillna(0.0)
        low_aligned = raw["low"].astype(float).reindex(index).ffill().bfill().fillna(0.0)

        o[sym] = open_aligned.to_numpy(dtype=np.float64, copy=False)
        h[sym] = high_aligned.to_numpy(dtype=np.float64, copy=False)
        l[sym] = low_aligned.to_numpy(dtype=np.float64, copy=False)
        c[sym] = close_aligned.to_numpy(dtype=np.float64, copy=False)
        tr[sym] = present_aligned.to_numpy(dtype=bool, copy=False)

    return HourlyOHLC(index=index, symbols=sym_list, open=o, high=h, low=l, close=c, tradable=tr)


def synthetic_hourly_ohlc_from_daily(data: MktdData, *, start: str | pd.Timestamp) -> HourlyOHLC:
    """Build a synthetic 24h-per-day OHLC from a daily MKTD.

    Each daily bar becomes 24 hourly bars whose close walks linearly from the
    daily open to the daily close, with intra-day high/low spread proportional
    to the daily range. Used for tests and as a fallback when real hourly
    data is missing for some symbols.
    """
    start_day = pd.to_datetime(start, utc=True).floor("D")
    T = data.num_timesteps
    S = data.num_symbols
    H = T * 24
    index = pd.date_range(start_day, periods=H, freq="h", tz="UTC")

    o: dict[str, np.ndarray] = {}
    h: dict[str, np.ndarray] = {}
    l: dict[str, np.ndarray] = {}
    c: dict[str, np.ndarray] = {}
    tr: dict[str, np.ndarray] = {}

    for si, sym in enumerate(data.symbols):
        opens = data.prices[:, si, P_OPEN].astype(np.float64)
        highs = data.prices[:, si, 1].astype(np.float64)
        lows = data.prices[:, si, 2].astype(np.float64)
        closes = data.prices[:, si, P_CLOSE].astype(np.float64)

        oo = np.empty(H, dtype=np.float64)
        hh = np.empty(H, dtype=np.float64)
        ll = np.empty(H, dtype=np.float64)
        cc = np.empty(H, dtype=np.float64)
        for di in range(T):
            base = di * 24
            # Walk close linearly from open->close across the day.
            for hi in range(24):
                frac = (hi + 1) / 24.0
                cc[base + hi] = opens[di] * (1.0 - frac) + closes[di] * frac
            # Bar opens equal to previous hourly close (or daily open at hour 0).
            oo[base] = opens[di]
            oo[base + 1 : base + 24] = cc[base : base + 23]
            # Spread the daily high/low across hours: hour with the daily extreme
            # picks up the actual extreme; others stay near the linear path.
            hi_pos = max(0, min(23, int(round(0.66 * 23))))
            lo_pos = max(0, min(23, int(round(0.33 * 23))))
            for hi in range(24):
                hh[base + hi] = max(oo[base + hi], cc[base + hi])
                ll[base + hi] = min(oo[base + hi], cc[base + hi])
            hh[base + hi_pos] = max(hh[base + hi_pos], highs[di])
            ll[base + lo_pos] = min(ll[base + lo_pos], lows[di])

        o[sym] = oo
        h[sym] = hh
        l[sym] = ll
        c[sym] = cc
        tr[sym] = np.ones(H, dtype=bool)

    return HourlyOHLC(
        index=index,
        symbols=list(data.symbols),
        open=o,
        high=h,
        low=l,
        close=c,
        tradable=tr,
    )


# ---------------------------------------------------------------------------
# Fill helpers
# ---------------------------------------------------------------------------


def _bar_intersects_level(low: float, high: float, level: float) -> bool:
    """True if a horizontal price level falls inside [low, high] inclusive."""
    return float(min(low, high)) <= float(level) <= float(max(low, high))


def _stop_loss_triggered(
    *,
    pos: Position,
    bar_low: float,
    bar_high: float,
    stop_pct: float,
) -> Optional[float]:
    """Return the stop-loss fill price if the bar crossed it, else None.

    For longs the stop is entry*(1-stop_pct) and triggers when bar_low <= stop.
    For shorts the stop is entry*(1+stop_pct) and triggers when bar_high >= stop.
    """
    if stop_pct is None or stop_pct <= 0.0:
        return None
    if pos.is_short:
        stop_price = pos.entry_price * (1.0 + stop_pct)
        if bar_high >= stop_price:
            return float(stop_price)
        return None
    stop_price = pos.entry_price * (1.0 - stop_pct)
    if bar_low <= stop_price:
        return float(stop_price)
    return None


def _take_profit_triggered(
    *,
    pos: Position,
    bar_low: float,
    bar_high: float,
    tp_pct: float,
) -> Optional[float]:
    if tp_pct is None or tp_pct <= 0.0:
        return None
    if pos.is_short:
        tp_price = pos.entry_price * (1.0 - tp_pct)
        if bar_low <= tp_price:
            return float(tp_price)
        return None
    tp_price = pos.entry_price * (1.0 + tp_pct)
    if bar_high >= tp_price:
        return float(tp_price)
    return None


# ---------------------------------------------------------------------------
# Main replay
# ---------------------------------------------------------------------------


def replay_intrabar(
    *,
    data: MktdData,
    actions: np.ndarray,
    hourly: HourlyOHLC,
    start_date: str | pd.Timestamp,
    max_steps: int,
    fee_rate: float = 0.001,
    fill_buffer_bps: float = 5.0,
    max_leverage: float = 1.0,
    stop_loss_pct: Optional[float] = None,
    take_profit_pct: Optional[float] = None,
    max_hold_hours: Optional[int] = None,
    initial_cash: float = INITIAL_CASH,
    periods_per_year: float = 8760.0,
) -> IntraBarReplayResult:
    """Walk hourly bars, executing the daily `actions` array with realistic intra-bar fills.

    Daily action semantics (single allocation/level bin, matches the prod
    `simulate_daily_policy` shape):

        action == 0           : flatten
        1..S                  : long  symbol (action - 1)
        S+1..2S               : short symbol (action - S - 1)

    Trade execution rules:

    - The "trade hour" for a calendar day is the first hour at/after midnight
      UTC where the target symbol is tradable. For 24/7 markets that's hour 0.
    - Entries fill at `bar.open ± fill_buffer` if that price falls inside the
      hour's [low, high]. Otherwise we walk forward through the rest of the
      day's hours and use the first hour whose [low, high] crosses the target.
    - Stop-loss / take-profit / max-hold are checked at every hour against
      the bar's [low, high] and triggered at the trigger price exactly.
    - Equity is marked to market every hour at the close.
    """
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if actions.shape[0] < max_steps:
        raise ValueError(f"actions length {actions.shape[0]} < max_steps {max_steps}")

    fill_buffer = _normalize_fill_buffer_bps(fill_buffer_bps) / 10_000.0
    start_day = pd.to_datetime(start_date, utc=True).floor("D")

    H = len(hourly.index)
    sym_names = [s.upper() for s in data.symbols]
    S = data.num_symbols

    # Build per-day list of (start_hi, end_hi_exclusive) within the hourly index.
    day_floor = hourly.index.floor("D")
    day_idx_per_hour = ((day_floor - start_day).days).to_numpy()
    day_starts: list[int] = []
    day_ends: list[int] = []
    cur_day = -1
    for hi in range(H):
        d = int(day_idx_per_hour[hi])
        if d != cur_day:
            if 0 <= cur_day < max_steps:
                day_ends.append(hi)
            cur_day = d
            if 0 <= d < max_steps:
                day_starts.append(hi)
            else:
                day_starts.append(-1)
                day_ends.append(-1)
    if 0 <= cur_day < max_steps:
        day_ends.append(H)
    while len(day_ends) < len(day_starts):
        day_ends.append(H)

    cash = float(initial_cash)
    pos: Optional[Position] = None
    initial_equity = float(initial_cash)
    peak_equity = initial_equity
    max_dd = 0.0
    equity_curve = np.full(H, initial_equity, dtype=np.float64)
    fills: list[IntraBarFill] = []
    num_trades = 0
    winning_trades = 0
    pos_open_hi: int = -1

    def _hour_close(sym_i: int, hi: int) -> float:
        return float(hourly.close[sym_names[sym_i]][hi])

    def _bar(sym_i: int, hi: int) -> tuple[float, float, float, float, bool]:
        name = sym_names[sym_i]
        return (
            float(hourly.open[name][hi]),
            float(hourly.high[name][hi]),
            float(hourly.low[name][hi]),
            float(hourly.close[name][hi]),
            bool(hourly.tradable[name][hi]),
        )

    def _record(hi: int, sym_i: int, side: str, price: float, qty: float, kind: str) -> None:
        equity_after = _compute_equity(cash, pos, price)
        fills.append(
            IntraBarFill(
                hourly_idx=int(hi),
                timestamp=hourly.index[hi],
                sym=int(sym_i),
                side=side,
                price=float(price),
                qty=float(qty),
                cash_after=float(cash),
                equity_after=float(equity_after),
                kind=kind,
            )
        )

    # Iterate over days; within each day walk every hour to mark-to-market and
    # check stop/tp/max-hold, and at the trade hour execute the daily action.
    num_days = min(len(day_starts), max_steps)
    for di in range(num_days):
        hi_start = day_starts[di]
        hi_end = day_ends[di] if di < len(day_ends) else H
        if hi_start < 0 or hi_start >= H:
            continue

        action = int(actions[di]) if di < len(actions) else 0
        if action < 0 or action > 2 * S:
            action = 0

        # ---- Stage 1: walk the day's hours, applying the action and exits.
        action_done = False
        for hi in range(hi_start, hi_end):
            # 1a) Stop-loss / take-profit / max-hold check on any open position.
            if pos is not None:
                _, hi_high, hi_low, _, _ = _bar(pos.sym, hi)
                exit_price: Optional[float] = None
                exit_kind: Optional[str] = None

                sl = _stop_loss_triggered(
                    pos=pos, bar_low=hi_low, bar_high=hi_high, stop_pct=stop_loss_pct or 0.0
                )
                tp = _take_profit_triggered(
                    pos=pos, bar_low=hi_low, bar_high=hi_high, tp_pct=take_profit_pct or 0.0
                )
                # Conservative ordering: a bar can hit both. Assume stop fires first.
                if sl is not None:
                    exit_price = sl
                    exit_kind = "stop"
                elif tp is not None:
                    exit_price = tp
                    exit_kind = "take_profit"
                elif (
                    max_hold_hours is not None
                    and max_hold_hours > 0
                    and pos_open_hi >= 0
                    and (hi - pos_open_hi) >= max_hold_hours
                ):
                    bar_open, _, _, bar_close, _ = _bar(pos.sym, hi)
                    exit_price = bar_open
                    exit_kind = "max_hold"

                if exit_price is not None:
                    side = "short_close" if pos.is_short else "long_close"
                    closed_sym = pos.sym
                    qty = pos.qty
                    cash, win = _close_position(cash, pos, exit_price, fee_rate)
                    pos = None
                    pos_open_hi = -1
                    num_trades += 1
                    winning_trades += int(win)
                    _record(hi, closed_sym, side, exit_price, qty, exit_kind or "exit")

            # 1b) Apply the daily action at the first tradable hour for the
            # target symbol — but only once per day.
            if not action_done and action != 0:
                if action <= S:
                    target_sym = action - 1
                    is_short = False
                else:
                    target_sym = action - S - 1
                    is_short = True

                bar_open, bar_high, bar_low, bar_close, target_tradable = _bar(target_sym, hi)
                if target_tradable and bar_open > 0.0:
                    # If we're already in this exact target, hold.
                    if pos is not None and pos.sym == target_sym and pos.is_short == is_short:
                        action_done = True
                    else:
                        # Close any other open position first (at this bar's open).
                        if pos is not None:
                            cur_open, cur_high, cur_low, cur_close, cur_tr = _bar(pos.sym, hi)
                            if cur_tr:
                                exit_px = cur_open if cur_open > 0.0 else cur_close
                                side = "short_close" if pos.is_short else "long_close"
                                closed_sym = pos.sym
                                qty = pos.qty
                                cash, win = _close_position(cash, pos, exit_px, fee_rate)
                                pos = None
                                pos_open_hi = -1
                                num_trades += 1
                                winning_trades += int(win)
                                _record(hi, closed_sym, side, exit_px, qty, "exit")
                            else:
                                # Can't close current pos this hour: try next hour.
                                continue

                        # Try to fill the entry inside this hour's [low, high].
                        target_price = bar_open * (1.0 + (fill_buffer if not is_short else -fill_buffer))
                        fill_px = _resolve_limit_fill_price(
                            low=bar_low,
                            high=bar_high,
                            target_price=target_price,
                            is_buy=not is_short,
                            fill_buffer_bps=0.0,  # already baked into target_price
                        )
                        if fill_px is None:
                            # Limit didn't cross this hour; walk forward to find it.
                            continue

                        if is_short:
                            cash, pos = _open_short(cash, target_sym, fill_px, fee_rate, max_leverage)
                            side = "short_open"
                        else:
                            cash, pos = _open_long(cash, target_sym, fill_px, fee_rate, max_leverage)
                            side = "long_open"
                        if pos is not None:
                            pos_open_hi = hi
                            _record(hi, target_sym, side, fill_px, pos.qty, "entry")
                        action_done = True
            elif not action_done and action == 0:
                # Flatten request: close at this hour's open if we have a position.
                if pos is not None:
                    cur_open, _, _, cur_close, cur_tr = _bar(pos.sym, hi)
                    if cur_tr:
                        exit_px = cur_open if cur_open > 0.0 else cur_close
                        side = "short_close" if pos.is_short else "long_close"
                        closed_sym = pos.sym
                        qty = pos.qty
                        cash, win = _close_position(cash, pos, exit_px, fee_rate)
                        pos = None
                        pos_open_hi = -1
                        num_trades += 1
                        winning_trades += int(win)
                        _record(hi, closed_sym, side, exit_px, qty, "exit")
                action_done = True

            # 1c) Mark-to-market at the bar close.
            if pos is None:
                eq = float(cash)
            else:
                eq = _compute_equity(cash, pos, _hour_close(pos.sym, hi))
            equity_curve[hi] = eq
            if eq > peak_equity:
                peak_equity = eq
            dd = (peak_equity - eq) / peak_equity if peak_equity > 0 else 0.0
            if dd > max_dd:
                max_dd = dd

    # Forward-fill equity beyond the last walked hour.
    last_walked = -1
    for hi in range(H - 1, -1, -1):
        if equity_curve[hi] != initial_equity or pos is not None:
            last_walked = hi
            break
    if last_walked >= 0 and last_walked + 1 < H:
        equity_curve[last_walked + 1 :] = equity_curve[last_walked]

    # Final flatten for total_return accounting at the last walked hour.
    if pos is not None and last_walked >= 0:
        exit_px = _hour_close(pos.sym, last_walked)
        side = "short_close" if pos.is_short else "long_close"
        closed_sym = pos.sym
        qty = pos.qty
        cash, win = _close_position(cash, pos, exit_px, fee_rate)
        pos = None
        num_trades += 1
        winning_trades += int(win)
        _record(last_walked, closed_sym, side, exit_px, qty, "exit")
        equity_curve[last_walked] = float(cash)

    final_equity = float(cash)
    total_return = (final_equity - initial_equity) / initial_equity if initial_equity != 0.0 else 0.0

    rets = (equity_curve[1:] - equity_curve[:-1]) / np.clip(equity_curve[:-1], 1e-12, None)
    neg = rets[rets < 0.0]
    if neg.size > 1:
        downside_dev = float(np.sqrt(np.mean(neg * neg)))
        ppy = float(periods_per_year) if periods_per_year > 0 else 8760.0
        mean_ret = float(np.mean(rets))
        sortino = float(mean_ret / downside_dev * np.sqrt(ppy)) if downside_dev > 0 else 0.0
    else:
        sortino = 0.0

    win_rate = float(winning_trades / num_trades) if num_trades > 0 else 0.0

    return IntraBarReplayResult(
        equity_curve=equity_curve,
        timestamps=hourly.index,
        fills=fills,
        total_return=float(total_return),
        sortino=float(sortino),
        max_drawdown=float(max_dd),
        num_trades=int(num_trades),
        num_orders=len(fills),
        win_rate=float(win_rate),
        initial_equity=float(initial_equity),
        final_equity=float(final_equity),
    )


# ---------------------------------------------------------------------------
# MarketsimTrace adapter
# ---------------------------------------------------------------------------


def build_hourly_marketsim_trace(
    *,
    hourly: HourlyOHLC,
    fills: list[IntraBarFill],
    equity_curve: np.ndarray,
    initial_equity: float,
):
    """Convert an intra-bar replay result into a MarketsimTrace.

    One frame per hourly bar. The frame's `prices_ohlc` is hourly OHLC for
    every symbol. Frames carry an `OrderTick` at the hour each fill happened,
    plus a `position_sym` reflecting the post-fill position.
    """
    from src.marketsim_video import MarketsimTrace, OrderTick

    H = len(hourly.index)
    S = len(hourly.symbols)
    sym_names = list(hourly.symbols)

    closes = np.zeros((H, S), dtype=np.float32)
    ohlc = np.zeros((H, S, 4), dtype=np.float32)
    for si, sym in enumerate(sym_names):
        closes[:, si] = hourly.close[sym].astype(np.float32, copy=False)
        ohlc[:, si, 0] = hourly.open[sym].astype(np.float32, copy=False)
        ohlc[:, si, 1] = hourly.high[sym].astype(np.float32, copy=False)
        ohlc[:, si, 2] = hourly.low[sym].astype(np.float32, copy=False)
        ohlc[:, si, 3] = hourly.close[sym].astype(np.float32, copy=False)

    trace = MarketsimTrace(symbols=sym_names, prices=closes, prices_ohlc=ohlc)

    fills_by_hi: dict[int, list[IntraBarFill]] = {}
    for fl in fills:
        fills_by_hi.setdefault(fl.hourly_idx, []).append(fl)

    pos_sym = -1
    pos_short = False
    for hi in range(H):
        bar_fills = fills_by_hi.get(hi, [])
        orders = []
        for fl in bar_fills:
            orders.append(OrderTick(sym=fl.sym, price=fl.price, is_short=("short" in fl.side)))
            if fl.side == "long_open":
                pos_sym = fl.sym
                pos_short = False
            elif fl.side == "short_open":
                pos_sym = fl.sym
                pos_short = True
            elif fl.side in ("long_close", "short_close"):
                pos_sym = -1
                pos_short = False
        eq = float(equity_curve[hi]) if hi < len(equity_curve) else float(equity_curve[-1])
        trace.record(
            step=hi,
            action_id=0,
            position_sym=pos_sym,
            position_is_short=pos_short,
            equity=eq * 10000.0 / max(initial_equity, 1e-9),
            orders=orders,
        )
    return trace
