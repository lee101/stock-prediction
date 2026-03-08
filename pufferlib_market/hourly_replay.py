"""Hourly replay evaluation utilities for the pufferlib_market daily RL policy.

This module is intentionally self-contained and lightweight:
- It can simulate the daily C environment dynamics in pure Python (for traces).
- It can "replay" a daily action sequence on higher-frequency (hourly) prices to
  compute higher-resolution risk metrics and execution counts.

The goal is not to perfectly model Alpaca execution; it's to catch obvious
simulation artifacts (e.g., too-frequent switching, market-hours constraints)
and to compute Sortino/max-drawdown on an hourly equity curve while training on
daily bars.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd

INITIAL_CASH = 10000.0

FEATURES_PER_SYM = 16
PRICE_FEATS = 5
SYM_NAME_LEN = 16
HEADER_SIZE = 64

P_OPEN = 0
P_HIGH = 1
P_LOW = 2
P_CLOSE = 3
P_VOL = 4


@dataclass(frozen=True)
class MktdData:
    version: int
    symbols: list[str]
    features: np.ndarray  # float32 [T, S, F]
    prices: np.ndarray  # float32 [T, S, 5] (O,H,L,C,V)
    tradable: Optional[np.ndarray]  # uint8/bool [T, S] (1=tradable)

    @property
    def num_symbols(self) -> int:
        return int(self.features.shape[1])

    @property
    def num_timesteps(self) -> int:
        return int(self.features.shape[0])


def read_mktd(path: str | Path) -> MktdData:
    path = Path(path)
    with path.open("rb") as f:
        header = f.read(HEADER_SIZE)
        if len(header) != HEADER_SIZE:
            raise ValueError(f"Short MKTD header: {path}")
        magic = header[:4]
        if magic != b"MKTD":
            raise ValueError(f"Bad MKTD magic in {path}: {magic!r}")

        # <4sIIIII40s
        version = int.from_bytes(header[4:8], "little", signed=False)
        num_symbols = int.from_bytes(header[8:12], "little", signed=False)
        num_timesteps = int.from_bytes(header[12:16], "little", signed=False)
        features_per_sym = int.from_bytes(header[16:20], "little", signed=False)
        price_features = int.from_bytes(header[20:24], "little", signed=False)

        if num_symbols <= 0 or num_symbols > 32:
            raise ValueError(f"Invalid num_symbols={num_symbols} in {path}")
        if num_timesteps <= 0:
            raise ValueError(f"Invalid num_timesteps={num_timesteps} in {path}")
        if features_per_sym <= 0:
            raise ValueError(f"Invalid features_per_sym={features_per_sym} in {path}")
        if price_features <= 0:
            raise ValueError(f"Invalid price_features={price_features} in {path}")
        if features_per_sym != FEATURES_PER_SYM:
            raise ValueError(
                f"Unsupported features_per_sym={features_per_sym} in {path} "
                f"(expected {FEATURES_PER_SYM})"
            )
        if price_features != PRICE_FEATS:
            raise ValueError(
                f"Unsupported price_features={price_features} in {path} "
                f"(expected {PRICE_FEATS})"
            )

        sym_table = f.read(num_symbols * SYM_NAME_LEN)
        if len(sym_table) != num_symbols * SYM_NAME_LEN:
            raise ValueError(f"Short MKTD symbol table: {path}")

        symbols: list[str] = []
        for i in range(num_symbols):
            raw = sym_table[i * SYM_NAME_LEN : (i + 1) * SYM_NAME_LEN]
            sym = raw.split(b"\x00", 1)[0].decode("ascii", errors="ignore").strip()
            if not sym:
                raise ValueError(f"Empty symbol at index {i} in {path}")
            symbols.append(sym)

        feat_count = num_timesteps * num_symbols * features_per_sym
        features = np.fromfile(f, dtype=np.float32, count=feat_count)
        if features.size != feat_count:
            raise ValueError(f"Short MKTD features array in {path}")
        features = features.reshape((num_timesteps, num_symbols, features_per_sym))

        price_count = num_timesteps * num_symbols * price_features
        prices = np.fromfile(f, dtype=np.float32, count=price_count)
        if prices.size != price_count:
            raise ValueError(f"Short MKTD price array in {path}")
        prices = prices.reshape((num_timesteps, num_symbols, price_features))

        tradable: Optional[np.ndarray] = None
        mask_count = num_timesteps * num_symbols

        # v2+ requires a mask. For forward compatibility, accept a mask if present for v1.
        mask_bytes = f.read()
        if mask_bytes:
            mask = np.frombuffer(mask_bytes, dtype=np.uint8, count=mask_count)
            if mask.size == mask_count:
                tradable = mask.reshape((num_timesteps, num_symbols))
            else:
                raise ValueError(
                    f"Trailing bytes in {path} do not match tradable mask size: "
                    f"got={len(mask_bytes)} expected={mask_count}"
                )
        elif version >= 2:
            raise ValueError(f"MKTD v{version} missing tradable mask: {path}")

    return MktdData(
        version=version,
        symbols=symbols,
        features=features,
        prices=prices,
        tradable=tradable,
    )


@dataclass
class Position:
    sym: int  # 0..S-1
    is_short: bool
    qty: float
    entry_price: float


@dataclass
class DailySimResult:
    actions: np.ndarray  # int32 [max_steps]
    total_return: float
    sortino: float
    max_drawdown: float
    num_trades: int
    win_rate: float
    avg_hold_steps: float


def _is_tradable(data: MktdData, t: int, sym: int) -> bool:
    if data.tradable is None:
        return True
    return bool(int(data.tradable[t, sym]) != 0)


def _compute_equity(cash: float, pos: Optional[Position], price: float) -> float:
    if pos is None:
        return float(cash)
    if pos.is_short:
        return float(cash - pos.qty * price)
    return float(cash + pos.qty * price)


def _close_position(cash: float, pos: Position, price: float, fee_rate: float) -> tuple[float, bool]:
    if pos.is_short:
        cost = pos.qty * price * (1.0 + fee_rate)
        cash -= cost
        win = (pos.entry_price - price) > 0
        return float(cash), bool(win)
    proceeds = pos.qty * price * (1.0 - fee_rate)
    cash += proceeds
    win = (price - pos.entry_price) > 0
    return float(cash), bool(win)


def _resolve_limit_fill_price(*, low: float, high: float, target_price: float) -> float | None:
    lo = float(min(low, high))
    hi = float(max(low, high))
    tp = float(target_price)
    if tp < lo or tp > hi:
        return None
    return tp


def _action_allocation_pct(*, alloc_idx: int, alloc_bins: int) -> float:
    bins = max(1, int(alloc_bins))
    if bins <= 1:
        return 1.0
    idx = max(0, min(int(alloc_idx), bins - 1))
    pct = float(idx + 1) / float(bins)
    return float(min(1.0, max(0.01, pct)))


def _action_level_offset_bps(*, level_idx: int, level_bins: int, max_offset_bps: float) -> float:
    bins = max(1, int(level_bins))
    max_bps = max(0.0, float(max_offset_bps))
    if bins <= 1 or max_bps <= 0.0:
        return 0.0
    idx = max(0, min(int(level_idx), bins - 1))
    frac = float(idx) / float(bins - 1)
    return (2.0 * frac - 1.0) * max_bps


def _open_long(cash: float, sym: int, price: float, fee_rate: float, max_leverage: float) -> tuple[float, Optional[Position]]:
    if price <= 0.0 or cash <= 0.0:
        return float(cash), None
    buy_budget = cash * max_leverage
    qty = buy_budget / (price * (1.0 + fee_rate))
    cost = qty * price * (1.0 + fee_rate)
    cash -= cost
    return float(cash), Position(sym=sym, is_short=False, qty=float(qty), entry_price=float(price))


def _open_short(cash: float, sym: int, price: float, fee_rate: float, max_leverage: float) -> tuple[float, Optional[Position]]:
    if price <= 0.0 or cash <= 0.0:
        return float(cash), None
    sell_budget = cash * max_leverage
    qty = sell_budget / (price * (1.0 + fee_rate))
    cash += qty * price * (1.0 - fee_rate)
    return float(cash), Position(sym=sym, is_short=True, qty=float(qty), entry_price=float(price))


def _open_long_limit(
    *,
    cash: float,
    sym: int,
    close_price: float,
    low_price: float,
    high_price: float,
    fee_rate: float,
    max_leverage: float,
    allocation_pct: float,
    level_offset_bps: float,
) -> tuple[float, Optional[Position]]:
    if close_price <= 0.0 or cash <= 0.0:
        return float(cash), None
    target_price = float(close_price) * (1.0 + float(level_offset_bps) / 10_000.0)
    fill_price = _resolve_limit_fill_price(low=float(low_price), high=float(high_price), target_price=target_price)
    if fill_price is None:
        return float(cash), None
    alloc = min(1.0, max(0.0, float(allocation_pct)))
    if alloc <= 0.0:
        return float(cash), None
    buy_budget = float(cash) * float(max_leverage) * alloc
    denom = fill_price * (1.0 + float(fee_rate))
    if buy_budget <= 0.0 or denom <= 0.0:
        return float(cash), None
    qty = buy_budget / denom
    cost = qty * denom
    if qty <= 0.0 or cost <= 0.0:
        return float(cash), None
    if cost > cash:
        cost = float(cash)
        qty = cost / denom
        if qty <= 0.0:
            return float(cash), None
    cash -= cost
    return float(cash), Position(sym=sym, is_short=False, qty=float(qty), entry_price=float(fill_price))


def _open_short_limit(
    *,
    cash: float,
    sym: int,
    close_price: float,
    low_price: float,
    high_price: float,
    fee_rate: float,
    max_leverage: float,
    allocation_pct: float,
    level_offset_bps: float,
) -> tuple[float, Optional[Position]]:
    if close_price <= 0.0 or cash <= 0.0:
        return float(cash), None
    target_price = float(close_price) * (1.0 + float(level_offset_bps) / 10_000.0)
    fill_price = _resolve_limit_fill_price(low=float(low_price), high=float(high_price), target_price=target_price)
    if fill_price is None:
        return float(cash), None
    alloc = min(1.0, max(0.0, float(allocation_pct)))
    if alloc <= 0.0:
        return float(cash), None
    sell_budget = float(cash) * float(max_leverage) * alloc
    denom = fill_price * (1.0 + float(fee_rate))
    if sell_budget <= 0.0 or denom <= 0.0:
        return float(cash), None
    qty = sell_budget / denom
    if qty <= 0.0:
        return float(cash), None
    cash += qty * fill_price * (1.0 - float(fee_rate))
    return float(cash), Position(sym=sym, is_short=True, qty=float(qty), entry_price=float(fill_price))


def _build_obs(
    data: MktdData,
    t: int,
    pos: Optional[Position],
    cash: float,
    hold_steps: int,
    step: int,
    max_steps: int,
) -> np.ndarray:
    S = data.num_symbols
    obs_size = S * FEATURES_PER_SYM + 5 + S
    obs = np.zeros((obs_size,), dtype=np.float32)

    obs[: S * FEATURES_PER_SYM] = data.features[t].reshape(-1)

    base = S * FEATURES_PER_SYM
    pos_val = 0.0
    unreal = 0.0
    if pos is not None:
        price = float(data.prices[t, pos.sym, P_CLOSE])
        if pos.is_short:
            pos_val = -(pos.qty * price)
            unreal = pos.qty * (pos.entry_price - price)
        else:
            pos_val = pos.qty * price
            unreal = pos.qty * (price - pos.entry_price)

    denom = float(INITIAL_CASH)
    obs[base + 0] = float(cash / denom)
    obs[base + 1] = float(pos_val / denom)
    obs[base + 2] = float(unreal / denom)
    obs[base + 3] = float(hold_steps / (max_steps if max_steps > 0 else 1))
    obs[base + 4] = float(step / (max_steps if max_steps > 0 else 1))

    # one-hot position (1 long, -1 short)
    if pos is not None:
        obs[base + 5 + pos.sym] = -1.0 if pos.is_short else 1.0

    return obs


def simulate_daily_policy(
    data: MktdData,
    policy_fn: Callable[[np.ndarray], int],
    *,
    max_steps: int,
    fee_rate: float = 0.001,
    max_leverage: float = 1.0,
    periods_per_year: float = 365.0,
    initial_cash: float = INITIAL_CASH,
    action_allocation_bins: int = 1,
    action_level_bins: int = 1,
    action_max_offset_bps: float = 0.0,
) -> DailySimResult:
    """Pure-python simulation matching the C env's daily step semantics.

    This is used to generate a daily action trace (and a baseline metric set)
    without relying on the compiled binding.
    """
    S = data.num_symbols
    T = data.num_timesteps
    if max_steps < 1 or max_steps >= T:
        raise ValueError(f"max_steps must be in [1, {T-1}] (got {max_steps})")

    cash = float(initial_cash)
    pos: Optional[Position] = None
    hold_steps = 0
    step = 0
    peak_equity = float(initial_cash)
    max_dd = 0.0
    initial_equity = float(initial_cash)

    num_trades = 0
    winning_trades = 0

    sum_ret = 0.0
    sum_neg_sq = 0.0
    ret_count = 0

    actions = np.zeros((max_steps,), dtype=np.int32)
    alloc_bins = max(1, int(action_allocation_bins))
    level_bins = max(1, int(action_level_bins))
    per_symbol_actions = alloc_bins * level_bins
    side_block = S * per_symbol_actions

    while True:
        t = step
        if t >= T:
            t = T - 1

        # current position info
        cur_sym = pos.sym if pos is not None else -1
        cur_tradable = _is_tradable(data, t, cur_sym) if cur_sym >= 0 else True

        # equity before action at time t
        price_cur = float(data.prices[t, cur_sym, P_CLOSE]) if cur_sym >= 0 else 0.0
        equity_before = _compute_equity(cash, pos, price_cur) if cur_sym >= 0 else float(cash)

        obs = _build_obs(data, t, pos, cash, hold_steps, step, max_steps)
        action = int(policy_fn(obs))
        if action < 0 or action > 2 * side_block:
            action = -1
        if step < max_steps:
            actions[step] = action

        # Execute action (same logic as c_step)
        if action == 0:
            if pos is not None:
                if cur_tradable:
                    cash, win = _close_position(cash, pos, price_cur, fee_rate)
                    num_trades += 1
                    winning_trades += int(win)
                    pos = None
                    hold_steps = 0
                else:
                    hold_steps += 1
        elif 1 <= action <= 2 * side_block:
            action_idx = action - 1
            is_short_target = action_idx >= side_block
            if is_short_target:
                action_idx -= side_block
            target_sym = action_idx // per_symbol_actions
            rem = action_idx % per_symbol_actions
            alloc_idx = rem // level_bins
            level_idx = rem % level_bins
            target_alloc = _action_allocation_pct(alloc_idx=alloc_idx, alloc_bins=alloc_bins)
            level_bps = _action_level_offset_bps(
                level_idx=level_idx,
                level_bins=level_bins,
                max_offset_bps=action_max_offset_bps,
            )
            target_pos_id = target_sym + S if is_short_target else target_sym
            target_tradable = _is_tradable(data, t, target_sym)
            if pos is not None and ((S + pos.sym) if pos.is_short else pos.sym) == target_pos_id:
                hold_steps += 1
            else:
                if not target_tradable:
                    if pos is not None:
                        hold_steps += 1
                elif pos is not None and not cur_tradable:
                    hold_steps += 1
                else:
                    if pos is not None:
                        cash, win = _close_position(cash, pos, price_cur, fee_rate)
                        num_trades += 1
                        winning_trades += int(win)
                    close_px = float(data.prices[t, target_sym, P_CLOSE])
                    low_px = float(data.prices[t, target_sym, P_LOW])
                    high_px = float(data.prices[t, target_sym, P_HIGH])
                    if is_short_target:
                        cash, pos = _open_short_limit(
                            cash=cash,
                            sym=target_sym,
                            close_price=close_px,
                            low_price=low_px,
                            high_price=high_px,
                            fee_rate=fee_rate,
                            max_leverage=max_leverage,
                            allocation_pct=target_alloc,
                            level_offset_bps=level_bps,
                        )
                    else:
                        cash, pos = _open_long_limit(
                            cash=cash,
                            sym=target_sym,
                            close_price=close_px,
                            low_price=low_px,
                            high_price=high_px,
                            fee_rate=fee_rate,
                            max_leverage=max_leverage,
                            allocation_pct=target_alloc,
                            level_offset_bps=level_bps,
                        )
                    if pos is not None and ((S + pos.sym) if pos.is_short else pos.sym) == target_pos_id:
                        hold_steps = 0
        else:
            if pos is not None:
                hold_steps += 1

        # advance time
        step += 1
        t_new = step
        if t_new >= T:
            t_new = T - 1

        # equity after market move at t_new
        if pos is None:
            equity_after = float(cash)
        else:
            price_new = float(data.prices[t_new, pos.sym, P_CLOSE])
            equity_after = _compute_equity(cash, pos, price_new)

        ret = 0.0
        if equity_before > 1e-6:
            ret = (equity_after - equity_before) / equity_before

        sum_ret += ret
        ret_count += 1
        if ret < 0.0:
            sum_neg_sq += ret * ret

        if equity_after > peak_equity:
            peak_equity = equity_after
        dd = (peak_equity - equity_after) / peak_equity if peak_equity > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

        done = (step >= max_steps) or (t_new >= T - 1) or (equity_after < initial_cash * 0.01)
        if done:
            # Close at t_new for final accounting (matches C env behavior).
            if pos is not None:
                price_end = float(data.prices[t_new, pos.sym, P_CLOSE])
                cash, win = _close_position(cash, pos, price_end, fee_rate)
                num_trades += 1
                winning_trades += int(win)
                pos = None

            final_equity = float(cash)
            total_return = (final_equity - initial_equity) / initial_equity

            sortino = 0.0
            if ret_count > 1 and sum_neg_sq > 0.0:
                mean_ret = sum_ret / ret_count
                downside_dev = float(np.sqrt(sum_neg_sq / ret_count))
                ppy = float(periods_per_year) if periods_per_year > 0 else 365.0
                sortino = float(mean_ret / downside_dev * np.sqrt(ppy))

            win_rate = float(winning_trades / num_trades) if num_trades > 0 else 0.0
            avg_hold = float(step / num_trades) if num_trades > 0 else 0.0

            return DailySimResult(
                actions=actions,
                total_return=float(total_return),
                sortino=float(sortino),
                max_drawdown=float(max_dd),
                num_trades=int(num_trades),
                win_rate=float(win_rate),
                avg_hold_steps=float(avg_hold),
            )


@dataclass
class HourlyReplayResult:
    total_return: float
    sortino: float
    max_drawdown: float
    num_trades: int
    num_orders: int
    win_rate: float
    equity_curve: np.ndarray  # float64 [H]
    orders_by_day: dict[str, int]


def _load_hourly_bars(symbol: str, hourly_data_root: Path) -> pd.DataFrame:
    symbol = symbol.upper()
    candidates = [
        hourly_data_root / "crypto" / f"{symbol}.csv",
        hourly_data_root / "stocks" / f"{symbol}.csv",
        hourly_data_root / f"{symbol}.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(f"No hourly bars for {symbol} under {hourly_data_root}")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df.columns = [str(c).lower() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates(subset="timestamp", keep="last")
    df = df.set_index("timestamp")
    if "close" not in df.columns:
        raise ValueError(f"{path} missing 'close' column")
    return df


@dataclass(frozen=True)
class HourlyMarket:
    index: pd.DatetimeIndex
    close: dict[str, np.ndarray]  # float64 [H]
    tradable: dict[str, np.ndarray]  # bool [H]


def load_hourly_market(
    symbols: Iterable[str],
    hourly_data_root: str | Path,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> HourlyMarket:
    """Load and align hourly close series for the given symbol list."""
    hourly_data_root = Path(hourly_data_root)
    start_ts = pd.to_datetime(start, utc=True)
    end_ts = pd.to_datetime(end, utc=True)
    if start_ts.tz is None:
        start_ts = start_ts.tz_localize("UTC")
    if end_ts.tz is None:
        end_ts = end_ts.tz_localize("UTC")
    if end_ts < start_ts:
        raise ValueError(f"end before start: {start_ts} > {end_ts}")

    # Inclusive hourly grid.
    index = pd.date_range(start_ts.floor("h"), end_ts.floor("h"), freq="h", tz="UTC")
    if len(index) < 2:
        raise ValueError("Hourly grid too small")

    close: dict[str, np.ndarray] = {}
    tradable: dict[str, np.ndarray] = {}

    for sym in symbols:
        raw = _load_hourly_bars(sym, hourly_data_root)
        raw = raw.loc[(raw.index >= index[0]) & (raw.index <= index[-1])]

        raw_close = raw["close"].astype(float)
        raw_hours = raw_close.index
        is_bar = pd.Series(True, index=raw_hours, dtype=bool)

        aligned_close = raw_close.reindex(index).ffill().bfill().fillna(0.0)
        aligned_tradable = is_bar.reindex(index, fill_value=False)

        close[sym.upper()] = aligned_close.to_numpy(dtype=np.float64, copy=False)
        tradable[sym.upper()] = aligned_tradable.to_numpy(dtype=bool, copy=False)

    return HourlyMarket(index=index, close=close, tradable=tradable)


def _compute_sortino(returns: np.ndarray, periods_per_year: float) -> float:
    if returns.size < 2:
        return 0.0
    mean_ret = float(np.mean(returns))
    neg = returns[returns < 0.0]
    if neg.size == 0:
        return 0.0
    downside_dev = float(np.sqrt(np.mean(neg * neg)))
    if downside_dev <= 0:
        return 0.0
    ppy = float(periods_per_year) if periods_per_year > 0 else 8760.0
    return float(mean_ret / downside_dev * np.sqrt(ppy))


def replay_hourly_frozen_daily_actions(
    *,
    data: MktdData,
    actions: np.ndarray,
    market: HourlyMarket,
    start_date: str,
    end_date: str,
    max_steps: int,
    fee_rate: float = 0.001,
    max_leverage: float = 1.0,
    periods_per_year: float = 8760.0,
    initial_cash: float = INITIAL_CASH,
) -> HourlyReplayResult:
    """Replay a daily action sequence on hourly prices.

    Trades execute once per calendar day at a single "trade hour" chosen as:
    - NYSE close hour for the reference stock on trading days (if any stock exists)
    - otherwise 23:00 UTC for that calendar day
    """
    symbols = [s.upper() for s in data.symbols]
    S = data.num_symbols
    if actions.shape[0] != max_steps:
        raise ValueError(f"actions length {actions.shape[0]} != max_steps {max_steps}")

    start_day = pd.to_datetime(start_date, utc=True).floor("D")
    end_day = pd.to_datetime(end_date, utc=True).floor("D")
    daily_days = pd.date_range(start_day, end_day, freq="D", tz="UTC")
    if len(daily_days) != data.num_timesteps:
        raise ValueError(
            f"Date range mismatch for daily data: days={len(daily_days)} timesteps={data.num_timesteps}. "
            "Provide the same start/end used during export_data_daily.py."
        )
    if max_steps >= len(daily_days):
        raise ValueError("max_steps must be < num_timesteps (needs t_new)")
    final_day_idx = max_steps  # terminal close uses t_new

    # Pick a reference "stock-like" symbol (if present) to find the daily "stock close hour".
    # In our daily exporter, stocks have tradable=0 on weekends/holidays while crypto is 1 daily.
    ref_stock = None
    ref_idx = None
    if data.tradable is not None:
        for i, sym in enumerate(symbols):
            mask = data.tradable[:, i]
            if bool(np.any(mask == 0)):
                ref_stock = sym
                ref_idx = i
                break

    # Precompute per-day trade timestamps.
    trade_ts: list[pd.Timestamp] = []
    for di, day in enumerate(daily_days):
        ts = day + pd.Timedelta(hours=23)
        if ref_stock is not None and ref_idx is not None and _is_tradable(data, di, ref_idx):
            # Last tradable hourly bar for the reference stock on that day.
            day_mask = (market.index.floor("D") == day)
            ref_tr = market.tradable[ref_stock]
            candidates = market.index[day_mask & ref_tr]
            if len(candidates) > 0:
                ts = candidates.max()
        trade_ts.append(ts)

    trade_ts_set = {t for t in trade_ts}

    cash = float(initial_cash)
    pos: Optional[Position] = None
    num_trades = 0
    winning_trades = 0
    num_orders = 0
    orders_by_day: dict[str, int] = {}

    equity_curve = np.zeros((len(market.index),), dtype=np.float64)
    peak_equity = float(initial_cash)
    max_dd = 0.0

    for hi, ts in enumerate(market.index):
        day = ts.floor("D")
        day_idx = int((day - start_day).days)

        # Execute scheduled daily action at the trade hour for that calendar day.
        if ts in trade_ts_set and 0 <= day_idx < max_steps:
            action = int(actions[day_idx])
            if action < 0 or action > 2 * S:
                action = 0

            cur_sym = pos.sym if pos is not None else -1
            cur_day_tr = _is_tradable(data, day_idx, cur_sym) if cur_sym >= 0 else True
            cur_hr_tr = bool(market.tradable[symbols[cur_sym]][hi]) if cur_sym >= 0 else True
            cur_tradable = bool(cur_day_tr and cur_hr_tr)

            # Convenience for trade prices
            def _hour_price(sym_i: int) -> float:
                return float(market.close[symbols[sym_i]][hi])

            def _count_order() -> None:
                nonlocal num_orders
                num_orders += 1
                key = str(day.date())
                orders_by_day[key] = orders_by_day.get(key, 0) + 1

            if action == 0:
                if pos is not None and cur_tradable:
                    cash, win = _close_position(cash, pos, _hour_price(pos.sym), fee_rate)
                    _count_order()
                    num_trades += 1
                    winning_trades += int(win)
                    pos = None
            elif 1 <= action <= S:
                target = action - 1
                target_day_tr = _is_tradable(data, day_idx, target)
                target_hr_tr = bool(market.tradable[symbols[target]][hi])
                target_tradable = bool(target_day_tr and target_hr_tr)
                if pos is not None and (not pos.is_short) and pos.sym == target:
                    pass
                else:
                    if not target_tradable:
                        pass
                    elif pos is not None and not cur_tradable:
                        pass
                    else:
                        if pos is not None:
                            cash, win = _close_position(cash, pos, _hour_price(pos.sym), fee_rate)
                            _count_order()
                            num_trades += 1
                            winning_trades += int(win)
                        cash, pos = _open_long(cash, target, _hour_price(target), fee_rate, max_leverage)
                        if pos is not None:
                            _count_order()
            else:
                target = action - S - 1
                target_day_tr = _is_tradable(data, day_idx, target)
                target_hr_tr = bool(market.tradable[symbols[target]][hi])
                target_tradable = bool(target_day_tr and target_hr_tr)
                if pos is not None and pos.is_short and pos.sym == target:
                    pass
                else:
                    if not target_tradable:
                        pass
                    elif pos is not None and not cur_tradable:
                        pass
                    else:
                        if pos is not None:
                            cash, win = _close_position(cash, pos, _hour_price(pos.sym), fee_rate)
                            _count_order()
                            num_trades += 1
                            winning_trades += int(win)
                        cash, pos = _open_short(cash, target, _hour_price(target), fee_rate, max_leverage)
                        if pos is not None:
                            _count_order()

        # Mark-to-market equity at this hour.
        if pos is None:
            equity = float(cash)
        else:
            px = float(market.close[symbols[pos.sym]][hi])
            equity = _compute_equity(cash, pos, px)
        equity_curve[hi] = equity
        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    # Terminal close at the final day trade hour (for total_return only).
    final_close_ts = trade_ts[final_day_idx]
    if final_close_ts in market.index and pos is not None:
        hi_end = int(market.index.get_loc(final_close_ts))
        px_end = float(market.close[symbols[pos.sym]][hi_end])
        cash, win = _close_position(cash, pos, px_end, fee_rate)
        num_trades += 1
        winning_trades += int(win)
        num_orders += 1
        key = str(final_close_ts.floor("D").date())
        orders_by_day[key] = orders_by_day.get(key, 0) + 1
        pos = None

    final_equity = float(cash)
    total_return = (final_equity - float(initial_cash)) / float(initial_cash)

    rets = (equity_curve[1:] - equity_curve[:-1]) / np.clip(equity_curve[:-1], 1e-12, None)
    sortino = _compute_sortino(rets.astype(np.float64, copy=False), periods_per_year)
    win_rate = float(winning_trades / num_trades) if num_trades > 0 else 0.0

    return HourlyReplayResult(
        total_return=float(total_return),
        sortino=float(sortino),
        max_drawdown=float(max_dd),
        num_trades=int(num_trades),
        num_orders=int(num_orders),
        win_rate=float(win_rate),
        equity_curve=equity_curve,
        orders_by_day=orders_by_day,
    )


def _build_obs_hourly_price(
    data: MktdData,
    t_day: int,
    pos: Optional[Position],
    cash: float,
    hold_days: int,
    step_day: int,
    max_steps_days: int,
    *,
    price_now_by_sym: Callable[[int], float],
) -> np.ndarray:
    """Observation for running a daily-trained policy at intra-day frequency.

    Per-symbol features come from the daily MKTD row for the calendar day.
    Portfolio fields are computed using the current hourly price.
    """
    S = data.num_symbols
    obs_size = S * FEATURES_PER_SYM + 5 + S
    obs = np.zeros((obs_size,), dtype=np.float32)
    obs[: S * FEATURES_PER_SYM] = data.features[t_day].reshape(-1)

    base = S * FEATURES_PER_SYM
    pos_val = 0.0
    unreal = 0.0
    if pos is not None:
        price = float(price_now_by_sym(pos.sym))
        if pos.is_short:
            pos_val = -(pos.qty * price)
            unreal = pos.qty * (pos.entry_price - price)
        else:
            pos_val = pos.qty * price
            unreal = pos.qty * (price - pos.entry_price)

    denom = float(INITIAL_CASH)
    obs[base + 0] = float(cash / denom)
    obs[base + 1] = float(pos_val / denom)
    obs[base + 2] = float(unreal / denom)
    obs[base + 3] = float(hold_days / (max_steps_days if max_steps_days > 0 else 1))
    obs[base + 4] = float(step_day / (max_steps_days if max_steps_days > 0 else 1))
    if pos is not None:
        obs[base + 5 + pos.sym] = -1.0 if pos.is_short else 1.0
    return obs


def simulate_hourly_policy(
    *,
    data: MktdData,
    policy_fn: Callable[[np.ndarray], int],
    market: HourlyMarket,
    start_date: str,
    end_date: str,
    max_steps_days: int,
    fee_rate: float = 0.001,
    max_leverage: float = 1.0,
    periods_per_year: float = 8760.0,
    initial_cash: float = INITIAL_CASH,
) -> HourlyReplayResult:
    """Execute the policy at hourly frequency using daily features + hourly mark-to-market.

    This is intentionally a "stress test" for double-execution/thrashing:
    we call the daily-trained policy every hour, with per-symbol features frozen
    to the current calendar day, but portfolio fields reflecting the current hour.
    """
    symbols = [s.upper() for s in data.symbols]
    S = data.num_symbols

    start_day = pd.to_datetime(start_date, utc=True).floor("D")
    end_day = pd.to_datetime(end_date, utc=True).floor("D")
    daily_days = pd.date_range(start_day, end_day, freq="D", tz="UTC")
    if len(daily_days) != data.num_timesteps:
        raise ValueError(
            f"Date range mismatch for daily data: days={len(daily_days)} timesteps={data.num_timesteps}. "
            "Provide the same start/end used during export_data_daily.py."
        )
    if max_steps_days >= len(daily_days):
        raise ValueError("max_steps_days must be < num_timesteps (needs terminal day)")
    final_day_idx = max_steps_days

    # Reference stock-like symbol to find a realistic terminal close hour.
    ref_stock = None
    ref_idx = None
    if data.tradable is not None:
        for i, sym in enumerate(symbols):
            mask = data.tradable[:, i]
            if bool(np.any(mask == 0)):
                ref_stock = sym
                ref_idx = i
                break

    trade_ts: list[pd.Timestamp] = []
    for di, day in enumerate(daily_days):
        ts = day + pd.Timedelta(hours=23)
        if ref_stock is not None and ref_idx is not None and _is_tradable(data, di, ref_idx):
            day_mask = (market.index.floor("D") == day)
            ref_tr = market.tradable[ref_stock]
            candidates = market.index[day_mask & ref_tr]
            if len(candidates) > 0:
                ts = candidates.max()
        trade_ts.append(ts)

    final_close_ts = trade_ts[final_day_idx]

    cash = float(initial_cash)
    pos: Optional[Position] = None
    hold_days = 0
    prev_day_idx: Optional[int] = None

    num_trades = 0
    winning_trades = 0
    num_orders = 0
    orders_by_day: dict[str, int] = {}

    equity_curve = np.zeros((len(market.index),), dtype=np.float64)
    peak_equity = float(initial_cash)
    max_dd = 0.0

    def _count_order(day_ts: pd.Timestamp) -> None:
        nonlocal num_orders
        num_orders += 1
        key = str(day_ts.floor("D").date())
        orders_by_day[key] = orders_by_day.get(key, 0) + 1

    for hi, ts in enumerate(market.index):
        day = ts.floor("D")
        day_idx = int((day - start_day).days)
        if day_idx < 0 or day_idx > final_day_idx:
            # Outside evaluation window: no actions, but still mark-to-market.
            if pos is None:
                equity = float(cash)
            else:
                px = float(market.close[symbols[pos.sym]][hi])
                equity = _compute_equity(cash, pos, px)
            equity_curve[hi] = equity
            if equity > peak_equity:
                peak_equity = equity
            dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
            continue

        # Increment hold_days once per day boundary while holding.
        if prev_day_idx is None:
            prev_day_idx = day_idx
        elif day_idx != prev_day_idx:
            if pos is not None:
                hold_days += 1
            prev_day_idx = day_idx

        # Stop issuing policy actions on the terminal day (we'll close at final_close_ts).
        if day_idx < max_steps_days:
            def _hour_price(sym_i: int) -> float:
                return float(market.close[symbols[sym_i]][hi])

            def _hour_tradable(sym_i: int) -> bool:
                return bool(market.tradable[symbols[sym_i]][hi])

            obs = _build_obs_hourly_price(
                data,
                t_day=day_idx,
                pos=pos,
                cash=cash,
                hold_days=hold_days,
                step_day=day_idx,
                max_steps_days=max_steps_days,
                price_now_by_sym=_hour_price,
            )
            action = int(policy_fn(obs))
            if action < 0 or action > 2 * S:
                action = 0

            cur_sym = pos.sym if pos is not None else -1
            cur_day_tr = _is_tradable(data, day_idx, cur_sym) if cur_sym >= 0 else True
            cur_hr_tr = _hour_tradable(cur_sym) if cur_sym >= 0 else True
            cur_tradable = bool(cur_day_tr and cur_hr_tr)

            if action == 0:
                if pos is not None and cur_tradable:
                    cash, win = _close_position(cash, pos, _hour_price(pos.sym), fee_rate)
                    _count_order(ts)
                    num_trades += 1
                    winning_trades += int(win)
                    pos = None
                    hold_days = 0
            elif 1 <= action <= S:
                target = action - 1
                target_day_tr = _is_tradable(data, day_idx, target)
                target_hr_tr = _hour_tradable(target)
                target_tradable = bool(target_day_tr and target_hr_tr)
                if pos is not None and (not pos.is_short) and pos.sym == target:
                    pass
                else:
                    if not target_tradable:
                        pass
                    elif pos is not None and not cur_tradable:
                        pass
                    else:
                        if pos is not None:
                            cash, win = _close_position(cash, pos, _hour_price(pos.sym), fee_rate)
                            _count_order(ts)
                            num_trades += 1
                            winning_trades += int(win)
                        cash, pos = _open_long(cash, target, _hour_price(target), fee_rate, max_leverage)
                        if pos is not None:
                            _count_order(ts)
                            hold_days = 0
            else:
                target = action - S - 1
                target_day_tr = _is_tradable(data, day_idx, target)
                target_hr_tr = _hour_tradable(target)
                target_tradable = bool(target_day_tr and target_hr_tr)
                if pos is not None and pos.is_short and pos.sym == target:
                    pass
                else:
                    if not target_tradable:
                        pass
                    elif pos is not None and not cur_tradable:
                        pass
                    else:
                        if pos is not None:
                            cash, win = _close_position(cash, pos, _hour_price(pos.sym), fee_rate)
                            _count_order(ts)
                            num_trades += 1
                            winning_trades += int(win)
                        cash, pos = _open_short(cash, target, _hour_price(target), fee_rate, max_leverage)
                        if pos is not None:
                            _count_order(ts)
                            hold_days = 0

        # Mark-to-market equity.
        if pos is None:
            equity = float(cash)
        else:
            px = float(market.close[symbols[pos.sym]][hi])
            equity = _compute_equity(cash, pos, px)
        equity_curve[hi] = equity
        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    # Terminal close for total_return only.
    if final_close_ts in market.index and pos is not None:
        hi_end = int(market.index.get_loc(final_close_ts))
        px_end = float(market.close[symbols[pos.sym]][hi_end])
        cash, win = _close_position(cash, pos, px_end, fee_rate)
        _count_order(final_close_ts)
        num_trades += 1
        winning_trades += int(win)
        pos = None

    final_equity = float(cash)
    total_return = (final_equity - float(initial_cash)) / float(initial_cash)

    rets = (equity_curve[1:] - equity_curve[:-1]) / np.clip(equity_curve[:-1], 1e-12, None)
    sortino = _compute_sortino(rets.astype(np.float64, copy=False), periods_per_year)
    win_rate = float(winning_trades / num_trades) if num_trades > 0 else 0.0

    return HourlyReplayResult(
        total_return=float(total_return),
        sortino=float(sortino),
        max_drawdown=float(max_dd),
        num_trades=int(num_trades),
        num_orders=int(num_orders),
        win_rate=float(win_rate),
        equity_curve=equity_curve,
        orders_by_day=orders_by_day,
    )
