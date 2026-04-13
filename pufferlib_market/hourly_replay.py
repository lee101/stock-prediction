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

from src.market_sim_early_exit import (
    evaluate_drawdown_vs_profit_early_exit,
    evaluate_metric_threshold_early_exit,
    print_early_exit,
)

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

        if num_symbols <= 0 or num_symbols > 1024:
            raise ValueError(f"Invalid num_symbols={num_symbols} in {path}")
        if num_timesteps <= 0:
            raise ValueError(f"Invalid num_timesteps={num_timesteps} in {path}")
        if features_per_sym <= 0:
            raise ValueError(f"Invalid features_per_sym={features_per_sym} in {path}")
        if price_features <= 0:
            raise ValueError(f"Invalid price_features={price_features} in {path}")
        # Accept any features_per_sym; the feature array is read generically below.
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


@dataclass(frozen=True)
class InitialPositionSpec:
    symbol: str
    side: str = "long"
    allocation_pct: float = 1.0


@dataclass(frozen=True)
class InitialPortfolioState:
    cash: float
    position: Optional[Position]
    initial_equity: float
    obs_scale: float
    mark_price: float


@dataclass
class DailySimResult:
    actions: np.ndarray  # int32 [max_steps]
    total_return: float
    sortino: float
    max_drawdown: float
    num_trades: int
    win_rate: float
    avg_hold_steps: float
    stopped_early: bool = False
    stop_reason: str = ""
    evaluated_steps: int = 0
    equity_curve: np.ndarray | None = None
    position_history: list[Position | None] | None = None


def _clone_position(pos: Optional[Position]) -> Optional[Position]:
    if pos is None:
        return None
    return Position(
        sym=int(pos.sym),
        is_short=bool(pos.is_short),
        qty=float(pos.qty),
        entry_price=float(pos.entry_price),
    )


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


def _short_borrow_fee(
    *,
    pos: Optional[Position],
    price: float,
    short_borrow_apr: float,
    periods_per_year: float,
) -> float:
    if pos is None or not pos.is_short:
        return 0.0
    if short_borrow_apr <= 0.0:
        return 0.0
    periods_per_year = float(periods_per_year) if periods_per_year > 0 else 8760.0
    if periods_per_year <= 0.0:
        return 0.0
    return float(max(0.0, pos.qty * price) * float(short_borrow_apr) / periods_per_year)


def _apply_short_borrow_cost(
    *,
    cash: float,
    pos: Optional[Position],
    price: float,
    short_borrow_apr: float,
    periods_per_year: float,
) -> tuple[float, float]:
    fee = _short_borrow_fee(
        pos=pos,
        price=price,
        short_borrow_apr=short_borrow_apr,
        periods_per_year=periods_per_year,
    )
    if fee <= 0.0:
        return float(cash), 0.0
    return float(cash - fee), float(fee)


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


def _normalize_fill_buffer_bps(fill_buffer_bps: float) -> float:
    value = float(fill_buffer_bps or 0.0)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"fill_buffer_bps must be finite and >= 0, got {fill_buffer_bps}.")
    return value


def _resolve_limit_fill_price(
    *,
    low: float,
    high: float,
    target_price: float,
    is_buy: bool,
    fill_buffer_bps: float = 0.0,
) -> float | None:
    lo = float(min(low, high))
    hi = float(max(low, high))
    tp = float(target_price)
    fill_buffer = _normalize_fill_buffer_bps(fill_buffer_bps) / 10_000.0
    if is_buy:
        trigger = max(0.0, tp * (1.0 - fill_buffer))
        if lo > trigger:
            return None
    else:
        trigger = tp * (1.0 + fill_buffer)
        if hi < trigger:
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
    fill_buffer_bps: float,
) -> tuple[float, Optional[Position]]:
    if close_price <= 0.0 or cash <= 0.0:
        return float(cash), None
    target_price = float(close_price) * (1.0 + float(level_offset_bps) / 10_000.0)
    fill_price = _resolve_limit_fill_price(
        low=float(low_price),
        high=float(high_price),
        target_price=target_price,
        is_buy=True,
        fill_buffer_bps=fill_buffer_bps,
    )
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
    fill_buffer_bps: float,
) -> tuple[float, Optional[Position]]:
    if close_price <= 0.0 or cash <= 0.0:
        return float(cash), None
    target_price = float(close_price) * (1.0 + float(level_offset_bps) / 10_000.0)
    fill_price = _resolve_limit_fill_price(
        low=float(low_price),
        high=float(high_price),
        target_price=target_price,
        is_buy=False,
        fill_buffer_bps=fill_buffer_bps,
    )
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
    *,
    portfolio_scale: float = INITIAL_CASH,
) -> np.ndarray:
    S = data.num_symbols
    F = int(data.features.shape[2])
    obs_size = S * F + 5 + S
    obs = np.zeros((obs_size,), dtype=np.float32)

    # Mirror C env's 1-bar observation lag (t_obs = t-1, clamped to 0).
    # The agent sees features from the *previous* bar so execution on the
    # current bar's open price does not introduce look-ahead bias.
    t_obs = max(0, t - 1)
    obs[: S * F] = data.features[t_obs].reshape(-1)

    base = S * F
    pos_val = 0.0
    unreal = 0.0
    if pos is not None:
        price = float(data.prices[t_obs, pos.sym, P_CLOSE])
        if pos.is_short:
            pos_val = -(pos.qty * price)
            unreal = pos.qty * (pos.entry_price - price)
        else:
            pos_val = pos.qty * price
            unreal = pos.qty * (price - pos.entry_price)

    denom = max(abs(float(portfolio_scale)), 1e-12)
    obs[base + 0] = float(cash / denom)
    obs[base + 1] = float(pos_val / denom)
    obs[base + 2] = float(unreal / denom)
    obs[base + 3] = float(hold_steps / (max_steps if max_steps > 0 else 1))
    obs[base + 4] = float(step / (max_steps if max_steps > 0 else 1))

    # one-hot position (1 long, -1 short)
    if pos is not None:
        obs[base + 5 + pos.sym] = -1.0 if pos.is_short else 1.0

    return obs


def _normalize_initial_position_spec(
    initial_position: InitialPositionSpec | None,
) -> InitialPositionSpec | None:
    if initial_position is None:
        return None
    symbol = str(initial_position.symbol).strip().upper()
    side = str(initial_position.side).strip().lower()
    allocation_pct = float(initial_position.allocation_pct)
    if not symbol:
        raise ValueError("initial_position.symbol must be non-empty")
    if side not in {"long", "short"}:
        raise ValueError(f"initial_position.side must be 'long' or 'short', got {initial_position.side!r}")
    if not np.isfinite(allocation_pct) or allocation_pct < 0.0 or allocation_pct > 1.0:
        raise ValueError(
            f"initial_position.allocation_pct must be finite and in [0, 1], got {initial_position.allocation_pct!r}"
        )
    if allocation_pct <= 0.0:
        return None
    return InitialPositionSpec(symbol=symbol, side=side, allocation_pct=allocation_pct)


def _first_positive_market_close(
    prices: np.ndarray,
    tradable: np.ndarray | None = None,
) -> float:
    values = np.asarray(prices, dtype=np.float64)
    mask = np.isfinite(values) & (values > 0.0)
    if tradable is not None:
        mask &= np.asarray(tradable, dtype=bool)
    idx = np.flatnonzero(mask)
    if len(idx) <= 0:
        return float(values[0]) if values.size > 0 else 0.0
    return float(values[int(idx[0])])


def _positive_price_or_fallback(primary: float, fallback: float) -> float:
    if np.isfinite(primary) and primary > 0.0:
        return float(primary)
    if np.isfinite(fallback) and fallback > 0.0:
        return float(fallback)
    return float(primary)


def _build_initial_portfolio_state(
    *,
    symbols: list[str],
    initial_cash: float,
    initial_position: InitialPositionSpec | None,
    fee_rate: float,
    max_leverage: float,
    price_by_sym: Callable[[int], float],
) -> InitialPortfolioState:
    cash = float(initial_cash)
    pos: Optional[Position] = None
    mark_price = 0.0

    spec = _normalize_initial_position_spec(initial_position)
    if spec is not None:
        try:
            sym_idx = symbols.index(spec.symbol)
        except ValueError as exc:
            raise ValueError(
                f"initial_position.symbol {spec.symbol!r} not found in symbols {symbols}"
            ) from exc
        mark_price = float(price_by_sym(sym_idx))
        if not np.isfinite(mark_price) or mark_price <= 0.0:
            raise ValueError(
                f"initial_position for {spec.symbol} requires a positive mark price, got {mark_price!r}"
            )
        open_kwargs = dict(
            cash=cash,
            sym=sym_idx,
            close_price=mark_price,
            low_price=mark_price,
            high_price=mark_price,
            fee_rate=fee_rate,
            max_leverage=max_leverage,
            allocation_pct=spec.allocation_pct,
            level_offset_bps=0.0,
            fill_buffer_bps=0.0,
        )
        if spec.side == "short":
            cash, pos = _open_short_limit(**open_kwargs)
        else:
            cash, pos = _open_long_limit(**open_kwargs)
        if pos is None:
            raise ValueError(
                f"Unable to open initial {spec.side} position for {spec.symbol} at price {mark_price:.6f}"
            )

    initial_equity = _compute_equity(cash, pos, mark_price) if pos is not None else float(cash)
    obs_scale = max(abs(float(initial_equity)), 1e-12)
    return InitialPortfolioState(
        cash=float(cash),
        position=pos,
        initial_equity=float(initial_equity),
        obs_scale=float(obs_scale),
        mark_price=float(mark_price),
    )


def simulate_daily_policy(
    data: MktdData,
    policy_fn: Callable[[np.ndarray], int],
    *,
    max_steps: int,
    fee_rate: float = 0.001,
    slippage_bps: float = 0.0,
    max_leverage: float = 1.0,
    periods_per_year: float = 365.0,
    short_borrow_apr: float = 0.0,
    initial_cash: float = INITIAL_CASH,
    initial_position: InitialPositionSpec | None = None,
    action_allocation_bins: int = 1,
    action_level_bins: int = 1,
    action_max_offset_bps: float = 0.0,
    fill_buffer_bps: float = 0.0,
    trailing_stop_pct: float = 0.0,
    max_hold_bars: int = 0,
    min_notional_usd: float = 0.0,
    enable_drawdown_profit_early_exit: bool = True,
    drawdown_profit_early_exit_verbose: bool = True,
    drawdown_profit_early_exit_min_steps: int = 20,
    drawdown_profit_early_exit_progress_fraction: float = 0.5,
    early_exit_max_drawdown: float | None = None,
    early_exit_min_sortino: float | None = None,
) -> DailySimResult:
    """Pure-python simulation matching the C env's daily step semantics.

    This is used to generate a daily action trace (and a baseline metric set)
    without relying on the compiled binding.

    Production-fidelity parameters:
        slippage_bps: Adverse fill slippage in basis points applied on top of
            fee_rate (e.g. 3bps for Alpaca crypto). Buy fills at price*(1+slip),
            sell fills at price*(1-slip). Default 0 (no extra slippage).
        trailing_stop_pct: Fraction below peak-since-entry that triggers a forced
            exit at the current bar's close (e.g. 0.003 = 0.3%). 0 = disabled.
            Matches production TRAILING_STOP_PCT = 0.003.
        max_hold_bars: Force-exit after holding this many bars (e.g. 6 for 6
            daily steps). 0 = disabled. Matches production MAX_HOLD_HOURS = 6.
        min_notional_usd: Skip opening a position if the dollar value of the
            position would be below this threshold (e.g. 12.0 for $12 minimum).
            0 = disabled. Matches production min notional check.
    """
    S = data.num_symbols
    T = data.num_timesteps
    if max_steps < 1 or max_steps >= T:
        raise ValueError(f"max_steps must be in [1, {T-1}] (got {max_steps})")
    fill_buffer_bps = _normalize_fill_buffer_bps(fill_buffer_bps)

    # Slippage: applied on top of fee_rate as adverse execution cost.
    # Buys fill at price*(1+slip), sells at price*(1-slip). We model this by
    # adding slippage_bps/10000 to the effective fee_rate for position opens and
    # using it symmetrically on closes. This exactly matches production where
    # Alpaca crypto has 0 commission but ~2-5bps market-impact slippage.
    slip_frac = max(0.0, float(slippage_bps)) / 10_000.0
    effective_fee = float(fee_rate) + slip_frac

    # Production constraints
    _trailing_stop = max(0.0, float(trailing_stop_pct))
    _max_hold = max(0, int(max_hold_bars))
    _min_notional = max(0.0, float(min_notional_usd))

    init_state = _build_initial_portfolio_state(
        symbols=[s.upper() for s in data.symbols],
        initial_cash=initial_cash,
        initial_position=initial_position,
        fee_rate=effective_fee,
        max_leverage=max_leverage,
        price_by_sym=lambda sym_idx: float(data.prices[0, sym_idx, P_CLOSE]),
    )

    cash = float(init_state.cash)
    pos: Optional[Position] = init_state.position
    pos_peak_price: float = 0.0   # peak close price since entry (for trailing stop)
    if pos is not None and not pos.is_short:
        pos_peak_price = float(pos.entry_price)
    hold_steps = 0
    step = 0
    peak_equity = float(init_state.initial_equity)
    max_dd = 0.0
    initial_equity = float(init_state.initial_equity)

    num_trades = 0
    winning_trades = 0

    sum_ret = 0.0
    sum_neg_sq = 0.0
    ret_count = 0

    actions = np.zeros((max_steps,), dtype=np.int32)
    equity_history: list[float] = [float(initial_equity)]
    position_history: list[Position | None] = []
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

        # Update peak price for trailing stop tracking (long positions only)
        if pos is not None and not pos.is_short and price_cur > 0.0:
            if price_cur > pos_peak_price:
                pos_peak_price = price_cur

        obs = _build_obs(
            data,
            t,
            pos,
            cash,
            hold_steps,
            step,
            max_steps,
            portfolio_scale=init_state.obs_scale,
        )
        action = int(policy_fn(obs))
        if action < 0 or action > 2 * side_block:
            action = -1
        if step < max_steps:
            actions[step] = action

        # Execute action (same logic as c_step)
        if action == 0:
            if pos is not None:
                if cur_tradable:
                    cash, win = _close_position(cash, pos, price_cur, effective_fee)
                    num_trades += 1
                    winning_trades += int(win)
                    pos = None
                    pos_peak_price = 0.0
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
                        cash, win = _close_position(cash, pos, price_cur, effective_fee)
                        num_trades += 1
                        winning_trades += int(win)
                        pos = None
                        pos_peak_price = 0.0
                    close_px = float(data.prices[t, target_sym, P_CLOSE])
                    low_px = float(data.prices[t, target_sym, P_LOW])
                    high_px = float(data.prices[t, target_sym, P_HIGH])
                    # Min notional check: skip if position value would be too small
                    open_budget = cash * max_leverage * target_alloc
                    if _min_notional > 0.0 and open_budget < _min_notional:
                        pass  # below min notional — stay flat
                    elif is_short_target:
                        cash, pos = _open_short_limit(
                            cash=cash,
                            sym=target_sym,
                            close_price=close_px,
                            low_price=low_px,
                            high_price=high_px,
                            fee_rate=effective_fee,
                            max_leverage=max_leverage,
                            allocation_pct=target_alloc,
                            level_offset_bps=level_bps,
                            fill_buffer_bps=fill_buffer_bps,
                        )
                        pos_peak_price = 0.0  # trailing stop not applied to shorts
                    else:
                        cash, pos = _open_long_limit(
                            cash=cash,
                            sym=target_sym,
                            close_price=close_px,
                            low_price=low_px,
                            high_price=high_px,
                            fee_rate=effective_fee,
                            max_leverage=max_leverage,
                            allocation_pct=target_alloc,
                            level_offset_bps=level_bps,
                            fill_buffer_bps=fill_buffer_bps,
                        )
                        # Initialise peak price to the fill price for trailing stop
                        pos_peak_price = float(pos.entry_price) if pos is not None else 0.0
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
            cash, borrow_fee = _apply_short_borrow_cost(
                cash=cash,
                pos=pos,
                price=price_new,
                short_borrow_apr=short_borrow_apr,
                periods_per_year=periods_per_year,
            )
            equity_after = _compute_equity(cash, pos, price_new)

            # --- Trailing stop (long positions only, matches production 0.3% below peak) ---
            if (
                _trailing_stop > 0.0
                and not pos.is_short
                and pos_peak_price > 0.0
                and price_new < pos_peak_price * (1.0 - _trailing_stop)
            ):
                cash, win = _close_position(cash, pos, price_new, effective_fee)
                num_trades += 1
                winning_trades += int(win)
                pos = None
                pos_peak_price = 0.0
                hold_steps = 0
                equity_after = float(cash)

            # --- Max hold: force exit after max_hold_bars bars ---
            elif _max_hold > 0 and pos is not None and hold_steps >= _max_hold:
                cash, win = _close_position(cash, pos, price_new, effective_fee)
                num_trades += 1
                winning_trades += int(win)
                pos = None
                pos_peak_price = 0.0
                hold_steps = 0
                equity_after = float(cash)

            # Update peak price for next bar
            elif pos is not None and not pos.is_short and price_new > pos_peak_price:
                pos_peak_price = price_new

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
        equity_history.append(float(equity_after))
        position_history.append(_clone_position(pos))

        early_exit = None
        if enable_drawdown_profit_early_exit:
            drawdown_profit_exit = evaluate_drawdown_vs_profit_early_exit(
                equity_history,
                total_steps=max_steps + 1,
                label="pufferlib_market.simulate_daily_policy",
                min_total_steps=drawdown_profit_early_exit_min_steps,
                progress_fraction=drawdown_profit_early_exit_progress_fraction,
            )
            if drawdown_profit_exit.should_stop:
                if drawdown_profit_early_exit_verbose:
                    print_early_exit(drawdown_profit_exit)
                early_exit = drawdown_profit_exit

        metric_threshold_exit = evaluate_metric_threshold_early_exit(
            equity_history,
            total_steps=max_steps + 1,
            label="pufferlib_market.simulate_daily_policy",
            periods_per_year=periods_per_year,
            max_drawdown_limit=early_exit_max_drawdown,
            min_sortino_limit=early_exit_min_sortino,
            min_total_steps=drawdown_profit_early_exit_min_steps,
            progress_fraction=drawdown_profit_early_exit_progress_fraction,
        )
        if metric_threshold_exit.should_stop:
            print_early_exit(metric_threshold_exit)
            early_exit = metric_threshold_exit

        done = (
            (early_exit is not None and early_exit.should_stop)
            or (step >= max_steps)
            or (t_new >= T - 1)
            or (equity_after < initial_equity * 0.01)
        )
        if done:
            # Close at t_new for final accounting (matches C env behavior).
            if pos is not None:
                price_end = float(data.prices[t_new, pos.sym, P_CLOSE])
                cash, win = _close_position(cash, pos, price_end, effective_fee)
                num_trades += 1
                winning_trades += int(win)
                pos = None
                pos_peak_price = 0.0

            final_equity = float(cash)
            total_return = (final_equity - initial_equity) / initial_equity if initial_equity != 0.0 else 0.0

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
                stopped_early=bool(early_exit is not None and early_exit.should_stop),
                stop_reason=early_exit.reason if early_exit is not None else "",
                evaluated_steps=int(step),
                equity_curve=np.asarray(equity_history, dtype=np.float64),
                position_history=position_history,
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
    aligned_index = df.index.floor("h")
    if not aligned_index.equals(df.index):
        df = (
            df.assign(_aligned_timestamp=aligned_index)
            .groupby("_aligned_timestamp", sort=True)
            .last()
        )
        df.index.name = "timestamp"
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


def _build_hourly_day_coverage(
    market: HourlyMarket,
    symbols: list[str],
    *,
    start_day: pd.Timestamp,
    num_days: int,
) -> dict[str, np.ndarray]:
    coverage: dict[str, np.ndarray] = {}
    market_days = market.index.floor("D")
    for sym in symbols:
        covered = np.zeros((num_days,), dtype=bool)
        tradable = np.asarray(
            market.tradable.get(sym.upper(), np.zeros((len(market.index),), dtype=bool)),
            dtype=bool,
        )
        for hi in np.flatnonzero(tradable):
            day_idx = int((market_days[hi] - start_day).days)
            if 0 <= day_idx < num_days:
                covered[day_idx] = True
        coverage[sym.upper()] = covered
    return coverage


def replay_hourly_frozen_daily_actions(
    *,
    data: MktdData,
    actions: np.ndarray,
    market: HourlyMarket,
    start_date: str,
    end_date: str,
    max_steps: int,
    fee_rate: float = 0.001,
    slippage_bps: float = 0.0,
    fill_buffer_bps: float = 0.0,
    max_leverage: float = 1.0,
    periods_per_year: float = 8760.0,
    short_borrow_apr: float = 0.0,
    initial_cash: float = INITIAL_CASH,
    initial_position: InitialPositionSpec | None = None,
    action_allocation_bins: int = 1,
    action_level_bins: int = 1,
    action_max_offset_bps: float = 0.0,
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
    fill_buffer_bps = _normalize_fill_buffer_bps(fill_buffer_bps)
    slip_frac = max(0.0, float(slippage_bps)) / 10_000.0
    effective_fee = float(fee_rate) + slip_frac

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
    hourly_day_coverage = _build_hourly_day_coverage(
        market,
        symbols,
        start_day=start_day,
        num_days=len(daily_days),
    )

    def _hour_price_at(sym_i: int, *, hi: int, day_idx: int) -> float:
        market_price = float(market.close[symbols[sym_i]][hi])
        daily_price = float(data.prices[min(max(day_idx, 0), data.num_timesteps - 1), sym_i, P_CLOSE])
        return _positive_price_or_fallback(market_price, daily_price)

    def _hour_tradable_at(sym_i: int, *, hi: int, day_idx: int, ts: pd.Timestamp) -> bool:
        if 0 <= day_idx < len(daily_days) and hourly_day_coverage[symbols[sym_i]][day_idx]:
            return bool(market.tradable[symbols[sym_i]][hi])
        if 0 <= day_idx < len(daily_days):
            return bool(_is_tradable(data, day_idx, sym_i) and ts == trade_ts[day_idx])
        return False

    init_state = _build_initial_portfolio_state(
        symbols=symbols,
        initial_cash=initial_cash,
        initial_position=initial_position,
        fee_rate=effective_fee,
        max_leverage=max_leverage,
        price_by_sym=lambda sym_idx: _positive_price_or_fallback(
            _first_positive_market_close(
                market.close[symbols[sym_idx]],
                market.tradable.get(symbols[sym_idx]),
            ),
            float(data.prices[0, sym_idx, P_CLOSE]),
        ),
    )

    cash = float(init_state.cash)
    pos: Optional[Position] = init_state.position
    num_trades = 0
    winning_trades = 0
    num_orders = 0
    orders_by_day: dict[str, int] = {}
    alloc_bins = max(1, int(action_allocation_bins))
    level_bins = max(1, int(action_level_bins))
    per_symbol_actions = alloc_bins * level_bins
    side_block = S * per_symbol_actions

    equity_curve = np.zeros((len(market.index),), dtype=np.float64)
    peak_equity = float(init_state.initial_equity)
    max_dd = 0.0
    stopped_early = False
    last_hi = -1
    last_ts: pd.Timestamp | None = None
    stopped_early = False
    last_hi = -1
    last_ts: pd.Timestamp | None = None

    for hi, ts in enumerate(market.index):
        day = ts.floor("D")
        day_idx = int((day - start_day).days)

        if hi > 0 and pos is not None:
            px_carry = _hour_price_at(pos.sym, hi=hi, day_idx=day_idx)
            cash, _ = _apply_short_borrow_cost(
                cash=cash,
                pos=pos,
                price=px_carry,
                short_borrow_apr=short_borrow_apr,
                periods_per_year=periods_per_year,
            )

        # Execute scheduled daily action at the trade hour for that calendar day.
        if ts in trade_ts_set and 0 <= day_idx < max_steps:
            action = int(actions[day_idx])
            if action < 0 or action > 2 * side_block:
                action = 0

            cur_sym = pos.sym if pos is not None else -1
            cur_day_tr = _is_tradable(data, day_idx, cur_sym) if cur_sym >= 0 else True
            cur_hr_tr = _hour_tradable_at(cur_sym, hi=hi, day_idx=day_idx, ts=ts) if cur_sym >= 0 else True
            cur_tradable = bool(cur_day_tr and cur_hr_tr)

            # Convenience for trade prices
            def _hour_price(sym_i: int) -> float:
                return _hour_price_at(sym_i, hi=hi, day_idx=day_idx)

            def _count_order() -> None:
                nonlocal num_orders
                num_orders += 1
                key = str(day.date())
                orders_by_day[key] = orders_by_day.get(key, 0) + 1

            if action == 0:
                if pos is not None and cur_tradable:
                    cash, win = _close_position(cash, pos, _hour_price(pos.sym), effective_fee)
                    _count_order()
                    num_trades += 1
                    winning_trades += int(win)
                    pos = None
            elif 1 <= action <= 2 * side_block:
                action_idx = action - 1
                is_short_target = action_idx >= side_block
                if is_short_target:
                    action_idx -= side_block
                target = action_idx // per_symbol_actions
                rem = action_idx % per_symbol_actions
                alloc_idx = rem // level_bins
                level_idx = rem % level_bins
                target_alloc = _action_allocation_pct(alloc_idx=alloc_idx, alloc_bins=alloc_bins)
                level_bps = _action_level_offset_bps(
                    level_idx=level_idx,
                    level_bins=level_bins,
                    max_offset_bps=action_max_offset_bps,
                )
                target_day_tr = _is_tradable(data, day_idx, target)
                target_hr_tr = _hour_tradable_at(target, hi=hi, day_idx=day_idx, ts=ts)
                target_tradable = bool(target_day_tr and target_hr_tr)
                if pos is not None and (pos.is_short == is_short_target) and pos.sym == target:
                    pass
                else:
                    if not target_tradable:
                        pass
                    elif pos is not None and not cur_tradable:
                        pass
                    else:
                        if pos is not None:
                            cash, win = _close_position(cash, pos, _hour_price(pos.sym), effective_fee)
                            _count_order()
                            num_trades += 1
                            winning_trades += int(win)
                        hour_price = _hour_price(target)
                        open_kwargs = dict(
                            cash=cash,
                            sym=target,
                            close_price=hour_price,
                            low_price=hour_price,
                            high_price=hour_price,
                            fee_rate=effective_fee,
                            max_leverage=max_leverage,
                            allocation_pct=target_alloc,
                            level_offset_bps=level_bps,
                            fill_buffer_bps=fill_buffer_bps,
                        )
                        if is_short_target:
                            cash, pos = _open_short_limit(**open_kwargs)
                        else:
                            cash, pos = _open_long_limit(**open_kwargs)
                        if pos is not None:
                            _count_order()

        # Mark-to-market equity at this hour.
        if pos is None:
            equity = float(cash)
        else:
            px = _hour_price_at(pos.sym, hi=hi, day_idx=day_idx)
            equity = _compute_equity(cash, pos, px)
        equity_curve[hi] = equity
        last_hi = hi
        last_ts = ts
        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
        early_exit = evaluate_drawdown_vs_profit_early_exit(
            equity_curve[: hi + 1],
            total_steps=len(market.index),
            label="pufferlib_market.replay_hourly_frozen_daily_actions",
        )
        if early_exit.should_stop:
            print_early_exit(early_exit)
            stopped_early = True
            break

    if stopped_early:
        if pos is not None and last_hi >= 0 and last_ts is not None:
            px_end = _hour_price_at(pos.sym, hi=last_hi, day_idx=int((last_ts.floor("D") - start_day).days))
            cash, win = _close_position(cash, pos, px_end, effective_fee)
            num_trades += 1
            winning_trades += int(win)
            num_orders += 1
            key = str(last_ts.floor("D").date())
            orders_by_day[key] = orders_by_day.get(key, 0) + 1
            pos = None
            equity_curve[last_hi] = float(cash)
        used_equity_curve = equity_curve[: max(last_hi + 1, 0)]
    else:
        # Terminal close at the final day trade hour (for total_return only).
        final_close_ts = trade_ts[final_day_idx]
        if final_close_ts in market.index and pos is not None:
            hi_end = int(market.index.get_loc(final_close_ts))
            px_end = _hour_price_at(pos.sym, hi=hi_end, day_idx=final_day_idx)
            cash, win = _close_position(cash, pos, px_end, effective_fee)
            num_trades += 1
            winning_trades += int(win)
            num_orders += 1
            key = str(final_close_ts.floor("D").date())
            orders_by_day[key] = orders_by_day.get(key, 0) + 1
            pos = None
        used_equity_curve = equity_curve

    final_equity = float(cash)
    initial_equity = float(init_state.initial_equity)
    total_return = (final_equity - initial_equity) / initial_equity if initial_equity != 0.0 else 0.0

    rets = (used_equity_curve[1:] - used_equity_curve[:-1]) / np.clip(used_equity_curve[:-1], 1e-12, None)
    sortino = _compute_sortino(rets.astype(np.float64, copy=False), periods_per_year)
    win_rate = float(winning_trades / num_trades) if num_trades > 0 else 0.0

    return HourlyReplayResult(
        total_return=float(total_return),
        sortino=float(sortino),
        max_drawdown=float(max_dd),
        num_trades=int(num_trades),
        num_orders=int(num_orders),
        win_rate=float(win_rate),
        equity_curve=used_equity_curve,
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
    portfolio_scale: float = INITIAL_CASH,
) -> np.ndarray:
    """Observation for running a daily-trained policy at intra-day frequency.

    Per-symbol features come from the daily MKTD row for the *previous* calendar
    day (1-bar lag matching the C env) to avoid look-ahead bias.
    Portfolio fields are computed using the current hourly price.
    """
    S = data.num_symbols
    F = int(data.features.shape[2])
    obs_size = S * F + 5 + S
    obs = np.zeros((obs_size,), dtype=np.float32)
    t_obs = max(0, t_day - 1)
    obs[: S * F] = data.features[t_obs].reshape(-1)

    base = S * F
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

    denom = max(abs(float(portfolio_scale)), 1e-12)
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
    slippage_bps: float = 0.0,
    fill_buffer_bps: float = 0.0,
    max_leverage: float = 1.0,
    periods_per_year: float = 8760.0,
    short_borrow_apr: float = 0.0,
    initial_cash: float = INITIAL_CASH,
    initial_position: InitialPositionSpec | None = None,
    action_allocation_bins: int = 1,
    action_level_bins: int = 1,
    action_max_offset_bps: float = 0.0,
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
    fill_buffer_bps = _normalize_fill_buffer_bps(fill_buffer_bps)
    slip_frac = max(0.0, float(slippage_bps)) / 10_000.0
    effective_fee = float(fee_rate) + slip_frac

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
    hourly_day_coverage = _build_hourly_day_coverage(
        market,
        symbols,
        start_day=start_day,
        num_days=len(daily_days),
    )

    def _hour_price_at(sym_i: int, *, hi: int, day_idx: int) -> float:
        market_price = float(market.close[symbols[sym_i]][hi])
        daily_price = float(data.prices[min(max(day_idx, 0), data.num_timesteps - 1), sym_i, P_CLOSE])
        return _positive_price_or_fallback(market_price, daily_price)

    def _hour_tradable_at(sym_i: int, *, hi: int, day_idx: int, ts: pd.Timestamp) -> bool:
        if 0 <= day_idx < len(daily_days) and hourly_day_coverage[symbols[sym_i]][day_idx]:
            return bool(market.tradable[symbols[sym_i]][hi])
        if 0 <= day_idx < len(daily_days):
            return bool(_is_tradable(data, day_idx, sym_i) and ts == trade_ts[day_idx])
        return False

    init_state = _build_initial_portfolio_state(
        symbols=symbols,
        initial_cash=initial_cash,
        initial_position=initial_position,
        fee_rate=effective_fee,
        max_leverage=max_leverage,
        price_by_sym=lambda sym_idx: _positive_price_or_fallback(
            _first_positive_market_close(
                market.close[symbols[sym_idx]],
                market.tradable.get(symbols[sym_idx]),
            ),
            float(data.prices[0, sym_idx, P_CLOSE]),
        ),
    )

    cash = float(init_state.cash)
    pos: Optional[Position] = init_state.position
    hold_days = 0
    prev_day_idx: Optional[int] = None

    num_trades = 0
    winning_trades = 0
    num_orders = 0
    orders_by_day: dict[str, int] = {}
    alloc_bins = max(1, int(action_allocation_bins))
    level_bins = max(1, int(action_level_bins))
    per_symbol_actions = alloc_bins * level_bins
    side_block = S * per_symbol_actions

    equity_curve = np.zeros((len(market.index),), dtype=np.float64)
    peak_equity = float(init_state.initial_equity)
    max_dd = 0.0
    stopped_early = False
    last_hi = -1
    last_ts: pd.Timestamp | None = None

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
                px = _hour_price_at(pos.sym, hi=hi, day_idx=day_idx)
                equity = _compute_equity(cash, pos, px)
            equity_curve[hi] = equity
            if equity > peak_equity:
                peak_equity = equity
            dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
            continue

        if hi > 0 and pos is not None:
            px_carry = _hour_price_at(pos.sym, hi=hi, day_idx=day_idx)
            cash, _ = _apply_short_borrow_cost(
                cash=cash,
                pos=pos,
                price=px_carry,
                short_borrow_apr=short_borrow_apr,
                periods_per_year=periods_per_year,
            )

        # Detect day boundary — but defer the hold_days increment until AFTER obs/action.
        # C env convention: build_observation() runs BEFORE the action each step, so the
        # first obs while holding a new position always shows hold_hours=0 (open_long() resets
        # to 0 and the next step's obs is built before any HOLD increments it).  If we
        # incremented hold_days here (before obs), the first hour of every new calendar day
        # would show hold_days=1 instead of the correct 0 on day-1-after-buying.
        was_holding_before_action = pos is not None
        day_crossed = False
        if prev_day_idx is None:
            prev_day_idx = day_idx
        elif day_idx != prev_day_idx:
            day_crossed = True
            prev_day_idx = day_idx

        # Stop issuing policy actions on the terminal day (we'll close at final_close_ts).
        if day_idx < max_steps_days:
            def _hour_price(sym_i: int) -> float:
                return _hour_price_at(sym_i, hi=hi, day_idx=day_idx)

            def _hour_tradable(sym_i: int) -> bool:
                return _hour_tradable_at(sym_i, hi=hi, day_idx=day_idx, ts=ts)

            obs = _build_obs_hourly_price(
                data,
                t_day=day_idx,
                pos=pos,
                cash=cash,
                hold_days=hold_days,
                step_day=day_idx,
                max_steps_days=max_steps_days,
                price_now_by_sym=_hour_price,
                portfolio_scale=init_state.obs_scale,
            )
            action = int(policy_fn(obs))
            if action < 0 or action > 2 * side_block:
                action = 0

            cur_sym = pos.sym if pos is not None else -1
            cur_day_tr = _is_tradable(data, day_idx, cur_sym) if cur_sym >= 0 else True
            cur_hr_tr = _hour_tradable(cur_sym) if cur_sym >= 0 else True
            cur_tradable = bool(cur_day_tr and cur_hr_tr)

            if action == 0:
                if pos is not None and cur_tradable:
                    cash, win = _close_position(cash, pos, _hour_price(pos.sym), effective_fee)
                    _count_order(ts)
                    num_trades += 1
                    winning_trades += int(win)
                    pos = None
                    hold_days = 0
            elif 1 <= action <= 2 * side_block:
                action_idx = action - 1
                is_short_target = action_idx >= side_block
                if is_short_target:
                    action_idx -= side_block
                target = action_idx // per_symbol_actions
                rem = action_idx % per_symbol_actions
                alloc_idx = rem // level_bins
                level_idx = rem % level_bins
                target_alloc = _action_allocation_pct(alloc_idx=alloc_idx, alloc_bins=alloc_bins)
                level_bps = _action_level_offset_bps(
                    level_idx=level_idx,
                    level_bins=level_bins,
                    max_offset_bps=action_max_offset_bps,
                )
                target_day_tr = _is_tradable(data, day_idx, target)
                target_hr_tr = _hour_tradable(target)
                target_tradable = bool(target_day_tr and target_hr_tr)
                if pos is not None and (pos.is_short == is_short_target) and pos.sym == target:
                    pass
                else:
                    if not target_tradable:
                        pass
                    elif pos is not None and not cur_tradable:
                        pass
                    else:
                        if pos is not None:
                            cash, win = _close_position(cash, pos, _hour_price(pos.sym), effective_fee)
                            _count_order(ts)
                            num_trades += 1
                            winning_trades += int(win)
                        hour_price = _hour_price(target)
                        open_kwargs = dict(
                            cash=cash,
                            sym=target,
                            close_price=hour_price,
                            low_price=hour_price,
                            high_price=hour_price,
                            fee_rate=effective_fee,
                            max_leverage=max_leverage,
                            allocation_pct=target_alloc,
                            level_offset_bps=level_bps,
                            fill_buffer_bps=fill_buffer_bps,
                        )
                        if is_short_target:
                            cash, pos = _open_short_limit(**open_kwargs)
                        else:
                            cash, pos = _open_long_limit(**open_kwargs)
                        if pos is not None:
                            _count_order(ts)
                            hold_days = 0

        # Deferred hold_days increment: apply the day-crossing increment NOW (after obs/action)
        # so the first obs of a new calendar day still shows the pre-increment value.
        # Only increment if we were already holding BEFORE this action (a fresh buy on this
        # same hour should not immediately bump the hold counter).
        if day_crossed and was_holding_before_action and pos is not None:
            hold_days += 1

        # Mark-to-market equity.
        if pos is None:
            equity = float(cash)
        else:
            px = _hour_price_at(pos.sym, hi=hi, day_idx=day_idx)
            equity = _compute_equity(cash, pos, px)
        equity_curve[hi] = equity
        last_hi = hi
        last_ts = ts
        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
        early_exit = evaluate_drawdown_vs_profit_early_exit(
            equity_curve[: hi + 1],
            total_steps=len(market.index),
            label="pufferlib_market.simulate_hourly_policy",
        )
        if early_exit.should_stop:
            print_early_exit(early_exit)
            stopped_early = True
            break

    if stopped_early:
        if pos is not None and last_hi >= 0 and last_ts is not None:
            px_end = _hour_price_at(pos.sym, hi=last_hi, day_idx=int((last_ts.floor("D") - start_day).days))
            cash, win = _close_position(cash, pos, px_end, effective_fee)
            _count_order(last_ts)
            num_trades += 1
            winning_trades += int(win)
            pos = None
            equity_curve[last_hi] = float(cash)
        used_equity_curve = equity_curve[: max(last_hi + 1, 0)]
    else:
        # Terminal close for total_return only.
        if final_close_ts in market.index and pos is not None:
            hi_end = int(market.index.get_loc(final_close_ts))
            px_end = _hour_price_at(pos.sym, hi=hi_end, day_idx=final_day_idx)
            cash, win = _close_position(cash, pos, px_end, effective_fee)
            _count_order(final_close_ts)
            num_trades += 1
            winning_trades += int(win)
            pos = None
        used_equity_curve = equity_curve

    final_equity = float(cash)
    initial_equity = float(init_state.initial_equity)
    total_return = (final_equity - initial_equity) / initial_equity if initial_equity != 0.0 else 0.0

    rets = (used_equity_curve[1:] - used_equity_curve[:-1]) / np.clip(used_equity_curve[:-1], 1e-12, None)
    sortino = _compute_sortino(rets.astype(np.float64, copy=False), periods_per_year)
    win_rate = float(winning_trades / num_trades) if num_trades > 0 else 0.0

    return HourlyReplayResult(
        total_return=float(total_return),
        sortino=float(sortino),
        max_drawdown=float(max_dd),
        num_trades=int(num_trades),
        num_orders=int(num_orders),
        win_rate=float(win_rate),
        equity_curve=used_equity_curve,
        orders_by_day=orders_by_day,
    )
