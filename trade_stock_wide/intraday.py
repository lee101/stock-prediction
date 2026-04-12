from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

from .planner import WidePlannerConfig, build_wide_plan
from .runtime_logging import WideRunLogger
from .types import DaySimulationResult, FillResult, WideCandidate, WideOrder


HOURLY_SEARCH_DIRS = (
    Path(""),
    Path("stocks"),
    Path("crypto"),
    Path("stocks/stocks"),
    Path("crypto/crypto"),
)


def _normalize_hourly_frame(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.rename(columns={column: str(column).strip().lower() for column in frame.columns}).copy()
    required = {"timestamp", "open", "high", "low", "close"}
    missing = sorted(required - set(renamed.columns))
    if missing:
        raise KeyError(f"Missing required hourly columns: {missing}")
    renamed["timestamp"] = pd.to_datetime(renamed["timestamp"], utc=True, errors="coerce")
    renamed = renamed.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    for column in ("open", "high", "low", "close", "volume", "trade_count", "vwap"):
        if column in renamed.columns:
            renamed[column] = pd.to_numeric(renamed[column], errors="coerce")
    return renamed.reset_index(drop=True)


def load_hourly_symbol_history(symbol: str, hourly_root: Path | str) -> pd.DataFrame:
    root = Path(hourly_root)
    for rel in HOURLY_SEARCH_DIRS:
        candidate = root / rel / f"{symbol.upper()}.csv"
        if not candidate.exists():
            continue
        return _normalize_hourly_frame(pd.read_csv(candidate))
    raise FileNotFoundError(f"Hourly history not found for {symbol} under {root}")


def _slice_symbol_session(history: pd.DataFrame, session_date: str | None) -> pd.DataFrame:
    if not session_date:
        return pd.DataFrame()
    session_day = pd.Timestamp(session_date, tz="UTC")
    next_day = session_day + pd.Timedelta(days=1)
    session = history[(history["timestamp"] >= session_day) & (history["timestamp"] < next_day)].copy()
    return session.reset_index(drop=True)


@dataclass(frozen=True)
class IntradayReplaySummary:
    filled_count: int
    trade_count: int
    total_pnl: float
    max_drawdown: float
    day_results: tuple[DaySimulationResult, ...]


@dataclass
class _OrderRuntime:
    order: WideOrder
    session: pd.DataFrame
    status: str = "pending"
    activated_at: pd.Timestamp | None = None
    filled_at: pd.Timestamp | None = None
    exit_price: float | None = None
    pnl: float = 0.0
    return_pct: float = 0.0
    hit_take_profit: bool = False


def _volume_adjusted_fill_buffer_bps(candidate: WideCandidate, planner: WidePlannerConfig) -> float:
    base_bps = float(planner.fill_buffer_bps)
    dollar_vol = candidate.dollar_vol_20d
    extra_bps = 0.0
    if dollar_vol is None:
        extra_bps = 2.0
    elif dollar_vol < 500_000:
        extra_bps = 8.0
    elif dollar_vol < 1_000_000:
        extra_bps = 6.0
    elif dollar_vol < 5_000_000:
        extra_bps = 4.0
    elif dollar_vol < 20_000_000:
        extra_bps = 2.0
    spread = candidate.spread_bps_estimate
    if spread is not None and spread >= 100.0:
        extra_bps = max(extra_bps, 4.0)
    return min(13.0, max(base_bps, base_bps + extra_bps))


def _buy_distance_to_entry_pct(candidate: WideCandidate, bar: pd.Series) -> float:
    low = float(bar["low"])
    if candidate.entry_price <= 0:
        return 1.0
    if low <= candidate.entry_price:
        return 0.0
    return max((low - candidate.entry_price) / candidate.entry_price, 0.0)


def _watch_triggered(candidate: WideCandidate, bar: pd.Series, planner: WidePlannerConfig) -> bool:
    return _buy_distance_to_entry_pct(candidate, bar) <= float(planner.watch_activation_pct)


def _watch_protected(candidate: WideCandidate, bar: pd.Series, planner: WidePlannerConfig) -> bool:
    return _buy_distance_to_entry_pct(candidate, bar) <= float(planner.steal_protection_pct)


def simulate_intraday_day(
    candidates: Sequence[WideCandidate],
    *,
    account_equity: float,
    hourly_by_symbol: Mapping[str, pd.DataFrame],
    config: WidePlannerConfig | None = None,
    day_index: int = 0,
    logger: WideRunLogger | None = None,
) -> DaySimulationResult:
    planner = config or WidePlannerConfig()
    orders = build_wide_plan(candidates, account_equity=account_equity, config=planner)
    max_gross_notional = account_equity * planner.max_leverage
    fee_rate = planner.fee_bps / 10_000.0
    realized_pnl = 0.0

    runtimes: dict[str, _OrderRuntime] = {}
    timeline: list[pd.Timestamp] = []
    seen_timestamps: set[pd.Timestamp] = set()
    for order in orders:
        candidate = order.candidate
        session = _slice_symbol_session(hourly_by_symbol.get(candidate.symbol, pd.DataFrame()), candidate.session_date)
        runtimes[candidate.symbol] = _OrderRuntime(order=order, session=session)
        if session.empty:
            if logger:
                logger.event(f"{candidate.session_date} no hourly session available for planned order", symbol=candidate.symbol)
            continue
        for ts in session["timestamp"]:
            if ts in seen_timestamps:
                continue
            seen_timestamps.add(ts)
            timeline.append(ts)
    timeline.sort()

    for ts in timeline:
        bars_at_ts: dict[str, pd.Series] = {}
        for symbol, runtime in runtimes.items():
            if runtime.session.empty:
                continue
            matches = runtime.session[runtime.session["timestamp"] == ts]
            if not matches.empty:
                bars_at_ts[symbol] = matches.iloc[-1]

        for runtime in runtimes.values():
            if runtime.status != "filled" or runtime.filled_at is None:
                continue
            candidate = runtime.order.candidate
            bar = bars_at_ts.get(candidate.symbol)
            if bar is None or ts <= runtime.filled_at:
                continue
            fill_buffer_bps = _volume_adjusted_fill_buffer_bps(candidate, planner)
            fill_buffer_frac = fill_buffer_bps / 10_000.0
            take_profit_trigger = candidate.take_profit_price * (1.0 + fill_buffer_frac)
            if float(bar["high"]) < take_profit_trigger:
                continue
            runtime.status = "closed"
            runtime.hit_take_profit = True
            runtime.exit_price = candidate.take_profit_price
            runtime.return_pct = ((candidate.take_profit_price - candidate.entry_price) / candidate.entry_price) - (2.0 * fee_rate)
            runtime.pnl = runtime.order.reserved_notional * runtime.return_pct
            realized_pnl += runtime.pnl
            if logger:
                logger.event(
                    (
                        f"{candidate.session_date} take-profit hit on {ts} exit={runtime.exit_price:.4f} "
                        f"pnl={runtime.pnl:+.2f}"
                    ),
                    symbol=candidate.symbol,
                )

        activations: list[tuple[float, float, float, int, _OrderRuntime]] = []
        for runtime in runtimes.values():
            if runtime.status != "pending":
                continue
            candidate = runtime.order.candidate
            bar = bars_at_ts.get(candidate.symbol)
            if bar is None or not _watch_triggered(candidate, bar, planner):
                continue
            distance_pct = _buy_distance_to_entry_pct(candidate, bar)
            activations.append(
                (
                    distance_pct,
                    -candidate.forecasted_pnl,
                    -candidate.score,
                    runtime.order.rank,
                    runtime,
                )
            )
        activations.sort()

        for _, _, _, _, runtime in activations:
            candidate = runtime.order.candidate

            def _reserved_notional() -> float:
                return sum(
                    item.order.reserved_notional
                    for item in runtimes.values()
                    if item.status in {"watching", "filled"}
                )

            remaining_capacity = max_gross_notional - _reserved_notional()
            if remaining_capacity >= runtime.order.reserved_notional:
                runtime.status = "watching"
                runtime.activated_at = ts
                if logger:
                    logger.event(
                        (
                            f"{candidate.session_date} watch activated on {ts} "
                            f"entry={candidate.entry_price:.4f} watch_window={planner.watch_activation_pct * 100:.2f}%"
                        ),
                        symbol=candidate.symbol,
                    )
                continue

            steal_candidates: list[tuple[float, float, float, int, _OrderRuntime]] = []
            for incumbent in runtimes.values():
                if incumbent.status != "watching":
                    continue
                incumbent_bar = bars_at_ts.get(incumbent.order.candidate.symbol)
                if incumbent_bar is not None and _watch_protected(incumbent.order.candidate, incumbent_bar, planner):
                    continue
                incumbent_distance = 1.0 if incumbent_bar is None else _buy_distance_to_entry_pct(incumbent.order.candidate, incumbent_bar)
                steal_candidates.append(
                    (
                        incumbent.order.candidate.forecasted_pnl,
                        incumbent.order.candidate.score,
                        -incumbent_distance,
                        -incumbent.order.rank,
                        incumbent,
                    )
                )
            if not steal_candidates:
                continue
            steal_candidates.sort()
            victim = steal_candidates[0][-1]
            if candidate.forecasted_pnl <= victim.order.candidate.forecasted_pnl:
                continue

            victim.status = "pending"
            victim.activated_at = None
            runtime.status = "watching"
            runtime.activated_at = ts
            if logger:
                logger.event(
                    (
                        f"{candidate.session_date} work steal on {ts}: {candidate.symbol} "
                        f"({candidate.forecasted_pnl:+.4f}) replaced {victim.order.candidate.symbol} "
                        f"({victim.order.candidate.forecasted_pnl:+.4f})"
                    ),
                    symbol=candidate.symbol,
                )
                logger.event(
                    f"{candidate.session_date} watch canceled by work steal from {candidate.symbol}",
                    symbol=victim.order.candidate.symbol,
                )

        fill_attempts: list[tuple[float, float, float, int, _OrderRuntime]] = []
        for runtime in runtimes.values():
            if runtime.status != "watching":
                continue
            candidate = runtime.order.candidate
            bar = bars_at_ts.get(candidate.symbol)
            if bar is None:
                continue
            fill_buffer_bps = _volume_adjusted_fill_buffer_bps(candidate, planner)
            fill_buffer_frac = fill_buffer_bps / 10_000.0
            entry_trigger = candidate.entry_price * (1.0 - fill_buffer_frac)
            if not (float(bar["low"]) <= entry_trigger <= float(bar["high"])):
                continue
            fill_attempts.append(
                (
                    _buy_distance_to_entry_pct(candidate, bar),
                    -candidate.forecasted_pnl,
                    -candidate.score,
                    runtime.order.rank,
                    runtime,
                )
            )
        fill_attempts.sort()
        for _, _, _, _, runtime in fill_attempts:
            candidate = runtime.order.candidate
            if runtime.status != "watching":
                continue
            runtime.status = "filled"
            runtime.filled_at = ts
            fill_buffer_bps = _volume_adjusted_fill_buffer_bps(candidate, planner)
            if logger:
                logger.event(
                    (
                        f"{candidate.session_date} filled at {candidate.entry_price:.4f} on {ts} "
                        f"penetration={fill_buffer_bps:.1f}bp"
                    ),
                    symbol=candidate.symbol,
                )

    fills: list[FillResult] = []
    for runtime in runtimes.values():
        candidate = runtime.order.candidate
        if runtime.status == "filled":
            runtime.status = "closed"
            runtime.exit_price = float(runtime.session.iloc[-1]["close"])
            runtime.return_pct = ((runtime.exit_price - candidate.entry_price) / candidate.entry_price) - (2.0 * fee_rate)
            runtime.pnl = runtime.order.reserved_notional * runtime.return_pct
            realized_pnl += runtime.pnl
            if logger:
                logger.event(
                    (
                        f"{candidate.session_date} session close exit={runtime.exit_price:.4f} "
                        f"pnl={runtime.pnl:+.2f}"
                    ),
                    symbol=candidate.symbol,
                )

        if runtime.exit_price is not None:
            fills.append(
                FillResult(
                    order=runtime.order,
                    filled=True,
                    notional=runtime.order.reserved_notional,
                    entry_price=candidate.entry_price,
                    exit_price=runtime.exit_price,
                    pnl=runtime.pnl,
                    return_pct=runtime.return_pct,
                    hit_take_profit=runtime.hit_take_profit,
                    work_steal_priority=candidate.entry_gap_pct,
                )
            )
            continue

        if not runtime.session.empty and logger:
            fill_buffer_bps = _volume_adjusted_fill_buffer_bps(candidate, planner)
            logger.event(
                (
                    f"{candidate.session_date} plan stayed pending all day "
                    f"entry={candidate.entry_price:.4f} watch_window={planner.watch_activation_pct * 100:.2f}% "
                    f"penetration={fill_buffer_bps:.1f}bp"
                ),
                symbol=candidate.symbol,
            )
        fills.append(
            FillResult(
                order=runtime.order,
                filled=False,
                notional=0.0,
                entry_price=None,
                exit_price=None,
                pnl=0.0,
                return_pct=0.0,
                hit_take_profit=False,
                work_steal_priority=candidate.entry_gap_pct,
            )
        )

    return DaySimulationResult(
        day_index=day_index,
        start_equity=account_equity,
        end_equity=account_equity + realized_pnl,
        realized_pnl=realized_pnl,
        max_gross_notional=max_gross_notional,
        top_symbols=tuple(order.candidate.symbol for order in orders),
        fills=tuple(fills),
    )


def load_hourly_histories(symbols: Iterable[str], hourly_root: Path | str) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for symbol in sorted({symbol.upper() for symbol in symbols}):
        try:
            out[symbol] = load_hourly_symbol_history(symbol, hourly_root)
        except FileNotFoundError:
            continue
    return out
