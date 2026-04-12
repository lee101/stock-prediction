from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping, Sequence

import math

import pandas as pd

from .types import WideCandidate, WideOrder


def _coerce_float(value: object, default: float | None = None) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def _first_finite(row: Mapping[str, object], keys: Sequence[str]) -> float | None:
    for key in keys:
        if key not in row:
            continue
        value = _coerce_float(row.get(key))
        if value is not None:
            return value
    return None


def _session_date_from_row(row: Mapping[str, object]) -> str | None:
    raw = row.get("timestamp")
    if raw is None:
        return None
    try:
        ts = pd.to_datetime(raw, utc=True, errors="coerce")
    except Exception:
        return None
    if pd.isna(ts):
        return None
    return ts.floor("D").date().isoformat()


def _refit_price_with_multiplier(
    *,
    last_close: float,
    raw_price: float | None,
    multiplier: float | None,
) -> float | None:
    if last_close <= 0:
        return raw_price
    derived_price = None if multiplier is None else last_close * (1.0 + multiplier)
    if raw_price is None:
        return derived_price
    if derived_price is None:
        return raw_price
    return 0.5 * raw_price + 0.5 * derived_price


@dataclass(frozen=True)
class StrategySpec:
    name: str
    forecast_keys: tuple[str, ...]
    avg_return_keys: tuple[str, ...]
    entry_keys: tuple[str, ...]
    take_profit_keys: tuple[str, ...]
    entry_multiplier_keys: tuple[str, ...] = ()
    take_profit_multiplier_keys: tuple[str, ...] = ()


STRATEGY_SPECS: tuple[StrategySpec, ...] = (
    StrategySpec(
        name="maxdiff",
        forecast_keys=("maxdiffprofit_profit", "maxdiff_return"),
        avg_return_keys=("maxdiff_avg_daily_return", "maxdiff_return"),
        entry_keys=("maxdiffprofit_low_price", "predicted_low"),
        take_profit_keys=("maxdiffprofit_high_price", "predicted_high"),
        entry_multiplier_keys=("maxdiffprofit_profit_low_multiplier",),
        take_profit_multiplier_keys=("maxdiffprofit_profit_high_multiplier",),
    ),
    StrategySpec(
        name="pctdiff",
        forecast_keys=("pctdiff_profit", "pctdiff_return"),
        avg_return_keys=("pctdiff_avg_daily_return", "pctdiff_return"),
        entry_keys=("pctdiff_entry_low_price", "predicted_low"),
        take_profit_keys=("pctdiff_takeprofit_high_price", "predicted_high"),
    ),
    StrategySpec(
        name="takeprofit",
        forecast_keys=("entry_takeprofit_profit", "entry_takeprofit_return"),
        avg_return_keys=("entry_takeprofit_avg_daily_return", "entry_takeprofit_return"),
        entry_keys=("entry_takeprofit_low_price", "takeprofit_low_price", "predicted_low"),
        take_profit_keys=("entry_takeprofit_high_price", "takeprofit_high_price", "predicted_high"),
    ),
    StrategySpec(
        name="highlow",
        forecast_keys=("highlow_profit", "highlow_return"),
        avg_return_keys=("highlow_avg_daily_return", "highlow_return"),
        entry_keys=("predicted_low",),
        take_profit_keys=("predicted_high",),
    ),
    StrategySpec(
        name="all_signals",
        forecast_keys=("all_signals_strategy_profit", "all_signals_strategy_return"),
        avg_return_keys=("all_signals_strategy_avg_daily_return", "all_signals_strategy_return"),
        entry_keys=("predicted_low",),
        take_profit_keys=("predicted_high",),
    ),
    StrategySpec(
        name="simple",
        forecast_keys=("simple_strategy_profit", "simple_strategy_return"),
        avg_return_keys=("simple_strategy_avg_daily_return", "simple_strategy_return"),
        entry_keys=("predicted_low",),
        take_profit_keys=("predicted_high",),
    ),
)


@dataclass(frozen=True)
class WidePlannerConfig:
    top_k: int = 6
    pair_notional_fraction: float = 0.50
    max_pair_notional_fraction: float = 0.50
    max_leverage: float = 2.0
    fee_bps: float = 10.0
    fill_buffer_bps: float = 5.0
    watch_activation_pct: float = 0.005
    steal_protection_pct: float = 0.004
    allow_short: bool = False


def validate_long_price_levels(
    *,
    symbol: str,
    entry_price: float,
    take_profit_price: float,
) -> None:
    normalized_symbol = str(symbol).strip().upper() or "UNKNOWN"
    if not math.isfinite(entry_price) or entry_price <= 0.0:
        raise ValueError(f"{normalized_symbol}: entry_price must be positive and finite")
    if not math.isfinite(take_profit_price) or take_profit_price <= 0.0:
        raise ValueError(f"{normalized_symbol}: take_profit_price must be positive and finite")
    if take_profit_price <= entry_price:
        raise ValueError(
            f"{normalized_symbol}: invalid long price plan; take_profit_price "
            f"({take_profit_price:.6f}) must be above entry_price ({entry_price:.6f})"
        )


def candidate_from_row(
    symbol: str,
    row: Mapping[str, object],
    *,
    day_index: int,
    allow_short: bool = False,
    require_realized_ohlc: bool = False,
) -> WideCandidate | None:
    del allow_short  # The initial wide planner is intentionally long-only.

    last_close = _first_finite(row, ("close", "last_close", "Close"))
    predicted_high = _first_finite(row, ("predicted_high", "maxdiffprofit_high_price", "high"))
    predicted_low = _first_finite(row, ("predicted_low", "maxdiffprofit_low_price", "low"))
    realized_close = _first_finite(row, ("close", "last_close", "Close"))
    realized_high = _first_finite(row, ("high", "High"))
    realized_low = _first_finite(row, ("low", "Low"))
    if not require_realized_ohlc:
        if realized_high is None:
            realized_high = predicted_high
        if realized_low is None:
            realized_low = predicted_low
    if (
        last_close is None
        or predicted_high is None
        or predicted_low is None
        or realized_close is None
        or realized_high is None
        or realized_low is None
    ):
        return None

    best: WideCandidate | None = None
    for spec in STRATEGY_SPECS:
        forecasted_pnl = _first_finite(row, spec.forecast_keys)
        avg_return = _first_finite(row, spec.avg_return_keys)
        entry_price = _refit_price_with_multiplier(
            last_close=last_close,
            raw_price=_first_finite(row, spec.entry_keys),
            multiplier=_first_finite(row, spec.entry_multiplier_keys),
        )
        take_profit_price = _refit_price_with_multiplier(
            last_close=last_close,
            raw_price=_first_finite(row, spec.take_profit_keys),
            multiplier=_first_finite(row, spec.take_profit_multiplier_keys),
        )
        if (
            forecasted_pnl is None
            or avg_return is None
            or entry_price is None
            or take_profit_price is None
            or forecasted_pnl <= 0.0
            or avg_return <= 0.0
            or entry_price <= 0.0
            or take_profit_price <= entry_price
            or entry_price >= last_close
        ):
            continue
        edge = (take_profit_price - entry_price) / entry_price
        score = (forecasted_pnl * 1.0) + (avg_return * 0.35) + (edge * 0.15)
        candidate = WideCandidate(
            symbol=symbol.upper(),
            strategy=spec.name,
            forecasted_pnl=forecasted_pnl,
            avg_return=avg_return,
            last_close=last_close,
            entry_price=entry_price,
            take_profit_price=take_profit_price,
            predicted_high=predicted_high,
            predicted_low=predicted_low,
            realized_close=realized_close,
            realized_high=realized_high,
            realized_low=realized_low,
            score=score,
            day_index=day_index,
            session_date=_session_date_from_row(row),
            dollar_vol_20d=_first_finite(row, ("dollar_vol_20d",)),
            spread_bps_estimate=_first_finite(row, ("spread_bps_estimate",)),
        )
        if best is None or (candidate.score, candidate.forecasted_pnl, candidate.avg_return) > (
            best.score,
            best.forecasted_pnl,
            best.avg_return,
        ):
            best = candidate
    return best


def build_daily_candidates(
    backtests_by_symbol: Mapping[str, pd.DataFrame],
    *,
    day_index: int = 0,
    allow_short: bool = False,
    require_realized_ohlc: bool = False,
) -> list[WideCandidate]:
    candidates: list[WideCandidate] = []
    for symbol, frame in backtests_by_symbol.items():
        if frame is None or frame.empty or day_index >= len(frame):
            continue
        row = frame.iloc[int(day_index)].to_dict()
        candidate = candidate_from_row(
            symbol,
            row,
            day_index=day_index,
            allow_short=allow_short,
            require_realized_ohlc=require_realized_ohlc,
        )
        if candidate is not None:
            candidates.append(candidate)
    return sorted(
        candidates,
        key=lambda item: (item.score, item.forecasted_pnl, item.avg_return),
        reverse=True,
    )


def build_wide_plan(
    candidates: Sequence[WideCandidate],
    *,
    account_equity: float,
    config: WidePlannerConfig | None = None,
) -> list[WideOrder]:
    planner = config or WidePlannerConfig()
    if account_equity <= 0:
        return []
    max_fraction = max(float(planner.max_pair_notional_fraction), 0.0)
    default_fraction = min(max(float(planner.pair_notional_fraction), 0.0), max_fraction)
    if default_fraction <= 0 and max_fraction <= 0:
        return []
    orders: list[WideOrder] = []
    for rank, candidate in enumerate(candidates[: max(int(planner.top_k), 0)], start=1):
        validate_long_price_levels(
            symbol=candidate.symbol,
            entry_price=candidate.entry_price,
            take_profit_price=candidate.take_profit_price,
        )
        candidate_fraction = candidate.allocation_fraction_of_equity
        if candidate_fraction is None:
            reserved_fraction = default_fraction
        else:
            reserved_fraction = min(max(float(candidate_fraction), 0.0), max_fraction)
        reserved_notional = account_equity * reserved_fraction
        if reserved_notional <= 0.0:
            continue
        orders.append(
            WideOrder(
                rank=rank,
                candidate=candidate,
                reserved_notional=reserved_notional,
                reserved_fraction_of_equity=reserved_fraction,
            )
        )
    return orders


def render_plan_text(
    orders: Sequence[WideOrder],
    *,
    account_equity: float,
    config: WidePlannerConfig | None = None,
    as_of: datetime | None = None,
) -> str:
    planner = config or WidePlannerConfig()
    ts = as_of or datetime.now(timezone.utc)
    if not orders:
        return (
            f"Wide plan for {ts.date().isoformat()}\n"
            "No valid long candidates met the positive-forecast and positive-return gates."
        )

    max_gross_notional = account_equity * planner.max_leverage
    max_fill_count = max(int(max_gross_notional // max(orders[0].reserved_notional, 1e-9)), 0)
    lines = [
        f"Wide plan for {ts.date().isoformat()}",
        (
            f"Account=${account_equity:,.2f} | pair_size={planner.pair_notional_fraction:.2f}x equity "
            f"| max_pair_size={planner.max_pair_notional_fraction:.2f}x equity "
            f"| max_leverage={planner.max_leverage:.2f}x | top_k={planner.top_k}"
        ),
        (
            f"Watch activation={planner.watch_activation_pct * 100:.2f}% "
            f"| steal protection={planner.steal_protection_pct * 100:.2f}% "
            f"| fill buffer={planner.fill_buffer_bps:.1f}bp"
        ),
        f"At most {max_fill_count} fills can coexist before the 2x cap is reached.",
    ]
    for order in orders:
        candidate = order.candidate
        lines.append(
            (
                f"{order.rank}. {candidate.symbol} {candidate.strategy} "
                f"forecast={candidate.forecasted_pnl:+.4f} avg={candidate.avg_return:+.4f} "
                f"entry={candidate.entry_price:.4f} tp={candidate.take_profit_price:.4f} "
                f"gap={candidate.entry_gap_pct * 100:.2f}% size={order.reserved_fraction_of_equity * 100:.1f}% "
                f"reserve=${order.reserved_notional:,.2f}"
            )
        )
    lines.append(
        "Deleverage companion: python scripts/deleverage_account_day_end.py --target 1.94 --once"
    )
    return "\n".join(lines)
