from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WideCandidate:
    symbol: str
    strategy: str
    forecasted_pnl: float
    avg_return: float
    last_close: float
    entry_price: float
    take_profit_price: float
    predicted_high: float
    predicted_low: float
    realized_close: float
    realized_high: float
    realized_low: float
    score: float
    day_index: int
    side: str = "buy"
    session_date: str | None = None
    dollar_vol_20d: float | None = None
    spread_bps_estimate: float | None = None
    allocation_fraction_of_equity: float | None = None
    rl_prior_score: float | None = None

    @property
    def entry_gap_pct(self) -> float:
        if self.last_close <= 0:
            return 0.0
        return max((self.last_close - self.entry_price) / self.last_close, 0.0)

    @property
    def expected_return_pct(self) -> float:
        if self.entry_price <= 0:
            return 0.0
        return (self.take_profit_price - self.entry_price) / self.entry_price


@dataclass(frozen=True)
class WideOrder:
    rank: int
    candidate: WideCandidate
    reserved_notional: float
    reserved_fraction_of_equity: float


@dataclass(frozen=True)
class FillResult:
    order: WideOrder
    filled: bool
    notional: float
    entry_price: float | None
    exit_price: float | None
    pnl: float
    return_pct: float
    hit_take_profit: bool
    work_steal_priority: float


@dataclass(frozen=True)
class DaySimulationResult:
    day_index: int
    start_equity: float
    end_equity: float
    realized_pnl: float
    max_gross_notional: float
    top_symbols: tuple[str, ...]
    fills: tuple[FillResult, ...]
