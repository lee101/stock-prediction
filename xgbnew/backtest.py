"""Backtest simulation for XGBoost daily open-to-close strategy.

Cost model (per trade):
    total_cost = spread_bps + 2 * commission_bps   (round-trip, in bps)

With leverage L (default 1.0):
    gross_return_leveraged  = L * actual_oc_return
    total_cost_leveraged    = L * (spread_bps + 2*commission_bps) / 10_000
    margin_cost             ≈ (L - 1) * annual_rate / 252  (negligible intraday)
    net_return              = gross_return_leveraged - total_cost_leveraged - margin_cost

Selection logic:
    Each day, score every stock using the XGBStockModel + optional Chronos2 blend.
    Pick the top-N by score.  Among ties, prefer higher Chronos2 oc_return.
    Only pick stocks that actually have an open/close recorded for that day.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from src.fees import get_fee_for_symbol
from .features import cross_sectional_regime_keep_by_date
from .model import XGBStockModel, combined_scores

logger = logging.getLogger(__name__)

ANNUAL_MARGIN_RATE = 0.0625   # 6.25% per year (Alpaca rate)
TRADING_DAYS_PER_YEAR = 252


@dataclass
class BacktestConfig:
    top_n: int = 2
    # Aggressive packing floor. When >0, each day fills at least this many
    # valid pick slots from the best-ranked names even if their score is below
    # ``min_score``. This is bounded by ``top_n`` and only applies after the
    # normal live-replicable universe filters, so it tests "stay invested" risk
    # without bypassing liquidity/spread/vol realism.
    min_picks: int = 0
    initial_cash: float = 10_000.0
    commission_bps: float = 0.0         # per side legacy override; prefer fee_rate
    leverage: float = 1.0               # 1.0 = no leverage, max 2.0
    xgb_weight: float = 0.5             # blend weight for XGB vs Chronos2
    min_score: float = 0.0              # min combined score to trade
    min_dollar_vol: float = 5e6         # skip illiquid stocks (min avg daily $ vol)
    min_vol_20d: float = 0.0            # 0 disables; else require annualised
                                        # 20-day realised vol ≥ this (so we
                                        # skip the dead-zone / bot-vol quartile
                                        # that LOBO flagged as a net drag).
    max_vol_20d: float = 0.0            # 0 disables; else require annualised
                                        # 20-day realised vol ≤ this (so we
                                        # skip crash-sensitive high-vol names
                                        # in stressed regimes). Band-pass with
                                        # min_vol_20d when both set.
    max_spread_bps: float = 30.0        # skip wide-spread stocks (max volume-based cost)
    # Cross-sectional (per-day) momentum filters — inference-only masks
    # based on a pick's rank within the day's pool. Default values
    # disable both: max_ret_20d_rank_pct=1.0 keeps everything up to the
    # 100th percentile; min_ret_5d_rank_pct=0.0 keeps everything from
    # the 0th percentile up. Motivated by project_xgb_oos_regime_inversion
    # diagnostic — on the 2025-07→2026-04 OOS the top-5 picks in
    # Q4-by-ret_20d (hot names) underperform by -27bps/day, and picks in
    # Q1-by-ret_5d (weak recent) underperform by -36bps/day; dropping
    # the bottom quartile by ret_5d alone flips mean target_oc from
    # -0.18%/day to +0.12%/day on top-1/day sim. Experimental — carries
    # overfit risk if the regime flips back.
    max_ret_20d_rank_pct: float = 1.0   # drop picks with ret_20d at strictly higher
                                        # per-day percentile than this. 1.0 disables;
                                        # 0.75 drops the "hot" top-25% on each day.
    min_ret_5d_rank_pct: float = 0.0    # drop picks with ret_5d at strictly lower
                                        # per-day percentile than this. 0.0 disables;
                                        # 0.25 drops the "weak recent" bottom-25%.
    chronos_col: str = "chronos_oc_return"
    fee_rate: float | None = None       # per-side fee fraction; defaults by symbol
    fill_buffer_bps: float = 5.0        # adverse entry/exit fill buffer around bar
    # Opportunistic work-stealing entry mode. When enabled, the model ranks a
    # larger daily watchlist, posts explicit buy limits below the open, and
    # only enters symbols whose observed low penetrates that limit by the
    # configured fill buffer. This approximates a live process that keeps
    # watching planned near-misses every few minutes without pretending that a
    # mere touch always fills. 0 disables and preserves open-to-close entries.
    opportunistic_watch_n: int = 0
    opportunistic_entry_discount_bps: float = 0.0
    regime_gate_window: int = 0         # 0 disables; 20/50/200 = SPY MA lookback
    vol_target_ann: float = 0.0         # 0 disables; else scale daily allocation by
                                        # min(1, vol_target_ann / realised_20d_ann_vol)
    # Per-PICK (cross-sectional) inverse-vol sizing. The SPY-based
    # vol_target_ann above responds to market-wide regime; this knob
    # responds to *this pick's* realised vol. The 2025-09+ tariff crash
    # is a cross-sectional shock (growth/high-beta names hammered while
    # SPY index vol stays moderate), so per-pick inv-vol is the lever
    # that can actually see it. Formula for each picked row:
    #   eff_lev = leverage * clip(
    #       inv_vol_target_ann / max(pick_vol_20d, inv_vol_floor),
    #       1/inv_vol_cap, inv_vol_cap)
    # 0 disables (pick_scale ≡ 1.0 — identity with pre-inv-vol sim).
    inv_vol_target_ann: float = 0.0
    inv_vol_floor: float = 0.05         # lower bound on pick_vol denom, prevents
                                        # blowup for ultra-low-vol (or missing) names
    inv_vol_cap: float = 3.0            # cap on the leverage multiplier (up AND down
                                        # symmetric). 3.0 ⇒ scale ∈ [1/3, 3].
    # Cross-sectional regime gate. On each day we compute the IQR of
    # ret_5d ACROSS the full pre-filter universe. Wide dispersion days
    # (tariff-crash style cross-sectional dislocation) are where the
    # ensemble's rank-order flips sign on fresh 2025-07→2026-04 true-OOS.
    # Gating OUT those days (trading only when IQR is below the threshold)
    # turned top-1/day cumulative from −37% to +43% on the fresh 5-seed
    # oos2025h1 ensemble. Leak-free — ret_5d is lag-1 (close[t-1]/close[t-6]),
    # so IQR computed on day T is known at the 9:30 am open.
    # 0.0 disables (legacy identity). Suggested: 0.042 from diagnostic.
    regime_cs_iqr_max: float = 0.0
    # Optional companion gate on cross-sectional skew of ret_5d. Positive
    # skew (few winners dominate) is the momentum-friendly regime; left-skew
    # (few catastrophic losers) flips the sign. 0.0 disables so we can
    # sweep iqr alone, both, or skew alone. NaN days keep the pre-filter
    # behaviour (no gate).
    regime_cs_skew_min: float = -1e9    # effectively disabled default
    allocation_mode: str = "equal"      # equal | softmax | score_norm — how we
                                        # weight the top_n picks within a day
    allocation_temp: float = 1.0        # softmax temperature (lower = more concentrated)
    hold_through: bool = False          # if tomorrow's picks intersect today's,
                                        # hold the overlap (skip sell-close+buy-open
                                        # round-trip). Saves 2x(fee+buffer) per held
                                        # day and captures overnight drift via the
                                        # close-to-close return on the continuation.
    # Missed-order Monte Carlo: each new pick has ``skip_prob`` chance of
    # being dropped (simulates Alpaca rejection / spread-too-wide /
    # stale-price). Hold-through continuations are NOT skipped — no new
    # order fires — so this tests order-entry risk, not position risk.
    # Seed is drawn once per simulate() call so sweeps are reproducible.
    # 0.0 disables (identity with pre-MC sim).
    skip_prob: float = 0.0
    skip_seed: int = 0
    # No-picks fallback. When the day's pick pool is empty (because no
    # symbol's score >= min_score, or every pick was filtered out, or
    # missed-order MC dropped them all), if ``no_picks_fallback_symbol``
    # is set we look the symbol up in the UNFILTERED test_df for that day
    # and buy it at leverage ``config.leverage * no_picks_fallback_alloc_scale``
    # (so the caller can run a small "keep market exposure" position on
    # low-conviction days). Empty string disables; recommended values are
    # "SPY" (broad) or "QQQ" (higher drift, higher DD). Fees and fill
    # buffer identical to a normal churn-day trade. Does NOT participate
    # in hold_through continuations — a fallback day always flattens the
    # hold_through state (so a real pick returning tomorrow triggers a
    # fresh buy, not a claim of "held through the SPY day").
    no_picks_fallback_symbol: str = ""
    no_picks_fallback_alloc_scale: float = 0.5
    # Conviction-scaled allocation. When enabled, each day's allocation
    # is scaled by clip((top_score - alloc_low) / (alloc_high - alloc_low), 0, 1).
    # Below alloc_low → 0% allocation (equivalent to "min_score gate at
    # alloc_low"); above alloc_high → full allocation; in between → linear.
    # Orthogonal to min_score: if min_score=0.0 the gate opens fully and
    # conviction scaling is the sole sizing lever.
    # 0 alloc days that would otherwise have a pick trigger the no-picks
    # fallback if configured.
    conviction_scaled_alloc: bool = False
    conviction_alloc_low: float = 0.55
    conviction_alloc_high: float = 0.85
    # Optional simulator circuit breaker. When positive, simulate() stops
    # immediately after the first traded day whose close-to-close equity
    # drawdown from the running peak reaches this percentage. This is for
    # large grid pruning and production-risk diagnostics; 0 preserves the
    # historical full-window simulation.
    stop_on_drawdown_pct: float = 0.0
    # Optional intraday circuit breaker. When positive, simulate() stops after
    # the first traded day whose OHLC-based portfolio adverse excursion reaches
    # this percentage. This catches high-leverage cells that recover by close
    # but would have been unacceptable live risk while the position was open.
    stop_on_intraday_drawdown_pct: float = 0.0
    # Optional Reg-T overnight gross-leverage cap. ``None`` (default)
    # preserves legacy behaviour. When set (e.g. 2.0), the per-pick effective
    # leverage used to compute returns is clipped to
    # ``min(eff_lev, overnight_max_gross_leverage)``. In daily-bar mode each
    # bar IS the overnight, so this matches the production
    # ``xgbnew/live_trader._eod_deleverage_tick`` semantic — positions can be
    # sized up to ``leverage`` intraday but must collapse back to the cap
    # before market close. Use this when scoring candidates that would be
    # deployed against the prod EOD deleverage tick.
    overnight_max_gross_leverage: float | None = None


@dataclass
class DayTrade:
    symbol: str
    score: float
    actual_open: float
    actual_close: float
    entry_fill_price: float
    exit_fill_price: float
    spread_bps: float
    commission_bps: float
    fee_rate: float
    fill_buffer_bps: float
    leverage: float
    gross_return_pct: float
    net_return_pct: float
    # Intraday unrealized excursions vs entry price, at leverage.
    # worst_dd:  max adverse excursion during the bar   (≥ 0 always — absolute)
    # best_runup: max favourable excursion during the bar (≥ 0 always)
    # Zero when actual_high / actual_low are missing from the row.
    intraday_worst_dd_pct:  float = 0.0
    intraday_best_runup_pct: float = 0.0


@dataclass
class DayResult:
    day: date
    equity_start: float
    equity_end: float
    daily_return_pct: float
    trades: list[DayTrade] = field(default_factory=list)
    n_candidates: int = 0
    # Portfolio-weighted intraday excursions (long-side, pct of equity at
    # day open, at leverage). Proxy for "how deep did we dip before close".
    # Computed via same weights as the realized daily return.
    intraday_worst_dd_pct:   float = 0.0
    intraday_best_runup_pct: float = 0.0


@dataclass
class BacktestResult:
    config: BacktestConfig
    day_results: list[DayResult]
    initial_cash: float
    final_equity: float
    total_return_pct: float
    monthly_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    total_trades: int
    avg_spread_bps: float
    avg_fee_bps: float
    directional_accuracy_pct: float  # % of picks that had gross_return > 0
    # Intraday unrealized excursions, aggregated across all traded days.
    # worst_intraday_dd_pct   = max over days of that day's portfolio DD
    # avg_intraday_dd_pct     = mean
    # worst_intraday_runup_pct = symmetric upside metric
    # 0.0 when high/low data unavailable.
    worst_intraday_dd_pct:   float = 0.0
    avg_intraday_dd_pct:     float = 0.0
    worst_intraday_runup_pct: float = 0.0
    # Equity-curve pain metrics — computed from the same running-max
    # decomposition as ``max_drawdown_pct`` but summarising duration
    # and integrated depth, not just the worst point.
    # time_under_water_pct = % of equity-curve samples below the prior peak
    # ulcer_index          = RMS of drawdown-depth % across the curve
    time_under_water_pct: float = 0.0
    ulcer_index:          float = 0.0
    stopped_early: bool = False
    stop_reason: str = ""


def _day_margin_cost(leverage: float) -> float:
    """Intraday margin cost fraction (open-to-close only)."""
    if leverage <= 1.0:
        return 0.0
    return (leverage - 1.0) * ANNUAL_MARGIN_RATE / TRADING_DAYS_PER_YEAR


def _inv_vol_pick_scale(
    pick_vol_20d: float,
    *,
    target_ann: float,
    floor: float,
    cap: float,
) -> float:
    """Per-pick leverage multiplier from 1/vol targeting. 1.0 when disabled.

    target_ann <= 0 ⇒ returns 1.0 (identity, no scaling).
    Missing / non-finite / non-positive vol ⇒ returns 1.0 (no surprise scale-up).
    Otherwise clip(target / max(vol, floor), 1/cap, cap).
    """
    if target_ann <= 0.0:
        return 1.0
    if not np.isfinite(pick_vol_20d) or pick_vol_20d <= 0.0:
        return 1.0
    denom = max(float(pick_vol_20d), float(floor))
    raw = float(target_ann) / denom
    lo = 1.0 / float(cap) if cap > 0 else 0.0
    hi = float(cap) if cap > 0 else raw
    return float(min(max(raw, lo), hi))


# Cap for annualised Sortino so the degenerate "almost no losses" case
# doesn't produce astronomical numbers downstream (e.g. grid sweep tables
# showing Sortino = 6.3e8 when 0-1 windows have losses).
_SORTINO_CAP = 100.0


def _sortino_semi(rets: np.ndarray, mean_r: float, ann_factor: float) -> float:
    """Annualised Sortino using the full-length semi-deviation.

    Semi-deviation = sqrt( sum(min(r, 0)^2) / (N-1) ). This is the textbook
    downside-deviation definition (Sortino 1994): squared losses weighted
    by the full sample size, not only by the count of losing periods. This
    stays stable when very few losses occur, instead of the len(down_r)-1
    denominator (which blows up to infinity once len(down_r) ≤ 1).

    The ratio is clipped to ±_SORTINO_CAP so tables and cross-window
    comparisons remain meaningful when downside is vanishingly small.
    """
    n = len(rets)
    if n < 2:
        return 0.0
    down = np.minimum(rets, 0.0)
    semi_var = float(np.sum(down * down) / (n - 1))
    semi_dev = float(np.sqrt(semi_var))
    if semi_dev < 1e-9:
        # No meaningful downside observed over the window — flag by
        # returning +cap for positive-mean runs, 0 otherwise.
        return _SORTINO_CAP if mean_r > 0 else 0.0
    raw = mean_r / semi_dev * ann_factor
    return float(np.clip(raw, -_SORTINO_CAP, _SORTINO_CAP))


def _equity_drawdown_metrics(eq: np.ndarray) -> tuple[float, float, float]:
    """Return (max_dd_frac, time_under_water_pct, ulcer_index).

    ``eq`` includes the initial equity as element 0. The initial point sets
    the first peak, but it is not an evaluated period; TuW and Ulcer therefore
    use drawdowns from eq[1:] only. Including eq[0] dilutes early losses and
    makes a first-day underwater state look artificially less painful.
    """
    if eq.size <= 1:
        return 0.0, 0.0, 0.0
    running_max = np.maximum.accumulate(eq)
    dd_frac = (running_max - eq) / running_max
    dd_path = dd_frac[1:]
    max_dd = float(np.max(dd_frac))
    tuw_pct = float(np.mean(dd_path > 0.0)) * 100.0
    ulcer = float(np.sqrt(np.mean(dd_path * dd_path))) * 100.0
    return max_dd, tuw_pct, ulcer


def _resolve_fee_rate(symbol: str, config: BacktestConfig) -> float:
    if config.fee_rate is not None and np.isfinite(config.fee_rate) and config.fee_rate >= 0.0:
        return float(config.fee_rate)
    return float(get_fee_for_symbol(symbol))


def _fill_prices(open_price: float, close_price: float, *, fill_buffer_bps: float) -> tuple[float, float]:
    buffer_frac = max(float(fill_buffer_bps), 0.0) / 10_000.0
    entry_fill = float(open_price) * (1.0 + buffer_frac)
    exit_fill = float(close_price) * (1.0 - buffer_frac)
    return entry_fill, exit_fill


def _limit_entry_fill_price(
    row: pd.Series,
    *,
    discount_bps: float,
    fill_buffer_bps: float,
) -> float | None:
    """Return a conservative explicit-limit entry fill, or None if unfilled.

    The limit is placed at ``open * (1 - discount_bps)``. To avoid the common
    OHLC look-ahead optimism where a one-tick touch is treated as certain
    execution, the bar low must move through the limit by ``fill_buffer_bps``.
    If it does, the fill price is the posted limit price, not a better low.
    """
    try:
        open_price = float(row.get("actual_open", 0.0) or 0.0)
        low_price = float(row.get("actual_low", 0.0) or 0.0)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(open_price) and np.isfinite(low_price)):
        return None
    if open_price <= 0.0 or low_price <= 0.0:
        return None

    discount = max(float(discount_bps), 0.0) / 10_000.0
    buffer_frac = max(float(fill_buffer_bps), 0.0) / 10_000.0
    limit_price = open_price * (1.0 - discount)
    required_low = limit_price * (1.0 - buffer_frac)
    if low_price <= required_low:
        return float(limit_price)
    return None


def _intraday_excursion_pct(
    row: pd.Series,
    entry_ref: float,
    leverage: float,
) -> tuple[float, float]:
    """Return (worst_dd_pct, best_runup_pct) for one long trade.

    ``entry_ref`` is the price from which we measure excursions — usually
    the entry fill (open-side) or the prior close on a continuation day.
    Uses ``actual_high`` / ``actual_low`` from the row when present; when
    the columns are missing or invalid (<= 0), returns (0.0, 0.0) so
    downstream aggregates are a proxy of "what we observed" and not an
    accidental floor.

    Long-direction only (XGB daily picks are always long). The returned
    values are POSITIVE percentages at the requested leverage:
      worst_dd  = max(0, (entry_ref - bar_low ) / entry_ref * L * 100)
      best_runup = max(0, (bar_high - entry_ref) / entry_ref * L * 100)
    """
    if entry_ref <= 0:
        return (0.0, 0.0)
    hi = float(row.get("actual_high", 0.0) or 0.0)
    lo = float(row.get("actual_low",  0.0) or 0.0)
    if not (np.isfinite(hi) and np.isfinite(lo)) or hi <= 0 or lo <= 0:
        return (0.0, 0.0)
    worst = max(0.0, (entry_ref - lo) / entry_ref) * float(leverage) * 100.0
    runup = max(0.0, (hi - entry_ref) / entry_ref) * float(leverage) * 100.0
    return (float(worst), float(runup))


def _build_regime_flags(
    spy_close_by_date: pd.Series | None,
    all_dates: pd.Index,
    *,
    window: int,
    available_at_open: bool = False,
) -> pd.Series:
    """Return a bool Series indexed by ``all_dates`` — True = gate CLOSED (skip day).

    Gate closed when SPY is strictly below its ``window``-day simple moving
    average. When ``available_at_open`` is true, the flag for date D uses SPY
    history through D-1's close so daily backtests do not look ahead to D's
    close. Dates missing from SPY coverage are treated as open (do not gate) so
    we never silently skip a trading day for data reasons.
    """
    closed = pd.Series(False, index=all_dates)
    if spy_close_by_date is None or window <= 0:
        return closed
    # Defensive: caller may pass a Series with duplicate date labels (e.g.,
    # multi-bar CSV collapsed incorrectly). Keep the last close per date so
    # the downstream reindex never errors out and never silently skips days.
    sp = spy_close_by_date.sort_index()
    if not sp.index.is_unique:
        sp = sp.groupby(level=0).last()
    ma = sp.rolling(window=int(window), min_periods=int(window)).mean()
    ratio = sp / ma
    below = ratio < 1.0
    if available_at_open:
        below = below.shift(1)
    # Align to all_dates — unknown dates stay False (open).
    aligned = below.reindex(all_dates).astype("boolean").fillna(False).astype(bool)
    return aligned


def _build_vol_scale(
    spy_close_by_date: pd.Series | None,
    all_dates: pd.Index,
    *,
    target_ann: float,
    lookback_days: int = 20,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
    available_at_open: bool = False,
) -> pd.Series:
    """Return a scalar Series (in [0, 1]) by date — min(1, target / realised).

    Uses SPY as the regime-vol proxy because the XGB signal is market-wide.
    When ``available_at_open`` is true, the scalar for date D uses SPY history
    through D-1's close so daily backtests do not look ahead to D's close.
    Dates with insufficient SPY history fall through to 1.0 (no scaling).
    """
    scale = pd.Series(1.0, index=all_dates)
    if spy_close_by_date is None or target_ann <= 0:
        return scale
    sp = spy_close_by_date.sort_index().astype(float)
    if not sp.index.is_unique:
        sp = sp.groupby(level=0).last()
    log_ret = np.log(sp / sp.shift(1))
    realised_ann = (
        log_ret.rolling(window=int(lookback_days), min_periods=int(lookback_days)).std()
        * np.sqrt(trading_days_per_year)
    )
    ratio = (float(target_ann) / realised_ann).clip(upper=1.0)
    if available_at_open:
        ratio = ratio.shift(1)
    aligned = ratio.reindex(all_dates).fillna(1.0).astype(float)
    return aligned


def _allocation_weights(
    scores: list[float] | np.ndarray,
    *,
    mode: str = "equal",
    temperature: float = 1.0,
) -> np.ndarray:
    """Return an array of per-pick weights summing to 1.0.

    ``mode`` selects packing policy:

    * ``"equal"``      — uniform 1/K (legacy behaviour)
    * ``"softmax"``    — exp(score / temperature) / sum(...)  (temperature>0)
    * ``"score_norm"`` — score / sum(score) with a non-negativity floor;
                         all-zero / all-negative → falls back to equal weights.

    The helper is deliberately total so callers don't have to branch on K==0
    or NaN scores — it always returns a valid non-negative distribution.
    """
    arr = np.asarray(list(scores), dtype=np.float64)
    n = arr.shape[0]
    if n == 0:
        return arr
    mode = (mode or "equal").lower()
    if mode == "equal":
        return np.full(n, 1.0 / n)
    if mode == "softmax":
        temp = float(temperature) if temperature and temperature > 0 else 1.0
        # Subtract max for numerical stability; NaNs fall through to equal.
        if not np.all(np.isfinite(arr)):
            return np.full(n, 1.0 / n)
        z = (arr - float(np.max(arr))) / temp
        e = np.exp(z)
        s = float(e.sum())
        if not np.isfinite(s) or s <= 0:
            return np.full(n, 1.0 / n)
        return e / s
    if mode == "score_norm":
        pos = np.clip(arr, a_min=0.0, a_max=None)
        s = float(pos.sum())
        if not np.isfinite(s) or s <= 0:
            return np.full(n, 1.0 / n)
        return pos / s
    raise ValueError(
        f"Unknown allocation_mode={mode!r}; expected equal|softmax|score_norm"
    )


def simulate(
    test_df: pd.DataFrame,
    model: XGBStockModel,
    config: BacktestConfig,
    *,
    precomputed_scores: pd.Series | np.ndarray | None = None,
    spy_close_by_date: pd.Series | None = None,
) -> BacktestResult:
    """Run the backtest on ``test_df``.

    ``test_df`` must have columns: date, symbol, actual_open, actual_close,
    spread_bps, dolvol_20d_log, target_oc, + DAILY_FEATURE_COLS + CHRONOS_FEATURE_COLS.

    ``spy_close_by_date`` is an optional daily SPY close series (index: date)
    used by the regime-gate and vol-target-sizing knobs on ``config``. Pass
    ``None`` if both knobs are disabled.
    """
    required = {"date", "symbol", "actual_open", "actual_close", "spread_bps"}
    missing = required - set(test_df.columns)
    if missing:
        raise ValueError(f"test_df missing columns: {missing}")

    if config.overnight_max_gross_leverage is not None:
        cap_val = float(config.overnight_max_gross_leverage)
        if not np.isfinite(cap_val) or cap_val <= 0.0:
            raise ValueError(
                "overnight_max_gross_leverage must be positive and finite "
                f"(got {config.overnight_max_gross_leverage!r})"
            )

    # Compute combined scores for every row
    if precomputed_scores is None:
        scores = combined_scores(
            test_df, model,
            xgb_weight=config.xgb_weight,
            chronos_col=config.chronos_col,
        )
    elif isinstance(precomputed_scores, pd.Series):
        scores = precomputed_scores.reindex(test_df.index)
    else:
        arr = np.asarray(precomputed_scores, dtype=np.float64)
        if arr.shape[0] != len(test_df):
            raise ValueError(
                f"precomputed_scores length {arr.shape[0]} does not match test_df length {len(test_df)}"
            )
        scores = pd.Series(arr, index=test_df.index, name="combined_score")
    test_df = test_df.copy()
    test_df["_score"] = scores.values

    # Snapshot rows before deployable inference filters. Hold-through live
    # keeps existing positions when no new pick clears the gate; this lookup
    # lets the simulator mark those stale holds close-to-close even if the
    # held symbol would not be in today's filtered pick pool.
    hold_lookup: dict[tuple[date, str], pd.Series] = {
        (row["date"], str(row["symbol"])): row
        for _, row in (
            test_df.dropna(subset=["date", "symbol"])
            .drop_duplicates(subset=["date", "symbol"], keep="last")
            .iterrows()
        )
    }

    # Snapshot fallback-symbol rows BEFORE any filter dropped them. We
    # key by (symbol, date) so the no-picks path can pull the bar back
    # even if dolvol / vol / spread / rank filters excluded it from the
    # pick pool. Returns None for the lookup when no fallback is
    # configured so the hot path pays zero overhead.
    fallback_sym = (config.no_picks_fallback_symbol or "").strip().upper()
    fallback_lookup: dict | None = None
    if fallback_sym:
        _fb_rows = test_df[test_df["symbol"].astype(str).str.upper() == fallback_sym]
        # Index rows by date for O(1) lookup; keep the most recent duplicate
        # (should never happen but defensive). This is built before filters
        # so an all-filtered day can still fall back to the configured symbol.
        fallback_lookup = {
            row["date"]: row
            for _, row in (
                _fb_rows.dropna(subset=["date"])
                .drop_duplicates(subset=["date"], keep="last")
                .iterrows()
            )
        }

    # Liquidity filter
    min_dolvol_log = np.log1p(config.min_dollar_vol)
    if "dolvol_20d_log" in test_df.columns:
        test_df = test_df[test_df["dolvol_20d_log"] >= min_dolvol_log]

    # Spread filter — skip stocks with unrealistically wide volume-based spreads
    if "spread_bps" in test_df.columns and config.max_spread_bps > 0:
        test_df = test_df[test_df["spread_bps"] <= config.max_spread_bps]

    # Realised-vol floor — drops the dead-zone names that LOBO flagged
    # as a net drag. Inference-only; training universe stays broader.
    if config.min_vol_20d > 0.0 and "vol_20d" in test_df.columns:
        test_df = test_df[test_df["vol_20d"] >= float(config.min_vol_20d)]

    # Realised-vol ceiling — symmetric "mask high-vol names at inference"
    # filter. High vol_20d is crash-sensitive in stressed regimes; the
    # symbol can still contribute positively to TRAINING (no feature-
    # distribution shift at fit time), but is dropped from the PICK pool.
    if config.max_vol_20d > 0.0 and "vol_20d" in test_df.columns:
        test_df = test_df[test_df["vol_20d"] <= float(config.max_vol_20d)]

    # Cross-sectional (per-day) momentum rank filters. Compute per-day
    # percentile ranks and apply both bounds. Cheap: single groupby.rank
    # call per active filter. Each pick's rank is relative to the same
    # day's pool AFTER the absolute filters above already ran.
    ret20_active = float(config.max_ret_20d_rank_pct) < 1.0
    ret5_active = float(config.min_ret_5d_rank_pct) > 0.0
    if ret20_active and "ret_20d" in test_df.columns:
        r20 = test_df.groupby("date")["ret_20d"].rank(pct=True, method="average")
        test_df = test_df[r20 <= float(config.max_ret_20d_rank_pct)]
    if ret5_active and "ret_5d" in test_df.columns:
        r5 = test_df.groupby("date")["ret_5d"].rank(pct=True, method="average")
        test_df = test_df[r5 >= float(config.min_ret_5d_rank_pct)]

    # Cross-sectional regime gate — shared with live so a swept day-level
    # dispersion/skew gate evaluates the same keep/drop decision at deploy.
    iqr_active  = float(config.regime_cs_iqr_max) > 0.0
    skew_active = float(config.regime_cs_skew_min) > -1e8
    if (iqr_active or skew_active) and "ret_5d" in test_df.columns:
        day_keep = cross_sectional_regime_keep_by_date(
            test_df,
            regime_cs_iqr_max=float(config.regime_cs_iqr_max),
            regime_cs_skew_min=float(config.regime_cs_skew_min),
        )
        keep_dates = set(day_keep[day_keep].index)
        test_df = test_df[test_df["date"].isin(keep_dates)]

    # Drop rows without valid actual prices
    test_df = test_df.dropna(subset=["actual_open", "actual_close"])
    test_df = test_df[(test_df["actual_open"] > 0) & (test_df["actual_close"] > 0)]
    if config.fee_rate is None:
        symbol_fee_rates = {
            str(sym): _resolve_fee_rate(str(sym), config)
            for sym in pd.unique(test_df["symbol"].astype(str))
        }
    else:
        symbol_fee_rates = {}

    unique_dates = pd.Index(sorted(test_df["date"].unique()))
    if fallback_lookup is not None:
        unique_dates = pd.Index(sorted(set(unique_dates).union(fallback_lookup.keys())))
    regime_closed = _build_regime_flags(
        spy_close_by_date,
        unique_dates,
        window=config.regime_gate_window,
        available_at_open=True,
    )
    vol_scale = _build_vol_scale(
        spy_close_by_date,
        unique_dates,
        target_ann=config.vol_target_ann,
        available_at_open=True,
    )

    equity = config.initial_cash
    peak_equity = config.initial_cash
    day_results: list[DayResult] = []
    stopped_early = False
    stop_reason = ""
    # Hold-through state: last day's pick set + per-symbol close prices. When
    # today's pick set equals yesterday's (as a set), we carry the positions
    # instead of sell-at-close + buy-at-open. The carry day's return is
    # close-to-close with zero fees/buffer (no trade happens).
    prev_pick_set: frozenset[str] | None = None
    prev_close_by_sym: dict[str, float] = {}

    # Missed-order RNG — seeded once at entry so the whole sim is
    # reproducible under the same (skip_prob, skip_seed). Pre-building it
    # here (rather than per-day) avoids any dependence on day ordering.
    _skip_prob = float(max(0.0, min(1.0, config.skip_prob)))
    _skip_rng = (
        np.random.default_rng(int(config.skip_seed))
        if _skip_prob > 0.0 else None
    )

    grouped_by_day = {day: day_df for day, day_df in test_df.groupby("date", sort=True)}
    empty_day_df = test_df.iloc[0:0]

    for day in unique_dates:
        day_df = grouped_by_day.get(day, empty_day_df)
        # Regime gate — if SPY under MA for the day, skip (stay in cash).
        if bool(regime_closed.get(day, False)):
            # Flatten hold state on gated days so a re-entry can't claim a
            # continuation across a cash day.
            prev_pick_set = None
            prev_close_by_sym = {}
            continue
        day_scale = float(vol_scale.get(day, 1.0))
        # Rank by combined score; among ties prefer stronger Chronos signal.
        sort_cols = ["_score"]
        ascending = [False]
        if config.chronos_col in day_df.columns:
            sort_cols.append(config.chronos_col)
            ascending.append(False)
        day_df = day_df.sort_values(sort_cols, ascending=ascending, kind="mergesort")

        # Materialize today's pick rows up to top_n (skipping invalid prices).
        pick_rows: list[pd.Series] = []
        opportunistic_entries: dict[str, float] = {}
        min_pick_floor = min(max(int(config.min_picks), 0), int(config.top_n))
        opportunistic_watch_n = max(int(config.opportunistic_watch_n), 0)
        opportunistic_discount_bps = max(
            float(config.opportunistic_entry_discount_bps), 0.0
        )
        if opportunistic_watch_n > 0 and opportunistic_discount_bps > 0.0:
            watch_rows: list[pd.Series] = []
            for _, row in day_df.iterrows():
                if len(watch_rows) >= opportunistic_watch_n:
                    break
                score = float(row.get("_score", 0.0) or 0.0)
                if score < config.min_score and len(watch_rows) >= min_pick_floor:
                    continue
                o = float(row["actual_open"])
                c = float(row["actual_close"])
                if o <= 0 or c <= 0:
                    continue
                watch_rows.append(row)

            for row in watch_rows:
                if len(pick_rows) >= config.top_n:
                    break
                entry_limit = _limit_entry_fill_price(
                    row,
                    discount_bps=opportunistic_discount_bps,
                    fill_buffer_bps=config.fill_buffer_bps,
                )
                if entry_limit is None:
                    continue
                pick_rows.append(row)
                opportunistic_entries[str(row["symbol"])] = entry_limit
        else:
            for _, row in day_df.iterrows():
                if len(pick_rows) >= config.top_n:
                    break
                score = float(row.get("_score", 0.0) or 0.0)
                if score < config.min_score and len(pick_rows) >= min_pick_floor:
                    continue
                o = float(row["actual_open"])
                c = float(row["actual_close"])
                if o <= 0 or c <= 0:
                    continue
                pick_rows.append(row)

        today_sym_list = [str(r["symbol"]) for r in pick_rows]
        today_pick_set = frozenset(today_sym_list)

        is_continuation = (
            bool(config.hold_through)
            and prev_pick_set is not None
            and len(today_pick_set) > 0
            and today_pick_set == prev_pick_set
            and all(
                prev_close_by_sym.get(sym, 0.0) > 0.0 for sym in today_sym_list
            )
        )

        # Missed-order Monte Carlo — only applies to churn days (new
        # orders firing). Draw once per pick; failures reduce today's
        # pick set, mimicking an Alpaca 403 / spread-too-wide / stale
        # price. Continuations carry the previous position, so no order
        # fires and nothing can be "missed".
        if _skip_rng is not None and not is_continuation and pick_rows:
            draws = _skip_rng.random(len(pick_rows))
            surviving = [
                r for r, u in zip(pick_rows, draws) if u >= _skip_prob
            ]
            pick_rows = surviving
            today_sym_list = [str(r["symbol"]) for r in pick_rows]
            today_pick_set = frozenset(today_sym_list)

        # Conviction-scaled allocation — sizes today's exposure by the
        # top-of-pool score. Computed from the DAY's top score before any
        # of the pick_rows assembly trims it; falls back to 0 if no pick.
        # When the scale lands at 0 we drop the pick rows entirely so the
        # no-picks fallback path can fire (keeps the "trade with conviction
        # or SPY else" semantics cleaner than simulating a zero-leverage
        # trade that pays neither fees nor gets upside).
        conviction_scale = 1.0
        if config.conviction_scaled_alloc:
            top_score = (
                float(pick_rows[0]["_score"]) if pick_rows
                else float(day_df["_score"].max()) if len(day_df) > 0
                else 0.0
            )
            lo = float(config.conviction_alloc_low)
            hi = float(config.conviction_alloc_high)
            span = max(hi - lo, 1e-9)
            conviction_scale = float(np.clip((top_score - lo) / span, 0.0, 1.0))
            if conviction_scale <= 0.0:
                pick_rows = []
                today_sym_list = []
                today_pick_set = frozenset()
                is_continuation = False

        # Live hold-through semantics: if today's model emits no valid pick
        # but the bot already holds yesterday's pick, it does not liquidate.
        # It simply keeps the position. Model that as a close-to-close
        # continuation using the unfiltered daily row for the held symbol.
        if (
            config.hold_through
            and not pick_rows
            and prev_pick_set is not None
            and prev_close_by_sym
        ):
            held_rows: list[pd.Series] = []
            missing_held = False
            for sym in sorted(prev_pick_set):
                row = hold_lookup.get((day, sym))
                if row is None:
                    missing_held = True
                    break
                c = float(row.get("actual_close", 0.0) or 0.0)
                if c <= 0 or prev_close_by_sym.get(sym, 0.0) <= 0:
                    missing_held = True
                    break
                held_rows.append(row)
            if not missing_held and held_rows:
                pick_rows = held_rows
                today_sym_list = [str(r["symbol"]) for r in pick_rows]
                today_pick_set = frozenset(today_sym_list)
                is_continuation = True

        trades: list[DayTrade] = []
        for row in pick_rows:
            o = float(row["actual_open"])
            c = float(row["actual_close"])
            spread = float(row.get("spread_bps", 25.0))
            if not np.isfinite(spread) or spread <= 0:
                spread = 25.0
            symbol = str(row["symbol"])

            pick_scale = _inv_vol_pick_scale(
                float(row.get("vol_20d", 0.0)),
                target_ann=config.inv_vol_target_ann,
                floor=config.inv_vol_floor,
                cap=config.inv_vol_cap,
            )
            eff_lev = config.leverage * pick_scale * conviction_scale
            if config.overnight_max_gross_leverage is not None:
                eff_lev = min(
                    float(eff_lev),
                    float(config.overnight_max_gross_leverage),
                )

            if is_continuation:
                # Position is held across the prior close — no trade fires
                # today. PnL is close-to-close (captures overnight + intraday).
                prev_c = prev_close_by_sym[symbol]
                gross_oc = (c - prev_c) / prev_c
                gross_leveraged = eff_lev * gross_oc
                # Only the intraday margin-hold cost applies (no fees, no
                # fill buffer since there's no trade).
                cost_frac = _day_margin_cost(eff_lev)
                net = gross_leveraged - cost_frac
                worst_dd, best_runup = _intraday_excursion_pct(
                    row, prev_c, eff_lev,
                )
                trades.append(DayTrade(
                    symbol=symbol,
                    score=float(row["_score"]),
                    actual_open=o,
                    actual_close=c,
                    entry_fill_price=prev_c,
                    exit_fill_price=c,
                    spread_bps=spread,
                    commission_bps=0.0,
                    fee_rate=0.0,
                    fill_buffer_bps=0.0,
                    leverage=eff_lev,
                    gross_return_pct=gross_oc * 100.0,
                    net_return_pct=net * 100.0,
                    intraday_worst_dd_pct=worst_dd,
                    intraday_best_runup_pct=best_runup,
                ))
                continue

            # Churn day — full open-to-close round-trip with fees + buffer.
            fee_rate = symbol_fee_rates.get(symbol, _resolve_fee_rate(symbol, config))
            entry_fill = opportunistic_entries.get(symbol)
            if entry_fill is None:
                entry_fill, exit_fill = _fill_prices(
                    o, c, fill_buffer_bps=config.fill_buffer_bps
                )
            else:
                exit_fill = c * (
                    1.0 - max(float(config.fill_buffer_bps), 0.0) / 10_000.0
                )
            if entry_fill <= 0 or exit_fill <= 0:
                continue

            gross_oc = (exit_fill - entry_fill) / entry_fill
            round_trip_return = (exit_fill * (1.0 - fee_rate) / (entry_fill * (1.0 + fee_rate))) - 1.0
            gross_leveraged = eff_lev * round_trip_return

            cost_frac = (
                eff_lev * (2.0 * config.commission_bps) / 10_000.0
                + _day_margin_cost(eff_lev)
            )
            net = gross_leveraged - cost_frac

            worst_dd, best_runup = _intraday_excursion_pct(
                row, entry_fill, eff_lev,
            )
            trades.append(DayTrade(
                symbol=symbol,
                score=float(row["_score"]),
                actual_open=o,
                actual_close=c,
                entry_fill_price=entry_fill,
                exit_fill_price=exit_fill,
                spread_bps=spread,
                commission_bps=config.commission_bps,
                fee_rate=fee_rate,
                fill_buffer_bps=float(config.fill_buffer_bps),
                leverage=eff_lev,
                gross_return_pct=gross_oc * 100.0,
                net_return_pct=net * 100.0,
                intraday_worst_dd_pct=worst_dd,
                intraday_best_runup_pct=best_runup,
            ))

        # No-picks fallback — only fires on churn days (not continuations).
        # When the pick pool is empty OR conviction-scaling produced a zero
        # allocation, look up the fallback symbol in the pre-filter
        # snapshot and trade it at a reduced exposure. The fallback
        # symbol is NEVER a hold_through carry — even if tomorrow's
        # picks land empty again we submit a fresh fallback order, since
        # the production live_trader path would do the same (no "was I
        # already in SPY" state machine in the bot).
        fell_back = False
        if (
            not trades
            and not is_continuation
            and fallback_lookup is not None
            and float(config.no_picks_fallback_alloc_scale) != 0.0
        ):
            fb_row = fallback_lookup.get(day)
            if fb_row is not None:
                fb_o = float(fb_row.get("actual_open", 0.0) or 0.0)
                fb_c = float(fb_row.get("actual_close", 0.0) or 0.0)
                if fb_o > 0 and fb_c > 0:
                    fb_spread = float(fb_row.get("spread_bps", 5.0))
                    if not np.isfinite(fb_spread) or fb_spread <= 0:
                        fb_spread = 5.0
                    fb_lev = float(config.leverage) * float(
                        config.no_picks_fallback_alloc_scale
                    )
                    if config.overnight_max_gross_leverage is not None:
                        fb_lev = min(
                            float(fb_lev),
                            float(config.overnight_max_gross_leverage),
                        )
                    fb_fee = _resolve_fee_rate(fallback_sym, config)
                    fb_entry, fb_exit = _fill_prices(
                        fb_o, fb_c, fill_buffer_bps=config.fill_buffer_bps
                    )
                    if fb_entry > 0 and fb_exit > 0:
                        fb_gross_oc = (fb_exit - fb_entry) / fb_entry
                        fb_round_trip = (
                            fb_exit * (1.0 - fb_fee) / (fb_entry * (1.0 + fb_fee))
                        ) - 1.0
                        fb_gross_lev = fb_lev * fb_round_trip
                        fb_cost = (
                            fb_lev * (2.0 * config.commission_bps) / 10_000.0
                            + _day_margin_cost(fb_lev)
                        )
                        fb_net = fb_gross_lev - fb_cost
                        fb_worst, fb_runup = _intraday_excursion_pct(
                            fb_row, fb_entry, fb_lev,
                        )
                        trades.append(DayTrade(
                            symbol=fallback_sym,
                            score=0.0,
                            actual_open=fb_o,
                            actual_close=fb_c,
                            entry_fill_price=fb_entry,
                            exit_fill_price=fb_exit,
                            spread_bps=fb_spread,
                            commission_bps=config.commission_bps,
                            fee_rate=fb_fee,
                            fill_buffer_bps=float(config.fill_buffer_bps),
                            leverage=fb_lev,
                            gross_return_pct=fb_gross_oc * 100.0,
                            net_return_pct=fb_net * 100.0,
                            intraday_worst_dd_pct=fb_worst,
                            intraday_best_runup_pct=fb_runup,
                        ))
                        fell_back = True

        # Update hold-through state for next day. Fallback days always
        # flatten the hold state — the fallback symbol isn't "held
        # through" tomorrow.
        if config.hold_through and trades and not fell_back:
            prev_pick_set = today_pick_set
            prev_close_by_sym = {t.symbol: float(t.actual_close) for t in trades}
        else:
            prev_pick_set = None
            prev_close_by_sym = {}

        if not trades:
            continue

        weights = _allocation_weights(
            [t.score for t in trades],
            mode=config.allocation_mode,
            temperature=config.allocation_temp,
        )
        rets = np.asarray([t.net_return_pct for t in trades], dtype=np.float64)
        daily_ret_pct = float(np.dot(weights, rets))
        # Vol-target sizing: scale today's exposure by SPY-regime vol scalar in [0, 1].
        daily_ret_pct *= day_scale
        equity_end = equity * (1.0 + daily_ret_pct / 100.0)

        # Portfolio-weighted intraday excursions (proxy for worst-within-day
        # DD on today's positions at today's allocation).
        dd_vec = np.asarray(
            [t.intraday_worst_dd_pct for t in trades], dtype=np.float64
        )
        runup_vec = np.asarray(
            [t.intraday_best_runup_pct for t in trades], dtype=np.float64
        )
        day_intraday_dd    = float(np.dot(weights, dd_vec))    * day_scale
        day_intraday_runup = float(np.dot(weights, runup_vec)) * day_scale

        day_results.append(DayResult(
            day=day,  # type: ignore[arg-type]
            equity_start=equity,
            equity_end=equity_end,
            daily_return_pct=daily_ret_pct,
            trades=trades,
            n_candidates=len(day_df),
            intraday_worst_dd_pct=day_intraday_dd,
            intraday_best_runup_pct=day_intraday_runup,
        ))
        equity = equity_end
        peak_equity = max(peak_equity, equity_end)
        if (
            float(config.stop_on_intraday_drawdown_pct) > 0.0
            and day_intraday_dd >= float(config.stop_on_intraday_drawdown_pct)
        ):
            stopped_early = True
            stop_reason = (
                "intraday_drawdown_pct>="
                f"{float(config.stop_on_intraday_drawdown_pct):g}"
            )
            break
        if (
            float(config.stop_on_drawdown_pct) > 0.0
            and peak_equity > 0.0
        ):
            realized_dd_pct = (peak_equity - equity_end) / peak_equity * 100.0
            if realized_dd_pct >= float(config.stop_on_drawdown_pct):
                stopped_early = True
                stop_reason = (
                    f"drawdown_pct>={float(config.stop_on_drawdown_pct):g}"
                )
                break

    return _compute_result(
        day_results,
        config,
        stopped_early=stopped_early,
        stop_reason=stop_reason,
    )


def _compute_result(
    day_results: list[DayResult],
    config: BacktestConfig,
    *,
    stopped_early: bool = False,
    stop_reason: str = "",
) -> BacktestResult:
    if not day_results:
        return BacktestResult(
            config=config, day_results=[], initial_cash=config.initial_cash,
            final_equity=config.initial_cash, total_return_pct=0.0,
            monthly_return_pct=0.0, annualized_return_pct=0.0,
            sharpe_ratio=0.0, sortino_ratio=0.0, max_drawdown_pct=0.0,
            win_rate_pct=0.0, total_trades=0, avg_spread_bps=0.0, avg_fee_bps=0.0,
            directional_accuracy_pct=0.0,
            stopped_early=stopped_early,
            stop_reason=stop_reason,
        )

    rets = np.array([r.daily_return_pct / 100.0 for r in day_results])
    eq = np.array([config.initial_cash] + [r.equity_end for r in day_results])
    n = len(day_results)

    total_ret = (eq[-1] - eq[0]) / eq[0]
    ann_ret = (1.0 + total_ret) ** (TRADING_DAYS_PER_YEAR / n) - 1.0
    monthly_ret = (1.0 + total_ret) ** (21.0 / n) - 1.0

    mean_r = float(np.mean(rets))
    std_r  = float(np.std(rets, ddof=1)) if n > 1 else 1e-9
    sharpe = mean_r / std_r * np.sqrt(TRADING_DAYS_PER_YEAR) if std_r > 0 else 0.0

    sortino = _sortino_semi(rets, mean_r, ann_factor=np.sqrt(TRADING_DAYS_PER_YEAR))

    max_dd, tuw_pct, ulcer = _equity_drawdown_metrics(eq)

    win_rate = float(np.mean(rets > 0)) * 100.0

    all_trades = [t for r in day_results for t in r.trades]
    spreads = [t.spread_bps for t in all_trades]
    fee_bps = [t.fee_rate * 10_000.0 for t in all_trades]
    dir_acc = (float(np.mean([t.gross_return_pct > 0 for t in all_trades])) * 100.0
               if all_trades else 0.0)

    dd_series   = np.asarray([r.intraday_worst_dd_pct   for r in day_results])
    runup_series = np.asarray([r.intraday_best_runup_pct for r in day_results])
    worst_intraday_dd    = float(np.max(dd_series))    if dd_series.size    else 0.0
    avg_intraday_dd      = float(np.mean(dd_series))   if dd_series.size    else 0.0
    worst_intraday_runup = float(np.max(runup_series)) if runup_series.size else 0.0

    return BacktestResult(
        config=config,
        day_results=day_results,
        initial_cash=config.initial_cash,
        final_equity=float(eq[-1]),
        total_return_pct=total_ret * 100.0,
        monthly_return_pct=monthly_ret * 100.0,
        annualized_return_pct=ann_ret * 100.0,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown_pct=max_dd * 100.0,
        win_rate_pct=win_rate,
        total_trades=len(all_trades),
        avg_spread_bps=float(np.mean(spreads)) if spreads else 0.0,
        avg_fee_bps=float(np.mean(fee_bps)) if fee_bps else 0.0,
        directional_accuracy_pct=dir_acc,
        worst_intraday_dd_pct=worst_intraday_dd,
        avg_intraday_dd_pct=avg_intraday_dd,
        worst_intraday_runup_pct=worst_intraday_runup,
        time_under_water_pct=tuw_pct,
        ulcer_index=ulcer,
        stopped_early=stopped_early,
        stop_reason=stop_reason,
    )


def print_summary(res: BacktestResult, label: str = "") -> None:
    lbl = f" [{label}]" if label else ""
    print(f"\n{'='*68}")
    print(f"  XGBoost Open→Close Backtest{lbl}")
    print(f"  top_n={res.config.top_n}  leverage={res.config.leverage:.1f}x"
          f"  xgb_weight={res.config.xgb_weight:.2f}")
    print(f"{'='*68}")
    print(f"  Initial cash      : ${res.initial_cash:,.0f}")
    print(f"  Final equity      : ${res.final_equity:,.2f}")
    print(f"  Total return      : {res.total_return_pct:+.2f}%")
    print(f"  Monthly return    : {res.monthly_return_pct:+.2f}%  (21-day equiv)")
    print(f"  Ann. return       : {res.annualized_return_pct:+.2f}%")
    print(f"  Sharpe (ann.)     : {res.sharpe_ratio:.3f}")
    print(f"  Sortino (ann.)    : {res.sortino_ratio:.3f}")
    print(f"  Max drawdown      : {res.max_drawdown_pct:.2f}%")
    print(f"  Win rate          : {res.win_rate_pct:.1f}%")
    print(f"  Directional acc.  : {res.directional_accuracy_pct:.1f}%")
    print(f"  Total trades      : {res.total_trades}")
    print(f"  Trading days      : {len(res.day_results)}")
    print(f"  Avg spread        : {res.avg_spread_bps:.1f} bps")
    print(f"  Avg fee/side      : {res.avg_fee_bps:.3f} bps")
    print(f"  Fill buffer/side  : {res.config.fill_buffer_bps:.1f} bps")
    print(f"  Commission/side   : {res.config.commission_bps:.3f} bps")
    print(f"{'='*68}")


__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "DayResult",
    "DayTrade",
    "_allocation_weights",
    "simulate",
    "simulate_hourly",
    "print_summary",
    "STOCK_HOURS_PER_YEAR",
    "CRYPTO_HOURS_PER_YEAR",
]


# ── Hourly backtest (bar-granular, same cost model) ───────────────────────────

# Regular US-equity session: 6.5 hr × 252 td = 1,638 bars/yr
STOCK_HOURS_PER_YEAR = 252 * 6.5
# Crypto trades 24/7: 24 hr × 365 d = 8,760 bars/yr  (≈ 5.35× higher budget)
CRYPTO_HOURS_PER_YEAR = 24 * 365
# Per-calendar-month bar counts (used for monthly-return annualisation)
STOCK_HOURS_PER_MONTH = 21 * 6.5          # ≈ 136.5
CRYPTO_HOURS_PER_MONTH = (24 * 365) / 12  # ≈ 730.0


def _hour_margin_cost(leverage: float, *, bars_per_year: float) -> float:
    """Intraday-hour margin cost fraction (held one bar)."""
    if leverage <= 1.0:
        return 0.0
    return (leverage - 1.0) * ANNUAL_MARGIN_RATE / float(bars_per_year)


def simulate_hourly(
    test_df: pd.DataFrame,
    model: XGBStockModel,
    config: BacktestConfig,
    *,
    bars_per_year: float = STOCK_HOURS_PER_YEAR,
    bars_per_month: float | None = None,
    kind_map: dict[str, str] | None = None,
    precomputed_scores: pd.Series | np.ndarray | None = None,
) -> BacktestResult:
    """Hourly analog of :func:`simulate`.

    Each bar is treated like a mini "day": pick top_n by score, buy at bar's
    open fill, sell at its close fill, costs applied per bar. PnL compounds
    across bars.

    ``test_df`` must carry columns: timestamp, symbol, actual_open, actual_close,
    spread_bps, target_oc, + HOURLY_FEATURE_COLS.

    ``bars_per_year`` controls annualisation; pass ``CRYPTO_HOURS_PER_YEAR``
    for a pure-crypto universe (24/7). For a mixed universe the caller should
    segment by kind and re-run.

    ``bars_per_month`` defaults to ``bars_per_year / 12`` so the "monthly
    return" reported on a stock-only OOS window is ≈ 136.5 bars regardless
    of window size.

    ``kind_map``: optional {symbol: 'stocks'|'crypto'} for per-symbol fee
    resolution. Falls back to ``get_fee_for_symbol`` heuristics.
    """
    if bars_per_month is None:
        bars_per_month = float(bars_per_year) / 12.0

    required = {"timestamp", "symbol", "actual_open", "actual_close", "spread_bps"}
    missing = required - set(test_df.columns)
    if missing:
        raise ValueError(f"test_df missing columns: {missing}")

    if config.overnight_max_gross_leverage is not None:
        cap_val = float(config.overnight_max_gross_leverage)
        if not np.isfinite(cap_val) or cap_val <= 0.0:
            raise ValueError(
                "overnight_max_gross_leverage must be positive and finite "
                f"(got {config.overnight_max_gross_leverage!r})"
            )

    if precomputed_scores is None:
        scores = combined_scores(
            test_df, model,
            xgb_weight=config.xgb_weight,
            chronos_col=config.chronos_col,
        )
    elif isinstance(precomputed_scores, pd.Series):
        scores = precomputed_scores.reindex(test_df.index)
    else:
        arr = np.asarray(precomputed_scores, dtype=np.float64)
        if arr.shape[0] != len(test_df):
            raise ValueError(
                f"precomputed_scores length {arr.shape[0]} != test_df length {len(test_df)}"
            )
        scores = pd.Series(arr, index=test_df.index, name="combined_score")

    df = test_df.copy()
    df["_score"] = scores.values

    # Drop obvious bad rows
    df = df.dropna(subset=["actual_open", "actual_close"])
    df = df[(df["actual_open"] > 0) & (df["actual_close"] > 0)]

    # Spread filter
    if config.max_spread_bps > 0 and "spread_bps" in df.columns:
        df = df[df["spread_bps"] <= config.max_spread_bps]

    # Resolve fees once per symbol
    symbol_fee_rates: dict[str, float] = {}
    if config.fee_rate is None:
        for sym in pd.unique(df["symbol"].astype(str)):
            symbol_fee_rates[str(sym)] = _resolve_fee_rate(str(sym), config)

    _hourly_eff_lev = float(config.leverage)
    if config.overnight_max_gross_leverage is not None:
        _hourly_eff_lev = min(
            _hourly_eff_lev, float(config.overnight_max_gross_leverage)
        )
    margin_per_bar = _hour_margin_cost(_hourly_eff_lev, bars_per_year=bars_per_year)

    equity = float(config.initial_cash)
    bar_results: list[DayResult] = []

    # Group by timestamp — each bar is one "decision"
    for bar_ts, bar_df in df.groupby("timestamp", sort=True):
        sort_cols = ["_score"]
        ascending = [False]
        if config.chronos_col in bar_df.columns:
            sort_cols.append(config.chronos_col)
            ascending.append(False)
        bar_df = bar_df.sort_values(sort_cols, ascending=ascending, kind="mergesort")

        picks = bar_df[bar_df["_score"] >= config.min_score].head(config.top_n * 3)

        trades: list[DayTrade] = []
        for _, row in picks.iterrows():
            if len(trades) >= config.top_n:
                break

            o = float(row["actual_open"])
            c = float(row["actual_close"])
            if o <= 0 or c <= 0:
                continue

            spread = float(row.get("spread_bps", 25.0))
            if not np.isfinite(spread) or spread <= 0:
                spread = 25.0

            symbol = str(row["symbol"])
            fee_rate = (
                float(config.fee_rate) if config.fee_rate is not None
                else symbol_fee_rates.get(symbol, _resolve_fee_rate(symbol, config))
            )

            entry_fill, exit_fill = _fill_prices(
                o, c, fill_buffer_bps=config.fill_buffer_bps
            )
            if entry_fill <= 0 or exit_fill <= 0:
                continue

            gross_oc = (exit_fill - entry_fill) / entry_fill
            round_trip_return = (
                exit_fill * (1.0 - fee_rate) / (entry_fill * (1.0 + fee_rate))
            ) - 1.0
            eff_lev = float(config.leverage)
            if config.overnight_max_gross_leverage is not None:
                eff_lev = min(eff_lev, float(config.overnight_max_gross_leverage))
            gross_leveraged = eff_lev * round_trip_return

            cost_frac = (
                eff_lev * (2.0 * config.commission_bps) / 10_000.0
                + margin_per_bar
            )
            net = gross_leveraged - cost_frac

            trades.append(DayTrade(
                symbol=symbol,
                score=float(row["_score"]),
                actual_open=o,
                actual_close=c,
                entry_fill_price=entry_fill,
                exit_fill_price=exit_fill,
                spread_bps=spread,
                commission_bps=config.commission_bps,
                fee_rate=fee_rate,
                fill_buffer_bps=float(config.fill_buffer_bps),
                leverage=eff_lev,
                gross_return_pct=gross_oc * 100.0,
                net_return_pct=net * 100.0,
            ))

        if not trades:
            continue

        weights = _allocation_weights(
            [t.score for t in trades],
            mode=config.allocation_mode,
            temperature=config.allocation_temp,
        )
        rets = np.asarray([t.net_return_pct for t in trades], dtype=np.float64)
        bar_ret_pct = float(np.dot(weights, rets))
        equity_end = equity * (1.0 + bar_ret_pct / 100.0)

        # DayResult.day stores the bar timestamp — caller can keep timestamp
        # precision even though the dataclass field is named ``day``.
        bar_day = bar_ts.to_pydatetime() if hasattr(bar_ts, "to_pydatetime") else bar_ts
        bar_results.append(DayResult(
            day=bar_day,  # type: ignore[arg-type]
            equity_start=equity,
            equity_end=equity_end,
            daily_return_pct=bar_ret_pct,
            trades=trades,
            n_candidates=len(bar_df),
        ))
        equity = equity_end

    return _compute_result_hourly(
        bar_results, config,
        bars_per_year=float(bars_per_year),
        bars_per_month=float(bars_per_month),
    )


def _compute_result_hourly(
    bar_results: list[DayResult],
    config: BacktestConfig,
    *,
    bars_per_year: float,
    bars_per_month: float,
) -> BacktestResult:
    """Build a ``BacktestResult`` for hourly bars.

    Mirrors ``_compute_result`` but uses bars-per-year / bars-per-month for
    annualisation. The ``day_results`` list carries one entry per traded bar,
    not per day, so the caller should not treat ``len(day_results)`` as a day
    count in hourly context.
    """
    if not bar_results:
        return BacktestResult(
            config=config, day_results=[], initial_cash=config.initial_cash,
            final_equity=config.initial_cash, total_return_pct=0.0,
            monthly_return_pct=0.0, annualized_return_pct=0.0,
            sharpe_ratio=0.0, sortino_ratio=0.0, max_drawdown_pct=0.0,
            win_rate_pct=0.0, total_trades=0, avg_spread_bps=0.0, avg_fee_bps=0.0,
            directional_accuracy_pct=0.0,
        )

    rets = np.array([r.daily_return_pct / 100.0 for r in bar_results])
    eq = np.array([config.initial_cash] + [r.equity_end for r in bar_results])
    n_bars = len(bar_results)

    total_ret = (eq[-1] - eq[0]) / eq[0]
    ann_ret = (1.0 + total_ret) ** (bars_per_year / n_bars) - 1.0
    monthly_ret = (1.0 + total_ret) ** (bars_per_month / n_bars) - 1.0

    mean_r = float(np.mean(rets))
    std_r = float(np.std(rets, ddof=1)) if n_bars > 1 else 1e-9
    sharpe = mean_r / std_r * np.sqrt(bars_per_year) if std_r > 0 else 0.0
    sortino = _sortino_semi(rets, mean_r, ann_factor=np.sqrt(bars_per_year))

    max_dd, tuw_pct, ulcer = _equity_drawdown_metrics(eq)

    win_rate = float(np.mean(rets > 0)) * 100.0

    all_trades = [t for r in bar_results for t in r.trades]
    spreads = [t.spread_bps for t in all_trades]
    fee_bps = [t.fee_rate * 10_000.0 for t in all_trades]
    dir_acc = (
        float(np.mean([t.gross_return_pct > 0 for t in all_trades])) * 100.0
        if all_trades else 0.0
    )

    return BacktestResult(
        config=config,
        day_results=bar_results,
        initial_cash=config.initial_cash,
        final_equity=float(eq[-1]),
        total_return_pct=total_ret * 100.0,
        monthly_return_pct=monthly_ret * 100.0,
        annualized_return_pct=ann_ret * 100.0,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown_pct=max_dd * 100.0,
        win_rate_pct=win_rate,
        total_trades=len(all_trades),
        avg_spread_bps=float(np.mean(spreads)) if spreads else 0.0,
        avg_fee_bps=float(np.mean(fee_bps)) if fee_bps else 0.0,
        directional_accuracy_pct=dir_acc,
        time_under_water_pct=tuw_pct,
        ulcer_index=ulcer,
    )
