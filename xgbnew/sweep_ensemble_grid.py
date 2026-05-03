"""Grid-sweep backtest knobs on a pre-trained ensemble.

Unlike ``xgbnew.eval_ensemble`` (which retrains each run), this loads a fixed
set of pre-trained XGBStockModel pickles, computes blended OOS scores ONCE,
then runs the windowed backtest over an arbitrary (leverage × min_score
× hold_through × top_n × fee_regime) grid.

Purpose: after deploying the live config (e.g. 5-seed alltrain @ lev=2.0
ms=0.85 hold_through), we want to know — without redoing 5× model training —
whether bumping lev to 2.5 or dropping ms to 0.80 would beat deploy on
strict-dominance (Δ median ≥ 0 AND Δ p10 ≥ 0 AND Δ neg ≤ 0) at 36× fees.

Example::

    python -m xgbnew.sweep_ensemble_grid \
        --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
        --data-root trainingdata \
        --model-paths analysis/xgbnew_daily/alltrain_ensemble_gpu/alltrain_seed0.pkl,\
analysis/xgbnew_daily/alltrain_ensemble_gpu/alltrain_seed7.pkl,\
analysis/xgbnew_daily/alltrain_ensemble_gpu/alltrain_seed42.pkl,\
analysis/xgbnew_daily/alltrain_ensemble_gpu/alltrain_seed73.pkl,\
analysis/xgbnew_daily/alltrain_ensemble_gpu/alltrain_seed197.pkl \
        --oos-start 2025-01-02 --oos-end 2026-04-19 \
        --window-days 30 --stride-days 7 \
        --leverage-grid 2.0,2.25,2.5,2.75,3.0 \
        --min-score-grid 0.80,0.85,0.90 \
        --hold-through \
        --fee-regimes deploy,prod10bps,stress36x \
        --output-dir analysis/xgbnew_ensemble_sweep
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from dataclasses import MISSING, dataclass, field
from datetime import date
from itertools import product
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from xgbnew.backtest import PRODUCTION_STOCK_FEE_RATE, BacktestConfig, simulate
from xgbnew.dataset import build_daily_dataset, load_chronos_cache, load_fm_latents
from xgbnew.features import (
    DAILY_DISPERSION_FEATURE_COLS,
    DAILY_RANK_FEATURE_COLS,
    LIVE_SUPPORTED_FEATURE_COLS,
)
from xgbnew.model import XGBStockModel
from xgbnew.model_registry import load_any_model


logger = logging.getLogger(__name__)


def _infer_required_fm_latents(
    feature_cols: tuple[str, ...],
    path: Path,
) -> int | None:
    """Return required latent count, rejecting sparse latent feature contracts."""
    latent_indices: list[int] = []
    for col in feature_cols:
        if not col.startswith("latent_"):
            continue
        suffix = col.removeprefix("latent_")
        if not suffix.isdecimal():
            raise ValueError(f"{path}: model feature_cols contains invalid FM latent column {col!r}")
        latent_indices.append(int(suffix))
    if not latent_indices:
        return None
    unique_indices = sorted(set(latent_indices))
    expected = list(range(unique_indices[-1] + 1))
    if unique_indices != expected:
        raise ValueError(
            f"{path}: model FM latent feature columns must be contiguous from latent_0; "
            f"got {unique_indices}"
        )
    return unique_indices[-1] + 1


# ── Fee regimes ───────────────────────────────────────────────────────────────
# "deploy" is the historical Alpaca fee baseline kept for compatibility.
# "prod10bps" is the current production-realism fee at the normal fill buffer.
# "stress36x" is the harsher realism-gate stress cell.
FEE_REGIMES: dict[str, dict[str, float]] = {
    "deploy": {"fee_rate": 0.0000278, "fill_buffer_bps": 5.0, "commission_bps": 0.0},
    "prod10bps": {
        "fee_rate": PRODUCTION_STOCK_FEE_RATE,
        "fill_buffer_bps": 5.0,
        "commission_bps": 0.0,
    },
    "stress36x": {"fee_rate": 0.001, "fill_buffer_bps": 15.0, "commission_bps": 10.0},
}


@dataclass
class CellResult:
    leverage: float
    min_score: float
    hold_through: bool
    top_n: int
    fee_regime: str
    n_windows: int
    median_monthly_pct: float
    p10_monthly_pct: float
    median_sortino: float
    worst_dd_pct: float
    n_neg: int
    mean_abs_neg_monthly_pct: float = 0.0
    monthly_return_pcts: list[float] = field(default_factory=list)
    window_sortino_values: list[float] = field(default_factory=list)
    window_drawdown_pcts: list[float] = field(default_factory=list)
    window_time_under_water_pcts: list[float] = field(default_factory=list)
    window_ulcer_indexes: list[float] = field(default_factory=list)
    window_active_day_pcts: list[float] = field(default_factory=list)
    window_worst_intraday_dd_pcts: list[float] = field(default_factory=list)
    window_avg_intraday_dd_pcts: list[float] = field(default_factory=list)
    window_start_dates: list[str] = field(default_factory=list)
    window_end_dates: list[str] = field(default_factory=list)
    short_n: int = 0
    max_short_score: float = 0.45
    short_allocation_scale: float = 0.5
    # Aggressive packing floor. 0 means classic min_score-gated behavior.
    min_picks: int = 0
    # Opportunistic work-stealing entries: watch more names than top_n and
    # enter only when a posted buy limit below the open is penetrated.
    opportunistic_watch_n: int = 0
    opportunistic_entry_discount_bps: float = 0.0
    # Uncertainty-adjusted sorting penalty. Scores used for selection are
    # ensemble_mean - penalty * ensemble_std across seed models. 0 disables.
    score_uncertainty_penalty: float = 0.0
    # Composite "one-scalar" optimization target — see compute_goodness().
    goodness_score: float = 0.0
    # Asymmetric-downside variant — see compute_robust_goodness().
    robust_goodness_score: float = 0.0
    # Robust downside score with explicit equity-curve pain penalties.
    pain_adjusted_goodness_score: float = 0.0
    # Fail-fast pruning fields. When enabled, a cell that breaches the
    # configured risk budget keeps observed partial metrics but its ranking
    # scores are forced to FAIL_FAST_SCORE so it cannot win a sweep.
    fail_fast_triggered: bool = False
    fail_fast_reason: str = ""
    # Intraday unrealized excursions, aggregated across windows.
    # worst_intraday_dd_pct = max over windows of that window's worst
    #   within-day unrealized drawdown (from entry fill to bar low, at
    #   leverage). 0.0 when actual_high/actual_low missing from dataset.
    worst_intraday_dd_pct: float = 0.0
    avg_intraday_dd_pct:   float = 0.0
    # Equity-curve pain metrics aggregated across windows.
    # time_under_water_pct = median over windows of %-of-days the equity
    #   curve is strictly below its prior peak.
    # ulcer_index          = median over windows of sqrt(mean(dd_pct^2)).
    # Paired median aggregation mirrors median_monthly_pct / median_sortino.
    time_under_water_pct: float = 0.0
    ulcer_index:          float = 0.0
    # Fraction of elapsed OOS days with at least one simulated trade.
    # This is diagnostic only, but important: monthly returns are annualised
    # over elapsed window days, not only active days, so sparse high-gate
    # strategies cannot look better merely because they sat in cash.
    median_active_day_pct: float = 0.0
    min_active_day_pct:    float = 0.0
    # Inference-side liquidity floor applied to the pick pool — stays at
    # 5M$ unless explicitly swept via --inference-min-dolvol-grid.
    inference_min_dolvol: float = 5_000_000.0
    # Inference-side volume-estimated spread ceiling. 30 matches live and
    # BacktestConfig defaults; 0 disables the spread filter.
    inference_max_spread_bps: float = 30.0
    # Inference-side realised-vol floor on vol_20d (annualised). 0 disables.
    # Swept via --inference-min-vol-grid.
    inference_min_vol_20d: float = 0.0
    # Inference-side realised-vol CEILING on vol_20d (annualised). 0
    # disables. Drops crash-sensitive high-vol names from the pick pool
    # at inference while keeping them in training. Swept via
    # --inference-max-vol-grid. Band-pass with inference_min_vol_20d.
    inference_max_vol_20d: float = 0.0
    # Missed-order Monte Carlo params — 0.0 = classic identity sim.
    skip_prob: float = 0.0
    skip_seed: int = 0
    # Per-cell fill-buffer override in bps. Sentinel -1.0 = "use the fee
    # regime's default FB". Non-negative = override. The sweep default is
    # sentinel so existing eval paths that don't pass ``fill_buffer_bps``
    # still see the historic 5-bps/15-bps regimes exactly.
    fill_buffer_bps: float = -1.0
    # SPY-based regime knobs (see BacktestConfig.{regime_gate_window,
    # vol_target_ann}). 0 = disabled for either axis so legacy cells
    # reproduce bit-for-bit when the grid is not swept.
    regime_gate_window: int = 0
    vol_target_ann: float = 0.0
    # Per-pick inv-vol sizing (see BacktestConfig.inv_vol_target_ann).
    # 0 disables. Swept via --inv-vol-target-grid. Floor + cap are
    # single-valued (sweep-level, not per-cell) — pattern mirrors
    # skip_prob/skip_seed where only the primary axis is gridded.
    inv_vol_target_ann: float = 0.0
    inv_vol_floor: float = 0.05
    inv_vol_cap: float = 3.0
    # Cross-sectional momentum rank filters (per-day). 1.0 / 0.0
    # respectively = disabled (matches BacktestConfig defaults).
    max_ret_20d_rank_pct: float = 1.0
    min_ret_5d_rank_pct:  float = 0.0
    # Cross-sectional regime-dispersion gate (BacktestConfig.regime_cs_iqr_max
    # / regime_cs_skew_min). Gates entire DAYS based on cross-sectional
    # ret_5d dispersion — first lever found to flip sign on the fresh
    # 2025-07→2026-04 true-OOS. 0.0 = disabled for iqr; -1e9 = disabled
    # for skew.
    regime_cs_iqr_max:  float = 0.0
    regime_cs_skew_min: float = -1e9
    # Correlation-aware packing gate — see BacktestConfig.corr_*.
    corr_window_days: int = 0
    corr_min_periods: int = 20
    corr_max_signed: float = 1.0
    # No-picks fallback — see BacktestConfig.no_picks_fallback_*.
    # Empty symbol = disabled (legacy hold-cash behaviour). When set,
    # sim buys this symbol at leverage * alloc_scale on days where no
    # scored candidate clears min_score (or every candidate was filtered
    # out). Use to keep market exposure on low-conviction days.
    no_picks_fallback_symbol: str = ""
    no_picks_fallback_alloc_scale: float = 0.0
    # Conviction-scaled allocation — see BacktestConfig.conviction_*.
    # Binary (trade/cash) and scaled (trade at size proportional to
    # top-score) policies are mutually compatible with no_picks_fallback.
    conviction_scaled_alloc: bool = False
    conviction_alloc_low:  float = 0.55
    conviction_alloc_high: float = 0.85
    # Per-day allocation weights across the selected top-N names.
    # equal = legacy; score_norm/softmax concentrate more capital into
    # higher-confidence names when top_n > 1; worksteal gives the first
    # executed fill 75% of gross exposure and later fills split the rest.
    allocation_mode: str = "equal"
    allocation_temp: float = 1.0
    # Cap largest allocation so at least this fraction of gross exposure is
    # outside the main pick when two or more trades fill.
    min_secondary_allocation: float = 0.0
    # Full rolling-window count available to this cell before fail-fast or
    # simulator early-stop pruning. Older checkpoint rows may omit this and
    # keep the legacy minimum-window compatibility path.
    expected_n_windows: int = 0
    # Ensemble feature mode used to build the scored OOS panel. Persisted so
    # promotion/deploy audits can verify live scoring supports the evaluated
    # feature schema.
    ensemble_needs_ranks: bool = False
    ensemble_needs_dispersion: bool = False


# Default weights: p10 drives the reward; worst-DD is a unit-for-unit
# penalty; any negative window is a 10pp penalty apiece. Keeping these
# here (not as CLI flags) because the output JSON now embeds them and
# the table uses the same numbers, so they need a single source of truth.
GOODNESS_WEIGHTS = {
    "p10_coef":   1.0,   # 1× p10 monthly %
    "dd_coef":    1.0,   # subtract worst_dd_pct
    "neg_coef": 100.0,   # subtract neg_frac * 100 (so 1/60 neg = −1.67pp)
}


# Robust variant — asymmetric downside. Rationale:
#   * dd_coef=1.5        — deeper DDs bite more than shallow ones
#   * neg_count_coef=50  — softer than 100 because we also add magnitude
#   * neg_magnitude_coef=2.0 — mean-of-losses term distinguishes 5× (−3%) mo
#                              from 5× (−15%) mo, which the plain count
#                              formula cannot do.
# See feedback: "1.2-1.5x on short-side volatility/drawdowns or realized
# losses" — this is that knob, as a separate sort-key so the existing
# frontier analysis is not disrupted.
ROBUST_GOODNESS_WEIGHTS = {
    "p10_coef":            1.0,
    "dd_coef":             1.5,
    "neg_count_coef":     50.0,
    "neg_magnitude_coef":  2.0,
}


# Pain-adjusted variant — starts from robust_goodness and then penalizes
# how long the strategy stays underwater plus the RMS depth of drawdowns.
# This matches the production preference for strategies that recover fast,
# not merely ones with acceptable endpoint DD.
PAIN_ADJUSTED_GOODNESS_WEIGHTS = {
    "tuw_coef":   0.25,   # 20% time underwater => -5 points
    "ulcer_coef": 1.00,   # ulcer 5 => -5 points
}


FAIL_FAST_SCORE = -1_000_000_000.0
PRODUCTION_TARGET_MEDIAN_MONTHLY_PCT = 27.0
PRODUCTION_TARGET_MAX_DD_PCT = 25.0
PRODUCTION_TARGET_MAX_NEG_WINDOWS = 0
PRODUCTION_TARGET_MIN_WINDOWS = 1
PRODUCTION_TARGET_EXIT_CODE = 3


STRATEGY_PARAM_FIELDS = (
    "leverage",
    "min_score",
    "hold_through",
    "top_n",
    "short_n",
    "max_short_score",
    "short_allocation_scale",
    "min_picks",
    "opportunistic_watch_n",
    "opportunistic_entry_discount_bps",
    "score_uncertainty_penalty",
    "inference_min_dolvol",
    "inference_max_spread_bps",
    "inference_min_vol_20d",
    "inference_max_vol_20d",
    "regime_gate_window",
    "vol_target_ann",
    "inv_vol_target_ann",
    "inv_vol_floor",
    "inv_vol_cap",
    "max_ret_20d_rank_pct",
    "min_ret_5d_rank_pct",
    "regime_cs_iqr_max",
    "regime_cs_skew_min",
    "corr_window_days",
    "corr_min_periods",
    "corr_max_signed",
    "no_picks_fallback_symbol",
    "no_picks_fallback_alloc_scale",
    "conviction_scaled_alloc",
    "conviction_alloc_low",
    "conviction_alloc_high",
    "allocation_mode",
    "allocation_temp",
    "min_secondary_allocation",
)


def compute_goodness(
    p10_monthly_pct: float,
    worst_dd_pct: float,
    n_neg: int,
    n_windows: int,
    weights: dict | None = None,
) -> float:
    """Composite optimization target in "%/month-equivalent" units.

    goodness = p10_coef * p10 − dd_coef * worst_dd − neg_coef * neg_frac

    Units: same scale as p10 monthly %. Higher is better. Designed so a
    strategy that lifts p10 by +5pp but widens worst_dd by +2pp still
    shows a clear +3 improvement, while ANY negative window is punished
    proportionally (1 neg window out of 60 = −1.67 goodness).
    """
    w = GOODNESS_WEIGHTS if weights is None else weights
    neg_frac = float(n_neg) / float(max(n_windows, 1))
    return (
        w["p10_coef"] * p10_monthly_pct
        - w["dd_coef"] * worst_dd_pct
        - w["neg_coef"] * neg_frac
    )


def compute_robust_goodness(
    monthlies: "list[float] | np.ndarray",
    worst_dd_pct: float,
    weights: dict | None = None,
) -> float:
    """Asymmetric-downside goodness: magnitude-aware, DD-weighted.

    robust = p10 − 1.5·worst_dd − 50·neg_frac − 2·mean_abs_neg_return

    where ``mean_abs_neg_return = Σ |m_i| for m_i<0  /  n_windows``  (so
    it's 0 when every window is positive, and grows with both the count
    AND the magnitude of losses). A 5/60-neg config at −3%/mo pays
    −0.5pp from the magnitude term; the same count at −15%/mo pays
    −2.5pp. Identical under the plain count formula, distinguished here.
    """
    w = ROBUST_GOODNESS_WEIGHTS if weights is None else weights
    arr = np.asarray(monthlies, dtype=float)
    n = int(arr.size)
    if n == 0:
        return 0.0
    p10 = float(np.percentile(arr, 10))
    neg_mask = arr < 0
    neg_count = int(neg_mask.sum())
    neg_frac = neg_count / n
    mean_abs_neg = float(-arr[neg_mask].sum()) / n if neg_count else 0.0
    return (
        w["p10_coef"] * p10
        - w["dd_coef"] * worst_dd_pct
        - w["neg_count_coef"] * neg_frac
        - w["neg_magnitude_coef"] * mean_abs_neg
    )


def compute_pain_adjusted_goodness(
    monthlies: "list[float] | np.ndarray",
    worst_dd_pct: float,
    time_under_water_pct: float,
    ulcer_index: float,
    *,
    robust_weights: dict | None = None,
    pain_weights: dict | None = None,
) -> float:
    """Robust goodness further penalized by equity-curve pain.

    This keeps the robust downside-aware ranking intact, then subtracts:

      0.25 * time_under_water_pct + 1.0 * ulcer_index

    so otherwise-similar strategies that spend less time below prior highs
    rank ahead of long-recovery variants.
    """
    pw = PAIN_ADJUSTED_GOODNESS_WEIGHTS if pain_weights is None else pain_weights
    robust = compute_robust_goodness(
        monthlies,
        worst_dd_pct,
        weights=robust_weights,
    )
    return (
        robust
        - pw["tuw_coef"] * float(time_under_water_pct)
        - pw["ulcer_coef"] * float(ulcer_index)
    )


def _build_windows(days, window_days: int, stride_days: int):
    if len(days) < window_days:
        return []
    out = []
    i = 0
    while i + window_days <= len(days):
        span = days[i : i + window_days]
        out.append((span[0], span[-1]))
        i += stride_days
    return out


def _monthly_return(total_pct: float, n_days: int) -> float:
    if n_days <= 0:
        return 0.0
    return (1.0 + total_pct / 100.0) ** (21.0 / n_days) - 1.0


def _elapsed_window_days(w_df: pd.DataFrame, res) -> int:
    """Elapsed trading days represented by a window result.

    ``simulate().day_results`` intentionally contains traded days only. For
    ranking monthly returns, cash days must still count as elapsed time;
    otherwise strict gates that fire once in a 30-day window get annualised as
    a one-day strategy. When the simulator stops early, count days only through
    the stop day so fail-fast partial metrics describe observed time.
    """
    if "date" not in w_df.columns:
        return 0
    dates = pd.Index(sorted(pd.unique(w_df["date"])))
    if len(dates) == 0:
        return 0
    if bool(getattr(res, "stopped_early", False)) and getattr(res, "day_results", None):
        stop_day = getattr(res.day_results[-1], "day", None)
        if stop_day is not None:
            return int(np.sum(dates <= stop_day))
    return int(len(dates))


def _load_symbols(path: Path) -> list[str]:
    out: list[str] = []
    with open(path) as f:
        for ln in f:
            s = ln.strip()
            if s and not s.startswith("#"):
                out.append(s)
    return out


def _blend_scores(oos_df: pd.DataFrame, models: list[XGBStockModel],
                  blend_mode: str) -> pd.Series:
    mat = np.stack(
        [m.predict_scores(oos_df).values for m in models], axis=0
    )
    if blend_mode == "mean":
        blended = mat.mean(axis=0)
    elif blend_mode == "median":
        blended = np.median(mat, axis=0)
    else:
        raise ValueError(f"unsupported blend_mode: {blend_mode}")
    return pd.Series(blended, index=oos_df.index, name="ensemble_score")


def _ensemble_score_mean_std(
    oos_df: pd.DataFrame,
    models: list[XGBStockModel],
    blend_mode: str,
) -> tuple[pd.Series, pd.Series]:
    """Return ensemble center and cross-seed dispersion for uncertainty sorting."""
    mat = np.stack(
        [m.predict_scores(oos_df).values for m in models], axis=0
    )
    if blend_mode == "mean":
        center = mat.mean(axis=0)
    elif blend_mode == "median":
        center = np.median(mat, axis=0)
    else:
        raise ValueError(f"unsupported blend_mode: {blend_mode}")
    spread = mat.std(axis=0)
    return (
        pd.Series(center, index=oos_df.index, name="ensemble_score"),
        pd.Series(spread, index=oos_df.index, name="ensemble_score_std"),
    )


def _uncertainty_adjusted_scores(
    scores: pd.Series,
    score_std: pd.Series | None,
    penalty: float,
) -> pd.Series:
    penalty = max(float(penalty or 0.0), 0.0)
    if penalty <= 0.0 or score_std is None:
        return scores
    adjusted = scores - penalty * score_std.reindex(scores.index).fillna(0.0)
    return adjusted.rename(f"{scores.name or 'score'}_uadj")


def _allocation_grid_pairs(
    allocation_modes: list[str],
    allocation_temps: list[float],
) -> list[tuple[str, float]]:
    """Return unique allocation configs, dropping temp-invariant duplicates."""
    pairs: list[tuple[str, float]] = []
    seen: set[tuple[str, float]] = set()
    for raw_mode in allocation_modes:
        mode = str(raw_mode or "equal").strip().lower()
        temps = allocation_temps if mode == "softmax" else [1.0]
        for raw_temp in temps:
            temp = float(raw_temp)
            pair = (mode, temp)
            if pair in seen:
                continue
            seen.add(pair)
            pairs.append(pair)
    return pairs


def _key_float(value: object) -> float:
    return round(float(value), 12)


def _resolved_fill_buffer_for_key(fee_regime: str, fill_buffer_bps: float | None) -> float:
    fees = FEE_REGIMES[fee_regime]
    if fill_buffer_bps is None or float(fill_buffer_bps) < 0.0:
        return float(fees["fill_buffer_bps"])
    return float(fill_buffer_bps)


def _cell_key_from_mapping(row: dict) -> tuple:
    """Stable identity for a sweep cell, independent of score metrics."""
    fee_regime = str(row.get("fee_regime", ""))
    fill_buffer_bps = row.get("fill_buffer_bps")
    fill_buffer_key = (
        _resolved_fill_buffer_for_key(fee_regime, fill_buffer_bps)
        if fee_regime in FEE_REGIMES
        else (-1.0 if fill_buffer_bps is None else float(fill_buffer_bps))
    )
    return (
        _key_float(row.get("leverage", 0.0)),
        _key_float(row.get("min_score", 0.0)),
        bool(row.get("hold_through", False)),
        int(row.get("top_n", 0)),
        int(row.get("short_n", 0)),
        _key_float(row.get("max_short_score", 0.45)),
        _key_float(row.get("short_allocation_scale", 0.5)),
        int(row.get("min_picks", 0)),
        int(row.get("opportunistic_watch_n", 0)),
        _key_float(row.get("opportunistic_entry_discount_bps", 0.0)),
        _key_float(row.get("score_uncertainty_penalty", 0.0)),
        fee_regime,
        _key_float(row.get("inference_min_dolvol", 5_000_000.0)),
        _key_float(row.get("inference_max_spread_bps", 30.0)),
        _key_float(row.get("inference_min_vol_20d", 0.0)),
        _key_float(row.get("inference_max_vol_20d", 0.0)),
        _key_float(row.get("skip_prob", 0.0)),
        int(row.get("skip_seed", 0)),
        _key_float(fill_buffer_key),
        int(row.get("regime_gate_window", 0)),
        _key_float(row.get("vol_target_ann", 0.0)),
        _key_float(row.get("inv_vol_target_ann", 0.0)),
        _key_float(row.get("inv_vol_floor", 0.05)),
        _key_float(row.get("inv_vol_cap", 3.0)),
        _key_float(row.get("max_ret_20d_rank_pct", 1.0)),
        _key_float(row.get("min_ret_5d_rank_pct", 0.0)),
        _key_float(row.get("regime_cs_iqr_max", 0.0)),
        _key_float(row.get("regime_cs_skew_min", -1e9)),
        int(row.get("corr_window_days", 0)),
        int(row.get("corr_min_periods", 20)),
        _key_float(row.get("corr_max_signed", 1.0)),
        str(row.get("no_picks_fallback_symbol", "") or ""),
        _key_float(row.get("no_picks_fallback_alloc_scale", 0.0)),
        bool(row.get("conviction_scaled_alloc", False)),
        _key_float(row.get("conviction_alloc_low", 0.55)),
        _key_float(row.get("conviction_alloc_high", 0.85)),
        str(row.get("allocation_mode", "equal") or "equal"),
        _key_float(row.get("allocation_temp", 1.0)),
        _key_float(row.get("min_secondary_allocation", 0.0)),
    )


def _strategy_key_from_mapping(row: dict) -> tuple:
    """Stable identity for deployable strategy knobs, excluding stress axes."""
    out = []
    for param_name in STRATEGY_PARAM_FIELDS:
        default = 0 if param_name in {"min_picks", "min_secondary_allocation"} else None
        value = row.get(param_name, default)
        if isinstance(value, float):
            out.append(_key_float(value))
        else:
            out.append(value)
    return tuple(out)


def _cell_key_from_values(
    *,
    leverage: float,
    min_score: float,
    hold_through: bool,
    top_n: int,
    short_n: int,
    max_short_score: float,
    short_allocation_scale: float,
    min_picks: int,
    opportunistic_watch_n: int,
    opportunistic_entry_discount_bps: float,
    score_uncertainty_penalty: float,
    fee_regime: str,
    inference_min_dolvol: float,
    inference_max_spread_bps: float,
    inference_min_vol_20d: float,
    inference_max_vol_20d: float,
    skip_prob: float,
    skip_seed: int,
    fill_buffer_bps: float | None,
    regime_gate_window: int,
    vol_target_ann: float,
    inv_vol_target_ann: float,
    inv_vol_floor: float,
    inv_vol_cap: float,
    max_ret_20d_rank_pct: float,
    min_ret_5d_rank_pct: float,
    regime_cs_iqr_max: float,
    regime_cs_skew_min: float,
    corr_window_days: int,
    corr_min_periods: int,
    corr_max_signed: float,
    no_picks_fallback_symbol: str,
    no_picks_fallback_alloc_scale: float,
    conviction_scaled_alloc: bool,
    conviction_alloc_low: float,
    conviction_alloc_high: float,
    allocation_mode: str,
    allocation_temp: float,
    min_secondary_allocation: float,
) -> tuple:
    return _cell_key_from_mapping(
        {
            "leverage": leverage,
            "min_score": min_score,
            "hold_through": hold_through,
            "top_n": top_n,
            "short_n": short_n,
            "max_short_score": max_short_score,
            "short_allocation_scale": short_allocation_scale,
            "min_picks": min_picks,
            "opportunistic_watch_n": opportunistic_watch_n,
            "opportunistic_entry_discount_bps": opportunistic_entry_discount_bps,
            "score_uncertainty_penalty": score_uncertainty_penalty,
            "fee_regime": fee_regime,
            "inference_min_dolvol": inference_min_dolvol,
            "inference_max_spread_bps": inference_max_spread_bps,
            "inference_min_vol_20d": inference_min_vol_20d,
            "inference_max_vol_20d": inference_max_vol_20d,
            "skip_prob": skip_prob,
            "skip_seed": skip_seed,
            "fill_buffer_bps": _resolved_fill_buffer_for_key(fee_regime, fill_buffer_bps),
            "regime_gate_window": regime_gate_window,
            "vol_target_ann": vol_target_ann,
            "inv_vol_target_ann": inv_vol_target_ann,
            "inv_vol_floor": inv_vol_floor,
            "inv_vol_cap": inv_vol_cap,
            "max_ret_20d_rank_pct": max_ret_20d_rank_pct,
            "min_ret_5d_rank_pct": min_ret_5d_rank_pct,
            "regime_cs_iqr_max": regime_cs_iqr_max,
            "regime_cs_skew_min": regime_cs_skew_min,
            "corr_window_days": corr_window_days,
            "corr_min_periods": corr_min_periods,
            "corr_max_signed": corr_max_signed,
            "no_picks_fallback_symbol": no_picks_fallback_symbol,
            "no_picks_fallback_alloc_scale": no_picks_fallback_alloc_scale,
            "conviction_scaled_alloc": conviction_scaled_alloc,
            "conviction_alloc_low": conviction_alloc_low,
            "conviction_alloc_high": conviction_alloc_high,
            "allocation_mode": allocation_mode,
            "allocation_temp": allocation_temp,
            "min_secondary_allocation": min_secondary_allocation,
        }
    )


def _cell_from_row(row: dict) -> CellResult:
    """Rehydrate a checkpoint row into the current CellResult shape."""
    kwargs = {}
    for field_name, field_def in CellResult.__dataclass_fields__.items():
        if field_name in row:
            kwargs[field_name] = row[field_name]
        elif field_def.default is not MISSING:
            kwargs[field_name] = field_def.default
    cell = CellResult(**kwargs)
    if "fill_buffer_bps" not in row and cell.fee_regime in FEE_REGIMES:
        cell.fill_buffer_bps = float(FEE_REGIMES[cell.fee_regime]["fill_buffer_bps"])
    return cell


def _resume_cells_from_rows(rows: list[dict]) -> dict[tuple, CellResult]:
    resumed: dict[tuple, CellResult] = {}
    for row in rows:
        resumed[_cell_key_from_mapping(row)] = _cell_from_row(row)
    return resumed


def _run_cell(
    oos_df: pd.DataFrame,
    scores: pd.Series,
    windows: list[tuple],
    leverage: float,
    min_score: float,
    hold_through: bool,
    top_n: int,
    fee_regime: str,
    short_n: int = 0,
    max_short_score: float = 0.45,
    short_allocation_scale: float = 0.5,
    min_picks: int = 0,
    opportunistic_watch_n: int = 0,
    opportunistic_entry_discount_bps: float = 0.0,
    score_uncertainty_penalty: float = 0.0,
    inference_min_dolvol: float = 5_000_000.0,
    inference_max_spread_bps: float = 30.0,
    inference_min_vol_20d: float = 0.0,
    inference_max_vol_20d: float = 0.0,
    skip_prob: float = 0.0,
    skip_seed: int = 0,
    fill_buffer_bps: float | None = None,
    regime_gate_window: int = 0,
    vol_target_ann: float = 0.0,
    spy_close_by_date: pd.Series | None = None,
    inv_vol_target_ann: float = 0.0,
    inv_vol_floor: float = 0.05,
    inv_vol_cap: float = 3.0,
    max_ret_20d_rank_pct: float = 1.0,
    min_ret_5d_rank_pct: float = 0.0,
    regime_cs_iqr_max: float = 0.0,
    regime_cs_skew_min: float = -1e9,
    corr_window_days: int = 0,
    corr_min_periods: int = 20,
    corr_max_signed: float = 1.0,
    no_picks_fallback_symbol: str = "",
    no_picks_fallback_alloc_scale: float = 0.0,
    conviction_scaled_alloc: bool = False,
    conviction_alloc_low: float = 0.55,
    conviction_alloc_high: float = 0.85,
    allocation_mode: str = "equal",
    allocation_temp: float = 1.0,
    min_secondary_allocation: float = 0.0,
    fail_fast_max_dd_pct: float = 0.0,
    fail_fast_max_intraday_dd_pct: float = 0.0,
    fail_fast_neg_windows: int = 0,
    overnight_max_gross_leverage: float | None = None,
) -> CellResult:
    fees = FEE_REGIMES[fee_regime]
    # Override fill_buffer if the caller explicitly set it; otherwise
    # fall through to the regime default so legacy behaviour is stable.
    fb_resolved = (
        float(fees["fill_buffer_bps"])
        if fill_buffer_bps is None or float(fill_buffer_bps) < 0.0
        else float(fill_buffer_bps)
    )
    cfg = BacktestConfig(
        top_n=int(top_n),
        short_n=int(short_n),
        max_short_score=float(max_short_score),
        short_allocation_scale=float(short_allocation_scale),
        min_picks=int(min_picks),
        opportunistic_watch_n=int(opportunistic_watch_n),
        opportunistic_entry_discount_bps=float(opportunistic_entry_discount_bps),
        leverage=float(leverage),
        xgb_weight=1.0,
        fee_rate=float(fees["fee_rate"]),
        fill_buffer_bps=fb_resolved,
        commission_bps=float(fees["commission_bps"]),
        min_dollar_vol=float(inference_min_dolvol),
        max_spread_bps=float(inference_max_spread_bps),
        min_vol_20d=float(inference_min_vol_20d),
        max_vol_20d=float(inference_max_vol_20d),
        hold_through=bool(hold_through),
        min_score=float(min_score),
        skip_prob=float(skip_prob),
        skip_seed=int(skip_seed),
        regime_gate_window=int(regime_gate_window),
        vol_target_ann=float(vol_target_ann),
        inv_vol_target_ann=float(inv_vol_target_ann),
        inv_vol_floor=float(inv_vol_floor),
        inv_vol_cap=float(inv_vol_cap),
        max_ret_20d_rank_pct=float(max_ret_20d_rank_pct),
        min_ret_5d_rank_pct=float(min_ret_5d_rank_pct),
        regime_cs_iqr_max=float(regime_cs_iqr_max),
        regime_cs_skew_min=float(regime_cs_skew_min),
        corr_window_days=int(corr_window_days),
        corr_min_periods=int(corr_min_periods),
        corr_max_signed=float(corr_max_signed),
        no_picks_fallback_symbol=str(no_picks_fallback_symbol or ""),
        no_picks_fallback_alloc_scale=float(no_picks_fallback_alloc_scale),
        conviction_scaled_alloc=bool(conviction_scaled_alloc),
        conviction_alloc_low=float(conviction_alloc_low),
        conviction_alloc_high=float(conviction_alloc_high),
        allocation_mode=str(allocation_mode or "equal"),
        allocation_temp=float(allocation_temp),
        min_secondary_allocation=float(min_secondary_allocation),
        stop_on_drawdown_pct=max(float(fail_fast_max_dd_pct), 0.0),
        stop_on_intraday_drawdown_pct=max(float(fail_fast_max_intraday_dd_pct), 0.0),
        overnight_max_gross_leverage=(
            None if overnight_max_gross_leverage is None
            else float(overnight_max_gross_leverage)
        ),
    )

    monthlies: list[float] = []
    sortinos: list[float] = []
    dds: list[float] = []
    intra_dds_worst: list[float] = []
    intra_dds_avg:   list[float] = []
    tuws: list[float] = []
    ulcers: list[float] = []
    active_day_pcts: list[float] = []
    window_start_dates: list[str] = []
    window_end_dates: list[str] = []
    fail_fast_reason = ""
    for w_start, w_end in windows:
        w_df = oos_df[(oos_df["date"] >= w_start) & (oos_df["date"] <= w_end)]
        if len(w_df) < 5:
            continue
        w_scores = scores.loc[w_df.index]
        res = simulate(
            w_df, None, cfg,  # type: ignore[arg-type]
            precomputed_scores=w_scores,
            spy_close_by_date=spy_close_by_date,
        )
        elapsed_days = max(_elapsed_window_days(w_df, res), 1)
        active_day_pcts.append(
            float(len(res.day_results)) / float(elapsed_days) * 100.0
        )
        monthly = _monthly_return(res.total_return_pct, elapsed_days) * 100.0
        monthlies.append(monthly)
        sortinos.append(res.sortino_ratio)
        dds.append(res.max_drawdown_pct)
        intra_dds_worst.append(res.worst_intraday_dd_pct)
        intra_dds_avg.append(res.avg_intraday_dd_pct)
        tuws.append(res.time_under_water_pct)
        ulcers.append(res.ulcer_index)
        window_start_dates.append(pd.Timestamp(w_start).date().isoformat())
        window_end_dates.append(pd.Timestamp(w_end).date().isoformat())
        if fail_fast_max_dd_pct > 0.0 and res.max_drawdown_pct >= fail_fast_max_dd_pct:
            fail_fast_reason = f"max_dd_pct>={fail_fast_max_dd_pct:g}"
            break
        if (
            fail_fast_max_intraday_dd_pct > 0.0
            and res.worst_intraday_dd_pct >= fail_fast_max_intraday_dd_pct
        ):
            fail_fast_reason = (
                f"intraday_dd_pct>={fail_fast_max_intraday_dd_pct:g}"
            )
            break
        if fail_fast_neg_windows > 0 and sum(m < 0.0 for m in monthlies) >= fail_fast_neg_windows:
            fail_fast_reason = f"neg_windows>={fail_fast_neg_windows:d}"
            break

    n = len(monthlies)
    if n == 0:
        empty = CellResult(
            leverage=leverage,
            min_score=min_score,
            hold_through=hold_through,
            top_n=top_n,
            short_n=int(short_n),
            max_short_score=float(max_short_score),
            short_allocation_scale=float(short_allocation_scale),
            fee_regime=fee_regime,
            n_windows=0,
            median_monthly_pct=0.0,
            p10_monthly_pct=0.0,
            median_sortino=0.0,
            worst_dd_pct=0.0,
            n_neg=0,
        )
        empty.min_picks = int(min_picks)
        empty.short_n = int(short_n)
        empty.max_short_score = float(max_short_score)
        empty.short_allocation_scale = float(short_allocation_scale)
        empty.opportunistic_watch_n = int(opportunistic_watch_n)
        empty.opportunistic_entry_discount_bps = float(opportunistic_entry_discount_bps)
        empty.score_uncertainty_penalty = float(score_uncertainty_penalty)
        empty.inference_min_dolvol = float(inference_min_dolvol)
        empty.inference_max_spread_bps = float(inference_max_spread_bps)
        empty.inference_min_vol_20d = float(inference_min_vol_20d)
        empty.inference_max_vol_20d = float(inference_max_vol_20d)
        empty.fill_buffer_bps = fb_resolved
        empty.allocation_mode = str(allocation_mode or "equal")
        empty.allocation_temp = float(allocation_temp)
        empty.min_secondary_allocation = float(min_secondary_allocation)
        empty.corr_window_days = int(corr_window_days)
        empty.corr_min_periods = int(corr_min_periods)
        empty.corr_max_signed = float(corr_max_signed)
        empty.expected_n_windows = int(len(windows))
        return empty

    arr = np.array(monthlies)
    p10 = float(np.percentile(arr, 10))
    worst_dd = float(np.max(dds))
    n_neg = int(np.sum(arr < 0))
    mean_abs_neg = float(-arr[arr < 0].sum()) / n if n_neg else 0.0
    goodness = compute_goodness(p10, worst_dd, n_neg, n)
    robust_goodness = compute_robust_goodness(arr, worst_dd)
    worst_intra = float(np.max(intra_dds_worst)) if intra_dds_worst else 0.0
    avg_intra   = float(np.mean(intra_dds_avg))  if intra_dds_avg   else 0.0
    tuw_med     = float(np.median(tuws))    if tuws    else 0.0
    ulcer_med   = float(np.median(ulcers))  if ulcers  else 0.0
    active_med  = float(np.median(active_day_pcts)) if active_day_pcts else 0.0
    active_min  = float(np.min(active_day_pcts)) if active_day_pcts else 0.0
    pain_adjusted_goodness = compute_pain_adjusted_goodness(
        arr,
        worst_dd,
        tuw_med,
        ulcer_med,
    )
    if fail_fast_reason:
        goodness = FAIL_FAST_SCORE
        robust_goodness = FAIL_FAST_SCORE
        pain_adjusted_goodness = FAIL_FAST_SCORE
    return CellResult(
        leverage=leverage,
        min_score=min_score,
        hold_through=hold_through,
        top_n=top_n,
        short_n=int(short_n),
        max_short_score=float(max_short_score),
        short_allocation_scale=float(short_allocation_scale),
        fee_regime=fee_regime,
        n_windows=n,
        median_monthly_pct=float(np.median(arr)),
        p10_monthly_pct=p10,
        median_sortino=float(np.median(sortinos)),
        worst_dd_pct=worst_dd,
        n_neg=n_neg,
        mean_abs_neg_monthly_pct=mean_abs_neg,
        monthly_return_pcts=[float(value) for value in arr],
        window_sortino_values=[float(value) for value in sortinos],
        window_drawdown_pcts=[float(value) for value in dds],
        window_time_under_water_pcts=[float(value) for value in tuws],
        window_ulcer_indexes=[float(value) for value in ulcers],
        window_active_day_pcts=[float(value) for value in active_day_pcts],
        window_worst_intraday_dd_pcts=[float(value) for value in intra_dds_worst],
        window_avg_intraday_dd_pcts=[float(value) for value in intra_dds_avg],
        window_start_dates=window_start_dates,
        window_end_dates=window_end_dates,
        min_picks=int(min_picks),
        opportunistic_watch_n=int(opportunistic_watch_n),
        opportunistic_entry_discount_bps=float(opportunistic_entry_discount_bps),
        score_uncertainty_penalty=float(score_uncertainty_penalty),
        goodness_score=goodness,
        robust_goodness_score=robust_goodness,
        pain_adjusted_goodness_score=pain_adjusted_goodness,
        fail_fast_triggered=bool(fail_fast_reason),
        fail_fast_reason=fail_fast_reason,
        worst_intraday_dd_pct=worst_intra,
        avg_intraday_dd_pct=avg_intra,
        time_under_water_pct=tuw_med,
        ulcer_index=ulcer_med,
        median_active_day_pct=active_med,
        min_active_day_pct=active_min,
        inference_min_dolvol=float(inference_min_dolvol),
        inference_max_spread_bps=float(inference_max_spread_bps),
        inference_min_vol_20d=float(inference_min_vol_20d),
        inference_max_vol_20d=float(inference_max_vol_20d),
        skip_prob=float(skip_prob),
        skip_seed=int(skip_seed),
        fill_buffer_bps=fb_resolved,
        regime_gate_window=int(regime_gate_window),
        vol_target_ann=float(vol_target_ann),
        inv_vol_target_ann=float(inv_vol_target_ann),
        inv_vol_floor=float(inv_vol_floor),
        inv_vol_cap=float(inv_vol_cap),
        max_ret_20d_rank_pct=float(max_ret_20d_rank_pct),
        min_ret_5d_rank_pct=float(min_ret_5d_rank_pct),
        regime_cs_iqr_max=float(regime_cs_iqr_max),
        regime_cs_skew_min=float(regime_cs_skew_min),
        corr_window_days=int(corr_window_days),
        corr_min_periods=int(corr_min_periods),
        corr_max_signed=float(corr_max_signed),
        no_picks_fallback_symbol=str(no_picks_fallback_symbol or ""),
        no_picks_fallback_alloc_scale=float(no_picks_fallback_alloc_scale),
        conviction_scaled_alloc=bool(conviction_scaled_alloc),
        conviction_alloc_low=float(conviction_alloc_low),
        conviction_alloc_high=float(conviction_alloc_high),
        allocation_mode=str(allocation_mode or "equal"),
        allocation_temp=float(allocation_temp),
        min_secondary_allocation=float(min_secondary_allocation),
        expected_n_windows=int(len(windows)),
    )


def _alltrain_seed_from_path(path: Path) -> int | None:
    prefix = "alltrain_seed"
    stem = path.stem
    if not stem.startswith(prefix):
        return None
    seed_text = stem[len(prefix):]
    try:
        return int(seed_text)
    except ValueError as exc:
        raise ValueError(f"{path}: filename must match alltrain_seed<seed>.pkl") from exc


def _validate_model_paths_for_sweep(model_paths: list[Path]) -> None:
    if not model_paths:
        raise ValueError("model path list is empty")
    normalized_paths = [path.expanduser().resolve(strict=False) for path in model_paths]
    if len(set(normalized_paths)) != len(normalized_paths):
        raise ValueError("model path list contains duplicates")
    for path in model_paths:
        _alltrain_seed_from_path(path)


def _float_grid(name: str, values: list[float] | None, default: list[float]) -> list[float]:
    try:
        vals = [float(x) for x in (values or default)]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} values must be numeric") from exc
    if any(not np.isfinite(x) for x in vals):
        raise ValueError(f"{name} values must be finite")
    return vals


def _int_grid(name: str, values: list[int] | None, default: list[int]) -> list[int]:
    vals: list[int] = []
    for raw in values or default:
        if isinstance(raw, bool):
            raise ValueError(f"{name} values must be integer-like")
        try:
            num = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} values must be integer-like") from exc
        if not np.isfinite(num) or not num.is_integer():
            raise ValueError(f"{name} values must be integer-like")
        vals.append(int(num))
    return vals


def _validate_sweep_grid_domains(
    *,
    window_days: int,
    stride_days: int,
    leverage_grid: list[float],
    min_score_grid: list[float],
    top_n_grid: list[int],
    short_n_grid: list[int] | None,
    max_short_score_grid: list[float] | None,
    short_allocation_scale_grid: list[float] | None,
    min_picks_grid: list[int] | None,
    opportunistic_watch_n_grid: list[int] | None,
    opportunistic_entry_discount_bps_grid: list[float] | None,
    min_dollar_vol: float,
    inference_min_dolvol_grid: list[float] | None,
    inference_max_spread_bps_grid: list[float] | None,
    inference_min_vol_grid: list[float] | None,
    inference_max_vol_grid: list[float] | None,
    skip_prob_grid: list[float] | None,
    fill_buffer_bps_grid: list[float] | None,
    regime_gate_window_grid: list[int] | None,
    vol_target_ann_grid: list[float] | None,
    inv_vol_target_grid: list[float] | None,
    max_ret_20d_rank_pct_grid: list[float] | None,
    min_ret_5d_rank_pct_grid: list[float] | None,
    regime_cs_iqr_max_grid: list[float] | None,
    regime_cs_skew_min_grid: list[float] | None,
    no_picks_fallback_symbol: str,
    no_picks_fallback_alloc_grid: list[float] | None,
    inv_vol_floor: float,
    inv_vol_cap: float,
    conviction_alloc_low: float,
    conviction_alloc_high: float,
    allocation_temp_grid: list[float] | None,
    min_secondary_allocation_grid: list[float] | None,
    score_uncertainty_penalty_grid: list[float] | None,
    fail_fast_max_dd_pct: float,
    fail_fast_max_intraday_dd_pct: float,
    fail_fast_neg_windows: int,
) -> None:
    window_count = _int_grid("window_days", [window_days], [30])[0]
    if window_count < 1:
        raise ValueError("window_days must be >= 1")
    stride_count = _int_grid("stride_days", [stride_days], [7])[0]
    if stride_count < 1:
        raise ValueError("stride_days must be >= 1")

    lev = _float_grid("leverage_grid", leverage_grid, [1.0])
    if any(x <= 0.0 for x in lev):
        raise ValueError("leverage_grid values must be > 0")
    min_scores = _float_grid("min_score_grid", min_score_grid, [0.0])
    if any(x < 0.0 or x > 1.0 for x in min_scores):
        raise ValueError("min_score_grid values must be between 0 and 1")
    topn = _int_grid("top_n_grid", top_n_grid, [1])
    if any(x < 1 for x in topn):
        raise ValueError("top_n_grid values must be >= 1")
    shortn = _int_grid("short_n_grid", short_n_grid, [0])
    if any(x < 0 for x in shortn):
        raise ValueError("short_n_grid values must be >= 0")
    max_short_scores = _float_grid("max_short_score_grid", max_short_score_grid, [0.45])
    if any(x < 0.0 or x > 1.0 for x in max_short_scores):
        raise ValueError("max_short_score_grid values must be between 0 and 1")
    short_alloc_scales = _float_grid("short_allocation_scale_grid", short_allocation_scale_grid, [0.5])
    if any(x < 0.0 for x in short_alloc_scales):
        raise ValueError("short_allocation_scale_grid values must be >= 0")
    minp = _int_grid("min_picks_grid", min_picks_grid, [0])
    if any(x < 0 for x in minp):
        raise ValueError("min_picks_grid values must be >= 0")
    if any(min_picks > top_n for min_picks, top_n in product(minp, topn)):
        raise ValueError("min_picks_grid values must be <= top_n_grid values")
    opp_watch = _int_grid("opportunistic_watch_n_grid", opportunistic_watch_n_grid, [0])
    if any(x < 0 for x in opp_watch):
        raise ValueError("opportunistic_watch_n_grid values must be >= 0")
    if any(w > 0 and w < top_n + short_n for w, top_n, short_n in product(opp_watch, topn, shortn)):
        raise ValueError("positive opportunistic_watch_n_grid values must be >= top_n_grid + short_n_grid values")

    base_min_dollar_vol = float(min_dollar_vol)
    if not np.isfinite(base_min_dollar_vol) or base_min_dollar_vol < 0.0:
        raise ValueError("min_dollar_vol must be finite and >= 0")
    nonnegative_grids = {
        "inference_min_dolvol_grid": _float_grid(
            "inference_min_dolvol_grid",
            inference_min_dolvol_grid,
            [base_min_dollar_vol],
        ),
        "inference_max_spread_bps_grid": _float_grid(
            "inference_max_spread_bps_grid",
            inference_max_spread_bps_grid,
            [30.0],
        ),
        "inference_min_vol_grid": _float_grid("inference_min_vol_grid", inference_min_vol_grid, [0.0]),
        "inference_max_vol_grid": _float_grid("inference_max_vol_grid", inference_max_vol_grid, [0.0]),
        "vol_target_ann_grid": _float_grid("vol_target_ann_grid", vol_target_ann_grid, [0.0]),
        "inv_vol_target_grid": _float_grid("inv_vol_target_grid", inv_vol_target_grid, [0.0]),
        "regime_cs_iqr_max_grid": _float_grid("regime_cs_iqr_max_grid", regime_cs_iqr_max_grid, [0.0]),
        "allocation_temp_grid": _float_grid("allocation_temp_grid", allocation_temp_grid, [1.0]),
        "min_secondary_allocation_grid": _float_grid(
            "min_secondary_allocation_grid",
            min_secondary_allocation_grid,
            [0.0],
        ),
        "score_uncertainty_penalty_grid": _float_grid(
            "score_uncertainty_penalty_grid",
            score_uncertainty_penalty_grid,
            [0.0],
        ),
        "opportunistic_entry_discount_bps_grid": _float_grid(
            "opportunistic_entry_discount_bps_grid",
            opportunistic_entry_discount_bps_grid,
            [0.0],
        ),
    }
    fb_sym = str(no_picks_fallback_symbol or "").strip()
    if fb_sym:
        nonnegative_grids["no_picks_fallback_alloc_grid"] = _float_grid(
            "no_picks_fallback_alloc_grid",
            no_picks_fallback_alloc_grid,
            [0.0],
        )
    for name, vals in nonnegative_grids.items():
        if any(x < 0.0 for x in vals):
            raise ValueError(f"{name} values must be >= 0")
    if any(x <= 0.0 for x in nonnegative_grids["allocation_temp_grid"]):
        raise ValueError("allocation_temp_grid values must be > 0")
    if any(x > 1.0 for x in nonnegative_grids["min_secondary_allocation_grid"]):
        raise ValueError("min_secondary_allocation_grid values must be between 0 and 1")

    skip_probs = _float_grid("skip_prob_grid", skip_prob_grid, [0.0])
    if any(x < 0.0 or x > 1.0 for x in skip_probs):
        raise ValueError("skip_prob_grid values must be between 0 and 1")
    fill_buffers = _float_grid("fill_buffer_bps_grid", fill_buffer_bps_grid, [-1.0])
    if any(x != -1.0 and x < 0.0 for x in fill_buffers):
        raise ValueError("fill_buffer_bps_grid values must be -1 or >= 0")
    rgw = _int_grid("regime_gate_window_grid", regime_gate_window_grid, [0])
    if any(x < 0 for x in rgw):
        raise ValueError("regime_gate_window_grid values must be >= 0")
    for name, vals in {
        "max_ret_20d_rank_pct_grid": _float_grid(
            "max_ret_20d_rank_pct_grid",
            max_ret_20d_rank_pct_grid,
            [1.0],
        ),
        "min_ret_5d_rank_pct_grid": _float_grid(
            "min_ret_5d_rank_pct_grid",
            min_ret_5d_rank_pct_grid,
            [0.0],
        ),
    }.items():
        if any(x < 0.0 or x > 1.0 for x in vals):
            raise ValueError(f"{name} values must be between 0 and 1")
    _float_grid("regime_cs_skew_min_grid", regime_cs_skew_min_grid, [-1e9])

    inv_floor = float(inv_vol_floor)
    inv_cap = float(inv_vol_cap)
    if not np.isfinite(inv_floor) or inv_floor <= 0.0:
        raise ValueError("inv_vol_floor must be finite and > 0")
    if not np.isfinite(inv_cap) or inv_cap < 1.0:
        raise ValueError("inv_vol_cap must be finite and >= 1")

    lo = float(conviction_alloc_low)
    hi = float(conviction_alloc_high)
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("conviction allocation bounds must be finite")
    if hi <= lo:
        raise ValueError("conviction_alloc_high must be > conviction_alloc_low")
    for name, val in {
        "fail_fast_max_dd_pct": float(fail_fast_max_dd_pct),
        "fail_fast_max_intraday_dd_pct": float(fail_fast_max_intraday_dd_pct),
    }.items():
        if not np.isfinite(val) or val < 0.0:
            raise ValueError(f"{name} must be finite and >= 0")
    ff_neg = _int_grid("fail_fast_neg_windows", [fail_fast_neg_windows], [0])[0]
    if ff_neg < 0:
        raise ValueError("fail_fast_neg_windows must be >= 0")


def run_sweep(
    *,
    symbols: list[str],
    data_root: Path,
    model_paths: list[Path],
    train_start: date,
    train_end: date,
    oos_start: date,
    oos_end: date,
    window_days: int,
    stride_days: int,
    leverage_grid: list[float],
    min_score_grid: list[float],
    hold_through_grid: list[bool],
    top_n_grid: list[int],
    fee_regimes: list[str],
    short_n_grid: list[int] | None = None,
    max_short_score_grid: list[float] | None = None,
    short_allocation_scale_grid: list[float] | None = None,
    min_picks_grid: list[int] | None = None,
    opportunistic_watch_n_grid: list[int] | None = None,
    opportunistic_entry_discount_bps_grid: list[float] | None = None,
    blend_mode: str = "mean",
    chronos_cache_path: Path | None = None,
    min_dollar_vol: float = 5_000_000.0,
    inference_min_dolvol_grid: list[float] | None = None,
    inference_max_spread_bps_grid: list[float] | None = None,
    inference_min_vol_grid: list[float] | None = None,
    inference_max_vol_grid: list[float] | None = None,
    skip_prob_grid: list[float] | None = None,
    skip_seeds: list[int] | None = None,
    fill_buffer_bps_grid: list[float] | None = None,
    regime_gate_window_grid: list[int] | None = None,
    vol_target_ann_grid: list[float] | None = None,
    spy_csv_path: Path | None = None,
    inv_vol_target_grid: list[float] | None = None,
    inv_vol_floor: float = 0.05,
    inv_vol_cap: float = 3.0,
    invert_scores: bool = False,
    max_ret_20d_rank_pct_grid: list[float] | None = None,
    min_ret_5d_rank_pct_grid: list[float] | None = None,
    regime_cs_iqr_max_grid: list[float] | None = None,
    regime_cs_skew_min_grid: list[float] | None = None,
    corr_window_days_grid: list[int] | None = None,
    corr_min_periods: int = 20,
    corr_max_signed_grid: list[float] | None = None,
    no_picks_fallback_symbol: str = "",
    no_picks_fallback_alloc_grid: list[float] | None = None,
    conviction_scaled_alloc_grid: list[bool] | None = None,
    conviction_alloc_low: float = 0.55,
    conviction_alloc_high: float = 0.85,
    allocation_mode_grid: list[str] | None = None,
    allocation_temp_grid: list[float] | None = None,
    min_secondary_allocation_grid: list[float] | None = None,
    score_uncertainty_penalty_grid: list[float] | None = None,
    fail_fast_max_dd_pct: float = 0.0,
    fail_fast_max_intraday_dd_pct: float = 0.0,
    fail_fast_neg_windows: int = 0,
    fast_features: bool = False,
    overnight_max_gross_leverage: float | None = None,
    progress_callback: Callable[[list[CellResult], int, int], None] | None = None,
    resume_rows: list[dict] | None = None,
    fm_latents_path: Path | None = None,
    fm_n_latents: int | None = None,
) -> list[CellResult]:
    """Run the full sweep. Returns a flat list of CellResult."""
    _validate_model_paths_for_sweep(model_paths)
    for p in model_paths:
        if not p.exists():
            raise FileNotFoundError(f"model path not found: {p}")
    for reg in fee_regimes:
        if reg not in FEE_REGIMES:
            raise ValueError(f"unknown fee regime: {reg}. "
                             f"Known: {list(FEE_REGIMES)}")
    _validate_sweep_grid_domains(
        window_days=window_days,
        stride_days=stride_days,
        leverage_grid=leverage_grid,
        min_score_grid=min_score_grid,
        top_n_grid=top_n_grid,
        short_n_grid=short_n_grid,
        max_short_score_grid=max_short_score_grid,
        short_allocation_scale_grid=short_allocation_scale_grid,
        min_picks_grid=min_picks_grid,
        opportunistic_watch_n_grid=opportunistic_watch_n_grid,
        opportunistic_entry_discount_bps_grid=opportunistic_entry_discount_bps_grid,
        min_dollar_vol=min_dollar_vol,
        inference_min_dolvol_grid=inference_min_dolvol_grid,
        inference_max_spread_bps_grid=inference_max_spread_bps_grid,
        inference_min_vol_grid=inference_min_vol_grid,
        inference_max_vol_grid=inference_max_vol_grid,
        skip_prob_grid=skip_prob_grid,
        fill_buffer_bps_grid=fill_buffer_bps_grid,
        regime_gate_window_grid=regime_gate_window_grid,
        vol_target_ann_grid=vol_target_ann_grid,
        inv_vol_target_grid=inv_vol_target_grid,
        max_ret_20d_rank_pct_grid=max_ret_20d_rank_pct_grid,
        min_ret_5d_rank_pct_grid=min_ret_5d_rank_pct_grid,
        regime_cs_iqr_max_grid=regime_cs_iqr_max_grid,
        regime_cs_skew_min_grid=regime_cs_skew_min_grid,
        no_picks_fallback_symbol=no_picks_fallback_symbol,
        no_picks_fallback_alloc_grid=no_picks_fallback_alloc_grid,
        inv_vol_floor=inv_vol_floor,
        inv_vol_cap=inv_vol_cap,
        conviction_alloc_low=conviction_alloc_low,
        conviction_alloc_high=conviction_alloc_high,
        allocation_temp_grid=allocation_temp_grid,
        min_secondary_allocation_grid=min_secondary_allocation_grid,
        score_uncertainty_penalty_grid=score_uncertainty_penalty_grid,
        fail_fast_max_dd_pct=fail_fast_max_dd_pct,
        fail_fast_max_intraday_dd_pct=fail_fast_max_intraday_dd_pct,
        fail_fast_neg_windows=fail_fast_neg_windows,
    )

    chronos_cache = {}
    if chronos_cache_path is not None and chronos_cache_path.exists():
        chronos_cache = load_chronos_cache(chronos_cache_path)

    fm_latents_df = None
    if fm_latents_path is not None:
        fm_latents_path = Path(fm_latents_path)
        if not fm_latents_path.exists():
            raise ValueError(f"--fm-latents-path not found: {fm_latents_path}")
        fm_latents_df = load_fm_latents(fm_latents_path)
        if fm_latents_df is not None:
            logger.info(
                "fm_latents loaded: rows=%d unique_syms=%d unique_dates=%d",
                len(fm_latents_df),
                fm_latents_df["symbol"].nunique(),
                fm_latents_df["date"].nunique(),
            )

    # Load models first so we can peek at their feature_cols and decide
    # whether the dataset needs cross-sectional ranks attached. Any family
    # registered in xgbnew.model_registry (xgb/lgb/cat/mlp/...) is accepted;
    # legacy pickles without a "family" key still dispatch to XGBStockModel.
    models = []
    for p in model_paths:
        logger.info("loading %s", p)
        models.append(load_any_model(p))

    first_features: tuple[str, ...] | None = None
    first_path: Path | None = None
    for model, path in zip(models, model_paths, strict=True):
        raw_features = getattr(model, "feature_cols", None)
        if (
            not isinstance(raw_features, (list, tuple))
            or not raw_features
            or not all(isinstance(col, str) and col for col in raw_features)
        ):
            raise ValueError(f"{path}: model feature_cols must be a non-empty list")
        features = tuple(raw_features)
        unsupported = sorted(set(features) - LIVE_SUPPORTED_FEATURE_COLS)
        if unsupported:
            raise ValueError(
                f"{path}: model feature_cols contains unsupported live features: "
                f"{unsupported}"
            )
        if first_features is None:
            first_features = features
            first_path = path
        elif features != first_features:
            raise ValueError(
                "Ensemble feature_cols mismatch: "
                f"{path} has {len(features)} features but {first_path} has "
                f"{len(first_features)}"
            )

    assert first_features is not None
    assert first_path is not None
    needs_ranks = any(c in first_features for c in DAILY_RANK_FEATURE_COLS)
    needs_disp = any(c in first_features for c in DAILY_DISPERSION_FEATURE_COLS)
    fm_latent_cols_in_model = [c for c in first_features if c.startswith("latent_")]
    needs_fm = bool(fm_latent_cols_in_model) or "fm_available" in first_features
    fm_n_latents_inferred = _infer_required_fm_latents(first_features, first_path)
    if fm_n_latents is not None and fm_n_latents_inferred is not None and fm_n_latents < fm_n_latents_inferred:
        raise ValueError(
            f"--fm-n-latents={fm_n_latents} is smaller than model feature_cols require "
            f"({fm_n_latents_inferred})"
        )
    if needs_fm and fm_latents_df is None:
        raise ValueError(
            "Ensemble feature_cols include foundation-model latents "
            f"({fm_latent_cols_in_model[:3]}…) but --fm-latents-path was not provided."
        )
    logger.info(
        "ensemble feature-mode: ranks=%s disp=%s fm_latents=%s (n_lat=%s)",
        needs_ranks, needs_disp, needs_fm,
        fm_n_latents_inferred if needs_fm else 0,
    )

    _t = time.perf_counter()
    train_df, _, oos_df = build_daily_dataset(
        data_root=data_root,
        symbols=symbols,
        train_start=train_start,
        train_end=train_end,
        val_start=oos_start, val_end=oos_end,
        test_start=oos_start, test_end=oos_end,
        chronos_cache=chronos_cache if chronos_cache else None,
        min_dollar_vol=min_dollar_vol,
        fast_features=bool(fast_features),
        include_cross_sectional_ranks=needs_ranks,
        include_cross_sectional_dispersion=needs_disp,
        fm_latents=fm_latents_df if needs_fm else None,
        fm_n_latents=(
            fm_n_latents
            if fm_n_latents is not None
            else fm_n_latents_inferred
        ),
    )
    logger.info("dataset built in %.1fs | train=%d oos=%d",
                time.perf_counter() - _t, len(train_df), len(oos_df))

    scores, score_std = _ensemble_score_mean_std(oos_df, models, blend_mode)
    if invert_scores:
        # Flip rank-order so "top-N" becomes the worst-scored names. Kept
        # in [0,1] so existing min_score gates stay meaningful on the
        # inverted distribution (callers should LOWER the gate).
        logger.info("invert_scores=True → flipping score ranks for short-side test")
        scores = (1.0 - scores).rename("ensemble_score_inv")
        score_std = score_std.rename("ensemble_score_inv_std")

    all_days = sorted(oos_df["date"].unique())
    windows = _build_windows(all_days, window_days, stride_days)
    if not windows:
        raise RuntimeError("no eval windows — check OOS date range")

    inf_grid = list(inference_min_dolvol_grid) if inference_min_dolvol_grid else [float(min_dollar_vol)]
    spread_grid = (
        list(inference_max_spread_bps_grid)
        if inference_max_spread_bps_grid else [30.0]
    )
    vol_grid = list(inference_min_vol_grid) if inference_min_vol_grid else [0.0]
    maxvol_grid = list(inference_max_vol_grid) if inference_max_vol_grid else [0.0]
    sp_grid  = [float(x) for x in (skip_prob_grid or [0.0])]
    ss_list  = [int(x) for x in (skip_seeds or [0])]
    # -1.0 sentinel = "use regime default fill_buffer_bps".
    fb_grid  = [float(x) for x in (fill_buffer_bps_grid or [-1.0])]
    opp_watch_grid = [int(x) for x in (opportunistic_watch_n_grid or [0])]
    opp_discount_grid = [
        float(x) for x in (opportunistic_entry_discount_bps_grid or [0.0])
    ]
    shortn_grid = [int(x) for x in (short_n_grid or [0])]
    max_short_score_grid = [float(x) for x in (max_short_score_grid or [0.45])]
    short_alloc_scale_grid = [float(x) for x in (short_allocation_scale_grid or [0.5])]
    rgw_grid = [int(x) for x in (regime_gate_window_grid or [0])]
    vta_grid = [float(x) for x in (vol_target_ann_grid or [0.0])]
    ivt_grid = [float(x) for x in (inv_vol_target_grid or [0.0])]
    r20g_grid = [float(x) for x in (max_ret_20d_rank_pct_grid or [1.0])]
    r5g_grid  = [float(x) for x in (min_ret_5d_rank_pct_grid or [0.0])]
    rgiqr_grid  = [float(x) for x in (regime_cs_iqr_max_grid  or [0.0])]
    rgskew_grid = [float(x) for x in (regime_cs_skew_min_grid or [-1e9])]
    corr_window_grid = [int(x) for x in (corr_window_days_grid or [0])]
    corr_max_grid = [float(x) for x in (corr_max_signed_grid or [1.0])]
    minp_grid = [int(x) for x in (min_picks_grid or [0])]
    # No-picks fallback axis — 0.0 means "no fallback" for legacy parity.
    # When the symbol is empty, alloc_grid is forced to [0.0] (no fallback
    # can fire without a target symbol).
    fb_sym = str(no_picks_fallback_symbol or "").strip().upper()
    fb_alloc_grid = (
        [float(x) for x in (no_picks_fallback_alloc_grid or [0.0])]
        if fb_sym else [0.0]
    )
    conv_grid = list(conviction_scaled_alloc_grid or [False])
    alloc_mode_grid = [str(x or "equal").strip().lower() for x in (allocation_mode_grid or ["equal"])]
    alloc_temp_grid = [float(x) for x in (allocation_temp_grid or [1.0])]
    msa_grid = [float(x) for x in (min_secondary_allocation_grid or [0.0])]
    sup_grid = [float(x) for x in (score_uncertainty_penalty_grid or [0.0])]
    if any(x < 0.0 for x in sup_grid):
        raise ValueError("score_uncertainty_penalty_grid values must be >= 0")
    valid_alloc_modes = {"equal", "score_norm", "softmax", "worksteal"}
    bad_alloc_modes = sorted(set(alloc_mode_grid) - valid_alloc_modes)
    if bad_alloc_modes:
        raise ValueError(
            f"Unknown allocation_mode values {bad_alloc_modes}; "
            "expected equal|score_norm|softmax|worksteal"
        )
    if int(corr_min_periods) < 2:
        raise ValueError("corr_min_periods must be >= 2")
    if any(x < 0 for x in corr_window_grid):
        raise ValueError("corr_window_days_grid values must be >= 0")
    if any((x < -1.0 or x > 1.0) for x in corr_max_grid):
        raise ValueError("corr_max_signed_grid values must be in [-1, 1]")
    alloc_pairs = _allocation_grid_pairs(alloc_mode_grid, alloc_temp_grid)
    dropped_alloc_cells = len(alloc_mode_grid) * len(alloc_temp_grid) - len(alloc_pairs)
    if dropped_alloc_cells > 0:
        logger.info(
            "allocation grid canonicalized: dropped %d temp-invariant duplicate cells",
            dropped_alloc_cells,
        )

    # SPY series used by BOTH knobs (regime gate + vol target).
    spy_close_by_date: pd.Series | None = None
    needs_spy = any(r > 0 for r in rgw_grid) or any(v > 0 for v in vta_grid)
    if needs_spy:
        if spy_csv_path is None or not spy_csv_path.exists():
            raise FileNotFoundError(
                f"regime_gate_window or vol_target_ann > 0 requires "
                f"--spy-csv; got {spy_csv_path!r}"
            )
        _spy_df = pd.read_csv(spy_csv_path, usecols=["timestamp", "close"])
        _spy_df["timestamp"] = pd.to_datetime(_spy_df["timestamp"], utc=True, errors="coerce")
        _spy_df = _spy_df.dropna(subset=["timestamp", "close"]).drop_duplicates(subset=["timestamp"])
        _spy_df["date"] = _spy_df["timestamp"].dt.date
        _spy_df = _spy_df.sort_values("timestamp")
        spy_close_by_date = (
            _spy_df.groupby("date")["close"].last().astype(float).sort_index()
        )
        logger.info(
            "loaded SPY closes: %d days (regime_gate_grid=%s, vol_target_ann_grid=%s)",
            len(spy_close_by_date), rgw_grid, vta_grid,
        )

    resume_by_key = _resume_cells_from_rows(resume_rows or [])
    if resume_by_key:
        logger.info("resume enabled: loaded %d completed checkpoint cells", len(resume_by_key))

    cells: list[CellResult] = []
    skip_pairs = [
        (float(sp), int(sseed))
        for sp in sp_grid
        for sseed in (ss_list if sp > 0 else [0])
    ]
    total = (
        len(leverage_grid) * len(min_score_grid)
        * len(hold_through_grid) * len(top_n_grid) * len(shortn_grid)
        * len(max_short_score_grid) * len(short_alloc_scale_grid)
        * len(minp_grid) * len(fee_regimes)
        * len(opp_watch_grid) * len(opp_discount_grid)
        * len(inf_grid) * len(spread_grid) * len(vol_grid) * len(maxvol_grid)
        * len(skip_pairs) * len(fb_grid)
        * len(rgw_grid) * len(vta_grid) * len(ivt_grid)
        * len(r20g_grid) * len(r5g_grid)
        * len(rgiqr_grid) * len(rgskew_grid)
        * len(corr_window_grid) * len(corr_max_grid)
        * len(fb_alloc_grid) * len(conv_grid)
        * len(alloc_pairs) * len(msa_grid) * len(sup_grid)
    )
    i = 0
    for (
        lev, ms, ht, tn, shortn, max_short_score, short_alloc_scale, minp,
        opp_watch, opp_disc, reg, inf_dv,
        inf_spread, inf_vol, inf_maxvol, skip_pair, fb, rgw, vta, ivt,
        r20g, r5g, rgiqr, rgskew,
        corr_window, corr_max,
        fb_alloc, conv, alloc_pair, min_secondary_alloc, sup,
    ) in product(
        leverage_grid, min_score_grid, hold_through_grid, top_n_grid,
        shortn_grid, max_short_score_grid, short_alloc_scale_grid, minp_grid,
        opp_watch_grid, opp_discount_grid, fee_regimes, inf_grid, spread_grid,
        vol_grid, maxvol_grid, skip_pairs, fb_grid, rgw_grid, vta_grid, ivt_grid,
        r20g_grid, r5g_grid, rgiqr_grid, rgskew_grid,
        corr_window_grid, corr_max_grid, fb_alloc_grid, conv_grid,
        alloc_pairs, msa_grid, sup_grid,
    ):
        alloc_mode, alloc_temp = alloc_pair
        sp, sseed = skip_pair
        cell_scores = _uncertainty_adjusted_scores(scores, score_std, float(sup))
        i += 1
        cell_key = _cell_key_from_values(
            leverage=lev,
            min_score=ms,
            hold_through=ht,
            top_n=tn,
            short_n=shortn,
            max_short_score=max_short_score,
            short_allocation_scale=short_alloc_scale,
            min_picks=minp,
            opportunistic_watch_n=opp_watch,
            opportunistic_entry_discount_bps=opp_disc,
            score_uncertainty_penalty=sup,
            fee_regime=reg,
            inference_min_dolvol=inf_dv,
            inference_max_spread_bps=inf_spread,
            inference_min_vol_20d=inf_vol,
            inference_max_vol_20d=inf_maxvol,
            skip_prob=sp,
            skip_seed=sseed,
            fill_buffer_bps=fb,
            regime_gate_window=rgw,
            vol_target_ann=vta,
            inv_vol_target_ann=ivt,
            inv_vol_floor=inv_vol_floor,
            inv_vol_cap=inv_vol_cap,
            max_ret_20d_rank_pct=r20g,
            min_ret_5d_rank_pct=r5g,
            regime_cs_iqr_max=rgiqr,
            regime_cs_skew_min=rgskew,
            corr_window_days=corr_window,
            corr_min_periods=int(corr_min_periods),
            corr_max_signed=corr_max,
            no_picks_fallback_symbol=fb_sym if fb_alloc != 0.0 else "",
            no_picks_fallback_alloc_scale=fb_alloc,
            conviction_scaled_alloc=bool(conv),
            conviction_alloc_low=conviction_alloc_low,
            conviction_alloc_high=conviction_alloc_high,
            allocation_mode=alloc_mode,
            allocation_temp=alloc_temp,
            min_secondary_allocation=min_secondary_alloc,
        )
        resumed_cell = resume_by_key.get(cell_key)
        if resumed_cell is not None:
            resumed_cell.ensemble_needs_ranks = bool(needs_ranks)
            resumed_cell.ensemble_needs_dispersion = bool(needs_disp)
            logger.info(
                "cell %d/%d lev=%.2f ms=%.2f ht=%s tn=%d sn=%d sms=%.2f sas=%.2f minp=%d "
                "opp=%d/%.1fbps up=%.2f reg=%s "
                "alloc=%s/%.2f resumed from checkpoint",
                i, total, lev, ms, ht, tn, shortn, max_short_score,
                short_alloc_scale, minp,
                opp_watch, opp_disc, sup, reg,
                alloc_mode, alloc_temp,
            )
            cells.append(resumed_cell)
            if progress_callback is not None:
                progress_callback(cells, i, total)
            continue
        cell = _run_cell(
            oos_df=oos_df, scores=cell_scores, windows=windows,
            leverage=lev, min_score=ms, hold_through=ht,
            top_n=tn, short_n=shortn, max_short_score=max_short_score,
            short_allocation_scale=short_alloc_scale,
            min_picks=minp,
            opportunistic_watch_n=opp_watch,
            opportunistic_entry_discount_bps=opp_disc,
            score_uncertainty_penalty=sup,
            fee_regime=reg,
            inference_min_dolvol=inf_dv,
            inference_max_spread_bps=inf_spread,
            inference_min_vol_20d=inf_vol,
            inference_max_vol_20d=inf_maxvol,
            skip_prob=sp, skip_seed=sseed,
            fill_buffer_bps=fb,
            regime_gate_window=rgw,
            vol_target_ann=vta,
            spy_close_by_date=spy_close_by_date,
            inv_vol_target_ann=ivt,
            inv_vol_floor=inv_vol_floor,
            inv_vol_cap=inv_vol_cap,
            max_ret_20d_rank_pct=r20g,
            min_ret_5d_rank_pct=r5g,
            regime_cs_iqr_max=rgiqr,
            regime_cs_skew_min=rgskew,
            corr_window_days=corr_window,
            corr_min_periods=int(corr_min_periods),
            corr_max_signed=corr_max,
            no_picks_fallback_symbol=fb_sym if fb_alloc != 0.0 else "",
            no_picks_fallback_alloc_scale=fb_alloc,
            conviction_scaled_alloc=bool(conv),
            conviction_alloc_low=conviction_alloc_low,
            conviction_alloc_high=conviction_alloc_high,
            allocation_mode=alloc_mode,
            allocation_temp=alloc_temp,
            min_secondary_allocation=min_secondary_alloc,
            fail_fast_max_dd_pct=float(fail_fast_max_dd_pct),
            fail_fast_max_intraday_dd_pct=float(fail_fast_max_intraday_dd_pct),
            fail_fast_neg_windows=int(fail_fast_neg_windows),
            overnight_max_gross_leverage=overnight_max_gross_leverage,
        )
        cell.ensemble_needs_ranks = bool(needs_ranks)
        cell.ensemble_needs_dispersion = bool(needs_disp)
        logger.info(
            "cell %d/%d lev=%.2f ms=%.2f ht=%s tn=%d sn=%d sms=%.2f sas=%.2f minp=%d "
            "opp=%d/%.1fbps up=%.2f reg=%s "
            "inf_dv=%.0e vol=[%.3f,%.3f] skp=%.2f/%d fb=%.1f "
            "spread<=%.1f "
            "rgw=%d vta=%.2f ivt=%.2f r20g=%.2f r5g=%.2f "
            "rgiqr=%.3f rgskew=%+.2f "
            "fb_sym=%s fb_alloc=%.2f conv=%s alloc=%s/%.2f "
            "med=%+.2f%% p10=%+.2f%% neg=%d/%d%s",
            i, total, lev, ms, ht, tn, shortn, max_short_score,
            short_alloc_scale, minp,
            opp_watch, opp_disc, sup, reg,
            inf_dv, inf_vol, inf_maxvol, sp, sseed,
            cell.fill_buffer_bps,
            inf_spread,
            rgw, vta, ivt, r20g, r5g,
            rgiqr, rgskew,
            fb_sym if fb_alloc != 0.0 else "",
            fb_alloc, conv, alloc_mode, alloc_temp,
            cell.median_monthly_pct, cell.p10_monthly_pct,
            cell.n_neg, cell.n_windows,
            f" FAIL_FAST({cell.fail_fast_reason})" if cell.fail_fast_triggered else "",
        )
        cells.append(cell)
        if progress_callback is not None:
            progress_callback(cells, i, total)
    return cells


def _cells_to_rows(cells: list[CellResult]) -> list[dict]:
    return [
        {
            "leverage": c.leverage, "min_score": c.min_score,
            "hold_through": c.hold_through, "top_n": c.top_n,
            "short_n": c.short_n,
            "max_short_score": c.max_short_score,
            "short_allocation_scale": c.short_allocation_scale,
            "min_picks": c.min_picks,
            "opportunistic_watch_n": c.opportunistic_watch_n,
            "opportunistic_entry_discount_bps": c.opportunistic_entry_discount_bps,
            "score_uncertainty_penalty": c.score_uncertainty_penalty,
            "fee_regime": c.fee_regime,
            "n_windows": c.n_windows,
            "median_monthly_pct": c.median_monthly_pct,
            "p10_monthly_pct": c.p10_monthly_pct,
            "median_sortino": c.median_sortino,
            "worst_dd_pct": c.worst_dd_pct,
            "n_neg": c.n_neg,
            "mean_abs_neg_monthly_pct": c.mean_abs_neg_monthly_pct,
            "monthly_return_pcts": c.monthly_return_pcts,
            "window_sortino_values": c.window_sortino_values,
            "window_drawdown_pcts": c.window_drawdown_pcts,
            "window_time_under_water_pcts": c.window_time_under_water_pcts,
            "window_ulcer_indexes": c.window_ulcer_indexes,
            "window_active_day_pcts": c.window_active_day_pcts,
            "window_worst_intraday_dd_pcts": c.window_worst_intraday_dd_pcts,
            "window_avg_intraday_dd_pcts": c.window_avg_intraday_dd_pcts,
            "window_start_dates": c.window_start_dates,
            "window_end_dates": c.window_end_dates,
            "goodness_score": c.goodness_score,
            "robust_goodness_score": c.robust_goodness_score,
            "pain_adjusted_goodness_score": c.pain_adjusted_goodness_score,
            "fail_fast_triggered": c.fail_fast_triggered,
            "fail_fast_reason": c.fail_fast_reason,
            "worst_intraday_dd_pct": c.worst_intraday_dd_pct,
            "avg_intraday_dd_pct":   c.avg_intraday_dd_pct,
            "time_under_water_pct":  c.time_under_water_pct,
            "ulcer_index":           c.ulcer_index,
            "median_active_day_pct": c.median_active_day_pct,
            "min_active_day_pct":    c.min_active_day_pct,
            "inference_min_dolvol":  c.inference_min_dolvol,
            "inference_max_spread_bps": c.inference_max_spread_bps,
            "inference_min_vol_20d": c.inference_min_vol_20d,
            "inference_max_vol_20d": c.inference_max_vol_20d,
            "skip_prob":             c.skip_prob,
            "skip_seed":             c.skip_seed,
            "fill_buffer_bps":       c.fill_buffer_bps,
            "regime_gate_window":    c.regime_gate_window,
            "vol_target_ann":        c.vol_target_ann,
            "inv_vol_target_ann":    c.inv_vol_target_ann,
            "inv_vol_floor":         c.inv_vol_floor,
            "inv_vol_cap":           c.inv_vol_cap,
            "max_ret_20d_rank_pct":  c.max_ret_20d_rank_pct,
            "min_ret_5d_rank_pct":   c.min_ret_5d_rank_pct,
            "regime_cs_iqr_max":     c.regime_cs_iqr_max,
            "regime_cs_skew_min":    c.regime_cs_skew_min,
            "corr_window_days":       c.corr_window_days,
            "corr_min_periods":       c.corr_min_periods,
            "corr_max_signed":        c.corr_max_signed,
            "no_picks_fallback_symbol":     c.no_picks_fallback_symbol,
            "no_picks_fallback_alloc_scale": c.no_picks_fallback_alloc_scale,
            "conviction_scaled_alloc":      c.conviction_scaled_alloc,
            "conviction_alloc_low":         c.conviction_alloc_low,
            "conviction_alloc_high":        c.conviction_alloc_high,
            "allocation_mode":              c.allocation_mode,
            "allocation_temp":              c.allocation_temp,
            "min_secondary_allocation":     c.min_secondary_allocation,
            "expected_n_windows":           c.expected_n_windows,
            "ensemble_needs_ranks":         c.ensemble_needs_ranks,
            "ensemble_needs_dispersion":    c.ensemble_needs_dispersion,
        }
        for c in cells
    ]


def _friction_robust_strategy_rows(rows: list[dict]) -> list[dict]:
    """Aggregate sweep rows across stress cells by deployable strategy params.

    The production rule is evaluated on the worst stress cell, not the
    easiest row. This summary groups identical deployable strategy knobs while
    treating fee_regime, fill_buffer_bps, skip_prob, and skip_seed as stress
    axes. Higher-is-better metrics use the minimum across the group;
    lower-is-better pain/risk metrics use the maximum.
    """
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        grouped.setdefault(_strategy_key_from_mapping(row), []).append(row)

    summaries: list[dict] = []
    high_good = (
        "median_monthly_pct",
        "p10_monthly_pct",
        "median_sortino",
        "goodness_score",
        "robust_goodness_score",
        "pain_adjusted_goodness_score",
        "median_active_day_pct",
        "min_active_day_pct",
    )
    low_good = (
        "worst_dd_pct",
        "worst_intraday_dd_pct",
        "avg_intraday_dd_pct",
        "time_under_water_pct",
        "ulcer_index",
        "n_neg",
    )
    for group_rows in grouped.values():
        first = group_rows[0]
        summary = {field: first.get(field) for field in STRATEGY_PARAM_FIELDS}
        for metric in high_good:
            summary[f"worst_{metric}"] = min(
                float(r.get(metric, 0.0)) for r in group_rows
            )
        for metric in low_good:
            raw = max(float(r.get(metric, 0.0)) for r in group_rows)
            summary[f"max_{metric}"] = int(raw) if metric == "n_neg" else raw
        summary["min_n_windows"] = min(int(r.get("n_windows", 0)) for r in group_rows)
        summary["max_expected_n_windows"] = max(
            int(r.get("expected_n_windows", 0)) for r in group_rows
        )
        summary["required_min_n_windows"] = max(
            PRODUCTION_TARGET_MIN_WINDOWS,
            int(summary["max_expected_n_windows"]),
        )
        summary["n_friction_cells"] = len(group_rows)
        summary["fee_regimes"] = sorted({str(r.get("fee_regime", "")) for r in group_rows})
        summary["fill_buffer_bps_values"] = sorted(
            {float(r.get("fill_buffer_bps", 0.0)) for r in group_rows}
        )
        summary["skip_prob_values"] = sorted(
            {float(r.get("skip_prob", 0.0)) for r in group_rows}
        )
        summary["skip_seed_values"] = sorted(
            {int(r.get("skip_seed", 0)) for r in group_rows}
        )
        worst_row = min(
            group_rows,
            key=lambda r: float(r.get("pain_adjusted_goodness_score", 0.0)),
        )
        summary["worst_fee_regime_by_pain"] = worst_row.get("fee_regime", "")
        summary["worst_fill_buffer_bps_by_pain"] = float(
            worst_row.get("fill_buffer_bps", 0.0)
        )
        summary["any_fail_fast_triggered"] = any(
            bool(r.get("fail_fast_triggered", False)) for r in group_rows
        )
        summary["production_target_pass"] = (
            not summary["any_fail_fast_triggered"]
            and summary["worst_median_monthly_pct"] >= PRODUCTION_TARGET_MEDIAN_MONTHLY_PCT
            and summary["max_worst_dd_pct"] <= PRODUCTION_TARGET_MAX_DD_PCT
            and summary["max_n_neg"] <= PRODUCTION_TARGET_MAX_NEG_WINDOWS
            and summary["min_n_windows"] >= summary["required_min_n_windows"]
        )
        summaries.append(summary)

    summaries.sort(
        key=lambda r: (
            bool(r["production_target_pass"]),
            float(r["worst_pain_adjusted_goodness_score"]),
            float(r["worst_median_monthly_pct"]),
        ),
        reverse=True,
    )
    return summaries


def _production_target_pass_rows(rows: list[dict]) -> list[dict]:
    """Return friction-robust strategy summaries that clear prod targets."""
    return [
        r for r in _friction_robust_strategy_rows(rows)
        if bool(r.get("production_target_pass", False))
    ]


def _production_target_exit_code(rows: list[dict], *, required: bool) -> int:
    if not required:
        return 0
    return 0 if _production_target_pass_rows(rows) else PRODUCTION_TARGET_EXIT_CODE


def _sweep_json_payload(
    *,
    symbols_file: Path | None = None,
    data_root: Path | None = None,
    spy_csv_path: Path | None = None,
    spy_csv_sha256: str | None = None,
    fm_latents_path: Path | None = None,
    fm_latents_sha256: str | None = None,
    fm_n_latents: int | None = None,
    blend_mode: str | None = None,
    model_paths: list[Path],
    model_sha256: list[str] | None = None,
    ensemble_manifest: dict | None = None,
    oos_start: str,
    oos_end: date,
    window_days: int,
    stride_days: int,
    fee_regimes: list[str],
    fail_fast_max_dd_pct: float,
    fail_fast_max_intraday_dd_pct: float,
    fail_fast_neg_windows: int,
    rows: list[dict],
    complete: bool,
) -> dict:
    robust_strategies = _friction_robust_strategy_rows(rows)
    production_passes = [
        r for r in robust_strategies
        if bool(r.get("production_target_pass", False))
    ]
    payload = {
        **({"symbols_file": str(symbols_file)} if symbols_file is not None else {}),
        **({"data_root": str(data_root)} if data_root is not None else {}),
        **({"spy_csv": str(spy_csv_path)} if spy_csv_path is not None else {}),
        **({"fm_latents_path": str(fm_latents_path)} if fm_latents_path is not None else {}),
        **({"blend_mode": str(blend_mode)} if blend_mode is not None else {}),
        "model_paths": [str(p) for p in model_paths],
        "oos_start": oos_start,
        "oos_end": str(oos_end),
        "window_days": int(window_days),
        "stride_days": int(stride_days),
        "fee_regimes": {k: FEE_REGIMES[k] for k in fee_regimes},
        "goodness_weights": GOODNESS_WEIGHTS,
        "robust_goodness_weights": ROBUST_GOODNESS_WEIGHTS,
        "pain_adjusted_goodness_weights": PAIN_ADJUSTED_GOODNESS_WEIGHTS,
        "ensemble_feature_mode": {
            "needs_ranks": any(bool(row.get("ensemble_needs_ranks", False)) for row in rows),
            "needs_dispersion": any(
                bool(row.get("ensemble_needs_dispersion", False)) for row in rows
            ),
        },
        "production_target": {
            "median_monthly_pct": PRODUCTION_TARGET_MEDIAN_MONTHLY_PCT,
            "max_dd_pct": PRODUCTION_TARGET_MAX_DD_PCT,
            "max_neg_windows": PRODUCTION_TARGET_MAX_NEG_WINDOWS,
            "min_windows": PRODUCTION_TARGET_MIN_WINDOWS,
            "expected_windows_required": True,
            "basis": (
                "worst fee_regime/fill_buffer_bps/skip_prob/skip_seed "
                "cell per strategy"
            ),
        },
        "fail_fast": {
            "max_dd_pct": float(fail_fast_max_dd_pct),
            "max_intraday_dd_pct": float(fail_fast_max_intraday_dd_pct),
            "neg_windows": int(fail_fast_neg_windows),
            "score": FAIL_FAST_SCORE,
        },
        "complete": bool(complete),
        "n_cells": len(rows),
        "n_friction_robust_strategies": len(robust_strategies),
        "n_production_target_pass": len(production_passes),
        "best_production_target_strategy": production_passes[0] if production_passes else None,
        "friction_robust_strategies": robust_strategies,
        "cells": rows,
    }
    fm_hash = (
        _optional_file_sha256(fm_latents_path)
        if fm_latents_sha256 is None and fm_latents_path is not None
        else fm_latents_sha256
    )
    if fm_hash is not None:
        payload["fm_latents_sha256"] = fm_hash
    if fm_n_latents is not None:
        payload["fm_n_latents"] = int(fm_n_latents)
    hashes = _model_sha256(model_paths) if model_sha256 is None else list(model_sha256)
    if hashes is not None:
        payload["model_sha256"] = hashes
    spy_hash = _optional_file_sha256(spy_csv_path) if spy_csv_sha256 is None else spy_csv_sha256
    if spy_hash is not None:
        payload["spy_csv_sha256"] = spy_hash
    manifest = _ensemble_manifest_metadata(model_paths) if ensemble_manifest is None else ensemble_manifest
    if manifest is not None:
        payload["ensemble_manifest"] = manifest
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_sha256(model_paths: list[Path]) -> list[str] | None:
    """Return per-model file hashes when all model files are available."""
    hashes: list[str] = []
    for path in model_paths:
        if not path.is_file():
            return None
        hashes.append(_file_sha256(path))
    return hashes


def _optional_file_sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    return _file_sha256(path)


def _ensemble_manifest_metadata(model_paths: list[Path]) -> dict | None:
    """Return shared alltrain manifest provenance when model paths have one."""
    if not model_paths:
        return None
    parents = {path.parent for path in model_paths}
    if len(parents) != 1:
        return None
    manifest_path = next(iter(parents)) / "alltrain_ensemble.json"
    if not manifest_path.is_file():
        return None
    metadata: dict = {
        "path": str(manifest_path),
        "sha256": _file_sha256(manifest_path),
    }
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return metadata
    if not isinstance(payload, dict):
        return metadata
    for key in ("trained_at", "train_start", "train_end", "seeds", "config"):
        if key in payload:
            metadata[key] = payload[key]
    return metadata


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def _load_checkpoint_rows(
    path: Path,
    *,
    expected_model_paths: list[Path] | None = None,
    expected_oos_start: str | None = None,
    expected_oos_end: date | str | None = None,
    expected_window_days: int | None = None,
    expected_stride_days: int | None = None,
) -> list[dict]:
    payload = json.loads(path.read_text())
    mismatches: list[str] = []
    if expected_model_paths is not None:
        expected = [str(p) for p in expected_model_paths]
        if payload.get("model_paths") != expected:
            mismatches.append("model_paths")
    if expected_oos_start is not None and payload.get("oos_start") != str(expected_oos_start):
        mismatches.append("oos_start")
    if expected_oos_end is not None and payload.get("oos_end") != str(expected_oos_end):
        mismatches.append("oos_end")
    if expected_window_days is not None and payload.get("window_days") != int(expected_window_days):
        mismatches.append("window_days")
    if expected_stride_days is not None and payload.get("stride_days") != int(expected_stride_days):
        mismatches.append("stride_days")
    if mismatches:
        joined = ", ".join(mismatches)
        raise ValueError(f"checkpoint {path} does not match current sweep: {joined}")
    rows = payload.get("cells")
    if not isinstance(rows, list):
        raise ValueError(f"checkpoint {path} does not contain a cells list")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"checkpoint {path} contains non-object cells")
    return rows


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-trained ensemble grid sweep.")
    p.add_argument("--symbols-file", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path("trainingdata"))
    p.add_argument("--chronos-cache", type=Path,
                   default=Path("analysis/xgbnew_daily/chronos_cache.parquet"))
    p.add_argument("--fm-latents-path", type=Path, default=None,
                   help="Optional foundation-model latents parquet "
                        "(scripts/build_chronos_bolt_latents.py output). "
                        "Required when the loaded ensemble's feature_cols "
                        "include latent_N / fm_available columns.")
    p.add_argument("--fm-n-latents", type=int, default=0,
                   help="Override the number of latent_N columns to attach. "
                        "0 (default) = auto-detect from the model's feature_cols.")
    p.add_argument("--model-paths", type=str, required=True,
                   help="Comma-separated pkl paths.")
    p.add_argument("--blend-mode", choices=["mean", "median"], default="mean")
    p.add_argument("--train-start", type=str, default="2020-01-01")
    p.add_argument("--train-end",   type=str, default="2024-12-31")
    p.add_argument("--oos-start",   type=str, default="2025-01-02")
    p.add_argument("--oos-end",     type=str, default="")
    p.add_argument("--window-days", type=int, default=30)
    p.add_argument("--stride-days", type=int, default=7)
    p.add_argument("--leverage-grid", type=str, default="2.0")
    p.add_argument("--min-score-grid", type=str, default="0.0")
    p.add_argument("--top-n-grid", type=str, default="1")
    p.add_argument("--short-n-grid", type=str, default="",
                   help="Optional bottom-ranked short slots per day. Empty "
                        "or 0 preserves long-only behavior.")
    p.add_argument("--max-short-score-grid", type=str, default="",
                   help="Maximum score eligible for short candidates. "
                        "Example: 0.45 shorts only low-upside names; 1.0 "
                        "always shorts the weakest names when short_n>0.")
    p.add_argument("--short-allocation-scale-grid", type=str, default="",
                   help="Relative allocation weight for short picks. 0.5 "
                        "means one short gets half the capital weight of one "
                        "long under equal allocation.")
    p.add_argument("--min-picks-grid", type=str, default="",
                   help="Aggressive packing floor grid. Comma-separated "
                        "integers; values greater than top_n are rejected. "
                        "0 or empty preserves classic min_score-gated "
                        "behavior; positive values force at least that many "
                        "best-ranked picks after live-replicable filters.")
    p.add_argument("--opportunistic-watch-n-grid", type=str, default="",
                   help="Work-stealing watchlist size grid. 0 or empty "
                        "disables. Positive values rank this many candidates "
                        "per day, post buy limits below the open, and enter "
                        "only triggered names up to top_n. Positive values "
                        "must be >= top_n.")
    p.add_argument("--opportunistic-entry-discount-bps-grid", type=str, default="",
                   help="Buy-limit discount below the day's open, in bps, "
                        "for opportunistic watchlist entries. Example: 30 "
                        "means buy at open*0.997 only if the bar low "
                        "penetrates that limit by fill_buffer_bps. 0 or "
                        "empty disables with watch_n=0.")
    p.add_argument("--hold-through", action="store_true",
                   help="Include hold_through=True in sweep.")
    p.add_argument("--no-hold-through", action="store_true",
                   help="Also include hold_through=False (for A/B).")
    p.add_argument("--fee-regimes", type=str, default="deploy",
                   help=f"Comma-separated regimes from {list(FEE_REGIMES)}")
    p.add_argument("--min-dollar-vol", type=float, default=5_000_000.0,
                   help="Training-universe liquidity floor (applied before "
                        "feature build). Inference-side floor defaults to "
                        "this unless --inference-min-dolvol-grid is set.")
    p.add_argument("--inference-min-dolvol-grid", type=str, default="",
                   help="Optional comma-separated inference-time "
                        "min_dollar_vol floors. Each value narrows the "
                        "pick pool AFTER model scoring — lets us sweep "
                        "'train broad, trade narrow'. Empty = single cell "
                        "at --min-dollar-vol.")
    p.add_argument("--inference-max-spread-bps-grid", type=str, default="",
                   help="Optional comma-separated inference-time spread "
                        "ceilings in bps. 30 matches the live/default "
                        "BacktestConfig filter; 0 disables the spread filter. "
                        "Lets sweeps test tighter liquidity filters without "
                        "changing training data.")
    p.add_argument("--inference-min-vol-grid", type=str, default="",
                   help="Optional comma-separated inference-time vol_20d "
                        "(annualised realised-vol) floors. 0 or empty "
                        "disables. Higher = drop more of the dead-zone / "
                        "bot-vol quartile.")
    p.add_argument("--inference-max-vol-grid", type=str, default="",
                   help="Optional comma-separated inference-time vol_20d "
                        "(annualised realised-vol) CEILINGS. 0 or empty "
                        "disables. Lower = drop more of the crash-sensitive "
                        "high-vol tail. Symbols stay in TRAINING — masked "
                        "only from the inference pick pool. Band-pass with "
                        "--inference-min-vol-grid.")
    p.add_argument("--skip-prob-grid", type=str, default="",
                   help="Missed-order Monte Carlo — comma-separated skip "
                        "probabilities, e.g. '0.0,0.05,0.10,0.20'. Each "
                        "value simulates 'Alpaca rejects this pick' at "
                        "that rate. Empty = single cell at 0.0.")
    p.add_argument("--skip-seeds", type=str, default="0,1,2",
                   help="Comma-separated RNG seeds for each non-zero skip "
                        "probability (gives MC variance). Ignored at "
                        "skip_prob=0.0 since the sim is deterministic.")
    p.add_argument("--fill-buffer-bps-grid", type=str, default="",
                   help="Override fill_buffer_bps for the cell. "
                        "Comma-separated bps, e.g. '3,5,8,15,30' — "
                        "tests how the config degrades as executions "
                        "get progressively worse. Empty = single cell at "
                        "the fee regime's default FB (deploy/prod10bps=5, "
                        "stress36x=15).")
    p.add_argument("--regime-gate-window-grid", type=str, default="",
                   help="Comma-separated SPY MA windows (e.g. '0,20,50,200'). "
                        "0 disables the gate; positive values stay-in-cash "
                        "when SPY closes below its N-day MA. Requires "
                        "--spy-csv.")
    p.add_argument("--vol-target-ann-grid", type=str, default="",
                   help="Comma-separated annualised vol targets (e.g. "
                        "'0.0,0.15,0.20,0.25'). 0 disables; positive values "
                        "scale daily allocation by min(1, target/SPY_20d_vol). "
                        "Requires --spy-csv.")
    p.add_argument("--spy-csv", type=Path,
                   default=Path("trainingdata/SPY.csv"),
                   help="SPY daily OHLCV CSV — used by --regime-gate-window-grid "
                        "and --vol-target-ann-grid. Ignored when both grids "
                        "are all-zero.")
    p.add_argument("--inv-vol-target-grid", type=str, default="",
                   help="Per-pick inv-vol sizing targets (annualised). "
                        "Comma-separated, e.g. '0.0,0.20,0.25,0.30'. 0 "
                        "disables (legacy sim). Positive scales each pick's "
                        "leverage by clip(target/max(vol_20d, floor), "
                        "1/cap, cap). Cross-sectional — responds to "
                        "individual pick vol, unlike --vol-target-ann-grid "
                        "(which reads SPY).")
    p.add_argument("--inv-vol-floor", type=float, default=0.05,
                   help="Denominator floor for inv-vol sizing. Any pick with "
                        "vol_20d below this is treated as at-floor so the "
                        "scale doesn't blow up. Default 0.05 (5%% ann).")
    p.add_argument("--inv-vol-cap", type=float, default=3.0,
                   help="Symmetric cap on the inv-vol multiplier. Scale is "
                        "clipped to [1/cap, cap]. Default 3.0 (so 0.333×..3×).")
    p.add_argument("--max-ret-20d-rank-pct-grid", type=str, default="",
                   help="Per-day cross-sectional upper rank-pct on ret_20d "
                        "applied to the pick pool. Comma-separated values in "
                        "(0,1], e.g. '1.0,0.75,0.50'. 1.0 disables (keeps all). "
                        "0.75 drops the top-25%% by ret_20d each day. "
                        "Motivated by regime-inversion diag: hot names crash "
                        "in tariff regimes.")
    p.add_argument("--min-ret-5d-rank-pct-grid", type=str, default="",
                   help="Per-day cross-sectional lower rank-pct on ret_5d "
                        "applied to the pick pool. Comma-separated values in "
                        "[0,1), e.g. '0.0,0.25,0.50'. 0.0 disables. 0.25 drops "
                        "the bottom-25%% by ret_5d each day. Composes "
                        "sequentially after --max-ret-20d-rank-pct-grid: "
                        "ret_5d ranks recomputed on the already-filtered pool.")
    p.add_argument("--regime-cs-iqr-max-grid", type=str, default="",
                   help="Cross-sectional regime-dispersion gate: "
                        "comma-separated upper bounds on per-day IQR of "
                        "ret_5d across the universe. 0 disables. Diagnostic "
                        "found 0.042 as the best absolute threshold on the "
                        "fresh 2025-07→2026-04 true-OOS (+0.67%%/day mean).")
    p.add_argument("--regime-cs-skew-min-grid", type=str, default="",
                   help="Cross-sectional regime-skew gate: comma-separated "
                        "lower bounds on per-day skew of ret_5d across the "
                        "universe. Very negative disables (defaults to -1e9). "
                        "Positive values keep only right-skew days (few "
                        "winners dominate the panel).")
    p.add_argument("--corr-window-days-grid", type=str, default="",
                   help="Correlation-aware packing windows in trading days. "
                        "0 or empty disables. Positive values compute a "
                        "leak-free trailing pairwise return correlation matrix "
                        "available at the open and skip redundant picks.")
    p.add_argument("--corr-min-periods", type=int, default=20,
                   help="Minimum overlapping return observations for "
                        "--corr-window-days-grid. Default 20.")
    p.add_argument("--corr-max-signed-grid", type=str, default="",
                   help="Maximum allowed signed correlation to an already "
                        "selected same-day pick. 1.0 disables. Example 0.6 "
                        "blocks long/long highly positive pairs and long/short "
                        "highly negative pairs that can lose together.")
    p.add_argument("--no-picks-fallback", type=str, default="",
                   help="Symbol to buy on days where no scored candidate "
                        "clears min_score (or conviction-scaling sizes to "
                        "zero). Typical: 'SPY' (broad market) or 'QQQ' "
                        "(higher drift, higher DD). Empty = hold cash "
                        "(legacy). Requires the symbol to exist in the "
                        "training universe so the build_daily_dataset "
                        "pipeline produces bars for it.")
    p.add_argument("--no-picks-fallback-alloc-grid", type=str, default="",
                   help="Comma-separated allocation scales for the no-picks "
                        "fallback, e.g. '0.25,0.5,1.0'. Each value is the "
                        "fraction of config.leverage to apply on no-picks "
                        "days. Empty = single cell at 0.0 (disabled). "
                        "Ignored when --no-picks-fallback is empty.")
    p.add_argument("--conviction-scaled-alloc-grid", type=str, default="",
                   help="Comma-separated booleans for conviction-scaled "
                        "allocation, e.g. '0,1' to A/B vs binary gate. "
                        "0 = off, 1 = on. Empty = single cell at 0 (off). "
                        "When on, each day's exposure is scaled by "
                        "clip((top_score - low) / (high - low), 0, 1).")
    p.add_argument("--conviction-alloc-low", type=float, default=0.55,
                   help="Lower edge of the conviction-scaled allocation "
                        "ramp (score at which exposure is 0%%). Default 0.55.")
    p.add_argument("--conviction-alloc-high", type=float, default=0.85,
                   help="Upper edge of the conviction-scaled allocation "
                        "ramp (score at which exposure is 100%%). "
                        "Default 0.85.")
    p.add_argument("--allocation-mode-grid", type=str, default="",
                   help="Comma-separated allocation weighting modes for "
                        "top_n>1: equal, score_norm, softmax, worksteal. Empty = "
                        "equal only. This changes sizing across selected "
                        "pairs without changing the pick set. worksteal "
                        "allocates 75%% of gross exposure to the first fill "
                        "and splits 25%% across later fills.")
    p.add_argument("--allocation-temp-grid", type=str, default="",
                   help="Comma-separated softmax temperatures for "
                        "--allocation-mode-grid softmax cells. Lower is "
                        "more concentrated. Ignored for equal/score_norm "
                        "because those modes are temperature-invariant. "
                        "Empty = 1.0.")
    p.add_argument("--min-secondary-allocation-grid", type=str, default="",
                   help="Comma-separated minimum non-main allocation floors. "
                        "0 disables. 0.20 caps the largest filled position at "
                        "80%% weight so top-2/top-3 packing keeps at least "
                        "20%% gross exposure outside the main pick.")
    p.add_argument("--score-uncertainty-penalty-grid", type=str, default="",
                   help="Comma-separated penalties for uncertainty-adjusted "
                        "sorting: adjusted_score = ensemble_mean - penalty "
                        "* ensemble_std across seed models. 0 or empty "
                        "preserves raw score sorting.")
    p.add_argument("--fail-fast-max-dd-pct", type=float, default=0.0,
                   help="Stop evaluating a cell once any completed window's "
                        "max drawdown reaches this percentage. 0 disables. "
                        "Pruned cells keep partial metrics but receive "
                        "FAIL_FAST_SCORE ranking values so they cannot win. "
                        "Example: 40 means prune after a 40%% window DD.")
    p.add_argument("--fail-fast-max-intraday-dd-pct", type=float, default=0.0,
                   help="Stop evaluating a cell once any window's OHLC-based "
                        "portfolio intraday drawdown reaches this percentage. "
                        "0 disables. This catches high-leverage cells that "
                        "recover by close but exceed live risk while open. "
                        "Example: 40 means prune after a 40%% intraday DD.")
    p.add_argument("--fail-fast-neg-windows", type=int, default=0,
                   help="Stop evaluating a cell once this many completed "
                        "windows have negative monthly returns. 0 disables. "
                        "Useful for large grids where early losing cells are "
                        "not production candidates regardless of later windows.")
    p.add_argument("--checkpoint-every-cells", type=int, default=0,
                   help="Write an atomic partial JSON checkpoint every N "
                        "completed cells. 0 disables. The checkpoint path "
                        "defaults to sweep_<timestamp>.partial.json under "
                        "--output-dir and uses the same schema as final JSON "
                        "with complete=false.")
    p.add_argument("--checkpoint-path", type=Path, default=None,
                   help="Optional explicit partial checkpoint path. Ignored "
                        "unless --checkpoint-every-cells is positive.")
    p.add_argument("--resume-from-checkpoint", type=Path, default=None,
                   help="Load completed cells from a previous partial/final "
                        "sweep JSON and skip matching grid cells. The final "
                        "output preserves the current grid order and recomputes "
                        "only missing cells.")
    p.add_argument("--require-production-target", action="store_true",
                   help="Exit nonzero after writing output if no friction-robust "
                        "strategy clears the project production target "
                        f"(median>={PRODUCTION_TARGET_MEDIAN_MONTHLY_PCT:g}%% "
                        f"and dd<={PRODUCTION_TARGET_MAX_DD_PCT:g}%% on the "
                        "worst fee/fill cell). Default research mode still "
                        "exits 0 even when all cells fail.")
    p.add_argument("--invert-scores", action="store_true",
                   help="Replace blended scores with 1 - scores so the sim "
                        "picks the bottom-N (originally worst) symbols. "
                        "Long-only sim is still used — POSITIVE median here "
                        "means the model rank-orders correctly (winners & "
                        "losers separable) and a proper short-side sim "
                        "WOULD profit. NEGATIVE median + negative PnL in "
                        "the regular sweep means the model is noise. "
                        "Experimental diagnostic only; LOWER --min-score-grid "
                        "when using this since the inverted distribution "
                        "skews toward lower values.")
    p.add_argument("--fast-features", action="store_true",
                   help="Use the Polars feature builder for large-universe sweeps.")
    p.add_argument("--overnight-max-gross-leverage", type=float, default=None,
                   help="Reg-T overnight gross-leverage cap. None (default) "
                        "preserves legacy uncapped behavior. Pass 2.0 to mirror "
                        "prod xgbnew/live_trader._eod_deleverage_tick — "
                        "candidate cells with leverage>cap will have their "
                        "per-pick effective leverage clipped at cap, matching "
                        "how the prod EOD tick auto-deleverages before close.")
    p.add_argument("--output-dir", type=Path,
                   default=Path("analysis/xgbnew_ensemble_sweep"))
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def _parse_float_list(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    symbols = _load_symbols(args.symbols_file)
    model_paths = [Path(p.strip()) for p in args.model_paths.split(",") if p.strip()]
    try:
        _validate_model_paths_for_sweep(model_paths)
    except ValueError as exc:
        print(f"[sweep] ERROR: {exc}", flush=True)
        return 2
    model_sha256 = _model_sha256(model_paths)
    ensemble_manifest = _ensemble_manifest_metadata(model_paths)
    spy_csv_sha256 = _optional_file_sha256(args.spy_csv)
    fm_latents_path = (
        args.fm_latents_path if getattr(args, "fm_latents_path", None) else None
    )
    fm_latents_sha256 = _optional_file_sha256(fm_latents_path)
    fm_n_latents = (
        int(args.fm_n_latents)
        if getattr(args, "fm_n_latents", 0) and int(args.fm_n_latents) > 0
        else None
    )
    ht_grid: list[bool] = []
    if args.hold_through:
        ht_grid.append(True)
    if args.no_hold_through:
        ht_grid.append(False)
    if not ht_grid:
        ht_grid = [False]

    oos_end = date.fromisoformat(args.oos_end) if args.oos_end else date.today()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = args.output_dir / f"sweep_{ts}.json"
    fee_regime_names = [r.strip() for r in args.fee_regimes.split(",") if r.strip()]
    checkpoint_every = int(args.checkpoint_every_cells)
    checkpoint_path = (
        args.checkpoint_path
        if args.checkpoint_path is not None
        else args.output_dir / f"sweep_{ts}.partial.json"
    )
    resume_rows = (
        _load_checkpoint_rows(
            args.resume_from_checkpoint,
            expected_model_paths=model_paths,
            expected_oos_start=args.oos_start,
            expected_oos_end=oos_end,
            expected_window_days=int(args.window_days),
            expected_stride_days=int(args.stride_days),
        )
        if args.resume_from_checkpoint is not None
        else None
    )
    if resume_rows is not None:
        logger.info(
            "loaded %d resume cells from %s",
            len(resume_rows),
            args.resume_from_checkpoint,
        )

    def _checkpoint(cells_so_far: list[CellResult], done: int, total: int) -> None:
        if checkpoint_every <= 0:
            return
        if done != total and done % checkpoint_every != 0:
            return
        rows_so_far = _cells_to_rows(cells_so_far)
        _write_json_atomic(
            checkpoint_path,
            _sweep_json_payload(
                symbols_file=args.symbols_file,
                data_root=args.data_root,
                spy_csv_path=args.spy_csv,
                spy_csv_sha256=spy_csv_sha256,
                fm_latents_path=fm_latents_path,
                fm_latents_sha256=fm_latents_sha256,
                fm_n_latents=fm_n_latents,
                blend_mode=args.blend_mode,
                model_paths=model_paths,
                model_sha256=model_sha256,
                ensemble_manifest=ensemble_manifest,
                oos_start=args.oos_start,
                oos_end=oos_end,
                window_days=int(args.window_days),
                stride_days=int(args.stride_days),
                fee_regimes=fee_regime_names,
                fail_fast_max_dd_pct=float(args.fail_fast_max_dd_pct),
                fail_fast_max_intraday_dd_pct=float(args.fail_fast_max_intraday_dd_pct),
                fail_fast_neg_windows=int(args.fail_fast_neg_windows),
                rows=rows_so_far,
                complete=False,
            ),
        )
        logger.info(
            "checkpoint wrote %s (%d/%d cells)",
            checkpoint_path, done, total,
        )

    cells = run_sweep(
        symbols=symbols,
        data_root=args.data_root,
        model_paths=model_paths,
        train_start=date.fromisoformat(args.train_start),
        train_end=date.fromisoformat(args.train_end),
        oos_start=date.fromisoformat(args.oos_start),
        oos_end=oos_end,
        window_days=int(args.window_days),
        stride_days=int(args.stride_days),
        leverage_grid=_parse_float_list(args.leverage_grid),
        min_score_grid=_parse_float_list(args.min_score_grid),
        hold_through_grid=ht_grid,
        top_n_grid=_parse_int_list(args.top_n_grid),
        short_n_grid=(
            _parse_int_list(args.short_n_grid)
            if args.short_n_grid else None
        ),
        max_short_score_grid=(
            _parse_float_list(args.max_short_score_grid)
            if args.max_short_score_grid else None
        ),
        short_allocation_scale_grid=(
            _parse_float_list(args.short_allocation_scale_grid)
            if args.short_allocation_scale_grid else None
        ),
        fee_regimes=fee_regime_names,
        min_picks_grid=(
            _parse_int_list(args.min_picks_grid)
            if args.min_picks_grid else None
        ),
        opportunistic_watch_n_grid=(
            _parse_int_list(args.opportunistic_watch_n_grid)
            if args.opportunistic_watch_n_grid else None
        ),
        opportunistic_entry_discount_bps_grid=(
            _parse_float_list(args.opportunistic_entry_discount_bps_grid)
            if args.opportunistic_entry_discount_bps_grid else None
        ),
        blend_mode=args.blend_mode,
        chronos_cache_path=args.chronos_cache,
        min_dollar_vol=float(args.min_dollar_vol),
        inference_min_dolvol_grid=(
            _parse_float_list(args.inference_min_dolvol_grid)
            if args.inference_min_dolvol_grid else None
        ),
        inference_max_spread_bps_grid=(
            _parse_float_list(args.inference_max_spread_bps_grid)
            if args.inference_max_spread_bps_grid else None
        ),
        inference_min_vol_grid=(
            _parse_float_list(args.inference_min_vol_grid)
            if args.inference_min_vol_grid else None
        ),
        inference_max_vol_grid=(
            _parse_float_list(args.inference_max_vol_grid)
            if args.inference_max_vol_grid else None
        ),
        skip_prob_grid=(
            _parse_float_list(args.skip_prob_grid)
            if args.skip_prob_grid else None
        ),
        skip_seeds=_parse_int_list(args.skip_seeds),
        fill_buffer_bps_grid=(
            _parse_float_list(args.fill_buffer_bps_grid)
            if args.fill_buffer_bps_grid else None
        ),
        regime_gate_window_grid=(
            _parse_int_list(args.regime_gate_window_grid)
            if args.regime_gate_window_grid else None
        ),
        vol_target_ann_grid=(
            _parse_float_list(args.vol_target_ann_grid)
            if args.vol_target_ann_grid else None
        ),
        spy_csv_path=args.spy_csv,
        inv_vol_target_grid=(
            _parse_float_list(args.inv_vol_target_grid)
            if args.inv_vol_target_grid else None
        ),
        inv_vol_floor=float(args.inv_vol_floor),
        inv_vol_cap=float(args.inv_vol_cap),
        invert_scores=bool(args.invert_scores),
        max_ret_20d_rank_pct_grid=(
            _parse_float_list(args.max_ret_20d_rank_pct_grid)
            if args.max_ret_20d_rank_pct_grid else None
        ),
        min_ret_5d_rank_pct_grid=(
            _parse_float_list(args.min_ret_5d_rank_pct_grid)
            if args.min_ret_5d_rank_pct_grid else None
        ),
        regime_cs_iqr_max_grid=(
            _parse_float_list(args.regime_cs_iqr_max_grid)
            if args.regime_cs_iqr_max_grid else None
        ),
        regime_cs_skew_min_grid=(
            _parse_float_list(args.regime_cs_skew_min_grid)
            if args.regime_cs_skew_min_grid else None
        ),
        corr_window_days_grid=(
            _parse_int_list(args.corr_window_days_grid)
            if args.corr_window_days_grid else None
        ),
        corr_min_periods=int(args.corr_min_periods),
        corr_max_signed_grid=(
            _parse_float_list(args.corr_max_signed_grid)
            if args.corr_max_signed_grid else None
        ),
        no_picks_fallback_symbol=str(args.no_picks_fallback or "").strip().upper(),
        no_picks_fallback_alloc_grid=(
            _parse_float_list(args.no_picks_fallback_alloc_grid)
            if args.no_picks_fallback_alloc_grid else None
        ),
        conviction_scaled_alloc_grid=(
            [bool(int(x)) for x in args.conviction_scaled_alloc_grid.split(",") if x.strip()]
            if args.conviction_scaled_alloc_grid else None
        ),
        conviction_alloc_low=float(args.conviction_alloc_low),
        conviction_alloc_high=float(args.conviction_alloc_high),
        allocation_mode_grid=(
            [x.strip().lower() for x in args.allocation_mode_grid.split(",") if x.strip()]
            if args.allocation_mode_grid else None
        ),
        allocation_temp_grid=(
            _parse_float_list(args.allocation_temp_grid)
            if args.allocation_temp_grid else None
        ),
        min_secondary_allocation_grid=(
            _parse_float_list(args.min_secondary_allocation_grid)
            if args.min_secondary_allocation_grid else None
        ),
        score_uncertainty_penalty_grid=(
            _parse_float_list(args.score_uncertainty_penalty_grid)
            if args.score_uncertainty_penalty_grid else None
        ),
        fail_fast_max_dd_pct=float(args.fail_fast_max_dd_pct),
        fail_fast_max_intraday_dd_pct=float(args.fail_fast_max_intraday_dd_pct),
        fail_fast_neg_windows=int(args.fail_fast_neg_windows),
        fast_features=bool(args.fast_features),
        overnight_max_gross_leverage=(
            None
            if args.overnight_max_gross_leverage is None
            else float(args.overnight_max_gross_leverage)
        ),
        progress_callback=_checkpoint if checkpoint_every > 0 else None,
        resume_rows=resume_rows,
        fm_latents_path=fm_latents_path,
        fm_n_latents=fm_n_latents,
    )

    rows = _cells_to_rows(cells)
    _write_json_atomic(
        out,
        _sweep_json_payload(
            symbols_file=args.symbols_file,
            data_root=args.data_root,
            spy_csv_path=args.spy_csv,
            spy_csv_sha256=spy_csv_sha256,
            fm_latents_path=fm_latents_path,
            fm_latents_sha256=fm_latents_sha256,
            fm_n_latents=fm_n_latents,
            blend_mode=args.blend_mode,
            model_paths=model_paths,
            model_sha256=model_sha256,
            ensemble_manifest=ensemble_manifest,
            oos_start=args.oos_start,
            oos_end=oos_end,
            window_days=int(args.window_days),
            stride_days=int(args.stride_days),
            fee_regimes=fee_regime_names,
            fail_fast_max_dd_pct=float(args.fail_fast_max_dd_pct),
            fail_fast_max_intraday_dd_pct=float(args.fail_fast_max_intraday_dd_pct),
            fail_fast_neg_windows=int(args.fail_fast_neg_windows),
            rows=rows,
            complete=True,
        ),
    )
    print(f"[sweep] wrote {out}  ({len(rows)} cells)", flush=True)

    # Pretty table — sorted by goodness descending for easy frontier read.
    # ddW    = worst realized (equity) drawdown across windows
    # idW    = worst INTRADAY unrealized drawdown (OHLC proxy)
    # Divergence between ddW and idW quantifies the "what we were briefly
    # exposed to" gap vs "what hit the equity curve at close".
    rows_sorted = sorted(rows, key=lambda r: -r["goodness_score"])
    inf_grid_active = len({r["inference_min_dolvol"] for r in rows}) > 1
    vol_grid_active = len({r["inference_min_vol_20d"] for r in rows}) > 1
    sp_grid_active  = len({r["skip_prob"] for r in rows}) > 1
    fb_grid_active  = len({r["fill_buffer_bps"] for r in rows}) > 1
    ff_active = any(r["fail_fast_triggered"] for r in rows)
    alloc_grid_active = (
        len({r["allocation_mode"] for r in rows}) > 1
        or len({r["allocation_temp"] for r in rows}) > 1
    )
    secondary_grid_active = len({r.get("min_secondary_allocation", 0.0) for r in rows}) > 1
    uncertainty_grid_active = len({r.get("score_uncertainty_penalty", 0.0) for r in rows}) > 1
    hdr = (f"\n{'lev':>5} {'ms':>5} {'ht':>3} {'tn':>3} {'mp':>3} {'reg':>10} "
           f"{'med%':>8} {'p10':>8} {'sort':>6} {'ddW':>6} {'idW':>6} "
           f"{'tuw%':>6} {'ulc':>6} {'act%':>6} {'neg':>6} "
           f"{'good':>8} {'robG':>8} {'painG':>8}")
    if ff_active:
        hdr += f" {'ff':>2}"
    if alloc_grid_active:
        hdr += f" {'alloc':>10} {'tmp':>5}"
    if secondary_grid_active:
        hdr += f" {'sec%':>5}"
    if uncertainty_grid_active:
        hdr += f" {'uPen':>5}"
    if inf_grid_active:
        hdr += f" {'inf$V':>8}"
    if vol_grid_active:
        hdr += f" {'infVol':>7}"
    if sp_grid_active:
        hdr += f" {'skip':>5} {'sd':>3}"
    if fb_grid_active:
        hdr += f" {'fb':>5}"
    print(hdr)
    print("-" * (115
                 + (3 if ff_active else 0)
                 + (17 if alloc_grid_active else 0)
                 + (6 if secondary_grid_active else 0)
                 + (6 if uncertainty_grid_active else 0)
                 + (9 if inf_grid_active else 0)
                 + (8 if vol_grid_active else 0)
                 + (9 if sp_grid_active else 0)
                 + (6 if fb_grid_active else 0)))
    for r in rows_sorted:
        line = (f"{r['leverage']:5.2f} {r['min_score']:5.2f} "
                f"{'Y' if r['hold_through'] else 'N':>3} {r['top_n']:3d} "
                f"{int(r.get('min_picks', 0)):3d} "
                f"{r['fee_regime']:>10} "
                f"{r['median_monthly_pct']:+8.2f} {r['p10_monthly_pct']:+8.2f} "
                f"{r['median_sortino']:6.2f} {r['worst_dd_pct']:6.2f} "
                f"{r['worst_intraday_dd_pct']:6.2f} "
                f"{r['time_under_water_pct']:6.2f} "
                f"{r['ulcer_index']:6.2f} "
                f"{r['median_active_day_pct']:6.2f} "
                f"{r['n_neg']:3d}/{r['n_windows']:3d} "
                f"{r['goodness_score']:+8.2f} "
                f"{r['robust_goodness_score']:+8.2f} "
                f"{r['pain_adjusted_goodness_score']:+8.2f}")
        if ff_active:
            line += f" {'Y' if r['fail_fast_triggered'] else 'N':>2}"
        if alloc_grid_active:
            line += f" {r['allocation_mode']:>10} {r['allocation_temp']:5.2f}"
        if secondary_grid_active:
            line += f" {float(r.get('min_secondary_allocation', 0.0)) * 100.0:5.1f}"
        if uncertainty_grid_active:
            line += f" {float(r.get('score_uncertainty_penalty', 0.0)):5.2f}"
        if inf_grid_active:
            line += f" {r['inference_min_dolvol']:8.2e}"
        if vol_grid_active:
            line += f" {r['inference_min_vol_20d']:7.3f}"
        if sp_grid_active:
            line += f" {r['skip_prob']:5.2f} {r['skip_seed']:3d}"
        if fb_grid_active:
            line += f" {r['fill_buffer_bps']:5.1f}"
        print(line)

    friction_rows = _friction_robust_strategy_rows(rows)
    friction_active = any(r["n_friction_cells"] > 1 for r in friction_rows)
    if friction_active:
        hdr2 = (
            f"\nFriction-robust strategy ranking "
            f"(target: med>={PRODUCTION_TARGET_MEDIAN_MONTHLY_PCT:g}%, "
            f"dd<={PRODUCTION_TARGET_MAX_DD_PCT:g}%, "
            f"neg<={PRODUCTION_TARGET_MAX_NEG_WINDOWS} on worst stress cell)\n"
            f"{'lev':>5} {'ms':>5} {'ht':>3} {'tn':>3} {'mp':>3} "
            f"{'medW':>8} {'p10W':>8} {'ddMax':>6} {'tuwMax':>7} "
            f"{'actMin':>7} {'negMax':>6} {'painW':>8} {'pass':>4} "
            f"{'worstReg':>10} {'fb':>5} {'cells':>5}"
        )
        print(hdr2)
        print("-" * 111)
        for r in friction_rows:
            print(
                f"{r['leverage']:5.2f} {r['min_score']:5.2f} "
                f"{'Y' if r['hold_through'] else 'N':>3} {r['top_n']:3d} "
                f"{int(r.get('min_picks', 0)):3d} "
                f"{r['worst_median_monthly_pct']:+8.2f} "
                f"{r['worst_p10_monthly_pct']:+8.2f} "
                f"{r['max_worst_dd_pct']:6.2f} "
                f"{r['max_time_under_water_pct']:7.2f} "
                f"{r['worst_min_active_day_pct']:7.2f} "
                f"{r['max_n_neg']:6d} "
                f"{r['worst_pain_adjusted_goodness_score']:+8.2f} "
                f"{'Y' if r['production_target_pass'] else 'N':>4} "
                f"{str(r['worst_fee_regime_by_pain']):>10} "
                f"{r['worst_fill_buffer_bps_by_pain']:5.1f} "
                f"{r['n_friction_cells']:5d}"
            )
    exit_code = _production_target_exit_code(
        rows,
        required=bool(args.require_production_target),
    )
    if exit_code != 0:
        print(
            "[sweep] no friction-robust strategy cleared production target "
            f"(median>={PRODUCTION_TARGET_MEDIAN_MONTHLY_PCT:g}%, "
            f"dd<={PRODUCTION_TARGET_MAX_DD_PCT:g}%, "
            f"neg<={PRODUCTION_TARGET_MAX_NEG_WINDOWS} on worst fee/fill cell)",
            flush=True,
        )
        return exit_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
