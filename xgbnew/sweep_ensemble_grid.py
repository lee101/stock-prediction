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
        --fee-regimes deploy,stress36x \
        --output-dir analysis/xgbnew_ensemble_sweep
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from xgbnew.backtest import BacktestConfig, simulate
from xgbnew.dataset import build_daily_dataset, load_chronos_cache
from xgbnew.features import DAILY_FEATURE_COLS, DAILY_RANK_FEATURE_COLS
from xgbnew.model import XGBStockModel

logger = logging.getLogger(__name__)


# ── Fee regimes ───────────────────────────────────────────────────────────────
# "deploy" mirrors real Alpaca costs. "stress36x" is the realism-gate stress
# (matches the 36× cell used in project_xgb_full_lever_stack_40pct.md).
FEE_REGIMES: dict[str, dict[str, float]] = {
    "deploy":    {"fee_rate": 0.0000278, "fill_buffer_bps": 5.0,  "commission_bps": 0.0},
    "stress36x": {"fee_rate": 0.001,     "fill_buffer_bps": 15.0, "commission_bps": 10.0},
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
    # Composite "one-scalar" optimization target — see compute_goodness().
    goodness_score: float = 0.0
    # Asymmetric-downside variant — see compute_robust_goodness().
    robust_goodness_score: float = 0.0
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
    # Inference-side liquidity floor applied to the pick pool — stays at
    # 5M$ unless explicitly swept via --inference-min-dolvol-grid.
    inference_min_dolvol: float = 5_000_000.0
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


def _run_cell(
    oos_df: pd.DataFrame,
    scores: pd.Series,
    windows: list[tuple],
    leverage: float,
    min_score: float,
    hold_through: bool,
    top_n: int,
    fee_regime: str,
    inference_min_dolvol: float = 5_000_000.0,
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
        leverage=float(leverage),
        xgb_weight=1.0,
        fee_rate=float(fees["fee_rate"]),
        fill_buffer_bps=fb_resolved,
        commission_bps=float(fees["commission_bps"]),
        min_dollar_vol=float(inference_min_dolvol),
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
    )

    dummy = XGBStockModel(device="cpu", n_estimators=1, max_depth=1, learning_rate=0.1)
    dummy.feature_cols = DAILY_FEATURE_COLS
    dummy._col_medians = np.zeros(len(DAILY_FEATURE_COLS), dtype=np.float32)
    dummy._fitted = True

    monthlies: list[float] = []
    sortinos: list[float] = []
    dds: list[float] = []
    intra_dds_worst: list[float] = []
    intra_dds_avg:   list[float] = []
    tuws: list[float] = []
    ulcers: list[float] = []
    for w_start, w_end in windows:
        w_df = oos_df[(oos_df["date"] >= w_start) & (oos_df["date"] <= w_end)]
        if len(w_df) < 5:
            continue
        w_scores = scores.loc[w_df.index]
        res = simulate(
            w_df, dummy, cfg,
            precomputed_scores=w_scores,
            spy_close_by_date=spy_close_by_date,
        )
        n_days = len(res.day_results)
        monthly = _monthly_return(res.total_return_pct, max(n_days, 1)) * 100.0
        monthlies.append(monthly)
        sortinos.append(res.sortino_ratio)
        dds.append(res.max_drawdown_pct)
        intra_dds_worst.append(res.worst_intraday_dd_pct)
        intra_dds_avg.append(res.avg_intraday_dd_pct)
        tuws.append(res.time_under_water_pct)
        ulcers.append(res.ulcer_index)

    n = len(monthlies)
    if n == 0:
        empty = CellResult(leverage, min_score, hold_through, top_n, fee_regime,
                           0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0)
        empty.fill_buffer_bps = fb_resolved
        return empty

    arr = np.array(monthlies)
    p10 = float(np.percentile(arr, 10))
    worst_dd = float(np.max(dds))
    n_neg = int(np.sum(arr < 0))
    goodness = compute_goodness(p10, worst_dd, n_neg, n)
    robust_goodness = compute_robust_goodness(arr, worst_dd)
    worst_intra = float(np.max(intra_dds_worst)) if intra_dds_worst else 0.0
    avg_intra   = float(np.mean(intra_dds_avg))  if intra_dds_avg   else 0.0
    tuw_med     = float(np.median(tuws))    if tuws    else 0.0
    ulcer_med   = float(np.median(ulcers))  if ulcers  else 0.0
    return CellResult(
        leverage=leverage,
        min_score=min_score,
        hold_through=hold_through,
        top_n=top_n,
        fee_regime=fee_regime,
        n_windows=n,
        median_monthly_pct=float(np.median(arr)),
        p10_monthly_pct=p10,
        median_sortino=float(np.median(sortinos)),
        worst_dd_pct=worst_dd,
        n_neg=n_neg,
        goodness_score=goodness,
        robust_goodness_score=robust_goodness,
        worst_intraday_dd_pct=worst_intra,
        avg_intraday_dd_pct=avg_intra,
        time_under_water_pct=tuw_med,
        ulcer_index=ulcer_med,
        inference_min_dolvol=float(inference_min_dolvol),
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
    )


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
    blend_mode: str = "mean",
    chronos_cache_path: Path | None = None,
    min_dollar_vol: float = 5_000_000.0,
    inference_min_dolvol_grid: list[float] | None = None,
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
) -> list[CellResult]:
    """Run the full sweep. Returns a flat list of CellResult."""
    for p in model_paths:
        if not p.exists():
            raise FileNotFoundError(f"model path not found: {p}")
    for reg in fee_regimes:
        if reg not in FEE_REGIMES:
            raise ValueError(f"unknown fee regime: {reg}. "
                             f"Known: {list(FEE_REGIMES)}")

    chronos_cache = {}
    if chronos_cache_path is not None and chronos_cache_path.exists():
        chronos_cache = load_chronos_cache(chronos_cache_path)

    # Load models first so we can peek at their feature_cols and decide
    # whether the dataset needs cross-sectional ranks attached.
    models: list[XGBStockModel] = []
    for p in model_paths:
        logger.info("loading %s", p)
        models.append(XGBStockModel.load(p))
    def _model_has_ranks(m) -> bool:
        fc = getattr(m, "feature_cols", None) or []
        return any(c in fc for c in DAILY_RANK_FEATURE_COLS)

    have_ranks = [_model_has_ranks(m) for m in models]
    needs_ranks = any(have_ranks)
    # All models must agree — a mixed ensemble (some with ranks, some without)
    # would silently produce different feature sets per member. Reject.
    if any(have_ranks) and not all(have_ranks):
        raise ValueError(
            "Ensemble mixes rank-trained and non-rank-trained models. "
            f"feature_cols per model: "
            f"{[len(m.feature_cols) for m in models]}"
        )
    logger.info("ensemble feature-mode: ranks=%s", needs_ranks)

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
        fast_features=False,
        include_cross_sectional_ranks=needs_ranks,
    )
    logger.info("dataset built in %.1fs | train=%d oos=%d",
                time.perf_counter() - _t, len(train_df), len(oos_df))

    scores = _blend_scores(oos_df, models, blend_mode)
    if invert_scores:
        # Flip rank-order so "top-N" becomes the worst-scored names. Kept
        # in [0,1] so existing min_score gates stay meaningful on the
        # inverted distribution (callers should LOWER the gate).
        logger.info("invert_scores=True → flipping score ranks for short-side test")
        scores = (1.0 - scores).rename("ensemble_score_inv")

    all_days = sorted(oos_df["date"].unique())
    windows = _build_windows(all_days, window_days, stride_days)
    if not windows:
        raise RuntimeError("no eval windows — check OOS date range")

    inf_grid = list(inference_min_dolvol_grid) if inference_min_dolvol_grid else [float(min_dollar_vol)]
    vol_grid = list(inference_min_vol_grid) if inference_min_vol_grid else [0.0]
    maxvol_grid = list(inference_max_vol_grid) if inference_max_vol_grid else [0.0]
    sp_grid  = [float(x) for x in (skip_prob_grid or [0.0])]
    ss_list  = [int(x) for x in (skip_seeds or [0])]
    # -1.0 sentinel = "use regime default fill_buffer_bps".
    fb_grid  = [float(x) for x in (fill_buffer_bps_grid or [-1.0])]
    rgw_grid = [int(x) for x in (regime_gate_window_grid or [0])]
    vta_grid = [float(x) for x in (vol_target_ann_grid or [0.0])]
    ivt_grid = [float(x) for x in (inv_vol_target_grid or [0.0])]
    r20g_grid = [float(x) for x in (max_ret_20d_rank_pct_grid or [1.0])]
    r5g_grid  = [float(x) for x in (min_ret_5d_rank_pct_grid or [0.0])]

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

    cells: list[CellResult] = []
    total = (
        len(leverage_grid) * len(min_score_grid)
        * len(hold_through_grid) * len(top_n_grid) * len(fee_regimes)
        * len(inf_grid) * len(vol_grid) * len(maxvol_grid)
        * len(sp_grid) * len(ss_list) * len(fb_grid)
        * len(rgw_grid) * len(vta_grid) * len(ivt_grid)
        * len(r20g_grid) * len(r5g_grid)
    )
    i = 0
    for lev in leverage_grid:
        for ms in min_score_grid:
            for ht in hold_through_grid:
                for tn in top_n_grid:
                    for reg in fee_regimes:
                        for inf_dv in inf_grid:
                            for inf_vol in vol_grid:
                                for inf_maxvol in maxvol_grid:
                                    for sp in sp_grid:
                                        # Only iterate seeds when skipping;
                                        # at skip=0 the sim is deterministic.
                                        seed_iter = ss_list if sp > 0 else [0]
                                        for sseed in seed_iter:
                                            for fb in fb_grid:
                                                for rgw in rgw_grid:
                                                    for vta in vta_grid:
                                                        for ivt in ivt_grid:
                                                            for r20g in r20g_grid:
                                                                for r5g in r5g_grid:
                                                                    i += 1
                                                                    cell = _run_cell(
                                                                        oos_df=oos_df, scores=scores, windows=windows,
                                                                        leverage=lev, min_score=ms, hold_through=ht,
                                                                        top_n=tn, fee_regime=reg,
                                                                        inference_min_dolvol=inf_dv,
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
                                                                    )
                                                                    logger.info(
                                                                        "cell %d/%d lev=%.2f ms=%.2f ht=%s tn=%d reg=%s "
                                                                        "inf_dv=%.0e vol=[%.3f,%.3f] skp=%.2f/%d fb=%.1f "
                                                                        "rgw=%d vta=%.2f ivt=%.2f r20g=%.2f r5g=%.2f "
                                                                        "med=%+.2f%% p10=%+.2f%% neg=%d/%d",
                                                                        i, total, lev, ms, ht, tn, reg,
                                                                        inf_dv, inf_vol, inf_maxvol, sp, sseed,
                                                                        cell.fill_buffer_bps,
                                                                        rgw, vta, ivt, r20g, r5g,
                                                                        cell.median_monthly_pct, cell.p10_monthly_pct,
                                                                        cell.n_neg, cell.n_windows,
                                                                    )
                                                                    cells.append(cell)
    return cells


def _cells_to_rows(cells: list[CellResult]) -> list[dict]:
    return [
        {
            "leverage": c.leverage, "min_score": c.min_score,
            "hold_through": c.hold_through, "top_n": c.top_n,
            "fee_regime": c.fee_regime,
            "n_windows": c.n_windows,
            "median_monthly_pct": c.median_monthly_pct,
            "p10_monthly_pct": c.p10_monthly_pct,
            "median_sortino": c.median_sortino,
            "worst_dd_pct": c.worst_dd_pct,
            "n_neg": c.n_neg,
            "goodness_score": c.goodness_score,
            "robust_goodness_score": c.robust_goodness_score,
            "worst_intraday_dd_pct": c.worst_intraday_dd_pct,
            "avg_intraday_dd_pct":   c.avg_intraday_dd_pct,
            "time_under_water_pct":  c.time_under_water_pct,
            "ulcer_index":           c.ulcer_index,
            "inference_min_dolvol":  c.inference_min_dolvol,
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
        }
        for c in cells
    ]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-trained ensemble grid sweep.")
    p.add_argument("--symbols-file", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path("trainingdata"))
    p.add_argument("--chronos-cache", type=Path,
                   default=Path("analysis/xgbnew_daily/chronos_cache.parquet"))
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
                        "the fee regime's default FB (deploy=5, "
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
    ht_grid: list[bool] = []
    if args.hold_through:
        ht_grid.append(True)
    if args.no_hold_through:
        ht_grid.append(False)
    if not ht_grid:
        ht_grid = [False]

    oos_end = date.fromisoformat(args.oos_end) if args.oos_end else date.today()
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
        fee_regimes=[r.strip() for r in args.fee_regimes.split(",") if r.strip()],
        blend_mode=args.blend_mode,
        chronos_cache_path=args.chronos_cache,
        min_dollar_vol=float(args.min_dollar_vol),
        inference_min_dolvol_grid=(
            _parse_float_list(args.inference_min_dolvol_grid)
            if args.inference_min_dolvol_grid else None
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
    )

    rows = _cells_to_rows(cells)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = args.output_dir / f"sweep_{ts}.json"
    out.write_text(json.dumps({
        "model_paths": [str(p) for p in model_paths],
        "oos_start": args.oos_start, "oos_end": str(oos_end),
        "window_days": int(args.window_days), "stride_days": int(args.stride_days),
        "fee_regimes": {k: FEE_REGIMES[k] for k in
                        [r.strip() for r in args.fee_regimes.split(",")]},
        "goodness_weights": GOODNESS_WEIGHTS,
        "robust_goodness_weights": ROBUST_GOODNESS_WEIGHTS,
        "cells": rows,
    }, indent=2))
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
    hdr = (f"\n{'lev':>5} {'ms':>5} {'ht':>3} {'tn':>3} {'reg':>10} "
           f"{'med%':>8} {'p10':>8} {'sort':>6} {'ddW':>6} {'idW':>6} "
           f"{'tuw%':>6} {'ulc':>6} {'neg':>6} {'good':>8} {'robG':>8}")
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
                 + (9 if inf_grid_active else 0)
                 + (8 if vol_grid_active else 0)
                 + (9 if sp_grid_active else 0)
                 + (6 if fb_grid_active else 0)))
    for r in rows_sorted:
        line = (f"{r['leverage']:5.2f} {r['min_score']:5.2f} "
                f"{'Y' if r['hold_through'] else 'N':>3} {r['top_n']:3d} "
                f"{r['fee_regime']:>10} "
                f"{r['median_monthly_pct']:+8.2f} {r['p10_monthly_pct']:+8.2f} "
                f"{r['median_sortino']:6.2f} {r['worst_dd_pct']:6.2f} "
                f"{r['worst_intraday_dd_pct']:6.2f} "
                f"{r['time_under_water_pct']:6.2f} "
                f"{r['ulcer_index']:6.2f} "
                f"{r['n_neg']:3d}/{r['n_windows']:3d} "
                f"{r['goodness_score']:+8.2f} "
                f"{r['robust_goodness_score']:+8.2f}")
        if inf_grid_active:
            line += f" {r['inference_min_dolvol']:8.2e}"
        if vol_grid_active:
            line += f" {r['inference_min_vol_20d']:7.3f}"
        if sp_grid_active:
            line += f" {r['skip_prob']:5.2f} {r['skip_seed']:3d}"
        if fb_grid_active:
            line += f" {r['fill_buffer_bps']:5.1f}"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
