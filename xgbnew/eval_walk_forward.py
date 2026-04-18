#!/usr/bin/env python3
"""Walk-forward OOS evaluator for the XGBoost daily open→close strategy.

Motivation
----------
``eval_multiwindow.py`` trains the model once on ``train_start..train_end``
and then evaluates rolling windows in the OOS span. The hyperparameter /
top-n search picks the best config on those same windows — which leaks
information *across windows* (a config that accidentally aligns with
post-2024 regime shifts gets rewarded without being penalised for it).

This harness is the true out-of-sample test: for every OOS fold
[t0, t1] we re-fit the model on only data strictly *before* t0, then
simulate the fold using that fresh model. The concatenated daily
returns are analysed as one equity curve. No config selection happens
inside the harness, so there is no cross-fold leakage.

Pure-function core
------------------
- ``build_refit_schedule(oos_start, oos_end, stride_days)`` — deterministic
  list of [fold_start, fold_end] closed intervals covering the OOS span.
- ``run_walk_forward_core(feat_df, schedule, ...)`` — iterates the
  schedule, re-fits an ``XGBStockModel`` per fold, and returns a
  ``WalkForwardResult`` with the concatenated daily returns and per-fold
  metadata.

Correctness invariants enforced by tests
----------------------------------------
1. Every training slice has ``max(date) < fold_start`` (no look-ahead).
2. ``train_window_days`` truly slides: slice size is bounded.
3. Running the harness on a fully-random target produces a Sharpe ≈ 0
   monthly distribution (sanity check for the equity aggregator).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from xgbnew.backtest import BacktestConfig, DayResult, simulate
from xgbnew.features import DAILY_FEATURE_COLS
from xgbnew.model import XGBStockModel

logger = logging.getLogger(__name__)


# ─── Scheduling ──────────────────────────────────────────────────────────────

def build_refit_schedule(
    oos_start: date,
    oos_end: date,
    refit_stride_days: int,
) -> list[tuple[date, date]]:
    """Return [(fold_start, fold_end), ...] closed intervals covering the span.

    Folds are contiguous calendar-day slices of length ``refit_stride_days``.
    The final fold may be shorter than stride if the span doesn't divide evenly.
    """
    if refit_stride_days <= 0:
        raise ValueError(f"refit_stride_days must be >0, got {refit_stride_days}")
    if oos_end < oos_start:
        raise ValueError(f"oos_end {oos_end} precedes oos_start {oos_start}")

    folds: list[tuple[date, date]] = []
    cur = oos_start
    step = timedelta(days=int(refit_stride_days))
    while cur <= oos_end:
        fold_end = min(cur + step - timedelta(days=1), oos_end)
        folds.append((cur, fold_end))
        cur = fold_end + timedelta(days=1)
    return folds


# ─── Result types ────────────────────────────────────────────────────────────

@dataclass
class FoldResult:
    fold_start: date
    fold_end: date
    train_rows: int
    train_date_min: date | None
    train_date_max: date | None
    oos_rows: int
    oos_days_traded: int
    monthly_return_pct: float
    day_results: list[DayResult] = field(default_factory=list)


@dataclass
class WalkForwardResult:
    folds: list[FoldResult]
    # Concatenated daily returns across all folds, in calendar order.
    daily_returns: pd.Series  # index=date, values=daily_return_pct/100 (fractional)
    fold_monthly_returns_pct: list[float]
    median_fold_monthly_pct: float
    p10_fold_monthly_pct: float
    n_neg_folds: int
    n_folds: int

    def summary_dict(self) -> dict:
        return {
            "n_folds": self.n_folds,
            "n_neg_folds": self.n_neg_folds,
            "median_fold_monthly_pct": self.median_fold_monthly_pct,
            "p10_fold_monthly_pct": self.p10_fold_monthly_pct,
            "fold_monthly_returns_pct": list(self.fold_monthly_returns_pct),
        }


# ─── Train/test slicing ──────────────────────────────────────────────────────

def _train_slice(
    feat_df: pd.DataFrame,
    fold_start: date,
    train_window_days: int | None,
) -> pd.DataFrame:
    """Return rows with date strictly < fold_start.

    If ``train_window_days`` is set, also constrain to
    ``date >= fold_start - train_window_days``.
    """
    mask = feat_df["date"] < fold_start
    slice_ = feat_df[mask]
    if train_window_days is not None and not slice_.empty:
        floor = fold_start - timedelta(days=int(train_window_days))
        slice_ = slice_[slice_["date"] >= floor]
    return slice_


def _fold_slice(feat_df: pd.DataFrame, fold_start: date, fold_end: date) -> pd.DataFrame:
    return feat_df[(feat_df["date"] >= fold_start) & (feat_df["date"] <= fold_end)]


# ─── Core walk-forward ───────────────────────────────────────────────────────

def run_walk_forward_core(
    feat_df: pd.DataFrame,
    schedule: Sequence[tuple[date, date]],
    *,
    xgb_params: dict,
    backtest_cfg: BacktestConfig,
    feature_cols: Sequence[str] = DAILY_FEATURE_COLS,
    min_train_rows: int = 500,
    train_window_days: int | None = None,
    device: str | None = None,
    progress: bool = False,
) -> WalkForwardResult:
    """Run the walk-forward loop and return aggregate + per-fold metrics.

    For each fold in ``schedule``:
      1. ``train_slice`` = rows with ``date < fold_start`` (optionally capped
         by ``train_window_days`` for a sliding window).
      2. Skip the fold if ``len(train_slice) < min_train_rows``.
      3. Fit a fresh ``XGBStockModel(**xgb_params, device=device)`` on it.
      4. Simulate the fold with ``simulate(...)`` and capture per-day results.

    The concatenated daily-return series is used to aggregate monthly stats
    (median, p10, neg count) across folds.
    """
    if "date" not in feat_df.columns:
        raise ValueError("feat_df must have a 'date' column")
    if "target_oc_up" not in feat_df.columns:
        raise ValueError("feat_df must have 'target_oc_up' (fit target)")

    fold_results: list[FoldResult] = []
    all_daily_returns: dict[date, float] = {}

    for i, (fold_start, fold_end) in enumerate(schedule):
        tr = _train_slice(feat_df, fold_start, train_window_days)
        oos = _fold_slice(feat_df, fold_start, fold_end)

        tr_max_date = pd.to_datetime(tr["date"]).max().date() if len(tr) else None
        tr_min_date = pd.to_datetime(tr["date"]).min().date() if len(tr) else None

        # Hard invariant — no look-ahead.
        if tr_max_date is not None and tr_max_date >= fold_start:
            raise RuntimeError(
                f"walk-forward leak: train max date {tr_max_date} >= fold_start {fold_start}"
            )

        if len(tr) < min_train_rows or len(oos) == 0:
            fold_results.append(FoldResult(
                fold_start=fold_start, fold_end=fold_end,
                train_rows=len(tr), train_date_min=tr_min_date, train_date_max=tr_max_date,
                oos_rows=len(oos), oos_days_traded=0, monthly_return_pct=0.0,
            ))
            if progress:
                print(
                    f"[wfwd] fold {i+1}/{len(schedule)} {fold_start}..{fold_end} "
                    f"SKIP train={len(tr)} rows oos={len(oos)} rows",
                    flush=True,
                )
            continue

        model = XGBStockModel(device=device, **xgb_params)
        model.fit(tr, feature_cols, verbose=False)
        bt = simulate(oos, model, backtest_cfg)

        for day_res in bt.day_results:
            all_daily_returns[day_res.day] = day_res.daily_return_pct / 100.0

        fold_results.append(FoldResult(
            fold_start=fold_start, fold_end=fold_end,
            train_rows=len(tr), train_date_min=tr_min_date, train_date_max=tr_max_date,
            oos_rows=len(oos), oos_days_traded=len(bt.day_results),
            monthly_return_pct=bt.monthly_return_pct,
            day_results=list(bt.day_results),
        ))

        if progress:
            print(
                f"[wfwd] fold {i+1}/{len(schedule)} {fold_start}..{fold_end} "
                f"train={len(tr)} ({tr_min_date}..{tr_max_date}) "
                f"oos_days={len(bt.day_results)} monthly={bt.monthly_return_pct:+.2f}%",
                flush=True,
            )

    # Aggregate
    if all_daily_returns:
        ret_series = pd.Series(all_daily_returns).sort_index()
    else:
        ret_series = pd.Series(dtype=float)

    fold_monthly = [fr.monthly_return_pct for fr in fold_results if fr.oos_days_traded > 0]
    median_fm = float(np.median(fold_monthly)) if fold_monthly else 0.0
    p10_fm = float(np.percentile(fold_monthly, 10)) if fold_monthly else 0.0
    n_neg = sum(1 for m in fold_monthly if m < 0)

    return WalkForwardResult(
        folds=fold_results,
        daily_returns=ret_series,
        fold_monthly_returns_pct=fold_monthly,
        median_fold_monthly_pct=median_fm,
        p10_fold_monthly_pct=p10_fm,
        n_neg_folds=n_neg,
        n_folds=len([fr for fr in fold_results if fr.oos_days_traded > 0]),
    )


__all__ = [
    "build_refit_schedule",
    "run_walk_forward_core",
    "FoldResult",
    "WalkForwardResult",
]
