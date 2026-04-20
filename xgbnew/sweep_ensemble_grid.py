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
) -> CellResult:
    fees = FEE_REGIMES[fee_regime]
    cfg = BacktestConfig(
        top_n=int(top_n),
        leverage=float(leverage),
        xgb_weight=1.0,
        fee_rate=float(fees["fee_rate"]),
        fill_buffer_bps=float(fees["fill_buffer_bps"]),
        commission_bps=float(fees["commission_bps"]),
        min_dollar_vol=5_000_000.0,
        hold_through=bool(hold_through),
        min_score=float(min_score),
    )

    dummy = XGBStockModel(device="cpu", n_estimators=1, max_depth=1, learning_rate=0.1)
    dummy.feature_cols = DAILY_FEATURE_COLS
    dummy._col_medians = np.zeros(len(DAILY_FEATURE_COLS), dtype=np.float32)
    dummy._fitted = True

    monthlies: list[float] = []
    sortinos: list[float] = []
    dds: list[float] = []
    for w_start, w_end in windows:
        w_df = oos_df[(oos_df["date"] >= w_start) & (oos_df["date"] <= w_end)]
        if len(w_df) < 5:
            continue
        w_scores = scores.loc[w_df.index]
        res = simulate(w_df, dummy, cfg, precomputed_scores=w_scores)
        n_days = len(res.day_results)
        monthly = _monthly_return(res.total_return_pct, max(n_days, 1)) * 100.0
        monthlies.append(monthly)
        sortinos.append(res.sortino_ratio)
        dds.append(res.max_drawdown_pct)

    n = len(monthlies)
    if n == 0:
        return CellResult(leverage, min_score, hold_through, top_n, fee_regime,
                          0, 0.0, 0.0, 0.0, 0.0, 0)

    arr = np.array(monthlies)
    return CellResult(
        leverage=leverage,
        min_score=min_score,
        hold_through=hold_through,
        top_n=top_n,
        fee_regime=fee_regime,
        n_windows=n,
        median_monthly_pct=float(np.median(arr)),
        p10_monthly_pct=float(np.percentile(arr, 10)),
        median_sortino=float(np.median(sortinos)),
        worst_dd_pct=float(np.max(dds)),
        n_neg=int(np.sum(arr < 0)),
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

    all_days = sorted(oos_df["date"].unique())
    windows = _build_windows(all_days, window_days, stride_days)
    if not windows:
        raise RuntimeError("no eval windows — check OOS date range")

    cells: list[CellResult] = []
    total = (
        len(leverage_grid) * len(min_score_grid)
        * len(hold_through_grid) * len(top_n_grid) * len(fee_regimes)
    )
    i = 0
    for lev in leverage_grid:
        for ms in min_score_grid:
            for ht in hold_through_grid:
                for tn in top_n_grid:
                    for reg in fee_regimes:
                        i += 1
                        cell = _run_cell(
                            oos_df=oos_df, scores=scores, windows=windows,
                            leverage=lev, min_score=ms, hold_through=ht,
                            top_n=tn, fee_regime=reg,
                        )
                        logger.info(
                            "cell %d/%d lev=%.2f ms=%.2f ht=%s tn=%d reg=%s "
                            "med=%+.2f%% p10=%+.2f%% neg=%d/%d",
                            i, total, lev, ms, ht, tn, reg,
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
    p.add_argument("--min-dollar-vol", type=float, default=5_000_000.0)
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
        "cells": rows,
    }, indent=2))
    print(f"[sweep] wrote {out}  ({len(rows)} cells)", flush=True)

    # Pretty table
    print(f"\n{'lev':>5} {'ms':>5} {'ht':>3} {'tn':>3} {'reg':>10} "
          f"{'med%':>8} {'p10':>8} {'sort':>6} {'ddW':>6} {'neg':>6}")
    print("-" * 74)
    for r in rows:
        print(f"{r['leverage']:5.2f} {r['min_score']:5.2f} "
              f"{'Y' if r['hold_through'] else 'N':>3} {r['top_n']:3d} "
              f"{r['fee_regime']:>10} "
              f"{r['median_monthly_pct']:+8.2f} {r['p10_monthly_pct']:+8.2f} "
              f"{r['median_sortino']:6.2f} {r['worst_dd_pct']:6.2f} "
              f"{r['n_neg']:3d}/{r['n_windows']:3d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
