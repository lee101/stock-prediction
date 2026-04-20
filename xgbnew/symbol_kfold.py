"""Symbol-level leave-one-bucket-out (LOBO) evaluation.

Answers "which chunks of the universe actually help or hurt?" by taking
a pre-trained ensemble, splitting the symbol universe into K buckets,
then re-running the windowed OOS backtest K+1 times:

  baseline  : full universe available each day
  bucket k  : held out — only symbols NOT in bucket k are pickable that day

The Δ against baseline is the bucket's *net contribution* at the current
config. A positive Δ (LOBO better than baseline) means that bucket is a
net drag on the live signal — an alpha-killer. A negative Δ means the
bucket is genuinely helping.

Bucketing strategies:
  * "alpha"       — deterministic alphabetical chunks (default, easy).
  * "liquidity"   — quartiles of 20-day log dollar volume (computed on
                    the first OOS day so buckets don't shift mid-run).
  * "volatility"  — quartiles of 20-day realised vol (same anchor).

This is CHEAP because the ensemble forward pass runs ONCE — every LOBO
cell just re-runs the windowed simulator over a masked score Series.

Usage (identical boilerplate to ``sweep_ensemble_grid``)::

    python -m xgbnew.symbol_kfold \\
        --symbols-file symbol_lists/stocks_wide_1000_v1.txt \\
        --data-root trainingdata \\
        --model-paths 'analysis/xgbnew_daily/alltrain_ensemble_gpu/alltrain_seed*.pkl' \\
        --oos-start 2025-01-02 --oos-end 2026-04-19 \\
        --leverage 2.0 --min-score 0.85 --hold-through \\
        --n-buckets 4 --bucket-mode liquidity \\
        --fee-regime deploy \\
        --output-dir analysis/xgbnew_symbol_kfold
"""
from __future__ import annotations

import argparse
import glob
import hashlib
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
from xgbnew.sweep_ensemble_grid import (
    FEE_REGIMES,
    _blend_scores,
    _build_windows,
    _monthly_return,
    compute_goodness,
)

logger = logging.getLogger(__name__)


@dataclass
class LOBOResult:
    bucket_id: int            # -1 = baseline (full universe)
    bucket_label: str
    bucket_symbols: list[str]
    n_symbols_in_bucket: int
    n_windows: int
    median_monthly_pct: float
    p10_monthly_pct: float
    median_sortino: float
    worst_dd_pct: float
    worst_intraday_dd_pct: float
    n_neg: int
    goodness_score: float


def _parse_symbols_file(path: Path) -> list[str]:
    out: list[str] = []
    with open(path) as f:
        for ln in f:
            s = ln.strip()
            if s and not s.startswith("#"):
                out.append(s)
    return out


def _resolve_model_paths(spec: str) -> list[Path]:
    """Accept comma-separated and glob patterns (e.g. ``'*/alltrain_seed*.pkl'``)."""
    out: list[Path] = []
    for tok in (s.strip() for s in spec.split(",")):
        if not tok:
            continue
        # If the token looks like a glob, expand it.
        if any(ch in tok for ch in ["*", "?", "["]):
            matches = sorted(glob.glob(tok))
            out.extend(Path(m) for m in matches)
        else:
            out.append(Path(tok))
    # De-dup while preserving order
    seen = set()
    uniq = []
    for p in out:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def _bucket_symbols(
    symbols: list[str],
    oos_df: pd.DataFrame,
    *,
    mode: str,
    n_buckets: int,
    hash_salt: str = "",
) -> dict[int, list[str]]:
    """Return {bucket_id -> [symbols]}. Every symbol assigned to exactly one bucket."""
    uniq = sorted(set(symbols))
    if mode == "alpha":
        # Deterministic hash-based spread so we don't cluster by first letter,
        # which correlates with listing exchange / sector.
        pairs = []
        for s in uniq:
            h = hashlib.sha1((hash_salt + s).encode()).hexdigest()
            pairs.append((int(h[:8], 16), s))
        pairs.sort()
        ordered = [s for _, s in pairs]
    elif mode == "liquidity":
        if "dolvol_20d_log" not in oos_df.columns:
            raise ValueError("liquidity mode needs dolvol_20d_log in oos_df")
        # Use the median across the OOS window per symbol so we don't pick
        # a pathological anchor day. groupby.median handles the duplicate
        # (symbol, date) rows gracefully without a reindex that chokes.
        liq = (oos_df.groupby("symbol")["dolvol_20d_log"]
               .median().reindex(uniq).fillna(-1.0).sort_values())
        ordered = list(liq.index)
    elif mode == "volatility":
        if "vol_20d" not in oos_df.columns:
            raise ValueError("volatility mode needs vol_20d in oos_df")
        v = (oos_df.groupby("symbol")["vol_20d"]
             .median().reindex(uniq).fillna(-1.0).sort_values())
        ordered = list(v.index)
    else:
        raise ValueError(f"unknown bucket mode: {mode}")

    if n_buckets < 2:
        raise ValueError("n_buckets must be >= 2")
    if len(ordered) < n_buckets:
        raise ValueError(f"only {len(ordered)} symbols, can't make {n_buckets} buckets")

    buckets: dict[int, list[str]] = {i: [] for i in range(n_buckets)}
    # np.array_split handles non-divisible nicely.
    chunks = np.array_split(np.asarray(ordered, dtype=object), n_buckets)
    for i, chunk in enumerate(chunks):
        buckets[i] = list(chunk)
    return buckets


def _run_cell(
    oos_df: pd.DataFrame,
    scores: pd.Series,
    windows: list[tuple],
    cfg: BacktestConfig,
    mask_symbols: set | None,
    bucket_id: int,
    bucket_label: str,
    bucket_symbols: list[str],
) -> LOBOResult:
    if mask_symbols:
        # Remove all rows for symbols in the held-out bucket before simulate.
        local_df = oos_df[~oos_df["symbol"].isin(mask_symbols)]
        local_scores = scores.loc[local_df.index]
    else:
        local_df = oos_df
        local_scores = scores

    dummy = XGBStockModel(device="cpu", n_estimators=1, max_depth=1, learning_rate=0.1)
    # Feature schema isn't consulted because we pass precomputed_scores,
    # but simulate checks the dummy model's fitted flag.
    dummy.feature_cols = DAILY_FEATURE_COLS
    dummy._col_medians = np.zeros(len(DAILY_FEATURE_COLS), dtype=np.float32)
    dummy._fitted = True

    monthlies: list[float] = []
    sortinos:  list[float] = []
    dds:       list[float] = []
    intra_dds: list[float] = []
    for w_start, w_end in windows:
        w_df = local_df[(local_df["date"] >= w_start) & (local_df["date"] <= w_end)]
        if len(w_df) < 5:
            continue
        w_scores = local_scores.loc[w_df.index]
        res = simulate(w_df, dummy, cfg, precomputed_scores=w_scores)
        n_days = len(res.day_results)
        monthly = _monthly_return(res.total_return_pct, max(n_days, 1)) * 100.0
        monthlies.append(monthly)
        sortinos.append(res.sortino_ratio)
        dds.append(res.max_drawdown_pct)
        intra_dds.append(res.worst_intraday_dd_pct)

    n = len(monthlies)
    if n == 0:
        return LOBOResult(
            bucket_id=bucket_id, bucket_label=bucket_label,
            bucket_symbols=bucket_symbols,
            n_symbols_in_bucket=len(bucket_symbols),
            n_windows=0, median_monthly_pct=0.0, p10_monthly_pct=0.0,
            median_sortino=0.0, worst_dd_pct=0.0, worst_intraday_dd_pct=0.0,
            n_neg=0, goodness_score=0.0,
        )
    arr = np.asarray(monthlies)
    p10 = float(np.percentile(arr, 10))
    worst_dd = float(np.max(dds))
    n_neg = int(np.sum(arr < 0))
    worst_intra = float(np.max(intra_dds)) if intra_dds else 0.0
    return LOBOResult(
        bucket_id=bucket_id, bucket_label=bucket_label,
        bucket_symbols=bucket_symbols,
        n_symbols_in_bucket=len(bucket_symbols),
        n_windows=n,
        median_monthly_pct=float(np.median(arr)),
        p10_monthly_pct=p10,
        median_sortino=float(np.median(sortinos)),
        worst_dd_pct=worst_dd,
        worst_intraday_dd_pct=worst_intra,
        n_neg=n_neg,
        goodness_score=compute_goodness(p10, worst_dd, n_neg, n),
    )


def run_kfold(
    *,
    symbols: list[str],
    data_root: Path,
    model_paths: list[Path],
    train_start: date, train_end: date,
    oos_start: date, oos_end: date,
    window_days: int, stride_days: int,
    leverage: float, min_score: float, hold_through: bool, top_n: int,
    fee_regime: str,
    n_buckets: int, bucket_mode: str,
    blend_mode: str = "mean",
    chronos_cache_path: Path | None = None,
    min_dollar_vol: float = 5_000_000.0,
) -> list[LOBOResult]:
    chronos_cache = {}
    if chronos_cache_path is not None and chronos_cache_path.exists():
        chronos_cache = load_chronos_cache(chronos_cache_path)

    models = [XGBStockModel.load(p) for p in model_paths]
    logger.info("loaded %d ensemble members", len(models))

    def _has_ranks(m) -> bool:
        fc = getattr(m, "feature_cols", None) or []
        return any(c in fc for c in DAILY_RANK_FEATURE_COLS)
    has_ranks = [_has_ranks(m) for m in models]
    if any(has_ranks) and not all(has_ranks):
        raise ValueError("mixed rank/no-rank ensemble — retrain all members")
    needs_ranks = any(has_ranks)

    train_df, _, oos_df = build_daily_dataset(
        data_root=data_root,
        symbols=symbols,
        train_start=train_start, train_end=train_end,
        val_start=oos_start, val_end=oos_end,
        test_start=oos_start, test_end=oos_end,
        chronos_cache=chronos_cache if chronos_cache else None,
        min_dollar_vol=min_dollar_vol,
        fast_features=False,
        include_cross_sectional_ranks=needs_ranks,
    )
    logger.info("oos rows=%d unique_symbols=%d",
                len(oos_df), oos_df["symbol"].nunique())

    scores = _blend_scores(oos_df, models, blend_mode)

    all_days = sorted(oos_df["date"].unique())
    windows = _build_windows(all_days, window_days, stride_days)
    if not windows:
        raise RuntimeError("no eval windows")

    fees = FEE_REGIMES[fee_regime]
    cfg = BacktestConfig(
        top_n=int(top_n), leverage=float(leverage),
        fee_rate=float(fees["fee_rate"]),
        fill_buffer_bps=float(fees["fill_buffer_bps"]),
        commission_bps=float(fees["commission_bps"]),
        min_dollar_vol=float(min_dollar_vol),
        hold_through=bool(hold_through),
        min_score=float(min_score),
        xgb_weight=1.0,
    )

    results: list[LOBOResult] = []

    # Baseline (nothing masked).
    logger.info("baseline cell (no mask)")
    results.append(_run_cell(
        oos_df=oos_df, scores=scores, windows=windows, cfg=cfg,
        mask_symbols=None, bucket_id=-1, bucket_label="baseline",
        bucket_symbols=sorted(oos_df["symbol"].unique().tolist()),
    ))

    buckets = _bucket_symbols(
        list(oos_df["symbol"].unique()),
        oos_df, mode=bucket_mode, n_buckets=n_buckets,
    )
    for bid, syms in buckets.items():
        label = f"{bucket_mode}_bucket_{bid}"
        logger.info("LOBO cell bucket=%s n_syms=%d", label, len(syms))
        results.append(_run_cell(
            oos_df=oos_df, scores=scores, windows=windows, cfg=cfg,
            mask_symbols=set(syms),
            bucket_id=bid, bucket_label=label, bucket_symbols=syms,
        ))
    return results


def _rows(results: list[LOBOResult]) -> list[dict]:
    return [
        {
            "bucket_id": r.bucket_id, "bucket_label": r.bucket_label,
            "n_symbols_in_bucket": r.n_symbols_in_bucket,
            "bucket_symbols": r.bucket_symbols,
            "n_windows": r.n_windows,
            "median_monthly_pct": r.median_monthly_pct,
            "p10_monthly_pct": r.p10_monthly_pct,
            "median_sortino": r.median_sortino,
            "worst_dd_pct": r.worst_dd_pct,
            "worst_intraday_dd_pct": r.worst_intraday_dd_pct,
            "n_neg": r.n_neg,
            "goodness_score": r.goodness_score,
        }
        for r in results
    ]


def _print_table(results: list[LOBOResult]) -> None:
    baseline = next((r for r in results if r.bucket_id == -1), None)
    if baseline is None:
        return
    print(f"\n{'bucket':>24} {'nsym':>5} {'med%':>8} {'Δmed':>7} "
          f"{'p10':>8} {'Δp10':>7} {'ddW':>6} {'idW':>6} {'good':>8} {'Δgood':>7}")
    print("-" * 104)
    # Baseline row first
    print(f"{baseline.bucket_label:>24} {baseline.n_symbols_in_bucket:5d} "
          f"{baseline.median_monthly_pct:+8.2f} {'':>7} "
          f"{baseline.p10_monthly_pct:+8.2f} {'':>7} "
          f"{baseline.worst_dd_pct:6.2f} {baseline.worst_intraday_dd_pct:6.2f} "
          f"{baseline.goodness_score:+8.2f} {'':>7}")
    for r in results:
        if r.bucket_id == -1:
            continue
        dmed  = r.median_monthly_pct - baseline.median_monthly_pct
        dp10  = r.p10_monthly_pct - baseline.p10_monthly_pct
        dgood = r.goodness_score - baseline.goodness_score
        print(f"{r.bucket_label:>24} {r.n_symbols_in_bucket:5d} "
              f"{r.median_monthly_pct:+8.2f} {dmed:+7.2f} "
              f"{r.p10_monthly_pct:+8.2f} {dp10:+7.2f} "
              f"{r.worst_dd_pct:6.2f} {r.worst_intraday_dd_pct:6.2f} "
              f"{r.goodness_score:+8.2f} {dgood:+7.2f}")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Leave-one-bucket-out symbol k-fold.")
    p.add_argument("--symbols-file", type=Path, required=True)
    p.add_argument("--data-root",    type=Path, default=Path("trainingdata"))
    p.add_argument("--chronos-cache", type=Path,
                   default=Path("analysis/xgbnew_daily/chronos_cache.parquet"))
    p.add_argument("--model-paths", type=str, required=True,
                   help="Comma-sep paths; globs OK")
    p.add_argument("--blend-mode", choices=["mean", "median"], default="mean")
    p.add_argument("--train-start", default="2020-01-01")
    p.add_argument("--train-end",   default="2024-12-31")
    p.add_argument("--oos-start",   default="2025-01-02")
    p.add_argument("--oos-end",     default="")
    p.add_argument("--window-days", type=int, default=30)
    p.add_argument("--stride-days", type=int, default=7)
    p.add_argument("--leverage",    type=float, default=2.0)
    p.add_argument("--min-score",   type=float, default=0.85)
    p.add_argument("--hold-through", action="store_true")
    p.add_argument("--top-n",       type=int, default=1)
    p.add_argument("--fee-regime",  default="deploy",
                   choices=list(FEE_REGIMES))
    p.add_argument("--n-buckets",   type=int, default=4)
    p.add_argument("--bucket-mode", default="liquidity",
                   choices=["alpha", "liquidity", "volatility"])
    p.add_argument("--min-dollar-vol", type=float, default=5_000_000.0)
    p.add_argument("--output-dir",  type=Path,
                   default=Path("analysis/xgbnew_symbol_kfold"))
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")

    symbols = _parse_symbols_file(args.symbols_file)
    model_paths = _resolve_model_paths(args.model_paths)
    if not model_paths:
        print("ERROR: no model paths resolved", file=sys.stderr)
        return 1

    oos_end = date.fromisoformat(args.oos_end) if args.oos_end else date.today()
    t0 = time.perf_counter()
    results = run_kfold(
        symbols=symbols,
        data_root=args.data_root,
        model_paths=model_paths,
        train_start=date.fromisoformat(args.train_start),
        train_end=date.fromisoformat(args.train_end),
        oos_start=date.fromisoformat(args.oos_start),
        oos_end=oos_end,
        window_days=int(args.window_days), stride_days=int(args.stride_days),
        leverage=float(args.leverage), min_score=float(args.min_score),
        hold_through=bool(args.hold_through), top_n=int(args.top_n),
        fee_regime=str(args.fee_regime),
        n_buckets=int(args.n_buckets), bucket_mode=str(args.bucket_mode),
        blend_mode=str(args.blend_mode),
        chronos_cache_path=args.chronos_cache,
        min_dollar_vol=float(args.min_dollar_vol),
    )
    logger.info("total walltime %.1fs", time.perf_counter() - t0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = args.output_dir / f"kfold_{ts}.json"
    out.write_text(json.dumps({
        "model_paths": [str(p) for p in model_paths],
        "oos_start": args.oos_start, "oos_end": str(oos_end),
        "leverage": args.leverage, "min_score": args.min_score,
        "hold_through": args.hold_through, "top_n": args.top_n,
        "fee_regime": args.fee_regime,
        "n_buckets": args.n_buckets, "bucket_mode": args.bucket_mode,
        "results": _rows(results),
    }, indent=2))
    print(f"[kfold] wrote {out}  ({len(results)} cells)", flush=True)

    _print_table(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
