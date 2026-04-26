#!/usr/bin/env python3
"""Cache ensemble daily scores and run quick portfolio diagnostics.

The normal sweep entry point is authoritative for promotion gates, but it
rebuilds the OOS feature panel and recomputes every model score per process.
This helper writes that expensive panel once, then evaluates a small set of
strategy hypotheses quickly through the same ``xgbnew.backtest.simulate``
path used by the production sweep.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import pickle
import sys
import time
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from xgbnew.dataset import build_daily_dataset
from xgbnew.features import (
    DAILY_DISPERSION_FEATURE_COLS,
    DAILY_RANK_FEATURE_COLS,
    LIVE_SUPPORTED_FEATURE_COLS,
)
from xgbnew.model_registry import load_any_model
from xgbnew.sweep_ensemble_grid import (
    FEE_REGIMES,
    CellResult,
    _build_windows,
    _friction_robust_strategy_rows,
    _run_cell,
)

LOG = logging.getLogger("xgb_cached_score_diagnostics")


def _parse_date(text: str) -> date:
    return date.fromisoformat(text)


def _parse_float_list(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def _parse_int_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _load_symbols(path: Path) -> list[str]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        sym = line.strip().split("#", 1)[0].strip().upper()
        if sym:
            out.append(sym)
    return out


def _load_models(model_paths: list[Path]) -> list[Any]:
    models = [load_any_model(p) for p in model_paths]
    if not models:
        raise ValueError("no models provided")
    first_features: tuple[str, ...] | None = None
    first_path: Path | None = None
    for model, path in zip(models, model_paths, strict=True):
        raw = getattr(model, "feature_cols", None)
        if not isinstance(raw, (list, tuple)) or not raw:
            raise ValueError(f"{path}: model has no feature_cols")
        features = tuple(str(c) for c in raw)
        unsupported = sorted(set(features) - LIVE_SUPPORTED_FEATURE_COLS)
        if unsupported:
            raise ValueError(f"{path}: unsupported live features {unsupported}")
        if first_features is None:
            first_features = features
            first_path = path
        elif features != first_features:
            raise ValueError(
                f"feature mismatch: {path} differs from {first_path}"
            )
    return models


def _score_matrix(oos_df: pd.DataFrame, models: list[Any]) -> tuple[pd.Series, pd.Series]:
    mat = np.stack(
        [np.asarray(model.predict_scores(oos_df), dtype=np.float64) for model in models],
        axis=0,
    )
    return (
        pd.Series(mat.mean(axis=0), index=oos_df.index, name="ensemble_score"),
        pd.Series(mat.std(axis=0), index=oos_df.index, name="ensemble_score_std"),
    )


def _build_cache(args: argparse.Namespace) -> dict[str, Any]:
    model_paths = [Path(p.strip()) for p in args.model_paths.split(",") if p.strip()]
    models = _load_models(model_paths)
    feature_cols = tuple(getattr(models[0], "feature_cols"))
    needs_ranks = any(c in feature_cols for c in DAILY_RANK_FEATURE_COLS)
    needs_disp = any(c in feature_cols for c in DAILY_DISPERSION_FEATURE_COLS)

    symbols = _load_symbols(args.symbols_file)
    oos_start = _parse_date(args.oos_start)
    oos_end = _parse_date(args.oos_end)
    train_start = _parse_date(args.train_start)
    train_end = oos_start - timedelta(days=1)

    t0 = time.perf_counter()
    _, _, oos_df = build_daily_dataset(
        data_root=args.data_root,
        symbols=symbols,
        train_start=train_start,
        train_end=train_end,
        val_start=oos_start,
        val_end=oos_end,
        test_start=oos_start,
        test_end=oos_end,
        min_dollar_vol=float(args.min_dollar_vol),
        fast_features=bool(args.fast_features),
        include_cross_sectional_ranks=needs_ranks,
        include_cross_sectional_dispersion=needs_disp,
    )
    LOG.info("built OOS panel rows=%d in %.1fs", len(oos_df), time.perf_counter() - t0)

    t0 = time.perf_counter()
    scores, score_std = _score_matrix(oos_df, models)
    LOG.info("scored %d models in %.1fs", len(models), time.perf_counter() - t0)

    payload = {
        "oos_df": oos_df,
        "scores": scores,
        "score_std": score_std,
        "meta": {
            "symbols_file": str(args.symbols_file),
            "model_paths": [str(p) for p in model_paths],
            "oos_start": args.oos_start,
            "oos_end": args.oos_end,
            "min_dollar_vol": float(args.min_dollar_vol),
            "fast_features": bool(args.fast_features),
            "needs_ranks": needs_ranks,
            "needs_dispersion": needs_disp,
        },
    }
    args.cache_path.parent.mkdir(parents=True, exist_ok=True)
    with args.cache_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return payload


def _load_or_build_cache(args: argparse.Namespace) -> dict[str, Any]:
    if args.cache_path.exists() and not args.rebuild_cache:
        with args.cache_path.open("rb") as f:
            return pickle.load(f)
    return _build_cache(args)


def _candidate_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    fee_regimes = [x.strip() for x in args.fee_regimes.split(",") if x.strip()]
    fill_buffers = _parse_float_list(args.fill_buffer_bps_grid)
    for (
        fee_regime,
        fill_buffer_bps,
        leverage,
        min_score,
        top_n,
        min_picks,
        invert,
        uncertainty_penalty,
        allocation_mode,
        allocation_temp,
        fallback_alloc,
        r20_max,
        r5_min,
        cs_iqr,
        cs_skew,
        inv_vol_target,
    ) in itertools.product(
        fee_regimes,
        fill_buffers,
        _parse_float_list(args.leverage_grid),
        _parse_float_list(args.min_score_grid),
        _parse_int_list(args.top_n_grid),
        _parse_int_list(args.min_picks_grid),
        [False, True] if args.include_invert else [False],
        _parse_float_list(args.score_uncertainty_penalty_grid),
        [x.strip() for x in args.allocation_mode_grid.split(",") if x.strip()],
        _parse_float_list(args.allocation_temp_grid),
        _parse_float_list(args.no_picks_fallback_alloc_grid),
        _parse_float_list(args.max_ret_20d_rank_pct_grid),
        _parse_float_list(args.min_ret_5d_rank_pct_grid),
        _parse_float_list(args.regime_cs_iqr_max_grid),
        _parse_float_list(args.regime_cs_skew_min_grid),
        _parse_float_list(args.inv_vol_target_grid),
    ):
        if min_picks > top_n:
            continue
        if allocation_mode != "softmax" and allocation_temp != 1.0:
            continue
        configs.append(
            {
                "fee_regime": fee_regime,
                "fill_buffer_bps": fill_buffer_bps,
                "leverage": leverage,
                "min_score": min_score,
                "top_n": top_n,
                "min_picks": min_picks,
                "invert_scores": invert,
                "score_uncertainty_penalty": uncertainty_penalty,
                "allocation_mode": allocation_mode,
                "allocation_temp": allocation_temp,
                "no_picks_fallback_symbol": (
                    args.no_picks_fallback if fallback_alloc > 0 else ""
                ),
                "no_picks_fallback_alloc_scale": fallback_alloc,
                "max_ret_20d_rank_pct": r20_max,
                "min_ret_5d_rank_pct": r5_min,
                "regime_cs_iqr_max": cs_iqr,
                "regime_cs_skew_min": cs_skew,
                "inv_vol_target_ann": inv_vol_target,
            }
        )
    max_cells = int(args.max_cells)
    if max_cells > 0:
        return configs[:max_cells]
    return configs


def _row(cell: CellResult, extra: dict[str, Any]) -> dict[str, Any]:
    out = asdict(cell)
    out.update(extra)
    return out


def _write_output(args: argparse.Namespace, payload: dict[str, Any], rows: list[dict[str, Any]], *, complete: bool) -> None:
    summaries = _friction_robust_strategy_rows(rows)
    out = {
        "complete": complete,
        "cache_path": str(args.cache_path),
        "cache_meta": payload.get("meta", {}),
        "n_rows": len(rows),
        "n_friction_robust_strategies": len(summaries),
        "best_friction_robust": summaries[0] if summaries else None,
        "friction_robust_strategies": summaries,
        "rows": rows,
        "fee_regimes": {k: FEE_REGIMES[k] for k in args.fee_regimes.split(",") if k},
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")


def _evaluate(args: argparse.Namespace, payload: dict[str, Any]) -> list[dict[str, Any]]:
    oos_df: pd.DataFrame = payload["oos_df"]
    scores: pd.Series = payload["scores"]
    score_std: pd.Series = payload["score_std"]
    days = sorted(oos_df["date"].unique())
    windows = _build_windows(days, int(args.window_days), int(args.stride_days))
    if not windows:
        raise RuntimeError("no windows for requested OOS range")

    configs = _candidate_grid(args)
    rows: list[dict[str, Any]] = []
    for i, cfg in enumerate(configs, start=1):
        base_scores = 1.0 - scores if cfg.pop("invert_scores") else scores
        penalty = float(cfg["score_uncertainty_penalty"])
        cell_scores = base_scores - penalty * score_std.reindex(base_scores.index).fillna(0.0)
        cell = _run_cell(
            oos_df=oos_df,
            scores=cell_scores,
            windows=windows,
            hold_through=True,
            inference_min_dolvol=float(args.inference_min_dolvol),
            inference_min_vol_20d=float(args.inference_min_vol),
            inference_max_vol_20d=float(args.inference_max_vol),
            fail_fast_max_dd_pct=float(args.fail_fast_max_dd_pct),
            fail_fast_neg_windows=int(args.fail_fast_neg_windows),
            **cfg,
        )
        rows.append(_row(cell, {"candidate_index": i}))
        if i % max(int(args.progress_every), 1) == 0:
            best = max(
                rows,
                key=lambda r: (
                    r.get("pain_adjusted_goodness_score", -1e9),
                    r.get("p10_monthly_pct", -1e9),
                ),
            )
            LOG.info(
                "evaluated %d/%d best med=%+.2f p10=%+.2f dd=%.2f neg=%d",
                i,
                len(configs),
                best["median_monthly_pct"],
                best["p10_monthly_pct"],
                best["worst_dd_pct"],
                best["n_neg"],
            )
        if int(args.checkpoint_every) > 0 and i % int(args.checkpoint_every) == 0:
            _write_output(args, payload, rows, complete=False)
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbols-file", type=Path, default=Path("symbol_lists/stocks_wide_1000_v1.txt"))
    p.add_argument("--data-root", type=Path, default=Path("trainingdata"))
    p.add_argument("--model-paths", required=True)
    p.add_argument("--train-start", default="2020-01-01")
    p.add_argument("--oos-start", default="2025-07-01")
    p.add_argument("--oos-end", default="2026-04-24")
    p.add_argument("--window-days", type=int, default=120)
    p.add_argument("--stride-days", type=int, default=30)
    p.add_argument("--min-dollar-vol", type=float, default=5_000_000.0)
    p.add_argument("--inference-min-dolvol", type=float, default=50_000_000.0)
    p.add_argument("--inference-min-vol", type=float, default=0.12)
    p.add_argument("--inference-max-vol", type=float, default=0.0)
    p.add_argument("--cache-path", type=Path, default=Path("analysis/xgbnew_daily/cache/oos_scores.pkl"))
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--fast-features", action="store_true")
    p.add_argument("--fee-regimes", default="deploy,stress36x")
    p.add_argument("--fill-buffer-bps-grid", default="5,15")
    p.add_argument("--leverage-grid", default="0.5,1.0,1.5,2.0")
    p.add_argument("--min-score-grid", default="0.0,0.55,0.70,0.85")
    p.add_argument("--top-n-grid", default="1,2,3")
    p.add_argument("--min-picks-grid", default="0,1")
    p.add_argument("--include-invert", action="store_true")
    p.add_argument("--score-uncertainty-penalty-grid", default="0,0.5,1.0")
    p.add_argument("--allocation-mode-grid", default="equal,score_norm,softmax")
    p.add_argument("--allocation-temp-grid", default="0.25,0.5,1.0")
    p.add_argument("--no-picks-fallback", default="")
    p.add_argument("--no-picks-fallback-alloc-grid", default="0")
    p.add_argument("--max-ret-20d-rank-pct-grid", default="1.0")
    p.add_argument("--min-ret-5d-rank-pct-grid", default="0.0")
    p.add_argument("--regime-cs-iqr-max-grid", default="0.0")
    p.add_argument("--regime-cs-skew-min-grid", default="-1000000000")
    p.add_argument("--inv-vol-target-grid", default="0.0")
    p.add_argument("--fail-fast-max-dd-pct", type=float, default=30.0)
    p.add_argument("--fail-fast-neg-windows", type=int, default=1)
    p.add_argument("--progress-every", type=int, default=25)
    p.add_argument("--checkpoint-every", type=int, default=1)
    p.add_argument("--max-cells", type=int, default=0)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    payload = _load_or_build_cache(args)
    rows = _evaluate(args, payload)
    _write_output(args, payload, rows, complete=True)
    best = _friction_robust_strategy_rows(rows)
    best = best[0] if best else None
    if best:
        print(json.dumps(best, indent=2, default=str))
    else:
        print("No friction-robust strategy survived the evaluated grid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
