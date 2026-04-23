"""Sweep over multi-day config (top_n_slots, min_prob, leverage, horizons_mask).

Reuses cached per-horizon models (saved under ``model_dir``) so one training
pass supports many evaluations. Writes a single JSON with all cells.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import xgboost as xgb

from .backtest import (
    MultiDayConfig,
    build_daily_candidate_table,
    simulate,
    _symbol_abs_ret_prior,
)
from .dataset import HORIZONS, build_multi_horizon_dataset
from .train import FEATURE_COLS, TrainConfig, train_per_horizon, score_per_horizon

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sweep_multiday")

REPO = Path(__file__).resolve().parents[1]


def _load_symbols(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")]


def _load_cached_models(model_dir: Path, horizons, seeds) -> dict[int, list[xgb.Booster]]:
    out: dict[int, list[xgb.Booster]] = {}
    for n in horizons:
        boosters = []
        for s in seeds:
            path = model_dir / f"fwd_{n}d_seed{s}.ubj"
            if not path.exists():
                return {}  # incomplete cache
            bst = xgb.Booster()
            bst.load_model(str(path))
            boosters.append(bst)
        out[n] = boosters
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=REPO / "trainingdata")
    p.add_argument("--symbols-file", type=Path,
                   default=REPO / "symbol_lists" / "stocks_wide_1000_v1.txt")
    p.add_argument("--train-end", type=str, required=True)
    p.add_argument("--test-end",  type=str, required=True)
    p.add_argument("--test-days", type=int, default=120)
    p.add_argument("--model-dir", type=Path, required=True,
                   help="Where to save/load trained per-horizon boosters")
    p.add_argument("--seeds",    type=str, default="0,7,42,73,197")
    p.add_argument("--n-estimators", type=int, default=600)
    p.add_argument("--max-depth",    type=int, default=6)
    p.add_argument("--learning-rate",type=float, default=0.05)
    p.add_argument("--device",      type=str, default="cuda")
    p.add_argument("--decision-lag", type=int, default=2)
    p.add_argument("--fee-bps",     type=float, default=10.0)
    p.add_argument("--fill-buffer-bps", type=float, default=5.0)
    p.add_argument("--force-retrain", action="store_true")
    # Sweep grids
    p.add_argument("--lev-grid",      type=str, default="1.0,1.25,1.5,2.0")
    p.add_argument("--slots-grid",    type=str, default="1,3,5")
    p.add_argument("--min-prob-grid", type=str, default="0.50,0.53,0.55")
    p.add_argument("--min-exp-ret-grid", type=str, default="0.0,0.005")
    p.add_argument("--horizon-masks", type=str,
                   default="1|1,2,3,5,10|2,3,5,10|3,5,10|1,3,5",
                   help="Pipe-separated horizon-subset lists for meta-selector")
    p.add_argument("--out",          type=Path, required=True)
    args = p.parse_args()

    train_end = date.fromisoformat(args.train_end)
    test_end  = date.fromisoformat(args.test_end)
    seeds     = tuple(int(x) for x in args.seeds.split(","))
    horizons  = HORIZONS

    symbols = _load_symbols(args.symbols_file)
    logger.info("Universe: %d symbols; train_end=%s test_end=%s", len(symbols), train_end, test_end)

    train_df, test_df_full = build_multi_horizon_dataset(
        args.data_root, symbols,
        train_end=train_end,
        test_start=train_end + timedelta(days=1),
        test_end=test_end,
        horizons=horizons,
        use_fast_features=True,
    )
    logger.info("train rows=%d test rows (full)=%d", len(train_df), len(test_df_full))

    test_days_sorted = sorted(test_df_full["date"].unique())
    if len(test_days_sorted) > args.test_days:
        cutoff = test_days_sorted[-args.test_days]
        test_df = test_df_full[test_df_full["date"] >= cutoff].copy()
    else:
        test_df = test_df_full.copy()
    logger.info("Final test window: %d trading days, %d rows",
                len(test_df["date"].unique()), len(test_df))

    # Load or train
    args.model_dir.mkdir(parents=True, exist_ok=True)
    models = None
    if not args.force_retrain:
        models = _load_cached_models(args.model_dir, horizons, seeds)
        if models:
            logger.info("Loaded cached models from %s", args.model_dir)
    if not models:
        cfg_train = TrainConfig(
            n_estimators=args.n_estimators, max_depth=args.max_depth,
            learning_rate=args.learning_rate, device=args.device, seeds=seeds,
        )
        models = train_per_horizon(
            train_df, horizons, cfg_train, out_dir=args.model_dir,
        )

    probs_all = score_per_horizon(test_df, models)
    abs_prior = _symbol_abs_ret_prior(train_df, horizons)

    lev_list = [float(x) for x in args.lev_grid.split(",")]
    slots_list = [int(x) for x in args.slots_grid.split(",")]
    mp_list = [float(x) for x in args.min_prob_grid.split(",")]
    mer_list = [float(x) for x in args.min_exp_ret_grid.split(",")]
    hmask_list = [
        tuple(int(h) for h in m.split(","))
        for m in args.horizon_masks.split("|")
    ]

    cells = []
    best_goodness = -1e18
    best_trades: list = []
    best_equity: pd.Series | None = None
    best_key: dict | None = None
    for hmask, lev, slots, mp, mer in itertools.product(
        hmask_list, lev_list, slots_list, mp_list, mer_list,
    ):
        cfg = MultiDayConfig(
            horizons=hmask,
            leverage=lev,
            fee_bps_per_side=args.fee_bps,
            fill_buffer_bps=args.fill_buffer_bps,
            decision_lag=args.decision_lag,
            min_prob=mp,
            min_expected_ret=mer,
            top_n_slots=slots,
        )
        probs_sel = {n: probs_all[n] for n in hmask if n in probs_all}
        cand = build_daily_candidate_table(test_df, probs_sel, abs_prior, cfg)
        res = simulate(test_df, cand, cfg)
        s = res.summary
        cells.append({
            "horizons": list(hmask), "lev": lev, "slots": slots,
            "min_prob": mp, "min_exp_ret": mer,
            **{k: v for k, v in s.items()},
        })
        # Goodness = med_monthly - 0.5*max_dd - 100*neg_frac (same spirit as xgbnew)
        neg = s.get("neg_window_frac") or 0.0
        goodness = s["median_monthly_pnl_pct"] - 0.5 * s["max_dd_pct"] - 100.0 * neg
        if goodness > best_goodness and s["n_trades"] >= 5:
            best_goodness = goodness
            best_trades = list(res.trades)
            best_equity = res.equity_by_date
            best_key = {
                "horizons": list(hmask), "lev": lev, "slots": slots,
                "min_prob": mp, "min_exp_ret": mer,
            }
        logger.info(
            "h=%s lev=%.2f slots=%d mp=%.2f mer=%.4f: n=%d med=%.2f%%/mo p10=%.2f dd=%.1f neg=%.0f%% avg_hold=%.1f",
            hmask, lev, slots, mp, mer,
            s["n_trades"], s["median_monthly_pnl_pct"], s["p10_monthly_pnl_pct"],
            s["max_dd_pct"], 100.0 * (s.get("neg_window_frac") or 0.0),
            s["avg_hold_days"],
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "train_end": args.train_end,
        "test_end":  args.test_end,
        "test_days_requested": args.test_days,
        "test_days_actual":    int(len(test_df["date"].unique())),
        "train_rows":          int(len(train_df)),
        "test_rows":           int(len(test_df)),
        "seeds":               list(seeds),
        "n_cells":             len(cells),
        "best_cell":           best_key,
        "best_goodness":       best_goodness,
        "cells":               cells,
    }, indent=2, default=str))
    logger.info("Wrote %s (%d cells)", args.out, len(cells))
    if best_trades and best_equity is not None and best_key is not None:
        trades_out = args.out.with_suffix(".best_trades.jsonl")
        with trades_out.open("w") as f:
            f.write(json.dumps({"_best_cell": best_key, "_best_goodness": best_goodness}) + "\n")
            for t in best_trades:
                f.write(json.dumps({
                    "entry_date": str(t.entry_date),
                    "exit_date":  str(t.exit_date),
                    "symbol":     t.symbol,
                    "horizon":    t.horizon,
                    "prob":       t.prob,
                    "expected_ret": t.expected_ret,
                    "gross_ret":  t.gross_ret,
                    "net_ret":    t.net_ret,
                    "hold_days":  t.hold_days,
                }) + "\n")
        logger.info("Wrote best-cell trades to %s (%d trades)",
                    trades_out, len(best_trades))


if __name__ == "__main__":
    main()
