"""End-to-end 120-day OOS simulation for the multi-horizon prototype.

Run:
    source .venv/bin/activate
    python -m xgbnew_multiday.run_120d_sim --train-end 2025-12-19 --test-end 2026-04-17

Defaults:
  * train period: everything up to ``train_end``
  * test period: 120 TRADING days up through ``test_end``
  * symbols: stocks_wide_1000_v1.txt (846 tickers — same as live xgb-daily-trader)
  * reports both:
      (a) multi-horizon meta-selector  — argmax over {1,2,3,5,10}
      (b) single-horizon baseline      — same model family forced to N=1
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import (
    MultiDayConfig,
    build_daily_candidate_table,
    simulate,
    _symbol_abs_ret_prior,
)
from .dataset import HORIZONS, build_multi_horizon_dataset
from .train import TrainConfig, train_per_horizon, score_per_horizon

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_120d_sim")


REPO = Path(__file__).resolve().parents[1]


def _load_symbols(path: Path) -> list[str]:
    lines = path.read_text().splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]


def _n_trading_days_before(target: date, n: int, all_dates: list[date]) -> date:
    """Return the date N trading days before target (using observed bars)."""
    before = [d for d in all_dates if d <= target]
    if len(before) <= n:
        return before[0]
    return before[-n - 1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=REPO / "trainingdata")
    p.add_argument("--symbols-file", type=Path,
                   default=REPO / "symbol_lists" / "stocks_wide_1000_v1.txt")
    p.add_argument("--train-end", type=str, required=True, help="YYYY-MM-DD inclusive")
    p.add_argument("--test-end",  type=str, required=True, help="YYYY-MM-DD inclusive")
    p.add_argument("--test-days", type=int, default=120)
    p.add_argument("--horizons", type=str, default="1,2,3,5,10")
    p.add_argument("--seeds",    type=str, default="0,7,42,73,197")
    p.add_argument("--n-estimators", type=int, default=600)
    p.add_argument("--max-depth",    type=int, default=6)
    p.add_argument("--learning-rate",type=float, default=0.05)
    p.add_argument("--device",      type=str, default="cuda")
    p.add_argument("--leverage",    type=float, default=1.0)
    p.add_argument("--fee-bps",     type=float, default=10.0)
    p.add_argument("--fill-buffer-bps", type=float, default=5.0)
    p.add_argument("--decision-lag", type=int, default=2)
    p.add_argument("--min-prob",    type=float, default=0.50)
    p.add_argument("--min-expected-ret", type=float, default=0.0)
    p.add_argument("--min-dollar-vol",   type=float, default=5e6)
    p.add_argument("--top-n-slots",      type=int, default=1,
                   help="Concurrent positions. 1 = single-slot, 3/5 = portfolio.")
    p.add_argument("--allocation-per-slot", type=float, default=1.0,
                   help="Fraction of equity allocated to each slot (1.0/K each).")
    p.add_argument("--out-dir",     type=Path,
                   default=REPO / "analysis" / "xgbnew_multiday")
    p.add_argument("--model-dir",   type=Path, default=None,
                   help="Directory to save trained models. If None, skip persist.")
    p.add_argument("--max-symbols", type=int, default=0,
                   help="If >0, truncate symbol list for smoke tests")
    p.add_argument("--baseline-only", action="store_true",
                   help="Train/eval single-horizon N=1 only (sanity check)")
    args = p.parse_args()

    train_end = date.fromisoformat(args.train_end)
    test_end  = date.fromisoformat(args.test_end)
    horizons  = tuple(int(x) for x in args.horizons.split(","))
    seeds     = tuple(int(x) for x in args.seeds.split(","))

    symbols = _load_symbols(args.symbols_file)
    if args.max_symbols > 0:
        symbols = symbols[: args.max_symbols]
    logger.info("Universe: %d symbols; train_end=%s test_end=%s", len(symbols), train_end, test_end)

    logger.info("Building multi-horizon dataset...")
    # Temporarily build with a very-wide test window, we'll slice later
    train_df, test_df_full = build_multi_horizon_dataset(
        args.data_root, symbols,
        train_end=train_end,
        test_start=train_end + timedelta(days=1),
        test_end=test_end,
        horizons=horizons,
        use_fast_features=True,
        min_dollar_vol=args.min_dollar_vol,
    )
    logger.info("train rows=%d test rows=%d", len(train_df), len(test_df_full))

    # Trim test to last N trading days
    test_days_sorted = sorted(test_df_full["date"].unique())
    if len(test_days_sorted) > args.test_days:
        cutoff = test_days_sorted[-args.test_days]
        test_df = test_df_full[test_df_full["date"] >= cutoff].copy()
    else:
        test_df = test_df_full.copy()
    logger.info("Final test window: %d trading days, %d rows",
                len(test_df["date"].unique()), len(test_df))

    # Train per horizon
    cfg_train = TrainConfig(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        device=args.device,
        seeds=seeds,
    )
    train_horizons = (1,) if args.baseline_only else horizons
    logger.info("Training horizons: %s", train_horizons)
    models = train_per_horizon(
        train_df, train_horizons, cfg_train,
        out_dir=args.model_dir,
    )

    logger.info("Scoring test set...")
    probs = score_per_horizon(test_df, models)

    abs_prior = _symbol_abs_ret_prior(train_df, train_horizons)

    results = {}
    for label, selector_horizons in [
        ("baseline_1d", (1,)),
        ("multi_horizon", train_horizons if not args.baseline_only else (1,)),
    ]:
        if label == "baseline_1d" and 1 not in probs:
            continue
        cfg = MultiDayConfig(
            horizons=selector_horizons,
            leverage=args.leverage,
            fee_bps_per_side=args.fee_bps,
            fill_buffer_bps=args.fill_buffer_bps,
            decision_lag=args.decision_lag,
            min_prob=args.min_prob,
            min_expected_ret=args.min_expected_ret,
            min_dollar_vol=args.min_dollar_vol,
            top_n_slots=args.top_n_slots,
            allocation_per_slot=args.allocation_per_slot,
        )
        # Filter probs to only selected horizons (so argmax only considers those)
        probs_sel = {n: probs[n] for n in selector_horizons if n in probs}
        cand = build_daily_candidate_table(test_df, probs_sel, abs_prior, cfg)
        res = simulate(test_df, cand, cfg)
        results[label] = {
            "config": {
                "horizons": list(selector_horizons),
                "leverage": args.leverage,
                "fee_bps_per_side": args.fee_bps,
                "fill_buffer_bps": args.fill_buffer_bps,
                "decision_lag": args.decision_lag,
                "min_prob": args.min_prob,
                "min_expected_ret": args.min_expected_ret,
            },
            "summary": res.summary,
        }
        logger.info("=== %s ===", label)
        for k, v in res.summary.items():
            logger.info("  %s = %s", k, v)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.train_end}_to_{args.test_end}"
    out_file = args.out_dir / f"multiday_sim_{tag}.json"
    out_file.write_text(json.dumps({
        "args": {
            "data_root": str(args.data_root),
            "symbols_file": str(args.symbols_file),
            "train_end": args.train_end,
            "test_end": args.test_end,
            "test_days_requested": args.test_days,
            "horizons": list(horizons),
            "seeds": list(seeds),
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
        },
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "test_days": int(len(test_df["date"].unique())),
        "results": results,
    }, indent=2, default=str))
    logger.info("Wrote %s", out_file)


if __name__ == "__main__":
    main()
