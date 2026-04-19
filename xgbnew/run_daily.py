#!/usr/bin/env python3
"""
XGBoost + Chronos2 daily open-to-close strategy.

Pipeline
--------
1. Load all 846-stock daily CSVs from trainingdata/
2. Compute vectorised no-lookahead technical features for every stock/day
3. Train XGBClassifier on 2021-2024 data (pure technical features)
4. Validate on 2025 data; print directional accuracy & val PnL
5. Attach Chronos2 forecast features to Jan-Apr 2026 test set
6. Backtest: each day, pick top-N by blended (XGB * w + Chronos2-rank * (1-w)) score
7. Simulate open-to-close with real spread estimates, commission, optional 2x leverage
8. Output: per-trade CSV, summary JSON, console table

Key flags
---------
  --top-n N              picks per day (default 2)
  --leverage L           1.0 = no leverage, 2.0 = max (default 1.0)
  --xgb-weight W         XGB weight in blend [0,1] (default 0.5)
  --chronos-cache DIR    path to Chronos2 JSON cache
  --symbols-file FILE    symbols list (default stocks_wide_1000_v1.txt)
  --min-dollar-vol M     min avg daily $ vol (default 5e6)
  --output-dir DIR       results output (default analysis/xgbnew_daily)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from loss_utils import TRADING_FEE
from xgbnew.dataset import build_daily_dataset, load_chronos_cache
from xgbnew.features import ALL_FEATURE_COLS, CHRONOS_FEATURE_COLS, DAILY_FEATURE_COLS
from xgbnew.model import XGBStockModel, combined_scores
from xgbnew.backtest import BacktestConfig, simulate, print_summary

logger = logging.getLogger(__name__)


def _load_symbols(path: Path) -> list[str]:
    symbols = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            symbols.append(s.split("#", 1)[0].strip().upper())
    return symbols


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols-file", type=Path,
                   default=REPO / "symbol_lists/stocks_wide_1000_v1.txt")
    p.add_argument("--symbols", default=None, help="Comma-separated symbol override")
    p.add_argument("--data-root", type=Path, default=REPO / "trainingdata")

    p.add_argument("--train-start", default="2021-01-01")
    p.add_argument("--train-end",   default="2024-12-31")
    p.add_argument("--val-start",   default="2025-01-01")
    p.add_argument("--val-end",     default="2025-11-30")
    p.add_argument("--test-start",  default="2026-01-05")
    p.add_argument("--test-end",    default="2026-04-09")

    p.add_argument("--top-n",      type=int,   default=2)
    p.add_argument("--leverage",   type=float, default=1.0,
                   help="Position leverage (1.0=none, 2.0=max)")
    p.add_argument("--xgb-weight", type=float, default=0.5,
                   help="XGB weight in blended score (0=pure Chronos2, 1=pure XGB)")
    p.add_argument("--commission-bps", type=float, default=0.0,
                   help="Legacy extra commission per side in bps (default 0; stock fee defaults are applied separately)")
    p.add_argument("--fee-rate", type=float, default=float(TRADING_FEE),
                   help="Per-side fee fraction (default: shared stock TRADING_FEE)")
    p.add_argument("--fill-buffer-bps", type=float, default=5.0,
                   help="Adverse fill buffer applied around open/close bars (default 5bps)")
    p.add_argument("--min-dollar-vol", type=float, default=5e6)
    p.add_argument("--initial-cash",   type=float, default=10_000.0)

    p.add_argument("--chronos-cache", type=Path,
                   default=REPO / "analysis/top2_backtest/forecast_cache")
    p.add_argument("--model-path", type=Path, default=None,
                   help="Load pre-trained model (skip training)")

    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--max-depth",    type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--random-state", type=int, default=42,
                   help="XGBoost training seed (default 42). DD-reduction sweep"
                        " validated seed=2 for minimal worst-DD across 7-seed cohort.")

    p.add_argument(
        "--device",
        default="cuda",
        help="XGBoost device (default 'cuda' when available). Pass 'cpu' to force CPU.",
    )
    p.add_argument("--output-dir", type=Path,
                   default=REPO / "analysis/xgbnew_daily")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    # ── Symbols ──────────────────────────────────────────────────────────────
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = _load_symbols(args.symbols_file)
    print(f"[xgb-daily] {len(symbols)} symbols, data root: {args.data_root}", flush=True)

    # ── Chronos2 cache ────────────────────────────────────────────────────────
    chronos_cache = {}
    if args.chronos_cache.exists():
        chronos_cache = load_chronos_cache(args.chronos_cache)
        print(f"[xgb-daily] Loaded Chronos2 cache: {len(chronos_cache)} days", flush=True)
    else:
        print(f"[xgb-daily] No Chronos2 cache at {args.chronos_cache} — tech features only",
              flush=True)

    # ── Build datasets ────────────────────────────────────────────────────────
    print("[xgb-daily] Building features (vectorised, no-lookahead)...", flush=True)
    t0 = time.perf_counter()

    train_df, val_df, test_df = build_daily_dataset(
        data_root=args.data_root,
        symbols=symbols,
        train_start=date.fromisoformat(args.train_start),
        train_end=date.fromisoformat(args.train_end),
        val_start=date.fromisoformat(args.val_start),
        val_end=date.fromisoformat(args.val_end),
        test_start=date.fromisoformat(args.test_start),
        test_end=date.fromisoformat(args.test_end),
        chronos_cache=chronos_cache,
        min_dollar_vol=args.min_dollar_vol,
    )
    print(f"[xgb-daily] Dataset built in {time.perf_counter()-t0:.1f}s: "
          f"train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,} rows",
          flush=True)

    if len(train_df) < 1000:
        print("ERROR: Too few training rows. Check --data-root and --symbols-file.",
              file=sys.stderr)
        return 1

    # ── Train or load model ───────────────────────────────────────────────────
    if args.model_path and Path(args.model_path).exists():
        print(f"[xgb-daily] Loading model from {args.model_path}", flush=True)
        model = XGBStockModel.load(args.model_path)
    else:
        print("[xgb-daily] Training XGBStockModel...", flush=True)
        model = XGBStockModel(
            device=args.device,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            random_state=args.random_state,
        )
        # Train on technical features only (Chronos2 = 0 during training)
        model.fit(train_df, DAILY_FEATURE_COLS, val_df=val_df,
                  verbose=args.verbose)

        if args.model_path:
            model.save(args.model_path)
            print(f"[xgb-daily] Model saved to {args.model_path}", flush=True)

    # ── Feature importances ───────────────────────────────────────────────────
    print("\n  Top-10 feature importances:", flush=True)
    imps = model.feature_importances()
    for feat, imp in imps.head(10).items():
        print(f"    {feat:<28} {imp:.4f}")

    # ── Val accuracy ──────────────────────────────────────────────────────────
    if len(val_df) > 0:
        val_scores = model.predict_scores(val_df)
        val_pred_up = (val_scores > 0.5).astype(int)
        val_acc = (val_pred_up.values == val_df["target_oc_up"].values).mean()
        print(f"\n  Val directional accuracy: {val_acc*100:.2f}%  (n={len(val_df):,})",
              flush=True)

    if len(test_df) == 0:
        print("ERROR: No test rows. Check --test-start / --test-end.", file=sys.stderr)
        return 1

    # ── Backtest variants ─────────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results_summary = []

    configs = [
        BacktestConfig(top_n=args.top_n, leverage=1.0,
                       xgb_weight=args.xgb_weight,
                       commission_bps=args.commission_bps,
                       fee_rate=args.fee_rate,
                       fill_buffer_bps=args.fill_buffer_bps,
                       initial_cash=args.initial_cash,
                       min_dollar_vol=args.min_dollar_vol),
    ]
    # If user requested leverage, also run at requested leverage
    if args.leverage > 1.0:
        configs.append(BacktestConfig(top_n=args.top_n, leverage=args.leverage,
                                      xgb_weight=args.xgb_weight,
                                      commission_bps=args.commission_bps,
                                      fee_rate=args.fee_rate,
                                      fill_buffer_bps=args.fill_buffer_bps,
                                      initial_cash=args.initial_cash,
                                      min_dollar_vol=args.min_dollar_vol))
    # Always compare: pure XGB, pure Chronos2, blended
    for xw in [1.0, 0.0, 0.5]:
        if xw != args.xgb_weight:
            configs.append(BacktestConfig(top_n=args.top_n, leverage=1.0,
                                          xgb_weight=xw,
                                          commission_bps=args.commission_bps,
                                          fee_rate=args.fee_rate,
                                          fill_buffer_bps=args.fill_buffer_bps,
                                          initial_cash=args.initial_cash,
                                          min_dollar_vol=args.min_dollar_vol))

    for cfg in configs:
        label = f"top{cfg.top_n}_lev{cfg.leverage:.1f}_xw{cfg.xgb_weight:.2f}"
        result = simulate(test_df, model, cfg)
        print_summary(result, label=label)
        results_summary.append({
            "label": label,
            "top_n": cfg.top_n,
            "leverage": cfg.leverage,
            "xgb_weight": cfg.xgb_weight,
            "total_return_pct": result.total_return_pct,
            "monthly_return_pct": result.monthly_return_pct,
            "sharpe": result.sharpe_ratio,
            "sortino": result.sortino_ratio,
            "max_dd_pct": result.max_drawdown_pct,
            "win_rate_pct": result.win_rate_pct,
            "dir_acc_pct": result.directional_accuracy_pct,
            "total_trades": result.total_trades,
            "avg_spread_bps": result.avg_spread_bps,
        })

        # Save per-trade CSV for the primary config
        if cfg == configs[0]:
            rows = []
            for dr in result.day_results:
                for t in dr.trades:
                    rows.append({
                        "date": str(dr.day), "symbol": t.symbol, "score": t.score,
                        "leverage": t.leverage, "actual_open": t.actual_open,
                        "actual_close": t.actual_close, "gross_return_pct": t.gross_return_pct,
                        "spread_bps": t.spread_bps, "commission_bps": t.commission_bps,
                        "net_return_pct": t.net_return_pct, "equity_end": dr.equity_end,
                    })
            pd.DataFrame(rows).to_csv(
                args.output_dir / f"trades_{label}.csv", index=False)

    # ── Comparison table ──────────────────────────────────────────────────────
    print("\n  Comparison across configs:")
    print(f"  {'Label':<40} {'Total%':>8} {'Monthly%':>9} {'Sharpe':>7} {'MaxDD%':>7} {'DirAcc%':>8}")
    print("  " + "-"*80)
    for r in results_summary:
        print(f"  {r['label']:<40} {r['total_return_pct']:>+8.2f} "
              f"{r['monthly_return_pct']:>+9.2f} {r['sharpe']:>7.3f} "
              f"{r['max_dd_pct']:>7.2f} {r['dir_acc_pct']:>8.1f}")

    # Save summary JSON
    ts = time.strftime("%Y%m%d_%H%M%S")
    summary_path = args.output_dir / f"summary_{ts}.json"
    summary_path.write_text(json.dumps({
        "train_start": args.train_start, "train_end": args.train_end,
        "val_start": args.val_start, "val_end": args.val_end,
        "test_start": args.test_start, "test_end": args.test_end,
        "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
        "results": results_summary,
    }, indent=2), encoding="utf-8")
    print(f"\n  Summary → {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
