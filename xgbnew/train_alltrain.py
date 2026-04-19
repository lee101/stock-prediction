#!/usr/bin/env python3
"""Final-stage XGB training on ALL data (no OOS holdout).

Trains the champion hyperparameters on the entire historical window
``--train-start``..``--train-end`` (defaults 2020-01-01 → today) and
saves the pickle to ``analysis/xgbnew_daily/live_model_alltrain.pkl``.

This is the "maximum-faith" final-stage deploy artefact: no held-out
validation is possible by construction, so we trust the separately-
validated champion hyperparams (see bonferroni / k-fold memory) and
stamp out one final model with every available row of training data.

Still emits the same feature-importances + training loss so we can spot
gross pathologies, but do not expect OOS metrics here.

Usage::

    python -m xgbnew.train_alltrain \\
        --symbols-file symbol_lists/stocks_wide_1000_v1.txt \\
        --data-root trainingdata \\
        --train-start 2020-01-01 \\
        --n-estimators 400 --max-depth 5 --learning-rate 0.03 \\
        --random-state 42 \\
        --device cpu \\
        --out analysis/xgbnew_daily/live_model_alltrain.pkl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from xgbnew.dataset import build_daily_dataset, load_chronos_cache
from xgbnew.features import DAILY_FEATURE_COLS
from xgbnew.model import XGBStockModel

logger = logging.getLogger(__name__)


def _load_symbols(path: Path) -> list[str]:
    syms: list[str] = []
    for line in path.read_text().splitlines():
        s = line.strip().upper()
        if s and not s.startswith("#"):
            syms.append(s)
    seen = set()
    return [s for s in syms if not (s in seen or seen.add(s))]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols-file", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=REPO / "trainingdata")
    p.add_argument("--chronos-cache", type=Path,
                   default=REPO / "analysis/top2_backtest/forecast_cache")
    p.add_argument("--train-start", default="2020-01-01")
    p.add_argument("--train-end", default="",
                   help="Empty = today.")
    p.add_argument("--min-dollar-vol", type=float, default=5_000_000.0)

    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--device", default="cpu",
                   help="CPU is the deployable path; CUDA only for exploration.")
    p.add_argument("--out", type=Path,
                   default=REPO / "analysis/xgbnew_daily/live_model_alltrain.pkl")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")

    symbols = _load_symbols(args.symbols_file)
    train_start = date.fromisoformat(args.train_start)
    train_end = date.fromisoformat(args.train_end) if args.train_end else date.today()

    chronos_cache = {}
    if args.chronos_cache.exists():
        chronos_cache = load_chronos_cache(args.chronos_cache)

    print(f"[xgb-alltrain] {len(symbols)} symbols | train {train_start} → {train_end} | "
          f"device={args.device}", flush=True)

    t0 = time.perf_counter()
    train_df, _, _ = build_daily_dataset(
        data_root=args.data_root,
        symbols=symbols,
        train_start=train_start, train_end=train_end,
        val_start=train_end, val_end=train_end,
        test_start=train_end, test_end=train_end,
        chronos_cache=chronos_cache if chronos_cache else None,
        min_dollar_vol=args.min_dollar_vol,
        fast_features=False,
    )
    print(f"[xgb-alltrain] dataset built in {time.perf_counter()-t0:.1f}s | "
          f"rows={len(train_df):,}  train_symbols={train_df['symbol'].nunique()}", flush=True)

    if len(train_df) < 5000:
        print("ERROR: not enough training data", file=sys.stderr)
        return 1

    print(f"[xgb-alltrain] training XGB (n_est={args.n_estimators}, d={args.max_depth}, "
          f"lr={args.learning_rate}, seed={args.random_state})...", flush=True)
    t_fit = time.perf_counter()
    model = XGBStockModel(
        device=args.device,
        n_estimators=int(args.n_estimators),
        max_depth=int(args.max_depth),
        learning_rate=float(args.learning_rate),
        random_state=int(args.random_state),
    )
    model.fit(train_df, DAILY_FEATURE_COLS, verbose=args.verbose)
    print(f"[xgb-alltrain] fit in {time.perf_counter()-t_fit:.1f}s", flush=True)

    imps = model.feature_importances().head(10)
    print("\n  Top-10 feature importances:")
    for feat, imp in imps.items():
        print(f"    {feat:<25} {imp:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)

    # Sidecar metadata
    meta = {
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "train_start": str(train_start),
        "train_end": str(train_end),
        "n_symbols_requested": len(symbols),
        "n_symbols_with_data": int(train_df["symbol"].nunique()),
        "n_rows": int(len(train_df)),
        "config": {
            "n_estimators": int(args.n_estimators),
            "max_depth": int(args.max_depth),
            "learning_rate": float(args.learning_rate),
            "random_state": int(args.random_state),
            "device": args.device,
            "min_dollar_vol": float(args.min_dollar_vol),
        },
        "feature_importances_top10": {k: float(v) for k, v in imps.items()},
    }
    meta_path = args.out.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\n  Model  → {args.out}")
    print(f"  Meta   → {meta_path}")
    print(f"  ⚠ No OOS metrics — this is an alltrain model; trust champion hyperparams only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
