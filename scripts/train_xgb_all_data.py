#!/usr/bin/env python3
"""Train the champion XGB config on 100% of trainingdata/ — final prod model.

Assumes the config (top_n=1, n_est=400, depth=5, lr=0.03, random_state=42)
has already been bonferroni-validated via walk-forward on the held-out span.
This script produces the final prod-weights by fitting on every available
day — no val split, no early stopping — so the live trader sees the best
signal Alpaca can give us. Lose market-sim accuracy on the final fit in
exchange for max production edge.

Output: a pickled XGBStockModel at --output-path, loadable by
``xgbnew/live_trader.py`` via ``--model-path``.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from xgbnew.dataset import build_daily_dataset
from xgbnew.features import DAILY_FEATURE_COLS
from xgbnew.model import XGBStockModel


def _load_symbols(path: Path) -> list[str]:
    return [l.strip().upper() for l in path.read_text().splitlines()
            if l.strip() and not l.startswith("#")]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols-file", type=Path,
                   default=REPO / "symbol_lists/stocks_wide_1000_v1.txt")
    p.add_argument("--data-root", type=Path, default=REPO / "trainingdata")
    p.add_argument("--start", default="2020-01-01",
                   help="Earliest training date inclusive")
    p.add_argument("--end", default="2026-04-17",
                   help="Latest training date inclusive — should be the most "
                        "recent day with universe-wide coverage")
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--min-dollar-vol", type=float, default=5e6)
    p.add_argument("--output-path", type=Path, required=True,
                   help="Pickle destination for the fitted model")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    symbols = _load_symbols(args.symbols_file)
    print(f"[train-all] universe: {len(symbols)} symbols  "
          f"data_root={args.data_root}", flush=True)

    # We abuse the train split to cover the entire span. val/test get
    # zero-day windows and are discarded.
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    train_df, _, _ = build_daily_dataset(
        data_root=args.data_root,
        symbols=symbols,
        train_start=start, train_end=end,
        val_start=end, val_end=end,
        test_start=end, test_end=end,
        chronos_cache={},
        min_dollar_vol=args.min_dollar_vol,
    )
    print(f"[train-all] rows={len(train_df):,}  "
          f"span={start}..{end}", flush=True)
    if len(train_df) < 1000:
        print("ERROR: too few training rows", file=sys.stderr)
        return 1

    model = XGBStockModel(
        device=args.device,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        random_state=args.random_state,
    )
    print(f"[train-all] fitting XGB(n_est={args.n_estimators} "
          f"depth={args.max_depth} lr={args.learning_rate} "
          f"seed={args.random_state})...", flush=True)
    model.fit(train_df, DAILY_FEATURE_COLS, val_df=None, verbose=True)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output_path)
    print(f"[train-all] saved → {args.output_path}", flush=True)

    print("\n  Top-10 feature importances:")
    for feat, imp in model.feature_importances().head(10).items():
        print(f"    {feat:<28} {imp:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
