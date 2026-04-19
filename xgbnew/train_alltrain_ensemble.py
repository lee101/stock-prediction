#!/usr/bin/env python3
"""Final-stage XGB ensemble training on ALL data (no OOS holdout).

Same concept as ``train_alltrain.py`` but trains N models with different
``random_state`` seeds and saves all of them. At inference time the
caller averages ``predict_proba`` across the N models — same pattern as
our RL ensemble. This buys seed-variance robustness on top of the
alltrain "maximum-faith" signal.

Writes:
    <out_dir>/alltrain_seed{seed}.pkl   (one per seed)
    <out_dir>/alltrain_ensemble.json    (manifest with seeds + metadata)

Usage::

    python -m xgbnew.train_alltrain_ensemble \\
        --symbols-file symbol_lists/stocks_wide_1000_v1.txt \\
        --data-root trainingdata \\
        --train-start 2020-01-01 \\
        --seeds 0,7,42,73,197 \\
        --n-estimators 400 --max-depth 5 --learning-rate 0.03 \\
        --device cpu \\
        --out-dir analysis/xgbnew_daily/alltrain_ensemble
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
    p.add_argument("--train-end", default="")
    p.add_argument("--min-dollar-vol", type=float, default=5_000_000.0)

    p.add_argument("--seeds", default="0,7,42,73,197",
                   help="Comma-separated seeds (one model per seed).")
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", type=Path,
                   default=REPO / "analysis/xgbnew_daily/alltrain_ensemble")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if len(seeds) < 2:
        print("ERROR: need at least 2 seeds", file=sys.stderr)
        return 1

    symbols = _load_symbols(args.symbols_file)
    train_start = date.fromisoformat(args.train_start)
    train_end = date.fromisoformat(args.train_end) if args.train_end else date.today()

    chronos_cache = {}
    if args.chronos_cache.exists():
        chronos_cache = load_chronos_cache(args.chronos_cache)

    print(f"[xgb-alltrain-ens] {len(symbols)} symbols | train {train_start} → {train_end} | "
          f"seeds={seeds} device={args.device}", flush=True)

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
    print(f"[xgb-alltrain-ens] dataset built in {time.perf_counter()-t0:.1f}s | "
          f"rows={len(train_df):,}  train_symbols={train_df['symbol'].nunique()}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for i, seed in enumerate(seeds, start=1):
        print(f"\n[xgb-alltrain-ens] {i}/{len(seeds)} training seed={seed}...", flush=True)
        t_fit = time.perf_counter()
        model = XGBStockModel(
            device=args.device,
            n_estimators=int(args.n_estimators),
            max_depth=int(args.max_depth),
            learning_rate=float(args.learning_rate),
            random_state=int(seed),
        )
        model.fit(train_df, DAILY_FEATURE_COLS, verbose=args.verbose)
        out_pkl = args.out_dir / f"alltrain_seed{seed}.pkl"
        model.save(out_pkl)
        saved.append({"seed": int(seed), "path": str(out_pkl),
                      "fit_seconds": round(time.perf_counter() - t_fit, 2)})
        print(f"  seed={seed} fit in {saved[-1]['fit_seconds']:.1f}s -> {out_pkl.name}", flush=True)

    manifest = {
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "train_start": str(train_start),
        "train_end": str(train_end),
        "n_symbols_requested": len(symbols),
        "n_symbols_with_data": int(train_df["symbol"].nunique()),
        "n_rows": int(len(train_df)),
        "seeds": seeds,
        "models": saved,
        "config": {
            "n_estimators": int(args.n_estimators),
            "max_depth": int(args.max_depth),
            "learning_rate": float(args.learning_rate),
            "device": args.device,
            "min_dollar_vol": float(args.min_dollar_vol),
        },
        "blend_recipe": "predict_proba mean across seeds then pick top_n=1",
    }
    manifest_path = args.out_dir / "alltrain_ensemble.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n[xgb-alltrain-ens] Manifest → {manifest_path}")
    print(f"[xgb-alltrain-ens] Models   → {len(saved)} files in {args.out_dir}")
    print(f"[xgb-alltrain-ens] ⚠ No OOS metrics — trust champion hyperparams only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
