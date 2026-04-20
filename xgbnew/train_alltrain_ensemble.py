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
from xgbnew.features import DAILY_FEATURE_COLS, DAILY_RANK_FEATURE_COLS
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
    p.add_argument("--include-ranks", action="store_true",
                   help="Add 5 per-day cross-sectional pct-rank features "
                        "(rank_ret_1d, rank_ret_5d, rank_vol_20d, "
                        "rank_dolvol_20d_log, rank_rsi_14). Saved into the "
                        "pkl feature_cols so predict-time knows to use them.")
    p.add_argument("--shapes", default="",
                   help="Architectural-diversity mode. Comma-separated tuples "
                        "'n_est:depth:lr:seed', one model per tuple. Overrides "
                        "--seeds / --n-estimators / --max-depth / --learning-rate. "
                        "Example: "
                        "'400:5:0.03:42,800:4:0.05:42,1600:5:0.01:42,300:7:0.02:42,200:3:0.10:42'")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def _parse_shapes(s: str) -> list[dict]:
    """Parse --shapes string into list of {n_est, depth, lr, seed} dicts.

    Format: 'n_est:depth:lr:seed' comma-separated. Empty string → [].
    """
    out: list[dict] = []
    for raw in s.split(","):
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split(":")
        if len(parts) != 4:
            raise ValueError(
                f"--shapes tuple must be 'n_est:depth:lr:seed', got {raw!r}"
            )
        out.append({
            "n_est": int(parts[0]),
            "depth": int(parts[1]),
            "lr":    float(parts[2]),
            "seed":  int(parts[3]),
        })
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")

    shapes = _parse_shapes(args.shapes) if args.shapes else []
    if shapes:
        # Architectural-diversity mode: shapes list drives training.
        seeds = [sp["seed"] for sp in shapes]
        if len(shapes) < 2:
            print("ERROR: --shapes needs at least 2 tuples", file=sys.stderr)
            return 1
    else:
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
        include_cross_sectional_ranks=bool(args.include_ranks),
    )
    print(f"[xgb-alltrain-ens] dataset built in {time.perf_counter()-t0:.1f}s | "
          f"rows={len(train_df):,}  train_symbols={train_df['symbol'].nunique()}", flush=True)

    feature_cols = list(DAILY_FEATURE_COLS)
    if args.include_ranks:
        feature_cols += list(DAILY_RANK_FEATURE_COLS)
    print(f"[xgb-alltrain-ens] feature_cols={len(feature_cols)} "
          f"ranks_on={bool(args.include_ranks)}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    if shapes:
        for i, sp in enumerate(shapes, start=1):
            seed = sp["seed"]
            print(f"\n[xgb-alltrain-ens] {i}/{len(shapes)} training "
                  f"n_est={sp['n_est']} d={sp['depth']} lr={sp['lr']} "
                  f"seed={seed}...", flush=True)
            t_fit = time.perf_counter()
            model = XGBStockModel(
                device=args.device,
                n_estimators=sp["n_est"],
                max_depth=sp["depth"],
                learning_rate=sp["lr"],
                random_state=seed,
            )
            model.fit(train_df, feature_cols, verbose=args.verbose)
            tag = f"n{sp['n_est']}_d{sp['depth']}_lr{str(sp['lr']).replace('.', '')}_s{seed}"
            out_pkl = args.out_dir / f"alltrain_{tag}.pkl"
            model.save(out_pkl)
            saved.append({"seed": seed, "n_estimators": sp["n_est"],
                          "max_depth": sp["depth"], "learning_rate": sp["lr"],
                          "path": str(out_pkl),
                          "fit_seconds": round(time.perf_counter() - t_fit, 2)})
            print(f"  {tag} fit in {saved[-1]['fit_seconds']:.1f}s -> {out_pkl.name}",
                  flush=True)
    else:
        for i, seed in enumerate(seeds, start=1):
            print(f"\n[xgb-alltrain-ens] {i}/{len(seeds)} training seed={seed}...",
                  flush=True)
            t_fit = time.perf_counter()
            model = XGBStockModel(
                device=args.device,
                n_estimators=int(args.n_estimators),
                max_depth=int(args.max_depth),
                learning_rate=float(args.learning_rate),
                random_state=int(seed),
            )
            model.fit(train_df, feature_cols, verbose=args.verbose)
            out_pkl = args.out_dir / f"alltrain_seed{seed}.pkl"
            model.save(out_pkl)
            saved.append({"seed": int(seed), "path": str(out_pkl),
                          "fit_seconds": round(time.perf_counter() - t_fit, 2)})
            print(f"  seed={seed} fit in {saved[-1]['fit_seconds']:.1f}s -> {out_pkl.name}",
                  flush=True)

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
            "include_ranks": bool(args.include_ranks),
            "feature_cols": feature_cols,
            "shapes_mode": bool(shapes),
            "shapes": shapes,
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
