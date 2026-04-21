#!/usr/bin/env python3
"""Train N-seed ensembles for any model family on the XGB daily feature frame.

Trains {xgb, lgb, cat, mlp} with the same dataset plumbing used by
``train_alltrain_ensemble.py`` so all families are evaluated on identical
inputs. The produced pickles follow the ``xgbnew.model_registry`` contract
(one "family" field per pickle; ``load_any_model`` dispatches at eval time).

Usage::

    # LightGBM 5-seed alltrain
    python -m xgbnew.train_ensemble_family \\
        --family lgb \\
        --symbols-file symbol_lists/stocks_wide_1000_v1.txt \\
        --train-start 2020-01-01 --train-end 2025-12-20 \\
        --seeds 0,7,42,73,197 \\
        --device cpu \\
        --out-dir analysis/xgbnew_daily/track1_oos120d_lgb

    # CatBoost on GPU
    python -m xgbnew.train_ensemble_family --family cat --device cuda ...

    # Small tabular MLP
    python -m xgbnew.train_ensemble_family --family mlp --device cuda ...
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
from xgbnew.features import (
    DAILY_DISPERSION_FEATURE_COLS,
    DAILY_FEATURE_COLS,
    DAILY_RANK_FEATURE_COLS,
)


def _load_symbols(path: Path) -> list[str]:
    syms: list[str] = []
    for line in path.read_text().splitlines():
        s = line.strip().upper()
        if s and not s.startswith("#"):
            syms.append(s)
    seen: set[str] = set()
    return [s for s in syms if not (s in seen or seen.add(s))]


FAMILY_CHOICES = ("xgb", "lgb", "cat", "mlp")


def _build_model(family: str, seed: int, device: str, args) -> object:
    if family == "xgb":
        from xgbnew.model import XGBStockModel
        return XGBStockModel(
            device=device,
            n_estimators=int(args.n_estimators),
            max_depth=int(args.max_depth),
            learning_rate=float(args.learning_rate),
            random_state=int(seed),
        )
    if family == "lgb":
        from xgbnew.model_lgb import LGBMStockModel
        return LGBMStockModel(
            device=device,
            n_estimators=int(args.n_estimators),
            num_leaves=int(args.lgb_num_leaves),
            max_depth=int(args.max_depth) if int(args.max_depth) > 0 else -1,
            learning_rate=float(args.learning_rate),
            random_state=int(seed),
        )
    if family == "cat":
        from xgbnew.model_cat import CatBoostStockModel
        return CatBoostStockModel(
            device=device,
            iterations=int(args.n_estimators),
            depth=int(args.max_depth) if int(args.max_depth) > 0 else 6,
            learning_rate=float(args.learning_rate),
            random_seed=int(seed),
        )
    if family == "mlp":
        from xgbnew.model_mlp import MLPStockModel
        hidden_dims = [int(h) for h in args.mlp_hidden.split(",") if h.strip()]
        return MLPStockModel(
            device=device,
            hidden_dims=tuple(hidden_dims),
            dropout=float(args.mlp_dropout),
            learning_rate=float(args.mlp_lr),
            batch_size=int(args.mlp_batch),
            epochs=int(args.mlp_epochs),
            early_stop_patience=int(args.mlp_patience),
            weight_decay=float(args.mlp_weight_decay),
            random_state=int(seed),
        )
    raise ValueError(f"Unknown family: {family}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", required=True, choices=FAMILY_CHOICES,
                   help="Model family to train.")
    p.add_argument("--symbols-file", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=REPO / "trainingdata")
    p.add_argument("--chronos-cache", type=Path,
                   default=REPO / "analysis/top2_backtest/forecast_cache")
    p.add_argument("--train-start", default="2020-01-01")
    p.add_argument("--train-end", default="",
                   help="YYYY-MM-DD (inclusive). Empty = today.")
    p.add_argument("--min-dollar-vol", type=float, default=5_000_000.0)
    p.add_argument("--val-frac", type=float, default=0.0,
                   help="If >0, hold out last val_frac of the training window "
                        "for val/early-stop. 0 = no val split (default).")

    p.add_argument("--seeds", default="0,7,42,73,197",
                   help="Comma-separated seeds (one model per seed).")
    p.add_argument("--n-estimators", type=int, default=400,
                   help="Trees / iterations (ignored for MLP).")
    p.add_argument("--max-depth", type=int, default=5,
                   help="Tree depth (-1 for LGB default; CAT default 6).")
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--device", default="cpu",
                   help="'cpu', 'cuda', or 'cuda:N'.")

    # LightGBM knobs
    p.add_argument("--lgb-num-leaves", type=int, default=31)

    # MLP knobs
    p.add_argument("--mlp-hidden", default="256,128,64")
    p.add_argument("--mlp-dropout", type=float, default=0.2)
    p.add_argument("--mlp-lr", type=float, default=1e-3)
    p.add_argument("--mlp-batch", type=int, default=8192)
    p.add_argument("--mlp-epochs", type=int, default=30)
    p.add_argument("--mlp-patience", type=int, default=5)
    p.add_argument("--mlp-weight-decay", type=float, default=1e-5)

    # Feature flags (same as train_alltrain_ensemble.py)
    p.add_argument("--include-ranks", action="store_true")
    p.add_argument("--include-dispersion", action="store_true")

    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if len(seeds) < 1:
        print("ERROR: need at least 1 seed", file=sys.stderr)
        return 1

    symbols = _load_symbols(args.symbols_file)
    train_start = date.fromisoformat(args.train_start)
    train_end = date.fromisoformat(args.train_end) if args.train_end else date.today()

    chronos_cache = {}
    if args.chronos_cache.exists():
        chronos_cache = load_chronos_cache(args.chronos_cache)

    print(f"[train-family {args.family}] {len(symbols)} symbols | "
          f"train {train_start} → {train_end} | seeds={seeds} | "
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
        include_cross_sectional_ranks=bool(args.include_ranks),
        include_cross_sectional_dispersion=bool(args.include_dispersion),
    )
    print(f"[train-family {args.family}] dataset built in {time.perf_counter()-t0:.1f}s "
          f"rows={len(train_df):,} symbols={train_df['symbol'].nunique()}", flush=True)

    # Optional time-ordered val split
    val_df = None
    if args.val_frac > 0:
        train_df = train_df.sort_values(["date", "symbol"]).reset_index(drop=True)
        cut = int(len(train_df) * (1.0 - float(args.val_frac)))
        val_df = train_df.iloc[cut:].copy()
        train_df = train_df.iloc[:cut].copy()
        print(f"[train-family {args.family}] val split: "
              f"train={len(train_df):,} val={len(val_df):,}", flush=True)

    feature_cols = list(DAILY_FEATURE_COLS)
    if args.include_ranks:
        feature_cols += list(DAILY_RANK_FEATURE_COLS)
    if args.include_dispersion:
        feature_cols += list(DAILY_DISPERSION_FEATURE_COLS)
    print(f"[train-family {args.family}] feature_cols={len(feature_cols)} "
          f"ranks_on={bool(args.include_ranks)} disp_on={bool(args.include_dispersion)}",
          flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    saved: list[dict] = []
    for i, seed in enumerate(seeds, start=1):
        print(f"\n[train-family {args.family}] {i}/{len(seeds)} seed={seed}...",
              flush=True)
        t_fit = time.perf_counter()
        model = _build_model(args.family, seed, args.device, args)
        model.fit(train_df, feature_cols, val_df=val_df, verbose=args.verbose)
        out_pkl = args.out_dir / f"alltrain_seed{seed}.pkl"
        model.save(out_pkl)
        saved.append({
            "seed": int(seed),
            "path": str(out_pkl),
            "fit_seconds": round(time.perf_counter() - t_fit, 2),
        })
        print(f"  seed={seed} fit in {saved[-1]['fit_seconds']:.1f}s "
              f"-> {out_pkl.name}", flush=True)

    manifest = {
        "family": args.family,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "train_start": str(train_start),
        "train_end": str(train_end),
        "n_symbols_requested": len(symbols),
        "n_symbols_with_data": int(train_df["symbol"].nunique()),
        "n_rows": int(len(train_df)),
        "seeds": seeds,
        "models": saved,
        "config": {
            "family": args.family,
            "n_estimators": int(args.n_estimators),
            "max_depth": int(args.max_depth),
            "learning_rate": float(args.learning_rate),
            "device": args.device,
            "min_dollar_vol": float(args.min_dollar_vol),
            "include_ranks": bool(args.include_ranks),
            "include_dispersion": bool(args.include_dispersion),
            "feature_cols": feature_cols,
            "val_frac": float(args.val_frac),
            "lgb_num_leaves": int(args.lgb_num_leaves),
            "mlp_hidden": args.mlp_hidden,
            "mlp_dropout": float(args.mlp_dropout),
            "mlp_lr": float(args.mlp_lr),
            "mlp_batch": int(args.mlp_batch),
            "mlp_epochs": int(args.mlp_epochs),
            "mlp_patience": int(args.mlp_patience),
        },
        "blend_recipe": "predict_scores mean across seeds",
    }
    manifest_path = args.out_dir / "alltrain_ensemble.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n[train-family {args.family}] Manifest → {manifest_path}")
    print(f"[train-family {args.family}] Models   → {len(saved)} files in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
