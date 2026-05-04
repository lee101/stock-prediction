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
import hashlib
import logging
import math
import sys
import time
from datetime import date
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from xgbnew.artifacts import save_model_atomic
from xgbnew.artifacts import write_json_atomic as _shared_write_json_atomic
from xgbnew.dataset import build_daily_dataset, load_chronos_cache, load_fm_latents
from xgbnew.features import (
    DAILY_DISPERSION_FEATURE_COLS,
    DAILY_FEATURE_COLS,
    DAILY_RANK_FEATURE_COLS,
)
from xgbnew.model import XGBStockModel


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict) -> None:
    _shared_write_json_atomic(path, payload)


def _load_symbols(path: Path) -> list[str]:
    syms: list[str] = []
    for line in path.read_text().splitlines():
        s = line.strip().upper()
        if s and not s.startswith("#"):
            syms.append(s)
    seen = set()
    return [s for s in syms if not (s in seen or seen.add(s))]


def _parse_seed_list(value: str) -> list[int]:
    seeds = [int(s.strip()) for s in value.split(",") if s.strip()]
    seen: set[int] = set()
    duplicates: list[int] = []
    for seed in seeds:
        if seed in seen and seed not in duplicates:
            duplicates.append(seed)
        seen.add(seed)
    if duplicates:
        raise ValueError(f"duplicate seeds are not allowed: {duplicates}")
    return seeds


def _fm_latent_columns(df) -> list[str]:
    cols = []
    for col in df.columns:
        col_name = str(col)
        if not col_name.startswith("latent_"):
            continue
        suffix = col_name.removeprefix("latent_")
        if suffix.isdecimal():
            cols.append((int(suffix), col_name))
    cols.sort()
    return [col for _idx, col in cols]


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
    p.add_argument("--include-dispersion", action="store_true",
                   help="Add 2 day-level cross-sectional dispersion features "
                        "(cs_iqr_ret5, cs_skew_ret5). Broadcast same value to "
                        "every symbol-row of the same date. Saved into pkl "
                        "feature_cols so predict-time uses them.")
    p.add_argument("--fm-latents-path", type=Path, default=None,
                   help="Optional foundation-model latents parquet "
                        "(scripts/build_chronos_bolt_latents.py output). "
                        "When provided, latent_0..latent_K and fm_available "
                        "are joined per (symbol, date) and added to the "
                        "training feature set. Saved into the pkl "
                        "feature_cols so predict-time uses them.")
    p.add_argument("--fm-n-latents", type=int, default=32,
                   help="Number of latent_N columns to keep from the "
                        "fm-latents parquet. Default 32 matches the PCA "
                        "output of build_chronos_bolt_latents.py.")
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


def _parse_date_arg(name: str, value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"--{name} must be an ISO date, got {value!r}") from exc


def _validate_positive_architecture(*, n_estimators: int, max_depth: int, learning_rate: float) -> None:
    if int(n_estimators) <= 0:
        raise ValueError("n_estimators must be positive")
    if int(max_depth) <= 0:
        raise ValueError("max_depth must be positive")
    if not math.isfinite(float(learning_rate)) or float(learning_rate) <= 0.0:
        raise ValueError("learning_rate must be finite and positive")


def _validate_args(args: argparse.Namespace, shapes: list[dict]) -> tuple[date, date]:
    train_start = _parse_date_arg("train-start", str(args.train_start))
    train_end = (
        _parse_date_arg("train-end", str(args.train_end))
        if args.train_end
        else date.today()
    )
    if train_start > train_end:
        raise ValueError("--train-start must be <= --train-end")
    if not math.isfinite(float(args.min_dollar_vol)) or float(args.min_dollar_vol) < 0.0:
        raise ValueError("--min-dollar-vol must be finite and nonnegative")
    if shapes:
        for idx, shape in enumerate(shapes, start=1):
            try:
                _validate_positive_architecture(
                    n_estimators=int(shape["n_est"]),
                    max_depth=int(shape["depth"]),
                    learning_rate=float(shape["lr"]),
                )
            except ValueError as exc:
                raise ValueError(f"--shapes tuple {idx}: {exc}") from exc
    else:
        _validate_positive_architecture(
            n_estimators=int(args.n_estimators),
            max_depth=int(args.max_depth),
            learning_rate=float(args.learning_rate),
        )
    return train_start, train_end


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")

    try:
        shapes = _parse_shapes(args.shapes) if args.shapes else []
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if shapes:
        # Architectural-diversity mode: shapes list drives training.
        seeds = [sp["seed"] for sp in shapes]
        if len(shapes) < 2:
            print("ERROR: --shapes needs at least 2 tuples", file=sys.stderr)
            return 1
    else:
        try:
            seeds = _parse_seed_list(args.seeds)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        if len(seeds) < 2:
            print("ERROR: need at least 2 seeds", file=sys.stderr)
            return 1

    try:
        train_start, train_end = _validate_args(args, shapes)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    fm_latents_df = None
    fm_latents_sha256 = None
    if args.fm_latents_path is not None:
        if int(args.fm_n_latents) <= 0:
            print(
                f"ERROR: --fm-n-latents must be positive when --fm-latents-path is set; "
                f"got {args.fm_n_latents}",
                file=sys.stderr,
            )
            return 1
        fm_latents_df = load_fm_latents(args.fm_latents_path)
        if fm_latents_df is None:
            print(f"ERROR: --fm-latents-path {args.fm_latents_path} not found",
                  file=sys.stderr)
            return 1
        latent_cols = _fm_latent_columns(fm_latents_df)
        if int(args.fm_n_latents) > len(latent_cols):
            print(
                f"ERROR: --fm-n-latents={args.fm_n_latents} exceeds artifact latent "
                f"columns ({len(latent_cols)})",
                file=sys.stderr,
            )
            return 1
        fm_latents_sha256 = _file_sha256(args.fm_latents_path)
        print(f"[xgb-alltrain-ens] fm latents: rows={len(fm_latents_df):,} "
              f"unique_symbols={fm_latents_df['symbol'].nunique()} "
              f"unique_dates={fm_latents_df['date'].nunique()} "
              f"cols={[c for c in fm_latents_df.columns if c.startswith('latent_')][:3]}…",
              flush=True)

    symbols = _load_symbols(args.symbols_file)

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
        include_cross_sectional_dispersion=bool(args.include_dispersion),
        fm_latents=fm_latents_df,
        fm_n_latents=int(args.fm_n_latents) if fm_latents_df is not None else None,
    )
    print(f"[xgb-alltrain-ens] dataset built in {time.perf_counter()-t0:.1f}s | "
          f"rows={len(train_df):,}  train_symbols={train_df['symbol'].nunique()}", flush=True)

    feature_cols = list(DAILY_FEATURE_COLS)
    if args.include_ranks:
        feature_cols += list(DAILY_RANK_FEATURE_COLS)
    if args.include_dispersion:
        feature_cols += list(DAILY_DISPERSION_FEATURE_COLS)
    if fm_latents_df is not None:
        n_lat = int(args.fm_n_latents)
        feature_cols += [f"latent_{i}" for i in range(n_lat)] + ["fm_available"]
    print(f"[xgb-alltrain-ens] feature_cols={len(feature_cols)} "
          f"ranks_on={bool(args.include_ranks)} disp_on={bool(args.include_dispersion)} "
          f"fm_latents_on={fm_latents_df is not None}", flush=True)

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
            save_model_atomic(model, out_pkl)
            saved.append({"seed": seed, "n_estimators": sp["n_est"],
                          "max_depth": sp["depth"], "learning_rate": sp["lr"],
                          "path": str(out_pkl),
                          "sha256": _file_sha256(out_pkl),
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
            save_model_atomic(model, out_pkl)
            saved.append({"seed": int(seed), "path": str(out_pkl),
                          "sha256": _file_sha256(out_pkl),
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
            "include_dispersion": bool(args.include_dispersion),
            "fm_latents_path": (
                str(args.fm_latents_path) if args.fm_latents_path else None
            ),
            "fm_latents_sha256": fm_latents_sha256,
            "fm_n_latents": (
                int(args.fm_n_latents) if fm_latents_df is not None else 0
            ),
            "feature_cols": feature_cols,
            "shapes_mode": bool(shapes),
            "shapes": shapes,
        },
        "blend_recipe": "predict_proba mean across seeds then pick top_n=1",
    }
    manifest_path = args.out_dir / "alltrain_ensemble.json"
    _write_json_atomic(manifest_path, manifest)
    print(f"\n[xgb-alltrain-ens] Manifest → {manifest_path}")
    print(f"[xgb-alltrain-ens] Models   → {len(saved)} files in {args.out_dir}")
    print("[xgb-alltrain-ens] ⚠ No OOS metrics — trust champion hyperparams only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
