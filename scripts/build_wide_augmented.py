#!/usr/bin/env python3
"""Build a wide augmented training dataset for daily RL trading.

Two augmentation axes:
  1. Session-shift (same as stocks17_augmented): use hourly data to shift
     the daily bar window by 0..N_OFFSETS-1 hours, giving N_OFFSETS daily
     variants of each trading day.
  2. Volatility-scale: after computing features, multiply all
     return/volatility/momentum features by a scale factor σ ∈ VOL_SCALES.
     This simulates different overall market volatility regimes (bull calm,
     bear volatile, etc.) without any lookahead bias.

Net data: len(OFFSETS) × len(VOL_SCALES) augmented variants, stacked along
the time axis exactly like the session-shift augmentation.

Feature indices scaled (all are ratio/percent based, not raw prices):
  0  return_1d       scale
  1  return_5d       scale
  2  return_20d      scale
  3  volatility_5d   scale
  4  volatility_20d  scale
  5  ma_delta_5d     scale
  6  ma_delta_20d    scale
  7  ma_delta_60d    scale
  8  atr_pct_14d     scale
  9  range_pct_1d    scale
  10 rsi_14          SKIP (ratio-of-gains, not linearly scalable)
  11 trend_60d       scale
  12 drawdown_20d    scale
  13 drawdown_60d    scale
  14 log_volume_z20d SKIP (z-score, already normalized)
  15 log_vol_delta   SKIP (log diff, already normalized)

Usage:
    python scripts/build_wide_augmented.py [--symbols SYM,...] [--out-dir ...]
"""
from __future__ import annotations

import argparse
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.build_augmented_daily_training import (
    _concat_binaries,
    build_augmented_csv,
)
from pufferlib_market.export_data_daily import export_binary, FEATURES_PER_SYM

# ─── Configuration ────────────────────────────────────────────────────────────

# Blue-chip S&P 500 stocks + major ETFs with 4+ year daily history.
# Chosen for diversity across sectors, all verified present in trainingdata/.
WIDE_SYMBOLS = [
    # Tech megacaps
    "AAPL", "MSFT", "NVDA", "GOOG", "GOOGL", "META", "AMZN", "AMD", "TSLA",
    "AVGO", "ADBE", "CRM", "INTC", "NFLX", "QCOM", "TXN", "AMAT", "ADI",
    "ASML", "NOW", "ORCL",
    # Finance
    "JPM", "BAC", "GS", "V", "MA", "COF", "SCHW", "BLK", "AXP", "C",
    "MS", "ICE",
    # Healthcare
    "LLY", "UNH", "JNJ", "ABBV", "ABT", "MRK", "DHR", "ISRG", "VRTX",
    "REGN", "GILD",
    # Consumer / Industrial
    "COST", "HD", "WMT", "MCD", "SBUX", "NKE", "PG", "KO", "PEP",
    "CAT", "DE", "HON",
    # Energy
    "XOM", "CVX", "COP",
    # Other tech/growth
    "PLTR", "CRWD", "DDOG", "PANW",
    # ETFs (index exposure)
    "SPY", "QQQ", "IWM", "VTI", "GLD",
    # Sector ETFs
    "XLK", "XLF", "XLV", "XLE", "XLI",
]

# Session-shift offsets: 0 = original daily bar, 1..N = shifted hourly variants.
SESSION_OFFSETS = [0, 1, 2, 3, 4]

# Volatility-scale factors applied to return/vol/momentum features.
# σ=1.0 is the original, σ<1 simulates low-vol regimes, σ>1 high-vol.
VOL_SCALES = [0.7, 1.0, 1.3]

# Feature indices that get scaled (return/vol/momentum ratios).
SCALE_FEAT_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13]

# Feature clip bounds (must match export_data_daily.compute_daily_features).
FEAT_CLIPS = {
    0: (-0.5,  0.5),   # return_1d
    1: (-1.0,  1.0),   # return_5d
    2: (-2.0,  2.0),   # return_20d
    3: ( 0.0,  1.0),   # volatility_5d
    4: ( 0.0,  1.0),   # volatility_20d
    5: (-0.5,  0.5),   # ma_delta_5d
    6: (-0.5,  0.5),   # ma_delta_20d
    7: (-0.5,  0.5),   # ma_delta_60d
    8: ( 0.0,  0.5),   # atr_pct_14d
    9: ( 0.0,  0.5),   # range_pct_1d
    11: (-3.0, 3.0),   # trend_60d
    12: (-1.0, 0.0),   # drawdown_20d
    13: (-1.0, 0.0),   # drawdown_60d
}

# ─── Helpers ──────────────────────────────────────────────────────────────────

HEADER_SIZE = 64
SYM_NAME_LEN = 16
PRICE_FEATURES = 5


def _read_mktd(path: Path):
    with open(path, "rb") as f:
        hdr_raw = f.read(HEADER_SIZE)
        magic, version, nsym, nts, nfeat, nprice = struct.unpack_from("<4sIIIII", hdr_raw)
        sym_table = f.read(nsym * SYM_NAME_LEN)
        feat = np.fromfile(f, dtype=np.float32, count=nts * nsym * nfeat).reshape(nts, nsym, nfeat)
        price = np.fromfile(f, dtype=np.float32, count=nts * nsym * nprice).reshape(nts, nsym, nprice)
        mask_raw = f.read()
        mask = np.frombuffer(mask_raw, dtype=np.uint8).reshape(nts, nsym) if mask_raw else None
    return dict(hdr=hdr_raw, sym_table=sym_table, feat=feat, price=price, mask=mask,
                nsym=nsym, nts=nts, nfeat=nfeat, nprice=nprice, version=version, magic=magic)


def _write_mktd(path: Path, ref: dict, feat: np.ndarray, price: np.ndarray, mask):
    nts = feat.shape[0]
    nfeat = feat.shape[2]  # may differ from ref["nfeat"] if cross-features appended
    new_hdr = bytearray(ref["hdr"])
    struct.pack_into("<I", new_hdr, 12, nts)
    struct.pack_into("<I", new_hdr, 16, nfeat)  # update features_per_sym in header
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(bytes(new_hdr))
        f.write(ref["sym_table"])
        f.write(feat.astype(np.float32).tobytes(order="C"))
        f.write(price.astype(np.float32).tobytes(order="C"))
        if mask is not None:
            f.write(mask.astype(np.uint8).tobytes(order="C"))
    print(f"  Wrote {path} ({nts} timesteps, {ref['nsym']} symbols, {nfeat} feats)")


def apply_vol_scale(feat: np.ndarray, sigma: float) -> np.ndarray:
    """Return feature array with return/vol features scaled by sigma and re-clipped."""
    if abs(sigma - 1.0) < 1e-6:
        return feat  # no-op for σ=1
    out = feat.copy()
    for fi in SCALE_FEAT_IDX:
        out[:, :, fi] *= sigma
        if fi in FEAT_CLIPS:
            lo, hi = FEAT_CLIPS[fi]
            out[:, :, fi] = out[:, :, fi].clip(lo, hi)
    return out


def _compute_cross_for_record(rec: dict, symbols: list[str], anchor: str = "SPY", window: int = 20) -> np.ndarray:
    """Compute cross-symbol features [T, S, 4] from close prices in an MKTD record."""
    from pufferlib_market.cross_symbol_features import compute_cross_features
    close_prices = rec["price"][:, :, 3]  # OHLCV index 3 = close
    return compute_cross_features(close_prices, symbols, window=window, anchor_symbol=anchor)


def append_cross_features(rec: dict, cross: np.ndarray) -> dict:
    """Return a new record with cross features appended to feat [T, S, F] → [T, S, F+4]."""
    feat = np.concatenate([rec["feat"], cross], axis=2)
    return {**rec, "feat": feat, "nfeat": feat.shape[2]}


def concat_mktd_list(parts: list[dict]) -> dict:
    """Concatenate MKTD records along time axis."""
    ref = parts[0]
    all_feat = np.concatenate([p["feat"] for p in parts], axis=0)
    all_price = np.concatenate([p["price"] for p in parts], axis=0)
    has_mask = all(p["mask"] is not None for p in parts)
    all_mask = np.concatenate([p["mask"] for p in parts], axis=0) if has_mask else None
    return {**ref, "feat": all_feat, "price": all_price, "mask": all_mask,
            "nts": all_feat.shape[0]}


# ─── Main builder ─────────────────────────────────────────────────────────────

def build_wide_augmented(
    *,
    symbols: list[str],
    hourly_root: Path,
    daily_root: Path,
    offsets: list[int],
    vol_scales: list[float],
    output_train: Path,
    output_val: Path,
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
    min_days: int = 200,
    cross_features: bool = False,
    cross_anchor: str = "SPY",
    cross_window: int = 20,
) -> None:
    print(f"Building wide augmented dataset:")
    print(f"  symbols: {len(symbols)} — {symbols[:5]}...")
    print(f"  session offsets: {offsets}")
    print(f"  vol scales: {vol_scales}")
    print(f"  train: {train_start} → {train_end}")
    print(f"  val:   {val_start} → {val_end}")

    # Filter symbols to those that have data in daily_root.
    available = []
    for sym in symbols:
        if (daily_root / f"{sym}.csv").exists():
            available.append(sym)
        else:
            print(f"  [skip] {sym}: no daily CSV")
    symbols = available
    print(f"  {len(symbols)} symbols with daily data")

    if len(symbols) > 128:
        print(f"  Truncating to 128 symbols (C env max)")
        symbols = symbols[:128]

    # Use a persistent intermediate directory (survives crashes, allows resume).
    cache_dir = output_train.parent / f".build_cache_{output_train.stem}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"  intermediate cache: {cache_dir}")

    # Build one base binary per session-offset for train and val.
    train_offset_bins: list[Path] = []
    val_offset_bins: list[Path] = []

    for offset in offsets:
        out_train = cache_dir / f"offset{offset}_train.bin"
        out_val   = cache_dir / f"offset{offset}_val.bin"

        # Resume: skip already-built offset binaries.
        if out_train.exists() and out_val.exists():
            print(f"\n--- Session offset {offset}: CACHED (skip) ---")
            train_offset_bins.append(out_train)
            val_offset_bins.append(out_val)
            continue

        print(f"\n--- Session offset {offset} ---")

        # Build augmented CSVs (same logic as build_augmented_daily_training.py).
        combined_train_csvs: dict[str, pd.DataFrame] = {}
        combined_val_csvs: dict[str, pd.DataFrame] = {}
        # Val is only needed for offset=0 (unshifted bars); skip val CSV
        # computation for other offsets to avoid wasted work + alignment errors.
        need_val = (offset == 0)

        for i, sym in enumerate(symbols):
            if i % 10 == 0:
                print(f"  [{i+1}/{len(symbols)}] processing {sym}...")
            df_train = build_augmented_csv(
                sym,
                hourly_root=hourly_root,
                daily_root=daily_root,
                offsets=[offset],
                start_date=train_start,
                end_date=train_end,
            )
            if df_train is not None:
                combined_train_csvs[sym] = df_train
            if need_val:
                df_val = build_augmented_csv(
                    sym,
                    hourly_root=hourly_root,
                    daily_root=daily_root,
                    offsets=[offset],
                    start_date=val_start,
                    end_date=val_end,
                )
                if df_val is not None:
                    combined_val_csvs[sym] = df_val

        # Verify all symbols present.
        sym_train = [s for s in symbols if s in combined_train_csvs]
        sym_val   = [s for s in symbols if s in combined_val_csvs]

        csv_dir_train = cache_dir / f"csv_offset{offset}_train"
        csv_dir_train.mkdir(exist_ok=True)
        for sym, df in combined_train_csvs.items():
            df.to_csv(csv_dir_train / f"{sym}.csv", index=False)
        export_binary(
            sym_train,
            csv_dir_train,
            out_train,
            start_date=train_start,
            end_date=train_end,
            min_days=min_days if offset == 0 else min(min_days, 50),
        )

        # Val: only build from offset=0 (unshifted daily bars).
        # Session-shifted val periods are shorter and may not align across
        # symbols, causing export_binary to fail. Val is used for checkpoint
        # selection only; σ=1.0 + offset=0 is the canonical unbiased split.
        if offset == 0:
            csv_dir_val = cache_dir / f"csv_offset{offset}_val"
            csv_dir_val.mkdir(exist_ok=True)
            for sym, df in combined_val_csvs.items():
                df.to_csv(csv_dir_val / f"{sym}.csv", index=False)
            export_binary(
                sym_val,
                csv_dir_val,
                out_val,
                start_date=val_start,
                end_date=val_end,
                min_days=20,
            )
            val_offset_bins.append(out_val)

        train_offset_bins.append(out_train)

        # ── Apply vol-scale augmentation across all session-offset binaries ──
        print(f"\n--- Applying vol-scale augmentation {vol_scales} ---")
        all_train_parts: list[dict] = []
        all_val_parts: list[dict] = []

        # Pre-compute cross-features once per session-offset binary (they are
        # vol-scale invariant since they derive from log-returns of prices).
        train_cross: dict[Path, np.ndarray] = {}
        val_cross: dict[Path, np.ndarray] = {}
        if cross_features:
            print(f"  Computing cross-symbol features (anchor={cross_anchor}, window={cross_window}d)...")
            for bp in train_offset_bins:
                rec = _read_mktd(bp)
                train_cross[bp] = _compute_cross_for_record(rec, symbols, cross_anchor, cross_window)
            for bp in val_offset_bins:
                rec = _read_mktd(bp)
                val_cross[bp] = _compute_cross_for_record(rec, symbols, cross_anchor, cross_window)

        for sigma in vol_scales:
            for bp in train_offset_bins:
                rec = _read_mktd(bp)
                rec["feat"] = apply_vol_scale(rec["feat"], sigma)
                if cross_features:
                    rec = append_cross_features(rec, train_cross[bp])
                all_train_parts.append(rec)
            for bp in val_offset_bins:
                rec = _read_mktd(bp)
                rec["feat"] = apply_vol_scale(rec["feat"], sigma)
                if cross_features:
                    rec = append_cross_features(rec, val_cross[bp])
                all_val_parts.append(rec)

        # Val: use only offset=0, σ=1.0 (unshifted, unscaled daily bars).
        # val_offset_bins always contains exactly one entry (offset0_val.bin).
        val_merged = _read_mktd(val_offset_bins[0])  # σ=1.0 unscaled
        if cross_features:
            val_merged = append_cross_features(val_merged, val_cross[val_offset_bins[0]])

        train_merged = concat_mktd_list(all_train_parts)

        base_days = all_train_parts[0]['nts'] if all_train_parts else 0
        print(f"\n  Train: {train_merged['nts']} timesteps "
              f"({len(train_offset_bins)} offsets × {len(vol_scales)} scales × "
              f"{base_days:.0f} base days, offset-weighted)")
        print(f"  Val:   {val_merged['nts']} timesteps (offset=0, σ=1.0 only)")

        _write_mktd(output_train, train_merged, train_merged["feat"],
                    train_merged["price"], train_merged["mask"])
        _write_mktd(output_val, val_merged, val_merged["feat"],
                    val_merged["price"], val_merged["mask"])

    print(f"\nDone!")
    print(f"  Train → {output_train}")
    print(f"  Val   → {output_val}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build wide augmented daily training binary")
    parser.add_argument("--symbols", default=",".join(WIDE_SYMBOLS),
                        help="Comma-separated symbol list (default: curated 65-symbol S&P500 list)")
    parser.add_argument("--hourly-root", default="trainingdatahourly/stocks")
    parser.add_argument("--daily-root", default="trainingdata")
    parser.add_argument("--offsets", default=",".join(map(str, SESSION_OFFSETS)))
    parser.add_argument("--vol-scales", default=",".join(map(str, VOL_SCALES)),
                        help="Comma-separated volatility scale factors (default: 0.7,1.0,1.3)")
    parser.add_argument("--output-train", default="pufferlib_market/data/wide_augmented_train.bin")
    parser.add_argument("--output-val", default="pufferlib_market/data/wide_augmented_val.bin")
    parser.add_argument("--train-start", default="2019-01-01")
    parser.add_argument("--train-end", default="2025-05-31")
    parser.add_argument("--val-start", default="2025-06-01")
    parser.add_argument("--val-end", default="2025-11-30")
    parser.add_argument("--min-days", type=int, default=200)
    args = parser.parse_args()

    build_wide_augmented(
        symbols=[s.strip().upper() for s in args.symbols.split(",") if s.strip()],
        hourly_root=Path(args.hourly_root),
        daily_root=Path(args.daily_root),
        offsets=[int(x) for x in args.offsets.split(",") if x.strip()],
        vol_scales=[float(x) for x in args.vol_scales.split(",") if x.strip()],
        output_train=Path(args.output_train),
        output_val=Path(args.output_val),
        train_start=args.train_start,
        train_end=args.train_end,
        val_start=args.val_start,
        val_end=args.val_end,
        min_days=args.min_days,
    )


if __name__ == "__main__":
    main()
