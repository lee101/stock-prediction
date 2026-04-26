#!/usr/bin/env python3
"""Build session-shift + vol-scale augmented crypto30 daily training binary.

For each of the 30 crypto symbols:
  1. Load hourly data (try USDT then USD naming)
  2. Generate 5 session-shifted daily variants (0-4 hour offsets)
  3. Export each offset to a binary
  4. Apply 3x vol-scale augmentation (0.7, 1.0, 1.3)
  5. Concatenate along time axis

Net: 5 offsets x 3 vol-scales = 15x augmentation.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pufferlib_market.export_data_daily import export_binary
from src.bar_aggregation import hourly_to_shifted_session_daily_ohlcv

CRYPTO30 = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "AVAXUSDT",
    "LINKUSDT", "AAVEUSDT", "LTCUSDT", "XRPUSDT", "DOTUSDT",
    "UNIUSDT", "NEARUSDT", "APTUSDT", "ICPUSDT", "SHIBUSDT",
    "ADAUSDT", "FILUSDT", "ARBUSDT", "OPUSDT", "INJUSDT",
    "SUIUSDT", "TIAUSDT", "SEIUSDT", "ATOMUSDT", "ALGOUSDT",
    "BCHUSDT", "BNBUSDT", "TRXUSDT", "PEPEUSDT", "MATICUSDT",
]

HOURLY_ROOT = REPO / "trainingdatahourly" / "crypto"
DAILY_ROOT = REPO / "trainingdata" / "train"
DATA_DIR = REPO / "pufferlib_market" / "data"

OFFSETS = [0, 1, 2, 3, 4]
VOL_SCALES = [0.7, 1.0, 1.3]

TRAIN_START = "2020-01-01"
TRAIN_END = "2025-09-30"
VAL_START = "2025-10-01"
VAL_END = "2026-04-10"

HEADER_SIZE = 64
SYM_NAME_LEN = 16

# Vol-scale config (same as build_crypto30_augmented.py)
SCALE_FEAT_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13]
FEAT_CLIPS = {
    0: (-0.5,  0.5), 1: (-1.0,  1.0), 2: (-2.0,  2.0),
    3: ( 0.0,  1.0), 4: ( 0.0,  1.0), 5: (-0.5,  0.5),
    6: (-0.5,  0.5), 7: (-0.5,  0.5), 8: ( 0.0,  0.5),
    9: ( 0.0,  0.5), 11: (-3.0, 3.0), 12: (-1.0, 0.0), 13: (-1.0, 0.0),
}


def read_mktd(path: Path):
    with open(path, "rb") as f:
        hdr_raw = f.read(HEADER_SIZE)
        magic, version, nsym, nts, nfeat, nprice = struct.unpack_from("<4sIIIII", hdr_raw)
        sym_table = f.read(nsym * SYM_NAME_LEN)
        feat = np.fromfile(f, dtype=np.float32, count=nts * nsym * nfeat).reshape(nts, nsym, nfeat)
        price = np.fromfile(f, dtype=np.float32, count=nts * nsym * nprice).reshape(nts, nsym, nprice)
        mask_raw = f.read()
        mask = np.frombuffer(mask_raw, dtype=np.uint8).reshape(nts, nsym) if mask_raw else None
    return dict(hdr=hdr_raw, sym_table=sym_table, feat=feat, price=price, mask=mask,
                nsym=nsym, nts=nts, nfeat=nfeat, nprice=nprice)


def write_mktd(path: Path, ref: dict, feat: np.ndarray, price: np.ndarray, mask):
    nts = feat.shape[0]
    new_hdr = bytearray(ref["hdr"])
    struct.pack_into("<I", new_hdr, 12, nts)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(bytes(new_hdr))
        f.write(ref["sym_table"])
        f.write(feat.astype(np.float32).tobytes(order="C"))
        f.write(price.astype(np.float32).tobytes(order="C"))
        if mask is not None:
            f.write(mask.astype(np.uint8).tobytes(order="C"))
    print(f"  Wrote {path} ({nts}ts, {ref['nsym']}sym, {feat.shape[2]}feat)")


def apply_vol_scale(feat: np.ndarray, sigma: float) -> np.ndarray:
    if abs(sigma - 1.0) < 1e-6:
        return feat
    out = feat.copy()
    for fi in SCALE_FEAT_IDX:
        out[:, :, fi] *= sigma
        if fi in FEAT_CLIPS:
            lo, hi = FEAT_CLIPS[fi]
            out[:, :, fi] = out[:, :, fi].clip(lo, hi)
    return out


def find_hourly_csv(sym: str) -> Path | None:
    """Find hourly CSV for a symbol, trying USDT then USD naming."""
    usdt = HOURLY_ROOT / f"{sym}.csv"
    if usdt.exists():
        return usdt
    usd = HOURLY_ROOT / f"{sym.replace('USDT', 'USD')}.csv"
    if usd.exists():
        return usd
    return None


def build_shifted_daily(sym: str, offset: int, start: str, end: str) -> pd.DataFrame | None:
    """Build shifted daily OHLCV from hourly data for one symbol+offset."""
    if offset == 0:
        # Use original daily CSV
        daily_path = DAILY_ROOT / f"{sym}.csv"
        if not daily_path.exists():
            return None
        df = pd.read_csv(daily_path)
        df.columns = [str(c).lower() for c in df.columns]
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        ts_start = pd.Timestamp(start, tz="UTC")
        ts_end = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
        df = df[(df["timestamp"] >= ts_start) & (df["timestamp"] < ts_end)]
        return df if not df.empty else None

    hourly_path = find_hourly_csv(sym)
    if hourly_path is None:
        # Fall back to original daily for symbols without hourly
        return build_shifted_daily(sym, 0, start, end)

    hourly = pd.read_csv(hourly_path)
    hourly.columns = [str(c).lower() for c in hourly.columns]

    # Force consistent symbol name
    if "symbol" in hourly.columns:
        hourly["symbol"] = sym

    try:
        shifted, stats = hourly_to_shifted_session_daily_ohlcv(
            hourly, offset_bars=offset, require_full_shift=(offset > 0),
        )
        if shifted.empty:
            return build_shifted_daily(sym, 0, start, end)

        shifted["timestamp"] = pd.to_datetime(shifted["timestamp"], utc=True, errors="coerce")
        shifted = shifted.dropna(subset=["timestamp"]).sort_values("timestamp")

        ts_start = pd.Timestamp(start, tz="UTC")
        ts_end = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
        shifted = shifted[(shifted["timestamp"] >= ts_start) & (shifted["timestamp"] < ts_end)]

        if shifted.empty:
            return build_shifted_daily(sym, 0, start, end)

        # Ensure required columns
        for col in ("volume", "trade_count", "vwap"):
            if col not in shifted.columns:
                shifted[col] = 0
        shifted["symbol"] = sym

        return shifted
    except Exception as e:
        print(f"  [{sym}] offset={offset}: {e}")
        return build_shifted_daily(sym, 0, start, end)


def build_offset_binary(offset: int, split: str, start: str, end: str, cache_dir: Path) -> Path | None:
    """Build one binary for a given offset and split."""
    out_path = cache_dir / f"offset{offset}_{split}.bin"
    if out_path.exists():
        print(f"  offset={offset} {split}: CACHED")
        return out_path

    csv_dir = cache_dir / f"csv_offset{offset}_{split}"
    csv_dir.mkdir(parents=True, exist_ok=True)

    valid_syms = []
    for sym in CRYPTO30:
        df = build_shifted_daily(sym, offset, start, end)
        if df is not None and len(df) >= 30:
            df.to_csv(csv_dir / f"{sym}.csv", index=False)
            valid_syms.append(sym)

    if not valid_syms:
        print(f"  offset={offset} {split}: no valid symbols!")
        return None

    print(f"  offset={offset} {split}: {len(valid_syms)}/{len(CRYPTO30)} symbols")
    export_binary(
        valid_syms, csv_dir, out_path,
        start_date=start, end_date=end,
        min_days=30,
        union_dates=True,
    )
    return out_path


def main():
    cache_dir = DATA_DIR / ".build_cache_crypto30_sessaug"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building crypto30 session-shift + vol-scale augmented dataset")
    print(f"  offsets: {OFFSETS}, vol_scales: {VOL_SCALES}")
    print(f"  total augmentation: {len(OFFSETS) * len(VOL_SCALES)}x")

    # Build per-offset binaries for train
    train_bins = []
    for offset in OFFSETS:
        p = build_offset_binary(offset, "train", TRAIN_START, TRAIN_END, cache_dir)
        if p:
            train_bins.append(p)

    # Val: only offset=0 (no augmentation for val)
    val_bin = build_offset_binary(0, "val", VAL_START, VAL_END, cache_dir)

    if not train_bins:
        print("ERROR: no training binaries produced!")
        return 1

    # Read all offset binaries and concatenate with vol-scaling
    all_parts = []
    for bin_path in train_bins:
        rec = read_mktd(bin_path)
        for sigma in VOL_SCALES:
            scaled_feat = apply_vol_scale(rec["feat"], sigma)
            all_parts.append((scaled_feat, rec["price"], rec.get("mask")))
            print(f"  {bin_path.stem} sigma={sigma}: {scaled_feat.shape[0]}ts")

    # Concatenate along time axis
    ref = read_mktd(train_bins[0])
    combined_feat = np.concatenate([p[0] for p in all_parts], axis=0)
    combined_price = np.concatenate([p[1] for p in all_parts], axis=0)
    has_mask = all(p[2] is not None for p in all_parts)
    combined_mask = np.concatenate([p[2] for p in all_parts], axis=0) if has_mask else None

    out_train = DATA_DIR / "crypto30_daily_sess_aug_train.bin"
    write_mktd(out_train, ref, combined_feat, combined_price, combined_mask)
    print(f"\nTrain: {combined_feat.shape[0]}ts = {len(OFFSETS)} offsets x {len(VOL_SCALES)} scales")

    # Val: just copy as-is (no augmentation)
    if val_bin:
        import shutil
        out_val = DATA_DIR / "crypto30_daily_sess_aug_val.bin"
        shutil.copy2(val_bin, out_val)
        val_rec = read_mktd(out_val)
        print(f"Val: {val_rec['nts']}ts ({val_rec['nsym']}sym)")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
