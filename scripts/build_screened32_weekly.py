#!/usr/bin/env python3
"""
Build screened32 weekly MKTD datasets with vol-scale augmentation.

Produces three binary files:
  screened32_weekly_train.bin       — 2019-2025-05 (train)
  screened32_weekly_val.bin         — 2025-06 to 2025-11 (OOS val)
  screened32_weekly_recent_val.bin  — 2025-12 to 2026-04 (recent bear market)

Augmentation: 7 vol scales = 7x training data.
With --chronos-cache adds 4 Chronos2 7d forecast features (features_per_sym=20).

Usage:
    python scripts/build_screened32_weekly.py
    python scripts/build_screened32_weekly.py --chronos-cache strategytraining/forecast_cache
"""
from __future__ import annotations

import argparse
import shutil
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np

from pufferlib_market.export_data_weekly import export_binary as _export_weekly
from scripts.build_screened_augmented import SCREENED_SYMBOLS

VOL_SCALES = [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5]


def _scale_prices_in_bin(src: Path, dst: Path, vol_scale: float) -> None:
    """Scale price data in a MKTD binary by vol_scale (features unchanged)."""
    if abs(vol_scale - 1.0) < 1e-9:
        shutil.copy2(src, dst)
        return

    with open(src, "rb") as fh:
        raw = bytearray(fh.read())

    _, _, num_symbols, num_timesteps, features_per_sym, price_features = struct.unpack_from("<4sIIIII", raw[:24])
    sym_table_size = num_symbols * 16
    feat_bytes = num_timesteps * num_symbols * features_per_sym * 4
    price_bytes = num_timesteps * num_symbols * price_features * 4
    price_offset = 64 + sym_table_size + feat_bytes

    prices = np.frombuffer(raw[price_offset: price_offset + price_bytes], dtype=np.float32).copy()
    prices *= np.float32(vol_scale)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as fh:
        fh.write(raw[:price_offset])
        fh.write(prices.tobytes())
        fh.write(raw[price_offset + price_bytes:])


def _concat_mktd_bins(bin_paths: list[Path], output_path: Path) -> None:
    """Concatenate MKTD v2 binaries along the timestep dimension."""
    if not bin_paths:
        raise ValueError("No input paths")

    with open(bin_paths[0], "rb") as fh:
        header = fh.read(64)
    magic, version, num_symbols, _, features_per_sym, price_features = struct.unpack_from("<4sIIIII", header)
    sym_table_size = num_symbols * 16

    all_features, all_prices, all_masks = [], [], []
    total_timesteps = 0

    for path in bin_paths:
        with open(path, "rb") as fh:
            raw = fh.read()
        _, _, n_sym, n_ts, n_feat, n_price = struct.unpack_from("<4sIIIII", raw[:24])
        assert n_sym == num_symbols and n_feat == features_per_sym

        off = 64 + sym_table_size
        feat_bytes = n_ts * n_sym * n_feat * 4
        price_bytes = n_ts * n_sym * n_price * 4
        mask_bytes = n_ts * n_sym

        all_features.append(np.frombuffer(raw[off: off + feat_bytes], dtype=np.float32).reshape(n_ts, n_sym, n_feat))
        off += feat_bytes
        all_prices.append(np.frombuffer(raw[off: off + price_bytes], dtype=np.float32).reshape(n_ts, n_sym, n_price))
        off += price_bytes
        all_masks.append(np.frombuffer(raw[off: off + mask_bytes], dtype=np.uint8).reshape(n_ts, n_sym))
        total_timesteps += n_ts

    with open(bin_paths[0], "rb") as fh:
        sym_table = fh.read()[64: 64 + sym_table_size]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as fh:
        fh.write(struct.pack("<4sIIIII40s", magic, version, num_symbols, total_timesteps,
                             features_per_sym, price_features, b"\x00" * 40))
        fh.write(sym_table)
        fh.write(np.concatenate(all_features, axis=0).tobytes(order="C"))
        fh.write(np.concatenate(all_prices, axis=0).tobytes(order="C"))
        fh.write(np.concatenate(all_masks, axis=0).tobytes(order="C"))

    print(f"  concat {len(bin_paths)} shards → {total_timesteps} weeks → {output_path}")


def build_weekly_augmented(
    symbols: list[str],
    data_root: Path,
    vol_scales: list[float],
    output_train: Path,
    output_val: Path,
    output_recent_val: Path,
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
    recent_val_start: str,
    recent_val_end: str,
    chronos_cache: Path | None,
    tmp_dir: Path,
) -> None:
    tmp_dir.mkdir(parents=True, exist_ok=True)

    base_train = tmp_dir / "base_train.bin"
    base_val = tmp_dir / "base_val.bin"
    base_recent_val = tmp_dir / "base_recent_val.bin"

    print("\n=== Building base weekly binaries ===")
    for out, start, end, label, min_w in [
        (base_train, train_start, train_end, "train", 26),
        (base_val, val_start, val_end, "val", 20),
        (base_recent_val, recent_val_start, recent_val_end, "recent_val", 10),
    ]:
        if out.exists():
            print(f"  {label}: exists, skip")
            continue
        print(f"  {label}: {start} → {end}")
        _export_weekly(symbols=symbols, data_root=data_root, output_path=out,
                       start_date=start, end_date=end, min_weeks=min_w, chronos_cache=chronos_cache)

    print(f"\n=== Applying {len(vol_scales)} vol scales to train ===")
    train_shards: list[Path] = []
    for scale in vol_scales:
        shard = tmp_dir / f"train_vol{scale:.2f}.bin"
        train_shards.append(shard)
        if shard.exists():
            print(f"  scale={scale}: exists, skip")
            continue
        print(f"  scale={scale}")
        _scale_prices_in_bin(base_train, shard, scale)

    print("\n=== Concatenating augmented train shards ===")
    _concat_mktd_bins(train_shards, output_train)

    if not output_val.exists():
        shutil.copy2(base_val, output_val)
        print(f"  val → {output_val}")
    if not output_recent_val.exists():
        shutil.copy2(base_recent_val, output_recent_val)
        print(f"  recent_val → {output_recent_val}")

    for path in [output_train, output_val, output_recent_val]:
        if path.exists():
            with open(path, "rb") as fh:
                _, _, n_sym, n_ts, n_feat, _ = struct.unpack_from("<4sIIIII", fh.read(24))
            print(f"  {path.name}: {n_sym} syms × {n_ts} weeks × {n_feat} feats")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build screened32 weekly augmented datasets")
    parser.add_argument("--out-prefix", default="screened32_weekly")
    parser.add_argument("--data-root", default="trainingdata")
    parser.add_argument("--chronos-cache", default=None,
                        help="Daily forecast parquet dir (adds 4 weekly Chronos features)")
    parser.add_argument("--train-start", default="2019-01-01")
    parser.add_argument("--train-end", default="2025-05-31")
    parser.add_argument("--val-start", default="2025-06-01")
    parser.add_argument("--val-end", default="2025-11-30")
    parser.add_argument("--recent-val-start", default="2025-12-01")
    parser.add_argument("--recent-val-end", default="2026-04-11")
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--tmp-dir", default=None)
    args = parser.parse_args()

    symbols = SCREENED_SYMBOLS
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    out_dir = REPO / "pufferlib_market" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(args.tmp_dir) if args.tmp_dir else REPO / ".tmp_weekly_build"
    chronos_cache = Path(args.chronos_cache) if args.chronos_cache else None

    print(f"Symbols ({len(symbols)}): {', '.join(symbols)}")
    print(f"Chronos cache: {chronos_cache or 'none (16 features)'}")

    build_weekly_augmented(
        symbols=symbols,
        data_root=REPO / args.data_root,
        vol_scales=VOL_SCALES,
        output_train=out_dir / f"{args.out_prefix}_train.bin",
        output_val=out_dir / f"{args.out_prefix}_val.bin",
        output_recent_val=out_dir / f"{args.out_prefix}_recent_val.bin",
        train_start=args.train_start, train_end=args.train_end,
        val_start=args.val_start, val_end=args.val_end,
        recent_val_start=args.recent_val_start, recent_val_end=args.recent_val_end,
        chronos_cache=chronos_cache,
        tmp_dir=tmp_dir,
    )

    print("\nDone. Run sweeps:")
    p = args.out_prefix
    print(f"  nohup bash scripts/sweep_screened32_weekly.sh C {p} > /tmp/s32w_C.log 2>&1 &")
    print(f"  nohup bash scripts/sweep_screened32_weekly.sh D {p} > /tmp/s32w_D.log 2>&1 &")


if __name__ == "__main__":
    main()
