#!/usr/bin/env python3
"""Build augmented daily stock training data via intrabar session shifting.

The key idea: load hourly OHLC data for each symbol and synthesise multiple
daily bar variants by shifting the session window forward by 1-4 bars.

  offset=0  → normal 9:30–16:00 ET daily bar
  offset=1  → bar starting 1h later (uses last N-1 hours of day i + first 1h of day i+1)
  offset=2  → shifted 2 hours
  …

Each offset produces a valid daily OHLC time series with slightly different
open/close/high/low values.  Concatenating offsets 0..K gives K×more training
rows without any lookahead bias (the shifted bar is still causal — it only
depends on already-elapsed hourly bars).

Then we run the standard export_data_daily pipeline on the augmented CSV to
produce a .bin file usable by the pufferlib_market C environment.

Usage:
    python scripts/build_augmented_daily_training.py \
        --symbols AAPL,MSFT,NVDA,GOOG,META,TSLA,SPY,QQQ,PLTR,JPM,V,AMZN,AMD,NFLX,COIN,CRWD,UBER \
        --hourly-root trainingdatahourly/stocks \
        --daily-root trainingdata \
        --offsets 0,1,2,3,4 \
        --output pufferlib_market/data/stocks17_augmented_train.bin \
        --start-date 2015-01-01 --end-date 2025-06-30
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.bar_aggregation import hourly_to_shifted_session_daily_ohlcv
from pufferlib_market.export_data_daily import export_binary, FEATURES_PER_SYM, PRICE_FEATURES


def _read_hourly(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = [str(c).lower() for c in df.columns]
    return df


def _read_daily(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = [str(c).lower() for c in df.columns]
    return df


def _generate_shifted_daily(
    hourly: pd.DataFrame,
    *,
    offset_bars: int,
    symbol: str,
) -> pd.DataFrame | None:
    """Return a daily OHLCV frame using shifted session bars, or None on failure."""
    try:
        shifted, stats = hourly_to_shifted_session_daily_ohlcv(
            hourly,
            offset_bars=offset_bars,
            require_full_shift=(offset_bars > 0),
        )
        if shifted.empty:
            return None
        # Normalize column names to match the daily data format
        shifted = shifted.rename(columns={"timestamp": "timestamp"})
        for col in ("open", "high", "low", "close"):
            if col not in shifted.columns:
                return None
        if "volume" not in shifted.columns:
            shifted["volume"] = 0
        shifted["trade_count"] = 0
        shifted["vwap"] = shifted["close"]
        shifted["symbol"] = symbol.upper()
        return shifted[["timestamp", "open", "high", "low", "close", "volume", "trade_count", "vwap", "symbol"]]
    except Exception as exc:
        print(f"  [{symbol}] offset={offset_bars}: {exc}", file=sys.stderr)
        return None


def build_augmented_csv(
    symbol: str,
    *,
    hourly_root: Path,
    daily_root: Path,
    offsets: list[int],
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame | None:
    """Build a concatenated daily DataFrame across all offsets for one symbol."""
    hourly_path = hourly_root / f"{symbol.upper()}.csv"
    daily_path = daily_root / f"{symbol.upper()}.csv"

    hourly = _read_hourly(hourly_path)
    daily_fallback = _read_daily(daily_path)

    parts: list[pd.DataFrame] = []

    for offset in offsets:
        if offset == 0 or hourly is None:
            # Use the original daily CSV for offset=0 (or when no hourly data available).
            # For symbols without hourly data, the same daily data is used for all offsets —
            # no augmentation benefit but ensures the symbol set stays consistent.
            if daily_fallback is None:
                continue
            part = daily_fallback.copy()
        else:
            part = _generate_shifted_daily(hourly, offset_bars=offset, symbol=symbol)
            if part is None:
                # Hourly shifting failed (e.g. ETF with pre-market data issues) — fall back
                # to original daily so this symbol still appears in the augmented segment.
                if daily_fallback is None:
                    continue
                print(f"  [{symbol}] offset={offset}: falling back to original daily", file=sys.stderr)
                part = daily_fallback.copy()

        # Normalize timestamps
        part = part.copy()
        part["timestamp"] = pd.to_datetime(part["timestamp"], utc=True, errors="coerce")
        part = part.dropna(subset=["timestamp"]).sort_values("timestamp")

        # Apply date range filter
        if start_date:
            ts_start = pd.Timestamp(start_date, tz="UTC")
            part = part[part["timestamp"] >= ts_start]
        if end_date:
            ts_end = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)
            part = part[part["timestamp"] < ts_end]

        # If shifted data is empty after filtering (e.g. hourly only covers recent period),
        # fall back to the original daily CSV so the symbol stays in the training set.
        if part.empty and offset != 0 and daily_fallback is not None:
            fb = daily_fallback.copy()
            fb["timestamp"] = pd.to_datetime(fb["timestamp"], utc=True, errors="coerce")
            fb = fb.dropna(subset=["timestamp"]).sort_values("timestamp")
            if start_date:
                fb = fb[fb["timestamp"] >= pd.Timestamp(start_date, tz="UTC")]
            if end_date:
                fb = fb[fb["timestamp"] < pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)]
            part = fb

        if not part.empty:
            parts.append(part.reset_index(drop=True))

    if not parts:
        return None

    combined = pd.concat(parts, ignore_index=True).sort_values("timestamp")
    combined = combined.drop_duplicates(subset=["timestamp"], keep="first")
    return combined.reset_index(drop=True)


def _concat_binaries(bin_paths: list[Path], output: Path) -> None:
    """Concatenate MKTD binary files with matching symbol/feature layout.

    MKTD format (export_data_daily.py):
      [64-byte header] [S×16 symbol table] [T×S×F float32 features]
      [T×S×5 float32 prices] [T×S uint8 tradable mask]

    This function reads each section explicitly so symbol tables are not
    duplicated and per-timestep arrays are concatenated correctly along axis 0.
    """
    import struct
    import numpy as np

    HEADER_SIZE = 64
    SYM_NAME_LEN = 16
    PRICE_FEATURES = 5

    def _read_mktd(path: Path):
        with open(path, "rb") as f:
            hdr_raw = f.read(HEADER_SIZE)
            if len(hdr_raw) != HEADER_SIZE:
                raise ValueError(f"Short header in {path}")
            # magic(4s) version(I) num_symbols(I) num_timesteps(I) features_per_sym(I) price_features(I) pad(40s)
            magic, version, syms, ts, feats, price_feats = struct.unpack_from("<4sIIIII", hdr_raw)
            sym_table = f.read(syms * SYM_NAME_LEN)
            feat_data = np.fromfile(f, dtype=np.float32, count=ts * syms * feats).reshape(ts, syms, feats)
            price_data = np.fromfile(f, dtype=np.float32, count=ts * syms * price_feats).reshape(ts, syms, price_feats)
            mask_raw = f.read()
            mask_data = np.frombuffer(mask_raw, dtype=np.uint8).reshape(ts, syms) if mask_raw else None
        return magic, version, syms, ts, feats, price_feats, hdr_raw, sym_table, feat_data, price_data, mask_data

    records = [_read_mktd(p) for p in bin_paths]

    # Validate all files have same layout
    ref = records[0]
    for i, r in enumerate(records):
        if r[0] != ref[0] or r[1] != ref[1] or r[2] != ref[2] or r[4] != ref[4] or r[5] != ref[5]:
            raise ValueError(
                f"Binary {bin_paths[i]} has incompatible layout vs {bin_paths[0]}: "
                f"magic={r[0]} ver={r[1]} syms={r[2]} feats={r[4]} price_feats={r[5]}"
            )

    total_timesteps = sum(r[3] for r in records)

    # Build merged header (update timestep_count at byte offset 12)
    new_header = bytearray(ref[6])
    struct.pack_into("<I", new_header, 12, total_timesteps)

    # Concatenate feature/price/mask arrays along timestep axis
    all_feats = np.concatenate([r[8] for r in records], axis=0)
    all_prices = np.concatenate([r[9] for r in records], axis=0)
    has_mask = all(r[10] is not None for r in records)
    all_masks = np.concatenate([r[10] for r in records], axis=0) if has_mask else None

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as f:
        f.write(bytes(new_header))
        f.write(ref[7])  # symbol table from first file (same for all)
        f.write(all_feats.tobytes(order="C"))
        f.write(all_prices.tobytes(order="C"))
        if all_masks is not None:
            f.write(all_masks.tobytes(order="C"))

    print(f"Concatenated {len(bin_paths)} binaries → {output} ({total_timesteps} total timesteps)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build augmented daily training data via session shifting")
    parser.add_argument("--symbols", required=True, help="Comma-separated symbol list")
    parser.add_argument("--hourly-root", default="trainingdatahourly/stocks", help="Hourly CSV directory")
    parser.add_argument("--daily-root", default="trainingdata", help="Daily CSV directory")
    parser.add_argument("--offsets", default="0,1,2,3,4", help="Comma-separated hour offsets to generate")
    parser.add_argument("--output", required=True, help="Output .bin path")
    parser.add_argument("--val-output", default=None, help="Optional val .bin output path")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--val-start-date", default=None)
    parser.add_argument("--val-end-date", default=None)
    parser.add_argument("--min-days", type=int, default=200)
    parser.add_argument("--cross-features", action="store_true", default=False,
                        help="Append 4 cross-symbol features (corr, beta, rel_return, breadth_rank)")
    parser.add_argument("--cross-anchor", default="SPY",
                        help="Anchor symbol for cross features (default: SPY)")
    parser.add_argument("--cross-window", type=int, default=20,
                        help="Rolling window in days for cross features (default: 20)")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    offsets = [int(x.strip()) for x in args.offsets.split(",") if x.strip()]
    hourly_root = Path(args.hourly_root)
    daily_root = Path(args.daily_root)
    output = Path(args.output)

    print(f"Building augmented training data: {len(symbols)} symbols, offsets={offsets}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="augmented_daily_"))
    print(f"Using temp dir: {tmp_dir}")

    try:
        # Per-offset: build symbol CSVs and export binary
        offset_bins: list[Path] = []
        for offset in offsets:
            offset_data_dir = tmp_dir / f"offset_{offset}"
            offset_data_dir.mkdir()

            missing = []
            for sym in symbols:
                df = build_augmented_csv(
                    sym,
                    hourly_root=hourly_root,
                    daily_root=daily_root,
                    offsets=[offset],
                    start_date=args.start_date,
                    end_date=args.end_date,
                )
                if df is None or df.empty:
                    missing.append(sym)
                    continue
                df.to_csv(offset_data_dir / f"{sym}.csv", index=False)

            if missing:
                print(f"  offset={offset}: MISSING {missing}")

            bin_path = tmp_dir / f"offset_{offset}.bin"
            print(f"  offset={offset}: exporting {len(symbols) - len(missing)} symbols → {bin_path.name}")
            # Augmented offsets (1+) often cover a shorter window than offset=0
            # (hourly data may only span a few months). Use a lower min_days so
            # these shorter-but-valid windows are included in the training mix.
            effective_min_days = args.min_days if offset == 0 else min(args.min_days, 50)
            try:
                export_binary(
                    symbols=symbols,
                    data_root=offset_data_dir,
                    output_path=bin_path,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    min_days=effective_min_days,
                    cross_features=args.cross_features,
                    cross_anchor=args.cross_anchor,
                    cross_window=args.cross_window,
                )
                offset_bins.append(bin_path)
            except Exception as e:
                print(f"  offset={offset}: export failed: {e}", file=sys.stderr)

        if not offset_bins:
            print("ERROR: No offset binaries generated.", file=sys.stderr)
            sys.exit(1)

        _concat_binaries(offset_bins, output)
        print(f"Train data → {output}")

        # Optional val set (offset=0 only, on val date range)
        if args.val_output and (args.val_start_date or args.val_end_date):
            val_output = Path(args.val_output)
            val_data_dir = tmp_dir / "val_offset_0"
            val_data_dir.mkdir()
            for sym in symbols:
                df = build_augmented_csv(
                    sym,
                    hourly_root=hourly_root,
                    daily_root=daily_root,
                    offsets=[0],
                    start_date=args.val_start_date,
                    end_date=args.val_end_date,
                )
                if df is not None and not df.empty:
                    df.to_csv(val_data_dir / f"{sym}.csv", index=False)
            try:
                export_binary(
                    symbols=symbols,
                    data_root=val_data_dir,
                    output_path=val_output,
                    start_date=args.val_start_date,
                    end_date=args.val_end_date,
                    min_days=50,  # val period is typically shorter than a full year
                    cross_features=args.cross_features,
                    cross_anchor=args.cross_anchor,
                    cross_window=args.cross_window,
                )
                print(f"Val data  → {val_output}")
            except Exception as e:
                print(f"Val export failed: {e}", file=sys.stderr)
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
