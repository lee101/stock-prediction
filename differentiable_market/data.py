from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .config import DataConfig


REQUIRED_COLUMNS = ("open", "high", "low", "close")


def _discover_files(cfg: DataConfig) -> List[Path]:
    root = cfg.root
    if not root.exists():
        raise FileNotFoundError(f"Data root {root} does not exist")
    files = sorted(root.glob(cfg.glob))
    if not files:
        raise FileNotFoundError(f"No files found under {root} with pattern {cfg.glob}")
    return files


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(col).strip().lower() for col in df.columns]
    if "timestamp" not in df.columns:
        raise ValueError(f"{path} missing 'timestamp' column")
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing OHLC columns {missing}")
    df = df[["timestamp", *REQUIRED_COLUMNS]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset="timestamp", keep="last")
    df = df.set_index("timestamp")
    df = df.astype(np.float32)
    return df


def _filter_symbols(files: Sequence[Path], cfg: DataConfig) -> List[Tuple[str, Path]]:
    selected: List[Tuple[str, Path]] = []
    excluded = {sym.lower() for sym in cfg.exclude_symbols}
    for path in files:
        symbol = path.stem.upper()
        if symbol.lower() in excluded:
            continue
        selected.append((symbol, path))
        if cfg.max_assets is not None and len(selected) >= cfg.max_assets:
            break
    if not selected:
        raise ValueError("No symbols selected after applying filters")
    return selected


def _cache_path(cfg: DataConfig) -> Path | None:
    if cfg.cache_dir is None:
        return None
    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = {
        "root": str(Path(cfg.root).resolve()),
        "glob": cfg.glob,
        "max_assets": cfg.max_assets,
        "normalize": cfg.normalize,
        "exclude": tuple(sorted(cfg.exclude_symbols)),
    }
    key_str = json.dumps(key, sort_keys=True)
    cache_name = f"ohlc_{abs(hash(key_str)) & 0xFFFFFFFFFFFFFFFF:x}.pt"
    return cache_dir / cache_name


def load_aligned_ohlc(cfg: DataConfig) -> tuple[torch.Tensor, List[str], pd.DatetimeIndex]:
    """Load OHLC tensors aligned across symbols with sufficient overlap."""
    cache_path = _cache_path(cfg)
    if cache_path and cache_path.exists():
        payload = torch.load(cache_path)
        return payload["ohlc"], payload["symbols"], pd.DatetimeIndex(payload["index"])

    files = _discover_files(cfg)
    symbols_and_paths = _filter_symbols(files, cfg)
    assets: list[tuple[str, pd.DataFrame]] = []
    for symbol, path in symbols_and_paths:
        df = _load_csv(path)
        if len(df) >= cfg.min_timesteps:
            assets.append((symbol, df))
    if not assets:
        raise ValueError("No assets meet minimum timestep requirement")

    assets.sort(key=lambda item: len(item[1]), reverse=True)

    symbols: list[str] = []
    aligned_frames: list[pd.DataFrame] = []
    common_index: pd.Index | None = None
    for symbol, df in assets:
        candidate_index = df.index if common_index is None else common_index.intersection(df.index)
        if len(candidate_index) < cfg.min_timesteps:
            continue
        # Reindex existing frames to the candidate intersection
        if common_index is not None and candidate_index is not common_index:
            aligned_frames = [frame.reindex(candidate_index) for frame in aligned_frames]
        frame = df.reindex(candidate_index)
        aligned_frames.append(frame)
        symbols.append(symbol)
        common_index = candidate_index
        if cfg.max_assets is not None and len(symbols) >= cfg.max_assets:
            break

    if common_index is None or len(common_index) < cfg.min_timesteps:
        raise ValueError("Not enough overlapping timestamps across symbols")
    if not aligned_frames:
        raise ValueError("Failed to align any assets with sufficient overlap")

    aligned = []
    for frame in aligned_frames:
        filled = frame.interpolate(method="time").ffill().bfill()
        aligned.append(filled.to_numpy(dtype=np.float32))

    stacked = np.stack(aligned, axis=0).transpose(1, 0, 2)
    ohlc = torch.from_numpy(stacked)
    index = pd.DatetimeIndex(common_index)

    if cache_path:
        torch.save({"ohlc": ohlc, "symbols": symbols, "index": index.to_numpy()}, cache_path)

    return ohlc, symbols, index


def split_train_eval(ohlc: torch.Tensor, split_ratio: float = 0.8) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 < split_ratio < 1.0:
        raise ValueError("split_ratio must be between 0 and 1")
    total_steps = ohlc.shape[0]
    split_idx = int(total_steps * split_ratio)
    if split_idx < 2 or total_steps - split_idx < 2:
        raise ValueError("Not enough timesteps for the requested split ratio")
    return ohlc[:split_idx].clone(), ohlc[split_idx:].clone()


def log_data_preview(ohlc: torch.Tensor, symbols: Sequence[str], index: Sequence[pd.Timestamp]) -> dict:
    if isinstance(index, pd.DatetimeIndex):
        idx = index
    else:
        idx = pd.DatetimeIndex(index)

    trading_days = int(len(idx))
    if trading_days >= 1:
        first_ts = idx[0]
        last_ts = idx[-1]
        calendar_span_days = int((last_ts - first_ts).days)
        if calendar_span_days <= 0:
            approx_trading_days_per_year = float("nan")
        else:
            approx_trading_days_per_year = trading_days / (calendar_span_days / 365.25)
    else:
        first_ts = last_ts = pd.Timestamp("NaT")
        calendar_span_days = 0
        approx_trading_days_per_year = float("nan")

    diffs = idx.to_series().diff().dt.days.iloc[1:] if trading_days > 1 else pd.Series(dtype="float64")
    max_gap_days = int(diffs.max()) if not diffs.empty and diffs.notna().any() else 0
    gap_days_count = int((diffs > 1).sum()) if not diffs.empty else 0

    if trading_days > 0:
        normalized_idx = idx.normalize()
        expected_range = pd.date_range(
            first_ts.normalize(),
            last_ts.normalize(),
            freq="B",
            tz=idx.tz,
        )
        missing_business_days = int(len(expected_range.difference(normalized_idx)))
    else:
        missing_business_days = 0

    def _approx_periods_per_year(series: Sequence[pd.Timestamp]) -> float:
        if len(series) < 2:
            return float("nan")
        if isinstance(series, pd.DatetimeIndex):
            datetimes = series
        else:
            datetimes = pd.DatetimeIndex(series)
        values = datetimes.asi8.astype(np.float64)
        diffs_ns = np.diff(values)
        diffs_ns = diffs_ns[diffs_ns > 0]
        if diffs_ns.size == 0:
            return float("nan")
        avg_ns = float(diffs_ns.mean())
        if not math.isfinite(avg_ns) or avg_ns <= 0.0:
            return float("nan")
        seconds_per_period = avg_ns / 1e9
        if seconds_per_period <= 0.0:
            return float("nan")
        seconds_per_year = 365.25 * 24 * 3600
        return float(seconds_per_year / seconds_per_period)

    preview = {
        "timesteps": int(ohlc.shape[0]),
        "assets": int(ohlc.shape[1]),
        "features": int(ohlc.shape[2]),
        "first_timestamp": str(first_ts),
        "last_timestamp": str(last_ts),
        "symbols": list(symbols[:10]),
        "calendar_span_days": calendar_span_days,
        "trading_days": trading_days,
        "approx_trading_days_per_year": approx_trading_days_per_year,
        "missing_business_days": missing_business_days,
        "max_gap_days": max_gap_days,
        "multi_day_gaps": gap_days_count,
        "estimated_periods_per_year": _approx_periods_per_year(idx),
    }
    return preview
