from __future__ import annotations

import json
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
    """
    Load OHLC tensors aligned across symbols.

    Returns:
        ohlc: tensor shaped [T, A, 4]
        symbols: list of symbol names ordered as in tensor
        index: pandas DatetimeIndex shared across symbols
    """
    cache_path = _cache_path(cfg)
    if cache_path and cache_path.exists():
        payload = torch.load(cache_path)
        return payload["ohlc"], payload["symbols"], pd.DatetimeIndex(payload["index"])

    files = _discover_files(cfg)
    symbols_and_paths = _filter_symbols(files, cfg)
    frames: list[pd.DataFrame] = []
    symbols: list[str] = []
    common_index: pd.Index | None = None
    for symbol, path in symbols_and_paths:
        df = _load_csv(path)
        if common_index is None:
            common_index = df.index
        else:
            common_index = common_index.intersection(df.index)
        frames.append(df)
        symbols.append(symbol)
    assert common_index is not None
    if len(common_index) < 10:
        raise ValueError("Not enough overlapping timestamps across symbols")
    aligned = []
    for df in frames:
        aligned_df = df.reindex(common_index)
        if aligned_df.isna().any().any():
            aligned_df = aligned_df.interpolate(method="time").ffill().bfill()
        aligned.append(aligned_df.to_numpy(dtype=np.float32))
    # shape [A, T, 4] -> transpose to [T, A, 4]
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
    preview = {
        "timesteps": int(ohlc.shape[0]),
        "assets": int(ohlc.shape[1]),
        "features": int(ohlc.shape[2]),
        "first_timestamp": str(index[0]),
        "last_timestamp": str(index[-1]),
        "symbols": list(symbols[:10]),
    }
    return preview
