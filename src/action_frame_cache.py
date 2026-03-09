from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd


def _frame_hash(frame: pd.DataFrame, *, feature_columns: Sequence[str]) -> str:
    columns = ["timestamp", *[str(col) for col in feature_columns if str(col) in frame.columns]]
    subset = frame.loc[:, columns].copy()
    float_cols = subset.select_dtypes(include=["float16", "float32", "float64"]).columns
    if len(float_cols) > 0:
        subset.loc[:, float_cols] = subset.loc[:, float_cols].round(8)
    hashed = pd.util.hash_pandas_object(subset, index=False, categorize=False)
    return hashlib.sha256(hashed.to_numpy(dtype="uint64", copy=False).tobytes()).hexdigest()[:20]


def _normalizer_hash(normalizer: Any) -> str:
    try:
        payload = pickle.dumps(normalizer, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        payload = repr(normalizer).encode("utf-8", errors="replace")
    return hashlib.sha256(payload).hexdigest()[:20]


def build_action_cache_key(
    *,
    symbol: str,
    checkpoint_path: Path,
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    normalizer: Any,
    sequence_length: int,
    horizon: int,
) -> str:
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    stat = checkpoint.stat()
    payload = {
        "symbol": str(symbol).upper(),
        "checkpoint": str(checkpoint),
        "checkpoint_size": int(stat.st_size),
        "checkpoint_mtime_ns": int(stat.st_mtime_ns),
        "frame_hash": _frame_hash(frame, feature_columns=feature_columns),
        "frame_rows": int(len(frame)),
        "frame_first_ts": str(pd.to_datetime(frame["timestamp"], utc=True).min()) if "timestamp" in frame.columns else "",
        "frame_last_ts": str(pd.to_datetime(frame["timestamp"], utc=True).max()) if "timestamp" in frame.columns else "",
        "feature_columns": [str(col) for col in feature_columns],
        "normalizer_hash": _normalizer_hash(normalizer),
        "sequence_length": int(sequence_length),
        "horizon": int(horizon),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]


def action_cache_path(*, cache_root: Path, symbol: str, cache_key: str) -> Path:
    return Path(cache_root) / str(symbol).upper() / f"{cache_key}.parquet"


def load_action_frame(*, cache_root: Path, symbol: str, cache_key: str) -> pd.DataFrame | None:
    path = action_cache_path(cache_root=cache_root, symbol=symbol, cache_key=cache_key)
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    return frame.reset_index(drop=True)


def save_action_frame(*, cache_root: Path, symbol: str, cache_key: str, actions: pd.DataFrame) -> Path:
    path = action_cache_path(cache_root=cache_root, symbol=symbol, cache_key=cache_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = actions.copy()
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame.to_parquet(path, index=False)
    return path


def load_or_generate_action_frame(
    *,
    cache_root: Path,
    symbol: str,
    checkpoint_path: Path,
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    normalizer: Any,
    sequence_length: int,
    horizon: int,
    generator: Callable[[], pd.DataFrame],
) -> tuple[pd.DataFrame, bool]:
    cache_key = build_action_cache_key(
        symbol=symbol,
        checkpoint_path=checkpoint_path,
        frame=frame,
        feature_columns=feature_columns,
        normalizer=normalizer,
        sequence_length=sequence_length,
        horizon=horizon,
    )
    cached = load_action_frame(cache_root=cache_root, symbol=symbol, cache_key=cache_key)
    if cached is not None:
        return cached, True
    actions = generator()
    save_action_frame(cache_root=cache_root, symbol=symbol, cache_key=cache_key, actions=actions)
    return actions.reset_index(drop=True), False


__all__ = [
    "action_cache_path",
    "build_action_cache_key",
    "load_action_frame",
    "load_or_generate_action_frame",
    "save_action_frame",
]
