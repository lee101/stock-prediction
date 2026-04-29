"""Build the dense (date × symbol × feature) panel tensor for the cross-attn
transformer.

Reuses ``xgbnew.dataset.build_daily_dataset`` so labels and features match the
XGB pipeline exactly. Caches the resulting numpy arrays to disk so we can
iterate quickly during model training.

Outputs (saved as a .npz):
  X:       (D, S, F)  feature matrix (float32, NaN-filled with 0)
  valid:   (D, S)     bool — True if symbol is present that day (passed
                            liquidity filter and has non-NaN core features)
  y:       (D, S)     float32 — target_oc_up (0/1), only meaningful where valid
  ret:     (D, S)     float32 — target_oc (continuous return), for evaluation
  dates:   (D,)       np.datetime64[D]
  symbols: (S,)       str
  feature_cols: (F,)  str
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from xgbnew.dataset import build_daily_dataset
from xgbnew.features import DAILY_FEATURE_COLS

logger = logging.getLogger(__name__)


# Subset of features used by the transformer — we drop spread_bps (an exact
# function of dolvol_20d_log via the lookup tier), keep the 15 listed in the
# task spec.
TRANSFORMER_FEATURE_COLS = list(DAILY_FEATURE_COLS)
assert len(TRANSFORMER_FEATURE_COLS) == 15


def _load_symbols(path: Path) -> list[str]:
    out: list[str] = []
    with open(path) as f:
        for ln in f:
            s = ln.strip()
            if s and not s.startswith("#"):
                out.append(s.upper())
    return out


def build_panel(
    data_root: Path,
    symbols_file: Path,
    train_start: date,
    train_end: date,
    val_start: date,
    val_end: date,
    test_start: date,
    test_end: date,
    cache_path: Path | None = None,
    fast_features: bool = True,
    min_dollar_vol: float = 1_000_000.0,
) -> dict:
    """Build the dense panel and return a dict; optionally cache to disk."""
    if cache_path is not None and Path(cache_path).exists():
        logger.info("Loading cached panel from %s", cache_path)
        loaded = np.load(cache_path, allow_pickle=True)
        return {k: loaded[k] for k in loaded.files}

    syms = _load_symbols(symbols_file)
    logger.info("build_panel: %d symbols requested", len(syms))

    tr, va, te = build_daily_dataset(
        Path(data_root), syms,
        train_start=train_start, train_end=train_end,
        val_start=val_start, val_end=val_end,
        test_start=test_start, test_end=test_end,
        fast_features=fast_features,
        min_dollar_vol=min_dollar_vol,
    )
    full = pd.concat([tr, va, te], ignore_index=True)
    logger.info("panel rows: %d (tr=%d va=%d te=%d)", len(full), len(tr), len(va), len(te))

    full["date"] = pd.to_datetime(full["date"]).dt.normalize().dt.date
    full = full.sort_values(["date", "symbol"]).reset_index(drop=True)

    dates = np.array(sorted(full["date"].unique()))
    syms_with_data = sorted(full["symbol"].unique())
    sym_to_idx = {s: i for i, s in enumerate(syms_with_data)}
    date_to_idx = {d: i for i, d in enumerate(dates)}

    D, S = len(dates), len(syms_with_data)
    F = len(TRANSFORMER_FEATURE_COLS)
    logger.info("panel shape D=%d S=%d F=%d", D, S, F)

    X = np.zeros((D, S, F), dtype=np.float32)
    valid = np.zeros((D, S), dtype=bool)
    y = np.zeros((D, S), dtype=np.float32)
    ret = np.zeros((D, S), dtype=np.float32)

    di = full["date"].map(date_to_idx).to_numpy()
    si = full["symbol"].map(sym_to_idx).to_numpy()
    feat_arr = full[TRANSFORMER_FEATURE_COLS].to_numpy(dtype=np.float32)
    # Replace NaNs in features with 0 — the model will see valid-mask via the
    # 'valid' tensor (only days with non-NaN core features are valid).
    feat_arr = np.where(np.isfinite(feat_arr), feat_arr, 0.0).astype(np.float32)
    X[di, si, :] = feat_arr
    valid[di, si] = True
    y[di, si] = full["target_oc_up"].to_numpy(dtype=np.float32)
    ret[di, si] = full["target_oc"].to_numpy(dtype=np.float32)

    out = dict(
        X=X, valid=valid, y=y, ret=ret,
        dates=np.array([np.datetime64(str(d)) for d in dates], dtype="datetime64[D]"),
        symbols=np.array(syms_with_data),
        feature_cols=np.array(TRANSFORMER_FEATURE_COLS),
        train_end=np.array(str(train_end)),
        val_end=np.array(str(val_end)),
        test_end=np.array(str(test_end)),
    )
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        logger.info("saving panel cache to %s", cache_path)
        np.savez_compressed(cache_path, **out)
    return out


# Per-feature robust normalization. The raw features include outliers from
# stock splits / data glitches (ret_20d with values 400+) that dominate std
# stats and squash the signal. We use winsorized mean/std (1st/99th
# percentile clip) computed on training rows only.
def fit_feature_stats(
    X: np.ndarray,
    valid: np.ndarray,
    train_end_idx: int,
    clip_q: float = 0.005,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute robust per-feature mean/std over training rows (date < train_end_idx).

    Winsorizes each feature at [clip_q, 1-clip_q] BEFORE computing mean/std so
    a handful of outlier rows (large stock splits, etc.) don't blow up the std.
    """
    train_X = X[:train_end_idx]            # (Dtr, S, F)
    train_v = valid[:train_end_idx]        # (Dtr, S)
    flat = train_X[train_v]                # (N, F)
    F = flat.shape[1]
    mean = np.zeros(F, dtype=np.float32)
    std  = np.zeros(F, dtype=np.float32)
    for f in range(F):
        col = flat[:, f]
        lo = np.quantile(col, clip_q)
        hi = np.quantile(col, 1.0 - clip_q)
        clipped = np.clip(col, lo, hi)
        mean[f] = clipped.mean()
        std[f]  = max(clipped.std(), 1e-6)
    return mean, std


def normalize_panel(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply z-score in-place-ish; broadcasts across (D, S) leading axes."""
    return ((X - mean) / std).astype(np.float32)


__all__ = [
    "build_panel",
    "fit_feature_stats",
    "normalize_panel",
    "TRANSFORMER_FEATURE_COLS",
]
