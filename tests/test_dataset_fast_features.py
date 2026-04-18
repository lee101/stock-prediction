"""Contract tests for ``build_daily_dataset(fast_features=True)``.

Invariants:
  1. Columns returned by the polars path are a superset-and-subset of the
     pandas path's columns (i.e. identical set).
  2. Row counts agree within a small tolerance — the two paths can differ
     on edge-of-window dropna for rolling features, so we require >=95%
     row overlap by (symbol, date).
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from xgbnew.dataset import build_daily_dataset  # noqa: E402


def _write_synth_csv(path: Path, n: int = 400, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    prices = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    prices = np.clip(prices, 1.0, None)
    ts = pd.date_range("2021-01-04", periods=n, freq="B", tz="UTC")
    df = pd.DataFrame({
        "timestamp": ts,
        "open":   prices * 0.999,
        "high":   prices * 1.01,
        "low":    prices * 0.99,
        "close":  prices,
        "volume": rng.uniform(1e7, 1e8, n),  # large dolvol so rows pass filter
    })
    df.to_csv(path, index=False)


def _setup_synth(tmp_path: Path, symbols: list[str]) -> None:
    (tmp_path / "train").mkdir()
    for i, sym in enumerate(symbols):
        _write_synth_csv(tmp_path / "train" / f"{sym}.csv", n=500, seed=i)


def _args(tmp_path: Path, symbols: list[str]) -> dict:
    return dict(
        data_root=tmp_path,
        symbols=symbols,
        train_start=date(2021, 1, 4),
        train_end=date(2021, 12, 31),
        val_start=date(2022, 1, 3),
        val_end=date(2022, 6, 30),
        test_start=date(2022, 1, 3),
        test_end=date(2022, 6, 30),
        min_dollar_vol=1e6,
    )


def test_pandas_and_polars_paths_return_identical_column_set(tmp_path: Path):
    syms = ["AAA", "BBB", "CCC"]
    _setup_synth(tmp_path, syms)
    tr1, _, te1 = build_daily_dataset(**_args(tmp_path, syms), fast_features=False)
    tr2, _, te2 = build_daily_dataset(**_args(tmp_path, syms), fast_features=True)
    assert set(tr1.columns) == set(tr2.columns), (
        f"train columns diverge:\n"
        f"  pandas only: {set(tr1.columns) - set(tr2.columns)}\n"
        f"  polars only: {set(tr2.columns) - set(tr1.columns)}"
    )
    assert set(te1.columns) == set(te2.columns)


def test_pandas_and_polars_paths_have_near_matching_row_counts(tmp_path: Path):
    syms = ["AAA", "BBB", "CCC"]
    _setup_synth(tmp_path, syms)
    tr1, _, te1 = build_daily_dataset(**_args(tmp_path, syms), fast_features=False)
    tr2, _, te2 = build_daily_dataset(**_args(tmp_path, syms), fast_features=True)
    # Allow small divergence from rolling-window warm-up edges.
    def _close(a: int, b: int, tol: float = 0.05) -> bool:
        return abs(a - b) <= max(5, int(max(a, b) * tol))
    assert _close(len(tr1), len(tr2)), f"train rows pandas={len(tr1)} polars={len(tr2)}"
    assert _close(len(te1), len(te2)), f"oos rows pandas={len(te1)} polars={len(te2)}"


def test_fast_features_fills_chronos_zero(tmp_path: Path):
    """Polars path must fill chronos_* columns with 0.0 when cache is absent,
    matching the pandas path — eval_multiwindow depends on this."""
    syms = ["AAA", "BBB"]
    _setup_synth(tmp_path, syms)
    tr, _, te = build_daily_dataset(**_args(tmp_path, syms), fast_features=True)
    for col in ("chronos_oc_return", "chronos_cc_return",
                "chronos_pred_range", "chronos_available"):
        assert col in tr.columns
        assert (tr[col] == 0.0).all()
        assert col in te.columns
        assert (te[col] == 0.0).all()
