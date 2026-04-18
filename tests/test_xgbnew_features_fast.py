"""Parity + smoke tests for ``xgbnew.features_fast``.

The polars-native feature builder must produce the same feature values as
``xgbnew.features.build_features_for_symbol`` within a tight tolerance
(the two paths use slightly different clip / min_periods heuristics, so
we tolerate small absolute differences and focus on correlation).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from xgbnew.features import DAILY_FEATURE_COLS, build_features_for_symbol
from xgbnew.features_fast import build_daily_features_fast


def _write_synthetic_csv(path: Path, n: int = 300, seed: int = 0) -> None:
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
        "volume": rng.uniform(1e5, 1e7, n),
    })
    df.to_csv(path, index=False)


def test_features_fast_matches_reference_path_on_key_columns(tmp_path: Path):
    # Set up a fake data_root with two symbols.
    (tmp_path / "train").mkdir()
    for i, sym in enumerate(("AAA", "BBB")):
        _write_synthetic_csv(tmp_path / "train" / f"{sym}.csv", n=400, seed=i)

    # Polars path
    fast_df = build_daily_features_fast(tmp_path, ["AAA", "BBB"])

    # Reference pandas path
    ref_parts = []
    for sym in ("AAA", "BBB"):
        raw = pd.read_csv(tmp_path / "train" / f"{sym}.csv")
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
        ref_parts.append(build_features_for_symbol(raw, symbol=sym))
    ref_df = pd.concat(ref_parts, ignore_index=True)
    ref_df = ref_df.dropna(subset=DAILY_FEATURE_COLS[:5])

    # Align on (symbol, date)
    fast_df = fast_df.sort_values(["symbol", "date"]).reset_index(drop=True)
    ref_df = ref_df.sort_values(["symbol", "date"]).reset_index(drop=True)
    fast_df["date"] = pd.to_datetime(fast_df["date"]).dt.date
    ref_df["date"] = pd.to_datetime(ref_df["date"]).dt.date

    merged = ref_df.merge(fast_df, on=["symbol", "date"], suffixes=("_ref", "_fast"))
    assert len(merged) > 200, f"expected many aligned rows, got {len(merged)}"

    # Return features — should match very tightly (same math).
    for col in ("ret_1d", "ret_2d", "ret_5d", "ret_10d", "ret_20d"):
        ref = merged[f"{col}_ref"]
        fast = merged[f"{col}_fast"]
        ok = ref.notna() & fast.notna()
        if ok.sum() < 10:
            continue
        np.testing.assert_allclose(
            fast[ok].values, ref[ok].values, rtol=1e-6, atol=1e-9,
            err_msg=f"mismatch on {col}",
        )

    # RSI and vol use rolling windows — correlation-level parity.
    for col in ("rsi_14", "vol_5d", "vol_20d"):
        ref = merged[f"{col}_ref"]
        fast = merged[f"{col}_fast"]
        ok = ref.notna() & fast.notna()
        if ok.sum() < 10:
            continue
        corr = np.corrcoef(fast[ok].values, ref[ok].values)[0, 1]
        # 0.98 tolerates tiny EWM warm-up / min_samples divergence across
        # the two paths without accepting structurally wrong formulas.
        assert corr > 0.98, f"corr for {col} is too low: {corr:.3f}"


def test_features_fast_handles_missing_symbol_gracefully(tmp_path: Path):
    (tmp_path / "train").mkdir()
    _write_synthetic_csv(tmp_path / "train" / "AAA.csv", n=400)
    # Ask for two symbols but only one exists — should still return the
    # existing one rather than blow up.
    out = build_daily_features_fast(tmp_path, ["AAA", "MISSING"])
    assert "AAA" in set(out["symbol"].unique())
    assert "MISSING" not in set(out["symbol"].unique())


def test_features_fast_rejects_empty_symbol_universe(tmp_path: Path):
    (tmp_path / "train").mkdir()
    with pytest.raises(FileNotFoundError):
        build_daily_features_fast(tmp_path, ["X", "Y"])
