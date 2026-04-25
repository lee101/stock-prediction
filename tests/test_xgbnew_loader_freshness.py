"""Regression test: xgbnew.dataset._load_symbol_csv must prefer the fresh
ROOT csv over the (often stale) ``train/`` snapshot.

Context — project_xgb_stale_training_csvs.md (2026-04-20 audit):

- ``update_daily_data.py`` refreshes ``trainingdata/{SYM}.csv`` daily.
- ``trainingdata/train/{SYM}.csv`` is only rewritten by the full-rebuild
  pipeline, which was last run 2026-04-10 while root went through 04-15.
- The old loader preferred ``train/`` because it was listed first in the
  subdir fall-through list. That silently trained the weekly ensemble on
  6-11 days of stale data, and degraded OOS-sweep candidate counts from
  686 → 213 → 1 in the tariff-crash window.

Locked-in behaviour:

1. Root wins when both exist — even when the ``train/`` copy has *more*
   rows (the legacy copy often has extra 2019 history but stops in the
   present; our retrain starts in 2020 so the extra 2019 history is never
   used anyway).
2. ``train/`` is still found if root is missing, so back-compat with
   hand-curated ``train/``-only symbols is preserved.
3. The same preference applies in the polars fast-features path
   (``xgbnew.features_fast._read_ohlcv_polars``).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _make_rows(start: str, n: int, close0: float = 100.0) -> list[dict]:
    """Generate n consecutive daily bars starting at `start` (UTC)."""
    rows = []
    ts = pd.Timestamp(start, tz="UTC")
    for i in range(n):
        c = close0 + i
        rows.append({
            "timestamp": ts.isoformat(),
            "open": c - 0.5, "high": c + 0.5, "low": c - 1.0,
            "close": c, "volume": 1_000_000,
        })
        ts = ts + pd.Timedelta(days=1)
    return rows


def test_root_csv_wins_over_train(tmp_path: Path) -> None:
    """Both root and train/ have SYM.csv — we must pick root (fresher)."""
    from xgbnew.dataset import _load_symbol_csv

    sym = "FAKE"
    # train/ is an older snapshot ending earlier, with a DISTINCTIVE close
    _write_csv(tmp_path / "train" / f"{sym}.csv", _make_rows("2024-01-02", 100, close0=50.0))
    # root is fresher — distinct close so the test can tell which path was chosen
    _write_csv(tmp_path / f"{sym}.csv", _make_rows("2024-01-02", 110, close0=200.0))

    df = _load_symbol_csv(sym, tmp_path)
    assert df is not None
    # If root won, first close is ≈200; if train won, ≈50.
    assert abs(float(df["close"].iloc[0]) - 200.0) < 1e-6, (
        f"loader returned close={df['close'].iloc[0]} — expected root-side (~200)"
    )
    # Root had 110 rows, train had 100 — fresher path is longer here too.
    assert len(df) == 110


def test_train_used_when_root_missing(tmp_path: Path) -> None:
    """Back-compat: if root CSV is absent, train/ is still found."""
    from xgbnew.dataset import _load_symbol_csv

    sym = "ONLYINTRAIN"
    _write_csv(tmp_path / "train" / f"{sym}.csv", _make_rows("2024-01-02", 100, close0=50.0))

    df = _load_symbol_csv(sym, tmp_path)
    assert df is not None
    assert abs(float(df["close"].iloc[0]) - 50.0) < 1e-6


def test_stocks_subdir_still_found(tmp_path: Path) -> None:
    """Legacy ``stocks/`` subdir is still in the fall-through chain."""
    from xgbnew.dataset import _load_symbol_csv

    sym = "INSTOCKS"
    _write_csv(tmp_path / "stocks" / f"{sym}.csv", _make_rows("2024-01-02", 100, close0=77.0))

    df = _load_symbol_csv(sym, tmp_path)
    assert df is not None
    assert abs(float(df["close"].iloc[0]) - 77.0) < 1e-6


def test_missing_symbol_returns_none(tmp_path: Path) -> None:
    from xgbnew.dataset import _load_symbol_csv
    assert _load_symbol_csv("DOESNOTEXIST", tmp_path) is None


def test_polars_path_prefers_root(tmp_path: Path) -> None:
    """Fast-features path must walk the same preference list."""
    pytest.importorskip("polars")
    from xgbnew.features_fast import _read_ohlcv_polars

    sym = "FAKE"
    _write_csv(tmp_path / "train" / f"{sym}.csv", _make_rows("2024-01-02", 100, close0=50.0))
    _write_csv(tmp_path / f"{sym}.csv", _make_rows("2024-01-02", 110, close0=200.0))

    df = _read_ohlcv_polars(tmp_path, [sym])  # subdir=None → walk preference list
    # Polars df — pull first close via to_pandas for compatibility with test.
    pdf = df.to_pandas()
    assert abs(float(pdf["close"].iloc[0]) - 200.0) < 1e-6
    assert len(pdf) == 110


def test_polars_explicit_subdir_is_honoured(tmp_path: Path) -> None:
    """When a subdir is explicitly named, we do NOT fall through."""
    pytest.importorskip("polars")
    from xgbnew.features_fast import _read_ohlcv_polars

    sym = "FAKE"
    _write_csv(tmp_path / "train" / f"{sym}.csv", _make_rows("2024-01-02", 100, close0=50.0))
    _write_csv(tmp_path / f"{sym}.csv", _make_rows("2024-01-02", 110, close0=200.0))

    df = _read_ohlcv_polars(tmp_path, [sym], subdir="train")
    pdf = df.to_pandas()
    # With subdir='train' explicitly, we read the older snapshot (close ≈ 50).
    assert abs(float(pdf["close"].iloc[0]) - 50.0) < 1e-6
    assert len(pdf) == 100
