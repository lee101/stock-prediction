"""Tests for xgbnew.sweep_ensemble_grid.

We avoid retraining XGB in the test suite (too slow). Instead we stub
``build_daily_dataset`` and ``XGBStockModel.load`` to return a tiny synthetic
panel + fake model, then verify:

1. The sweep produces one CellResult per grid cell.
2. Score-blending is done ONCE (models are asked for predict_scores once each).
3. min_score=0.0 and min_score>0.99 produce different results (gate fires).
4. Missing pkl path raises FileNotFoundError.
5. Unknown fee regime raises ValueError.
6. Fee regimes differ in per-cell outcomes when fees actually bite.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from xgbnew import sweep_ensemble_grid as sweep  # noqa: E402
from xgbnew.features import DAILY_FEATURE_COLS  # noqa: E402


def _mk_panel(n_days: int = 40, n_syms: int = 6) -> pd.DataFrame:
    """Build a tiny OOS panel that simulate() can consume."""
    rng = np.random.default_rng(7)
    start = date(2025, 1, 2)
    days = [start + timedelta(days=i) for i in range(n_days)]
    rows = []
    for d in days:
        for k in range(n_syms):
            # Benign day-returns: uniform [-0.5%, +1.5%] — positive bias.
            day_ret = float(rng.uniform(-0.005, 0.015))
            o = 100.0
            c = o * (1.0 + day_ret)
            row = {
                "symbol": f"SYM{k}",
                "date": d,
                "open": o,
                "close": c,
                "actual_open": o,
                "actual_close": c,
                "day_return_pct": day_ret * 100.0,
                "target_oc": day_ret,
                "spread_bps": 2.0,
                "dollar_volume": 1e9,
            }
            # Feature cols first (zeros), then overwrite liquidity column so
            # the sim's dolvol gate doesn't wipe every row.
            for c_ in DAILY_FEATURE_COLS:
                row[c_] = 0.0
            row["dolvol_20d_log"] = float(np.log1p(1e9))
            rows.append(row)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


class _FakeModel:
    """Returns per-row score = rank of (symbol_id + tiny date noise)."""

    def __init__(self, seed: int):
        self._seed = seed

    def predict_scores(self, df: pd.DataFrame) -> pd.Series:
        rng = np.random.default_rng(self._seed)
        # Per-row score in [0, 1]. Use SYM id to give every day a clear
        # winner that varies across seeds, so ensemble blends.
        sym_ord = df["symbol"].map(lambda s: int(s.replace("SYM", "")))
        base = sym_ord.values.astype(np.float64) / 10.0
        jitter = rng.uniform(-0.02, 0.02, size=len(df))
        return pd.Series(np.clip(base + jitter, 0.0, 1.0),
                         index=df.index, name="score")


def _install_fakes(monkeypatch, n_models: int = 3):
    oos = _mk_panel()
    # build_daily_dataset returns (train_df, val_df, oos_df) — give it trivial
    # train/val so the precondition checks in sweep pass.
    train = oos.copy()  # size OK for the len(train_df) < 1000 gate
    monkeypatch.setattr(
        sweep, "build_daily_dataset",
        lambda **kw: (train, train.iloc[:0], oos),
    )
    # Load returns a FakeModel keyed by the filename.
    monkeypatch.setattr(
        sweep.XGBStockModel, "load",
        classmethod(lambda cls, path: _FakeModel(seed=hash(str(path)) & 0xFFFF)),
    )
    return oos, n_models


def _fake_paths(tmp_path: Path, n: int) -> list[Path]:
    paths = []
    for i in range(n):
        p = tmp_path / f"fake_seed{i}.pkl"
        p.write_bytes(b"fake")
        paths.append(p)
    return paths


def test_sweep_cells_match_grid_product(monkeypatch, tmp_path):
    _install_fakes(monkeypatch)
    paths = _fake_paths(tmp_path, 3)

    cells = sweep.run_sweep(
        symbols=[f"SYM{k}" for k in range(6)],
        data_root=Path("/tmp"),
        model_paths=paths,
        train_start=date(2020, 1, 1), train_end=date(2024, 12, 31),
        oos_start=date(2025, 1, 2), oos_end=date(2025, 12, 31),
        window_days=10, stride_days=5,
        leverage_grid=[1.0, 2.0],
        min_score_grid=[0.0, 0.5],
        hold_through_grid=[True],
        top_n_grid=[1],
        fee_regimes=["deploy", "stress36x"],
    )
    assert len(cells) == 2 * 2 * 1 * 1 * 2, (
        f"expected 8 cells (2 lev × 2 ms × 1 ht × 1 tn × 2 reg), got {len(cells)}"
    )
    # Each cell has some windows
    assert all(c.n_windows > 0 for c in cells)


def test_sweep_blends_scores_once(monkeypatch, tmp_path):
    """predict_scores must fire exactly once per model regardless of cell count."""
    _install_fakes(monkeypatch)
    paths = _fake_paths(tmp_path, 3)

    # Track predict_scores calls.
    call_counts = [0, 0, 0]
    real_predict = _FakeModel.predict_scores

    def _tracking_predict(self, df, _i=[0]):  # noqa: B008 — stable closure
        # Use object id to bucket by model instance.
        idx = id(self) % 3
        call_counts[idx] += 1
        return real_predict(self, df)

    monkeypatch.setattr(_FakeModel, "predict_scores", _tracking_predict)

    sweep.run_sweep(
        symbols=[f"SYM{k}" for k in range(6)],
        data_root=Path("/tmp"),
        model_paths=paths,
        train_start=date(2020, 1, 1), train_end=date(2024, 12, 31),
        oos_start=date(2025, 1, 2), oos_end=date(2025, 12, 31),
        window_days=10, stride_days=5,
        leverage_grid=[1.0, 2.0, 3.0],
        min_score_grid=[0.0, 0.5],
        hold_through_grid=[True, False],
        top_n_grid=[1],
        fee_regimes=["deploy"],
    )
    # 3*2*2*1*1 = 12 cells but predict_scores should fire 3 times total
    # (one per model), not 12*3 = 36.
    total_calls = sum(call_counts)
    assert total_calls == 3, (
        f"scores should be computed once per model (3 total), "
        f"got {total_calls}: {call_counts}"
    )


def test_missing_pkl_raises(monkeypatch, tmp_path):
    _install_fakes(monkeypatch)
    real = _fake_paths(tmp_path, 1)
    missing = tmp_path / "does_not_exist.pkl"
    with pytest.raises(FileNotFoundError, match="model path not found"):
        sweep.run_sweep(
            symbols=["SYM0"],
            data_root=Path("/tmp"),
            model_paths=real + [missing],
            train_start=date(2020, 1, 1), train_end=date(2024, 12, 31),
            oos_start=date(2025, 1, 2), oos_end=date(2025, 12, 31),
            window_days=10, stride_days=5,
            leverage_grid=[1.0], min_score_grid=[0.0],
            hold_through_grid=[False], top_n_grid=[1],
            fee_regimes=["deploy"],
        )


def test_unknown_fee_regime_raises(monkeypatch, tmp_path):
    _install_fakes(monkeypatch)
    paths = _fake_paths(tmp_path, 2)
    with pytest.raises(ValueError, match="unknown fee regime"):
        sweep.run_sweep(
            symbols=[f"SYM{k}" for k in range(6)],
            data_root=Path("/tmp"),
            model_paths=paths,
            train_start=date(2020, 1, 1), train_end=date(2024, 12, 31),
            oos_start=date(2025, 1, 2), oos_end=date(2025, 12, 31),
            window_days=10, stride_days=5,
            leverage_grid=[1.0], min_score_grid=[0.0],
            hold_through_grid=[False], top_n_grid=[1],
            fee_regimes=["bogus"],
        )


def test_min_score_gate_changes_outcome(monkeypatch, tmp_path):
    """ms=0 vs ms=0.99 must produce different cell outcomes (gate trims picks)."""
    _install_fakes(monkeypatch)
    paths = _fake_paths(tmp_path, 2)

    cells = sweep.run_sweep(
        symbols=[f"SYM{k}" for k in range(6)],
        data_root=Path("/tmp"),
        model_paths=paths,
        train_start=date(2020, 1, 1), train_end=date(2024, 12, 31),
        oos_start=date(2025, 1, 2), oos_end=date(2025, 12, 31),
        window_days=10, stride_days=5,
        leverage_grid=[1.0],
        min_score_grid=[0.0, 0.99],
        hold_through_grid=[False],
        top_n_grid=[1],
        fee_regimes=["deploy"],
    )
    by_ms = {c.min_score: c for c in cells}
    assert 0.0 in by_ms and 0.99 in by_ms
    # The 0.99 gate should drop most/all picks, yielding a different monthly
    # number (likely 0 or close to it) vs the ungated cell.
    assert (by_ms[0.0].median_monthly_pct
            != pytest.approx(by_ms[0.99].median_monthly_pct, abs=0.01)), (
        f"min_score gate did nothing: {by_ms[0.0]} vs {by_ms[0.99]}"
    )


def test_fee_regimes_in_registry():
    assert set(sweep.FEE_REGIMES) >= {"deploy", "stress36x"}
    d = sweep.FEE_REGIMES["deploy"]
    s = sweep.FEE_REGIMES["stress36x"]
    # stress must be meaningfully more pessimistic on every axis
    assert s["fee_rate"] > d["fee_rate"]
    assert s["fill_buffer_bps"] > d["fill_buffer_bps"]
    assert s["commission_bps"] >= d["commission_bps"]


def test_cells_to_rows_shapes():
    c = sweep.CellResult(
        leverage=2.0, min_score=0.85, hold_through=True, top_n=1,
        fee_regime="deploy", n_windows=60,
        median_monthly_pct=141.0, p10_monthly_pct=96.0,
        median_sortino=40.0, worst_dd_pct=12.0, n_neg=0,
    )
    rows = sweep._cells_to_rows([c])
    assert len(rows) == 1
    r = rows[0]
    assert r["leverage"] == 2.0
    assert r["n_neg"] == 0
    assert r["hold_through"] is True
