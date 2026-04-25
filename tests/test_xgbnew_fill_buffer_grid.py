"""Fill-buffer sensitivity grid tests for xgbnew.sweep_ensemble_grid.

Validates:
  * --fill-buffer-bps-grid plumbs into BacktestConfig.fill_buffer_bps.
  * Empty grid → single cell at fee-regime's historic default (5 / 15).
  * Increasing FB monotonically reduces median PnL (cost discipline).
  * -1.0 sentinel resolves to regime default (legacy callers unchanged).
  * JSON row carries fill_buffer_bps.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from xgbnew import sweep_ensemble_grid as sweep
from xgbnew.features import DAILY_FEATURE_COLS


def _mk_panel(n_days: int = 40, n_syms: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    start = date(2025, 1, 2)
    days = [start + timedelta(days=i) for i in range(n_days)]
    rows = []
    for d in days:
        for k in range(n_syms):
            day_ret = float(rng.uniform(-0.003, 0.012))
            o = 100.0
            c = o * (1.0 + day_ret)
            row = {
                "symbol": f"SYM{k}", "date": d,
                "open": o, "close": c,
                "actual_open": o, "actual_close": c,
                "actual_high": max(o, c) * 1.002,
                "actual_low":  min(o, c) * 0.998,
                "day_return_pct": day_ret * 100.0,
                "target_oc": day_ret,
                "spread_bps": 2.0, "dollar_volume": 1e9,
            }
            for fc in DAILY_FEATURE_COLS:
                row[fc] = 0.0
            row["dolvol_20d_log"] = np.log1p(1e9)
            row["vol_20d"] = 0.20
            rows.append(row)
    return pd.DataFrame(rows)


def _install_fakes(monkeypatch):
    """Stub build_daily_dataset + model.load so the sweep runs without training."""
    panel = _mk_panel()

    def fake_build(*args, **kwargs):
        # build_daily_dataset returns (train_df, val_df, oos_df). The sweep
        # uses only train (for the size gate) and oos (scored). Passing the
        # panel as both keeps _install_fakes short.
        return (panel.copy(), panel.iloc[0:0].copy(), panel.copy())

    # Score = open-relative to close so higher close-over-open gets picked.
    def fake_predict(self, df, chronos_cache=None):
        s = (df["actual_close"].values / df["actual_open"].values) - 1.0
        return pd.Series(s, index=df.index, name="xgb_score")

    monkeypatch.setattr(sweep, "build_daily_dataset", fake_build)
    monkeypatch.setattr("xgbnew.model.XGBStockModel.predict_scores",
                        fake_predict)

    def fake_load(path, **kw):
        m = MagicMock()
        m.predict_scores = fake_predict.__get__(m)
        m.feature_cols = DAILY_FEATURE_COLS
        return m
    monkeypatch.setattr(sweep, "load_any_model", fake_load)


def _fake_paths(tmp_path: Path, n: int) -> list[Path]:
    paths = []
    for i in range(n):
        p = tmp_path / f"m{i}.pkl"
        p.write_bytes(b"x")
        paths.append(p)
    return paths


def test_fill_buffer_sentinel_matches_regime_default(monkeypatch, tmp_path):
    """When grid is empty, sentinel -1.0 resolves to the fee regime's FB.

    deploy regime fb=5.0 → resulting cell reports fill_buffer_bps=5.0.
    """
    _install_fakes(monkeypatch)
    paths = _fake_paths(tmp_path, 2)
    cells = sweep.run_sweep(
        symbols=[f"SYM{k}" for k in range(6)],
        data_root=Path("/tmp"),
        model_paths=paths,
        train_start=date(2020, 1, 1), train_end=date(2024, 12, 31),
        oos_start=date(2025, 1, 2), oos_end=date(2025, 12, 31),
        window_days=10, stride_days=5,
        leverage_grid=[1.0], min_score_grid=[0.0],
        hold_through_grid=[False], top_n_grid=[1],
        fee_regimes=["deploy"],
        # No fill_buffer_bps_grid → single cell at regime default (5.0).
    )
    assert len(cells) == 1
    assert cells[0].fill_buffer_bps == pytest.approx(5.0, abs=1e-9)


def test_fill_buffer_grid_produces_one_cell_per_fb(monkeypatch, tmp_path):
    _install_fakes(monkeypatch)
    paths = _fake_paths(tmp_path, 2)
    fb_grid = [3.0, 5.0, 8.0, 15.0, 30.0]
    cells = sweep.run_sweep(
        symbols=[f"SYM{k}" for k in range(6)],
        data_root=Path("/tmp"),
        model_paths=paths,
        train_start=date(2020, 1, 1), train_end=date(2024, 12, 31),
        oos_start=date(2025, 1, 2), oos_end=date(2025, 12, 31),
        window_days=10, stride_days=5,
        leverage_grid=[1.0], min_score_grid=[0.0],
        hold_through_grid=[False], top_n_grid=[1],
        fee_regimes=["deploy"],
        fill_buffer_bps_grid=fb_grid,
    )
    assert len(cells) == len(fb_grid)
    got = sorted(c.fill_buffer_bps for c in cells)
    assert got == sorted(fb_grid)


def test_fill_buffer_monotonically_reduces_pnl(monkeypatch, tmp_path):
    """Higher FB = worse fills = lower median monthly PnL (cost discipline)."""
    _install_fakes(monkeypatch)
    paths = _fake_paths(tmp_path, 2)
    fb_grid = [3.0, 8.0, 30.0]
    cells = sweep.run_sweep(
        symbols=[f"SYM{k}" for k in range(6)],
        data_root=Path("/tmp"),
        model_paths=paths,
        train_start=date(2020, 1, 1), train_end=date(2024, 12, 31),
        oos_start=date(2025, 1, 2), oos_end=date(2025, 12, 31),
        window_days=10, stride_days=5,
        leverage_grid=[1.0], min_score_grid=[0.0],
        hold_through_grid=[False], top_n_grid=[1],
        fee_regimes=["deploy"],
        fill_buffer_bps_grid=fb_grid,
    )
    by_fb = {c.fill_buffer_bps: c for c in cells}
    assert by_fb[3.0].median_monthly_pct > by_fb[8.0].median_monthly_pct
    assert by_fb[8.0].median_monthly_pct > by_fb[30.0].median_monthly_pct


def test_fill_buffer_column_in_cells_to_rows(monkeypatch, tmp_path):
    _install_fakes(monkeypatch)
    paths = _fake_paths(tmp_path, 2)
    cells = sweep.run_sweep(
        symbols=[f"SYM{k}" for k in range(6)],
        data_root=Path("/tmp"),
        model_paths=paths,
        train_start=date(2020, 1, 1), train_end=date(2024, 12, 31),
        oos_start=date(2025, 1, 2), oos_end=date(2025, 12, 31),
        window_days=10, stride_days=5,
        leverage_grid=[1.0], min_score_grid=[0.0],
        hold_through_grid=[False], top_n_grid=[1],
        fee_regimes=["deploy"],
        fill_buffer_bps_grid=[3.0, 15.0],
    )
    rows = sweep._cells_to_rows(cells)
    assert all("fill_buffer_bps" in r for r in rows)
    assert sorted(r["fill_buffer_bps"] for r in rows) == [3.0, 15.0]


def test_fill_buffer_negative_sentinel_is_regime_default(monkeypatch, tmp_path):
    """Explicit -1.0 in grid = use regime default, same as empty grid."""
    _install_fakes(monkeypatch)
    paths = _fake_paths(tmp_path, 2)
    cells_none = sweep.run_sweep(
        symbols=[f"SYM{k}" for k in range(6)],
        data_root=Path("/tmp"),
        model_paths=paths,
        train_start=date(2020, 1, 1), train_end=date(2024, 12, 31),
        oos_start=date(2025, 1, 2), oos_end=date(2025, 12, 31),
        window_days=10, stride_days=5,
        leverage_grid=[1.0], min_score_grid=[0.0],
        hold_through_grid=[False], top_n_grid=[1],
        fee_regimes=["deploy"],
    )
    cells_sentinel = sweep.run_sweep(
        symbols=[f"SYM{k}" for k in range(6)],
        data_root=Path("/tmp"),
        model_paths=paths,
        train_start=date(2020, 1, 1), train_end=date(2024, 12, 31),
        oos_start=date(2025, 1, 2), oos_end=date(2025, 12, 31),
        window_days=10, stride_days=5,
        leverage_grid=[1.0], min_score_grid=[0.0],
        hold_through_grid=[False], top_n_grid=[1],
        fee_regimes=["deploy"],
        fill_buffer_bps_grid=[-1.0],
    )
    assert cells_none[0].fill_buffer_bps == pytest.approx(5.0)
    assert cells_sentinel[0].fill_buffer_bps == pytest.approx(5.0)
    assert cells_none[0].median_monthly_pct == pytest.approx(
        cells_sentinel[0].median_monthly_pct, abs=1e-9
    )
