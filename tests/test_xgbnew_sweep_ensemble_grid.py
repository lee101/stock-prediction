"""Tests for xgbnew.sweep_ensemble_grid.

We avoid retraining XGB in the test suite (too slow). Instead we stub
``build_daily_dataset`` and ``load_any_model`` to return a tiny synthetic
panel + fake model, then verify:

1. The sweep produces one CellResult per grid cell.
2. Score-blending is done ONCE (models are asked for predict_scores once each).
3. min_score=0.0 and min_score>0.99 produce different results (gate fires).
4. Missing pkl path raises FileNotFoundError.
5. Unknown fee regime raises ValueError.
6. Fee regimes differ in per-cell outcomes when fees actually bite.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from xgbnew import sweep_ensemble_grid as sweep
from xgbnew.features import DAILY_FEATURE_COLS


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

    feature_cols = list(DAILY_FEATURE_COLS)

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
        sweep, "load_any_model",
        lambda path: _FakeModel(seed=hash(str(path)) & 0xFFFF),
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


def test_run_sweep_progress_callback_receives_partial_cells(monkeypatch, tmp_path):
    _install_fakes(monkeypatch)
    paths = _fake_paths(tmp_path, 2)
    events: list[tuple[int, int, int]] = []

    def _progress(cells_so_far, done, total):
        events.append((len(cells_so_far), done, total))

    cells = sweep.run_sweep(
        symbols=[f"SYM{k}" for k in range(6)],
        data_root=Path("/tmp"),
        model_paths=paths,
        train_start=date(2020, 1, 1), train_end=date(2024, 12, 31),
        oos_start=date(2025, 1, 2), oos_end=date(2025, 12, 31),
        window_days=10, stride_days=5,
        leverage_grid=[1.0, 2.0],
        min_score_grid=[0.0],
        hold_through_grid=[True],
        top_n_grid=[1],
        fee_regimes=["deploy", "stress36x"],
        progress_callback=_progress,
    )

    assert len(cells) == 4
    assert events == [(1, 1, 4), (2, 2, 4), (3, 3, 4), (4, 4, 4)]


def test_run_sweep_includes_opportunistic_grid_in_cell_identity(monkeypatch, tmp_path):
    _install_fakes(monkeypatch)
    paths = _fake_paths(tmp_path, 2)

    calls: list[tuple[int, float]] = []

    def _fake_run_cell(**kwargs):
        calls.append((
            int(kwargs["opportunistic_watch_n"]),
            float(kwargs["opportunistic_entry_discount_bps"]),
        ))
        return sweep.CellResult(
            leverage=float(kwargs["leverage"]),
            min_score=float(kwargs["min_score"]),
            hold_through=bool(kwargs["hold_through"]),
            top_n=int(kwargs["top_n"]),
            min_picks=int(kwargs["min_picks"]),
            opportunistic_watch_n=int(kwargs["opportunistic_watch_n"]),
            opportunistic_entry_discount_bps=float(kwargs["opportunistic_entry_discount_bps"]),
            fee_regime=str(kwargs["fee_regime"]),
            n_windows=5,
            median_monthly_pct=1.0,
            p10_monthly_pct=1.0,
            median_sortino=1.0,
            worst_dd_pct=1.0,
            n_neg=0,
        )

    monkeypatch.setattr(sweep, "_run_cell", _fake_run_cell)
    cells = sweep.run_sweep(
        symbols=[f"SYM{k}" for k in range(6)],
        data_root=Path("/tmp"),
        model_paths=paths,
        train_start=date(2020, 1, 1), train_end=date(2024, 12, 31),
        oos_start=date(2025, 1, 2), oos_end=date(2025, 12, 31),
        window_days=10, stride_days=5,
        leverage_grid=[1.0],
        min_score_grid=[0.0],
        hold_through_grid=[False],
        top_n_grid=[1],
        fee_regimes=["deploy"],
        opportunistic_watch_n_grid=[0, 5],
        opportunistic_entry_discount_bps_grid=[0.0, 30.0],
    )

    assert calls == [(0, 0.0), (0, 30.0), (5, 0.0), (5, 30.0)]
    rows = sweep._cells_to_rows(cells)
    assert rows[-1]["opportunistic_watch_n"] == 5
    assert rows[-1]["opportunistic_entry_discount_bps"] == 30.0


def test_run_sweep_includes_max_spread_grid_in_cell_identity(monkeypatch, tmp_path):
    _install_fakes(monkeypatch)
    paths = _fake_paths(tmp_path, 2)

    calls: list[float] = []

    def _fake_run_cell(**kwargs):
        calls.append(float(kwargs["inference_max_spread_bps"]))
        return sweep.CellResult(
            leverage=float(kwargs["leverage"]),
            min_score=float(kwargs["min_score"]),
            hold_through=bool(kwargs["hold_through"]),
            top_n=int(kwargs["top_n"]),
            fee_regime=str(kwargs["fee_regime"]),
            n_windows=5,
            median_monthly_pct=1.0,
            p10_monthly_pct=1.0,
            median_sortino=1.0,
            worst_dd_pct=1.0,
            n_neg=0,
            inference_max_spread_bps=float(kwargs["inference_max_spread_bps"]),
        )

    monkeypatch.setattr(sweep, "_run_cell", _fake_run_cell)
    cells = sweep.run_sweep(
        symbols=[f"SYM{k}" for k in range(6)],
        data_root=Path("/tmp"),
        model_paths=paths,
        train_start=date(2020, 1, 1), train_end=date(2024, 12, 31),
        oos_start=date(2025, 1, 2), oos_end=date(2025, 12, 31),
        window_days=10, stride_days=5,
        leverage_grid=[1.0],
        min_score_grid=[0.0],
        hold_through_grid=[False],
        top_n_grid=[1],
        fee_regimes=["deploy"],
        inference_max_spread_bps_grid=[12.0, 30.0],
    )

    assert calls == [12.0, 30.0]
    rows = sweep._cells_to_rows(cells)
    assert [row["inference_max_spread_bps"] for row in rows] == [12.0, 30.0]


def test_run_sweep_resume_rows_skips_completed_cells(monkeypatch, tmp_path):
    _install_fakes(monkeypatch)
    paths = _fake_paths(tmp_path, 2)
    resumed = sweep.CellResult(
        leverage=1.0,
        min_score=0.0,
        hold_through=True,
        top_n=1,
        fee_regime="deploy",
        n_windows=7,
        median_monthly_pct=123.0,
        p10_monthly_pct=100.0,
        median_sortino=9.0,
        worst_dd_pct=4.0,
        n_neg=0,
        goodness_score=96.0,
        robust_goodness_score=94.0,
        pain_adjusted_goodness_score=93.0,
        fill_buffer_bps=sweep.FEE_REGIMES["deploy"]["fill_buffer_bps"],
    )
    calls: list[float] = []
    events: list[tuple[int, int, int]] = []

    def _fake_run_cell(**kwargs):
        calls.append(float(kwargs["leverage"]))
        return sweep.CellResult(
            leverage=float(kwargs["leverage"]),
            min_score=float(kwargs["min_score"]),
            hold_through=bool(kwargs["hold_through"]),
            top_n=int(kwargs["top_n"]),
            fee_regime=str(kwargs["fee_regime"]),
            n_windows=5,
            median_monthly_pct=456.0,
            p10_monthly_pct=400.0,
            median_sortino=8.0,
            worst_dd_pct=6.0,
            n_neg=0,
            goodness_score=394.0,
            fill_buffer_bps=sweep.FEE_REGIMES[str(kwargs["fee_regime"])]["fill_buffer_bps"],
        )

    def _progress(cells_so_far, done, total):
        events.append((len(cells_so_far), done, total))

    monkeypatch.setattr(sweep, "_run_cell", _fake_run_cell)
    resume_rows = sweep._cells_to_rows([resumed])
    # Older checkpoints did not always include fill_buffer_bps; resume should
    # still match those rows against the fee-regime default.
    resume_rows[0].pop("fill_buffer_bps")

    cells = sweep.run_sweep(
        symbols=[f"SYM{k}" for k in range(6)],
        data_root=Path("/tmp"),
        model_paths=paths,
        train_start=date(2020, 1, 1), train_end=date(2024, 12, 31),
        oos_start=date(2025, 1, 2), oos_end=date(2025, 12, 31),
        window_days=10, stride_days=5,
        leverage_grid=[1.0, 2.0],
        min_score_grid=[0.0],
        hold_through_grid=[True],
        top_n_grid=[1],
        fee_regimes=["deploy"],
        progress_callback=_progress,
        resume_rows=resume_rows,
    )

    assert calls == [2.0]
    assert [c.leverage for c in cells] == [1.0, 2.0]
    assert cells[0].median_monthly_pct == 123.0
    assert cells[0].fill_buffer_bps == sweep.FEE_REGIMES["deploy"]["fill_buffer_bps"]
    assert cells[1].median_monthly_pct == 456.0
    assert events == [(1, 1, 2), (2, 2, 2)]


def test_load_checkpoint_rows_validates_shape(tmp_path):
    good = tmp_path / "checkpoint.json"
    good.write_text(json.dumps({
        "model_paths": ["model_a.pkl"],
        "oos_start": "2025-01-02",
        "oos_end": "2025-04-01",
        "window_days": 30,
        "stride_days": 7,
        "cells": [{"leverage": 1.0}],
    }))
    assert sweep._load_checkpoint_rows(good) == [{"leverage": 1.0}]
    assert sweep._load_checkpoint_rows(
        good,
        expected_model_paths=[Path("model_a.pkl")],
        expected_oos_start="2025-01-02",
        expected_oos_end=date(2025, 4, 1),
        expected_window_days=30,
        expected_stride_days=7,
    ) == [{"leverage": 1.0}]
    with pytest.raises(ValueError, match="model_paths"):
        sweep._load_checkpoint_rows(good, expected_model_paths=[Path("other.pkl")])

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"cells": {"leverage": 1.0}}))
    with pytest.raises(ValueError, match="cells list"):
        sweep._load_checkpoint_rows(bad)


def test_sweep_blends_scores_once(monkeypatch, tmp_path):
    """predict_scores must fire exactly once per model regardless of cell count."""
    _install_fakes(monkeypatch)
    paths = _fake_paths(tmp_path, 3)

    # Track predict_scores calls.
    call_counts = [0, 0, 0]
    real_predict = _FakeModel.predict_scores

    def _tracking_predict(self, df):
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


def test_run_sweep_rejects_duplicate_model_paths(tmp_path):
    path = tmp_path / "fake_seed0.pkl"
    path.write_bytes(b"fake")

    with pytest.raises(ValueError, match="model path list contains duplicates"):
        sweep.run_sweep(
            symbols=["SYM0"],
            data_root=Path("/tmp"),
            model_paths=[path, path],
            train_start=date(2020, 1, 1), train_end=date(2024, 12, 31),
            oos_start=date(2025, 1, 2), oos_end=date(2025, 12, 31),
            window_days=10, stride_days=5,
            leverage_grid=[1.0], min_score_grid=[0.0],
            hold_through_grid=[False], top_n_grid=[1],
            fee_regimes=["deploy"],
        )


def test_validate_model_paths_allows_same_seed_in_distinct_paths(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    path0_left = left / "alltrain_seed0.pkl"
    path0_right = right / "alltrain_seed0.pkl"
    path0_left.write_bytes(b"fake")
    path0_right.write_bytes(b"fake")

    sweep._validate_model_paths_for_sweep([path0_left, path0_right])


def test_run_sweep_rejects_malformed_alltrain_seed_path(tmp_path):
    path = tmp_path / "alltrain_seedx.pkl"
    path.write_bytes(b"fake")

    with pytest.raises(ValueError, match="filename must match alltrain_seed<seed>.pkl"):
        sweep.run_sweep(
            symbols=["SYM0"],
            data_root=Path("/tmp"),
            model_paths=[path],
            train_start=date(2020, 1, 1), train_end=date(2024, 12, 31),
            oos_start=date(2025, 1, 2), oos_end=date(2025, 12, 31),
            window_days=10, stride_days=5,
            leverage_grid=[1.0], min_score_grid=[0.0],
            hold_through_grid=[False], top_n_grid=[1],
            fee_regimes=["deploy"],
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"window_days": 0}, "window_days must be >= 1"),
        ({"stride_days": 0}, "stride_days must be >= 1"),
        ({"leverage_grid": [float("nan")]}, "leverage_grid values must be finite"),
        ({"leverage_grid": [0.0]}, "leverage_grid values must be > 0"),
        ({"min_score_grid": [1.5]}, "min_score_grid values must be between 0 and 1"),
        ({"top_n_grid": [0]}, "top_n_grid values must be >= 1"),
        ({"min_picks_grid": [-1]}, "min_picks_grid values must be >= 0"),
        (
            {"top_n_grid": [2], "min_picks_grid": [3]},
            "min_picks_grid values must be <= top_n_grid values",
        ),
        ({"skip_prob_grid": [1.5]}, "skip_prob_grid values must be between 0 and 1"),
        ({"fill_buffer_bps_grid": [-0.5]}, "fill_buffer_bps_grid values must be -1 or >= 0"),
        ({"inference_max_spread_bps_grid": [-1.0]}, "inference_max_spread_bps_grid values"),
        ({"allocation_temp_grid": [0.0]}, "allocation_temp_grid values must be > 0"),
        ({"min_secondary_allocation_grid": [1.2]}, "min_secondary_allocation_grid values"),
        ({"max_ret_20d_rank_pct_grid": [1.5]}, "max_ret_20d_rank_pct_grid values"),
        ({"inv_vol_floor": float("nan")}, "inv_vol_floor must be finite and > 0"),
        ({"inv_vol_cap": 0.5}, "inv_vol_cap must be finite and >= 1"),
        ({"conviction_alloc_low": 0.7, "conviction_alloc_high": 0.7}, "conviction_alloc_high"),
        ({"fail_fast_max_dd_pct": float("inf")}, "fail_fast_max_dd_pct must be finite"),
    ],
)
def test_run_sweep_rejects_invalid_grid_domains_before_model_load(
    tmp_path,
    overrides,
    message,
):
    path = tmp_path / "alltrain_seed0.pkl"
    path.write_bytes(b"not-a-real-model")
    kwargs = {
        "symbols": ["SYM0"],
        "data_root": Path("/tmp"),
        "model_paths": [path],
        "train_start": date(2020, 1, 1),
        "train_end": date(2024, 12, 31),
        "oos_start": date(2025, 1, 2),
        "oos_end": date(2025, 12, 31),
        "window_days": 10,
        "stride_days": 5,
        "leverage_grid": [1.0],
        "min_score_grid": [0.0],
        "hold_through_grid": [False],
        "top_n_grid": [1],
        "fee_regimes": ["deploy"],
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=message):
        sweep.run_sweep(**kwargs)


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
    assert set(sweep.FEE_REGIMES) >= {"deploy", "prod10bps", "stress36x"}
    d = sweep.FEE_REGIMES["deploy"]
    p = sweep.FEE_REGIMES["prod10bps"]
    s = sweep.FEE_REGIMES["stress36x"]
    assert p["fee_rate"] == pytest.approx(0.001)
    assert p["fill_buffer_bps"] == d["fill_buffer_bps"]
    assert p["commission_bps"] == d["commission_bps"]
    assert p["fee_rate"] > d["fee_rate"]
    # stress must be meaningfully more pessimistic on every axis
    assert s["fee_rate"] >= p["fee_rate"]
    assert s["fill_buffer_bps"] > p["fill_buffer_bps"]
    assert s["commission_bps"] > p["commission_bps"]


def test_cells_to_rows_shapes():
    c = sweep.CellResult(
        leverage=2.0, min_score=0.85, hold_through=True, top_n=1,
        fee_regime="deploy", n_windows=60,
        median_monthly_pct=141.0, p10_monthly_pct=96.0,
        median_sortino=40.0, worst_dd_pct=12.0, n_neg=0,
        score_uncertainty_penalty=0.75,
        goodness_score=84.0,
        no_picks_fallback_symbol="SPY",
        no_picks_fallback_alloc_scale=0.5,
        conviction_scaled_alloc=True,
        conviction_alloc_low=0.40,
        conviction_alloc_high=0.90,
        allocation_mode="softmax",
        allocation_temp=0.25,
        ensemble_needs_ranks=True,
        ensemble_needs_dispersion=True,
    )
    rows = sweep._cells_to_rows([c])
    assert len(rows) == 1
    r = rows[0]
    assert r["leverage"] == 2.0
    assert r["n_neg"] == 0
    assert r["hold_through"] is True
    assert r["min_picks"] == 0
    assert r["score_uncertainty_penalty"] == 0.75
    assert r["goodness_score"] == 84.0
    assert r["inference_max_spread_bps"] == 30.0
    assert r["no_picks_fallback_symbol"] == "SPY"
    assert r["no_picks_fallback_alloc_scale"] == 0.5
    assert r["conviction_scaled_alloc"] is True
    assert r["conviction_alloc_low"] == 0.40
    assert r["conviction_alloc_high"] == 0.90
    assert r["allocation_mode"] == "softmax"
    assert r["allocation_temp"] == 0.25
    assert r["fail_fast_triggered"] is False
    assert r["fail_fast_reason"] == ""
    assert r["ensemble_needs_ranks"] is True
    assert r["ensemble_needs_dispersion"] is True


def test_uncertainty_adjusted_scores_penalize_seed_disagreement():
    scores = pd.Series([0.70, 0.69], index=["A", "B"], name="ensemble_score")
    score_std = pd.Series([0.29, 0.0], index=["A", "B"])

    adjusted = sweep._uncertainty_adjusted_scores(scores, score_std, penalty=1.0)

    assert adjusted.loc["A"] == pytest.approx(0.41)
    assert adjusted.loc["B"] == pytest.approx(0.69)
    assert adjusted.sort_values(ascending=False).index.tolist() == ["B", "A"]


def test_run_sweep_rejects_negative_uncertainty_penalty(monkeypatch, tmp_path):
    _install_fakes(monkeypatch)
    paths = _fake_paths(tmp_path, 2)

    with pytest.raises(ValueError, match="score_uncertainty_penalty_grid values"):
        sweep.run_sweep(
            symbols=[f"SYM{k}" for k in range(6)],
            data_root=Path("/tmp"),
            model_paths=paths,
            train_start=date(2020, 1, 1), train_end=date(2024, 12, 31),
            oos_start=date(2025, 1, 2), oos_end=date(2025, 12, 31),
            window_days=10, stride_days=5,
            leverage_grid=[1.0],
            min_score_grid=[0.0],
            hold_through_grid=[True],
            top_n_grid=[1],
            fee_regimes=["deploy"],
            score_uncertainty_penalty_grid=[-0.1],
        )


def test_sweep_json_payload_and_atomic_write(tmp_path):
    rows = [{
        "leverage": 2.0,
        "min_score": 0.85,
        "hold_through": True,
        "top_n": 1,
        "fee_regime": "deploy",
        "n_windows": 4,
        "median_monthly_pct": 30.0,
        "p10_monthly_pct": 20.0,
        "median_sortino": 2.0,
        "worst_dd_pct": 10.0,
        "n_neg": 0,
        "goodness_score": 10.0,
        "robust_goodness_score": 9.0,
        "pain_adjusted_goodness_score": 8.0,
        "fill_buffer_bps": 5.0,
        "ensemble_needs_ranks": True,
        "ensemble_needs_dispersion": True,
    }]
    payload = sweep._sweep_json_payload(
        symbols_file=Path("symbol_lists/test_universe.txt"),
        data_root=Path("trainingdata_test"),
        spy_csv_path=Path("trainingdata_test/SPY.csv"),
        blend_mode="mean",
        model_paths=[Path("model_a.pkl")],
        oos_start="2025-01-02",
        oos_end=date(2025, 4, 1),
        window_days=30,
        stride_days=7,
        fee_regimes=["deploy"],
        fail_fast_max_dd_pct=40.0,
        fail_fast_max_intraday_dd_pct=35.0,
        fail_fast_neg_windows=2,
        rows=rows,
        complete=False,
    )
    out = tmp_path / "nested" / "partial.json"
    sweep._write_json_atomic(out, payload)

    loaded = json.loads(out.read_text())
    assert loaded["complete"] is False
    assert loaded["symbols_file"] == "symbol_lists/test_universe.txt"
    assert loaded["data_root"] == "trainingdata_test"
    assert loaded["spy_csv"] == "trainingdata_test/SPY.csv"
    assert loaded["blend_mode"] == "mean"
    assert loaded["n_cells"] == 1
    assert loaded["cells"] == rows
    assert loaded["fail_fast"]["max_dd_pct"] == 40.0
    assert loaded["fail_fast"]["max_intraday_dd_pct"] == 35.0
    assert loaded["fail_fast"]["neg_windows"] == 2
    assert loaded["fee_regimes"]["deploy"] == sweep.FEE_REGIMES["deploy"]
    assert loaded["production_target"]["median_monthly_pct"] == 27.0
    assert loaded["production_target"]["max_dd_pct"] == 25.0
    assert loaded["production_target"]["max_neg_windows"] == 0
    assert loaded["production_target"]["min_windows"] == 1
    assert loaded["production_target"]["expected_windows_required"] is True
    assert loaded["ensemble_feature_mode"] == {
        "needs_ranks": True,
        "needs_dispersion": True,
    }
    assert loaded["n_friction_robust_strategies"] == 1
    assert loaded["n_production_target_pass"] == 1
    assert loaded["best_production_target_strategy"]["production_target_pass"] is True
    assert loaded["friction_robust_strategies"][0]["production_target_pass"] is True
    assert not list(out.parent.glob("*.tmp"))


def test_sweep_json_payload_records_model_hashes_when_files_exist(tmp_path):
    model_a = tmp_path / "model_a.pkl"
    model_b = tmp_path / "model_b.pkl"
    model_a.write_bytes(b"model-a")
    model_b.write_bytes(b"model-b")

    payload = sweep._sweep_json_payload(
        model_paths=[model_a, model_b],
        oos_start="2025-01-02",
        oos_end=date(2025, 4, 1),
        window_days=30,
        stride_days=7,
        fee_regimes=["deploy"],
        fail_fast_max_dd_pct=40.0,
        fail_fast_max_intraday_dd_pct=35.0,
        fail_fast_neg_windows=2,
        rows=[],
        complete=True,
    )

    assert payload["model_sha256"] == [
        hashlib.sha256(b"model-a").hexdigest(),
        hashlib.sha256(b"model-b").hexdigest(),
    ]


def test_sweep_json_payload_records_spy_csv_hash_when_file_exists(tmp_path):
    spy_csv = tmp_path / "SPY.csv"
    spy_csv.write_text("timestamp,close\n2026-01-02T21:00:00Z,100\n", encoding="utf-8")

    payload = sweep._sweep_json_payload(
        spy_csv_path=spy_csv,
        model_paths=[Path("model.pkl")],
        oos_start="2025-01-02",
        oos_end=date(2025, 4, 1),
        window_days=30,
        stride_days=7,
        fee_regimes=["deploy"],
        fail_fast_max_dd_pct=40.0,
        fail_fast_max_intraday_dd_pct=35.0,
        fail_fast_neg_windows=2,
        rows=[],
        complete=True,
    )

    assert payload["spy_csv"] == str(spy_csv)
    assert payload["spy_csv_sha256"] == hashlib.sha256(spy_csv.read_bytes()).hexdigest()


def test_sweep_json_payload_reuses_supplied_spy_csv_hash(monkeypatch, tmp_path):
    spy_csv = tmp_path / "SPY.csv"
    spy_csv.write_text("timestamp,close\n2026-01-02T21:00:00Z,100\n", encoding="utf-8")

    def _fail_if_rehashed(_path):
        raise AssertionError("checkpoint payload should reuse precomputed SPY hash")

    monkeypatch.setattr(sweep, "_optional_file_sha256", _fail_if_rehashed)

    payload = sweep._sweep_json_payload(
        spy_csv_path=spy_csv,
        spy_csv_sha256="precomputed",
        model_paths=[Path("model.pkl")],
        oos_start="2025-01-02",
        oos_end=date(2025, 4, 1),
        window_days=30,
        stride_days=7,
        fee_regimes=["deploy"],
        fail_fast_max_dd_pct=40.0,
        fail_fast_max_intraday_dd_pct=35.0,
        fail_fast_neg_windows=2,
        rows=[],
        complete=True,
    )

    assert payload["spy_csv_sha256"] == "precomputed"


def test_sweep_json_payload_records_fm_latent_metadata(tmp_path):
    fm_latents = tmp_path / "latents.parquet"
    fm_latents.write_bytes(b"fm-latents")

    payload = sweep._sweep_json_payload(
        fm_latents_path=fm_latents,
        fm_n_latents=32,
        model_paths=[Path("model.pkl")],
        oos_start="2025-01-02",
        oos_end=date(2025, 4, 1),
        window_days=30,
        stride_days=7,
        fee_regimes=["deploy"],
        fail_fast_max_dd_pct=40.0,
        fail_fast_max_intraday_dd_pct=35.0,
        fail_fast_neg_windows=2,
        rows=[],
        complete=True,
    )

    assert payload["fm_latents_path"] == str(fm_latents)
    assert payload["fm_latents_sha256"] == hashlib.sha256(fm_latents.read_bytes()).hexdigest()
    assert payload["fm_n_latents"] == 32


def test_sweep_json_payload_reuses_supplied_fm_latent_hash(monkeypatch, tmp_path):
    fm_latents = tmp_path / "latents.parquet"
    fm_latents.write_bytes(b"fm-latents")

    def _fail_if_rehashed(path):
        if path is None:
            return None
        raise AssertionError("checkpoint payload should reuse precomputed FM latent hash")

    monkeypatch.setattr(sweep, "_optional_file_sha256", _fail_if_rehashed)

    payload = sweep._sweep_json_payload(
        fm_latents_path=fm_latents,
        fm_latents_sha256="precomputed",
        fm_n_latents=16,
        model_paths=[Path("model.pkl")],
        oos_start="2025-01-02",
        oos_end=date(2025, 4, 1),
        window_days=30,
        stride_days=7,
        fee_regimes=["deploy"],
        fail_fast_max_dd_pct=40.0,
        fail_fast_max_intraday_dd_pct=35.0,
        fail_fast_neg_windows=2,
        rows=[],
        complete=True,
    )

    assert payload["fm_latents_sha256"] == "precomputed"
    assert payload["fm_n_latents"] == 16


def test_sweep_json_payload_reuses_supplied_model_hashes(monkeypatch, tmp_path):
    model = tmp_path / "model.pkl"
    model.write_bytes(b"model")

    def _fail_if_rehashed(_paths):
        raise AssertionError("checkpoint payload should reuse precomputed hashes")

    monkeypatch.setattr(sweep, "_model_sha256", _fail_if_rehashed)

    payload = sweep._sweep_json_payload(
        model_paths=[model],
        model_sha256=["precomputed"],
        oos_start="2025-01-02",
        oos_end=date(2025, 4, 1),
        window_days=30,
        stride_days=7,
        fee_regimes=["deploy"],
        fail_fast_max_dd_pct=40.0,
        fail_fast_max_intraday_dd_pct=35.0,
        fail_fast_neg_windows=2,
        rows=[],
        complete=True,
    )

    assert payload["model_sha256"] == ["precomputed"]


def test_sweep_json_payload_records_ensemble_manifest_metadata(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model = model_dir / "alltrain_seed0.pkl"
    manifest = model_dir / "alltrain_ensemble.json"
    model.write_bytes(b"model")
    manifest_payload = {
        "trained_at": "2026-04-25T00:00:00",
        "train_start": "2020-01-01",
        "train_end": "2026-04-25",
        "seeds": [0],
        "config": {"feature_cols": ["ret_1d"]},
    }
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")

    payload = sweep._sweep_json_payload(
        model_paths=[model],
        oos_start="2025-01-02",
        oos_end=date(2025, 4, 1),
        window_days=30,
        stride_days=7,
        fee_regimes=["deploy"],
        fail_fast_max_dd_pct=40.0,
        fail_fast_max_intraday_dd_pct=35.0,
        fail_fast_neg_windows=2,
        rows=[],
        complete=True,
    )

    assert payload["ensemble_manifest"] == {
        "path": str(manifest),
        "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        **manifest_payload,
    }


def test_run_sweep_records_model_feature_mode(monkeypatch, tmp_path):
    oos = _mk_panel()
    train = oos.copy()
    build_kwargs: list[dict] = []

    def _fake_build_daily_dataset(**kwargs):
        build_kwargs.append(kwargs)
        return train, train.iloc[:0], oos

    class _PanelFeatureFakeModel(_FakeModel):
        feature_cols = (
            ["rank_ret_1d"]
            + ["cs_iqr_ret5"]
        )

    monkeypatch.setattr(sweep, "build_daily_dataset", _fake_build_daily_dataset)
    monkeypatch.setattr(
        sweep,
        "load_any_model",
        lambda path: _PanelFeatureFakeModel(seed=hash(str(path)) & 0xFFFF),
    )
    paths = _fake_paths(tmp_path, 2)

    cells = sweep.run_sweep(
        symbols=[f"SYM{k}" for k in range(6)],
        data_root=Path("/tmp"),
        model_paths=paths,
        train_start=date(2020, 1, 1), train_end=date(2024, 12, 31),
        oos_start=date(2025, 1, 2), oos_end=date(2025, 12, 31),
        window_days=10, stride_days=5,
        leverage_grid=[1.0],
        min_score_grid=[0.0],
        hold_through_grid=[True],
        top_n_grid=[1],
        fee_regimes=["deploy"],
    )

    assert build_kwargs[0]["include_cross_sectional_ranks"] is True
    assert build_kwargs[0]["include_cross_sectional_dispersion"] is True
    assert cells
    assert all(cell.ensemble_needs_ranks is True for cell in cells)
    assert all(cell.ensemble_needs_dispersion is True for cell in cells)
    rows = sweep._cells_to_rows(cells)
    assert all(row["ensemble_needs_ranks"] is True for row in rows)
    assert all(row["ensemble_needs_dispersion"] is True for row in rows)


def test_run_sweep_rejects_unsupported_model_feature_cols(monkeypatch, tmp_path):
    class _BadFeatureModel(_FakeModel):
        feature_cols = ["ret_1d", "future_alpha_leak"]

    monkeypatch.setattr(
        sweep,
        "load_any_model",
        lambda path: _BadFeatureModel(seed=hash(str(path)) & 0xFFFF),
    )
    paths = _fake_paths(tmp_path, 1)

    with pytest.raises(ValueError, match="unsupported live features"):
        sweep.run_sweep(
            symbols=[f"SYM{k}" for k in range(6)],
            data_root=Path("/tmp"),
            model_paths=paths,
            train_start=date(2020, 1, 1), train_end=date(2024, 12, 31),
            oos_start=date(2025, 1, 2), oos_end=date(2025, 12, 31),
            window_days=10, stride_days=5,
            leverage_grid=[1.0],
            min_score_grid=[0.0],
            hold_through_grid=[True],
            top_n_grid=[1],
            fee_regimes=["deploy"],
        )


def test_run_sweep_rejects_sparse_fm_latent_model_feature_cols(monkeypatch, tmp_path):
    class _SparseFmFeatureModel(_FakeModel):
        feature_cols = ["ret_1d", "latent_0", "latent_2", "fm_available"]

    monkeypatch.setattr(
        sweep,
        "load_any_model",
        lambda path: _SparseFmFeatureModel(seed=hash(str(path)) & 0xFFFF),
    )
    paths = _fake_paths(tmp_path, 1)

    with pytest.raises(ValueError, match="contiguous from latent_0"):
        sweep.run_sweep(
            symbols=[f"SYM{k}" for k in range(6)],
            data_root=Path("/tmp"),
            model_paths=paths,
            train_start=date(2020, 1, 1), train_end=date(2024, 12, 31),
            oos_start=date(2025, 1, 2), oos_end=date(2025, 12, 31),
            window_days=10, stride_days=5,
            leverage_grid=[1.0],
            min_score_grid=[0.0],
            hold_through_grid=[True],
            top_n_grid=[1],
            fee_regimes=["deploy"],
        )


def test_run_sweep_rejects_missing_explicit_fm_latents_path(tmp_path):
    paths = _fake_paths(tmp_path, 1)

    with pytest.raises(ValueError, match="--fm-latents-path not found"):
        sweep.run_sweep(
            symbols=[f"SYM{k}" for k in range(6)],
            data_root=Path("/tmp"),
            model_paths=paths,
            train_start=date(2020, 1, 1), train_end=date(2024, 12, 31),
            oos_start=date(2025, 1, 2), oos_end=date(2025, 12, 31),
            window_days=10, stride_days=5,
            leverage_grid=[1.0],
            min_score_grid=[0.0],
            hold_through_grid=[True],
            top_n_grid=[1],
            fee_regimes=["deploy"],
            fm_latents_path=tmp_path / "missing_latents.parquet",
        )


def test_run_sweep_rejects_fm_n_latents_below_model_requirement(monkeypatch, tmp_path):
    class _FmFeatureModel(_FakeModel):
        feature_cols = ["ret_1d", "latent_0", "latent_1", "latent_2", "fm_available"]

    monkeypatch.setattr(
        sweep,
        "load_any_model",
        lambda path: _FmFeatureModel(seed=hash(str(path)) & 0xFFFF),
    )
    paths = _fake_paths(tmp_path, 1)

    with pytest.raises(ValueError, match="smaller than model feature_cols require"):
        sweep.run_sweep(
            symbols=[f"SYM{k}" for k in range(6)],
            data_root=Path("/tmp"),
            model_paths=paths,
            train_start=date(2020, 1, 1), train_end=date(2024, 12, 31),
            oos_start=date(2025, 1, 2), oos_end=date(2025, 12, 31),
            window_days=10, stride_days=5,
            leverage_grid=[1.0],
            min_score_grid=[0.0],
            hold_through_grid=[True],
            top_n_grid=[1],
            fee_regimes=["deploy"],
            fm_n_latents=2,
        )


def test_run_sweep_rejects_mixed_ensemble_feature_contract(monkeypatch, tmp_path):
    paths = _fake_paths(tmp_path, 2)
    features_by_path = {
        paths[0]: ["ret_1d", "ret_5d"],
        paths[1]: ["ret_5d", "ret_1d"],
    }

    class _FeatureModel(_FakeModel):
        def __init__(self, seed: int, feature_cols: list[str]):
            super().__init__(seed)
            self.feature_cols = feature_cols

    monkeypatch.setattr(
        sweep,
        "load_any_model",
        lambda path: _FeatureModel(
            seed=hash(str(path)) & 0xFFFF,
            feature_cols=features_by_path[Path(path)],
        ),
    )

    with pytest.raises(ValueError, match="Ensemble feature_cols mismatch"):
        sweep.run_sweep(
            symbols=[f"SYM{k}" for k in range(6)],
            data_root=Path("/tmp"),
            model_paths=paths,
            train_start=date(2020, 1, 1), train_end=date(2024, 12, 31),
            oos_start=date(2025, 1, 2), oos_end=date(2025, 12, 31),
            window_days=10, stride_days=5,
            leverage_grid=[1.0],
            min_score_grid=[0.0],
            hold_through_grid=[True],
            top_n_grid=[1],
            fee_regimes=["deploy"],
        )


def test_production_target_gate_counts_only_friction_robust_passes():
    good = {
        "leverage": 2.0,
        "min_score": 0.85,
        "hold_through": True,
        "top_n": 1,
        "fee_regime": "deploy",
        "n_windows": 4,
        "expected_n_windows": 4,
        "median_monthly_pct": 30.0,
        "p10_monthly_pct": 20.0,
        "median_sortino": 2.0,
        "worst_dd_pct": 10.0,
        "n_neg": 0,
        "pain_adjusted_goodness_score": 8.0,
        "fill_buffer_bps": 5.0,
        "fail_fast_triggered": False,
    }
    bad = {
        **good,
        "leverage": 2.5,
        "median_monthly_pct": 20.0,
        "worst_dd_pct": 12.0,
        "pain_adjusted_goodness_score": 7.0,
    }
    fail_fast = {
        **good,
        "leverage": 3.0,
        "median_monthly_pct": 50.0,
        "worst_dd_pct": 8.0,
        "fail_fast_triggered": True,
    }
    negative_window = {
        **good,
        "leverage": 3.5,
        "median_monthly_pct": 60.0,
        "worst_dd_pct": 8.0,
        "n_neg": 1,
        "pain_adjusted_goodness_score": 12.0,
    }
    no_windows = {
        **good,
        "leverage": 4.0,
        "median_monthly_pct": 70.0,
        "worst_dd_pct": 8.0,
        "n_windows": 0,
        "pain_adjusted_goodness_score": 13.0,
    }
    incomplete_windows = {
        **good,
        "leverage": 4.5,
        "median_monthly_pct": 80.0,
        "worst_dd_pct": 8.0,
        "n_windows": 3,
        "expected_n_windows": 4,
        "pain_adjusted_goodness_score": 14.0,
    }
    rows = [bad, fail_fast, negative_window, no_windows, incomplete_windows, good]

    passes = sweep._production_target_pass_rows(rows)

    assert [r["leverage"] for r in passes] == [2.0]
    summaries = sweep._friction_robust_strategy_rows(rows)
    incomplete = next(r for r in summaries if r["leverage"] == 4.5)
    assert incomplete["min_n_windows"] == 3
    assert incomplete["max_expected_n_windows"] == 4
    assert incomplete["required_min_n_windows"] == 4
    assert incomplete["production_target_pass"] is False
    assert sweep._production_target_exit_code(rows, required=False) == 0
    assert sweep._production_target_exit_code(rows, required=True) == 0
    assert sweep._production_target_exit_code([bad], required=True) == sweep.PRODUCTION_TARGET_EXIT_CODE
    assert sweep._production_target_exit_code([negative_window], required=True) == sweep.PRODUCTION_TARGET_EXIT_CODE
    assert sweep._production_target_exit_code([no_windows], required=True) == sweep.PRODUCTION_TARGET_EXIT_CODE
    assert sweep._production_target_exit_code([incomplete_windows], required=True) == sweep.PRODUCTION_TARGET_EXIT_CODE


def test_require_production_target_flag_parses():
    args = sweep.parse_args([
        "--symbols-file", "symbols.txt",
        "--model-paths", "model.pkl",
        "--require-production-target",
    ])

    assert args.require_production_target is True


def test_friction_robust_strategy_rows_take_worst_fee_cell():
    base = {
        "leverage": 2.0,
        "min_score": 0.85,
        "hold_through": True,
        "top_n": 1,
        "n_windows": 8,
        "p10_monthly_pct": 26.0,
        "median_sortino": 3.0,
        "n_neg": 0,
        "goodness_score": 15.0,
        "robust_goodness_score": 13.0,
        "pain_adjusted_goodness_score": 11.0,
        "worst_intraday_dd_pct": 18.0,
        "avg_intraday_dd_pct": 4.0,
        "time_under_water_pct": 20.0,
        "ulcer_index": 3.0,
        "inference_min_dolvol": 5_000_000.0,
        "inference_min_vol_20d": 0.0,
        "inference_max_vol_20d": 0.0,
        "skip_prob": 0.0,
        "skip_seed": 0,
        "regime_gate_window": 0,
        "vol_target_ann": 0.0,
        "inv_vol_target_ann": 0.0,
        "inv_vol_floor": 0.05,
        "inv_vol_cap": 3.0,
        "max_ret_20d_rank_pct": 1.0,
        "min_ret_5d_rank_pct": 0.0,
        "regime_cs_iqr_max": 0.0,
        "regime_cs_skew_min": -1e9,
        "no_picks_fallback_symbol": "",
        "no_picks_fallback_alloc_scale": 0.0,
        "conviction_scaled_alloc": False,
        "conviction_alloc_low": 0.55,
        "conviction_alloc_high": 0.85,
        "allocation_mode": "equal",
        "allocation_temp": 1.0,
        "fail_fast_triggered": False,
    }
    deploy = {
        **base,
        "fee_regime": "deploy",
        "fill_buffer_bps": 5.0,
        "median_monthly_pct": 34.0,
        "worst_dd_pct": 12.0,
    }
    stress = {
        **base,
        "fee_regime": "stress36x",
        "fill_buffer_bps": 15.0,
        "median_monthly_pct": 24.0,
        "worst_dd_pct": 28.0,
        "n_neg": 2,
        "pain_adjusted_goodness_score": -8.0,
        "time_under_water_pct": 55.0,
    }

    summaries = sweep._friction_robust_strategy_rows([deploy, stress])

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["n_friction_cells"] == 2
    assert summary["fee_regimes"] == ["deploy", "stress36x"]
    assert summary["fill_buffer_bps_values"] == [5.0, 15.0]
    assert summary["skip_prob_values"] == [0.0]
    assert summary["skip_seed_values"] == [0]
    assert summary["worst_median_monthly_pct"] == 24.0
    assert summary["max_worst_dd_pct"] == 28.0
    assert summary["max_time_under_water_pct"] == 55.0
    assert summary["max_n_neg"] == 2
    assert summary["worst_pain_adjusted_goodness_score"] == -8.0
    assert summary["worst_fee_regime_by_pain"] == "stress36x"
    assert summary["production_target_pass"] is False


def test_friction_robust_strategy_rows_treat_skip_as_stress_axis():
    base = {
        "leverage": 2.0,
        "min_score": 0.85,
        "hold_through": True,
        "top_n": 1,
        "n_windows": 8,
        "expected_n_windows": 8,
        "p10_monthly_pct": 30.0,
        "median_sortino": 3.0,
        "n_neg": 0,
        "goodness_score": 15.0,
        "robust_goodness_score": 13.0,
        "pain_adjusted_goodness_score": 11.0,
        "worst_intraday_dd_pct": 18.0,
        "avg_intraday_dd_pct": 4.0,
        "time_under_water_pct": 20.0,
        "ulcer_index": 3.0,
        "fee_regime": "stress36x",
        "fill_buffer_bps": 15.0,
        "inference_min_dolvol": 5_000_000.0,
        "inference_min_vol_20d": 0.0,
        "inference_max_vol_20d": 0.0,
        "regime_gate_window": 0,
        "vol_target_ann": 0.0,
        "inv_vol_target_ann": 0.0,
        "inv_vol_floor": 0.05,
        "inv_vol_cap": 3.0,
        "max_ret_20d_rank_pct": 1.0,
        "min_ret_5d_rank_pct": 0.0,
        "regime_cs_iqr_max": 0.0,
        "regime_cs_skew_min": -1e9,
        "no_picks_fallback_symbol": "",
        "no_picks_fallback_alloc_scale": 0.0,
        "conviction_scaled_alloc": False,
        "conviction_alloc_low": 0.55,
        "conviction_alloc_high": 0.85,
        "allocation_mode": "equal",
        "allocation_temp": 1.0,
        "fail_fast_triggered": False,
    }
    live_like = {
        **base,
        "skip_prob": 0.0,
        "skip_seed": 0,
        "median_monthly_pct": 33.0,
        "worst_dd_pct": 10.0,
    }
    missed_order_stress = {
        **base,
        "skip_prob": 0.25,
        "skip_seed": 7,
        "median_monthly_pct": 23.0,
        "worst_dd_pct": 27.0,
        "n_neg": 1,
        "pain_adjusted_goodness_score": -5.0,
    }

    summaries = sweep._friction_robust_strategy_rows([live_like, missed_order_stress])

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["n_friction_cells"] == 2
    assert summary["skip_prob_values"] == [0.0, 0.25]
    assert summary["skip_seed_values"] == [0, 7]
    assert summary["worst_median_monthly_pct"] == 23.0
    assert summary["max_worst_dd_pct"] == 27.0
    assert summary["max_n_neg"] == 1
    assert summary["production_target_pass"] is False


def test_allocation_grid_pairs_drop_temp_invariant_duplicates():
    pairs = sweep._allocation_grid_pairs(
        ["equal", "score_norm", "softmax"],
        [0.25, 1.0],
    )
    assert pairs == [
        ("equal", 1.0),
        ("score_norm", 1.0),
        ("softmax", 0.25),
        ("softmax", 1.0),
    ]


def test_fallback_and_conviction_axes_match_grid_product(monkeypatch, tmp_path):
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
        min_score_grid=[0.0],
        hold_through_grid=[False],
        top_n_grid=[1],
        fee_regimes=["deploy"],
        no_picks_fallback_symbol="SPY",
        no_picks_fallback_alloc_grid=[0.25, 0.50],
        conviction_scaled_alloc_grid=[False, True],
        conviction_alloc_low=0.40,
        conviction_alloc_high=0.90,
        min_picks_grid=[0, 1],
    )

    assert len(cells) == 8
    assert sorted({c.no_picks_fallback_alloc_scale for c in cells}) == [0.25, 0.50]
    assert sorted({c.conviction_scaled_alloc for c in cells}) == [False, True]
    assert sorted({c.min_picks for c in cells}) == [0, 1]
    for c in cells:
        assert c.no_picks_fallback_symbol == "SPY"
        assert c.conviction_alloc_low == pytest.approx(0.40)
        assert c.conviction_alloc_high == pytest.approx(0.90)


def test_allocation_mode_axes_match_grid_product(monkeypatch, tmp_path):
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
        min_score_grid=[0.0],
        hold_through_grid=[False],
        top_n_grid=[2],
        fee_regimes=["deploy"],
        allocation_mode_grid=["equal", "score_norm", "softmax", "worksteal"],
        allocation_temp_grid=[0.25, 1.0],
    )

    assert len(cells) == 5
    pairs = [(c.allocation_mode, c.allocation_temp) for c in cells]
    assert pairs == [
        ("equal", 1.0),
        ("score_norm", 1.0),
        ("softmax", 0.25),
        ("softmax", 1.0),
        ("worksteal", 1.0),
    ]


def test_allocation_mode_rejects_unknown_value(monkeypatch, tmp_path):
    _install_fakes(monkeypatch)
    paths = _fake_paths(tmp_path, 1)

    with pytest.raises(ValueError, match="allocation_mode"):
        sweep.run_sweep(
            symbols=[f"SYM{k}" for k in range(6)],
            data_root=Path("/tmp"),
            model_paths=paths,
            train_start=date(2020, 1, 1), train_end=date(2024, 12, 31),
            oos_start=date(2025, 1, 2), oos_end=date(2025, 12, 31),
            window_days=10, stride_days=5,
            leverage_grid=[1.0],
            min_score_grid=[0.0],
            hold_through_grid=[False],
            top_n_grid=[2],
            fee_regimes=["deploy"],
            allocation_mode_grid=["martingale"],
        )


def test_min_secondary_allocation_axis_matches_grid_product(monkeypatch, tmp_path):
    _install_fakes(monkeypatch)
    paths = _fake_paths(tmp_path, 2)
    calls: list[float] = []

    def _fake_run_cell(**kwargs):
        calls.append(float(kwargs["min_secondary_allocation"]))
        return sweep.CellResult(
            leverage=float(kwargs["leverage"]),
            min_score=float(kwargs["min_score"]),
            hold_through=bool(kwargs["hold_through"]),
            top_n=int(kwargs["top_n"]),
            fee_regime=str(kwargs["fee_regime"]),
            n_windows=5,
            median_monthly_pct=1.0,
            p10_monthly_pct=1.0,
            median_sortino=1.0,
            worst_dd_pct=1.0,
            n_neg=0,
            allocation_mode=str(kwargs["allocation_mode"]),
            allocation_temp=float(kwargs["allocation_temp"]),
            min_secondary_allocation=float(kwargs["min_secondary_allocation"]),
        )

    monkeypatch.setattr(sweep, "_run_cell", _fake_run_cell)
    cells = sweep.run_sweep(
        symbols=[f"SYM{k}" for k in range(6)],
        data_root=Path("/tmp"),
        model_paths=paths,
        train_start=date(2020, 1, 1), train_end=date(2024, 12, 31),
        oos_start=date(2025, 1, 2), oos_end=date(2025, 12, 31),
        window_days=10, stride_days=5,
        leverage_grid=[1.0],
        min_score_grid=[0.0],
        hold_through_grid=[False],
        top_n_grid=[2],
        fee_regimes=["deploy"],
        allocation_mode_grid=["softmax"],
        allocation_temp_grid=[0.25],
        min_secondary_allocation_grid=[0.0, 0.2],
    )

    assert calls == [0.0, 0.2]
    rows = sweep._cells_to_rows(cells)
    assert [r["min_secondary_allocation"] for r in rows] == [0.0, 0.2]


def test_corr_gate_axes_match_grid_product(monkeypatch, tmp_path):
    _install_fakes(monkeypatch)
    paths = _fake_paths(tmp_path, 2)

    calls: list[tuple[int, int, float]] = []

    def _fake_run_cell(**kwargs):
        calls.append((
            int(kwargs["corr_window_days"]),
            int(kwargs["corr_min_periods"]),
            float(kwargs["corr_max_signed"]),
        ))
        return sweep.CellResult(
            leverage=float(kwargs["leverage"]),
            min_score=float(kwargs["min_score"]),
            hold_through=bool(kwargs["hold_through"]),
            top_n=int(kwargs["top_n"]),
            fee_regime=str(kwargs["fee_regime"]),
            n_windows=5,
            median_monthly_pct=1.0,
            p10_monthly_pct=1.0,
            median_sortino=1.0,
            worst_dd_pct=1.0,
            n_neg=0,
            corr_window_days=int(kwargs["corr_window_days"]),
            corr_min_periods=int(kwargs["corr_min_periods"]),
            corr_max_signed=float(kwargs["corr_max_signed"]),
        )

    monkeypatch.setattr(sweep, "_run_cell", _fake_run_cell)
    cells = sweep.run_sweep(
        symbols=[f"SYM{k}" for k in range(6)],
        data_root=Path("/tmp"),
        model_paths=paths,
        train_start=date(2020, 1, 1), train_end=date(2024, 12, 31),
        oos_start=date(2025, 1, 2), oos_end=date(2025, 12, 31),
        window_days=10, stride_days=5,
        leverage_grid=[1.0],
        min_score_grid=[0.0],
        hold_through_grid=[False],
        top_n_grid=[2],
        fee_regimes=["deploy"],
        corr_window_days_grid=[0, 60],
        corr_min_periods=12,
        corr_max_signed_grid=[1.0, 0.6],
    )

    assert calls == [(0, 12, 1.0), (0, 12, 0.6), (60, 12, 1.0), (60, 12, 0.6)]
    rows = sweep._cells_to_rows(cells)
    assert rows[-1]["corr_window_days"] == 60
    assert rows[-1]["corr_min_periods"] == 12
    assert rows[-1]["corr_max_signed"] == 0.6


def test_goodness_score_formula():
    # Baseline: p10=96, worst_dd=7.18, 0 neg → 96 - 7.18 - 0 = 88.82
    g = sweep.compute_goodness(p10_monthly_pct=96.0, worst_dd_pct=7.18,
                                n_neg=0, n_windows=60)
    assert g == pytest.approx(88.82, abs=1e-6)

    # 1 neg in 60 → extra penalty = 100 * (1/60) = 1.6667
    g2 = sweep.compute_goodness(p10_monthly_pct=96.0, worst_dd_pct=7.18,
                                n_neg=1, n_windows=60)
    assert g2 == pytest.approx(88.82 - 100.0 / 60.0, abs=1e-6)


def test_goodness_score_responds_to_levers():
    """Improving p10 should lift goodness; widening DD should lower it."""
    base = sweep.compute_goodness(96.0, 7.0, 0, 60)
    better_p10 = sweep.compute_goodness(108.0, 7.0, 0, 60)
    worse_dd = sweep.compute_goodness(96.0, 9.0, 0, 60)
    worse_neg = sweep.compute_goodness(96.0, 7.0, 1, 60)
    assert better_p10 > base
    assert worse_dd < base
    assert worse_neg < base


def test_cells_populated_with_goodness(monkeypatch, tmp_path):
    """Run the sweep end-to-end and verify goodness_score is plumbed."""
    _install_fakes(monkeypatch)
    paths = _fake_paths(tmp_path, 2)

    cells = sweep.run_sweep(
        symbols=[f"SYM{k}" for k in range(6)],
        data_root=Path("/tmp"),
        model_paths=paths,
        train_start=date(2020, 1, 1), train_end=date(2024, 12, 31),
        oos_start=date(2025, 1, 2), oos_end=date(2025, 12, 31),
        window_days=10, stride_days=5,
        leverage_grid=[1.0, 2.0],
        min_score_grid=[0.0],
        hold_through_grid=[True],
        top_n_grid=[1],
        fee_regimes=["deploy"],
    )
    for c in cells:
        # goodness matches the public formula for this cell.
        expected = sweep.compute_goodness(
            c.p10_monthly_pct, c.worst_dd_pct, c.n_neg, c.n_windows,
        )
        assert c.goodness_score == pytest.approx(expected, abs=1e-9)


def test_run_cell_monthly_uses_elapsed_window_days_not_traded_days(monkeypatch):
    """Sparse-gate cells should not get one-day annualisation."""
    oos = _mk_panel(n_days=10, n_syms=3)
    scores = pd.Series(1.0, index=oos.index)
    windows = [(oos["date"].min(), oos["date"].max())]
    trade_day = sorted(oos["date"].unique())[0]

    def _fake_simulate(*args, **kwargs):
        return SimpleNamespace(
            day_results=[SimpleNamespace(day=trade_day)],
            total_return_pct=10.0,
            sortino_ratio=3.0,
            max_drawdown_pct=0.0,
            worst_intraday_dd_pct=0.0,
            avg_intraday_dd_pct=0.0,
            time_under_water_pct=0.0,
            ulcer_index=0.0,
            stopped_early=False,
        )

    monkeypatch.setattr(sweep, "simulate", _fake_simulate)

    cell = sweep._run_cell(
        oos_df=oos,
        scores=scores,
        windows=windows,
        leverage=1.0,
        min_score=0.0,
        hold_through=True,
        top_n=1,
        fee_regime="deploy",
    )

    assert cell.median_monthly_pct == pytest.approx(
        sweep._monthly_return(10.0, 10) * 100.0
    )
    assert cell.median_monthly_pct < sweep._monthly_return(10.0, 1) * 100.0
    assert cell.median_active_day_pct == pytest.approx(10.0)
    assert cell.min_active_day_pct == pytest.approx(10.0)


def test_elapsed_window_days_stopped_early_counts_days_through_stop():
    oos = _mk_panel(n_days=10, n_syms=2)
    days = sorted(oos["date"].unique())
    res = SimpleNamespace(
        stopped_early=True,
        day_results=[SimpleNamespace(day=days[4])],
    )

    assert sweep._elapsed_window_days(oos, res) == 5


def test_fail_fast_max_dd_prunes_cell_and_forces_losing_rank(monkeypatch):
    """A breached max-DD budget should stop more windows and never rank high."""
    oos = _mk_panel(n_days=30, n_syms=6)
    scores = pd.Series(1.0, index=oos.index)
    windows = sweep._build_windows(sorted(oos["date"].unique()), 10, 5)
    calls = {"n": 0}

    def _fake_simulate(*args, **kwargs):
        calls["n"] += 1
        assert args[2].stop_on_drawdown_pct == pytest.approx(40.0)
        return SimpleNamespace(
            day_results=[object()] * 10,
            total_return_pct=200.0,
            sortino_ratio=10.0,
            max_drawdown_pct=41.0,
            worst_intraday_dd_pct=42.0,
            avg_intraday_dd_pct=11.0,
            time_under_water_pct=90.0,
            ulcer_index=20.0,
        )

    monkeypatch.setattr(sweep, "simulate", _fake_simulate)

    cell = sweep._run_cell(
        oos_df=oos,
        scores=scores,
        windows=windows,
        leverage=3.0,
        min_score=0.0,
        hold_through=True,
        top_n=1,
        fee_regime="deploy",
        fail_fast_max_dd_pct=40.0,
    )

    assert calls["n"] == 1
    assert cell.n_windows == 1
    assert cell.fail_fast_triggered is True
    assert cell.fail_fast_reason == "max_dd_pct>=40"
    assert cell.goodness_score == sweep.FAIL_FAST_SCORE
    assert cell.robust_goodness_score == sweep.FAIL_FAST_SCORE
    assert cell.pain_adjusted_goodness_score == sweep.FAIL_FAST_SCORE
    assert cell.safety_goodness_score == sweep.FAIL_FAST_SCORE


def test_fail_fast_max_intraday_dd_prunes_cell_and_forces_losing_rank(monkeypatch):
    """Intraday risk breach should prune even if close-to-close DD is fine."""
    oos = _mk_panel(n_days=30, n_syms=6)
    scores = pd.Series(1.0, index=oos.index)
    windows = sweep._build_windows(sorted(oos["date"].unique()), 10, 5)
    calls = {"n": 0}

    def _fake_simulate(*args, **kwargs):
        calls["n"] += 1
        assert args[2].stop_on_intraday_drawdown_pct == pytest.approx(40.0)
        return SimpleNamespace(
            day_results=[object()] * 10,
            total_return_pct=25.0,
            sortino_ratio=2.0,
            max_drawdown_pct=3.0,
            worst_intraday_dd_pct=45.0,
            avg_intraday_dd_pct=12.0,
            time_under_water_pct=15.0,
            ulcer_index=2.0,
        )

    monkeypatch.setattr(sweep, "simulate", _fake_simulate)

    cell = sweep._run_cell(
        oos_df=oos,
        scores=scores,
        windows=windows,
        leverage=3.0,
        min_score=0.0,
        hold_through=True,
        top_n=1,
        fee_regime="deploy",
        fail_fast_max_intraday_dd_pct=40.0,
    )

    assert calls["n"] == 1
    assert cell.n_windows == 1
    assert cell.fail_fast_triggered is True
    assert cell.fail_fast_reason == "intraday_dd_pct>=40"
    assert cell.goodness_score == sweep.FAIL_FAST_SCORE
    assert cell.robust_goodness_score == sweep.FAIL_FAST_SCORE
    assert cell.pain_adjusted_goodness_score == sweep.FAIL_FAST_SCORE
    assert cell.safety_goodness_score == sweep.FAIL_FAST_SCORE


def test_fail_fast_neg_windows_prunes_after_threshold(monkeypatch):
    """Negative-window bail should stop once the configured count is reached."""
    oos = _mk_panel(n_days=30, n_syms=6)
    scores = pd.Series(1.0, index=oos.index)
    windows = sweep._build_windows(sorted(oos["date"].unique()), 10, 5)
    returns = iter([-10.0, -8.0, 100.0])
    calls = {"n": 0}

    def _fake_simulate(*args, **kwargs):
        calls["n"] += 1
        return SimpleNamespace(
            day_results=[object()] * 10,
            total_return_pct=next(returns),
            sortino_ratio=-1.0,
            max_drawdown_pct=5.0,
            worst_intraday_dd_pct=6.0,
            avg_intraday_dd_pct=2.0,
            time_under_water_pct=75.0,
            ulcer_index=4.0,
        )

    monkeypatch.setattr(sweep, "simulate", _fake_simulate)

    cell = sweep._run_cell(
        oos_df=oos,
        scores=scores,
        windows=windows,
        leverage=2.0,
        min_score=0.0,
        hold_through=True,
        top_n=1,
        fee_regime="deploy",
        fail_fast_neg_windows=2,
    )

    assert calls["n"] == 2
    assert cell.n_neg == 2
    assert cell.fail_fast_triggered is True
    assert cell.fail_fast_reason == "neg_windows>=2"
    assert cell.pain_adjusted_goodness_score == sweep.FAIL_FAST_SCORE
    assert cell.safety_goodness_score == sweep.FAIL_FAST_SCORE


# ─── robust_goodness tests ────────────────────────────────────────────────────

def test_robust_goodness_all_positive_matches_expected():
    # All-positive deploy-style case: 0 neg, 0 magnitude penalty;
    # robust = p10 − 1.5·worst_dd. Use a flat distribution so p10
    # is unambiguously the shared value.
    monthlies = [100.0] * 60
    worst_dd = 5.34
    rg = sweep.compute_robust_goodness(monthlies, worst_dd)
    assert rg == pytest.approx(100.0 - 1.5 * 5.34, abs=1e-6)


def test_robust_goodness_dd_penalty_is_1p5x_plain():
    """Doubling DD should subtract exactly 1.5x more than compute_goodness."""
    monthlies = [10.0] * 60
    g_base = sweep.compute_goodness(10.0, 10.0, 0, 60)
    g_dd2 = sweep.compute_goodness(10.0, 20.0, 0, 60)
    rg_base = sweep.compute_robust_goodness(monthlies, 10.0)
    rg_dd2 = sweep.compute_robust_goodness(monthlies, 20.0)
    # Plain: −1pp per 1pp DD. Robust: −1.5pp per 1pp DD.
    assert (g_base - g_dd2) == pytest.approx(10.0, abs=1e-9)
    assert (rg_base - rg_dd2) == pytest.approx(15.0, abs=1e-9)


def test_robust_goodness_distinguishes_loss_magnitude():
    """Same neg count, different magnitudes → robust differs, plain doesn't."""
    shallow = [-3.0] * 5 + [40.0] * 55   # 5 neg at −3%, p10 ≈ −3
    deep    = [-15.0] * 5 + [40.0] * 55  # 5 neg at −15%, p10 ≈ −15
    # compute_goodness at identical DD / n_neg is identical iff p10 matches;
    # these cases differ in p10 so they'll already differ, but the
    # *magnitude* term adds an additional gap — exactly what we want.
    rg_shallow = sweep.compute_robust_goodness(shallow, 10.0)
    rg_deep = sweep.compute_robust_goodness(deep, 10.0)
    # Magnitude term alone: shallow pays 2 * (15/60) = 0.5,
    # deep pays 2 * (75/60) = 2.5 → extra spread 2.0 on top of p10 gap.
    gap = rg_shallow - rg_deep
    assert gap > 0, "deep losses should be strictly worse"
    # The magnitude term contributes EXACTLY +2.0 to the gap.
    p10_gap = np.percentile(shallow, 10) - np.percentile(deep, 10)
    # Actually: mean_abs_neg_shallow = 15/60 = 0.25
    #           mean_abs_neg_deep    = 75/60 = 1.25
    #   gap from magnitude = 2 * (1.25 − 0.25) = 2.0
    assert gap == pytest.approx(p10_gap + 2.0, abs=1e-6)


def test_robust_goodness_zero_on_empty():
    assert sweep.compute_robust_goodness([], 10.0) == 0.0


def test_robust_goodness_custom_weights():
    monthlies = [-5.0] * 5 + [30.0] * 55
    default = sweep.compute_robust_goodness(monthlies, 10.0)
    zero_dd_penalty = sweep.compute_robust_goodness(
        monthlies, 10.0,
        weights={**sweep.ROBUST_GOODNESS_WEIGHTS, "dd_coef": 0.0},
    )
    # Dropping the DD penalty only should raise goodness by exactly 1.5*10.
    assert (zero_dd_penalty - default) == pytest.approx(15.0, abs=1e-9)


def test_pain_adjusted_goodness_penalizes_tuw_and_ulcer():
    monthlies = [20.0] * 60
    base = sweep.compute_pain_adjusted_goodness(monthlies, 5.0, 0.0, 0.0)
    pain = sweep.compute_pain_adjusted_goodness(monthlies, 5.0, 20.0, 3.0)
    expected_penalty = (
        sweep.PAIN_ADJUSTED_GOODNESS_WEIGHTS["tuw_coef"] * 20.0
        + sweep.PAIN_ADJUSTED_GOODNESS_WEIGHTS["ulcer_coef"] * 3.0
    )
    assert (base - pain) == pytest.approx(expected_penalty, abs=1e-9)


def test_pain_adjusted_goodness_custom_weights():
    monthlies = [15.0] * 60
    default = sweep.compute_pain_adjusted_goodness(monthlies, 5.0, 12.0, 2.0)
    no_pain = sweep.compute_pain_adjusted_goodness(
        monthlies,
        5.0,
        12.0,
        2.0,
        pain_weights={**sweep.PAIN_ADJUSTED_GOODNESS_WEIGHTS, "tuw_coef": 0.0, "ulcer_coef": 0.0},
    )
    expected_delta = (
        sweep.PAIN_ADJUSTED_GOODNESS_WEIGHTS["tuw_coef"] * 12.0
        + sweep.PAIN_ADJUSTED_GOODNESS_WEIGHTS["ulcer_coef"] * 2.0
    )
    assert (no_pain - default) == pytest.approx(expected_delta, abs=1e-9)


def test_interval_loss_uses_worst_comparable_window_risk():
    losses = sweep.compute_window_interval_loss_pcts(
        monthlies=[25.0, -55.0, 10.0],
        drawdowns=[5.0, 35.0, 20.0],
        intraday_drawdowns=[7.0, 30.0, 45.0],
    )

    assert losses.tolist() == pytest.approx([7.0, 55.0, 45.0])


def test_safety_goodness_penalizes_hidden_crash_tail():
    smooth = sweep.compute_safety_goodness(
        [28.0] * 10,
        drawdowns=[6.0] * 10,
        intraday_drawdowns=[7.0] * 10,
        time_under_water_pct=5.0,
        ulcer_index=1.0,
    )
    crashy = sweep.compute_safety_goodness(
        [31.0] * 9 + [-55.0],
        drawdowns=[6.0] * 9 + [35.0],
        intraday_drawdowns=[7.0] * 9 + [40.0],
        time_under_water_pct=35.0,
        ulcer_index=8.0,
    )

    assert crashy < smooth


def test_safety_goodness_custom_weights():
    monthlies = [15.0] * 9 + [-20.0]
    default = sweep.compute_safety_goodness(
        monthlies,
        drawdowns=[4.0] * 10,
        intraday_drawdowns=[6.0] * 10,
    )
    no_interval_penalty = sweep.compute_safety_goodness(
        monthlies,
        drawdowns=[4.0] * 10,
        intraday_drawdowns=[6.0] * 10,
        weights={
            **sweep.SAFETY_GOODNESS_WEIGHTS,
            "p90_interval_loss_coef": 0.0,
            "worst_interval_loss_coef": 0.0,
        },
    )

    assert no_interval_penalty > default


def test_robust_goodness_in_cell_result_and_rows():
    c = sweep.CellResult(
        leverage=2.0, min_score=0.85, hold_through=True, top_n=1,
        fee_regime="deploy", n_windows=60,
        median_monthly_pct=141.0, p10_monthly_pct=96.0,
        median_sortino=40.0, worst_dd_pct=7.18, n_neg=0,
        goodness_score=88.82, robust_goodness_score=85.23,
        pain_adjusted_goodness_score=74.56,
        safety_goodness_score=66.0,
        p05_monthly_pct=95.0,
        worst_monthly_pct=94.0,
        p90_interval_loss_pct=8.0,
        worst_interval_loss_pct=9.0,
    )
    rows = sweep._cells_to_rows([c])
    assert rows[0]["robust_goodness_score"] == pytest.approx(85.23)
    assert rows[0]["pain_adjusted_goodness_score"] == pytest.approx(74.56)
    assert rows[0]["safety_goodness_score"] == pytest.approx(66.0)
    assert rows[0]["p05_monthly_pct"] == pytest.approx(95.0)
    assert rows[0]["worst_monthly_pct"] == pytest.approx(94.0)
    assert rows[0]["p90_interval_loss_pct"] == pytest.approx(8.0)
    assert rows[0]["worst_interval_loss_pct"] == pytest.approx(9.0)


def test_robust_goodness_sweep_end_to_end(monkeypatch, tmp_path):
    """End-to-end sweep populates robust_goodness_score alongside goodness."""
    _install_fakes(monkeypatch)
    paths = _fake_paths(tmp_path, 2)
    cells = sweep.run_sweep(
        symbols=[f"SYM{k}" for k in range(6)],
        data_root=Path("/tmp"),
        model_paths=paths,
        train_start=date(2020, 1, 1), train_end=date(2024, 12, 31),
        oos_start=date(2025, 1, 2), oos_end=date(2025, 12, 31),
        window_days=10, stride_days=5,
        leverage_grid=[1.0, 2.0],
        min_score_grid=[0.0],
        hold_through_grid=[True],
        top_n_grid=[1],
        fee_regimes=["deploy"],
    )
    for c in cells:
        # robust_goodness must be a finite float, and should not equal
        # plain goodness in general (different coefficients).
        assert np.isfinite(c.robust_goodness_score)
        # With DD=0 and all monthlies positive they coincide; otherwise not.
        # Just sanity-check the column is populated:
        assert c.robust_goodness_score != 0.0 or c.worst_dd_pct == 0.0


def test_tuw_and_ulcer_in_cell_result_and_rows():
    """TuW + Ulcer round-trip through CellResult → _cells_to_rows → JSON row."""
    c = sweep.CellResult(
        leverage=2.0, min_score=0.85, hold_through=True, top_n=1,
        fee_regime="deploy", n_windows=60,
        median_monthly_pct=141.0, p10_monthly_pct=96.0,
        median_sortino=40.0, worst_dd_pct=7.18, n_neg=0,
        goodness_score=88.82, robust_goodness_score=85.23,
        pain_adjusted_goodness_score=74.56,
        time_under_water_pct=42.5, ulcer_index=3.17,
    )
    rows = sweep._cells_to_rows([c])
    assert rows[0]["time_under_water_pct"] == pytest.approx(42.5)
    assert rows[0]["ulcer_index"] == pytest.approx(3.17)
    assert rows[0]["pain_adjusted_goodness_score"] == pytest.approx(74.56)


def test_tuw_and_ulcer_sweep_end_to_end(monkeypatch, tmp_path):
    """End-to-end sweep populates tuw + ulcer aggregates."""
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
        hold_through_grid=[True], top_n_grid=[1],
        fee_regimes=["deploy"],
    )
    assert cells, "expected at least one cell"
    for c in cells:
        # TuW is a percentage in [0, 100]; Ulcer ≥ 0.
        assert 0.0 <= c.time_under_water_pct <= 100.0
        assert c.ulcer_index >= 0.0
        # Ulcer ≤ max-DD in %. RMS ≤ max by definition.
        assert c.ulcer_index <= c.worst_dd_pct + 1e-9
        assert np.isfinite(c.pain_adjusted_goodness_score)
        assert c.pain_adjusted_goodness_score <= c.robust_goodness_score + 1e-9


# ── SPY regime-gate + vol-target grids (task #92) ──────────────────────────

def _write_fake_spy_csv(path: Path, n: int = 120) -> Path:
    """Synthetic SPY CSV with pd.to_datetime-parsable timestamps."""
    rng = np.random.default_rng(5)
    ts = pd.date_range("2024-08-01", periods=n, freq="B", tz="UTC")
    price = 400.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, n))
    pd.DataFrame({"timestamp": ts, "close": price}).to_csv(path, index=False)
    return path


def test_sweep_vol_target_axis_matches_product(monkeypatch, tmp_path):
    """vol_target_ann_grid widens the cell total linearly."""
    _install_fakes(monkeypatch)
    paths = _fake_paths(tmp_path, 2)
    spy = _write_fake_spy_csv(tmp_path / "SPY.csv")

    cells = sweep.run_sweep(
        symbols=[f"SYM{k}" for k in range(6)],
        data_root=Path("/tmp"),
        model_paths=paths,
        train_start=date(2020, 1, 1), train_end=date(2024, 12, 31),
        oos_start=date(2025, 1, 2), oos_end=date(2025, 12, 31),
        window_days=10, stride_days=5,
        leverage_grid=[1.0], min_score_grid=[0.0],
        hold_through_grid=[True], top_n_grid=[1],
        fee_regimes=["deploy"],
        vol_target_ann_grid=[0.0, 0.15, 0.25],
        spy_csv_path=spy,
    )
    # 1 × 1 × 1 × 1 × 1 × (1 inf_dv × 1 vol × 1 maxvol × 1 sp × 1 ss × 1 fb)
    # × (1 rgw × 3 vta) = 3.
    assert len(cells) == 3
    vtas = sorted(c.vol_target_ann for c in cells)
    assert vtas == [0.0, 0.15, 0.25]


def test_sweep_vol_target_requires_spy_csv(monkeypatch, tmp_path):
    """Positive vol_target_ann without --spy-csv must fail loud."""
    _install_fakes(monkeypatch)
    paths = _fake_paths(tmp_path, 1)
    with pytest.raises(FileNotFoundError, match="spy"):
        sweep.run_sweep(
            symbols=[f"SYM{k}" for k in range(6)],
            data_root=Path("/tmp"),
            model_paths=paths,
            train_start=date(2020, 1, 1), train_end=date(2024, 12, 31),
            oos_start=date(2025, 1, 2), oos_end=date(2025, 12, 31),
            window_days=10, stride_days=5,
            leverage_grid=[1.0], min_score_grid=[0.0],
            hold_through_grid=[True], top_n_grid=[1],
            fee_regimes=["deploy"],
            vol_target_ann_grid=[0.15],
            spy_csv_path=None,
        )


def test_sweep_vol_target_changes_results(monkeypatch, tmp_path):
    """vta=0 and vta=0.10 (aggressive shrink) must not produce bit-identical PnL."""
    _install_fakes(monkeypatch)
    paths = _fake_paths(tmp_path, 2)
    spy = _write_fake_spy_csv(tmp_path / "SPY.csv", n=200)

    cells = sweep.run_sweep(
        symbols=[f"SYM{k}" for k in range(6)],
        data_root=Path("/tmp"),
        model_paths=paths,
        train_start=date(2020, 1, 1), train_end=date(2024, 12, 31),
        oos_start=date(2025, 1, 2), oos_end=date(2025, 12, 31),
        window_days=10, stride_days=5,
        leverage_grid=[2.0], min_score_grid=[0.0],
        hold_through_grid=[False], top_n_grid=[1],
        fee_regimes=["deploy"],
        vol_target_ann_grid=[0.0, 0.05],  # 0.05 will scale most days < 1
        spy_csv_path=spy,
    )
    by_vta = {c.vol_target_ann: c for c in cells}
    assert set(by_vta) == {0.0, 0.05}
    # 0.05 ann-vol target aggressively shrinks allocation → lower median %.
    # Assert the two cells do not produce identical median monthly.
    assert by_vta[0.0].median_monthly_pct != by_vta[0.05].median_monthly_pct


# ── Per-pick inv-vol sizing axis ───────────────────────────────────────────

def _install_fakes_with_vol(monkeypatch, sym_vols: dict[str, float]) -> None:
    """Like _install_fakes but stamps vol_20d per symbol on the panel."""
    oos = _mk_panel()
    oos["vol_20d"] = oos["symbol"].map(sym_vols).astype(float)
    train = oos.copy()
    monkeypatch.setattr(
        sweep, "build_daily_dataset",
        lambda **kw: (train, train.iloc[:0], oos),
    )
    monkeypatch.setattr(
        sweep, "load_any_model",
        lambda path: _FakeModel(seed=hash(str(path)) & 0xFFFF),
    )


def test_sweep_inv_vol_axis_matches_product(monkeypatch, tmp_path):
    """inv_vol_target_grid multiplies cell count linearly."""
    sym_vols = {f"SYM{k}": 0.10 + 0.05 * k for k in range(6)}
    _install_fakes_with_vol(monkeypatch, sym_vols)
    paths = _fake_paths(tmp_path, 2)

    cells = sweep.run_sweep(
        symbols=list(sym_vols),
        data_root=Path("/tmp"),
        model_paths=paths,
        train_start=date(2020, 1, 1), train_end=date(2024, 12, 31),
        oos_start=date(2025, 1, 2), oos_end=date(2025, 12, 31),
        window_days=10, stride_days=5,
        leverage_grid=[1.0], min_score_grid=[0.0],
        hold_through_grid=[False], top_n_grid=[1],
        fee_regimes=["deploy"],
        inv_vol_target_grid=[0.0, 0.20, 0.30],
    )
    assert len(cells) == 3
    assert sorted(c.inv_vol_target_ann for c in cells) == [0.0, 0.20, 0.30]
    # Floor/cap should echo back (defaults).
    for c in cells:
        assert c.inv_vol_floor == pytest.approx(0.05)
        assert c.inv_vol_cap == pytest.approx(3.0)


def test_sweep_invert_scores_flips_picks(monkeypatch, tmp_path):
    """invert_scores=True should pick the opposite set of symbols and
    produce a different PnL when the model rank-orders the panel."""
    # Panel with a clear monotone bias in score (via _FakeModel: SYM0<<SYM5).
    sym_vols = {f"SYM{k}": 0.10 for k in range(6)}
    _install_fakes_with_vol(monkeypatch, sym_vols)
    paths = _fake_paths(tmp_path, 1)

    cells_norm = sweep.run_sweep(
        symbols=list(sym_vols),
        data_root=Path("/tmp"),
        model_paths=paths,
        train_start=date(2020, 1, 1), train_end=date(2024, 12, 31),
        oos_start=date(2025, 1, 2), oos_end=date(2025, 12, 31),
        window_days=10, stride_days=5,
        leverage_grid=[1.0], min_score_grid=[0.0],
        hold_through_grid=[False], top_n_grid=[1],
        fee_regimes=["deploy"],
        invert_scores=False,
    )
    cells_inv = sweep.run_sweep(
        symbols=list(sym_vols),
        data_root=Path("/tmp"),
        model_paths=paths,
        train_start=date(2020, 1, 1), train_end=date(2024, 12, 31),
        oos_start=date(2025, 1, 2), oos_end=date(2025, 12, 31),
        window_days=10, stride_days=5,
        leverage_grid=[1.0], min_score_grid=[0.0],
        hold_through_grid=[False], top_n_grid=[1],
        fee_regimes=["deploy"],
        invert_scores=True,
    )
    assert len(cells_norm) == 1 and len(cells_inv) == 1
    # PnL must differ (different picks → different trades).
    assert (cells_norm[0].median_monthly_pct
            != pytest.approx(cells_inv[0].median_monthly_pct, abs=0.01)), (
        f"invert_scores had no effect: {cells_norm[0]} vs {cells_inv[0]}"
    )


def test_sweep_inv_vol_changes_results(monkeypatch, tmp_path):
    """inv-vol target=0 vs 0.25 must produce different PnL when vol varies."""
    # Spread vol widely so scales push and pull across picks.
    sym_vols = {f"SYM{k}": 0.08 + 0.10 * k for k in range(6)}
    _install_fakes_with_vol(monkeypatch, sym_vols)
    paths = _fake_paths(tmp_path, 2)

    cells = sweep.run_sweep(
        symbols=list(sym_vols),
        data_root=Path("/tmp"),
        model_paths=paths,
        train_start=date(2020, 1, 1), train_end=date(2024, 12, 31),
        oos_start=date(2025, 1, 2), oos_end=date(2025, 12, 31),
        window_days=10, stride_days=5,
        leverage_grid=[2.0], min_score_grid=[0.0],
        hold_through_grid=[False], top_n_grid=[1],
        fee_regimes=["deploy"],
        inv_vol_target_grid=[0.0, 0.25],
    )
    by_ivt = {c.inv_vol_target_ann: c for c in cells}
    assert set(by_ivt) == {0.0, 0.25}
    assert by_ivt[0.0].median_monthly_pct != by_ivt[0.25].median_monthly_pct


def test_sweep_momentum_rank_axis_matches_product(monkeypatch, tmp_path):
    """max_ret_20d_rank_pct_grid × min_ret_5d_rank_pct_grid multiplies
    cell count cartesian-linearly. Defaults (1.0 / 0.0) echo through.
    """
    sym_vols = {f"SYM{k}": 0.10 + 0.05 * k for k in range(6)}
    _install_fakes_with_vol(monkeypatch, sym_vols)
    paths = _fake_paths(tmp_path, 1)

    cells = sweep.run_sweep(
        symbols=list(sym_vols),
        data_root=Path("/tmp"),
        model_paths=paths,
        train_start=date(2020, 1, 1), train_end=date(2024, 12, 31),
        oos_start=date(2025, 1, 2), oos_end=date(2025, 12, 31),
        window_days=10, stride_days=5,
        leverage_grid=[1.0], min_score_grid=[0.0],
        hold_through_grid=[False], top_n_grid=[1],
        fee_regimes=["deploy"],
        max_ret_20d_rank_pct_grid=[1.0, 0.75, 0.50],
        min_ret_5d_rank_pct_grid=[0.0, 0.25],
    )
    # 3 × 2 = 6 cells.
    assert len(cells) == 6
    pairs = {(c.max_ret_20d_rank_pct, c.min_ret_5d_rank_pct) for c in cells}
    assert pairs == {
        (1.0, 0.0), (1.0, 0.25),
        (0.75, 0.0), (0.75, 0.25),
        (0.50, 0.0), (0.50, 0.25),
    }
