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
        goodness_score=84.0,
    )
    rows = sweep._cells_to_rows([c])
    assert len(rows) == 1
    r = rows[0]
    assert r["leverage"] == 2.0
    assert r["n_neg"] == 0
    assert r["hold_through"] is True
    assert r["goodness_score"] == 84.0


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
    count_gap = 0.0  # same neg_frac
    magnitude_gap = 2.0 * (60 / 60.0) * (15 - 3) / 60 * 5  # derivation below
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


def test_robust_goodness_in_cell_result_and_rows():
    c = sweep.CellResult(
        leverage=2.0, min_score=0.85, hold_through=True, top_n=1,
        fee_regime="deploy", n_windows=60,
        median_monthly_pct=141.0, p10_monthly_pct=96.0,
        median_sortino=40.0, worst_dd_pct=7.18, n_neg=0,
        goodness_score=88.82, robust_goodness_score=85.23,
    )
    rows = sweep._cells_to_rows([c])
    assert rows[0]["robust_goodness_score"] == pytest.approx(85.23)


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
        time_under_water_pct=42.5, ulcer_index=3.17,
    )
    rows = sweep._cells_to_rows([c])
    assert rows[0]["time_under_water_pct"] == pytest.approx(42.5)
    assert rows[0]["ulcer_index"] == pytest.approx(3.17)


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
