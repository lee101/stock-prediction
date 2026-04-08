"""Unit tests for scripts/eval_100d.py aggregate math + md rendering.

These tests do NOT exercise the marketsim (that's covered by the fp4 tests).
They only verify the pure-python aggregation functions used to turn the
raw ``_run_slippage_sweep`` output into a 27%/month-target report.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path


_REPO = Path(__file__).resolve().parents[1]
_MOD_PATH = _REPO / "scripts" / "eval_100d.py"

_spec = importlib.util.spec_from_file_location("_eval_100d_under_test", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
assert _spec.loader is not None
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]


def test_monthly_from_total_21day_window_matches_explicit_formula():
    # A 21-day window with total return = 0.05 should give ~5% monthly.
    assert math.isclose(
        _mod._monthly_from_total(0.05, 21),
        (1.05) ** (21 / 21) - 1,
        rel_tol=1e-9,
    )


def test_monthly_from_total_100day_window_27pct_month():
    # A 100-day total return that would imply 27%/month: (1.27) ** (100/21) - 1.
    target = (1.27) ** (100 / 21) - 1
    got = _mod._monthly_from_total(target, 100)
    assert math.isclose(got, 0.27, rel_tol=1e-6), f"expected ~0.27, got {got}"


def test_monthly_from_total_zero_and_neg_window():
    assert _mod._monthly_from_total(0.10, 0) == 0.0
    assert _mod._monthly_from_total(0.10, -5) == 0.0


def test_aggregate_picks_worst_slip():
    summaries = {
        "0":  {"median_return": 0.12, "p10_return": -0.05, "mean_return": 0.09,
               "sortino": 3.0, "max_drawdown": 0.07, "n_windows": 20, "n_neg": 4},
        "5":  {"median_return": 0.05, "p10_return": -0.10, "mean_return": 0.03,
               "sortino": 1.1, "max_drawdown": 0.09, "n_windows": 20, "n_neg": 7},
        "20": {"median_return": -0.02, "p10_return": -0.18, "mean_return": -0.01,
               "sortino": -0.4, "max_drawdown": 0.13, "n_windows": 20, "n_neg": 12},
    }
    agg = _mod._aggregate(summaries, window_days=100)
    # worst_slip_monthly should come from the "20" cell.
    worst = agg["worst_slip_monthly"]
    expected_worst = _mod._monthly_from_total(-0.02, 100)
    assert math.isclose(worst, expected_worst, rel_tol=1e-9)
    assert set(agg["by_slippage"].keys()) == {"0", "5", "20"}
    # Each cell must carry both total and monthly variants.
    for cell in agg["by_slippage"].values():
        assert "median_total_return" in cell
        assert "median_monthly_return" in cell
        assert "p10_monthly_return" in cell


def test_aggregate_handles_summary_subkey():
    # _run_slippage_sweep emits cell[bps] = <summary dict> directly,
    # but _same_backend_eval nests it under a "summary" key. _aggregate
    # should handle both shapes without raising.
    summaries = {
        "0": {"summary": {"median_return": 0.20, "p10_return": 0.05, "mean_return": 0.18,
                          "sortino": 2.5, "max_drawdown": 0.06, "n_windows": 10, "n_neg": 1}},
    }
    agg = _mod._aggregate(summaries, window_days=100)
    cell = agg["by_slippage"]["0"]
    assert cell["median_total_return"] == 0.20
    assert cell["median_monthly_return"] > 0.0


def test_render_md_reports_pass_when_worst_slip_beats_target():
    # A 100d window returning 200% => (1+2.0) ** (21/100) - 1 = 25.7% monthly,
    # which is under the 27% target. Use 260% so worst slip clears it.
    summaries = {
        "0": {"median_return": 3.0, "p10_return": 2.0, "mean_return": 2.5,
              "sortino": 5.0, "max_drawdown": 0.05, "n_windows": 20, "n_neg": 0},
        "5": {"median_return": 2.6, "p10_return": 1.7, "mean_return": 2.2,
              "sortino": 4.8, "max_drawdown": 0.06, "n_windows": 20, "n_neg": 0},
    }
    agg = _mod._aggregate(summaries, window_days=100)
    md = _mod._render_md(
        ckpt=Path("fake/best.pt"),
        aggregate=agg,
        window_days=100,
        n_windows=20,
        target_monthly=0.27,
        eval_result={"backend": "pufferlib_market"},
    )
    assert "PASS" in md
    assert "100d unseen-data eval" in md


def test_aggregate_carries_failed_fast_marker_through():
    # _run_slippage_sweep marks the bailed cell with failed_fast=True; the
    # aggregate function should still produce valid numbers (we treat the
    # failed-fast cell like a normal one — its absurd values will simply
    # show up in the markdown so the operator can see why it bailed).
    summaries = {
        "0": {"median_return": -0.10, "p10_return": -0.30, "mean_return": -0.20,
              "sortino": -2.0, "max_drawdown": 0.25, "n_windows": 3, "n_neg": 3,
              "failed_fast": True, "failed_reason": "max_dd > 0.20"},
    }
    agg = _mod._aggregate(summaries, window_days=100)
    assert agg["worst_slip_monthly"] < 0.0


def test_render_md_reports_fail_when_worst_slip_under_target():
    summaries = {
        "0": {"median_return": 0.05, "p10_return": -0.02, "mean_return": 0.03,
              "sortino": 0.5, "max_drawdown": 0.08, "n_windows": 20, "n_neg": 6},
    }
    agg = _mod._aggregate(summaries, window_days=100)
    md = _mod._render_md(
        ckpt=Path("fake/best.pt"),
        aggregate=agg,
        window_days=100,
        n_windows=20,
        target_monthly=0.27,
        eval_result={"backend": "pufferlib_market"},
    )
    assert "FAIL" in md
