"""Unit tests for scripts/eval_100d.py aggregate math + md rendering.

These tests do NOT exercise the marketsim (that's covered by the fp4 tests).
They only verify the pure-python aggregation functions used to turn the
raw ``_run_slippage_sweep`` output into a 27%/month-target report.
"""
from __future__ import annotations

import importlib.util
import hashlib
import math
from pathlib import Path
import json

import pytest

_REPO = Path(__file__).resolve().parents[1]
_MOD_PATH = _REPO / "scripts" / "eval_100d.py"

_spec = importlib.util.spec_from_file_location("_eval_100d_under_test", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
assert _spec.loader is not None
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]


def _summary_cell(
    *,
    median_return: float = 3.0,
    p10_return: float = 2.0,
    mean_return: float = 2.5,
    sortino: float = 5.0,
    max_drawdown: float = 0.05,
    n_windows: int = 4,
    n_neg: int = 0,
) -> dict[str, float | int]:
    return {
        "median_return": median_return,
        "p10_return": p10_return,
        "mean_return": mean_return,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "n_windows": n_windows,
        "n_neg": n_neg,
    }


def _good_by_slippage(*bps: int) -> dict[str, dict[str, float | int]]:
    return {str(b): _summary_cell() for b in bps}


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
               "sortino": 1.1, "max_drawdown": 0.09, "worst_max_drawdown": 0.18,
               "n_windows": 20, "n_neg": 7},
        "20": {"median_return": -0.02, "p10_return": -0.18, "mean_return": -0.01,
               "sortino": -0.4, "max_drawdown": 0.13, "n_windows": 20, "n_neg": 12},
    }
    agg = _mod._aggregate(summaries, window_days=100)
    # worst_slip_monthly should come from the "20" cell.
    worst = agg["worst_slip_monthly"]
    expected_worst = _mod._monthly_from_total(-0.02, 100)
    assert math.isclose(worst, expected_worst, rel_tol=1e-9)
    assert set(agg["by_slippage"].keys()) == {"0", "5", "20"}
    assert agg["worst_slip_bps_by_monthly"] == "20"
    assert agg["max_slip_median_max_drawdown"] == 0.13
    assert agg["max_slip_worst_max_drawdown"] == 0.18
    assert agg["worst_slip_bps_by_drawdown"] == "5"
    assert agg["max_slip_negative_windows"] == 12
    assert agg["worst_slip_bps_by_negative_windows"] == "20"
    assert agg["min_slip_n_windows"] == 20
    assert agg["worst_slip_bps_by_n_windows"] == "0"
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
    assert cell["worst_max_drawdown"] == 0.06


def test_aggregate_preserves_invalid_count_metrics_for_gate_failure():
    summaries = {
        "0": {
            "median_return": 3.0,
            "p10_return": 2.0,
            "mean_return": 2.5,
            "sortino": 5.0,
            "max_drawdown": 0.05,
            "n_windows": True,
            "n_neg": False,
        },
    }
    agg = _mod._aggregate(summaries, window_days=100)

    gate = _mod._promotion_status(
        agg,
        target_monthly=0.27,
        max_dd_target=0.25,
        window_days=100,
        min_window_days=100,
        decision_lag=2,
        min_decision_lag=2,
        slippage_bps=[0, 20],
        min_max_slippage_bps=20,
        min_completed_windows=1,
    )

    assert math.isnan(agg["max_slip_negative_windows"])
    assert math.isnan(agg["min_slip_n_windows"])
    assert agg["worst_slip_bps_by_negative_windows"] == "0"
    assert agg["worst_slip_bps_by_n_windows"] == "0"
    assert gate["passed"] is False
    assert gate["failures"] == [
        "max_slip_negative_windows is not an integer",
        "min_slip_n_windows is not an integer",
    ]


def test_summarise_window_metrics_tracks_worst_drawdown_tail():
    summary = _mod._summarise_window_metrics([
        {"total_return": 0.10, "sortino": 3.0, "max_drawdown": 0.05},
        {"total_return": 0.20, "sortino": 4.0, "max_drawdown": 0.30},
        {"total_return": 0.30, "sortino": 5.0, "max_drawdown": 0.04},
    ])

    assert summary["max_drawdown"] == 0.05
    assert summary["worst_max_drawdown"] == 0.30


def test_eval_generic_summary_tracks_worst_drawdown_tail():
    from fp4.bench.eval_generic import _summarise_windows

    summary = _summarise_windows(
        returns=[0.10, 0.20, 0.30],
        sortinos=[3.0, 4.0, 5.0],
        maxdds=[0.05, 0.30, 0.04],
    )

    assert summary["max_drawdown"] == 0.05
    assert summary["worst_max_drawdown"] == 0.30


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
        checkpoint_sha256="abc123",
        aggregate=agg,
        window_days=100,
        n_windows=20,
        target_monthly=0.27,
        max_dd_target=0.25,
        eval_result={"backend": "pufferlib_market"},
    )
    assert "PASS" in md
    assert "100d unseen-data eval" in md
    assert "checkpoint_sha256: `abc123`" in md


def test_promotion_status_fails_high_drawdown_even_when_return_clears():
    summaries = {
        "0": {"median_return": 3.0, "p10_return": 2.0, "mean_return": 2.5,
              "sortino": 5.0, "max_drawdown": 0.30, "n_windows": 20, "n_neg": 0},
        "5": {"median_return": 2.6, "p10_return": 1.7, "mean_return": 2.2,
              "sortino": 4.8, "max_drawdown": 0.10, "n_windows": 20, "n_neg": 0},
    }
    agg = _mod._aggregate(summaries, window_days=100)

    gate = _mod._promotion_status(agg, target_monthly=0.27, max_dd_target=0.25)
    md = _mod._render_md(
        ckpt=Path("fake/best.pt"),
        checkpoint_sha256="abc123",
        aggregate=agg,
        window_days=100,
        n_windows=20,
        target_monthly=0.27,
        max_dd_target=0.25,
        eval_result={"backend": "pufferlib_market"},
    )

    assert gate["passed"] is False
    assert "max_slip_worst_max_drawdown" in gate["failures"][0]
    assert "FAIL" in md
    assert "max DD 30.00% vs cap 25.00%" in md


def test_promotion_status_fails_short_window_even_when_metrics_clear():
    summaries = {
        "0": {"median_return": 3.0, "p10_return": 2.0, "mean_return": 2.5,
              "sortino": 5.0, "max_drawdown": 0.05, "n_windows": 20, "n_neg": 0},
    }
    agg = _mod._aggregate(summaries, window_days=30)

    gate = _mod._promotion_status(
        agg,
        target_monthly=0.27,
        max_dd_target=0.25,
        window_days=30,
        min_window_days=100,
    )

    assert gate["passed"] is False
    assert gate["failures"] == ["window_days 30 < 100"]


def test_promotion_status_fails_low_decision_lag_even_when_metrics_clear():
    summaries = {
        "0": {"median_return": 3.0, "p10_return": 2.0, "mean_return": 2.5,
              "sortino": 5.0, "max_drawdown": 0.05, "n_windows": 20, "n_neg": 0},
    }
    agg = _mod._aggregate(summaries, window_days=100)

    gate = _mod._promotion_status(
        agg,
        target_monthly=0.27,
        max_dd_target=0.25,
        window_days=100,
        min_window_days=100,
        decision_lag=1,
        min_decision_lag=2,
    )

    assert gate["passed"] is False
    assert gate["failures"] == ["decision_lag 1 < 2"]


def test_promotion_status_fails_low_slippage_coverage_even_when_metrics_clear():
    summaries = {
        "0": {"median_return": 3.0, "p10_return": 2.0, "mean_return": 2.5,
              "sortino": 5.0, "max_drawdown": 0.05, "n_windows": 20, "n_neg": 0},
    }
    agg = _mod._aggregate(summaries, window_days=100)

    gate = _mod._promotion_status(
        agg,
        target_monthly=0.27,
        max_dd_target=0.25,
        window_days=100,
        min_window_days=100,
        decision_lag=2,
        min_decision_lag=2,
        slippage_bps=[0, 5],
        min_max_slippage_bps=20,
    )

    assert gate["passed"] is False
    assert gate["failures"] == ["max_slippage_bps 5 < 20"]


def test_promotion_status_fails_negative_windows_even_when_return_and_dd_clear():
    summaries = {
        "0": {"median_return": 3.0, "p10_return": 2.0, "mean_return": 2.5,
              "sortino": 5.0, "max_drawdown": 0.05, "n_windows": 20, "n_neg": 0},
        "20": {"median_return": 2.8, "p10_return": 2.0, "mean_return": 2.5,
               "sortino": 5.0, "max_drawdown": 0.05, "n_windows": 20, "n_neg": 1},
    }
    agg = _mod._aggregate(summaries, window_days=100)

    gate = _mod._promotion_status(
        agg,
        target_monthly=0.27,
        max_dd_target=0.25,
        window_days=100,
        min_window_days=100,
        decision_lag=2,
        min_decision_lag=2,
        slippage_bps=[0, 20],
        min_max_slippage_bps=20,
    )

    assert gate["passed"] is False
    assert gate["negative_windows"] == 1
    assert gate["max_negative_windows"] == 0
    assert gate["failures"] == ["negative_windows 1 > 0"]


def test_promotion_status_fails_incomplete_windows_even_when_metrics_clear():
    summaries = {
        "0": {"median_return": 3.0, "p10_return": 2.0, "mean_return": 2.5,
              "sortino": 5.0, "max_drawdown": 0.05, "n_windows": 20, "n_neg": 0},
        "20": {"median_return": 2.8, "p10_return": 2.0, "mean_return": 2.5,
               "sortino": 5.0, "max_drawdown": 0.05, "n_windows": 3, "n_neg": 0},
    }
    agg = _mod._aggregate(summaries, window_days=100)

    gate = _mod._promotion_status(
        agg,
        target_monthly=0.27,
        max_dd_target=0.25,
        window_days=100,
        min_window_days=100,
        decision_lag=2,
        min_decision_lag=2,
        slippage_bps=[0, 20],
        min_max_slippage_bps=20,
        min_completed_windows=20,
    )

    assert gate["passed"] is False
    assert gate["completed_windows"] == 3
    assert gate["min_completed_windows"] == 20
    assert gate["failures"] == ["completed_windows 3 < 20"]


def test_promotion_status_rejects_bool_window_counts():
    aggregate = _mod._aggregate(_good_by_slippage(0, 20), window_days=100)
    aggregate["max_slip_negative_windows"] = False
    aggregate["min_slip_n_windows"] = True

    gate = _mod._promotion_status(
        aggregate,
        target_monthly=0.27,
        max_dd_target=0.25,
        window_days=100,
        min_window_days=100,
        decision_lag=2,
        min_decision_lag=2,
        slippage_bps=[0, 20],
        min_max_slippage_bps=20,
        min_completed_windows=4,
    )

    assert gate["passed"] is False
    assert gate["failures"] == [
        "max_slip_negative_windows is not an integer",
        "min_slip_n_windows is not an integer",
    ]


def test_promotion_status_fails_nonfinite_required_metric():
    aggregate = _mod._aggregate(_good_by_slippage(0, 20), window_days=100)
    aggregate["worst_slip_monthly"] = float("nan")

    gate = _mod._promotion_status(
        aggregate,
        target_monthly=0.27,
        max_dd_target=0.25,
        window_days=100,
        min_window_days=100,
        decision_lag=2,
        min_decision_lag=2,
        slippage_bps=[0, 20],
        min_max_slippage_bps=20,
        min_completed_windows=4,
    )

    assert gate["passed"] is False
    assert gate["failures"] == ["worst_slip_monthly is not finite"]


def test_promotion_status_can_disable_negative_window_gate_for_smoke():
    summaries = {
        "0": {"median_return": 3.0, "p10_return": 2.0, "mean_return": 2.5,
              "sortino": 5.0, "max_drawdown": 0.05, "n_windows": 20, "n_neg": 1},
        "20": {"median_return": 2.8, "p10_return": 2.0, "mean_return": 2.5,
               "sortino": 5.0, "max_drawdown": 0.05, "n_windows": 20, "n_neg": 1},
    }
    agg = _mod._aggregate(summaries, window_days=100)

    gate = _mod._promotion_status(
        agg,
        target_monthly=0.27,
        max_dd_target=0.25,
        window_days=100,
        min_window_days=100,
        decision_lag=2,
        min_decision_lag=2,
        slippage_bps=[0, 20],
        min_max_slippage_bps=20,
        max_negative_windows=-1,
    )

    assert gate["passed"] is True
    assert gate["negative_windows"] == 1
    assert gate["max_negative_windows"] == -1


def test_evaluated_slippage_bps_uses_returned_cells_not_requested_grid():
    agg = _mod._aggregate(_good_by_slippage(0), window_days=100)

    gate = _mod._promotion_status(
        agg,
        target_monthly=0.27,
        max_dd_target=0.25,
        window_days=100,
        min_window_days=100,
        decision_lag=2,
        min_decision_lag=2,
        slippage_bps=_mod._evaluated_slippage_bps(agg),
        min_max_slippage_bps=20,
    )

    assert gate["passed"] is False
    assert gate["failures"] == ["max_slippage_bps 0 < 20"]


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
        checkpoint_sha256="abc123",
        aggregate=agg,
        window_days=100,
        n_windows=20,
        target_monthly=0.27,
        max_dd_target=0.25,
        eval_result={"backend": "pufferlib_market"},
    )
    assert "FAIL" in md


def test_main_routes_hourly_intrabar_and_writes_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    ckpt = tmp_path / "best.pt"
    val = tmp_path / "val.bin"
    hourly_root = tmp_path / "hourly"
    out_dir = tmp_path / "out"
    ckpt.write_bytes(b"x")
    val.write_bytes(b"x")
    hourly_root.mkdir()

    seen = {}

    def _fake_eval(**kwargs):
        seen.update(kwargs)
        return {
            "status": "ok",
            "backend": "pufferlib_market_intrabar_hourly",
            "by_slippage": _good_by_slippage(0, 20),
        }

    monkeypatch.setattr(_mod, "_evaluate_intrabar_hourly", _fake_eval)

    rc = _mod.main([
        "--checkpoint", str(ckpt),
        "--val-data", str(val),
        "--execution-granularity", "hourly_intrabar",
        "--hourly-data-root", str(hourly_root),
        "--daily-start-date", "2026-01-01",
        "--n-windows", "4",
        "--window-days", "100",
        "--out-dir", str(out_dir),
    ])

    assert rc == 0
    assert seen["daily_start_date"] == "2026-01-01"
    assert Path(seen["hourly_data_root"]) == hourly_root.resolve()

    payload = json.loads((out_dir / "best_eval100d.json").read_text())
    md = (out_dir / "best_eval100d.md").read_text()
    assert payload["raw"]["backend"] == "pufferlib_market_intrabar_hourly"
    assert payload["checkpoint_sha256"] == hashlib.sha256(b"x").hexdigest()
    assert payload["max_dd_target"] == 0.25
    assert payload["min_window_days"] == 100
    assert payload["decision_lag"] == 2
    assert payload["min_decision_lag"] == 2
    assert payload["min_max_slippage_bps"] == 20
    assert payload["promotion_gate"]["passed"] is True
    assert payload["promotion_gate"]["max_slippage_bps"] == 20
    assert payload["promotion_gate"]["completed_windows"] == 4
    assert payload["promotion_gate"]["min_completed_windows"] == 4
    assert seen["decision_lag"] == 2
    assert "decision_lag: 2" in md
    assert f"checkpoint_sha256: `{hashlib.sha256(b'x').hexdigest()}`" in md
    assert "slippage_bps: 0,5,10,20" in md
    assert "completed_windows: 4 (min 4)" in md
    assert "pufferlib_market_intrabar_hourly" in md


def test_main_rejects_high_drawdown_even_when_return_target_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    ckpt = tmp_path / "best.pt"
    val = tmp_path / "val.bin"
    hourly_root = tmp_path / "hourly"
    out_dir = tmp_path / "out"
    ckpt.write_bytes(b"x")
    val.write_bytes(b"x")
    hourly_root.mkdir()

    def _fake_eval(**_kwargs):
        return {
            "status": "ok",
            "backend": "pufferlib_market_intrabar_hourly",
            "by_slippage": {
                "0": _summary_cell(max_drawdown=0.30),
                "20": _summary_cell(max_drawdown=0.30),
            },
        }

    monkeypatch.setattr(_mod, "_evaluate_intrabar_hourly", _fake_eval)

    rc = _mod.main([
        "--checkpoint", str(ckpt),
        "--val-data", str(val),
        "--execution-granularity", "hourly_intrabar",
        "--hourly-data-root", str(hourly_root),
        "--daily-start-date", "2026-01-01",
        "--n-windows", "4",
        "--window-days", "100",
        "--out-dir", str(out_dir),
    ])

    assert rc == 3
    payload = json.loads((out_dir / "best_eval100d.json").read_text())
    md = (out_dir / "best_eval100d.md").read_text()
    assert payload["promotion_gate"]["passed"] is False
    assert payload["promotion_gate"]["failures"] == [
        "max_slip_worst_max_drawdown 30.00% > 25.00%"
    ]
    assert "FAIL" in md


def test_main_rejects_negative_windows_even_when_return_and_dd_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    ckpt = tmp_path / "best.pt"
    val = tmp_path / "val.bin"
    hourly_root = tmp_path / "hourly"
    out_dir = tmp_path / "out"
    ckpt.write_bytes(b"x")
    val.write_bytes(b"x")
    hourly_root.mkdir()

    def _fake_eval(**_kwargs):
        return {
            "status": "ok",
            "backend": "pufferlib_market_intrabar_hourly",
            "by_slippage": {
                "0": _summary_cell(n_neg=0),
                "20": _summary_cell(median_return=2.8, n_neg=1),
            },
        }

    monkeypatch.setattr(_mod, "_evaluate_intrabar_hourly", _fake_eval)

    rc = _mod.main([
        "--checkpoint", str(ckpt),
        "--val-data", str(val),
        "--execution-granularity", "hourly_intrabar",
        "--hourly-data-root", str(hourly_root),
        "--daily-start-date", "2026-01-01",
        "--n-windows", "4",
        "--window-days", "100",
        "--out-dir", str(out_dir),
    ])

    assert rc == 3
    payload = json.loads((out_dir / "best_eval100d.json").read_text())
    md = (out_dir / "best_eval100d.md").read_text()
    assert payload["max_negative_windows"] == 0
    assert payload["promotion_gate"]["negative_windows"] == 1
    assert payload["promotion_gate"]["failures"] == ["negative_windows 1 > 0"]
    assert "negative_windows: 1 (cap 0)" in md


def test_main_allows_explicit_negative_window_override_for_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    ckpt = tmp_path / "best.pt"
    val = tmp_path / "val.bin"
    hourly_root = tmp_path / "hourly"
    out_dir = tmp_path / "out"
    ckpt.write_bytes(b"x")
    val.write_bytes(b"x")
    hourly_root.mkdir()

    def _fake_eval(**_kwargs):
        return {
            "status": "ok",
            "backend": "pufferlib_market_intrabar_hourly",
            "by_slippage": {
                "0": _summary_cell(n_neg=1),
                "20": _summary_cell(median_return=2.8, n_neg=1),
            },
        }

    monkeypatch.setattr(_mod, "_evaluate_intrabar_hourly", _fake_eval)

    rc = _mod.main([
        "--checkpoint", str(ckpt),
        "--val-data", str(val),
        "--execution-granularity", "hourly_intrabar",
        "--hourly-data-root", str(hourly_root),
        "--daily-start-date", "2026-01-01",
        "--n-windows", "4",
        "--window-days", "100",
        "--max-negative-windows", "-1",
        "--out-dir", str(out_dir),
    ])

    assert rc == 0
    payload = json.loads((out_dir / "best_eval100d.json").read_text())
    assert payload["promotion_gate"]["passed"] is True
    assert payload["promotion_gate"]["negative_windows"] == 1
    assert payload["promotion_gate"]["max_negative_windows"] == -1


def test_main_rejects_incomplete_returned_windows_even_when_metrics_clear(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    ckpt = tmp_path / "best.pt"
    val = tmp_path / "val.bin"
    hourly_root = tmp_path / "hourly"
    out_dir = tmp_path / "out"
    ckpt.write_bytes(b"x")
    val.write_bytes(b"x")
    hourly_root.mkdir()

    def _fake_eval(**_kwargs):
        return {
            "status": "ok",
            "backend": "pufferlib_market_intrabar_hourly",
            "by_slippage": {
                "0": _summary_cell(n_windows=4),
                "20": _summary_cell(median_return=2.8, n_windows=3),
            },
        }

    monkeypatch.setattr(_mod, "_evaluate_intrabar_hourly", _fake_eval)

    rc = _mod.main([
        "--checkpoint", str(ckpt),
        "--val-data", str(val),
        "--execution-granularity", "hourly_intrabar",
        "--hourly-data-root", str(hourly_root),
        "--daily-start-date", "2026-01-01",
        "--n-windows", "4",
        "--window-days", "100",
        "--out-dir", str(out_dir),
    ])

    assert rc == 3
    payload = json.loads((out_dir / "best_eval100d.json").read_text())
    md = (out_dir / "best_eval100d.md").read_text()
    assert payload["min_completed_windows"] == 4
    assert payload["promotion_gate"]["completed_windows"] == 3
    assert payload["promotion_gate"]["failures"] == ["completed_windows 3 < 4"]
    assert "completed_windows: 3 (min 4)" in md


def test_main_allows_explicit_lower_min_completed_windows_for_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    ckpt = tmp_path / "best.pt"
    val = tmp_path / "val.bin"
    hourly_root = tmp_path / "hourly"
    out_dir = tmp_path / "out"
    ckpt.write_bytes(b"x")
    val.write_bytes(b"x")
    hourly_root.mkdir()

    def _fake_eval(**_kwargs):
        return {
            "status": "ok",
            "backend": "pufferlib_market_intrabar_hourly",
            "by_slippage": {
                "0": _summary_cell(n_windows=4),
                "20": _summary_cell(median_return=2.8, n_windows=3),
            },
        }

    monkeypatch.setattr(_mod, "_evaluate_intrabar_hourly", _fake_eval)

    rc = _mod.main([
        "--checkpoint", str(ckpt),
        "--val-data", str(val),
        "--execution-granularity", "hourly_intrabar",
        "--hourly-data-root", str(hourly_root),
        "--daily-start-date", "2026-01-01",
        "--n-windows", "4",
        "--window-days", "100",
        "--min-completed-windows", "3",
        "--out-dir", str(out_dir),
    ])

    assert rc == 0
    payload = json.loads((out_dir / "best_eval100d.json").read_text())
    assert payload["min_completed_windows"] == 3
    assert payload["promotion_gate"]["passed"] is True
    assert payload["promotion_gate"]["completed_windows"] == 3
    assert payload["promotion_gate"]["min_completed_windows"] == 3


def test_main_rejects_short_window_even_when_metrics_clear(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    ckpt = tmp_path / "best.pt"
    val = tmp_path / "val.bin"
    hourly_root = tmp_path / "hourly"
    out_dir = tmp_path / "out"
    ckpt.write_bytes(b"x")
    val.write_bytes(b"x")
    hourly_root.mkdir()

    def _fake_eval(**_kwargs):
        return {
            "status": "ok",
            "backend": "pufferlib_market_intrabar_hourly",
            "by_slippage": {
                "0": _summary_cell(median_return=0.50, p10_return=0.40, mean_return=0.45),
                "20": _summary_cell(median_return=0.50, p10_return=0.40, mean_return=0.45),
            },
        }

    monkeypatch.setattr(_mod, "_evaluate_intrabar_hourly", _fake_eval)

    rc = _mod.main([
        "--checkpoint", str(ckpt),
        "--val-data", str(val),
        "--execution-granularity", "hourly_intrabar",
        "--hourly-data-root", str(hourly_root),
        "--daily-start-date", "2026-01-01",
        "--n-windows", "4",
        "--window-days", "30",
        "--out-dir", str(out_dir),
    ])

    assert rc == 3
    payload = json.loads((out_dir / "best_eval100d.json").read_text())
    assert payload["window_days"] == 30
    assert payload["min_window_days"] == 100
    assert payload["promotion_gate"]["failures"] == ["window_days 30 < 100"]


def test_main_allows_explicit_lower_min_window_for_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    ckpt = tmp_path / "best.pt"
    val = tmp_path / "val.bin"
    hourly_root = tmp_path / "hourly"
    out_dir = tmp_path / "out"
    ckpt.write_bytes(b"x")
    val.write_bytes(b"x")
    hourly_root.mkdir()

    def _fake_eval(**_kwargs):
        return {
            "status": "ok",
            "backend": "pufferlib_market_intrabar_hourly",
            "by_slippage": {
                "0": _summary_cell(median_return=0.50, p10_return=0.40, mean_return=0.45),
                "20": _summary_cell(median_return=0.50, p10_return=0.40, mean_return=0.45),
            },
        }

    monkeypatch.setattr(_mod, "_evaluate_intrabar_hourly", _fake_eval)

    rc = _mod.main([
        "--checkpoint", str(ckpt),
        "--val-data", str(val),
        "--execution-granularity", "hourly_intrabar",
        "--hourly-data-root", str(hourly_root),
        "--daily-start-date", "2026-01-01",
        "--n-windows", "4",
        "--window-days", "30",
        "--min-window-days", "30",
        "--out-dir", str(out_dir),
    ])

    assert rc == 0
    payload = json.loads((out_dir / "best_eval100d.json").read_text())
    assert payload["promotion_gate"]["passed"] is True
    assert payload["promotion_gate"]["min_window_days"] == 30


def test_main_rejects_low_decision_lag_even_when_metrics_clear(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    ckpt = tmp_path / "best.pt"
    val = tmp_path / "val.bin"
    hourly_root = tmp_path / "hourly"
    out_dir = tmp_path / "out"
    ckpt.write_bytes(b"x")
    val.write_bytes(b"x")
    hourly_root.mkdir()

    def _fake_eval(**_kwargs):
        return {
            "status": "ok",
            "backend": "pufferlib_market_intrabar_hourly",
            "by_slippage": _good_by_slippage(0, 20),
        }

    monkeypatch.setattr(_mod, "_evaluate_intrabar_hourly", _fake_eval)

    rc = _mod.main([
        "--checkpoint", str(ckpt),
        "--val-data", str(val),
        "--execution-granularity", "hourly_intrabar",
        "--hourly-data-root", str(hourly_root),
        "--daily-start-date", "2026-01-01",
        "--n-windows", "4",
        "--window-days", "100",
        "--decision-lag", "1",
        "--out-dir", str(out_dir),
    ])

    assert rc == 3
    payload = json.loads((out_dir / "best_eval100d.json").read_text())
    assert payload["decision_lag"] == 1
    assert payload["min_decision_lag"] == 2
    assert payload["promotion_gate"]["failures"] == ["decision_lag 1 < 2"]


def test_main_allows_explicit_lower_min_decision_lag_for_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    ckpt = tmp_path / "best.pt"
    val = tmp_path / "val.bin"
    hourly_root = tmp_path / "hourly"
    out_dir = tmp_path / "out"
    ckpt.write_bytes(b"x")
    val.write_bytes(b"x")
    hourly_root.mkdir()

    def _fake_eval(**_kwargs):
        return {
            "status": "ok",
            "backend": "pufferlib_market_intrabar_hourly",
            "by_slippage": _good_by_slippage(0, 20),
        }

    monkeypatch.setattr(_mod, "_evaluate_intrabar_hourly", _fake_eval)

    rc = _mod.main([
        "--checkpoint", str(ckpt),
        "--val-data", str(val),
        "--execution-granularity", "hourly_intrabar",
        "--hourly-data-root", str(hourly_root),
        "--daily-start-date", "2026-01-01",
        "--n-windows", "4",
        "--window-days", "100",
        "--decision-lag", "1",
        "--min-decision-lag", "1",
        "--out-dir", str(out_dir),
    ])

    assert rc == 0
    payload = json.loads((out_dir / "best_eval100d.json").read_text())
    assert payload["promotion_gate"]["passed"] is True
    assert payload["promotion_gate"]["decision_lag"] == 1
    assert payload["promotion_gate"]["min_decision_lag"] == 1


def test_main_rejects_low_slippage_coverage_even_when_metrics_clear(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    ckpt = tmp_path / "best.pt"
    val = tmp_path / "val.bin"
    hourly_root = tmp_path / "hourly"
    out_dir = tmp_path / "out"
    ckpt.write_bytes(b"x")
    val.write_bytes(b"x")
    hourly_root.mkdir()

    def _fake_eval(**_kwargs):
        return {
            "status": "ok",
            "backend": "pufferlib_market_intrabar_hourly",
            "by_slippage": _good_by_slippage(0, 5),
        }

    monkeypatch.setattr(_mod, "_evaluate_intrabar_hourly", _fake_eval)

    rc = _mod.main([
        "--checkpoint", str(ckpt),
        "--val-data", str(val),
        "--execution-granularity", "hourly_intrabar",
        "--hourly-data-root", str(hourly_root),
        "--daily-start-date", "2026-01-01",
        "--n-windows", "4",
        "--window-days", "100",
        "--slippage-bps", "0,5",
        "--out-dir", str(out_dir),
    ])

    assert rc == 3
    payload = json.loads((out_dir / "best_eval100d.json").read_text())
    assert payload["slippage_bps"] == [0, 5]
    assert payload["min_max_slippage_bps"] == 20
    assert payload["promotion_gate"]["failures"] == ["max_slippage_bps 5 < 20"]


def test_main_allows_explicit_lower_min_slippage_for_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    ckpt = tmp_path / "best.pt"
    val = tmp_path / "val.bin"
    hourly_root = tmp_path / "hourly"
    out_dir = tmp_path / "out"
    ckpt.write_bytes(b"x")
    val.write_bytes(b"x")
    hourly_root.mkdir()

    def _fake_eval(**_kwargs):
        return {
            "status": "ok",
            "backend": "pufferlib_market_intrabar_hourly",
            "by_slippage": _good_by_slippage(0, 5),
        }

    monkeypatch.setattr(_mod, "_evaluate_intrabar_hourly", _fake_eval)

    rc = _mod.main([
        "--checkpoint", str(ckpt),
        "--val-data", str(val),
        "--execution-granularity", "hourly_intrabar",
        "--hourly-data-root", str(hourly_root),
        "--daily-start-date", "2026-01-01",
        "--n-windows", "4",
        "--window-days", "100",
        "--slippage-bps", "0,5",
        "--min-max-slippage-bps", "5",
        "--out-dir", str(out_dir),
    ])

    assert rc == 0
    payload = json.loads((out_dir / "best_eval100d.json").read_text())
    assert payload["promotion_gate"]["passed"] is True
    assert payload["promotion_gate"]["max_slippage_bps"] == 5
    assert payload["promotion_gate"]["min_max_slippage_bps"] == 5


def test_main_rejects_when_evaluator_returns_less_slippage_than_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    ckpt = tmp_path / "best.pt"
    val = tmp_path / "val.bin"
    hourly_root = tmp_path / "hourly"
    out_dir = tmp_path / "out"
    ckpt.write_bytes(b"x")
    val.write_bytes(b"x")
    hourly_root.mkdir()

    def _fake_eval(**_kwargs):
        return {
            "status": "ok",
            "backend": "same_as_train",
            "by_slippage": _good_by_slippage(0),
        }

    monkeypatch.setattr(_mod, "_evaluate_intrabar_hourly", _fake_eval)

    rc = _mod.main([
        "--checkpoint", str(ckpt),
        "--val-data", str(val),
        "--execution-granularity", "hourly_intrabar",
        "--hourly-data-root", str(hourly_root),
        "--daily-start-date", "2026-01-01",
        "--n-windows", "4",
        "--window-days", "100",
        "--out-dir", str(out_dir),
    ])

    assert rc == 3
    payload = json.loads((out_dir / "best_eval100d.json").read_text())
    assert payload["slippage_bps"] == [0, 5, 10, 20]
    assert payload["promotion_gate"]["max_slippage_bps"] == 0
    assert payload["promotion_gate"]["failures"] == ["max_slippage_bps 0 < 20"]


def test_main_failed_fast_artifact_has_promotion_gate_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    ckpt = tmp_path / "best.pt"
    val = tmp_path / "val.bin"
    hourly_root = tmp_path / "hourly"
    out_dir = tmp_path / "out"
    ckpt.write_bytes(b"x")
    val.write_bytes(b"x")
    hourly_root.mkdir()

    def _fake_eval(**_kwargs):
        return {
            "status": "failed_fast",
            "backend": "pufferlib_market_intrabar_hourly",
            "failed_reason": "window 0 max_drawdown=0.300 > 0.200",
            "by_slippage": {
                "0": {
                    **_summary_cell(median_return=-0.20, p10_return=-0.20, mean_return=-0.20,
                                    max_drawdown=0.30),
                    "failed_fast": True,
                    "failed_reason": "window 0 max_drawdown=0.300 > 0.200",
                },
            },
        }

    monkeypatch.setattr(_mod, "_evaluate_intrabar_hourly", _fake_eval)

    rc = _mod.main([
        "--checkpoint", str(ckpt),
        "--val-data", str(val),
        "--execution-granularity", "hourly_intrabar",
        "--hourly-data-root", str(hourly_root),
        "--daily-start-date", "2026-01-01",
        "--n-windows", "4",
        "--window-days", "100",
        "--out-dir", str(out_dir),
    ])

    assert rc == 3
    payload = json.loads((out_dir / "best_eval100d.json").read_text())
    md = (out_dir / "best_eval100d.md").read_text()
    assert payload["checkpoint_sha256"] == hashlib.sha256(b"x").hexdigest()
    assert "aggregate" in payload
    assert payload["promotion_gate"]["passed"] is False
    assert payload["promotion_gate"]["max_slippage_bps"] == 0
    assert "failed_fast: window 0 max_drawdown=0.300 > 0.200" in payload["promotion_gate"]["failures"]
    assert f"checkpoint_sha256: `{hashlib.sha256(b'x').hexdigest()}`" in md
    assert "promotion_gate: FAIL" in md


def test_main_non_ok_artifact_records_checkpoint_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    ckpt = tmp_path / "best.pt"
    val = tmp_path / "val.bin"
    hourly_root = tmp_path / "hourly"
    out_dir = tmp_path / "out"
    ckpt.write_bytes(b"x")
    val.write_bytes(b"x")
    hourly_root.mkdir()

    monkeypatch.setattr(
        _mod,
        "_evaluate_intrabar_hourly",
        lambda **_kwargs: {"status": "skip", "reason": "not enough data"},
    )

    rc = _mod.main([
        "--checkpoint", str(ckpt),
        "--val-data", str(val),
        "--execution-granularity", "hourly_intrabar",
        "--hourly-data-root", str(hourly_root),
        "--daily-start-date", "2026-01-01",
        "--out-dir", str(out_dir),
    ])

    assert rc == 1
    payload = json.loads((out_dir / "best_eval100d.json").read_text())
    assert payload["checkpoint"] == str(ckpt.resolve())
    assert payload["checkpoint_sha256"] == hashlib.sha256(b"x").hexdigest()
    assert payload["raw"] == {"status": "skip", "reason": "not enough data"}
