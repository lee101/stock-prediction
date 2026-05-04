"""Unit tests for scripts/eval_100d.py aggregate math + md rendering.

These tests do NOT exercise the marketsim (that's covered by the fp4 tests).
They only verify the pure-python aggregation functions used to turn the
raw ``_run_slippage_sweep`` output into a 27%/month-target report.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path

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


def test_total_from_monthly_inverts_monthly_from_total():
    total = _mod._total_from_monthly(0.27, 100)
    assert math.isclose(_mod._monthly_from_total(total, 100), 0.27, rel_tol=1e-9)


def test_module_usage_documents_promotable_hourly_eval_command():
    doc = _mod.__doc__ or ""

    assert "--execution-granularity hourly_intrabar" in doc
    assert "--hourly-data-root" in doc
    assert "--daily-start-date" in doc
    assert "--allow-daily-promotion" in doc
    assert "retained for smoke/legacy inspection" in doc


def test_eval100d_artifacts_use_shared_atomic_writers():
    source = _MOD_PATH.read_text(encoding="utf-8")

    assert "write_json_atomic" in source
    assert "write_text_atomic" in source
    assert not re.search(r"_eval100d\.(?:json|md).*?\.write_text\(", source, re.DOTALL)


def test_monthly_from_total_zero_and_neg_window():
    assert _mod._monthly_from_total(0.10, 0) == 0.0
    assert _mod._monthly_from_total(0.10, -5) == 0.0


def test_median_target_impossible_only_after_strict_majority_below_target():
    target_total = _mod._total_from_monthly(0.27, 100)
    rows = [
        {"total_return": target_total - 0.01},
        {"total_return": target_total - 0.02},
    ]
    impossible, below_target, threshold = _mod._median_target_impossible(
        rows,
        n_windows=4,
        window_days=100,
        target_monthly=0.27,
    )
    assert impossible is False
    assert below_target == 2
    assert math.isclose(threshold, target_total, rel_tol=1e-12)

    impossible, below_target, _threshold = _mod._median_target_impossible(
        [*rows, {"total_return": target_total - 0.03}],
        n_windows=4,
        window_days=100,
        target_monthly=0.27,
    )
    assert impossible is True
    assert below_target == 3


def test_parse_int_csv_rejects_malformed_slippage_grid():
    assert _mod._parse_int_csv("0,5,10,20", label="slippage_bps") == [0, 5, 10, 20]
    with pytest.raises(ValueError, match="empty entry"):
        _mod._parse_int_csv("0,,20", label="slippage_bps")
    with pytest.raises(ValueError, match="duplicate entry"):
        _mod._parse_int_csv("0,5,5,20", label="slippage_bps")
    with pytest.raises(ValueError, match="non-integer entry"):
        _mod._parse_int_csv("0,stress", label="slippage_bps")
    with pytest.raises(ValueError, match="entry below 0"):
        _mod._parse_int_csv("-5,0,20", label="slippage_bps", min_value=0)


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


def test_eval_generic_slippage_sweep_forwards_borrow_apr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import struct

    import torch
    from fp4.bench import eval_generic
    from pufferlib_market import binding

    data_path = tmp_path / "mktd.bin"
    data_path.write_bytes(struct.pack("<4sIIIII", b"MKTD", 1, 1, 5, 2, 0) + b"\0" * 40)
    captured: dict[str, float] = {}
    term_buf_holder = {}

    class _Policy:
        def to(self, _device):
            return self

        def __call__(self, obs):
            return torch.zeros((obs.shape[0], 2), dtype=torch.float32, device=obs.device)

    def _vec_init(obs_bufs, act_bufs, rew_bufs, term_bufs, trunc_bufs, *args, **kwargs):
        del obs_bufs, act_bufs, rew_bufs, trunc_bufs, args
        captured["short_borrow_apr"] = float(kwargs["short_borrow_apr"])
        term_buf_holder["term_bufs"] = term_bufs
        return object()

    monkeypatch.setattr(binding, "shared", lambda **_kwargs: None)
    monkeypatch.setattr(binding, "vec_init", _vec_init)
    monkeypatch.setattr(binding, "vec_set_offsets", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(binding, "vec_reset", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        binding,
        "vec_step",
        lambda *_args, **_kwargs: term_buf_holder["term_bufs"].fill(1),
    )
    monkeypatch.setattr(binding, "vec_env_at", lambda _vec_handle, idx: idx)
    monkeypatch.setattr(
        binding,
        "env_get",
        lambda _env_handle: {
            "total_return": 3.0,
            "sortino": 5.0,
            "max_drawdown": 0.05,
        },
    )
    monkeypatch.setattr(binding, "vec_close", lambda *_args, **_kwargs: None)

    result = eval_generic._run_slippage_sweep(
        _Policy(),
        data_path,
        slippages_bps=[0],
        n_windows=1,
        eval_hours=2,
        fee_rate=0.001,
        max_leverage=1.5,
        short_borrow_apr=0.0625,
        seed=1,
        fail_fast=False,
    )

    assert result["status"] == "ok"
    assert captured["short_borrow_apr"] == 0.0625


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


def test_promotion_status_requires_hourly_intrabar_when_requested():
    agg = _mod._aggregate(_good_by_slippage(0, 5, 10, 20), window_days=100)

    gate = _mod._promotion_status(
        agg,
        target_monthly=0.27,
        max_dd_target=0.25,
        window_days=100,
        min_window_days=100,
        decision_lag=2,
        min_decision_lag=2,
        execution_granularity="daily",
        require_hourly_intrabar=True,
        slippage_bps=[0, 5, 10, 20],
        min_max_slippage_bps=20,
        required_slippage_bps=[0, 5, 10, 20],
        min_completed_windows=4,
    )

    assert gate["passed"] is False
    assert gate["execution_granularity"] == "daily"
    assert gate["require_hourly_intrabar"] is True
    assert gate["failures"] == [
        "execution_granularity daily is not promotable; use hourly_intrabar "
        "or --allow-daily-promotion for smoke/legacy checks"
    ]


def test_promotion_status_requires_full_production_slippage_grid():
    agg = _mod._aggregate(_good_by_slippage(20), window_days=100)

    gate = _mod._promotion_status(
        agg,
        target_monthly=0.27,
        max_dd_target=0.25,
        window_days=100,
        min_window_days=100,
        decision_lag=2,
        min_decision_lag=2,
        slippage_bps=[20],
        min_max_slippage_bps=20,
        required_slippage_bps=[0, 5, 10, 20],
    )

    assert gate["passed"] is False
    assert gate["missing_required_slippage_bps"] == [0, 5, 10]
    assert gate["failures"] == ["missing_required_slippage_bps 0,5,10"]


def test_promotion_status_rejects_negative_slippage_evidence():
    agg = _mod._aggregate(_good_by_slippage(-5, 0, 5, 10, 20), window_days=100)

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
        required_slippage_bps=[0, 5, 10, 20],
    )

    assert gate["passed"] is False
    assert gate["max_slippage_bps"] == 20
    assert gate["missing_required_slippage_bps"] == []
    assert gate["failures"] == ["slippage_bps must be non-negative"]


def test_promotion_status_rejects_negative_required_slippage():
    agg = _mod._aggregate(_good_by_slippage(0, 5, 10, 20), window_days=100)

    gate = _mod._promotion_status(
        agg,
        target_monthly=0.27,
        max_dd_target=0.25,
        window_days=100,
        min_window_days=100,
        decision_lag=2,
        min_decision_lag=2,
        slippage_bps=[0, 5, 10, 20],
        min_max_slippage_bps=20,
        required_slippage_bps=[-5, 0, 5, 10, 20],
    )

    assert gate["passed"] is False
    assert gate["failures"] == [
        "required_slippage_bps must be non-negative",
        "missing_required_slippage_bps -5",
    ]


def test_promotion_status_requires_hourly_max_hold_guard():
    agg = _mod._aggregate(_good_by_slippage(0, 5, 10, 20), window_days=100)

    disabled = _mod._promotion_status(
        agg,
        target_monthly=0.27,
        max_dd_target=0.25,
        window_days=100,
        min_window_days=100,
        decision_lag=2,
        min_decision_lag=2,
        slippage_bps=[0, 5, 10, 20],
        min_max_slippage_bps=20,
        required_slippage_bps=[0, 5, 10, 20],
        hourly_max_hold_hours=0,
        max_hourly_hold_hours=6,
    )
    loose = _mod._promotion_status(
        agg,
        target_monthly=0.27,
        max_dd_target=0.25,
        window_days=100,
        min_window_days=100,
        decision_lag=2,
        min_decision_lag=2,
        slippage_bps=[0, 5, 10, 20],
        min_max_slippage_bps=20,
        required_slippage_bps=[0, 5, 10, 20],
        hourly_max_hold_hours=12,
        max_hourly_hold_hours=6,
    )

    assert disabled["passed"] is False
    assert disabled["failures"] == ["hourly_max_hold_hours 0 disables max-hold guard"]
    assert loose["passed"] is False
    assert loose["failures"] == ["hourly_max_hold_hours 12 > 6"]


def test_promotion_status_requires_hourly_fill_buffer():
    agg = _mod._aggregate(_good_by_slippage(0, 5, 10, 20), window_days=100)

    gate = _mod._promotion_status(
        agg,
        target_monthly=0.27,
        max_dd_target=0.25,
        window_days=100,
        min_window_days=100,
        decision_lag=2,
        min_decision_lag=2,
        slippage_bps=[0, 5, 10, 20],
        min_max_slippage_bps=20,
        required_slippage_bps=[0, 5, 10, 20],
        hourly_fill_buffer_bps=0.0,
        min_hourly_fill_buffer_bps=5.0,
    )

    assert gate["passed"] is False
    assert gate["hourly_fill_buffer_bps"] == 0.0
    assert gate["min_hourly_fill_buffer_bps"] == 5.0
    assert gate["failures"] == ["hourly_fill_buffer_bps 0 < 5"]


def test_promotion_status_rejects_invalid_production_realism_thresholds():
    agg = _mod._aggregate(_good_by_slippage(0, 5, 10, 20), window_days=100)

    gate = _mod._promotion_status(
        agg,
        target_monthly=0.27,
        max_dd_target=0.25,
        window_days=100,
        min_window_days=100,
        decision_lag=2,
        min_decision_lag=2,
        fee_rate=0.001,
        min_fee_rate=float("nan"),
        short_borrow_apr=0.0625,
        min_short_borrow_apr=-1.0,
        max_leverage=1.5,
        max_leverage_target=float("inf"),
        slippage_bps=[0, 5, 10, 20],
        required_slippage_bps=[0, 5, 10, 20],
        hourly_fill_buffer_bps=5.0,
        min_hourly_fill_buffer_bps=-1.0,
        hourly_max_hold_hours=6,
        max_hourly_hold_hours=-1,
    )

    assert gate["passed"] is False
    assert gate["failures"] == [
        "min_fee_rate must be finite and non-negative",
        "min_short_borrow_apr must be finite and non-negative",
        "max_leverage_target must be finite and non-negative",
        "min_hourly_fill_buffer_bps must be finite and non-negative",
        "max_hourly_hold_hours must be non-negative",
    ]


def test_promotion_status_rejects_invalid_realism_values_even_for_smoke_overrides():
    agg = _mod._aggregate(_good_by_slippage(0, 5, 10, 20), window_days=100)

    gate = _mod._promotion_status(
        agg,
        target_monthly=0.27,
        max_dd_target=0.25,
        window_days=100,
        min_window_days=100,
        decision_lag=2,
        min_decision_lag=2,
        fee_rate=float("nan"),
        min_fee_rate=0.0,
        short_borrow_apr=-0.01,
        min_short_borrow_apr=0.0,
        max_leverage=float("inf"),
        max_leverage_target=0.0,
        slippage_bps=[0, 5, 10, 20],
        min_max_slippage_bps=0,
        required_slippage_bps=[],
        hourly_fill_buffer_bps=-1.0,
        min_hourly_fill_buffer_bps=0.0,
        hourly_max_hold_hours=0,
        max_hourly_hold_hours=0,
    )

    assert gate["passed"] is False
    assert gate["failures"] == [
        "fee_rate must be finite and non-negative",
        "short_borrow_apr must be finite and non-negative",
        "max_leverage must be finite and positive",
        "hourly_fill_buffer_bps must be finite and non-negative",
    ]


def test_promotion_status_rejects_invalid_core_gate_thresholds():
    agg = _mod._aggregate(_good_by_slippage(0, 5, 10, 20), window_days=100)

    gate = _mod._promotion_status(
        agg,
        target_monthly=float("nan"),
        max_dd_target=float("inf"),
        window_days=100,
        min_window_days=-1,
        decision_lag=2,
        min_decision_lag=-1,
        slippage_bps=[0, 5, 10, 20],
        min_max_slippage_bps=-1,
        required_slippage_bps=[0, 5, 10, 20],
        max_negative_windows=-1,
        min_completed_windows=-1,
    )

    assert gate["passed"] is False
    assert gate["failures"] == [
        "monthly_target must be finite and non-negative",
        "max_dd_target must be finite and non-negative",
        "min_window_days must be non-negative",
        "min_decision_lag must be non-negative",
        "min_max_slippage_bps must be non-negative",
        "min_completed_windows must be non-negative",
    ]


def test_promotion_status_requires_production_fee_rate():
    agg = _mod._aggregate(_good_by_slippage(0, 5, 10, 20), window_days=100)

    gate = _mod._promotion_status(
        agg,
        target_monthly=0.27,
        max_dd_target=0.25,
        window_days=100,
        min_window_days=100,
        decision_lag=2,
        min_decision_lag=2,
        fee_rate=0.0,
        min_fee_rate=0.001,
        slippage_bps=[0, 5, 10, 20],
        min_max_slippage_bps=20,
        required_slippage_bps=[0, 5, 10, 20],
    )

    assert gate["passed"] is False
    assert gate["fee_rate"] == 0.0
    assert gate["min_fee_rate"] == 0.001
    assert gate["failures"] == ["fee_rate 0 < 0.001"]


def test_promotion_status_requires_production_borrow_apr():
    agg = _mod._aggregate(_good_by_slippage(0, 5, 10, 20), window_days=100)

    gate = _mod._promotion_status(
        agg,
        target_monthly=0.27,
        max_dd_target=0.25,
        window_days=100,
        min_window_days=100,
        decision_lag=2,
        min_decision_lag=2,
        fee_rate=0.001,
        min_fee_rate=0.001,
        short_borrow_apr=0.0,
        min_short_borrow_apr=0.0625,
        slippage_bps=[0, 5, 10, 20],
        min_max_slippage_bps=20,
        required_slippage_bps=[0, 5, 10, 20],
    )

    assert gate["passed"] is False
    assert gate["short_borrow_apr"] == 0.0
    assert gate["min_short_borrow_apr"] == 0.0625
    assert gate["failures"] == ["short_borrow_apr 0 < 0.0625"]


def test_promotion_status_rejects_over_levered_eval():
    agg = _mod._aggregate(_good_by_slippage(0, 5, 10, 20), window_days=100)

    gate = _mod._promotion_status(
        agg,
        target_monthly=0.27,
        max_dd_target=0.25,
        window_days=100,
        min_window_days=100,
        decision_lag=2,
        min_decision_lag=2,
        fee_rate=0.001,
        min_fee_rate=0.001,
        max_leverage=3.0,
        max_leverage_target=2.0,
        slippage_bps=[0, 5, 10, 20],
        min_max_slippage_bps=20,
        required_slippage_bps=[0, 5, 10, 20],
    )

    assert gate["passed"] is False
    assert gate["max_leverage"] == 3.0
    assert gate["max_leverage_target"] == 2.0
    assert gate["failures"] == ["max_leverage 3 > 2"]


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
            "by_slippage": _good_by_slippage(0, 5, 10, 20),
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
    assert seen["target_monthly"] == 0.27
    assert seen["short_borrow_apr"] == 0.0625
    assert seen["max_leverage"] == 1.5
    assert seen["fill_buffer_bps"] == 5.0
    assert seen["max_hold_hours"] == 6

    payload = json.loads((out_dir / "best_eval100d.json").read_text())
    md = (out_dir / "best_eval100d.md").read_text()
    assert payload["raw"]["backend"] == "pufferlib_market_intrabar_hourly"
    assert payload["checkpoint_sha256"] == hashlib.sha256(b"x").hexdigest()
    assert payload["max_dd_target"] == 0.25
    assert payload["min_window_days"] == 100
    assert payload["decision_lag"] == 2
    assert payload["min_decision_lag"] == 2
    assert payload["fee_rate"] == 0.001
    assert payload["min_fee_rate"] == 0.001
    assert payload["short_borrow_apr"] == 0.0625
    assert payload["min_short_borrow_apr"] == 0.0625
    assert payload["max_leverage"] == 1.5
    assert payload["max_leverage_target"] == 2.0
    assert payload["min_max_slippage_bps"] == 20
    assert payload["required_slippage_bps"] == [0, 5, 10, 20]
    assert payload["hourly_fill_buffer_bps"] == 5.0
    assert payload["min_hourly_fill_buffer_bps"] == 5.0
    assert payload["hourly_max_hold_hours"] == 6
    assert payload["max_hourly_hold_hours"] == 6
    assert payload["promotion_gate"]["passed"] is True
    assert payload["promotion_gate"]["fee_rate"] == 0.001
    assert payload["promotion_gate"]["short_borrow_apr"] == 0.0625
    assert payload["promotion_gate"]["min_short_borrow_apr"] == 0.0625
    assert payload["promotion_gate"]["max_leverage"] == 1.5
    assert payload["promotion_gate"]["max_leverage_target"] == 2.0
    assert payload["promotion_gate"]["max_slippage_bps"] == 20
    assert payload["promotion_gate"]["missing_required_slippage_bps"] == []
    assert payload["promotion_gate"]["hourly_fill_buffer_bps"] == 5.0
    assert payload["promotion_gate"]["hourly_max_hold_hours"] == 6
    assert payload["promotion_gate"]["completed_windows"] == 4
    assert payload["promotion_gate"]["min_completed_windows"] == 4
    assert seen["decision_lag"] == 2
    assert "decision_lag: 2" in md
    assert "fee_rate: 0.001 (min 0.001)" in md
    assert "short_borrow_apr: 0.0625 (min 0.0625)" in md
    assert "max_leverage: 1.5 (max 2)" in md
    assert f"checkpoint_sha256: `{hashlib.sha256(b'x').hexdigest()}`" in md
    assert "slippage_bps: 0,5,10,20" in md
    assert "required_slippage_bps: 0,5,10,20" in md
    assert "hourly_fill_buffer_bps: 5 (min 5)" in md
    assert "hourly_max_hold_hours: 6 (max 6)" in md
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
                "5": _summary_cell(max_drawdown=0.30),
                "10": _summary_cell(max_drawdown=0.30),
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
                "5": _summary_cell(n_neg=0),
                "10": _summary_cell(n_neg=0),
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
                "5": _summary_cell(n_neg=1),
                "10": _summary_cell(n_neg=1),
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
                "5": _summary_cell(n_windows=4),
                "10": _summary_cell(n_windows=4),
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
                "5": _summary_cell(n_windows=4),
                "10": _summary_cell(n_windows=4),
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
                "5": _summary_cell(median_return=0.50, p10_return=0.40, mean_return=0.45),
                "10": _summary_cell(median_return=0.50, p10_return=0.40, mean_return=0.45),
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
    assert (out_dir / "best_eval100d.json").exists()


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
                "5": _summary_cell(median_return=0.50, p10_return=0.40, mean_return=0.45),
                "10": _summary_cell(median_return=0.50, p10_return=0.40, mean_return=0.45),
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


def test_main_rejects_disabled_hourly_max_hold_even_when_metrics_clear(
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
            "by_slippage": _good_by_slippage(0, 5, 10, 20),
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
        "--hourly-max-hold-hours", "0",
        "--out-dir", str(out_dir),
    ])

    assert rc == 3
    assert (out_dir / "best_eval100d.json").exists()


def test_main_allows_explicit_hourly_max_hold_override_for_smoke(
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
            "by_slippage": _good_by_slippage(0, 5, 10, 20),
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
        "--hourly-max-hold-hours", "0",
        "--max-hourly-hold-hours-target", "0",
        "--out-dir", str(out_dir),
    ])

    assert rc == 0
    payload = json.loads((out_dir / "best_eval100d.json").read_text())
    assert payload["promotion_gate"]["passed"] is True
    assert payload["promotion_gate"]["hourly_max_hold_hours"] == 0
    assert payload["promotion_gate"]["max_hourly_hold_hours"] == 0


def test_main_rejects_low_hourly_fill_buffer_even_when_metrics_clear(
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
            "by_slippage": _good_by_slippage(0, 5, 10, 20),
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
        "--hourly-fill-buffer-bps", "0",
        "--out-dir", str(out_dir),
    ])

    assert rc == 3
    assert (out_dir / "best_eval100d.json").exists()


def test_main_allows_explicit_hourly_fill_buffer_override_for_smoke(
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
            "by_slippage": _good_by_slippage(0, 5, 10, 20),
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
        "--hourly-fill-buffer-bps", "0",
        "--min-hourly-fill-buffer-bps", "0",
        "--out-dir", str(out_dir),
    ])

    assert rc == 0
    payload = json.loads((out_dir / "best_eval100d.json").read_text())
    assert payload["promotion_gate"]["passed"] is True
    assert payload["promotion_gate"]["hourly_fill_buffer_bps"] == 0.0
    assert payload["promotion_gate"]["min_hourly_fill_buffer_bps"] == 0.0


@pytest.mark.parametrize(
    ("extra_args", "message"),
    [
        (["--n-windows", "0"], "n_windows must be positive"),
        (["--fee-rate", "nan", "--min-fee-rate", "0"], "fee_rate must be finite"),
        (
            ["--short-borrow-apr", "-0.01", "--min-short-borrow-apr", "0"],
            "short_borrow_apr must be finite",
        ),
        (
            ["--max-leverage", "0", "--max-leverage-target", "0"],
            "max_leverage must be finite and positive",
        ),
        (
            ["--hourly-fill-buffer-bps", "-1", "--min-hourly-fill-buffer-bps", "0"],
            "hourly_fill_buffer_bps must be finite",
        ),
    ],
)
def test_main_rejects_invalid_simulation_inputs_before_evaluator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    extra_args: list[str],
    message: str,
):
    ckpt = tmp_path / "best.pt"
    val = tmp_path / "val.bin"
    hourly_root = tmp_path / "hourly"
    out_dir = tmp_path / "out"
    ckpt.write_bytes(b"x")
    val.write_bytes(b"x")
    hourly_root.mkdir()
    called = False

    def _fake_eval(**_kwargs):
        nonlocal called
        called = True
        return {
            "status": "ok",
            "backend": "pufferlib_market_intrabar_hourly",
            "by_slippage": _good_by_slippage(0, 5, 10, 20),
        }

    monkeypatch.setattr(_mod, "_evaluate_intrabar_hourly", _fake_eval)

    rc = _mod.main([
        "--checkpoint", str(ckpt),
        "--val-data", str(val),
        "--execution-granularity", "hourly_intrabar",
        "--hourly-data-root", str(hourly_root),
        "--daily-start-date", "2026-01-01",
        "--window-days", "100",
        "--out-dir", str(out_dir),
        *extra_args,
    ])

    assert rc == 2
    assert called is False
    assert message in capsys.readouterr().err
    assert not out_dir.exists()


@pytest.mark.parametrize(
    ("extra_args", "message"),
    [
        (["--window-days", "30"], "window_days 30 < 100"),
        (["--decision-lag", "1"], "decision_lag 1 < 2"),
        (["--slippage-bps", "0,5"], "max_slippage_bps 5 < 20"),
        (["--fee-rate", "0"], "fee_rate 0 < 0.001"),
        (["--short-borrow-apr", "0"], "short_borrow_apr 0 < 0.0625"),
        (["--max-leverage", "3"], "max_leverage 3 > 2"),
        (
            ["--hourly-fill-buffer-bps", "0"],
            "hourly_fill_buffer_bps 0 < 5",
        ),
        (
            ["--hourly-max-hold-hours", "0"],
            "hourly_max_hold_hours 0 disables max-hold guard",
        ),
    ],
)
def test_main_rejects_static_promotion_mismatches_before_evaluator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    extra_args: list[str],
    message: str,
):
    ckpt = tmp_path / "best.pt"
    val = tmp_path / "val.bin"
    hourly_root = tmp_path / "hourly"
    out_dir = tmp_path / "out"
    ckpt.write_bytes(b"x")
    val.write_bytes(b"x")
    hourly_root.mkdir()
    called = False

    def _fake_eval(**_kwargs):
        nonlocal called
        called = True
        return {
            "status": "ok",
            "backend": "pufferlib_market_intrabar_hourly",
            "by_slippage": _good_by_slippage(0, 5, 10, 20),
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
        *extra_args,
    ])

    assert rc == 3
    assert called is False
    assert message in capsys.readouterr().err
    payload = json.loads((out_dir / "best_eval100d.json").read_text())
    md = (out_dir / "best_eval100d.md").read_text()
    assert payload["raw"] == {"status": "static_promotion_preflight_failed"}
    assert payload["aggregate"]["by_slippage"] == {}
    assert payload["aggregate"]["worst_slip_monthly"] == 0.0
    assert payload["aggregate"]["max_slip_worst_max_drawdown"] == 0.0
    assert payload["aggregate"]["min_slip_n_windows"] == 0
    assert payload["promotion_gate"]["static_preflight"] is True
    assert payload["promotion_gate"]["completed_windows"] == 0
    assert payload["promotion_gate"]["negative_windows"] == 0
    assert payload["promotion_gate"]["max_slippage_bps"] in {5, 20}
    assert "missing_required_slippage_bps" in payload["promotion_gate"]
    assert payload["promotion_gate"]["worst_slip_monthly"] == 0.0
    assert payload["promotion_gate"]["max_slip_worst_max_drawdown"] == 0.0
    assert message in payload["promotion_gate"]["failures"]
    assert "STATIC_PREFLIGHT_FAIL" in md


def test_main_rejects_daily_execution_without_explicit_smoke_override(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    ckpt = tmp_path / "best.pt"
    val = tmp_path / "val.bin"
    out_dir = tmp_path / "out"
    ckpt.write_bytes(b"x")
    val.write_bytes(b"x")

    rc = _mod.main([
        "--checkpoint", str(ckpt),
        "--val-data", str(val),
        "--n-windows", "4",
        "--window-days", "100",
        "--out-dir", str(out_dir),
    ])

    message = (
        "execution_granularity daily is not promotable; use hourly_intrabar "
        "or --allow-daily-promotion for smoke/legacy checks"
    )
    assert rc == 3
    assert message in capsys.readouterr().err
    payload = json.loads((out_dir / "best_eval100d.json").read_text())
    assert payload["raw"] == {"status": "static_promotion_preflight_failed"}
    assert payload["execution_granularity"] == "daily"
    assert payload["require_hourly_intrabar"] is True
    assert payload["promotion_gate"]["execution_granularity"] == "daily"
    assert payload["promotion_gate"]["require_hourly_intrabar"] is True
    assert message in payload["promotion_gate"]["failures"]


def test_main_allows_daily_execution_only_with_explicit_smoke_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from fp4.bench import eval_generic

    ckpt = tmp_path / "best.pt"
    val = tmp_path / "val.bin"
    out_dir = tmp_path / "out"
    ckpt.write_bytes(b"x")
    val.write_bytes(b"x")
    seen = {}

    def _fake_eval(checkpoint, cfg, repo):
        seen["checkpoint"] = checkpoint
        seen["cfg"] = cfg
        seen["repo"] = repo
        return {
            "status": "ok",
            "backend": "pufferlib_market",
            "by_slippage": _good_by_slippage(0, 5, 10, 20),
        }

    monkeypatch.setattr(eval_generic, "evaluate_policy_file", _fake_eval)

    rc = _mod.main([
        "--checkpoint", str(ckpt),
        "--val-data", str(val),
        "--allow-daily-promotion",
        "--n-windows", "4",
        "--window-days", "100",
        "--out-dir", str(out_dir),
    ])

    assert rc == 0
    assert seen["checkpoint"] == ckpt.resolve()
    payload = json.loads((out_dir / "best_eval100d.json").read_text())
    md = (out_dir / "best_eval100d.md").read_text()
    assert payload["execution_granularity"] == "daily"
    assert payload["require_hourly_intrabar"] is False
    assert payload["promotion_gate"]["passed"] is True
    assert payload["promotion_gate"]["execution_granularity"] == "daily"
    assert payload["promotion_gate"]["require_hourly_intrabar"] is False
    assert "execution_granularity: daily (hourly_intrabar not required)" in md


def test_main_rejects_low_fee_rate_even_when_metrics_clear(
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
            "by_slippage": _good_by_slippage(0, 5, 10, 20),
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
        "--fee-rate", "0",
        "--out-dir", str(out_dir),
    ])

    assert rc == 3
    assert (out_dir / "best_eval100d.json").exists()


def test_main_allows_explicit_low_fee_rate_override_for_smoke(
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
            "by_slippage": _good_by_slippage(0, 5, 10, 20),
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
        "--fee-rate", "0",
        "--min-fee-rate", "0",
        "--out-dir", str(out_dir),
    ])

    assert rc == 0
    payload = json.loads((out_dir / "best_eval100d.json").read_text())
    assert payload["promotion_gate"]["passed"] is True
    assert payload["promotion_gate"]["fee_rate"] == 0.0
    assert payload["promotion_gate"]["min_fee_rate"] == 0.0


def test_main_rejects_low_borrow_apr_even_when_metrics_clear(
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
            "by_slippage": _good_by_slippage(0, 5, 10, 20),
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
        "--short-borrow-apr", "0",
        "--out-dir", str(out_dir),
    ])

    assert rc == 3
    assert (out_dir / "best_eval100d.json").exists()


def test_main_allows_explicit_low_borrow_apr_override_for_smoke(
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
            "by_slippage": _good_by_slippage(0, 5, 10, 20),
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
        "--short-borrow-apr", "0",
        "--min-short-borrow-apr", "0",
        "--out-dir", str(out_dir),
    ])

    assert rc == 0
    payload = json.loads((out_dir / "best_eval100d.json").read_text())
    assert payload["promotion_gate"]["passed"] is True
    assert payload["promotion_gate"]["short_borrow_apr"] == 0.0
    assert payload["promotion_gate"]["min_short_borrow_apr"] == 0.0


def test_main_rejects_over_levered_eval_even_when_metrics_clear(
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
            "by_slippage": _good_by_slippage(0, 5, 10, 20),
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
        "--max-leverage", "3",
        "--out-dir", str(out_dir),
    ])

    assert rc == 3
    assert (out_dir / "best_eval100d.json").exists()


def test_main_allows_explicit_high_leverage_override_for_smoke(
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
            "by_slippage": _good_by_slippage(0, 5, 10, 20),
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
        "--max-leverage", "3",
        "--max-leverage-target", "0",
        "--out-dir", str(out_dir),
    ])

    assert rc == 0
    payload = json.loads((out_dir / "best_eval100d.json").read_text())
    assert payload["promotion_gate"]["passed"] is True
    assert payload["promotion_gate"]["max_leverage"] == 3.0
    assert payload["promotion_gate"]["max_leverage_target"] == 0.0


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
            "by_slippage": _good_by_slippage(0, 5, 10, 20),
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
    assert (out_dir / "best_eval100d.json").exists()


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
            "by_slippage": _good_by_slippage(0, 5, 10, 20),
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
    assert (out_dir / "best_eval100d.json").exists()


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
        "--required-slippage-bps", "0,5",
        "--out-dir", str(out_dir),
    ])

    assert rc == 0
    payload = json.loads((out_dir / "best_eval100d.json").read_text())
    assert payload["promotion_gate"]["passed"] is True
    assert payload["promotion_gate"]["max_slippage_bps"] == 5
    assert payload["promotion_gate"]["min_max_slippage_bps"] == 5
    assert payload["promotion_gate"]["required_slippage_bps"] == [0, 5]


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
    assert payload["promotion_gate"]["failures"] == [
        "max_slippage_bps 0 < 20",
        "missing_required_slippage_bps 5,10,20",
    ]


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
    assert payload["aggregate"]["by_slippage"] == {}
    assert payload["aggregate"]["worst_slip_monthly"] == 0.0
    assert payload["promotion_gate"]["passed"] is False
    assert payload["promotion_gate"]["max_slippage_bps"] == 0
    assert payload["promotion_gate"]["missing_required_slippage_bps"] == [0, 5, 10, 20]
    assert (
        "evaluator_status skip: not enough data"
        in payload["promotion_gate"]["failures"]
    )
    assert payload["execution_granularity"] == "hourly_intrabar"
    assert payload["require_hourly_intrabar"] is True
    md = (out_dir / "best_eval100d.md").read_text()
    assert "EVALUATOR_NON_OK" in md
    assert "promotion_gate: FAIL" in md
