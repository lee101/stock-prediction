"""Unit tests for the DSR / Bonferroni helper in ``xgbnew.analyze_sweep``.

Checks:
  - Moment computation is correct on synthetic data.
  - SR threshold formula: zero when only one trial.
  - DSR = 1.0 for a confidently-positive trial.
  - DSR ~ 0.5 when observed Sharpe sits on the null threshold.
  - Sortino with no downside returns +inf (positive mean) or 0 (non-positive mean).
  - Seed-dispersion grouping correctly pools configs that share hyperparams.
  - Top-level ``analyze_sweep`` round-trip on a synthetic multiwindow JSON.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from xgbnew.analyze_sweep import (
    _moments,
    _sortino,
    analyze_sweep,
    deflated_sharpe,
    sr_threshold,
)


def test_moments_of_symmetric_distribution():
    xs = [-2.0, -1.0, 0.0, 1.0, 2.0]
    mean, sd, skew, kurt = _moments(xs)
    assert mean == pytest.approx(0.0, abs=1e-9)
    # population sd of {-2,-1,0,1,2} = sqrt(2)
    assert sd == pytest.approx(math.sqrt(2.0), abs=1e-6)
    assert skew == pytest.approx(0.0, abs=1e-9)
    # {-2,-1,0,1,2}: pop var=2, pop m4=6.8, excess kurt=6.8/4 - 3 = -1.3
    assert kurt == pytest.approx(-1.3, abs=1e-6)


def test_moments_handles_short_input():
    assert _moments([]) == (0.0, 0.0, 0.0, 0.0)
    assert _moments([1.5]) == (0.0, 0.0, 0.0, 0.0)


def test_sr_threshold_single_trial_is_zero():
    assert sr_threshold(0.5, n_trials=1) == 0.0
    assert sr_threshold(0.0, n_trials=100) == 0.0


def test_sr_threshold_monotone_in_n_trials():
    # More trials ⇒ higher null-max Sharpe threshold.
    sd_var = 0.1
    t10 = sr_threshold(sd_var, 10)
    t100 = sr_threshold(sd_var, 100)
    t1000 = sr_threshold(sd_var, 1000)
    assert t10 < t100 < t1000


def test_deflated_sharpe_saturates_for_strong_signal():
    # SR_hat huge, sr_star modest, n_obs modest ⇒ DSR ≈ 1.0.
    dsr = deflated_sharpe(sr_hat=3.0, n_obs=34, skew=0.0, excess_kurt=0.0, sr_star=0.5)
    assert dsr > 0.9999


def test_deflated_sharpe_neutral_at_threshold():
    # SR_hat == sr_star ⇒ z=0 ⇒ DSR = Φ(0) = 0.5.
    dsr = deflated_sharpe(sr_hat=0.5, n_obs=34, skew=0.0, excess_kurt=0.0, sr_star=0.5)
    assert dsr == pytest.approx(0.5, abs=1e-6)


def test_deflated_sharpe_handles_negative_skew_penalty():
    # DSR denom variance is (1 - skew·SR + (kurt/4)·SR²). For positive skew the
    # denom shrinks, so the z-stat grows and DSR rises. Negative skew inflates
    # the denom and lowers DSR. Use a small Sharpe so we stay below saturation.
    base = deflated_sharpe(sr_hat=0.7, n_obs=20, skew=0.0, excess_kurt=0.0, sr_star=0.5)
    pos_skew = deflated_sharpe(sr_hat=0.7, n_obs=20, skew=0.8, excess_kurt=0.0, sr_star=0.5)
    neg_skew = deflated_sharpe(sr_hat=0.7, n_obs=20, skew=-0.8, excess_kurt=0.0, sr_star=0.5)
    assert pos_skew > base > neg_skew
    # All should sit in (0, 1).
    assert 0.0 < neg_skew < base < pos_skew < 1.0


def test_sortino_returns_inf_when_no_downside():
    out = _sortino([1.0, 2.0, 3.0, 4.0])
    assert math.isinf(out) and out > 0.0


def test_sortino_zero_when_mean_nonpositive_and_no_downside():
    assert _sortino([0.0, 0.0, 0.0]) == 0.0


def test_sortino_finite_with_negatives():
    out = _sortino([-1.0, 2.0, 3.0])
    # mean=4/3, downside=[-1] (single entry), ds=1, sortino≈1.333
    assert out == pytest.approx(4.0 / 3.0, abs=1e-6)


def _write_fake_sweep(tmp: Path, trials: list[dict]) -> Path:
    """Persist a multiwindow-compatible JSON for ``analyze_sweep``."""
    doc = {
        "train_start": "2021-01-01",
        "train_end": "2023-12-31",
        "oos_start": "2024-01-02",
        "oos_end": "2026-04-18",
        "sweep_results": trials,
    }
    path = tmp / "multiwindow_fake.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _wins(rets: list[float]) -> list[dict]:
    return [{"monthly_return_pct": r} for r in rets]


def test_analyze_sweep_reports_best_and_seed_dispersion(tmp_path):
    # Three configs at same hyperparams (seed replicas) + one lucky config.
    trials = [
        {
            "config": {
                "n_estimators": 400, "max_depth": 5, "learning_rate": 0.03,
                "top_n": 1, "xgb_weight": 1.0, "leverage": 1.0, "random_state": 0,
            },
            "windows": _wins([5, 7, 9, 4, 6, 8]),
        },
        {
            "config": {
                "n_estimators": 400, "max_depth": 5, "learning_rate": 0.03,
                "top_n": 1, "xgb_weight": 1.0, "leverage": 1.0, "random_state": 1,
            },
            "windows": _wins([3, 6, 4, 2, 5, 7]),
        },
        {
            "config": {
                "n_estimators": 400, "max_depth": 5, "learning_rate": 0.03,
                "top_n": 1, "xgb_weight": 1.0, "leverage": 1.0, "random_state": 2,
            },
            "windows": _wins([4, 5, 3, 6, 4, 5]),
        },
        {
            # One-off lucky config at different hyperparams.
            "config": {
                "n_estimators": 800, "max_depth": 7, "learning_rate": 0.03,
                "top_n": 2, "xgb_weight": 1.0, "leverage": 1.0, "random_state": 99,
            },
            "windows": _wins([12, 14, 11, 13, 15, 12]),
        },
    ]
    path = _write_fake_sweep(tmp_path, trials)
    summary = analyze_sweep(path)
    assert summary["n_trials"] == 4
    # Best by median monthly is the "lucky" cell at config depth=7 top_n=2.
    best_cfg = summary["best"]["config"]
    assert best_cfg["max_depth"] == 7
    assert best_cfg["top_n"] == 2
    # DSR should be high — mean 12+, std small, with 4 trials.
    assert summary["best_deflated_sharpe"] >= 0.9
    # Seed dispersion should have one group with 3 seeds (the replica cell).
    groups = summary["seed_dispersion_per_hyperparam_cell"]
    assert len(groups) == 1
    g = groups[0]
    assert g["n_seeds"] == 3
    assert set(g["seeds"]) == {0, 1, 2}
    # medians across seeds for the replica cell: 6.5, 4.5, 4.5
    assert g["median_monthly_median"] == pytest.approx(4.5, abs=1e-6)


def test_analyze_sweep_rejects_empty_sweep(tmp_path):
    path = _write_fake_sweep(tmp_path, [])
    with pytest.raises(SystemExit):
        analyze_sweep(path)
