from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.chronos2_objective import (
    build_correlation_cohort_map,
    build_correlation_matrix,
    compute_chronos_objective,
    compute_error_smoothness,
    select_correlation_cohort,
)


def test_compute_error_smoothness_handles_short_series() -> None:
    assert compute_error_smoothness([0.12]) == 0.0
    value = compute_error_smoothness([0.1, 0.2, 0.3, 0.4])
    assert value == pytest.approx(0.1)


def test_compute_chronos_objective_includes_smoothness_and_direction() -> None:
    actual = np.array([101.0, 99.0, 102.0], dtype=np.float64)
    predicted = np.array([100.5, 99.5, 101.5], dtype=np.float64)
    metrics = compute_chronos_objective(
        actual_close=actual,
        predicted_close=predicted,
        reference_close=100.0,
        smoothness_weight=0.5,
        direction_bonus=0.1,
    )

    abs_return_errors = np.abs((predicted - 100.0) / 100.0 - (actual - 100.0) / 100.0)
    expected_mae = float(np.mean(abs_return_errors))
    expected_smooth = float(np.mean(np.abs(np.diff(abs_return_errors))))
    expected_direction = 1.0
    expected_objective = expected_mae + 0.5 * expected_smooth - 0.1 * expected_direction

    assert metrics.n_samples == 3
    assert metrics.pct_return_mae == pytest.approx(expected_mae)
    assert metrics.pct_return_mae_smoothness == pytest.approx(expected_smooth)
    assert metrics.direction_accuracy == pytest.approx(expected_direction)
    assert metrics.objective == pytest.approx(expected_objective)


def test_compute_chronos_objective_rejects_invalid_reference() -> None:
    with pytest.raises(ValueError, match="reference_close"):
        compute_chronos_objective(
            actual_close=[1.0, 2.0],
            predicted_close=[1.1, 2.1],
            reference_close=0.0,
        )


def test_build_correlation_matrix_and_cohort_selection() -> None:
    idx = pd.date_range("2026-01-01", periods=120, freq="h", tz="UTC")
    base = pd.Series(np.linspace(100.0, 120.0, len(idx)), index=idx)
    peer_pos = base * 1.01
    peer_neg = 1.0 / base
    uncorrelated = pd.Series(np.sin(np.arange(len(idx))), index=idx)

    corr = build_correlation_matrix(
        {
            "AAA": base,
            "BBB": peer_pos,
            "CCC": peer_neg,
            "DDD": uncorrelated,
        },
        lookback=96,
        min_periods=24,
    )
    assert not corr.empty
    cohort_default = select_correlation_cohort(
        symbol="AAA",
        corr_matrix=corr,
        max_size=3,
        min_abs_corr=0.2,
        include_negative=False,
    )
    assert "BBB" in cohort_default
    assert "CCC" not in cohort_default

    cohort_with_negative = select_correlation_cohort(
        symbol="AAA",
        corr_matrix=corr,
        max_size=3,
        min_abs_corr=0.2,
        include_negative=True,
    )
    assert "BBB" in cohort_with_negative
    assert "CCC" in cohort_with_negative

    cohort_map = build_correlation_cohort_map(
        corr,
        max_size=2,
        min_abs_corr=0.2,
        include_negative=True,
    )
    assert "AAA" in cohort_map
    assert len(cohort_map["AAA"]) <= 2


def test_build_correlation_matrix_keeps_overlap_when_one_symbol_is_disjoint() -> None:
    idx = pd.date_range("2026-01-01", periods=120, freq="h", tz="UTC")
    base = pd.Series(np.linspace(100.0, 120.0, len(idx)), index=idx)
    peer = base * 1.02
    disjoint_idx = pd.date_range("2025-12-01", periods=48, freq="h", tz="UTC")
    disjoint = pd.Series(np.linspace(10.0, 12.0, len(disjoint_idx)), index=disjoint_idx)

    corr = build_correlation_matrix(
        {
            "AAA": base,
            "BBB": peer,
            "ZZZ": disjoint,
        },
        lookback=120,
        min_periods=24,
    )

    assert not corr.empty
    assert corr.loc["AAA", "BBB"] > 0.99
    cohort = select_correlation_cohort(
        symbol="AAA",
        corr_matrix=corr,
        max_size=2,
        min_abs_corr=0.2,
        include_negative=False,
    )
    assert "BBB" in cohort
