from unified_hourly_experiment.chronos_nonregression_sweep import (
    float_token,
    parse_float_list,
    parse_int_list,
    parse_optional_int_list,
    select_best_robust_candidate,
    should_promote,
    should_promote_on_windows,
)


def test_float_token_strips_trailing_zeros() -> None:
    assert float_token(1e-4) == "0p0001"
    assert float_token(5e-5) == "0p00005"
    assert float_token(0.125) == "0p125"


def test_parse_lists() -> None:
    assert parse_int_list(" 1,2, 3 ") == [1, 2, 3]
    assert parse_float_list("5e-5, 1e-4 ,0.2") == [5e-5, 1e-4, 0.2]
    assert parse_optional_int_list("") == []
    assert parse_optional_int_list("168, 336") == [168, 336]


def test_should_promote_requires_candidate_better() -> None:
    assert should_promote(
        current_test_mae_percent=10.0,
        candidate_test_mae_percent=9.9,
        min_improvement_abs=0.0,
        min_improvement_rel=0.0,
    )
    assert not should_promote(
        current_test_mae_percent=10.0,
        candidate_test_mae_percent=10.0,
        min_improvement_abs=0.0,
        min_improvement_rel=0.0,
    )
    assert not should_promote(
        current_test_mae_percent=10.0,
        candidate_test_mae_percent=10.1,
        min_improvement_abs=0.0,
        min_improvement_rel=0.0,
    )


def test_should_promote_respects_abs_threshold() -> None:
    assert not should_promote(
        current_test_mae_percent=10.0,
        candidate_test_mae_percent=9.95,
        min_improvement_abs=0.1,
        min_improvement_rel=0.0,
    )
    assert should_promote(
        current_test_mae_percent=10.0,
        candidate_test_mae_percent=9.8,
        min_improvement_abs=0.1,
        min_improvement_rel=0.0,
    )


def test_should_promote_respects_rel_threshold() -> None:
    assert not should_promote(
        current_test_mae_percent=10.0,
        candidate_test_mae_percent=9.7,
        min_improvement_abs=0.0,
        min_improvement_rel=0.05,
    )
    assert should_promote(
        current_test_mae_percent=10.0,
        candidate_test_mae_percent=9.4,
        min_improvement_abs=0.0,
        min_improvement_rel=0.05,
    )


def test_should_promote_on_windows_requires_mean_gain_and_limited_regression() -> None:
    ok, details = should_promote_on_windows(
        current_by_window={168: 5.0, 336: 4.0, 672: 3.0},
        candidate_by_window={168: 4.8, 336: 3.9, 672: 2.8},
        max_window_regression=0.0,
        min_mean_improvement=0.0,
    )
    assert ok
    assert details["mean_improvement_test_mae_percent"] > 0.0
    assert details["max_window_regression"] <= 0.0


def test_should_promote_on_windows_blocks_large_single_window_regression() -> None:
    ok, details = should_promote_on_windows(
        current_by_window={168: 5.0, 336: 4.0, 672: 3.0},
        candidate_by_window={168: 5.2, 336: 3.4, 672: 2.0},
        max_window_regression=0.05,
        min_mean_improvement=0.0,
    )
    assert not ok
    assert details["max_window_regression"] > 0.05


def test_should_promote_on_windows_respects_min_mean_improvement() -> None:
    ok, details = should_promote_on_windows(
        current_by_window={168: 5.0, 336: 4.0, 672: 3.0},
        candidate_by_window={168: 4.95, 336: 3.98, 672: 2.97},
        max_window_regression=0.0,
        min_mean_improvement=0.05,
    )
    assert not ok
    assert details["mean_improvement_test_mae_percent"] < 0.05


def test_select_best_robust_candidate_prefers_mean_improvement_then_test_mae() -> None:
    c1 = {"save_name": "a", "test_mae_percent": 1.0}
    d1 = {"mean_improvement_test_mae_percent": 0.05}
    c2 = {"save_name": "b", "test_mae_percent": 0.8}
    d2 = {"mean_improvement_test_mae_percent": 0.04}
    c3 = {"save_name": "c", "test_mae_percent": 0.9}
    d3 = {"mean_improvement_test_mae_percent": 0.05}

    best = select_best_robust_candidate([(c1, d1), (c2, d2), (c3, d3)])
    assert best is not None
    assert best["save_name"] == "c"


def test_select_best_robust_candidate_empty() -> None:
    assert select_best_robust_candidate([]) is None
