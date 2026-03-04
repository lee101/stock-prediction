from unified_hourly_experiment.chronos_nonregression_sweep import (
    float_token,
    parse_float_list,
    parse_int_list,
    should_promote,
)


def test_float_token_strips_trailing_zeros() -> None:
    assert float_token(1e-4) == "0p0001"
    assert float_token(5e-5) == "0p00005"
    assert float_token(0.125) == "0p125"


def test_parse_lists() -> None:
    assert parse_int_list(" 1,2, 3 ") == [1, 2, 3]
    assert parse_float_list("5e-5, 1e-4 ,0.2") == [5e-5, 1e-4, 0.2]


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
