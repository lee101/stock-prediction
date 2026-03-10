from __future__ import annotations

from unified_hourly_experiment.auto_meta_optimize import parse_float_list, rank_key


def test_parse_float_list() -> None:
    assert parse_float_list("0.1, 0.2,0.3") == [0.1, 0.2, 0.3]


def test_rank_key_prefers_higher_sortino_then_return() -> None:
    a = {
        "min_sortino": 0.4,
        "mean_sortino": 1.0,
        "min_return_pct": 0.2,
        "mean_return_pct": 0.7,
        "mean_max_drawdown_pct": 0.4,
    }
    b = {
        "min_sortino": 0.1,
        "mean_sortino": 2.0,
        "min_return_pct": 0.5,
        "mean_return_pct": 1.2,
        "mean_max_drawdown_pct": 0.2,
    }
    assert rank_key(a) > rank_key(b)

