from __future__ import annotations

from unified_hourly_experiment.sweep_meta_portfolio import parse_sit_out_threshold_values


def test_parse_sit_out_threshold_values_disabled() -> None:
    values = parse_sit_out_threshold_values(
        sit_out_if_negative=False,
        sit_out_threshold=0.2,
        sit_out_thresholds="0.1,0.3",
    )
    assert values == [None]


def test_parse_sit_out_threshold_values_legacy_single() -> None:
    values = parse_sit_out_threshold_values(
        sit_out_if_negative=True,
        sit_out_threshold=0.25,
        sit_out_thresholds="",
    )
    assert values == [0.25]


def test_parse_sit_out_threshold_values_multi() -> None:
    values = parse_sit_out_threshold_values(
        sit_out_if_negative=True,
        sit_out_threshold=0.0,
        sit_out_thresholds="-0.0015,-0.001,0.0",
    )
    assert values == [-0.0015, -0.001, 0.0]
