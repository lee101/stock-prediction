from __future__ import annotations

import pytest

from scripts.sweep_binance33_xgb import _monthly_equivalent_return, _passes_production_target


def test_monthly_equivalent_return_keeps_30_day_return_unchanged() -> None:
    assert _monthly_equivalent_return(0.27, 30) == pytest.approx(0.27)


def test_monthly_equivalent_return_converts_120_day_return() -> None:
    assert _monthly_equivalent_return(0.68956464, 120) == pytest.approx(0.14010199, rel=1e-6)


def test_monthly_equivalent_return_requires_positive_days() -> None:
    with pytest.raises(ValueError, match="days must be positive"):
        _monthly_equivalent_return(0.1, 0)


def test_passes_production_target_requires_smooth_positive_worst_slippage_row() -> None:
    row = {
        "failed_fast": 0,
        "median_monthly_pct": 28.0,
        "p10_monthly_pct": 3.0,
        "neg_windows": 0,
        "p90_dd_pct": 12.0,
    }

    assert _passes_production_target(row, target_monthly_pct=27.0, max_dd_pct=20.0)


@pytest.mark.parametrize(
    ("override", "expected_reason"),
    [
        ({"failed_fast": 1}, "failed fast"),
        ({"median_monthly_pct": 26.9}, "median below target"),
        ({"p10_monthly_pct": -0.1}, "negative p10"),
        ({"neg_windows": 1}, "negative windows"),
        ({"p90_dd_pct": 20.1}, "drawdown above target"),
    ],
)
def test_passes_production_target_rejects_weak_rows(
    override: dict[str, float | int],
    expected_reason: str,
) -> None:
    row = {
        "failed_fast": 0,
        "median_monthly_pct": 28.0,
        "p10_monthly_pct": 3.0,
        "neg_windows": 0,
        "p90_dd_pct": 12.0,
        **override,
    }

    assert not _passes_production_target(row, target_monthly_pct=27.0, max_dd_pct=20.0), expected_reason
