"""Fee-default contracts for legacy XGB evaluators."""
from __future__ import annotations

from pathlib import Path

import pytest
from xgbnew.backtest import PRODUCTION_STOCK_FEE_RATE
from xgbnew.eval_ensemble import parse_args as parse_ensemble_args
from xgbnew.eval_hourly_multiwindow import (
    _validate_realism_args as validate_hourly_realism_args,
)
from xgbnew.eval_hourly_multiwindow import parse_args as parse_hourly_multiwindow_args
from xgbnew.eval_kfold import parse_args as parse_kfold_args
from xgbnew.run_daily import parse_args as parse_run_daily_args


def test_eval_ensemble_default_fee_is_production_stress_fee(tmp_path: Path) -> None:
    args = parse_ensemble_args(
        [
            "--symbols-file",
            str(tmp_path / "symbols.txt"),
        ]
    )
    assert args.fee_rate == PRODUCTION_STOCK_FEE_RATE


def test_eval_kfold_default_fee_is_production_stress_fee(tmp_path: Path) -> None:
    args = parse_kfold_args(
        [
            "--symbols-file",
            str(tmp_path / "symbols.txt"),
        ]
    )
    assert args.fee_rate == PRODUCTION_STOCK_FEE_RATE


def test_run_daily_default_fee_is_production_stress_fee() -> None:
    args = parse_run_daily_args([])
    assert args.fee_rate == PRODUCTION_STOCK_FEE_RATE


def test_eval_hourly_multiwindow_default_fee_is_production_stress_fee() -> None:
    args = parse_hourly_multiwindow_args([])
    assert args.fee_rate == PRODUCTION_STOCK_FEE_RATE


@pytest.mark.parametrize(
    ("flag", "value", "expected"),
    [
        ("--fee-rate", "nan", "fee_rate must be finite and non-negative"),
        ("--fee-rate", "-1", "fee_rate must be finite and non-negative"),
        (
            "--fill-buffer-bps",
            "-1",
            "fill_buffer_bps must be finite and non-negative",
        ),
        (
            "--commission-bps",
            "inf",
            "commission_bps must be finite and non-negative",
        ),
    ],
)
def test_eval_hourly_multiwindow_rejects_invalid_realism_args(
    flag: str,
    value: str,
    expected: str,
) -> None:
    args = parse_hourly_multiwindow_args([flag, value])
    assert validate_hourly_realism_args(args) == [expected]
