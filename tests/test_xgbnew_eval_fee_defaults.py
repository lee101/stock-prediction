"""Fee-default contracts for legacy XGB evaluators."""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from xgbnew import (
    eval_ensemble,
    eval_hourly_multiwindow,
    eval_kfold,
    eval_multiwindow,
    eval_pretrained,
    run_daily,
    run_hourly,
)
from xgbnew.backtest import PRODUCTION_STOCK_FEE_RATE
from xgbnew.eval_ensemble import (
    _validate_realism_args as validate_ensemble_realism_args,
)
from xgbnew.eval_ensemble import parse_args as parse_ensemble_args
from xgbnew.eval_hourly_multiwindow import (
    _validate_realism_args as validate_hourly_realism_args,
)
from xgbnew.eval_hourly_multiwindow import parse_args as parse_hourly_multiwindow_args
from xgbnew.eval_kfold import (
    _validate_realism_args as validate_kfold_realism_args,
)
from xgbnew.eval_kfold import parse_args as parse_kfold_args
from xgbnew.eval_multiwindow import (
    _validate_realism_args as validate_multiwindow_realism_args,
)
from xgbnew.eval_multiwindow import parse_args as parse_multiwindow_args
from xgbnew.eval_pretrained import (
    _validate_realism_args as validate_pretrained_realism_args,
)
from xgbnew.eval_pretrained import parse_args as parse_pretrained_args
from xgbnew.run_daily import (
    _validate_realism_args as validate_run_daily_realism_args,
)
from xgbnew.run_daily import parse_args as parse_run_daily_args
from xgbnew.run_hourly import (
    _validate_realism_args as validate_run_hourly_realism_args,
)
from xgbnew.run_hourly import parse_args as parse_run_hourly_args


@pytest.mark.parametrize(
    "module",
    [
        eval_ensemble,
        eval_hourly_multiwindow,
        eval_kfold,
        eval_multiwindow,
        eval_pretrained,
    ],
)
def test_evaluator_json_outputs_use_shared_atomic_writer(module) -> None:
    source = inspect.getsource(module)
    assert "write_json_atomic" in source
    assert ".write_text(json.dumps" not in source


def test_eval_ensemble_default_fee_is_production_stress_fee(tmp_path: Path) -> None:
    args = parse_ensemble_args(
        [
            "--symbols-file",
            str(tmp_path / "symbols.txt"),
        ]
    )
    assert args.fee_rate == PRODUCTION_STOCK_FEE_RATE


def test_eval_multiwindow_default_fee_is_production_stress_fee() -> None:
    args = parse_multiwindow_args([])
    assert args.fee_rate == PRODUCTION_STOCK_FEE_RATE


def test_eval_kfold_default_fee_is_production_stress_fee(tmp_path: Path) -> None:
    args = parse_kfold_args(
        [
            "--symbols-file",
            str(tmp_path / "symbols.txt"),
        ]
    )
    assert args.fee_rate == PRODUCTION_STOCK_FEE_RATE


def test_eval_pretrained_default_fee_is_production_stress_fee(tmp_path: Path) -> None:
    args = parse_pretrained_args(
        [
            "--model-path",
            str(tmp_path / "missing.pkl"),
            "--symbols-file",
            str(tmp_path / "symbols.txt"),
        ]
    )
    assert args.fee_rate == PRODUCTION_STOCK_FEE_RATE


@pytest.mark.parametrize(
    ("parse", "validate", "base_args"),
    [
        (parse_multiwindow_args, validate_multiwindow_realism_args, []),
        (
            parse_ensemble_args,
            validate_ensemble_realism_args,
            ["--symbols-file", "symbols.txt"],
        ),
        (
            parse_kfold_args,
            validate_kfold_realism_args,
            ["--symbols-file", "symbols.txt"],
        ),
        (
            parse_pretrained_args,
            validate_pretrained_realism_args,
            ["--model-path", "model.pkl", "--symbols-file", "symbols.txt"],
        ),
    ],
)
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
def test_daily_evaluators_reject_invalid_realism_args(
    parse,
    validate,
    base_args: list[str],
    flag: str,
    value: str,
    expected: str,
) -> None:
    args = parse([*base_args, flag, value])
    assert validate(args) == [expected]


@pytest.mark.parametrize(
    ("main", "argv", "later_error"),
    [
        (eval_multiwindow.main, ["--fee-rate", "nan"], "No such file"),
        (
            eval_ensemble.main,
            ["--symbols-file", "missing-symbols.txt", "--fee-rate", "nan"],
            "missing-symbols",
        ),
        (
            eval_kfold.main,
            ["--symbols-file", "missing-symbols.txt", "--fee-rate", "nan"],
            "missing-symbols",
        ),
        (
            eval_pretrained.main,
            [
                "--model-path",
                "missing-model.pkl",
                "--symbols-file",
                "missing-symbols.txt",
                "--fee-rate",
                "nan",
            ],
            "model pickle not found",
        ),
    ],
)
def test_daily_evaluator_mains_reject_invalid_realism_before_io(
    main,
    argv: list[str],
    later_error: str,
    capsys,
) -> None:
    rc = main(argv)

    assert rc == 2
    stderr = capsys.readouterr().err
    assert "ERROR: fee_rate must be finite and non-negative" in stderr
    assert later_error not in stderr


def test_run_daily_default_fee_is_production_stress_fee() -> None:
    args = parse_run_daily_args([])
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
def test_run_daily_rejects_invalid_realism_args(
    flag: str,
    value: str,
    expected: str,
) -> None:
    args = parse_run_daily_args([flag, value])
    assert validate_run_daily_realism_args(args) == [expected]


def test_run_daily_main_rejects_invalid_realism_before_data_work(capsys) -> None:
    rc = run_daily.main(["--fee-rate", "nan"])

    assert rc == 2
    assert "ERROR: fee_rate must be finite and non-negative" in capsys.readouterr().err


def test_eval_hourly_multiwindow_default_fee_is_production_stress_fee() -> None:
    args = parse_hourly_multiwindow_args([])
    assert args.fee_rate == PRODUCTION_STOCK_FEE_RATE


def test_run_hourly_default_fee_is_production_stress_fee() -> None:
    args = parse_run_hourly_args([])
    assert args.fee_rate == PRODUCTION_STOCK_FEE_RATE
    assert args.fill_buffer_bps == 5.0
    assert args.commission_bps == 0.0


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
def test_run_hourly_rejects_invalid_realism_args(
    flag: str,
    value: str,
    expected: str,
) -> None:
    args = parse_run_hourly_args([flag, value])
    assert validate_run_hourly_realism_args(args) == [expected]


def test_run_hourly_main_rejects_invalid_realism_before_file_work(
    capsys,
    tmp_path: Path,
) -> None:
    rc = run_hourly.main(
        [
            "--mktd-file",
            str(tmp_path / "missing.mktd"),
            "--fee-rate",
            "nan",
        ]
    )

    assert rc == 2
    stderr = capsys.readouterr().err
    assert "ERROR: fee_rate must be finite and non-negative" in stderr
    assert "MKTD file not found" not in stderr
