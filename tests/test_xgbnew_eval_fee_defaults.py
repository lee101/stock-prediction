"""Fee-default contracts for legacy daily XGB evaluators."""
from __future__ import annotations

from pathlib import Path

from xgbnew.backtest import PRODUCTION_STOCK_FEE_RATE
from xgbnew.eval_ensemble import parse_args as parse_ensemble_args
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
