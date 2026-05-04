from __future__ import annotations

import ast
from pathlib import Path

import gpu_trading_env
import pytest
from pufferlib_market.realism import (
    PRODUCTION_DECISION_LAG,
    PRODUCTION_FEE_BPS,
    PRODUCTION_FILL_BUFFER_BPS,
    PRODUCTION_SHORT_BORROW_APR,
    require_production_decision_lag,
)


SHORT_BORROW_EVALUATOR_ENTRYPOINTS = [
    Path("pufferlib_market/evaluate.py"),
    Path("pufferlib_market/evaluate_fast.py"),
    Path("pufferlib_market/evaluate_holdout.py"),
    Path("pufferlib_market/evaluate_multiperiod.py"),
    Path("pufferlib_market/evaluate_sliding.py"),
    Path("pufferlib_market/evaluate_tail.py"),
    Path("pufferlib_market/evaluate_ttt.py"),
    Path("pufferlib_market/meta_replay_eval.py"),
    Path("pufferlib_market/replay_eval.py"),
]

DECISION_LAG_EVALUATOR_ENTRYPOINTS = [
    Path("pufferlib_market/evaluate_holdout.py"),
    Path("pufferlib_market/evaluate_multiperiod.py"),
    Path("pufferlib_market/evaluate_tail.py"),
]


def _argument_defaults(path: Path, flag: str) -> list[ast.expr]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    defaults: list[ast.expr] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == flag
        ):
            continue
        for keyword in node.keywords:
            if keyword.arg == "default":
                defaults.append(keyword.value)
    return defaults


def _has_argument(path: Path, flag: str) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == flag
        for node in ast.walk(tree)
    )


def test_pufferlib_production_realism_defaults_match_project_standard() -> None:
    assert PRODUCTION_DECISION_LAG == 2
    assert PRODUCTION_FEE_BPS == 10.0
    assert PRODUCTION_FILL_BUFFER_BPS == 5.0
    assert PRODUCTION_SHORT_BORROW_APR == 0.0625


def test_gpu_trading_env_cost_defaults_match_pufferlib_realism() -> None:
    assert gpu_trading_env.PRODUCTION_FEE_BPS == PRODUCTION_FEE_BPS
    assert gpu_trading_env.PRODUCTION_FILL_BUFFER_BPS == PRODUCTION_FILL_BUFFER_BPS


def test_low_lag_validation_requires_explicit_diagnostic_opt_in() -> None:
    assert require_production_decision_lag(2, allow_low_lag_diagnostics=False) == 2
    assert require_production_decision_lag(0, allow_low_lag_diagnostics=True) == 0
    with pytest.raises(ValueError, match="decision_lag below 2 requires --allow-low-lag-diagnostics"):
        require_production_decision_lag(1, allow_low_lag_diagnostics=False)
    with pytest.raises(ValueError, match="--decision-lag must be >= 0"):
        require_production_decision_lag(-1, allow_low_lag_diagnostics=True)


@pytest.mark.parametrize("path", SHORT_BORROW_EVALUATOR_ENTRYPOINTS)
def test_pufferlib_evaluator_cli_defaults_use_production_short_borrow_apr(path: Path) -> None:
    defaults = _argument_defaults(path, "--short-borrow-apr")

    assert len(defaults) == 1
    default = defaults[0]
    assert isinstance(default, ast.Name)
    assert default.id == "PRODUCTION_SHORT_BORROW_APR"


@pytest.mark.parametrize("path", DECISION_LAG_EVALUATOR_ENTRYPOINTS)
def test_pufferlib_evaluator_cli_defaults_use_production_decision_lag(path: Path) -> None:
    defaults = _argument_defaults(path, "--decision-lag")

    assert len(defaults) == 1
    default = defaults[0]
    assert isinstance(default, ast.Name)
    assert default.id == "PRODUCTION_DECISION_LAG"


@pytest.mark.parametrize("path", DECISION_LAG_EVALUATOR_ENTRYPOINTS)
def test_low_lag_diagnostics_require_explicit_cli_flag(path: Path) -> None:
    assert _has_argument(path, "--allow-low-lag-diagnostics")
