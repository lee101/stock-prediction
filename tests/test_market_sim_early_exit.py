from __future__ import annotations

import ast
import importlib
import runpy
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import src.market_sim_early_exit as market_sim_early_exit
from src.market_sim_early_exit import (
    evaluate_baseline_comparability_early_exit,
    evaluate_drawdown_vs_profit_early_exit,
)


def test_early_exit_waits_until_halfway() -> None:
    decision = evaluate_drawdown_vs_profit_early_exit(
        [100.0, 110.0, 105.0],
        total_steps=10,
        label="unit",
    )
    assert not decision.should_stop


def test_early_exit_ignores_tiny_windows() -> None:
    decision = evaluate_drawdown_vs_profit_early_exit(
        [100.0, 80.0, 70.0],
        total_steps=3,
        label="unit",
    )
    assert not decision.should_stop


def test_early_exit_triggers_when_drawdown_exceeds_profit_after_halfway() -> None:
    decision = evaluate_drawdown_vs_profit_early_exit(
        [100.0, 140.0, 120.0, 105.0],
        total_steps=4,
        min_total_steps=4,
        label="unit",
    )
    assert decision.should_stop
    assert "max drawdown" in decision.reason
    assert decision.max_drawdown > decision.total_return


def test_baseline_comparability_early_exit_triggers_for_clearly_weaker_path() -> None:
    decision = evaluate_baseline_comparability_early_exit(
        [100.0, 92.0, 88.0, 84.0],
        total_steps=4,
        min_total_steps=4,
        label="unit",
        periods_per_year=24.0 * 365.0,
        baseline_total_return=0.05,
        baseline_sortino=1.0,
        baseline_max_drawdown=0.02,
        stage1_progress=0.25,
        stage2_progress=0.50,
        stage3_progress=0.75,
        return_tolerance=0.01,
        sortino_tolerance=0.25,
        max_drawdown_tolerance=0.01,
    )
    assert decision.should_stop
    assert "baseline gate" in decision.reason
    assert decision.max_drawdown > 0.03


def test_fast_marketsim_eval_import_does_not_disable_global_early_exit() -> None:
    importlib.import_module("fast_marketsim_eval")
    importlib.import_module("pufferlib_market.fast_marketsim_eval")

    decision = market_sim_early_exit.evaluate_drawdown_vs_profit_early_exit(
        [100.0, 140.0, 120.0, 105.0],
        total_steps=4,
        min_total_steps=4,
        label="unit",
    )

    assert decision.should_stop


def test_root_market_sim_early_exit_aliases_src_module() -> None:
    sys.modules.pop("market_sim_early_exit", None)
    root_module = importlib.import_module("market_sim_early_exit")

    assert root_module is market_sim_early_exit
    assert sys.modules["market_sim_early_exit"] is market_sim_early_exit


def test_root_market_sim_early_exit_monkeypatches_canonical_module() -> None:
    sys.modules.pop("market_sim_early_exit", None)
    root_module = importlib.import_module("market_sim_early_exit")
    original = market_sim_early_exit.evaluate_metric_threshold_early_exit

    def _disabled(*_args, **_kwargs):
        return market_sim_early_exit.EarlyExitDecision(False, 0.0, 0.0, 0.0)

    try:
        root_module.evaluate_metric_threshold_early_exit = _disabled
        assert market_sim_early_exit.evaluate_metric_threshold_early_exit is _disabled
    finally:
        market_sim_early_exit.evaluate_metric_threshold_early_exit = original


def test_root_market_sim_early_exit_alias_is_stable_when_imported_first() -> None:
    code = """
from market_sim_early_exit import EarlyExitDecision
import market_sim_early_exit
import src.market_sim_early_exit as src_module
assert market_sim_early_exit is src_module
assert EarlyExitDecision is src_module.EarlyExitDecision
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_root_market_sim_early_exit_shim_exposes_public_api_under_runpy() -> None:
    repo = Path(__file__).resolve().parent.parent

    namespace = runpy.run_path(str(repo / "market_sim_early_exit.py"))

    assert namespace["EarlyExitDecision"] is market_sim_early_exit.EarlyExitDecision
    assert (
        namespace["evaluate_metric_threshold_early_exit"]
        is market_sim_early_exit.evaluate_metric_threshold_early_exit
    )


def test_pytest_collect_hook_restores_legacy_early_exit_monkeypatch(
    restore_market_sim_early_exit_functions: Callable[[], None],
) -> None:
    original = market_sim_early_exit.evaluate_drawdown_vs_profit_early_exit

    def _disabled(*_args, **_kwargs):
        return market_sim_early_exit.EarlyExitDecision(False, 0.0, 0.0, 0.0)

    market_sim_early_exit.evaluate_drawdown_vs_profit_early_exit = _disabled
    restore_market_sim_early_exit_functions()

    assert market_sim_early_exit.evaluate_drawdown_vs_profit_early_exit is original


def test_eval_scripts_do_not_monkeypatch_early_exit_at_import_time() -> None:
    repo = Path(__file__).resolve().parent.parent
    script_paths = [
        repo / "comprehensive_marketsim_eval.py",
        repo / "eval_daily_with_hourly_replay.py",
        repo / "eval_ensemble_marketsim.py",
        repo / "eval_long_daily.py",
        repo / "eval_stocks12_seeds.py",
        repo / "full_120d_marketsim_compare.py",
        repo / "reeval_crypto_and_leverage.py",
        repo / "sweep_crypto10_reg_combo.py",
        repo / "sweep_mixed32_tp_wd.py",
        repo / "sweep_stocks40_daily.py",
        repo / "sweep_stocks_leverage_daily.py",
        repo / "sweep_tp_fine_grain.py",
        repo / "sweep_tp15_seeds.py",
        repo / "verify_mass_daily_seeds.py",
        repo / "scripts" / "eval_stocks12_seeds.py",
    ]
    patched_names = {
        "evaluate_drawdown_vs_profit_early_exit",
        "evaluate_metric_threshold_early_exit",
    }

    offenders: list[str] = []
    for path in script_paths:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in tree.body:
            targets: list[ast.expr] = []
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            elif isinstance(node, ast.AugAssign):
                targets = [node.target]

            for target in targets:
                if isinstance(target, ast.Attribute) and target.attr in patched_names:
                    offenders.append(f"{path.relative_to(repo)}:{node.lineno}")

    assert offenders == []
