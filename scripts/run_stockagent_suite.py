from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import typer

from stock.state import get_state_dir
from stockagent.reporting import (
    SummaryError,
    format_summary,
    load_state_snapshot,
    summarize_trades,
)

try:
    import pytest  # noqa: WPS433
except ImportError as exc:  # pragma: no cover - pytest should be installed
    raise SystemExit("pytest is required for run_stockagent_suite") from exc


app = typer.Typer(help="Run stockagent test suites and print summarized PnL telemetry.")


@dataclass(frozen=True)
class SuiteConfig:
    tests: Sequence[str]
    default_suffix: Optional[str] = "sim"
    description: str = ""


SUITES: dict[str, SuiteConfig] = {
    "stockagent": SuiteConfig(
        tests=("tests/prod/agents/stockagent",),
        description="Stateful GPT-5 planner harness.",
    ),
    "stockagentindependant": SuiteConfig(
        tests=("tests/prod/agents/stockagentindependant",),
        description="Stateless plan generator checks.",
    ),
    "stockagent2": SuiteConfig(
        tests=("tests/prod/agents/stockagent2",),
        description="Experimental second-generation agent tests.",
    ),
    "stockagentcombined": SuiteConfig(
        tests=(
            "tests/prod/agents/stockagentcombined/test_stockagentcombined.py",
            "tests/prod/agents/stockagentcombined/test_stockagentcombined_plans.py",
            "tests/prod/agents/stockagentcombined/test_stockagentcombined_cli.py",
            "tests/prod/agents/stockagentcombined/test_stockagentcombined_entrytakeprofit.py",
            "tests/prod/agents/stockagentcombined/test_stockagentcombined_profit_shutdown.py",
        ),
        description="Combined planner + executor regression tests.",
    ),
}


def _resolve_suites(selected: Sequence[str]) -> tuple[List[str], dict[str, str]]:
    if not selected:
        return ["stockagent"], {}
    overrides: dict[str, str] = {}
    entries: List[str] = []
    for token in selected:
        if token == "all":
            entries.extend(name for name in SUITES if name not in entries)
            continue
        name, _, suffix = token.partition(":")
        if name == "all":
            raise typer.BadParameter("Custom suffix overrides are not supported with 'all'.")
        if name not in SUITES:
            valid = ", ".join(sorted(SUITES))
            raise typer.BadParameter(f"Unknown suite '{name}'. Valid options: {valid}, all")
        if name not in entries:
            entries.append(name)
        if suffix:
            overrides[name] = suffix
    return entries, overrides


def _unknown_suites(selected: Sequence[str]) -> List[str]:
    unknown = []
    for token in selected:
        name = token.split(":", 1)[0]
        if name != "all" and name not in SUITES:
            unknown.append(name)
    return unknown


def _ensure_valid(selected: Sequence[str]) -> None:
    unknown = _unknown_suites(selected)
    if unknown:
        valid = ", ".join(sorted(SUITES))
        raise typer.BadParameter(f"Unknown suite(s): {', '.join(unknown)}. Valid options: {valid}, all")


def _run_pytest(paths: Sequence[str], extra_args: Sequence[str]) -> int:
    args = list(paths) + list(extra_args)
    typer.echo(f"[pytest] Running {' '.join(args) or 'default arguments'}")
    return pytest.main(args)


def _render_summary(
    suite_name: str,
    *,
    state_suffix: Optional[str],
    state_dir: Optional[Path],
    overrides: dict[str, str],
) -> str:
    config = SUITES[suite_name]
    suffix = overrides.get(suite_name, state_suffix if state_suffix is not None else config.default_suffix)
    snapshot = load_state_snapshot(state_dir=state_dir, state_suffix=suffix)
    directory_value = snapshot.get("__directory__")
    directory = Path(directory_value) if isinstance(directory_value, str) else (state_dir or get_state_dir())
    summary = summarize_trades(snapshot=snapshot, directory=directory, suffix=suffix)
    return format_summary(summary, label=suite_name)


@app.command()
def main(
    suite: List[str] = typer.Option(
        None,
        "--suite",
        "-s",
        help="Test suite(s) to execute (stockagent, stockagentindependant, stockagent2, stockagentcombined, all). "
        "Use NAME:SUFFIX to override the state suffix for a specific suite.",
    ),
    pytest_arg: List[str] = typer.Option(
        None,
        "--pytest-arg",
        help="Additional arguments forwarded to pytest (use multiple --pytest-arg entries).",
    ),
    state_suffix: Optional[str] = typer.Option(
        None,
        "--state-suffix",
        help="Explicit state suffix override (defaults to suite configuration / environment).",
    ),
    state_dir: Optional[Path] = typer.Option(
        None,
        "--state-dir",
        help="Override the strategy_state directory to read results from.",
    ),
    skip_tests: bool = typer.Option(
        False,
        "--skip-tests",
        help="Skip pytest execution and only print the summaries.",
    ),
) -> None:
    _ensure_valid(suite or ["stockagent"])
    suites, overrides = _resolve_suites(suite or ["stockagent"])
    extra_args = pytest_arg if pytest_arg else []

    exit_code = 0
    if not skip_tests:
        test_paths: list[str] = []
        for name in suites:
            config = SUITES[name]
            test_paths.extend(config.tests)
        exit_code = _run_pytest(test_paths, extra_args)
        if exit_code != 0:
            typer.secho(f"Pytest returned exit code {exit_code}", fg=typer.colors.RED)

    for name in suites:
        typer.echo("")
        typer.secho(f"=== {name} summary ===", fg=typer.colors.CYAN)
        try:
            summary_text = _render_summary(name, state_suffix=state_suffix, state_dir=state_dir, overrides=overrides)
            typer.echo(summary_text)
        except SummaryError as exc:
            typer.secho(f"Summary unavailable: {exc}", fg=typer.colors.YELLOW)

    if exit_code != 0:
        raise typer.Exit(exit_code)


def entrypoint() -> None:
    app()


if __name__ == "__main__":
    entrypoint()
