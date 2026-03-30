"""CLI ergonomics tests for binance_worksteal entrypoints."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binance_worksteal import backtest, evaluate_symbols, reporting, sim_vs_live_audit, sweep, sweep_expanded
from binance_worksteal.cli import (
    add_date_range_args,
    build_data_coverage_warnings,
    build_load_summary_warnings,
    build_partial_universe_hint,
    build_strict_retry_command,
    build_require_full_universe_error,
    build_symbol_selection_cli_args,
    load_bars_with_summary,
    print_data_coverage_summary,
    print_loaded_symbol_summary,
    print_window_span_coverage_summary,
    resolve_paired_date_range_or_print_error,
    resolve_cli_symbols,
    resolve_cli_symbols_or_print_error,
    summarize_loaded_symbols,
    validate_date_range_or_print_error,
)
from binance_worksteal.config_io import default_best_config_path, default_best_overrides_path
from binance_worksteal.reporting import (
    announce_sweep_artifacts,
    build_empty_sweep_run_summary,
    build_preview_run_summary,
    build_sweep_follow_up_commands,
    build_sweep_run_summary,
    default_next_steps_script_path,
    default_sidecar_json_path,
    prepare_sweep_recommendation_artifacts,
    print_warning_summary,
    run_with_optional_summary,
    write_summary_json_or_warn,
)
from binance_worksteal.strategy import TradeLog, WorkStealConfig


def _write_universe(tmp_path: Path) -> Path:
    path = tmp_path / "universe.yaml"
    path.write_text(
        "symbols:\n"
        "  - symbol: BTCUSD\n"
        "  - symbol: ETHUSD\n",
        encoding="utf-8",
    )
    return path


def test_resolve_cli_symbols_prefers_explicit_symbols(tmp_path):
    universe_file = _write_universe(tmp_path)

    symbols, source = resolve_cli_symbols(
        symbols_arg=["solusd", " adausd "],
        universe_file=str(universe_file),
        default_symbols=["BTCUSD"],
    )

    assert symbols == ["SOLUSD", "ADAUSD"]
    assert source == "command line --symbols"


def test_resolve_cli_symbols_or_print_error_returns_none_on_invalid_input(tmp_path, capsys):
    missing_path = tmp_path / "missing.yaml"

    resolved = resolve_cli_symbols_or_print_error(
        symbols_arg=None,
        universe_file=str(missing_path),
        default_symbols=["BTCUSD"],
    )

    out = capsys.readouterr().out.strip()
    assert resolved is None
    assert out == f"ERROR: Universe file not found: {missing_path}"


def test_build_symbol_selection_cli_args_prefers_symbols():
    args = build_symbol_selection_cli_args(
        symbols_arg=["solusd", " adausd "],
        universe_file="ignored.yaml",
    )

    assert args == ["--symbols", "SOLUSD", "ADAUSD"]


def test_build_sweep_follow_up_commands_includes_symbol_eval_for_fixed_window():
    commands = build_sweep_follow_up_commands(
        config_file="best.yaml",
        data_dir="trainingdata/train",
        symbol_args=["--symbols", "BTCUSD"],
        start_date="2026-01-01",
        end_date="2026-01-31",
        eval_days=None,
        eval_windows=None,
        eval_start_date="2026-01-01",
        eval_end_date="2026-01-31",
    )

    assert [item["name"] for item in commands] == ["backtest", "sim_vs_live_audit", "evaluate_symbols"]
    assert commands[-1]["argv"][-4:] == ["--start", "2026-01-01", "--end", "2026-01-31"]



def test_build_symbol_listing_summary_always_includes_config_file_key():
    payload = reporting.build_symbol_listing_summary(
        tool="example",
        symbol_source="built-in default universe",
        symbols=["BTCUSD", "ETHUSD"],
        data_dir="trainingdata/train",
        config_file=None,
    )

    assert payload["tool"] == "example"
    assert payload["list_symbols_only"] is True
    assert payload["data_dir"] == "trainingdata/train"
    assert "config_file" in payload
    assert payload["config_file"] is None


def test_build_cli_error_summary_always_includes_config_file_key():
    payload = reporting.build_cli_error_summary(
        tool="example",
        error="ERROR: boom",
        config_file=None,
    )

    assert payload["tool"] == "example"
    assert payload["error"] == "ERROR: boom"
    assert "config_file" in payload
    assert payload["config_file"] is None



def test_build_preview_and_listing_summaries_share_symbol_selection_fields():
    preview = build_preview_run_summary(
        tool="example",
        data_dir="trainingdata/train",
        symbol_source="command line --symbols",
        symbols=["BTCUSD"],
        config_file=None,
        requested_symbol_count=3,
    )
    listing = reporting.build_symbol_listing_summary(
        tool="example",
        data_dir="trainingdata/train",
        symbol_source="command line --symbols",
        symbols=["BTCUSD"],
        config_file=None,
        requested_symbol_count=3,
    )

    for field in ["tool", "data_dir", "symbol_source", "requested_symbol_count", "symbol_count", "symbols", "config_file"]:
        assert listing[field] == preview[field]
    assert preview["requested_symbol_count"] == 3
    assert preview["symbol_count"] == 1

def test_prepare_sweep_recommendation_artifacts_writes_best_config_and_next_steps(tmp_path):
    output_csv = tmp_path / "sweep_results.csv"
    recommendation = prepare_sweep_recommendation_artifacts(
        ranked_results=[{"dip_pct": 0.18, "profit_target_pct": 0.12}],
        base_config=WorkStealConfig(initial_cash=5000.0, sma_filter_period=20),
        swept_fields=["dip_pct", "profit_target_pct"],
        output_csv=output_csv,
        data_dir="trainingdata/train",
        symbols_arg=["btcusd"],
        universe_file=None,
        start_date="2026-01-01",
        end_date="2026-02-01",
        eval_days=60,
        eval_windows=3,
    )

    best_config_path = default_best_config_path(output_csv)
    best_overrides_path = default_best_overrides_path(output_csv)
    next_steps_script_path = default_next_steps_script_path(output_csv)
    assert recommendation["recommended_config_file"] == str(best_config_path)
    assert recommendation["recommended_config"].dip_pct == pytest.approx(0.18)
    assert recommendation["recommended_config"].profit_target_pct == pytest.approx(0.12)
    assert recommendation["recommended_overrides_file"] == str(best_overrides_path)
    assert recommendation["recommended_overrides"]["initial_cash"] == pytest.approx(5000.0)
    assert recommendation["follow_up_config_file"] == str(best_overrides_path)
    assert recommendation["follow_up_config_kind"] == "standalone_overrides"
    assert recommendation["next_steps_script_file"] == str(next_steps_script_path)
    assert best_config_path.exists()
    assert best_overrides_path.exists()
    assert next_steps_script_path.exists()
    assert next_steps_script_path.stat().st_mode & 0o111
    payload = yaml.safe_load(best_config_path.read_text(encoding="utf-8"))
    assert payload["config"]["dip_pct"] == pytest.approx(0.18)
    overrides_payload = yaml.safe_load(best_overrides_path.read_text(encoding="utf-8"))
    assert overrides_payload["config"]["dip_pct"] == pytest.approx(0.18)
    assert overrides_payload["config"]["initial_cash"] == pytest.approx(5000.0)
    assert [item["name"] for item in recommendation["next_steps"]] == [
        "backtest",
        "sim_vs_live_audit",
        "evaluate_symbols",
    ]
    assert "--config-file" in recommendation["next_steps"][0]["argv"]
    assert str(best_overrides_path) in recommendation["next_steps"][0]["argv"]
    next_steps_script = next_steps_script_path.read_text(encoding="utf-8")
    assert next_steps_script.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert recommendation["next_steps"][0]["command"] in next_steps_script


def test_prepare_sweep_recommendation_artifacts_builds_fixed_window_eval_follow_up(tmp_path):
    output_csv = tmp_path / "sweep_results.csv"
    recommendation = prepare_sweep_recommendation_artifacts(
        ranked_results=[{"dip_pct": 0.18}],
        base_config=WorkStealConfig(initial_cash=5000.0),
        swept_fields=["dip_pct"],
        output_csv=output_csv,
        data_dir="trainingdata/train",
        symbols_arg=["btcusd"],
        universe_file=None,
        start_date="2026-01-01",
        end_date="2026-02-01",
        eval_start_date="2026-01-01",
        eval_end_date="2026-02-01",
    )

    assert [item["name"] for item in recommendation["next_steps"]] == [
        "backtest",
        "sim_vs_live_audit",
        "evaluate_symbols",
    ]
    assert recommendation["next_steps"][-1]["argv"][-4:] == [
        "--start",
        "2026-01-01",
        "--end",
        "2026-02-01",
    ]
    assert recommendation["next_steps_script_file"] == str(default_next_steps_script_path(output_csv))
    fixed_window_script = default_next_steps_script_path(output_csv).read_text(encoding="utf-8")
    assert "--start 2026-01-01 --end 2026-02-01" in fixed_window_script


def test_print_warning_summary_lists_warnings(capsys):
    print_warning_summary(["missing data for 1 symbol: ETHUSD", "1 symbol starts after requested range start: ETHUSD"])

    out = capsys.readouterr().out
    assert "Warnings:" in out
    assert "  - missing data for 1 symbol: ETHUSD" in out
    assert "  - 1 symbol starts after requested range start: ETHUSD" in out


def test_prepare_sweep_recommendation_artifacts_returns_empty_when_no_results(tmp_path):
    recommendation = prepare_sweep_recommendation_artifacts(
        ranked_results=[],
        base_config=WorkStealConfig(),
        swept_fields=["dip_pct"],
        output_csv=tmp_path / "sweep_results.csv",
        data_dir="trainingdata/train",
        symbols_arg=["BTCUSD"],
        universe_file=None,
        start_date="2026-01-01",
        end_date="2026-02-01",
    )

    assert recommendation == {
        "recommended_config": None,
        "recommended_config_file": None,
        "recommended_overrides": None,
        "recommended_overrides_file": None,
        "follow_up_config_file": None,
        "follow_up_config_kind": None,
        "next_steps_script_file": None,
        "next_steps": [],
    }


def test_write_next_steps_script_or_warn_reports_write_failures(tmp_path, monkeypatch, capsys):
    output_csv = tmp_path / "sweep_results.csv"
    script_path = default_next_steps_script_path(output_csv)

    monkeypatch.setattr(reporting, "write_text_atomic", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")))

    written = reporting.write_next_steps_script_or_warn(
        output_csv,
        [{
            "name": "backtest",
            "description": "Replay the recommended config.",
            "command": "python -m binance_worksteal.backtest --config-file best.yaml",
        }],
    )

    out = capsys.readouterr().out.strip()
    assert written is None
    assert out == f"WARN: failed to write next steps shell script to {script_path}: disk full"



def test_build_sweep_artifact_manifest_includes_generated_follow_up_files():
    artifacts = reporting.build_sweep_artifact_manifest(
        recommendation={
            "recommended_config_file": "results.best_config.yaml",
            "recommended_overrides_file": "results.best_overrides.yaml",
            "next_steps_script_file": "results.next_steps.sh",
        },
        output_csv="results.csv",
        include_output_csv=True,
        summary_json_file="results.summary.json",
    )

    assert artifacts == [
        {
            "name": "summary_json",
            "path": "results.summary.json",
            "description": "Structured JSON run summary.",
        },
        {
            "name": "results_csv",
            "path": "results.csv",
            "description": "Sweep results CSV.",
        },
        {
            "name": "recommended_config",
            "path": "results.best_config.yaml",
            "description": "Full best-config YAML snapshot.",
        },
        {
            "name": "recommended_overrides",
            "path": "results.best_overrides.yaml",
            "description": "Minimal best-config overrides YAML.",
        },
        {
            "name": "next_steps_script",
            "path": "results.next_steps.sh",
            "description": "Runnable shell script with suggested follow-up commands.",
        },
    ]


def test_announce_sweep_artifacts_prints_manifest_and_optional_reproduce(capsys):
    artifacts = announce_sweep_artifacts(
        recommendation={
            "recommended_config_file": "results.best_config.yaml",
            "recommended_overrides_file": None,
            "next_steps_script_file": None,
        },
        output_csv="results.csv",
        summary_json_file="results.summary.json",
        module="binance_worksteal.sweep",
        argv=["--symbols", "BTCUSD", "--output", "results.csv"],
        include_output_csv=True,
    )

    out = capsys.readouterr().out
    assert artifacts == [
        {
            "name": "summary_json",
            "path": "results.summary.json",
            "description": "Structured JSON run summary.",
        },
        {
            "name": "results_csv",
            "path": "results.csv",
            "description": "Sweep results CSV.",
        },
        {
            "name": "recommended_config",
            "path": "results.best_config.yaml",
            "description": "Full best-config YAML snapshot.",
        },
    ]
    assert "Generated artifacts:" in out
    assert "  summary_json: results.summary.json" in out
    assert "  results_csv: results.csv" in out
    assert "Reproduce:" in out
    assert "python -m binance_worksteal.sweep --symbols BTCUSD --output results.csv" in out


def test_build_sweep_run_summary_merges_common_and_extra_fields():
    payload = build_sweep_run_summary(
        tool="sweep",
        data_dir="trainingdata/train",
        symbol_source="command line --symbols",
        symbols=["BTCUSD", "ETHUSD"],
        load_summary={
            "loaded_symbol_count": 1,
            "loaded_symbols": ["BTCUSD"],
            "missing_symbol_count": 1,
            "missing_symbols": ["ETHUSD"],
        },
        data_coverage={"available_date_span": {"start_date": "2026-01-01", "end_date": "2026-01-31"}},
        config_file="best.yaml",
        base_config=WorkStealConfig(initial_cash=2500.0),
        output_csv="results.csv",
        swept_fields=["dip_pct", "profit_target_pct"],
        windows=[("2026-01-01", "2026-01-31")],
        results_count=3,
        recommendation={
            "recommended_config": {"dip_pct": 0.15},
            "recommended_config_file": "results.best_config.yaml",
            "recommended_overrides": {"dip_pct": 0.15},
            "recommended_overrides_file": "results.best_overrides.yaml",
            "follow_up_config_file": "results.best_overrides.yaml",
            "follow_up_config_kind": "standalone_overrides",
            "next_steps_script_file": "results.next_steps.sh",
            "next_steps": [{"name": "backtest", "command": "python -m ..."}],
        },
        best_result={"dip_pct": 0.15, "min_sortino": 1.2},
        top_results=[{"dip_pct": 0.15, "min_sortino": 1.2}],
        extra={"worker_count": 4},
    )

    assert payload["tool"] == "sweep"
    assert payload["output_csv"] == "results.csv"
    assert payload["swept_fields"] == ["dip_pct", "profit_target_pct"]
    assert payload["windows"] == [{"start_date": "2026-01-01", "end_date": "2026-01-31"}]
    assert payload["results_count"] == 3
    assert payload["recommended_overrides_file"] == "results.best_overrides.yaml"
    assert payload["follow_up_config_kind"] == "standalone_overrides"
    assert payload["next_steps_script_file"] == "results.next_steps.sh"
    assert payload["next_steps"] == [{"name": "backtest", "command": "python -m ..."}]
    assert [item["name"] for item in payload["artifacts"]] == [
        "results_csv",
        "recommended_config",
        "recommended_overrides",
        "next_steps_script",
    ]
    assert [item["path"] for item in payload["artifacts"]] == [
        "results.csv",
        "results.best_config.yaml",
        "results.best_overrides.yaml",
        "results.next_steps.sh",
    ]
    assert payload["best_result"] == {"dip_pct": 0.15, "min_sortino": 1.2}
    assert payload["top_results"] == [{"dip_pct": 0.15, "min_sortino": 1.2}]
    assert payload["worker_count"] == 4
    assert payload["warnings"] == ["missing data for 1 symbol: ETHUSD"]
    assert payload["loaded_symbol_count"] == 1
    assert payload["missing_symbols"] == ["ETHUSD"]
    assert payload["partial_universe_hint"] == (
        "Hint: rerun with --require-full-universe to fail instead of continuing with a partial universe."
    )


def test_build_empty_sweep_run_summary_returns_shared_empty_shape():
    payload = build_empty_sweep_run_summary(
        tool="sweep",
        data_dir="trainingdata/train",
        symbol_source="command line --symbols",
        symbols=["BTCUSD", "ETHUSD"],
        load_summary={
            "loaded_symbol_count": 1,
            "loaded_symbols": ["BTCUSD"],
            "missing_symbol_count": 1,
            "missing_symbols": ["ETHUSD"],
            "universe_complete": False,
        },
        data_coverage=None,
        config_file=None,
        base_config=WorkStealConfig(initial_cash=2500.0),
        output_csv="results.csv",
        swept_fields=["dip_pct"],
        extra={"error": "ERROR: missing data", "worker_count": 4},
    )

    assert payload["results_count"] == 0
    assert payload["windows"] == []
    assert payload["recommended_config"] is None
    assert payload["recommended_overrides"] is None
    assert payload["follow_up_config_file"] is None
    assert payload["next_steps"] == []
    assert payload["error"] == "ERROR: missing data"
    assert payload["worker_count"] == 4


def test_print_loaded_symbol_summary_reports_missing_symbols(capsys):
    summary = print_loaded_symbol_summary(["BTCUSD", "ETHUSD", "SOLUSD"], {"BTCUSD": object(), "SOLUSD": object()})

    out = capsys.readouterr().out.strip().splitlines()
    assert out == [
        "Loaded 2/3 symbols with data",
        "WARN: missing data for 1 symbol: ETHUSD",
        "Hint: rerun with --require-full-universe to fail instead of continuing with a partial universe.",
    ]
    assert summary["loaded_symbols"] == ["BTCUSD", "SOLUSD"]
    assert summary["missing_symbols"] == ["ETHUSD"]
    assert summary["universe_complete"] is False
    assert summary["partial_universe_hint"] == (
        "Hint: rerun with --require-full-universe to fail instead of continuing with a partial universe."
    )


def test_print_loaded_symbol_summary_includes_retry_command_when_provided(capsys):
    retry_command = build_strict_retry_command(
        module="binance_worksteal.backtest",
        argv=["--symbols", "BTCUSD", "ETHUSD"],
    )

    summary = print_loaded_symbol_summary(
        ["BTCUSD", "ETHUSD", "SOLUSD"],
        {"BTCUSD": object(), "SOLUSD": object()},
        strict_retry_command=retry_command,
    )

    out = capsys.readouterr().out.strip().splitlines()
    assert out == [
        "Loaded 2/3 symbols with data",
        "WARN: missing data for 1 symbol: ETHUSD",
        "Hint: rerun with --require-full-universe to fail instead of continuing with a partial universe.",
        "Retry command: python -m binance_worksteal.backtest --symbols BTCUSD ETHUSD --require-full-universe",
    ]
    assert summary["strict_retry_command"] == retry_command


def test_build_require_full_universe_error_uses_missing_preview():
    summary = summarize_loaded_symbols(["BTCUSD", "ETHUSD", "SOLUSD"], ["BTCUSD"])

    message = build_require_full_universe_error(summary)

    assert message == "ERROR: --require-full-universe found missing data for 2 symbols: ETHUSD, SOLUSD"


def test_build_partial_universe_hint_only_for_partial_loads():
    partial_summary = summarize_loaded_symbols(["BTCUSD", "ETHUSD"], ["BTCUSD"])
    empty_summary = summarize_loaded_symbols(["BTCUSD", "ETHUSD"], [])
    complete_summary = summarize_loaded_symbols(["BTCUSD", "ETHUSD"], ["BTCUSD", "ETHUSD"])

    assert build_partial_universe_hint(partial_summary) == (
        "Hint: rerun with --require-full-universe to fail instead of continuing with a partial universe."
    )
    assert build_partial_universe_hint(
        partial_summary,
        strict_retry_command="python -m binance_worksteal.backtest --symbols BTCUSD ETHUSD --require-full-universe",
    ) == (
        "Hint: rerun with --require-full-universe to fail instead of continuing with a partial universe.\n"
        "Retry command: python -m binance_worksteal.backtest --symbols BTCUSD ETHUSD --require-full-universe"
    )
    assert build_partial_universe_hint(empty_summary) is None
    assert build_partial_universe_hint(complete_summary) is None



def test_build_load_summary_warnings_formats_missing_symbols():
    summary = summarize_loaded_symbols(["BTCUSD", "ETHUSD", "SOLUSD"], ["BTCUSD"])

    warnings = build_load_summary_warnings(summary)

    assert warnings == ["missing data for 2 symbols: ETHUSD, SOLUSD"]



def test_build_data_coverage_warnings_formats_partial_range_coverage():
    warnings = build_data_coverage_warnings(
        {
            "coverage_range_label": "window span",
            "coverage_start_date": "2026-01-01",
            "coverage_end_date": "2026-01-05",
            "symbols_missing_range_start": ["ETHUSD"],
            "symbols_missing_range_end": ["SOLUSD"],
        }
    )

    assert warnings == [
        "1 symbol starts after window span start: ETHUSD",
        "1 symbol ends before window span end: SOLUSD",
    ]


def test_load_bars_with_summary_reports_empty_data(capsys):
    def fake_loader(data_dir, symbols):
        assert data_dir == "trainingdata/train"
        assert list(symbols) == ["BTCUSD", "ETHUSD"]
        return {}

    loaded = load_bars_with_summary(
        data_dir="trainingdata/train",
        requested_symbols=["BTCUSD", "ETHUSD"],
        load_bars=fake_loader,
        loading_message="Loading test symbols",
        no_data_message="ERROR: No data",
    )

    out = capsys.readouterr().out.strip().splitlines()
    assert loaded is None
    assert out == [
        "Loading test symbols",
        "Loaded 0/2 symbols with data",
        "WARN: missing data for 2 symbols: BTCUSD, ETHUSD",
        "ERROR: No data",
    ]


def test_load_bars_with_summary_can_return_summary_on_empty(capsys):
    def fake_loader(data_dir, symbols):
        assert data_dir == "trainingdata/train"
        assert list(symbols) == ["BTCUSD", "ETHUSD"]
        return {}

    loaded = load_bars_with_summary(
        data_dir="trainingdata/train",
        requested_symbols=["BTCUSD", "ETHUSD"],
        load_bars=fake_loader,
        loading_message="Loading test symbols",
        no_data_message="ERROR: No data",
        return_summary_on_empty=True,
    )

    out = capsys.readouterr().out.strip().splitlines()
    assert loaded is not None
    loaded_bars, load_summary = loaded
    assert loaded_bars == {}
    assert load_summary["requested_symbol_count"] == 2
    assert load_summary["loaded_symbol_count"] == 0
    assert load_summary["loaded_symbols"] == []
    assert load_summary["missing_symbol_count"] == 2
    assert load_summary["missing_symbols"] == ["BTCUSD", "ETHUSD"]
    assert load_summary["universe_complete"] is False
    assert out == [
        "Loading test symbols",
        "Loaded 0/2 symbols with data",
        "WARN: missing data for 2 symbols: BTCUSD, ETHUSD",
        "ERROR: No data",
    ]


def test_load_bars_with_summary_returns_consistent_triple_when_empty_and_failure_enabled(capsys):
    def fake_loader(data_dir, symbols):
        assert data_dir == "trainingdata/train"
        assert list(symbols) == ["BTCUSD", "ETHUSD"]
        return {}

    loaded_bars, load_summary, load_failure = load_bars_with_summary(
        data_dir="trainingdata/train",
        requested_symbols=["BTCUSD", "ETHUSD"],
        load_bars=fake_loader,
        loading_message="Loading test symbols",
        no_data_message="ERROR: No data",
        return_summary_on_empty=True,
        return_failure_on_error=True,
    )

    out = capsys.readouterr().out.strip().splitlines()
    assert loaded_bars == {}
    assert load_summary["requested_symbol_count"] == 2
    assert load_summary["loaded_symbol_count"] == 0
    assert load_summary["missing_symbols"] == ["BTCUSD", "ETHUSD"]
    assert load_failure is None
    assert out == [
        "Loading test symbols",
        "Loaded 0/2 symbols with data",
        "WARN: missing data for 2 symbols: BTCUSD, ETHUSD",
        "ERROR: No data",
    ]


def test_load_bars_with_summary_returns_consistent_triple_on_success_when_empty_and_failure_enabled(capsys):
    def fake_loader(data_dir, symbols):
        assert data_dir == "trainingdata/train"
        assert list(symbols) == ["BTCUSD", "ETHUSD"]
        return {"BTCUSD": object()}

    loaded_bars, load_summary, load_failure = load_bars_with_summary(
        data_dir="trainingdata/train",
        requested_symbols=["BTCUSD", "ETHUSD"],
        load_bars=fake_loader,
        loading_message="Loading test symbols",
        no_data_message="ERROR: No data",
        return_summary_on_empty=True,
        return_failure_on_error=True,
    )

    out = capsys.readouterr().out.strip().splitlines()
    assert list(loaded_bars) == ["BTCUSD"]
    assert load_summary["loaded_symbol_count"] == 1
    assert load_summary["missing_symbols"] == ["ETHUSD"]
    assert load_failure is None
    assert out == [
        "Loading test symbols",
        "Loaded 1/2 symbols with data",
        "WARN: missing data for 1 symbol: ETHUSD",
        "Hint: rerun with --require-full-universe to fail instead of continuing with a partial universe.",
    ]


def test_load_bars_with_summary_can_return_failure_on_loader_error(capsys):
    def fake_loader(data_dir, symbols):
        assert data_dir == "trainingdata/train"
        assert list(symbols) == ["BTCUSD", "ETHUSD"]
        raise OSError("disk failure")

    loaded = load_bars_with_summary(
        data_dir="trainingdata/train",
        requested_symbols=["BTCUSD", "ETHUSD"],
        load_bars=fake_loader,
        loading_message="Loading test symbols",
        no_data_message="ERROR: No data",
        return_failure_on_error=True,
    )

    out = capsys.readouterr().out.strip().splitlines()
    assert loaded is not None
    loaded_bars, load_summary, load_failure = loaded
    assert loaded_bars == {}
    assert load_summary["requested_symbol_count"] == 2
    assert load_summary["loaded_symbol_count"] == 0
    assert load_summary["missing_symbols"] == ["BTCUSD", "ETHUSD"]
    assert load_failure == {"error": "ERROR: disk failure", "error_type": "OSError"}
    assert out == [
        "Loading test symbols",
        "ERROR: disk failure",
    ]

def test_load_bars_with_summary_returns_consistent_triple_on_loader_error_when_empty_and_failure_enabled(capsys):
    def fake_loader(data_dir, symbols):
        assert data_dir == "trainingdata/train"
        assert list(symbols) == ["BTCUSD", "ETHUSD"]
        raise OSError("disk failure")

    loaded_bars, load_summary, load_failure = load_bars_with_summary(
        data_dir="trainingdata/train",
        requested_symbols=["BTCUSD", "ETHUSD"],
        load_bars=fake_loader,
        loading_message="Loading test symbols",
        no_data_message="ERROR: No data",
        return_summary_on_empty=True,
        return_failure_on_error=True,
    )

    out = capsys.readouterr().out.strip().splitlines()
    assert loaded_bars == {}
    assert load_summary["requested_symbol_count"] == 2
    assert load_summary["loaded_symbol_count"] == 0
    assert load_summary["missing_symbols"] == ["BTCUSD", "ETHUSD"]
    assert load_failure == {"error": "ERROR: disk failure", "error_type": "OSError"}
    assert out == [
        "Loading test symbols",
        "ERROR: disk failure",
    ]


def test_load_bars_with_summary_returns_failure_on_invalid_loader_result(capsys):
    def fake_loader(data_dir, symbols):
        assert data_dir == "trainingdata/train"
        assert list(symbols) == ["BTCUSD", "ETHUSD"]
        return []

    loaded_bars, load_summary, load_failure = load_bars_with_summary(
        data_dir="trainingdata/train",
        requested_symbols=["BTCUSD", "ETHUSD"],
        load_bars=fake_loader,
        loading_message="Loading test symbols",
        no_data_message="ERROR: No data",
        return_summary_on_empty=True,
        return_failure_on_error=True,
        strict_retry_command="python -m binance_worksteal.backtest --symbols BTCUSD ETHUSD --require-full-universe",
    )

    out = capsys.readouterr().out.strip().splitlines()
    assert loaded_bars == {}
    assert load_summary["requested_symbol_count"] == 2
    assert load_summary["loaded_symbol_count"] == 0
    assert load_summary["missing_symbols"] == ["BTCUSD", "ETHUSD"]
    assert load_summary["strict_retry_command"] == (
        "python -m binance_worksteal.backtest --symbols BTCUSD ETHUSD --require-full-universe"
    )
    assert load_failure == {
        "error": "ERROR: load_bars returned list, expected a symbol->bars mapping",
        "error_type": "TypeError",
    }
    assert out == [
        "Loading test symbols",
        "ERROR: load_bars returned list, expected a symbol->bars mapping",
    ]


def test_print_data_coverage_summary_reports_partial_range_coverage(capsys):
    bars = {
        "BTCUSD": pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=5, tz="UTC", freq="D"),
            }
        ),
        "ETHUSD": pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-03", periods=2, tz="UTC", freq="D"),
            }
        ),
    }

    summary = print_data_coverage_summary(
        bars,
        start_date="2026-01-02",
        end_date="2026-01-05",
    )

    out = capsys.readouterr().out.strip().splitlines()
    assert out == [
        "Data coverage: 2026-01-01 to 2026-01-05 across 2 loaded symbols",
        "Requested range coverage: 1/2 loaded symbols fully cover 2026-01-02 to 2026-01-05",
        "WARN: 1 symbol starts after requested range start: ETHUSD",
        "WARN: 1 symbol ends before requested range end: ETHUSD",
    ]
    assert summary["overall_start_date"] == "2026-01-01"
    assert summary["overall_end_date"] == "2026-01-05"
    assert summary["symbols_covering_range_count"] == 1
    assert summary["symbols_missing_range_start"] == ["ETHUSD"]
    assert summary["symbols_missing_range_end"] == ["ETHUSD"]


def test_print_window_span_coverage_summary_uses_combined_window_range(capsys):
    bars = {
        "BTCUSD": pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=5, tz="UTC", freq="D"),
            }
        ),
        "ETHUSD": pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-02", periods=3, tz="UTC", freq="D"),
            }
        ),
    }

    summary = print_window_span_coverage_summary(
        bars,
        [("2026-01-02", "2026-01-04"), ("2026-01-01", "2026-01-05")],
    )

    out = capsys.readouterr().out.strip().splitlines()
    assert out == [
        "Data coverage: 2026-01-01 to 2026-01-05 across 2 loaded symbols",
        "Window span coverage: 1/2 loaded symbols fully cover 2026-01-01 to 2026-01-05",
        "WARN: 1 symbol starts after window span start: ETHUSD",
        "WARN: 1 symbol ends before window span end: ETHUSD",
    ]
    assert summary["symbols_covering_range_count"] == 1
    assert summary["symbols_missing_range_start"] == ["ETHUSD"]
    assert summary["symbols_missing_range_end"] == ["ETHUSD"]


def test_add_date_range_args_accepts_short_aliases():
    parser = argparse.ArgumentParser()
    add_date_range_args(parser, include_days=True, days_default=60)

    args = parser.parse_args(["--start", "2026-01-01", "--end", "2026-01-31"])

    assert args.start_date == "2026-01-01"
    assert args.end_date == "2026-01-31"
    assert args.days == 60


def test_resolve_paired_date_range_or_print_error_requires_both_bounds(capsys):
    resolved = resolve_paired_date_range_or_print_error(start_date="2026-01-01", end_date=None)

    out = capsys.readouterr().out.strip()
    assert resolved is None
    assert out == "ERROR: --start/--start-date and --end/--end-date must be provided together."


def test_validate_date_range_or_print_error_rejects_invalid_start_date(capsys):
    resolved = validate_date_range_or_print_error(start_date="not-a-date", end_date="2026-01-31")

    out = capsys.readouterr().out.strip()
    assert resolved is None
    assert out == "ERROR: Invalid --start/--start-date value: 'not-a-date'"


def test_validate_date_range_or_print_error_rejects_reversed_range(capsys):
    resolved = validate_date_range_or_print_error(start_date="2026-02-01", end_date="2026-01-31")

    out = capsys.readouterr().out.strip()
    assert resolved is None
    assert out == "ERROR: --start/--start-date must be on or before --end/--end-date."


@pytest.mark.parametrize(
    "entrypoint",
    [
        backtest.main,
        sweep.main,
        sweep_expanded.main,
        evaluate_symbols.main,
        sim_vs_live_audit.main,
    ],
)
def test_list_symbols_uses_universe_file(entrypoint, tmp_path, capsys):
    universe_file = _write_universe(tmp_path)

    rc = entrypoint(["--universe-file", str(universe_file), "--list-symbols"])

    out = capsys.readouterr().out.strip().splitlines()
    assert rc == 0
    assert out[0] == f"Resolved 2 symbols from universe file {universe_file}:"
    assert out[1:] == ["BTCUSD", "ETHUSD"]


def test_backtest_list_symbols_prefers_symbols_over_universe_file(tmp_path, capsys):
    universe_file = _write_universe(tmp_path)

    rc = backtest.main(
        ["--universe-file", str(universe_file), "--symbols", "solusd", "dogeusd", "--list-symbols"]
    )

    out = capsys.readouterr().out.strip().splitlines()
    assert rc == 0
    assert out[0] == "Resolved 2 symbols from command line --symbols:"
    assert out[1:] == ["SOLUSD", "DOGEUSD"]


def test_backtest_list_symbols_supports_string_format_universe_file(tmp_path, capsys):
    universe_file = tmp_path / "universe.yaml"
    universe_file.write_text("symbols:\n  - btcusdt\n  - solusd\n", encoding="utf-8")

    rc = backtest.main(["--universe-file", str(universe_file), "--list-symbols"])

    out = capsys.readouterr().out.strip().splitlines()
    assert rc == 0
    assert out[0] == f"Resolved 2 symbols from universe file {universe_file}:"
    assert out[1:] == ["BTCUSD", "SOLUSD"]


@pytest.mark.parametrize("entrypoint", [sweep.main, sweep_expanded.main])
def test_sweep_list_symbols_does_not_write_default_sidecar(entrypoint, tmp_path, capsys):
    universe_file = _write_universe(tmp_path)
    output_csv = tmp_path / "results.csv"
    sidecar = default_sidecar_json_path(output_csv)

    rc = entrypoint([
        "--universe-file",
        str(universe_file),
        "--list-symbols",
        "--output",
        str(output_csv),
    ])

    out = capsys.readouterr().out.strip().splitlines()
    assert rc == 0
    assert out[0] == f"Resolved 2 symbols from universe file {universe_file}:"
    assert out[1:] == ["BTCUSD", "ETHUSD"]
    assert not sidecar.exists()


@pytest.mark.parametrize(
    ("entrypoint", "tool", "module_name", "output_name", "summary_name"),
    [
        (sweep.main, "sweep", "binance_worksteal.sweep", "results.csv", "symbols_summary.json"),
        (
            sweep_expanded.main,
            "sweep_expanded",
            "binance_worksteal.sweep_expanded",
            "expanded.csv",
            "expanded_symbols_summary.json",
        ),
    ],
)
def test_sweep_list_symbols_summary_file_prints_artifacts(
    entrypoint,
    tool,
    module_name,
    output_name,
    summary_name,
    tmp_path,
    capsys,
):
    universe_file = _write_universe(tmp_path)
    output_csv = tmp_path / output_name
    summary_path = tmp_path / summary_name

    rc = entrypoint([
        "--universe-file",
        str(universe_file),
        "--list-symbols",
        "--output",
        str(output_csv),
        "--summary-json",
        str(summary_path),
    ])

    out = capsys.readouterr().out
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert f"Resolved 2 symbols from universe file {universe_file}:" in out
    assert "Generated artifacts:" in out
    assert f"  summary_json: {summary_path}" in out
    assert "Reproduce:" in out
    assert payload["invocation"]["command"] in out
    assert "Wrote summary JSON" not in out
    assert payload["tool"] == tool
    assert payload["list_symbols_only"] is True
    assert payload["summary_json_file"] == str(summary_path)
    assert payload["artifacts"] == [
        {
            "name": "summary_json",
            "path": str(summary_path),
            "description": "Structured JSON run summary.",
        }
    ]
    assert payload["invocation"]["module"] == module_name


def test_backtest_list_symbols_summary_dash_prints_json_to_stdout(tmp_path, capsys):
    universe_file = _write_universe(tmp_path)
    config_path = tmp_path / "backtest.yaml"
    config_path.write_text("dip_pct: 0.18\n", encoding="utf-8")

    rc = backtest.main([
        "--universe-file",
        str(universe_file),
        "--config-file",
        str(config_path),
        "--list-symbols",
        "--summary-json",
        "-",
    ])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 0
    assert payload["tool"] == "backtest"
    assert payload["list_symbols_only"] is True
    assert payload["symbol_source"] == f"universe file {universe_file}"
    assert payload["symbols"] == ["BTCUSD", "ETHUSD"]
    assert payload["data_dir"] == "trainingdata/train"
    assert payload["config_file"] == str(config_path)
    assert payload["status"] == "success"
    assert payload["exit_code"] == 0
    assert payload["invocation"]["module"] == "binance_worksteal.backtest"
    assert payload["invocation"]["argv"][-2:] == ["--summary-json", "-"]
    assert captured.err.splitlines()[0] == f"Resolved 2 symbols from universe file {universe_file}:"


def test_evaluate_symbols_list_symbols_reports_effective_and_ignored_candidate_symbols(tmp_path, capsys):
    universe_file = _write_universe(tmp_path)
    config_path = tmp_path / "evaluate.yaml"
    config_path.write_text("dip_pct: 0.18\n", encoding="utf-8")

    rc = evaluate_symbols.main([
        "--universe-file",
        str(universe_file),
        "--config-file",
        str(config_path),
        "--candidate-symbols",
        "solusd",
        "ethusd",
        "--list-symbols",
        "--summary-json",
        "-",
    ])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    err_lines = captured.err.strip().splitlines()
    assert rc == 0
    assert payload["tool"] == "evaluate_symbols"
    assert payload["list_symbols_only"] is True
    assert payload["config_file"] == str(config_path)
    assert payload["symbols"] == ["BTCUSD", "ETHUSD"]
    assert payload["candidate_symbols"] == ["SOLUSD"]
    assert payload["candidate_symbol_count"] == 1
    assert payload["candidate_symbol_source"] == "command line --candidate-symbols"
    assert payload["ignored_candidate_symbols"] == ["ETHUSD"]
    assert payload["ignored_candidate_symbol_count"] == 1
    assert payload["status"] == "success"
    assert payload["exit_code"] == 0
    assert err_lines[0] == f"Resolved 2 symbols from universe file {universe_file}:"
    assert "Resolved 1 symbols from command line --candidate-symbols:" in err_lines
    assert "SOLUSD" in err_lines
    assert "Ignored 1 candidate symbols already in base universe: ETHUSD" in err_lines


def test_evaluate_symbols_invalid_config_summary_dash_includes_candidate_context(tmp_path, capsys):
    config_path = tmp_path / "evaluate_config.yaml"
    config_path.write_text("unknown_field: 1\n", encoding="utf-8")

    rc = evaluate_symbols.main([
        "--symbols",
        "BTCUSD",
        "ETHUSD",
        "--candidate-symbols",
        "ethusd",
        "solusd",
        "--config-file",
        str(config_path),
        "--summary-json",
        "-",
    ])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 1
    assert payload["tool"] == "evaluate_symbols"
    assert payload["error"] == f"ERROR: Unsupported WorkStealConfig fields in {config_path}: unknown_field"
    assert payload["error_type"] == "ValueError"
    assert payload["config_file"] == str(config_path)
    assert payload["candidate_symbols"] == ["SOLUSD"]
    assert payload["candidate_symbol_count"] == 1
    assert payload["candidate_symbol_source"] == "command line --candidate-symbols"
    assert payload["ignored_candidate_symbols"] == ["ETHUSD"]
    assert payload["ignored_candidate_symbol_count"] == 1
    assert payload["status"] == "error"
    assert payload["exit_code"] == 1
    assert captured.err.strip() == f"ERROR: Unsupported WorkStealConfig fields in {config_path}: unknown_field"


def test_evaluate_symbols_invalid_date_summary_dash_includes_candidate_context(capsys):
    rc = evaluate_symbols.main([
        "--symbols",
        "BTCUSD",
        "--candidate-symbols",
        "solusd",
        "--start",
        "bad-date",
        "--end",
        "2026-01-31",
        "--summary-json",
        "-",
    ])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 1
    assert payload["tool"] == "evaluate_symbols"
    assert payload["error"] == "ERROR: Invalid --start/--start-date value: 'bad-date'"
    assert payload["error_type"] == "ValueError"
    assert payload["config_file"] is None
    assert payload["candidate_symbols"] == ["SOLUSD"]
    assert payload["candidate_symbol_count"] == 1
    assert payload["candidate_symbol_source"] == "command line --candidate-symbols"
    assert payload["ignored_candidate_symbols"] == []
    assert payload["ignored_candidate_symbol_count"] == 0
    assert payload["status"] == "error"
    assert payload["exit_code"] == 1
    assert captured.err.strip() == "ERROR: Invalid --start/--start-date value: 'bad-date'"


@pytest.mark.parametrize(
    "entrypoint",
    [
        backtest.main,
        sweep.main,
        sweep_expanded.main,
        evaluate_symbols.main,
        sim_vs_live_audit.main,
    ],
)
def test_invalid_universe_file_returns_error(entrypoint, tmp_path, capsys):
    missing_path = tmp_path / "missing.yaml"

    rc = entrypoint(["--universe-file", str(missing_path), "--list-symbols"])

    out = capsys.readouterr().out.strip()
    assert rc == 1
    assert out == f"ERROR: Universe file not found: {missing_path}"


@pytest.mark.parametrize(
    "entrypoint",
    [
        backtest.main,
        sweep.main,
        sweep_expanded.main,
        evaluate_symbols.main,
        sim_vs_live_audit.main,
    ],
)
def test_malformed_universe_file_returns_error(entrypoint, tmp_path, capsys):
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text("symbols: [BTCUSD\n", encoding="utf-8")

    rc = entrypoint(["--universe-file", str(bad_path), "--list-symbols"])

    out = capsys.readouterr().out.strip()
    assert rc == 1
    assert out.startswith(f"ERROR: Invalid universe YAML in {bad_path}:")


@pytest.mark.parametrize(
    "entrypoint",
    [
        backtest.main,
        sweep.main,
        sweep_expanded.main,
        evaluate_symbols.main,
        sim_vs_live_audit.main,
    ],
)
def test_invalid_universe_entry_returns_error(entrypoint, tmp_path, capsys):
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text("symbols:\n  - fee_tier: usdt\n", encoding="utf-8")

    rc = entrypoint(["--universe-file", str(bad_path), "--list-symbols"])

    out = capsys.readouterr().out.strip()
    assert rc == 1
    assert out.startswith(f"ERROR: Symbol entry missing 'symbol' field at index 0 in {bad_path}:")


@pytest.mark.parametrize(
    ("entrypoint", "tool", "module_name"),
    [
        (backtest.main, "backtest", "binance_worksteal.backtest"),
        (sweep.main, "sweep", "binance_worksteal.sweep"),
        (sweep_expanded.main, "sweep_expanded", "binance_worksteal.sweep_expanded"),
        (evaluate_symbols.main, "evaluate_symbols", "binance_worksteal.evaluate_symbols"),
        (sim_vs_live_audit.main, "sim_vs_live_audit", "binance_worksteal.sim_vs_live_audit"),
    ],
)
def test_invalid_universe_file_summary_dash_writes_structured_error(entrypoint, tool, module_name, tmp_path, capsys):
    missing_path = tmp_path / "missing.yaml"

    rc = entrypoint(["--universe-file", str(missing_path), "--list-symbols", "--summary-json", "-"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 1
    assert payload["tool"] == tool
    assert payload["error"] == f"ERROR: Universe file not found: {missing_path}"
    assert payload["error_type"] == "FileNotFoundError"
    assert payload["config_file"] is None
    assert payload["list_symbols_only"] is True
    assert payload["status"] == "error"
    assert payload["exit_code"] == 1
    assert payload["invocation"]["module"] == module_name
    assert captured.err.strip() == f"ERROR: Universe file not found: {missing_path}"


@pytest.mark.parametrize(
    ("entrypoint", "tool", "module_name"),
    [
        (backtest.main, "backtest", "binance_worksteal.backtest"),
        (sweep.main, "sweep", "binance_worksteal.sweep"),
        (sweep_expanded.main, "sweep_expanded", "binance_worksteal.sweep_expanded"),
        (evaluate_symbols.main, "evaluate_symbols", "binance_worksteal.evaluate_symbols"),
        (sim_vs_live_audit.main, "sim_vs_live_audit", "binance_worksteal.sim_vs_live_audit"),
    ],
)
def test_invalid_universe_entry_summary_dash_writes_structured_error(entrypoint, tool, module_name, tmp_path, capsys):
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text("symbols:\n  - fee_tier: usdt\n", encoding="utf-8")

    rc = entrypoint(["--universe-file", str(bad_path), "--list-symbols", "--summary-json", "-"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 1
    assert payload["tool"] == tool
    assert payload["error"].startswith(
        f"ERROR: Symbol entry missing 'symbol' field at index 0 in {bad_path}:"
    )
    assert payload["error_type"] == "ValueError"
    assert payload["list_symbols_only"] is True
    assert payload["status"] == "error"
    assert payload["exit_code"] == 1
    assert payload["invocation"]["module"] == module_name
    assert captured.err.strip().startswith(
        f"ERROR: Symbol entry missing 'symbol' field at index 0 in {bad_path}:"
    )


@pytest.mark.parametrize(
    ("entrypoint", "tool", "module_name"),
    [
        (backtest.main, "backtest", "binance_worksteal.backtest"),
        (sweep.main, "sweep", "binance_worksteal.sweep"),
        (sweep_expanded.main, "sweep_expanded", "binance_worksteal.sweep_expanded"),
        (evaluate_symbols.main, "evaluate_symbols", "binance_worksteal.evaluate_symbols"),
        (sim_vs_live_audit.main, "sim_vs_live_audit", "binance_worksteal.sim_vs_live_audit"),
    ],
)
def test_invalid_universe_boolean_field_summary_dash_writes_structured_error(entrypoint, tool, module_name, tmp_path, capsys):
    bad_path = tmp_path / "bad_bool.yaml"
    bad_path.write_text(
        "symbols:\n"
        "  - symbol: BTCUSD\n"
        "    margin_eligible: maybe\n",
        encoding="utf-8",
    )

    rc = entrypoint(["--universe-file", str(bad_path), "--list-symbols", "--summary-json", "-"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 1
    assert payload["tool"] == tool
    assert payload["error"] == f"ERROR: Invalid margin_eligible for BTCUSD in {bad_path}: 'maybe'"
    assert payload["error_type"] == "ValueError"
    assert payload["list_symbols_only"] is True
    assert payload["status"] == "error"
    assert payload["exit_code"] == 1
    assert payload["invocation"]["module"] == module_name
    assert captured.err.strip() == f"ERROR: Invalid margin_eligible for BTCUSD in {bad_path}: 'maybe'"


@pytest.mark.parametrize(
    ("entrypoint", "tool", "output_name"),
    [
        (sweep.main, "sweep", "sweep_results.csv"),
        (sweep_expanded.main, "sweep_expanded", "sweep_expanded.csv"),
    ],
)
def test_sweep_invalid_universe_file_writes_default_error_sidecar(entrypoint, tool, output_name, tmp_path, capsys):
    missing_path = tmp_path / "missing.yaml"
    output_csv = tmp_path / output_name

    rc = entrypoint(["--universe-file", str(missing_path), "--output", str(output_csv)])

    captured = capsys.readouterr()
    summary_path = default_sidecar_json_path(output_csv)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 1
    assert payload["tool"] == tool
    assert payload["error"] == f"ERROR: Universe file not found: {missing_path}"
    assert payload["error_type"] == "FileNotFoundError"
    assert payload["output_csv"] == str(output_csv)
    assert payload["status"] == "error"
    assert payload["exit_code"] == 1
    assert captured.out.splitlines()[0] == f"ERROR: Universe file not found: {missing_path}"


@pytest.mark.parametrize(
    ("module", "entrypoint"),
    [
        (backtest, backtest.main),
        (evaluate_symbols, evaluate_symbols.main),
        (sim_vs_live_audit, sim_vs_live_audit.main),
    ],
)
def test_print_config_exits_before_loading_data(module, entrypoint, tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "dip_pct: 0.18\n"
        "sma_filter_period: 7\n",
        encoding="utf-8",
    )

    def fail_load(*args, **kwargs):
        raise AssertionError("load_daily_bars should not run for --print-config")

    monkeypatch.setattr(module, "load_daily_bars", fail_load)

    rc = entrypoint(["--print-config", "--config-file", str(config_path), "--dip-pct", "0.25"])

    out = capsys.readouterr().out
    payload = yaml.safe_load(out)
    assert rc == 0
    assert payload["config"]["dip_pct"] == pytest.approx(0.25)
    assert payload["config"]["sma_filter_period"] == 7


@pytest.mark.parametrize(
    ("module", "entrypoint"),
    [
        (backtest, backtest.main),
        (sim_vs_live_audit, sim_vs_live_audit.main),
    ],
)
def test_print_config_ignores_invalid_runtime_dates(module, entrypoint, monkeypatch, capsys):
    def fail_load(*args, **kwargs):
        raise AssertionError("load_daily_bars should not run for --print-config")

    monkeypatch.setattr(module, "load_daily_bars", fail_load)

    rc = entrypoint(["--print-config", "--start", "bad-date"])

    out = capsys.readouterr().out
    payload = yaml.safe_load(out)
    assert rc == 0
    assert "config" in payload


@pytest.mark.parametrize(
    ("module", "entrypoint"),
    [
        (backtest, backtest.main),
        (evaluate_symbols, evaluate_symbols.main),
        (sim_vs_live_audit, sim_vs_live_audit.main),
    ],
)
def test_explain_config_exits_before_loading_data(module, entrypoint, tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "dip_pct: 0.18\n"
        "sma_filter_period: 7\n",
        encoding="utf-8",
    )

    def fail_load(*args, **kwargs):
        raise AssertionError("load_daily_bars should not run for --explain-config")

    monkeypatch.setattr(module, "load_daily_bars", fail_load)

    rc = entrypoint(["--explain-config", "--config-file", str(config_path), "--dip-pct", "0.25"])

    out = capsys.readouterr().out
    payload = yaml.safe_load(out)
    assert rc == 0
    assert payload["config"]["dip_pct"] == pytest.approx(0.25)
    assert payload["sources"]["dip_pct"] == "cli"
    assert payload["sources"]["sma_filter_period"] == "config_file"
    assert payload["changed_fields"]["dip_pct"]["config_file_value"] == pytest.approx(0.18)


@pytest.mark.parametrize(
    ("module", "entrypoint"),
    [
        (sweep, sweep.main),
        (sweep_expanded, sweep_expanded.main),
    ],
)
def test_sweep_print_config_exits_before_loading_data(module, entrypoint, tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "dip_pct: 0.18\n"
        "max_positions: 7\n",
        encoding="utf-8",
    )

    def fail_load(*args, **kwargs):
        raise AssertionError("load_daily_bars should not run for --print-config")

    monkeypatch.setattr(module, "load_daily_bars", fail_load)

    rc = entrypoint(
        ["--print-config", "--config-file", str(config_path), "--cash", "25000", "--realistic"]
        if entrypoint is sweep.main
        else ["--print-config", "--config-file", str(config_path), "--cash", "25000"]
    )

    out = capsys.readouterr().out
    payload = yaml.safe_load(out)
    assert rc == 0
    assert payload["config"]["dip_pct"] == pytest.approx(0.18)
    assert payload["config"]["max_positions"] == 7
    assert payload["config"]["initial_cash"] == pytest.approx(25000.0)
    if entrypoint is sweep.main:
        assert payload["config"]["realistic_fill"] is True
        assert payload["config"]["daily_checkpoint_only"] is True


def test_sweep_explain_config_shows_realistic_flag_sources(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("dip_pct: 0.18\n", encoding="utf-8")

    def fail_load(*args, **kwargs):
        raise AssertionError("load_daily_bars should not run for --explain-config")

    monkeypatch.setattr(sweep, "load_daily_bars", fail_load)

    rc = sweep.main(
        ["--explain-config", "--config-file", str(config_path), "--cash", "25000", "--realistic"]
    )

    out = capsys.readouterr().out
    payload = yaml.safe_load(out)
    assert rc == 0
    assert payload["sources"]["realistic_fill"] == "cli"
    assert payload["sources"]["daily_checkpoint_only"] == "cli"
    assert payload["changed_fields"]["realistic_fill"]["value"] is True
    assert payload["changed_fields"]["daily_checkpoint_only"]["value"] is True


def test_backtest_writes_summary_json(tmp_path, monkeypatch, capsys):
    summary_path = tmp_path / "backtest_summary.json"
    bars = {
        "BTCUSD": pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-01-31", tz="UTC")],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000.0],
                "symbol": ["BTCUSD"],
            }
        )
    }
    trades = [
        TradeLog(
            timestamp=pd.Timestamp("2026-01-31", tz="UTC"),
            symbol="BTCUSD",
            side="buy",
            price=90.0,
            quantity=1.0,
            notional=90.0,
            fee=0.09,
            reason="dip_buy",
            direction="long",
        ),
        TradeLog(
            timestamp=pd.Timestamp("2026-02-01", tz="UTC"),
            symbol="BTCUSD",
            side="sell",
            price=95.0,
            quantity=1.0,
            notional=95.0,
            fee=0.095,
            pnl=4.815,
            reason="profit_target",
            direction="long",
        ),
    ]

    monkeypatch.setattr(backtest, "load_daily_bars", lambda data_dir, symbols: bars)
    monkeypatch.setattr(
        backtest,
        "run_worksteal_backtest",
        lambda *args, **kwargs: (
            pd.DataFrame({"timestamp": [pd.Timestamp("2026-01-31", tz="UTC")], "equity": [10050.0]}),
            trades,
            {"final_equity": 10050.0, "total_return_pct": 0.5, "n_days": 2},
        ),
    )
    monkeypatch.setattr(backtest, "print_results", lambda *args, **kwargs: None)

    rc = backtest.main(["--symbols", "BTCUSD", "--summary-json", str(summary_path)])

    out = capsys.readouterr().out
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert "Generated artifacts:" in out
    assert f"  summary_json: {summary_path}" in out
    assert "Reproduce:" in out
    assert payload["invocation"]["command"] in out
    assert "Wrote summary JSON" not in out
    assert payload["summary_schema_version"] == 1
    assert payload["tool"] == "backtest"
    assert payload["requested_symbol_count"] == 1
    assert payload["symbol_count"] == 1
    assert payload["metrics"]["final_equity"] == 10050.0
    assert payload["trade_counts"]["entries"] == 1
    assert payload["per_symbol_pnl"]["BTCUSD"] == pytest.approx(4.815)


def test_backtest_require_full_universe_returns_error_and_summary(tmp_path, monkeypatch, capsys):
    summary_path = tmp_path / "backtest_summary.json"
    bars = {
        "BTCUSD": pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-01-31", tz="UTC")],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000.0],
                "symbol": ["BTCUSD"],
            }
        )
    }

    monkeypatch.setattr(backtest, "load_daily_bars", lambda data_dir, symbols: bars)

    def fail_run(*args, **kwargs):
        raise AssertionError("run_worksteal_backtest should not run when --require-full-universe fails")

    monkeypatch.setattr(backtest, "run_worksteal_backtest", fail_run)

    rc = backtest.main([
        "--symbols", "BTCUSD", "ETHUSD",
        "--require-full-universe",
        "--summary-json", str(summary_path),
    ])

    out = capsys.readouterr().out.strip().splitlines()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 1
    assert "ERROR: --require-full-universe found missing data for 1 symbol: ETHUSD" in out
    assert "Diagnostic artifacts:" in out
    assert f"  summary_json: {summary_path}" in out
    assert "Reproduce:" in out
    assert "Wrote summary JSON" not in "\n".join(out)
    assert payload["require_full_universe"] is True
    assert payload["universe_complete"] is False
    assert payload["missing_symbols"] == ["ETHUSD"]
    assert payload["error"] == "ERROR: --require-full-universe found missing data for 1 symbol: ETHUSD"
    assert payload["exit_code"] == 1
    assert payload["status"] == "error"


def test_backtest_writes_summary_json_on_no_data(tmp_path, monkeypatch, capsys):
    summary_path = tmp_path / "backtest_summary.json"

    monkeypatch.setattr(backtest, "load_daily_bars", lambda data_dir, symbols: {})

    def fail_run(*args, **kwargs):
        raise AssertionError("run_worksteal_backtest should not run when no data loads")

    monkeypatch.setattr(backtest, "run_worksteal_backtest", fail_run)

    rc = backtest.main(["--symbols", "BTCUSD", "--summary-json", str(summary_path)])

    out = capsys.readouterr().out.strip().splitlines()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 1
    assert "ERROR: No data loaded" in out
    assert "Diagnostic artifacts:" in out
    assert f"  summary_json: {summary_path}" in out
    assert "Reproduce:" in out
    assert "Wrote summary JSON" not in "\n".join(out)
    assert payload["loaded_symbol_count"] == 0
    assert payload["missing_symbols"] == ["BTCUSD"]
    assert payload["error"] == "ERROR: No data loaded"
    assert payload["artifacts"] == [
        {
            "name": "summary_json",
            "path": str(summary_path),
            "description": "Structured JSON run summary.",
        }
    ]
    assert payload["exit_code"] == 1
    assert payload["status"] == "error"


def test_backtest_writes_summary_json_on_loader_error(tmp_path, monkeypatch, capsys):
    summary_path = tmp_path / "backtest_summary.json"

    def raise_load_error(data_dir, symbols):
        raise OSError("bad backtest dir")

    monkeypatch.setattr(backtest, "load_daily_bars", raise_load_error)

    def fail_run(*args, **kwargs):
        raise AssertionError("run_worksteal_backtest should not run when load_daily_bars fails")

    monkeypatch.setattr(backtest, "run_worksteal_backtest", fail_run)

    rc = backtest.main(["--symbols", "BTCUSD", "--summary-json", str(summary_path)])

    out = capsys.readouterr().out.strip().splitlines()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 1
    assert "ERROR: bad backtest dir" in out
    assert "Diagnostic artifacts:" in out
    assert f"  summary_json: {summary_path}" in out
    assert "Reproduce:" in out
    assert "Wrote summary JSON" not in "\n".join(out)
    assert payload["loaded_symbol_count"] == 0
    assert payload["missing_symbols"] == ["BTCUSD"]
    assert payload["error"] == "ERROR: bad backtest dir"
    assert payload["load_failure"] == {"error": "ERROR: bad backtest dir", "error_type": "OSError"}
    assert payload["exit_code"] == 1
    assert payload["status"] == "error"


def test_backtest_writes_summary_json_on_invalid_loader_result(tmp_path, monkeypatch, capsys):
    summary_path = tmp_path / "backtest_summary.json"

    monkeypatch.setattr(backtest, "load_daily_bars", lambda data_dir, symbols: [])

    def fail_run(*args, **kwargs):
        raise AssertionError("run_worksteal_backtest should not run when load_daily_bars returns an invalid type")

    monkeypatch.setattr(backtest, "run_worksteal_backtest", fail_run)

    rc = backtest.main(["--symbols", "BTCUSD", "--summary-json", str(summary_path)])

    out = capsys.readouterr().out.strip().splitlines()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 1
    assert "ERROR: load_bars returned list, expected a symbol->bars mapping" in out
    assert "Diagnostic artifacts:" in out
    assert f"  summary_json: {summary_path}" in out
    assert "Reproduce:" in out
    assert "Wrote summary JSON" not in "\n".join(out)
    assert payload["loaded_symbol_count"] == 0
    assert payload["missing_symbols"] == ["BTCUSD"]
    assert payload["error"] == "ERROR: load_bars returned list, expected a symbol->bars mapping"
    assert payload["load_failure"] == {
        "error": "ERROR: load_bars returned list, expected a symbol->bars mapping",
        "error_type": "TypeError",
    }
    assert payload["strict_retry_command"] == (
        "python -m binance_worksteal.backtest --symbols BTCUSD --summary-json "
        f"{summary_path} --require-full-universe"
    )
    assert payload["exit_code"] == 1
    assert payload["status"] == "error"


def test_backtest_reports_missing_symbols_in_output_and_summary(tmp_path, monkeypatch, capsys):
    summary_path = tmp_path / "backtest_summary.json"
    bars = {
        "BTCUSD": pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-01-31", tz="UTC")],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000.0],
                "symbol": ["BTCUSD"],
            }
        )
    }

    monkeypatch.setattr(backtest, "load_daily_bars", lambda data_dir, symbols: bars)
    monkeypatch.setattr(
        backtest,
        "run_worksteal_backtest",
        lambda *args, **kwargs: (
            pd.DataFrame({"timestamp": [pd.Timestamp("2026-01-31", tz="UTC")], "equity": [10000.0]}),
            [],
            {"final_equity": 10000.0, "total_return_pct": 0.0, "n_days": 1},
        ),
    )
    monkeypatch.setattr(backtest, "print_results", lambda *args, **kwargs: None)

    rc = backtest.main([
        "--symbols", "BTCUSD", "ETHUSD",
        "--start", "2026-01-31",
        "--end", "2026-01-31",
        "--summary-json", str(summary_path),
    ])

    captured = capsys.readouterr().out
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert "Loaded 1/2 symbols with data" in captured
    assert "WARN: missing data for 1 symbol: ETHUSD" in captured
    assert payload["requested_symbol_count"] == 2
    assert payload["loaded_symbol_count"] == 1
    assert payload["loaded_symbols"] == ["BTCUSD"]
    assert payload["missing_symbol_count"] == 1
    assert payload["missing_symbols"] == ["ETHUSD"]
    assert payload["strict_retry_command"] == (
        "python -m binance_worksteal.backtest --symbols BTCUSD ETHUSD --start 2026-01-31 "
        f"--end 2026-01-31 --summary-json {summary_path} --require-full-universe"
    )
    assert payload["warnings"] == ["missing data for 1 symbol: ETHUSD"]
    assert "Warnings:" in captured
    assert "  - missing data for 1 symbol: ETHUSD" in captured
    assert "Retry command: python -m binance_worksteal.backtest" in captured


def test_backtest_reports_date_coverage_in_output_and_summary(tmp_path, monkeypatch, capsys):
    summary_path = tmp_path / "backtest_summary.json"
    bars = {
        "BTCUSD": pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", "2026-01-03T00:00:00Z"],
                    utc=True,
                ),
                "open": [100.0, 101.0, 102.0],
                "high": [101.0, 102.0, 103.0],
                "low": [99.0, 100.0, 101.0],
                "close": [100.0, 101.0, 102.0],
                "volume": [1000.0, 1000.0, 1000.0],
                "symbol": ["BTCUSD", "BTCUSD", "BTCUSD"],
            }
        ),
        "ETHUSD": pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2026-01-02T00:00:00Z", "2026-01-03T00:00:00Z"],
                    utc=True,
                ),
                "open": [200.0, 201.0],
                "high": [201.0, 202.0],
                "low": [199.0, 200.0],
                "close": [200.0, 201.0],
                "volume": [1000.0, 1000.0],
                "symbol": ["ETHUSD", "ETHUSD"],
            }
        ),
    }

    monkeypatch.setattr(backtest, "load_daily_bars", lambda data_dir, symbols: bars)
    monkeypatch.setattr(
        backtest,
        "run_worksteal_backtest",
        lambda *args, **kwargs: (
            pd.DataFrame({"timestamp": [pd.Timestamp("2026-01-03", tz="UTC")], "equity": [10000.0]}),
            [],
            {"final_equity": 10000.0, "total_return_pct": 0.0, "n_days": 3},
        ),
    )
    monkeypatch.setattr(backtest, "print_results", lambda *args, **kwargs: None)

    rc = backtest.main(
        [
            "--symbols", "BTCUSD", "ETHUSD",
            "--start", "2026-01-01",
            "--end", "2026-01-03",
            "--summary-json", str(summary_path),
        ]
    )

    captured = capsys.readouterr().out
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert "Data coverage: 2026-01-01 to 2026-01-03 across 2 loaded symbols" in captured
    assert "Requested range coverage: 1/2 loaded symbols fully cover 2026-01-01 to 2026-01-03" in captured
    assert "WARN: 1 symbol starts after requested range start: ETHUSD" in captured
    assert payload["data_coverage"]["overall_start_date"] == "2026-01-01"
    assert payload["data_coverage"]["overall_end_date"] == "2026-01-03"
    assert payload["data_coverage"]["symbols_covering_range_count"] == 1
    assert payload["data_coverage"]["symbols_missing_range_start"] == ["ETHUSD"]
    assert payload["warnings"] == ["1 symbol starts after requested range start: ETHUSD"]
    assert "Warnings:" in captured
    assert "  - 1 symbol starts after requested range start: ETHUSD" in captured


def test_backtest_summary_write_failure_is_nonfatal(monkeypatch, capsys):
    bars = {
        "BTCUSD": pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-01-31", tz="UTC")],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000.0],
                "symbol": ["BTCUSD"],
            }
        )
    }

    monkeypatch.setattr(backtest, "load_daily_bars", lambda data_dir, symbols: bars)
    monkeypatch.setattr(
        backtest,
        "run_worksteal_backtest",
        lambda *args, **kwargs: (
            pd.DataFrame({"timestamp": [pd.Timestamp("2026-01-31", tz="UTC")], "equity": [10050.0]}),
            [],
            {"final_equity": 10050.0, "total_return_pct": 0.5, "n_days": 2},
        ),
    )
    monkeypatch.setattr(backtest, "print_results", lambda *args, **kwargs: None)
    monkeypatch.setattr(reporting, "safe_write_summary_json", lambda path, payload: (None, "disk full"))

    rc = backtest.main(["--symbols", "BTCUSD", "--summary-json", "/tmp/fail.json"])

    out = capsys.readouterr().out
    assert rc == 0
    assert "WARN: failed to write summary JSON to /tmp/fail.json: disk full" in out


def test_write_summary_json_or_warn_writes_file(tmp_path, capsys):
    summary_path = tmp_path / "summary.json"

    written_path = write_summary_json_or_warn(summary_path, {"tool": "example", "symbols": ["BTCUSD"]})

    out = capsys.readouterr().out.strip()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert written_path == summary_path
    assert out == f"Wrote summary JSON to {summary_path}"
    assert payload["tool"] == "example"
    assert payload["symbols"] == ["BTCUSD"]


def test_write_summary_json_or_warn_emits_warning_on_failure(monkeypatch, capsys):
    monkeypatch.setattr(reporting, "safe_write_summary_json", lambda path, payload: (None, "disk full"))

    written_path = write_summary_json_or_warn("/tmp/fail.json", {"tool": "example"})

    out = capsys.readouterr().out.strip()
    assert written_path is None
    assert out == "WARN: failed to write summary JSON to /tmp/fail.json: disk full"


def test_write_summary_json_or_warn_writes_to_stdout_when_path_is_dash(capsys):
    written_path = write_summary_json_or_warn("-", {"tool": "example", "symbols": ["BTCUSD"]})

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert written_path is None
    assert captured.err == ""
    assert payload["tool"] == "example"
    assert payload["symbols"] == ["BTCUSD"]
    assert "generated_at_utc" in payload


def test_write_summary_json_or_warn_dash_failure_goes_to_stderr(monkeypatch, capsys):
    monkeypatch.setattr(reporting, "safe_write_summary_json", lambda path, payload: (None, "disk full"))

    written_path = write_summary_json_or_warn("-", {"tool": "example"})

    captured = capsys.readouterr()
    assert written_path is None
    assert captured.out == ""
    assert captured.err.strip() == "WARN: failed to write summary JSON to -: disk full"


def test_run_with_optional_summary_writes_file_and_returns_runner_status(tmp_path, capsys):
    summary_path = tmp_path / "summary.json"

    def runner():
        print("human output")
        return 7, {
            "tool": "example",
            "artifacts": [{"name": "other", "path": "other.txt", "description": "Other artifact."}],
        }

    rc = run_with_optional_summary(
        summary_path,
        runner,
        module="binance_worksteal.example",
        argv=["--symbols", "BTCUSD"],
    )

    captured = capsys.readouterr()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 7
    assert "human output" in captured.out
    assert "Diagnostic artifacts:" in captured.out
    assert f"  summary_json: {summary_path}" in captured.out
    assert "Reproduce:" in captured.out
    assert payload["invocation"]["command"] in captured.out
    assert "Wrote summary JSON" not in captured.out
    assert payload["tool"] == "example"
    assert payload["exit_code"] == 7
    assert payload["status"] == "error"
    assert payload["summary_json_file"] == str(summary_path)
    assert [item["name"] for item in payload["artifacts"]] == ["summary_json", "other"]
    assert payload["invocation"]["module"] == "binance_worksteal.example"
    assert payload["invocation"]["argv"] == ["--symbols", "BTCUSD"]
    assert payload["invocation"]["command"] == "python -m binance_worksteal.example --symbols BTCUSD"
    assert payload["invocation"]["cwd"] == str(Path.cwd())


def test_run_with_optional_summary_adds_summary_artifact_when_payload_has_none(tmp_path, capsys):
    summary_path = tmp_path / "summary.json"

    def runner():
        print("human output")
        return 0, {"tool": "example"}

    rc = run_with_optional_summary(
        summary_path,
        runner,
        module="binance_worksteal.example",
        argv=["--symbols", "BTCUSD"],
        announce_artifact_manifest_on_success=True,
    )

    captured = capsys.readouterr()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert "human output" in captured.out
    assert "Generated artifacts:" in captured.out
    assert f"  summary_json: {summary_path}" in captured.out
    assert "Reproduce:" in captured.out
    assert payload["invocation"]["command"] in captured.out
    assert "Wrote summary JSON" not in captured.out
    assert payload["summary_json_file"] == str(summary_path)
    assert payload["artifacts"] == [
        {
            "name": "summary_json",
            "path": str(summary_path),
            "description": "Structured JSON run summary.",
        }
    ]


def test_run_with_optional_summary_dash_redirects_human_output_to_stderr(capsys):
    def runner():
        print("human output")
        return 0, {"tool": "example"}

    rc = run_with_optional_summary("-", runner)

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 0
    assert payload["tool"] == "example"
    assert payload["exit_code"] == 0
    assert payload["status"] == "success"
    assert captured.err.strip() == "human output"



def test_run_with_optional_summary_catches_runner_exception_and_writes_summary(tmp_path, capsys):
    summary_path = tmp_path / "summary.json"

    def runner():
        print("human output")
        raise RuntimeError("boom")

    rc = run_with_optional_summary(
        summary_path,
        runner,
        module="binance_worksteal.example",
        argv=["--symbols", "BTCUSD"],
    )

    captured = capsys.readouterr()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 1
    assert "human output" in captured.out
    assert "ERROR: boom" in captured.out
    assert "Diagnostic artifacts:" in captured.out
    assert f"  summary_json: {summary_path}" in captured.out
    assert "Reproduce:" in captured.out
    assert payload["invocation"]["command"] in captured.out
    assert "Wrote summary JSON" not in captured.out
    assert payload["tool"] == "example"
    assert payload["error"] == "ERROR: boom"
    assert payload["error_type"] == "RuntimeError"
    assert payload["exit_code"] == 1
    assert payload["status"] == "error"
    assert payload["summary_json_file"] == str(summary_path)
    assert payload["artifacts"] == [
        {
            "name": "summary_json",
            "path": str(summary_path),
            "description": "Structured JSON run summary.",
        }
    ]
    assert payload["invocation"]["module"] == "binance_worksteal.example"



def test_run_with_optional_summary_dash_catches_runner_exception_and_keeps_json_on_stdout(capsys):
    def runner():
        print("human output")
        raise RuntimeError("boom")

    rc = run_with_optional_summary("-", runner, module="binance_worksteal.example", argv=["--symbols", "BTCUSD"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 1
    assert payload["tool"] == "example"
    assert payload["error"] == "ERROR: boom"
    assert payload["error_type"] == "RuntimeError"
    assert payload["exit_code"] == 1
    assert payload["status"] == "error"
    err_lines = captured.err.strip().splitlines()
    assert err_lines == ["human output", "ERROR: boom"]



def test_run_with_optional_summary_rejects_invalid_payload_and_writes_error_summary(tmp_path, capsys):
    summary_path = tmp_path / "summary.json"

    def runner():
        print("human output")
        return 0, ["bad"]

    rc = run_with_optional_summary(
        summary_path,
        runner,
        module="binance_worksteal.example",
        argv=["--symbols", "BTCUSD"],
    )

    captured = capsys.readouterr()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 1
    assert "human output" in captured.out
    assert "ERROR: runner returned list summary payload, expected a mapping" in captured.out
    assert "Diagnostic artifacts:" in captured.out
    assert f"  summary_json: {summary_path}" in captured.out
    assert "Reproduce:" in captured.out
    assert payload["invocation"]["command"] in captured.out
    assert "Wrote summary JSON" not in captured.out
    assert payload["tool"] == "example"
    assert payload["error"] == "ERROR: runner returned list summary payload, expected a mapping"
    assert payload["error_type"] == "TypeError"
    assert payload["exit_code"] == 1
    assert payload["status"] == "error"
    assert payload["summary_json_file"] == str(summary_path)
    assert payload["invocation"]["module"] == "binance_worksteal.example"



def test_run_with_optional_summary_dash_rejects_invalid_payload_and_keeps_json_on_stdout(capsys):
    def runner():
        print("human output")
        return 0, ["bad"]

    rc = run_with_optional_summary("-", runner, module="binance_worksteal.example", argv=["--symbols", "BTCUSD"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 1
    assert payload["tool"] == "example"
    assert payload["error"] == "ERROR: runner returned list summary payload, expected a mapping"
    assert payload["error_type"] == "TypeError"
    assert payload["exit_code"] == 1
    assert payload["status"] == "error"
    err_lines = captured.err.strip().splitlines()
    assert err_lines == ["human output", "ERROR: runner returned list summary payload, expected a mapping"]


def test_backtest_preview_run_skips_data_loading_and_writes_summary(tmp_path, monkeypatch, capsys):
    summary_path = tmp_path / "preview.json"

    def fail_load(*args, **kwargs):
        raise AssertionError("load_daily_bars should not run for --preview-run")

    monkeypatch.setattr(backtest, "load_daily_bars", fail_load)

    rc = backtest.main(
        [
            "--symbols",
            "btcusd",
            "ethusd",
            "--days",
            "45",
            "--preview-run",
            "--summary-json",
            str(summary_path),
        ]
    )

    out = capsys.readouterr().out
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert "backtest run preview:" in out
    assert "date_mode: auto_last_n_days" in out
    assert "symbol_count: 2" in out
    assert payload["preview_only"] is True
    assert payload["tool"] == "backtest"
    assert payload["requested_symbol_count"] == 2
    assert payload["symbols"] == ["BTCUSD", "ETHUSD"]
    assert payload["days"] == 45
    assert payload["exit_code"] == 0
    assert payload["status"] == "success"
    assert payload["invocation"]["module"] == "binance_worksteal.backtest"
    assert payload["invocation"]["argv"] == [
        "--symbols",
        "btcusd",
        "ethusd",
        "--days",
        "45",
        "--preview-run",
        "--summary-json",
        str(summary_path),
    ]
    assert payload["invocation"]["command"].startswith("python -m binance_worksteal.backtest ")


def test_evaluate_symbols_writes_summary_json(tmp_path, monkeypatch, capsys):
    summary_path = tmp_path / "eval_summary.json"

    monkeypatch.setattr(
        evaluate_symbols,
        "run_evaluation",
        lambda **kwargs: (
            "example output",
            [{"symbol": "BTCUSD", "marginal_contribution": 0.25, "standalone_sortino": 1.5}],
            {
                "windows": [("2026-01-01", "2026-01-31")],
                "avg_full_return": 2.5,
                "avg_full_sortino": 1.2,
                "base_symbol_count": 1,
                "evaluated_symbol_count": 1,
            },
        ),
    )

    rc = evaluate_symbols.main(["--symbols", "BTCUSD", "--summary-json", str(summary_path)])

    out = capsys.readouterr().out
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert "example output" in out
    assert "Generated artifacts:" in out
    assert f"  summary_json: {summary_path}" in out
    assert "Reproduce:" in out
    assert payload["invocation"]["command"] in out
    assert "Wrote summary JSON" not in out
    assert payload["summary_schema_version"] == 1
    assert payload["tool"] == "evaluate_symbols"
    assert payload["exit_code"] == 0
    assert payload["status"] == "success"
    assert payload["requested_symbol_count"] == 1
    assert payload["rows"][0]["symbol"] == "BTCUSD"
    assert payload["avg_full_sortino"] == 1.2


def test_evaluate_symbols_preview_run_dedupes_and_reports_ignored_candidates(tmp_path, monkeypatch, capsys):
    summary_path = tmp_path / "preview.json"

    def fail_run_evaluation(**kwargs):
        raise AssertionError("run_evaluation should not run for --preview-run")

    monkeypatch.setattr(evaluate_symbols, "run_evaluation", fail_run_evaluation)

    rc = evaluate_symbols.main(
        [
            "--symbols",
            "BTCUSD",
            "ETHUSD",
            "--candidate-symbols",
            "ethusd",
            "solusd",
            "--preview-run",
            "--summary-json",
            str(summary_path),
        ]
    )

    out = capsys.readouterr().out
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["candidate_symbols"] == ["SOLUSD"]
    assert payload["candidate_symbol_count"] == 1
    assert payload["ignored_candidate_symbols"] == ["ETHUSD"]
    assert payload["ignored_candidate_symbol_count"] == 1
    assert "ignored_candidate_symbol_count: 1" in out
    assert "ignored_candidate_symbols:" in out
    assert "ETHUSD" in out


def test_evaluate_symbols_preview_run_normalizes_candidates_and_skips_execution(tmp_path, monkeypatch, capsys):
    summary_path = tmp_path / "preview.json"

    def fail_run_evaluation(**kwargs):
        raise AssertionError("run_evaluation should not run for --preview-run")

    monkeypatch.setattr(evaluate_symbols, "run_evaluation", fail_run_evaluation)

    rc = evaluate_symbols.main(
        [
            "--symbols",
            "BTCUSD",
            "--candidate-symbols",
            "ethusd",
            "solusd",
            "--start",
            "2026-01-01",
            "--end",
            "2026-01-31",
            "--preview-run",
            "--summary-json",
            str(summary_path),
        ]
    )

    out = capsys.readouterr().out
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert "evaluate_symbols run preview:" in out
    assert "date_mode: fixed_range" in out
    assert "candidate_symbol_count: 2" in out
    assert payload["preview_only"] is True
    assert payload["window_mode"] == "fixed"
    assert payload["candidate_symbols"] == ["ETHUSD", "SOLUSD"]
    assert payload["candidate_symbol_count"] == 2
    assert payload["start_date"] == "2026-01-01"
    assert payload["end_date"] == "2026-01-31"


def test_sweep_preview_run_skips_data_loading_and_reports_grid(tmp_path, monkeypatch, capsys):
    summary_path = tmp_path / "preview.json"
    output_csv = tmp_path / "results.csv"

    def fail_load(*args, **kwargs):
        raise AssertionError("load_daily_bars should not run for --preview-run")

    monkeypatch.setattr(sweep, "load_daily_bars", fail_load)

    rc = sweep.main(
        [
            "--symbols",
            "BTCUSD",
            "--output",
            str(output_csv),
            "--n-trials",
            "123",
            "--realistic",
            "--preview-run",
            "--summary-json",
            str(summary_path),
        ]
    )

    out = capsys.readouterr().out
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert "sweep run preview:" in out
    assert "n_trials: 123" in out
    assert "execution_mode: REALISTIC (touch-fill + next-bar execution)" in out
    assert "Generated artifacts:" in out
    assert f"  summary_json: {summary_path}" in out
    assert "Reproduce:" in out
    assert payload["invocation"]["command"] in out
    assert "Wrote summary JSON" not in out
    assert payload["preview_only"] is True
    assert payload["n_trials_requested"] == 123
    assert payload["total_grid_size"] > 0
    assert payload["output_csv"] == str(output_csv)
    assert payload["execution_mode"] == "REALISTIC (touch-fill + next-bar execution)"


def test_sweep_preview_run_does_not_write_default_sidecar_summary(tmp_path, monkeypatch, capsys):
    output_csv = tmp_path / "results.csv"
    summary_path = default_sidecar_json_path(output_csv)

    def fail_load(*args, **kwargs):
        raise AssertionError("load_daily_bars should not run for --preview-run")

    monkeypatch.setattr(sweep, "load_daily_bars", fail_load)

    rc = sweep.main(
        [
            "--symbols",
            "BTCUSD",
            "--output",
            str(output_csv),
            "--preview-run",
        ]
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert "sweep run preview:" in out
    assert "Wrote summary JSON" not in out
    assert summary_path.exists() is False


def test_sweep_expanded_preview_run_skips_data_loading(tmp_path, monkeypatch, capsys):
    summary_path = tmp_path / "preview.json"
    output_csv = tmp_path / "expanded.csv"

    def fail_load(*args, **kwargs):
        raise AssertionError("load_daily_bars should not run for --preview-run")

    monkeypatch.setattr(sweep_expanded, "load_daily_bars", fail_load)

    rc = sweep_expanded.main(
        [
            "--symbols",
            "BTCUSD",
            "--output",
            str(output_csv),
            "--max-trials",
            "77",
            "--workers",
            "4",
            "--preview-run",
            "--summary-json",
            str(summary_path),
        ]
    )

    out = capsys.readouterr().out
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert "sweep_expanded run preview:" in out
    assert "max_trials: 77" in out
    assert "workers: 4" in out
    assert "Generated artifacts:" in out
    assert f"  summary_json: {summary_path}" in out
    assert "Reproduce:" in out
    assert payload["invocation"]["command"] in out
    assert "Wrote summary JSON" not in out
    assert payload["preview_only"] is True
    assert payload["max_trials_requested"] == 77
    assert payload["worker_count"] == 4
    assert payload["output_csv"] == str(output_csv)


def test_sweep_expanded_preview_run_does_not_write_default_sidecar_summary(tmp_path, monkeypatch, capsys):
    output_csv = tmp_path / "expanded.csv"
    summary_path = default_sidecar_json_path(output_csv)

    def fail_load(*args, **kwargs):
        raise AssertionError("load_daily_bars should not run for --preview-run")

    monkeypatch.setattr(sweep_expanded, "load_daily_bars", fail_load)

    rc = sweep_expanded.main(
        [
            "--symbols",
            "BTCUSD",
            "--output",
            str(output_csv),
            "--preview-run",
        ]
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert "sweep_expanded run preview:" in out
    assert "Wrote summary JSON" not in out
    assert summary_path.exists() is False


def test_sim_vs_live_audit_preview_run_skips_data_loading(tmp_path, monkeypatch, capsys):
    summary_path = tmp_path / "preview.json"

    def fail_load(*args, **kwargs):
        raise AssertionError("load_daily_bars should not run for --preview-run")

    monkeypatch.setattr(sim_vs_live_audit, "load_daily_bars", fail_load)

    rc = sim_vs_live_audit.main(
        [
            "--symbols",
            "BTCUSD",
            "--days",
            "14",
            "--preview-run",
            "--summary-json",
            str(summary_path),
        ]
    )

    out = capsys.readouterr().out
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert "sim_vs_live_audit run preview:" in out
    assert "date_mode: auto_last_n_days" in out
    assert payload["preview_only"] is True
    assert payload["days"] == 14
    assert payload["tool"] == "sim_vs_live_audit"


def test_evaluate_symbols_summary_json_includes_candidate_load_details(tmp_path, monkeypatch):
    summary_path = tmp_path / "eval_summary.json"

    monkeypatch.setattr(
        evaluate_symbols,
        "run_evaluation",
        lambda **kwargs: (
            "example output",
            [{"symbol": "BTCUSD", "marginal_contribution": 0.25, "standalone_sortino": 1.5}],
            {
                "windows": [("2026-01-01", "2026-01-31")],
                "avg_full_return": 2.5,
                "avg_full_sortino": 1.2,
                "base_symbol_count": 1,
                "evaluated_symbol_count": 2,
                "candidate_symbols": ["ETHUSD", "SOLUSD"],
                "candidate_symbol_count": 2,
                "candidate_loaded_symbol_count": 1,
                "candidate_loaded_symbols": ["ETHUSD"],
                "candidate_missing_symbol_count": 1,
                "candidate_missing_symbols": ["SOLUSD"],
            },
        ),
    )

    rc = evaluate_symbols.main(
        ["--symbols", "BTCUSD", "--candidate-symbols", "ethusd", "solusd", "--summary-json", str(summary_path)]
    )

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["candidate_symbols"] == ["ETHUSD", "SOLUSD"]
    assert payload["candidate_symbol_count"] == 2
    assert payload["candidate_loaded_symbol_count"] == 1
    assert payload["candidate_loaded_symbols"] == ["ETHUSD"]
    assert payload["candidate_missing_symbol_count"] == 1
    assert payload["candidate_missing_symbols"] == ["SOLUSD"]


def test_evaluate_symbols_returns_nonzero_on_runner_error(monkeypatch, capsys):
    monkeypatch.setattr(
        evaluate_symbols,
        "run_evaluation",
        lambda **kwargs: (
            "ERROR: No data loaded",
            [],
            {
                "windows": [],
                "avg_full_return": 0.0,
                "avg_full_sortino": 0.0,
                "base_symbol_count": 0,
                "evaluated_symbol_count": 0,
            },
        ),
    )

    rc = evaluate_symbols.main(["--symbols", "BTCUSD"])

    out = capsys.readouterr().out.strip().splitlines()
    assert rc == 1
    assert out[-1] == "ERROR: No data loaded"


def test_evaluate_symbols_writes_summary_json_on_runner_error(tmp_path, monkeypatch, capsys):
    summary_path = tmp_path / "eval_summary.json"
    monkeypatch.setattr(
        evaluate_symbols,
        "run_evaluation",
        lambda **kwargs: (
            "ERROR: Not enough data for rolling windows",
            [],
            {
                "windows": [],
                "avg_full_return": 0.0,
                "avg_full_sortino": 0.0,
                "base_symbol_count": 0,
                "evaluated_symbol_count": 0,
            },
        ),
    )

    rc = evaluate_symbols.main(["--symbols", "BTCUSD", "--summary-json", str(summary_path)])

    out = capsys.readouterr().out.strip().splitlines()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 1
    assert "ERROR: Not enough data for rolling windows" in out
    assert "Diagnostic artifacts:" in out
    assert f"  summary_json: {summary_path}" in out
    assert "Reproduce:" in out
    assert "Wrote summary JSON" not in "\n".join(out)
    assert payload["summary_text"] == "ERROR: Not enough data for rolling windows"
    assert payload["rows"] == []
    assert payload["status"] == "error"
    assert payload["exit_code"] == 1


def test_evaluate_symbols_returns_nonzero_when_runner_produces_no_rows(monkeypatch, capsys):
    monkeypatch.setattr(
        evaluate_symbols,
        "run_evaluation",
        lambda **kwargs: (
            "example output",
            [],
            {
                "windows": [("2026-01-01", "2026-01-31")],
                "avg_full_return": 2.5,
                "avg_full_sortino": 1.2,
                "base_symbol_count": 1,
                "evaluated_symbol_count": 0,
                "evaluation_failure_count": 1,
                "evaluation_failures": [{"stage": "standalone", "symbol": "BTCUSD"}],
                "skipped_symbols": ["BTCUSD"],
            },
        ),
    )

    rc = evaluate_symbols.main(["--symbols", "BTCUSD"])

    out = capsys.readouterr().out.strip().splitlines()
    assert rc == 1
    assert out[-2] == "example output"
    assert out[-1] == "ERROR: No valid symbol evaluation results"


def test_evaluate_symbols_writes_summary_json_for_empty_rows_failure(tmp_path, monkeypatch, capsys):
    summary_path = tmp_path / "eval_summary.json"
    monkeypatch.setattr(
        evaluate_symbols,
        "run_evaluation",
        lambda **kwargs: (
            "example output",
            [],
            {
                "windows": [("2026-01-01", "2026-01-31")],
                "avg_full_return": 2.5,
                "avg_full_sortino": 1.2,
                "base_symbol_count": 1,
                "evaluated_symbol_count": 0,
                "evaluation_failure_count": 2,
                "evaluation_failures": [{"stage": "standalone", "symbol": "BTCUSD"}],
                "skipped_symbols": ["BTCUSD"],
            },
        ),
    )

    rc = evaluate_symbols.main(["--symbols", "BTCUSD", "--summary-json", str(summary_path)])

    out = capsys.readouterr().out.strip().splitlines()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 1
    assert "Diagnostic artifacts:" in out
    assert f"  summary_json: {summary_path}" in out
    assert "Reproduce:" in out
    assert "Wrote summary JSON" not in "\n".join(out)
    assert payload["rows"] == []
    assert payload["evaluation_failure_count"] == 2
    assert payload["skipped_symbols"] == ["BTCUSD"]
    assert payload["summary_text"].endswith("ERROR: No valid symbol evaluation results")


def test_evaluate_symbols_runner_error_summary_dash_prints_json_to_stdout(monkeypatch, capsys):
    monkeypatch.setattr(
        evaluate_symbols,
        "run_evaluation",
        lambda **kwargs: (
            "ERROR: Not enough data for rolling windows",
            [],
            {
                "windows": [],
                "avg_full_return": 0.0,
                "avg_full_sortino": 0.0,
                "base_symbol_count": 0,
                "evaluated_symbol_count": 0,
            },
        ),
    )

    rc = evaluate_symbols.main(["--symbols", "BTCUSD", "--summary-json", "-"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 1
    assert payload["summary_text"] == "ERROR: Not enough data for rolling windows"
    assert payload["rows"] == []
    assert payload["exit_code"] == 1
    assert payload["status"] == "error"
    assert "Wrote summary JSON" not in captured.out
    assert "ERROR: Not enough data for rolling windows" in captured.err
    assert "Using 1 symbols from command line --symbols" in captured.err


def test_evaluate_symbols_empty_rows_summary_dash_prints_json_to_stdout(monkeypatch, capsys):
    monkeypatch.setattr(
        evaluate_symbols,
        "run_evaluation",
        lambda **kwargs: (
            "example output",
            [],
            {
                "windows": [("2026-01-01", "2026-01-31")],
                "avg_full_return": 2.5,
                "avg_full_sortino": 1.2,
                "base_symbol_count": 1,
                "evaluated_symbol_count": 0,
                "evaluation_failure_count": 2,
                "evaluation_failures": [{"stage": "standalone", "symbol": "BTCUSD"}],
                "skipped_symbols": ["BTCUSD"],
            },
        ),
    )

    rc = evaluate_symbols.main(["--symbols", "BTCUSD", "--summary-json", "-"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 1
    assert payload["rows"] == []
    assert payload["evaluation_failure_count"] == 2
    assert payload["summary_text"].endswith("ERROR: No valid symbol evaluation results")
    assert payload["exit_code"] == 1
    assert payload["status"] == "error"
    assert "Wrote summary JSON" not in captured.out
    assert "example output" in captured.err
    assert "ERROR: No valid symbol evaluation results" in captured.err


def test_backtest_summary_json_dash_prints_json_to_stdout(tmp_path, monkeypatch, capsys):
    bars = {
        "BTCUSD": pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-01-31", tz="UTC")],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000.0],
                "symbol": ["BTCUSD"],
            }
        )
    }

    monkeypatch.setattr(backtest, "load_daily_bars", lambda data_dir, symbols: bars)
    monkeypatch.setattr(
        backtest,
        "run_worksteal_backtest",
        lambda *args, **kwargs: (
            pd.DataFrame({"timestamp": [pd.Timestamp("2026-01-31", tz="UTC")], "equity": [10050.0]}),
            [],
            {"final_equity": 10050.0, "total_return_pct": 0.5, "n_days": 2},
        ),
    )
    monkeypatch.setattr(backtest, "print_results", lambda *args, **kwargs: None)

    rc = backtest.main(["--symbols", "BTCUSD", "--summary-json", "-"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 0
    assert captured.out.strip().startswith("{")
    assert "Wrote summary JSON" not in captured.out
    assert payload["tool"] == "backtest"
    assert payload["symbol_count"] == 1
    assert "Using 1 symbols from command line --symbols" in captured.err
    assert "Loading data for 1 symbols" in captured.err


def test_backtest_runtime_failure_writes_summary_json(tmp_path, monkeypatch, capsys):
    summary_path = tmp_path / "backtest_summary.json"
    bars = {
        "BTCUSD": pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-01-31", tz="UTC")],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000.0],
                "symbol": ["BTCUSD"],
            }
        )
    }

    monkeypatch.setattr(backtest, "load_daily_bars", lambda data_dir, symbols: bars)

    def raise_backtest(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(backtest, "run_worksteal_backtest", raise_backtest)
    monkeypatch.setattr(backtest, "print_results", lambda *args, **kwargs: None)

    rc = backtest.main(["--symbols", "BTCUSD", "--summary-json", str(summary_path)])

    out = capsys.readouterr().out.strip().splitlines()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 1
    assert any(line.startswith("ERROR: backtest run_worksteal_backtest Python backtest evaluation failed") for line in out)
    assert "Diagnostic artifacts:" in out
    assert f"  summary_json: {summary_path}" in out
    assert "Reproduce:" in out
    assert "Wrote summary JSON" not in "\n".join(out)
    assert "error" not in payload
    assert payload["backtest_failure"]["stage"] == "run_worksteal_backtest"
    assert payload["backtest_failure"]["error_type"] == "RuntimeError"
    assert payload["backtest_failure"]["error"] == "boom"
    assert payload["metrics"] is None
    assert payload["trade_counts"]["entries"] == 0
    assert payload["exit_code"] == 1
    assert payload["status"] == "error"



def test_backtest_summary_json_dash_prints_runtime_failure_to_stdout(tmp_path, monkeypatch, capsys):
    bars = {
        "BTCUSD": pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-01-31", tz="UTC")],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000.0],
                "symbol": ["BTCUSD"],
            }
        )
    }

    monkeypatch.setattr(backtest, "load_daily_bars", lambda data_dir, symbols: bars)
    monkeypatch.setattr(
        backtest,
        "run_worksteal_backtest",
        lambda *args, **kwargs: (
            pd.DataFrame({"timestamp": [pd.Timestamp("2026-01-31", tz="UTC")], "equity": [10050.0]}),
            [],
            {"final_equity": 10050.0, "total_return_pct": 0.5, "n_days": 2},
        ),
    )

    def raise_print(*args, **kwargs):
        raise ValueError("bad render")

    monkeypatch.setattr(backtest, "print_results", raise_print)

    rc = backtest.main(["--symbols", "BTCUSD", "--summary-json", "-"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 1
    assert payload["tool"] == "backtest"
    assert payload["backtest_failure"]["stage"] == "print_results"
    assert payload["backtest_failure"]["error_type"] == "ValueError"
    assert payload["backtest_failure"]["error"] == "bad render"
    assert payload["metrics"]["final_equity"] == 10050.0
    assert payload["trade_counts"]["entries"] == 0
    assert payload["exit_code"] == 1
    assert payload["status"] == "error"
    assert "Wrote summary JSON" not in captured.out
    assert "ERROR: backtest print_results reporting evaluation failed" in captured.err



def test_backtest_accepts_start_end_aliases(monkeypatch):
    seen = {}
    bars = {
        "BTCUSD": pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-01-31", tz="UTC")],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000.0],
                "symbol": ["BTCUSD"],
            }
        )
    }

    monkeypatch.setattr(backtest, "load_daily_bars", lambda data_dir, symbols: bars)

    def fake_run(*args, **kwargs):
        seen["start_date"] = kwargs["start_date"]
        seen["end_date"] = kwargs["end_date"]
        return pd.DataFrame(), [], {"final_equity": 10000.0}

    monkeypatch.setattr(backtest, "run_worksteal_backtest", fake_run)
    monkeypatch.setattr(backtest, "print_results", lambda *args, **kwargs: None)

    rc = backtest.main(["--symbols", "BTCUSD", "--start", "2026-01-01", "--end", "2026-01-31"])

    assert rc == 0
    assert seen == {"start_date": "2026-01-01", "end_date": "2026-01-31"}


@pytest.mark.parametrize(
    ("argv", "expected_error"),
    [
        (
            ["--symbols", "BTCUSD", "--start", "bad-date", "--end", "2026-01-31"],
            "ERROR: Invalid --start/--start-date value: 'bad-date'",
        ),
        (
            ["--symbols", "BTCUSD", "--start", "2026-02-01", "--end", "2026-01-31"],
            "ERROR: --start/--start-date must be on or before --end/--end-date.",
        ),
    ],
)
def test_backtest_invalid_date_range_returns_error_before_loading_data(argv, expected_error, monkeypatch, capsys):
    def fail_load(*args, **kwargs):
        raise AssertionError("load_daily_bars should not run for invalid date ranges")

    monkeypatch.setattr(backtest, "load_daily_bars", fail_load)

    rc = backtest.main(argv)

    out = capsys.readouterr().out.strip().splitlines()
    assert rc == 1
    assert out[-1] == expected_error


@pytest.mark.parametrize(
    ("entrypoint", "tool", "module_name", "argv", "expected_error"),
    [
        (
            backtest.main,
            "backtest",
            "binance_worksteal.backtest",
            ["--symbols", "BTCUSD", "--start", "bad-date", "--end", "2026-01-31", "--summary-json", "-"],
            "ERROR: Invalid --start/--start-date value: 'bad-date'",
        ),
        (
            evaluate_symbols.main,
            "evaluate_symbols",
            "binance_worksteal.evaluate_symbols",
            ["--symbols", "BTCUSD", "--start", "bad-date", "--end", "2026-01-31", "--summary-json", "-"],
            "ERROR: Invalid --start/--start-date value: 'bad-date'",
        ),
        (
            sim_vs_live_audit.main,
            "sim_vs_live_audit",
            "binance_worksteal.sim_vs_live_audit",
            ["--symbols", "BTCUSD", "--start", "bad-date", "--end", "2026-01-31", "--summary-json", "-"],
            "ERROR: Invalid --start/--start-date value: 'bad-date'",
        ),
    ],
)
def test_invalid_date_range_summary_dash_writes_structured_error(entrypoint, tool, module_name, argv, expected_error, monkeypatch, capsys):
    def fail_load(*args, **kwargs):
        raise AssertionError("load_daily_bars should not run for invalid date ranges")

    if entrypoint is backtest.main:
        monkeypatch.setattr(backtest, "load_daily_bars", fail_load)
    elif entrypoint is evaluate_symbols.main:
        monkeypatch.setattr(evaluate_symbols, "load_daily_bars", fail_load)
    else:
        monkeypatch.setattr(sim_vs_live_audit, "load_daily_bars", fail_load)

    rc = entrypoint(argv)

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 1
    assert payload["tool"] == tool
    assert payload["error"] == expected_error
    assert payload["error_type"] == "ValueError"
    assert payload["status"] == "error"
    assert payload["exit_code"] == 1
    assert payload["invocation"]["module"] == module_name
    assert captured.err.strip() == expected_error


def test_backtest_uses_config_file_overrides(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "backtest_config.yaml"
    config_path.write_text(
        "config:\n"
        "  dip_pct: 0.18\n"
        "  sma_filter_period: 7\n"
        "  realistic_fill: true\n",
        encoding="utf-8",
    )
    seen = {}
    bars = {
        "BTCUSD": pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-01-31", tz="UTC")],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000.0],
                "symbol": ["BTCUSD"],
            }
        )
    }

    monkeypatch.setattr(backtest, "load_daily_bars", lambda data_dir, symbols: bars)

    def fake_run(all_bars, config, start_date, end_date):
        seen["dip_pct"] = config.dip_pct
        seen["sma_filter_period"] = config.sma_filter_period
        seen["realistic_fill"] = config.realistic_fill
        return pd.DataFrame(), [], {"final_equity": 10000.0}

    monkeypatch.setattr(backtest, "run_worksteal_backtest", fake_run)
    monkeypatch.setattr(backtest, "print_results", lambda *args, **kwargs: None)

    rc = backtest.main(["--symbols", "BTCUSD", "--config-file", str(config_path)])

    out = capsys.readouterr().out
    assert rc == 0
    assert f"Loaded config overrides from {config_path}" in out
    assert seen == {"dip_pct": 0.18, "sma_filter_period": 7, "realistic_fill": True}


def test_backtest_cli_flags_override_config_file(tmp_path, monkeypatch):
    config_path = tmp_path / "backtest_config.yaml"
    config_path.write_text("dip_pct: 0.18\n", encoding="utf-8")
    seen = {}
    bars = {
        "BTCUSD": pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-01-31", tz="UTC")],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000.0],
                "symbol": ["BTCUSD"],
            }
        )
    }

    monkeypatch.setattr(backtest, "load_daily_bars", lambda data_dir, symbols: bars)

    def fake_run(all_bars, config, start_date, end_date):
        seen["dip_pct"] = config.dip_pct
        return pd.DataFrame(), [], {"final_equity": 10000.0}

    monkeypatch.setattr(backtest, "run_worksteal_backtest", fake_run)
    monkeypatch.setattr(backtest, "print_results", lambda *args, **kwargs: None)

    rc = backtest.main(["--symbols", "BTCUSD", "--config-file", str(config_path), "--dip-pct", "0.25"])

    assert rc == 0
    assert seen == {"dip_pct": 0.25}


def test_backtest_invalid_config_file_returns_error(tmp_path, capsys):
    config_path = tmp_path / "backtest_config.yaml"
    config_path.write_text("unknown_field: 1\n", encoding="utf-8")

    rc = backtest.main(["--symbols", "BTCUSD", "--config-file", str(config_path)])

    out = capsys.readouterr().out.strip()
    assert rc == 1
    assert out == f"ERROR: Unsupported WorkStealConfig fields in {config_path}: unknown_field"


@pytest.mark.parametrize(
    ("entrypoint", "tool", "module_name", "config_name"),
    [
        (backtest.main, "backtest", "binance_worksteal.backtest", "backtest_config.yaml"),
        (evaluate_symbols.main, "evaluate_symbols", "binance_worksteal.evaluate_symbols", "evaluate_config.yaml"),
        (sim_vs_live_audit.main, "sim_vs_live_audit", "binance_worksteal.sim_vs_live_audit", "audit_config.yaml"),
    ],
)
def test_invalid_config_file_summary_dash_writes_structured_error(entrypoint, tool, module_name, config_name, tmp_path, capsys):
    config_path = tmp_path / config_name
    config_path.write_text("unknown_field: 1\n", encoding="utf-8")

    rc = entrypoint(["--symbols", "BTCUSD", "--config-file", str(config_path), "--summary-json", "-"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 1
    assert payload["tool"] == tool
    assert payload["error"] == f"ERROR: Unsupported WorkStealConfig fields in {config_path}: unknown_field"
    assert payload["error_type"] == "ValueError"
    assert payload["status"] == "error"
    assert payload["exit_code"] == 1
    assert payload["invocation"]["module"] == module_name
    assert captured.err.strip() == f"ERROR: Unsupported WorkStealConfig fields in {config_path}: unknown_field"


@pytest.mark.parametrize(
    ("entrypoint", "tool", "module_name", "config_name"),
    [
        (backtest.main, "backtest", "binance_worksteal.backtest", "backtest_missing.yaml"),
        (evaluate_symbols.main, "evaluate_symbols", "binance_worksteal.evaluate_symbols", "evaluate_missing.yaml"),
        (sim_vs_live_audit.main, "sim_vs_live_audit", "binance_worksteal.sim_vs_live_audit", "audit_missing.yaml"),
    ],
)
def test_missing_config_file_summary_dash_writes_structured_error(entrypoint, tool, module_name, config_name, tmp_path, capsys):
    config_path = tmp_path / config_name

    rc = entrypoint(["--symbols", "BTCUSD", "--config-file", str(config_path), "--summary-json", "-"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 1
    assert payload["tool"] == tool
    assert payload["error"] == f"ERROR: Config file not found: {config_path}"
    assert payload["error_type"] == "FileNotFoundError"
    assert payload["config_file"] == str(config_path)
    assert payload["status"] == "error"
    assert payload["exit_code"] == 1
    assert payload["invocation"]["module"] == module_name
    assert captured.err.strip() == f"ERROR: Config file not found: {config_path}"


def test_sweep_writes_default_summary_sidecar(tmp_path, monkeypatch, capsys):
    output_csv = tmp_path / "sweep_results.csv"
    bars = {
        "BTCUSD": pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-01-31", tz="UTC")],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000.0],
            }
        )
    }

    monkeypatch.setattr(sweep, "load_daily_bars", lambda data_dir, symbols: bars)
    monkeypatch.setattr(sweep, "build_windows", lambda all_bars, window_days, n_windows: [("2026-01-31", "2026-01-31")])
    monkeypatch.setattr(
        sweep,
        "run_sweep",
        lambda *args, **kwargs: [
            {"min_sortino": 1.5, "mean_sortino": 2.0, "mean_return_pct": 3.0, "total_n_trades": 4}
        ],
    )

    rc = sweep.main(["--symbols", "BTCUSD", "--output", str(output_csv), "--n-trials", "1"])

    out = capsys.readouterr().out
    summary_path = default_sidecar_json_path(output_csv)
    best_config_path = default_best_config_path(output_csv)
    best_overrides_path = default_best_overrides_path(output_csv)
    next_steps_script_path = default_next_steps_script_path(output_csv)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert "Wrote summary JSON" not in out
    assert "Wrote recommended config YAML" not in out
    assert "Wrote recommended overrides YAML" not in out
    assert "Wrote next steps shell script" not in out
    assert "Generated artifacts:" in out
    assert "Warnings:" not in out
    assert "Reproduce:" in out
    assert f"python -m binance_worksteal.sweep --symbols BTCUSD --output {output_csv} --n-trials 1" in out
    assert str(summary_path) in out
    assert str(output_csv) in out
    assert str(best_config_path) in out
    assert str(best_overrides_path) in out
    assert str(next_steps_script_path) in out
    assert "Next steps:" in out
    assert "python -m binance_worksteal.backtest" in out
    assert "python -m binance_worksteal.sim_vs_live_audit" in out
    assert "python -m binance_worksteal.evaluate_symbols" in out
    assert payload["summary_schema_version"] == 1
    assert payload["tool"] == "sweep"
    assert payload["requested_symbol_count"] == 1
    assert payload["config"]["initial_cash"] == pytest.approx(10000.0)
    assert payload["base_config"]["initial_cash"] == pytest.approx(10000.0)
    assert payload["output_csv"] == str(output_csv)
    assert payload["recommended_config_file"] == str(best_config_path)
    assert payload["recommended_config"]["initial_cash"] == pytest.approx(10000.0)
    assert payload["recommended_overrides_file"] == str(best_overrides_path)
    assert payload["recommended_overrides"] == {}
    assert payload["follow_up_config_file"] == str(best_overrides_path)
    assert payload["next_steps_script_file"] == str(next_steps_script_path)
    assert payload["summary_json_file"] == str(summary_path)
    assert payload["follow_up_config_kind"] == "standalone_overrides"
    assert [item["name"] for item in payload["artifacts"]] == [
        "summary_json",
        "results_csv",
        "recommended_config",
        "recommended_overrides",
        "next_steps_script",
    ]
    assert [item["name"] for item in payload["next_steps"]] == [
        "backtest",
        "sim_vs_live_audit",
        "evaluate_symbols",
    ]
    assert str(best_overrides_path) in payload["next_steps"][0]["argv"]
    assert payload["skipped_backtest_failure_count"] == 0
    assert payload["backtest_failure_samples"] == []
    assert payload["c_sim_runtime_fallback_count"] == 0
    assert payload["best_result"]["min_sortino"] == 1.5
    best_config_payload = yaml.safe_load(best_config_path.read_text(encoding="utf-8"))
    assert best_config_payload["config"]["initial_cash"] == pytest.approx(10000.0)
    best_overrides_payload = yaml.safe_load(best_overrides_path.read_text(encoding="utf-8"))
    assert best_overrides_payload["config"] == {}
    next_steps_script = next_steps_script_path.read_text(encoding="utf-8")
    assert next_steps_script.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert "python -m binance_worksteal.backtest" in next_steps_script


def test_sweep_require_full_universe_returns_error_and_summary(tmp_path, monkeypatch, capsys):
    output_csv = tmp_path / "sweep_results.csv"
    summary_path = default_sidecar_json_path(output_csv)
    bars = {
        "BTCUSD": pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-01-31", tz="UTC")],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000.0],
            }
        )
    }

    monkeypatch.setattr(sweep, "load_daily_bars", lambda data_dir, symbols: bars)

    def fail_windows(*args, **kwargs):
        raise AssertionError("build_windows should not run when --require-full-universe fails")

    monkeypatch.setattr(sweep, "build_windows", fail_windows)

    rc = sweep.main([
        "--symbols", "BTCUSD", "ETHUSD",
        "--output", str(output_csv),
        "--require-full-universe",
        "--summary-json", str(summary_path),
    ])

    out = capsys.readouterr().out.strip().splitlines()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 1
    assert "ERROR: --require-full-universe found missing data for 1 symbol: ETHUSD" in out
    assert "Diagnostic artifacts:" in out
    assert f"  summary_json: {summary_path}" in out
    assert "Reproduce:" in out
    assert "Wrote summary JSON" not in "\n".join(out)
    assert payload["require_full_universe"] is True
    assert payload["universe_complete"] is False
    assert payload["missing_symbols"] == ["ETHUSD"]
    assert payload["results_count"] == 0
    assert payload["error"] == "ERROR: --require-full-universe found missing data for 1 symbol: ETHUSD"
    assert payload["exit_code"] == 1
    assert payload["status"] == "error"


def test_sweep_writes_summary_json_on_no_data(tmp_path, monkeypatch, capsys):
    output_csv = tmp_path / "sweep_results.csv"
    summary_path = default_sidecar_json_path(output_csv)

    monkeypatch.setattr(sweep, "load_daily_bars", lambda data_dir, symbols: {})

    def fail_windows(*args, **kwargs):
        raise AssertionError("build_windows should not run when no data loads")

    monkeypatch.setattr(sweep, "build_windows", fail_windows)

    rc = sweep.main([
        "--symbols", "BTCUSD",
        "--output", str(output_csv),
        "--summary-json", str(summary_path),
    ])

    out = capsys.readouterr().out.strip().splitlines()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 1
    assert "ERROR: No data" in out
    assert "Diagnostic artifacts:" in out
    assert f"  summary_json: {summary_path}" in out
    assert "Reproduce:" in out
    assert "Wrote summary JSON" not in "\n".join(out)
    assert payload["results_count"] == 0
    assert payload["loaded_symbol_count"] == 0
    assert payload["missing_symbols"] == ["BTCUSD"]
    assert payload["error"] == "ERROR: No data"
    assert payload["exit_code"] == 1
    assert payload["status"] == "error"


def test_sweep_writes_summary_json_on_loader_error(tmp_path, monkeypatch, capsys):
    output_csv = tmp_path / "sweep_results.csv"
    summary_path = default_sidecar_json_path(output_csv)

    def raise_load_error(data_dir, symbols):
        raise OSError("bad sweep dir")

    monkeypatch.setattr(sweep, "load_daily_bars", raise_load_error)

    def fail_windows(*args, **kwargs):
        raise AssertionError("build_windows should not run when load_daily_bars fails")

    monkeypatch.setattr(sweep, "build_windows", fail_windows)

    rc = sweep.main([
        "--symbols", "BTCUSD",
        "--output", str(output_csv),
        "--summary-json", str(summary_path),
    ])

    out = capsys.readouterr().out.strip().splitlines()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 1
    assert "ERROR: bad sweep dir" in out
    assert "Diagnostic artifacts:" in out
    assert f"  summary_json: {summary_path}" in out
    assert "Reproduce:" in out
    assert "Wrote summary JSON" not in "\n".join(out)
    assert payload["results_count"] == 0
    assert payload["loaded_symbol_count"] == 0
    assert payload["missing_symbols"] == ["BTCUSD"]
    assert payload["error"] == "ERROR: bad sweep dir"
    assert payload["load_failure"] == {"error": "ERROR: bad sweep dir", "error_type": "OSError"}
    assert payload["exit_code"] == 1
    assert payload["status"] == "error"


def test_sweep_summary_includes_failure_metadata_from_runner(tmp_path, monkeypatch):
    output_csv = tmp_path / "sweep_results.csv"
    summary_path = default_sidecar_json_path(output_csv)
    bars = {
        "BTCUSD": pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-01-31", tz="UTC")],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000.0],
            }
        )
    }

    monkeypatch.setattr(sweep, "load_daily_bars", lambda data_dir, symbols: bars)
    monkeypatch.setattr(sweep, "build_windows", lambda all_bars, window_days, n_windows: [("2026-01-31", "2026-01-31")])
    monkeypatch.setattr(
        sweep,
        "run_sweep",
        lambda *args, **kwargs: (
            [{"min_sortino": 1.5, "mean_sortino": 2.0, "mean_return_pct": 3.0, "total_n_trades": 4}],
            {
                "skipped_backtest_failure_count": 2,
                "backtest_failure_samples": ["sample failure"],
                "suppressed_backtest_failure_count": 1,
                "c_sim_available": True,
                "c_sim_incompatibility_detected": False,
                "c_sim_incompatibility_issues": None,
                "c_sim_runtime_fallback_count": 1,
                "c_sim_runtime_fallback_samples": ["csim failure"],
            },
        ),
    )

    rc = sweep.main(["--symbols", "BTCUSD", "--output", str(output_csv), "--summary-json", str(summary_path)])

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["skipped_backtest_failure_count"] == 2
    assert payload["backtest_failure_samples"] == ["sample failure"]
    assert payload["suppressed_backtest_failure_count"] == 1
    assert payload["c_sim_runtime_fallback_count"] == 1
    assert payload["c_sim_runtime_fallback_samples"] == ["csim failure"]


def test_sweep_prints_warning_summary_for_partial_universe(tmp_path, monkeypatch, capsys):
    output_csv = tmp_path / "sweep_results.csv"
    bars = {
        "BTCUSD": pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-01-31", tz="UTC")],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000.0],
            }
        )
    }

    monkeypatch.setattr(sweep, "load_daily_bars", lambda data_dir, symbols: bars)
    monkeypatch.setattr(sweep, "build_windows", lambda all_bars, window_days, n_windows: [("2026-01-31", "2026-01-31")])
    monkeypatch.setattr(
        sweep,
        "run_sweep",
        lambda *args, **kwargs: [
            {"min_sortino": 1.5, "mean_sortino": 2.0, "mean_return_pct": 3.0, "total_n_trades": 4}
        ],
    )

    rc = sweep.main(["--symbols", "BTCUSD", "ETHUSD", "--output", str(output_csv), "--n-trials", "1"])

    out = capsys.readouterr().out
    summary_path = default_sidecar_json_path(output_csv)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["warnings"] == ["missing data for 1 symbol: ETHUSD"]
    assert "Warnings:" in out
    assert "  - missing data for 1 symbol: ETHUSD" in out


def test_sweep_returns_nonzero_when_runner_finds_no_valid_results(tmp_path, monkeypatch, capsys):
    output_csv = tmp_path / "sweep_results.csv"
    summary_path = default_sidecar_json_path(output_csv)
    bars = {
        "BTCUSD": pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-01-31", tz="UTC")],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000.0],
            }
        )
    }

    monkeypatch.setattr(sweep, "load_daily_bars", lambda data_dir, symbols: bars)
    monkeypatch.setattr(sweep, "build_windows", lambda all_bars, window_days, n_windows: [("2026-01-01", "2026-01-31")])
    monkeypatch.setattr(
        sweep,
        "run_sweep",
        lambda *args, **kwargs: (
            [],
            {
                "skipped_backtest_failure_count": 2,
                "backtest_failure_samples": ["sample failure"],
                "suppressed_backtest_failure_count": 0,
                "c_sim_available": False,
                "c_sim_incompatibility_detected": False,
                "c_sim_incompatibility_issues": None,
                "c_sim_runtime_fallback_count": 0,
                "c_sim_runtime_fallback_samples": [],
            },
        ),
    )

    rc = sweep.main(["--symbols", "BTCUSD", "--output", str(output_csv), "--summary-json", str(summary_path)])

    out = capsys.readouterr().out
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 1
    assert "ERROR: No valid sweep results" in out
    assert payload["follow_up_config_file"] is None
    assert payload["follow_up_config_kind"] is None
    assert payload["exit_code"] == 1
    assert payload["status"] == "error"
    assert payload["results_count"] == 0
    assert [item["name"] for item in payload["artifacts"]] == ["summary_json"]
    assert payload["skipped_backtest_failure_count"] == 2


def test_sweep_accepts_windows_alias(tmp_path, monkeypatch):
    output_csv = tmp_path / "sweep_results.csv"
    seen = {}
    bars = {
        "BTCUSD": pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-01-31", tz="UTC")],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000.0],
            }
        )
    }

    monkeypatch.setattr(sweep, "load_daily_bars", lambda data_dir, symbols: bars)

    def fake_build_windows(all_bars, window_days, n_windows):
        seen["window_days"] = window_days
        seen["n_windows"] = n_windows
        return [("2026-01-01", "2026-01-31")]

    monkeypatch.setattr(sweep, "build_windows", fake_build_windows)
    monkeypatch.setattr(
        sweep,
        "run_sweep",
        lambda *args, **kwargs: [{"min_sortino": 1.0, "mean_sortino": 1.0, "mean_return_pct": 1.0, "total_n_trades": 1}],
    )

    rc = sweep.main(["--symbols", "BTCUSD", "--output", str(output_csv), "--windows", "4"])

    assert rc == 0
    assert seen == {"window_days": 60, "n_windows": 4}


def test_sweep_accepts_start_end_aliases_for_fixed_window(tmp_path, monkeypatch):
    output_csv = tmp_path / "sweep_results.csv"
    seen = {}
    bars = {
        "BTCUSD": pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-01-31", tz="UTC")],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000.0],
            }
        )
    }

    monkeypatch.setattr(sweep, "load_daily_bars", lambda data_dir, symbols: bars)

    def fail_build_windows(*args, **kwargs):
        raise AssertionError("build_windows should not run when explicit start/end are provided")

    def fake_run_sweep(*args, **kwargs):
        seen["windows"] = args[1]
        return [{"min_sortino": 1.0, "mean_sortino": 1.0, "mean_return_pct": 1.0, "total_n_trades": 1}]

    monkeypatch.setattr(sweep, "build_windows", fail_build_windows)
    monkeypatch.setattr(sweep, "run_sweep", fake_run_sweep)

    rc = sweep.main(
        [
            "--symbols",
            "BTCUSD",
            "--output",
            str(output_csv),
            "--n-trials",
            "1",
            "--start",
            "2026-01-01",
            "--end",
            "2026-01-31",
        ]
    )

    assert rc == 0
    assert seen == {"windows": [("2026-01-01", "2026-01-31")]}


def test_sweep_requires_paired_date_range_bounds(tmp_path, monkeypatch, capsys):
    output_csv = tmp_path / "sweep_results.csv"
    def fail_load(*args, **kwargs):
        raise AssertionError("load_daily_bars should not run for invalid partial date ranges")

    monkeypatch.setattr(sweep, "load_daily_bars", fail_load)
    monkeypatch.setattr(
        sweep,
        "run_sweep",
        lambda *args, **kwargs: [{"min_sortino": 1.0, "mean_sortino": 1.0, "mean_return_pct": 1.0, "total_n_trades": 1}],
    )

    rc = sweep.main(["--symbols", "BTCUSD", "--output", str(output_csv), "--start", "2026-01-01"])

    out = capsys.readouterr().out.strip().splitlines()
    assert rc == 1
    assert out[-1] == "ERROR: --start/--start-date and --end/--end-date must be provided together."


def test_sweep_errors_when_no_rolling_windows_fit_loaded_data(tmp_path, monkeypatch, capsys):
    output_csv = tmp_path / "sweep_results.csv"
    bars = {
        "BTCUSD": pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-01-31", tz="UTC")],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000.0],
            }
        )
    }

    monkeypatch.setattr(sweep, "load_daily_bars", lambda data_dir, symbols: bars)

    def fail_run_sweep(*args, **kwargs):
        raise AssertionError("run_sweep should not run when no rolling windows fit")

    monkeypatch.setattr(sweep, "run_sweep", fail_run_sweep)

    rc = sweep.main(["--symbols", "BTCUSD", "--output", str(output_csv), "--days", "60", "--n-windows", "3"])

    out = capsys.readouterr().out.strip().splitlines()
    assert rc == 1
    assert out[-1] == "ERROR: Not enough data for rolling windows"


def test_sweep_no_window_failure_writes_default_error_sidecar(tmp_path, monkeypatch, capsys):
    output_csv = tmp_path / "sweep_results.csv"
    bars = {
        "BTCUSD": pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-01-31", tz="UTC")],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000.0],
            }
        )
    }

    monkeypatch.setattr(sweep, "load_daily_bars", lambda data_dir, symbols: bars)
    monkeypatch.setattr(
        sweep,
        "run_sweep",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("run_sweep should not run when no rolling windows fit")),
    )

    rc = sweep.main(["--symbols", "BTCUSD", "--output", str(output_csv), "--days", "60", "--n-windows", "3"])

    captured = capsys.readouterr()
    summary_path = default_sidecar_json_path(output_csv)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 1
    assert captured.out.strip().splitlines()[-1] == "ERROR: Not enough data for rolling windows"
    assert "Wrote summary JSON" not in captured.out
    assert payload["tool"] == "sweep"
    assert payload["error"] == "ERROR: Not enough data for rolling windows"
    assert payload["requested_window_count"] == 3
    assert payload["window_days"] == 60
    assert payload["status"] == "error"
    assert payload["exit_code"] == 1


def test_sweep_warns_when_only_some_rolling_windows_fit(tmp_path, monkeypatch, capsys):
    output_csv = tmp_path / "sweep_results.csv"
    bars = {
        "BTCUSD": pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-10-01", periods=100, freq="D", tz="UTC"),
                "open": [100.0] * 100,
                "high": [101.0] * 100,
                "low": [99.0] * 100,
                "close": [100.0] * 100,
                "volume": [1000.0] * 100,
            }
        )
    }

    monkeypatch.setattr(sweep, "load_daily_bars", lambda data_dir, symbols: bars)
    monkeypatch.setattr(
        sweep,
        "run_sweep",
        lambda *args, **kwargs: [{"min_sortino": 1.0, "mean_sortino": 1.0, "mean_return_pct": 1.0, "total_n_trades": 1}],
    )

    rc = sweep.main(["--symbols", "BTCUSD", "--output", str(output_csv), "--days", "60", "--n-windows", "3"])

    out = capsys.readouterr().out.strip().splitlines()
    assert rc == 0
    assert "WARN: only 1/3 rolling windows of 60 days fit within loaded data coverage" in out


@pytest.mark.parametrize(
    ("argv", "expected_error"),
    [
        (
            ["--symbols", "BTCUSD", "--output", "ignored.csv", "--start", "bad-date", "--end", "2026-01-31"],
            "ERROR: Invalid --start/--start-date value: 'bad-date'",
        ),
        (
            ["--symbols", "BTCUSD", "--output", "ignored.csv", "--start", "2026-02-01", "--end", "2026-01-31"],
            "ERROR: --start/--start-date must be on or before --end/--end-date.",
        ),
    ],
)
def test_sweep_invalid_date_range_returns_error_before_loading_data(
    argv, expected_error, tmp_path, monkeypatch, capsys
):
    argv = [str(tmp_path / "ignored.csv") if arg == "ignored.csv" else arg for arg in argv]

    def fail_load(*args, **kwargs):
        raise AssertionError("load_daily_bars should not run for invalid date ranges")

    monkeypatch.setattr(sweep, "load_daily_bars", fail_load)
    monkeypatch.setattr(
        sweep,
        "run_sweep",
        lambda *args, **kwargs: [{"min_sortino": 1.0, "mean_sortino": 1.0, "mean_return_pct": 1.0, "total_n_trades": 1}],
    )

    rc = sweep.main(argv)

    out = capsys.readouterr().out.strip().splitlines()
    assert rc == 1
    assert out[-1] == expected_error


@pytest.mark.parametrize(
    ("entrypoint", "tool", "output_name", "argv", "expected_error"),
    [
        (
            sweep.main,
            "sweep",
            "sweep_results.csv",
            ["--symbols", "BTCUSD", "--output", "{output}", "--start", "bad-date", "--end", "2026-01-31"],
            "ERROR: Invalid --start/--start-date value: 'bad-date'",
        ),
        (
            sweep_expanded.main,
            "sweep_expanded",
            "sweep_expanded.csv",
            ["--symbols", "BTCUSD", "--output", "{output}", "--start", "bad-date", "--end", "2026-01-31"],
            "ERROR: Invalid --start/--start-date value: 'bad-date'",
        ),
    ],
)
def test_sweep_invalid_date_range_writes_default_error_sidecar(entrypoint, tool, output_name, argv, expected_error, tmp_path, capsys):
    output_csv = tmp_path / output_name
    rendered_argv = [str(output_csv) if token == "{output}" else token for token in argv]

    rc = entrypoint(rendered_argv)

    captured = capsys.readouterr()
    summary_path = default_sidecar_json_path(output_csv)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 1
    assert payload["tool"] == tool
    assert payload["error"] == expected_error
    assert payload["error_type"] == "ValueError"
    assert payload["output_csv"] == str(output_csv)
    assert payload["status"] == "error"
    assert payload["exit_code"] == 1
    assert captured.out.splitlines()[0] == expected_error


def test_sweep_uses_config_file_as_baseline(tmp_path, monkeypatch, capsys):
    output_csv = tmp_path / "sweep_results.csv"
    config_path = tmp_path / "sweep_config.yaml"
    config_path.write_text(
        "config:\n"
        "  initial_cash: 5000\n"
        "  base_asset_symbol: ETHUSD\n"
        "  realistic_fill: true\n"
        "  daily_checkpoint_only: true\n",
        encoding="utf-8",
    )
    bars = {
        "BTCUSD": pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-01-31", tz="UTC")],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000.0],
            }
        )
    }
    seen = {}

    monkeypatch.setattr(sweep, "load_daily_bars", lambda data_dir, symbols: bars)
    monkeypatch.setattr(sweep, "build_windows", lambda all_bars, window_days, n_windows: [("2026-01-01", "2026-01-31")])

    def fake_run_sweep(*args, **kwargs):
        config = kwargs["base_config"]
        seen["initial_cash"] = config.initial_cash
        seen["base_asset_symbol"] = config.base_asset_symbol
        seen["realistic_fill"] = config.realistic_fill
        seen["daily_checkpoint_only"] = config.daily_checkpoint_only
        return [{"min_sortino": 1.0, "mean_sortino": 1.0, "mean_return_pct": 1.0, "total_n_trades": 1}]

    monkeypatch.setattr(sweep, "run_sweep", fake_run_sweep)

    rc = sweep.main(
        ["--symbols", "BTCUSD", "--output", str(output_csv), "--n-trials", "1", "--config-file", str(config_path)]
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert f"Loaded config overrides from {config_path}" in out
    assert seen == {
        "initial_cash": 5000.0,
        "base_asset_symbol": "ETHUSD",
        "realistic_fill": True,
        "daily_checkpoint_only": True,
    }


def test_sweep_cli_flags_override_config_file_baseline(tmp_path, monkeypatch):
    output_csv = tmp_path / "sweep_results.csv"
    config_path = tmp_path / "sweep_config.yaml"
    config_path.write_text(
        "config:\n"
        "  initial_cash: 5000\n"
        "  realistic_fill: false\n"
        "  daily_checkpoint_only: false\n",
        encoding="utf-8",
    )
    bars = {
        "BTCUSD": pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-01-31", tz="UTC")],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000.0],
            }
        )
    }
    seen = {}

    monkeypatch.setattr(sweep, "load_daily_bars", lambda data_dir, symbols: bars)
    monkeypatch.setattr(sweep, "build_windows", lambda all_bars, window_days, n_windows: [("2026-01-01", "2026-01-31")])

    def fake_run_sweep(*args, **kwargs):
        config = kwargs["base_config"]
        seen["initial_cash"] = config.initial_cash
        seen["realistic_fill"] = config.realistic_fill
        seen["daily_checkpoint_only"] = config.daily_checkpoint_only
        return [{"min_sortino": 1.0, "mean_sortino": 1.0, "mean_return_pct": 1.0, "total_n_trades": 1}]

    monkeypatch.setattr(sweep, "run_sweep", fake_run_sweep)

    rc = sweep.main(
        [
            "--symbols",
            "BTCUSD",
            "--output",
            str(output_csv),
            "--n-trials",
            "1",
            "--config-file",
            str(config_path),
            "--cash",
            "9000",
            "--realistic",
        ]
    )

    assert rc == 0
    assert seen == {
        "initial_cash": 9000.0,
        "realistic_fill": True,
        "daily_checkpoint_only": True,
    }


def test_sweep_invalid_config_file_returns_error(tmp_path, capsys):
    output_csv = tmp_path / "sweep_results.csv"
    config_path = tmp_path / "sweep_config.yaml"
    config_path.write_text("unknown_field: 1\n", encoding="utf-8")

    rc = sweep.main(
        ["--symbols", "BTCUSD", "--output", str(output_csv), "--n-trials", "1", "--config-file", str(config_path)]
    )

    out = capsys.readouterr().out.strip()
    assert rc == 1
    assert out == f"ERROR: Unsupported WorkStealConfig fields in {config_path}: unknown_field"


@pytest.mark.parametrize(
    ("entrypoint", "tool", "output_name", "config_name"),
    [
        (sweep.main, "sweep", "sweep_results.csv", "sweep_config.yaml"),
        (sweep_expanded.main, "sweep_expanded", "sweep_expanded.csv", "expanded_config.yaml"),
    ],
)
def test_sweep_invalid_config_file_writes_default_error_sidecar(entrypoint, tool, output_name, config_name, tmp_path, capsys):
    output_csv = tmp_path / output_name
    config_path = tmp_path / config_name
    config_path.write_text("unknown_field: 1\n", encoding="utf-8")

    argv = ["--symbols", "BTCUSD", "--output", str(output_csv), "--config-file", str(config_path)]
    if entrypoint is sweep.main:
        argv.extend(["--n-trials", "1"])
    else:
        argv.extend(["--max-trials", "1"])

    rc = entrypoint(argv)

    captured = capsys.readouterr()
    summary_path = default_sidecar_json_path(output_csv)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 1
    assert payload["tool"] == tool
    assert payload["error"] == f"ERROR: Unsupported WorkStealConfig fields in {config_path}: unknown_field"
    assert payload["error_type"] == "ValueError"
    assert payload["output_csv"] == str(output_csv)
    assert payload["status"] == "error"
    assert payload["exit_code"] == 1
    assert captured.out.splitlines()[0] == f"ERROR: Unsupported WorkStealConfig fields in {config_path}: unknown_field"


def test_sweep_summary_distinguishes_partial_realism_from_config_file(tmp_path, monkeypatch):
    output_csv = tmp_path / "sweep_results.csv"
    config_path = tmp_path / "sweep_config.yaml"
    config_path.write_text(
        "config:\n"
        "  realistic_fill: true\n",
        encoding="utf-8",
    )
    bars = {
        "BTCUSD": pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-01-31", tz="UTC")],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000.0],
            }
        )
    }

    monkeypatch.setattr(sweep, "load_daily_bars", lambda data_dir, symbols: bars)
    monkeypatch.setattr(sweep, "build_windows", lambda all_bars, window_days, n_windows: [("2026-01-01", "2026-01-31")])
    monkeypatch.setattr(
        sweep,
        "run_sweep",
        lambda *args, **kwargs: [{"min_sortino": 1.0, "mean_sortino": 1.0, "mean_return_pct": 1.0, "total_n_trades": 1}],
    )

    rc = sweep.main(
        ["--symbols", "BTCUSD", "--output", str(output_csv), "--n-trials", "1", "--config-file", str(config_path)]
    )

    summary_path = default_sidecar_json_path(output_csv)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["execution_mode"] == "TOUCH_FILL_ONLY"
    assert payload["realistic"] is False
    assert payload["realistic_fill"] is True
    assert payload["daily_checkpoint_only"] is False


def test_sweep_expanded_writes_default_summary_sidecar(tmp_path, monkeypatch, capsys):
    output_csv = tmp_path / "expanded_results.csv"
    bars = {
        "BTCUSD": pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-01-31", tz="UTC")],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000.0],
            }
        )
    }

    monkeypatch.setattr(sweep_expanded, "load_daily_bars", lambda data_dir, symbols: bars)
    monkeypatch.setattr(
        sweep_expanded,
        "build_windows",
        lambda all_bars, window_days, n_windows: [("2026-01-01", "2026-01-31")],
    )
    monkeypatch.setattr(
        sweep_expanded,
        "run_sweep",
        lambda *args, **kwargs: [
            {"safety_score": 4.2, "mean_sortino": 2.5, "mean_return_pct": 3.0, "total_n_trades": 6}
        ],
    )

    rc = sweep_expanded.main(["--symbols", "BTCUSD", "--output", str(output_csv), "--max-trials", "1"])

    out = capsys.readouterr().out
    summary_path = default_sidecar_json_path(output_csv)
    best_config_path = default_best_config_path(output_csv)
    best_overrides_path = default_best_overrides_path(output_csv)
    next_steps_script_path = default_next_steps_script_path(output_csv)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert "Wrote summary JSON" not in out
    assert "Wrote recommended config YAML" not in out
    assert "Wrote recommended overrides YAML" not in out
    assert "Wrote next steps shell script" not in out
    assert "Generated artifacts:" in out
    assert "Reproduce:" in out
    assert f"python -m binance_worksteal.sweep_expanded --symbols BTCUSD --output {output_csv} --max-trials 1" in out
    assert str(summary_path) in out
    assert str(output_csv) in out
    assert str(best_config_path) in out
    assert str(best_overrides_path) in out
    assert str(next_steps_script_path) in out
    assert "Next steps:" in out
    assert "python -m binance_worksteal.backtest" in out
    assert "python -m binance_worksteal.sim_vs_live_audit" in out
    assert "python -m binance_worksteal.evaluate_symbols" in out
    assert payload["summary_schema_version"] == 1
    assert payload["tool"] == "sweep_expanded"
    assert payload["requested_symbol_count"] == 1
    assert payload["config"]["initial_cash"] == pytest.approx(10000.0)
    assert payload["base_config"]["initial_cash"] == pytest.approx(10000.0)
    assert payload["recommended_config_file"] == str(best_config_path)
    assert payload["recommended_config"]["initial_cash"] == pytest.approx(10000.0)
    assert payload["recommended_overrides_file"] == str(best_overrides_path)
    assert payload["recommended_overrides"] == {"sma_filter_period": 20}
    assert payload["follow_up_config_file"] == str(best_overrides_path)
    assert payload["next_steps_script_file"] == str(next_steps_script_path)
    assert payload["summary_json_file"] == str(summary_path)
    assert payload["follow_up_config_kind"] == "standalone_overrides"
    assert [item["name"] for item in payload["artifacts"]] == [
        "summary_json",
        "results_csv",
        "recommended_config",
        "recommended_overrides",
        "next_steps_script",
    ]
    assert [item["name"] for item in payload["next_steps"]] == [
        "backtest",
        "sim_vs_live_audit",
        "evaluate_symbols",
    ]
    assert str(best_overrides_path) in payload["next_steps"][0]["argv"]
    assert payload["skipped_backtest_failure_count"] == 0
    assert payload["backtest_failure_samples"] == []
    assert payload["c_batch_runtime_fallback_count"] == 0
    assert payload["best_result"]["safety_score"] == 4.2
    best_config_payload = yaml.safe_load(best_config_path.read_text(encoding="utf-8"))
    assert best_config_payload["config"]["initial_cash"] == pytest.approx(10000.0)
    best_overrides_payload = yaml.safe_load(best_overrides_path.read_text(encoding="utf-8"))
    assert best_overrides_payload["config"] == {"sma_filter_period": 20}
    next_steps_script = next_steps_script_path.read_text(encoding="utf-8")
    assert next_steps_script.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert "python -m binance_worksteal.backtest" in next_steps_script


def test_sweep_expanded_summary_includes_failure_metadata_from_runner(tmp_path, monkeypatch):
    output_csv = tmp_path / "expanded_results.csv"
    summary_path = default_sidecar_json_path(output_csv)
    bars = {
        "BTCUSD": pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-01-31", tz="UTC")],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000.0],
            }
        )
    }

    monkeypatch.setattr(sweep_expanded, "load_daily_bars", lambda data_dir, symbols: bars)
    monkeypatch.setattr(sweep_expanded, "build_windows", lambda all_bars, window_days, n_windows: [("2026-01-01", "2026-01-31")])
    monkeypatch.setattr(
        sweep_expanded,
        "run_sweep",
        lambda *args, **kwargs: (
            [{"safety_score": 4.2, "mean_sortino": 2.5, "mean_return_pct": 3.0, "total_n_trades": 6}],
            {
                "skipped_backtest_failure_count": 3,
                "backtest_failure_samples": ["python failure"],
                "suppressed_backtest_failure_count": 2,
                "c_batch_available": True,
                "c_batch_used": True,
                "c_batch_incompatibility_detected": False,
                "c_batch_incompatibility_issues": None,
                "c_batch_runtime_fallback_count": 1,
                "c_batch_runtime_fallback_samples": ["batch failure"],
            },
        ),
    )

    rc = sweep_expanded.main(["--symbols", "BTCUSD", "--output", str(output_csv), "--summary-json", str(summary_path)])

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["skipped_backtest_failure_count"] == 3
    assert payload["backtest_failure_samples"] == ["python failure"]
    assert payload["suppressed_backtest_failure_count"] == 2
    assert payload["c_batch_used"] is True
    assert payload["c_batch_runtime_fallback_count"] == 1
    assert payload["c_batch_runtime_fallback_samples"] == ["batch failure"]


def test_sweep_expanded_writes_summary_json_on_no_data(tmp_path, monkeypatch, capsys):
    output_csv = tmp_path / "expanded_results.csv"
    summary_path = default_sidecar_json_path(output_csv)

    monkeypatch.setattr(sweep_expanded, "load_daily_bars", lambda data_dir, symbols: {})

    def fail_windows(*args, **kwargs):
        raise AssertionError("build_windows should not run when no data loads")

    monkeypatch.setattr(sweep_expanded, "build_windows", fail_windows)

    rc = sweep_expanded.main([
        "--symbols", "BTCUSD",
        "--output", str(output_csv),
        "--summary-json", str(summary_path),
    ])

    out = capsys.readouterr().out.strip().splitlines()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 1
    assert "ERROR: No data" in out
    assert "Diagnostic artifacts:" in out
    assert f"  summary_json: {summary_path}" in out
    assert "Reproduce:" in out
    assert "Wrote summary JSON" not in "\n".join(out)
    assert payload["results_count"] == 0
    assert payload["loaded_symbol_count"] == 0
    assert payload["missing_symbols"] == ["BTCUSD"]
    assert payload["error"] == "ERROR: No data"
    assert payload["exit_code"] == 1
    assert payload["status"] == "error"


def test_sweep_expanded_writes_summary_json_on_loader_error(tmp_path, monkeypatch, capsys):
    output_csv = tmp_path / "expanded_results.csv"
    summary_path = default_sidecar_json_path(output_csv)

    def raise_load_error(data_dir, symbols):
        raise OSError("bad expanded sweep dir")

    monkeypatch.setattr(sweep_expanded, "load_daily_bars", raise_load_error)

    def fail_windows(*args, **kwargs):
        raise AssertionError("build_windows should not run when load_daily_bars fails")

    monkeypatch.setattr(sweep_expanded, "build_windows", fail_windows)

    rc = sweep_expanded.main([
        "--symbols", "BTCUSD",
        "--output", str(output_csv),
        "--summary-json", str(summary_path),
    ])

    out = capsys.readouterr().out.strip().splitlines()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 1
    assert "ERROR: bad expanded sweep dir" in out
    assert "Diagnostic artifacts:" in out
    assert f"  summary_json: {summary_path}" in out
    assert "Reproduce:" in out
    assert "Wrote summary JSON" not in "\n".join(out)
    assert payload["results_count"] == 0
    assert payload["loaded_symbol_count"] == 0
    assert payload["missing_symbols"] == ["BTCUSD"]
    assert payload["error"] == "ERROR: bad expanded sweep dir"
    assert payload["load_failure"] == {"error": "ERROR: bad expanded sweep dir", "error_type": "OSError"}
    assert payload["exit_code"] == 1
    assert payload["status"] == "error"


def test_sweep_expanded_returns_nonzero_when_runner_finds_no_valid_results(tmp_path, monkeypatch, capsys):
    output_csv = tmp_path / "expanded_results.csv"
    summary_path = default_sidecar_json_path(output_csv)
    bars = {
        "BTCUSD": pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-01-31", tz="UTC")],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000.0],
            }
        )
    }

    monkeypatch.setattr(sweep_expanded, "load_daily_bars", lambda data_dir, symbols: bars)
    monkeypatch.setattr(sweep_expanded, "build_windows", lambda all_bars, window_days, n_windows: [("2026-01-01", "2026-01-31")])
    monkeypatch.setattr(
        sweep_expanded,
        "run_sweep",
        lambda *args, **kwargs: (
            [],
            {
                "skipped_backtest_failure_count": 3,
                "backtest_failure_samples": ["python failure"],
                "suppressed_backtest_failure_count": 0,
                "c_batch_available": False,
                "c_batch_used": False,
                "c_batch_incompatibility_detected": False,
                "c_batch_incompatibility_issues": None,
                "c_batch_runtime_fallback_count": 0,
                "c_batch_runtime_fallback_samples": [],
            },
        ),
    )

    rc = sweep_expanded.main(["--symbols", "BTCUSD", "--output", str(output_csv), "--summary-json", str(summary_path)])

    out = capsys.readouterr().out
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 1
    assert "ERROR: No valid expanded sweep results" in out
    assert payload["follow_up_config_file"] is None
    assert payload["follow_up_config_kind"] is None
    assert payload["results_count"] == 0
    assert [item["name"] for item in payload["artifacts"]] == ["summary_json"]
    assert payload["skipped_backtest_failure_count"] == 3


def test_sweep_expanded_accepts_n_windows_alias(tmp_path, monkeypatch):
    output_csv = tmp_path / "expanded_results.csv"
    seen = {}
    bars = {
        "BTCUSD": pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-01-31", tz="UTC")],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000.0],
            }
        )
    }

    monkeypatch.setattr(sweep_expanded, "load_daily_bars", lambda data_dir, symbols: bars)

    def fake_build_windows(all_bars, window_days, n_windows):
        seen["window_days"] = window_days
        seen["n_windows"] = n_windows
        return [("2026-01-01", "2026-01-31")]

    monkeypatch.setattr(sweep_expanded, "build_windows", fake_build_windows)
    monkeypatch.setattr(
        sweep_expanded,
        "run_sweep",
        lambda *args, **kwargs: [{"safety_score": 1.0, "mean_sortino": 1.0, "mean_return_pct": 1.0, "total_n_trades": 1}],
    )

    rc = sweep_expanded.main(["--symbols", "BTCUSD", "--output", str(output_csv), "--n-windows", "4"])

    assert rc == 0
    assert seen == {"window_days": 60, "n_windows": 4}


def test_sweep_expanded_accepts_start_end_aliases_for_fixed_window(tmp_path, monkeypatch):
    output_csv = tmp_path / "expanded_results.csv"
    seen = {}
    bars = {
        "BTCUSD": pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-01-31", tz="UTC")],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000.0],
            }
        )
    }

    monkeypatch.setattr(sweep_expanded, "load_daily_bars", lambda data_dir, symbols: bars)

    def fail_build_windows(*args, **kwargs):
        raise AssertionError("build_windows should not run when explicit start/end are provided")

    def fake_run_sweep(*args, **kwargs):
        seen["windows"] = args[1]
        return [{"safety_score": 1.0, "mean_sortino": 1.0, "mean_return_pct": 1.0, "total_n_trades": 1}]

    monkeypatch.setattr(sweep_expanded, "build_windows", fail_build_windows)
    monkeypatch.setattr(sweep_expanded, "run_sweep", fake_run_sweep)

    rc = sweep_expanded.main(
        [
            "--symbols",
            "BTCUSD",
            "--output",
            str(output_csv),
            "--max-trials",
            "1",
            "--start",
            "2026-01-01",
            "--end",
            "2026-01-31",
        ]
    )

    assert rc == 0
    assert seen == {"windows": [("2026-01-01", "2026-01-31")]}


def test_sweep_expanded_requires_paired_date_range_bounds(tmp_path, monkeypatch, capsys):
    output_csv = tmp_path / "expanded_results.csv"
    def fail_load(*args, **kwargs):
        raise AssertionError("load_daily_bars should not run for invalid partial date ranges")

    monkeypatch.setattr(sweep_expanded, "load_daily_bars", fail_load)
    monkeypatch.setattr(
        sweep_expanded,
        "run_sweep",
        lambda *args, **kwargs: [{"safety_score": 1.0, "mean_sortino": 1.0, "mean_return_pct": 1.0, "total_n_trades": 1}],
    )

    rc = sweep_expanded.main(
        ["--symbols", "BTCUSD", "--output", str(output_csv), "--start", "2026-01-01"]
    )

    out = capsys.readouterr().out.strip().splitlines()
    assert rc == 1
    assert out[-1] == "ERROR: --start/--start-date and --end/--end-date must be provided together."


def test_sweep_expanded_errors_when_no_rolling_windows_fit_loaded_data(tmp_path, monkeypatch, capsys):
    output_csv = tmp_path / "expanded_results.csv"
    bars = {
        "BTCUSD": pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-01-31", tz="UTC")],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000.0],
            }
        )
    }

    monkeypatch.setattr(sweep_expanded, "load_daily_bars", lambda data_dir, symbols: bars)

    def fail_run_sweep(*args, **kwargs):
        raise AssertionError("run_sweep should not run when no rolling windows fit")

    monkeypatch.setattr(sweep_expanded, "run_sweep", fail_run_sweep)

    rc = sweep_expanded.main(
        ["--symbols", "BTCUSD", "--output", str(output_csv), "--days", "60", "--windows", "3"]
    )

    out = capsys.readouterr().out.strip().splitlines()
    assert rc == 1
    assert out[-1] == "ERROR: Not enough data for rolling windows"


def test_sweep_expanded_no_window_failure_writes_default_error_sidecar(tmp_path, monkeypatch, capsys):
    output_csv = tmp_path / "expanded_results.csv"
    bars = {
        "BTCUSD": pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-01-31", tz="UTC")],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000.0],
            }
        )
    }

    monkeypatch.setattr(sweep_expanded, "load_daily_bars", lambda data_dir, symbols: bars)
    monkeypatch.setattr(
        sweep_expanded,
        "run_sweep",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("run_sweep should not run when no rolling windows fit")),
    )

    rc = sweep_expanded.main(
        ["--symbols", "BTCUSD", "--output", str(output_csv), "--days", "60", "--windows", "3"]
    )

    captured = capsys.readouterr()
    summary_path = default_sidecar_json_path(output_csv)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 1
    assert captured.out.strip().splitlines()[-1] == "ERROR: Not enough data for rolling windows"
    assert "Wrote summary JSON" not in captured.out
    assert payload["tool"] == "sweep_expanded"
    assert payload["error"] == "ERROR: Not enough data for rolling windows"
    assert payload["requested_window_count"] == 3
    assert payload["window_days"] == 60
    assert payload["status"] == "error"
    assert payload["exit_code"] == 1


@pytest.mark.parametrize(
    ("argv", "expected_error"),
    [
        (
            ["--symbols", "BTCUSD", "--output", "ignored.csv", "--start", "bad-date", "--end", "2026-01-31"],
            "ERROR: Invalid --start/--start-date value: 'bad-date'",
        ),
        (
            ["--symbols", "BTCUSD", "--output", "ignored.csv", "--start", "2026-02-01", "--end", "2026-01-31"],
            "ERROR: --start/--start-date must be on or before --end/--end-date.",
        ),
    ],
)
def test_sweep_expanded_invalid_date_range_returns_error_before_loading_data(
    argv, expected_error, tmp_path, monkeypatch, capsys
):
    argv = [str(tmp_path / "ignored.csv") if arg == "ignored.csv" else arg for arg in argv]

    def fail_load(*args, **kwargs):
        raise AssertionError("load_daily_bars should not run for invalid date ranges")

    monkeypatch.setattr(sweep_expanded, "load_daily_bars", fail_load)
    monkeypatch.setattr(
        sweep_expanded,
        "run_sweep",
        lambda *args, **kwargs: [{"safety_score": 1.0, "mean_sortino": 1.0, "mean_return_pct": 1.0, "total_n_trades": 1}],
    )

    rc = sweep_expanded.main(argv)

    out = capsys.readouterr().out.strip().splitlines()
    assert rc == 1
    assert out[-1] == expected_error


def test_sweep_expanded_uses_config_file_as_baseline(tmp_path, monkeypatch, capsys):
    output_csv = tmp_path / "expanded_results.csv"
    config_path = tmp_path / "expanded_config.yaml"
    config_path.write_text(
        "config:\n"
        "  initial_cash: 7000\n"
        "  base_asset_symbol: ETHUSD\n"
        "  max_hold_days: 30\n",
        encoding="utf-8",
    )
    bars = {
        "BTCUSD": pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-01-31", tz="UTC")],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000.0],
            }
        )
    }
    seen = {}

    monkeypatch.setattr(sweep_expanded, "load_daily_bars", lambda data_dir, symbols: bars)
    monkeypatch.setattr(
        sweep_expanded,
        "build_windows",
        lambda all_bars, window_days, n_windows: [("2026-01-01", "2026-01-31")],
    )

    def fake_run_sweep(*args, **kwargs):
        config = kwargs["base_config"]
        seen["initial_cash"] = config.initial_cash
        seen["base_asset_symbol"] = config.base_asset_symbol
        seen["max_hold_days"] = config.max_hold_days
        return [{"safety_score": 1.0, "mean_sortino": 1.0, "mean_return_pct": 1.0, "total_n_trades": 1}]

    monkeypatch.setattr(sweep_expanded, "run_sweep", fake_run_sweep)

    rc = sweep_expanded.main(
        ["--symbols", "BTCUSD", "--output", str(output_csv), "--max-trials", "1", "--config-file", str(config_path)]
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert f"Loaded config overrides from {config_path}" in out
    assert seen == {
        "initial_cash": 7000.0,
        "base_asset_symbol": "ETHUSD",
        "max_hold_days": 30,
    }


def test_sweep_expanded_cash_flag_overrides_config_file(tmp_path, monkeypatch):
    output_csv = tmp_path / "expanded_results.csv"
    config_path = tmp_path / "expanded_config.yaml"
    config_path.write_text("initial_cash: 7000\n", encoding="utf-8")
    bars = {
        "BTCUSD": pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-01-31", tz="UTC")],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000.0],
            }
        )
    }
    seen = {}

    monkeypatch.setattr(sweep_expanded, "load_daily_bars", lambda data_dir, symbols: bars)
    monkeypatch.setattr(
        sweep_expanded,
        "build_windows",
        lambda all_bars, window_days, n_windows: [("2026-01-01", "2026-01-31")],
    )

    def fake_run_sweep(*args, **kwargs):
        seen["initial_cash"] = kwargs["base_config"].initial_cash
        return [{"safety_score": 1.0, "mean_sortino": 1.0, "mean_return_pct": 1.0, "total_n_trades": 1}]

    monkeypatch.setattr(sweep_expanded, "run_sweep", fake_run_sweep)

    rc = sweep_expanded.main(
        [
            "--symbols",
            "BTCUSD",
            "--output",
            str(output_csv),
            "--max-trials",
            "1",
            "--config-file",
            str(config_path),
            "--cash",
            "9000",
        ]
    )

    assert rc == 0
    assert seen == {"initial_cash": 9000.0}


def test_sweep_expanded_invalid_config_file_returns_error(tmp_path, capsys):
    output_csv = tmp_path / "expanded_results.csv"
    config_path = tmp_path / "expanded_config.yaml"
    config_path.write_text("unknown_field: 1\n", encoding="utf-8")

    rc = sweep_expanded.main(
        [
            "--symbols",
            "BTCUSD",
            "--output",
            str(output_csv),
            "--max-trials",
            "1",
            "--config-file",
            str(config_path),
        ]
    )

    out = capsys.readouterr().out.strip()
    assert rc == 1
    assert out == f"ERROR: Unsupported WorkStealConfig fields in {config_path}: unknown_field"


def test_sim_vs_live_audit_writes_summary_json(tmp_path, monkeypatch, capsys):
    summary_path = tmp_path / "audit_summary.json"
    bars = {
        "BTCUSD": pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-01-31", tz="UTC")],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000.0],
                "symbol": ["BTCUSD"],
            }
        )
    }
    audit_df = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-01-31", tz="UTC"),
                "symbol": "BTCUSD",
                "sma_blocks": False,
                "proximity_blocks": False,
                "momentum_blocks": False,
                "is_candidate": True,
                "would_fill_realistic": True,
                "proximity_bps": 5.0,
            }
        ]
    )

    monkeypatch.setattr(sim_vs_live_audit, "load_daily_bars", lambda data_dir, symbols: bars)
    monkeypatch.setattr(sim_vs_live_audit, "audit_entries", lambda *args, **kwargs: audit_df)
    monkeypatch.setattr(sim_vs_live_audit, "print_audit_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(sim_vs_live_audit, "run_comparison", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        sim_vs_live_audit,
        "build_comparison_summary",
        lambda *args, **kwargs: {"default": {"sortino": 1.0}, "realistic": {"sortino": 0.8}},
    )

    rc = sim_vs_live_audit.main(
        ["--symbols", "BTCUSD", "--start", "2026-01-01", "--end", "2026-01-31", "--summary-json", str(summary_path)]
    )

    out = capsys.readouterr().out
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert "Generated artifacts:" in out
    assert f"  summary_json: {summary_path}" in out
    assert "Reproduce:" in out
    assert payload["invocation"]["command"] in out
    assert "Wrote summary JSON" not in out
    assert payload["summary_schema_version"] == 1
    assert payload["tool"] == "sim_vs_live_audit"
    assert payload["requested_symbol_count"] == 1
    assert payload["audit_summary"]["candidate_count"] == 1
    assert payload["comparison"]["realistic"]["sortino"] == 0.8


def test_sim_vs_live_audit_returns_error_on_audit_entries_failure(monkeypatch, capsys):
    bars = {
        "BTCUSD": pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-01-31", tz="UTC")],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000.0],
                "symbol": ["BTCUSD"],
            }
        )
    }

    monkeypatch.setattr(sim_vs_live_audit, "load_daily_bars", lambda data_dir, symbols: bars)

    def fail_audit_entries(*args, **kwargs):
        raise RuntimeError("audit boom")

    monkeypatch.setattr(sim_vs_live_audit, "audit_entries", fail_audit_entries)

    rc = sim_vs_live_audit.main(
        ["--symbols", "BTCUSD", "--start", "2026-01-01", "--end", "2026-01-31"]
    )

    out = capsys.readouterr().out
    assert rc == 1
    assert "ERROR: sim_vs_live_audit audit_entries python evaluation failed" in out
    assert "RuntimeError: audit boom" in out


def test_sim_vs_live_audit_writes_failure_summary_json(tmp_path, monkeypatch, capsys):
    summary_path = tmp_path / "audit_summary.json"
    bars = {
        "BTCUSD": pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-01-31", tz="UTC")],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000.0],
                "symbol": ["BTCUSD"],
            }
        )
    }
    audit_df = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-01-31", tz="UTC"),
                "symbol": "BTCUSD",
                "sma_blocks": False,
                "proximity_blocks": False,
                "momentum_blocks": False,
                "is_candidate": True,
                "would_fill_realistic": True,
                "proximity_bps": 5.0,
            }
        ]
    )

    monkeypatch.setattr(sim_vs_live_audit, "load_daily_bars", lambda data_dir, symbols: bars)
    monkeypatch.setattr(sim_vs_live_audit, "audit_entries", lambda *args, **kwargs: audit_df)
    monkeypatch.setattr(sim_vs_live_audit, "print_audit_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        sim_vs_live_audit,
        "build_comparison_summary",
        lambda *args, **kwargs: {"default": {"sortino": 1.0}, "realistic": {"sortino": 0.8}},
    )

    def fail_run_comparison(*args, **kwargs):
        raise RuntimeError("comparison boom")

    monkeypatch.setattr(sim_vs_live_audit, "run_comparison", fail_run_comparison)

    rc = sim_vs_live_audit.main(
        [
            "--symbols",
            "BTCUSD",
            "--start",
            "2026-01-01",
            "--end",
            "2026-01-31",
            "--summary-json",
            str(summary_path),
        ]
    )

    out = capsys.readouterr().out
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 1
    assert "ERROR: sim_vs_live_audit run_comparison reporting evaluation failed" in out
    assert "Diagnostic artifacts:" in out
    assert f"  summary_json: {summary_path}" in out
    assert "Reproduce:" in out
    assert "Wrote summary JSON" not in out
    assert payload["exit_code"] == 1
    assert payload["status"] == "error"
    assert payload["audit_failure"]["stage"] == "run_comparison"
    assert payload["audit_failure"]["error_type"] == "RuntimeError"
    assert payload["comparison"]["default"]["sortino"] == 1.0
    assert payload["audit_summary"]["candidate_count"] == 1


def test_sim_vs_live_audit_writes_summary_json_on_no_data(tmp_path, monkeypatch, capsys):
    summary_path = tmp_path / "audit_summary.json"

    monkeypatch.setattr(sim_vs_live_audit, "load_daily_bars", lambda data_dir, symbols: {})

    def fail_audit_entries(*args, **kwargs):
        raise AssertionError("audit_entries should not run when no data loads")

    monkeypatch.setattr(sim_vs_live_audit, "audit_entries", fail_audit_entries)

    rc = sim_vs_live_audit.main(
        ["--symbols", "BTCUSD", "--start", "2026-01-01", "--end", "2026-01-31", "--summary-json", str(summary_path)]
    )

    out = capsys.readouterr().out.strip().splitlines()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 1
    assert "ERROR: No data loaded" in out
    assert "Diagnostic artifacts:" in out
    assert f"  summary_json: {summary_path}" in out
    assert "Reproduce:" in out
    assert "Wrote summary JSON" not in "\n".join(out)
    assert payload["loaded_symbol_count"] == 0
    assert payload["missing_symbols"] == ["BTCUSD"]
    assert payload["audit_summary"] is None
    assert payload["comparison"] is None
    assert payload["audit_failure"]["stage"] == "load_bars"
    assert payload["audit_failure"]["error_type"] == "NoDataLoaded"
    assert payload["exit_code"] == 1
    assert payload["status"] == "error"


def test_sim_vs_live_audit_writes_summary_json_on_loader_error(tmp_path, monkeypatch, capsys):
    summary_path = tmp_path / "audit_summary.json"

    def raise_load_error(data_dir, symbols):
        raise OSError("bad audit dir")

    monkeypatch.setattr(sim_vs_live_audit, "load_daily_bars", raise_load_error)

    def fail_audit_entries(*args, **kwargs):
        raise AssertionError("audit_entries should not run when load_daily_bars fails")

    monkeypatch.setattr(sim_vs_live_audit, "audit_entries", fail_audit_entries)

    rc = sim_vs_live_audit.main(
        ["--symbols", "BTCUSD", "--start", "2026-01-01", "--end", "2026-01-31", "--summary-json", str(summary_path)]
    )

    out = capsys.readouterr().out.strip().splitlines()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 1
    assert "ERROR: bad audit dir" in out
    assert "Diagnostic artifacts:" in out
    assert f"  summary_json: {summary_path}" in out
    assert "Reproduce:" in out
    assert "Wrote summary JSON" not in "\n".join(out)
    assert payload["loaded_symbol_count"] == 0
    assert payload["missing_symbols"] == ["BTCUSD"]
    assert payload["audit_summary"] is None
    assert payload["comparison"] is None
    assert payload["audit_failure"]["stage"] == "load_bars"
    assert payload["audit_failure"]["error_type"] == "OSError"
    assert payload["audit_failure"]["error"] == "ERROR: bad audit dir"
    assert payload["exit_code"] == 1
    assert payload["status"] == "error"


def test_evaluate_symbols_accepts_n_windows_alias(monkeypatch, capsys):
    seen = {}

    def fake_run_evaluation(**kwargs):
        seen["window_days"] = kwargs["window_days"]
        seen["n_windows"] = kwargs["n_windows"]
        return "ok", [{"symbol": "BTCUSD", "marginal_contribution": 0.1}], {
            "windows": [],
            "avg_full_return": 0.0,
            "avg_full_sortino": 0.0,
            "base_symbol_count": 1,
            "evaluated_symbol_count": 1,
        }

    monkeypatch.setattr(evaluate_symbols, "run_evaluation", fake_run_evaluation)

    rc = evaluate_symbols.main(["--symbols", "BTCUSD", "--n-windows", "4"])

    out = capsys.readouterr().out
    assert rc == 0
    assert "ok" in out
    assert seen == {"window_days": 60, "n_windows": 4}


def test_evaluate_symbols_accepts_start_date_end_date_aliases(monkeypatch, capsys):
    seen = {}

    def fake_run_evaluation(**kwargs):
        seen["start_date"] = kwargs["start_date"]
        seen["end_date"] = kwargs["end_date"]
        return "ok", [{"symbol": "BTCUSD", "marginal_contribution": 0.1}], {
            "windows": [("2026-01-01", "2026-01-31")],
            "window_mode": "fixed",
            "start_date": "2026-01-01",
            "end_date": "2026-01-31",
            "avg_full_return": 0.0,
            "avg_full_sortino": 0.0,
            "base_symbol_count": 1,
            "evaluated_symbol_count": 1,
        }

    monkeypatch.setattr(evaluate_symbols, "run_evaluation", fake_run_evaluation)

    rc = evaluate_symbols.main(["--symbols", "BTCUSD", "--start-date", "2026-01-01", "--end-date", "2026-01-31"])

    out = capsys.readouterr().out
    assert rc == 0
    assert "ok" in out
    assert seen == {"start_date": "2026-01-01", "end_date": "2026-01-31"}


def test_evaluate_symbols_requires_paired_date_range_bounds(monkeypatch, capsys):
    def fail_run_evaluation(**kwargs):
        raise AssertionError("run_evaluation should not run for invalid partial date ranges")

    monkeypatch.setattr(evaluate_symbols, "run_evaluation", fail_run_evaluation)

    rc = evaluate_symbols.main(["--symbols", "BTCUSD", "--start", "2026-01-01"])

    out = capsys.readouterr().out.strip().splitlines()
    assert rc == 1
    assert out[-1] == "ERROR: --start/--start-date and --end/--end-date must be provided together."


def test_evaluate_symbols_uses_config_file_overrides(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "evaluate_config.yaml"
    config_path.write_text(
        "config:\n"
        "  dip_pct: 0.18\n"
        "  sma_filter_period: 7\n"
        "  realistic_fill: true\n",
        encoding="utf-8",
    )
    seen = {}

    def fake_run_evaluation(**kwargs):
        config = kwargs["config"]
        seen["dip_pct"] = config.dip_pct
        seen["sma_filter_period"] = config.sma_filter_period
        seen["realistic_fill"] = config.realistic_fill
        return "ok", [{"symbol": "BTCUSD", "marginal_contribution": 0.1}], {
            "windows": [],
            "avg_full_return": 0.0,
            "avg_full_sortino": 0.0,
            "base_symbol_count": 1,
            "evaluated_symbol_count": 1,
        }

    monkeypatch.setattr(evaluate_symbols, "run_evaluation", fake_run_evaluation)

    rc = evaluate_symbols.main(["--symbols", "BTCUSD", "--config-file", str(config_path)])

    out = capsys.readouterr().out
    assert rc == 0
    assert f"Loaded config overrides from {config_path}" in out
    assert seen == {"dip_pct": 0.18, "sma_filter_period": 7, "realistic_fill": True}


def test_evaluate_symbols_cli_flags_override_config_file(tmp_path, monkeypatch):
    config_path = tmp_path / "evaluate_config.yaml"
    config_path.write_text("dip_pct: 0.18\n", encoding="utf-8")
    seen = {}

    def fake_run_evaluation(**kwargs):
        seen["dip_pct"] = kwargs["config"].dip_pct
        return "ok", [{"symbol": "BTCUSD", "marginal_contribution": 0.1}], {
            "windows": [],
            "avg_full_return": 0.0,
            "avg_full_sortino": 0.0,
            "base_symbol_count": 1,
            "evaluated_symbol_count": 1,
        }

    monkeypatch.setattr(evaluate_symbols, "run_evaluation", fake_run_evaluation)

    rc = evaluate_symbols.main(
        ["--symbols", "BTCUSD", "--config-file", str(config_path), "--dip-pct", "0.25"]
    )

    assert rc == 0
    assert seen == {"dip_pct": 0.25}


def test_evaluate_symbols_invalid_config_file_returns_error(tmp_path, capsys):
    config_path = tmp_path / "evaluate_config.yaml"
    config_path.write_text("unknown_field: 1\n", encoding="utf-8")

    rc = evaluate_symbols.main(["--symbols", "BTCUSD", "--config-file", str(config_path)])

    out = capsys.readouterr().out.strip()
    assert rc == 1
    assert out == f"ERROR: Unsupported WorkStealConfig fields in {config_path}: unknown_field"


def test_sim_vs_live_audit_accepts_start_date_end_date_aliases(monkeypatch):
    seen = {}
    bars = {
        "BTCUSD": pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-01-31", tz="UTC")],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000.0],
                "symbol": ["BTCUSD"],
            }
        )
    }
    audit_df = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-01-31", tz="UTC"),
                "symbol": "BTCUSD",
                "sma_blocks": False,
                "proximity_blocks": False,
                "momentum_blocks": False,
                "is_candidate": True,
                "would_fill_realistic": True,
                "proximity_bps": 5.0,
            }
        ]
    )

    monkeypatch.setattr(sim_vs_live_audit, "load_daily_bars", lambda data_dir, symbols: bars)

    def fake_audit_entries(*args, **kwargs):
        seen["start_date"] = kwargs["start_date"]
        seen["end_date"] = kwargs["end_date"]
        return audit_df

    monkeypatch.setattr(sim_vs_live_audit, "audit_entries", fake_audit_entries)
    monkeypatch.setattr(sim_vs_live_audit, "print_audit_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        sim_vs_live_audit,
        "build_comparison_summary",
        lambda *args, **kwargs: {"default": {"sortino": 1.0}, "realistic": {"sortino": 0.8}},
    )
    monkeypatch.setattr(sim_vs_live_audit, "run_comparison", lambda *args, **kwargs: None)

    rc = sim_vs_live_audit.main(
        ["--symbols", "BTCUSD", "--start-date", "2026-01-01", "--end-date", "2026-01-31"]
    )

    assert rc == 0
    assert seen == {"start_date": "2026-01-01", "end_date": "2026-01-31"}


def test_sim_vs_live_audit_uses_config_file_overrides(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "audit_config.yaml"
    config_path.write_text(
        "config:\n"
        "  dip_pct: 0.18\n"
        "  sma_filter_period: 7\n"
        "  realistic_fill: true\n",
        encoding="utf-8",
    )
    seen = {}
    bars = {
        "BTCUSD": pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-01-31", tz="UTC")],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000.0],
                "symbol": ["BTCUSD"],
            }
        )
    }
    audit_df = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-01-31", tz="UTC"),
                "symbol": "BTCUSD",
                "sma_blocks": False,
                "proximity_blocks": False,
                "momentum_blocks": False,
                "is_candidate": True,
                "would_fill_realistic": True,
                "proximity_bps": 5.0,
            }
        ]
    )

    monkeypatch.setattr(sim_vs_live_audit, "load_daily_bars", lambda data_dir, symbols: bars)

    def fake_audit_entries(all_bars, config, start_date, end_date):
        seen["dip_pct"] = config.dip_pct
        seen["sma_filter_period"] = config.sma_filter_period
        seen["realistic_fill"] = config.realistic_fill
        return audit_df

    monkeypatch.setattr(sim_vs_live_audit, "audit_entries", fake_audit_entries)
    monkeypatch.setattr(sim_vs_live_audit, "print_audit_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        sim_vs_live_audit,
        "build_comparison_summary",
        lambda *args, **kwargs: {"default": {"sortino": 1.0}, "realistic": {"sortino": 0.8}},
    )
    monkeypatch.setattr(sim_vs_live_audit, "run_comparison", lambda *args, **kwargs: None)

    rc = sim_vs_live_audit.main(
        [
            "--symbols",
            "BTCUSD",
            "--start",
            "2026-01-01",
            "--end",
            "2026-01-31",
            "--config-file",
            str(config_path),
        ]
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert f"Loaded config overrides from {config_path}" in out
    assert seen == {"dip_pct": 0.18, "sma_filter_period": 7, "realistic_fill": True}


def test_sim_vs_live_audit_cli_flags_override_config_file(tmp_path, monkeypatch):
    config_path = tmp_path / "audit_config.yaml"
    config_path.write_text("dip_pct: 0.18\n", encoding="utf-8")
    seen = {}
    bars = {
        "BTCUSD": pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-01-31", tz="UTC")],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000.0],
                "symbol": ["BTCUSD"],
            }
        )
    }
    audit_df = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-01-31", tz="UTC"),
                "symbol": "BTCUSD",
                "sma_blocks": False,
                "proximity_blocks": False,
                "momentum_blocks": False,
                "is_candidate": True,
                "would_fill_realistic": True,
                "proximity_bps": 5.0,
            }
        ]
    )

    monkeypatch.setattr(sim_vs_live_audit, "load_daily_bars", lambda data_dir, symbols: bars)

    def fake_audit_entries(all_bars, config, start_date, end_date):
        seen["dip_pct"] = config.dip_pct
        return audit_df

    monkeypatch.setattr(sim_vs_live_audit, "audit_entries", fake_audit_entries)
    monkeypatch.setattr(sim_vs_live_audit, "print_audit_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        sim_vs_live_audit,
        "build_comparison_summary",
        lambda *args, **kwargs: {"default": {"sortino": 1.0}, "realistic": {"sortino": 0.8}},
    )
    monkeypatch.setattr(sim_vs_live_audit, "run_comparison", lambda *args, **kwargs: None)

    rc = sim_vs_live_audit.main(
        [
            "--symbols",
            "BTCUSD",
            "--start",
            "2026-01-01",
            "--end",
            "2026-01-31",
            "--config-file",
            str(config_path),
            "--dip-pct",
            "0.25",
        ]
    )

    assert rc == 0
    assert seen == {"dip_pct": 0.25}


def test_sim_vs_live_audit_invalid_config_file_returns_error(tmp_path, capsys):
    config_path = tmp_path / "audit_config.yaml"
    config_path.write_text("unknown_field: 1\n", encoding="utf-8")

    rc = sim_vs_live_audit.main(["--symbols", "BTCUSD", "--config-file", str(config_path)])

    out = capsys.readouterr().out.strip()
    assert rc == 1
    assert out == (
        f"ERROR: Unsupported WorkStealConfig fields in {config_path}: unknown_field"
    )


def test_sim_vs_live_audit_malformed_config_file_returns_error(tmp_path, capsys):
    config_path = tmp_path / "audit_config.yaml"
    config_path.write_text("config: [\n", encoding="utf-8")

    rc = sim_vs_live_audit.main(["--symbols", "BTCUSD", "--config-file", str(config_path)])

    out = capsys.readouterr().out.strip()
    assert rc == 1
    assert out == f"ERROR: Invalid config file format: {config_path}"


@pytest.mark.parametrize(
    ("argv", "expected_error"),
    [
        (
            ["--symbols", "BTCUSD", "--start", "bad-date", "--end", "2026-01-31"],
            "ERROR: Invalid --start/--start-date value: 'bad-date'",
        ),
        (
            ["--symbols", "BTCUSD", "--start", "2026-02-01", "--end", "2026-01-31"],
            "ERROR: --start/--start-date must be on or before --end/--end-date.",
        ),
    ],
)
def test_sim_vs_live_audit_invalid_date_range_returns_error_before_loading_data(
    argv, expected_error, monkeypatch, capsys
):
    def fail_load(*args, **kwargs):
        raise AssertionError("load_daily_bars should not run for invalid date ranges")

    monkeypatch.setattr(sim_vs_live_audit, "load_daily_bars", fail_load)

    rc = sim_vs_live_audit.main(argv)

    out = capsys.readouterr().out.strip().splitlines()
    assert rc == 1
    assert out[-1] == expected_error


def test_build_comparison_summary_uses_default_and_realistic_configs(monkeypatch):
    seen = []

    def fake_backtest(all_bars, config, start_date, end_date):
        seen.append((bool(config.realistic_fill), bool(config.daily_checkpoint_only)))
        if config.realistic_fill:
            return pd.DataFrame(), [], {"sortino": 0.8, "entries_executed": 2, "fill_rate": 0.2}
        return pd.DataFrame(), [], {"sortino": 1.1, "entries_executed": 4, "fill_rate": 0.4}

    monkeypatch.setattr(sim_vs_live_audit, "run_worksteal_backtest", fake_backtest)

    summary = sim_vs_live_audit.build_comparison_summary(
        all_bars={"BTCUSD": pd.DataFrame()},
        config=backtest.WorkStealConfig(),
        start_date="2026-01-01",
        end_date="2026-01-31",
    )

    assert seen == [(False, False), (True, True)]
    assert summary["default"]["sortino"] == 1.1
    assert summary["realistic"]["sortino"] == 0.8
