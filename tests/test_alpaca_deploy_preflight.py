from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO / "scripts" / "alpaca_deploy_preflight.py"


def _load_module():
    spec = spec_from_file_location("alpaca_deploy_preflight", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_extract_flag_csv_values_handles_split_and_equals_forms() -> None:
    mod = _load_module()
    split = mod.extract_flag_csv_values("python bot.py --stock-symbols AAPL,MSFT,msft", "--stock-symbols")
    inline = mod.extract_flag_csv_values("python bot.py --stock-symbols=DBX,TRIP,dbx", "--stock-symbols")
    assert split == ["AAPL", "MSFT"]
    assert inline == ["DBX", "TRIP"]


def test_extract_flag_csv_values_handles_nargs_style_values() -> None:
    mod = _load_module()
    out = mod.extract_flag_csv_values(
        "python bot.py --stock-symbols YELP NET dbx --live --interval 3600",
        "--stock-symbols",
    )
    assert out == ["YELP", "NET", "DBX"]


def test_parse_git_status_porcelain_extracts_dirty_paths_and_ahead_behind() -> None:
    mod = _load_module()
    text = "\n".join(
        [
            "## main...origin/main [ahead 2, behind 4]",
            " M alpacaprod.md",
            "M  unified_orchestrator/orchestrator.py",
            "R  old.py -> new.py",
        ]
    )
    out = mod.parse_git_status_porcelain(text)
    assert out.branch_line == "main...origin/main [ahead 2, behind 4]"
    assert out.ahead == 2
    assert out.behind == 4
    assert out.dirty_paths == ["alpacaprod.md", "unified_orchestrator/orchestrator.py", "new.py"]


def test_repo_relative_dirty_paths_outside_watchlist_filters_expected_paths() -> None:
    mod = _load_module()
    dirty = ["alpacaprod.md", "unified_orchestrator/orchestrator.py", "README.md"]
    watched = ("unified_orchestrator/orchestrator.py",)
    assert mod.repo_relative_dirty_paths_outside_watchlist(dirty, watched) == ["README.md", "alpacaprod.md"]


def test_parse_supervisor_pid_extracts_running_pid() -> None:
    mod = _load_module()
    text = "unified-stock-trader             RUNNING   pid 153050, uptime 0:27:23\n"
    assert mod._parse_supervisor_pid(text) == 153050


def test_build_service_report_marks_restart_reasons_and_apply_blockers(monkeypatch) -> None:
    mod = _load_module()
    spec = mod.ServiceSpec(
        name="unit-test-svc",
        manager="supervisor",
        actual_name="unit-test-svc",
        config_path=Path("/tmp/unit-test-svc.conf"),
        watched_repo_files=("watched.py",),
        symbols_flag="--stock-symbols",
        ownership_service_name="owner",
    )
    git_status = mod.GitStatusSummary(
        branch_line="main...origin/main [ahead 2, behind 4]",
        dirty_paths=["README.md", "watched.py"],
        ahead=2,
        behind=4,
    )

    monkeypatch.setattr(mod, "get_supervisor_pid", lambda _program: 123)
    monkeypatch.setattr(
        mod,
        "read_supervisor_command",
        lambda _path: "python bot.py --stock-symbols AAPL,MSFT",
    )
    monkeypatch.setattr(
        mod,
        "_read_runtime_cmd",
        lambda _pid: "python bot.py --stock-symbols AAPL,MSFT,GOOG",
    )
    monkeypatch.setattr(
        mod,
        "_read_process_start_utc",
        lambda _pid: "2026-03-28T09:10:36+00:00",
    )
    monkeypatch.setattr(
        mod,
        "files_newer_than_process",
        lambda _pid, _paths: ["watched.py"],
    )
    monkeypatch.setattr(
        mod,
        "get_service_owned_symbols",
        lambda service_name: {
            "owner": ["AAPL", "MSFT"],
        }.get(service_name, []),
    )

    report = mod.build_service_report(spec, git_status)

    assert report.running is True
    assert report.runtime_symbols == ["AAPL", "MSFT", "GOOG"]
    assert report.configured_symbols == ["AAPL", "MSFT"]
    assert report.runtime_symbols_outside_ownership == ["GOOG"]
    assert report.configured_symbols_outside_ownership == []
    assert report.restart_reasons == [
        "watched_files_newer_than_process",
        "runtime_command_differs_from_config",
        "runtime_symbols_do_not_match_service_ownership",
    ]
    assert report.apply_blockers == [
        "dirty_repo_outside_watchlist:1",
        "branch_behind_origin:4",
    ]
    assert report.safe_to_apply is False


def test_symbols_outside_ownership_filters_non_owned_symbols() -> None:
    mod = _load_module()
    out = mod.symbols_outside_ownership(["YELP", "NET", "DBX"], ["NET", "DBX"])
    assert out == ["YELP"]


def test_extract_systemd_execstart_parses_unit_text() -> None:
    mod = _load_module()
    text = "\n".join(
        [
            "[Service]",
            "Environment=PYTHONUNBUFFERED=1",
            "ExecStart=/bin/bash -lc 'echo hi'",
        ]
    )
    assert mod._extract_systemd_execstart(text) == "/bin/bash -lc 'echo hi'"


def test_daily_rl_trader_watchlist_includes_inference_loader_files() -> None:
    mod = _load_module()
    spec = mod.SPECS["daily-rl-trader"]

    assert spec.watched_repo_files == (
        "trade_daily_stock_prod.py",
        "pufferlib_market/inference.py",
        "pufferlib_market/inference_daily.py",
        "pufferlib_market/checkpoint_loader.py",
        "systemd/daily-rl-trader.service",
        "unified_orchestrator/service_config.json",
    )
    assert spec.repo_config_path == REPO / "systemd" / "daily-rl-trader.service"


def test_build_service_report_flags_installed_config_drift_from_repo(monkeypatch) -> None:
    mod = _load_module()
    spec = mod.ServiceSpec(
        name="unit-test-svc",
        manager="systemd",
        actual_name="unit-test-svc.service",
        config_path=Path("/etc/systemd/system/unit-test-svc.service"),
        repo_config_path=REPO / "systemd" / "daily-rl-trader.service",
    )
    git_status = mod.GitStatusSummary(branch_line="main...origin/main", dirty_paths=[])

    monkeypatch.setattr(mod, "get_systemd_pid", lambda _unit: None)
    monkeypatch.setattr(mod, "read_systemd_execstart", lambda _unit: "python live.py --live")
    monkeypatch.setattr(mod, "read_repo_configured_command", lambda _spec: "python live.py --paper")
    monkeypatch.setattr(mod, "_read_runtime_cmd", lambda _pid: None)
    monkeypatch.setattr(mod, "_read_process_start_utc", lambda _pid: None)
    monkeypatch.setattr(mod, "files_newer_than_process", lambda _pid, _paths: [])

    report = mod.build_service_report(spec, git_status)

    assert report.repo_configured_cmd == "python live.py --paper"
    assert report.restart_reasons == ["installed_config_differs_from_repo"]


def test_llm_stock_trader_spec_exists_and_tracks_symbol_ownership() -> None:
    mod = _load_module()
    spec = mod.SPECS["llm-stock-trader"]

    assert spec.manager == "supervisor"
    assert spec.symbols_flag == "--stock-symbols"
    assert spec.ownership_service_name == "llm-stock-trader"
