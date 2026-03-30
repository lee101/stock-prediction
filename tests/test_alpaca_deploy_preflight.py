from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO / "scripts" / "alpaca_deploy_preflight.py"


def _load_module():
    spec = spec_from_file_location("alpaca_deploy_preflight", SCRIPT_PATH)
    module = module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_extract_flag_csv_values_handles_split_and_equals_forms() -> None:
    mod = _load_module()
    split = mod.extract_flag_csv_values("python bot.py --stock-symbols AAPL,MSFT,msft", "--stock-symbols")
    inline = mod.extract_flag_csv_values("python bot.py --stock-symbols=DBX,TRIP,dbx", "--stock-symbols")
    assert split == ["AAPL", "MSFT"]
    assert inline == ["DBX", "TRIP"]


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
        peer_service_name="peer",
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
            "peer": ["GOOG", "TSLA"],
        }.get(service_name, []),
    )

    report = mod.build_service_report(spec, git_status)

    assert report.running is True
    assert report.runtime_symbols == ["AAPL", "MSFT", "GOOG"]
    assert report.configured_symbols == ["AAPL", "MSFT"]
    assert report.overlap_with_peer == ["GOOG"]
    assert report.restart_reasons == [
        "watched_files_newer_than_process",
        "runtime_command_differs_from_config",
        "runtime_symbol_overlap_with_peer_service",
    ]
    assert report.apply_blockers == [
        "dirty_repo_outside_watchlist:1",
        "branch_behind_origin:4",
    ]
    assert report.safe_to_apply is False
