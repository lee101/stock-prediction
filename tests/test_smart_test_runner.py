from __future__ import annotations

import argparse
import json
import os
import signal
import shlex
import stat
import sys
import threading
from pathlib import Path

import pytest
import scripts._smart_test_runner as runner_impl
import scripts.smart_test_runner as cli_runner
import smart_test_runner as runner


def test_is_project_test_file_excludes_hidden_cache_entries() -> None:
    assert runner._is_project_test_file(Path("tests/test_ok.py")) is True
    assert runner._is_project_test_file(Path("tests/prod/trading/test_ok.py")) is True
    assert runner._is_project_test_file(Path("tests/experimental/test_hidden_regression.py")) is False
    assert runner._is_project_test_file(Path("tests/.uvcache/archive-v0/pkg/test_vendor.py")) is False
    assert runner._is_project_test_file(Path("tests/helper.py")) is False


def test_map_file_to_tests_ignores_cached_vendor_python_files() -> None:
    assert runner.map_file_to_tests("tests/.uvcache/archive-v0/pkg/setup.py") == []
    assert runner.map_file_to_tests("tests/.uvcache/archive-v0/pkg/test_vendor.py") == []


def test_map_file_to_tests_ignores_experimental_tests() -> None:
    assert runner.map_file_to_tests("tests/experimental/rl/test_realistic_rl_env.py") == []


def test_map_file_to_tests_gracefully_handles_missing_grep(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "foo.py"
    source.write_text("print('ok')\n", encoding="utf-8")

    def _missing_grep(*args, **kwargs):
        raise FileNotFoundError("grep")

    monkeypatch.setattr(runner_impl.subprocess, "run", _missing_grep)

    assert runner.map_file_to_tests(str(source)) == []


def test_map_file_to_tests_uses_cached_import_index_for_dotted_imports(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "src" / "autoresearch_stock" / "prepare.py"
    test_file = tmp_path / "tests" / "test_prepare_runtime.py"
    source.parent.mkdir(parents=True)
    test_file.parent.mkdir(parents=True)
    source.write_text("def noop():\n    return None\n", encoding="utf-8")
    test_file.write_text("import src.autoresearch_stock.prepare as stock_prepare\n", encoding="utf-8")

    monkeypatch.setattr(runner_impl, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(runner_impl, "_TEST_IMPORT_INDEX", None)

    assert runner.map_file_to_tests(str(source)) == ["tests/test_prepare_runtime.py"]


def test_test_import_index_reuses_cached_index_when_signature_is_unchanged(monkeypatch, tmp_path: Path) -> None:
    test_file = tmp_path / "tests" / "test_prepare_runtime.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("import src.autoresearch_stock.prepare as stock_prepare\n", encoding="utf-8")

    monkeypatch.setattr(runner_impl, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(runner_impl, "_TEST_IMPORT_INDEX", None)
    monkeypatch.setattr(runner_impl, "_TEST_IMPORT_INDEX_SIGNATURE", None)

    calls: list[int] = []
    original_build = runner_impl._build_test_import_index

    def _recording_build(test_files=None):
        calls.append(1)
        return original_build(test_files)

    monkeypatch.setattr(runner_impl, "_build_test_import_index", _recording_build)

    first = runner_impl._test_import_index()
    second = runner_impl._test_import_index()

    assert calls == [1]
    assert first is second


def test_test_import_index_serializes_concurrent_rebuilds(monkeypatch, tmp_path: Path) -> None:
    test_file = tmp_path / "tests" / "test_prepare_runtime.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("import src.autoresearch_stock.prepare as stock_prepare\n", encoding="utf-8")

    monkeypatch.setattr(runner_impl, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(runner_impl, "_TEST_IMPORT_INDEX", None)
    monkeypatch.setattr(runner_impl, "_TEST_IMPORT_INDEX_SIGNATURE", None)

    entered = threading.Event()
    release = threading.Event()
    build_calls: list[int] = []
    original_build = runner_impl._build_test_import_index

    def _blocking_build(test_files=None):
        build_calls.append(1)
        entered.set()
        release.wait(timeout=5)
        return original_build(test_files)

    monkeypatch.setattr(runner_impl, "_build_test_import_index", _blocking_build)

    results: list[dict[str, set[str]]] = []

    def _worker() -> None:
        results.append(runner_impl._test_import_index())

    first = threading.Thread(target=_worker)
    second = threading.Thread(target=_worker)
    first.start()
    assert entered.wait(timeout=5)
    second.start()
    release.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert build_calls == [1]
    assert len(results) == 2
    assert results[0] is results[1]
    assert results[0]["prepare"] == {"tests/test_prepare_runtime.py"}


def test_test_import_index_uses_persistent_cache_when_signature_matches(monkeypatch, tmp_path: Path) -> None:
    test_file = tmp_path / "tests" / "test_prepare_runtime.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("import src.autoresearch_stock.prepare as stock_prepare\n", encoding="utf-8")

    cache_path = tmp_path / ".pytest_cache" / "smart-test-runner" / "import-index.json"
    cache_path.parent.mkdir(parents=True)
    signature = (("tests/test_prepare_runtime.py", test_file.stat().st_mtime_ns),)
    payload = {
        "signature": [[signature[0][0], signature[0][1]]],
        "index": {"prepare": ["tests/test_prepare_runtime.py"]},
    }
    cache_path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(runner_impl, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(runner_impl, "_DEFAULT_IMPORT_INDEX_CACHE_PATH", cache_path)
    monkeypatch.setattr(runner_impl, "_TEST_IMPORT_INDEX", None)
    monkeypatch.setattr(runner_impl, "_TEST_IMPORT_INDEX_SIGNATURE", None)

    def _unexpected_build(test_files=None):
        del test_files
        raise AssertionError("persistent cache should have been used")

    monkeypatch.setattr(runner_impl, "_build_test_import_index", _unexpected_build)

    assert runner_impl._test_import_index() == {"prepare": {"tests/test_prepare_runtime.py"}}


def test_test_import_index_rebuilds_when_persistent_cache_is_invalid(monkeypatch, tmp_path: Path) -> None:
    test_file = tmp_path / "tests" / "test_prepare_runtime.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("import src.autoresearch_stock.prepare as stock_prepare\n", encoding="utf-8")

    cache_path = tmp_path / ".pytest_cache" / "smart-test-runner" / "import-index.json"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text("{not-json", encoding="utf-8")

    monkeypatch.setattr(runner_impl, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(runner_impl, "_DEFAULT_IMPORT_INDEX_CACHE_PATH", cache_path)
    monkeypatch.setattr(runner_impl, "_TEST_IMPORT_INDEX", None)
    monkeypatch.setattr(runner_impl, "_TEST_IMPORT_INDEX_SIGNATURE", None)

    calls: list[int] = []
    original_build = runner_impl._build_test_import_index

    def _recording_build(test_files=None):
        calls.append(1)
        return original_build(test_files)

    monkeypatch.setattr(runner_impl, "_build_test_import_index", _recording_build)

    index = runner_impl._test_import_index()
    assert index["prepare"] == {"tests/test_prepare_runtime.py"}
    assert calls == [1]


def test_prioritize_tests_excludes_experimental_tree_from_remaining_tests() -> None:
    _, remaining = runner.prioritize_tests(set())

    assert "tests/experimental/rl/test_realistic_rl_env.py" not in remaining


def test_run_tests_uses_current_interpreter(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run_with_tee(cmd, *, output_log_path):
        captured["cmd"] = cmd
        captured["output_log_path"] = output_log_path
        return 0

    monkeypatch.setattr(runner_impl, "_run_subprocess_with_tee", _fake_run_with_tee)

    ok = runner.run_tests(
        ["tests/test_smart_test_runner.py"],
        runner_impl.TestLane.PRIORITY,
        verbose=False,
        dry_run=False,
    )

    assert ok is True
    assert captured["cmd"][0] == sys.executable
    assert captured["cmd"][1:3] == ["-m", "pytest"]
    assert captured["cmd"][3] == "--basetemp"
    assert "smart-test-runner/priority-" in captured["cmd"][4]
    assert captured["cmd"][4].endswith("/basetemp")


def test_run_tests_dry_run_previews_real_lane_command(
    monkeypatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(runner_impl, "_NESTED_PYTEST_BASETEMP_ROOT", tmp_path / "smart-test-runner")

    ok = runner.run_tests(
        ["tests/test_smart_test_runner.py"],
        runner_impl.TestLane.REMAINING,
        verbose=False,
        dry_run=True,
    )

    output = capsys.readouterr().out
    assert ok is True
    assert "DRY RUN: Would execute:" in output
    assert f"{sys.executable} -m pytest tests/test_smart_test_runner.py --ignore=tests/experimental --maxfail=20" in output
    assert " -v" not in output
    assert f"Nested basetemp: {tmp_path / 'smart-test-runner' / 'remaining-<random>' / 'basetemp'}" in output


def test_run_tests_cleans_up_nested_basetemp(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(runner_impl, "_NESTED_PYTEST_BASETEMP_ROOT", tmp_path / "smart-test-runner")

    def _fake_run_with_tee(cmd, *, output_log_path):
        del output_log_path
        captured["cmd"] = cmd
        basetemp = Path(cmd[4])
        basetemp.mkdir(parents=True, exist_ok=True)
        (basetemp / "sentinel.txt").write_text("ok", encoding="utf-8")
        return 0

    monkeypatch.setattr(runner_impl, "_run_subprocess_with_tee", _fake_run_with_tee)

    ok = runner.run_tests(
        ["tests/test_smart_test_runner.py"],
        runner_impl.TestLane.PRIORITY,
        verbose=False,
        dry_run=False,
    )

    assert ok is True
    run_root = Path(captured["cmd"][4]).parent
    assert not run_root.exists()
    assert not (tmp_path / "smart-test-runner").exists()


def test_nested_pytest_basetemp_root_honors_env_override(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SMART_TEST_RUNNER_BASETEMP_ROOT", str(tmp_path / "custom-root"))

    assert runner_impl.nested_pytest_basetemp_root() == tmp_path / "custom-root"


def test_run_tests_preserves_custom_basetemp_root_directory(monkeypatch, tmp_path: Path) -> None:
    custom_root = tmp_path / "custom-root"
    monkeypatch.setenv("SMART_TEST_RUNNER_BASETEMP_ROOT", str(custom_root))

    def _fake_run_with_tee(cmd, *, output_log_path):
        del output_log_path
        basetemp = Path(cmd[4])
        basetemp.mkdir(parents=True, exist_ok=True)
        return 0

    monkeypatch.setattr(runner_impl, "_run_subprocess_with_tee", _fake_run_with_tee)

    ok = runner.run_tests(
        ["tests/test_smart_test_runner.py"],
        runner_impl.TestLane.PRIORITY,
        verbose=False,
        dry_run=False,
    )

    assert ok is True
    assert custom_root.exists()


def test_default_summary_json_path_lives_under_pytest_cache() -> None:
    assert ".pytest_cache" in runner_impl._DEFAULT_SUMMARY_JSON_PATH.parts
    assert runner_impl._DEFAULT_SUMMARY_JSON_PATH.name == "last-run.json"


def test_format_log_open_command_prefers_less_and_falls_back_to_cat(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "failure.log"

    monkeypatch.setattr(runner_impl.shutil, "which", lambda name: "/usr/bin/less" if name == "less" else None)
    assert runner_impl._format_log_open_command(path) == f"less {path}"

    monkeypatch.setattr(runner_impl.shutil, "which", lambda _name: None)
    assert runner_impl._format_log_open_command(path) == f"cat {path}"


def test_run_subprocess_with_tee_writes_log_and_stdout(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    output_log_path = tmp_path / "priority.log"

    returncode = runner_impl._run_subprocess_with_tee(
        [
            sys.executable,
            "-c",
            "print('alpha'); print('beta')",
        ],
        output_log_path=output_log_path,
    )

    assert returncode == 0
    assert capsys.readouterr().out == "alpha\nbeta\n"
    assert output_log_path.read_text(encoding="utf-8") == "alpha\nbeta\n"


def test_run_tests_preserves_log_and_reports_interrupt(
    monkeypatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "priority.log"
    latest_log_path = tmp_path / "latest-priority.log"

    def _interrupting_run_with_tee(cmd, *, output_log_path):
        del cmd
        output_log_path.write_text("partial output\n", encoding="utf-8")
        raise KeyboardInterrupt

    monkeypatch.setattr(runner_impl, "_pytest_output_log_path", lambda _label: log_path)
    monkeypatch.setattr(runner_impl, "_latest_pytest_output_log_path", lambda _label: latest_log_path)
    monkeypatch.setattr(runner_impl, "_run_subprocess_with_tee", _interrupting_run_with_tee)

    result = runner_impl._run_test_lane(
        ["tests/test_smart_test_runner.py"],
        runner_impl.TestLane.PRIORITY,
        verbose=False,
        dry_run=False,
    )

    output = capsys.readouterr().out
    assert result.ok is False
    assert result.status is runner_impl.TestRunStatus.INTERRUPTED
    assert result.output_log_path == log_path
    assert "🛑 PRIORITY TESTS INTERRUPTED" in output
    assert f"Latest failure log: {latest_log_path}" in output
    assert latest_log_path.read_text(encoding="utf-8") == "partial output\n"


def test_pytest_log_has_terminal_summary_detects_complete_and_truncated_logs(tmp_path: Path) -> None:
    complete_log = tmp_path / "complete.log"
    truncated_log = tmp_path / "truncated.log"
    complete_log.write_text(
        "tests/test_ok.py ..\n"
        "=========================== short test summary info ============================\n"
        "2 passed in 0.12s\n",
        encoding="utf-8",
    )
    truncated_log.write_text(
        "tests/test_ok.py ..\n"
        "tests/test_more.py .....\n",
        encoding="utf-8",
    )

    assert runner_impl._pytest_log_has_terminal_summary(complete_log) is True
    assert runner_impl._pytest_log_has_terminal_summary(truncated_log) is False


def test_run_tests_reports_signal_terminated_pytest(
    monkeypatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "priority.log"
    latest_log_path = tmp_path / "latest-priority.log"

    def _killed_run_with_tee(cmd, *, output_log_path):
        del cmd
        output_log_path.write_text("partial output\n", encoding="utf-8")
        return -signal.SIGKILL

    monkeypatch.setattr(runner_impl, "_pytest_output_log_path", lambda _label: log_path)
    monkeypatch.setattr(runner_impl, "_latest_pytest_output_log_path", lambda _label: latest_log_path)
    monkeypatch.setattr(runner_impl, "_run_subprocess_with_tee", _killed_run_with_tee)

    result = runner_impl._run_test_lane(
        ["tests/test_smart_test_runner.py"],
        runner_impl.TestLane.PRIORITY,
        verbose=False,
        dry_run=False,
    )

    output = capsys.readouterr().out
    assert result.ok is False
    assert result.status is runner_impl.TestRunStatus.FAILED
    assert "Pytest process was terminated by signal 9 (SIGKILL)." in output


def test_run_tests_reports_truncated_failure_log_as_external_interrupt(
    monkeypatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "priority.log"
    latest_log_path = tmp_path / "latest-priority.log"

    def _truncated_run_with_tee(cmd, *, output_log_path):
        del cmd
        output_log_path.write_text(
            "tests/test_one.py ....\n"
            "tests/test_two.py ......\n",
            encoding="utf-8",
        )
        return 1

    monkeypatch.setattr(runner_impl, "_pytest_output_log_path", lambda _label: log_path)
    monkeypatch.setattr(runner_impl, "_latest_pytest_output_log_path", lambda _label: latest_log_path)
    monkeypatch.setattr(runner_impl, "_run_subprocess_with_tee", _truncated_run_with_tee)

    result = runner_impl._run_test_lane(
        ["tests/test_smart_test_runner.py"],
        runner_impl.TestLane.PRIORITY,
        verbose=False,
        dry_run=False,
    )

    output = capsys.readouterr().out
    assert result.ok is False
    assert result.status is runner_impl.TestRunStatus.FAILED
    assert "without writing a terminal summary" in output


def test_format_rerun_command_shell_quotes_arguments(monkeypatch) -> None:
    command = [
        "/tmp/python with space",
        "-m",
        "pytest",
        "tests/test file.py",
        "--summary-json",
        "/tmp/summary dir/last-run.json",
    ]
    monkeypatch.setattr(runner_impl, "_build_pytest_command", lambda *args, **kwargs: command)

    rendered = runner_impl.format_rerun_command(
        ["tests/test file.py"],
        label=runner_impl.TestLane.PRIORITY,
        verbose=False,
    )

    assert rendered == shlex.join(command)


def test_update_latest_failure_log_creates_empty_file_when_output_log_is_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    output_log_path = tmp_path / "missing.log"
    latest_log_path = tmp_path / "latest-priority.log"

    monkeypatch.setattr(runner_impl, "_latest_pytest_output_log_path", lambda _label: latest_log_path)

    resolved = runner_impl._update_latest_failure_log(runner_impl.TestLane.PRIORITY, output_log_path)

    assert resolved == latest_log_path
    assert latest_log_path.exists()
    assert latest_log_path.read_text(encoding="utf-8") == ""


def test_resolved_summary_json_path_can_be_disabled(tmp_path: Path) -> None:
    override = tmp_path / "custom-summary.json"

    assert runner_impl._resolved_summary_json_path(None, disabled=False) == runner_impl._DEFAULT_SUMMARY_JSON_PATH
    assert runner_impl._resolved_summary_json_path(str(override), disabled=False) == override
    assert runner_impl._resolved_summary_json_path(str(override), disabled=True) is None


def test_resolved_runner_config_reports_env_and_cli_sources(monkeypatch, tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    basetemp_root = tmp_path / "custom-basetemp"
    monkeypatch.setenv(runner_impl._BASE_TEMP_ROOT_ENV_VAR, str(basetemp_root))

    args = argparse.Namespace(
        base_branch="develop",
        dry_run=False,
        no_summary_json=False,
        priority_only=False,
        remaining_only=True,
        summary_json=str(summary_path),
        verbose=False,
        print_config=False,
    )

    config = runner_impl._resolved_runner_config(args)

    assert config.base_branch == "develop"
    assert config.selected_lane is runner_impl.SelectedLane.REMAINING_ONLY
    assert config.basetemp_root == str(basetemp_root)
    assert config.basetemp_root_source is runner_impl.BasetempRootSource.ENV
    assert config.summary_json_enabled is True
    assert config.summary_json_path == str(summary_path)
    assert config.summary_json_source is runner_impl.SummaryJsonSource.CLI
    assert config.import_index_cache_path == str(runner_impl._DEFAULT_IMPORT_INDEX_CACHE_PATH)


def test_build_run_summary_uses_explicit_summary_contract(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runner_impl, "_utc_now_iso", lambda: "2026-04-08T00:00:05+00:00")
    monkeypatch.setattr(runner_impl.time, "monotonic", lambda: 105.0)

    summary = runner_impl._build_run_summary(
        args=argparse.Namespace(base_branch="main", dry_run=False, verbose=False),
        changed_files={"src/foo.py"},
        exit_code=0,
        priority_duration_seconds=1.25,
        priority_output_log_path=None,
        priority_status=runner_impl.TestRunStatus.PASSED,
        priority_tests=["tests/test_a.py"],
        remaining_duration_seconds=None,
        remaining_output_log_path=None,
        remaining_status=runner_impl.TestRunStatus.NOT_RUN,
        remaining_tests=["tests/test_b.py"],
        run_started_at="2026-04-08T00:00:00+00:00",
        run_started_monotonic=100.0,
        selected_lane=runner_impl.SelectedLane.PRIORITY_ONLY,
        summary_json_path=tmp_path / "summary.json",
    )

    assert summary.base_branch == "main"
    assert summary.changed_files == ["src/foo.py"]
    assert summary.selected_lane is runner_impl.SelectedLane.PRIORITY_ONLY
    assert summary.summary_json_path == str(tmp_path / "summary.json")
    assert summary.priority_output_log_path is None
    assert summary.remaining_output_log_path is None
    assert summary.total_duration_seconds == 5.0
    assert summary.run_finished_at_utc == "2026-04-08T00:00:05+00:00"
    assert summary.priority_status is runner_impl.TestRunStatus.PASSED
    assert summary.remaining_status is runner_impl.TestRunStatus.NOT_RUN
    assert summary.priority_rerun_command is not None
    assert summary.remaining_rerun_command is not None


def test_preferred_project_python_returns_sole_repo_candidate(tmp_path: Path, monkeypatch) -> None:
    versioned = tmp_path / ".venv313" / "bin" / "python"
    versioned.parent.mkdir(parents=True)
    versioned.write_text("", encoding="utf-8")
    monkeypatch.setattr(runner_impl, "_REPO_ROOT", tmp_path)
    monkeypatch.delenv(runner_impl._PYTHON_ENV_VAR, raising=False)

    assert runner_impl._preferred_project_python() == versioned


def test_preferred_project_python_returns_none_for_ambiguous_candidates(tmp_path: Path, monkeypatch) -> None:
    plain = tmp_path / ".venv" / "bin" / "python"
    versioned = tmp_path / ".venv313" / "bin" / "python"
    plain.parent.mkdir(parents=True)
    versioned.parent.mkdir(parents=True)
    plain.write_text("", encoding="utf-8")
    versioned.write_text("", encoding="utf-8")
    monkeypatch.setattr(runner_impl, "_REPO_ROOT", tmp_path)
    monkeypatch.delenv(runner_impl._PYTHON_ENV_VAR, raising=False)

    assert runner_impl._preferred_project_python() is None


def test_preferred_project_python_honors_override(monkeypatch, tmp_path: Path) -> None:
    override = tmp_path / "custom-python"
    monkeypatch.setenv(runner_impl._PYTHON_ENV_VAR, str(override))

    assert runner_impl._preferred_project_python() == override


def test_ensure_repo_managed_interpreter_reexecs_into_repo_venv(
    monkeypatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    preferred = tmp_path / ".venv313" / "bin" / "python"
    preferred.parent.mkdir(parents=True)
    preferred.write_text("", encoding="utf-8")
    monkeypatch.setattr(runner_impl, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(sys, "executable", "/usr/bin/python3")
    monkeypatch.setattr(sys, "argv", ["smart_test_runner.py", "--priority-only"])

    captured: dict[str, object] = {}

    def _fake_execve(path, argv, env):
        captured["path"] = path
        captured["argv"] = argv
        captured["env"] = env
        raise SystemExit(0)

    monkeypatch.setattr(runner_impl.os, "execve", _fake_execve)

    with pytest.raises(SystemExit) as exc_info:
        runner_impl.ensure_repo_managed_interpreter()

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "Re-executing smart test runner with repo-managed interpreter" in output
    assert captured["path"] == str(preferred)
    assert captured["argv"] == [str(preferred), "smart_test_runner.py", "--priority-only"]
    assert captured["env"][runner_impl._REEXEC_ENV_VAR] == "1"


def test_ensure_repo_managed_interpreter_skips_repo_managed_python(
    monkeypatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    preferred = tmp_path / ".venv313" / "bin" / "python"
    preferred.parent.mkdir(parents=True)
    preferred.write_text("", encoding="utf-8")
    monkeypatch.setattr(runner_impl, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(sys, "executable", str(preferred))

    runner_impl.ensure_repo_managed_interpreter()

    assert capsys.readouterr().out == ""


def test_ensure_repo_managed_interpreter_warns_after_failed_reexec(
    monkeypatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    preferred = tmp_path / ".venv313" / "bin" / "python"
    preferred.parent.mkdir(parents=True)
    preferred.write_text("", encoding="utf-8")
    monkeypatch.setattr(runner_impl, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(sys, "executable", "/usr/bin/python3")
    monkeypatch.setenv(runner_impl._REEXEC_ENV_VAR, "1")

    runner_impl.ensure_repo_managed_interpreter()

    output = capsys.readouterr().out
    assert "Current interpreter is not a repo-managed .venv Python" in output
    assert str(preferred) in output


def test_ensure_repo_managed_interpreter_warns_when_multiple_repo_venvs_exist(
    monkeypatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    plain = tmp_path / ".venv" / "bin" / "python"
    versioned = tmp_path / ".venv313" / "bin" / "python"
    plain.parent.mkdir(parents=True)
    versioned.parent.mkdir(parents=True)
    plain.write_text("", encoding="utf-8")
    versioned.write_text("", encoding="utf-8")
    monkeypatch.setattr(runner_impl, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(sys, "executable", "/usr/bin/python3")

    runner_impl.ensure_repo_managed_interpreter()

    output = capsys.readouterr().out
    assert "refusing to guess" in output
    assert runner_impl._PYTHON_ENV_VAR in output


def test_run_tests_prints_rerun_command_on_failure(
    monkeypatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "priority.log"
    latest_log_path = tmp_path / "latest-priority.log"

    def _fake_run_with_tee(cmd, *, output_log_path):
        del cmd
        assert output_log_path == log_path
        output_log_path.write_text("failing output\n", encoding="utf-8")
        return 1

    monkeypatch.setattr(runner_impl, "_pytest_output_log_path", lambda _label: log_path)
    monkeypatch.setattr(runner_impl, "_latest_pytest_output_log_path", lambda _label: latest_log_path)
    monkeypatch.setattr(runner_impl, "_run_subprocess_with_tee", _fake_run_with_tee)

    ok = runner.run_tests(
        ["tests/test_smart_test_runner.py"],
        runner_impl.TestLane.PRIORITY,
        verbose=True,
        dry_run=False,
    )

    output = capsys.readouterr().out
    assert ok is False
    assert f"Pytest output log: {log_path}" in output
    assert f"Latest failure log: {latest_log_path}" in output
    assert f"Open log with: less {latest_log_path}" in output
    assert latest_log_path.read_text(encoding="utf-8") == "failing output\n"
    assert "Rerun command:" in output
    assert f"{sys.executable} -m pytest tests/test_smart_test_runner.py -v --ignore=tests/experimental -x" in output


def test_run_tests_reports_failed_subprocess_start(
    monkeypatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "priority.log"
    latest_log_path = tmp_path / "latest-priority.log"

    def _missing_pytest(*args, **kwargs):
        raise FileNotFoundError("pytest")

    monkeypatch.setattr(runner_impl, "_pytest_output_log_path", lambda _label: log_path)
    monkeypatch.setattr(runner_impl, "_latest_pytest_output_log_path", lambda _label: latest_log_path)
    monkeypatch.setattr(runner_impl, "_run_subprocess_with_tee", _missing_pytest)

    ok = runner.run_tests(
        ["tests/test_smart_test_runner.py"],
        runner_impl.TestLane.PRIORITY,
        verbose=False,
        dry_run=False,
    )

    output = capsys.readouterr().out
    assert ok is False
    assert "TESTS FAILED TO START" in output
    assert f"Pytest output log: {log_path}" in output
    assert f"Latest failure log: {latest_log_path}" in output
    assert f"Open log with: less {latest_log_path}" in output
    assert latest_log_path.exists()
    assert "Rerun command:" in output


def test_scripts_entrypoint_delegates_to_shared_implementation(monkeypatch) -> None:
    called = False
    repo_root = Path(__file__).resolve().parents[1]

    def _fake_main() -> None:
        nonlocal called
        called = True

    monkeypatch.setattr("scripts._smart_test_runner.main", _fake_main)
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if entry != str(repo_root)])

    cli_runner.main()

    assert called is True
    assert sys.path[0] == str(repo_root)


def test_main_reports_remaining_failures_without_claiming_all_tests_passed(
    monkeypatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(runner_impl, "get_changed_files", lambda _base_branch: set())
    monkeypatch.setattr(runner_impl, "prioritize_tests", lambda _changed: (["tests/test_a.py"], ["tests/test_b.py"]))
    monkeypatch.setattr(sys, "argv", ["smart_test_runner.py"])

    calls: list[runner_impl.TestLane] = []

    def _fake_run_test_lane(test_files, label, verbose=False, dry_run=False):
        del test_files, verbose, dry_run
        calls.append(label)
        return runner_impl.LaneRunResult(
            ok=(label is runner_impl.TestLane.PRIORITY),
            output_log_path=None,
        )

    monkeypatch.setattr(runner_impl, "_run_test_lane", _fake_run_test_lane)

    with pytest.raises(SystemExit) as exc_info:
        runner_impl.main()

    output = capsys.readouterr().out
    assert exc_info.value.code == 0
    assert calls == [runner_impl.TestLane.PRIORITY, runner_impl.TestLane.REMAINING]
    assert "⚠️  PRIORITY TESTS PASSED; REMAINING TESTS HAD FAILURES" in output
    assert "✅ ALL TESTS PASSED" not in output


def test_get_changed_files_gracefully_handles_missing_git(monkeypatch) -> None:
    def _missing_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(runner_impl.subprocess, "run", _missing_git)

    assert runner_impl.get_changed_files("main") == set()


def test_active_repo_pytest_processes_ignores_current_process_and_foreign_repos(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runner_impl, "_REPO_ROOT", tmp_path)

    class _Completed:
        stdout = (
            f"{os.getpid()} /usr/bin/python -m pytest tests/test_smart_test_runner.py\n"
            f"1234 {tmp_path}/.venv313/bin/python -m pytest -q\n"
            "5678 /other/repo/.venv/bin/python -m pytest -q\n"
            "6789 pgrep -af pytest\n"
        )

    monkeypatch.setattr(runner_impl.subprocess, "run", lambda *args, **kwargs: _Completed())

    assert runner_impl._active_repo_pytest_processes() == [f"1234 {tmp_path}/.venv313/bin/python -m pytest -q"]


def test_assert_no_competing_repo_pytest_processes_raises_with_preview(monkeypatch) -> None:
    monkeypatch.setattr(
        runner_impl,
        "_active_repo_pytest_processes",
        lambda: [f"{1000 + idx} /repo/.venv/bin/python -m pytest -q" for idx in range(7)],
    )

    with pytest.raises(RuntimeError, match="Detected competing repo-local pytest processes"):
        runner_impl._assert_no_competing_repo_pytest_processes()


def test_should_check_for_competing_repo_pytest_processes_skips_pytest_harness(monkeypatch) -> None:
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/test_smart_test_runner.py::test")
    assert runner_impl._should_check_for_competing_repo_pytest_processes() is False

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    assert runner_impl._should_check_for_competing_repo_pytest_processes() is True


def test_main_priority_only_runs_only_priority_lane(
    monkeypatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(runner_impl, "get_changed_files", lambda _base_branch: {"trade_daily_stock_prod.py"})
    monkeypatch.setattr(runner_impl, "prioritize_tests", lambda _changed: (["tests/test_a.py"], ["tests/test_b.py"]))
    monkeypatch.setattr(sys, "argv", ["smart_test_runner.py", "--priority-only"])

    calls: list[runner_impl.TestLane] = []

    def _fake_run_test_lane(test_files, label, verbose=False, dry_run=False):
        del test_files, verbose, dry_run
        calls.append(label)
        return runner_impl.LaneRunResult(ok=True, output_log_path=None)

    monkeypatch.setattr(runner_impl, "_run_test_lane", _fake_run_test_lane)

    with pytest.raises(SystemExit) as exc_info:
        runner_impl.main()

    output = capsys.readouterr().out
    assert exc_info.value.code == 0
    assert calls == [runner_impl.TestLane.PRIORITY]
    assert "Selected lane: priority only" in output
    assert "✅ PRIORITY TESTS PASSED" in output


def test_main_fails_cleanly_when_competing_repo_pytest_processes_are_active(
    monkeypatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "summary.json"
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(runner_impl, "ensure_repo_managed_interpreter", lambda: None)
    monkeypatch.setattr(runner_impl, "get_changed_files", lambda _base_branch: {"src/foo.py"})
    monkeypatch.setattr(runner_impl, "prioritize_tests", lambda _changed: (["tests/test_a.py"], ["tests/test_b.py"]))
    monkeypatch.setattr(
        runner_impl,
        "_assert_no_competing_repo_pytest_processes",
        lambda: (_ for _ in ()).throw(
            RuntimeError("Detected competing repo-local pytest processes.\n  - 1234 /repo/.venv/bin/python -m pytest -q")
        ),
    )
    monkeypatch.setattr(sys, "argv", ["smart_test_runner.py", "--summary-json", str(summary_path)])

    def _unexpected_run_test_lane(*args, **kwargs):
        raise AssertionError("runner should fail before starting any lanes")

    monkeypatch.setattr(runner_impl, "_run_test_lane", _unexpected_run_test_lane)

    with pytest.raises(SystemExit) as exc_info:
        runner_impl.main()

    output = capsys.readouterr().out
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert exc_info.value.code == 1
    assert "Detected competing repo-local pytest processes" in output
    assert payload["priority_status"] == "not_run"
    assert payload["remaining_status"] == "not_run"
    assert payload["exit_code"] == 1


def test_main_remaining_only_runs_only_remaining_lane_and_fails_on_failure(
    monkeypatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(runner_impl, "get_changed_files", lambda _base_branch: set())
    monkeypatch.setattr(runner_impl, "prioritize_tests", lambda _changed: (["tests/test_a.py"], ["tests/test_b.py"]))
    monkeypatch.setattr(sys, "argv", ["smart_test_runner.py", "--remaining-only"])

    calls: list[runner_impl.TestLane] = []

    def _fake_run_test_lane(test_files, label, verbose=False, dry_run=False):
        del test_files, verbose, dry_run
        calls.append(label)
        return runner_impl.LaneRunResult(ok=False, output_log_path=None)

    monkeypatch.setattr(runner_impl, "_run_test_lane", _fake_run_test_lane)

    with pytest.raises(SystemExit) as exc_info:
        runner_impl.main()

    output = capsys.readouterr().out
    assert exc_info.value.code == 1
    assert calls == [runner_impl.TestLane.REMAINING]
    assert "Selected lane: remaining only" in output
    assert "❌ REMAINING TESTS FAILED" in output


def test_main_writes_summary_json_for_priority_only(
    monkeypatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "summary.json"
    monkeypatch.setattr(runner_impl, "ensure_repo_managed_interpreter", lambda: None)
    monkeypatch.setattr(runner_impl, "get_changed_files", lambda _base_branch: {"src/foo.py"})
    monkeypatch.setattr(runner_impl, "prioritize_tests", lambda _changed: (["tests/test_a.py"], ["tests/test_b.py"]))
    monkeypatch.setattr(sys, "argv", ["smart_test_runner.py", "--priority-only", "--summary-json", str(summary_path)])

    def _fake_run_test_lane(test_files, label, verbose=False, dry_run=False):
        del test_files, label, verbose, dry_run
        return runner_impl.LaneRunResult(ok=True, output_log_path=None)

    monkeypatch.setattr(runner_impl, "_run_test_lane", _fake_run_test_lane)

    with pytest.raises(SystemExit) as exc_info:
        runner_impl.main()

    output = capsys.readouterr().out
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert exc_info.value.code == 0
    assert f"Summary JSON: {summary_path}" in output
    assert payload["selected_lane"] == "priority_only"
    assert payload["changed_files"] == ["src/foo.py"]
    assert payload["cwd"] == str(Path.cwd())
    assert payload["priority_status"] == "passed"
    assert payload["priority_duration_seconds"] is not None
    assert payload["priority_output_log_path"] is None
    assert payload["python_executable"] == sys.executable
    assert payload["remaining_status"] == "not_run"
    assert payload["remaining_duration_seconds"] is None
    assert payload["remaining_output_log_path"] is None
    assert payload["run_finished_at_utc"]
    assert payload["run_started_at_utc"]
    assert payload["exit_code"] == 0
    assert payload["summary_json_path"] == str(summary_path)
    assert payload["total_duration_seconds"] >= 0
    assert payload["priority_rerun_command"].endswith("tests/test_a.py --ignore=tests/experimental -x")


def test_main_print_config_exits_without_running_tests(
    monkeypatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    basetemp_root = tmp_path / "custom-basetemp"
    summary_path = tmp_path / "summary.json"
    monkeypatch.setenv(runner_impl._BASE_TEMP_ROOT_ENV_VAR, str(basetemp_root))
    monkeypatch.setattr(runner_impl, "ensure_repo_managed_interpreter", lambda: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["smart_test_runner.py", "--remaining-only", "--summary-json", str(summary_path), "--print-config"],
    )

    def _unexpected_run_test_lane(*args, **kwargs):
        raise AssertionError("print-config should exit before running tests")

    monkeypatch.setattr(runner_impl, "_run_test_lane", _unexpected_run_test_lane)

    with pytest.raises(SystemExit) as exc_info:
        runner_impl.main()

    output = capsys.readouterr().out
    assert exc_info.value.code == 0
    assert "Smart Test Runner Config" in output
    assert f"Basetemp root:       {basetemp_root}" in output
    assert "Basetemp source:     env" in output
    assert f"Summary JSON:        {summary_path}" in output
    assert "Summary source:      cli" in output
    assert "Selected lane:       remaining_only" in output


def test_main_sets_private_permissions_on_summary_json(
    monkeypatch,
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "summary.json"

    monkeypatch.setattr(runner_impl, "ensure_repo_managed_interpreter", lambda: None)
    monkeypatch.setattr(runner_impl, "get_changed_files", lambda _base_branch: {"src/foo.py"})
    monkeypatch.setattr(runner_impl, "prioritize_tests", lambda _changed: (["tests/test_a.py"], ["tests/test_b.py"]))
    monkeypatch.setattr(sys, "argv", ["smart_test_runner.py", "--priority-only", "--summary-json", str(summary_path)])
    monkeypatch.setattr(
        runner_impl,
        "_run_test_lane",
        lambda *args, **kwargs: runner_impl.LaneRunResult(ok=True, output_log_path=None),
    )

    with pytest.raises(SystemExit) as exc_info:
        runner_impl.main()

    assert exc_info.value.code == 0
    assert stat.S_IMODE(summary_path.stat().st_mode) == 0o600


def test_run_subprocess_with_tee_rejects_symlinked_repo_managed_log_parent(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(runner_impl, "_REPO_ROOT", tmp_path)
    log_root = tmp_path / ".pytest_cache" / "smart-test-runner" / "logs"
    redirect_root = tmp_path / "redirected-logs"
    redirect_root.mkdir(parents=True)
    log_root.parent.mkdir(parents=True, exist_ok=True)
    log_root.symlink_to(redirect_root, target_is_directory=True)

    with pytest.raises(RuntimeError, match="symlinked path"):
        runner_impl._run_subprocess_with_tee(
            [sys.executable, "-c", "print('ok')"],
            output_log_path=log_root / "priority.log",
        )

    assert not any(redirect_root.iterdir())


def test_write_summary_json_rejects_symlinked_repo_managed_parent(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(runner_impl, "_REPO_ROOT", tmp_path)
    summary_root = tmp_path / ".pytest_cache" / "smart-test-runner"
    redirect_root = tmp_path / "redirected-summary"
    redirect_root.mkdir(parents=True)
    summary_root.parent.mkdir(parents=True, exist_ok=True)
    summary_root.symlink_to(redirect_root, target_is_directory=True)
    monkeypatch.setattr(runner_impl, "_DEFAULT_SUMMARY_JSON_PATH", summary_root / "last-run.json")

    summary = runner_impl._build_run_summary(
        args=argparse.Namespace(base_branch="main", dry_run=False, verbose=False),
        changed_files={"src/foo.py"},
        exit_code=0,
        priority_duration_seconds=1.0,
        priority_output_log_path=None,
        priority_status=runner_impl.TestRunStatus.PASSED,
        priority_tests=["tests/test_a.py"],
        remaining_duration_seconds=None,
        remaining_output_log_path=None,
        remaining_status=runner_impl.TestRunStatus.NOT_RUN,
        remaining_tests=[],
        run_started_at="2026-04-08T00:00:00+00:00",
        run_started_monotonic=0.0,
        selected_lane=runner_impl.SelectedLane.PRIORITY_ONLY,
        summary_json_path=runner_impl._DEFAULT_SUMMARY_JSON_PATH,
    )

    with pytest.raises(RuntimeError, match="Failed to write smart test runner summary JSON"):
        runner_impl._write_summary_json(None, summary)

    assert not any(redirect_root.iterdir())


def test_main_writes_summary_json_for_fail_fast_priority_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "summary.json"
    output_log_path = tmp_path / "priority.log"
    monkeypatch.setattr(runner_impl, "ensure_repo_managed_interpreter", lambda: None)
    monkeypatch.setattr(runner_impl, "get_changed_files", lambda _base_branch: {"src/foo.py"})
    monkeypatch.setattr(runner_impl, "prioritize_tests", lambda _changed: (["tests/test_a.py"], ["tests/test_b.py"]))
    monkeypatch.setattr(sys, "argv", ["smart_test_runner.py", "--summary-json", str(summary_path)])

    def _fake_run_test_lane(test_files, label, verbose=False, dry_run=False):
        del test_files, verbose, dry_run
        return runner_impl.LaneRunResult(
            ok=(label is not runner_impl.TestLane.PRIORITY),
            output_log_path=output_log_path if label is runner_impl.TestLane.PRIORITY else None,
        )

    monkeypatch.setattr(runner_impl, "_run_test_lane", _fake_run_test_lane)

    with pytest.raises(SystemExit) as exc_info:
        runner_impl.main()

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert exc_info.value.code == 1
    assert payload["selected_lane"] == "priority_and_remaining"
    assert payload["priority_status"] == "failed"
    assert payload["priority_output_log_path"] == str(output_log_path)
    assert payload["remaining_status"] == "not_run"
    assert payload["remaining_output_log_path"] is None
    assert payload["exit_code"] == 1


def test_main_writes_summary_json_for_priority_interrupt(
    monkeypatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "summary.json"
    output_log_path = tmp_path / "priority.log"
    monkeypatch.setattr(runner_impl, "ensure_repo_managed_interpreter", lambda: None)
    monkeypatch.setattr(runner_impl, "get_changed_files", lambda _base_branch: {"src/foo.py"})
    monkeypatch.setattr(runner_impl, "prioritize_tests", lambda _changed: (["tests/test_a.py"], ["tests/test_b.py"]))
    monkeypatch.setattr(sys, "argv", ["smart_test_runner.py", "--summary-json", str(summary_path)])

    def _fake_run_test_lane(test_files, label, verbose=False, dry_run=False):
        del test_files, verbose, dry_run
        return runner_impl.LaneRunResult(
            ok=False,
            output_log_path=output_log_path if label is runner_impl.TestLane.PRIORITY else None,
            status=runner_impl.TestRunStatus.INTERRUPTED if label is runner_impl.TestLane.PRIORITY else None,
        )

    monkeypatch.setattr(runner_impl, "_run_test_lane", _fake_run_test_lane)

    with pytest.raises(SystemExit) as exc_info:
        runner_impl.main()

    output = capsys.readouterr().out
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert exc_info.value.code == 130
    assert "PRIORITY TESTS INTERRUPTED" in output
    assert payload["priority_status"] == "interrupted"
    assert payload["priority_output_log_path"] == str(output_log_path)
    assert payload["remaining_status"] == "not_run"
    assert payload["exit_code"] == 130


def test_main_writes_summary_json_for_remaining_only_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "summary.json"
    output_log_path = tmp_path / "remaining.log"
    monkeypatch.setattr(runner_impl, "ensure_repo_managed_interpreter", lambda: None)
    monkeypatch.setattr(runner_impl, "get_changed_files", lambda _base_branch: {"src/foo.py"})
    monkeypatch.setattr(runner_impl, "prioritize_tests", lambda _changed: (["tests/test_a.py"], ["tests/test_b.py"]))
    monkeypatch.setattr(
        sys,
        "argv",
        ["smart_test_runner.py", "--remaining-only", "--summary-json", str(summary_path)],
    )

    def _fake_run_test_lane(test_files, label, verbose=False, dry_run=False):
        del test_files, verbose, dry_run
        return runner_impl.LaneRunResult(
            ok=False,
            output_log_path=output_log_path if label is runner_impl.TestLane.REMAINING else None,
        )

    monkeypatch.setattr(runner_impl, "_run_test_lane", _fake_run_test_lane)

    with pytest.raises(SystemExit) as exc_info:
        runner_impl.main()

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert exc_info.value.code == 1
    assert payload["selected_lane"] == "remaining_only"
    assert payload["priority_status"] == "not_run"
    assert payload["priority_output_log_path"] is None
    assert payload["remaining_status"] == "failed"
    assert payload["remaining_output_log_path"] == str(output_log_path)
    assert payload["exit_code"] == 1


def test_main_fails_cleanly_when_summary_json_cannot_be_written(
    monkeypatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "summary.json"
    monkeypatch.setattr(runner_impl, "ensure_repo_managed_interpreter", lambda: None)
    monkeypatch.setattr(runner_impl, "get_changed_files", lambda _base_branch: {"src/foo.py"})
    monkeypatch.setattr(runner_impl, "prioritize_tests", lambda _changed: (["tests/test_a.py"], ["tests/test_b.py"]))
    monkeypatch.setattr(sys, "argv", ["smart_test_runner.py", "--priority-only", "--summary-json", str(summary_path)])
    monkeypatch.setattr(
        runner_impl,
        "_run_test_lane",
        lambda *args, **kwargs: runner_impl.LaneRunResult(ok=True, output_log_path=None),
    )

    def _fail_write(_target, _content):
        raise OSError("disk full")

    monkeypatch.setattr(runner_impl, "_write_private_text_artifact", _fail_write)

    with pytest.raises(SystemExit) as exc_info:
        runner_impl.main()

    output = capsys.readouterr().out
    assert exc_info.value.code == 1
    assert "Failed to write smart test runner summary JSON" in output
    assert "disk full" in output


def test_main_writes_summary_json_for_dry_run(
    monkeypatch,
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "summary.json"
    monkeypatch.setattr(runner_impl, "ensure_repo_managed_interpreter", lambda: None)
    monkeypatch.setattr(runner_impl, "get_changed_files", lambda _base_branch: {"src/foo.py"})
    monkeypatch.setattr(runner_impl, "prioritize_tests", lambda _changed: (["tests/test_a.py"], ["tests/test_b.py"]))
    monkeypatch.setattr(sys, "argv", ["smart_test_runner.py", "--dry-run", "--summary-json", str(summary_path)])

    calls: list[tuple[runner_impl.TestLane, bool]] = []

    def _fake_run_test_lane(test_files, label, verbose=False, dry_run=False):
        del test_files, verbose
        calls.append((label, dry_run))
        return runner_impl.LaneRunResult(ok=True, output_log_path=None)

    monkeypatch.setattr(runner_impl, "_run_test_lane", _fake_run_test_lane)

    with pytest.raises(SystemExit) as exc_info:
        runner_impl.main()

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert exc_info.value.code == 0
    assert calls == [
        (runner_impl.TestLane.PRIORITY, True),
        (runner_impl.TestLane.REMAINING, True),
    ]
    assert payload["dry_run"] is True
    assert payload["priority_status"] == "dry_run"
    assert payload["priority_duration_seconds"] is not None
    assert payload["priority_output_log_path"] is None
    assert payload["remaining_status"] == "dry_run"
    assert payload["remaining_duration_seconds"] is not None
    assert payload["remaining_output_log_path"] is None
    assert payload["exit_code"] == 0
    assert payload["total_duration_seconds"] >= 0


def test_main_writes_default_summary_json_path(
    monkeypatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / ".smart-test-runner" / "last-run.json"
    monkeypatch.setattr(runner_impl, "_NESTED_PYTEST_BASETEMP_ROOT", tmp_path / ".smart-test-runner")
    monkeypatch.setattr(runner_impl, "_DEFAULT_SUMMARY_JSON_PATH", summary_path)
    monkeypatch.setattr(runner_impl, "ensure_repo_managed_interpreter", lambda: None)
    monkeypatch.setattr(runner_impl, "get_changed_files", lambda _base_branch: {"src/foo.py"})
    monkeypatch.setattr(runner_impl, "prioritize_tests", lambda _changed: (["tests/test_a.py"], ["tests/test_b.py"]))
    monkeypatch.setattr(sys, "argv", ["smart_test_runner.py", "--priority-only"])
    monkeypatch.setattr(
        runner_impl,
        "_run_test_lane",
        lambda *args, **kwargs: runner_impl.LaneRunResult(ok=True, output_log_path=None),
    )

    with pytest.raises(SystemExit) as exc_info:
        runner_impl.main()

    output = capsys.readouterr().out
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert exc_info.value.code == 0
    assert f"Summary JSON: {summary_path}" in output
    assert payload["cwd"] == str(Path.cwd())
    assert payload["python_executable"] == sys.executable
    assert payload["summary_json_path"] == str(summary_path)
    assert payload["priority_status"] == "passed"
    assert payload["priority_output_log_path"] is None


def test_main_can_disable_default_summary_json(
    monkeypatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / ".smart-test-runner" / "last-run.json"
    monkeypatch.setattr(runner_impl, "_NESTED_PYTEST_BASETEMP_ROOT", tmp_path / ".smart-test-runner")
    monkeypatch.setattr(runner_impl, "_DEFAULT_SUMMARY_JSON_PATH", summary_path)
    monkeypatch.setattr(runner_impl, "ensure_repo_managed_interpreter", lambda: None)
    monkeypatch.setattr(runner_impl, "get_changed_files", lambda _base_branch: {"src/foo.py"})
    monkeypatch.setattr(runner_impl, "prioritize_tests", lambda _changed: (["tests/test_a.py"], ["tests/test_b.py"]))
    monkeypatch.setattr(sys, "argv", ["smart_test_runner.py", "--priority-only", "--no-summary-json"])
    monkeypatch.setattr(
        runner_impl,
        "_run_test_lane",
        lambda *args, **kwargs: runner_impl.LaneRunResult(ok=True, output_log_path=None),
    )

    with pytest.raises(SystemExit) as exc_info:
        runner_impl.main()

    output = capsys.readouterr().out
    assert exc_info.value.code == 0
    assert "Summary JSON:" not in output
    assert not summary_path.exists()
