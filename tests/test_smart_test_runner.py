from __future__ import annotations

import subprocess
import sys
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


def test_prioritize_tests_excludes_experimental_tree_from_remaining_tests() -> None:
    _, remaining = runner.prioritize_tests(set())

    assert "tests/experimental/rl/test_realistic_rl_env.py" not in remaining


def test_run_tests_uses_current_interpreter(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run(cmd, *args, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(runner_impl.subprocess, "run", _fake_run)

    ok = runner.run_tests(["tests/test_smart_test_runner.py"], "priority", verbose=False, dry_run=False)

    assert ok is True
    assert captured["cmd"][0] == sys.executable
    assert captured["cmd"][1:3] == ["-m", "pytest"]


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

    calls: list[str] = []

    def _fake_run_tests(test_files, label, verbose=False, dry_run=False):
        del test_files, verbose, dry_run
        calls.append(label)
        return label == "priority"

    monkeypatch.setattr(runner_impl, "run_tests", _fake_run_tests)

    with pytest.raises(SystemExit) as exc_info:
        runner_impl.main()

    output = capsys.readouterr().out
    assert exc_info.value.code == 0
    assert calls == ["priority", "remaining"]
    assert "⚠️  PRIORITY TESTS PASSED; REMAINING TESTS HAD FAILURES" in output
    assert "✅ ALL TESTS PASSED" not in output
