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


def test_map_file_to_tests_gracefully_handles_missing_grep(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "foo.py"
    source.write_text("print('ok')\n", encoding="utf-8")

    def _missing_grep(*args, **kwargs):
        raise FileNotFoundError("grep")

    monkeypatch.setattr(runner_impl.subprocess, "run", _missing_grep)

    assert runner.map_file_to_tests(str(source)) == []


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
    assert captured["cmd"][3] == "--basetemp"
    assert "smart-test-runner/priority-" in captured["cmd"][4]
    assert captured["cmd"][4].endswith("/basetemp")


def test_run_tests_cleans_up_nested_basetemp(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(runner_impl, "_NESTED_PYTEST_BASETEMP_ROOT", tmp_path / "smart-test-runner")

    def _fake_run(cmd, *args, **kwargs):
        captured["cmd"] = cmd
        basetemp = Path(cmd[4])
        basetemp.mkdir(parents=True, exist_ok=True)
        (basetemp / "sentinel.txt").write_text("ok", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(runner_impl.subprocess, "run", _fake_run)

    ok = runner.run_tests(["tests/test_smart_test_runner.py"], "priority", verbose=False, dry_run=False)

    assert ok is True
    run_root = Path(captured["cmd"][4]).parent
    assert not run_root.exists()


def test_nested_pytest_basetemp_root_honors_env_override(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SMART_TEST_RUNNER_BASETEMP_ROOT", str(tmp_path / "custom-root"))

    assert runner_impl.nested_pytest_basetemp_root() == tmp_path / "custom-root"


def test_run_tests_prints_rerun_command_on_failure(
    monkeypatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _fake_run(cmd, *args, **kwargs):
        del args, kwargs
        return subprocess.CompletedProcess(cmd, 1)

    monkeypatch.setattr(runner_impl.subprocess, "run", _fake_run)

    ok = runner.run_tests(["tests/test_smart_test_runner.py"], "priority", verbose=True, dry_run=False)

    output = capsys.readouterr().out
    assert ok is False
    assert "Rerun command:" in output
    assert f"{sys.executable} -m pytest tests/test_smart_test_runner.py -v --ignore=tests/experimental -x" in output


def test_run_tests_reports_failed_subprocess_start(
    monkeypatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _missing_pytest(*args, **kwargs):
        raise FileNotFoundError("pytest")

    monkeypatch.setattr(runner_impl.subprocess, "run", _missing_pytest)

    ok = runner.run_tests(["tests/test_smart_test_runner.py"], "priority", verbose=False, dry_run=False)

    output = capsys.readouterr().out
    assert ok is False
    assert "TESTS FAILED TO START" in output
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


def test_get_changed_files_gracefully_handles_missing_git(monkeypatch) -> None:
    def _missing_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(runner_impl.subprocess, "run", _missing_git)

    assert runner_impl.get_changed_files("main") == set()


def test_main_priority_only_runs_only_priority_lane(
    monkeypatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(runner_impl, "get_changed_files", lambda _base_branch: {"trade_daily_stock_prod.py"})
    monkeypatch.setattr(runner_impl, "prioritize_tests", lambda _changed: (["tests/test_a.py"], ["tests/test_b.py"]))
    monkeypatch.setattr(sys, "argv", ["smart_test_runner.py", "--priority-only"])

    calls: list[str] = []

    def _fake_run_tests(test_files, label, verbose=False, dry_run=False):
        del test_files, verbose, dry_run
        calls.append(label)
        return True

    monkeypatch.setattr(runner_impl, "run_tests", _fake_run_tests)

    with pytest.raises(SystemExit) as exc_info:
        runner_impl.main()

    output = capsys.readouterr().out
    assert exc_info.value.code == 0
    assert calls == ["priority"]
    assert "Selected lane: priority only" in output
    assert "✅ PRIORITY TESTS PASSED" in output


def test_main_remaining_only_runs_only_remaining_lane_and_fails_on_failure(
    monkeypatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(runner_impl, "get_changed_files", lambda _base_branch: set())
    monkeypatch.setattr(runner_impl, "prioritize_tests", lambda _changed: (["tests/test_a.py"], ["tests/test_b.py"]))
    monkeypatch.setattr(sys, "argv", ["smart_test_runner.py", "--remaining-only"])

    calls: list[str] = []

    def _fake_run_tests(test_files, label, verbose=False, dry_run=False):
        del test_files, verbose, dry_run
        calls.append(label)
        return False

    monkeypatch.setattr(runner_impl, "run_tests", _fake_run_tests)

    with pytest.raises(SystemExit) as exc_info:
        runner_impl.main()

    output = capsys.readouterr().out
    assert exc_info.value.code == 1
    assert calls == ["remaining"]
    assert "Selected lane: remaining only" in output
    assert "❌ REMAINING TESTS FAILED" in output
