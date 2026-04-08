#!/usr/bin/env python3
"""
Shared implementation for the smart test runner.

Keep the actual logic here so both the root-level compatibility module and the
scripts/ CLI entry point use the same code path.
"""

import argparse
import json
import os
import re
import signal
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from dataclasses import asdict, dataclass
from enum import StrEnum
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock


_IGNORED_TEST_PARTS = {".uvcache", ".pytest_cache", "__pycache__"}
_IGNORED_TEST_ROOTS = {"tests/experimental"}
_REPO_ROOT = Path(__file__).resolve().parents[1]
_NESTED_PYTEST_BASETEMP_ROOT = _REPO_ROOT / ".smart-test-runner"
_DEFAULT_SUMMARY_JSON_PATH = _REPO_ROOT / ".pytest_cache" / "smart-test-runner" / "last-run.json"
_DEFAULT_PYTEST_OUTPUT_LOG_DIR = _REPO_ROOT / ".pytest_cache" / "smart-test-runner" / "logs"
_DEFAULT_IMPORT_INDEX_CACHE_PATH = _REPO_ROOT / ".pytest_cache" / "smart-test-runner" / "import-index.json"
_CHANGED_FILE_PREVIEW_LIMIT = 10
_SUBPROCESS_INTERRUPT_WAIT_TIMEOUT_SECONDS = 5
_COMPETING_PYTEST_PROCESS_PREVIEW_LIMIT = 5
_BASE_TEMP_ROOT_ENV_VAR = "SMART_TEST_RUNNER_BASETEMP_ROOT"
_PYTHON_ENV_VAR = "SMART_TEST_RUNNER_PYTHON"
_REEXEC_ENV_VAR = "SMART_TEST_RUNNER_REEXECUTED"
_TEST_IMPORT_INDEX: dict[str, set[str]] | None = None
_TEST_IMPORT_INDEX_SIGNATURE: tuple[tuple[str, int], ...] | None = None
_TEST_IMPORT_INDEX_LOCK = RLock()


class TestLane(StrEnum):
    PRIORITY = "priority"
    REMAINING = "remaining"


class TestRunStatus(StrEnum):
    NOT_RUN = "not_run"
    PASSED = "passed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    DRY_RUN = "dry_run"


class SelectedLane(StrEnum):
    PRIORITY_ONLY = "priority_only"
    REMAINING_ONLY = "remaining_only"
    PRIORITY_AND_REMAINING = "priority_and_remaining"


class BasetempRootSource(StrEnum):
    DEFAULT = "default"
    ENV = "env"


class SummaryJsonSource(StrEnum):
    DEFAULT = "default"
    CLI = "cli"
    DISABLED = "disabled"


@dataclass(frozen=True)
class TestRunSummary:
    base_branch: str
    changed_files: list[str]
    cwd: str
    dry_run: bool
    exit_code: int
    python_executable: str
    remaining_duration_seconds: float | None
    remaining_output_log_path: str | None
    summary_json_path: str | None
    priority_duration_seconds: float | None
    priority_output_log_path: str | None
    priority_rerun_command: str | None
    priority_status: TestRunStatus
    priority_tests: list[str]
    run_finished_at_utc: str
    run_started_at_utc: str
    remaining_rerun_command: str | None
    remaining_status: TestRunStatus
    remaining_tests: list[str]
    selected_lane: SelectedLane
    total_duration_seconds: float


@dataclass(frozen=True)
class LaneRunResult:
    ok: bool
    output_log_path: Path | None
    status: TestRunStatus | None = None


@dataclass(frozen=True)
class ResolvedRunnerConfig:
    base_branch: str
    basetemp_root: str
    basetemp_root_source: BasetempRootSource
    import_index_cache_path: str
    python_executable: str
    repo_root: str
    selected_lane: SelectedLane
    summary_json_enabled: bool
    summary_json_path: str | None
    summary_json_source: SummaryJsonSource


def nested_pytest_basetemp_root() -> Path:
    override = os.getenv(_BASE_TEMP_ROOT_ENV_VAR)
    if override:
        return Path(override).expanduser()
    return _NESTED_PYTEST_BASETEMP_ROOT


def _repo_managed_python_candidates() -> list[Path]:
    return sorted(_REPO_ROOT.glob(".venv*/bin/python"))


def _preferred_project_python() -> Path | None:
    override = os.getenv(_PYTHON_ENV_VAR)
    if override:
        return Path(override).expanduser()

    candidates = _repo_managed_python_candidates()
    if len(candidates) == 1:
        return candidates[0]
    return None


def _using_repo_managed_python() -> bool:
    executable = Path(sys.executable).absolute()
    try:
        relative = executable.relative_to(_REPO_ROOT)
    except ValueError:
        return False
    return any(part.startswith(".venv") for part in relative.parts)


def ensure_repo_managed_interpreter() -> None:
    preferred = _preferred_project_python()
    if preferred is None or _using_repo_managed_python():
        if preferred is None and not _using_repo_managed_python():
            maybe_warn_about_interpreter()
        return

    if os.getenv(_REEXEC_ENV_VAR) == "1":
        maybe_warn_about_interpreter()
        return

    print("\n↪ Re-executing smart test runner with repo-managed interpreter.")
    print(f"  Current interpreter: {sys.executable}")
    print(f"  Repo interpreter: {preferred}")
    env = os.environ.copy()
    env[_REEXEC_ENV_VAR] = "1"
    os.execve(str(preferred), [str(preferred), *sys.argv], env)


def maybe_warn_about_interpreter() -> None:
    preferred = _preferred_project_python()
    if _using_repo_managed_python():
        return

    print("\n⚠️  Current interpreter is not a repo-managed .venv Python.")
    print(f"  Running with: {sys.executable}")
    if preferred is not None:
        print(f"  Suggested repo interpreter: {preferred}")
        print("  Example:")
        print(f"    source {preferred.parent.parent}/bin/activate && python scripts/smart_test_runner.py")
        return

    candidates = [path.parents[1].name for path in _repo_managed_python_candidates()]
    if candidates:
        print("  Multiple repo-managed interpreters found; refusing to guess.")
        print(f"  Candidates: {', '.join(candidates)}")
        print(f"  Set {_PYTHON_ENV_VAR}=/abs/path/to/python or activate the intended .venv first.")


def _repo_relative_path(path: Path) -> Path:
    try:
        return path.resolve().relative_to(_REPO_ROOT)
    except ValueError:
        return path


def _is_project_test_file(path: Path) -> bool:
    """Return True for repo-owned pytest modules, excluding cached/vendor files."""
    relative_path = _repo_relative_path(path)
    if relative_path.suffix != ".py" or not relative_path.name.startswith("test_"):
        return False
    normalized = relative_path.as_posix()
    if any(normalized == root or normalized.startswith(f"{root}/") for root in _IGNORED_TEST_ROOTS):
        return False
    return not any(part in _IGNORED_TEST_PARTS or part.startswith(".") for part in relative_path.parts)


def _iter_project_test_files() -> list[Path]:
    test_files: list[Path] = []
    root = _REPO_ROOT / "tests"
    if not root.exists():
        return test_files
    for test_file in root.rglob("test_*.py"):
        if _is_project_test_file(test_file):
            test_files.append(test_file)
    return test_files


def _record_imported_module(index: dict[str, set[str]], module_name: str, test_file: Path) -> None:
    normalized_test = _repo_relative_path(test_file).as_posix()
    for part in module_name.split("."):
        if part:
            index.setdefault(part, set()).add(normalized_test)


def _test_import_index_signature(test_files: list[Path]) -> tuple[tuple[str, int], ...]:
    signature: list[tuple[str, int]] = []
    for test_file in test_files:
        try:
            stat = test_file.stat()
        except OSError:
            continue
        signature.append((_repo_relative_path(test_file).as_posix(), stat.st_mtime_ns))
    return tuple(signature)


def _build_test_import_index(test_files: list[Path] | None = None) -> dict[str, set[str]]:
    index: dict[str, set[str]] = {}
    from_import_re = re.compile(r"^\s*from\s+([A-Za-z_][\w\.]*)\s+import\b")
    import_re = re.compile(r"^\s*import\s+(.+)$")

    candidate_files = _iter_project_test_files() if test_files is None else test_files
    for test_file in candidate_files:
        try:
            lines = test_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue

        for line in lines:
            from_match = from_import_re.match(line)
            if from_match is not None:
                _record_imported_module(index, from_match.group(1), test_file)
                continue

            import_match = import_re.match(line)
            if import_match is None:
                continue

            for raw_name in import_match.group(1).split(","):
                module_name = raw_name.strip().split(" as ", 1)[0].strip()
                if module_name:
                    _record_imported_module(index, module_name, test_file)

    return index


def _load_cached_test_import_index(
    signature: tuple[tuple[str, int], ...],
    *,
    cache_path: Path | None = None,
) -> dict[str, set[str]] | None:
    resolved_cache_path = _DEFAULT_IMPORT_INDEX_CACHE_PATH if cache_path is None else cache_path
    try:
        _ensure_safe_repo_managed_artifact_path(resolved_cache_path)
        payload = json.loads(resolved_cache_path.read_text(encoding="utf-8"))
    except (OSError, RuntimeError, json.JSONDecodeError):
        return None

    raw_signature = payload.get("signature")
    raw_index = payload.get("index")
    if not isinstance(raw_signature, list) or not isinstance(raw_index, dict):
        return None

    normalized_signature: list[tuple[str, int]] = []
    for item in raw_signature:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not isinstance(item[1], int)
        ):
            return None
        normalized_signature.append((item[0], item[1]))
    if tuple(normalized_signature) != signature:
        return None

    normalized_index: dict[str, set[str]] = {}
    for key, value in raw_index.items():
        if not isinstance(key, str) or not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            return None
        normalized_index[key] = set(value)
    return normalized_index


def _write_cached_test_import_index(
    signature: tuple[tuple[str, int], ...],
    index: dict[str, set[str]],
    *,
    cache_path: Path | None = None,
) -> None:
    resolved_cache_path = _DEFAULT_IMPORT_INDEX_CACHE_PATH if cache_path is None else cache_path
    payload = {
        "signature": [[path, mtime_ns] for path, mtime_ns in signature],
        "index": {key: sorted(value) for key, value in sorted(index.items())},
    }
    _write_private_text_artifact(
        resolved_cache_path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )


def _test_import_index() -> dict[str, set[str]]:
    global _TEST_IMPORT_INDEX
    global _TEST_IMPORT_INDEX_SIGNATURE
    test_files = _iter_project_test_files()
    signature = _test_import_index_signature(test_files)
    with _TEST_IMPORT_INDEX_LOCK:
        if _TEST_IMPORT_INDEX is None or _TEST_IMPORT_INDEX_SIGNATURE != signature:
            cached = _load_cached_test_import_index(signature)
            if cached is not None:
                _TEST_IMPORT_INDEX = cached
            else:
                _TEST_IMPORT_INDEX = _build_test_import_index(test_files)
                with suppress(OSError, RuntimeError):
                    _write_cached_test_import_index(signature, _TEST_IMPORT_INDEX)
            _TEST_IMPORT_INDEX_SIGNATURE = signature
        return _TEST_IMPORT_INDEX


def get_changed_files(base_branch: str = "main") -> set[str]:
    """Get list of changed files compared to base branch or last commit."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", f"{base_branch}...HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        files = result.stdout.strip().split("\n")
        if files and files[0]:
            return {f for f in files if f}
    except (subprocess.CalledProcessError, OSError):
        pass

    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        staged_result = subprocess.run(
            ["git", "diff", "--name-only", "--cached"],
            capture_output=True,
            text=True,
            check=True,
        )
        files = set(result.stdout.strip().split("\n") + staged_result.stdout.strip().split("\n"))
        return {f for f in files if f}
    except (subprocess.CalledProcessError, OSError):
        return set()


def _active_repo_pytest_processes() -> list[str]:
    try:
        result = subprocess.run(
            ["pgrep", "-af", "pytest"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return []

    active: list[str] = []
    current_pid = str(os.getpid())
    repo_root = str(_REPO_ROOT)
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        pid, _sep, command = line.partition(" ")
        if not pid or pid == current_pid:
            continue
        normalized_command = command.strip()
        if not normalized_command:
            continue
        if "pgrep -af pytest" in normalized_command:
            continue
        if repo_root not in normalized_command:
            continue
        active.append(line)
    return active


def _assert_no_competing_repo_pytest_processes() -> None:
    active = _active_repo_pytest_processes()
    if not active:
        return

    preview = active[:_COMPETING_PYTEST_PROCESS_PREVIEW_LIMIT]
    details = "\n".join(f"  - {entry}" for entry in preview)
    if len(active) > _COMPETING_PYTEST_PROCESS_PREVIEW_LIMIT:
        details += f"\n  - ... and {len(active) - _COMPETING_PYTEST_PROCESS_PREVIEW_LIMIT} more"
    raise RuntimeError(
        "Detected competing repo-local pytest processes. "
        "Stop other pytest runs before using smart_test_runner.py.\n"
        f"{details}"
    )


def _should_check_for_competing_repo_pytest_processes() -> bool:
    return "PYTEST_CURRENT_TEST" not in os.environ


def map_file_to_tests(file_path: str) -> list[str]:
    """Map a source file to its corresponding test files."""
    tests: list[str] = []
    path = Path(file_path)

    if "test" in path.parts or file_path.startswith("tests/"):
        if not path.exists() or not _is_project_test_file(path):
            return []
        return [_repo_relative_path(path).as_posix()]

    if path.suffix != ".py":
        return []

    stem = path.stem
    test_patterns = [
        f"tests/test_{stem}.py",
        f"tests/prod/test_{stem}.py",
        f"tests/prod/**/test_{stem}.py",
    ]

    special_mappings = {
        "loss_utils.py": [
            "tests/test_close_at_eod.py",
            "tests/test_maxdiff_pnl.py",
        ],
        "trade_stock_e2e.py": [
            "tests/prod/trading/test_trade_stock_e2e.py",
            "tests/experimental/integration/integ/test_trade_stock_e2e_integ.py",
        ],
        "backtest_test3_inline.py": [
            "tests/prod/backtesting/test_backtest3.py",
        ],
    }

    if path.name in special_mappings:
        tests.extend(special_mappings[path.name])

    for pattern in test_patterns:
        if "*" in pattern:
            for test_file in _REPO_ROOT.glob(pattern):
                if _is_project_test_file(test_file):
                    tests.append(_repo_relative_path(test_file).as_posix())
        elif (_REPO_ROOT / pattern).exists() and _is_project_test_file(_REPO_ROOT / pattern):
            tests.append(pattern)

    if stem and path.exists():
        tests.extend(sorted(_test_import_index().get(stem, set())))

    return list(set(tests))


def prioritize_tests(changed_files: set[str]) -> tuple[list[str], list[str]]:
    """Return priority and remaining test file lists for the current diff."""
    priority_tests = set()

    for file_path in changed_files:
        tests = map_file_to_tests(file_path)
        priority_tests.update(tests)

    critical_tests = [
        "tests/prod/trading/test_trade_stock_e2e.py",
        "tests/test_close_at_eod.py",
        "tests/test_maxdiff_pnl.py",
    ]

    for test in critical_tests:
        if Path(test).exists():
            priority_tests.add(test)

    all_tests = {_repo_relative_path(test_file).as_posix() for test_file in _iter_project_test_files()}

    remaining_tests = all_tests - priority_tests
    return sorted(priority_tests), sorted(remaining_tests)


def _lane_failure_args(label: TestLane) -> list[str]:
    if label is TestLane.PRIORITY:
        return ["-x"]
    return ["--maxfail=20"]


def _cleanup_nested_pytest_root(path: Path) -> None:
    try:
        path.rmdir()
    except OSError:
        return


def _pytest_output_log_path(label: TestLane) -> Path:
    _DEFAULT_PYTEST_OUTPUT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return _DEFAULT_PYTEST_OUTPUT_LOG_DIR / f"{label.value}-{timestamp}.log"


def _latest_pytest_output_log_path(label: TestLane) -> Path:
    _DEFAULT_PYTEST_OUTPUT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    return _DEFAULT_PYTEST_OUTPUT_LOG_DIR / f"latest-{label.value}.log"


def _cleanup_output_log_dir() -> None:
    try:
        _DEFAULT_PYTEST_OUTPUT_LOG_DIR.rmdir()
    except OSError:
        return


def _build_pytest_command(
    test_files: list[str],
    label: TestLane,
    *,
    verbose: bool = False,
    nested_basetemp: str | None = None,
) -> list[str]:
    cmd = [sys.executable, "-m", "pytest"]
    if nested_basetemp is not None:
        cmd.extend(["--basetemp", nested_basetemp])
    cmd.extend(test_files)
    if verbose:
        cmd.append("-v")
    cmd.extend(["--ignore=tests/experimental"])
    cmd.extend(_lane_failure_args(label))
    return cmd


def _ensure_safe_repo_managed_artifact_path(target: Path) -> None:
    absolute_target = target.expanduser().absolute()
    try:
        absolute_target.relative_to(_REPO_ROOT)
    except ValueError:
        return

    current = absolute_target
    while True:
        if current.is_symlink():
            raise RuntimeError(
                f"Refusing to write smart test runner artifact through symlinked path: {current}"
            )
        if current == _REPO_ROOT:
            break
        parent = current.parent
        if parent == current:
            break
        current = parent


def _write_private_text_artifact(target: Path, content: str) -> None:
    _ensure_safe_repo_managed_artifact_path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
        text=True,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.chmod(0o600)
        os.replace(tmp_path, target)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass


def _run_subprocess_with_tee(cmd: list[str], *, output_log_path: Path) -> int:
    _ensure_safe_repo_managed_artifact_path(output_log_path)
    output_log_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{output_log_path.name}.",
        suffix=".tmp",
        dir=output_log_path.parent,
        text=True,
    )
    tmp_path = Path(tmp_name)
    process: subprocess.Popen[str] | None = None
    interrupted = False
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log_handle.write(line)
            returncode = process.wait()
    except KeyboardInterrupt:
        interrupted = True
        if process is not None and process.poll() is None:
            process.terminate()
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=_SUBPROCESS_INTERRUPT_WAIT_TIMEOUT_SECONDS)
            if process.poll() is None:
                process.kill()
                with suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=_SUBPROCESS_INTERRUPT_WAIT_TIMEOUT_SECONDS)
    finally:
        with suppress(OSError):
            tmp_path.chmod(0o600)
        if process is not None and process.stdout is not None:
            with suppress(Exception):
                process.stdout.close()
        try:
            os.replace(tmp_path, output_log_path)
        except FileNotFoundError:
            pass
        except OSError:
            with suppress(FileNotFoundError, OSError):
                tmp_path.unlink()
            raise
        else:
            with suppress(FileNotFoundError, OSError):
                tmp_path.unlink()
    if interrupted:
        raise KeyboardInterrupt
    return returncode


def _update_latest_failure_log(label: TestLane, output_log_path: Path) -> Path:
    latest_log_path = _latest_pytest_output_log_path(label)
    if output_log_path.exists():
        content = output_log_path.read_text(encoding="utf-8")
    else:
        content = ""
    _write_private_text_artifact(latest_log_path, content)
    return latest_log_path


def _format_log_open_command(path: Path) -> str:
    quoted_path = shlex.quote(str(path))
    if shutil.which("less") is not None:
        return f"less {quoted_path}"
    return f"cat {quoted_path}"


def _report_lane_failure(
    *,
    label: TestLane,
    message: str,
    output_log_path: Path,
    status: TestRunStatus,
    test_files: list[str],
    verbose: bool,
    failure_detail: str | None = None,
) -> LaneRunResult:
    latest_log_path = _update_latest_failure_log(label, output_log_path)
    print(f"\n{message}")
    if failure_detail:
        print(f"Detail: {failure_detail}")
    print(f"Pytest output log: {output_log_path}")
    print(f"Latest failure log: {latest_log_path}")
    print(f"Open log with: {_format_log_open_command(latest_log_path)}")
    print("Rerun command:")
    print(f"  {format_rerun_command(test_files, label=label, verbose=verbose)}")
    return LaneRunResult(ok=False, output_log_path=output_log_path, status=status)


def _pytest_log_has_terminal_summary(output_log_path: Path) -> bool:
    try:
        log_text = output_log_path.read_text(encoding="utf-8")
    except OSError:
        return False

    if not log_text.strip():
        return False

    tail = "\n".join(log_text.splitlines()[-20:])
    if "short test summary info" in tail:
        return True
    return bool(
        re.search(
            r"\b\d+\s+(?:passed|failed|error|errors|skipped|xfailed|xpassed|deselected)\b|no tests ran",
            tail,
        )
    )


def _subprocess_failure_detail(returncode: int, output_log_path: Path) -> str | None:
    if returncode < 0:
        signal_number = -returncode
        with suppress(ValueError):
            signal_name = signal.Signals(signal_number).name
            return f"Pytest process was terminated by signal {signal_number} ({signal_name})."
        return f"Pytest process was terminated by signal {signal_number}."
    if not _pytest_log_has_terminal_summary(output_log_path):
        return (
            f"Pytest exited with code {returncode} without writing a terminal summary; "
            "the process may have been killed or interrupted externally."
        )
    return None


def _run_test_lane(
    test_files: list[str],
    label: TestLane,
    *,
    verbose: bool = False,
    dry_run: bool = False,
) -> LaneRunResult:
    """Run pytest for the selected files and return a structured lane result."""
    if not test_files:
        print(f"No {label.value} tests to run")
        return LaneRunResult(ok=True, output_log_path=None, status=TestRunStatus.PASSED)

    print(f"\n{'=' * 80}")
    print(label.value.upper())
    print(f"{'=' * 80}")
    print(f"Running {len(test_files)} test(s):")
    for test in test_files:
        print(f"  - {test}")
    print()

    if dry_run:
        dry_run_cmd = _build_pytest_command(test_files, label, verbose=verbose)
        print("DRY RUN: Would execute:")
        print(f"  {' '.join(shlex.quote(part) for part in dry_run_cmd)}")
        print(f"  Nested basetemp: {nested_pytest_basetemp_root() / f'{label.value}-<random>' / 'basetemp'}")
        return LaneRunResult(ok=True, output_log_path=None, status=TestRunStatus.DRY_RUN)

    basetemp_root = nested_pytest_basetemp_root()
    basetemp_root.mkdir(parents=True, exist_ok=True)
    nested_run_root = Path(tempfile.mkdtemp(prefix=f"{label.value}-", dir=basetemp_root))
    output_log_path = _pytest_output_log_path(label)
    nested_basetemp = str(nested_run_root / "basetemp")
    cmd = _build_pytest_command(
        test_files,
        label,
        verbose=verbose,
        nested_basetemp=nested_basetemp,
    )

    try:
        returncode = _run_subprocess_with_tee(cmd, output_log_path=output_log_path)
    except KeyboardInterrupt:
        return _report_lane_failure(
            label=label,
            message=f"🛑 {label.value.upper()} TESTS INTERRUPTED",
            output_log_path=output_log_path,
            status=TestRunStatus.INTERRUPTED,
            test_files=test_files,
            verbose=verbose,
        )
    except (OSError, RuntimeError) as exc:
        return _report_lane_failure(
            label=label,
            message=f"❌ {label.value.upper()} TESTS FAILED TO START: {exc}",
            output_log_path=output_log_path,
            status=TestRunStatus.FAILED,
            test_files=test_files,
            verbose=verbose,
        )
    finally:
        shutil.rmtree(nested_run_root, ignore_errors=True)
        if basetemp_root == _NESTED_PYTEST_BASETEMP_ROOT:
            _cleanup_nested_pytest_root(basetemp_root)
    if returncode != 0:
        return _report_lane_failure(
            label=label,
            message=f"❌ {label.value.upper()} TESTS FAILED",
            output_log_path=output_log_path,
            status=TestRunStatus.FAILED,
            test_files=test_files,
            verbose=verbose,
            failure_detail=_subprocess_failure_detail(returncode, output_log_path),
        )

    output_log_path.unlink(missing_ok=True)
    _cleanup_output_log_dir()
    print(f"\n✅ {label.value.upper()} TESTS PASSED")
    return LaneRunResult(ok=True, output_log_path=None, status=TestRunStatus.PASSED)


def run_tests(test_files: list[str], label: TestLane, verbose: bool = False, dry_run: bool = False) -> bool:
    """Run pytest for the selected files and return success status."""
    return _run_test_lane(test_files, label, verbose=verbose, dry_run=dry_run).ok


def format_rerun_command(test_files: list[str], *, label: TestLane, verbose: bool = False) -> str:
    cmd = _build_pytest_command(test_files, label, verbose=verbose)
    return shlex.join(cmd)


def _build_run_summary(
    *,
    args: argparse.Namespace,
    changed_files: set[str],
    exit_code: int,
    priority_duration_seconds: float | None,
    priority_output_log_path: Path | None,
    priority_status: TestRunStatus,
    priority_tests: list[str],
    remaining_duration_seconds: float | None,
    remaining_output_log_path: Path | None,
    remaining_status: TestRunStatus,
    remaining_tests: list[str],
    run_started_at: str,
    run_started_monotonic: float,
    selected_lane: SelectedLane,
    summary_json_path: Path | None,
) -> TestRunSummary:
    return TestRunSummary(
        base_branch=args.base_branch,
        changed_files=sorted(changed_files),
        cwd=str(Path.cwd()),
        dry_run=args.dry_run,
        exit_code=exit_code,
        python_executable=sys.executable,
        remaining_duration_seconds=remaining_duration_seconds,
        remaining_output_log_path=None if remaining_output_log_path is None else str(remaining_output_log_path),
        summary_json_path=None if summary_json_path is None else str(summary_json_path),
        priority_duration_seconds=priority_duration_seconds,
        priority_output_log_path=None if priority_output_log_path is None else str(priority_output_log_path),
        priority_rerun_command=(
            format_rerun_command(priority_tests, label=TestLane.PRIORITY, verbose=args.verbose)
            if priority_tests
            else None
        ),
        priority_status=priority_status,
        priority_tests=priority_tests,
        run_finished_at_utc=_utc_now_iso(),
        run_started_at_utc=run_started_at,
        remaining_rerun_command=(
            format_rerun_command(remaining_tests, label=TestLane.REMAINING, verbose=args.verbose)
            if remaining_tests
            else None
        ),
        remaining_status=remaining_status,
        remaining_tests=remaining_tests,
        selected_lane=selected_lane,
        total_duration_seconds=round(time.monotonic() - run_started_monotonic, 6),
    )


def _write_summary_json(path: str | None, summary: TestRunSummary) -> Path | None:
    target = _DEFAULT_SUMMARY_JSON_PATH if not path else Path(path).expanduser()
    try:
        _write_private_text_artifact(target, json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n")
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(f"Failed to write smart test runner summary JSON to {target}: {exc}") from exc
    return target


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolved_summary_json_path(path: str | None, *, disabled: bool) -> Path | None:
    if disabled:
        return None
    return _DEFAULT_SUMMARY_JSON_PATH if not path else Path(path).expanduser()


def _lane_result_status(result: LaneRunResult, *, dry_run: bool) -> TestRunStatus:
    if result.status is not None:
        return result.status
    if dry_run:
        return TestRunStatus.DRY_RUN
    return TestRunStatus.PASSED if result.ok else TestRunStatus.FAILED


def _selected_lane_from_args(args: argparse.Namespace) -> SelectedLane:
    return (
        SelectedLane.PRIORITY_ONLY
        if args.priority_only
        else SelectedLane.REMAINING_ONLY if args.remaining_only else SelectedLane.PRIORITY_AND_REMAINING
    )


def _resolved_runner_config(args: argparse.Namespace) -> ResolvedRunnerConfig:
    summary_path = _resolved_summary_json_path(args.summary_json, disabled=args.no_summary_json)
    basetemp_override = os.getenv(_BASE_TEMP_ROOT_ENV_VAR)
    return ResolvedRunnerConfig(
        base_branch=args.base_branch,
        basetemp_root=str(nested_pytest_basetemp_root()),
        basetemp_root_source=BasetempRootSource.ENV if basetemp_override else BasetempRootSource.DEFAULT,
        import_index_cache_path=str(_DEFAULT_IMPORT_INDEX_CACHE_PATH),
        python_executable=sys.executable,
        repo_root=str(_REPO_ROOT),
        selected_lane=_selected_lane_from_args(args),
        summary_json_enabled=summary_path is not None,
        summary_json_path=None if summary_path is None else str(summary_path),
        summary_json_source=(
            SummaryJsonSource.DISABLED
            if args.no_summary_json
            else SummaryJsonSource.CLI if args.summary_json else SummaryJsonSource.DEFAULT
        ),
    )


def _print_runner_config(config: ResolvedRunnerConfig) -> None:
    print("Smart Test Runner Config")
    print("=" * 80)
    print(f"Repo root:           {config.repo_root}")
    print(f"Python executable:   {config.python_executable}")
    print(f"Selected lane:       {config.selected_lane.value}")
    print(f"Base branch:         {config.base_branch}")
    print(f"Basetemp root:       {config.basetemp_root}")
    print(f"Basetemp source:     {config.basetemp_root_source}")
    print(f"Import index cache:  {config.import_index_cache_path}")
    print(f"Summary JSON:        {config.summary_json_path or 'disabled'}")
    print(f"Summary source:      {config.summary_json_source}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smart test runner with change-based prioritization")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose pytest output")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Show what would be tested without running")
    parser.add_argument("--base-branch", "-b", default="main", help="Base branch for comparison (default: main)")
    parser.add_argument("--summary-json", help="Write a machine-readable run summary to this path")
    parser.add_argument(
        "--no-summary-json",
        action="store_true",
        help="Disable summary JSON output, including the default .pytest_cache summary file",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved runner configuration and exit",
    )
    lane_group = parser.add_mutually_exclusive_group()
    lane_group.add_argument(
        "--priority-only",
        action="store_true",
        help="Run only the fail-fast priority lane",
    )
    lane_group.add_argument(
        "--remaining-only",
        action="store_true",
        help="Run only the remaining non-priority lane",
    )
    args = parser.parse_args()

    ensure_repo_managed_interpreter()
    if args.print_config:
        _print_runner_config(_resolved_runner_config(args))
        sys.exit(0)
    print("Smart Test Runner")
    print("=" * 80)

    changed_files = get_changed_files(args.base_branch)
    if changed_files:
        print(f"\nDetected {len(changed_files)} changed file(s):")
        for file_path in sorted(changed_files)[:_CHANGED_FILE_PREVIEW_LIMIT]:
            print(f"  - {file_path}")
        if len(changed_files) > _CHANGED_FILE_PREVIEW_LIMIT:
            print(f"  ... and {len(changed_files) - _CHANGED_FILE_PREVIEW_LIMIT} more")
    else:
        print("\nNo changed files detected (running all tests)")

    priority_tests, remaining_tests = prioritize_tests(changed_files)
    priority_status = TestRunStatus.NOT_RUN
    remaining_status = TestRunStatus.NOT_RUN

    print("\nTest execution plan:")
    print(f"  Priority tests (fail-fast): {len(priority_tests)}")
    print(f"  Remaining tests: {len(remaining_tests)}")
    if args.priority_only:
        print("  Selected lane: priority only")
    elif args.remaining_only:
        print("  Selected lane: remaining only")
    else:
        print("  Selected lane: priority + remaining")

    selected_lane = _selected_lane_from_args(args)
    summary_json_path = _resolved_summary_json_path(args.summary_json, disabled=args.no_summary_json)
    run_started_at = _utc_now_iso()
    run_started_monotonic = time.monotonic()
    priority_duration_seconds: float | None = None
    priority_output_log_path: Path | None = None
    remaining_duration_seconds: float | None = None
    remaining_output_log_path: Path | None = None
    exit_code = 0

    def _emit_summary_or_exit() -> None:
        nonlocal exit_code
        try:
            summary_path = (
                None
                if summary_json_path is None
                else _write_summary_json(
                    str(summary_json_path),
                    _build_run_summary(
                        args=args,
                        changed_files=changed_files,
                        exit_code=exit_code,
                        priority_duration_seconds=priority_duration_seconds,
                        priority_output_log_path=priority_output_log_path,
                        priority_status=priority_status,
                        priority_tests=priority_tests,
                        remaining_duration_seconds=remaining_duration_seconds,
                        remaining_output_log_path=remaining_output_log_path,
                        remaining_status=remaining_status,
                        remaining_tests=remaining_tests,
                        run_started_at=run_started_at,
                        run_started_monotonic=run_started_monotonic,
                        selected_lane=selected_lane,
                        summary_json_path=summary_json_path,
                    ),
                )
            )
        except RuntimeError as exc:
            print(f"\n❌ {exc}")
            exit_code = 1
            sys.exit(exit_code)
        if summary_path is not None:
            print(f"Summary JSON: {summary_path}")

    if not args.dry_run and _should_check_for_competing_repo_pytest_processes():
        try:
            _assert_no_competing_repo_pytest_processes()
        except RuntimeError as exc:
            print(f"\n❌ {exc}")
            exit_code = 1
            _emit_summary_or_exit()
            sys.exit(exit_code)

    if args.remaining_only:
        remaining_started_monotonic = time.monotonic()
        remaining_result = _run_test_lane(remaining_tests, TestLane.REMAINING, verbose=args.verbose, dry_run=args.dry_run)
        remaining_duration_seconds = round(time.monotonic() - remaining_started_monotonic, 6)
        remaining_output_log_path = remaining_result.output_log_path
        remaining_ok = remaining_result.ok
        remaining_status = _lane_result_status(remaining_result, dry_run=args.dry_run)
        if remaining_status is TestRunStatus.INTERRUPTED:
            exit_code = 130
    else:
        priority_started_monotonic = time.monotonic()
        priority_result = _run_test_lane(priority_tests, TestLane.PRIORITY, verbose=args.verbose, dry_run=args.dry_run)
        priority_duration_seconds = round(time.monotonic() - priority_started_monotonic, 6)
        priority_output_log_path = priority_result.output_log_path
        priority_ok = priority_result.ok
        priority_status = _lane_result_status(priority_result, dry_run=args.dry_run)
        if not priority_ok:
            if priority_status is TestRunStatus.INTERRUPTED:
                print("\n🛑 PRIORITY TESTS INTERRUPTED - Stopping here")
                exit_code = 130
            else:
                print("\n❌ PRIORITY TESTS FAILED - Stopping here (fail-fast)")
                exit_code = 1
            _emit_summary_or_exit()
            sys.exit(exit_code)
        if args.priority_only:
            remaining_ok = True
        else:
            remaining_started_monotonic = time.monotonic()
            remaining_result = _run_test_lane(
                remaining_tests,
                TestLane.REMAINING,
                verbose=args.verbose,
                dry_run=args.dry_run,
            )
            remaining_duration_seconds = round(time.monotonic() - remaining_started_monotonic, 6)
            remaining_output_log_path = remaining_result.output_log_path
            remaining_ok = remaining_result.ok
            remaining_status = _lane_result_status(remaining_result, dry_run=args.dry_run)
            if not remaining_ok:
                if remaining_status is TestRunStatus.INTERRUPTED:
                    print("\n🛑 REMAINING TESTS INTERRUPTED")
                    exit_code = 130
                else:
                    print("\n⚠️  SOME REMAINING TESTS FAILED (non-fatal)")

    print("\n" + "=" * 80)
    if args.priority_only:
        if priority_status is TestRunStatus.INTERRUPTED:
            print("🛑 PRIORITY TESTS INTERRUPTED")
        else:
            print("✅ PRIORITY TESTS PASSED")
    elif args.remaining_only:
        if remaining_status is TestRunStatus.INTERRUPTED:
            print("🛑 REMAINING TESTS INTERRUPTED")
        elif remaining_ok:
            print("✅ REMAINING TESTS PASSED")
        else:
            print("❌ REMAINING TESTS FAILED")
            exit_code = 1
    elif remaining_status is TestRunStatus.INTERRUPTED:
        print("🛑 PRIORITY TESTS PASSED; REMAINING TESTS WERE INTERRUPTED")
    elif remaining_ok:
        print("✅ ALL TESTS PASSED")
    else:
        print("⚠️  PRIORITY TESTS PASSED; REMAINING TESTS HAD FAILURES")
    print("=" * 80)
    _emit_summary_or_exit()
    sys.exit(exit_code)
