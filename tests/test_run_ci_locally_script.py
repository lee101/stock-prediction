from __future__ import annotations

import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "run_ci_locally.sh"


def _run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )


def test_run_ci_locally_help_includes_job_selection_flags() -> None:
    result = _run_script("--help")

    assert result.returncode == 0
    assert "--job NAME" in result.stdout
    assert "Valid jobs: lint, tests, typecheck" in result.stdout
    assert "--dry-run" in result.stdout
    assert "python scripts/dev_setup_status.py" in result.stdout


def test_run_ci_locally_dry_run_reports_selected_job_without_running_ci() -> None:
    result = _run_script("--dry-run", "--job", "lint")

    assert result.returncode == 0
    assert "Selected jobs: lint" in result.stdout
    assert "Dry run: no commands executed" in result.stdout
    assert "JOB 1: Lint & Format Check" not in result.stdout


def test_run_ci_locally_rejects_unknown_job() -> None:
    result = _run_script("--job", "bogus")

    combined = result.stdout + result.stderr
    assert result.returncode == 2
    assert "Unknown job: bogus" in combined


def test_run_ci_locally_rejects_unknown_option() -> None:
    result = _run_script("--bogus")

    combined = result.stdout + result.stderr
    assert result.returncode == 2
    assert "Unknown option: --bogus" in combined


def test_run_ci_locally_requires_job_value() -> None:
    result = _run_script("--job")

    combined = result.stdout + result.stderr
    assert result.returncode == 2
    assert "--job requires a value" in combined
