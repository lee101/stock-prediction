from __future__ import annotations

from pathlib import Path


def test_pytest_ini_does_not_force_shared_repo_basetemp() -> None:
    text = (Path(__file__).resolve().parents[1] / "pytest.ini").read_text(encoding="utf-8")

    assert "--basetemp" not in text
    assert ".pytest_tmp" in text
