from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binance_worksteal.io_utils import write_text_atomic


def test_write_text_atomic_writes_file_without_leaking_temp_files(tmp_path):
    path = tmp_path / "summary.json"
    body = '{"tool": "example"}\n'

    written = write_text_atomic(path, body)

    assert written == path
    assert path.read_text(encoding="utf-8") == body
    assert list(tmp_path.glob('.summary.json.tmp.*')) == []


def test_write_text_atomic_cleans_temp_file_on_replace_failure(tmp_path, monkeypatch):
    path = tmp_path / "summary.json"
    body = '{"tool": "example"}\n'
    captured = {}

    def _raise_replace(src, dst):
        captured["src"] = Path(src)
        captured["dst"] = Path(dst)
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", _raise_replace)

    with pytest.raises(OSError, match="disk full"):
        write_text_atomic(path, body)

    assert captured["dst"] == path
    assert not path.exists()
    assert not captured["src"].exists()
    assert list(tmp_path.glob('.summary.json.tmp.*')) == []


def test_write_text_atomic_preserves_existing_mode(tmp_path):
    path = tmp_path / "script.sh"
    path.write_text("echo old\n", encoding="utf-8")
    path.chmod(0o755)

    write_text_atomic(path, "echo new\n")

    assert path.read_text(encoding="utf-8") == "echo new\n"
    assert path.stat().st_mode & 0o777 == 0o755


def test_write_text_atomic_explicit_mode_overrides_existing_mode(tmp_path):
    path = tmp_path / "script.sh"
    path.write_text("echo old\n", encoding="utf-8")
    path.chmod(0o755)

    write_text_atomic(path, "echo new\n", mode=0o600)

    assert path.read_text(encoding="utf-8") == "echo new\n"
    assert path.stat().st_mode & 0o777 == 0o600
