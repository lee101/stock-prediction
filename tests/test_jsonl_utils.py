from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import unified_orchestrator.jsonl_utils as jsonl_utils
from unified_orchestrator.jsonl_utils import append_jsonl_row, iter_jsonl_lines_reverse


def test_iter_jsonl_lines_reverse_handles_chunk_boundaries_and_blank_lines(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    rows = [
        '{"id": 1, "payload": "alphaalpha"}',
        "",
        '{"id": 2, "payload": "betabeta"}',
        '{"id": 3, "payload": "gammagamma"}',
    ]
    path.write_text("\n".join(rows), encoding="utf-8")

    out = list(iter_jsonl_lines_reverse(path, chunk_size=7))

    assert out == [
        '{"id": 3, "payload": "gammagamma"}',
        '{"id": 2, "payload": "betabeta"}',
        '{"id": 1, "payload": "alphaalpha"}',
    ]


def test_iter_jsonl_lines_reverse_handles_missing_trailing_newline(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"id": 1}\n{"id": 2}', encoding="utf-8")

    out = list(iter_jsonl_lines_reverse(path, chunk_size=4))

    assert out == ['{"id": 2}', '{"id": 1}']


def test_append_jsonl_row_round_trips_payload(tmp_path: Path):
    path = tmp_path / "events.jsonl"

    append_jsonl_row(path, {"b": 1, "a": 2}, sort_keys=True)

    assert path.read_text(encoding="utf-8") == '{"a": 2, "b": 1}\n'


def test_append_jsonl_row_uses_file_lock_when_available(tmp_path: Path, monkeypatch):
    path = tmp_path / "events.jsonl"
    calls: list[object] = []

    class _FakeFcntl:
        LOCK_EX = "lock_ex"
        LOCK_UN = "lock_un"

        @staticmethod
        def flock(_fileno: int, op: object) -> None:
            calls.append(op)

    monkeypatch.setattr(jsonl_utils, "_fcntl", _FakeFcntl)

    append_jsonl_row(path, {"id": 1})

    assert calls == ["lock_ex", "lock_un"]


def test_append_jsonl_row_handles_threaded_writers_without_invalid_json(tmp_path: Path):
    path = tmp_path / "events.jsonl"

    def _write_one(idx: int) -> None:
        append_jsonl_row(path, {"id": idx})

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(_write_one, range(80)))

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert len(rows) == 80
    assert sorted(row["id"] for row in rows) == list(range(80))
