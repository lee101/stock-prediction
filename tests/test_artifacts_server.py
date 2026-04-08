from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from src import artifacts_server
from src.artifacts_server import create_app


def test_artifacts_server_lists_and_serves(tmp_path: Path) -> None:
    (tmp_path / "run1").mkdir()
    (tmp_path / "run1" / "video.mp4").write_bytes(b"fake-mp4-bytes")
    (tmp_path / "top.txt").write_text("hello", encoding="utf-8")

    client = TestClient(create_app(tmp_path))

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["ok"] is True

    response = client.get("/")
    assert response.status_code == 200
    names = {entry["name"] for entry in response.json()["entries"]}
    assert {"run1", "top.txt"} <= names

    response = client.get("/list/run1")
    assert response.status_code == 200
    assert any(entry["name"] == "video.mp4" for entry in response.json()["entries"])

    response = client.get("/files/run1/video.mp4")
    assert response.status_code == 200
    assert response.content == b"fake-mp4-bytes"

    response = client.get("/list/does-not-exist")
    assert response.status_code == 404


def test_artifacts_server_path_traversal_blocked(tmp_path: Path) -> None:
    TestClient(create_app(tmp_path))
    artifacts_server.ARTIFACTS_ROOT = tmp_path.resolve()
    with pytest.raises(HTTPException, match="path outside artifacts root"):
        artifacts_server._safe_join("../../etc/passwd")


def test_artifacts_server_safe_entry_path_blocks_outside_root(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")

    with pytest.raises(HTTPException, match="path outside artifacts root"):
        artifacts_server._safe_entry_path(outside, root=tmp_path)


def test_artifacts_server_skips_external_symlink_and_blocks_fetch(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    artifacts_root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (artifacts_root / "safe.txt").write_text("ok", encoding="utf-8")
    (artifacts_root / "escape.txt").symlink_to(outside)

    client = TestClient(create_app(artifacts_root))

    root_entries = {entry["name"] for entry in client.get("/").json()["entries"]}
    assert "safe.txt" in root_entries
    assert "escape.txt" not in root_entries

    response = client.get("/files/escape.txt")
    assert response.status_code == 400
    assert response.json()["detail"] == "path outside artifacts root"


def test_artifacts_server_paginates_root_listing(tmp_path: Path) -> None:
    for name in ["c.txt", "a.txt", "b.txt"]:
        (tmp_path / name).write_text(name, encoding="utf-8")

    client = TestClient(create_app(tmp_path))
    response = client.get("/?limit=2&offset=1")

    assert response.status_code == 200
    payload = response.json()
    assert [entry["name"] for entry in payload["entries"]] == ["b.txt", "c.txt"]
    assert payload["offset"] == 1
    assert payload["limit"] == 2
    assert payload["returned_entries"] == 2
    assert payload["total_entries"] == 3
    assert payload["has_more"] is False
    assert payload["next_offset"] is None


def test_artifacts_server_paginates_subdirectory_listing(tmp_path: Path) -> None:
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    for idx in range(5):
        (run_dir / f"file{idx}.txt").write_text(str(idx), encoding="utf-8")

    client = TestClient(create_app(tmp_path))
    response = client.get("/list/run1?limit=2&offset=2")

    assert response.status_code == 200
    payload = response.json()
    assert payload["path"] == "run1"
    assert [entry["name"] for entry in payload["entries"]] == ["file2.txt", "file3.txt"]
    assert payload["returned_entries"] == 2
    assert payload["total_entries"] == 5
    assert payload["has_more"] is True
    assert payload["next_offset"] == 4


def test_artifacts_server_apps_keep_separate_roots(tmp_path: Path) -> None:
    root_a = tmp_path / "artifacts-a"
    root_b = tmp_path / "artifacts-b"
    root_a.mkdir()
    root_b.mkdir()
    (root_a / "only-a.txt").write_text("a", encoding="utf-8")
    (root_b / "only-b.txt").write_text("b", encoding="utf-8")

    client_a = TestClient(create_app(root_a))
    client_b = TestClient(create_app(root_b))

    payload_a = client_a.get("/").json()
    payload_b = client_b.get("/").json()

    assert payload_a["root"] == str(root_a.resolve())
    assert [entry["name"] for entry in payload_a["entries"]] == ["only-a.txt"]
    assert payload_b["root"] == str(root_b.resolve())
    assert [entry["name"] for entry in payload_b["entries"]] == ["only-b.txt"]


def test_artifacts_server_reuses_cached_directory_listing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")

    real_list_children = artifacts_server._list_children
    calls: list[Path] = []

    def _counting_list_children(dir_path: Path, *, root: Path | None = None) -> list[dict[str, object]]:
        calls.append(dir_path)
        return real_list_children(dir_path, root=root)

    monkeypatch.setattr(artifacts_server, "_list_children", _counting_list_children)
    monkeypatch.setattr(artifacts_server, "LISTING_CACHE_TTL_SECONDS", 60.0)
    artifacts_server._LISTING_CACHE.clear()

    client = TestClient(create_app(tmp_path))
    first = client.get("/")
    second = client.get("/")

    assert first.status_code == 200
    assert second.status_code == 200
    assert len(calls) == 1


def test_artifacts_server_invalidates_cached_listing_when_directory_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")

    real_list_children = artifacts_server._list_children
    calls: list[Path] = []

    def _counting_list_children(dir_path: Path, *, root: Path | None = None) -> list[dict[str, object]]:
        calls.append(dir_path)
        return real_list_children(dir_path, root=root)

    monkeypatch.setattr(artifacts_server, "_list_children", _counting_list_children)
    monkeypatch.setattr(artifacts_server, "LISTING_CACHE_TTL_SECONDS", 60.0)
    artifacts_server._LISTING_CACHE.clear()

    client = TestClient(create_app(tmp_path))
    first = client.get("/")
    assert first.status_code == 200
    assert [entry["name"] for entry in first.json()["entries"]] == ["a.txt"]

    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    second = client.get("/")

    assert second.status_code == 200
    assert [entry["name"] for entry in second.json()["entries"]] == ["a.txt", "b.txt"]
    assert len(calls) == 2
