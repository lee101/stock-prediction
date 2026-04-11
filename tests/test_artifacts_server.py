from __future__ import annotations

import threading
from dataclasses import replace
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
    artifacts_server._clear_listing_cache()

    client = TestClient(create_app(tmp_path))

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert response.json()["root_exists"] is True
    assert response.json()["config"] == {
        "root_source": "argument",
        "default_list_limit": artifacts_server.DEFAULT_LIST_LIMIT,
        "max_list_limit": artifacts_server.MAX_LIST_LIMIT,
        "listing_cache_ttl_seconds": artifacts_server.LISTING_CACHE_TTL_SECONDS,
        "listing_cache_max_entries": artifacts_server.LISTING_CACHE_MAX_ENTRIES,
        "listing_cache_inflight_wait_seconds": artifacts_server.LISTING_CACHE_INFLIGHT_WAIT_SECONDS,
    }
    assert response.json()["cache"] == {
        "entries": 0,
        "hits": 0,
        "misses": 0,
        "stores": 0,
        "waits": 0,
        "ttl_seconds": artifacts_server.LISTING_CACHE_TTL_SECONDS,
        "max_entries": artifacts_server.LISTING_CACHE_MAX_ENTRIES,
    }

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


def test_artifacts_server_hidden_path_helper_flags_hidden_segments() -> None:
    assert artifacts_server._has_hidden_path_segment(".hidden.txt") is True
    assert artifacts_server._has_hidden_path_segment("dir/.secret/file.txt") is True
    assert artifacts_server._has_hidden_path_segment("visible/file.txt") is False
    assert artifacts_server._has_hidden_path_segment("../visible/file.txt") is False


def test_artifacts_server_env_helpers_accept_valid_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARTIFACTS_TEST_INT", "42")
    monkeypatch.setenv("ARTIFACTS_TEST_FLOAT", "2.5")

    assert artifacts_server._read_env_int("ARTIFACTS_TEST_INT", 1, minimum=1) == 42
    assert artifacts_server._read_env_float("ARTIFACTS_TEST_FLOAT", 1.0, minimum=0.0) == 2.5


@pytest.mark.parametrize(
    ("name", "value", "reader", "default", "minimum", "match"),
    [
        ("ARTIFACTS_TEST_INT", "abc", artifacts_server._read_env_int, 1, 1, "must be an integer"),
        ("ARTIFACTS_TEST_INT", "0", artifacts_server._read_env_int, 1, 1, "must be >="),
        ("ARTIFACTS_TEST_FLOAT", "abc", artifacts_server._read_env_float, 1.0, 0.0, "must be a float"),
        ("ARTIFACTS_TEST_FLOAT", "-1", artifacts_server._read_env_float, 1.0, 0.0, "must be >="),
    ],
)
def test_artifacts_server_env_helpers_reject_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
    reader,
    default,
    minimum,
    match: str,
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(RuntimeError, match=match):
        reader(name, default, minimum=minimum)


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


def test_artifacts_server_hides_dotfiles_and_hidden_directories(tmp_path: Path) -> None:
    (tmp_path / ".hidden.txt").write_text("secret", encoding="utf-8")
    (tmp_path / ".private").mkdir()
    (tmp_path / ".private" / "note.txt").write_text("secret", encoding="utf-8")
    (tmp_path / "visible.txt").write_text("ok", encoding="utf-8")

    client = TestClient(create_app(tmp_path))

    response = client.get("/")
    assert response.status_code == 200
    names = {entry["name"] for entry in response.json()["entries"]}
    assert "visible.txt" in names
    assert ".hidden.txt" not in names
    assert ".private" not in names

    hidden_file = client.get("/files/.hidden.txt")
    assert hidden_file.status_code == 404
    assert hidden_file.json()["detail"] == "file not found"

    hidden_dir = client.get("/list/.private")
    assert hidden_dir.status_code == 404
    assert hidden_dir.json()["detail"] == "file not found"


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
    health_a = client_a.get("/health").json()
    health_b = client_b.get("/health").json()

    assert payload_a["root"] == str(root_a.resolve())
    assert health_a["config"]["root_source"] == "argument"
    assert [entry["name"] for entry in payload_a["entries"]] == ["only-a.txt"]
    assert payload_b["root"] == str(root_b.resolve())
    assert health_b["config"]["root_source"] == "argument"
    assert [entry["name"] for entry in payload_b["entries"]] == ["only-b.txt"]
    assert health_a["cache"]["entries"] == 1
    assert health_b["cache"]["entries"] == 1
    assert health_a["cache"]["misses"] == 1
    assert health_b["cache"]["misses"] == 1


def test_artifacts_server_create_app_accepts_explicit_settings(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    custom_settings = replace(
        artifacts_server._DEFAULT_SETTINGS,
        default_list_limit=1,
        max_list_limit=3,
        listing_cache_ttl_seconds=5.0,
        listing_cache_max_entries=8,
    )

    client = TestClient(create_app(tmp_path, settings=custom_settings))

    payload = client.get("/").json()
    health = client.get("/health").json()

    assert payload["limit"] == 1
    assert payload["returned_entries"] == 1
    assert health["config"] == {
        "root_source": "argument",
        "default_list_limit": 1,
        "max_list_limit": 3,
        "listing_cache_ttl_seconds": 5.0,
        "listing_cache_max_entries": 8,
        "listing_cache_inflight_wait_seconds": artifacts_server.LISTING_CACHE_INFLIGHT_WAIT_SECONDS,
    }


def test_artifacts_server_reuses_cached_directory_listing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")

    real_list_children = artifacts_server._list_children
    calls: list[Path] = []

    def _counting_list_children(
        dir_path: Path,
        *,
        root: Path | None = None,
    ) -> list[artifacts_server.ArtifactEntry]:
        calls.append(dir_path)
        return real_list_children(dir_path, root=root)

    monkeypatch.setattr(artifacts_server, "_list_children", _counting_list_children)
    monkeypatch.setattr(artifacts_server, "LISTING_CACHE_TTL_SECONDS", 60.0)
    artifacts_server._clear_listing_cache()

    client = TestClient(create_app(tmp_path))
    first = client.get("/")
    second = client.get("/")

    assert first.status_code == 200
    assert second.status_code == 200
    assert len(calls) == 1
    stats = client.get("/health").json()["cache"]
    assert stats["entries"] == 1
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["stores"] == 1
    assert stats["waits"] == 0


def test_artifacts_server_invalidates_cached_listing_when_directory_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")

    real_list_children = artifacts_server._list_children
    calls: list[Path] = []

    def _counting_list_children(
        dir_path: Path,
        *,
        root: Path | None = None,
    ) -> list[artifacts_server.ArtifactEntry]:
        calls.append(dir_path)
        return real_list_children(dir_path, root=root)

    monkeypatch.setattr(artifacts_server, "_list_children", _counting_list_children)
    monkeypatch.setattr(artifacts_server, "LISTING_CACHE_TTL_SECONDS", 60.0)
    artifacts_server._clear_listing_cache()

    client = TestClient(create_app(tmp_path))
    first = client.get("/")
    assert first.status_code == 200
    assert [entry["name"] for entry in first.json()["entries"]] == ["a.txt"]

    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    second = client.get("/")

    assert second.status_code == 200
    assert [entry["name"] for entry in second.json()["entries"]] == ["a.txt", "b.txt"]
    assert len(calls) == 2
    stats = client.get("/health").json()["cache"]
    assert stats["entries"] == 1
    assert stats["hits"] == 0
    assert stats["misses"] == 2
    assert stats["stores"] == 2
    assert stats["waits"] == 0


def test_artifacts_server_cache_handles_concurrent_misses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")

    real_list_children = artifacts_server._list_children
    start_event = threading.Event()
    release_event = threading.Event()
    calls: list[Path] = []

    def _blocking_list_children(
        dir_path: Path,
        *,
        root: Path | None = None,
    ) -> list[artifacts_server.ArtifactEntry]:
        calls.append(dir_path)
        start_event.set()
        release_event.wait(timeout=2)
        return real_list_children(dir_path, root=root)

    monkeypatch.setattr(artifacts_server, "_list_children", _blocking_list_children)
    monkeypatch.setattr(artifacts_server, "LISTING_CACHE_TTL_SECONDS", 60.0)
    artifacts_server._clear_listing_cache()

    results: list[list[dict[str, object]]] = []
    errors: list[BaseException] = []

    def _worker() -> None:
        try:
            results.append(artifacts_server._cached_list_children(tmp_path, root=tmp_path))
        except BaseException as exc:  # pragma: no cover - failures are asserted below
            errors.append(exc)

    threads = [threading.Thread(target=_worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    assert start_event.wait(timeout=2)
    release_event.set()
    for thread in threads:
        thread.join(timeout=2)

    assert all(not thread.is_alive() for thread in threads)
    assert not errors
    assert len(results) == 2
    assert results[0] == results[1]
    assert len(calls) == 1
    stats = artifacts_server._listing_cache_stats()
    assert stats["entries"] == 1
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["stores"] == 1
    assert stats["waits"] == 1


def test_artifacts_server_cache_wait_times_out_when_inflight_never_finishes(tmp_path: Path) -> None:
    cache_state = artifacts_server._new_listing_cache_state()
    cache_key = (str(tmp_path.resolve()), str(tmp_path.resolve()))
    cache_state.inflight[cache_key] = threading.Event()

    with pytest.raises(RuntimeError, match="artifact listing cache wait timed out"):
        artifacts_server._cached_list_children(
            tmp_path,
            root=tmp_path,
            cache_state=cache_state,
            settings=artifacts_server._DEFAULT_SETTINGS,
        )
