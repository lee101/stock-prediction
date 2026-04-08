"""Tiny FastAPI app exposing models/artifacts/ over HTTP.

Run:
    uvicorn src.artifacts_server:app --host 0.0.0.0 --port 8765

Endpoints:
    GET  /            -> JSON tree of artifacts (lazy, one level)
    GET  /list/<sub>  -> JSON listing for a subdirectory
    GET  /files/<...> -> raw file (guarded file response)
    GET  /health      -> {"ok": true}

Designed to be read-only and safe to expose: paths are normalized and
constrained to ARTIFACTS_ROOT.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse


def _resolve_root() -> Path:
    env = os.environ.get("ARTIFACTS_ROOT")
    if env:
        root = Path(env).resolve()
    else:
        # Default: <repo>/models/artifacts
        root = (Path(__file__).resolve().parent.parent / "models" / "artifacts").resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


ARTIFACTS_ROOT: Path = _resolve_root()
DEFAULT_LIST_LIMIT = 200
MAX_LIST_LIMIT = 1000
LISTING_CACHE_TTL_SECONDS = 1.0
LISTING_CACHE_MAX_ENTRIES = 256


@dataclass(frozen=True)
class _DirectoryListingCacheEntry:
    directory_mtime_ns: int
    expires_at_monotonic: float
    entries: tuple[dict[str, object], ...]


def _safe_join(sub: str, *, root: Path | None = None) -> Path:
    root_path = (root or ARTIFACTS_ROOT).resolve()
    p = (root_path / sub).resolve()
    if root_path not in p.parents and p != root_path:
        raise HTTPException(status_code=400, detail="path outside artifacts root")
    return p


def _safe_entry_path(p: Path, *, root: Path | None = None) -> Path:
    root_path = (root or ARTIFACTS_ROOT).resolve()
    try:
        relative = p.relative_to(root_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="path outside artifacts root") from exc
    _safe_join(str(relative), root=root_path)
    return p


def _entry_payload_from_stat(
    *,
    path: Path,
    root: Path,
    stat_result: os.stat_result,
    is_dir: bool,
) -> dict[str, object]:
    return {
        "name": path.name,
        "is_dir": is_dir,
        "size": stat_result.st_size if not is_dir else None,
        "mtime": stat_result.st_mtime,
        "url": None if is_dir else f"/files/{path.relative_to(root).as_posix()}",
    }


def _list_children(dir_path: Path, *, root: Path | None = None) -> list[dict[str, object]]:
    root_path = (root or ARTIFACTS_ROOT).resolve()
    children: list[dict[str, object]] = []
    with os.scandir(dir_path) as iterator:
        entries = list(iterator)
    for child in entries:
        child_path = dir_path / child.name
        try:
            safe_child_path = _safe_entry_path(child_path, root=root_path)
            is_dir = child.is_dir()
            stat_result = child.stat()
        except (HTTPException, OSError):
            continue
        children.append(
            _entry_payload_from_stat(
                path=safe_child_path,
                root=root_path,
                stat_result=stat_result,
                is_dir=is_dir,
            )
        )
    children.sort(key=lambda entry: str(entry["name"]))
    return children


_LISTING_CACHE: dict[tuple[str, str], _DirectoryListingCacheEntry] = {}


def _prune_listing_cache() -> None:
    while len(_LISTING_CACHE) > LISTING_CACHE_MAX_ENTRIES:
        oldest_key = min(
            _LISTING_CACHE,
            key=lambda key: _LISTING_CACHE[key].expires_at_monotonic,
        )
        _LISTING_CACHE.pop(oldest_key, None)


def _cached_list_children(dir_path: Path, *, root: Path | None = None) -> list[dict[str, object]]:
    root_path = (root or ARTIFACTS_ROOT).resolve()
    resolved_dir_path = dir_path.resolve()
    directory_stat = resolved_dir_path.stat()
    cache_key = (str(root_path), str(resolved_dir_path))
    now_monotonic = time.monotonic()
    cached = _LISTING_CACHE.get(cache_key)
    if (
        cached is not None
        and cached.directory_mtime_ns == directory_stat.st_mtime_ns
        and cached.expires_at_monotonic >= now_monotonic
    ):
        return list(cached.entries)

    children = _list_children(resolved_dir_path, root=root_path)
    _LISTING_CACHE[cache_key] = _DirectoryListingCacheEntry(
        directory_mtime_ns=directory_stat.st_mtime_ns,
        expires_at_monotonic=now_monotonic + LISTING_CACHE_TTL_SECONDS,
        entries=tuple(children),
    )
    _prune_listing_cache()
    return children


def _list_response(
    *,
    dir_path: Path,
    root_key: str,
    root_value: str,
    offset: int,
    limit: int,
    root: Path | None = None,
) -> dict[str, object]:
    root_path = (root or ARTIFACTS_ROOT).resolve()
    children = _cached_list_children(dir_path, root=root_path)
    page = children[offset: offset + limit]
    next_offset = offset + len(page)
    return {
        root_key: root_value,
        "entries": page,
        "offset": offset,
        "limit": limit,
        "returned_entries": len(page),
        "total_entries": len(children),
        "has_more": next_offset < len(children),
        "next_offset": next_offset if next_offset < len(children) else None,
    }


def create_app(root: Path | None = None) -> FastAPI:
    root_path = ARTIFACTS_ROOT if root is None else Path(root).resolve()
    root_path.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="marketsim-artifacts", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "root": str(root_path)}

    @app.get("/")
    def index(
        offset: int = Query(0, ge=0),
        limit: int = Query(DEFAULT_LIST_LIMIT, ge=1, le=MAX_LIST_LIMIT),
    ) -> JSONResponse:
        return JSONResponse(
            _list_response(
                dir_path=root_path,
                root_key="root",
                root_value=str(root_path),
                offset=offset,
                limit=limit,
                root=root_path,
            )
        )

    @app.get("/list/{sub:path}")
    def list_sub(
        sub: str,
        offset: int = Query(0, ge=0),
        limit: int = Query(DEFAULT_LIST_LIMIT, ge=1, le=MAX_LIST_LIMIT),
    ) -> JSONResponse:
        p = _safe_join(sub, root=root_path)
        if not p.exists() or not p.is_dir():
            raise HTTPException(status_code=404, detail=f"not a directory: {sub}")
        return JSONResponse(
            _list_response(
                dir_path=p,
                root_key="path",
                root_value=sub,
                offset=offset,
                limit=limit,
                root=root_path,
            )
        )

    @app.get("/files/{sub:path}")
    def serve_file(sub: str) -> FileResponse:
        p = _safe_join(sub, root=root_path)
        if not p.exists() or not p.is_file():
            raise HTTPException(status_code=404, detail=f"not a file: {sub}")
        return FileResponse(p)

    return app


app = create_app()
