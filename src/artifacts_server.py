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
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, TypedDict

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse


type ArtifactRootSource = Literal["default", "env", "argument"]


def _read_env_int(name: str, default: int, *, minimum: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None or not str(raw_value).strip():
        return default
    try:
        value = int(str(raw_value).strip())
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {raw_value!r}") from exc
    if value < minimum:
        raise RuntimeError(f"{name} must be >= {minimum}, got {value}")
    return value


def _read_env_float(name: str, default: float, *, minimum: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None or not str(raw_value).strip():
        return default
    try:
        value = float(str(raw_value).strip())
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a float, got {raw_value!r}") from exc
    if value < minimum:
        raise RuntimeError(f"{name} must be >= {minimum}, got {value}")
    return value


def _resolve_root() -> Path:
    env = os.environ.get("ARTIFACTS_ROOT")
    if env:
        root = Path(env).resolve()
    else:
        # Default: <repo>/models/artifacts
        root = (Path(__file__).resolve().parent.parent / "models" / "artifacts").resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


class ArtifactEntry(TypedDict):
    name: str
    is_dir: bool
    size: int | None
    mtime: float
    url: str | None


@dataclass(frozen=True)
class _DirectoryListingCacheEntry:
    directory_mtime_ns: int
    expires_at_monotonic: float
    entries: tuple[ArtifactEntry, ...]


@dataclass(frozen=True)
class ArtifactServerSettings:
    root: Path
    root_source: ArtifactRootSource
    default_list_limit: int
    max_list_limit: int
    listing_cache_ttl_seconds: float
    listing_cache_max_entries: int


def _build_default_settings() -> ArtifactServerSettings:
    default_list_limit = _read_env_int("ARTIFACTS_DEFAULT_LIST_LIMIT", 200, minimum=1)
    return ArtifactServerSettings(
        root=_resolve_root(),
        root_source="env" if os.environ.get("ARTIFACTS_ROOT") else "default",
        default_list_limit=default_list_limit,
        max_list_limit=_read_env_int(
            "ARTIFACTS_MAX_LIST_LIMIT",
            1000,
            minimum=default_list_limit,
        ),
        listing_cache_ttl_seconds=_read_env_float(
            "ARTIFACTS_LISTING_CACHE_TTL_SECONDS",
            1.0,
            minimum=0.0,
        ),
        listing_cache_max_entries=_read_env_int(
            "ARTIFACTS_LISTING_CACHE_MAX_ENTRIES",
            256,
            minimum=1,
        ),
    )


_DEFAULT_SETTINGS = _build_default_settings()
ARTIFACTS_ROOT_SOURCE = _DEFAULT_SETTINGS.root_source
ARTIFACTS_ROOT: Path = _DEFAULT_SETTINGS.root
DEFAULT_LIST_LIMIT = _DEFAULT_SETTINGS.default_list_limit
MAX_LIST_LIMIT = _DEFAULT_SETTINGS.max_list_limit
LISTING_CACHE_TTL_SECONDS = _DEFAULT_SETTINGS.listing_cache_ttl_seconds
LISTING_CACHE_MAX_ENTRIES = _DEFAULT_SETTINGS.listing_cache_max_entries
LISTING_CACHE_INFLIGHT_WAIT_SECONDS = 30.0


@dataclass
class _ListingCacheState:
    entries: dict[tuple[str, str], _DirectoryListingCacheEntry]
    inflight: dict[tuple[str, str], threading.Event]
    lock: threading.Lock
    hits: int = 0
    misses: int = 0
    stores: int = 0
    waits: int = 0

    def clear(self) -> None:
        with self.lock:
            self.entries.clear()
            self.inflight.clear()
            self.hits = 0
            self.misses = 0
            self.stores = 0
            self.waits = 0

    def snapshot(self, *, settings: ArtifactServerSettings) -> ListingCacheStats:
        with self.lock:
            return {
                "entries": len(self.entries),
                "hits": self.hits,
                "misses": self.misses,
                "stores": self.stores,
                "waits": self.waits,
                "ttl_seconds": settings.listing_cache_ttl_seconds,
                "max_entries": settings.listing_cache_max_entries,
            }

    def prune_locked(self, *, settings: ArtifactServerSettings) -> None:
        while len(self.entries) > settings.listing_cache_max_entries:
            oldest_key = min(
                self.entries,
                key=lambda key: self.entries[key].expires_at_monotonic,
            )
            self.entries.pop(oldest_key, None)


def _new_listing_cache_state() -> _ListingCacheState:
    return _ListingCacheState(
        entries={},
        inflight={},
        lock=threading.Lock(),
    )


class ListingCacheStats(TypedDict):
    entries: int
    hits: int
    misses: int
    stores: int
    waits: int
    ttl_seconds: float
    max_entries: int


class ArtifactServerConfig(TypedDict):
    root_source: ArtifactRootSource
    default_list_limit: int
    max_list_limit: int
    listing_cache_ttl_seconds: float
    listing_cache_max_entries: int
    listing_cache_inflight_wait_seconds: float


class ArtifactHealthPayload(TypedDict):
    ok: bool
    root: str
    root_exists: bool
    config: ArtifactServerConfig
    cache: ListingCacheStats


class ArtifactListPayload(TypedDict, total=False):
    root: str
    path: str
    entries: list[ArtifactEntry]
    offset: int
    limit: int
    returned_entries: int
    total_entries: int
    has_more: bool
    next_offset: int | None


def _has_hidden_path_segment(path_like: str | Path) -> bool:
    path = Path(path_like)
    return any(part.startswith(".") for part in path.parts if part not in ("", ".", ".."))


def _safe_join(sub: str, *, root: Path | None = None) -> Path:
    root_path = (root or ARTIFACTS_ROOT).resolve()
    if _has_hidden_path_segment(sub):
        raise HTTPException(status_code=404, detail="file not found")
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
) -> ArtifactEntry:
    return {
        "name": path.name,
        "is_dir": is_dir,
        "size": stat_result.st_size if not is_dir else None,
        "mtime": stat_result.st_mtime,
        "url": None if is_dir else f"/files/{path.relative_to(root).as_posix()}",
    }


def _list_children(dir_path: Path, *, root: Path | None = None) -> list[ArtifactEntry]:
    root_path = (root or ARTIFACTS_ROOT).resolve()
    children: list[ArtifactEntry] = []
    with os.scandir(dir_path) as iterator:
        entries = list(iterator)
    for child in entries:
        if _has_hidden_path_segment(child.name):
            continue
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


_DEFAULT_CACHE_STATE = _new_listing_cache_state()


def _clear_listing_cache(*, cache_state: _ListingCacheState = _DEFAULT_CACHE_STATE) -> None:
    cache_state.clear()


def _listing_cache_stats(
    *,
    settings: ArtifactServerSettings = _DEFAULT_SETTINGS,
    cache_state: _ListingCacheState = _DEFAULT_CACHE_STATE,
) -> ListingCacheStats:
    return cache_state.snapshot(settings=settings)


def _server_config_payload(*, settings: ArtifactServerSettings) -> ArtifactServerConfig:
    return {
        "root_source": settings.root_source,
        "default_list_limit": settings.default_list_limit,
        "max_list_limit": settings.max_list_limit,
        "listing_cache_ttl_seconds": settings.listing_cache_ttl_seconds,
        "listing_cache_max_entries": settings.listing_cache_max_entries,
        "listing_cache_inflight_wait_seconds": LISTING_CACHE_INFLIGHT_WAIT_SECONDS,
    }


def _cached_list_children(
    dir_path: Path,
    *,
    root: Path | None = None,
    settings: ArtifactServerSettings = _DEFAULT_SETTINGS,
    cache_state: _ListingCacheState = _DEFAULT_CACHE_STATE,
) -> list[ArtifactEntry]:
    root_path = (root or settings.root).resolve()
    resolved_dir_path = dir_path.resolve()
    directory_stat = resolved_dir_path.stat()
    cache_key = (str(root_path), str(resolved_dir_path))
    now_monotonic = time.monotonic()
    producer_event: threading.Event | None = None
    with cache_state.lock:
        cached = cache_state.entries.get(cache_key)
        if (
            cached is not None
            and cached.directory_mtime_ns == directory_stat.st_mtime_ns
            and cached.expires_at_monotonic >= now_monotonic
        ):
            cache_state.hits += 1
            return list(cached.entries)

        inflight_event = cache_state.inflight.get(cache_key)
        if inflight_event is None:
            producer_event = threading.Event()
            cache_state.inflight[cache_key] = producer_event
            cache_state.misses += 1
        else:
            cache_state.waits += 1

    if producer_event is None:
        if not inflight_event.wait(timeout=LISTING_CACHE_INFLIGHT_WAIT_SECONDS):
            raise RuntimeError(
                "artifact listing cache wait timed out for "
                f"{resolved_dir_path}"
            )
        with cache_state.lock:
            cached = cache_state.entries.get(cache_key)
            if (
                cached is not None
                and cached.directory_mtime_ns == directory_stat.st_mtime_ns
                and cached.expires_at_monotonic >= time.monotonic()
            ):
                cache_state.hits += 1
                return list(cached.entries)
        return _cached_list_children(
            resolved_dir_path,
            root=root_path,
            settings=settings,
            cache_state=cache_state,
        )

    try:
        children = _list_children(resolved_dir_path, root=root_path)
        with cache_state.lock:
            cache_state.entries[cache_key] = _DirectoryListingCacheEntry(
                directory_mtime_ns=directory_stat.st_mtime_ns,
                expires_at_monotonic=now_monotonic + settings.listing_cache_ttl_seconds,
                entries=tuple(children),
            )
            cache_state.stores += 1
            cache_state.prune_locked(settings=settings)
        return children
    finally:
        with cache_state.lock:
            event = cache_state.inflight.pop(cache_key, None)
            if event is not None:
                event.set()


def _list_response(
    *,
    dir_path: Path,
    root_key: str,
    root_value: str,
    offset: int,
    limit: int,
    root: Path | None = None,
    settings: ArtifactServerSettings = _DEFAULT_SETTINGS,
    cache_state: _ListingCacheState = _DEFAULT_CACHE_STATE,
) -> ArtifactListPayload:
    root_path = (root or settings.root).resolve()
    children = _cached_list_children(
        dir_path,
        root=root_path,
        settings=settings,
        cache_state=cache_state,
    )
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


def create_app(
    root: Path | None = None,
    *,
    settings: ArtifactServerSettings = _DEFAULT_SETTINGS,
) -> FastAPI:
    effective_settings = settings if root is None else replace(
        settings,
        root=Path(root).resolve(),
        root_source="argument",
    )
    root_path = effective_settings.root
    cache_state = _new_listing_cache_state()
    root_path.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="marketsim-artifacts", version="0.1.0")

    @app.get("/health")
    def health() -> ArtifactHealthPayload:
        return {
            "ok": True,
            "root": str(root_path),
            "root_exists": root_path.exists(),
            "config": _server_config_payload(settings=effective_settings),
            "cache": _listing_cache_stats(
                settings=effective_settings,
                cache_state=cache_state,
            ),
        }

    @app.get("/")
    def index(
        offset: int = Query(0, ge=0),
        limit: int = Query(
            effective_settings.default_list_limit,
            ge=1,
            le=effective_settings.max_list_limit,
        ),
    ) -> JSONResponse:
        return JSONResponse(
            _list_response(
                dir_path=root_path,
                root_key="root",
                root_value=str(root_path),
                offset=offset,
                limit=limit,
                root=root_path,
                settings=effective_settings,
                cache_state=cache_state,
            )
        )

    @app.get("/list/{sub:path}")
    def list_sub(
        sub: str,
        offset: int = Query(0, ge=0),
        limit: int = Query(
            effective_settings.default_list_limit,
            ge=1,
            le=effective_settings.max_list_limit,
        ),
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
                settings=effective_settings,
                cache_state=cache_state,
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
