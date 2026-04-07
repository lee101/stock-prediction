"""Tiny FastAPI app exposing models/artifacts/ over HTTP.

Run:
    uvicorn src.artifacts_server:app --host 0.0.0.0 --port 8765

Endpoints:
    GET  /            -> JSON tree of artifacts (lazy, one level)
    GET  /list/<sub>  -> JSON listing for a subdirectory
    GET  /files/<...> -> raw file (mounted via StaticFiles)
    GET  /health      -> {"ok": true}

Designed to be read-only and safe to expose: paths are normalized and
constrained to ARTIFACTS_ROOT.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles


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


def _safe_join(sub: str) -> Path:
    p = (ARTIFACTS_ROOT / sub).resolve()
    if ARTIFACTS_ROOT not in p.parents and p != ARTIFACTS_ROOT:
        raise HTTPException(status_code=400, detail="path outside artifacts root")
    return p


def _entry(p: Path) -> dict:
    st = p.stat()
    return {
        "name": p.name,
        "is_dir": p.is_dir(),
        "size": st.st_size if p.is_file() else None,
        "mtime": st.st_mtime,
        "url": None if p.is_dir() else f"/files/{p.relative_to(ARTIFACTS_ROOT).as_posix()}",
    }


def create_app(root: Path | None = None) -> FastAPI:
    global ARTIFACTS_ROOT
    if root is not None:
        ARTIFACTS_ROOT = Path(root).resolve()
        ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="marketsim-artifacts", version="0.1.0")
    app.mount("/files", StaticFiles(directory=str(ARTIFACTS_ROOT)), name="files")

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "root": str(ARTIFACTS_ROOT)}

    @app.get("/")
    def index() -> JSONResponse:
        items = sorted(ARTIFACTS_ROOT.iterdir(), key=lambda p: p.name) if ARTIFACTS_ROOT.exists() else []
        return JSONResponse({"root": str(ARTIFACTS_ROOT), "entries": [_entry(p) for p in items]})

    @app.get("/list/{sub:path}")
    def list_sub(sub: str) -> JSONResponse:
        p = _safe_join(sub)
        if not p.exists() or not p.is_dir():
            raise HTTPException(status_code=404, detail=f"not a directory: {sub}")
        items = sorted(p.iterdir(), key=lambda x: x.name)
        return JSONResponse({"path": sub, "entries": [_entry(x) for x in items]})

    return app


app = create_app()
