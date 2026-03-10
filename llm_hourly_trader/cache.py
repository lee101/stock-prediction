"""Disk cache for Gemini API responses to avoid repeating calls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent / "api_cache"


def _cache_key(model: str, prompt: str) -> str:
    h = hashlib.sha256(f"{model}::{prompt}".encode()).hexdigest()[:16]
    return h


def get_cached(model: str, prompt: str) -> dict | None:
    key = _cache_key(model, prompt)
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def set_cached(model: str, prompt: str, response: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(model, prompt)
    path = CACHE_DIR / f"{key}.json"
    path.write_text(json.dumps(response, default=str))
