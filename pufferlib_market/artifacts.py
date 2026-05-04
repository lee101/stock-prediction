"""Artifact write helpers for pufferlib-market evaluators."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch


def _replace_path(source: Path, target: Path) -> None:
    source.replace(target)


def write_json_atomic(path: Path, payload: dict[str, Any], *, sort_keys: bool = False) -> None:
    """Write JSON via same-directory temp file so artifacts are never partial."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=sort_keys) + "\n", encoding="utf-8")
        _replace_path(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def save_torch_atomic(payload: Any, path: Path) -> None:
    """Save a torch checkpoint without exposing a partially written target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, tmp_path)
        _replace_path(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)
