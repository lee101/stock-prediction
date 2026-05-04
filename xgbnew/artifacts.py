"""Artifact write helpers for XGB research/eval scripts."""

from __future__ import annotations

import csv
import json
import os
import pickle
from pathlib import Path
from typing import Any


def _replace_path(source: Path, target: Path) -> None:
    source.replace(target)


def _tmp_path_for(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.tmp")


def write_text_atomic(path: Path, text: str) -> None:
    """Write text via same-directory temp file and atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _tmp_path_for(path)
    try:
        tmp_path.write_text(text, encoding="utf-8")
        _replace_path(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def write_json_atomic(
    path: Path,
    payload: dict[str, Any],
    *,
    default: Any = None,
    sort_keys: bool = False,
) -> None:
    write_text_atomic(path, json.dumps(payload, indent=2, default=default, sort_keys=sort_keys) + "\n")


def write_pickle_atomic(path: Path, payload: Any) -> None:
    """Write a pickle without exposing partial model artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _tmp_path_for(path)
    try:
        with tmp_path.open("wb") as handle:
            pickle.dump(payload, handle)
        _replace_path(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def write_dataframe_csv_atomic(path: Path, frame: Any) -> None:
    """Write a DataFrame CSV without exposing partial output files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _tmp_path_for(path)
    try:
        frame.to_csv(tmp_path, index=False)
        _replace_path(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def write_dict_rows_csv_atomic(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    fieldnames: list[str],
) -> None:
    """Write dict rows to CSV without exposing partial output files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _tmp_path_for(path)
    try:
        with tmp_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        _replace_path(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def save_model_atomic(model: Any, path: Path) -> None:
    """Save a model through its ``save(path)`` method without exposing partial output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _tmp_path_for(path)
    try:
        model.save(tmp_path)
        if not tmp_path.exists():
            raise FileNotFoundError(f"model save did not create temp artifact: {tmp_path}")
        _replace_path(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)
