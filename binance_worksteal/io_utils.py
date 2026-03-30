"""Filesystem helpers for binance_worksteal."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path


def write_text_atomic(
    path: str | Path,
    body: str,
    *,
    encoding: str = "utf-8",
    mode: int | None = None,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preserved_mode: int | None = None
    if mode is None:
        try:
            preserved_mode = output_path.stat().st_mode & 0o777
        except FileNotFoundError:
            preserved_mode = None
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.tmp.",
        dir=str(output_path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        final_mode = mode if mode is not None else preserved_mode
        if final_mode is not None:
            os.chmod(tmp_path, final_mode)
        os.replace(tmp_path, output_path)
        return output_path
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
