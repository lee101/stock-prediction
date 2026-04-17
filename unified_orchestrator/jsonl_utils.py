from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator

try:  # pragma: no cover - platform dependent
    import fcntl as _fcntl
except Exception:  # pragma: no cover - non-POSIX fallback
    _fcntl = None


def iter_jsonl_lines_reverse(path: Path, *, chunk_size: int = 65_536) -> Iterator[str]:
    """Yield non-empty JSONL lines from the end of a file backwards."""
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive")

    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell()
        remainder = b""

        while position > 0:
            read_size = min(int(chunk_size), position)
            position -= read_size
            handle.seek(position)
            chunk = handle.read(read_size)
            buffer = chunk + remainder
            lines = buffer.split(b"\n")
            remainder = lines[0]
            for raw_line in reversed(lines[1:]):
                line = raw_line.strip()
                if line:
                    yield line.decode("utf-8", errors="replace")

        final_line = remainder.strip()
        if final_line:
            yield final_line.decode("utf-8", errors="replace")


def append_jsonl_row(
    path: Path,
    payload: Any,
    *,
    sort_keys: bool = False,
    default: Any = None,
) -> None:
    """Append one JSONL row with a best-effort cross-process file lock."""

    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, sort_keys=sort_keys, default=default) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        if _fcntl is not None:
            _fcntl.flock(handle.fileno(), _fcntl.LOCK_EX)
        try:
            handle.write(line)
            handle.flush()
        finally:
            if _fcntl is not None:
                _fcntl.flock(handle.fileno(), _fcntl.LOCK_UN)
