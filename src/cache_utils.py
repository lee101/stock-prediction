"""Utilities for managing cache directories used by external ML libraries."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


_HF_ENV_VARS: Sequence[str] = ("HF_HOME", "TRANSFORMERS_CACHE", "HUGGINGFACE_HUB_CACHE")
_CACHE_SENTINEL = ".cache-write-test"


def _expand_path(path_like: str) -> Path:
    """Expand user and environment components and return a Path."""
    return Path(path_like).expanduser()


def _candidate_paths(extra_candidates: Optional[Iterable[Path]] = None) -> List[Path]:
    """Return the ordered list of cache candidates to probe for writability."""
    candidates: List[Path] = []

    for env_key in _HF_ENV_VARS:
        env_value = os.getenv(env_key)
        if not env_value:
            continue
        expanded = _expand_path(env_value)
        if expanded not in candidates:
            candidates.append(expanded)

    repo_root = Path(__file__).resolve().parent.parent
    defaults = [
        repo_root / "cache" / "huggingface",
        Path.cwd() / ".hf_cache",
        Path.home() / ".cache" / "huggingface",
        repo_root / "compiled_models" / "huggingface",
    ]
    if extra_candidates:
        defaults = list(extra_candidates) + defaults

    for path in defaults:
        if path not in candidates:
            candidates.append(path)

    return candidates


def _is_writable(path: Path) -> bool:
    """Return True if ``path`` can be created and written to."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception:
        return False

    sentinel = path / _CACHE_SENTINEL
    try:
        with sentinel.open("w", encoding="utf-8") as handle:
            handle.write("ok")
    except Exception:
        return False
    finally:
        try:
            sentinel.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            # If cleanup fails, we leave the sentinel in place; not critical.
            pass
    return True


def ensure_huggingface_cache_dir(
    *,
    logger: Optional[logging.Logger] = None,
    extra_candidates: Optional[Iterable[Path]] = None,
) -> Path:
    """
    Ensure that a writable Hugging Face cache directory is available.

    The function attempts the following, in order:

        1. Use any directories referenced by the standard HF cache environment vars.
        2. Fall back to repository-local cache directories.
        3. Fall back to the user's home cache directory.

    Once a writable directory is found, all relevant environment variables are updated
    to reference it.  A ``RuntimeError`` is raised if no candidate directories are
    writable.
    """
    selected: Optional[Path] = None

    for candidate in _candidate_paths(extra_candidates=extra_candidates):
        if _is_writable(candidate):
            selected = candidate
            break

    if selected is None:
        message = (
            "Unable to locate a writable Hugging Face cache directory. "
            "Set HF_HOME or TRANSFORMERS_CACHE to a writable path."
        )
        if logger:
            logger.error(message)
        raise RuntimeError(message)

    resolved = selected.resolve()
    for env_key in _HF_ENV_VARS:
        os.environ[env_key] = str(resolved)

    if logger:
        logger.info(f"Using Hugging Face cache directory: {resolved}")
    return resolved


def find_hf_snapshot_dir(
    repo_id: str,
    *,
    logger: Optional[logging.Logger] = None,
    extra_candidates: Optional[Iterable[Path]] = None,
) -> Optional[Path]:
    """
    Locate the most recent cached snapshot directory for ``repo_id``.

    Args:
        repo_id: Hugging Face repository identifier, e.g. ``"amazon/chronos-2"``.
        logger: Optional logger for debug output.
        extra_candidates: Optional iterable of additional cache roots to probe
            before falling back to the default candidate search order.

    Returns:
        Path to the newest snapshot directory containing a ``config.json`` file,
        or ``None`` if no cached snapshot exists.
    """

    repo_fragment = repo_id.replace("/", "--")
    candidates = _candidate_paths(extra_candidates=extra_candidates)

    for cache_root in candidates:
        hub_dir = cache_root / "hub" / f"models--{repo_fragment}" / "snapshots"
        try:
            snapshot_dirs = sorted(
                (path for path in hub_dir.iterdir() if path.is_dir()),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        except FileNotFoundError:
            continue
        except PermissionError:
            if logger:
                logger.debug("Skipping Hugging Face cache %s (permission denied)", hub_dir)
            continue
        except OSError as exc:
            if logger:
                logger.debug("Skipping Hugging Face cache %s (%s)", hub_dir, exc)
            continue

        for snapshot_dir in snapshot_dirs:
            if (snapshot_dir / "config.json").exists():
                if logger:
                    logger.debug("Found cached snapshot for %s at %s", repo_id, snapshot_dir)
                return snapshot_dir

    if logger:
        logger.debug("No cached snapshot found for %s across %d candidates", repo_id, len(candidates))
    return None
