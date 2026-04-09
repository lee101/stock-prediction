"""Lightweight rotating-file logger for long-running training processes.

Usage::

    from fp4.log_utils import RotatingFileLogger

    logger = RotatingFileLogger("train.log")          # in cwd
    logger = RotatingFileLogger(Path("runs/run1/train.log"), max_bytes=50_000_000)

    logger.info("epoch %d loss %.4f", epoch, loss)
    logger.warning("lr schedule exhausted")

Wraps ``logging.handlers.RotatingFileHandler`` so log files never grow past
``max_bytes`` (default 50 MB).  Up to ``backup_count`` rotated copies are kept.
"""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Union


def RotatingFileLogger(
    path: Union[str, Path],
    *,
    max_bytes: int = 50_000_000,
    backup_count: int = 3,
    level: int = logging.INFO,
    fmt: str = "%(asctime)s %(levelname)s %(message)s",
    name: str | None = None,
) -> logging.Logger:
    """Return a stdlib Logger that writes to a size-capped rotating file.

    Parameters
    ----------
    path : str or Path
        Log file path.  Parent dirs are created automatically.
    max_bytes : int
        Maximum bytes per log file before rotation (default 50 MB).
    backup_count : int
        Number of rotated backups to keep (default 3).
    level : int
        Logging level (default INFO).
    fmt : str
        Log line format string.
    name : str or None
        Logger name.  Defaults to the file stem.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger_name = name or path.stem
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers on repeated calls.
    if not logger.handlers:
        handler = RotatingFileHandler(
            str(path),
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)

    return logger
