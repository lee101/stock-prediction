"""Shared logging helpers for faltrain modules.

Every faltrain logger should emit to stdout so local runs mirror production
behaviour. This module centralises the setup so repeated calls remain idempotent
and honour environment overrides.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional, Union

LogLevel = Union[int, str]

_STDOUT_HANDLER_MARKER = "_faltrain_stdout_handler"

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _resolve_level(level: Optional[LogLevel]) -> int:
    """Interpret ``level`` or environment overrides into a logging level."""
    candidate: Optional[Union[int, str]] = level
    if candidate is None:
        env_value = os.getenv("FALTRAIN_LOG_LEVEL")
        if env_value:
            candidate = env_value
    if candidate is None:
        return logging.INFO
    if isinstance(candidate, int):
        return candidate
    text = str(candidate).strip()
    if not text:
        return logging.INFO
    if text.isdigit():
        return int(text)
    upper = text.upper()
    resolved = getattr(logging, upper, None)
    if isinstance(resolved, int):
        return resolved
    numeric = logging.getLevelName(upper)
    if isinstance(numeric, int):
        return numeric
    return logging.INFO


def _formatter(fmt: Optional[str], datefmt: Optional[str]) -> logging.Formatter:
    format_str = fmt or os.getenv("FALTRAIN_LOG_FORMAT") or _DEFAULT_FORMAT
    date_format = datefmt or os.getenv("FALTRAIN_LOG_DATEFMT") or _DEFAULT_DATEFMT
    return logging.Formatter(format_str, date_format)


def _ensure_stdout_handler(
    logger: logging.Logger,
    *,
    fmt: Optional[str],
    datefmt: Optional[str],
) -> logging.Handler:
    """Attach a stdout stream handler if one is not already present."""
    for handler in logger.handlers:
        marker = getattr(handler, _STDOUT_HANDLER_MARKER, False)
        stream = getattr(handler, "stream", None)
        if marker or stream is sys.stdout:
            handler.setFormatter(_formatter(fmt, datefmt))
            setattr(handler, _STDOUT_HANDLER_MARKER, True)
            return handler

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_formatter(fmt, datefmt))
    handler.setLevel(logging.NOTSET)
    setattr(handler, _STDOUT_HANDLER_MARKER, True)
    logger.addHandler(handler)
    return handler


def configure_stdout_logging(
    level: Optional[LogLevel] = None,
    *,
    fmt: Optional[str] = None,
    datefmt: Optional[str] = None,
) -> logging.Logger:
    """Ensure the root logger emits to stdout with the configured level."""
    logger = logging.getLogger()
    _ensure_stdout_handler(logger, fmt=fmt, datefmt=datefmt)
    logger.setLevel(_resolve_level(level))
    return logger


def std_logger(
    name: str,
    level: Optional[LogLevel] = None,
    *,
    fmt: Optional[str] = None,
    datefmt: Optional[str] = None,
    propagate: bool = False,
) -> logging.Logger:
    """Return a logger configured to stream to stdout."""
    logger = logging.getLogger(name)
    had_stdout_handler = any(
        getattr(handler, _STDOUT_HANDLER_MARKER, False) or getattr(handler, "stream", None) is sys.stdout
        for handler in logger.handlers
    )
    _ensure_stdout_handler(logger, fmt=fmt, datefmt=datefmt)
    env_override = os.getenv("FALTRAIN_LOG_LEVEL")
    if level is not None or env_override or not had_stdout_handler:
        logger.setLevel(_resolve_level(level))
    logger.propagate = propagate
    return logger
