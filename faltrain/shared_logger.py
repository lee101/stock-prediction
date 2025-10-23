#!/usr/bin/env python3
"""
Shared logger utility with timestamp formatting using stdlib only
"""

import logging
import sys
import time
from contextlib import contextmanager
from typing import Iterator


def get_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """
    Get a logger with consistent timestamp formatting.

    Args:
        name: Logger name (typically __name__)
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)

    # Avoid adding multiple handlers if already configured
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # Create formatter with timestamp
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Prevent propagation to avoid duplicate messages
    logger.propagate = False

    return logger


def setup_logging(level: int = logging.INFO) -> None:
    """
    Setup basic logging configuration with timestamps.

    Args:
        level: Logging level (default: INFO)
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        stream=sys.stdout
    )


@contextmanager
def log_timing(
    logger: logging.Logger,
    message: str,
    *,
    level: int = logging.INFO,
    error_level: int = logging.ERROR,
) -> Iterator[None]:
    """Log start, completion, and duration for a code section."""
    start = time.perf_counter()
    logger.log(level, "START %s", message)
    try:
        yield
    except Exception:
        elapsed = time.perf_counter() - start
        logger.log(error_level, "FAIL  %s (%.3fs)", message, elapsed, exc_info=True)
        raise
    else:
        elapsed = time.perf_counter() - start
        logger.log(level, "DONE  %s (%.3fs)", message, elapsed)
