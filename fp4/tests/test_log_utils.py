"""Tests for fp4.log_utils.RotatingFileLogger."""
from __future__ import annotations

import tempfile
from pathlib import Path

from fp4.log_utils import RotatingFileLogger


def test_rotating_logger_creates_file():
    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "subdir" / "test.log"
        logger = RotatingFileLogger(log_path)
        logger.info("hello world")
        # Flush handlers
        for h in logger.handlers:
            h.flush()
        assert log_path.exists()
        content = log_path.read_text()
        assert "hello world" in content


def test_rotating_logger_respects_max_bytes():
    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "small.log"
        # Tiny max to force rotation
        logger = RotatingFileLogger(log_path, max_bytes=200, backup_count=2)
        for i in range(50):
            logger.info("line %d " + "x" * 20, i)
        for h in logger.handlers:
            h.flush()
        # Should have rotated: backup files exist
        backups = list(Path(td).glob("small.log.*"))
        assert len(backups) > 0, "Expected at least one backup file after rotation"
        # Main file should be under max_bytes (approximately)
        assert log_path.stat().st_size < 400


def test_no_duplicate_handlers():
    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "dup.log"
        logger1 = RotatingFileLogger(log_path, name="dedup_test")
        logger2 = RotatingFileLogger(log_path, name="dedup_test")
        assert logger1 is logger2
        assert len(logger1.handlers) == 1
