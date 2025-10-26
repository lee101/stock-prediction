from __future__ import annotations

import logging

import pytest

from falmarket.shared_logger import get_logger, log_timing


def _fresh_logger(name: str) -> logging.Logger:
    logger = get_logger(name)
    logger.setLevel(logging.INFO)
    return logger


def test_log_timing_success_logs_start_and_done(capsys: pytest.CaptureFixture[str]) -> None:
    logger = _fresh_logger("falmarket.tests.success")

    with log_timing(logger, "success-case"):
        pass

    captured = capsys.readouterr().out
    assert "START success-case" in captured
    assert "DONE  success-case" in captured


def test_log_timing_failure_logs_exception(capsys: pytest.CaptureFixture[str]) -> None:
    logger = _fresh_logger("falmarket.tests.failure")

    with pytest.raises(RuntimeError, match="boom"):
        with log_timing(logger, "failure-case"):
            raise RuntimeError("boom")

    captured = capsys.readouterr().out
    assert "START failure-case" in captured
    assert "FAIL  failure-case" in captured
