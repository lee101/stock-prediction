import logging
import sys

import pytest

from faltrain.logger_utils import configure_stdout_logging, std_logger


@pytest.fixture
def restore_root_logger():
    root = logging.getLogger()
    original_level = root.level
    original_handlers = list(root.handlers)
    try:
        yield
    finally:
        root.handlers = original_handlers
        root.setLevel(original_level)


def _cleanup_logger(name: str) -> None:
    logger = logging.getLogger(name)
    logger.handlers = []
    logger.propagate = True
    logger.manager.loggerDict.pop(name, None)


def test_std_logger_attaches_stdout_once(restore_root_logger):
    name = "faltrain.test.std_logger"
    try:
        logger = std_logger(name, level="debug")
        stdout_handlers = [h for h in logger.handlers if getattr(h, "stream", None) is sys.stdout]
        assert stdout_handlers, "expected stdout handler to be attached"

        handler_count = len(logger.handlers)
        same_logger = std_logger(name)
        assert same_logger is logger
        assert len(same_logger.handlers) == handler_count
        assert logger.level == logging.DEBUG
    finally:
        _cleanup_logger(name)


def test_configure_stdout_logging_respects_overrides(monkeypatch, restore_root_logger):
    monkeypatch.setenv("FALTRAIN_LOG_LEVEL", "warning")
    root = configure_stdout_logging()
    assert root.level == logging.WARNING

    handler = next((h for h in root.handlers if getattr(h, "stream", None) is sys.stdout), None)
    assert handler is not None, "expected stdout handler on root logger"
    formatter = handler.formatter
    assert formatter is not None

    configure_stdout_logging(level="ERROR", fmt="%(message)s")
    assert logging.getLogger().level == logging.ERROR
    record = logging.LogRecord(
        name="faltrain.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=0,
        msg="hello",
        args=(),
        exc_info=None,
    )
    assert handler.format(record) == "hello"
