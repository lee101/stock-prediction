from contextlib import contextmanager
from datetime import datetime

from loguru import logger


@contextmanager
def log_time(prefix=""):
    """log the time taken in a with block
    prefix: the prefix text to show
    """
    start_time = datetime.now()
    logger.info("{}: start: {}".format(prefix, start_time))

    try:
        yield
    finally:
        end_time = datetime.now()
        logger.info("{}: end: {}".format(prefix, end_time))
        logger.info("{}: elapsed: {}".format(prefix, end_time - start_time))


import time


def debounce(seconds, key_func=None):
    def decorator(func):
        last_called = {}

        def debounced(*args, **kwargs):
            key = key_func(*args, **kwargs) if key_func else None
            elapsed = time.time() - last_called.get(key, 0.0)
            if elapsed >= seconds:
                last_called[key] = time.time()
                return func(*args, **kwargs)

        return debounced

    return decorator
