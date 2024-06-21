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
