from datetime import datetime, timezone
from functools import wraps
from unittest.mock import patch


def freeze_time(time_str: str):
    """A lightweight freeze_time decorator compatible with the tests."""
    frozen = datetime.fromisoformat(time_str)
    if frozen.tzinfo is None:
        frozen = frozen.replace(tzinfo=timezone.utc)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with patch('src.date_utils.datetime') as mock_datetime:
                mock_datetime.now.return_value = frozen
                mock_datetime.side_effect = lambda *a, **kw: datetime(*a, **kw)
                return func(*args, **kwargs)

        return wrapper

    return decorator
