import time
import unittest.mock
from datetime import datetime, timedelta
from freezegun import freeze_time
from tests.test_data_utils import get_time


def test_foo():
    # last_three_days = [
    #     datetime.now() - timedelta(days=3),
    #     datetime.now() - timedelta(days=2),
    #     datetime.now() - timedelta(days=1),
    # ]
    last_three_days = [
        "2012-01-14"
    ]
    with freeze_time(last_three_days[0], ignore=['transformers']):
        # assert get_time() == 12345
        assert datetime.now() == datetime(2012, 1, 14)

