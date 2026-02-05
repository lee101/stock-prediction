from datetime import datetime, timezone

from src.exit_watch_utils import floor_bucket, intrahour_triggered, update_intrahour_baseline


def test_floor_bucket_minutes():
    now = datetime(2026, 2, 5, 12, 44, tzinfo=timezone.utc)
    bucket = floor_bucket(now, bucket_minutes=30)
    assert bucket.hour == 12
    assert bucket.minute == 30
    assert bucket.second == 0


def test_update_baseline_below_min_resets():
    now = datetime(2026, 2, 5, 12, 10, tzinfo=timezone.utc)
    baseline_price, baseline_hour, state = update_intrahour_baseline(
        current_price=80.0,
        now=now,
        baseline_price=100.0,
        baseline_hour=floor_bucket(now, bucket_minutes=60),
        min_price=81.0,
        bucket_minutes=60,
    )
    assert baseline_price is None
    assert baseline_hour is None
    assert state == "below_min"


def test_update_baseline_resets_on_bucket_change():
    now = datetime(2026, 2, 5, 12, 10, tzinfo=timezone.utc)
    baseline_price, baseline_hour, state = update_intrahour_baseline(
        current_price=100.0,
        now=now,
        baseline_price=None,
        baseline_hour=None,
        min_price=81.0,
        bucket_minutes=60,
    )
    assert state == "reset"
    assert baseline_price == 100.0

    later = datetime(2026, 2, 5, 13, 5, tzinfo=timezone.utc)
    baseline_price2, baseline_hour2, state2 = update_intrahour_baseline(
        current_price=101.0,
        now=later,
        baseline_price=baseline_price,
        baseline_hour=baseline_hour,
        min_price=81.0,
        bucket_minutes=60,
    )
    assert state2 == "reset"
    assert baseline_price2 == 101.0
    assert baseline_hour2.hour == 13


def test_intrahour_triggered_threshold():
    assert not intrahour_triggered(
        current_price=100.49, baseline_price=100.0, pct_above=0.005
    )
    assert intrahour_triggered(
        current_price=100.5, baseline_price=100.0, pct_above=0.005
    )
