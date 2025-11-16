import pytest

from src.maxdiff_exit_tracker import ExitTargetTracker


def test_tracker_defaults_to_full_position():
    tracker = ExitTargetTracker(None)
    qty, remaining, done = tracker.plan_order(10.0)
    assert qty == pytest.approx(10.0)
    assert remaining == pytest.approx(0.0)
    assert done is False


def test_tracker_limits_to_requested_quantity():
    tracker = ExitTargetTracker(5.0)
    qty, remaining, done = tracker.plan_order(10.0)
    assert qty == pytest.approx(5.0)
    assert remaining == pytest.approx(5.0)
    assert done is False

    qty, remaining, done = tracker.plan_order(5.0)
    assert qty == pytest.approx(0.0)
    assert remaining == pytest.approx(0.0)
    assert done is True


def test_tracker_clamps_to_available_position_and_detects_zero():
    tracker = ExitTargetTracker(100.0)
    qty, remaining, done = tracker.plan_order(40.0)
    assert qty == pytest.approx(40.0)
    assert remaining == pytest.approx(40.0)
    assert done is False

    qty, remaining, done = tracker.plan_order(0.0)
    assert qty == pytest.approx(0.0)
    assert remaining == pytest.approx(0.0)
    assert done is True
