from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.rolling_risk_metrics import RollingRiskMetrics


def _make_frame(start: datetime, closes: list[float], *, spacing: timedelta = timedelta(minutes=1)) -> pd.DataFrame:
    timestamps = [start + spacing * idx for idx in range(len(closes))]
    return pd.DataFrame({"timestamp": timestamps, "close": closes})


def test_update_and_volatility_estimation():
    tracker = RollingRiskMetrics(window_duration=timedelta(minutes=30), min_history=3)
    start = datetime(2025, 1, 1, 12, tzinfo=timezone.utc)
    closes_initial = [100.0, 101.5, 101.0, 102.4]
    frame_initial = _make_frame(start, closes_initial)

    added = tracker.update_from_price_frame("AAPL", frame_initial)
    assert added == len(closes_initial) - 1

    stats = tracker.get_symbol_volatility_stats("AAPL")
    assert stats is not None
    expected_returns = frame_initial["close"].pct_change().dropna()
    assert stats.sample_count == len(expected_returns)
    assert stats.realized_volatility == pytest.approx(expected_returns.std(ddof=1))

    # Feed additional data chunk with a discontinuous first bar to ensure bridging logic works
    follow_start = start + timedelta(minutes=len(closes_initial))
    closes_follow_up = [102.9, 103.5, 103.0, 104.2]
    frame_follow_up = _make_frame(follow_start, closes_follow_up)
    added_follow_up = tracker.update_from_price_frame("AAPL", frame_follow_up)

    # Bridging adds one extra return (gap between historical and first new bar)
    assert added_follow_up == len(closes_follow_up)

    combined_closes = closes_initial + closes_follow_up
    combined_returns = pd.Series(combined_closes).pct_change().dropna()
    stats = tracker.get_symbol_volatility_stats("AAPL")
    assert stats is not None
    assert stats.sample_count == len(combined_returns)
    assert stats.realized_volatility == pytest.approx(combined_returns.std(ddof=1))


def test_window_trimming_limits_history():
    tracker = RollingRiskMetrics(window_duration=timedelta(minutes=2), min_history=2)
    start = datetime(2025, 2, 1, 9, tzinfo=timezone.utc)
    frame_old = _make_frame(start, [100.0, 101.0, 102.0])
    tracker.update_from_price_frame("MSFT", frame_old)
    assert len(tracker.get_symbol_returns("MSFT")) == len(frame_old) - 1

    # Advance far enough that the rolling window should discard earlier returns
    new_start = start + timedelta(minutes=10)
    frame_recent = _make_frame(new_start, [103.0, 104.0, 105.0, 104.5])
    tracker.update_from_price_frame("MSFT", frame_recent)

    series = tracker.get_symbol_returns("MSFT")
    assert not series.empty
    assert series.index.min() >= new_start
    # With a tight window the first bridged return rolls off immediately
    assert len(series) == len(frame_recent) - 1


def test_correlation_matrix_reacts_to_new_data():
    tracker = RollingRiskMetrics(window_duration=timedelta(minutes=3), min_history=2)
    start = datetime(2025, 3, 1, 14, tzinfo=timezone.utc)

    frame_aaa = _make_frame(start, [100.0, 101.0, 102.0, 103.0])
    frame_bbb = _make_frame(start, [200.0, 202.0, 204.0, 206.0])

    tracker.update_from_price_frame("AAA", frame_aaa)
    tracker.update_from_price_frame("BBB", frame_bbb)

    matrix = tracker.get_correlation_matrix(min_overlap=3)
    assert "AAA" in matrix.columns and "BBB" in matrix.columns
    assert matrix.loc["AAA", "BBB"] == pytest.approx(1.0, rel=1e-5)

    snapshot = tracker.build_snapshot()
    assert set(snapshot.symbol_stats) == {"AAA", "BBB"}
    assert snapshot.correlation_matrix.loc["AAA", "BBB"] == pytest.approx(matrix.loc["AAA", "BBB"])

    # Feed new data well outside the rolling window to force a recalculated correlation
    future_start = start + timedelta(minutes=10)
    frame_aaa_future = _make_frame(future_start, [104.0, 105.5, 107.0])
    frame_bbb_future = _make_frame(future_start, [205.0, 203.0, 201.5])

    tracker.update_from_price_frame("AAA", frame_aaa_future)
    tracker.update_from_price_frame("BBB", frame_bbb_future)

    updated_matrix = tracker.get_correlation_matrix(min_overlap=2)
    assert updated_matrix.loc["AAA", "BBB"] <= -0.85
