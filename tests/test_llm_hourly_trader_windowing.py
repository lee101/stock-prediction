from __future__ import annotations

import pandas as pd
import pytest

from llm_hourly_trader.windowing import parse_utc_timestamp, resolve_window_end


def test_parse_utc_timestamp_localizes_naive_values() -> None:
    ts = parse_utc_timestamp("2026-03-14 00:00:00")
    assert ts == pd.Timestamp("2026-03-14 00:00:00", tz="UTC")


def test_resolve_window_end_uses_earliest_common_cutoff() -> None:
    cutoffs = [
        pd.Timestamp("2026-03-14 00:00:00", tz="UTC"),
        pd.Timestamp("2026-03-13 22:00:00", tz="UTC"),
        pd.Timestamp("2026-03-14 03:00:00", tz="UTC"),
    ]
    assert resolve_window_end(cutoffs) == pd.Timestamp("2026-03-13 22:00:00", tz="UTC")


def test_resolve_window_end_accepts_explicit_earlier_timestamp() -> None:
    cutoffs = [
        pd.Timestamp("2026-03-14 00:00:00", tz="UTC"),
        pd.Timestamp("2026-03-13 22:00:00", tz="UTC"),
    ]
    resolved = resolve_window_end(cutoffs, "2026-03-13T12:00:00Z")
    assert resolved == pd.Timestamp("2026-03-13 12:00:00", tz="UTC")


def test_resolve_window_end_rejects_timestamp_after_common_cutoff() -> None:
    cutoffs = [
        pd.Timestamp("2026-03-14 00:00:00", tz="UTC"),
        pd.Timestamp("2026-03-13 22:00:00", tz="UTC"),
    ]
    with pytest.raises(ValueError, match="exceeds earliest forecast cutoff"):
        resolve_window_end(cutoffs, "2026-03-13T23:00:00Z")
