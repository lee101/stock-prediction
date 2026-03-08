from __future__ import annotations

from binanceleveragesui.feature_horizon_meta_search import (
    _horizon_label,
    _parse_horizon_sets,
    _parse_windows,
)


def test_parse_horizon_sets_deduplicates_and_sorts() -> None:
    parsed = _parse_horizon_sets("12,1;1,6;1,6;")

    assert parsed == [(1, 12), (1, 6)]


def test_parse_windows_skips_invalid_duplicates() -> None:
    parsed = _parse_windows("7,1,7,30")

    assert parsed == [7, 1, 30]


def test_horizon_label_formats_consistently() -> None:
    assert _horizon_label((1, 6, 12)) == "h1_h6_h12"
