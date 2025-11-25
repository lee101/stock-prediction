import pandas as pd

from alpaca_data_wrapper import _merge_and_dedup


def test_merge_and_dedup_overwrites_duplicates():
    idx = pd.to_datetime(["2025-01-01 00:00Z", "2025-01-01 01:00Z"])
    existing = pd.DataFrame({"close": [1.0, 2.0]}, index=idx)

    new_idx = pd.to_datetime(["2025-01-01 01:00Z", "2025-01-01 02:00Z"])
    new = pd.DataFrame({"close": [2.5, 3.0]}, index=new_idx)

    merged = _merge_and_dedup(existing, new)

    assert len(merged) == 3
    assert merged.loc[pd.Timestamp("2025-01-01 01:00Z"), "close"] == 2.5
    assert merged.iloc[-1]["close"] == 3.0
