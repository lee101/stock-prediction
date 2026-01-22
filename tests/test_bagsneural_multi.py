from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from bagsneural.train_multi import build_multi_token_dataset


def _make_df(rows: int) -> pd.DataFrame:
    timestamps = [datetime(2026, 1, 1) + timedelta(minutes=10 * i) for i in range(rows)]
    base = [1.0 + 0.01 * i for i in range(rows)]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": base,
            "high": [v * 1.01 for v in base],
            "low": [v * 0.99 for v in base],
            "close": [v * 1.005 for v in base],
        }
    )


def test_build_multi_token_dataset():
    df_a = _make_df(20)
    df_b = _make_df(25)
    train, val = build_multi_token_dataset(
        {"A": df_a, "B": df_b},
        context_bars=4,
        horizon=2,
        cost_bps=5.0,
        min_return=0.0,
        size_scale=0.01,
        val_split=0.2,
    )

    assert train[0].shape[0] > 0
    assert val[0].shape[0] > 0
    assert train[0].shape[1] == 4 * 3
