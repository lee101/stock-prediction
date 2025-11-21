import numpy as np
import pandas as pd

from neuraldailytraining.symbol_groups import (
    build_correlation_groups,
    group_ids_from_clusters,
)


def _frame(prices):
    dates = pd.date_range("2024-01-01", periods=len(prices), freq="D", tz="UTC")
    return pd.DataFrame({"date": dates, "close": prices})


def test_correlation_groups_cluster_similar_equities():
    frames = {
        "AAA": _frame(np.linspace(100, 110, 50)),
        "BBB": _frame(np.linspace(50, 55, 50)),  # Strongly correlated trend
        "CCC": _frame(np.concatenate([np.linspace(20, 30, 25), np.linspace(30, 10, 25)])),
    }

    clusters = build_correlation_groups(
        frames,
        min_corr=0.95,
        max_group_size=4,
        window_days=120,
        min_overlap=10,
        split_crypto=False,
    )
    mapping = group_ids_from_clusters(clusters)

    assert mapping["AAA"] == mapping["BBB"]
    assert mapping["CCC"] != mapping["AAA"]


def test_correlation_groups_separate_crypto_and_equity():
    frames = {
        "AAPL": _frame(np.linspace(100, 105, 40)),
        "MSFT": _frame(np.linspace(200, 210, 40)),
        "BTC-USD": _frame(np.linspace(30000, 32000, 40)),
    }

    clusters = build_correlation_groups(
        frames,
        min_corr=0.5,
        max_group_size=4,
        window_days=90,
        min_overlap=5,
        split_crypto=True,
    )

    equity_groups = {name for name in clusters if name.startswith("corr_equity")}
    crypto_groups = {name for name in clusters if name.startswith("corr_crypto")}

    assert equity_groups  # at least one equity cluster
    assert crypto_groups  # crypto isolated into its own cluster set
    assert any("BTC-USD" in symbols for grp, symbols in clusters.items() if grp in crypto_groups)
