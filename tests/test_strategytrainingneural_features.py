import pandas as pd

from pathlib import Path

from strategytrainingneural.feature_builder import FeatureBuilder
from strategytrainingneural import current_symbols


def test_feature_builder_roundtrip():
    df = pd.DataFrame(
        {
            "rolling_sharpe": [0.5, 1.5, -0.2],
            "rolling_sortino": [0.4, 1.0, -0.1],
            "mode": ["normal", "probe", "blocked"],
            "gate_config": ["-", "Stock", "Global"],
        }
    )
    builder = FeatureBuilder(
        numeric_columns=["rolling_sharpe", "rolling_sortino"],
        categorical_columns=["mode", "gate_config"],
    )
    features = builder.fit_transform(df)
    assert features.shape[0] == len(df)
    assert features.shape[1] == len(builder.spec.feature_names)

    # Transforming a subset with the stored spec should keep column order.
    subset = df.iloc[[0]].copy()
    transformed = builder.transform(subset)
    assert transformed.shape[1] == features.shape[1]


def test_current_symbols_split(tmp_path: Path, monkeypatch):
    sample = tmp_path / "trade_stock_e2e.py"
    sample.write_text(
        """
def main():
    symbols = ["AAPL", "ETH-USD", "NVDA", "BTC-USD"]
""",
        encoding="utf-8",
    )
    symbols = current_symbols.load_current_symbols(sample)
    assert symbols == ["AAPL", "ETH-USD", "NVDA", "BTC-USD"]
    stocks, crypto = current_symbols.split_by_asset_class(symbols)
    assert stocks == ["AAPL", "NVDA"]
    assert crypto == ["ETH-USD", "BTC-USD"]
