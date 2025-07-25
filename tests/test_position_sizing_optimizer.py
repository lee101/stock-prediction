import pandas as pd
from src.position_sizing_optimizer import (
    constant_sizing,
    expected_return_sizing,
    volatility_scaled_sizing,
    backtest_position_sizing,
    optimize_position_sizing,
    top_n_expected_return_sizing,
    backtest_position_sizing_series,
)


def test_constant_sizing():
    preds = pd.Series([0.1, 0.2, 0.3])
    result = constant_sizing(preds, factor=2)
    assert (result == 2).all()


def test_constant_sizing_dataframe():
    preds = pd.DataFrame({"a": [0.1, 0.2], "b": [0.3, -0.1]})
    result = constant_sizing(preds, factor=1.5)
    assert result.shape == preds.shape
    assert (result == 1.5).all().all()


def test_optimize_position_sizing():
    actual = pd.Series([0.01, 0.02, -0.01, 0.03, -0.04])
    preds = pd.Series([0.5, 0.3, -0.1, 0.7, -0.2])
    results = optimize_position_sizing(actual, preds, trading_fee=0.001, risk_factor=1.0)
    # expected_return and vol_scaled should outperform constant
    assert results["expected_return"] > results["constant"]
    assert results["vol_scaled"] > results["constant"]
    # vol_scaled should also outperform expected_return for this data
    assert results["vol_scaled"] > results["expected_return"]


def test_risk_factor_and_clipping():
    actual = pd.Series([0.02, 0.01])
    preds = pd.Series([0.5, 0.6])
    results_low = optimize_position_sizing(actual, preds, risk_factor=0.5)
    results_high = optimize_position_sizing(actual, preds, risk_factor=2.0, max_abs_size=0.5)
    # Risk factor increases sizing but clipping limits the effect
    assert results_high["expected_return"] >= results_low["expected_return"]


def test_top_n_expected_return_sizing():
    preds = pd.DataFrame(
        {
            "asset1": [0.2, -0.1, 0.3],
            "asset2": [0.1, 0.4, -0.2],
            "asset3": [-0.05, 0.2, 0.1],
        }
    )
    sizes = top_n_expected_return_sizing(preds, n=2, leverage=1.0)
    # At each row no more than two non-zero positions
    assert (sizes.gt(0).sum(axis=1) <= 2).all()
    # Allocation per row sums to 1 when there is at least one positive prediction
    sums = sizes.sum(axis=1)
    assert sums.iloc[0] == 1.0
    assert sums.iloc[1] == 1.0


def test_backtest_position_sizing_series_dataframe():
    actual = pd.DataFrame({"a": [0.01, -0.02], "b": [0.03, 0.04]})
    predicted = actual.shift(1).fillna(0)
    sizes = constant_sizing(predicted, factor=1.0)
    pnl = backtest_position_sizing_series(actual, predicted, lambda _: sizes)
    assert isinstance(pnl, pd.Series)
    assert len(pnl) == 2


def test_optimize_position_sizing_sharpe():
    actual = pd.Series([0.01, 0.02, -0.01, 0.02])
    preds = actual.shift(1).fillna(0)
    results = optimize_position_sizing(actual, preds)
    assert "constant_sharpe" in results
    assert isinstance(results["constant_sharpe"], float)


def test_risk_free_rate_effect():
    actual = pd.Series([0.01, 0.02, -0.01, 0.03])
    preds = actual.shift(1).fillna(0)
    res_zero = optimize_position_sizing(actual, preds, risk_free_rate=0.0)
    res_high = optimize_position_sizing(actual, preds, risk_free_rate=0.1)
    assert res_high["constant_sharpe"] != res_zero["constant_sharpe"]
