import pandas as pd
import numpy as np
from typing import Callable, Dict, Union, Optional


Returns = Union[pd.Series, pd.DataFrame]


def constant_sizing(predicted_returns: Returns, factor: float = 1.0) -> Returns:
    """Return a constant position size for each input element."""
    if isinstance(predicted_returns, pd.DataFrame):
        return pd.DataFrame(
            factor, index=predicted_returns.index, columns=predicted_returns.columns
        )
    return pd.Series(factor, index=predicted_returns.index)


def expected_return_sizing(predicted_returns: Returns, risk_factor: float = 1.0) -> Returns:
    """Size positions proportional to the predicted return."""
    return predicted_returns.fillna(0.0) * risk_factor


def volatility_scaled_sizing(predicted_returns: Returns, window: int = 5) -> Returns:
    """Scale position size by the rolling standard deviation of predictions."""
    vol = predicted_returns.abs().rolling(window=window, min_periods=1).std()
    if isinstance(vol, pd.DataFrame):
        fill_value = vol.mean().replace(0.0, np.nan).fillna(1.0)
        vol = vol.replace(0.0, np.nan).fillna(fill_value)
    else:
        vol = vol.replace(0.0, np.nan).fillna(vol.mean() or 1.0)
    return predicted_returns / vol


def top_n_expected_return_sizing(
    predicted_returns: pd.DataFrame, n: int, leverage: float = 1.0
) -> pd.DataFrame:
    """Allocate leverage equally across the top ``n`` positive predictions."""
    if not isinstance(predicted_returns, pd.DataFrame):
        raise TypeError("predicted_returns must be a DataFrame for top-n sizing")

    positive = predicted_returns.clip(lower=0)
    ranks = positive.rank(axis=1, ascending=False, method="first")
    selected = ranks.le(n)
    counts = selected.sum(axis=1).replace(0, np.nan)
    sizes = selected.div(counts, axis=0).fillna(0.0) * leverage
    return sizes


def sharpe_ratio(pnl_series: pd.Series, periods_per_year: int = 252, risk_free_rate: float = 0.0) -> float:
    """Compute the annualised Sharpe ratio of a pnl series."""
    excess = pnl_series - risk_free_rate / periods_per_year
    denominator = pnl_series.std(ddof=0) or 1e-9
    return np.sqrt(periods_per_year) * excess.mean() / denominator


def backtest_position_sizing_series(
    actual_returns: Returns,
    predicted_returns: Returns,
    sizing_func: Callable[[Returns], Returns],
    trading_fee: float = 0.0,
) -> pd.Series:
    """Return a pnl series for the provided sizing strategy."""
    sizes = sizing_func(predicted_returns)
    if isinstance(actual_returns, pd.DataFrame):
        pnl_series = (sizes * actual_returns).sum(axis=1) - sizes.abs().sum(axis=1) * trading_fee
    else:
        pnl_series = sizes * actual_returns - sizes.abs() * trading_fee
    return pnl_series


def backtest_position_sizing(
    actual_returns: Returns,
    predicted_returns: Returns,
    sizing_func: Callable[[Returns], Returns],
    trading_fee: float = 0.0,
) -> float:
    """Calculate total pnl for a given sizing strategy."""
    pnl_series = backtest_position_sizing_series(
        actual_returns, predicted_returns, sizing_func, trading_fee
    )
    pnl = float(pnl_series.sum())
    return pnl


def optimize_position_sizing(
    actual_returns: Returns,
    predicted_returns: Returns,
    trading_fee: float = 0.0,
    risk_factor: float = 1.0,
    max_abs_size: Optional[float] = None,
    risk_free_rate: float = 0.0,
) -> Dict[str, float]:
    """Return pnl and Sharpe ratio for several sizing strategies."""
    strategies: Dict[str, Callable[[Returns], Returns]] = {
        "constant": lambda p: constant_sizing(p, factor=risk_factor),
        "expected_return": lambda p: expected_return_sizing(p, risk_factor=risk_factor),
        "vol_scaled": volatility_scaled_sizing,
    }
    results: Dict[str, float] = {}
    for name, fn in strategies.items():
        sizes = fn(predicted_returns)
        if max_abs_size is not None:
            sizes = sizes.clip(-max_abs_size, max_abs_size)
        pnl_series = backtest_position_sizing_series(
            actual_returns,
            predicted_returns,
            lambda _: sizes,
            trading_fee,
        )
        results[name] = pnl_series.sum()
        results[f"{name}_sharpe"] = sharpe_ratio(pnl_series, risk_free_rate=risk_free_rate)
    
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run position sizing optimizer")
    parser.add_argument("csv", help="CSV file with a Close column")
    parser.add_argument("--risk-free-rate", type=float, default=0.0, help="annual risk free rate")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    returns = df["Close"].pct_change().dropna()
    predicted_returns = returns.shift(1).fillna(0.0)

    results = optimize_position_sizing(returns, predicted_returns, risk_free_rate=args.risk_free_rate)
    for key, val in results.items():
        print(f"{key}: {val:.4f}")
