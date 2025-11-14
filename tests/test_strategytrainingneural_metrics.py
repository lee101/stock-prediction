import pandas as pd
import torch

from strategytrainingneural.metrics import aggregate_daily_pnl, combine_sortino_and_return
from strategytrainingneural.trade_windows import load_trade_window_metrics


def test_aggregate_daily_pnl():
    weights = torch.tensor([1.0, 0.5, 0.0, -0.5], dtype=torch.float32)
    returns = torch.tensor([0.02, -0.01, 0.015, 0.03], dtype=torch.float32)
    day_index = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    daily = aggregate_daily_pnl(weights, returns, day_index)
    assert daily.shape[0] == 2
    expected_day0 = 1.0 * 0.02 + 0.5 * -0.01
    expected_day1 = 0.0 * 0.015 + (-0.5) * 0.03
    assert torch.allclose(daily[0], torch.tensor(expected_day0))
    assert torch.allclose(daily[1], torch.tensor(expected_day1))


def test_combine_sortino_and_return_positive_signal():
    daily = torch.tensor([0.01, 0.02, -0.005, 0.015], dtype=torch.float32)
    result = combine_sortino_and_return(daily, trading_days=4, return_weight=0.1)
    assert result.score > 0
    assert result.sortino > 0
    assert result.annual_return > 0


def test_trade_window_loader(tmp_path):
    parquet_path = tmp_path / "trades.parquet"
    df = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "AAPL", "ETH-USD"],
            "strategy": ["simple_strategy", "simple_strategy", "simple_strategy", "simple_strategy"],
            "exit_timestamp": [
                "2025-01-01T00:00:00Z",
                "2025-01-02T00:00:00Z",
                "2025-01-03T00:00:00Z",
                "2025-01-01T00:00:00Z",
            ],
            "pnl": [10.0, -5.0, 8.0, 12.0],
            "pnl_pct": [0.01, -0.005, 0.008, 0.02],
        }
    )
    df.to_parquet(parquet_path)
    dataset = load_trade_window_metrics([str(parquet_path)], asset_class="stock", window_days=2, min_trades=2)
    assert not dataset.empty
    assert set(dataset["day_class"].unique()) == {"stock"}
