import argparse

from train_neural_daily_with_validation import (
    _epoch_run_name,
    _resolve_simulation_start_date,
    _resolve_training_symbols,
    _validation_metric_value,
)


def test_epoch_run_name_is_stable() -> None:
    assert _epoch_run_name("daily_run", 3) == "daily_run_epoch003"


def test_validation_metric_value_uses_requested_field() -> None:
    summary = {
        "pnl": 0.15,
        "sortino": 1.2,
        "goodness_score": 2.5,
    }

    assert _validation_metric_value(summary, "pnl") == 0.15
    assert _validation_metric_value(summary, "sortino") == 1.2
    assert _validation_metric_value(summary, "goodness") == 2.5


def test_resolve_simulation_start_date_prefers_recent_window() -> None:
    class _DummySimulator:
        def _available_dates(self):
            return [f"2024-01-0{i}" for i in range(1, 7)]

    assert _resolve_simulation_start_date(_DummySimulator(), 3, None) == "2024-01-04"
    assert _resolve_simulation_start_date(_DummySimulator(), 3, "2024-01-02") == "2024-01-02"


def test_resolve_training_symbols_supports_broad_mixed(tmp_path) -> None:
    for symbol in ("AAPL", "BTCUSD", "MSFT", "JPM", "XOM"):
        (tmp_path / f"{symbol}.csv").write_text("timestamp,open,high,low,close,volume\n", encoding="utf-8")

    args = argparse.Namespace(
        symbols=None,
        symbol_source="broad_mixed",
        data_root=str(tmp_path),
        max_symbols=4,
        per_group_cap=1,
    )

    symbols = _resolve_training_symbols(args)

    assert symbols[0] == "BTCUSD"
    assert len(symbols) == 4
    assert "AAPL" in symbols
    assert any(symbol in symbols for symbol in ("MSFT", "JPM", "XOM"))
