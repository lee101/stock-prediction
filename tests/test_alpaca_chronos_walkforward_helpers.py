from __future__ import annotations

import pytest

try:
    from experiments.alpaca_pufferlib_hourly_crypto_chronos_walkforward_20260211.run_walkforward import (
        _align_fold_symbols,
        _rank_key,
        _resolve_single_fee,
        _summarize_fold_best,
    )
except (ImportError, ModuleNotFoundError):
    pytest.skip("Required module experiments.alpaca_pufferlib_hourly_crypto_chronos_walkforward_20260211 not available", allow_module_level=True)


def test_align_fold_symbols_preserves_train_order() -> None:
    train = ["BTCUSD", "ETHUSD", "SOLUSD", "LINKUSD"]
    eval_syms = ["SOLUSD", "BTCUSD", "LINKUSD"]
    assert _align_fold_symbols(train, eval_syms) == ["BTCUSD", "SOLUSD", "LINKUSD"]


def test_rank_key_prefers_annualized_then_sortino() -> None:
    a = {"annualized_return": 0.10, "sortino": 1.0}
    b = {"annualized_return": 0.05, "sortino": 5.0}
    c = {"annualized_return": 0.10, "sortino": 2.0}
    ranked = sorted([a, b, c], key=_rank_key, reverse=True)
    assert ranked == [c, a, b]


def test_summarize_fold_best_aggregates_metrics() -> None:
    rows = [
        {"metrics": {"total_return": 0.10, "annualized_return": 0.50, "sortino": 1.2, "max_drawdown": 0.10}},
        {"metrics": {"total_return": -0.20, "annualized_return": -0.40, "sortino": -0.5, "max_drawdown": 0.30}},
    ]
    summary = _summarize_fold_best(rows)
    assert summary["mean_return"] == pytest.approx(-0.05)
    assert summary["mean_annualized_return"] == pytest.approx(0.05)
    assert summary["positive_folds"] == 1.0
    assert summary["num_folds"] == 2.0


def test_resolve_single_fee_prefers_explicit() -> None:
    assert _resolve_single_fee(["BTCUSD", "ETHUSD"], explicit_fee=0.0012) == pytest.approx(0.0012)


def test_resolve_single_fee_raises_on_mixed_symbol_fees(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_get_fee(symbol: str) -> float:
        return {"BTCUSD": 0.0015, "AAPL": 0.0005}[symbol]

    monkeypatch.setattr("src.fees.get_fee_for_symbol", _fake_get_fee)
    with pytest.raises(ValueError, match="single fee_rate"):
        _resolve_single_fee(["BTCUSD", "AAPL"], explicit_fee=None)
