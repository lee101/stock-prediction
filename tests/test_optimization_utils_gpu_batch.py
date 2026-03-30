from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from loss_utils import (
    calculate_profit_torch_with_entry_buysell_profit_values,
    calculate_trading_profit_torch_with_entry_buysell,
)
import src.optimization_utils_gpu_batch as gpu_batch


def _sample_entry_exit_inputs():
    close_actuals = [
        torch.tensor([0.02, -0.01, 0.015, 0.005], dtype=torch.float32),
        torch.tensor([-0.015, 0.01, -0.005], dtype=torch.float32),
        torch.tensor([0.01, 0.0, -0.02, 0.03, -0.01], dtype=torch.float32),
    ]
    positions_list = [
        torch.tensor([1.0, -1.0, 1.0, 1.0], dtype=torch.float32),
        torch.tensor([-1.0, -1.0, 1.0], dtype=torch.float32),
        torch.tensor([1.0, 1.0, -1.0, -1.0, 1.0], dtype=torch.float32),
    ]
    high_actuals = [values + 0.02 for values in close_actuals]
    low_actuals = [values - 0.015 for values in close_actuals]
    high_preds = [values + 0.01 for values in close_actuals]
    low_preds = [values - 0.01 for values in close_actuals]
    return close_actuals, positions_list, high_actuals, high_preds, low_actuals, low_preds


def _sample_always_on_inputs():
    close_actuals, _positions, high_actuals, high_preds, low_actuals, low_preds = _sample_entry_exit_inputs()
    buy_indicators = [
        torch.tensor([1.0, 1.0, 0.0, 1.0], dtype=torch.float32),
        torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32),
        torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0], dtype=torch.float32),
    ]
    sell_indicators = [
        torch.tensor([0.0, -1.0, -1.0, 0.0], dtype=torch.float32),
        torch.tensor([-1.0, -1.0, 0.0], dtype=torch.float32),
        torch.tensor([-1.0, 0.0, 0.0, -1.0, -1.0], dtype=torch.float32),
    ]
    return (
        close_actuals,
        buy_indicators,
        sell_indicators,
        high_actuals,
        high_preds,
        low_actuals,
        low_preds,
    )


def _sample_always_on_batch(device: str = "cpu") -> dict:
    (
        close_actuals,
        buy_indicators,
        sell_indicators,
        high_actuals,
        high_preds,
        low_actuals,
        low_preds,
    ) = _sample_always_on_inputs()
    max_len = max(len(values) for values in close_actuals)
    batch = {
        "close": torch.zeros(len(close_actuals), max_len, device=device),
        "buy": torch.zeros(len(close_actuals), max_len, device=device),
        "sell": torch.zeros(len(close_actuals), max_len, device=device),
        "high_actual": torch.zeros(len(close_actuals), max_len, device=device),
        "high_pred": torch.zeros(len(close_actuals), max_len, device=device),
        "low_actual": torch.zeros(len(close_actuals), max_len, device=device),
        "low_pred": torch.zeros(len(close_actuals), max_len, device=device),
        "mask": torch.zeros(len(close_actuals), max_len, dtype=torch.bool, device=device),
    }
    for idx, close_actual in enumerate(close_actuals):
        length = len(close_actual)
        batch["close"][idx, :length] = close_actual.to(device)
        batch["buy"][idx, :length] = buy_indicators[idx].to(device)
        batch["sell"][idx, :length] = sell_indicators[idx].to(device)
        batch["high_actual"][idx, :length] = high_actuals[idx].to(device)
        batch["high_pred"][idx, :length] = high_preds[idx].to(device)
        batch["low_actual"][idx, :length] = low_actuals[idx].to(device)
        batch["low_pred"][idx, :length] = low_preds[idx].to(device)
        batch["mask"][idx, :length] = True
    return batch


def test_calculate_batch_entry_exit_profit_matches_sequential() -> None:
    close_actuals, positions_list, high_actuals, high_preds, low_actuals, low_preds = _sample_entry_exit_inputs()
    batch = gpu_batch._prepare_batched_data(
        close_actuals,
        positions_list,
        high_actuals,
        high_preds,
        low_actuals,
        low_preds,
        max_len=max(len(values) for values in close_actuals),
        device="cpu",
    )

    actual = gpu_batch._calculate_batch_entry_exit_profit(
        batch,
        high_mult=0.004,
        low_mult=-0.003,
        close_at_eod=False,
        trading_fee=0.001,
    )

    expected = []
    for close_actual, positions, high_actual, high_pred, low_actual, low_pred in zip(
        close_actuals,
        positions_list,
        high_actuals,
        high_preds,
        low_actuals,
        low_preds,
        strict=True,
    ):
        expected.append(
            calculate_trading_profit_torch_with_entry_buysell(
                None,
                None,
                close_actual,
                positions,
                high_actual,
                high_pred + 0.004,
                low_actual,
                low_pred - 0.003,
                close_at_eod=False,
                trading_fee=0.001,
            )
        )

    assert torch.allclose(actual, torch.stack(expected), atol=1e-6)


def test_calculate_batch_entry_exit_profit_matches_sequential_close_at_eod() -> None:
    close_actuals, positions_list, high_actuals, high_preds, low_actuals, low_preds = _sample_entry_exit_inputs()
    batch = gpu_batch._prepare_batched_data(
        close_actuals,
        positions_list,
        high_actuals,
        high_preds,
        low_actuals,
        low_preds,
        max_len=max(len(values) for values in close_actuals),
        device="cpu",
    )

    actual = gpu_batch._calculate_batch_entry_exit_profit(
        batch,
        high_mult=0.001,
        low_mult=-0.004,
        close_at_eod=True,
        trading_fee=0.0009,
    )

    expected = []
    for close_actual, positions, high_actual, high_pred, low_actual, low_pred in zip(
        close_actuals,
        positions_list,
        high_actuals,
        high_preds,
        low_actuals,
        low_preds,
        strict=True,
    ):
        expected.append(
            calculate_trading_profit_torch_with_entry_buysell(
                None,
                None,
                close_actual,
                positions,
                high_actual,
                high_pred + 0.001,
                low_actual,
                low_pred - 0.004,
                close_at_eod=True,
                trading_fee=0.0009,
            )
        )

    assert torch.allclose(actual, torch.stack(expected), atol=1e-6)


def test_calculate_batch_always_on_profit_matches_sequential() -> None:
    batch = _sample_always_on_batch()

    actual = gpu_batch._calculate_batch_always_on_profit(
        batch,
        high_mult=0.002,
        low_mult=-0.001,
        close_at_eod=True,
        trading_fee=0.0005,
        is_crypto=False,
    )

    expected = []
    for idx in range(batch["close"].shape[0]):
        length = int(batch["mask"][idx].sum().item())
        buy_returns = calculate_profit_torch_with_entry_buysell_profit_values(
            batch["close"][idx, :length],
            batch["high_actual"][idx, :length],
            batch["high_pred"][idx, :length] + 0.002,
            batch["low_actual"][idx, :length],
            batch["low_pred"][idx, :length] - 0.001,
            batch["buy"][idx, :length],
            close_at_eod=True,
            trading_fee=0.0005,
        )
        sell_returns = calculate_profit_torch_with_entry_buysell_profit_values(
            batch["close"][idx, :length],
            batch["high_actual"][idx, :length],
            batch["high_pred"][idx, :length] + 0.002,
            batch["low_actual"][idx, :length],
            batch["low_pred"][idx, :length] - 0.001,
            batch["sell"][idx, :length],
            close_at_eod=True,
            trading_fee=0.0005,
        )
        expected.append(buy_returns.sum() + sell_returns.sum())

    assert torch.allclose(actual, torch.stack(expected), atol=1e-6)


def test_calculate_batch_always_on_profit_crypto_matches_sequential() -> None:
    batch = _sample_always_on_batch()

    actual = gpu_batch._calculate_batch_always_on_profit(
        batch,
        high_mult=0.003,
        low_mult=-0.002,
        close_at_eod=False,
        trading_fee=0.0004,
        is_crypto=True,
    )

    expected = []
    for idx in range(batch["close"].shape[0]):
        length = int(batch["mask"][idx].sum().item())
        buy_returns = calculate_profit_torch_with_entry_buysell_profit_values(
            batch["close"][idx, :length],
            batch["high_actual"][idx, :length],
            batch["high_pred"][idx, :length] + 0.003,
            batch["low_actual"][idx, :length],
            batch["low_pred"][idx, :length] - 0.002,
            batch["buy"][idx, :length],
            close_at_eod=False,
            trading_fee=0.0004,
        )
        expected.append(buy_returns.sum())

    assert torch.allclose(actual, torch.stack(expected), atol=1e-6)


def test_optimize_batch_always_on_returns_chunked_results(monkeypatch) -> None:
    (
        close_actuals,
        buy_indicators,
        sell_indicators,
        high_actuals,
        high_preds,
        low_actuals,
        low_preds,
    ) = _sample_always_on_inputs()

    def _fake_direct(objective, bounds, maxfun):
        return SimpleNamespace(x=np.asarray([0.002, -0.001], dtype=np.float64), fun=0.0)

    monkeypatch.setattr(gpu_batch, "direct", _fake_direct)

    results = gpu_batch.optimize_batch_always_on(
        close_actuals,
        buy_indicators,
        sell_indicators,
        high_actuals,
        high_preds,
        low_actuals,
        low_preds,
        batch_size=2,
        device="cpu",
        close_at_eod=True,
        trading_fee=0.0005,
        is_crypto=False,
    )

    assert len(results) == 3
    assert [result[:2] for result in results] == [(0.002, -0.001)] * 3

    for result, close_actual, buy_indicator, sell_indicator, high_actual, high_pred, low_actual, low_pred in zip(
        results,
        close_actuals,
        buy_indicators,
        sell_indicators,
        high_actuals,
        high_preds,
        low_actuals,
        low_preds,
        strict=True,
    ):
        buy_returns = calculate_profit_torch_with_entry_buysell_profit_values(
            close_actual,
            high_actual,
            high_pred + 0.002,
            low_actual,
            low_pred - 0.001,
            buy_indicator,
            close_at_eod=True,
            trading_fee=0.0005,
        )
        sell_returns = calculate_profit_torch_with_entry_buysell_profit_values(
            close_actual,
            high_actual,
            high_pred + 0.002,
            low_actual,
            low_pred - 0.001,
            sell_indicator,
            close_at_eod=True,
            trading_fee=0.0005,
        )
        expected = (buy_returns.sum() + sell_returns.sum()).item()
        assert abs(result[2] - expected) < 1e-6


def test_optimize_batch_always_on_crypto_ignores_sell_indicators(monkeypatch) -> None:
    (
        close_actuals,
        buy_indicators,
        sell_indicators,
        high_actuals,
        high_preds,
        low_actuals,
        low_preds,
    ) = _sample_always_on_inputs()
    sell_indicators = [torch.full_like(indicator, -1.0) for indicator in sell_indicators]

    def _fake_direct(objective, bounds, maxfun):
        return SimpleNamespace(x=np.asarray([0.003, -0.002], dtype=np.float64), fun=0.0)

    monkeypatch.setattr(gpu_batch, "direct", _fake_direct)

    results = gpu_batch.optimize_batch_always_on(
        close_actuals,
        buy_indicators,
        sell_indicators,
        high_actuals,
        high_preds,
        low_actuals,
        low_preds,
        batch_size=2,
        device="cpu",
        close_at_eod=False,
        trading_fee=0.0004,
        is_crypto=True,
    )

    assert len(results) == 3
    assert [result[:2] for result in results] == [(0.003, -0.002)] * 3

    for result, close_actual, buy_indicator, high_actual, high_pred, low_actual, low_pred in zip(
        results,
        close_actuals,
        buy_indicators,
        high_actuals,
        high_preds,
        low_actuals,
        low_preds,
        strict=True,
    ):
        buy_returns = calculate_profit_torch_with_entry_buysell_profit_values(
            close_actual,
            high_actual,
            high_pred + 0.003,
            low_actual,
            low_pred - 0.002,
            buy_indicator,
            close_at_eod=False,
            trading_fee=0.0004,
        )
        assert abs(result[2] - buy_returns.sum().item()) < 1e-6


def test_optimize_batch_entry_exit_returns_chunked_results(monkeypatch) -> None:
    close_actuals, positions_list, high_actuals, high_preds, low_actuals, low_preds = _sample_entry_exit_inputs()

    def _fake_direct(objective, bounds, maxfun):
        return SimpleNamespace(x=np.asarray([0.003, -0.002], dtype=np.float64), fun=0.0)

    monkeypatch.setattr(gpu_batch, "direct", _fake_direct)

    results = gpu_batch.optimize_batch_entry_exit(
        close_actuals,
        positions_list,
        high_actuals,
        high_preds,
        low_actuals,
        low_preds,
        batch_size=2,
        device="cpu",
        trading_fee=0.0007,
    )

    assert len(results) == 3
    assert [result[:2] for result in results] == [(0.003, -0.002)] * 3

    for result, close_actual, positions, high_actual, high_pred, low_actual, low_pred in zip(
        results,
        close_actuals,
        positions_list,
        high_actuals,
        high_preds,
        low_actuals,
        low_preds,
        strict=True,
    ):
        expected = calculate_trading_profit_torch_with_entry_buysell(
            None,
            None,
            close_actual,
            positions,
            high_actual,
            high_pred + 0.003,
            low_actual,
            low_pred - 0.002,
            trading_fee=0.0007,
        ).item()
        assert abs(result[2] - expected) < 1e-6
