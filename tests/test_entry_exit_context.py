import torch

from src.optimization_utils import (
    _evaluate_entry_exit_profit,
    _prepare_entry_exit_context,
)
from loss_utils import calculate_trading_profit_torch_with_entry_buysell


def _sample_tensors(n: int = 64):
    torch.manual_seed(123)
    close_actual = torch.randn(n)
    positions = torch.randn(n)
    high_actual = close_actual + torch.abs(torch.randn(n)) * 0.01
    high_pred = torch.randn(n) * 0.01
    low_actual = close_actual - torch.abs(torch.randn(n)) * 0.01
    low_pred = torch.randn(n) * 0.01
    return close_actual, positions, high_actual, high_pred, low_actual, low_pred


def _reference_profit(args, high_mult, low_mult, close_at_eod):
    close_actual, positions, high_actual, high_pred, low_actual, low_pred = args
    return calculate_trading_profit_torch_with_entry_buysell(
        None,
        None,
        close_actual,
        positions,
        high_actual,
        high_pred + high_mult,
        low_actual,
        low_pred + low_mult,
        close_at_eod=close_at_eod,
    ).item()


def test_context_matches_reference_close_and_intraday():
    args = _sample_tensors(32)
    ctx = _prepare_entry_exit_context(*args)

    for close_at_eod in (False, True):
        for high_mult, low_mult in [(0.0, 0.0), (0.01, -0.015), (-0.02, 0.005)]:
            optimized = _evaluate_entry_exit_profit(
                ctx,
                high_mult=high_mult,
                low_mult=low_mult,
                close_at_eod=close_at_eod,
                trading_fee=None,
            ).sum().item()
            reference = _reference_profit(args, high_mult, low_mult, close_at_eod)
            assert torch.isclose(
                torch.tensor(optimized),
                torch.tensor(reference),
                atol=1e-6,
            ), f"Mismatch close_at_eod={close_at_eod} high={high_mult} low={low_mult}"
