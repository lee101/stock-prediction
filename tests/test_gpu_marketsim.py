"""Tests for GPU-native market simulator."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trainingefficiency.gpu_marketsim import (
    gpu_simulate_binary,
    compute_sortino_gpu,
    compute_max_drawdown_gpu,
    compute_total_return_gpu,
    compute_num_trades_gpu,
    compute_metrics_gpu,
    metrics_to_scalars,
)
from differentiable_loss_utils import simulate_hourly_trades_binary


def _is_cuda_resource_pressure_error(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


def _skip_for_cuda_resource_pressure(exc: BaseException) -> None:
    if _is_cuda_resource_pressure_error(exc):
        pytest.skip(f"GPU marketsim test skipped under shared-GPU resource pressure: {exc}")


def _allocate_or_skip(factory, *args, **kwargs):
    try:
        return factory(*args, **kwargs)
    except Exception as exc:
        _skip_for_cuda_resource_pressure(exc)
        raise


def _to_device_or_skip(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    try:
        return tensor.to(device)
    except Exception as exc:
        _skip_for_cuda_resource_pressure(exc)
        raise


def _make_market_data(steps=100, seed=42, device="cpu"):
    torch.manual_seed(seed)
    close_start = 100.0
    returns = _allocate_or_skip(torch.randn, steps, device=device) * 0.01
    closes = close_start * torch.cumprod(1 + returns, dim=0)
    highs = closes * (1 + _allocate_or_skip(torch.rand, steps, device=device) * 0.02)
    lows = closes * (1 - _allocate_or_skip(torch.rand, steps, device=device) * 0.02)
    opens = closes * (1 + (_allocate_or_skip(torch.rand, steps, device=device) - 0.5) * 0.01)
    return highs, lows, closes, opens


def _make_actions(closes, device=None):
    device = closes.device if device is None else device
    steps = closes.shape[-1]
    buy_prices = closes * (1 - 0.005 * _allocate_or_skip(torch.rand, steps, device=device))
    sell_prices = closes * (1 + 0.005 * _allocate_or_skip(torch.rand, steps, device=device))
    buy_amounts = _allocate_or_skip(torch.rand, steps, device=device) * 0.5
    sell_amounts = _allocate_or_skip(torch.rand, steps, device=device) * 0.5
    return buy_prices, sell_prices, buy_amounts, sell_amounts


class TestGpuSimulateVsReference:
    """Compare gpu_simulate_binary against simulate_hourly_trades_binary."""

    def test_parity_basic(self):
        torch.manual_seed(42)
        highs, lows, closes, opens = _make_market_data(80)
        buy_p, sell_p, buy_a, sell_a = _make_actions(closes)

        ref = simulate_hourly_trades_binary(
            highs=highs, lows=lows, closes=closes, opens=opens,
            buy_prices=buy_p, sell_prices=sell_p,
            trade_intensity=buy_a,
            buy_trade_intensity=buy_a,
            sell_trade_intensity=sell_a,
            maker_fee=0.001, max_leverage=1.0,
        )

        gpu = gpu_simulate_binary(
            highs=highs, lows=lows, closes=closes, opens=opens,
            buy_prices=buy_p, sell_prices=sell_p,
            buy_amounts=buy_a, sell_amounts=sell_a,
            maker_fee=0.001, max_leverage=1.0,
        )

        torch.testing.assert_close(ref.returns, gpu["returns"], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(ref.portfolio_values, gpu["portfolio_values"], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(ref.inventory_path, gpu["inventory_path"], atol=1e-6, rtol=1e-5)

    def test_parity_leverage(self):
        torch.manual_seed(99)
        highs, lows, closes, opens = _make_market_data(60)
        buy_p, sell_p, buy_a, sell_a = _make_actions(closes)

        ref = simulate_hourly_trades_binary(
            highs=highs, lows=lows, closes=closes,
            buy_prices=buy_p, sell_prices=sell_p,
            trade_intensity=buy_a,
            buy_trade_intensity=buy_a,
            sell_trade_intensity=sell_a,
            maker_fee=0.001, max_leverage=2.0,
        )

        gpu = gpu_simulate_binary(
            highs=highs, lows=lows, closes=closes,
            buy_prices=buy_p, sell_prices=sell_p,
            buy_amounts=buy_a, sell_amounts=sell_a,
            maker_fee=0.001, max_leverage=2.0,
        )

        torch.testing.assert_close(ref.returns, gpu["returns"], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(ref.portfolio_values, gpu["portfolio_values"], atol=1e-5, rtol=1e-5)

    def test_parity_margin(self):
        torch.manual_seed(77)
        highs, lows, closes, opens = _make_market_data(50)
        buy_p, sell_p, buy_a, sell_a = _make_actions(closes)

        ref = simulate_hourly_trades_binary(
            highs=highs, lows=lows, closes=closes,
            buy_prices=buy_p, sell_prices=sell_p,
            trade_intensity=buy_a,
            buy_trade_intensity=buy_a,
            sell_trade_intensity=sell_a,
            maker_fee=0.001, max_leverage=2.0,
            margin_annual_rate=0.0625,
        )

        gpu = gpu_simulate_binary(
            highs=highs, lows=lows, closes=closes,
            buy_prices=buy_p, sell_prices=sell_p,
            buy_amounts=buy_a, sell_amounts=sell_a,
            maker_fee=0.001, max_leverage=2.0,
            margin_annual_rate=0.0625,
        )

        torch.testing.assert_close(ref.returns, gpu["returns"], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(ref.portfolio_values, gpu["portfolio_values"], atol=1e-5, rtol=1e-5)

    def test_parity_fill_buffer(self):
        torch.manual_seed(11)
        highs, lows, closes, opens = _make_market_data(40)
        buy_p, sell_p, buy_a, sell_a = _make_actions(closes)

        ref = simulate_hourly_trades_binary(
            highs=highs, lows=lows, closes=closes,
            buy_prices=buy_p, sell_prices=sell_p,
            trade_intensity=buy_a,
            buy_trade_intensity=buy_a,
            sell_trade_intensity=sell_a,
            maker_fee=0.001, fill_buffer_pct=0.0005,
        )

        gpu = gpu_simulate_binary(
            highs=highs, lows=lows, closes=closes,
            buy_prices=buy_p, sell_prices=sell_p,
            buy_amounts=buy_a, sell_amounts=sell_a,
            maker_fee=0.001, fill_buffer_pct=0.0005,
        )

        torch.testing.assert_close(ref.returns, gpu["returns"], atol=1e-6, rtol=1e-5)

    def test_parity_short(self):
        torch.manual_seed(33)
        highs, lows, closes, opens = _make_market_data(50)
        buy_p, sell_p, buy_a, sell_a = _make_actions(closes)

        ref = simulate_hourly_trades_binary(
            highs=highs, lows=lows, closes=closes,
            buy_prices=buy_p, sell_prices=sell_p,
            trade_intensity=buy_a,
            buy_trade_intensity=buy_a,
            sell_trade_intensity=sell_a,
            maker_fee=0.001, max_leverage=2.0,
            can_short=True,
        )

        gpu = gpu_simulate_binary(
            highs=highs, lows=lows, closes=closes,
            buy_prices=buy_p, sell_prices=sell_p,
            buy_amounts=buy_a, sell_amounts=sell_a,
            maker_fee=0.001, max_leverage=2.0,
            can_short=True,
        )

        torch.testing.assert_close(ref.returns, gpu["returns"], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(ref.portfolio_values, gpu["portfolio_values"], atol=1e-5, rtol=1e-5)

    def test_parity_no_long(self):
        torch.manual_seed(55)
        highs, lows, closes, opens = _make_market_data(40)
        buy_p, sell_p, buy_a, sell_a = _make_actions(closes)

        ref = simulate_hourly_trades_binary(
            highs=highs, lows=lows, closes=closes,
            buy_prices=buy_p, sell_prices=sell_p,
            trade_intensity=buy_a,
            buy_trade_intensity=buy_a,
            sell_trade_intensity=sell_a,
            maker_fee=0.001, can_long=False, can_short=True, max_leverage=2.0,
        )

        gpu = gpu_simulate_binary(
            highs=highs, lows=lows, closes=closes,
            buy_prices=buy_p, sell_prices=sell_p,
            buy_amounts=buy_a, sell_amounts=sell_a,
            maker_fee=0.001, can_long=False, can_short=True, max_leverage=2.0,
        )

        torch.testing.assert_close(ref.returns, gpu["returns"], atol=1e-6, rtol=1e-5)

    def test_parity_decision_lag(self):
        torch.manual_seed(88)
        highs, lows, closes, opens = _make_market_data(60)
        buy_p, sell_p, buy_a, sell_a = _make_actions(closes)

        ref = simulate_hourly_trades_binary(
            highs=highs, lows=lows, closes=closes, opens=opens,
            buy_prices=buy_p, sell_prices=sell_p,
            trade_intensity=buy_a,
            buy_trade_intensity=buy_a,
            sell_trade_intensity=sell_a,
            maker_fee=0.001, decision_lag_bars=1,
        )

        gpu = gpu_simulate_binary(
            highs=highs, lows=lows, closes=closes, opens=opens,
            buy_prices=buy_p, sell_prices=sell_p,
            buy_amounts=buy_a, sell_amounts=sell_a,
            maker_fee=0.001, decision_lag_bars=1,
        )

        torch.testing.assert_close(ref.returns, gpu["returns"], atol=1e-6, rtol=1e-5)

    def test_parity_batched(self):
        torch.manual_seed(42)
        B, T = 4, 50
        closes = 100 + torch.randn(B, T) * 2
        highs = closes + torch.rand(B, T) * 2
        lows = closes - torch.rand(B, T) * 2
        buy_p = closes * 0.998
        sell_p = closes * 1.002
        buy_a = torch.rand(B, T) * 0.3
        sell_a = torch.rand(B, T) * 0.3

        ref = simulate_hourly_trades_binary(
            highs=highs, lows=lows, closes=closes,
            buy_prices=buy_p, sell_prices=sell_p,
            trade_intensity=buy_a,
            buy_trade_intensity=buy_a,
            sell_trade_intensity=sell_a,
            maker_fee=0.001,
        )

        gpu = gpu_simulate_binary(
            highs=highs, lows=lows, closes=closes,
            buy_prices=buy_p, sell_prices=sell_p,
            buy_amounts=buy_a, sell_amounts=sell_a,
            maker_fee=0.001,
        )

        torch.testing.assert_close(ref.returns, gpu["returns"], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(ref.portfolio_values, gpu["portfolio_values"], atol=1e-5, rtol=1e-5)

    def test_parity_market_order_entry(self):
        torch.manual_seed(22)
        highs, lows, closes, opens = _make_market_data(40)
        buy_p, sell_p, buy_a, sell_a = _make_actions(closes)

        ref = simulate_hourly_trades_binary(
            highs=highs, lows=lows, closes=closes, opens=opens,
            buy_prices=buy_p, sell_prices=sell_p,
            trade_intensity=buy_a,
            buy_trade_intensity=buy_a,
            sell_trade_intensity=sell_a,
            maker_fee=0.001, market_order_entry=True,
        )

        gpu = gpu_simulate_binary(
            highs=highs, lows=lows, closes=closes, opens=opens,
            buy_prices=buy_p, sell_prices=sell_p,
            buy_amounts=buy_a, sell_amounts=sell_a,
            maker_fee=0.001, market_order_entry=True,
        )

        torch.testing.assert_close(ref.returns, gpu["returns"], atol=1e-6, rtol=1e-5)


class TestSortino:
    def test_positive_returns(self):
        returns = torch.tensor([0.01, 0.02, 0.01, -0.005, 0.015])
        s = compute_sortino_gpu(returns)
        assert s.item() > 0

    def test_negative_returns(self):
        returns = torch.tensor([-0.01, -0.02, -0.01, -0.005, -0.015])
        s = compute_sortino_gpu(returns)
        assert s.item() < 0

    def test_zero_returns(self):
        returns = torch.zeros(10)
        s = compute_sortino_gpu(returns)
        assert abs(s.item()) < 1e-3

    def test_matches_reference(self):
        torch.manual_seed(42)
        returns = torch.randn(200) * 0.01
        gpu_sort = compute_sortino_gpu(returns, 8760.0)
        mean_r = returns.mean()
        ds = torch.clamp(-returns, min=0.0)
        ds_std = (ds.square().mean() + 1e-8).sqrt()
        ref_sort = (mean_r / ds_std) * (8760.0 ** 0.5)
        torch.testing.assert_close(gpu_sort, ref_sort, atol=1e-5, rtol=1e-5)

    def test_batched(self):
        torch.manual_seed(42)
        returns = torch.randn(3, 100) * 0.01
        s = compute_sortino_gpu(returns)
        assert s.shape == (3,)


class TestMaxDrawdown:
    def test_no_drawdown(self):
        returns = torch.tensor([0.01, 0.01, 0.01, 0.01])
        dd = compute_max_drawdown_gpu(returns)
        assert dd.item() < 1e-8

    def test_drawdown(self):
        returns = torch.tensor([0.1, -0.15, -0.05, 0.02])
        dd = compute_max_drawdown_gpu(returns)
        assert dd.item() > 0

    def test_matches_manual(self):
        returns = torch.tensor([0.05, -0.10, 0.02, -0.03])
        cum = torch.cumsum(returns, dim=-1)
        running_max = torch.cummax(cum, dim=-1).values
        expected_dd = (running_max - cum).max().item()
        dd = compute_max_drawdown_gpu(returns).item()
        assert abs(dd - expected_dd) < 1e-7

    def test_batched(self):
        returns = torch.randn(5, 50) * 0.01
        dd = compute_max_drawdown_gpu(returns)
        assert dd.shape == (5,)


class TestTotalReturn:
    def test_simple(self):
        values = torch.tensor([100.0, 105.0, 110.0])
        tr = compute_total_return_gpu(values)
        assert abs(tr.item() - 0.1) < 1e-6

    def test_loss(self):
        values = torch.tensor([100.0, 95.0, 90.0])
        tr = compute_total_return_gpu(values)
        assert abs(tr.item() - (-0.1)) < 1e-6


class TestNumTrades:
    def test_count(self):
        buys = torch.tensor([0.5, 0.0, 0.3, 0.0])
        sells = torch.tensor([0.0, 0.0, 0.0, 0.4])
        n = compute_num_trades_gpu(buys, sells)
        assert n.item() == 3

    def test_batched(self):
        buys = torch.tensor([[0.5, 0.0], [0.0, 0.3]])
        sells = torch.tensor([[0.0, 0.1], [0.0, 0.0]])
        n = compute_num_trades_gpu(buys, sells)
        assert n.shape == (2,)
        assert n[0].item() == 2
        assert n[1].item() == 1


class TestMetricsToScalars:
    def test_conversion(self):
        m = {"sortino": torch.tensor(2.5), "max_drawdown": torch.tensor(0.05)}
        s = metrics_to_scalars(m)
        assert isinstance(s["sortino"], float)
        assert abs(s["sortino"] - 2.5) < 1e-6


class TestMaxHoldBars:
    def test_force_close(self):
        torch.manual_seed(42)
        T = 20
        closes = torch.full((T,), 100.0)
        highs = torch.full((T,), 101.0)
        lows = torch.full((T,), 99.0)
        opens = torch.full((T,), 100.0)
        buy_prices = torch.full((T,), 99.5)
        sell_prices = torch.full((T,), 200.0)
        buy_amounts = torch.full((T,), 0.5)
        sell_amounts = torch.full((T,), 0.0)

        result_no_hold = gpu_simulate_binary(
            highs=highs, lows=lows, closes=closes, opens=opens,
            buy_prices=buy_prices, sell_prices=sell_prices,
            buy_amounts=buy_amounts, sell_amounts=sell_amounts,
            maker_fee=0.001, max_hold_bars=0,
        )

        result_hold = gpu_simulate_binary(
            highs=highs, lows=lows, closes=closes, opens=opens,
            buy_prices=buy_prices, sell_prices=sell_prices,
            buy_amounts=buy_amounts, sell_amounts=sell_amounts,
            maker_fee=0.001, max_hold_bars=3,
        )

        inv_no = result_no_hold["inventory_path"]
        inv_hold = result_hold["inventory_path"]
        assert inv_no[-1].item() > 0
        found_reset = False
        for i in range(3, T):
            if inv_hold[i].item() < inv_hold[i-1].item() - 1e-6:
                found_reset = True
                break
        assert found_reset, "max_hold_bars should force position close"


class TestComputeMetrics:
    def test_full_pipeline(self):
        torch.manual_seed(42)
        highs, lows, closes, opens = _make_market_data(100)
        buy_p, sell_p, buy_a, sell_a = _make_actions(closes)

        sim = gpu_simulate_binary(
            highs=highs, lows=lows, closes=closes, opens=opens,
            buy_prices=buy_p, sell_prices=sell_p,
            buy_amounts=buy_a, sell_amounts=sell_a,
            maker_fee=0.001,
        )

        metrics = compute_metrics_gpu(sim)
        scalars = metrics_to_scalars(metrics)

        assert "sortino" in scalars
        assert "max_drawdown" in scalars
        assert "total_return" in scalars
        assert "num_trades" in scalars
        assert isinstance(scalars["sortino"], float)


class TestEdgeCases:
    def test_zero_amounts(self):
        T = 10
        closes = torch.full((T,), 100.0)
        highs = torch.full((T,), 101.0)
        lows = torch.full((T,), 99.0)
        result = gpu_simulate_binary(
            highs=highs, lows=lows, closes=closes,
            buy_prices=torch.full((T,), 99.0),
            sell_prices=torch.full((T,), 101.0),
            buy_amounts=torch.zeros(T),
            sell_amounts=torch.zeros(T),
        )
        assert (result["executed_buys"] == 0).all()
        assert (result["executed_sells"] == 0).all()

    def test_single_step(self):
        result = gpu_simulate_binary(
            highs=torch.tensor([101.0]),
            lows=torch.tensor([99.0]),
            closes=torch.tensor([100.0]),
            buy_prices=torch.tensor([99.5]),
            sell_prices=torch.tensor([100.5]),
            buy_amounts=torch.tensor([0.5]),
            sell_amounts=torch.tensor([0.5]),
        )
        assert result["returns"].shape == (1,)

    def test_no_fills(self):
        T = 10
        closes = torch.full((T,), 100.0)
        highs = torch.full((T,), 100.5)
        lows = torch.full((T,), 99.5)
        result = gpu_simulate_binary(
            highs=highs, lows=lows, closes=closes,
            buy_prices=torch.full((T,), 98.0),
            sell_prices=torch.full((T,), 102.0),
            buy_amounts=torch.full((T,), 0.5),
            sell_amounts=torch.full((T,), 0.5),
        )
        assert (result["executed_buys"] == 0).all()
        assert (result["executed_sells"] == 0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCuda:
    def test_stays_on_gpu(self):
        device = torch.device("cuda")
        highs, lows, closes, opens = _make_market_data(50, device=device)
        buy_p, sell_p, buy_a, sell_a = _make_actions(closes)

        result = gpu_simulate_binary(
            highs=highs, lows=lows, closes=closes, opens=opens,
            buy_prices=buy_p, sell_prices=sell_p,
            buy_amounts=buy_a, sell_amounts=sell_a,
        )

        for k, v in result.items():
            if isinstance(v, torch.Tensor):
                assert v.device.type == "cuda", f"{k} not on cuda"

    def test_cuda_parity_with_cpu(self):
        torch.manual_seed(42)
        highs_cpu, lows_cpu, closes_cpu, opens_cpu = _make_market_data(60)
        buy_p_cpu, sell_p_cpu, buy_a_cpu, sell_a_cpu = _make_actions(closes_cpu)

        cpu_result = gpu_simulate_binary(
            highs=highs_cpu, lows=lows_cpu, closes=closes_cpu, opens=opens_cpu,
            buy_prices=buy_p_cpu, sell_prices=sell_p_cpu,
            buy_amounts=buy_a_cpu, sell_amounts=sell_a_cpu,
            maker_fee=0.001, max_leverage=2.0, margin_annual_rate=0.0625,
        )

        device = torch.device("cuda")
        gpu_result = gpu_simulate_binary(
            highs=_to_device_or_skip(highs_cpu, device), lows=_to_device_or_skip(lows_cpu, device),
            closes=_to_device_or_skip(closes_cpu, device), opens=_to_device_or_skip(opens_cpu, device),
            buy_prices=_to_device_or_skip(buy_p_cpu, device), sell_prices=_to_device_or_skip(sell_p_cpu, device),
            buy_amounts=_to_device_or_skip(buy_a_cpu, device), sell_amounts=_to_device_or_skip(sell_a_cpu, device),
            maker_fee=0.001, max_leverage=2.0, margin_annual_rate=0.0625,
        )

        torch.testing.assert_close(
            cpu_result["returns"],
            gpu_result["returns"].cpu(),
            atol=1e-5, rtol=1e-4,
        )
