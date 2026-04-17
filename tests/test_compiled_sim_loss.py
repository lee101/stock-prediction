from __future__ import annotations

from pathlib import Path

import pytest
import torch

from differentiable_loss_utils import HOURLY_PERIODS_PER_YEAR, compute_loss_by_type
from trainingefficiency.compiled_sim_loss import (
    _sim_loop_sortino,
    compiled_sim_and_loss,
    compiled_sim_trajectory,
    get_compiled_sim_loss,
)
from trainingefficiency.compiled_sim_visual import trajectory_to_marketsim_trace, write_comparison_html
from trainingefficiency.fast_differentiable_sim import simulate_hourly_trades_fast
import trainingefficiency.compiled_sim_loss as compiled_module


def test_get_compiled_sim_loss_skips_compile_when_disabled(monkeypatch) -> None:
    monkeypatch.setenv("TORCH_NO_COMPILE", "1")
    monkeypatch.setattr(compiled_module, "_compiled_sim_loss", None)

    def _unexpected_compile(*_args, **_kwargs):
        raise AssertionError("torch.compile should not run when TORCH_NO_COMPILE is set")

    monkeypatch.setattr(compiled_module.torch, "compile", _unexpected_compile)

    fn = get_compiled_sim_loss()

    assert fn is _sim_loop_sortino


def test_compiled_sim_and_loss_matches_vectorized_sortino(monkeypatch) -> None:
    monkeypatch.setattr(compiled_module, "get_compiled_sim_loss", lambda: _sim_loop_sortino)

    torch.manual_seed(0)
    shape = (2, 8)
    base = 100.0 + torch.randn(shape)
    closes = base
    highs = base + torch.rand(shape)
    lows = base - torch.rand(shape)
    buy_prices = base - 0.5 * torch.rand(shape)
    sell_prices = base + 0.5 * torch.rand(shape)
    buy_frac = torch.rand(shape) * 0.25
    sell_frac = torch.rand(shape) * 0.25
    max_leverage = torch.ones(shape)
    can_short = torch.zeros(shape[0])
    can_long = torch.ones(shape[0])

    loss, score, sortino, annual_return = compiled_sim_and_loss(
        highs=highs,
        lows=lows,
        closes=closes,
        buy_prices=buy_prices,
        sell_prices=sell_prices,
        buy_frac=buy_frac,
        sell_frac=sell_frac,
        max_leverage=max_leverage,
        can_short=can_short,
        can_long=can_long,
        initial_cash=1.0,
        initial_inventory=0.0,
        maker_fee=0.001,
        temperature=0.01,
        fill_buffer_pct=0.0005,
        margin_annual_rate=0.0,
        periods_per_year=HOURLY_PERIODS_PER_YEAR,
        return_weight=0.08,
        decision_lag_bars=1,
    )

    sim = simulate_hourly_trades_fast(
        highs=highs,
        lows=lows,
        closes=closes,
        buy_prices=buy_prices,
        sell_prices=sell_prices,
        trade_intensity=buy_frac,
        buy_trade_intensity=buy_frac,
        sell_trade_intensity=sell_frac,
        maker_fee=0.001,
        initial_cash=1.0,
        initial_inventory=0.0,
        temperature=0.01,
        max_leverage=max_leverage,
        can_short=can_short,
        can_long=can_long,
        decision_lag_bars=1,
        fill_buffer_pct=0.0005,
        margin_annual_rate=0.0,
    )
    ref_loss, ref_score, ref_sortino, ref_annual_return = compute_loss_by_type(
        sim.returns,
        "sortino",
        periods_per_year=HOURLY_PERIODS_PER_YEAR,
        return_weight=0.08,
    )

    assert torch.allclose(loss, ref_loss, atol=1e-6, rtol=1e-6)
    assert torch.allclose(score, ref_score, atol=1e-6, rtol=1e-6)
    assert torch.allclose(sortino, ref_sortino, atol=1e-6, rtol=1e-6)
    assert torch.allclose(annual_return, ref_annual_return, atol=1e-6, rtol=1e-6)


def test_compiled_sim_trajectory_matches_vectorized_state() -> None:
    torch.manual_seed(0)
    shape = (2, 8)
    base = 100.0 + torch.randn(shape)
    closes = base
    highs = base + torch.rand(shape)
    lows = base - torch.rand(shape)
    buy_prices = base - 0.5 * torch.rand(shape)
    sell_prices = base + 0.5 * torch.rand(shape)
    buy_frac = torch.rand(shape) * 0.25
    sell_frac = torch.rand(shape) * 0.25
    max_leverage = torch.full(shape, 1.5)
    can_short = torch.ones(shape[0])
    can_long = torch.ones(shape[0])

    fused = compiled_sim_trajectory(
        highs=highs,
        lows=lows,
        closes=closes,
        buy_prices=buy_prices,
        sell_prices=sell_prices,
        buy_frac=buy_frac,
        sell_frac=sell_frac,
        max_leverage=max_leverage,
        can_short=can_short,
        can_long=can_long,
        initial_cash=1.0,
        initial_inventory=0.0,
        maker_fee=0.001,
        temperature=0.01,
        fill_buffer_pct=0.0005,
        margin_annual_rate=0.0625,
        periods_per_year=HOURLY_PERIODS_PER_YEAR,
        return_weight=0.08,
        decision_lag_bars=1,
    )

    baseline = simulate_hourly_trades_fast(
        highs=highs,
        lows=lows,
        closes=closes,
        buy_prices=buy_prices,
        sell_prices=sell_prices,
        trade_intensity=buy_frac,
        buy_trade_intensity=buy_frac,
        sell_trade_intensity=sell_frac,
        maker_fee=0.001,
        initial_cash=1.0,
        initial_inventory=0.0,
        temperature=0.01,
        max_leverage=max_leverage,
        can_short=can_short,
        can_long=can_long,
        decision_lag_bars=1,
        fill_buffer_pct=0.0005,
        margin_annual_rate=0.0625,
    )
    ref_loss, ref_score, ref_sortino, ref_annual_return = compute_loss_by_type(
        baseline.returns,
        "sortino",
        periods_per_year=HOURLY_PERIODS_PER_YEAR,
        return_weight=0.08,
    )

    assert torch.allclose(fused.returns, baseline.returns, atol=1e-6, rtol=1e-6)
    assert torch.allclose(fused.portfolio_values, baseline.portfolio_values, atol=1e-6, rtol=1e-6)
    assert torch.allclose(fused.inventory_path, baseline.inventory_path, atol=1e-6, rtol=1e-6)
    assert torch.allclose(fused.executed_buys, baseline.executed_buys, atol=1e-6, rtol=1e-6)
    assert torch.allclose(fused.executed_sells, baseline.executed_sells, atol=1e-6, rtol=1e-6)
    assert torch.allclose(fused.buy_fill_probability, baseline.buy_fill_probability, atol=1e-6, rtol=1e-6)
    assert torch.allclose(fused.sell_fill_probability, baseline.sell_fill_probability, atol=1e-6, rtol=1e-6)
    assert torch.allclose(fused.loss, ref_loss, atol=1e-6, rtol=1e-6)
    assert torch.allclose(fused.score, ref_score, atol=1e-6, rtol=1e-6)
    assert torch.allclose(fused.sortino, ref_sortino, atol=1e-6, rtol=1e-6)
    assert torch.allclose(fused.annual_return, ref_annual_return, atol=1e-6, rtol=1e-6)


def test_visual_helpers_render_comparison_html(tmp_path: Path) -> None:
    pytest.importorskip("plotly.graph_objects")

    torch.manual_seed(0)
    shape = (1, 6)
    base = 100.0 + torch.randn(shape)
    trajectory = compiled_sim_trajectory(
        highs=base + 1.0,
        lows=base - 1.0,
        closes=base,
        buy_prices=base * 0.999,
        sell_prices=base * 1.001,
        buy_frac=torch.full(shape, 0.1),
        sell_frac=torch.full(shape, 0.05),
        max_leverage=torch.ones(shape),
        can_short=torch.ones(shape[0]),
        can_long=torch.ones(shape[0]),
        maker_fee=0.001,
        decision_lag_bars=0,
    )

    trace = trajectory_to_marketsim_trace(trajectory, sample_index=0, symbol="TEST")
    assert trace.num_steps() == 6
    assert trace.symbols == ["TEST"]

    out = tmp_path / "comparison.html"
    write_comparison_html(
        out_path=out,
        baseline=trajectory,
        fused=trajectory,
        sample_index=0,
        title="compiled visual parity",
    )
    content = out.read_text().lower()
    assert out.exists()
    assert "plotly" in content
    assert "compiled visual parity" in content
