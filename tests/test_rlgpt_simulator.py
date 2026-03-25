from __future__ import annotations

import torch

from RLgpt.config import SimulatorConfig
from RLgpt.simulator import compute_trading_objective, simulate_daily_plans


def test_simulator_executes_daily_buy_then_sell_plan():
    config = SimulatorConfig(
        initial_cash=1_000.0,
        shared_unit_budget=4.0,
        max_units_per_asset=4.0,
        maker_fee_bps=0.0,
        slippage_bps=0.0,
        fill_buffer_bps=0.0,
        fill_temperature_bps=0.1,
        carry_inventory=False,
    )
    sim_out = simulate_daily_plans(
        hourly_open=torch.tensor([[[100.0, 50.0], [100.0, 50.0]]]),
        hourly_high=torch.tensor([[[100.4, 50.1], [101.5, 50.1]]]),
        hourly_low=torch.tensor([[[98.5, 49.9], [99.8, 49.9]]]),
        hourly_close=torch.tensor([[[99.5, 50.0], [101.0, 50.0]]]),
        hourly_mask=torch.ones(1, 2, 2),
        daily_anchor=torch.tensor([[100.0, 50.0]]),
        plans={
            "allocation_logits": torch.tensor([[8.0, -8.0]]),
            "center_offset_bps": torch.tensor([[0.0, 0.0]]),
            "half_spread_bps": torch.tensor([[1.0, 1.0]]),
            "max_long_fraction": torch.tensor([[1.0, 1.0]]),
            "max_short_fraction": torch.tensor([[0.0, 0.0]]),
            "trade_fraction": torch.tensor([[1.0, 1.0]]),
            "budget_scale": torch.tensor([[1.0]]),
        },
        config=config,
    )

    assert sim_out["allocation"][0, 0].item() > 0.999
    assert sim_out["final_inventory"].abs().sum().item() == 0.0
    assert sim_out["final_equity"].item() > config.initial_cash


def test_simulator_respects_shared_cash_constraint():
    config = SimulatorConfig(
        initial_cash=100.0,
        shared_unit_budget=4.0,
        max_units_per_asset=4.0,
        maker_fee_bps=0.0,
        slippage_bps=0.0,
        fill_buffer_bps=0.0,
        fill_temperature_bps=0.1,
        carry_inventory=True,
    )
    sim_out = simulate_daily_plans(
        hourly_open=torch.tensor([[[100.0, 100.0]]]),
        hourly_high=torch.tensor([[[100.0, 100.0]]]),
        hourly_low=torch.tensor([[[99.0, 99.0]]]),
        hourly_close=torch.tensor([[[100.0, 100.0]]]),
        hourly_mask=torch.ones(1, 1, 2),
        daily_anchor=torch.tensor([[100.0, 100.0]]),
        plans={
            "allocation_logits": torch.tensor([[0.0, 0.0]]),
            "center_offset_bps": torch.tensor([[0.0, 0.0]]),
            "half_spread_bps": torch.tensor([[1.0, 1.0]]),
            "max_long_fraction": torch.tensor([[1.0, 1.0]]),
            "max_short_fraction": torch.tensor([[0.0, 0.0]]),
            "trade_fraction": torch.tensor([[1.0, 1.0]]),
            "budget_scale": torch.tensor([[1.0]]),
        },
        config=config,
    )

    assert sim_out["final_inventory"].sum().item() <= 1.01
    assert sim_out["final_equity"].item() <= 100.1


def test_objective_penalizes_ending_inventory():
    base = {
        "pnl_path": torch.tensor([[2.0, 3.0]]),
        "turnover_path": torch.tensor([[10.0, 10.0]]),
        "initial_equity": torch.tensor([100.0]),
        "final_equity": torch.tensor([105.0]),
    }
    _, with_inventory = compute_trading_objective(
        {
            **base,
            "final_inventory": torch.tensor([[3.0]]),
        },
        SimulatorConfig(inventory_penalty=0.5),
    )
    _, flat = compute_trading_objective(
        {
            **base,
            "final_inventory": torch.tensor([[0.0]]),
        },
        SimulatorConfig(inventory_penalty=0.5),
    )

    assert flat["score"].item() > with_inventory["score"].item()
