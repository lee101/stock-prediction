from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO / "scripts" / "simulate_multistage_refiner.py"


def _load_module():
    spec = spec_from_file_location("simulate_multistage_refiner", SCRIPT_PATH)
    module = module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_args_defaults_are_stable() -> None:
    mod = _load_module()

    args = mod._parse_args([])

    assert args.symbols == ["AAPL", "MSFT", "NVDA"]
    assert args.bars_per_symbol == 240
    assert args.warmup_bars == 48
    assert args.seed == 7
    assert args.output_dir == "analysis/multistage_refiner_sim"
    assert args.no_wandb is False
    assert args.r2_dest is None


def test_baseline_plan_from_rl_respects_direction_and_caps_allocation() -> None:
    mod = _load_module()

    long_signal = mod.RLSignal(
        symbol_idx=0,
        symbol_name="AAPL",
        direction="long",
        confidence=0.8,
        logit_gap=2.0,
        allocation_pct=3.0,
        level_offset_bps=0.0,
    )
    hold_signal = mod.RLSignal(
        symbol_idx=1,
        symbol_name="MSFT",
        direction="hold",
        confidence=0.4,
        logit_gap=0.5,
        allocation_pct=0.5,
        level_offset_bps=0.0,
    )

    long_plan = mod._baseline_plan_from_rl(long_signal, current_price=100.0)
    hold_plan = mod._baseline_plan_from_rl(hold_signal, current_price=100.0)

    assert long_plan.direction == "long"
    assert long_plan.buy_price < 100.0
    assert long_plan.sell_price > 100.0
    assert long_plan.allocation_pct == 100.0

    assert hold_plan.direction == "hold"
    assert hold_plan.buy_price == 0.0
    assert hold_plan.sell_price == 0.0
    assert hold_plan.allocation_pct == 0.0
