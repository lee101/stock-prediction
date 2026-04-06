from __future__ import annotations

import importlib
import sys

def test_daily_mixed_hybrid_root_module_reexports_src_implementation() -> None:
    sys.modules.pop("daily_mixed_hybrid", None)
    sys.modules.pop("src.daily_mixed_hybrid", None)

    root_module = importlib.import_module("daily_mixed_hybrid")
    src_module = importlib.import_module("src.daily_mixed_hybrid")

    assert root_module is src_module
    assert root_module.build_candidate_plan is src_module.build_candidate_plan
    assert root_module.refine_trade_plan_multistage is src_module.refine_trade_plan_multistage
    assert root_module.DEFAULT_REFINEMENT_PASSES is src_module.DEFAULT_REFINEMENT_PASSES
    assert root_module._forecast_block is src_module._forecast_block
