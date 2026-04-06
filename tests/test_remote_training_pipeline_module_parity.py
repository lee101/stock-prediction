from __future__ import annotations

import importlib
import sys


def test_remote_training_pipeline_root_module_reexports_src_implementation() -> None:
    sys.modules.pop("remote_training_pipeline", None)
    sys.modules.pop("src.remote_training_pipeline", None)

    root_module = importlib.import_module("remote_training_pipeline")
    src_module = importlib.import_module("src.remote_training_pipeline")

    assert root_module is src_module
    assert root_module.parse_csv_tokens is src_module.parse_csv_tokens
    assert root_module.normalize_symbols is src_module.normalize_symbols
    assert root_module.render_remote_pipeline_script is src_module.render_remote_pipeline_script
    assert root_module.build_remote_hourly_chronos_rl_plan is src_module.build_remote_hourly_chronos_rl_plan
    assert root_module.build_remote_autoresearch_plan is src_module.build_remote_autoresearch_plan
    assert root_module.LARGE_UNIVERSE_STOCK_A40_DESCRIPTIONS is src_module.LARGE_UNIVERSE_STOCK_A40_DESCRIPTIONS
    assert root_module.resolve_daily_symbol_path is src_module.resolve_daily_symbol_path
