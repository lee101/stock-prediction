from __future__ import annotations

import importlib


def test_chronos2_root_module_reexports_src_implementation() -> None:
    root_module = importlib.import_module("models.chronos2_wrapper")
    src_module = importlib.import_module("src.models.chronos2_wrapper")

    assert root_module.Chronos2OHLCWrapper is src_module.Chronos2OHLCWrapper
    assert root_module.DEFAULT_QUANTILE_LEVELS is src_module.DEFAULT_QUANTILE_LEVELS
    assert root_module.DEFAULT_TARGET_COLUMNS is src_module.DEFAULT_TARGET_COLUMNS
    assert root_module._resolve_model_source is src_module._resolve_model_source
    assert root_module._hash_dataframe_for_cache is src_module._hash_dataframe_for_cache


def test_root_multiscale_module_uses_shared_chronos2_wrapper() -> None:
    root_multiscale = importlib.import_module("models.multiscale_chronos")
    src_module = importlib.import_module("src.models.chronos2_wrapper")

    assert root_multiscale.Chronos2OHLCWrapper is src_module.Chronos2OHLCWrapper
