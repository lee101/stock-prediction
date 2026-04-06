from __future__ import annotations

import importlib


def test_preaug_forecast_config_root_module_aliases_src_module():
    root_module = importlib.import_module("preaug.forecast_config")
    src_module = importlib.import_module("src.preaug.forecast_config")

    assert root_module is src_module
    assert root_module.ForecastTag is src_module.ForecastTag
    assert root_module.ForecastConfig is src_module.ForecastConfig
    assert root_module.ForecastConfigSelector is src_module.ForecastConfigSelector


def test_preaug_package_uses_src_forecast_config_defaults():
    preaug_module = importlib.import_module("preaug")
    selector = preaug_module.ForecastConfigSelector()

    config = selector.get_config_for_symbols(["BTCUSD", "ETHUSD"])

    assert config.use_cross_learning is False
