from __future__ import annotations

import json
from pathlib import Path

from src.preaug.forecast_config import ForecastConfigSelector, ForecastTag


def test_forecast_tag_classifies_u_suffix_stock_as_stock() -> None:
    tag = ForecastTag.from_symbols_and_targets(["MU"], targets=("close",))
    assert tag.asset_type == "stock"


def test_forecast_tag_classifies_mixed_stock_and_btc_u_as_mixed() -> None:
    tag = ForecastTag.from_symbols_and_targets(["AAPL", "BTCU"], targets=("close",))
    assert tag.asset_type == "mixed"


def test_forecast_tag_classifies_u_quote_crypto_only_list_as_crypto() -> None:
    tag = ForecastTag.from_symbols_and_targets(["BTCU", "ETHU"], targets=("close",))
    assert tag.asset_type == "crypto"


def test_forecast_config_selector_disables_cross_learning_for_multi_crypto_ohlc() -> None:
    selector = ForecastConfigSelector()

    config = selector.get_config_for_symbols(["BTCUSD", "ETHUSD"])

    assert config.use_cross_learning is False


def test_forecast_config_selector_prefers_loaded_exact_config(tmp_path: Path) -> None:
    tag = ForecastTag.from_symbols_and_targets(["BTCUSD", "ETHUSD"])
    config_path = tmp_path / "forecast_config.json"
    config_path.write_text(
        json.dumps(
            {
                "configs": {
                    tag.tag_key: {
                        "symbols": list(tag.symbols),
                        "targets": list(tag.targets),
                        "asset_type": tag.asset_type,
                        "use_multivariate": True,
                        "use_cross_learning": True,
                        "use_multiscale": True,
                        "multiscale_method": "weighted",
                        "multiscale_skip_rates": [1, 3, 5],
                        "batch_size": 12,
                        "context_length": 1024,
                        "mae_improvement": 42.0,
                        "source": "fixture",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    selector = ForecastConfigSelector(config_paths=[config_path])
    config = selector.get_config(tag)

    assert config.use_multivariate is True
    assert config.use_cross_learning is True
    assert config.use_multiscale is True
    assert config.multiscale_method == "weighted"
    assert config.multiscale_skip_rates == (1, 3, 5)
    assert config.batch_size == 12
    assert config.context_length == 1024
    assert config.mae_improvement == 42.0
    assert config.source == "fixture"


def test_forecast_config_selector_loads_per_symbol_multiscale_config(tmp_path: Path) -> None:
    multiscale_path = tmp_path / "multiscale.json"
    multiscale_path.write_text(
        json.dumps(
            {
                "symbol_configs": {
                    "AAPL": {
                        "method": "median",
                        "skip_rates": [1, 2, 4],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    selector = ForecastConfigSelector(multiscale_config_paths=[multiscale_path])
    config = selector.get_config_for_symbols(["AAPL"])

    assert config.use_multivariate is True
    assert config.use_cross_learning is False
    assert config.use_multiscale is True
    assert config.multiscale_method == "median"
    assert config.multiscale_skip_rates == (1, 2, 4)
