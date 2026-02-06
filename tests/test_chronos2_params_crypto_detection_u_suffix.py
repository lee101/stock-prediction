from __future__ import annotations

from typing import Any, Optional

import pytest

import src.chronos2_params as chronos_params


@pytest.fixture()
def params_module():
    chronos_params._chronos2_params_cache.clear()  # type: ignore[attr-defined]
    yield chronos_params
    chronos_params._chronos2_params_cache.clear()  # type: ignore[attr-defined]


def _no_configs(_: str, __: str, store: Optional[Any] = None) -> None:
    return None


def test_resolve_chronos2_params_does_not_treat_u_suffix_stock_as_crypto(monkeypatch, params_module) -> None:
    monkeypatch.setattr(params_module, "load_best_config", _no_configs)
    monkeypatch.setattr(params_module, "CHRONOS2_USE_MULTIVARIATE", True)
    monkeypatch.setattr(params_module, "_get_symbol_multivariate_setting", lambda symbol, default: default)

    params = params_module.resolve_chronos2_params("MU", frequency="daily")
    assert params["use_multivariate"] is True


def test_resolve_chronos2_params_treats_btc_u_as_crypto(monkeypatch, params_module) -> None:
    monkeypatch.setattr(params_module, "load_best_config", _no_configs)
    monkeypatch.setattr(params_module, "CHRONOS2_USE_MULTIVARIATE", True)
    monkeypatch.setattr(params_module, "_get_symbol_multivariate_setting", lambda symbol, default: default)

    params = params_module.resolve_chronos2_params("BTCU", frequency="daily")
    assert params["use_multivariate"] is False
