from __future__ import annotations

import importlib
import sys


def test_root_crypto_alpaca_looper_api_delegates_to_src_module() -> None:
    sys.modules.pop("crypto_loop.crypto_alpaca_looper_api", None)
    sys.modules.pop("src.crypto_loop.crypto_alpaca_looper_api", None)

    root_module = importlib.import_module("crypto_loop.crypto_alpaca_looper_api")
    src_module = importlib.import_module("src.crypto_loop.crypto_alpaca_looper_api")

    assert root_module is src_module
    assert root_module.submit_order is src_module.submit_order
    assert root_module.get_orders is src_module.get_orders
    assert root_module.FakeOrder is src_module.FakeOrder
