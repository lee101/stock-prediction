from __future__ import annotations

import importlib


def test_torch_backend_root_module_aliases_src_module():
    root_module = importlib.import_module("torch_backend")
    src_module = importlib.import_module("src.torch_backend")

    assert root_module is src_module
    assert root_module.configure_tf32_backends is src_module.configure_tf32_backends
    assert (
        root_module.maybe_set_float32_precision
        is src_module.maybe_set_float32_precision
    )
