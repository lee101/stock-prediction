from __future__ import annotations

import importlib
import sys


def test_maxdiff_entry_guard_root_module_aliases_src_module():
    sys.modules.pop("maxdiff_entry_guard", None)
    sys.modules.pop("src.maxdiff_entry_guard", None)

    root_module = importlib.import_module("maxdiff_entry_guard")
    src_module = importlib.import_module("src.maxdiff_entry_guard")

    assert root_module is src_module


def test_maxdiff_entry_guard_root_and_src_share_live_functions():
    sys.modules.pop("maxdiff_entry_guard", None)
    sys.modules.pop("src.maxdiff_entry_guard", None)

    root_module = importlib.import_module("maxdiff_entry_guard")
    src_module = importlib.import_module("src.maxdiff_entry_guard")

    assert root_module._effective_entry_quantities is src_module._effective_entry_quantities
    assert root_module._sum_order_qty is src_module._sum_order_qty
    assert root_module.EntryQuantitySnapshot is src_module.EntryQuantitySnapshot
