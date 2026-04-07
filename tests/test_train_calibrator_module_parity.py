from __future__ import annotations

import importlib
import sys


def test_train_calibrator_root_module_reexports_rl_binance_implementation() -> None:
    sys.modules.pop("train_calibrator", None)
    sys.modules.pop("rl_trading_agent_binance.train_calibrator", None)

    root_module = importlib.import_module("train_calibrator")
    package_module = importlib.import_module("rl_trading_agent_binance.train_calibrator")

    assert root_module is package_module
    assert root_module.time_split is package_module.time_split
    assert root_module.prepare_symbol_tensors is package_module.prepare_symbol_tensors
    assert root_module.run_sim is package_module.run_sim
    assert root_module.train_one_symbol is package_module.train_one_symbol
