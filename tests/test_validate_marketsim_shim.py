from __future__ import annotations

import importlib.util
import runpy
from pathlib import Path

from pufferlib_market import validate_marketsim as package_module


REPO = Path(__file__).resolve().parents[1]
ROOT_SHIM = REPO / "validate_marketsim.py"


def _load_root_shim():
    spec = importlib.util.spec_from_file_location("root_validate_marketsim", ROOT_SHIM)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_root_validate_marketsim_shim_reexports_canonical_main() -> None:
    shim = _load_root_shim()

    assert shim.main is package_module.main


def test_root_validate_marketsim_shim_reexports_core_public_helpers() -> None:
    shim = _load_root_shim()

    assert shim.load_hourly_bars is package_module.load_hourly_bars
    assert shim.load_daily_bars is package_module.load_daily_bars
    assert shim.FEE_TIERS == package_module.FEE_TIERS
    assert shim.SLIPPAGE_BPS == package_module.SLIPPAGE_BPS


def test_root_validate_marketsim_shim_exports_explicit_public_surface() -> None:
    shim = _load_root_shim()

    assert shim.__all__ == (
        "DailyPPOTrader",
        "FEE_TIERS",
        "PPOTrader",
        "SLIPPAGE_BPS",
        "TradingSignal",
        "compute_daily_feature_history",
        "compute_hourly_feature_snapshot",
        "load_daily_bars",
        "load_hourly_bars",
        "main",
    )


def test_root_validate_marketsim_shim_script_mode_delegates_to_canonical_main(
    monkeypatch,
) -> None:
    calls: list[str] = []

    def _fake_main() -> None:
        calls.append("main")

    monkeypatch.setattr(package_module, "main", _fake_main)

    runpy.run_path(str(ROOT_SHIM), run_name="__main__")

    assert calls == ["main"]
