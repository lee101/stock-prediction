from __future__ import annotations

from pathlib import Path

from falmarket.app import MarketSimulatorApp


def test_local_python_modules_cover_required_packages() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    modules = MarketSimulatorApp.local_python_modules

    expected_paths = {
        "falmarket": repo_root / "falmarket",
        "fal_marketsimulator": repo_root / "fal_marketsimulator",
        "faltrain": repo_root / "faltrain",
        "marketsimulator": repo_root / "marketsimulator",
        "trade_stock_e2e": repo_root / "trade_stock_e2e.py",
        "trade_stock_e2e_trained": repo_root / "trade_stock_e2e_trained.py",
        "alpaca_wrapper": repo_root / "alpaca_wrapper.py",
        "backtest_test3_inline": repo_root / "backtest_test3_inline.py",
        "data_curate_daily": repo_root / "data_curate_daily.py",
        "env_real": repo_root / "env_real.py",
        "jsonshelve": repo_root / "jsonshelve.py",
        "src": repo_root / "src",
        "stock": repo_root / "stock",
        "utils": repo_root / "utils",
        "traininglib": repo_root / "traininglib",
    }

    for module_name, path in expected_paths.items():
        assert module_name in modules, f"{module_name} missing from local_python_modules"
        assert path.exists(), f"{path} is missing for module {module_name}"

    assert len(modules) == len(set(modules)), "local_python_modules contains duplicates"
