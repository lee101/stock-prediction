from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


MODULE_DIR = Path(__file__).resolve().parents[1] / "rl-trading-agent-binance"


def _load_module(name: str, relative_path: str):
    if str(MODULE_DIR) not in sys.path:
        sys.path.insert(0, str(MODULE_DIR))
    module_path = MODULE_DIR / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


validate_hybrid = _load_module(
    "validate_hybrid_cycle_snapshots_testshim",
    "validate_hybrid_cycle_snapshots.py",
)


def test_pull_margin_orders_dedupes_boundary_overlap(monkeypatch) -> None:
    order = {
        "orderId": 123,
        "time": 1_700_000_000_000,
        "updateTime": 1_700_000_000_100,
        "side": "BUY",
        "status": "NEW",
        "price": "0.1",
        "origQty": "10",
        "executedQty": "0",
    }

    monkeypatch.setattr(
        validate_hybrid,
        "get_all_margin_orders",
        lambda *args, **kwargs: [dict(order)],
    )

    rows = validate_hybrid.pull_margin_orders(
        ["DOGEUSDT"],
        pd.Timestamp("2026-01-01 00:00:00+00:00"),
        pd.Timestamp("2026-01-03 00:00:00+00:00"),
    )

    assert len(rows) == 1
    assert rows[0]["orderId"] == 123
    assert rows[0]["symbol"] == "DOGEUSDT"


def test_pull_margin_trades_dedupes_boundary_overlap(monkeypatch) -> None:
    trade = {
        "id": 77,
        "orderId": 123,
        "time": 1_700_000_000_000,
        "isBuyer": True,
        "price": "0.1",
        "qty": "10",
    }

    monkeypatch.setattr(
        validate_hybrid,
        "get_margin_trades",
        lambda *args, **kwargs: [dict(trade)],
    )

    rows = validate_hybrid.pull_margin_trades(
        ["DOGEUSDT"],
        pd.Timestamp("2026-01-01 00:00:00+00:00"),
        pd.Timestamp("2026-01-03 00:00:00+00:00"),
    )

    assert len(rows) == 1
    assert rows[0]["id"] == 77
    assert rows[0]["symbol"] == "DOGEUSDT"
