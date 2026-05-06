from __future__ import annotations

import pytest
import scripts.binance_margin_dust_cleanup as cleanup


def test_build_cleanup_plan_filters_assets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cleanup,
        "get_small_liability_exchange_assets",
        lambda: [
            {"asset": "TON", "liabilityQty": "0.04"},
            {"asset": "DOGE", "liabilityQty": "0.22"},
        ],
    )
    monkeypatch.setattr(
        cleanup,
        "get_margin_dust_assets",
        lambda: {
            "details": [
                {"asset": "ADA", "toBNB": "0.1"},
                {"asset": "ZEC", "toBNB": "0.2"},
            ],
            "totalTransferBNB": "0.3",
        },
    )

    plan = cleanup.build_cleanup_plan(
        include_liability_assets={"TON"},
        include_margin_dust_assets={"ZEC"},
    )

    assert plan.liability_assets == ["TON"]
    assert plan.margin_dust_assets == ["ZEC"]
    assert plan.total_margin_dust_bnb == pytest.approx(0.3)


def test_execute_cleanup_requires_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALLOW_BINANCE_DUST_CLEANUP", raising=False)
    plan = cleanup.DustCleanupPlan(
        liability_assets=["TON"],
        margin_dust_assets=[],
        liability_rows=[],
        margin_dust_rows=[],
        total_margin_dust_bnb=0.0,
        execute=False,
    )

    with pytest.raises(RuntimeError, match="ALLOW_BINANCE_DUST_CLEANUP"):
        cleanup.execute_cleanup(plan)


def test_execute_cleanup_calls_exchange_apis(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOW_BINANCE_DUST_CLEANUP", "1")
    calls = {}
    monkeypatch.setattr(
        cleanup,
        "exchange_small_liability_assets",
        lambda assets: calls.setdefault("liability", list(assets)) or {"ok": True},
    )
    monkeypatch.setattr(
        cleanup,
        "transfer_margin_dust",
        lambda assets: calls.setdefault("dust", list(assets)) or {"ok": True},
    )
    plan = cleanup.DustCleanupPlan(
        liability_assets=["TON"],
        margin_dust_assets=["ADA"],
        liability_rows=[],
        margin_dust_rows=[],
        total_margin_dust_bnb=0.1,
        execute=False,
    )

    executed = cleanup.execute_cleanup(plan)

    assert executed.execute is True
    assert calls == {"liability": ["TON"], "dust": ["ADA"]}
