from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

if "fal" not in globals():
    import types

    class _FalAppMeta(type):
        def __new__(mcls, cls_name, bases, namespace, **kwargs):
            cls = super().__new__(mcls, cls_name, bases, dict(namespace))
            cls._fal_options = kwargs
            return cls

    class _FalApp(metaclass=_FalAppMeta):
        def __init__(self, *args, **kwargs):
            pass

        @classmethod
        def endpoint(cls, path: str):
            def decorator(func):
                return func

            return decorator

    fal_stub = types.ModuleType("fal")
    fal_stub.App = _FalApp
    fal_stub.endpoint = _FalApp.endpoint
    globals()["fal"] = fal_stub
    import sys

    sys.modules["fal"] = fal_stub

import falmarket.app as fal_app


def test_simulate_endpoint_returns_response(monkeypatch):
    payload = {
        "timeline": [{"step": 1, "timestamp": "2024-01-01T12:00:00", "picked": {"AAPL": {}}, "analyzed_count": 5}],
        "summary": {"cash": 10500.0, "equity": 11000.0, "positions": {"AAPL": 3}, "initial_cash": 10000.0},
        "run_seconds": 1.5,
    }

    monkeypatch.setattr(fal_app, "simulate_trading", lambda **kwargs: json.loads(json.dumps(payload)))
    monkeypatch.setattr(
        fal_app.MarketSimulatorApp,
        "_prefetch_reference_artifacts",
        lambda self: None,
    )
    monkeypatch.setattr(
        fal_app.MarketSimulatorApp,
        "_sync_hyperparams",
        lambda self, direction: None,
    )
    monkeypatch.setattr(
        fal_app.MarketSimulatorApp,
        "_sync_compiled_models",
        lambda self, direction: None,
    )

    app = fal_app.create_app()
    request = fal_app.SimulationRequest(symbols=["AAPL"], steps=1)
    response = app.simulate(request)

    assert response.summary["cash"] == pytest.approx(10500.0)
    assert response.timeline[0]["step"] == 1
    assert response.run_seconds == pytest.approx(1.5)
    assert response.started_at <= datetime.now(timezone.utc)
