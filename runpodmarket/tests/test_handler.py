from __future__ import annotations

import importlib
import sys
from typing import Dict

import pytest


@pytest.fixture()
def handler_module(monkeypatch):
    """Import runpodmarket.handler with serverless auto-start disabled."""

    monkeypatch.setenv("RUNPODMARKET_DISABLE_SERVERLESS", "1")
    if "runpodmarket.handler" in sys.modules:
        del sys.modules["runpodmarket.handler"]
    module = importlib.import_module("runpodmarket.handler")
    return module


def test_handle_job_runs_simulation(monkeypatch, handler_module):
    recorded: Dict[str, object] = {}
    timeline = [{"step": 1, "picked": {}}]
    summary = {"equity": 123.45}
    results = {"timeline": timeline, "summary": summary, "run_seconds": 9.5}

    def fake_simulate_trading(**kwargs):
        recorded.update(kwargs)
        return results

    downloads = []

    monkeypatch.setattr(handler_module, "simulate_trading", fake_simulate_trading)
    monkeypatch.setattr(handler_module, "_download_model_artifacts", lambda force: downloads.append(("models", force)))
    monkeypatch.setattr(handler_module, "_download_training_data", lambda force: downloads.append(("training", force)))

    response = handler_module.handle_job(
        {
            "symbols": [" tsla ", "MSFT", "TSLA"],
            "steps": 4,
            "step_size": 2,
            "initial_cash": 50_000,
            "top_k": 3,
            "kronos_only": True,
            "compact_logs": False,
        }
    )

    assert response["status"] == "success"
    run = response["run"]
    assert run["summary"] == summary
    assert run["timeline"] == timeline
    assert run["parameters"]["steps"] == 4
    assert run["parameters"]["download_models"] is True
    assert recorded["symbols"] == ["TSLA", "MSFT"]
    assert recorded["step_size"] == 2
    assert ("models", False) in downloads
    assert ("training", False) in downloads


def test_handle_job_force_download(monkeypatch, handler_module):
    monkeypatch.setattr(handler_module, "simulate_trading", lambda **_: {"timeline": [], "summary": {}, "run_seconds": 1.0})
    calls = []
    monkeypatch.setattr(handler_module, "_download_model_artifacts", lambda force: calls.append(("models", force)))
    monkeypatch.setattr(handler_module, "_download_training_data", lambda force: calls.append(("training", force)))

    handler_module.handle_job({"force_download": True})

    assert ("models", True) in calls
    assert ("training", True) in calls


def test_handle_job_validation_error(handler_module):
    response = handler_module.handle_job({"symbols": [], "steps": 0})
    assert response["status"] == "error"
    assert isinstance(response["error"], list)
    assert any(entry["loc"][-1] == "symbols" for entry in response["error"])


def test_handler_wraps_exceptions(monkeypatch, handler_module):
    def boom(_payload):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(handler_module, "handle_job", boom)
    outcome = handler_module.handler({"input": {"symbols": ["AAPL"]}})
    assert outcome["status"] == "error"
    assert outcome["type"] == "RuntimeError"
