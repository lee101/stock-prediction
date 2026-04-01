from __future__ import annotations

import importlib.util
import os
import uuid
from pathlib import Path


_ENV_REAL_PATH = Path(__file__).resolve().parent.parent / "env_real.py"
_ALPACA_ENV_KEYS = (
    "ALP_PAPER",
    "PAPER",
    "ALP_KEY_ID",
    "ALP_SECRET_KEY",
    "ALP_KEY_ID_PAPER",
    "ALP_SECRET_KEY_PAPER",
    "ALP_KEY_ID_PROD",
    "ALP_SECRET_KEY_PROD",
    "ALP_ENDPOINT",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "ANTHROPIC_API_KEY",
    "XAI_API_KEY",
    "ZHIPU_API_KEY",
)


def _load_real_env_module(monkeypatch, **env_values):
    for key in _ALPACA_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in env_values.items():
        monkeypatch.setenv(key, value)

    module_name = f"env_real_contract_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, _ENV_REAL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_env_real_defaults_to_paper_mode(monkeypatch):
    env_real = _load_real_env_module(monkeypatch)

    assert env_real.PAPER is True
    assert env_real.ALP_ENDPOINT == "https://paper-api.alpaca.markets"
    assert hasattr(env_real, "ALP_KEY_ID_PAPER")
    assert hasattr(env_real, "ACTIVE_ALP_KEY_ID")
    assert env_real.ACTIVE_ALP_KEY_ID == env_real.ALP_KEY_ID
    assert env_real.ACTIVE_ALP_SECRET_KEY == env_real.ALP_SECRET_KEY
    assert "placeholder" in env_real.ACTIVE_ALP_KEY_ID.lower()
    assert "placeholder" in env_real.ACTIVE_ALP_SECRET_KEY.lower()
    assert "placeholder" in env_real.OPENAI_API_KEY.lower()
    assert "placeholder" in env_real.GEMINI_API_KEY.lower()
    assert "OPENAI_API_KEY" not in os.environ
    assert "GEMINI_API_KEY" not in os.environ


def test_env_real_prefers_documented_alp_paper_alias(monkeypatch):
    env_real = _load_real_env_module(
        monkeypatch,
        PAPER="0",
        ALP_PAPER="1",
        ALP_KEY_ID="paper-key",
        ALP_SECRET_KEY="paper-secret",
        ALP_KEY_ID_PROD="live-key",
        ALP_SECRET_KEY_PROD="live-secret",
    )

    assert env_real.PAPER is True
    assert env_real.ALP_ENDPOINT == "https://paper-api.alpaca.markets"
    assert env_real.ACTIVE_ALP_KEY_ID == "paper-key"
    assert env_real.ACTIVE_ALP_SECRET_KEY == "paper-secret"


def test_env_real_live_mode_uses_prod_active_credentials(monkeypatch):
    env_real = _load_real_env_module(
        monkeypatch,
        ALP_PAPER="0",
        ALP_KEY_ID="paper-key",
        ALP_SECRET_KEY="paper-secret",
        ALP_KEY_ID_PROD="live-key",
        ALP_SECRET_KEY_PROD="live-secret",
    )

    assert env_real.PAPER is False
    assert env_real.ALP_KEY_ID_PAPER == "paper-key"
    assert env_real.ALP_SECRET_KEY_PAPER == "paper-secret"
    assert env_real.ALP_KEY_ID_PROD == "live-key"
    assert env_real.ALP_SECRET_KEY_PROD == "live-secret"
    assert env_real.ACTIVE_ALP_KEY_ID == "live-key"
    assert env_real.ACTIVE_ALP_SECRET_KEY == "live-secret"
    assert env_real.ALP_KEY_ID == "live-key"
    assert env_real.ALP_SECRET_KEY == "live-secret"
    assert env_real.ALP_ENDPOINT == "https://api.alpaca.markets"
