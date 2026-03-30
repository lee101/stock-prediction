from __future__ import annotations

from src.trading_server.client import TradingServerClient
from src.trading_server.settings import MAX_WRITER_TTL_SECONDS, TradingServerSettings


def test_claim_writer_uses_shared_writer_ttl_default(monkeypatch) -> None:
    monkeypatch.setenv("TRADING_SERVER_WRITER_TTL_SECONDS", "333")
    client = TradingServerClient(account="paper_test", bot_id="paper_test_v1", session_id="session-a")
    captured: dict[str, object] = {}

    def fake_post(path: str, payload: dict[str, object]) -> dict[str, object]:
        captured["path"] = path
        captured["payload"] = payload
        return {"ok": True}

    client._post = fake_post  # type: ignore[method-assign]

    result = client.claim_writer()

    assert result == {"ok": True}
    assert captured["path"] == "/api/v1/writer/claim"
    assert captured["payload"]["ttl_seconds"] == TradingServerSettings.from_env().writer_ttl_seconds  # type: ignore[index]


def test_heartbeat_writer_explicit_ttl_overrides_shared_default(monkeypatch) -> None:
    monkeypatch.setenv("TRADING_SERVER_WRITER_TTL_SECONDS", "333")
    client = TradingServerClient(account="paper_test", bot_id="paper_test_v1", session_id="session-a")
    captured: dict[str, object] = {}

    def fake_post(path: str, payload: dict[str, object]) -> dict[str, object]:
        captured["path"] = path
        captured["payload"] = payload
        return {"ok": True}

    client._post = fake_post  # type: ignore[method-assign]

    result = client.heartbeat_writer(ttl_seconds=45)

    assert result == {"ok": True}
    assert captured["path"] == "/api/v1/writer/heartbeat"
    assert captured["payload"]["ttl_seconds"] == 45  # type: ignore[index]


def test_claim_writer_clamps_shared_writer_ttl_default_to_contract_max(monkeypatch) -> None:
    monkeypatch.setenv("TRADING_SERVER_WRITER_TTL_SECONDS", "999999")
    client = TradingServerClient(account="paper_test", bot_id="paper_test_v1", session_id="session-a")
    captured: dict[str, object] = {}

    def fake_post(path: str, payload: dict[str, object]) -> dict[str, object]:
        captured["path"] = path
        captured["payload"] = payload
        return {"ok": True}

    client._post = fake_post  # type: ignore[method-assign]

    result = client.claim_writer()

    assert result == {"ok": True}
    assert captured["path"] == "/api/v1/writer/claim"
    assert captured["payload"]["ttl_seconds"] == TradingServerSettings.from_env().writer_ttl_seconds  # type: ignore[index]
    assert captured["payload"]["ttl_seconds"] == MAX_WRITER_TTL_SECONDS  # type: ignore[index]
