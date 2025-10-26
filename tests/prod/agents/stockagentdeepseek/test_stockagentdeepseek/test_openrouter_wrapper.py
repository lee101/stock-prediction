import json
from types import SimpleNamespace

import pytest

import openrouter_wrapper


class DummyCompletions:
    def __init__(self, responses):
        self.responses = responses if isinstance(responses, list) else [responses]
        self.calls = 0
        self.kwargs_list = []

    def create(self, **kwargs):
        self.kwargs_list.append(json.loads(json.dumps(kwargs)))
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        if isinstance(response, Exception):
            raise response
        return response


class DummyClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=DummyCompletions(responses))


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(openrouter_wrapper, "APIError", Exception, raising=False)
    openrouter_wrapper.reset_client()
    yield
    openrouter_wrapper.reset_client()


def test_openrouter_uses_cache(monkeypatch):
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=" hello "))])
    client = DummyClient(response)
    monkeypatch.setattr(openrouter_wrapper, "_ensure_client", lambda: client)

    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "prompt"},
    ]

    first = openrouter_wrapper.call_openrouter_chat(
        messages,
        model="deepseek/deepseek-r1",
        max_tokens=64,
        cache_ttl=60,
    )
    second = openrouter_wrapper.call_openrouter_chat(
        messages,
        model="deepseek/deepseek-r1",
        max_tokens=64,
        cache_ttl=60,
    )

    assert first.strip() == "hello"
    assert second.strip() == "hello"
    assert client.chat.completions.calls == 1


def test_openrouter_fallback(monkeypatch):
    error = Exception("context length exceeded")
    final = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=" fallback ok "))])
    client = DummyClient([error, error, error, final])
    monkeypatch.setattr(openrouter_wrapper, "_ensure_client", lambda: client)

    messages = [{"role": "user", "content": "payload"}]

    output = openrouter_wrapper.call_openrouter_chat(
        messages,
        model="primary-model",
        fallback_models=["fallback-model"],
        max_tokens=128,
        cache_ttl=None,
    )

    assert output.strip() == "fallback ok"
    assert client.chat.completions.calls == 4
    first_kwargs = client.chat.completions.kwargs_list[0]
    assert first_kwargs["model"] == "primary-model"
    fallback_kwargs = client.chat.completions.kwargs_list[-1]
    assert fallback_kwargs["model"] == "fallback-model"
