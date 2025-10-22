import json
from types import SimpleNamespace

import pytest

import deepseek_wrapper
from src.cache import cache


@pytest.fixture(autouse=True)
def _reset_cache():
    cache.clear()
    yield
    cache.clear()
    deepseek_wrapper.reset_client()


class DummyCompletions:
    def __init__(self, responses):
        self.responses = responses if isinstance(responses, list) else [responses]
        self.kwargs_list = []
        self.calls = 0

    def create(self, **kwargs):
        self.kwargs_list.append(json.loads(json.dumps(kwargs)))
        index = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        result = self.responses[index]
        if isinstance(result, Exception):
            raise result
        return result


class DummyClient:
    def __init__(self, responses):
        self.completions = DummyCompletions(responses)
        self.chat = SimpleNamespace(completions=self.completions)


def test_call_deepseek_chat_returns_stripped_text_and_caches() -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="  plan payload  "))]
    )
    client = DummyClient(response)
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "Generate a plan"},
    ]

    first = deepseek_wrapper.call_deepseek_chat(
        messages,
        client=client,
        cache_ttl=30,
        max_output_tokens=128,
    )
    second = deepseek_wrapper.call_deepseek_chat(
        messages,
        client=client,
        cache_ttl=30,
        max_output_tokens=128,
    )

    assert first == "plan payload"
    assert second == "plan payload"
    assert client.completions.calls == 1


def test_call_deepseek_chat_retries_after_context_error() -> None:
    error = deepseek_wrapper.BadRequestError("maximum context length exceeded")
    final_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="trimmed plan"))]
    )
    client = DummyClient([error, final_response])

    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "instruction payload"},
        {
            "role": "user",
            "content": "heavy payload " + "X" * (deepseek_wrapper.MAX_CONTEXT_TOKENS),
        },
    ]

    result = deepseek_wrapper.call_deepseek_chat(
        messages,
        client=client,
        cache_ttl=None,
        max_output_tokens=128,
    )

    assert result == "trimmed plan"
    assert client.completions.calls == 2

    first_call_messages = client.completions.kwargs_list[0]["messages"]
    second_call_messages = client.completions.kwargs_list[1]["messages"]

    assert len(first_call_messages) == 3
    assert len(second_call_messages) == 2
    assert second_call_messages[0]["role"] == "system"
    assert second_call_messages[1]["content"] == "instruction payload"
