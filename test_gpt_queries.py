import asyncio
import copy
import os
import importlib
import sys
import types
from types import SimpleNamespace

import pytest

# Ensure the OpenAI key exists before importing the module under test
os.environ.setdefault("OPENAI_API_KEY", "test-key")

# Provide a lightweight stub for the openai package if it's unavailable.
if "openai" not in sys.modules:
    stub_module = types.ModuleType("openai")

    def _not_implemented(*args, **kwargs):
        raise RuntimeError("Stub OpenAI client cannot be used directly. Provide a monkeypatched client.")

    class _StubAsyncOpenAI:
        def __init__(self, api_key: str):
            self.api_key = api_key
            self.responses = types.SimpleNamespace(create=_not_implemented)

    class _StubOpenAI:
        def __init__(self, api_key: str):
            self.api_key = api_key
            self.responses = types.SimpleNamespace(create=_not_implemented)

    stub_module.AsyncOpenAI = _StubAsyncOpenAI
    stub_module.OpenAI = _StubOpenAI
    sys.modules["openai"] = stub_module

if "diskcache" not in sys.modules:
    diskcache_stub = types.ModuleType("diskcache")

    class _StubCache:
        def __init__(self, *args, **kwargs):
            self._store = {}

        def memoize(self, *args, **kwargs):
            def decorator(func):
                def wrapper(*f_args, **f_kwargs):
                    key = (f_args, tuple(sorted(f_kwargs.items())))
                    if key not in self._store:
                        self._store[key] = func(*f_args, **f_kwargs)
                    return self._store[key]

                wrapper.__cache_key__ = lambda *f_args, **f_kwargs: (f_args, tuple(sorted(f_kwargs.items())))
                return wrapper

            return decorator

        def get(self, key):
            return self._store.get(key)

        def set(self, key, value, expire=None):
            self._store[key] = value

        def clear(self):
            self._store.clear()

    diskcache_stub.Cache = _StubCache
    sys.modules["diskcache"] = diskcache_stub

gpt5_queries = importlib.import_module("gpt5_queries")
from src.cache import cache as global_cache

global_cache.clear()


@pytest.fixture(autouse=True)
def _clear_cache_between_tests():
    global_cache.clear()
    yield
    global_cache.clear()


class DummyResponse:
    def __init__(self, output=None, output_text=None, status="completed", incomplete_reason=None):
        self.output = output or []
        if output_text is not None:
            self.output_text = output_text
        self.status = status
        if incomplete_reason is not None:
            self.incomplete_details = SimpleNamespace(reason=incomplete_reason)
        else:
            self.incomplete_details = None


class DummyResponses:
    def __init__(self, response):
        self._responses = response if isinstance(response, list) else [response]
        self.kwargs = None
        self._call_index = 0
        self.calls = []

    async def create(self, **kwargs):
        self.kwargs = kwargs
        self.calls.append(copy.deepcopy(kwargs))
        idx = self._call_index
        if idx >= len(self._responses):
            idx = len(self._responses) - 1
        self._call_index += 1
        response = self._responses[idx]
        if isinstance(response, Exception):
            raise response
        return response


class DummyClient:
    def __init__(self, response):
        self.responses = DummyResponses(response)


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_query_returns_output_text(monkeypatch):
    dummy_client = DummyClient(DummyResponse(output_text=" 0.1234 "))
    monkeypatch.setattr(gpt5_queries, "gpt5_client", dummy_client)

    result = _run(
        gpt5_queries.query_to_gpt5_async(
            prompt="first prompt",
            extra_data={"max_output_tokens": 16},
            model="gpt-5-mini",
        )
    )

    assert result == "0.1234"
    assert dummy_client.responses.kwargs is not None
    assert dummy_client.responses.kwargs["model"] == "gpt-5-mini"
    assert dummy_client.responses.kwargs["max_output_tokens"] == 16
    assert dummy_client.responses.kwargs["reasoning"] == {"effort": "high"}


def test_query_collects_nested_text(monkeypatch):
    text_piece_one = SimpleNamespace(value="line one")
    text_piece_two = SimpleNamespace(value="line two")
    content_one = SimpleNamespace(text=text_piece_one)
    content_two = SimpleNamespace(text=text_piece_two)
    block = SimpleNamespace(content=[content_one, content_two])
    dummy_client = DummyClient(DummyResponse(output=[block]))
    monkeypatch.setattr(gpt5_queries, "gpt5_client", dummy_client)

    result = _run(
            gpt5_queries.query_to_gpt5_async(
                prompt="second prompt",
                extra_data={"max_output_tokens": 64, "temperature": 0.5, "reasoning_effort": "medium"},
                model="gpt-5-pro",
            )
        )

    assert result == "line one\nline two"
    assert dummy_client.responses.kwargs is not None
    assert "temperature" not in dummy_client.responses.kwargs
    assert dummy_client.responses.kwargs["model"] == "gpt-5-pro"
    assert dummy_client.responses.kwargs["reasoning"] == {"effort": "medium"}


def test_query_retries_on_incomplete_reasoning(monkeypatch):
    incomplete = DummyResponse(status="incomplete", incomplete_reason="max_output_tokens")
    final = DummyResponse(output_text="7.25")
    dummy_client = DummyClient([incomplete, final])
    monkeypatch.setattr(gpt5_queries, "gpt5_client", dummy_client)

    result = _run(
        gpt5_queries.query_to_gpt5_async(
            prompt="retry prompt",
            extra_data={"max_output_tokens": 128},
            model="gpt-5-mini",
        )
    )

    assert result == "7.25"
    calls = dummy_client.responses.calls
    assert len(calls) == 2
    assert calls[0]["max_output_tokens"] == 128
    assert calls[0]["reasoning"]["effort"] == "high"
    assert calls[1]["max_output_tokens"] == 1152
    assert calls[1]["reasoning"]["effort"] == "high"


def test_query_reasoning_can_downgrade_when_unlocked(monkeypatch):
    incomplete = DummyResponse(status="incomplete", incomplete_reason="max_output_tokens")
    final = DummyResponse(output_text="9.01")
    dummy_client = DummyClient([incomplete, final])
    monkeypatch.setattr(gpt5_queries, "gpt5_client", dummy_client)

    result = _run(
        gpt5_queries.query_to_gpt5_async(
            prompt="retry prompt",
            extra_data={"max_output_tokens": 128, "lock_reasoning_effort": False},
            model="gpt-5-mini",
        )
    )

    assert result == "9.01"
    calls = dummy_client.responses.calls
    assert len(calls) == 2
    assert calls[0]["reasoning"]["effort"] == "high"
    assert calls[1]["reasoning"]["effort"] == "medium"


def test_query_retries_on_exception(monkeypatch):
    exception = RuntimeError("network failure")
    final = DummyResponse(output_text="1.23")
    dummy_client = DummyClient([exception, final])
    monkeypatch.setattr(gpt5_queries, "gpt5_client", dummy_client)

    async def _sleep_stub(seconds):
        return None

    monkeypatch.setattr(asyncio, "sleep", _sleep_stub)

    result = _run(
        gpt5_queries.query_to_gpt5_async(
            prompt="exception prompt",
            extra_data={"max_output_tokens": 64, "max_exception_retries": 2, "exception_retry_backoff": 0},
            model="gpt-5-mini",
        )
    )

    assert result == "1.23"
    assert len(dummy_client.responses.calls) == 2


def test_query_uses_disk_cache(monkeypatch):
    first_client = DummyClient(DummyResponse(output_text="cached value"))
    monkeypatch.setattr(gpt5_queries, "gpt5_client", first_client)

    prompt = "cache me prompt"
    extra = {"max_output_tokens": 32}

    first_result = _run(
        gpt5_queries.query_to_gpt5_async(
            prompt=prompt,
            extra_data=extra,
            model="gpt-5-mini",
        )
    )

    assert first_result == "cached value"
    assert len(first_client.responses.calls) == 1

    second_client = DummyClient(DummyResponse(output_text="should not be used"))
    monkeypatch.setattr(gpt5_queries, "gpt5_client", second_client)

    cached_result = _run(
        gpt5_queries.query_to_gpt5_async(
            prompt=prompt,
            extra_data=extra,
            model="gpt-5-mini",
        )
    )

    assert cached_result == "cached value"
    assert len(second_client.responses.calls) == 0


def test_query_cache_bypass(monkeypatch):
    prompt = "bypass prompt"
    extra = {"max_output_tokens": 16, "cache_bypass": True}

    first_client = DummyClient(DummyResponse(output_text="first result"))
    monkeypatch.setattr(gpt5_queries, "gpt5_client", first_client)

    first_run = _run(
        gpt5_queries.query_to_gpt5_async(
            prompt=prompt,
            extra_data=extra,
            model="gpt-5-mini",
        )
    )

    assert first_run == "first result"
    assert len(first_client.responses.calls) == 1

    second_client = DummyClient(DummyResponse(output_text="second result"))
    monkeypatch.setattr(gpt5_queries, "gpt5_client", second_client)

    second_run = _run(
        gpt5_queries.query_to_gpt5_async(
            prompt=prompt,
            extra_data=extra,
            model="gpt-5-mini",
        )
    )

    assert second_run == "second result"
    assert len(second_client.responses.calls) == 1
