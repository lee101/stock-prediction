"""Helper utilities for calling OpenRouter-hosted models with fallbacks."""

from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from typing import Iterable, Mapping, MutableMapping, Sequence

from loguru import logger

from llm_utils import (
    estimate_messages_tokens,
    is_context_error,
    normalize_for_cache,
    response_text,
    shrink_messages,
)
from src.cache import cache

try:  # pragma: no cover - falls back to stubs in test environments
    from openai import APIError, OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


OPENROUTER_BASE_URL = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "StockTradingSuite")
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", "https://github.com/lee101/stock-trading-suite")
MAX_CONTEXT_TOKENS = int(os.getenv("OPENROUTER_CONTEXT_LIMIT", "32768"))
_CACHE_NAMESPACE = "openrouter_chat_v1"
_DEFAULT_FALLBACK_MODELS: tuple[str, ...] = (
    "neversleep/llama-3.1-lumimaid-8b",
    "gryphe/mythomax-l2-13b",
)

_client: OpenAI | None = None


def reset_client() -> None:
    """Reset the cached OpenRouter client (used in tests)."""
    global _client
    _client = None


def _ensure_client() -> OpenAI:
    global _client
    if _client is not None:
        return _client
    if OpenAI is None:  # pragma: no cover
        raise RuntimeError("The openai package is required for OpenRouter calls.")
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY environment variable is not set.")
    _client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
    return _client


def _call_single_model(
    model: str,
    base_messages: Sequence[Mapping[str, object]],
    *,
    max_tokens: int,
    temperature: float | None,
    stop: Sequence[str] | None,
    cache_ttl: int | None,
    max_attempts: int,
    client: OpenAI | None,
) -> str:
    working_messages: list[MutableMapping[str, object]] = [dict(message) for message in base_messages]
    attempts = max(1, max_attempts)

    for attempt in range(1, attempts + 1):
        while estimate_messages_tokens(working_messages) > MAX_CONTEXT_TOKENS:
            new_messages = shrink_messages(working_messages)
            if new_messages == working_messages:
                break
            working_messages = new_messages

        cache_key_payload = {
            "model": model,
            "messages": normalize_for_cache(working_messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": list(stop) if stop else None,
        }
        cache_key = hashlib.sha256(
            json.dumps(cache_key_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

        cached = cache.get((_CACHE_NAMESPACE, cache_key))
        if cached is not None:
            logger.debug("OpenRouter cache hit for key %s", cache_key)
            return str(cached)

        client_instance = client or _ensure_client()
        try:
            response = client_instance.chat.completions.create(  # type: ignore[attr-defined]
                model=model,
                messages=deepcopy(working_messages),
                max_tokens=max_tokens,
                temperature=temperature,
                stop=list(stop) if stop else None,
                stream=False,
                extra_headers={
                    "HTTP-Referer": OPENROUTER_HTTP_REFERER,
                    "X-Title": OPENROUTER_APP_NAME,
                },
            )
        except APIError as exc:  # pragma: no cover - depends on network behaviour
            if is_context_error(exc) and attempt < attempts:
                logger.warning("OpenRouter context limit hit for %s; retrying trimmed payload.", model)
                working_messages = shrink_messages(working_messages)
                continue
            raise

        text = response_text(response)
        if not text:
            raise RuntimeError(f"OpenRouter response from {model} did not contain content.")

        if cache_ttl is not None and cache_ttl >= 0:
            cache.set((_CACHE_NAMESPACE, cache_key), text, expire=cache_ttl)
        return text

    raise RuntimeError(f"OpenRouter request exhausted retries for model {model}.")


def call_openrouter_chat(
    messages: Sequence[Mapping[str, object]],
    *,
    model: str,
    max_tokens: int = 2048,
    temperature: float | None = None,
    stop: Sequence[str] | None = None,
    cache_ttl: int | None = 1800,
    max_attempts: int = 3,
    fallback_models: Iterable[str] | None = None,
    client: OpenAI | None = None,
) -> str:
    """Call a model via OpenRouter, optionally cascading through fallbacks."""
    if not messages:
        raise ValueError("messages must not be empty.")
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY environment variable is not set.")

    models_chain = [model]
    for fallback in fallback_models or ():
        if fallback and fallback not in models_chain:
            models_chain.append(fallback)

    base_messages = [dict(message) for message in messages]
    last_error: Exception | None = None
    for current_model in models_chain:
        try:
            return _call_single_model(
                current_model,
                base_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                cache_ttl=cache_ttl,
                max_attempts=max_attempts,
                client=client,
            )
        except Exception as exc:  # pragma: no cover - depends on runtime behaviour
            last_error = exc
            logger.warning("OpenRouter model %s failed with error: %s", current_model, exc)
            continue
    if last_error is not None:
        raise last_error
    raise RuntimeError("OpenRouter call failed without attempting any models.")


def call_openrouter_chat_with_fallback(
    messages: Sequence[Mapping[str, object]],
    *,
    primary_model: str,
    fallback_models: Iterable[str] | None = None,
    max_tokens: int = 2048,
    temperature: float | None = None,
    stop: Sequence[str] | None = None,
    cache_ttl: int | None = 1800,
    max_attempts: int = 3,
    client: OpenAI | None = None,
) -> str:
    """Invoke OpenRouter with a sensible default fallback chain."""
    models = list(fallback_models or _DEFAULT_FALLBACK_MODELS)
    return call_openrouter_chat(
        messages,
        model=primary_model,
        fallback_models=models,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop,
        cache_ttl=cache_ttl,
        max_attempts=max_attempts,
        client=client,
    )


__all__ = [
    "call_openrouter_chat",
    "call_openrouter_chat_with_fallback",
    "reset_client",
    "OPENROUTER_BASE_URL",
    "MAX_CONTEXT_TOKENS",
]
