"""Convenience helpers for calling DeepSeek chat models with caching and retries."""

from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from typing import Any, Mapping, MutableMapping, Sequence

from loguru import logger

from src.cache import cache
from llm_utils import (
    estimate_messages_tokens,
    is_context_error,
    normalize_for_cache,
    response_text,
    shrink_messages,
)

try:  # pragma: no cover - falls back to stubs in test environments
    from openai import APIError, BadRequestError, OpenAI  # type: ignore
except Exception:  # pragma: no cover - openai optional for tests
    OpenAI = None  # type: ignore

    class APIError(Exception):
        """Fallback API error when openai package is unavailable."""

    class BadRequestError(APIError):
        """Fallback bad request error."""


DEFAULT_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
MAX_CONTEXT_TOKENS = int(os.getenv("DEEPSEEK_CONTEXT_LIMIT", "32768"))
MAX_ATTEMPTS = int(os.getenv("DEEPSEEK_MAX_ATTEMPTS", "3"))
_CACHE_NAMESPACE = "deepseek_chat_v1"
_OPENROUTER_DEFAULT_MODEL = os.getenv("OPENROUTER_DEEPSEEK_MODEL", "deepseek/deepseek-r1")
_OPENROUTER_FALLBACK_MODELS = tuple(
    filter(
        None,
        json.loads(os.getenv("OPENROUTER_FALLBACK_MODELS", "[]"))
        if os.getenv("OPENROUTER_FALLBACK_MODELS")
        else ["neversleep/llama-3.1-lumimaid-8b", "gryphe/mythomax-l2-13b"],
    )
)
_DISABLE_OPENROUTER = os.getenv("DEEPSEEK_DISABLE_OPENROUTER", "").strip().lower() in {"1", "true", "yes", "on"}

_client: OpenAI | None = None


def reset_client() -> None:
    """Reset the cached OpenAI client (used by tests)."""
    global _client
    _client = None


def _ensure_client() -> OpenAI:
    global _client
    if _client is not None:
        return _client
    if OpenAI is None:  # pragma: no cover - ensures helpful error outside tests
        raise RuntimeError("The openai package is required for DeepSeek calls.")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY environment variable is not set.")
    _client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
    return _client


def _call_openrouter_if_available(
    messages: Sequence[Mapping[str, Any]],
    *,
    model: str,
    max_output_tokens: int,
    temperature: float | None,
    cache_ttl: int | None,
    max_attempts: int,
) -> str | None:
    if _DISABLE_OPENROUTER:
        return None
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_key:
        return None
    try:
        from openrouter_wrapper import call_openrouter_chat_with_fallback
    except ImportError as exc:  # pragma: no cover - fallback if optional dependency missing
        logger.warning("OpenRouter wrapper unavailable (%s); using direct DeepSeek API.", exc)
        return None

    try:
        return call_openrouter_chat_with_fallback(
            messages,
            primary_model=model if model.startswith("deepseek/") else _OPENROUTER_DEFAULT_MODEL,
            fallback_models=_OPENROUTER_FALLBACK_MODELS,
            max_tokens=max_output_tokens,
            temperature=temperature,
            cache_ttl=cache_ttl,
            max_attempts=max_attempts,
        )
    except Exception as exc:
        logger.warning("OpenRouter DeepSeek attempt failed (%s); falling back to direct API.", exc)
        return None


def call_deepseek_chat(
    messages: Sequence[Mapping[str, Any]],
    *,
    model: str = DEFAULT_MODEL,
    max_output_tokens: int = 2048,
    temperature: float | None = None,
    cache_ttl: int | None = 1800,
    max_attempts: int = MAX_ATTEMPTS,
    client: OpenAI | None = None,
) -> str:
    """Send a chat completion request to DeepSeek with disk caching and retries."""
    if not messages:
        raise ValueError("messages must not be empty.")

    openrouter_result = _call_openrouter_if_available(
        messages,
        model=model,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        cache_ttl=cache_ttl,
        max_attempts=max_attempts,
    )
    if openrouter_result is not None:
        return openrouter_result

    working_messages: list[MutableMapping[str, Any]] = [dict(message) for message in messages]
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
            "max_tokens": max_output_tokens,
            "temperature": temperature,
        }
        cache_key = hashlib.sha256(
            json.dumps(cache_key_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

        cached = cache.get((_CACHE_NAMESPACE, cache_key))
        if cached is not None:
            logger.debug("DeepSeek cache hit for key %s", cache_key)
            return str(cached)

        client_instance = client or _ensure_client()
        try:
            response = client_instance.chat.completions.create(  # type: ignore[attr-defined]
                model=model,
                messages=deepcopy(working_messages),
                max_tokens=max_output_tokens,
                temperature=temperature,
                stream=False,
            )
        except BadRequestError as exc:
                if is_context_error(exc) and attempt < attempts:
                    logger.warning("DeepSeek context limit hit; retrying with trimmed messages (attempt %s).", attempt)
                    working_messages = shrink_messages(working_messages)
                    continue
                raise
        except APIError as exc:  # pragma: no cover - exercised in integration environments
            if is_context_error(exc) and attempt < attempts:
                logger.warning("DeepSeek API context error; retrying trimmed payload (attempt %s).", attempt)
                working_messages = shrink_messages(working_messages)
                continue
            raise

        text = response_text(response)
        if not text:
            raise RuntimeError("DeepSeek response did not contain any content.")

        if cache_ttl is not None and cache_ttl >= 0:
            cache.set((_CACHE_NAMESPACE, cache_key), text, expire=cache_ttl)
        return text

    raise RuntimeError("DeepSeek chat request exceeded retry attempts without a valid response.")


__all__ = [
    "call_deepseek_chat",
    "reset_client",
    "DEFAULT_MODEL",
    "DEEPSEEK_BASE_URL",
    "MAX_CONTEXT_TOKENS",
]
