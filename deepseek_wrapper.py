"""Convenience helpers for calling DeepSeek chat models with caching and retries."""

from __future__ import annotations

import hashlib
import json
import math
import os
from copy import deepcopy
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from loguru import logger

from src.cache import cache

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


def _estimate_content_tokens(content: Any) -> int:
    if content is None:
        return 0
    if isinstance(content, str):
        return max(1, math.ceil(len(content) / 4))
    if isinstance(content, (bytes, bytearray)):
        return max(1, math.ceil(len(content) / 4))
    if isinstance(content, Mapping):
        text = content.get("text") or content.get("content")
        if isinstance(text, str):
            return _estimate_content_tokens(text)
        return _estimate_content_tokens(json.dumps(content, ensure_ascii=False, sort_keys=True))
    if isinstance(content, Iterable) and not isinstance(content, (str, bytes, bytearray)):
        total = 0
        for item in content:
            total += _estimate_content_tokens(item)
        return total
    return _estimate_content_tokens(str(content))


def _estimate_messages_tokens(messages: Sequence[Mapping[str, Any]]) -> int:
    total = 0
    for message in messages:
        total += _estimate_content_tokens(message.get("content"))
    return total


def _truncate_string(content: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    approx_chars = max_tokens * 4
    if len(content) <= approx_chars:
        return content
    # Keep the tail, which usually contains the freshest context.
    truncated = content[-approx_chars:]
    return f"(truncated to fit context)\n{truncated}"


def _shrink_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    trimmed = [dict(message) for message in messages]
    user_indexes = [idx for idx, message in enumerate(trimmed) if message.get("role") == "user"]
    if not user_indexes:
        return trimmed

    # First strategy: drop the most token-heavy user message when multiple are present.
    if len(user_indexes) > 1:
        largest_idx = max(user_indexes, key=lambda idx: _estimate_content_tokens(trimmed[idx].get("content")))
        trimmed.pop(largest_idx)
        return trimmed

    # Second strategy: truncate the remaining user payload by half.
    target_idx = user_indexes[-1]
    content = trimmed[target_idx].get("content")
    current_tokens = _estimate_content_tokens(content)
    if current_tokens <= 1:
        return trimmed

    new_token_budget = max(1, current_tokens // 2)
    if isinstance(content, str):
        trimmed[target_idx]["content"] = _truncate_string(content, new_token_budget)
        return trimmed
    if isinstance(content, list):
        # Remove earliest list entries until we fit, then truncate the last part if needed.
        new_content: list[Any] = []
        for item in reversed(content):
            new_content.insert(0, item)
            if _estimate_content_tokens(new_content) >= new_token_budget:
                break
        while new_content and _estimate_content_tokens(new_content) > new_token_budget:
            new_content.pop(0)
        trimmed[target_idx]["content"] = new_content
        return trimmed
    trimmed[target_idx]["content"] = _truncate_string(str(content), new_token_budget)
    return trimmed


def _normalize_for_cache(messages: Sequence[Mapping[str, Any]]) -> tuple[tuple[str, str], ...]:
    normalized: list[tuple[str, str]] = []
    for message in messages:
        role = str(message.get("role", ""))
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, Mapping):
                    text = item.get("text") or item.get("content") or ""
                    parts.append(str(text))
                else:
                    parts.append(str(item))
            normalized.append((role, "\n".join(parts)))
        else:
            normalized.append((role, "" if content is None else str(content)))
    return tuple(normalized)


def _response_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        raise RuntimeError("DeepSeek response contained no choices.")
    choice = choices[0]
    message = getattr(choice, "message", choice)
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, Mapping):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
            elif item:
                parts.append(str(item))
        if parts:
            return "\n".join(part.strip() for part in parts if part).strip()
    if content is not None:
        return str(content).strip()
    return ""


def _is_context_error(error: Exception) -> bool:
    message = str(error).lower()
    return "context" in message and ("length" in message or "token" in message)


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
    working_messages: list[MutableMapping[str, Any]] = [dict(message) for message in messages]
    attempts = max(1, max_attempts)

    for attempt in range(1, attempts + 1):
        while _estimate_messages_tokens(working_messages) > MAX_CONTEXT_TOKENS:
            new_messages = _shrink_messages(working_messages)
            if new_messages == working_messages:
                break
            working_messages = new_messages

        cache_key_payload = {
            "model": model,
            "messages": _normalize_for_cache(working_messages),
            "max_output_tokens": max_output_tokens,
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
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                stream=False,
            )
        except BadRequestError as exc:
            if _is_context_error(exc) and attempt < attempts:
                logger.warning("DeepSeek context limit hit; retrying with trimmed messages (attempt %s).", attempt)
                working_messages = _shrink_messages(working_messages)
                continue
            raise
        except APIError as exc:  # pragma: no cover - exercised in integration environments
            if _is_context_error(exc) and attempt < attempts:
                logger.warning("DeepSeek API context error; retrying trimmed payload (attempt %s).", attempt)
                working_messages = _shrink_messages(working_messages)
                continue
            raise

        text = _response_text(response)
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
