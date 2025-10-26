"""Shared helpers for working with chat-style LLM messages."""

from __future__ import annotations

import json
import math
from typing import Any, Iterable, Mapping, MutableMapping, Sequence


USER_ROLE = "user"


def estimate_content_tokens(content: Any) -> int:
    """Rudimentary token estimator used for trimming oversized prompts."""
    if content is None:
        return 0
    if isinstance(content, str):
        return max(1, math.ceil(len(content) / 4))
    if isinstance(content, (bytes, bytearray)):
        return max(1, math.ceil(len(content) / 4))
    if isinstance(content, Mapping):
        text = content.get("text") or content.get("content")
        if isinstance(text, str):
            return estimate_content_tokens(text)
        return estimate_content_tokens(json.dumps(content, ensure_ascii=False, sort_keys=True))
    if isinstance(content, Iterable) and not isinstance(content, (str, bytes, bytearray)):
        total = 0
        for item in content:
            total += estimate_content_tokens(item)
        return total
    return estimate_content_tokens(str(content))


def estimate_messages_tokens(messages: Sequence[Mapping[str, Any]]) -> int:
    """Estimate the total token usage for a batch of messages."""
    total = 0
    for message in messages:
        total += estimate_content_tokens(message.get("content"))
    return total


def truncate_string(content: str, max_tokens: int) -> str:
    """Trim a string so that it roughly fits under the provided token budget."""
    if max_tokens <= 0:
        return ""
    approx_chars = max_tokens * 4
    if len(content) <= approx_chars:
        return content
    truncated = content[-approx_chars:]
    return f"(truncated to fit context)\n{truncated}"


def shrink_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    user_role: str = USER_ROLE,
) -> list[dict[str, Any]]:
    """Attempt to shrink messages by dropping or truncating user content."""
    trimmed = [dict(message) for message in messages]
    user_indexes = [idx for idx, message in enumerate(trimmed) if message.get("role") == user_role]
    if not user_indexes:
        return trimmed

    if len(user_indexes) > 1:
        largest_idx = max(user_indexes, key=lambda idx: estimate_content_tokens(trimmed[idx].get("content")))
        trimmed.pop(largest_idx)
        return trimmed

    target_idx = user_indexes[-1]
    content = trimmed[target_idx].get("content")
    current_tokens = estimate_content_tokens(content)
    if current_tokens <= 1:
        return trimmed

    new_token_budget = max(1, current_tokens // 2)
    if isinstance(content, str):
        trimmed[target_idx]["content"] = truncate_string(content, new_token_budget)
        return trimmed
    if isinstance(content, list):
        new_content: list[Any] = []
        for item in reversed(content):
            new_content.insert(0, item)
            if estimate_content_tokens(new_content) >= new_token_budget:
                break
        while new_content and estimate_content_tokens(new_content) > new_token_budget:
            new_content.pop(0)
        trimmed[target_idx]["content"] = new_content
        return trimmed
    trimmed[target_idx]["content"] = truncate_string(str(content), new_token_budget)
    return trimmed


def normalize_for_cache(messages: Sequence[Mapping[str, Any]]) -> tuple[tuple[str, str], ...]:
    """Create a cache-friendly representation of messages."""
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


def response_text(response: Any) -> str:
    """Extract text content from an OpenAI chat completion response."""
    choices = getattr(response, "choices", None)
    if not choices:
        raise RuntimeError("LLM response contained no choices.")
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


def is_context_error(error: Exception) -> bool:
    """Return True if the exception looks like a context-length error."""
    message = str(error).lower()
    return "context" in message and ("length" in message or "token" in message)
