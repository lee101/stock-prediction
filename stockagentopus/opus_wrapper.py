"""Convenience helpers for calling Claude Opus 4.5 with caching and retries."""

from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Mapping, MutableMapping, Sequence

from loguru import logger
from pydantic import BaseModel, Field

from src.cache import cache

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore


DEFAULT_MODEL = "claude-opus-4-5-20251101"


# Pydantic models for structured outputs
class ActionType(str, Enum):
    buy = "buy"
    sell = "sell"
    exit = "exit"
    hold = "hold"


class ExecutionSession(str, Enum):
    market_open = "market_open"
    market_close = "market_close"


class TradingInstructionModel(BaseModel):
    """A single trading instruction."""
    symbol: str = Field(description="Stock ticker symbol (e.g. AAPL, MSFT)")
    action: ActionType = Field(description="Trading action: buy, sell, exit, or hold")
    quantity: float = Field(description="Number of shares to trade", ge=0)
    execution_session: ExecutionSession = Field(default=ExecutionSession.market_open, description="When to execute: market_open or market_close")
    entry_price: float | None = Field(default=None, description="Limit entry price")
    exit_price: float | None = Field(default=None, description="Target exit price")
    exit_reason: str | None = Field(default=None, description="Reason for exit")
    notes: str | None = Field(default=None, description="Additional notes")


class TradingPlanModel(BaseModel):
    """A trading plan for a single day."""
    target_date: str = Field(description="Target date in YYYY-MM-DD format")
    instructions: list[TradingInstructionModel] = Field(default_factory=list, description="List of trading instructions")
    risk_notes: str | None = Field(default=None, description="Risk management notes")
    focus_symbols: list[str] = Field(default_factory=list, description="Symbols to focus on")
    stop_trading_symbols: list[str] = Field(default_factory=list, description="Symbols to avoid trading")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


MAX_OUTPUT_TOKENS = int(os.getenv("OPUS_MAX_OUTPUT_TOKENS", "20000"))
MAX_CONTEXT_TOKENS = int(os.getenv("OPUS_CONTEXT_LIMIT", "200000"))
MAX_ATTEMPTS = int(os.getenv("OPUS_MAX_ATTEMPTS", "3"))
DEFAULT_TEMPERATURE = float(os.getenv("OPUS_TEMPERATURE", "1.0"))
_CACHE_NAMESPACE = "opus_chat_v1"
_CACHE_DIR = Path(os.getenv("OPUS_CACHE_DIR", ".opus_cache"))

_client: "anthropic.Anthropic | None" = None


def reset_client() -> None:
    """Reset the cached Anthropic client (used by tests)."""
    global _client
    _client = None


def _ensure_client() -> "anthropic.Anthropic":
    global _client
    if _client is not None:
        return _client
    if anthropic is None:
        raise RuntimeError("The anthropic package is required for Opus calls. pip install anthropic")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set.")
    _client = anthropic.Anthropic(api_key=api_key)
    return _client


def _estimate_tokens(messages: Sequence[Mapping[str, Any]]) -> int:
    """Rough token estimate based on character count."""
    total_chars = sum(
        len(str(msg.get("content", "")))
        for msg in messages
    )
    return total_chars // 4


def _normalize_for_cache(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalize messages for cache key generation."""
    normalized = []
    for msg in messages:
        normalized.append({
            "role": msg.get("role", "user"),
            "content": str(msg.get("content", "")),
        })
    return normalized


def _disk_cache_path(cache_key: str) -> Path:
    """Get the disk cache file path for a given key."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / f"{cache_key}.json"


def _load_disk_cache(cache_key: str) -> str | None:
    """Load cached response from disk."""
    path = _disk_cache_path(cache_key)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get("response")
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _save_disk_cache(cache_key: str, response: str, metadata: dict[str, Any] | None = None) -> None:
    """Save response to disk cache."""
    path = _disk_cache_path(cache_key)
    data = {
        "response": response,
        "metadata": metadata or {},
    }
    try:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as e:
        logger.warning("Failed to save disk cache: %s", e)


def call_opus_chat(
    messages: Sequence[Mapping[str, Any]],
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = MAX_OUTPUT_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    system_prompt: str | None = None,
    cache_ttl: int | None = 3600,
    use_disk_cache: bool = True,
    max_attempts: int = MAX_ATTEMPTS,
) -> str:
    """Send a chat completion request to Claude Opus 4.5 with caching.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.
        model: Model ID to use (default: claude-opus-4-5-20251101).
        max_tokens: Maximum tokens in response.
        temperature: Sampling temperature (default 1.0 for creative thinking).
        system_prompt: Optional system prompt.
        cache_ttl: In-memory cache TTL in seconds (None to disable).
        use_disk_cache: Whether to use on-disk caching for reproducibility.
        max_attempts: Maximum retry attempts.

    Returns:
        The model's text response.
    """
    if not messages:
        raise ValueError("messages must not be empty.")

    working_messages = [dict(msg) for msg in messages]

    cache_key_payload = {
        "model": model,
        "messages": _normalize_for_cache(working_messages),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system_prompt,
    }
    cache_key = hashlib.sha256(
        json.dumps(cache_key_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()

    if use_disk_cache:
        disk_cached = _load_disk_cache(cache_key)
        if disk_cached is not None:
            logger.debug("Opus disk cache hit for key %s", cache_key[:16])
            return disk_cached

    if cache_ttl is not None:
        mem_cached = cache.get((_CACHE_NAMESPACE, cache_key))
        if mem_cached is not None:
            logger.debug("Opus memory cache hit for key %s", cache_key[:16])
            return str(mem_cached)

    client = _ensure_client()

    anthropic_messages = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in working_messages
        if msg.get("role") in ("user", "assistant")
    ]

    for attempt in range(1, max_attempts + 1):
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": anthropic_messages,
            }
            if system_prompt:
                kwargs["system"] = system_prompt

            response = client.messages.create(**kwargs)

            text_parts = []
            for block in response.content:
                if hasattr(block, "text"):
                    text_parts.append(block.text)

            text = "".join(text_parts)
            if not text:
                raise RuntimeError("Opus response did not contain any text content.")

            if cache_ttl is not None and cache_ttl >= 0:
                cache.set((_CACHE_NAMESPACE, cache_key), text, expire=cache_ttl)

            if use_disk_cache:
                _save_disk_cache(cache_key, text, {
                    "model": model,
                    "input_tokens": getattr(response.usage, "input_tokens", 0),
                    "output_tokens": getattr(response.usage, "output_tokens", 0),
                    "stop_reason": response.stop_reason,
                })

            return text

        except Exception as exc:
            logger.warning("Opus API call failed (attempt %d/%d): %s", attempt, max_attempts, exc)
            if attempt >= max_attempts:
                raise RuntimeError(f"Opus chat request failed after {max_attempts} attempts: {exc}") from exc

    raise RuntimeError("Opus chat request exceeded retry attempts without a valid response.")


def call_opus_structured(
    messages: Sequence[Mapping[str, Any]],
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = MAX_OUTPUT_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    system_prompt: str | None = None,
    cache_ttl: int | None = 3600,
    use_disk_cache: bool = True,
    max_attempts: int = MAX_ATTEMPTS,
) -> TradingPlanModel:
    """Call Claude Opus 4.5 with structured outputs to get a validated trading plan.

    Uses the beta structured outputs API for guaranteed schema compliance.
    """
    if not messages:
        raise ValueError("messages must not be empty.")

    working_messages = [dict(msg) for msg in messages]

    cache_key_payload = {
        "model": model,
        "messages": _normalize_for_cache(working_messages),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system_prompt,
        "structured": True,
    }
    cache_key = hashlib.sha256(
        json.dumps(cache_key_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()

    # Check disk cache
    if use_disk_cache:
        disk_cached = _load_disk_cache(cache_key)
        if disk_cached is not None:
            logger.debug("Opus structured disk cache hit for key %s", cache_key[:16])
            return TradingPlanModel.model_validate_json(disk_cached)

    # Check memory cache
    if cache_ttl is not None:
        mem_cached = cache.get((_CACHE_NAMESPACE, cache_key))
        if mem_cached is not None:
            logger.debug("Opus structured memory cache hit for key %s", cache_key[:16])
            return TradingPlanModel.model_validate_json(str(mem_cached))

    client = _ensure_client()

    anthropic_messages = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in working_messages
        if msg.get("role") in ("user", "assistant")
    ]

    for attempt in range(1, max_attempts + 1):
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": anthropic_messages,
                "betas": ["structured-outputs-2025-11-13"],
                "output_format": TradingPlanModel,
            }
            if system_prompt:
                kwargs["system"] = system_prompt

            # Use the beta.messages.parse API for structured outputs
            response = client.beta.messages.parse(**kwargs)

            # Extract the parsed model from the response
            if hasattr(response, "content") and response.content:
                for block in response.content:
                    if hasattr(block, "parsed") and block.parsed is not None:
                        result = block.parsed
                        result_json = result.model_dump_json()

                        if cache_ttl is not None and cache_ttl >= 0:
                            cache.set((_CACHE_NAMESPACE, cache_key), result_json, expire=cache_ttl)

                        if use_disk_cache:
                            _save_disk_cache(cache_key, result_json, {
                                "model": model,
                                "input_tokens": getattr(response.usage, "input_tokens", 0),
                                "output_tokens": getattr(response.usage, "output_tokens", 0),
                                "stop_reason": response.stop_reason,
                            })

                        return result

            raise RuntimeError("Opus structured response did not contain parsed content.")

        except Exception as exc:
            logger.warning("Opus structured API call failed (attempt %d/%d): %s", attempt, max_attempts, exc)
            if attempt >= max_attempts:
                raise RuntimeError(f"Opus structured request failed after {max_attempts} attempts: {exc}") from exc

    raise RuntimeError("Opus structured request exceeded retry attempts without a valid response.")


__all__ = [
    "call_opus_chat",
    "call_opus_structured",
    "reset_client",
    "DEFAULT_MODEL",
    "MAX_OUTPUT_TOKENS",
    "TradingPlanModel",
    "TradingInstructionModel",
    "ActionType",
    "ExecutionSession",
]
