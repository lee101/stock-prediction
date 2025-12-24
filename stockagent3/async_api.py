"""Async API calls with retries for Claude (stockagent3)."""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any, Optional

import anthropic
from loguru import logger


def _get_api_key() -> str:
    key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
    if not key:
        try:
            from env_real import ANTHROPIC_API_KEY, CLAUDE_API_KEY
        except ImportError as exc:
            raise RuntimeError("API key required: set ANTHROPIC_API_KEY or CLAUDE_API_KEY") from exc
        key = ANTHROPIC_API_KEY or CLAUDE_API_KEY
    if not key:
        raise RuntimeError("API key required: set ANTHROPIC_API_KEY or CLAUDE_API_KEY")
    return key


DEFAULT_MAX_RETRIES = 8
DEFAULT_RETRY_DELAY = 2.0


async def async_api_call(
    client: anthropic.AsyncAnthropic,
    *,
    model: str,
    system: str,
    user_prompt: str,
    max_tokens: int = 2048,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
    timeout: Optional[float] = None,
) -> str:
    last_error: Optional[str] = None

    for attempt in range(max_retries + 1):
        try:
            request = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user_prompt}],
            )
            if timeout is None:
                response = await request
            else:
                response = await asyncio.wait_for(request, timeout=timeout)
            return response.content[0].text
        except asyncio.TimeoutError:
            last_error = f"Timeout after {timeout}s"
            logger.warning("API call timeout (attempt %d/%d)", attempt + 1, max_retries + 1)
        except anthropic.RateLimitError as exc:
            last_error = str(exc)
            wait_time = retry_delay * (2 ** attempt)
            logger.warning("Rate limited, waiting %ss (attempt %d/%d)", wait_time, attempt + 1, max_retries + 1)
            await asyncio.sleep(wait_time)
        except anthropic.APIError as exc:
            last_error = str(exc)
            logger.warning("API error: %s (attempt %d/%d)", exc, attempt + 1, max_retries + 1)
        except Exception as exc:
            last_error = str(exc)
            logger.warning("Unexpected error: %s (attempt %d/%d)", exc, attempt + 1, max_retries + 1)

        if attempt < max_retries:
            await asyncio.sleep(retry_delay * (2 ** attempt))

    raise RuntimeError(f"API call failed after {max_retries + 1} attempts: {last_error}")


async def async_api_call_with_thinking(
    client: anthropic.AsyncAnthropic,
    *,
    model: str,
    system: str,
    user_prompt: str,
    max_tokens: int = 64000,
    thinking_budget: int = 60000,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
) -> str:
    last_error: Optional[str] = None

    for attempt in range(max_retries + 1):
        try:
            response_text = ""
            async with client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                temperature=1,
                system=system,
                messages=[{"role": "user", "content": user_prompt}],
                thinking={"type": "enabled", "budget_tokens": thinking_budget},
            ) as stream:
                async for event in stream:
                    if hasattr(event, "type") and event.type == "content_block_delta":
                        if hasattr(event.delta, "text"):
                            response_text += event.delta.text
            return response_text
        except anthropic.RateLimitError as exc:
            last_error = str(exc)
            wait_time = retry_delay * (2 ** attempt)
            logger.warning("Rate limited, waiting %ss (attempt %d/%d)", wait_time, attempt + 1, max_retries + 1)
            await asyncio.sleep(wait_time)
        except anthropic.APIError as exc:
            last_error = str(exc)
            logger.warning("API error: %s (attempt %d/%d)", exc, attempt + 1, max_retries + 1)
        except Exception as exc:
            last_error = str(exc)
            logger.warning("Unexpected error: %s (attempt %d/%d)", exc, attempt + 1, max_retries + 1)

        if attempt < max_retries:
            await asyncio.sleep(retry_delay * (2 ** attempt))

    raise RuntimeError(f"Thinking API call failed after {max_retries + 1} attempts: {last_error}")


def parse_json_robust(text: str) -> dict[str, Any]:
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    patterns = [
        r"```json\s*(.*?)\s*```",
        r"```\s*(.*?)\s*```",
        r"\{[\s\S]*\}",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                json_str = match.group(1) if "```" in pattern else match.group(0)
                return json.loads(json_str)
            except json.JSONDecodeError:
                continue

    return {}


__all__ = [
    "_get_api_key",
    "async_api_call",
    "async_api_call_with_thinking",
    "parse_json_robust",
]
