"""Async API calls with retries for Claude."""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

import anthropic
from loguru import logger


def _get_api_key() -> str:
    """Get API key from environment or env_real.py fallback."""
    key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
    if not key:
        # Try importing from env_real.py
        try:
            from env_real import ANTHROPIC_API_KEY, CLAUDE_API_KEY
            key = ANTHROPIC_API_KEY or CLAUDE_API_KEY
            if key:
                return key
        except ImportError:
            pass
        raise RuntimeError("API key required: set ANTHROPIC_API_KEY or CLAUDE_API_KEY")
    return key


# Default retry settings - increased for high-volume parallel requests
DEFAULT_MAX_RETRIES = 8
DEFAULT_RETRY_DELAY = 2.0  # seconds (exponential backoff)
DEFAULT_TIMEOUT = 300.0  # seconds (5 min for thinking)


async def async_api_call(
    client: anthropic.AsyncAnthropic,
    model: str,
    system: str,
    user_prompt: str,
    max_tokens: int = 2048,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
    timeout: float = DEFAULT_TIMEOUT,
) -> str:
    """Make an async API call to Claude with retries.

    Args:
        client: Async Anthropic client
        model: Model ID to use
        system: System prompt
        user_prompt: User message
        max_tokens: Max response tokens
        max_retries: Number of retries on failure
        retry_delay: Delay between retries (exponential backoff)
        timeout: Request timeout in seconds

    Returns:
        Response text from Claude
    """
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            response = await asyncio.wait_for(
                client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user_prompt}],
                ),
                timeout=timeout,
            )
            return response.content[0].text

        except asyncio.TimeoutError:
            last_error = f"Timeout after {timeout}s"
            logger.warning(f"API call timeout (attempt {attempt + 1}/{max_retries + 1})")

        except anthropic.RateLimitError as e:
            last_error = str(e)
            wait_time = retry_delay * (2 ** attempt)
            logger.warning(f"Rate limited, waiting {wait_time}s (attempt {attempt + 1}/{max_retries + 1})")
            await asyncio.sleep(wait_time)

        except anthropic.APIError as e:
            last_error = str(e)
            logger.warning(f"API error: {e} (attempt {attempt + 1}/{max_retries + 1})")

        except Exception as e:
            last_error = str(e)
            logger.warning(f"Unexpected error: {e} (attempt {attempt + 1}/{max_retries + 1})")

        # Exponential backoff
        if attempt < max_retries:
            await asyncio.sleep(retry_delay * (2 ** attempt))

    raise RuntimeError(f"API call failed after {max_retries + 1} attempts: {last_error}")


async def async_api_call_with_thinking(
    client: anthropic.AsyncAnthropic,
    model: str,
    system: str,
    user_prompt: str,
    max_tokens: int = 64000,
    thinking_budget: int = 60000,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
) -> str:
    """Make an async API call with extended thinking.

    Args:
        client: Async Anthropic client
        model: Model ID (should be Opus for thinking)
        system: System prompt
        user_prompt: User message
        max_tokens: Max total tokens (thinking + response)
        thinking_budget: Token budget for thinking
        max_retries: Number of retries on failure
        retry_delay: Delay between retries

    Returns:
        Response text from Claude (excluding thinking blocks)
    """
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            response_text = ""

            # Use streaming for extended thinking
            async with client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                temperature=1,  # Required for thinking
                system=system,  # Actual system prompt
                messages=[{"role": "user", "content": user_prompt}],
                thinking={
                    "type": "enabled",
                    "budget_tokens": thinking_budget
                }
            ) as stream:
                async for event in stream:
                    if hasattr(event, "type") and event.type == "content_block_delta":
                        if hasattr(event.delta, "text"):
                            response_text += event.delta.text

            return response_text

        except anthropic.RateLimitError as e:
            last_error = str(e)
            wait_time = retry_delay * (2 ** attempt)
            logger.warning(f"Rate limited, waiting {wait_time}s (attempt {attempt + 1}/{max_retries + 1})")
            await asyncio.sleep(wait_time)

        except anthropic.APIError as e:
            last_error = str(e)
            logger.warning(f"API error: {e} (attempt {attempt + 1}/{max_retries + 1})")

        except Exception as e:
            last_error = str(e)
            logger.warning(f"Unexpected error: {e} (attempt {attempt + 1}/{max_retries + 1})")

        if attempt < max_retries:
            await asyncio.sleep(retry_delay * (2 ** attempt))

    raise RuntimeError(f"Thinking API call failed after {max_retries + 1} attempts: {last_error}")


def parse_json_robust(text: str) -> dict[str, Any]:
    """Extract JSON from response with robust fallback for truncated responses."""
    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try code blocks
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

    # Fallback: extract individual fields via regex (for truncated JSON)
    result = {}

    # Extract basis point values
    for field in ["entry_bps", "exit_bps", "stop_loss_bps"]:
        match = re.search(rf'"{field}":\s*([+-]?\d+)', text)
        if match:
            result[field] = int(match.group(1))

    # Extract float values
    for field in ["alloc", "confidence", "overall_confidence"]:
        match = re.search(rf'"{field}":\s*(\d+\.?\d*)', text)
        if match:
            result[field] = float(match.group(1))

    # Extract string values
    for field in ["direction", "time_horizon", "reasoning", "rationale"]:
        match = re.search(rf'"{field}":\s*"([^"]*)"', text)
        if match:
            result[field] = match.group(1)

    # Extract allocations block
    alloc_match = re.search(r'"allocations":\s*\{([^}]+(?:\{[^}]*\}[^}]*)*)\}', text)
    if alloc_match:
        alloc_text = alloc_match.group(1)
        allocations = {}
        # Find each symbol allocation
        symbol_pattern = r'"(SYMBOL_[A-Z])":\s*\{([^}]+)\}'
        for sym_match in re.finditer(symbol_pattern, alloc_text):
            sym_name = sym_match.group(1)
            sym_data = sym_match.group(2)
            sym_alloc = {}
            for field in ["alloc", "confidence"]:
                f_match = re.search(rf'"{field}":\s*(\d+\.?\d*)', sym_data)
                if f_match:
                    sym_alloc[field] = float(f_match.group(1))
            dir_match = re.search(r'"direction":\s*"(\w+)"', sym_data)
            if dir_match:
                sym_alloc["direction"] = dir_match.group(1)
            if sym_alloc:
                allocations[sym_name] = sym_alloc
        if allocations:
            result["allocations"] = allocations

    if result:
        return result

    raise ValueError(f"Could not parse JSON from: {text[:300]}")
