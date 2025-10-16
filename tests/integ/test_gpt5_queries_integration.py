"""
Live integration checks for the GPT-5 query helpers.

These tests intentionally hit the real GPT-5 API. They are skipped automatically
unless ``OPENAI_API_KEY`` is present in the environment, so CI or local runs
without credentials will fast-skip instead of failing.
"""

from __future__ import annotations

import asyncio
import json
import os

import pytest

from gpt5_queries import query_gpt5_structured, query_to_gpt5_async

OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
pytestmark = pytest.mark.integration


def _require_api_key() -> str:
    api_key = os.getenv(OPENAI_API_KEY_ENV)
    if not api_key:
        pytest.skip(f"{OPENAI_API_KEY_ENV} not set; skipping live GPT-5 integration test.")
    return api_key


@pytest.mark.requires_openai
def test_query_gpt5_structured_live_round_trip() -> None:
    _require_api_key()

    schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "echo": {"type": "string"},
        },
        "required": ["status", "echo"],
    }

    response = query_gpt5_structured(
        system_message="You are a concise integration test bot.",
        user_prompt="Respond with JSON containing status='ok' and echo='success'.",
        response_schema=schema,
        max_output_tokens=64,
    )

    payload = json.loads(response)
    assert payload["status"].lower() == "ok"
    assert "success" in payload["echo"].lower()


@pytest.mark.requires_openai
@pytest.mark.asyncio
async def test_query_to_gpt5_async_live_round_trip() -> None:
    _require_api_key()

    prompt = (
        "Provide a short sentence that contains the word 'integration' and end with a period."
        " Respond with plain text (no JSON)."
    )
    extra = {
        "cache_bypass": True,
        "timeout": 60,
        "max_output_tokens": 128,
    }

    response = await query_to_gpt5_async(
        prompt,
        system_message="You are verifying live GPT-5 access for integration tests.",
        extra_data=extra,
        model=os.getenv("GPT5_MODEL", "gpt-5-mini"),
    )

    assert response is not None
    normalized = response.strip().lower()
    assert "integration" in normalized
    assert normalized.endswith(".")
