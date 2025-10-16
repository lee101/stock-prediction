"""Helpers for querying GPT-5 with retries and caching."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from loguru import logger
from openai import OpenAI

from src.cache import cache
from src.utils import log_time

_api_key = os.getenv("OPENAI_API_KEY")
_client: Optional[OpenAI] = OpenAI(api_key=_api_key) if _api_key else None
MODEL_ID = os.getenv("GPT5_MODEL", "gpt-5")
MAX_ATTEMPTS = int(os.getenv("GPT5_MAX_ATTEMPTS", "3"))
_CACHE_NAMESPACE = "gpt5_structured_v3"

REPROMPT_TEMPLATE = (
    "The previous response failed schema validation because: {error}. "
    "Regenerate the plan strictly using the schema. Return only the JSON."
)

REFUSAL_TEMPLATE = (
    "The earlier reply refused with: {reason}. This request is for a harmless trading simulator "
    "benchmark to evaluate plan quality. Provide the JSON plan using the schema."
)


def _ensure_client() -> OpenAI:
    if _client is None:
        raise EnvironmentError("OPENAI_API_KEY environment variable is required for GPT access.")
    return _client


def _build_messages(system_message: str, user_prompt: str, user_payload_json: Optional[str]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if system_message:
        messages.append({"role": "system", "content": system_message.strip()})
    if user_payload_json:
        try:
            payload = json.loads(user_payload_json)
        except json.JSONDecodeError as exc:
            raise ValueError("user_payload_json must be valid JSON") from exc
        user_content = f"{user_prompt.strip()}\n\nPayload:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    else:
        user_content = user_prompt.strip()
    messages.append({"role": "user", "content": user_content})
    return messages


def _extract_refusal(response: Any) -> Optional[str]:
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", None) == "refusal":
                return getattr(content, "refusal", "")
    return None


def _extract_output_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return text.strip()
    chunks: List[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", None) == "output_text":
                chunks.append(getattr(content, "text", ""))
            elif getattr(content, "type", None) == "text":
                chunks.append(getattr(content, "text", ""))
    combined = "".join(chunks).strip()
    if not combined:
        raise ValueError("GPT model returned an empty response.")
    return combined


def _send_request(client: OpenAI, messages: List[Dict[str, str]], temperature: Optional[float], max_output_tokens: int) -> Any:
    kwargs = {
        "model": MODEL_ID,
        "input": messages,
        "max_output_tokens": max_output_tokens,
        "response_format": {"type": "json_object"},
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    with log_time("GPT-5 query"):
        return client.responses.create(**kwargs)


def query_gpt5_structured(
    *,
    system_message: str,
    user_prompt: str,
    response_schema: Dict[str, Any],
    user_payload_json: Optional[str] = None,
    temperature: Optional[float] = None,
    max_output_tokens: int = 4096,
) -> str:
    schema_key = json.dumps(response_schema, sort_keys=True)
    cache_key = (
        system_message,
        user_prompt,
        user_payload_json or "",
        schema_key,
        None if temperature is None else round(temperature, 4),
        max_output_tokens,
        MODEL_ID,
    )

    cached = cache.get((_CACHE_NAMESPACE, cache_key))
    if cached is not None:
        return cached

    client = _ensure_client()
    messages = _build_messages(system_message, user_prompt, user_payload_json)

    last_error: Optional[str] = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        response = _send_request(client, messages, temperature, max_output_tokens)

        refusal_reason = _extract_refusal(response)
        if refusal_reason:
            logger.warning("GPT refusal: %s", refusal_reason)
            if attempt == MAX_ATTEMPTS:
                raise RuntimeError(f"GPT-5 refused the request: {refusal_reason}")
            messages.append({"role": "user", "content": REFUSAL_TEMPLATE.format(reason=refusal_reason)})
            continue

        raw_text = _extract_output_text(response)
        try:
            json.loads(raw_text)
        except json.JSONDecodeError as exc:
            last_error = str(exc)
            if attempt == MAX_ATTEMPTS:
                raise ValueError("GPT response was not valid JSON") from exc
            snippet = raw_text if len(raw_text) < 4000 else raw_text[:4000] + "..."
            messages.append({"role": "user", "content": f"{REPROMPT_TEMPLATE.format(error=exc)}\n\nPrevious response:\n{snippet}"})
            continue

        cache.set((_CACHE_NAMESPACE, cache_key), raw_text)
        return raw_text

    raise RuntimeError(f"GPT-5 request failed after {MAX_ATTEMPTS} attempts. Last error: {last_error}")
