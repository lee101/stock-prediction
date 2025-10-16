import asyncio
import hashlib
import json
import math
import os
from collections.abc import Iterable, Sequence
from decimal import Decimal
from typing import Any, Dict, FrozenSet, Iterable as TypingIterable, List, Optional, Tuple

from loguru import logger
from openai import AsyncOpenAI

from src.cache import cache
from src.utils import log_time

try:
    import numpy as np  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - numpy is optional
    np = None  # type: ignore[assignment]

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is required for GPT-5 queries.")

gpt5_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

_CACHE_VERSION = "1"
_CACHE_NAMESPACE = f"gpt5_async:{_CACHE_VERSION}"
_CACHE_DEFAULT_IGNORED_EXTRA_KEYS = frozenset({
    "timeout",
    "max_retries",
    "max_exception_retries",
    "exception_retry_backoff",
    "cache_bypass",
    "cache_expire_seconds",
    "cache_include_keys",
    "cache_exclude_keys",
})


def _coerce_reasoning_effort(effort: Optional[str]) -> str:
    allowed = ("minimal", "low", "medium", "high")
    if effort is None:
        return "high"
    effort_lower = effort.lower()
    return effort_lower if effort_lower in allowed else "high"


def _reasoning_fallback_chain(initial_effort: str) -> Sequence[str]:
    """Return an ordered list of reasoning efforts to try."""
    hierarchy: Dict[str, List[str]] = {
        "high": ["medium", "low", "minimal"],
        "medium": ["low", "minimal"],
        "low": ["minimal"],
        "minimal": [],
    }
    fallbacks = hierarchy.get(initial_effort, [])
    ordered: List[str] = [initial_effort]
    ordered.extend(e for e in fallbacks if e not in ordered)
    return tuple(ordered)


def _extract_text_from_response(response: Any) -> Optional[str]:
    """Attempt to extract text content from a Responses API object."""
    text_out = getattr(response, "output_text", None)
    if isinstance(text_out, str) and text_out.strip():
        return text_out.strip()

    collected_parts: List[str] = []
    try:
        output_blocks = getattr(response, "output", None)
        if isinstance(output_blocks, Iterable):
            for block in output_blocks:
                block_content = getattr(block, "content", None)
                if block_content is None and isinstance(block, dict):
                    block_content = block.get("content")
                if not block_content:
                    continue
                for item in block_content:
                    candidate = None
                    if hasattr(item, "text"):
                        candidate = getattr(item, "text")
                    elif isinstance(item, dict):
                        candidate = item.get("text") or item.get("value")
                    if candidate is None:
                        continue
                    if isinstance(candidate, str):
                        collected_parts.append(candidate)
                        continue
                    # Handle SDK objects exposing nested value/text fields.
                    nested_value = getattr(candidate, "value", None)
                    if isinstance(nested_value, str):
                        collected_parts.append(nested_value)
                        continue
                    nested_text = getattr(candidate, "text", None)
                    if isinstance(nested_text, str):
                        collected_parts.append(nested_text)
                        continue
    except Exception as exc:
        logger.error(f"Failed to traverse GPT-5 response structure: {exc}")

    if collected_parts:
        merged = "\n".join(part for part in collected_parts if isinstance(part, str) and part.strip())
        return merged.strip() if merged.strip() else None
    return None


def _normalize_value(value: Any) -> Any:
    """Convert values into JSON-stable, cache-friendly representations."""
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        return value
    if isinstance(value, Decimal):
        return float(value)
    if np is not None:
        if isinstance(value, np.generic):  # type: ignore[misc]
            return value.item()
        if isinstance(value, np.ndarray):  # type: ignore[misc]
            return value.tolist()
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        try:
            tensor = value.detach().cpu().numpy()
            return tensor.tolist()
        except Exception:  # pragma: no cover - best effort fallback
            return repr(value)
    if isinstance(value, dict):
        normalized_items = []
        for key in sorted(value.keys(), key=lambda item: str(item)):
            normalized_items.append((str(key), _normalize_value(value[key])))
        return {key: val for key, val in normalized_items}
    if isinstance(value, (list, tuple)):
        return [_normalize_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized_items = [_normalize_value(item) for item in value]
        sortable_index = []
        for item in normalized_items:
            if isinstance(item, (dict, list)):
                sortable_index.append(json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
            else:
                sortable_index.append(repr(item))
        sorted_pairs = sorted(zip(sortable_index, normalized_items), key=lambda pair: pair[0])
        return [item for _, item in sorted_pairs]
    if hasattr(value, "tolist"):
        try:
            return value.tolist()  # pragma: no cover - numpy/pandas objects
        except Exception:
            pass
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return _normalize_value(vars(value))
    return repr(value)


def _normalize_stop_sequences(stop_sequences: Optional[TypingIterable[str]]) -> Optional[Tuple[str, ...]]:
    if stop_sequences is None:
        return None
    normalized: List[str] = []
    for item in stop_sequences:
        normalized.append(str(item))
    return tuple(sorted(normalized))


def _normalize_extra_data(
    extra_data: Optional[Dict[str, Any]],
    include_keys: TypingIterable[str] = (),
    exclude_keys: TypingIterable[str] = (),
) -> Dict[str, Any]:
    if not extra_data:
        return {}

    include = {str(key) for key in include_keys}
    exclude = {str(key) for key in exclude_keys}
    normalized: Dict[str, Any] = {}

    for key, value in extra_data.items():
        key_str = str(key)
        if key_str in exclude:
            continue
        if key_str in _CACHE_DEFAULT_IGNORED_EXTRA_KEYS and key_str not in include:
            continue
        normalized[key_str] = _normalize_value(value)

    return normalized


def _build_cache_payload(
    *,
    prompt: str,
    stop_sequences: Optional[TypingIterable[str]],
    extra_data: Optional[Dict[str, Any]],
    prefill: Optional[str],
    system_message: Optional[str],
    model: str,
    include_keys: TypingIterable[str],
    exclude_keys: TypingIterable[str],
) -> Dict[str, Any]:
    return {
        "version": _CACHE_VERSION,
        "prompt": prompt.strip() if isinstance(prompt, str) else str(prompt),
        "system_message": system_message.strip() if isinstance(system_message, str) else system_message,
        "prefill": prefill.strip() if isinstance(prefill, str) else prefill,
        "stop_sequences": _normalize_stop_sequences(stop_sequences),
        "extra": _normalize_extra_data(extra_data, include_keys, exclude_keys),
        "model": model,
    }


def _build_cache_key(payload: Dict[str, Any]) -> str:
    serialized = json.dumps(_normalize_value(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return f"{_CACHE_NAMESPACE}:{digest}"


def _coerce_cache_ttl(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if value <= 0:
            return 0
        return int(value)
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        logger.warning("Ignoring invalid cache_expire_seconds value: {!r}", value)
        return None
    if parsed <= 0:
        return 0
    return int(parsed)


async def query_to_gpt5_async(
    prompt: str,
    stop_sequences: Optional[FrozenSet[str]] = None,
    extra_data: Optional[Dict[str, Any]] = None,
    prefill: Optional[str] = None,
    system_message: Optional[str] = None,
    model: str = "gpt-5-mini",
) -> Optional[str]:
    """Async GPT-5 query using the OpenAI Responses API with caching."""
    if extra_data and not isinstance(extra_data, dict):
        extra_data = dict(extra_data)  # type: ignore[arg-type]
    else:
        extra_data = extra_data or {}
    extra_data = dict(extra_data)

    include_keys_raw = extra_data.pop("cache_include_keys", ())
    exclude_keys_raw = extra_data.pop("cache_exclude_keys", ())
    cache_bypass = bool(extra_data.pop("cache_bypass", False))
    cache_expire_raw = extra_data.pop("cache_expire_seconds", None)

    include_keys: Tuple[str, ...] = tuple(str(item) for item in include_keys_raw) if include_keys_raw else ()
    exclude_keys: Tuple[str, ...] = tuple(str(item) for item in exclude_keys_raw) if exclude_keys_raw else ()
    cache_ttl = _coerce_cache_ttl(cache_expire_raw)
    if cache_ttl == 0:
        cache_bypass = True

    prompt_clean = prompt.strip()
    system_clean = system_message.strip() if isinstance(system_message, str) else None
    prefill_clean = prefill.strip() if isinstance(prefill, str) else None

    cache_key: Optional[str] = None
    if not cache_bypass:
        payload = _build_cache_payload(
            prompt=prompt_clean,
            stop_sequences=stop_sequences,
            extra_data=extra_data,
            prefill=prefill_clean,
            system_message=system_clean,
            model=model,
            include_keys=include_keys,
            exclude_keys=exclude_keys,
        )
        cache_key = _build_cache_key(payload)
        try:
            cached_response = cache.get(cache_key)
        except Exception as exc:  # pragma: no cover - cache backend failure paths
            logger.warning("Failed to read GPT-5 cache entry: {}", exc)
            cached_response = None
        if cached_response is not None:
            logger.info("GPT-5 cache hit for digest {}", cache_key[-12:])
            return cached_response

    timeout = extra_data.get("timeout", 45)

    messages: List[Dict[str, str]] = []
    if system_clean:
        messages.append({
            "role": "system",
            "content": system_clean,
        })

    messages.append({
        "role": "user",
        "content": prompt_clean,
    })

    if prefill_clean:
        messages.append({
            "role": "assistant",
            "content": prefill_clean,
        })

    base_max_tokens = int(extra_data.get("max_output_tokens", 8192))
    if base_max_tokens <= 0:
        raise ValueError("max_output_tokens must be a positive integer.")

    max_token_cap = int(extra_data.get("max_output_tokens_cap", 60000))
    if max_token_cap < base_max_tokens:
        max_token_cap = base_max_tokens

    token_growth_factor = float(extra_data.get("token_growth_factor", 2.0))
    if token_growth_factor < 1.0:
        token_growth_factor = 1.0

    min_token_increment = int(extra_data.get("min_token_increment", 1024))
    if min_token_increment < 0:
        min_token_increment = 0

    lock_reasoning_effort = bool(extra_data.get("lock_reasoning_effort", True))

    request_args: Dict[str, Any] = {
        "model": model,
        "input": messages,
        "max_output_tokens": base_max_tokens,
    }

    # OpenAI GPT-5 endpoints reject unsupported parameters like temperature/top_p.
    # Silently drop them while logging for visibility.
    if "temperature" in extra_data:
        logger.warning("Ignoring unsupported temperature setting for GPT-5: {}", extra_data["temperature"])
    if "top_p" in extra_data:
        logger.warning("Ignoring unsupported top_p setting for GPT-5: {}", extra_data["top_p"])

    reasoning_effort = _coerce_reasoning_effort(extra_data.get("reasoning_effort"))
    reasoning_chain = _reasoning_fallback_chain(reasoning_effort)
    if lock_reasoning_effort and reasoning_chain:
        reasoning_chain = (reasoning_chain[0],)
    current_reasoning_index = 0
    request_args["reasoning"] = {"effort": reasoning_chain[current_reasoning_index]}

    if stop_sequences:
        request_args["stop"] = list(stop_sequences)

    max_retries = int(extra_data.get("max_retries", 6))
    max_exception_retries = int(extra_data.get("max_exception_retries", 3))
    exception_retry_backoff = float(extra_data.get("exception_retry_backoff", 5.0))
    if exception_retry_backoff < 0:
        exception_retry_backoff = 0.0
    attempt = 0
    exception_attempt = 0
    response = None  # type: ignore[assignment]
    have_adjusted_reasoning = False

    with log_time("GPT-5 async query"):
        while True:
            try:
                response = await asyncio.wait_for(
                    gpt5_client.responses.create(**request_args),
                    timeout=timeout,
                )
            except Exception as exc:
                logger.error("Error querying GPT-5: {!r}", exc)
                if exception_attempt >= max_exception_retries:
                    return None
                exception_attempt += 1
                backoff_seconds = exception_retry_backoff * exception_attempt
                if backoff_seconds > 0:
                    logger.info(
                        "Retrying GPT-5 request after exception in {:.2f}s (attempt {}/{}).",
                        backoff_seconds,
                        exception_attempt,
                        max_exception_retries,
                    )
                    await asyncio.sleep(backoff_seconds)
                continue

            text_out = _extract_text_from_response(response)
            if text_out:
                logger.info("GPT-5 response text:\n{}", text_out)
                if cache_key and not cache_bypass:
                    expire_param = None if cache_ttl is None else cache_ttl
                    try:
                        cache.set(cache_key, text_out, expire=expire_param)
                        logger.info("GPT-5 cache store for digest {}", cache_key[-12:])
                    except Exception as exc:
                        logger.warning("Failed to persist GPT-5 cache entry: {}", exc)
                return text_out

            status = getattr(response, "status", None)
            incomplete = getattr(response, "incomplete_details", None)
            reason = getattr(incomplete, "reason", None) if incomplete else None

            if status == "incomplete" and reason == "max_output_tokens" and attempt < max_retries:
                attempt += 1
                current_tokens = int(request_args["max_output_tokens"])
                proposed = max(
                    current_tokens + min_token_increment,
                    int(current_tokens * token_growth_factor),
                )
                new_limit = min(proposed, max_token_cap)
                if new_limit == current_tokens and current_tokens >= max_token_cap:
                    logger.warning(
                        "GPT-5 response truncated at token cap {} with no extractable text; cannot raise further.",
                        current_tokens,
                    )
                else:
                    logger.info(
                        "GPT-5 response truncated at {} tokens. Retrying with max_output_tokens={}.",
                        current_tokens,
                        new_limit,
                    )
                    request_args["max_output_tokens"] = new_limit
                if not lock_reasoning_effort and (current_reasoning_index + 1) < len(reasoning_chain):
                    current_reasoning_index += 1
                    new_effort = reasoning_chain[current_reasoning_index]
                    request_args["reasoning"]["effort"] = new_effort
                    have_adjusted_reasoning = True
                    logger.info("Reducing GPT-5 reasoning effort to '{}' after repeated truncation.", new_effort)
                elif not lock_reasoning_effort and not have_adjusted_reasoning and reasoning_chain[-1] != "minimal":
                    # As a safety valve, fall back to minimal effort even if the chain did not include it.
                    request_args["reasoning"]["effort"] = "minimal"
                    have_adjusted_reasoning = True
                    logger.info("Forcing GPT-5 reasoning effort to 'minimal' after repeated truncation.")
                continue
            break

    logger.error("GPT-5 response contained no extractable text.")
    try:
        logger.debug("GPT-5 raw response repr: {}", repr(response))
    except Exception:
        pass
    return None
