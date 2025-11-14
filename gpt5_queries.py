"""Helpers for querying GPT-5 with retries, caching, and structured outputs."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from types import ModuleType
from decimal import Decimal
from typing import Any, Dict, FrozenSet, Iterable as TypingIterable, List, Optional, Tuple

from loguru import logger
from openai import AsyncOpenAI, OpenAI

from src.cache import cache
from src.utils import log_time

np: ModuleType | None = None

try:
    import numpy as np  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - numpy is optional
    np = None

jsonschema: ModuleType | None = None
_JsonSchemaValidationError: type[Exception] | None = None

try:  # pragma: no cover - jsonschema is optional but strongly recommended
    import jsonschema
    from jsonschema import ValidationError as _JsonSchemaValidationError
except ImportError:  # pragma: no cover
    jsonschema = None
    _JsonSchemaValidationError = None

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
_ASYNC_CLIENT: Optional[AsyncOpenAI] = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
_SYNC_CLIENT: Optional[OpenAI] = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
gpt5_client: Optional[AsyncOpenAI] = _ASYNC_CLIENT

MODEL_ID = os.getenv("GPT5_MODEL", "gpt-5")
MAX_ATTEMPTS = int(os.getenv("GPT5_MAX_ATTEMPTS", "3"))
_STRUCTURED_CACHE_NAMESPACE = "gpt5_structured_v3"
_STRUCTURED_MAX_TOKEN_CAP = 16384
STRUCTURED_DEFAULT_REASONING = os.getenv("GPT5_STRUCTURED_REASONING", "high")

REPROMPT_TEMPLATE = (
    "The previous response failed schema validation because: {error}. "
    "Regenerate the plan strictly using the schema. Return only the JSON."
)

SCHEMA_REPROMPT_TEMPLATE = (
    "The prior JSON violated the required schema: {error}. Update the specified fields and resend JSON only."
)

REFUSAL_TEMPLATE = (
    "The earlier reply refused with: {reason}. This request is for a harmless trading simulator "
    "benchmark to evaluate plan quality. Provide the JSON plan using the schema."
)


@dataclass(frozen=True)
class ValidationIssue:
    """Structured description of a schema validation failure."""

    path: Tuple[Any, ...]
    path_display: str
    message: str
    fix_hint: str
    issue_type: str
    value_snippet: Optional[str] = None
    field: Optional[str] = None


_RE_REQUIRED_PROPERTY = re.compile(r"""['"](?P<field>[^'"]+)['"] is a required property""")
_RE_ADDITIONAL_PROPERTIES = re.compile(
    r"Additional properties are not allowed \((?P<props>.+?) (?:was|were) unexpected\)"
)


def _humanize_location(path_display: str) -> str:
    return "payload" if not path_display or path_display == "<root>" else path_display


def _summarize_fragment(fragment: Any, *, max_chars: int = 400) -> Optional[str]:
    if fragment is None:
        return "null"
    try:
        if isinstance(fragment, (dict, list)):
            text = json.dumps(fragment, ensure_ascii=False, indent=2)
        else:
            text = json.dumps(fragment, ensure_ascii=False)
    except (TypeError, ValueError):
        text = repr(fragment)
    text = text.strip()
    if not text:
        return None
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


def _issues_from_jsonschema_error(error: Any) -> List[ValidationIssue]:
    path_tokens = tuple(getattr(error, "absolute_path", ()) or getattr(error, "path", ()))
    location = _format_json_path(path_tokens)
    validator = getattr(error, "validator", "")
    message = getattr(error, "message", "Schema violation")
    instance = getattr(error, "instance", None)
    issues: List[ValidationIssue] = []

    if validator == "required":
        required = getattr(error, "validator_value", ())
        missing: List[str] = []
        if isinstance(instance, dict):
            missing = [field for field in required if field not in instance]
        if not missing:
            match = _RE_REQUIRED_PROPERTY.search(message)
            if match:
                missing = [match.group("field")]
        if not missing:
            missing = list(required) if isinstance(required, (list, tuple, set)) else []
        parent_display = _humanize_location(location)
        for field in missing or ("<missing>",):
            issue_path = path_tokens + (field,)
            issue_location = _format_json_path(issue_path)
            issues.append(
                ValidationIssue(
                    path=issue_path,
                    path_display=issue_location,
                    message=f"{parent_display} is missing {field}",
                    fix_hint=f"Add '{field}' under {parent_display} with a value that satisfies the schema.",
                    issue_type="missing_required",
                    value_snippet=_summarize_fragment(instance),
                    field=field,
                )
            )
        return issues

    if validator == "type":
        expected_raw = getattr(error, "validator_value", None)
        if isinstance(expected_raw, (list, tuple, set, frozenset)):
            expected_display = ", ".join(sorted(str(val) for val in expected_raw))
        else:
            expected_display = str(expected_raw)
        actual_type = type(instance).__name__
        issue_type = "null_disallowed" if instance is None else "type_mismatch"
        human_location = _humanize_location(location)
        issues.append(
            ValidationIssue(
                path=path_tokens,
                path_display=location,
                message=f"{human_location} must be of type {expected_display}; received {actual_type}",
                fix_hint=(
                    f"Replace null with a {expected_display} value."
                    if instance is None
                    else f"Provide a value matching type {expected_display}."
                ),
                issue_type=issue_type,
                value_snippet=_summarize_fragment(instance),
            )
        )
        return issues

    if validator == "enum":
        allowed_values = getattr(error, "validator_value", ())
        allowed_display = ", ".join(str(val) for val in allowed_values) if allowed_values else "<enum>"
        human_location = _humanize_location(location)
        issues.append(
            ValidationIssue(
                path=path_tokens,
                path_display=location,
                message=f"{human_location} must be one of: {allowed_display}",
                fix_hint=f"Choose one of the allowed options: {allowed_display}.",
                issue_type="enum",
                value_snippet=_summarize_fragment(instance),
            )
        )
        return issues

    if validator in {"minimum", "exclusiveMinimum"}:
        comparator = ">=" if validator == "minimum" else ">"
        threshold = getattr(error, "validator_value", None)
        human_location = _humanize_location(location)
        issues.append(
            ValidationIssue(
                path=path_tokens,
                path_display=location,
                message=f"{human_location} must be {comparator} {threshold}",
                fix_hint=f"Ensure the value is {comparator} {threshold}.",
                issue_type="range",
                value_snippet=_summarize_fragment(instance),
            )
        )
        return issues

    if validator == "additionalProperties":
        human_location = _humanize_location(location)
        props_match = _RE_ADDITIONAL_PROPERTIES.search(message)
        raw_props = props_match.group("props") if props_match else ""
        extras = [prop.strip(" '\"") for prop in raw_props.split(",")] if raw_props else []
        extras_display = ", ".join(extras) if extras else message
        issues.append(
            ValidationIssue(
                path=path_tokens,
                path_display=location,
                message=f"{human_location} includes unexpected properties: {extras_display}",
                fix_hint="Remove the properties that are not part of the schema.",
                issue_type="additional_properties",
                value_snippet=_summarize_fragment(instance),
            )
        )
        return issues

    human_location = _humanize_location(location)
    issues.append(
        ValidationIssue(
            path=path_tokens,
            path_display=location,
            message=f"{message} (at {human_location})",
            fix_hint="Adjust the field to satisfy the schema requirement.",
            issue_type="schema",
            value_snippet=_summarize_fragment(instance),
        )
    )
    return issues


def collect_structured_payload_issues(payload: Dict[str, Any], schema: Dict[str, Any]) -> List[ValidationIssue]:
    """Return a stable list of schema/business-rule violations for a payload."""
    issues_map: Dict[Tuple[str, str], ValidationIssue] = {}

    def add_issue(issue: ValidationIssue, *, overwrite: bool = False) -> None:
        key = (issue.path_display, issue.issue_type)
        if overwrite and key in issues_map:
            del issues_map[key]
        if key not in issues_map:
            issues_map[key] = issue

    if jsonschema is not None:
        validator = jsonschema.Draft202012Validator(schema)
        sorted_errors = sorted(validator.iter_errors(payload), key=lambda err: list(getattr(err, "path", ())))
        for error in sorted_errors:
            for issue in _issues_from_jsonschema_error(error):
                add_issue(issue)

    plan_prefix: Tuple[Any, ...] = ()
    plan_display_prefix = ""
    plan_candidate = payload.get("plan") if isinstance(payload, dict) else None
    if isinstance(plan_candidate, dict):
        plan = plan_candidate
        plan_prefix = ("plan",)
        plan_display_prefix = "plan."
    else:
        plan = payload

    if not isinstance(plan, dict):
        display = "plan" if plan_display_prefix else "<root>"
        add_issue(
            ValidationIssue(
                path=plan_prefix or ("<root>",),
                path_display=display,
                message=f"{display} must be an object",
                fix_hint="Return the trading plan as an object with target_date, instructions, and optional metadata.",
                issue_type="business_structure",
                value_snippet=_summarize_fragment(plan),
            ),
            overwrite=True,
        )
        return list(issues_map.values())

    instructions = plan.get("instructions")
    if not isinstance(instructions, list):
        path = plan_prefix + ("instructions",)
        display = f"{plan_display_prefix}instructions"
        add_issue(
            ValidationIssue(
                path=path,
                path_display=display,
                message=f"{display} must be an array",
                fix_hint="Provide instructions as an array of instruction objects.",
                issue_type="business_structure",
                value_snippet=_summarize_fragment(instructions),
            ),
            overwrite=True,
        )
        return list(issues_map.values())

    for idx, instruction in enumerate(instructions):
        if not isinstance(instruction, dict):
            path = plan_prefix + ("instructions", idx)
            display = f"{plan_display_prefix}instructions[{idx}]"
            add_issue(
                ValidationIssue(
                    path=path,
                    path_display=display,
                    message=f"{display} must be an object",
                    fix_hint="Ensure each instruction is an object with symbol/action/quantity/etc.",
                    issue_type="business_structure",
                    value_snippet=_summarize_fragment(instruction),
                ),
                overwrite=True,
            )
            continue

        quantity = instruction.get("quantity")
        if quantity is None:
            path = plan_prefix + ("instructions", idx, "quantity")
            display = f"{plan_display_prefix}instructions[{idx}].quantity"
            add_issue(
                ValidationIssue(
                    path=path,
                    path_display=display,
                    message=f"{display} is missing quantity",
                    fix_hint="Add a numeric quantity matching the planned action.",
                    issue_type="missing_required",
                    value_snippet=_summarize_fragment(instruction),
                    field="quantity",
                ),
                overwrite=True,
            )
            continue

        try:
            quantity_val = float(quantity)
        except (TypeError, ValueError):
            path = plan_prefix + ("instructions", idx, "quantity")
            display = f"{plan_display_prefix}instructions[{idx}].quantity"
            add_issue(
                ValidationIssue(
                    path=path,
                    path_display=display,
                    message=f"{display} must be numeric",
                    fix_hint="Set quantity to a numeric value (integer or float).",
                    issue_type="type_mismatch",
                    value_snippet=_summarize_fragment(quantity),
                ),
                overwrite=True,
            )
            continue

        action_raw = instruction.get("action")
        action = str(action_raw).lower() if action_raw is not None else ""
        if action in {"buy", "sell"} and quantity_val <= 0.0:
            path = plan_prefix + ("instructions", idx, "quantity")
            display = f"{plan_display_prefix}instructions[{idx}].quantity"
            add_issue(
                ValidationIssue(
                    path=path,
                    path_display=display,
                    message=f"{display} must be greater than zero for action '{action}'",
                    fix_hint="Use a strictly positive quantity when action is buy/sell.",
                    issue_type="business_rule",
                    value_snippet=_summarize_fragment(instruction),
                ),
                overwrite=True,
            )

    return list(issues_map.values())


def _build_schema_retry_message(
    issues: Sequence[ValidationIssue],
    *,
    raw_text: str,
    max_snippet_chars: int = 4000,
    max_issues: int = 6,
) -> str:
    if not issues:
        snippet = raw_text if len(raw_text) <= max_snippet_chars else raw_text[:max_snippet_chars] + "..."
        return f"{SCHEMA_REPROMPT_TEMPLATE.format(error='unknown issues')}\n\nPrevious response:\n{snippet}"

    lead = SCHEMA_REPROMPT_TEMPLATE.format(error=issues[0].message)
    lines: List[str] = []
    for issue in issues[:max_issues]:
        location = _humanize_location(issue.path_display)
        entry = f"- {location}: {issue.message}"
        if issue.fix_hint and issue.fix_hint not in issue.message:
            entry = f"- {location}: {issue.message}. Fix: {issue.fix_hint}"
        lines.append(entry)

    message = (
        f"{lead}\n\n"
        "Only adjust the fields listed below and resend the complete JSON payload. "
        "All other fields should remain unchanged unless a change is required for consistency.\n"
        "Issues detected:\n"
        + "\n".join(lines)
    )

    fragments: List[str] = []
    for issue in issues[:3]:
        if issue.value_snippet:
            location = _humanize_location(issue.path_display)
            fragments.append(f"{location} sample:\n{issue.value_snippet}")
    if fragments:
        message += "\n\nContext:\n" + "\n\n".join(fragments)

    snippet = raw_text if len(raw_text) <= max_snippet_chars else raw_text[:max_snippet_chars] + "..."
    message += f"\n\nPrevious response:\n{snippet}"
    return message


def _ensure_sync_client() -> OpenAI:
    if _SYNC_CLIENT is None:
        raise EnvironmentError("OPENAI_API_KEY environment variable is required for GPT access.")
    return _SYNC_CLIENT


def _ensure_async_client() -> AsyncOpenAI:
    if _ASYNC_CLIENT is None:
        raise EnvironmentError("OPENAI_API_KEY environment variable is required for GPT access.")
    return _ASYNC_CLIENT


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


def _schema_digest(schema: Dict[str, Any]) -> str:
    serialized = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()[:16]


def _normalize_schema(node: Any) -> Any:
    if isinstance(node, dict):
        normalized: Dict[str, Any] = {}
        for key, value in node.items():
            normalized[key] = _normalize_schema(value)
        if normalized.get("type") == "object" and "additionalProperties" not in normalized:
            normalized["additionalProperties"] = False
        return normalized
    if isinstance(node, list):
        return [_normalize_schema(item) for item in node]
    return node


def _send_structured_request(
    client: OpenAI,
    messages: List[Dict[str, str]],
    temperature: Optional[float],
    max_output_tokens: int,
    response_schema: Dict[str, Any],
    reasoning_effort: Optional[str],
) -> Any:
    normalized_schema = _normalize_schema(response_schema)
    schema_name = f"schema_{_schema_digest(normalized_schema)}"
    text_format: Dict[str, Any] = {
        "type": "json_schema",
        "name": schema_name,
        "schema": normalized_schema,
    }
    kwargs = {
        "model": MODEL_ID,
        "input": messages,
        "max_output_tokens": max_output_tokens,
        "text": {"format": text_format},
    }
    effort = _coerce_reasoning_effort(reasoning_effort or STRUCTURED_DEFAULT_REASONING)
    kwargs["reasoning"] = {"effort": effort}
    if temperature is not None:
        kwargs["temperature"] = temperature
    with log_time("GPT-5 query"):
        return client.responses.create(**kwargs)


def _format_json_path(path: TypingIterable[Any]) -> str:
    tokens = list(path)
    if not tokens:
        return "<root>"
    formatted: List[str] = []
    for token in tokens:
        if isinstance(token, int):
            formatted.append(f"[{token}]")
        else:
            formatted.append(f".{token}")
    joined = "".join(formatted)
    if joined.startswith("."):
        joined = joined[1:]
    return joined or "<root>"


def validate_structured_payload(payload: Dict[str, Any], schema: Dict[str, Any]) -> Optional[str]:
    """
    Validate a structured GPT payload against the supplied JSON schema and business rules.

    Returns
    -------
    Optional[str]
        None when validation succeeds, otherwise a human-readable error string that pinpoints the violation.
    """
    issues = collect_structured_payload_issues(payload, schema)
    return issues[0].message if issues else None


def query_gpt5_structured(
    *,
    system_message: str,
    user_prompt: str,
    response_schema: Dict[str, Any],
    user_payload_json: Optional[str] = None,
    temperature: Optional[float] = None,
    max_output_tokens: int = 4096,
    reasoning_effort: Optional[str] = None,
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
        (reasoning_effort or STRUCTURED_DEFAULT_REASONING).lower(),
    )

    cached = cache.get((_STRUCTURED_CACHE_NAMESPACE, cache_key))
    if cached is not None:
        return cached

    client = _ensure_sync_client()
    messages = _build_messages(system_message, user_prompt, user_payload_json)

    last_error: Optional[str] = None
    current_max_tokens = max_output_tokens
    attempt = 1
    while attempt <= MAX_ATTEMPTS:
        response = _send_structured_request(
            client,
            messages,
            temperature,
            current_max_tokens,
            response_schema,
            reasoning_effort,
        )

        status = getattr(response, "status", None)
        if status == "incomplete":
            incomplete = getattr(response, "incomplete_details", None)
            reason = getattr(incomplete, "reason", None) if incomplete else None
            if reason == "max_output_tokens" and current_max_tokens < _STRUCTURED_MAX_TOKEN_CAP:
                next_tokens = min(
                    _STRUCTURED_MAX_TOKEN_CAP,
                    max(current_max_tokens * 2, current_max_tokens + 128),
                )
                logger.info(
                    "GPT-5 structured response truncated at %d tokens. Retrying with max_output_tokens=%d.",
                    current_max_tokens,
                    next_tokens,
                )
                current_max_tokens = next_tokens
                continue

        refusal_reason = _extract_refusal(response)
        if refusal_reason:
            logger.warning("GPT refusal: %s", refusal_reason)
            if attempt == MAX_ATTEMPTS:
                raise RuntimeError(f"GPT-5 refused the request: {refusal_reason}")
            messages.append({"role": "user", "content": REFUSAL_TEMPLATE.format(reason=refusal_reason)})
            attempt += 1
            continue

        raw_text = _extract_output_text(response)
        try:
            json.loads(raw_text)
        except json.JSONDecodeError as exc:
            last_error = str(exc)
            if attempt == MAX_ATTEMPTS:
                raise ValueError("GPT response was not valid JSON") from exc
            snippet = raw_text if len(raw_text) < 4000 else raw_text[:4000] + "..."
            messages.append(
                {
                    "role": "user",
                    "content": f"{REPROMPT_TEMPLATE.format(error=exc)}\n\nPrevious response:\n{snippet}",
                }
            )
            attempt += 1
            continue

        payload = json.loads(raw_text)
        issues = collect_structured_payload_issues(payload, response_schema)
        if issues:
            validation_summary = issues[0].message
            last_error = validation_summary
            if attempt == MAX_ATTEMPTS:
                joined = "; ".join(issue.message for issue in issues[:3])
                raise ValueError(f"GPT response failed schema validation: {joined}")
            reprompt = _build_schema_retry_message(issues, raw_text=raw_text)
            messages.append({"role": "user", "content": reprompt})
            attempt += 1
            continue

        cache.set((_STRUCTURED_CACHE_NAMESPACE, cache_key), raw_text)
        return raw_text

    raise RuntimeError(f"GPT-5 request failed after {MAX_ATTEMPTS} attempts. Last error: {last_error}")


_ASYNC_CACHE_VERSION = "1"
_ASYNC_CACHE_NAMESPACE = f"gpt5_async:{_ASYNC_CACHE_VERSION}"
_ASYNC_CACHE_DEFAULT_IGNORED_EXTRA_KEYS = frozenset(
    {
        "timeout",
        "max_retries",
        "max_exception_retries",
        "exception_retry_backoff",
        "cache_bypass",
        "cache_expire_seconds",
        "cache_include_keys",
        "cache_exclude_keys",
    }
)


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
                    nested_value = getattr(candidate, "value", None)
                    if isinstance(nested_value, str):
                        collected_parts.append(nested_value)
                        continue
                    nested_text = getattr(candidate, "text", None)
                    if isinstance(nested_text, str):
                        collected_parts.append(nested_text)
                        continue
    except Exception as exc:
        logger.error("Failed to traverse GPT-5 response structure: %s", exc)

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
                sortable_index.append(
                    json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
                )
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
        if key_str in _ASYNC_CACHE_DEFAULT_IGNORED_EXTRA_KEYS and key_str not in include:
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
        "version": _ASYNC_CACHE_VERSION,
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
    return f"{_ASYNC_CACHE_NAMESPACE}:{digest}"


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
        logger.warning("Ignoring invalid cache_expire_seconds value: %r", value)
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
            logger.warning("Failed to read GPT-5 cache entry: %s", exc)
            cached_response = None
        if cached_response is not None:
            logger.info("GPT-5 cache hit for digest %s", cache_key[-12:])
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

    base_max_tokens = int(extra_data.get("max_output_tokens", 4096))
    max_token_cap = int(extra_data.get("max_token_cap", 16384))
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

    if "temperature" in extra_data:
        logger.warning("Ignoring unsupported temperature setting for GPT-5: %s", extra_data["temperature"])
    if "top_p" in extra_data:
        logger.warning("Ignoring unsupported top_p setting for GPT-5: %s", extra_data["top_p"])

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
    response: Any | None = None
    have_adjusted_reasoning = False

    client = _ensure_async_client()
    global gpt5_client
    if gpt5_client is not None:
        client = gpt5_client

    with log_time("GPT-5 async query"):
        while True:
            try:
                response = await asyncio.wait_for(
                    client.responses.create(**request_args),
                    timeout=timeout,
                )
            except Exception as exc:
                logger.error("Error querying GPT-5: %r", exc)
                if exception_attempt >= max_exception_retries:
                    return None
                exception_attempt += 1
                backoff_seconds = exception_retry_backoff * exception_attempt
                if backoff_seconds > 0:
                    logger.info(
                        "Retrying GPT-5 request after exception in %.2fs (attempt %d/%d).",
                        backoff_seconds,
                        exception_attempt,
                        max_exception_retries,
                    )
                    await asyncio.sleep(backoff_seconds)
                continue

            text_out = _extract_text_from_response(response)
            if text_out:
                logger.info("GPT-5 response text:\n%s", text_out)
                if cache_key and not cache_bypass:
                    expire_param = None if cache_ttl is None else cache_ttl
                    try:
                        cache.set(cache_key, text_out, expire=expire_param)
                        logger.info("GPT-5 cache store for digest %s", cache_key[-12:])
                    except Exception as exc:
                        logger.warning("Failed to persist GPT-5 cache entry: %s", exc)
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
                        "GPT-5 response truncated at token cap %d with no extractable text; cannot raise further.",
                        current_tokens,
                    )
                else:
                    logger.info(
                        "GPT-5 response truncated at %d tokens. Retrying with max_output_tokens=%d.",
                        current_tokens,
                        new_limit,
                    )
                    request_args["max_output_tokens"] = new_limit
                if not lock_reasoning_effort and (current_reasoning_index + 1) < len(reasoning_chain):
                    current_reasoning_index += 1
                    new_effort = reasoning_chain[current_reasoning_index]
                    request_args["reasoning"]["effort"] = new_effort
                    have_adjusted_reasoning = True
                    logger.info("Reducing GPT-5 reasoning effort to '%s' after repeated truncation.", new_effort)
                elif not lock_reasoning_effort and not have_adjusted_reasoning and reasoning_chain[-1] != "minimal":
                    request_args["reasoning"]["effort"] = "minimal"
                    have_adjusted_reasoning = True
                    logger.info("Forcing GPT-5 reasoning effort to 'minimal' after repeated truncation.")
                continue
            break

    logger.error("GPT-5 response contained no extractable text.")
    try:
        logger.debug("GPT-5 raw response repr: %r", response)
    except Exception:
        pass
    return None
