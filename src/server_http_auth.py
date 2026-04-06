from __future__ import annotations

import hmac


def normalize_auth_token(value: str | None) -> str | None:
    normalized = str(value or "").strip()
    return normalized or None


def format_bearer_auth_header(token: str) -> str:
    normalized = normalize_auth_token(token)
    if normalized is None:
        raise ValueError("auth token must not be empty")
    return f"Bearer {normalized}"


def extract_bearer_auth_token(authorization: str | None) -> str | None:
    if authorization is None:
        return None
    raw_value = str(authorization).strip()
    if not raw_value:
        return None
    scheme, _, credentials = raw_value.partition(" ")
    if scheme.lower() != "bearer":
        return None
    return normalize_auth_token(credentials)


def classify_bearer_auth_failure(authorization: str | None) -> str:
    if authorization is None:
        return "missing"
    raw_value = str(authorization).strip()
    if not raw_value:
        return "missing"
    scheme, _, credentials = raw_value.partition(" ")
    if scheme.lower() != "bearer":
        return "unsupported_scheme"
    if normalize_auth_token(credentials) is None:
        return "missing_token"
    return "mismatch"


def bearer_auth_matches(*, expected_token: str | None, authorization: str | None) -> bool:
    normalized_expected = normalize_auth_token(expected_token)
    if normalized_expected is None:
        return True
    provided = extract_bearer_auth_token(authorization)
    if provided is None:
        return False
    return hmac.compare_digest(provided, normalized_expected)
