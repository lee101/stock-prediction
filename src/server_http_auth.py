"""HTTP Bearer token authentication helpers for trading servers."""

from __future__ import annotations

import secrets


def normalize_auth_token(token: str | None) -> str | None:
    """Normalize an auth token by stripping whitespace.

    Returns *None* when the token is empty or was already *None*.
    """
    if token is None:
        return None
    token = token.strip()
    return token if token else None


def format_bearer_auth_header(token: str) -> str:
    """Format a token as an HTTP ``Authorization: Bearer`` header value."""
    return f"Bearer {token}"


def bearer_auth_matches(
    *,
    expected_token: str | None,
    authorization: str | None,
) -> bool:
    """Return *True* when *authorization* carries a Bearer token matching *expected_token*.

    Uses constant-time comparison to prevent timing side-channels.
    """
    if expected_token is None:
        # No auth configured -- allow all requests.
        return True
    if authorization is None:
        return False

    parts = authorization.split(None, 1)
    if len(parts) != 2:
        return False
    scheme, provided_token = parts
    if scheme.lower() != "bearer":
        return False
    return secrets.compare_digest(provided_token, expected_token)


def classify_bearer_auth_failure(authorization: str | None) -> str:
    """Return a short, sanitised description of why Bearer auth failed.

    The description is safe to log without leaking token values.
    """
    if authorization is None:
        return "missing"
    parts = authorization.split(None, 1)
    if len(parts) == 0:
        return "empty"
    scheme = parts[0]
    if scheme.lower() != "bearer":
        return f"wrong_scheme({scheme})"
    if len(parts) < 2 or not parts[1].strip():
        return "bearer_no_token"
    return "bearer_token_mismatch"


__all__ = [
    "bearer_auth_matches",
    "classify_bearer_auth_failure",
    "format_bearer_auth_header",
    "normalize_auth_token",
]
