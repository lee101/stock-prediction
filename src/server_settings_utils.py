from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PathResolution:
    path: Path
    source: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {
            "value": str(self.path),
            "source": self.source,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class IntResolution:
    value: int
    source: str
    detail: str

    def as_dict(self) -> dict[str, int | str]:
        return {
            "value": self.value,
            "source": self.source,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class SecretResolution:
    secret: str | None
    source: str
    detail: str

    def as_dict(self) -> dict[str, bool | str]:
        return {
            "value": self.secret is not None,
            "source": self.source,
            "detail": self.detail,
        }


def clamp_int(value: int, *, minimum: int, maximum: int | None = None) -> int:
    clamped = max(int(value), minimum)
    if maximum is not None:
        clamped = min(clamped, maximum)
    return clamped


def resolve_env_int(name: str, default: int, *, minimum: int, maximum: int | None = None) -> int:
    return resolve_env_int_resolution(
        name,
        default,
        minimum=minimum,
        maximum=maximum,
    ).value


def _clamp_detail(*, parsed: int, clamped: int, minimum: int, maximum: int | None) -> str | None:
    if clamped == parsed:
        return None
    if parsed < minimum:
        return f"clamped to {clamped} (minimum {minimum})"
    if maximum is not None and parsed > maximum:
        return f"clamped to {clamped} (maximum {maximum})"
    return f"clamped to {clamped}"


def resolve_env_int_resolution(
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int | None = None,
) -> IntResolution:
    raw_value = os.getenv(name)
    if raw_value is None:
        return IntResolution(
            value=default,
            source="default",
            detail=f"built-in default {default}",
        )
    try:
        parsed = int(str(raw_value).strip())
    except (TypeError, ValueError):
        return IntResolution(
            value=default,
            source="env-invalid",
            detail=f"{name}={raw_value!r} is invalid; using built-in default {default}",
        )
    clamped = clamp_int(parsed, minimum=minimum, maximum=maximum)
    clamp_detail = _clamp_detail(parsed=parsed, clamped=clamped, minimum=minimum, maximum=maximum)
    detail = f"{name}={raw_value!r}"
    if clamp_detail is not None:
        detail = f"{detail}; {clamp_detail}"
    return IntResolution(
        value=clamped,
        source="env",
        detail=detail,
    )


def resolve_explicit_or_env_int(
    explicit: int | None,
    *,
    env_name: str,
    default: int,
    minimum: int,
    maximum: int | None = None,
) -> int:
    return resolve_explicit_or_env_int_resolution(
        explicit,
        env_name=env_name,
        default=default,
        minimum=minimum,
        maximum=maximum,
    ).value


def resolve_explicit_or_env_int_resolution(
    explicit: int | None,
    *,
    env_name: str,
    default: int,
    minimum: int,
    maximum: int | None = None,
    explicit_label: str | None = None,
) -> IntResolution:
    if explicit is None:
        return resolve_env_int_resolution(
            env_name,
            default,
            minimum=minimum,
            maximum=maximum,
        )
    parsed = int(explicit)
    clamped = clamp_int(parsed, minimum=minimum, maximum=maximum)
    label = explicit_label or env_name.lower()
    detail = f"explicit {label}={explicit!r}"
    clamp_detail = _clamp_detail(parsed=parsed, clamped=clamped, minimum=minimum, maximum=maximum)
    if clamp_detail is not None:
        detail = f"{detail}; {clamp_detail}"
    return IntResolution(
        value=clamped,
        source="explicit",
        detail=detail,
    )


def resolve_optional_secret(
    explicit: str | None,
    *,
    env_name: str,
    explicit_label: str | None = None,
) -> str | None:
    return resolve_optional_secret_resolution(
        explicit,
        env_name=env_name,
        explicit_label=explicit_label,
    ).secret


def resolve_optional_secret_resolution(
    explicit: str | None,
    *,
    env_name: str,
    explicit_label: str | None = None,
) -> SecretResolution:
    label = explicit_label or env_name.lower()
    if explicit is not None:
        normalized = str(explicit).strip()
        if normalized:
            return SecretResolution(
                secret=normalized,
                source="explicit",
                detail=f"explicit {label} configured",
            )
        return SecretResolution(
            secret=None,
            source="explicit-disabled",
            detail=f"explicit {label} disabled",
        )

    raw_value = os.getenv(env_name)
    if raw_value is None:
        return SecretResolution(
            secret=None,
            source="default",
            detail="disabled",
        )

    normalized = str(raw_value).strip()
    if normalized:
        return SecretResolution(
            secret=normalized,
            source="env",
            detail=f"{env_name} configured",
        )
    return SecretResolution(
        secret=None,
        source="env-disabled",
        detail=f"{env_name} is empty; disabled",
    )


def resolve_repo_relative_path(
    path: str | Path | None,
    *,
    repo_root: Path,
    env_name: str,
    default_path: str | Path,
    explicit_label: str = "path",
) -> PathResolution:
    if path is not None:
        raw_path = path
        source = "explicit"
        detail = f"explicit {explicit_label}={str(path)!r}"
    else:
        env_value = os.getenv(env_name)
        if env_value is not None:
            raw_path = env_value
            source = "env"
            detail = f"{env_name}={env_value!r}"
        else:
            raw_path = default_path
            source = "default"
            detail = f"built-in default {str(default_path)!r}"
    resolved = Path(raw_path).expanduser()
    if not resolved.is_absolute():
        resolved = repo_root / resolved
    return PathResolution(path=resolved, source=source, detail=detail)
