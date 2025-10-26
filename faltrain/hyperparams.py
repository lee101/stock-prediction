"""Hyperparameter resolution utilities for FAL training."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

from faltrain.logger_utils import std_logger

try:  # pragma: no cover - boto3 is optional at runtime
    import boto3
    from botocore.exceptions import ClientError
except Exception:  # pragma: no cover - allow operating without boto3
    boto3 = None  # type: ignore
    ClientError = None  # type: ignore

LOG = std_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _deduplicate(paths: Iterable[Path]) -> Tuple[Path, ...]:
    seen = {}
    ordered: Tuple[Path, ...] = tuple()
    for path in paths:
        resolved = path
        if resolved not in seen:
            seen[resolved] = None
            ordered += (resolved,)
    return ordered


def _coerce_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        return (_REPO_ROOT / path).resolve()
    return path


def _default_search_roots() -> Tuple[Path, ...]:
    roots: Sequence[Path] = ()

    env_root_raw = os.getenv("HYPERPARAM_ROOT")
    if env_root_raw:
        roots = (*roots, _coerce_path(env_root_raw))

    repo_default = (_REPO_ROOT / "hyperparams").resolve()
    roots = (*roots, repo_default)

    data_path = Path("/data/stock/hyperparams")
    roots = (*roots, data_path)

    extra_raw = os.getenv("HYPERPARAM_PATHS")
    if extra_raw:
        for token in extra_raw.split(os.pathsep):
            token = token.strip()
            if not token:
                continue
            roots = (*roots, _coerce_path(token))

    return _deduplicate(path for path in roots if path)


@dataclass(frozen=True)
class HyperparamResult:
    """Container describing a resolved hyperparameter payload."""

    payload: Dict[str, Any]
    source: str
    kind: str
    path: Optional[Path] = None

    @property
    def model(self) -> Optional[str]:
        return self.payload.get("model")

    @property
    def symbol(self) -> Optional[str]:
        return self.payload.get("symbol")

    @property
    def config(self) -> Dict[str, Any]:
        config = self.payload.get("config", {})
        if isinstance(config, dict):
            return dict(config)
        return {}

    @property
    def validation(self) -> Dict[str, Any]:
        block = self.payload.get("validation", {})
        if isinstance(block, dict):
            return dict(block)
        return {}

    @property
    def test(self) -> Dict[str, Any]:
        block = self.payload.get("test", {})
        if isinstance(block, dict):
            return dict(block)
        return {}


class HyperparamResolver:
    """Resolve hyperparameter payloads across local and remote backends."""

    def __init__(
        self,
        *,
        search_roots: Optional[Sequence[Path]] = None,
        bucket: Optional[str] = None,
        remote_prefix: Optional[str] = None,
        endpoint_url: Optional[str] = None,
    ) -> None:
        self._search_roots = _deduplicate(
            Path(root).resolve() for root in (search_roots or _default_search_roots())
        )
        self._bucket = bucket or os.getenv("R2_BUCKET", "models")
        self._remote_prefix = (remote_prefix or os.getenv("HYPERPARAM_REMOTE_PREFIX", "stock")).strip("/")
        self._endpoint_url = endpoint_url or os.getenv("R2_ENDPOINT")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def load(
        self,
        symbol: str,
        model: str,
        *,
        prefer_best: bool = True,
        allow_remote: bool = True,
        s3_client: Any | None = None,
    ) -> Optional[HyperparamResult]:
        symbol = symbol.upper().strip()
        model = model.lower().strip()

        if prefer_best:
            result = self._load_from_roots(symbol, "best", model)
            if result is not None:
                return result

        result = self._load_from_roots(symbol, model, model)
        if result is not None:
            return result

        if not allow_remote:
            return None

        return self._load_remote(symbol, model, prefer_best=prefer_best, s3_client=s3_client)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _load_from_roots(
        self,
        symbol: str,
        section: str,
        required_model: Optional[str],
    ) -> Optional[HyperparamResult]:
        filename = f"{symbol}.json"
        for root in self._search_roots:
            path = root / section / filename
            payload = self._read_json(path)
            if not payload:
                continue
            payload_model = str(payload.get("model", "")).lower() or None
            if required_model and payload_model and payload_model != required_model:
                continue
            source = f"file://{path}"
            return HyperparamResult(payload=payload, source=source, kind=section, path=path)
        return None

    def _read_json(self, path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        try:
            with path.open("r") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            LOG.warning("Failed to load hyperparams from %s: %s", path, exc)
            return None
        if not isinstance(data, dict):
            LOG.warning("Hyperparam payload at %s is not a JSON object.", path)
            return None
        return data

    def _load_remote(
        self,
        symbol: str,
        model: str,
        *,
        prefer_best: bool,
        s3_client: Any | None,
    ) -> Optional[HyperparamResult]:
        if not self._bucket:
            return None
        if not self._remote_prefix:
            return None
        if self._endpoint_url is None and s3_client is None:
            return None

        client = s3_client
        if client is None:
            if boto3 is None:
                LOG.debug("boto3 unavailable; skipping remote hyperparam lookup for %s/%s.", model, symbol)
                return None
            client = boto3.client("s3", endpoint_url=self._endpoint_url)

        keys = []
        if prefer_best:
            keys.append(f"{self._remote_prefix}/best/{symbol}.json")
        keys.append(f"{self._remote_prefix}/{model}/{symbol}.json")

        for key in keys:
            try:
                response = client.get_object(Bucket=self._bucket, Key=key)
            except Exception as exc:  # pragma: no cover - defensive guard
                if _is_missing_object(exc):
                    continue
                LOG.warning("Failed to fetch remote hyperparams %s (bucket=%s): %s", key, self._bucket, exc)
                continue

            body = response.get("Body")
            if body is None:
                LOG.warning("S3 response for %s missing Body.", key)
                continue
            try:
                raw = body.read()
            except Exception as exc:  # pragma: no cover - defensive
                LOG.warning("Failed reading hyperparam body for %s: %s", key, exc)
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                LOG.warning("Hyperparam JSON decode error for %s: %s", key, exc)
                continue
            if not isinstance(payload, dict):
                continue
            if prefer_best and "best/" in key:
                payload_model = str(payload.get("model", "")).lower()
                if payload_model and payload_model != model:
                    continue
            source = f"s3://{self._bucket}/{key}"
            return HyperparamResult(payload=payload, source=source, kind="best" if "best/" in key else model)

        return None


def _is_missing_object(exc: Exception) -> bool:
    if isinstance(exc, FileNotFoundError):
        return True
    if ClientError and isinstance(exc, ClientError):  # type: ignore[arg-type]
        code = exc.response.get("Error", {}).get("Code")  # type: ignore[attr-defined]
        if code in {"NoSuchKey", "404", "NotFound"}:
            return True
    message = str(exc).lower()
    return "nosuchkey" in message or "not found" in message


__all__ = ["HyperparamResolver", "HyperparamResult"]
