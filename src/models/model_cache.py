from __future__ import annotations

import json
import os
import re
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


__all__ = [
    "ModelCacheError",
    "ModelCacheManager",
    "dtype_to_token",
    "device_to_token",
]


_SANITIZE_PATTERN = re.compile(r"[^a-zA-Z0-9._-]+")

class ModelCacheError(RuntimeError):
    """Raised when persisting or loading compiled model artifacts fails."""


def _sanitize_identifier(identifier: str) -> str:
    cleaned = _SANITIZE_PATTERN.sub("-", identifier.strip())
    cleaned = cleaned.strip("-")
    return cleaned or "default"


def dtype_to_token(dtype: Any) -> str:
    """
    Convert a torch dtype (or string/None) to a stable, filesystem friendly token.
    """
    try:
        import torch
    except Exception:  # pragma: no cover - torch missing when dependency stubs are used
        if dtype is None:
            return "fp32"
        if isinstance(dtype, str):
            return dtype.lower()
        return str(dtype)

    if dtype is None:
        return "fp32"
    if isinstance(dtype, str):
        value = dtype.lower()
        aliases = {
            "float32": "fp32",
            "fp32": "fp32",
            "float16": "fp16",
            "fp16": "fp16",
            "half": "fp16",
            "bfloat16": "bf16",
            "bf16": "bf16",
        }
        return aliases.get(value, value)
    if dtype == torch.float32:
        return "fp32"
    if dtype == torch.float16:
        return "fp16"
    if hasattr(torch, "bfloat16") and dtype == torch.bfloat16:  # pragma: no cover - bfloat16 missing on CPU
        return "bf16"
    return str(dtype).replace("torch.", "")


def device_to_token(device: Any) -> str:
    """Return a stable token representing a device string."""

    if device is None:
        return "cpu"
    value = str(device).strip().lower()
    if not value:
        return "cpu"
    # Normalise CUDA devices so ``cuda`` and ``cuda:0`` share the same token.
    if value.startswith("cuda"):
        return "cuda"
    if value.startswith("gpu"):
        return "cuda"
    if value.startswith("cpu"):
        return "cpu"
    if value.startswith("mps"):
        return "mps"
    return _sanitize_identifier(value) or "cpu"


@dataclass
class ModelCacheManager:
    """
    Helper that manages compiled model artifacts and metadata for a namespace.
    """

    namespace: str
    root: Optional[Path] = None

    def __post_init__(self) -> None:
        base_root = self.root if self.root is not None else Path(os.getenv("COMPILED_MODELS_DIR", "compiled_models"))
        self.root = Path(base_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._ns_root = self.root / _sanitize_identifier(self.namespace)
        self._ns_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Directory helpers
    # ------------------------------------------------------------------ #
    def _base_dir(
        self,
        model_id: str,
        dtype_token: str,
        variant_token: Optional[str] = None,
    ) -> Path:
        base = self._ns_root / _sanitize_identifier(model_id) / _sanitize_identifier(dtype_token)
        if variant_token:
            variant = _sanitize_identifier(variant_token)
            if not variant:
                variant = "default"
            base = base / variant
        return base

    def _resolve_dir(
        self,
        model_id: str,
        dtype_token: str,
        variant_token: Optional[str],
        *,
        suffix: str,
        ensure: bool = False,
    ) -> Path:
        base = self._base_dir(model_id, dtype_token, variant_token)
        path = base / suffix
        if ensure:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def weights_dir(
        self,
        model_id: str,
        dtype_token: str,
        variant_token: Optional[str] = None,
    ) -> Path:
        return self._resolve_dir(model_id, dtype_token, variant_token, suffix="weights")

    def compilation_dir(
        self,
        model_id: str,
        dtype_token: str,
        variant_token: Optional[str] = None,
    ) -> Path:
        return self._resolve_dir(model_id, dtype_token, variant_token, suffix="torch_inductor")

    def metadata_path(
        self,
        model_id: str,
        dtype_token: str,
        variant_token: Optional[str] = None,
    ) -> Path:
        return self._resolve_dir(model_id, dtype_token, variant_token, suffix="metadata.json")

    # ------------------------------------------------------------------ #
    # Metadata helpers
    # ------------------------------------------------------------------ #
    def load_metadata(
        self,
        model_id: str,
        dtype_token: str,
        variant_token: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        path = self.metadata_path(model_id, dtype_token, variant_token)
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except FileNotFoundError:
            if variant_token is not None:
                legacy_path = self.metadata_path(model_id, dtype_token, None)
                try:
                    with legacy_path.open("r", encoding="utf-8") as handle:
                        return json.load(handle)
                except FileNotFoundError:
                    return None
                except json.JSONDecodeError:
                    return None
            return None
        except json.JSONDecodeError:
            return None

    def metadata_matches(self, metadata: Dict[str, Any], expected: Dict[str, Any]) -> bool:
        for key, value in expected.items():
            if metadata.get(key) != value:
                return False
        return True

    def write_metadata(
        self,
        model_id: str,
        dtype_token: str,
        metadata: Dict[str, Any],
        variant_token: Optional[str] = None,
    ) -> None:
        path = self.metadata_path(model_id, dtype_token, variant_token)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = dict(metadata)
        metadata.setdefault(
            "created_at",
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        tmp_path = path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")
        tmp_path.replace(path)

    # ------------------------------------------------------------------ #
    # Artifact helpers
    # ------------------------------------------------------------------ #
    def has_cached_weights(
        self,
        model_id: str,
        dtype_token: str,
        variant_token: Optional[str] = None,
    ) -> bool:
        weights = self.weights_dir(model_id, dtype_token, variant_token)
        if not weights.exists():
            if variant_token is not None:
                legacy = self.weights_dir(model_id, dtype_token, None)
                if not legacy.exists():
                    return False
                return any(legacy.iterdir())
            return False
        return any(weights.iterdir())

    def reset_cache(
        self,
        model_id: str,
        dtype_token: str,
        variant_token: Optional[str] = None,
    ) -> None:
        base = self._base_dir(model_id, dtype_token, variant_token)
        if base.exists():
            shutil.rmtree(base)

    # ------------------------------------------------------------------ #
    # Environments
    # ------------------------------------------------------------------ #
    @contextmanager
    def compilation_env(
        self,
        model_id: str,
        dtype_token: str,
        variant_token: Optional[str] = None,
    ):
        """
        Context manager that points TORCHINDUCTOR_CACHE_DIR at the cache location.
        """
        compile_dir = self.compilation_dir(model_id, dtype_token, variant_token)
        compile_dir.mkdir(parents=True, exist_ok=True)
        env_key = "TORCHINDUCTOR_CACHE_DIR"
        previous = os.environ.get(env_key)
        os.environ[env_key] = str(compile_dir)
        try:
            yield compile_dir
        finally:
            if previous is None:
                os.environ.pop(env_key, None)
            else:
                os.environ[env_key] = previous

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def persist_model_state(
        self,
        *,
        model_id: str,
        dtype_token: str,
        model: Any,
        metadata: Dict[str, Any],
        force: bool = False,
        variant_token: Optional[str] = None,
    ) -> None:
        """
        Persist model weights and metadata to the cache directory.

        The method first attempts ``save_pretrained`` (HuggingFace compatible) and
        falls back to ``state_dict`` when unavailable.
        """
        weights_dir = self.weights_dir(model_id, dtype_token, variant_token)
        if force and weights_dir.exists():
            shutil.rmtree(weights_dir)
        weights_dir.mkdir(parents=True, exist_ok=True)

        fmt = "state_dict"
        saved = False
        if hasattr(model, "save_pretrained"):
            try:
                model.save_pretrained(  # type: ignore[attr-defined]
                    str(weights_dir),
                    safe_serialization=True,
                )
                fmt = "pretrained"
                saved = True
            except TypeError:
                # Older APIs may not support ``safe_serialization``.
                try:
                    model.save_pretrained(str(weights_dir))  # type: ignore[attr-defined]
                    fmt = "pretrained"
                    saved = True
                except Exception:
                    saved = False
            except Exception:
                saved = False

        if not saved:
            try:
                import torch
            except Exception as exc:  # pragma: no cover - torch missing
                raise ModelCacheError("Unable to persist model state without torch.") from exc
            state_path = weights_dir / "model_state.pt"
            torch.save(model.state_dict(), state_path)  # type: ignore[arg-type]
            metadata["state_path"] = state_path.name
            fmt = "state_dict"

        metadata = dict(metadata)
        metadata["data_format"] = fmt
        self.write_metadata(model_id, dtype_token, metadata, variant_token)

    def load_pretrained_path(
        self,
        model_id: str,
        dtype_token: str,
        variant_token: Optional[str] = None,
    ) -> Optional[Path]:
        weights_dir = self.weights_dir(model_id, dtype_token, variant_token)
        if not weights_dir.exists():
            if variant_token is not None:
                legacy = self.weights_dir(model_id, dtype_token, None)
                if not legacy.exists():
                    return None
                weights_dir = legacy
            else:
                return None
        config = weights_dir / "config.json"
        if config.exists():
            return weights_dir
        # If set is empty (state dict only) we return None
        return None

    def state_dict_path(
        self,
        model_id: str,
        dtype_token: str,
        metadata: Optional[Dict[str, Any]] = None,
        variant_token: Optional[str] = None,
    ) -> Optional[Path]:
        weights_dir = self.weights_dir(model_id, dtype_token, variant_token)
        if not weights_dir.exists():
            if variant_token is not None:
                legacy = self.weights_dir(model_id, dtype_token, None)
                if not legacy.exists():
                    return None
                weights_dir = legacy
            else:
                return None
        if metadata is None:
            metadata = self.load_metadata(model_id, dtype_token, variant_token)
        if metadata:
            candidate = metadata.get("state_path")
            if candidate:
                path = weights_dir / candidate
                if path.exists():
                    return path
        fallback = weights_dir / "model_state.pt"
        if fallback.exists():
            return fallback
        return None
