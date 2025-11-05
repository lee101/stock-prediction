from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_ROOT = Path(os.getenv("HYPERPARAM_ROOT", "hyperparams"))


@dataclass
class HyperparamRecord:
    config: Dict[str, Any]
    validation: Dict[str, Any]
    test: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self, symbol: str, model: str, windows: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "symbol": symbol,
            "model": model,
            "config": self.config,
            "validation": self.validation,
            "test": self.test,
            "windows": windows,
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "HyperparamRecord":
        metadata = payload.get("metadata", {})
        return cls(
            config=payload["config"],
            validation=payload["validation"],
            test=payload["test"],
            metadata=metadata,
        )


class HyperparamStore:
    def __init__(self, root: Path | str = DEFAULT_ROOT):
        self.root = Path(root)

    def _model_dir(self, model: str) -> Path:
        return self.root / model

    def _config_path(self, model: str, symbol: str) -> Path:
        return self._model_dir(model) / f"{symbol}.json"

    def _selection_dir(self) -> Path:
        return self.root / "best"

    def _selection_path(self, symbol: str) -> Path:
        return self._selection_dir() / f"{symbol}.json"

    def save(
        self,
        model: str,
        symbol: str,
        record: HyperparamRecord,
        windows: Dict[str, Any],
    ) -> Path:
        model_dir = self._model_dir(model)
        model_dir.mkdir(parents=True, exist_ok=True)
        path = self._config_path(model, symbol)
        payload = record.to_payload(symbol, model, windows)
        with path.open("w") as fp:
            json.dump(payload, fp, indent=2)
        return path

    def load(self, model: str, symbol: str) -> Optional[HyperparamRecord]:
        path = self._config_path(model, symbol)
        if not path.exists():
            return None
        with path.open("r") as fp:
            payload = json.load(fp)
        return HyperparamRecord.from_payload(payload)

    def save_selection(self, symbol: str, payload: Dict[str, Any]) -> Path:
        selection_dir = self._selection_dir()
        selection_dir.mkdir(parents=True, exist_ok=True)
        path = self._selection_path(symbol)
        with path.open("w") as fp:
            json.dump(payload, fp, indent=2)
        return path

    def load_selection(self, symbol: str) -> Optional[Dict[str, Any]]:
        path = self._selection_path(symbol)
        if not path.exists():
            return None
        with path.open("r") as fp:
            return json.load(fp)


_DEFAULT_STORE = HyperparamStore()


def save_best_config(
    model: str,
    symbol: str,
    config: Dict[str, Any],
    validation: Dict[str, Any],
    test: Dict[str, Any],
    windows: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    store: Optional[HyperparamStore] = None,
) -> Path:
    record = HyperparamRecord(config=config, validation=validation, test=test, metadata=metadata or {})
    target_store = store or _DEFAULT_STORE
    return target_store.save(model, symbol, record, windows)


def load_best_config(
    model: str,
    symbol: str,
    store: Optional[HyperparamStore] = None,
) -> Optional[HyperparamRecord]:
    target_store = store or _DEFAULT_STORE
    return target_store.load(model, symbol)


def save_model_selection(
    symbol: str,
    model: str,
    config: Dict[str, Any],
    validation: Dict[str, Any],
    test: Dict[str, Any],
    windows: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    config_path: Optional[str] = None,
    store: Optional[HyperparamStore] = None,
) -> Path:
    payload: Dict[str, Any] = {
        "symbol": symbol,
        "model": model,
        "config": config,
        "validation": validation,
        "test": test,
        "windows": windows,
    }
    if config_path is not None:
        payload["config_path"] = config_path
    if metadata:
        payload["metadata"] = metadata
    target_store = store or _DEFAULT_STORE
    return target_store.save_selection(symbol, payload)


def load_model_selection(
    symbol: str,
    store: Optional[HyperparamStore] = None,
) -> Optional[Dict[str, Any]]:
    target_store = store or _DEFAULT_STORE
    return target_store.load_selection(symbol)


def save_close_policy(
    symbol: str,
    close_policy: str,
    comparison_results: Optional[Dict[str, Any]] = None,
    store: Optional[HyperparamStore] = None,
) -> Path:
    """
    Save the best close policy for a symbol.

    Args:
        symbol: The trading symbol
        close_policy: Either 'INSTANT_CLOSE' or 'KEEP_OPEN'
        comparison_results: Optional dict with comparison metrics
        store: Optional HyperparamStore instance

    Returns:
        Path to the saved file
    """
    payload: Dict[str, Any] = {
        "symbol": symbol,
        "close_policy": close_policy,
    }
    if comparison_results:
        payload["comparison"] = comparison_results

    target_store = store or _DEFAULT_STORE
    close_policy_dir = target_store.root / "close_policy"
    close_policy_dir.mkdir(parents=True, exist_ok=True)

    path = close_policy_dir / f"{symbol}.json"
    with path.open("w") as fp:
        json.dump(payload, fp, indent=2)
    return path


def load_close_policy(
    symbol: str,
    store: Optional[HyperparamStore] = None,
) -> Optional[str]:
    """
    Load the best close policy for a symbol.

    Args:
        symbol: The trading symbol
        store: Optional HyperparamStore instance

    Returns:
        Either 'INSTANT_CLOSE' or 'KEEP_OPEN', or None if not found
    """
    target_store = store or _DEFAULT_STORE
    close_policy_dir = target_store.root / "close_policy"
    path = close_policy_dir / f"{symbol}.json"

    if not path.exists():
        return None

    with path.open("r") as fp:
        payload = json.load(fp)

    return payload.get("close_policy")
