#!/usr/bin/env python3
"""Normalize XGB ensemble manifest model paths and hashes after directory moves."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_int_seed(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("boolean seed is invalid")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and (stripped.isdigit() or (stripped[0] in "+-" and stripped[1:].isdigit())):
            return int(stripped)
    raise ValueError(f"seed must be an integer, got {value!r}")


def normalize_manifest(ensemble_dir: Path) -> dict:
    ensemble_dir = Path(ensemble_dir).resolve()
    manifest_path = ensemble_dir / "alltrain_ensemble.json"
    if not manifest_path.is_file():
        raise ValueError(f"missing manifest: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{manifest_path}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{manifest_path}: expected JSON object")
    models = payload.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("manifest models must be a non-empty list")

    seen_seeds: set[int] = set()
    for raw_model in models:
        if not isinstance(raw_model, dict):
            raise ValueError("manifest models contains non-object entry")
        try:
            seed = _manifest_int_seed(raw_model["seed"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"manifest model seed invalid: {raw_model!r}") from exc
        if seed in seen_seeds:
            raise ValueError(f"manifest models contains duplicate seed {seed}")
        seen_seeds.add(seed)
        model_path = ensemble_dir / f"alltrain_seed{seed}.pkl"
        if not model_path.is_file():
            raise ValueError(f"missing model for seed {seed}: {model_path}")
        raw_model["path"] = str(model_path)
        raw_model["sha256"] = _file_sha256(model_path)

    tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    tmp_path.write_text(f"{json.dumps(payload, indent=2, sort_keys=True)}\n", encoding="utf-8")
    tmp_path.replace(manifest_path)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ensemble_dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        payload = normalize_manifest(args.ensemble_dir)
    except ValueError as exc:
        print(f"[xgb-ensemble-manifest-normalize] FAIL {exc}")
        return 2
    models = payload.get("models") or []
    print(
        "[xgb-ensemble-manifest-normalize] OK "
        f"dir={Path(args.ensemble_dir).resolve()} models={len(models)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
