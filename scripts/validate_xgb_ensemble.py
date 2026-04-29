#!/usr/bin/env python3
"""Validate a staged xgbnew alltrain ensemble before live rotation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xgbnew.model import XGBStockModel
from xgbnew.features import (
    FM_LATENT_FEATURE_COLS,
    LIVE_SUPPORTED_FEATURE_COLS,
)


DEFAULT_SEEDS = (0, 7, 42, 73, 197)
DEFAULT_MIN_PKL_BYTES = 100 * 1024


@dataclass(frozen=True)
class ModelCheck:
    seed: int
    path: str
    sha256: str
    n_features: int
    feature_cols: tuple[str, ...]
    device: str | None


@dataclass(frozen=True)
class EnsembleCheck:
    ensemble_dir: str | None
    train_end: str | None
    seeds: tuple[int, ...]
    models: tuple[ModelCheck, ...]


def _load_manifest(ensemble_dir: Path) -> dict:
    manifest_path = ensemble_dir / "alltrain_ensemble.json"
    if not manifest_path.is_file():
        raise ValueError(f"missing manifest: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{manifest_path}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{manifest_path}: expected JSON object")
    return payload


def _manifest_seeds(payload: dict) -> tuple[int, ...]:
    raw = payload.get("seeds")
    if not isinstance(raw, list):
        raise ValueError("manifest seeds must be a list")
    try:
        return tuple(sorted(_manifest_int_seed(seed) for seed in raw))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"manifest seeds invalid: {raw!r}") from exc


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_model_contract(model: XGBStockModel, path: Path, *, sha256: str) -> ModelCheck:
    if not getattr(model, "_fitted", False):
        raise ValueError(f"{path}: model is not marked fitted")
    feature_cols = getattr(model, "feature_cols", None)
    if not isinstance(feature_cols, list) or not feature_cols:
        raise ValueError(f"{path}: feature_cols must be a non-empty list")
    if not all(isinstance(col, str) and col for col in feature_cols):
        raise ValueError(f"{path}: feature_cols contains invalid names")
    unsupported = sorted(set(feature_cols) - LIVE_SUPPORTED_FEATURE_COLS)
    if unsupported:
        raise ValueError(
            f"{path}: feature_cols contains unsupported live features: {unsupported}"
        )
    offline_only = sorted(set(feature_cols).intersection(FM_LATENT_FEATURE_COLS))
    if offline_only:
        raise ValueError(
            f"{path}: feature_cols require offline FM latents not available in live trading: "
            f"{offline_only}"
        )
    medians = np.asarray(getattr(model, "_col_medians", []), dtype=np.float32)
    if medians.shape != (len(feature_cols),):
        raise ValueError(
            f"{path}: col_medians shape {medians.shape} does not match "
            f"{len(feature_cols)} features"
        )
    if not np.isfinite(medians).all():
        raise ValueError(f"{path}: col_medians contains non-finite values")
    if not hasattr(getattr(model, "clf", None), "predict_proba"):
        raise ValueError(f"{path}: classifier lacks predict_proba")

    smoke_df = pd.DataFrame([dict.fromkeys(feature_cols, 0.0)])
    scores = model.predict_scores(smoke_df)
    if len(scores) != 1:
        raise ValueError(f"{path}: smoke prediction returned {len(scores)} rows")
    score = float(scores.iloc[0])
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"{path}: smoke prediction out of range: {score!r}")

    seed_text = path.stem.removeprefix("alltrain_seed")
    try:
        seed = int(seed_text)
    except ValueError:
        raise ValueError(f"{path}: filename must match alltrain_seed<seed>.pkl")
    return ModelCheck(
        seed=seed,
        path=str(path),
        sha256=sha256,
        n_features=len(feature_cols),
        feature_cols=tuple(feature_cols),
        device=getattr(model, "device", None),
    )


def _manifest_model_hashes(payload: dict) -> dict[int, str] | None:
    entries = _manifest_model_entries(payload)
    if entries is None:
        return None
    hashes: dict[int, str] = {}
    for seed, raw_model in entries.items():
        if "sha256" not in raw_model:
            raise ValueError(f"manifest model sha256 missing for seed {seed}")
        sha = str(raw_model["sha256"]).strip().lower()
        if not len(sha) == 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError(f"manifest model sha256 invalid for seed {seed}")
        hashes[seed] = sha
    return hashes


def _manifest_model_entries(payload: dict) -> dict[int, dict] | None:
    raw_models = payload.get("models")
    if raw_models is None:
        return None
    if not isinstance(raw_models, list):
        raise ValueError("manifest models must be a list")
    entries: dict[int, dict] = {}
    for raw_model in raw_models:
        if not isinstance(raw_model, dict):
            raise ValueError("manifest models contains non-object entry")
        try:
            seed = _manifest_int_seed(raw_model["seed"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"manifest model seed invalid: {raw_model!r}") from exc
        if seed in entries:
            raise ValueError(f"manifest models contains duplicate seed {seed}")
        entries[seed] = raw_model
    return entries


def _resolve_manifest_model_path(raw_path: str, manifest_dir: Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else manifest_dir / path


def validate_ensemble_dir(
    ensemble_dir: Path,
    *,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    min_pkl_bytes: int = DEFAULT_MIN_PKL_BYTES,
) -> EnsembleCheck:
    ensemble_dir = Path(ensemble_dir)
    if not ensemble_dir.is_dir():
        raise ValueError(f"ensemble dir does not exist: {ensemble_dir}")

    manifest = _load_manifest(ensemble_dir)
    expected_seeds = tuple(sorted(_manifest_int_seed(seed) for seed in seeds))
    found_seeds = _manifest_seeds(manifest)
    if found_seeds != expected_seeds:
        raise ValueError(f"manifest seeds={list(found_seeds)} expected={list(expected_seeds)}")

    model_paths = tuple(ensemble_dir / f"alltrain_seed{seed}.pkl" for seed in expected_seeds)
    check = validate_model_paths(
        model_paths,
        seeds=expected_seeds,
        min_pkl_bytes=min_pkl_bytes,
        require_manifest=True,
    )
    return check


def validate_model_paths(
    model_paths: tuple[Path, ...],
    *,
    seeds: tuple[int, ...] = (),
    min_pkl_bytes: int = DEFAULT_MIN_PKL_BYTES,
    ensemble_dir: Path | None = None,
    train_end: str | None = None,
    require_manifest: bool = False,
) -> EnsembleCheck:
    if not model_paths:
        raise ValueError("model path list is empty")
    normalized_paths = [Path(path).expanduser().resolve(strict=False) for path in model_paths]
    if len(set(normalized_paths)) != len(normalized_paths):
        raise ValueError("model path list contains duplicates")
    expected_seeds = tuple(sorted(_manifest_int_seed(seed) for seed in seeds))
    model_parents = {path.parent for path in normalized_paths}
    manifest: dict | None = None
    if require_manifest:
        if len(model_parents) != 1:
            raise ValueError("--require-manifest requires all model paths to share one parent dir")
        ensemble_dir = next(iter(model_parents))
        manifest = _load_manifest(ensemble_dir)

    checks: list[ModelCheck] = []
    for raw_path in model_paths:
        path = Path(raw_path)
        if not path.is_file():
            raise ValueError(f"missing model: {path}")
        size = path.stat().st_size
        if size < int(min_pkl_bytes):
            raise ValueError(f"{path}: {size} bytes < {int(min_pkl_bytes)}")
        try:
            model = XGBStockModel.load(path)
        except Exception as exc:
            raise ValueError(f"{path}: failed to load with XGBStockModel.load: {exc}") from exc
        check = _validate_model_contract(model, path, sha256=_file_sha256(path))
        checks.append(check)

    found_seeds = tuple(sorted(check.seed for check in checks))
    if len(set(found_seeds)) != len(found_seeds):
        raise ValueError(f"model path seeds contain duplicates: {list(found_seeds)}")
    if expected_seeds and found_seeds != expected_seeds:
        raise ValueError(f"model path seeds={list(found_seeds)} expected={list(expected_seeds)}")
    first_features = checks[0].feature_cols
    for check in checks[1:]:
        if check.feature_cols != first_features:
            raise ValueError(
                f"{check.path}: feature_cols differ from {checks[0].path} "
                f"({len(check.feature_cols)} vs {len(first_features)} features)"
            )
    if manifest is not None:
        found_manifest_seeds = _manifest_seeds(manifest)
        expected_manifest_seeds = expected_seeds or found_seeds
        if found_manifest_seeds != expected_manifest_seeds:
            raise ValueError(
                f"manifest seeds={list(found_manifest_seeds)} "
                f"expected={list(expected_manifest_seeds)}"
            )
        manifest_features = manifest.get("config", {}).get("feature_cols")
        if manifest_features is not None:
            if not isinstance(manifest_features, list) or not all(
                isinstance(col, str) and col for col in manifest_features
            ):
                raise ValueError("manifest config.feature_cols is invalid")
            if tuple(manifest_features) != first_features:
                raise ValueError("manifest config.feature_cols differs from model feature_cols")
        manifest_model_entries = _manifest_model_entries(manifest)
        if manifest_model_entries is None:
            raise ValueError("manifest models must be present when manifest is required")
        found_entry_seeds = tuple(sorted(manifest_model_entries))
        if found_entry_seeds != found_seeds:
            raise ValueError(
                f"manifest model seeds={list(found_entry_seeds)} "
                f"expected={list(found_seeds)}"
            )
        manifest_dir = Path(ensemble_dir)
        for check in checks:
            raw_manifest_path = manifest_model_entries[check.seed].get("path")
            if raw_manifest_path is None:
                raise ValueError(f"manifest model path missing for seed {check.seed}")
            if not isinstance(raw_manifest_path, str) or not raw_manifest_path.strip():
                raise ValueError(f"manifest model path invalid for seed {check.seed}")
            manifest_path = _resolve_manifest_model_path(
                raw_manifest_path.strip(),
                manifest_dir,
            )
            if manifest_path.resolve(strict=False) != Path(check.path).resolve(strict=False):
                raise ValueError(f"manifest model path differs for seed {check.seed}")
        manifest_hashes = _manifest_model_hashes(manifest)
        if manifest_hashes is not None:
            found_hash_seeds = tuple(sorted(manifest_hashes))
            if found_hash_seeds != found_seeds:
                raise ValueError(
                    f"manifest sha256 seeds={list(found_hash_seeds)} "
                    f"expected={list(found_seeds)}"
                )
            for check in checks:
                manifest_sha = manifest_hashes[check.seed]
                if manifest_sha != check.sha256:
                    raise ValueError(f"manifest sha256 differs for seed {check.seed}")
        train_end = manifest.get("train_end")

    return EnsembleCheck(
        ensemble_dir=str(ensemble_dir) if ensemble_dir is not None else None,
        train_end=train_end,
        seeds=found_seeds,
        models=tuple(checks),
    )


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid seed list: {value!r}") from exc


def _parse_model_paths(value: str) -> tuple[Path, ...]:
    return tuple(Path(part.strip()) for part in value.split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ensemble_dir", type=Path, nargs="?")
    parser.add_argument(
        "--model-paths",
        type=_parse_model_paths,
        help="Comma-separated exact model pickle paths to validate instead of an ensemble dir.",
    )
    parser.add_argument("--seeds", type=_parse_seeds, default=DEFAULT_SEEDS)
    parser.add_argument("--min-pkl-bytes", type=int, default=DEFAULT_MIN_PKL_BYTES)
    parser.add_argument(
        "--require-manifest",
        action="store_true",
        help=(
            "When validating --model-paths, require all paths to share a parent "
            "with alltrain_ensemble.json whose seed/features match the models."
        ),
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.model_paths:
            check = validate_model_paths(
                tuple(args.model_paths),
                seeds=tuple(args.seeds),
                min_pkl_bytes=int(args.min_pkl_bytes),
                require_manifest=bool(args.require_manifest),
            )
        else:
            if args.ensemble_dir is None:
                print("[xgb-ensemble-validate] FAIL ensemble_dir or --model-paths is required")
                return 2
            check = validate_ensemble_dir(
                args.ensemble_dir,
                seeds=tuple(args.seeds),
                min_pkl_bytes=int(args.min_pkl_bytes),
            )
    except ValueError as exc:
        print(f"[xgb-ensemble-validate] FAIL {exc}")
        return 2
    if args.json:
        print(json.dumps({
            "ensemble_dir": check.ensemble_dir,
            "train_end": check.train_end,
            "seeds": list(check.seeds),
            "models": [model.__dict__ for model in check.models],
        }, indent=2, sort_keys=True))
    else:
        print(
            "[xgb-ensemble-validate] OK "
            f"dir={check.ensemble_dir} train_end={check.train_end} "
            f"models={len(check.models)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
