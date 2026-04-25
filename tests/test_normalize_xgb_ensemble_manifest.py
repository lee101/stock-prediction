from __future__ import annotations

import hashlib
import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO / "scripts" / "normalize_xgb_ensemble_manifest.py"


def _load_module():
    spec = spec_from_file_location("normalize_xgb_ensemble_manifest", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_normalize_manifest_rewrites_staging_paths_and_hashes(tmp_path: Path) -> None:
    mod = _load_module()
    live_dir = tmp_path / "alltrain_ensemble_gpu"
    live_dir.mkdir()
    staging_dir = tmp_path / "alltrain_ensemble_gpu_staging_20260425T000000Z"
    for seed in (0, 7):
        (live_dir / f"alltrain_seed{seed}.pkl").write_bytes(f"model-{seed}".encode())
    manifest = {
        "seeds": [0, 7],
        "models": [
            {
                "seed": 0,
                "path": str(staging_dir / "alltrain_seed0.pkl"),
            },
            {
                "seed": 7,
                "path": str(staging_dir / "alltrain_seed7.pkl"),
                "sha256": "0" * 64,
            },
        ],
    }
    (live_dir / "alltrain_ensemble.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    mod.normalize_manifest(live_dir)

    payload = json.loads((live_dir / "alltrain_ensemble.json").read_text(encoding="utf-8"))
    models = {int(model["seed"]): model for model in payload["models"]}
    for seed in (0, 7):
        model_path = live_dir / f"alltrain_seed{seed}.pkl"
        assert models[seed]["path"] == str(model_path.resolve())
        assert models[seed]["sha256"] == _sha256(model_path)


def test_normalize_manifest_rejects_boolean_model_seed(tmp_path: Path) -> None:
    mod = _load_module()
    live_dir = tmp_path / "alltrain_ensemble_gpu"
    live_dir.mkdir()
    (live_dir / "alltrain_seed0.pkl").write_bytes(b"model")
    (live_dir / "alltrain_ensemble.json").write_text(
        json.dumps({"models": [{"seed": False, "path": "alltrain_seed0.pkl"}]}),
        encoding="utf-8",
    )

    try:
        mod.normalize_manifest(live_dir)
    except ValueError as exc:
        assert "manifest model seed invalid" in str(exc)
    else:
        raise AssertionError("expected boolean manifest model seed to fail")


def test_normalize_manifest_rejects_fractional_model_seed(tmp_path: Path) -> None:
    mod = _load_module()
    live_dir = tmp_path / "alltrain_ensemble_gpu"
    live_dir.mkdir()
    (live_dir / "alltrain_seed7.pkl").write_bytes(b"model")
    (live_dir / "alltrain_ensemble.json").write_text(
        json.dumps({"models": [{"seed": 7.9, "path": "alltrain_seed7.pkl"}]}),
        encoding="utf-8",
    )

    try:
        mod.normalize_manifest(live_dir)
    except ValueError as exc:
        assert "manifest model seed invalid" in str(exc)
    else:
        raise AssertionError("expected fractional manifest model seed to fail")


def test_normalize_manifest_uses_canonical_seed_filename(tmp_path: Path) -> None:
    mod = _load_module()
    live_dir = tmp_path / "alltrain_ensemble_gpu"
    live_dir.mkdir()
    seed0 = live_dir / "alltrain_seed0.pkl"
    seed7 = live_dir / "alltrain_seed7.pkl"
    seed0.write_bytes(b"model-0")
    seed7.write_bytes(b"model-7")
    (live_dir / "alltrain_ensemble.json").write_text(
        json.dumps(
            {
                "models": [
                    {"seed": 7, "path": str(tmp_path / "stale" / "alltrain_seed0.pkl")},
                ],
            }
        ),
        encoding="utf-8",
    )

    mod.normalize_manifest(live_dir)

    payload = json.loads((live_dir / "alltrain_ensemble.json").read_text(encoding="utf-8"))
    model = payload["models"][0]
    assert model["path"] == str(seed7.resolve())
    assert model["sha256"] == _sha256(seed7)


def test_normalize_manifest_rejects_duplicate_model_seed(tmp_path: Path) -> None:
    mod = _load_module()
    live_dir = tmp_path / "alltrain_ensemble_gpu"
    live_dir.mkdir()
    (live_dir / "alltrain_seed0.pkl").write_bytes(b"model")
    (live_dir / "alltrain_ensemble.json").write_text(
        json.dumps(
            {
                "models": [
                    {"seed": 0, "path": "alltrain_seed0.pkl"},
                    {"seed": 0, "path": "alltrain_seed0.pkl"},
                ],
            }
        ),
        encoding="utf-8",
    )

    try:
        mod.normalize_manifest(live_dir)
    except ValueError as exc:
        assert "duplicate seed 0" in str(exc)
    else:
        raise AssertionError("expected duplicate manifest model seed to fail")
