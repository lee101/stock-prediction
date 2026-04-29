from __future__ import annotations

import json
import hashlib
import pickle
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO / "scripts" / "validate_xgb_ensemble.py"
DEFAULT_FEATURE_COLS = ["ret_1d", "ret_5d"]


class _FakeClassifier:
    def predict_proba(self, x):
        return np.tile(np.array([[0.35, 0.65]], dtype=np.float32), (len(x), 1))


def _load_module():
    spec = spec_from_file_location("validate_xgb_ensemble", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_fake_model(path: Path, *, feature_cols: list[str] | None = None) -> None:
    if feature_cols is None:
        feature_cols = list(DEFAULT_FEATURE_COLS)
    payload = {
        "clf": _FakeClassifier(),
        "feature_cols": feature_cols,
        "col_medians": np.zeros(len(feature_cols), dtype=np.float32),
        "device": "cpu",
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_ensemble(tmp_path: Path, seeds: tuple[int, ...] = (0, 7)) -> Path:
    ensemble_dir = tmp_path / "ensemble"
    ensemble_dir.mkdir()
    models = []
    for seed in seeds:
        path = ensemble_dir / f"alltrain_seed{seed}.pkl"
        _write_fake_model(path)
        models.append({"seed": seed, "path": str(path), "sha256": _sha256(path)})
    (ensemble_dir / "alltrain_ensemble.json").write_text(
        json.dumps({
            "seeds": list(seeds),
            "train_end": "2026-04-25",
            "models": models,
            "config": {"feature_cols": DEFAULT_FEATURE_COLS},
        }),
        encoding="utf-8",
    )
    return ensemble_dir


def test_validate_ensemble_dir_loads_each_model_and_smoke_predicts(tmp_path) -> None:
    mod = _load_module()
    ensemble_dir = _write_ensemble(tmp_path)

    check = mod.validate_ensemble_dir(
        ensemble_dir,
        seeds=(0, 7),
        min_pkl_bytes=1,
    )

    assert check.train_end == "2026-04-25"
    assert check.seeds == (0, 7)
    assert [model.seed for model in check.models] == [0, 7]
    assert [model.sha256 for model in check.models] == [
        _sha256(ensemble_dir / "alltrain_seed0.pkl"),
        _sha256(ensemble_dir / "alltrain_seed7.pkl"),
    ]
    assert [model.n_features for model in check.models] == [2, 2]
    assert [model.feature_cols for model in check.models] == [
        tuple(DEFAULT_FEATURE_COLS),
        tuple(DEFAULT_FEATURE_COLS),
    ]


def test_validate_model_paths_loads_exact_paths_without_manifest(tmp_path) -> None:
    mod = _load_module()
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model_paths = (
        model_dir / "alltrain_seed7.pkl",
        model_dir / "alltrain_seed0.pkl",
    )
    for path in model_paths:
        _write_fake_model(path)

    check = mod.validate_model_paths(model_paths, seeds=(0, 7), min_pkl_bytes=1)

    assert check.ensemble_dir is None
    assert check.seeds == (0, 7)
    assert [model.path for model in check.models] == [str(path) for path in model_paths]


def test_validate_model_paths_rejects_offline_fm_latent_features(tmp_path) -> None:
    mod = _load_module()
    path = tmp_path / "alltrain_seed0.pkl"
    _write_fake_model(path, feature_cols=["ret_1d", "latent_0", "fm_available"])

    with pytest.raises(ValueError, match="offline FM latents not available in live trading"):
        mod.validate_model_paths((path,), min_pkl_bytes=1)


def test_validate_model_paths_rejects_duplicate_paths(tmp_path) -> None:
    mod = _load_module()
    path = tmp_path / "alltrain_seed0.pkl"
    _write_fake_model(path)

    with pytest.raises(ValueError, match="model path list contains duplicates"):
        mod.validate_model_paths((path, path), min_pkl_bytes=1)


def test_validate_model_paths_rejects_normalized_duplicate_paths(tmp_path, monkeypatch) -> None:
    mod = _load_module()
    path = tmp_path / "alltrain_seed0.pkl"
    _write_fake_model(path)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="model path list contains duplicates"):
        mod.validate_model_paths((Path("alltrain_seed0.pkl"), path), min_pkl_bytes=1)


def test_validate_model_paths_rejects_duplicate_seed_filenames(tmp_path) -> None:
    mod = _load_module()
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    path0_left = left / "alltrain_seed0.pkl"
    path0_right = right / "alltrain_seed0.pkl"
    _write_fake_model(path0_left)
    _write_fake_model(path0_right)

    with pytest.raises(ValueError, match="model path seeds contain duplicates"):
        mod.validate_model_paths((path0_left, path0_right), min_pkl_bytes=1)


def test_validate_model_paths_rejects_unparseable_seed_filename(tmp_path) -> None:
    mod = _load_module()
    path = tmp_path / "candidate.pkl"
    _write_fake_model(path)

    with pytest.raises(ValueError, match="filename must match alltrain_seed<seed>.pkl"):
        mod.validate_model_paths((path,), min_pkl_bytes=1)


def test_validate_model_paths_can_require_matching_manifest(tmp_path) -> None:
    mod = _load_module()
    ensemble_dir = tmp_path / "models"
    ensemble_dir.mkdir()
    model_paths = (
        ensemble_dir / "alltrain_seed0.pkl",
        ensemble_dir / "alltrain_seed7.pkl",
    )
    for path in model_paths:
        _write_fake_model(path)
    (ensemble_dir / "alltrain_ensemble.json").write_text(
        json.dumps(
            {
                "seeds": [0, 7],
                "train_end": "2026-04-25",
                "models": [
                    {"seed": 0, "path": str(model_paths[0]), "sha256": _sha256(model_paths[0])},
                    {"seed": 7, "path": str(model_paths[1]), "sha256": _sha256(model_paths[1])},
                ],
                "config": {"feature_cols": DEFAULT_FEATURE_COLS},
            }
        ),
        encoding="utf-8",
    )

    check = mod.validate_model_paths(
        model_paths,
        seeds=(0, 7),
        min_pkl_bytes=1,
        require_manifest=True,
    )

    assert check.ensemble_dir == str(ensemble_dir.resolve())
    assert check.train_end == "2026-04-25"


def test_validate_model_paths_require_manifest_rejects_hash_mismatch(tmp_path) -> None:
    mod = _load_module()
    ensemble_dir = tmp_path / "models"
    ensemble_dir.mkdir()
    path0 = ensemble_dir / "alltrain_seed0.pkl"
    path7 = ensemble_dir / "alltrain_seed7.pkl"
    _write_fake_model(path0)
    _write_fake_model(path7)
    (ensemble_dir / "alltrain_ensemble.json").write_text(
        json.dumps(
            {
                "seeds": [0, 7],
                "models": [
                    {"seed": 0, "path": str(path0), "sha256": "0" * 64},
                    {"seed": 7, "path": str(path7), "sha256": _sha256(path7)},
                ],
                "config": {"feature_cols": DEFAULT_FEATURE_COLS},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="manifest sha256 differs for seed 0"):
        mod.validate_model_paths(
            (path0, path7),
            seeds=(0, 7),
            min_pkl_bytes=1,
            require_manifest=True,
        )


def test_validate_model_paths_require_manifest_rejects_missing_hash(tmp_path) -> None:
    mod = _load_module()
    ensemble_dir = tmp_path / "models"
    ensemble_dir.mkdir()
    path0 = ensemble_dir / "alltrain_seed0.pkl"
    path7 = ensemble_dir / "alltrain_seed7.pkl"
    _write_fake_model(path0)
    _write_fake_model(path7)
    (ensemble_dir / "alltrain_ensemble.json").write_text(
        json.dumps(
            {
                "seeds": [0, 7],
                "models": [
                    {"seed": 0, "path": str(path0), "sha256": _sha256(path0)},
                    {"seed": 7, "path": str(path7)},
                ],
                "config": {"feature_cols": DEFAULT_FEATURE_COLS},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="manifest model sha256 missing for seed 7"):
        mod.validate_model_paths(
            (path0, path7),
            seeds=(0, 7),
            min_pkl_bytes=1,
            require_manifest=True,
        )


def test_validate_model_paths_require_manifest_rejects_missing_models(tmp_path) -> None:
    mod = _load_module()
    ensemble_dir = tmp_path / "models"
    ensemble_dir.mkdir()
    path0 = ensemble_dir / "alltrain_seed0.pkl"
    _write_fake_model(path0)
    (ensemble_dir / "alltrain_ensemble.json").write_text(
        json.dumps(
            {
                "seeds": [0],
                "config": {"feature_cols": DEFAULT_FEATURE_COLS},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="manifest models must be present"):
        mod.validate_model_paths((path0,), seeds=(0,), min_pkl_bytes=1, require_manifest=True)


def test_validate_model_paths_require_manifest_rejects_missing_model_path(tmp_path) -> None:
    mod = _load_module()
    ensemble_dir = tmp_path / "models"
    ensemble_dir.mkdir()
    path0 = ensemble_dir / "alltrain_seed0.pkl"
    _write_fake_model(path0)
    (ensemble_dir / "alltrain_ensemble.json").write_text(
        json.dumps(
            {
                "seeds": [0],
                "models": [{"seed": 0, "sha256": _sha256(path0)}],
                "config": {"feature_cols": DEFAULT_FEATURE_COLS},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="manifest model path missing for seed 0"):
        mod.validate_model_paths((path0,), seeds=(0,), min_pkl_bytes=1, require_manifest=True)


def test_validate_model_paths_require_manifest_rejects_model_path_mismatch(tmp_path) -> None:
    mod = _load_module()
    ensemble_dir = tmp_path / "models"
    ensemble_dir.mkdir()
    path0 = ensemble_dir / "alltrain_seed0.pkl"
    path7 = ensemble_dir / "alltrain_seed7.pkl"
    _write_fake_model(path0)
    _write_fake_model(path7)
    (ensemble_dir / "alltrain_ensemble.json").write_text(
        json.dumps(
            {
                "seeds": [0, 7],
                "models": [
                    {
                        "seed": 0,
                        "path": str(tmp_path / "staging" / "alltrain_seed0.pkl"),
                        "sha256": _sha256(path0),
                    },
                    {"seed": 7, "path": str(path7), "sha256": _sha256(path7)},
                ],
                "config": {"feature_cols": DEFAULT_FEATURE_COLS},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="manifest model path differs for seed 0"):
        mod.validate_model_paths(
            (path0, path7),
            seeds=(0, 7),
            min_pkl_bytes=1,
            require_manifest=True,
        )


def test_validate_ensemble_dir_rejects_manifest_hash_mismatch(tmp_path) -> None:
    mod = _load_module()
    ensemble_dir = tmp_path / "models"
    ensemble_dir.mkdir()
    path0 = ensemble_dir / "alltrain_seed0.pkl"
    path7 = ensemble_dir / "alltrain_seed7.pkl"
    _write_fake_model(path0)
    _write_fake_model(path7)
    (ensemble_dir / "alltrain_ensemble.json").write_text(
        json.dumps(
            {
                "seeds": [0, 7],
                "models": [
                    {"seed": 0, "path": str(path0), "sha256": "0" * 64},
                    {"seed": 7, "path": str(path7), "sha256": _sha256(path7)},
                ],
                "config": {"feature_cols": DEFAULT_FEATURE_COLS},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="manifest sha256 differs for seed 0"):
        mod.validate_ensemble_dir(ensemble_dir, seeds=(0, 7), min_pkl_bytes=1)


def test_validate_model_paths_require_manifest_rejects_missing_manifest(tmp_path) -> None:
    mod = _load_module()
    path = tmp_path / "alltrain_seed0.pkl"
    _write_fake_model(path)

    with pytest.raises(ValueError, match="missing manifest"):
        mod.validate_model_paths((path,), seeds=(0,), min_pkl_bytes=1, require_manifest=True)


def test_validate_model_paths_require_manifest_rejects_mixed_parents(tmp_path) -> None:
    mod = _load_module()
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    path0 = left / "alltrain_seed0.pkl"
    path7 = right / "alltrain_seed7.pkl"
    _write_fake_model(path0)
    _write_fake_model(path7)

    with pytest.raises(ValueError, match="share one parent dir"):
        mod.validate_model_paths(
            (path0, path7),
            seeds=(0, 7),
            min_pkl_bytes=1,
            require_manifest=True,
        )


def test_validate_model_paths_require_manifest_rejects_feature_mismatch(tmp_path) -> None:
    mod = _load_module()
    ensemble_dir = tmp_path / "models"
    ensemble_dir.mkdir()
    (ensemble_dir / "alltrain_ensemble.json").write_text(
        json.dumps(
            {
                "seeds": [0, 7],
                "config": {"feature_cols": list(reversed(DEFAULT_FEATURE_COLS))},
            }
        ),
        encoding="utf-8",
    )
    path0 = ensemble_dir / "alltrain_seed0.pkl"
    path7 = ensemble_dir / "alltrain_seed7.pkl"
    _write_fake_model(path0)
    _write_fake_model(path7)

    with pytest.raises(ValueError, match="manifest config.feature_cols differs"):
        mod.validate_model_paths(
            (path0, path7),
            seeds=(0, 7),
            min_pkl_bytes=1,
            require_manifest=True,
        )


def test_validate_ensemble_dir_rejects_manifest_feature_mismatch(tmp_path) -> None:
    mod = _load_module()
    ensemble_dir = tmp_path / "models"
    ensemble_dir.mkdir()
    (ensemble_dir / "alltrain_ensemble.json").write_text(
        json.dumps(
            {
                "seeds": [0, 7],
                "config": {"feature_cols": list(reversed(DEFAULT_FEATURE_COLS))},
            }
        ),
        encoding="utf-8",
    )
    path0 = ensemble_dir / "alltrain_seed0.pkl"
    path7 = ensemble_dir / "alltrain_seed7.pkl"
    _write_fake_model(path0)
    _write_fake_model(path7)

    with pytest.raises(ValueError, match="manifest config.feature_cols differs"):
        mod.validate_ensemble_dir(ensemble_dir, seeds=(0, 7), min_pkl_bytes=1)


def test_validate_model_paths_rejects_exact_path_seed_mismatch(tmp_path) -> None:
    mod = _load_module()
    path = tmp_path / "alltrain_seed0.pkl"
    _write_fake_model(path)

    with pytest.raises(ValueError, match=r"model path seeds=\[0\] expected=\[0, 7\]"):
        mod.validate_model_paths((path,), seeds=(0, 7), min_pkl_bytes=1)


def test_validate_model_paths_rejects_mixed_feature_contracts(tmp_path) -> None:
    mod = _load_module()
    path0 = tmp_path / "alltrain_seed0.pkl"
    path7 = tmp_path / "alltrain_seed7.pkl"
    _write_fake_model(path0, feature_cols=DEFAULT_FEATURE_COLS)
    _write_fake_model(path7, feature_cols=list(reversed(DEFAULT_FEATURE_COLS)))

    with pytest.raises(ValueError, match="feature_cols differ"):
        mod.validate_model_paths((path0, path7), seeds=(0, 7), min_pkl_bytes=1)


def test_validate_ensemble_dir_rejects_manifest_seed_mismatch(tmp_path) -> None:
    mod = _load_module()
    ensemble_dir = _write_ensemble(tmp_path, seeds=(0,))

    with pytest.raises(ValueError, match=r"manifest seeds=\[0\] expected=\[0, 7\]"):
        mod.validate_ensemble_dir(ensemble_dir, seeds=(0, 7), min_pkl_bytes=1)


def test_validate_ensemble_dir_rejects_boolean_manifest_seeds(tmp_path) -> None:
    mod = _load_module()
    ensemble_dir = tmp_path / "models"
    ensemble_dir.mkdir()
    _write_fake_model(ensemble_dir / "alltrain_seed0.pkl")
    (ensemble_dir / "alltrain_ensemble.json").write_text(
        json.dumps({"seeds": [False]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="manifest seeds invalid"):
        mod.validate_ensemble_dir(ensemble_dir, seeds=(0,), min_pkl_bytes=1)


def test_validate_ensemble_dir_rejects_fractional_manifest_seeds(tmp_path) -> None:
    mod = _load_module()
    ensemble_dir = tmp_path / "models"
    ensemble_dir.mkdir()
    _write_fake_model(ensemble_dir / "alltrain_seed7.pkl")
    (ensemble_dir / "alltrain_ensemble.json").write_text(
        json.dumps({"seeds": [7.9]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="manifest seeds invalid"):
        mod.validate_ensemble_dir(ensemble_dir, seeds=(7,), min_pkl_bytes=1)


def test_validate_model_paths_rejects_boolean_manifest_model_seed(tmp_path) -> None:
    mod = _load_module()
    ensemble_dir = tmp_path / "models"
    ensemble_dir.mkdir()
    path0 = ensemble_dir / "alltrain_seed0.pkl"
    _write_fake_model(path0)
    (ensemble_dir / "alltrain_ensemble.json").write_text(
        json.dumps(
            {
                "seeds": [0],
                "models": [{"seed": False, "path": str(path0), "sha256": _sha256(path0)}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="manifest model seed invalid"):
        mod.validate_model_paths((path0,), seeds=(0,), min_pkl_bytes=1, require_manifest=True)


def test_validate_model_paths_rejects_fractional_manifest_model_seed(tmp_path) -> None:
    mod = _load_module()
    ensemble_dir = tmp_path / "models"
    ensemble_dir.mkdir()
    path7 = ensemble_dir / "alltrain_seed7.pkl"
    _write_fake_model(path7)
    (ensemble_dir / "alltrain_ensemble.json").write_text(
        json.dumps(
            {
                "seeds": [7],
                "models": [{"seed": 7.9, "path": str(path7), "sha256": _sha256(path7)}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="manifest model seed invalid"):
        mod.validate_model_paths((path7,), seeds=(7,), min_pkl_bytes=1, require_manifest=True)


def test_validate_ensemble_dir_rejects_bad_model_metadata(tmp_path) -> None:
    mod = _load_module()
    ensemble_dir = _write_ensemble(tmp_path, seeds=(0,))
    _write_fake_model(ensemble_dir / "alltrain_seed0.pkl", feature_cols=[])

    with pytest.raises(ValueError, match="feature_cols must be a non-empty list"):
        mod.validate_ensemble_dir(ensemble_dir, seeds=(0,), min_pkl_bytes=1)


def test_validate_model_paths_allows_live_supported_optional_features(tmp_path) -> None:
    mod = _load_module()
    path = tmp_path / "alltrain_seed0.pkl"
    _write_fake_model(
        path,
        feature_cols=[
            "ret_1d",
            "chronos_oc_return",
            "rank_ret_5d",
            "cs_iqr_ret5",
        ],
    )

    check = mod.validate_model_paths((path,), seeds=(0,), min_pkl_bytes=1)

    assert check.models[0].feature_cols == (
        "ret_1d",
        "chronos_oc_return",
        "rank_ret_5d",
        "cs_iqr_ret5",
    )


def test_validate_model_paths_rejects_features_live_cannot_build(tmp_path) -> None:
    mod = _load_module()
    path = tmp_path / "alltrain_seed0.pkl"
    _write_fake_model(path, feature_cols=["ret_1d", "future_alpha_leak"])

    with pytest.raises(ValueError, match="unsupported live features"):
        mod.validate_model_paths((path,), seeds=(0,), min_pkl_bytes=1)
