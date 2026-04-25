from __future__ import annotations

import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "xgbnew" / "train_alltrain_ensemble.py"


def _load_module():
    spec = spec_from_file_location("xgbnew_train_alltrain_ensemble", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_write_json_atomic_replaces_manifest_and_removes_temp(tmp_path: Path) -> None:
    mod = _load_module()
    manifest = tmp_path / "ensemble" / "alltrain_ensemble.json"
    manifest.parent.mkdir()
    manifest.write_text("{not-json", encoding="utf-8")

    mod._write_json_atomic(manifest, {"seeds": [0, 7], "models": []})

    assert json.loads(manifest.read_text(encoding="utf-8")) == {
        "seeds": [0, 7],
        "models": [],
    }
    assert not list(manifest.parent.glob(f".{manifest.name}.*.tmp"))


def test_parse_seed_list_rejects_duplicate_production_seed() -> None:
    mod = _load_module()

    try:
        mod._parse_seed_list("0,7,0,42")
    except ValueError as exc:
        assert "duplicate seeds are not allowed: [0]" in str(exc)
    else:
        raise AssertionError("expected duplicate production seed to fail")


def test_parse_seed_list_accepts_unique_seed_strings() -> None:
    mod = _load_module()

    assert mod._parse_seed_list("0, 7,42") == [0, 7, 42]
