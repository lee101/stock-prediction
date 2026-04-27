"""Tests for scripts/xgbcat_risk_parity_wide_120d.sh."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "xgbcat_risk_parity_wide_120d.sh"


def _make_fake_repo(tmp_path: Path) -> Path:
    fake_repo = tmp_path / "repo"
    (fake_repo / ".venv313" / "bin").mkdir(parents=True)
    (fake_repo / ".venv313" / "bin" / "activate").write_text(
        "export XGBCAT_TEST_VENV=1\n",
        encoding="utf-8",
    )
    symbols = fake_repo / "symbol_lists" / "stocks_wide_fresh0401_photonics_2500_v1.txt"
    symbols.parent.mkdir(parents=True)
    symbols.write_text("AAPL\nMSFT\n", encoding="utf-8")
    for model_dir in [
        fake_repo / "analysis" / "xgbnew_daily" / "track1_oos120d_xgb",
        fake_repo / "analysis" / "xgbnew_daily" / "track1_oos120d_cat",
    ]:
        model_dir.mkdir(parents=True)
        (model_dir / "alltrain_seed0.pkl").write_text("model\n", encoding="utf-8")
    return fake_repo


def _run_script(fake_repo: Path, **env_overrides: str) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "REPO": str(fake_repo),
        "DRY_RUN": "1",
    }
    env.update(env_overrides)
    return subprocess.run(
        ["bash", str(SCRIPT)],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_xgbcat_risk_parity_script_syntax_is_valid() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_xgbcat_risk_parity_preflight_passes_in_dry_run(tmp_path: Path) -> None:
    fake_repo = _make_fake_repo(tmp_path)

    proc = _run_script(fake_repo)

    assert proc.returncode == 0
    assert "dry run: preflight passed" in proc.stdout
    assert "xgb_model_count=1" in proc.stdout
    assert "cat_model_count=1" in proc.stdout
    assert "manifest=" in proc.stdout
    assert not (fake_repo / "logs" / "track1" / "xgbcat_risk_parity_wide_120d.log").exists()
    manifest_path = (
        fake_repo
        / "analysis"
        / "xgbnew_daily"
        / "xgbcat_risk_parity_wide_120d"
        / "run_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["script"] == "scripts/xgbcat_risk_parity_wide_120d.sh"
    assert manifest["venv"] == ".venv313"
    assert manifest["xgb_model_count"] == 1
    assert manifest["cat_model_count"] == 1
    assert manifest["symbols_sha256"] == _sha256(
        fake_repo / "symbol_lists" / "stocks_wide_fresh0401_photonics_2500_v1.txt",
    )
    assert [model["family"] for model in manifest["models"]] == ["xgb", "cat"]
    assert all(len(model["sha256"]) == 64 for model in manifest["models"])
    assert manifest["config"]["oos_start"] == "2025-12-18"
    assert manifest["fixed_flags"]["require_production_target"] is True
    model_list = Path(manifest["model_paths_file"])
    assert model_list.read_text(encoding="utf-8").count("alltrain_seed0.pkl") == 2


def test_xgbcat_risk_parity_manifest_records_config_overrides(tmp_path: Path) -> None:
    fake_repo = _make_fake_repo(tmp_path)
    out = fake_repo / "custom-out"

    proc = _run_script(
        fake_repo,
        OOS_START="2026-01-05",
        LEVERAGE_GRID="1.5",
        OUT_SHOULD_NOT_BE_USED="ignored",
    )

    assert proc.returncode == 0
    default_manifest = (
        fake_repo
        / "analysis"
        / "xgbnew_daily"
        / "xgbcat_risk_parity_wide_120d"
        / "run_manifest.json"
    )
    assert default_manifest.exists()
    manifest = json.loads(default_manifest.read_text(encoding="utf-8"))
    assert manifest["config"]["oos_start"] == "2026-01-05"
    assert manifest["config"]["leverage_grid"] == "1.5"
    assert not out.exists()


def test_xgbcat_risk_parity_uses_venv313_by_default() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'VENV="${VENV:-.venv313}"' in text
    assert 'source "$VENV/bin/activate"' in text
    assert "XGB=$(ls " not in text
    assert "CAT=$(ls " not in text


def test_xgbcat_risk_parity_fails_before_sweep_when_symbols_missing(tmp_path: Path) -> None:
    fake_repo = _make_fake_repo(tmp_path)
    (fake_repo / "symbol_lists" / "stocks_wide_fresh0401_photonics_2500_v1.txt").unlink()
    (fake_repo / "symbol_lists" / "stocks_wide_1000_v1.txt").unlink(missing_ok=True)

    proc = _run_script(fake_repo)

    assert proc.returncode == 2
    assert "missing symbols file" in proc.stderr
    assert "dry run: preflight passed" not in proc.stdout


def test_xgbcat_risk_parity_fails_before_sweep_when_models_missing(tmp_path: Path) -> None:
    fake_repo = _make_fake_repo(tmp_path)
    for path in (fake_repo / "analysis" / "xgbnew_daily" / "track1_oos120d_cat").glob("*.pkl"):
        path.unlink()

    proc = _run_script(fake_repo)

    assert proc.returncode == 2
    assert "no CatBoost alltrain_seed*.pkl models found" in proc.stderr
    assert "dry run: preflight passed" not in proc.stdout


def test_xgbcat_risk_parity_rejects_duplicate_model_artifacts(tmp_path: Path) -> None:
    fake_repo = _make_fake_repo(tmp_path)
    shared_dir = fake_repo / "analysis" / "xgbnew_daily" / "track1_oos120d_xgb"

    proc = _run_script(
        fake_repo,
        CAT_DIR=str(shared_dir),
    )

    assert proc.returncode == 2
    assert "duplicate model artifact" in proc.stderr
    assert "dry run: preflight passed" not in proc.stdout
