from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "evaluate_screened32_candidates.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("evaluate_screened32_candidates", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_resolve_candidate_checkpoint_prefers_val_best(tmp_path: Path):
    module = _load_module()
    root = tmp_path / "ckpts"
    trial = root / "cand_a"
    trial.mkdir(parents=True)
    (trial / "best.pt").write_bytes(b"best")
    (trial / "val_best.pt").write_bytes(b"val")
    resolved = module.resolve_candidate_checkpoint(root, "cand_a")
    assert resolved.name == "val_best.pt"


def test_resolve_candidate_checkpoint_falls_back_to_latest_pt(tmp_path: Path):
    module = _load_module()
    root = tmp_path / "ckpts"
    trial = root / "cand_b"
    trial.mkdir(parents=True)
    older = trial / "older.pt"
    newer = trial / "newer.pt"
    older.write_bytes(b"old")
    newer.write_bytes(b"new")
    resolved = module.resolve_candidate_checkpoint(root, "cand_b")
    assert resolved.name == "newer.pt"


def test_load_ranked_rows_filters_errors_and_sorts(tmp_path: Path):
    module = _load_module()
    leaderboard = tmp_path / "leaderboard.csv"
    leaderboard.write_text(
        "description,rank_score,error,holdout_robust_score\n"
        "bad,10,boom,1\n"
        "ok2,2,,3\n"
        "ok1,5,,2\n",
        encoding="utf-8",
    )
    rows = module.load_ranked_rows(
        leaderboard_path=leaderboard,
        sort_field="rank_score",
        require_blank_error=True,
    )
    assert [row["description"] for row in rows] == ["ok1", "ok2"]
