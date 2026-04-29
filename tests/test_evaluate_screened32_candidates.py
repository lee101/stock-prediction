from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


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


@pytest.mark.parametrize("description", ["", "../outside", "nested/name", "/abs/path", ".", ".."])
def test_require_safe_component_rejects_unsafe_descriptions(description: str):
    module = _load_module()

    with pytest.raises(ValueError, match=r"safe path component|non-empty"):
        module.require_safe_component(description)


def test_resolve_candidate_checkpoint_rejects_traversal(tmp_path: Path):
    module = _load_module()
    root = tmp_path / "ckpts"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "best.pt").write_bytes(b"outside")

    with pytest.raises(ValueError, match="safe path component"):
        module.resolve_candidate_checkpoint(root, "../outside")


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


def test_run_candidate_eval_forwards_short_borrow_apr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_module()
    commands: list[list[str]] = []
    output_path = tmp_path / "report.json"

    def fake_run(cmd, **kwargs):
        commands.append([str(part) for part in cmd])
        output_path.write_text('{"recommendation": {"status": "ok"}}', encoding="utf-8")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    args = SimpleNamespace(
        data_path=tmp_path / "data.bin",
        baseline_checkpoint=tmp_path / "baseline.pt",
        baseline_extra_checkpoints=[tmp_path / "extra.pt"],
        horizons_days="100",
        slippage_bps="0,5,10,20",
        n_windows=30,
        seed=1337,
        recent_within_days=140,
        fee_rate=0.001,
        fill_buffer_bps=5.0,
        short_borrow_apr=0.0625,
        decision_lag=2,
        exhaustive=False,
        disable_shorts=True,
    )

    report = module._run_candidate_eval(
        candidate_checkpoint=tmp_path / "candidate.pt",
        output_path=output_path,
        args=args,
    )

    assert report["recommendation"]["status"] == "ok"
    assert len(commands) == 1
    cmd = commands[0]
    assert cmd[cmd.index("--short-borrow-apr") + 1] == "0.0625"
    assert "--allow-shorts" not in cmd


def test_main_rejects_unsafe_selected_description_before_eval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_module()
    leaderboard = tmp_path / "leaderboard.csv"
    leaderboard.write_text(
        "description,rank_score,error,holdout_robust_score\n"
        "good,10,,1\n"
        "../outside,9,,1\n",
        encoding="utf-8",
    )
    checkpoint_root = tmp_path / "ckpts"
    (checkpoint_root / "good").mkdir(parents=True)
    (checkpoint_root / "good" / "best.pt").write_bytes(b"good")
    out_dir = tmp_path / "out"
    run_calls = 0

    def fail_if_called(**kwargs):
        nonlocal run_calls
        run_calls += 1
        raise AssertionError("_run_candidate_eval should not be called")

    monkeypatch.setattr(module, "_run_candidate_eval", fail_if_called)

    rc = module.main(
        [
            "--leaderboard",
            str(leaderboard),
            "--checkpoint-root",
            str(checkpoint_root),
            "--top-k",
            "2",
            "--out-dir",
            str(out_dir),
        ]
    )

    assert rc == 2
    assert run_calls == 0
    assert not (out_dir / "summary.json").exists()
