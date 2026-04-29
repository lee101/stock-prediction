from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.eval_gpu_pool_top_candidates as eval_top
from scripts.eval_gpu_pool_top_candidates import (
    build_eval_command,
    load_leaderboard,
    load_eval_artifact_summary,
    main,
    pick_top_rows,
    promotion_gate_passed,
    require_leaderboard_field,
    require_safe_leaderboard_id,
    render_summary_markdown,
    resolve_checkpoint_path,
    validate_candidate_ids,
    validate_unique_eval_dirs,
    wrapper_exit_code_for_eval_returncode,
    write_text_atomic,
)


def test_pick_top_rows_sorts_descending() -> None:
    rows = [
        {"description": "a", "val_return": "0.10"},
        {"description": "b", "val_return": "0.30"},
        {"description": "c", "val_return": "-0.20"},
    ]

    top = pick_top_rows(rows, sort_by="val_return", top_k=2)

    assert [row["description"] for row in top] == ["b", "a"]


def test_load_leaderboard_validates_required_columns(tmp_path: Path) -> None:
    leaderboard = tmp_path / "leaderboard.csv"
    leaderboard.write_text("description,gpu_id\ntrial_a,0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns: val_return"):
        load_leaderboard(leaderboard, required_fields=("description", "gpu_id", "val_return"))


def test_load_leaderboard_rejects_duplicate_columns(tmp_path: Path) -> None:
    leaderboard = tmp_path / "leaderboard.csv"
    leaderboard.write_text(
        "description,gpu_id,val_return,val_return\ntrial_a,0,0.10,0.20\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate columns: val_return"):
        load_leaderboard(leaderboard, required_fields=("description", "gpu_id", "val_return"))


def test_load_leaderboard_rejects_rows_with_extra_fields(tmp_path: Path) -> None:
    leaderboard = tmp_path / "leaderboard.csv"
    leaderboard.write_text(
        "description,gpu_id,val_return\ntrial_a,0,0.10,unexpected\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="row 2 has more fields"):
        load_leaderboard(leaderboard, required_fields=("description", "gpu_id", "val_return"))


@pytest.mark.parametrize("top_k", [0, -1])
def test_pick_top_rows_rejects_non_positive_top_k(top_k: int) -> None:
    with pytest.raises(ValueError, match="top_k must be positive"):
        pick_top_rows([{"description": "a", "val_return": "0.10"}], sort_by="val_return", top_k=top_k)


def test_pick_top_rows_rejects_empty_leaderboard() -> None:
    with pytest.raises(ValueError, match="leaderboard has no candidate rows"):
        pick_top_rows([], sort_by="val_return", top_k=1)


@pytest.mark.parametrize("sort_by", ["", " ", "../score", "score/name", "score name"])
def test_pick_top_rows_rejects_unsafe_sort_metric_names(sort_by: str) -> None:
    with pytest.raises(ValueError, match="sort_by"):
        pick_top_rows([{"description": "a", "val_return": "0.10"}], sort_by=sort_by, top_k=1)


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ({"description": "bad", "val_return": "nan"}, "non-finite val_return"),
        ({"description": "bad", "val_return": ""}, "missing sort metric val_return"),
        ({"description": "bad"}, "missing sort metric val_return"),
        ({"description": "bad", "val_return": "not-a-number"}, "non-numeric val_return"),
    ],
)
def test_pick_top_rows_rejects_invalid_sort_metrics(
    row: dict[str, str],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        pick_top_rows([row], sort_by="val_return", top_k=1)


def test_resolve_checkpoint_path_prefers_best(tmp_path: Path) -> None:
    trial_dir = tmp_path / "gpu0" / "trial_a"
    trial_dir.mkdir(parents=True)
    (trial_dir / "final.pt").write_bytes(b"f")
    (trial_dir / "best.pt").write_bytes(b"b")

    resolved = resolve_checkpoint_path(tmp_path, "trial_a", "0")

    assert resolved == trial_dir / "best.pt"


@pytest.mark.parametrize(
    "value",
    ["trial_a", "trial-1", "trial.best", "trial=wide_120d"],
)
def test_require_safe_leaderboard_id_accepts_safe_names(value: str) -> None:
    assert require_safe_leaderboard_id(value, field_name="description") == value


@pytest.mark.parametrize(
    "value",
    ["", " ", "../outside", "trial/a", r"trial\\a", ".", "..", "trial name", "-starts_dash"],
)
def test_require_safe_leaderboard_id_rejects_unsafe_names(value: str) -> None:
    with pytest.raises(ValueError, match="safe path component|non-empty"):
        require_safe_leaderboard_id(value, field_name="description")


def test_require_leaderboard_field_rejects_missing_field() -> None:
    with pytest.raises(ValueError, match="missing required field gpu_id"):
        require_leaderboard_field({"description": "trial_a"}, "gpu_id")


def test_require_leaderboard_field_rejects_unsafe_field_value() -> None:
    with pytest.raises(ValueError, match="safe path component"):
        require_leaderboard_field({"description": "../outside", "gpu_id": "0"}, "description")


def test_validate_candidate_ids_rejects_unsafe_selected_row_before_eval() -> None:
    with pytest.raises(ValueError, match="safe path component"):
        validate_candidate_ids(
            [
                {"description": "safe_trial", "gpu_id": "0", "val_return": "0.30"},
                {"description": "unsafe/trial", "gpu_id": "1", "val_return": "0.20"},
            ]
        )


def test_validate_unique_eval_dirs_rejects_duplicate_selected_descriptions() -> None:
    with pytest.raises(ValueError, match="reuse eval output directories"):
        validate_unique_eval_dirs(
            [
                ("trial_a", "0", {"description": "trial_a", "gpu_id": "0"}),
                ("trial_a", "1", {"description": "trial_a", "gpu_id": "1"}),
            ]
        )


def test_validate_unique_eval_dirs_rejects_same_candidate_duplicate() -> None:
    with pytest.raises(ValueError, match="reuse eval output directories"):
        validate_unique_eval_dirs(
            [
                ("trial_a", "0", {"description": "trial_a", "gpu_id": "0"}),
                ("trial_a", "0", {"description": "trial_a", "gpu_id": "0"}),
            ]
        )


def test_resolve_checkpoint_path_rejects_path_traversal(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "best.pt").write_bytes(b"checkpoint")

    with pytest.raises(ValueError, match="safe path component"):
        resolve_checkpoint_path(tmp_path / "checkpoints", "../outside", "0")


def test_build_eval_command_includes_hourly_intrabar() -> None:
    cmd = build_eval_command(
        checkpoint_path=Path("ckpts/best.pt"),
        val_data=Path("val.bin"),
        out_dir=Path("out"),
        hourly_data_root=Path("trainingdatahourly/stocks"),
        daily_start_date="2025-08-01T00:00:00+00:00",
    )

    assert cmd[0].endswith("python")
    assert cmd[cmd.index("--n-windows") + 1] == "30"
    assert cmd[cmd.index("--window-days") + 1] == "100"
    assert cmd[cmd.index("--min-window-days") + 1] == "100"
    assert cmd[cmd.index("--fail-fast-max-dd") + 1] == "0.20"
    assert cmd[cmd.index("--fail-fast-min-completed") + 1] == "3"
    assert cmd[cmd.index("--monthly-target") + 1] == "0.27"
    assert cmd[cmd.index("--max-dd-target") + 1] == "0.25"
    assert cmd[cmd.index("--max-negative-windows") + 1] == "0"
    assert cmd[cmd.index("--min-completed-windows") + 1] == "30"
    assert "--execution-granularity" in cmd
    assert cmd[cmd.index("--execution-granularity") + 1] == "hourly_intrabar"
    assert cmd[cmd.index("--decision-lag") + 1] == "2"
    assert cmd[cmd.index("--min-decision-lag") + 1] == "2"
    assert cmd[cmd.index("--slippage-bps") + 1] == "0,5,10,20"
    assert cmd[cmd.index("--required-slippage-bps") + 1] == "0,5,10,20"
    assert cmd[cmd.index("--min-max-slippage-bps") + 1] == "20"
    assert cmd[cmd.index("--fee-rate") + 1] == "0.001"
    assert cmd[cmd.index("--min-fee-rate") + 1] == "0.001"
    assert cmd[cmd.index("--short-borrow-apr") + 1] == "0.0625"
    assert cmd[cmd.index("--min-short-borrow-apr") + 1] == "0.0625"
    assert cmd[cmd.index("--max-leverage") + 1] == "1.5"
    assert cmd[cmd.index("--max-leverage-target") + 1] == "2.0"
    assert cmd[cmd.index("--hourly-fill-buffer-bps") + 1] == "5.0"
    assert cmd[cmd.index("--min-hourly-fill-buffer-bps") + 1] == "5.0"
    assert cmd[cmd.index("--hourly-max-hold-hours") + 1] == "6"
    assert cmd[cmd.index("--max-hourly-hold-hours-target") + 1] == "6"


def test_load_eval_artifact_summary_extracts_promotion_gate(tmp_path: Path) -> None:
    out_dir = tmp_path / "eval"
    out_dir.mkdir()
    artifact = out_dir / "best_eval100d.json"
    artifact.write_text(
        json.dumps(
            {
                "raw": {"status": "ok"},
                "aggregate": {"worst_slip_monthly": 0.31},
                "promotion_gate": {
                    "passed": False,
                    "failures": ["completed_windows 3 < 4"],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = load_eval_artifact_summary(out_dir)

    assert summary["eval_artifact"] == str(artifact)
    assert summary["eval_artifact_status"] == "ok"
    assert summary["raw_status"] == "ok"
    assert summary["worst_slip_monthly"] == 0.31
    assert summary["promotion_gate_passed"] is False
    assert summary["promotion_failures"] == ["completed_windows 3 < 4"]


@pytest.mark.parametrize(
    ("gate", "expected"),
    [
        ({"passed": True}, True),
        ({"passed": False}, False),
        ({"pass": True}, True),
        ({"pass": False}, False),
        ({}, None),
        (None, None),
    ],
)
def test_promotion_gate_passed_supports_current_and_legacy_keys(
    gate: object,
    expected: bool | None,
) -> None:
    assert promotion_gate_passed(gate) is expected


def test_load_eval_artifact_summary_supports_legacy_promotion_pass_key(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "eval"
    out_dir.mkdir()
    artifact = out_dir / "best_eval100d.json"
    artifact.write_text(
        json.dumps(
            {
                "raw": {"status": "ok"},
                "aggregate": {"worst_slip_monthly": 0.31},
                "promotion_gate": {"pass": True, "failures": []},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = load_eval_artifact_summary(out_dir)

    assert summary["eval_artifact"] == str(artifact)
    assert summary["promotion_gate_passed"] is True


def test_load_eval_artifact_summary_handles_missing_artifact(tmp_path: Path) -> None:
    assert load_eval_artifact_summary(tmp_path) == {
        "eval_artifact": None,
        "eval_artifact_status": "missing",
    }


def test_load_eval_artifact_summary_uses_checkpoint_specific_artifact(tmp_path: Path) -> None:
    out_dir = tmp_path / "eval"
    out_dir.mkdir()
    stale = out_dir / "aaa_stale_eval100d.json"
    stale.write_text(
        json.dumps(
            {
                "raw": {"status": "ok"},
                "aggregate": {"worst_slip_monthly": 0.99},
                "promotion_gate": {"passed": True, "failures": []},
            }
        ),
        encoding="utf-8",
    )
    expected = out_dir / "best_eval100d.json"
    expected.write_text(
        json.dumps(
            {
                "raw": {"status": "ok"},
                "aggregate": {"worst_slip_monthly": 0.12},
                "promotion_gate": {"passed": False, "failures": ["under target"]},
            }
        ),
        encoding="utf-8",
    )

    summary = load_eval_artifact_summary(out_dir, checkpoint_stem="best")

    assert summary["eval_artifact"] == str(expected)
    assert summary["worst_slip_monthly"] == 0.12
    assert summary["promotion_gate_passed"] is False
    assert summary["promotion_failures"] == ["under target"]


def test_load_eval_artifact_summary_rejects_ambiguous_artifacts_without_checkpoint(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "eval"
    out_dir.mkdir()
    (out_dir / "a_eval100d.json").write_text("{}", encoding="utf-8")
    (out_dir / "b_eval100d.json").write_text("{}", encoding="utf-8")

    summary = load_eval_artifact_summary(out_dir)

    assert summary["eval_artifact_status"] == "ambiguous"
    assert summary["eval_artifact"] is None
    assert summary["eval_artifact_candidates"] == [
        str(out_dir / "a_eval100d.json"),
        str(out_dir / "b_eval100d.json"),
    ]


def test_load_eval_artifact_summary_handles_invalid_json(tmp_path: Path) -> None:
    out_dir = tmp_path / "eval"
    out_dir.mkdir()
    artifact = out_dir / "best_eval100d.json"
    artifact.write_text("{not-json\n", encoding="utf-8")

    summary = load_eval_artifact_summary(out_dir)

    assert summary["eval_artifact"] == str(artifact)
    assert summary["eval_artifact_status"] == "invalid_json"
    assert "eval_artifact_error" in summary


def test_render_summary_markdown_highlights_status_metrics_and_failures() -> None:
    markdown = render_summary_markdown(
        [
            {
                "description": "trial_a",
                "gpu_id": "0",
                "status": "ok",
                "returncode": 0,
                "promotion_gate_passed": True,
                "worst_slip_monthly": 0.31,
                "eval_dir": "evals/trial_a",
            },
            {
                "description": "trial_b",
                "gpu_id": "1",
                "status": "failed",
                "returncode": 2,
                "promotion_gate_passed": False,
                "worst_slip_monthly": 0.12,
                "promotion_failures": ["worst_slip_monthly 12.00% < 27.00%"],
                "eval_dir": "evals/trial_b",
            },
            {
                "description": "missing_trial",
                "gpu_id": "2",
                "status": "failed",
                "returncode": None,
                "wrapper_failure": "checkpoint_missing",
                "eval_dir": "evals/missing_trial",
            },
        ]
    )

    assert "# GPU Pool Candidate Eval Summary" in markdown
    assert "| trial_a | 0 | ok | 0 | pass | +31.00% | - | evals/trial_a |" in markdown
    assert "worst_slip_monthly 12.00% < 27.00%" in markdown
    assert "checkpoint_missing" in markdown


def test_write_text_atomic_overwrites_without_leaving_temp_file(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "summary.json"

    write_text_atomic(target, '{"status":"old"}\n')
    write_text_atomic(target, '{"status":"new"}\n')

    assert target.read_text(encoding="utf-8") == '{"status":"new"}\n'
    assert list(target.parent.glob(".summary.json.*.tmp")) == []


def test_wrapper_exit_code_for_eval_returncode_maps_signal_to_contract_failure() -> None:
    assert wrapper_exit_code_for_eval_returncode(3) == 3
    assert wrapper_exit_code_for_eval_returncode(-9) == 2


def test_main_returns_first_failed_eval_code_and_writes_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    leaderboard = tmp_path / "leaderboard.csv"
    leaderboard.write_text(
        "description,gpu_id,val_return\ntrial_a,0,0.30\ntrial_b,1,0.20\n",
        encoding="utf-8",
    )
    checkpoint_root = tmp_path / "checkpoints"
    for gpu_id, description in [("0", "trial_a"), ("1", "trial_b")]:
        trial_dir = checkpoint_root / f"gpu{gpu_id}" / description
        trial_dir.mkdir(parents=True)
        (trial_dir / "best.pt").write_bytes(b"checkpoint")
    out_root = tmp_path / "evals"
    returncodes = iter([0, 3])

    def fake_run_eval(cmd, *, log_path: Path) -> int:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("fake eval\n", encoding="utf-8")
        artifact = log_path.parent / "best_eval100d.json"
        passed = "trial_a" in str(log_path)
        (log_path.parent / "aaa_stale_eval100d.json").write_text(
            json.dumps(
                {
                    "raw": {"status": "ok"},
                    "aggregate": {"worst_slip_monthly": 0.99},
                    "promotion_gate": {"passed": True, "failures": []},
                }
            ),
            encoding="utf-8",
        )
        artifact.write_text(
            json.dumps(
                {
                    "raw": {"status": "ok"},
                    "aggregate": {"worst_slip_monthly": 0.31 if passed else 0.12},
                    "promotion_gate": {
                        "passed": passed,
                        "failures": [] if passed else ["worst_slip_monthly 12.00% < 27.00%"],
                    },
                }
            ),
            encoding="utf-8",
        )
        assert "--execution-granularity" in cmd
        return next(returncodes)

    monkeypatch.setattr(eval_top, "run_eval", fake_run_eval)

    rc = main(
        [
            "--leaderboard",
            str(leaderboard),
            "--checkpoint-root",
            str(checkpoint_root),
            "--val-data",
            str(tmp_path / "val.bin"),
            "--hourly-data-root",
            str(tmp_path / "hourly"),
            "--daily-start-date",
            "2025-08-01T00:00:00+00:00",
            "--out-root",
            str(out_root),
            "--top-k",
            "2",
        ]
    )

    summary = json.loads((out_root / "summary.json").read_text(encoding="utf-8"))
    markdown = (out_root / "summary.md").read_text(encoding="utf-8")
    assert rc == 3
    assert [row["returncode"] for row in summary] == [0, 3]
    assert [row["status"] for row in summary] == ["ok", "failed"]
    assert [row["eval_artifact_status"] for row in summary] == ["ok", "ok"]
    assert [row["promotion_gate_passed"] for row in summary] == [True, False]
    assert summary[1]["promotion_failures"] == ["worst_slip_monthly 12.00% < 27.00%"]
    assert "| trial_a | 0 | ok | 0 | pass | +31.00% | - |" in markdown
    assert "worst_slip_monthly 12.00% < 27.00%" in markdown


def test_main_maps_signaled_eval_returncode_to_contract_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    leaderboard = tmp_path / "leaderboard.csv"
    leaderboard.write_text("description,gpu_id,val_return\ntrial_a,0,0.30\n", encoding="utf-8")
    checkpoint_root = tmp_path / "checkpoints"
    trial_dir = checkpoint_root / "gpu0" / "trial_a"
    trial_dir.mkdir(parents=True)
    (trial_dir / "best.pt").write_bytes(b"checkpoint")
    out_root = tmp_path / "evals"

    def fake_run_eval(cmd, *, log_path: Path) -> int:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("killed by signal\n", encoding="utf-8")
        (log_path.parent / "best_eval100d.json").write_text(
            json.dumps(
                {
                    "raw": {"status": "error"},
                    "aggregate": {"worst_slip_monthly": None},
                    "promotion_gate": {"passed": False, "failures": ["eval signaled"]},
                }
            ),
            encoding="utf-8",
        )
        assert "--execution-granularity" in cmd
        return -9

    monkeypatch.setattr(eval_top, "run_eval", fake_run_eval)

    rc = main(
        [
            "--leaderboard",
            str(leaderboard),
            "--checkpoint-root",
            str(checkpoint_root),
            "--val-data",
            str(tmp_path / "val.bin"),
            "--hourly-data-root",
            str(tmp_path / "hourly"),
            "--daily-start-date",
            "2025-08-01T00:00:00+00:00",
            "--out-root",
            str(out_root),
            "--top-k",
            "1",
        ]
    )

    summary = json.loads((out_root / "summary.json").read_text(encoding="utf-8"))
    assert rc == 2
    assert summary[0]["returncode"] == -9
    assert summary[0]["status"] == "failed"
    assert summary[0]["promotion_gate_passed"] is False


def test_main_returns_contract_failure_for_invalid_leaderboard_schema(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch,
) -> None:
    leaderboard = tmp_path / "leaderboard.csv"
    leaderboard.write_text("description,gpu_id\ntrial_a,0\n", encoding="utf-8")
    out_root = tmp_path / "evals"
    eval_calls: list[Path] = []

    def fake_run_eval(cmd, *, log_path: Path) -> int:
        eval_calls.append(log_path)
        return 0

    monkeypatch.setattr(eval_top, "run_eval", fake_run_eval)

    rc = main(
        [
            "--leaderboard",
            str(leaderboard),
            "--checkpoint-root",
            str(tmp_path / "checkpoints"),
            "--val-data",
            str(tmp_path / "val.bin"),
            "--hourly-data-root",
            str(tmp_path / "hourly"),
            "--daily-start-date",
            "2025-08-01T00:00:00+00:00",
            "--out-root",
            str(out_root),
            "--top-k",
            "1",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 2
    assert eval_calls == []
    assert "missing required columns: val_return" in captured.err
    assert not (out_root / "summary.json").exists()


def test_main_returns_contract_failure_for_missing_leaderboard(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch,
) -> None:
    out_root = tmp_path / "evals"
    eval_calls: list[Path] = []

    def fake_run_eval(cmd, *, log_path: Path) -> int:
        eval_calls.append(log_path)
        return 0

    monkeypatch.setattr(eval_top, "run_eval", fake_run_eval)

    rc = main(
        [
            "--leaderboard",
            str(tmp_path / "missing.csv"),
            "--checkpoint-root",
            str(tmp_path / "checkpoints"),
            "--val-data",
            str(tmp_path / "val.bin"),
            "--hourly-data-root",
            str(tmp_path / "hourly"),
            "--daily-start-date",
            "2025-08-01T00:00:00+00:00",
            "--out-root",
            str(out_root),
            "--top-k",
            "1",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 2
    assert eval_calls == []
    assert "missing.csv" in captured.err
    assert not (out_root / "summary.json").exists()


def test_main_validates_selected_candidate_ids_before_any_eval(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch,
) -> None:
    leaderboard = tmp_path / "leaderboard.csv"
    leaderboard.write_text(
        "description,gpu_id,val_return\nsafe_trial,0,0.30\nunsafe/trial,1,0.20\n",
        encoding="utf-8",
    )
    checkpoint_root = tmp_path / "checkpoints"
    trial_dir = checkpoint_root / "gpu0" / "safe_trial"
    trial_dir.mkdir(parents=True)
    (trial_dir / "best.pt").write_bytes(b"checkpoint")
    out_root = tmp_path / "evals"
    eval_calls: list[Path] = []

    def fake_run_eval(cmd, *, log_path: Path) -> int:
        eval_calls.append(log_path)
        return 0

    monkeypatch.setattr(eval_top, "run_eval", fake_run_eval)

    rc = main(
        [
            "--leaderboard",
            str(leaderboard),
            "--checkpoint-root",
            str(checkpoint_root),
            "--val-data",
            str(tmp_path / "val.bin"),
            "--hourly-data-root",
            str(tmp_path / "hourly"),
            "--daily-start-date",
            "2025-08-01T00:00:00+00:00",
            "--out-root",
            str(out_root),
            "--top-k",
            "2",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 2
    assert eval_calls == []
    assert "safe path component" in captured.err
    assert not (out_root / "summary.json").exists()


def test_main_rejects_selected_rows_that_would_share_eval_directory(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch,
) -> None:
    leaderboard = tmp_path / "leaderboard.csv"
    leaderboard.write_text(
        "description,gpu_id,val_return\ntrial_a,0,0.40\ntrial_a,1,0.30\n",
        encoding="utf-8",
    )
    checkpoint_root = tmp_path / "checkpoints"
    for gpu_id in ["0", "1"]:
        trial_dir = checkpoint_root / f"gpu{gpu_id}" / "trial_a"
        trial_dir.mkdir(parents=True)
        (trial_dir / "best.pt").write_bytes(b"checkpoint")
    out_root = tmp_path / "evals"
    eval_calls: list[Path] = []

    def fake_run_eval(cmd, *, log_path: Path) -> int:
        eval_calls.append(log_path)
        return 0

    monkeypatch.setattr(eval_top, "run_eval", fake_run_eval)

    rc = main(
        [
            "--leaderboard",
            str(leaderboard),
            "--checkpoint-root",
            str(checkpoint_root),
            "--val-data",
            str(tmp_path / "val.bin"),
            "--hourly-data-root",
            str(tmp_path / "hourly"),
            "--daily-start-date",
            "2025-08-01T00:00:00+00:00",
            "--out-root",
            str(out_root),
            "--top-k",
            "2",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 2
    assert eval_calls == []
    assert "reuse eval output directories" in captured.err
    assert not (out_root / "summary.json").exists()


def test_main_fails_when_successful_eval_writes_no_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    leaderboard = tmp_path / "leaderboard.csv"
    leaderboard.write_text("description,gpu_id,val_return\ntrial_a,0,0.30\n", encoding="utf-8")
    checkpoint_root = tmp_path / "checkpoints"
    trial_dir = checkpoint_root / "gpu0" / "trial_a"
    trial_dir.mkdir(parents=True)
    (trial_dir / "best.pt").write_bytes(b"checkpoint")
    out_root = tmp_path / "evals"

    def fake_run_eval(cmd, *, log_path: Path) -> int:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("fake eval without artifact\n", encoding="utf-8")
        assert "--execution-granularity" in cmd
        return 0

    monkeypatch.setattr(eval_top, "run_eval", fake_run_eval)

    rc = main(
        [
            "--leaderboard",
            str(leaderboard),
            "--checkpoint-root",
            str(checkpoint_root),
            "--val-data",
            str(tmp_path / "val.bin"),
            "--hourly-data-root",
            str(tmp_path / "hourly"),
            "--daily-start-date",
            "2025-08-01T00:00:00+00:00",
            "--out-root",
            str(out_root),
            "--top-k",
            "1",
        ]
    )

    summary = json.loads((out_root / "summary.json").read_text(encoding="utf-8"))
    assert rc == 2
    assert summary[0]["status"] == "failed"
    assert summary[0]["returncode"] == 0
    assert summary[0]["eval_artifact_status"] == "missing"
    assert summary[0]["wrapper_failure"] == "eval_artifact_missing"


def test_main_fails_when_successful_eval_artifact_does_not_pass_gate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    leaderboard = tmp_path / "leaderboard.csv"
    leaderboard.write_text("description,gpu_id,val_return\ntrial_a,0,0.30\n", encoding="utf-8")
    checkpoint_root = tmp_path / "checkpoints"
    trial_dir = checkpoint_root / "gpu0" / "trial_a"
    trial_dir.mkdir(parents=True)
    (trial_dir / "best.pt").write_bytes(b"checkpoint")
    out_root = tmp_path / "evals"

    def fake_run_eval(cmd, *, log_path: Path) -> int:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("fake eval with failed gate\n", encoding="utf-8")
        (log_path.parent / "best_eval100d.json").write_text(
            json.dumps(
                {
                    "raw": {"status": "ok"},
                    "aggregate": {"worst_slip_monthly": 0.12},
                    "promotion_gate": {
                        "passed": False,
                        "failures": ["worst_slip_monthly 12.00% < 27.00%"],
                    },
                }
            ),
            encoding="utf-8",
        )
        assert "--execution-granularity" in cmd
        return 0

    monkeypatch.setattr(eval_top, "run_eval", fake_run_eval)

    rc = main(
        [
            "--leaderboard",
            str(leaderboard),
            "--checkpoint-root",
            str(checkpoint_root),
            "--val-data",
            str(tmp_path / "val.bin"),
            "--hourly-data-root",
            str(tmp_path / "hourly"),
            "--daily-start-date",
            "2025-08-01T00:00:00+00:00",
            "--out-root",
            str(out_root),
            "--top-k",
            "1",
        ]
    )

    summary = json.loads((out_root / "summary.json").read_text(encoding="utf-8"))
    assert rc == 2
    assert summary[0]["status"] == "failed"
    assert summary[0]["returncode"] == 0
    assert summary[0]["eval_artifact_status"] == "ok"
    assert summary[0]["promotion_gate_passed"] is False
    assert summary[0]["wrapper_failure"] == "eval_success_without_passing_promotion_gate"


def test_main_fails_when_successful_eval_artifact_raw_status_is_not_ok(
    tmp_path: Path,
    monkeypatch,
) -> None:
    leaderboard = tmp_path / "leaderboard.csv"
    leaderboard.write_text("description,gpu_id,val_return\ntrial_a,0,0.30\n", encoding="utf-8")
    checkpoint_root = tmp_path / "checkpoints"
    trial_dir = checkpoint_root / "gpu0" / "trial_a"
    trial_dir.mkdir(parents=True)
    (trial_dir / "best.pt").write_bytes(b"checkpoint")
    out_root = tmp_path / "evals"

    def fake_run_eval(cmd, *, log_path: Path) -> int:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("fake eval with contradictory artifact\n", encoding="utf-8")
        (log_path.parent / "best_eval100d.json").write_text(
            json.dumps(
                {
                    "raw": {"status": "error"},
                    "aggregate": {"worst_slip_monthly": 0.31},
                    "promotion_gate": {"passed": True, "failures": []},
                }
            ),
            encoding="utf-8",
        )
        assert "--execution-granularity" in cmd
        return 0

    monkeypatch.setattr(eval_top, "run_eval", fake_run_eval)

    rc = main(
        [
            "--leaderboard",
            str(leaderboard),
            "--checkpoint-root",
            str(checkpoint_root),
            "--val-data",
            str(tmp_path / "val.bin"),
            "--hourly-data-root",
            str(tmp_path / "hourly"),
            "--daily-start-date",
            "2025-08-01T00:00:00+00:00",
            "--out-root",
            str(out_root),
            "--top-k",
            "1",
        ]
    )

    summary = json.loads((out_root / "summary.json").read_text(encoding="utf-8"))
    assert rc == 2
    assert summary[0]["status"] == "failed"
    assert summary[0]["returncode"] == 0
    assert summary[0]["raw_status"] == "error"
    assert summary[0]["promotion_gate_passed"] is True
    assert summary[0]["wrapper_failure"] == "eval_success_with_raw_status_error"


def test_main_records_missing_checkpoint_and_continues_to_next_candidate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    leaderboard = tmp_path / "leaderboard.csv"
    leaderboard.write_text(
        "description,gpu_id,val_return\nmissing_trial,0,0.40\ntrial_b,1,0.30\n",
        encoding="utf-8",
    )
    checkpoint_root = tmp_path / "checkpoints"
    trial_dir = checkpoint_root / "gpu1" / "trial_b"
    trial_dir.mkdir(parents=True)
    (trial_dir / "best.pt").write_bytes(b"checkpoint")
    out_root = tmp_path / "evals"
    seen_logs: list[Path] = []

    def fake_run_eval(cmd, *, log_path: Path) -> int:
        seen_logs.append(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("fake eval\n", encoding="utf-8")
        (log_path.parent / "best_eval100d.json").write_text(
            json.dumps(
                {
                    "raw": {"status": "ok"},
                    "aggregate": {"worst_slip_monthly": 0.31},
                    "promotion_gate": {"passed": True, "failures": []},
                }
            ),
            encoding="utf-8",
        )
        assert "--execution-granularity" in cmd
        return 0

    monkeypatch.setattr(eval_top, "run_eval", fake_run_eval)

    rc = main(
        [
            "--leaderboard",
            str(leaderboard),
            "--checkpoint-root",
            str(checkpoint_root),
            "--val-data",
            str(tmp_path / "val.bin"),
            "--hourly-data-root",
            str(tmp_path / "hourly"),
            "--daily-start-date",
            "2025-08-01T00:00:00+00:00",
            "--out-root",
            str(out_root),
            "--top-k",
            "2",
        ]
    )

    summary = json.loads((out_root / "summary.json").read_text(encoding="utf-8"))
    markdown = (out_root / "summary.md").read_text(encoding="utf-8")
    assert rc == 2
    assert len(seen_logs) == 1
    assert seen_logs[0] == out_root / "trial_b" / "eval100d.log"
    assert [row["description"] for row in summary] == ["missing_trial", "trial_b"]
    assert summary[0]["status"] == "failed"
    assert summary[0]["wrapper_failure"] == "checkpoint_missing"
    assert summary[0]["eval_artifact_status"] == "not_run"
    assert summary[1]["status"] == "ok"
    assert summary[1]["promotion_gate_passed"] is True
    assert "checkpoint_missing" in markdown
    assert "| trial_b | 1 | ok | 0 | pass | +31.00% | - |" in markdown


def test_main_records_eval_launch_exception_and_continues_to_next_candidate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    leaderboard = tmp_path / "leaderboard.csv"
    leaderboard.write_text(
        "description,gpu_id,val_return\nbroken_trial,0,0.40\ntrial_b,1,0.30\n",
        encoding="utf-8",
    )
    checkpoint_root = tmp_path / "checkpoints"
    for gpu_id, description in [("0", "broken_trial"), ("1", "trial_b")]:
        trial_dir = checkpoint_root / f"gpu{gpu_id}" / description
        trial_dir.mkdir(parents=True)
        (trial_dir / "best.pt").write_bytes(b"checkpoint")
    out_root = tmp_path / "evals"
    seen_logs: list[Path] = []

    def fake_run_eval(cmd, *, log_path: Path) -> int:
        seen_logs.append(log_path)
        if "broken_trial" in str(log_path):
            raise OSError("subprocess launch failed")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("fake eval\n", encoding="utf-8")
        (log_path.parent / "best_eval100d.json").write_text(
            json.dumps(
                {
                    "raw": {"status": "ok"},
                    "aggregate": {"worst_slip_monthly": 0.31},
                    "promotion_gate": {"passed": True, "failures": []},
                }
            ),
            encoding="utf-8",
        )
        assert "--execution-granularity" in cmd
        return 0

    monkeypatch.setattr(eval_top, "run_eval", fake_run_eval)

    rc = main(
        [
            "--leaderboard",
            str(leaderboard),
            "--checkpoint-root",
            str(checkpoint_root),
            "--val-data",
            str(tmp_path / "val.bin"),
            "--hourly-data-root",
            str(tmp_path / "hourly"),
            "--daily-start-date",
            "2025-08-01T00:00:00+00:00",
            "--out-root",
            str(out_root),
            "--top-k",
            "2",
        ]
    )

    summary = json.loads((out_root / "summary.json").read_text(encoding="utf-8"))
    markdown = (out_root / "summary.md").read_text(encoding="utf-8")
    assert rc == 2
    assert seen_logs == [
        out_root / "broken_trial" / "eval100d.log",
        out_root / "trial_b" / "eval100d.log",
    ]
    assert [row["description"] for row in summary] == ["broken_trial", "trial_b"]
    assert summary[0]["status"] == "failed"
    assert summary[0]["wrapper_failure"] == "eval_launch_failed"
    assert summary[0]["wrapper_error"] == "subprocess launch failed"
    assert summary[0]["eval_artifact_status"] == "not_run"
    assert summary[1]["status"] == "ok"
    assert summary[1]["promotion_gate_passed"] is True
    assert "eval_launch_failed" in markdown
    assert "| trial_b | 1 | ok | 0 | pass | +31.00% | - |" in markdown
