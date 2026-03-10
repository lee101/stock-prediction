from __future__ import annotations

import json
import sys
from pathlib import Path

from src.autoresearch_stock import agent_scheduler


def _resolve_record_path(repo_root: Path, value: str | None) -> Path:
    assert value is not None
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _write_minimal_repo(root: Path) -> None:
    train_path = root / "src/autoresearch_stock/train.py"
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_path.write_text(
        "from __future__ import annotations\n\n"
        "def main() -> int:\n"
        "    return 0\n",
        encoding="utf-8",
    )


def test_parse_train_log_extracts_metrics(tmp_path: Path) -> None:
    log_path = tmp_path / "train.log"
    log_path.write_text(
        "\n".join(
            [
                "robust_score:      -12.345678",
                "val_loss:          0.004321",
                "training_seconds:  300.0",
                "total_seconds:     305.4",
                "peak_vram_mb:      165.1",
                "scenario_count:    3",
                "total_trade_count: 706",
                "train_samples:     9638",
                "num_steps:         17861",
                "frequency:         hourly",
                "hold_bars:         6",
            ]
        ),
        encoding="utf-8",
    )

    metrics = agent_scheduler.parse_train_log(log_path)

    assert metrics.robust_score == -12.345678
    assert metrics.val_loss == 0.004321
    assert metrics.training_seconds == 300.0
    assert metrics.total_seconds == 305.4
    assert metrics.peak_vram_mb == 165.1
    assert metrics.scenario_count == 3
    assert metrics.total_trade_count == 706
    assert metrics.train_samples == 9638
    assert metrics.num_steps == 17861
    assert metrics.frequency == "hourly"
    assert metrics.hold_bars == 6


def test_run_turn_records_failed_result_when_agent_json_is_invalid(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    analysis_dir = tmp_path / "analysis"
    experiment_bundle_root = tmp_path / "experiments"
    _write_minimal_repo(repo_root)

    def fake_run_subprocess(
        *,
        command: list[str],
        cwd: Path,
        stdout_path: Path,
        stderr_path: Path,
        stdin_text: str | None = None,
    ) -> int:
        del command, cwd, stdin_text
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return 0

    def fake_load_agent_result(path: Path) -> dict[str, object]:
        del path
        raise json.JSONDecodeError("bad json", "", 0)

    monkeypatch.setattr(agent_scheduler, "_run_subprocess", fake_run_subprocess)
    monkeypatch.setattr(agent_scheduler, "_load_agent_result", fake_load_agent_result)

    record = agent_scheduler.run_turn(
        repo_root=repo_root,
        analysis_dir=analysis_dir,
        experiment_bundle_root=experiment_bundle_root,
        python_path=Path(sys.executable),
        selection=agent_scheduler.TurnSelection(turn_index=1, backend="codex", frequency="hourly"),
        codex_model=None,
        codex_reasoning_effort="high",
        claude_model=None,
        claude_effort="high",
        prompt_sections=[],
        extra_prompt=None,
    )

    assert record.status == "failed"
    assert "invalid agent JSON output" in record.summary
    assert record.agent_exit_code == 0
    assert record.train_py_changed is False
    assert record.diff_path is None
    assert record.notes
    assert "invalid agent JSON output" in record.notes[0]
    assert record.experiment_dir is not None
    bundle_dir = _resolve_record_path(repo_root, record.experiment_dir)
    assert bundle_dir.exists()
    assert (bundle_dir / "prompt.md").exists()
    assert (bundle_dir / "train.py.before").exists()
    assert (bundle_dir / "train.py.after").exists()
    assert (bundle_dir / "metadata.json").exists()


def test_collect_backend_statuses_disables_claude_by_default(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []

    def fake_probe_backend(
        backend: str,
        *,
        repo_root: Path,
        schema_path: Path,
        probe_dir: Path,
        codex_model: str | None,
        codex_reasoning_effort: str,
        claude_model: str | None,
        claude_effort: str,
    ) -> agent_scheduler.BackendStatus:
        del repo_root, schema_path, probe_dir, codex_model, codex_reasoning_effort, claude_model, claude_effort
        calls.append(backend)
        return agent_scheduler.BackendStatus(name=backend, available=True, reason="ok")

    monkeypatch.setattr(agent_scheduler, "probe_backend", fake_probe_backend)

    statuses = agent_scheduler.collect_backend_statuses(
        ["codex", "claude"],
        skip_probe=False,
        allow_claude=False,
        repo_root=tmp_path,
        schema_path=agent_scheduler.TURN_SCHEMA_PATH,
        probe_dir=tmp_path / "probe",
        codex_model=None,
        codex_reasoning_effort="xhigh",
        claude_model=None,
        claude_effort="high",
    )

    assert calls == ["codex"]
    assert [status.name for status in statuses] == ["codex", "claude"]
    assert statuses[0].available is True
    assert statuses[1].available is False
    assert "disabled by default" in statuses[1].reason


def test_resolve_python_path_preserves_virtualenv_symlink_path(tmp_path: Path) -> None:
    bin_dir = tmp_path / ".venv312/bin"
    bin_dir.mkdir(parents=True)
    link_path = bin_dir / "python"
    link_path.symlink_to(sys.executable)

    resolved = agent_scheduler.resolve_python_path(".venv312/bin/python", repo_root=tmp_path)

    assert resolved == link_path.absolute()


def test_resolve_repo_path_uses_repo_root_for_relative_paths(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    resolved = agent_scheduler.resolve_repo_path("analysis/test", repo_root=repo_root)

    assert resolved == (repo_root / "analysis/test").resolve()


def test_load_prompt_files_reads_repo_relative_paths(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    prompt_path = repo_root / "docs/prompt.md"
    prompt_path.parent.mkdir(parents=True)
    prompt_path.write_text("hello prompt", encoding="utf-8")

    sections = agent_scheduler.load_prompt_files(["docs/prompt.md"], repo_root=repo_root)

    assert sections == [("docs/prompt.md", "hello prompt")]


def test_run_turn_writes_prompt_packs_and_bundle_files(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    analysis_dir = tmp_path / "analysis"
    experiment_bundle_root = tmp_path / "experiments"
    _write_minimal_repo(repo_root)

    def fake_run_subprocess(
        *,
        command: list[str],
        cwd: Path,
        stdout_path: Path,
        stderr_path: Path,
        stdin_text: str | None = None,
    ) -> int:
        del command, cwd, stdin_text
        run_dir = stdout_path.parent
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        (run_dir / "agent_result.json").write_text(
            json.dumps(
                {
                    "status": "success",
                    "summary": "ok",
                    "touched_files": ["src/autoresearch_stock/train.py"],
                    "train_log": str(run_dir / "train_hourly.log"),
                    "robust_score": -1.0,
                    "val_loss": 0.1,
                    "training_seconds": 300.0,
                    "total_seconds": 305.0,
                    "peak_vram_mb": 123.0,
                    "num_steps": 456,
                    "notes": ["note"],
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "train_hourly.log").write_text(
            "\n".join(
                [
                    "robust_score:      -1.000000",
                    "val_loss:          0.100000",
                    "training_seconds:  300.0",
                    "total_seconds:     305.0",
                    "peak_vram_mb:      123.0",
                    "scenario_count:    3",
                    "total_trade_count: 10",
                    "train_samples:     20",
                    "num_steps:         456",
                    "frequency:         hourly",
                    "hold_bars:         6",
                ]
            ),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(agent_scheduler, "_run_subprocess", fake_run_subprocess)
    prompt_sections = [("src/autoresearch_stock/prompts/hourly_selectivity.md", "prompt text")]

    record = agent_scheduler.run_turn(
        repo_root=repo_root,
        analysis_dir=analysis_dir,
        experiment_bundle_root=experiment_bundle_root,
        python_path=Path(sys.executable),
        selection=agent_scheduler.TurnSelection(turn_index=1, backend="codex", frequency="hourly"),
        codex_model=None,
        codex_reasoning_effort="high",
        claude_model=None,
        claude_effort="high",
        prompt_sections=prompt_sections,
        extra_prompt="extra",
    )

    bundle_dir = _resolve_record_path(repo_root, record.experiment_dir)
    assert record.status == "success"
    assert record.train_log_path is not None
    assert bundle_dir.exists()
    assert (bundle_dir / "benchmark_command.sh").exists()
    assert (bundle_dir / "agent_result.json").exists()
    assert (bundle_dir / "train_hourly.log").exists()
    assert (bundle_dir / "metadata.json").exists()
    copied_prompt = bundle_dir / "prompt_packs/00_hourly_selectivity.md"
    assert copied_prompt.exists()
    assert copied_prompt.read_text(encoding="utf-8").strip() == "prompt text"
