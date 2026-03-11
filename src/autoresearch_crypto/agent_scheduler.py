"""Agent scheduler for crypto RL autoresearch.

Fork of src/autoresearch_stock/agent_scheduler.py adapted for crypto training.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


DEFAULT_ANALYSIS_DIR = Path("analysis/autoresearch_crypto_agent")
DEFAULT_EXPERIMENT_BUNDLE_ROOT = Path("experiments/autoresearch_crypto_agent")
DEFAULT_EXPERIMENT_CODE_ROOT = Path("src/autoresearch_crypto/experiments")
DEFAULT_PYTHON = Path(".venv313/bin/python")
DEFAULT_BACKENDS = ("codex",)
DEFAULT_FREQUENCIES = ("crypto",)
TURN_SCHEMA_PATH = Path(__file__).with_name("agent_turn_schema.json")
STATE_FILENAME = "state.json"
HISTORY_FILENAME = "turns.jsonl"
LEADERBOARD_FILENAME = "leaderboard.tsv"
PROBE_TIMEOUT_PROMPT = 'Reply with JSON only: {"status":"ok","summary":"probe","touched_files":[],"train_log":null,"robust_score":null,"val_loss":null,"training_seconds":null,"total_seconds":null,"peak_vram_mb":null,"num_steps":null,"notes":[]}'

TRAIN_PY_RELATIVE = "src/autoresearch_crypto/train.py"
MODULE_NAME = "autoresearch_crypto"

MetricValue = float | int | str | None
BackendName = Literal["codex", "claude"]


@dataclass
class BackendStatus:
    name: str
    available: bool
    reason: str


@dataclass
class TurnSelection:
    turn_index: int
    backend: str
    frequency: str


@dataclass
class ParsedTrainMetrics:
    robust_score: float | None = None
    val_loss: float | None = None
    training_seconds: float | None = None
    total_seconds: float | None = None
    peak_vram_mb: float | None = None
    scenario_count: int | None = None
    total_trade_count: int | None = None
    num_steps: int | None = None
    num_epochs: int | None = None
    symbols: str | None = None


@dataclass
class TurnRecord:
    turn_index: int
    started_at_utc: str
    finished_at_utc: str
    elapsed_seconds: float
    backend: str
    frequency: str
    status: str
    summary: str
    experiment_dir: str | None
    prompt_path: str
    result_path: str
    stdout_path: str
    stderr_path: str
    train_log_path: str | None
    diff_path: str | None
    touched_files: list[str]
    train_py_changed: bool
    train_py_before_sha256: str
    train_py_after_sha256: str
    agent_exit_code: int
    robust_score: float | None
    val_loss: float | None
    training_seconds: float | None
    total_seconds: float | None
    peak_vram_mb: float | None
    scenario_count: int | None
    total_trade_count: int | None
    num_steps: int | None
    notes: list[str]


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _timestamp_token(dt: datetime | None = None) -> str:
    return (dt or _utc_now()).strftime("%Y%m%dT%H%M%SZ")


def _parse_csv_list(raw: str | None, *, lower: bool = False) -> list[str]:
    if raw is None:
        return []
    values = [token.strip() for token in str(raw).split(",") if token.strip()]
    if lower:
        values = [value.lower() for value in values]
    return values


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path, *, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    _ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    _ensure_parent(path)
    path.write_text(text, encoding="utf-8")


def _sha256_text(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _snapshot_train_py(repo_root: Path) -> tuple[str, str]:
    path = repo_root / TRAIN_PY_RELATIVE
    text = path.read_text(encoding="utf-8")
    return _sha256_text(text), text


def _write_diff(path: Path, before: str, after: str, *, fromfile: str, tofile: str) -> bool:
    if before == after:
        return False
    diff = "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=fromfile,
            tofile=tofile,
        )
    )
    _ensure_parent(path)
    path.write_text(diff, encoding="utf-8")
    return True


def _load_history(history_path: Path, *, tail: int = 5) -> list[dict[str, Any]]:
    if not history_path.exists():
        return []
    rows = []
    with history_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows[-tail:]


def _append_history(history_path: Path, row: dict[str, Any]) -> None:
    _ensure_parent(history_path)
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _append_leaderboard_row(path: Path, record: TurnRecord) -> None:
    _ensure_parent(path)
    header = (
        "turn\tstarted_at_utc\tbackend\tfrequency\tstatus\trobust_score\tval_loss\ttraining_seconds\t"
        "total_seconds\tpeak_vram_mb\tscenario_count\ttotal_trade_count\tnum_steps\t"
        "train_py_changed\ttouched_files\tsummary\n"
    )
    write_header = not path.exists()
    with path.open("a", encoding="utf-8") as handle:
        if write_header:
            handle.write(header)
        fields = [
            str(record.turn_index),
            record.started_at_utc,
            record.backend,
            record.frequency,
            record.status,
            "" if record.robust_score is None else f"{record.robust_score:.6f}",
            "" if record.val_loss is None else f"{record.val_loss:.6f}",
            "" if record.training_seconds is None else f"{record.training_seconds:.1f}",
            "" if record.total_seconds is None else f"{record.total_seconds:.1f}",
            "" if record.peak_vram_mb is None else f"{record.peak_vram_mb:.1f}",
            "" if record.scenario_count is None else str(record.scenario_count),
            "" if record.total_trade_count is None else str(record.total_trade_count),
            "" if record.num_steps is None else str(record.num_steps),
            "1" if record.train_py_changed else "0",
            ",".join(record.touched_files),
            record.summary.replace("\t", " ").replace("\n", " ").strip(),
        ]
        handle.write("\t".join(fields) + "\n")


def parse_train_log(log_path: Path) -> ParsedTrainMetrics:
    metrics = ParsedTrainMetrics()
    if not log_path.exists():
        return metrics

    pattern = re.compile(r"^([a-z_]+):\s+(.+?)\s*$")
    text = log_path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        key, raw_value = match.groups()
        value: MetricValue
        try:
            value = float(raw_value)
        except ValueError:
            value = raw_value

        if key == "robust_score" and isinstance(value, float):
            metrics.robust_score = value
        elif key == "val_loss" and isinstance(value, float):
            metrics.val_loss = value
        elif key == "training_seconds" and isinstance(value, float):
            metrics.training_seconds = value
        elif key == "total_seconds" and isinstance(value, float):
            metrics.total_seconds = value
        elif key == "peak_vram_mb" and isinstance(value, float):
            metrics.peak_vram_mb = value
        elif key == "scenario_count" and isinstance(value, float):
            metrics.scenario_count = int(value)
        elif key == "total_trade_count" and isinstance(value, float):
            metrics.total_trade_count = int(value)
        elif key == "num_steps" and isinstance(value, float):
            metrics.num_steps = int(value)
        elif key == "num_epochs" and isinstance(value, float):
            metrics.num_epochs = int(value)
        elif key == "symbols" and isinstance(value, str):
            metrics.symbols = value
    return metrics


def _relative_or_absolute(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except Exception:
        return str(path.resolve())


def resolve_python_path(raw_path: str, *, repo_root: Path) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return Path(os.path.abspath(str(candidate)))


def resolve_repo_path(raw_path: str, *, repo_root: Path) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve()


def load_prompt_files(raw_values: list[str], *, repo_root: Path) -> list[tuple[str, str]]:
    prompt_sections: list[tuple[str, str]] = []
    for raw in raw_values:
        for token in _parse_csv_list(raw):
            path = Path(token).expanduser()
            if not path.is_absolute():
                path = repo_root / path
            resolved = path.resolve()
            text = resolved.read_text(encoding="utf-8").strip()
            prompt_sections.append((_relative_or_absolute(resolved, repo_root), text))
    return prompt_sections


def _copy_if_exists(source: Path | None, destination: Path) -> None:
    if source is None or not source.exists():
        return
    _ensure_parent(destination)
    shutil.copyfile(source, destination)


def _history_summary_lines(history_rows: list[dict[str, Any]]) -> list[str]:
    if not history_rows:
        return ["No prior turns recorded yet."]
    lines: list[str] = []
    for row in history_rows:
        robust = row.get("robust_score")
        robust_text = "n/a" if robust is None else f"{float(robust):.6f}"
        val_loss = row.get("val_loss")
        val_text = "n/a" if val_loss is None else f"{float(val_loss):.6f}"
        lines.append(
            f"- turn {row.get('turn_index')} | backend={row.get('backend')} "
            f"| status={row.get('status')} | robust_score={robust_text} | val_loss={val_text} "
            f"| summary={row.get('summary', '').strip()}"
        )
    return lines


def build_turn_prompt(
    *,
    repo_root: Path,
    run_dir: Path,
    experiment_dir: Path,
    frequency: str,
    python_path: Path,
    recent_history: list[dict[str, Any]],
    prompt_sections: list[tuple[str, str]],
    extra_prompt: str | None,
) -> str:
    train_log = run_dir / "train_crypto.log"
    result_json = run_dir / "agent_result.json"
    experiment_code_root = repo_root / DEFAULT_EXPERIMENT_CODE_ROOT
    prompt = f"""
You are running one bounded optimization turn for the crypto RL autoresearch framework.

Repository root: {repo_root}
Python interpreter: {python_path}
Run directory: {run_dir}
Experiment bundle directory: {experiment_dir}
Experiment code root: {experiment_code_root}
Training log path: {train_log}
Result file path: {result_json}

Primary goal: maximize `robust_score`
Secondary goal: minimize `val_loss`

The model is a transformer policy that outputs limit order prices and trade amounts.
Training uses a differentiable market simulator with realistic fees (10bps), margin
interest (6.25%), 2x leverage, and fill probabilities. Evaluation uses binary fills
on 30-day holdout data across DOGEUSD and AAVEUSD.

Hard rules:
- Keep changes deterministic and replayable.
- Isolate multi-file experiments under `src/autoresearch_crypto/experiments/<slug>/`.
- Keep `src/autoresearch_crypto/train.py` as a small dispatcher.
- Do NOT edit `src/autoresearch_crypto/prepare.py`.
- Do NOT fake the benchmark. Run one real training run with the fixed harness.
- Use the exact interpreter shown above.

Read first:
- `src/autoresearch_crypto/program.md`
- `src/autoresearch_crypto/prepare.py`
- `src/autoresearch_crypto/train.py`
- `binanceneural/config.py` (TrainingConfig, PolicyConfig)
- `binanceneural/model.py` (BinancePolicyBase, build_policy)
- `differentiable_loss_utils.py` (simulate_hourly_trades, compute_loss_by_type)

Recent history:
{os.linesep.join(_history_summary_lines(recent_history))}

Required training command:
`{python_path} -m {MODULE_NAME}.train > {train_log} 2>&1`

Required workflow:
1. Read the current training code and recent history.
2. Make one coherent improvement attempt.
3. Run the training command above exactly once.
4. Parse the summary lines from the log.
5. Return structured JSON response only.

Response format:
- `status`: `"success"`, `"no_change"`, or `"failed"`
- `summary`: one short sentence
- `touched_files`: changed file paths
- `train_log`: path to the log file
- `robust_score`, `val_loss`, `training_seconds`, `total_seconds`, `peak_vram_mb`, `num_steps`: parsed or null
- `notes`: short list of key observations
""".strip()
    if prompt_sections:
        prompt += "\n\nLoaded prompt packs:\n"
        for relative_path, text in prompt_sections:
            prompt += f"\n### {relative_path}\n{text}\n"
    if extra_prompt:
        prompt += "\n\nExtra instruction:\n" + extra_prompt.strip()
    return prompt + "\n"


def _build_codex_command(
    *,
    repo_root: Path,
    schema_path: Path,
    result_path: Path,
    model: str | None,
    reasoning_effort: str,
) -> list[str]:
    cmd = [
        "codex",
        "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        "--skip-git-repo-check",
        "-C",
        str(repo_root),
        "--output-schema",
        str(schema_path),
        "-o",
        str(result_path),
        "-c",
        f'model_reasoning_effort="{reasoning_effort}"',
        "-",
    ]
    if model:
        cmd[2:2] = ["--model", model]
    return cmd


def _build_claude_command(
    *,
    schema_path: Path,
    model: str | None,
    effort: str,
    prompt: str,
) -> list[str]:
    schema_text = schema_path.read_text(encoding="utf-8")
    cmd = [
        "claude",
        "-p",
        "--dangerously-skip-permissions",
        "--output-format",
        "json",
        "--json-schema",
        schema_text,
        "--effort",
        effort,
    ]
    if model:
        cmd.extend(["--model", model])
    cmd.append(prompt)
    return cmd


def _run_subprocess(
    *,
    command: list[str],
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    stdin_text: str | None = None,
) -> int:
    _ensure_parent(stdout_path)
    _ensure_parent(stderr_path)
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            input=stdin_text,
            text=True,
            stdout=stdout_handle,
            stderr=stderr_handle,
            check=False,
        )
    return int(proc.returncode)


def _load_agent_result(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    return json.loads(text)


def probe_backend(
    backend: BackendName,
    *,
    repo_root: Path,
    schema_path: Path,
    probe_dir: Path,
    codex_model: str | None,
    codex_reasoning_effort: str,
    claude_model: str | None,
    claude_effort: str,
) -> BackendStatus:
    if shutil.which(backend) is None:
        return BackendStatus(name=backend, available=False, reason="command not found in PATH")

    probe_dir.mkdir(parents=True, exist_ok=True)
    result_path = probe_dir / f"{backend}_probe_result.json"
    stdout_path = probe_dir / f"{backend}_probe_stdout.txt"
    stderr_path = probe_dir / f"{backend}_probe_stderr.txt"
    prompt = PROBE_TIMEOUT_PROMPT

    if backend == "codex":
        command = _build_codex_command(
            repo_root=repo_root,
            schema_path=schema_path,
            result_path=result_path,
            model=codex_model,
            reasoning_effort=codex_reasoning_effort,
        )
        return_code = _run_subprocess(
            command=command,
            cwd=repo_root,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            stdin_text=prompt,
        )
    else:
        command = _build_claude_command(
            schema_path=schema_path,
            model=claude_model,
            effort=claude_effort,
            prompt=prompt,
        )
        return_code = _run_subprocess(
            command=command,
            cwd=repo_root,
            stdout_path=result_path,
            stderr_path=stderr_path,
        )
        if result_path.exists() and not stdout_path.exists():
            shutil.copyfile(result_path, stdout_path)

    if return_code != 0:
        reason = stderr_path.read_text(encoding="utf-8", errors="replace").strip()
        if not reason:
            reason = stdout_path.read_text(encoding="utf-8", errors="replace").strip()
        return BackendStatus(name=backend, available=False, reason=reason or f"exit code {return_code}")

    try:
        payload = _load_agent_result(result_path if backend == "codex" else stdout_path)
    except Exception as exc:
        return BackendStatus(name=backend, available=False, reason=f"invalid JSON output: {exc}")

    if payload.get("status") is None:
        return BackendStatus(name=backend, available=False, reason="probe did not return structured status")
    return BackendStatus(name=backend, available=True, reason="ok")


def collect_backend_statuses(
    requested_backends: list[str],
    *,
    skip_probe: bool,
    allow_claude: bool,
    repo_root: Path,
    schema_path: Path,
    probe_dir: Path,
    codex_model: str | None,
    codex_reasoning_effort: str,
    claude_model: str | None,
    claude_effort: str,
) -> list[BackendStatus]:
    backend_statuses: list[BackendStatus] = []
    if skip_probe:
        for backend in requested_backends:
            if backend == "claude" and not allow_claude:
                backend_statuses.append(
                    BackendStatus(name=backend, available=False, reason="Claude backend disabled; pass --allow-claude.")
                )
            elif shutil.which(backend) is not None:
                backend_statuses.append(BackendStatus(name=backend, available=True, reason="probe skipped"))
            else:
                backend_statuses.append(BackendStatus(name=backend, available=False, reason="command not found in PATH"))
        return backend_statuses

    for backend_name in requested_backends:
        if backend_name == "claude" and not allow_claude:
            backend_statuses.append(
                BackendStatus(name=backend_name, available=False, reason="Claude backend disabled; pass --allow-claude.")
            )
            continue
        backend_statuses.append(
            probe_backend(
                backend_name,
                repo_root=repo_root,
                schema_path=schema_path,
                probe_dir=probe_dir,
                codex_model=codex_model,
                codex_reasoning_effort=codex_reasoning_effort,
                claude_model=claude_model,
                claude_effort=claude_effort,
            )
        )
    return backend_statuses


def select_turn(
    *,
    turn_index: int,
    backends: list[str],
    frequencies: list[str],
) -> TurnSelection:
    backend = backends[(turn_index - 1) % len(backends)]
    frequency = frequencies[(turn_index - 1) % len(frequencies)]
    return TurnSelection(turn_index=turn_index, backend=backend, frequency=frequency)


def run_turn(
    *,
    repo_root: Path,
    analysis_dir: Path,
    experiment_bundle_root: Path,
    python_path: Path,
    selection: TurnSelection,
    codex_model: str | None,
    codex_reasoning_effort: str,
    claude_model: str | None,
    claude_effort: str,
    prompt_sections: list[tuple[str, str]],
    extra_prompt: str | None,
) -> TurnRecord:
    started_at = _utc_now()
    turn_name = f"turn_{selection.turn_index:04d}_{selection.backend}_{_timestamp_token(started_at)}"
    run_dir = analysis_dir / turn_name
    run_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir = experiment_bundle_root / turn_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = run_dir / "prompt.md"
    result_path = run_dir / "agent_result.json"
    stdout_path = run_dir / "backend_stdout.txt"
    stderr_path = run_dir / "backend_stderr.txt"
    diff_path = run_dir / "train_py.diff"
    history_path = analysis_dir / HISTORY_FILENAME

    before_sha, before_text = _snapshot_train_py(repo_root)
    prompt = build_turn_prompt(
        repo_root=repo_root,
        run_dir=run_dir,
        experiment_dir=experiment_dir,
        frequency=selection.frequency,
        python_path=python_path,
        recent_history=_load_history(history_path, tail=5),
        prompt_sections=prompt_sections,
        extra_prompt=extra_prompt,
    )
    _write_text(prompt_path, prompt)
    _write_text(experiment_dir / "prompt.md", prompt)
    _write_text(
        experiment_dir / "benchmark_command.sh",
        "\n".join([
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f'cd "{repo_root}"',
            f'"{python_path}" -m {MODULE_NAME}.train > "{run_dir / "train_crypto.log"}" 2>&1',
            "",
        ]),
    )
    _write_text(experiment_dir / "train.py.before", before_text)
    if prompt_sections:
        for index, (relative_path, text) in enumerate(prompt_sections):
            label = Path(relative_path).name
            _write_text(experiment_dir / "prompt_packs" / f"{index:02d}_{label}", text + "\n")

    if selection.backend == "codex":
        command = _build_codex_command(
            repo_root=repo_root,
            schema_path=TURN_SCHEMA_PATH,
            result_path=result_path,
            model=codex_model,
            reasoning_effort=codex_reasoning_effort,
        )
        exit_code = _run_subprocess(
            command=command,
            cwd=repo_root,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            stdin_text=prompt,
        )
    else:
        command = _build_claude_command(
            schema_path=TURN_SCHEMA_PATH,
            model=claude_model,
            effort=claude_effort,
            prompt=prompt,
        )
        exit_code = _run_subprocess(
            command=command,
            cwd=repo_root,
            stdout_path=result_path,
            stderr_path=stderr_path,
        )
        if result_path.exists() and not stdout_path.exists():
            shutil.copyfile(result_path, stdout_path)

    finished_at = _utc_now()
    after_sha, after_text = _snapshot_train_py(repo_root)
    changed = _write_diff(
        diff_path,
        before_text,
        after_text,
        fromfile=f"{TRAIN_PY_RELATIVE}.before",
        tofile=f"{TRAIN_PY_RELATIVE}.after",
    )
    _write_text(experiment_dir / "train.py.after", after_text)

    agent_result_path = result_path if selection.backend == "codex" else stdout_path
    agent_result: dict[str, Any] = {}
    result_error: str | None = None
    try:
        agent_result = _load_agent_result(agent_result_path)
    except Exception as exc:
        result_error = f"invalid agent JSON: {exc}"

    train_log_path: Path | None = None
    raw_train_log = agent_result.get("train_log")
    if isinstance(raw_train_log, str) and raw_train_log.strip():
        candidate = Path(raw_train_log.strip())
        train_log_path = candidate if candidate.is_absolute() else (repo_root / candidate)
    else:
        default_path = run_dir / "train_crypto.log"
        if default_path.exists():
            train_log_path = default_path

    parsed_metrics = ParsedTrainMetrics()
    if train_log_path is not None:
        parsed_metrics = parse_train_log(train_log_path)

    _copy_if_exists(stdout_path, experiment_dir / "backend_stdout.txt")
    _copy_if_exists(stderr_path, experiment_dir / "backend_stderr.txt")
    _copy_if_exists(result_path, experiment_dir / "agent_result.json")
    _copy_if_exists(diff_path if changed else None, experiment_dir / "train_py.diff")
    _copy_if_exists(train_log_path, experiment_dir / "train_crypto.log")

    notes = []
    if isinstance(agent_result.get("notes"), list):
        notes = [str(item) for item in agent_result.get("notes", [])]
    if result_error:
        notes.insert(0, result_error)
    if exit_code != 0 and not notes:
        stderr_tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-1000:]
        notes = [stderr_tail.strip() or f"backend exited with code {exit_code}"]

    record = TurnRecord(
        turn_index=selection.turn_index,
        started_at_utc=started_at.isoformat(),
        finished_at_utc=finished_at.isoformat(),
        elapsed_seconds=float((finished_at - started_at).total_seconds()),
        backend=selection.backend,
        frequency=selection.frequency,
        status=str(agent_result.get("status") or ("failed" if exit_code != 0 or result_error else "unknown")),
        summary=str(agent_result.get("summary") or (result_error or "")),
        experiment_dir=_relative_or_absolute(experiment_dir, repo_root),
        prompt_path=_relative_or_absolute(prompt_path, repo_root),
        result_path=_relative_or_absolute(result_path, repo_root),
        stdout_path=_relative_or_absolute(stdout_path, repo_root),
        stderr_path=_relative_or_absolute(stderr_path, repo_root),
        train_log_path=None if train_log_path is None else _relative_or_absolute(train_log_path, repo_root),
        diff_path=_relative_or_absolute(diff_path, repo_root) if changed else None,
        touched_files=[str(item) for item in agent_result.get("touched_files", [])] if isinstance(agent_result.get("touched_files"), list) else [],
        train_py_changed=bool(changed),
        train_py_before_sha256=before_sha,
        train_py_after_sha256=after_sha,
        agent_exit_code=int(exit_code),
        robust_score=parsed_metrics.robust_score if parsed_metrics.robust_score is not None else agent_result.get("robust_score"),
        val_loss=parsed_metrics.val_loss if parsed_metrics.val_loss is not None else agent_result.get("val_loss"),
        training_seconds=parsed_metrics.training_seconds if parsed_metrics.training_seconds is not None else agent_result.get("training_seconds"),
        total_seconds=parsed_metrics.total_seconds if parsed_metrics.total_seconds is not None else agent_result.get("total_seconds"),
        peak_vram_mb=parsed_metrics.peak_vram_mb if parsed_metrics.peak_vram_mb is not None else agent_result.get("peak_vram_mb"),
        scenario_count=parsed_metrics.scenario_count,
        total_trade_count=parsed_metrics.total_trade_count,
        num_steps=parsed_metrics.num_steps if parsed_metrics.num_steps is not None else agent_result.get("num_steps"),
        notes=notes,
    )
    _write_json(
        experiment_dir / "metadata.json",
        {
            "record": asdict(record),
            "prompt_packs": [path for path, _ in prompt_sections],
        },
    )
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Schedule optimization turns for autoresearch_crypto.")
    parser.add_argument("--analysis-dir", default=str(DEFAULT_ANALYSIS_DIR))
    parser.add_argument("--experiment-bundle-root", default=str(DEFAULT_EXPERIMENT_BUNDLE_ROOT))
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--backends", default="codex")
    parser.add_argument("--frequencies", default="crypto")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--max-turns", type=int, default=1)
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--allow-claude", action="store_true")
    parser.add_argument("--codex-model", default=None)
    parser.add_argument("--codex-reasoning-effort", default="xhigh")
    parser.add_argument("--claude-model", default=None)
    parser.add_argument("--claude-effort", default="high")
    parser.add_argument("--prompt-file", action="append", default=[])
    parser.add_argument("--extra-prompt", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    analysis_dir = resolve_repo_path(args.analysis_dir, repo_root=repo_root)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    experiment_bundle_root = resolve_repo_path(args.experiment_bundle_root, repo_root=repo_root)
    experiment_bundle_root.mkdir(parents=True, exist_ok=True)
    python_path = resolve_python_path(args.python, repo_root=repo_root)
    if not python_path.exists():
        raise SystemExit(f"Python interpreter not found: {python_path}")
    prompt_sections = load_prompt_files(list(args.prompt_file), repo_root=repo_root)

    requested_backends = _parse_csv_list(args.backends, lower=True) or list(DEFAULT_BACKENDS)
    requested_frequencies = _parse_csv_list(args.frequencies, lower=True) or list(DEFAULT_FREQUENCIES)

    probe_dir = analysis_dir / "probe"
    backend_statuses = collect_backend_statuses(
        requested_backends,
        skip_probe=bool(args.skip_probe),
        allow_claude=bool(args.allow_claude),
        repo_root=repo_root,
        schema_path=TURN_SCHEMA_PATH,
        probe_dir=probe_dir,
        codex_model=args.codex_model,
        codex_reasoning_effort=args.codex_reasoning_effort,
        claude_model=args.claude_model,
        claude_effort=args.claude_effort,
    )
    _write_json(analysis_dir / "backend_status.json", [asdict(item) for item in backend_statuses])
    for status in backend_statuses:
        print(f"backend={status.name} available={status.available} reason={status.reason}", flush=True)

    available_backends = [item.name for item in backend_statuses if item.available]
    if not available_backends:
        raise SystemExit("No usable agent backends available.")

    state_path = analysis_dir / STATE_FILENAME
    state = _read_json(state_path, default={"next_turn_index": 1})

    turns_to_run = int(args.max_turns)
    turns_completed = 0
    while True:
        turn_index = int(state.get("next_turn_index", 1))
        selection = select_turn(
            turn_index=turn_index,
            backends=available_backends,
            frequencies=requested_frequencies,
        )
        turn_started = time.time()
        record = run_turn(
            repo_root=repo_root,
            analysis_dir=analysis_dir,
            experiment_bundle_root=experiment_bundle_root,
            python_path=python_path,
            selection=selection,
            codex_model=args.codex_model,
            codex_reasoning_effort=args.codex_reasoning_effort,
            claude_model=args.claude_model,
            claude_effort=args.claude_effort,
            prompt_sections=prompt_sections,
            extra_prompt=args.extra_prompt,
        )

        history_path = analysis_dir / HISTORY_FILENAME
        leaderboard_path = analysis_dir / LEADERBOARD_FILENAME
        _append_history(history_path, asdict(record))
        _append_leaderboard_row(leaderboard_path, record)

        state["next_turn_index"] = int(turn_index + 1)
        state["last_turn"] = {
            "turn_index": record.turn_index,
            "backend": record.backend,
            "frequency": record.frequency,
            "status": record.status,
            "robust_score": record.robust_score,
            "val_loss": record.val_loss,
            "summary": record.summary,
        }
        _write_json(state_path, state)
        robust_text = "n/a" if record.robust_score is None else f"{record.robust_score:.6f}"
        val_text = "n/a" if record.val_loss is None else f"{record.val_loss:.6f}"
        print(
            f"turn={record.turn_index} backend={record.backend} "
            f"status={record.status} robust_score={robust_text} val_loss={val_text} "
            f"summary={record.summary}",
            flush=True,
        )

        turns_completed += 1
        if not args.loop or turns_completed >= turns_to_run:
            break

        next_due = turn_started + float(args.interval_seconds)
        sleep_seconds = max(0.0, next_due - time.time())
        if sleep_seconds > 0.0:
            time.sleep(sleep_seconds)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
