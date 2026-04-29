#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


REPO = Path(__file__).resolve().parents[1]
EVAL_OUTPUT_CONTRACT_FAILURE_RC = 2
SAFE_LEADERBOARD_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._=-]*$")


def load_leaderboard(
    path: Path,
    *,
    required_fields: Sequence[str] = (),
) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        raw_fieldnames = reader.fieldnames
        if raw_fieldnames is None:
            raise ValueError(f"leaderboard {path} is missing a header row")
        fieldnames = [str(name).strip() for name in raw_fieldnames]
        if any(name == "" for name in fieldnames):
            raise ValueError(f"leaderboard {path} contains an empty header field")
        duplicates = sorted({name for name in fieldnames if fieldnames.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"leaderboard {path} contains duplicate columns: {','.join(duplicates)}"
            )
        missing = sorted(set(str(field) for field in required_fields) - set(fieldnames))
        if missing:
            raise ValueError(
                f"leaderboard {path} missing required columns: {','.join(missing)}"
            )
        reader.fieldnames = fieldnames
        rows = list(reader)
    for row_index, row in enumerate(rows, start=2):
        if None in row:
            raise ValueError(
                f"leaderboard {path} row {row_index} has more fields than the header"
            )
    return rows


def require_safe_leaderboard_id(value: str, *, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"leaderboard {field_name} must be non-empty")
    if (
        normalized in {".", ".."}
        or "/" in normalized
        or "\\" in normalized
        or not SAFE_LEADERBOARD_ID_RE.fullmatch(normalized)
    ):
        raise ValueError(f"leaderboard {field_name} is not a safe path component: {value!r}")
    return normalized


def require_leaderboard_field(row: dict[str, str], field_name: str) -> str:
    raw = row.get(field_name)
    if raw is None:
        description = str(row.get("description") or "<unknown>")
        raise ValueError(f"leaderboard row {description} missing required field {field_name}")
    return require_safe_leaderboard_id(str(raw), field_name=field_name)


def validate_candidate_ids(rows: Sequence[dict[str, str]]) -> list[tuple[str, str, dict[str, str]]]:
    return [
        (
            require_leaderboard_field(row, "description"),
            require_leaderboard_field(row, "gpu_id"),
            row,
        )
        for row in rows
    ]


def validate_unique_eval_dirs(
    candidate_ids: Sequence[tuple[str, str, dict[str, str]]],
) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for description, gpu_id, _row in candidate_ids:
        if description in seen:
            duplicates.append(f"{description} (gpu_id {gpu_id})")
        seen.add(description)
    if duplicates:
        raise ValueError(
            "selected leaderboard rows would reuse eval output directories: "
            + ", ".join(duplicates)
        )


def pick_top_rows(
    rows: Sequence[dict[str, str]],
    *,
    sort_by: str,
    top_k: int,
) -> list[dict[str, str]]:
    sort_by = require_safe_leaderboard_id(sort_by, field_name="sort_by")
    if int(top_k) <= 0:
        raise ValueError(f"top_k must be positive, got {int(top_k)}")
    if not rows:
        raise ValueError("leaderboard has no candidate rows")

    def sort_metric(row: dict[str, str]) -> float:
        raw = row.get(sort_by)
        description = str(row.get("description") or "<unknown>")
        if raw is None or str(raw).strip() == "":
            raise ValueError(f"leaderboard row {description} missing sort metric {sort_by}")
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(
                f"leaderboard row {description} has non-numeric {sort_by}: {raw}"
            ) from exc
        if not math.isfinite(value):
            raise ValueError(
                f"leaderboard row {description} has non-finite {sort_by}: {raw}"
            )
        return value

    ranked = sorted(rows, key=sort_metric, reverse=True)
    return ranked[: int(top_k)]


def resolve_checkpoint_path(checkpoint_root: Path, description: str, gpu_id: str) -> Path:
    description = require_safe_leaderboard_id(description, field_name="description")
    gpu_id = require_safe_leaderboard_id(gpu_id, field_name="gpu_id")
    trial_dir = checkpoint_root / f"gpu{gpu_id}" / description
    for name in ("best.pt", "val_best.pt", "final.pt"):
        candidate = trial_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No checkpoint found under {trial_dir}")


def build_eval_command(
    *,
    checkpoint_path: Path,
    val_data: Path,
    out_dir: Path,
    hourly_data_root: Path,
    daily_start_date: str,
) -> list[str]:
    return [
        sys.executable,
        "-u",
        "scripts/eval_100d.py",
        "--checkpoint",
        str(checkpoint_path),
        "--val-data",
        str(val_data),
        "--out-dir",
        str(out_dir),
        "--n-windows",
        "30",
        "--window-days",
        "100",
        "--min-window-days",
        "100",
        "--fail-fast-max-dd",
        "0.20",
        "--fail-fast-min-completed",
        "3",
        "--monthly-target",
        "0.27",
        "--max-dd-target",
        "0.25",
        "--max-negative-windows",
        "0",
        "--min-completed-windows",
        "30",
        "--decision-lag",
        "2",
        "--min-decision-lag",
        "2",
        "--slippage-bps",
        "0,5,10,20",
        "--required-slippage-bps",
        "0,5,10,20",
        "--min-max-slippage-bps",
        "20",
        "--fee-rate",
        "0.001",
        "--min-fee-rate",
        "0.001",
        "--short-borrow-apr",
        "0.0625",
        "--min-short-borrow-apr",
        "0.0625",
        "--max-leverage",
        "1.5",
        "--max-leverage-target",
        "2.0",
        "--execution-granularity",
        "hourly_intrabar",
        "--hourly-data-root",
        str(hourly_data_root),
        "--daily-start-date",
        str(daily_start_date),
        "--hourly-fill-buffer-bps",
        "5.0",
        "--min-hourly-fill-buffer-bps",
        "5.0",
        "--hourly-max-hold-hours",
        "6",
        "--max-hourly-hold-hours-target",
        "6",
    ]


def run_eval(cmd: Sequence[str], *, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(shlex.quote(str(part)) for part in cmd) + "\n\n")
        handle.flush()
        proc = subprocess.run(list(cmd), cwd=str(REPO), check=False, stdout=handle, stderr=subprocess.STDOUT, text=True)
    return int(proc.returncode)


def wrapper_exit_code_for_eval_returncode(returncode: int) -> int:
    if int(returncode) < 0:
        return EVAL_OUTPUT_CONTRACT_FAILURE_RC
    return int(returncode)


def promotion_gate_passed(gate: Any) -> bool | None:
    if not isinstance(gate, dict):
        return None
    if gate.get("passed") is True or gate.get("pass") is True:
        return True
    if gate.get("passed") is False or gate.get("pass") is False:
        return False
    return None


def load_eval_artifact_summary(
    out_dir: Path,
    *,
    checkpoint_stem: str | None = None,
) -> dict[str, Any]:
    if checkpoint_stem is not None:
        artifact = out_dir / f"{checkpoint_stem}_eval100d.json"
        if not artifact.exists():
            return {"eval_artifact": str(artifact), "eval_artifact_status": "missing"}
    else:
        artifacts = sorted(out_dir.glob("*_eval100d.json"))
        if not artifacts:
            return {"eval_artifact": None, "eval_artifact_status": "missing"}
        if len(artifacts) > 1:
            return {
                "eval_artifact": None,
                "eval_artifact_status": "ambiguous",
                "eval_artifact_candidates": [str(path) for path in artifacts],
            }
        artifact = artifacts[0]
    try:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "eval_artifact": str(artifact),
            "eval_artifact_status": "invalid_json",
            "eval_artifact_error": str(exc),
        }
    gate = payload.get("promotion_gate")
    aggregate = payload.get("aggregate")
    raw = payload.get("raw")
    summary: dict[str, Any] = {
        "eval_artifact": str(artifact),
        "eval_artifact_status": "ok",
        "raw_status": raw.get("status") if isinstance(raw, dict) else None,
        "worst_slip_monthly": (
            aggregate.get("worst_slip_monthly") if isinstance(aggregate, dict) else None
        ),
        "promotion_gate_passed": promotion_gate_passed(gate),
        "promotion_failures": gate.get("failures") if isinstance(gate, dict) else None,
    }
    return summary


def _format_monthly_return(value: Any) -> str:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(parsed):
        return "-"
    return f"{100.0 * parsed:+.2f}%"


def _markdown_cell(value: Any) -> str:
    text = "-" if value is None else str(value)
    return text.replace("\n", " ").replace("|", "\\|")


def _summary_failure_reason(row: dict[str, Any]) -> str:
    wrapper_failure = row.get("wrapper_failure")
    if wrapper_failure:
        return str(wrapper_failure)
    failures = row.get("promotion_failures")
    if isinstance(failures, list) and failures:
        return "; ".join(str(item) for item in failures)
    artifact_status = row.get("eval_artifact_status")
    if artifact_status not in {None, "ok"}:
        return f"eval_artifact_{artifact_status}"
    return "-"


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


def render_summary_markdown(summary: Sequence[dict[str, Any]]) -> str:
    headers = [
        "description",
        "gpu_id",
        "status",
        "returncode",
        "promotion_gate",
        "worst_slip_monthly",
        "failure",
        "eval_dir",
    ]
    lines = [
        "# GPU Pool Candidate Eval Summary",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in summary:
        gate = row.get("promotion_gate_passed")
        gate_label = "pass" if gate is True else "fail" if gate is False else "-"
        values = [
            row.get("description"),
            row.get("gpu_id"),
            row.get("status"),
            row.get("returncode"),
            gate_label,
            _format_monthly_return(row.get("worst_slip_monthly")),
            _summary_failure_reason(row),
            row.get("eval_dir"),
        ]
        lines.append("| " + " | ".join(_markdown_cell(value) for value in values) + " |")
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run hourly eval_100d on top gpu_pool leaderboard candidates.")
    parser.add_argument("--leaderboard", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--val-data", type=Path, required=True)
    parser.add_argument("--hourly-data-root", type=Path, required=True)
    parser.add_argument("--daily-start-date", required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--sort-by", default="val_return")
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        sort_by = require_safe_leaderboard_id(str(args.sort_by), field_name="sort_by")
        rows = load_leaderboard(
            Path(args.leaderboard),
            required_fields=("description", "gpu_id", sort_by),
        )
        winners = pick_top_rows(rows, sort_by=sort_by, top_k=int(args.top_k))
        candidate_ids = validate_candidate_ids(winners)
        validate_unique_eval_dirs(candidate_ids)
    except (OSError, ValueError) as exc:
        print(f"eval_gpu_pool_top_candidates: {exc}", file=sys.stderr)
        return EVAL_OUTPUT_CONTRACT_FAILURE_RC
    summary: list[dict[str, Any]] = []
    first_failure_rc = 0
    for description, gpu_id, row in candidate_ids:
        out_dir = Path(args.out_root) / description
        try:
            checkpoint_path = resolve_checkpoint_path(Path(args.checkpoint_root), description, gpu_id)
        except FileNotFoundError as exc:
            summary.append(
                {
                    "description": description,
                    "gpu_id": gpu_id,
                    "checkpoint": None,
                    "returncode": None,
                    "status": "failed",
                    "wrapper_failure": "checkpoint_missing",
                    "wrapper_error": str(exc),
                    "eval_dir": str(out_dir),
                    "eval_artifact": None,
                    "eval_artifact_status": "not_run",
                }
            )
            if first_failure_rc == 0:
                first_failure_rc = EVAL_OUTPUT_CONTRACT_FAILURE_RC
            continue
        eval_cmd = build_eval_command(
            checkpoint_path=checkpoint_path,
            val_data=Path(args.val_data),
            out_dir=out_dir,
            hourly_data_root=Path(args.hourly_data_root),
            daily_start_date=str(args.daily_start_date),
        )
        try:
            rc = run_eval(eval_cmd, log_path=out_dir / "eval100d.log")
        except Exception as exc:
            summary.append(
                {
                    "description": description,
                    "gpu_id": gpu_id,
                    "checkpoint": str(checkpoint_path),
                    "returncode": None,
                    "status": "failed",
                    "wrapper_failure": "eval_launch_failed",
                    "wrapper_error": str(exc),
                    "eval_dir": str(out_dir),
                    "eval_artifact": None,
                    "eval_artifact_status": "not_run",
                }
            )
            if first_failure_rc == 0:
                first_failure_rc = EVAL_OUTPUT_CONTRACT_FAILURE_RC
            continue
        row_summary = {
            "description": description,
            "gpu_id": gpu_id,
            "checkpoint": str(checkpoint_path),
            "returncode": int(rc),
            "status": "ok" if int(rc) == 0 else "failed",
            "eval_dir": str(out_dir),
        }
        artifact_summary = load_eval_artifact_summary(
            out_dir,
            checkpoint_stem=checkpoint_path.stem,
        )
        row_summary.update(artifact_summary)
        output_contract_failure = None
        if artifact_summary.get("eval_artifact_status") != "ok":
            output_contract_failure = f"eval_artifact_{artifact_summary.get('eval_artifact_status')}"
        elif int(rc) == 0 and artifact_summary.get("raw_status") != "ok":
            output_contract_failure = (
                f"eval_success_with_raw_status_{artifact_summary.get('raw_status')}"
            )
        elif int(rc) == 0 and artifact_summary.get("promotion_gate_passed") is not True:
            output_contract_failure = "eval_success_without_passing_promotion_gate"
        if output_contract_failure is not None:
            row_summary["status"] = "failed"
            row_summary["wrapper_failure"] = output_contract_failure
        summary.append(row_summary)
        if int(rc) != 0 and first_failure_rc == 0:
            first_failure_rc = wrapper_exit_code_for_eval_returncode(int(rc))
        elif output_contract_failure is not None and first_failure_rc == 0:
            first_failure_rc = EVAL_OUTPUT_CONTRACT_FAILURE_RC
    summary_path = Path(args.out_root) / "summary.json"
    write_text_atomic(summary_path, f"{json.dumps(summary, indent=2, sort_keys=True)}\n")
    write_text_atomic(Path(args.out_root) / "summary.md", render_summary_markdown(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return first_failure_rc


if __name__ == "__main__":
    raise SystemExit(main())
