from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.binance_hybrid_process_audit import (
    BinanceHybridProcessMatchResult,
    audit_running_hybrid_process_matches_launch,
)
from src.binance_live_process_audit import (
    BinanceLiveProcessAuditResult,
    audit_running_binance_live_processes,
)


@dataclass(frozen=True)
class BinanceHybridMachineAuditResult:
    launch_script: str
    process_audit: BinanceLiveProcessAuditResult
    hybrid_process_match: BinanceHybridProcessMatchResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "launch_script": self.launch_script,
            "process_audit": self.process_audit.to_dict(),
            "hybrid_process_match": self.hybrid_process_match.to_dict(),
        }


def audit_binance_hybrid_machine_state(
    launch_script: str | Path,
) -> BinanceHybridMachineAuditResult:
    launch_path = str(Path(launch_script).resolve())
    process_audit = audit_running_binance_live_processes()
    hybrid_process_match = audit_running_hybrid_process_matches_launch(launch_path)
    return BinanceHybridMachineAuditResult(
        launch_script=launch_path,
        process_audit=process_audit,
        hybrid_process_match=hybrid_process_match,
    )


def build_machine_audit_launch_mismatch_issues(
    result: BinanceHybridMachineAuditResult,
) -> list[str]:
    hybrid_process_match = result.hybrid_process_match
    if hybrid_process_match.ok:
        return []
    if hybrid_process_match.mismatched_fields:
        return [hybrid_process_match.reason]
    if str(hybrid_process_match.reason).startswith("unable to parse running hybrid process command"):
        return [hybrid_process_match.reason]
    return []


def build_machine_audit_health_issues(
    result: BinanceHybridMachineAuditResult,
) -> list[str]:
    issues: list[str] = []
    if not result.process_audit.ok:
        issues.append(result.process_audit.reason)
    hybrid_process_match = result.hybrid_process_match
    if hybrid_process_match.ok:
        return issues
    if hybrid_process_match.reason == "no running hybrid process found":
        issues.append(hybrid_process_match.reason)
    return issues


def build_machine_deploy_preflight_reason(
    result: BinanceHybridMachineAuditResult,
) -> str | None:
    if not result.process_audit.ok:
        return f"binance live process isolation failed: {result.process_audit.reason}"
    if not result.hybrid_process_match.ok:
        return (
            "binance live hybrid process drift detected: "
            f"{result.hybrid_process_match.reason}"
        )
    return None
