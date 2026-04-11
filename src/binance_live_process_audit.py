from __future__ import annotations

import shlex
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass


_PS_COMMAND = ("ps", "-eo", "pid=,args=")
_PROCESS_SIGNATURES: tuple[tuple[str, str, str], ...] = (
    ("hybrid", "script", "trade_binance_live.py"),
    ("meta_margin", "module", "binanceleveragesui.trade_margin_meta"),
    ("selector", "module", "binanceexp1.trade_binance_selector"),
    ("hourly", "module", "binanceexp1.trade_binance_hourly"),
    ("worksteal_daily", "module", "binance_worksteal.trade_live"),
)


@dataclass(frozen=True)
class BinanceLiveProcess:
    pid: int
    kind: str
    command: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BinanceLiveProcessAuditResult:
    ok: bool
    reason: str
    processes: tuple[BinanceLiveProcess, ...]
    counts_by_kind: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "processes": [process.to_dict() for process in self.processes],
            "counts_by_kind": dict(self.counts_by_kind),
        }


_CACHE_ONLY_KINDS = frozenset({"selector", "hourly"})


def _run_ps() -> str:
    completed = subprocess.run(
        _PS_COMMAND,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(f"failed to inspect running processes: returncode={completed.returncode} stderr={stderr}")
    return completed.stdout


def _classify_process(command: str) -> str | None:
    normalized = str(command or "").strip()
    if not normalized:
        return None
    try:
        tokens = shlex.split(normalized)
    except ValueError:
        return None
    if not tokens:
        return None
    executable = tokens[0].rsplit("/", 1)[-1].lower()
    if "python" not in executable:
        return None
    matched_kind: str | None = None
    for kind, signature_type, target in _PROCESS_SIGNATURES:
        if signature_type == "module":
            for index, token in enumerate(tokens[:-1]):
                if token == "-m" and tokens[index + 1] == target:
                    matched_kind = kind
                    break
        elif signature_type == "script" and any(token.endswith(target) for token in tokens[1:]):
            matched_kind = kind
        if matched_kind is not None:
            break
    return matched_kind


def _is_cache_only_process(process: BinanceLiveProcess) -> bool:
    if process.kind not in _CACHE_ONLY_KINDS:
        return False
    try:
        tokens = shlex.split(str(process.command or "").strip())
    except ValueError:
        return False
    return "--cache-only" in tokens


def list_running_binance_live_processes(ps_text: str | None = None) -> tuple[BinanceLiveProcess, ...]:
    raw = _run_ps() if ps_text is None else str(ps_text)
    processes: list[BinanceLiveProcess] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, separator, command = stripped.partition(" ")
        if not separator:
            continue
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        kind = _classify_process(command)
        if kind is None:
            continue
        processes.append(
            BinanceLiveProcess(
                pid=pid,
                kind=kind,
                command=command.strip(),
            )
        )
    return tuple(processes)


def audit_running_binance_live_processes(
    ps_text: str | None = None,
    *,
    allowed_counts_by_kind: dict[str, int] | None = None,
) -> BinanceLiveProcessAuditResult:
    processes = list_running_binance_live_processes(ps_text)
    counts = Counter(process.kind for process in processes)
    allowed_counts = {"hybrid": 1}
    if allowed_counts_by_kind:
        for kind, count in allowed_counts_by_kind.items():
            allowed_counts[str(kind)] = max(0, int(count))

    processes_by_kind: dict[str, list[BinanceLiveProcess]] = defaultdict(list)
    for process in processes:
        processes_by_kind[process.kind].append(process)

    conflict_parts: list[str] = []
    for kind, kind_processes in sorted(processes_by_kind.items()):
        relevant_processes = kind_processes
        if kind in _CACHE_ONLY_KINDS:
            relevant_processes = [process for process in kind_processes if not _is_cache_only_process(process)]
            if not relevant_processes:
                continue
        allowed = allowed_counts.get(kind, 0)
        if len(relevant_processes) <= allowed:
            continue
        pid_list = ",".join(str(process.pid) for process in relevant_processes)
        if allowed <= 0:
            conflict_parts.append(f"{kind}(pid={pid_list})")
        else:
            conflict_parts.append(f"{kind} count {len(relevant_processes)}>{allowed} (pid={pid_list})")

    if conflict_parts:
        return BinanceLiveProcessAuditResult(
            ok=False,
            reason="conflicting Binance live writers detected: " + "; ".join(conflict_parts),
            processes=processes,
            counts_by_kind=dict(counts),
        )

    if processes:
        return BinanceLiveProcessAuditResult(
            ok=True,
            reason="Binance live process set is isolated",
            processes=processes,
            counts_by_kind=dict(counts),
        )

    return BinanceLiveProcessAuditResult(
        ok=True,
        reason="no Binance live writer processes found",
        processes=processes,
        counts_by_kind=dict(counts),
    )
