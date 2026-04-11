from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.binance_hybrid_launch import (
    BinanceHybridLaunchConfig,
    parse_launch_script,
    parse_trade_binance_live_command,
)
from src.binance_live_process_audit import BinanceLiveProcess, list_running_binance_live_processes


@dataclass(frozen=True)
class BinanceHybridProcessMatchResult:
    ok: bool
    reason: str
    launch_script: str
    pid: int | None = None
    mismatched_fields: tuple[str, ...] = ()
    running_checkpoint: str | None = None
    expected_checkpoint: str | None = None
    running_config: BinanceHybridLaunchConfig | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_symbol_list(symbols: list[str]) -> tuple[str, ...]:
    return tuple(str(symbol).strip().upper() for symbol in symbols if str(symbol).strip())


def _compare_configs(
    launch_cfg: BinanceHybridLaunchConfig,
    running_cfg: BinanceHybridLaunchConfig,
) -> tuple[str, ...]:
    mismatches: list[str] = []
    if str(launch_cfg.model) != str(running_cfg.model):
        mismatches.append("model")
    if _normalize_symbol_list(launch_cfg.symbols) != _normalize_symbol_list(running_cfg.symbols):
        mismatches.append("symbols")
    if str(launch_cfg.execution_mode) != str(running_cfg.execution_mode):
        mismatches.append("execution_mode")
    if not math.isclose(float(launch_cfg.leverage), float(running_cfg.leverage), rel_tol=0.0, abs_tol=1e-9):
        mismatches.append("leverage")
    if launch_cfg.interval != running_cfg.interval:
        mismatches.append("interval")
    if str(launch_cfg.fallback_mode) != str(running_cfg.fallback_mode):
        mismatches.append("fallback_mode")
    if str(launch_cfg.rl_checkpoint or "") != str(running_cfg.rl_checkpoint or ""):
        mismatches.append("rl_checkpoint")
    return tuple(mismatches)


def _parse_running_hybrid_process(process: BinanceLiveProcess) -> BinanceHybridLaunchConfig:
    return parse_trade_binance_live_command(
        process.command,
        source=f"pid={process.pid}",
        require_rl_checkpoint=False,
    )


def audit_running_hybrid_process_matches_launch(
    launch_script: str | Path,
    *,
    ps_text: str | None = None,
) -> BinanceHybridProcessMatchResult:
    launch_cfg = parse_launch_script(launch_script, require_rl_checkpoint=False)
    launch_script_resolved = str(Path(launch_script).resolve())
    hybrid_processes = [process for process in list_running_binance_live_processes(ps_text) if process.kind == "hybrid"]
    if not hybrid_processes:
        return BinanceHybridProcessMatchResult(
            ok=False,
            reason="no running hybrid process found",
            launch_script=launch_script_resolved,
            expected_checkpoint=launch_cfg.rl_checkpoint,
        )
    if len(hybrid_processes) > 1:
        return BinanceHybridProcessMatchResult(
            ok=False,
            reason="multiple running hybrid processes found: " + ",".join(str(process.pid) for process in hybrid_processes),
            launch_script=launch_script_resolved,
            pid=hybrid_processes[0].pid,
            expected_checkpoint=launch_cfg.rl_checkpoint,
        )

    process = hybrid_processes[0]
    try:
        running_cfg = _parse_running_hybrid_process(process)
    except Exception as exc:
        return BinanceHybridProcessMatchResult(
            ok=False,
            reason=f"unable to parse running hybrid process command: {exc}",
            launch_script=launch_script_resolved,
            pid=process.pid,
            expected_checkpoint=launch_cfg.rl_checkpoint,
        )

    mismatched_fields = _compare_configs(launch_cfg, running_cfg)
    if mismatched_fields:
        return BinanceHybridProcessMatchResult(
            ok=False,
            reason=(
                "running hybrid process does not match launch config: "
                + ", ".join(mismatched_fields)
            ),
            launch_script=launch_script_resolved,
            pid=process.pid,
            mismatched_fields=mismatched_fields,
            running_checkpoint=running_cfg.rl_checkpoint,
            expected_checkpoint=launch_cfg.rl_checkpoint,
            running_config=running_cfg,
        )

    return BinanceHybridProcessMatchResult(
        ok=True,
        reason="running hybrid process matches launch config",
        launch_script=launch_script_resolved,
        pid=process.pid,
        running_checkpoint=running_cfg.rl_checkpoint,
        expected_checkpoint=launch_cfg.rl_checkpoint,
        running_config=running_cfg,
    )
