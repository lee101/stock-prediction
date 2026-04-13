from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from src.binan.hybrid_cycle_trace import DEFAULT_TRACE_DIR, load_cycle_snapshots
from src.binance_hybrid_launch import DEFAULT_LAUNCH_SCRIPT, parse_launch_script
from src.binance_hybrid_machine_audit import (
    audit_binance_hybrid_machine_state,
    build_machine_audit_health_issues,
    build_machine_audit_launch_mismatch_issues,
)
from src.binance_hybrid_snapshot_activity import (
    find_missing_exit_order_coverage,
    find_unexpected_snapshot_activity,
)


_UNHEALTHY_SNAPSHOT_STATUSES = frozenset(
    {
        "account_guard_blocked",
        "failed",
        "context_error",
        "gemini_error_rl_flat",
        "invalid_allocation_plan",
        "invalid_allocation_plan_rl_flat",
        "rl_signal_error",
        "no_contexts",
        "no_active_tradable_symbols",
    }
)

_UNHEALTHY_HYBRID_FALLBACK_SOURCES = frozenset(
    {
        "rl_only_fallback",
        "rl_flat_no_action",
        "rl_flat_cash_flatten",
    }
)


def _normalize_checkpoint_path(path: str | Path | None) -> str | None:
    if path is None or str(path).strip() == "":
        return None
    return str(Path(path).expanduser().resolve())


def _normalize_symbols(symbols: list[str] | tuple[str, ...] | None) -> list[str]:
    return [str(symbol).strip().upper() for symbol in list(symbols or []) if str(symbol).strip()]


def _parse_expected_symbols(value: str | None) -> list[str]:
    return _normalize_symbols(str(value or "").replace(",", " ").split())


def _parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = str(value).strip().replace("Z", "+00:00")
    ts = datetime.fromisoformat(normalized)
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


def _tail_lines(path: Path, max_lines: int, *, start_offset: int = 0) -> str:
    if not path.exists():
        return ""
    max_lines = max(1, int(max_lines))
    start_offset = max(0, int(start_offset))
    file_size = path.stat().st_size
    read_offset = start_offset if 0 < start_offset <= file_size else 0
    with open(path, "rb") as handle:
        if read_offset:
            handle.seek(read_offset)
        payload = handle.read()
    lines = payload.decode("utf-8", errors="replace").splitlines(keepends=True)
    return "".join(lines[-max_lines:])


@dataclass(frozen=True)
class DeployVerificationResult:
    ok: bool
    reason: str
    launch_script: str
    expected_checkpoint: str | None
    launch_checkpoint: str | None
    expected_leverage: float | None
    launch_leverage: float
    expected_symbols: list[str]
    launch_symbols: list[str]
    log_path: str
    log_confirmed: bool
    latest_live_cycle_started_at: str | None = None
    latest_live_cycle_checkpoint: str | None = None
    latest_live_cycle_requested_leverage: float | None = None
    latest_live_cycle_status: str | None = None
    latest_live_cycle_allocation_source: str | None = None
    latest_live_cycle_missing_exit_price_symbols: tuple[str, ...] = ()
    latest_live_cycle_missing_exit_order_symbols: tuple[str, ...] = ()
    healthy_live_cycle_count: int = 0
    required_healthy_live_cycles: int = 0
    snapshot_checked: bool = False
    snapshot_confirmed: bool | None = None
    snapshot_reason: str | None = None
    machine_checked: bool = False
    machine_confirmed: bool | None = None
    machine_reason: str | None = None
    machine_launch_mismatch_issues: tuple[str, ...] = ()
    machine_health_issues: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _extract_live_snapshot_fields(snapshot: dict[str, object]) -> dict[str, object]:
    latest_live_cycle_checkpoint_raw = str(snapshot.get("rl_checkpoint") or "").strip()
    latest_live_cycle_checkpoint = (
        _normalize_checkpoint_path(latest_live_cycle_checkpoint_raw)
        if latest_live_cycle_checkpoint_raw
        else None
    )
    requested_leverage = snapshot.get("requested_leverage")
    try:
        latest_live_cycle_requested_leverage = (
            float(requested_leverage) if requested_leverage is not None else None
        )
    except (TypeError, ValueError):
        latest_live_cycle_requested_leverage = None

    requested_symbols = _normalize_symbols(snapshot.get("requested_tradable_symbols"))
    active_symbols = _normalize_symbols(snapshot.get("active_tradable_symbols"))
    snapshot_symbols = requested_symbols or active_symbols
    missing_exit_price_symbols, missing_exit_order_symbols = find_missing_exit_order_coverage(
        snapshot,
        snapshot_symbols,
    )

    return {
        "started_at": str(snapshot.get("cycle_started_at") or ""),
        "checkpoint": latest_live_cycle_checkpoint,
        "requested_leverage": latest_live_cycle_requested_leverage,
        "status": str(snapshot.get("status") or "").strip() or None,
        "allocation_source": str(snapshot.get("allocation_source") or "").strip() or None,
        "snapshot_symbols": snapshot_symbols,
        "missing_exit_price_symbols": tuple(missing_exit_price_symbols),
        "missing_exit_order_symbols": tuple(missing_exit_order_symbols),
        "snapshot": snapshot,
    }


def evaluate_live_snapshot_health(
    snapshot: dict[str, object],
    *,
    expected_checkpoint_norm: str | None,
    expect_no_rl_checkpoint: bool,
    expected_symbols_list: list[str],
    launch_symbols: list[str],
    expected_leverage: float | None,
) -> tuple[bool | None, str | None, dict[str, object]]:
    meta = _extract_live_snapshot_fields(snapshot)
    latest_live_cycle_checkpoint = meta["checkpoint"]
    latest_live_cycle_requested_leverage = meta["requested_leverage"]
    latest_live_cycle_status = meta["status"]
    latest_live_cycle_allocation_source = meta["allocation_source"]
    snapshot_symbols = meta["snapshot_symbols"]
    allowed_symbols = expected_symbols_list or launch_symbols
    unexpected_symbols, unexpected_order_symbols = find_unexpected_snapshot_activity(
        meta["snapshot"],  # type: ignore[arg-type]
        allowed_symbols,
    )
    missing_exit_price_symbols = meta["missing_exit_price_symbols"]
    missing_exit_order_symbols = meta["missing_exit_order_symbols"]

    snapshot_confirmed: bool | None = True
    snapshot_reason: str | None = None
    if expect_no_rl_checkpoint and latest_live_cycle_checkpoint is not None:
        snapshot_confirmed = False
        snapshot_reason = "latest live snapshot still records rl_checkpoint"
    elif latest_live_cycle_checkpoint and latest_live_cycle_checkpoint != expected_checkpoint_norm:
        snapshot_confirmed = False
        snapshot_reason = "latest live snapshot checkpoint does not match expected checkpoint"
    elif expected_symbols_list and not snapshot_symbols:
        snapshot_confirmed = False
        snapshot_reason = "latest live snapshot does not record tradable symbols"
    elif expected_symbols_list and snapshot_symbols != expected_symbols_list:
        snapshot_confirmed = False
        snapshot_reason = "latest live snapshot symbols do not match expected symbols"
    elif unexpected_symbols:
        snapshot_confirmed = False
        snapshot_reason = (
            "latest live snapshot shows unexpected symbol activity: "
            + ", ".join(unexpected_symbols)
        )
    elif unexpected_order_symbols:
        snapshot_confirmed = False
        snapshot_reason = (
            "latest live snapshot shows unexpected order symbols: "
            + ", ".join(unexpected_order_symbols)
        )
    elif missing_exit_price_symbols:
        snapshot_confirmed = False
        snapshot_reason = (
            "latest live snapshot is missing exit prices for open positions: "
            + ", ".join(missing_exit_price_symbols)
        )
    elif missing_exit_order_symbols:
        snapshot_confirmed = False
        snapshot_reason = (
            "latest live snapshot is missing closing sell orders for open positions: "
            + ", ".join(missing_exit_order_symbols)
        )
    elif expected_leverage is not None and latest_live_cycle_requested_leverage is None:
        snapshot_confirmed = False
        snapshot_reason = "latest live snapshot does not record requested leverage"
    elif (
        expected_leverage is not None
        and not math.isclose(latest_live_cycle_requested_leverage, float(expected_leverage), rel_tol=0.0, abs_tol=1e-9)
    ):
        snapshot_confirmed = False
        snapshot_reason = "latest live snapshot leverage does not match expected leverage"
    elif latest_live_cycle_status in _UNHEALTHY_SNAPSHOT_STATUSES:
        snapshot_confirmed = False
        detail = str((meta["snapshot"] or {}).get("error") or "").strip()
        snapshot_reason = (
            f"latest live snapshot status is unhealthy: {latest_live_cycle_status}"
            + (f" ({detail})" if detail else "")
        )
    elif (
        not expect_no_rl_checkpoint
        and latest_live_cycle_allocation_source in _UNHEALTHY_HYBRID_FALLBACK_SOURCES
    ):
        snapshot_confirmed = False
        detail = ""
        allocation_plan = (meta["snapshot"] or {}).get("allocation_plan")
        if isinstance(allocation_plan, dict):
            detail = str(allocation_plan.get("reasoning") or "").strip()
        snapshot_reason = (
            "latest live snapshot shows Gemini fallback allocation source: "
            f"{latest_live_cycle_allocation_source}"
            + (f" ({detail})" if detail else "")
        )
    elif not expect_no_rl_checkpoint and latest_live_cycle_checkpoint is None:
        snapshot_confirmed = False
        snapshot_reason = "latest live snapshot does not record rl_checkpoint"
    else:
        snapshot_reason = (
            "latest live snapshot confirms Gemini-only mode and symbols"
            if expect_no_rl_checkpoint
            else "latest live snapshot matches expected checkpoint and symbols"
        )

    return snapshot_confirmed, snapshot_reason, meta


def verify_deployed_binance_hybrid(
    *,
    launch_script: str | Path = DEFAULT_LAUNCH_SCRIPT,
    expected_checkpoint: str | Path | None = None,
    expect_no_rl_checkpoint: bool = False,
    expected_symbols: str | None = None,
    expected_leverage: float | None = None,
    log_path: str | Path,
    trace_dir: str | Path = DEFAULT_TRACE_DIR,
    deployed_after: str | None = None,
    log_offset_bytes: int = 0,
    max_log_lines: int = 200,
    require_live_snapshot: bool = False,
    minimum_healthy_live_cycles: int = 0,
    wait_timeout_seconds: float | None = None,
    poll_interval_seconds: float = 5.0,
    require_machine_state_match: bool = False,
) -> DeployVerificationResult:
    if expect_no_rl_checkpoint and expected_checkpoint is not None:
        raise ValueError("expected_checkpoint must be omitted when expect_no_rl_checkpoint=True")
    if not expect_no_rl_checkpoint and expected_checkpoint is None:
        raise ValueError("expected_checkpoint is required unless expect_no_rl_checkpoint=True")
    if int(minimum_healthy_live_cycles) < 0:
        raise ValueError("minimum_healthy_live_cycles must be >= 0")
    if int(log_offset_bytes) < 0:
        raise ValueError("log_offset_bytes must be >= 0")

    launch_cfg = parse_launch_script(launch_script, require_rl_checkpoint=not expect_no_rl_checkpoint)
    expected_checkpoint_norm = _normalize_checkpoint_path(expected_checkpoint)
    launch_checkpoint = _normalize_checkpoint_path(launch_cfg.rl_checkpoint)
    launch_leverage = float(launch_cfg.leverage)
    expected_symbols_list = _parse_expected_symbols(expected_symbols)
    launch_symbols = _normalize_symbols(launch_cfg.symbols)
    log_path_resolved = str(Path(log_path).expanduser().resolve())

    launch_reason: str | None = None
    if expect_no_rl_checkpoint:
        if launch_checkpoint is not None:
            launch_reason = "launch still defines an rl_checkpoint"
    elif launch_checkpoint != expected_checkpoint_norm:
        launch_reason = "launch checkpoint does not match expected checkpoint"
    elif expected_symbols_list and launch_symbols != expected_symbols_list:
        launch_reason = "launch symbols do not match expected symbols"
    elif expected_leverage is not None and not math.isclose(launch_leverage, float(expected_leverage), rel_tol=0.0, abs_tol=1e-9):
        launch_reason = "launch leverage does not match expected leverage"

    if launch_reason is not None:
        return DeployVerificationResult(
            ok=False,
            reason=launch_reason,
            launch_script=str(Path(launch_script).resolve()),
            expected_checkpoint=expected_checkpoint_norm,
            launch_checkpoint=launch_checkpoint,
            expected_leverage=expected_leverage,
            launch_leverage=launch_leverage,
            expected_symbols=expected_symbols_list,
            launch_symbols=launch_symbols,
            log_path=log_path_resolved,
            log_confirmed=False,
        )

    log_text = _tail_lines(Path(log_path_resolved), max_lines=max_log_lines, start_offset=log_offset_bytes)
    expected_log_snippet = "Hybrid mode: RL=disabled" if expect_no_rl_checkpoint else f"Hybrid mode: RL={expected_checkpoint_norm}"
    log_confirmed = expected_log_snippet in log_text
    if not log_confirmed:
        return DeployVerificationResult(
            ok=False,
            reason="supervisor log does not show the expected checkpoint startup line",
            launch_script=str(Path(launch_script).resolve()),
            expected_checkpoint=expected_checkpoint_norm,
            launch_checkpoint=launch_checkpoint,
            expected_leverage=expected_leverage,
            launch_leverage=launch_leverage,
            expected_symbols=expected_symbols_list,
            launch_symbols=launch_symbols,
            log_path=log_path_resolved,
            log_confirmed=False,
        )

    deployed_after_ts = _parse_utc(deployed_after)
    snapshot_start = deployed_after_ts.isoformat() if deployed_after_ts is not None else None
    minimum_healthy_live_cycles = max(0, int(minimum_healthy_live_cycles))
    require_live_snapshot = require_live_snapshot or minimum_healthy_live_cycles > 0
    wait_timeout = float(wait_timeout_seconds or 0.0)
    snapshot_evaluations: list[tuple[bool | None, str | None, dict[str, object]]] = []
    if wait_timeout > 0.0:
        deadline = time.monotonic() + wait_timeout
        while True:
            live_snapshots = load_cycle_snapshots(
                log_dir=Path(trace_dir),
                start=snapshot_start,
                live_only=True,
            )
            snapshot_evaluations = [
                evaluate_live_snapshot_health(
                    snapshot,
                    expected_checkpoint_norm=expected_checkpoint_norm,
                    expect_no_rl_checkpoint=expect_no_rl_checkpoint,
                    expected_symbols_list=expected_symbols_list,
                    launch_symbols=launch_symbols,
                    expected_leverage=expected_leverage,
                )
                for snapshot in live_snapshots
            ]
            healthy_live_cycle_count = sum(1 for confirmed, _reason, _meta in snapshot_evaluations if confirmed is True)
            saw_failure = any(confirmed is False for confirmed, _reason, _meta in snapshot_evaluations)
            if minimum_healthy_live_cycles > 0:
                if saw_failure or healthy_live_cycle_count >= minimum_healthy_live_cycles or time.monotonic() >= deadline:
                    break
            elif live_snapshots or time.monotonic() >= deadline:
                break
            time.sleep(max(0.0, float(poll_interval_seconds)))
    else:
        live_snapshots = load_cycle_snapshots(
            log_dir=Path(trace_dir),
            start=snapshot_start,
            live_only=True,
        )
        snapshot_evaluations = [
            evaluate_live_snapshot_health(
                snapshot,
                expected_checkpoint_norm=expected_checkpoint_norm,
                expect_no_rl_checkpoint=expect_no_rl_checkpoint,
                expected_symbols_list=expected_symbols_list,
                launch_symbols=launch_symbols,
                expected_leverage=expected_leverage,
            )
            for snapshot in live_snapshots
        ]

    latest_live_cycle_started_at: str | None = None
    latest_live_cycle_checkpoint: str | None = None
    latest_live_cycle_requested_leverage: float | None = None
    latest_live_cycle_status: str | None = None
    latest_live_cycle_allocation_source: str | None = None
    latest_live_cycle_missing_exit_price_symbols: tuple[str, ...] = ()
    latest_live_cycle_missing_exit_order_symbols: tuple[str, ...] = ()
    healthy_live_cycle_count = 0
    snapshot_checked = False
    snapshot_confirmed: bool | None = None
    snapshot_reason: str | None = None

    if live_snapshots:
        latest_confirmed, latest_reason, latest_meta = snapshot_evaluations[-1]
        latest_live_cycle_started_at = str(latest_meta["started_at"] or "")
        latest_live_cycle_checkpoint = latest_meta["checkpoint"]  # type: ignore[assignment]
        latest_live_cycle_requested_leverage = latest_meta["requested_leverage"]  # type: ignore[assignment]
        latest_live_cycle_status = latest_meta["status"]  # type: ignore[assignment]
        latest_live_cycle_allocation_source = latest_meta["allocation_source"]  # type: ignore[assignment]
        latest_live_cycle_missing_exit_price_symbols = latest_meta["missing_exit_price_symbols"]  # type: ignore[assignment]
        latest_live_cycle_missing_exit_order_symbols = latest_meta["missing_exit_order_symbols"]  # type: ignore[assignment]
        snapshot_checked = True
        healthy_live_cycle_count = sum(1 for confirmed, _reason, _meta in snapshot_evaluations if confirmed is True)

        if minimum_healthy_live_cycles > 0:
            first_failure = next(
                (
                    (confirmed, reason, meta)
                    for confirmed, reason, meta in snapshot_evaluations
                    if confirmed is False
                ),
                None,
            )
            if first_failure is not None:
                _failed_confirmed, failed_reason, _failed_meta = first_failure
                snapshot_confirmed = False
                snapshot_reason = failed_reason
            elif healthy_live_cycle_count < minimum_healthy_live_cycles:
                snapshot_confirmed = False
                snapshot_reason = (
                    f"observed {healthy_live_cycle_count} healthy live cycles after deploy; "
                    f"need at least {minimum_healthy_live_cycles}"
                )
                if latest_reason and latest_confirmed is not True:
                    snapshot_reason = f"{snapshot_reason} (latest: {latest_reason})"
            else:
                snapshot_confirmed = True
                snapshot_reason = (
                    f"observed {healthy_live_cycle_count} healthy live cycles after deploy"
                )
        else:
            snapshot_confirmed = latest_confirmed
            snapshot_reason = latest_reason
    elif require_live_snapshot:
        return DeployVerificationResult(
            ok=False,
            reason="no live cycle snapshot observed after deploy",
            launch_script=str(Path(launch_script).resolve()),
            expected_checkpoint=expected_checkpoint_norm,
            launch_checkpoint=launch_checkpoint,
            expected_leverage=expected_leverage,
            launch_leverage=launch_leverage,
            expected_symbols=expected_symbols_list,
            launch_symbols=launch_symbols,
            log_path=log_path_resolved,
            log_confirmed=True,
            latest_live_cycle_requested_leverage=latest_live_cycle_requested_leverage,
            latest_live_cycle_status=latest_live_cycle_status,
            latest_live_cycle_allocation_source=latest_live_cycle_allocation_source,
            latest_live_cycle_missing_exit_price_symbols=latest_live_cycle_missing_exit_price_symbols,
            latest_live_cycle_missing_exit_order_symbols=latest_live_cycle_missing_exit_order_symbols,
            healthy_live_cycle_count=healthy_live_cycle_count,
            required_healthy_live_cycles=minimum_healthy_live_cycles,
            snapshot_checked=False,
            snapshot_confirmed=False,
            snapshot_reason="no live cycle snapshot observed after deploy",
        )

    if snapshot_checked and snapshot_confirmed is False:
        return DeployVerificationResult(
            ok=False,
            reason=snapshot_reason or "latest live snapshot does not match expected deploy config",
            launch_script=str(Path(launch_script).resolve()),
            expected_checkpoint=expected_checkpoint_norm,
            launch_checkpoint=launch_checkpoint,
            expected_leverage=expected_leverage,
            launch_leverage=launch_leverage,
            expected_symbols=expected_symbols_list,
            launch_symbols=launch_symbols,
            log_path=log_path_resolved,
            log_confirmed=True,
            latest_live_cycle_started_at=latest_live_cycle_started_at,
            latest_live_cycle_checkpoint=latest_live_cycle_checkpoint,
            latest_live_cycle_requested_leverage=latest_live_cycle_requested_leverage,
            latest_live_cycle_status=latest_live_cycle_status,
            latest_live_cycle_allocation_source=latest_live_cycle_allocation_source,
            latest_live_cycle_missing_exit_price_symbols=latest_live_cycle_missing_exit_price_symbols,
            latest_live_cycle_missing_exit_order_symbols=latest_live_cycle_missing_exit_order_symbols,
            healthy_live_cycle_count=healthy_live_cycle_count,
            required_healthy_live_cycles=minimum_healthy_live_cycles,
            snapshot_checked=True,
            snapshot_confirmed=False,
            snapshot_reason=snapshot_reason,
        )

    machine_checked = False
    machine_confirmed: bool | None = None
    machine_reason: str | None = None
    machine_launch_mismatch_issues: tuple[str, ...] = ()
    machine_health_issues: tuple[str, ...] = ()
    if require_machine_state_match:
        machine_checked = True
        machine_audit = audit_binance_hybrid_machine_state(launch_script)
        machine_launch_mismatch_issues = tuple(build_machine_audit_launch_mismatch_issues(machine_audit))
        machine_health_issues = tuple(build_machine_audit_health_issues(machine_audit))
        machine_issues = [*machine_launch_mismatch_issues, *machine_health_issues]
        machine_confirmed = not machine_issues
        if machine_confirmed:
            machine_reason = "running process matches launch config and Binance live writer set is isolated"
        else:
            machine_reason = "; ".join(machine_issues)
            return DeployVerificationResult(
                ok=False,
                reason=machine_reason,
                launch_script=str(Path(launch_script).resolve()),
                expected_checkpoint=expected_checkpoint_norm,
                launch_checkpoint=launch_checkpoint,
                expected_leverage=expected_leverage,
                launch_leverage=launch_leverage,
                expected_symbols=expected_symbols_list,
                launch_symbols=launch_symbols,
                log_path=log_path_resolved,
                log_confirmed=True,
                latest_live_cycle_started_at=latest_live_cycle_started_at,
                latest_live_cycle_checkpoint=latest_live_cycle_checkpoint,
                latest_live_cycle_requested_leverage=latest_live_cycle_requested_leverage,
                latest_live_cycle_status=latest_live_cycle_status,
                latest_live_cycle_allocation_source=latest_live_cycle_allocation_source,
                latest_live_cycle_missing_exit_price_symbols=latest_live_cycle_missing_exit_price_symbols,
                latest_live_cycle_missing_exit_order_symbols=latest_live_cycle_missing_exit_order_symbols,
                healthy_live_cycle_count=healthy_live_cycle_count,
                required_healthy_live_cycles=minimum_healthy_live_cycles,
                snapshot_checked=snapshot_checked,
                snapshot_confirmed=snapshot_confirmed,
                snapshot_reason=snapshot_reason,
                machine_checked=machine_checked,
                machine_confirmed=machine_confirmed,
                machine_reason=machine_reason,
                machine_launch_mismatch_issues=machine_launch_mismatch_issues,
                machine_health_issues=machine_health_issues,
            )

    return DeployVerificationResult(
        ok=True,
        reason="deploy verification passed",
        launch_script=str(Path(launch_script).resolve()),
        expected_checkpoint=expected_checkpoint_norm,
        launch_checkpoint=launch_checkpoint,
        expected_leverage=expected_leverage,
        launch_leverage=launch_leverage,
        expected_symbols=expected_symbols_list,
        launch_symbols=launch_symbols,
        log_path=log_path_resolved,
        log_confirmed=True,
        latest_live_cycle_started_at=latest_live_cycle_started_at,
        latest_live_cycle_checkpoint=latest_live_cycle_checkpoint,
        latest_live_cycle_requested_leverage=latest_live_cycle_requested_leverage,
        latest_live_cycle_status=latest_live_cycle_status,
        latest_live_cycle_allocation_source=latest_live_cycle_allocation_source,
        latest_live_cycle_missing_exit_price_symbols=latest_live_cycle_missing_exit_price_symbols,
        latest_live_cycle_missing_exit_order_symbols=latest_live_cycle_missing_exit_order_symbols,
        healthy_live_cycle_count=healthy_live_cycle_count,
        required_healthy_live_cycles=minimum_healthy_live_cycles,
        snapshot_checked=snapshot_checked,
        snapshot_confirmed=snapshot_confirmed,
        snapshot_reason=snapshot_reason,
        machine_checked=machine_checked,
        machine_confirmed=machine_confirmed,
        machine_reason=machine_reason,
        machine_launch_mismatch_issues=machine_launch_mismatch_issues,
        machine_health_issues=machine_health_issues,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify Binance hybrid deploy state against launch config, logs, and recent live snapshots.")
    parser.add_argument("--launch-script", default=str(DEFAULT_LAUNCH_SCRIPT))
    parser.add_argument("--expected-checkpoint", default="")
    parser.add_argument("--expect-no-rl-checkpoint", action="store_true")
    parser.add_argument("--expected-symbols", default="")
    parser.add_argument("--expected-leverage", type=float, default=None)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--trace-dir", default=str(DEFAULT_TRACE_DIR))
    parser.add_argument("--deployed-after", default="")
    parser.add_argument("--log-offset-bytes", type=int, default=0)
    parser.add_argument("--max-log-lines", type=int, default=200)
    parser.add_argument("--require-live-snapshot", action="store_true")
    parser.add_argument("--min-healthy-live-cycles", type=int, default=0)
    parser.add_argument("--wait-timeout-seconds", type=float, default=0.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=5.0)
    parser.add_argument("--require-machine-state-match", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.expect_no_rl_checkpoint and args.expected_checkpoint:
        raise SystemExit("--expected-checkpoint cannot be used with --expect-no-rl-checkpoint")
    if not args.expect_no_rl_checkpoint and not args.expected_checkpoint:
        raise SystemExit("--expected-checkpoint is required unless --expect-no-rl-checkpoint is set")
    result = verify_deployed_binance_hybrid(
        launch_script=args.launch_script,
        expected_checkpoint=args.expected_checkpoint or None,
        expect_no_rl_checkpoint=args.expect_no_rl_checkpoint,
        expected_symbols=args.expected_symbols,
        expected_leverage=args.expected_leverage,
        log_path=args.log_path,
        trace_dir=args.trace_dir,
        deployed_after=args.deployed_after or None,
        log_offset_bytes=args.log_offset_bytes,
        max_log_lines=args.max_log_lines,
        require_live_snapshot=args.require_live_snapshot,
        minimum_healthy_live_cycles=args.min_healthy_live_cycles,
        wait_timeout_seconds=args.wait_timeout_seconds,
        poll_interval_seconds=args.poll_interval_seconds,
        require_machine_state_match=args.require_machine_state_match,
    )

    print(f"Launch:   {result.launch_script}")
    print(f"Expected: {result.expected_checkpoint or 'RL disabled'}")
    print(f"Launch RL:{' ' if result.launch_checkpoint else ''} {result.launch_checkpoint or 'disabled'}")
    print(f"Leverage: {result.launch_leverage:.2f}x" + (f" (expected {result.expected_leverage:.2f}x)" if result.expected_leverage is not None else ""))
    print(f"Symbols:  {','.join(result.launch_symbols)}")
    print(f"Log:      {result.log_path}")
    print(f"Log OK:   {'yes' if result.log_confirmed else 'no'}")
    if result.latest_live_cycle_started_at:
        print(f"Snapshot: {result.latest_live_cycle_started_at}")
    if result.latest_live_cycle_checkpoint:
        print(f"Snap RL:  {result.latest_live_cycle_checkpoint}")
    if result.latest_live_cycle_status:
        print(f"Snap St:  {result.latest_live_cycle_status}")
    if result.latest_live_cycle_allocation_source:
        print(f"Snap Src: {result.latest_live_cycle_allocation_source}")
    if result.latest_live_cycle_missing_exit_price_symbols:
        print(f"Snap No Exit Px: {','.join(result.latest_live_cycle_missing_exit_price_symbols)}")
    if result.latest_live_cycle_missing_exit_order_symbols:
        print(f"Snap No Exit Ord:{','.join(result.latest_live_cycle_missing_exit_order_symbols)}")
    if result.required_healthy_live_cycles > 0:
        print(
            f"Healthy:  {result.healthy_live_cycle_count}/{result.required_healthy_live_cycles}"
        )
    if result.snapshot_reason:
        print(f"Snapshot: {result.snapshot_reason}")
    if result.machine_checked:
        print(f"Mach OK:  {'yes' if result.machine_confirmed else 'no'}")
        if result.machine_reason:
            print(f"Machine: {result.machine_reason}")
    print(f"Decision: {'ALLOW' if result.ok else 'BLOCK'} -- {result.reason}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
