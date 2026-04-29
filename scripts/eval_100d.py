"""End-to-end 100-day unseen-data eval harness.

Loads an fp4 / pufferlib_market checkpoint, runs it through the C marketsim
on rolling windows of N unseen days each, aggregates per-window
returns/sortino/max_dd, estimates annualised monthly return, and asserts
the configured promotion gate (default: >=27%/month, <=25% max DD, and
zero losing OOS windows).

Emits two sibling artifacts next to the checkpoint:

  - ``<ckpt_stem>_eval100d.json`` — raw per-slippage summaries + aggregate
  - ``<ckpt_stem>_eval100d.md``   — one-page markdown for the leaderboard

Usage::

    python scripts/eval_100d.py \
        --checkpoint pufferlib_market/checkpoints/stocks12_v5_rsi/tp05_s42/best.pt \
        --val-data pufferlib_market/data/stocks12_daily_v5_rsi_val.bin \
        --execution-granularity hourly_intrabar \
        --hourly-data-root data/hourly \
        --daily-start-date 2025-12-18 \
        --n-windows 30 --window-days 100 --monthly-target 0.27

The C marketsim is the ground truth (binary fills, realistic fees +
slippage). Production promotion additionally requires hourly intrabar replay
so the 5 bps fill-through buffer and 6h max-hold guard are exercised. Daily
execution is retained for smoke/legacy inspection but cannot pass the gate
unless ``--allow-daily-promotion`` is set explicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _monthly_from_total(total_return: float, window_days: int,
                        trading_days_per_month: float = 21.0) -> float:
    """Annualise a total window return into a per-month compound rate.

    monthly = (1 + total) ** (trading_days_per_month / window_days) - 1
    """
    if window_days <= 0:
        return 0.0
    try:
        return math.expm1(math.log1p(float(total_return)) * (trading_days_per_month / float(window_days)))
    except Exception:
        return 0.0


def _total_from_monthly(monthly_return: float, window_days: int,
                        trading_days_per_month: float = 21.0) -> float:
    """Convert a monthly compound target into the equivalent window return."""
    if window_days <= 0:
        return 0.0
    try:
        return math.expm1(
            math.log1p(float(monthly_return)) * (float(window_days) / trading_days_per_month)
        )
    except Exception:
        return math.inf


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary_int_metric(summary: Dict[str, Any], key: str, default: int = 0) -> int | None:
    value = summary.get(key, default)
    if isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out) or not out.is_integer() or out < 0:
        return None
    return int(out)


def _parse_int_csv(
    value: str,
    *,
    label: str,
    allow_empty: bool = False,
    min_value: int | None = None,
) -> list[int]:
    if value.strip() == "":
        if allow_empty:
            return []
        raise ValueError(f"{label} must not be empty")
    out: list[int] = []
    seen: set[int] = set()
    for idx, raw in enumerate(value.split(",")):
        token = raw.strip()
        if token == "":
            raise ValueError(f"{label} contains an empty entry at position {idx}")
        try:
            parsed = int(token)
        except ValueError as exc:
            raise ValueError(f"{label} contains non-integer entry: {token}") from exc
        if min_value is not None and parsed < int(min_value):
            raise ValueError(f"{label} contains entry below {int(min_value)}: {parsed}")
        if parsed in seen:
            raise ValueError(f"{label} contains duplicate entry: {parsed}")
        seen.add(parsed)
        out.append(parsed)
    return out


def _validate_main_numeric_args(
    args: argparse.Namespace,
    *,
    min_completed_windows: int,
) -> list[str]:
    errors: list[str] = []

    def finite_nonnegative(name: str, value: float) -> None:
        parsed = float(value)
        if not math.isfinite(parsed) or parsed < 0.0:
            errors.append(f"{name} must be finite and non-negative")

    def finite_positive(name: str, value: float) -> None:
        parsed = float(value)
        if not math.isfinite(parsed) or parsed <= 0.0:
            errors.append(f"{name} must be finite and positive")

    def int_nonnegative(name: str, value: int) -> None:
        if int(value) < 0:
            errors.append(f"{name} must be non-negative")

    def int_positive(name: str, value: int) -> None:
        if int(value) <= 0:
            errors.append(f"{name} must be positive")

    int_positive("n_windows", int(args.n_windows))
    int_positive("window_days", int(args.window_days))
    int_nonnegative("min_window_days", int(args.min_window_days))
    int_nonnegative("decision_lag", int(args.decision_lag))
    int_nonnegative("min_decision_lag", int(args.min_decision_lag))
    int_nonnegative("min_max_slippage_bps", int(args.min_max_slippage_bps))
    int_nonnegative("min_completed_windows", int(min_completed_windows))

    finite_nonnegative("monthly_target", float(args.monthly_target))
    finite_nonnegative("max_dd_target", float(args.max_dd_target))
    finite_nonnegative("fee_rate", float(args.fee_rate))
    finite_nonnegative("min_fee_rate", float(args.min_fee_rate))
    finite_nonnegative("short_borrow_apr", float(args.short_borrow_apr))
    finite_nonnegative("min_short_borrow_apr", float(args.min_short_borrow_apr))
    finite_positive("max_leverage", float(args.max_leverage))
    finite_nonnegative("max_leverage_target", float(args.max_leverage_target))
    finite_nonnegative("fail_fast_max_dd", float(args.fail_fast_max_dd))
    int_nonnegative("fail_fast_min_completed", int(args.fail_fast_min_completed))

    if str(args.execution_granularity) == "hourly_intrabar":
        finite_nonnegative("hourly_fill_buffer_bps", float(args.hourly_fill_buffer_bps))
        finite_nonnegative("min_hourly_fill_buffer_bps", float(args.min_hourly_fill_buffer_bps))
        int_nonnegative("hourly_max_hold_hours", int(args.hourly_max_hold_hours))
        int_nonnegative(
            "max_hourly_hold_hours_target",
            int(args.max_hourly_hold_hours_target),
        )
        finite_nonnegative("hourly_stop_loss_pct", float(args.hourly_stop_loss_pct))
        finite_nonnegative("hourly_take_profit_pct", float(args.hourly_take_profit_pct))

    return errors


def _static_promotion_preflight(
    args: argparse.Namespace,
    *,
    slippages: list[int],
    required_slippages: list[int],
    min_completed_windows: int,
) -> list[str]:
    errors: list[str] = []

    if (
        not bool(args.allow_daily_promotion)
        and str(args.execution_granularity) != "hourly_intrabar"
    ):
        errors.append(
            "execution_granularity daily is not promotable; use hourly_intrabar "
            "or --allow-daily-promotion for smoke/legacy checks"
        )
    if int(args.window_days) < int(args.min_window_days):
        errors.append(f"window_days {int(args.window_days)} < {int(args.min_window_days)}")
    if int(args.decision_lag) < int(args.min_decision_lag):
        errors.append(
            f"decision_lag {int(args.decision_lag)} < {int(args.min_decision_lag)}"
        )
    if int(min_completed_windows) > int(args.n_windows):
        errors.append(
            f"min_completed_windows {int(min_completed_windows)} > n_windows {int(args.n_windows)}"
        )

    evaluated_max_slippage = max(slippages, default=0)
    if evaluated_max_slippage < int(args.min_max_slippage_bps):
        errors.append(
            f"max_slippage_bps {evaluated_max_slippage} < {int(args.min_max_slippage_bps)}"
        )
    missing_required = sorted(set(required_slippages) - set(slippages))
    if missing_required:
        errors.append(
            "missing_required_slippage_bps "
            f"{','.join(str(x) for x in missing_required)}"
        )

    fee_rate = float(args.fee_rate)
    min_fee_rate = float(args.min_fee_rate)
    if min_fee_rate > 0.0 and fee_rate < min_fee_rate:
        errors.append(f"fee_rate {fee_rate:g} < {min_fee_rate:g}")
    short_borrow_apr = float(args.short_borrow_apr)
    min_short_borrow_apr = float(args.min_short_borrow_apr)
    if min_short_borrow_apr > 0.0 and short_borrow_apr < min_short_borrow_apr:
        errors.append(f"short_borrow_apr {short_borrow_apr:g} < {min_short_borrow_apr:g}")
    max_leverage = float(args.max_leverage)
    max_leverage_target = float(args.max_leverage_target)
    if max_leverage_target > 0.0 and max_leverage > max_leverage_target:
        errors.append(f"max_leverage {max_leverage:g} > {max_leverage_target:g}")

    if str(args.execution_granularity) == "hourly_intrabar":
        fill_buffer = float(args.hourly_fill_buffer_bps)
        min_fill_buffer = float(args.min_hourly_fill_buffer_bps)
        if min_fill_buffer > 0.0 and fill_buffer < min_fill_buffer:
            errors.append(f"hourly_fill_buffer_bps {fill_buffer:g} < {min_fill_buffer:g}")
        max_hold = int(args.hourly_max_hold_hours)
        max_hold_target = int(args.max_hourly_hold_hours_target)
        if max_hold_target > 0:
            if max_hold <= 0:
                errors.append("hourly_max_hold_hours 0 disables max-hold guard")
            elif max_hold > max_hold_target:
                errors.append(f"hourly_max_hold_hours {max_hold} > {max_hold_target}")

    return errors


def _write_static_preflight_artifacts(
    *,
    out_dir: Path,
    ckpt: Path,
    val: Path,
    checkpoint_sha256: str,
    args: argparse.Namespace,
    slippages: list[int],
    required_slippages: list[int],
    min_completed_windows: int,
    failures: list[str],
) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregate = _aggregate({}, window_days=int(args.window_days))
    hourly_fill_buffer_for_gate = (
        float(args.hourly_fill_buffer_bps)
        if str(args.execution_granularity) == "hourly_intrabar"
        else None
    )
    hourly_max_hold_for_gate = (
        int(args.hourly_max_hold_hours)
        if str(args.execution_granularity) == "hourly_intrabar"
        else None
    )
    gate = _promotion_status(
        aggregate,
        target_monthly=float(args.monthly_target),
        max_dd_target=float(args.max_dd_target),
        window_days=int(args.window_days),
        min_window_days=int(args.min_window_days),
        decision_lag=int(args.decision_lag),
        min_decision_lag=int(args.min_decision_lag),
        fee_rate=float(args.fee_rate),
        min_fee_rate=float(args.min_fee_rate),
        short_borrow_apr=float(args.short_borrow_apr),
        min_short_borrow_apr=float(args.min_short_borrow_apr),
        max_leverage=float(args.max_leverage),
        max_leverage_target=float(args.max_leverage_target),
        execution_granularity=str(args.execution_granularity),
        require_hourly_intrabar=not bool(args.allow_daily_promotion),
        slippage_bps=slippages,
        min_max_slippage_bps=int(args.min_max_slippage_bps),
        required_slippage_bps=required_slippages,
        hourly_fill_buffer_bps=hourly_fill_buffer_for_gate,
        min_hourly_fill_buffer_bps=float(args.min_hourly_fill_buffer_bps),
        hourly_max_hold_hours=hourly_max_hold_for_gate,
        max_hourly_hold_hours=int(args.max_hourly_hold_hours_target),
        max_negative_windows=int(args.max_negative_windows),
        min_completed_windows=int(min_completed_windows),
    )
    gate["static_preflight"] = True
    gate["failures"] = list(dict.fromkeys([*list(gate["failures"]), *failures]))
    md = (
        f"# 100d unseen-data eval — `{ckpt.name}`\n\n"
        f"- checkpoint_sha256: `{checkpoint_sha256}`\n"
        "- **status**: STATIC_PREFLIGHT_FAIL\n"
        f"- failures: {'; '.join(failures)}\n"
        f"- windows: {int(args.n_windows)} × {int(args.window_days)}d\n"
        f"- decision_lag: {int(args.decision_lag)}\n"
        f"- slippage_bps: {','.join(str(x) for x in slippages)}\n"
        f"- required_slippage_bps: {','.join(str(x) for x in required_slippages)}\n"
        "- promotion_gate: FAIL\n"
    )
    (out_dir / f"{ckpt.stem}_eval100d.md").write_text(md)
    (out_dir / f"{ckpt.stem}_eval100d.json").write_text(json.dumps({
        "checkpoint": str(ckpt),
        "checkpoint_sha256": checkpoint_sha256,
        "val_data": str(val),
        "raw": {"status": "static_promotion_preflight_failed"},
        "aggregate": aggregate,
        "promotion_gate": gate,
        "monthly_target": float(args.monthly_target),
        "max_dd_target": float(args.max_dd_target),
        "min_window_days": int(args.min_window_days),
        "decision_lag": int(args.decision_lag),
        "min_decision_lag": int(args.min_decision_lag),
        "fee_rate": float(args.fee_rate),
        "min_fee_rate": float(args.min_fee_rate),
        "short_borrow_apr": float(args.short_borrow_apr),
        "min_short_borrow_apr": float(args.min_short_borrow_apr),
        "max_leverage": float(args.max_leverage),
        "max_leverage_target": float(args.max_leverage_target),
        "execution_granularity": str(args.execution_granularity),
        "require_hourly_intrabar": not bool(args.allow_daily_promotion),
        "min_max_slippage_bps": int(args.min_max_slippage_bps),
        "required_slippage_bps": required_slippages,
        "hourly_fill_buffer_bps": gate["hourly_fill_buffer_bps"],
        "min_hourly_fill_buffer_bps": gate["min_hourly_fill_buffer_bps"],
        "hourly_max_hold_hours": gate["hourly_max_hold_hours"],
        "max_hourly_hold_hours": gate["max_hourly_hold_hours"],
        "max_negative_windows": int(args.max_negative_windows),
        "min_completed_windows": int(min_completed_windows),
        "n_windows": int(args.n_windows),
        "window_days": int(args.window_days),
        "slippage_bps": slippages,
    }, indent=2, default=str))
    return md


def _aggregate(summaries_by_bps: Dict[str, Any], window_days: int) -> Dict[str, Any]:
    """Flatten {bps: {...}} into a single aggregate dict.

    For each slippage cell we compute the implied median monthly return
    from ``median_return`` (a per-window total), then return the minimum
    across slippages (conservative: "does the worst slip still meet target?").
    Drawdown is tracked both as the displayed median-window DD and the stricter
    worst-window DD used by the promotion gate when available.
    """
    out: Dict[str, Any] = {
        "by_slippage": {},
        "worst_slip_monthly": None,
        "worst_slip_bps_by_monthly": None,
        "max_slip_median_max_drawdown": None,
        "max_slip_worst_max_drawdown": None,
        "worst_slip_bps_by_drawdown": None,
        "max_slip_negative_windows": None,
        "worst_slip_bps_by_negative_windows": None,
        "min_slip_n_windows": None,
        "worst_slip_bps_by_n_windows": None,
    }
    worst: float | None = None
    worst_bps: str | None = None
    max_dd: float | None = None
    max_dd_bps: str | None = None
    max_neg: int | None = None
    max_neg_bps: str | None = None
    invalid_neg_count = False
    min_n_windows: int | None = None
    min_n_windows_bps: str | None = None
    invalid_window_count = False
    for bps, cell in summaries_by_bps.items():
        # Both _run_slippage_sweep and _same_backend_eval emit slightly
        # different shapes; normalise here.
        summary = cell.get("summary") or cell
        median_ret = float(summary.get("median_return", 0.0))
        p10_ret = float(summary.get("p10_return", 0.0))
        mean_ret = float(summary.get("mean_return", 0.0))
        median_sortino = float(summary.get("sortino", 0.0))
        median_dd = abs(float(summary.get("max_drawdown", 0.0)))
        worst_dd = abs(float(summary.get("worst_max_drawdown", median_dd)))
        n = _summary_int_metric(summary, "n_windows")
        n_neg = _summary_int_metric(summary, "n_neg")
        if n is None:
            invalid_window_count = True
            min_n_windows_bps = str(bps)
        if n_neg is None:
            invalid_neg_count = True
            max_neg_bps = str(bps)
        monthly = _monthly_from_total(median_ret, window_days)
        p10_monthly = _monthly_from_total(p10_ret, window_days)
        cell_out = {
            "median_total_return": median_ret,
            "p10_total_return": p10_ret,
            "mean_total_return": mean_ret,
            "median_monthly_return": monthly,
            "p10_monthly_return": p10_monthly,
            "median_sortino": median_sortino,
            "median_max_drawdown": median_dd,
            "worst_max_drawdown": worst_dd,
            "n_windows": math.nan if n is None else n,
            "n_negative_windows": math.nan if n_neg is None else n_neg,
        }
        out["by_slippage"][str(bps)] = cell_out
        if worst is None or monthly < worst:
            worst = monthly
            worst_bps = str(bps)
        if max_dd is None or worst_dd > max_dd:
            max_dd = worst_dd
            max_dd_bps = str(bps)
        if n_neg is not None and not invalid_neg_count and (max_neg is None or n_neg > max_neg):
            max_neg = n_neg
            max_neg_bps = str(bps)
        if n is not None and not invalid_window_count and (min_n_windows is None or n < min_n_windows):
            min_n_windows = n
            min_n_windows_bps = str(bps)
    out["worst_slip_monthly"] = float(worst if worst is not None else 0.0)
    out["worst_slip_bps_by_monthly"] = worst_bps
    out["max_slip_median_max_drawdown"] = max(
        (float(c["median_max_drawdown"]) for c in out["by_slippage"].values()),
        default=0.0,
    )
    out["max_slip_worst_max_drawdown"] = float(max_dd if max_dd is not None else 0.0)
    out["worst_slip_bps_by_drawdown"] = max_dd_bps
    out["max_slip_negative_windows"] = (
        math.nan if invalid_neg_count else int(max_neg if max_neg is not None else 0)
    )
    out["worst_slip_bps_by_negative_windows"] = max_neg_bps
    out["min_slip_n_windows"] = (
        math.nan
        if invalid_window_count
        else int(min_n_windows if min_n_windows is not None else 0)
    )
    out["worst_slip_bps_by_n_windows"] = min_n_windows_bps
    return out


def _finite_metric(aggregate: Dict[str, Any], key: str, failures: list[str]) -> tuple[float, bool]:
    value = aggregate.get(key)
    try:
        out = float(value)
    except (TypeError, ValueError):
        failures.append(f"{key} is not finite")
        return 0.0, False
    if not math.isfinite(out):
        failures.append(f"{key} is not finite")
        return 0.0, False
    return out, True


def _integer_metric(aggregate: Dict[str, Any], key: str, failures: list[str]) -> tuple[int, bool]:
    value = aggregate.get(key)
    if isinstance(value, bool):
        failures.append(f"{key} is not an integer")
        return 0, False
    try:
        out_float = float(value)
    except (TypeError, ValueError):
        failures.append(f"{key} is not an integer")
        return 0, False
    if not math.isfinite(out_float) or not out_float.is_integer():
        failures.append(f"{key} is not an integer")
        return 0, False
    return int(out_float), True


def _promotion_status(
    aggregate: Dict[str, Any],
    *,
    target_monthly: float,
    max_dd_target: float,
    window_days: int | None = None,
    min_window_days: int = 0,
    decision_lag: int | None = None,
    min_decision_lag: int = 0,
    fee_rate: float | None = None,
    min_fee_rate: float = 0.0,
    short_borrow_apr: float | None = None,
    min_short_borrow_apr: float = 0.0,
    max_leverage: float | None = None,
    max_leverage_target: float = 0.0,
    execution_granularity: str | None = None,
    require_hourly_intrabar: bool = False,
    slippage_bps: list[int] | None = None,
    min_max_slippage_bps: int = 0,
    required_slippage_bps: list[int] | None = None,
    hourly_fill_buffer_bps: float | None = None,
    min_hourly_fill_buffer_bps: float = 0.0,
    hourly_max_hold_hours: int | None = None,
    max_hourly_hold_hours: int = 0,
    max_negative_windows: int = 0,
    min_completed_windows: int = 0,
) -> Dict[str, Any]:
    failures: list[str] = []
    worst_m, worst_m_ok = _finite_metric(aggregate, "worst_slip_monthly", failures)
    if "max_slip_worst_max_drawdown" in aggregate:
        worst_dd, worst_dd_ok = _finite_metric(aggregate, "max_slip_worst_max_drawdown", failures)
    else:
        worst_dd, worst_dd_ok = _finite_metric(aggregate, "max_slip_median_max_drawdown", failures)
    median_dd, _median_dd_ok = _finite_metric(aggregate, "max_slip_median_max_drawdown", failures)
    monthly_target = float(target_monthly)
    if not math.isfinite(monthly_target) or monthly_target < 0.0:
        failures.append("monthly_target must be finite and non-negative")
    max_dd_cap = float(max_dd_target)
    if not math.isfinite(max_dd_cap) or max_dd_cap < 0.0:
        failures.append("max_dd_target must be finite and non-negative")
    if worst_m_ok and math.isfinite(monthly_target) and worst_m < monthly_target:
        failures.append(
            f"worst_slip_monthly {worst_m * 100:.2f}% < "
            f"{monthly_target * 100:.2f}%"
        )
    if worst_dd_ok and math.isfinite(max_dd_cap) and worst_dd > max_dd_cap:
        failures.append(
            f"max_slip_worst_max_drawdown {worst_dd * 100:.2f}% > "
            f"{max_dd_cap * 100:.2f}%"
        )
    min_window = int(min_window_days)
    if min_window < 0:
        failures.append("min_window_days must be non-negative")
    elif window_days is not None and int(window_days) < min_window:
        failures.append(f"window_days {int(window_days)} < {min_window}")
    min_lag = int(min_decision_lag)
    if min_lag < 0:
        failures.append("min_decision_lag must be non-negative")
    elif decision_lag is not None and int(decision_lag) < min_lag:
        failures.append(f"decision_lag {int(decision_lag)} < {min_lag}")
    fee = None if fee_rate is None else float(fee_rate)
    min_fee = float(min_fee_rate)
    if not math.isfinite(min_fee) or min_fee < 0.0:
        failures.append("min_fee_rate must be finite and non-negative")
    if fee is not None:
        if not math.isfinite(fee) or fee < 0.0:
            failures.append("fee_rate must be finite and non-negative")
        elif min_fee > 0.0 and fee < min_fee:
            failures.append(f"fee_rate {fee:g} < {min_fee:g}")
    borrow = None if short_borrow_apr is None else float(short_borrow_apr)
    min_borrow = float(min_short_borrow_apr)
    if not math.isfinite(min_borrow) or min_borrow < 0.0:
        failures.append("min_short_borrow_apr must be finite and non-negative")
    if borrow is not None:
        if not math.isfinite(borrow) or borrow < 0.0:
            failures.append("short_borrow_apr must be finite and non-negative")
        elif min_borrow > 0.0 and borrow < min_borrow:
            failures.append(f"short_borrow_apr {borrow:g} < {min_borrow:g}")
    leverage = None if max_leverage is None else float(max_leverage)
    leverage_limit = float(max_leverage_target)
    if not math.isfinite(leverage_limit) or leverage_limit < 0.0:
        failures.append("max_leverage_target must be finite and non-negative")
    if leverage is not None:
        if not math.isfinite(leverage) or leverage <= 0.0:
            failures.append("max_leverage must be finite and positive")
        elif leverage_limit > 0.0 and leverage > leverage_limit:
            failures.append(f"max_leverage {leverage:g} > {leverage_limit:g}")
    granularity = None if execution_granularity is None else str(execution_granularity)
    if require_hourly_intrabar and granularity != "hourly_intrabar":
        failures.append(
            "execution_granularity daily is not promotable; use hourly_intrabar "
            "or --allow-daily-promotion for smoke/legacy checks"
        )
    min_max_slip = int(min_max_slippage_bps)
    if min_max_slip < 0:
        failures.append("min_max_slippage_bps must be non-negative")
    evaluated_slippage_set: set[int] = set()
    evaluated_max_slippage_bps = None
    if slippage_bps is not None:
        evaluated_slippage_set = {int(x) for x in slippage_bps}
        if any(x < 0 for x in evaluated_slippage_set):
            failures.append("slippage_bps must be non-negative")
        evaluated_max_slippage_bps = max(evaluated_slippage_set, default=0)
    if (
        evaluated_max_slippage_bps is not None
        and min_max_slip >= 0
        and int(evaluated_max_slippage_bps) < min_max_slip
    ):
        failures.append(
            f"max_slippage_bps {int(evaluated_max_slippage_bps)} < "
            f"{min_max_slip}"
        )
    required_slippage_set = (
        set() if required_slippage_bps is None else {int(x) for x in required_slippage_bps}
    )
    if any(x < 0 for x in required_slippage_set):
        failures.append("required_slippage_bps must be non-negative")
    missing_required_slippage = sorted(required_slippage_set - evaluated_slippage_set)
    if missing_required_slippage:
        failures.append(
            "missing_required_slippage_bps "
            f"{','.join(str(x) for x in missing_required_slippage)}"
        )
    fill_buffer = (
        None if hourly_fill_buffer_bps is None else float(hourly_fill_buffer_bps)
    )
    min_fill_buffer = float(min_hourly_fill_buffer_bps)
    if not math.isfinite(min_fill_buffer) or min_fill_buffer < 0.0:
        failures.append("min_hourly_fill_buffer_bps must be finite and non-negative")
    if fill_buffer is not None:
        if not math.isfinite(fill_buffer) or fill_buffer < 0.0:
            failures.append("hourly_fill_buffer_bps must be finite and non-negative")
        elif min_fill_buffer > 0.0 and fill_buffer < min_fill_buffer:
            failures.append(
                f"hourly_fill_buffer_bps {fill_buffer:g} < {min_fill_buffer:g}"
            )
    max_hold_limit = int(max_hourly_hold_hours)
    if max_hold_limit < 0:
        failures.append("max_hourly_hold_hours must be non-negative")
    elif hourly_max_hold_hours is not None and max_hold_limit > 0:
        if int(hourly_max_hold_hours) <= 0:
            failures.append("hourly_max_hold_hours 0 disables max-hold guard")
        elif int(hourly_max_hold_hours) > max_hold_limit:
            failures.append(
                f"hourly_max_hold_hours {int(hourly_max_hold_hours)} > "
                f"{max_hold_limit}"
            )
    negative_windows, negative_windows_ok = _integer_metric(
        aggregate, "max_slip_negative_windows", failures
    )
    negative_window_cap = int(max_negative_windows)
    if (
        negative_windows_ok
        and negative_window_cap >= 0
        and negative_windows > negative_window_cap
    ):
        failures.append(f"negative_windows {negative_windows} > {negative_window_cap}")
    completed_windows, completed_windows_ok = _integer_metric(
        aggregate, "min_slip_n_windows", failures
    )
    min_completed = int(min_completed_windows)
    if min_completed < 0:
        failures.append("min_completed_windows must be non-negative")
    if (
        completed_windows_ok
        and min_completed > 0
        and completed_windows < min_completed
    ):
        failures.append(
            f"completed_windows {completed_windows} < {min_completed}"
        )
    return {
        "passed": not failures,
        "failures": failures,
        "worst_slip_monthly": worst_m,
        "max_slip_median_max_drawdown": median_dd,
        "max_slip_worst_max_drawdown": worst_dd,
        "monthly_target": monthly_target,
        "max_dd_target": max_dd_cap,
        "window_days": None if window_days is None else int(window_days),
        "min_window_days": min_window,
        "decision_lag": None if decision_lag is None else int(decision_lag),
        "min_decision_lag": min_lag,
        "fee_rate": fee,
        "min_fee_rate": min_fee,
        "short_borrow_apr": borrow,
        "min_short_borrow_apr": min_borrow,
        "max_leverage": leverage,
        "max_leverage_target": leverage_limit,
        "execution_granularity": granularity,
        "require_hourly_intrabar": bool(require_hourly_intrabar),
        "max_slippage_bps": evaluated_max_slippage_bps,
        "min_max_slippage_bps": min_max_slip,
        "required_slippage_bps": sorted(required_slippage_set),
        "missing_required_slippage_bps": missing_required_slippage,
        "hourly_fill_buffer_bps": fill_buffer,
        "min_hourly_fill_buffer_bps": min_fill_buffer,
        "hourly_max_hold_hours": (
            None if hourly_max_hold_hours is None else int(hourly_max_hold_hours)
        ),
        "max_hourly_hold_hours": max_hold_limit,
        "negative_windows": negative_windows,
        "max_negative_windows": negative_window_cap,
        "completed_windows": completed_windows,
        "min_completed_windows": min_completed,
    }


def _evaluated_slippage_bps(aggregate: Dict[str, Any]) -> list[int]:
    values: list[int] = []
    for bps in aggregate.get("by_slippage", {}).keys():
        try:
            values.append(int(bps))
        except (TypeError, ValueError):
            continue
    return sorted(set(values))


def _render_md(
    ckpt: Path,
    checkpoint_sha256: str,
    aggregate: Dict[str, Any],
    window_days: int,
    n_windows: int,
    target_monthly: float,
    max_dd_target: float,
    eval_result: Dict[str, Any],
    min_window_days: int = 0,
    decision_lag: int | None = None,
    min_decision_lag: int = 0,
    fee_rate: float | None = None,
    min_fee_rate: float = 0.0,
    short_borrow_apr: float | None = None,
    min_short_borrow_apr: float = 0.0,
    max_leverage: float | None = None,
    max_leverage_target: float = 0.0,
    execution_granularity: str | None = None,
    require_hourly_intrabar: bool = False,
    slippage_bps: list[int] | None = None,
    min_max_slippage_bps: int = 0,
    required_slippage_bps: list[int] | None = None,
    hourly_fill_buffer_bps: float | None = None,
    min_hourly_fill_buffer_bps: float = 0.0,
    hourly_max_hold_hours: int | None = None,
    max_hourly_hold_hours: int = 0,
    max_negative_windows: int = 0,
    min_completed_windows: int = 0,
) -> str:
    gate = _promotion_status(
        aggregate,
        target_monthly=float(target_monthly),
        max_dd_target=float(max_dd_target),
        window_days=int(window_days),
        min_window_days=int(min_window_days),
        decision_lag=decision_lag,
        min_decision_lag=int(min_decision_lag),
        fee_rate=fee_rate,
        min_fee_rate=float(min_fee_rate),
        short_borrow_apr=short_borrow_apr,
        min_short_borrow_apr=float(min_short_borrow_apr),
        max_leverage=max_leverage,
        max_leverage_target=float(max_leverage_target),
        execution_granularity=execution_granularity,
        require_hourly_intrabar=bool(require_hourly_intrabar),
        slippage_bps=_evaluated_slippage_bps(aggregate),
        min_max_slippage_bps=int(min_max_slippage_bps),
        required_slippage_bps=required_slippage_bps,
        hourly_fill_buffer_bps=hourly_fill_buffer_bps,
        min_hourly_fill_buffer_bps=float(min_hourly_fill_buffer_bps),
        hourly_max_hold_hours=hourly_max_hold_hours,
        max_hourly_hold_hours=int(max_hourly_hold_hours),
        max_negative_windows=int(max_negative_windows),
        min_completed_windows=int(min_completed_windows),
    )
    worst_m = float(gate["worst_slip_monthly"])
    worst_dd = float(gate["max_slip_worst_max_drawdown"])
    ok = "PASS" if bool(gate["passed"]) else "FAIL"
    lines: List[str] = []
    lines.append(f"# 100d unseen-data eval — `{ckpt.name}`")
    lines.append("")
    lines.append(f"- checkpoint_sha256: `{checkpoint_sha256}`")
    lines.append(
        f"- **status**: {ok}  "
        f"({worst_m * 100:.2f}%/month vs target {target_monthly * 100:.2f}%/month; "
        f"max DD {worst_dd * 100:.2f}% vs cap {max_dd_target * 100:.2f}%)"
    )
    if gate["failures"]:
        lines.append(f"- failures: {'; '.join(gate['failures'])}")
    lines.append(f"- windows: {n_windows} × {window_days}d  (total {n_windows * window_days}d unseen)")
    if decision_lag is not None:
        lines.append(f"- decision_lag: {int(decision_lag)}")
    if fee_rate is not None:
        lines.append(
            f"- fee_rate: {float(fee_rate):g} "
            f"(min {float(min_fee_rate):g})"
        )
    if short_borrow_apr is not None:
        lines.append(
            f"- short_borrow_apr: {float(short_borrow_apr):g} "
            f"(min {float(min_short_borrow_apr):g})"
        )
    if max_leverage is not None:
        lines.append(
            f"- max_leverage: {float(max_leverage):g} "
            f"(max {float(max_leverage_target):g})"
        )
    if execution_granularity is not None:
        requirement = "required" if require_hourly_intrabar else "not required"
        lines.append(
            f"- execution_granularity: {execution_granularity} "
            f"(hourly_intrabar {requirement})"
        )
    if slippage_bps is not None:
        lines.append(f"- slippage_bps: {','.join(str(int(x)) for x in slippage_bps)}")
    if required_slippage_bps is not None:
        lines.append(
            "- required_slippage_bps: "
            f"{','.join(str(int(x)) for x in required_slippage_bps)}"
        )
    if hourly_fill_buffer_bps is not None:
        lines.append(
            f"- hourly_fill_buffer_bps: {float(hourly_fill_buffer_bps):g} "
            f"(min {float(min_hourly_fill_buffer_bps):g})"
        )
    if hourly_max_hold_hours is not None:
        lines.append(
            f"- hourly_max_hold_hours: {int(hourly_max_hold_hours)} "
            f"(max {int(max_hourly_hold_hours)})"
        )
    lines.append(
        f"- negative_windows: {int(gate['negative_windows'])} "
        f"(cap {int(gate['max_negative_windows'])})"
    )
    lines.append(
        f"- completed_windows: {int(gate['completed_windows'])} "
        f"(min {int(gate['min_completed_windows'])})"
    )
    lines.append(f"- backend: {eval_result.get('backend', 'pufferlib_market')}")
    if 'shape_mismatch' in eval_result:
        lines.append(f"- shape_mismatch: `{eval_result['shape_mismatch']}`")
    if 'videos' in eval_result and isinstance(eval_result['videos'], dict):
        vids = eval_result['videos']
        if 'mp4' in vids:
            lines.append(f"- video: `{vids['mp4']}`")
        if 'html' in vids:
            lines.append(f"- scrubber: `{vids['html']}`")
    lines.append("")
    lines.append(
        "| slip_bps | median total | median monthly | p10 total | p10 monthly | "
        "sortino | med dd | worst dd | n_neg |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for bps, cell in sorted(aggregate["by_slippage"].items(), key=lambda kv: int(kv[0])):
        lines.append(
            f"| {bps} "
            f"| {cell['median_total_return'] * 100:+.2f}% "
            f"| {cell['median_monthly_return'] * 100:+.2f}% "
            f"| {cell['p10_total_return'] * 100:+.2f}% "
            f"| {cell['p10_monthly_return'] * 100:+.2f}% "
            f"| {cell['median_sortino']:.2f} "
            f"| {cell['median_max_drawdown'] * 100:.2f}% "
            f"| {cell['worst_max_drawdown'] * 100:.2f}% "
            f"| {cell['n_negative_windows']}/{cell['n_windows']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _slice_mktd_window(data, *, start: int, steps: int):
    from pufferlib_market.hourly_replay import MktdData

    end = int(start) + int(steps) + 1
    if start < 0 or end > data.num_timesteps:
        raise ValueError(f"Invalid MKTD slice start={start} steps={steps} for T={data.num_timesteps}")
    tradable = None if data.tradable is None else data.tradable[start:end].copy()
    return MktdData(
        version=int(data.version),
        symbols=list(data.symbols),
        features=data.features[start:end].copy(),
        prices=data.prices[start:end].copy(),
        tradable=tradable,
    )


def _summarise_window_metrics(rows: List[Dict[str, float]]) -> Dict[str, Any]:
    if not rows:
        return {"error": "no windows completed"}
    returns = np.asarray([float(r["total_return"]) for r in rows], dtype=np.float64)
    sortinos = np.asarray([float(r["sortino"]) for r in rows], dtype=np.float64)
    maxdds = np.asarray([float(r["max_drawdown"]) for r in rows], dtype=np.float64)
    abs_maxdds = np.abs(maxdds)
    return {
        "p10_return": float(np.percentile(returns, 10)),
        "median_return": float(np.percentile(returns, 50)),
        "p90_return": float(np.percentile(returns, 90)),
        "mean_return": float(np.mean(returns)),
        "sortino": float(np.median(sortinos)),
        "max_drawdown": float(np.median(abs_maxdds)),
        "worst_max_drawdown": float(np.max(abs_maxdds)),
        "n_neg": int(np.sum(returns < 0.0)),
        "n_windows": int(returns.size),
    }


def _median_target_impossible(
    rows: List[Dict[str, float]],
    *,
    n_windows: int,
    window_days: int,
    target_monthly: float,
) -> tuple[bool, int, float]:
    """Return true when remaining windows cannot rescue the final median target."""
    planned_windows = int(n_windows)
    if planned_windows <= 0:
        return False, 0, 0.0
    target_total = _total_from_monthly(float(target_monthly), int(window_days))
    if not math.isfinite(target_total):
        return False, 0, target_total
    below_target = 0
    for row in rows:
        try:
            total_return = float(row["total_return"])
        except (KeyError, TypeError, ValueError):
            below_target += 1
            continue
        if not math.isfinite(total_return) or total_return < target_total:
            below_target += 1
    return below_target > (planned_windows // 2), below_target, target_total


def _evaluate_intrabar_hourly(
    *,
    checkpoint: Path,
    val_data: Path,
    hourly_data_root: Path,
    daily_start_date: str,
    n_windows: int,
    window_days: int,
    slippages: List[int],
    fee_rate: float,
    short_borrow_apr: float,
    max_leverage: float,
    seed: int,
    fill_buffer_bps: float,
    max_hold_hours: int,
    stop_loss_pct: float,
    take_profit_pct: float,
    trade_hour_mode: str,
    decision_lag: int,
    target_monthly: float,
    fail_fast: bool,
    fail_fast_max_dd: float,
    fail_fast_min_completed: int,
):
    import torch

    from pufferlib_market.evaluate_multiperiod import load_policy, make_policy_fn
    from pufferlib_market.hourly_replay import read_mktd
    from pufferlib_market.intrabar_replay import load_hourly_ohlc, simulate_daily_policy_intrabar

    data = read_mktd(val_data)
    num_symbols = data.num_symbols
    features_per_sym = int(data.features.shape[2])
    loaded = load_policy(
        str(checkpoint),
        num_symbols,
        arch="auto",
        hidden_size=None,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        features_per_sym=features_per_sym,
    )

    full_start_day = np.datetime64(str(daily_start_date))
    full_end_day = full_start_day + np.timedelta64(data.num_timesteps - 1, "D")
    hourly = load_hourly_ohlc(
        data.symbols,
        hourly_data_root,
        start=f"{str(full_start_day)} 00:00",
        end=f"{str(full_end_day)} 23:00",
    )

    window_len = int(window_days) + 1
    max_offset = data.num_timesteps - window_len
    if max_offset < 0:
        return {"status": "skip", "reason": f"val data too short: {data.num_timesteps} < {window_len}"}

    rng = np.random.default_rng(int(seed))
    starts = rng.choice(max_offset + 1, size=int(n_windows), replace=(max_offset + 1 < int(n_windows)))
    by_slippage: Dict[str, Any] = {}

    for bps in slippages:
        cell_rows: List[Dict[str, float]] = []
        failed_fast_reason: str | None = None
        for idx, start in enumerate(starts.tolist()):
            window_data = _slice_mktd_window(data, start=int(start), steps=int(window_days))
            policy_fn = make_policy_fn(
                loaded.policy,
                num_symbols=num_symbols,
                deterministic=True,
                decision_lag=int(decision_lag),
                device=next(loaded.policy.parameters()).device,
            )
            window_start_day = full_start_day + np.timedelta64(int(start), "D")
            result = simulate_daily_policy_intrabar(
                data=window_data,
                policy_fn=policy_fn,
                hourly=hourly,
                start_date=str(window_start_day),
                max_steps=int(window_days),
                fee_rate=float(fee_rate) + float(bps) / 10_000.0,
                short_borrow_apr=float(short_borrow_apr),
                fill_buffer_bps=float(fill_buffer_bps),
                max_leverage=float(max_leverage),
                stop_loss_pct=(float(stop_loss_pct) if stop_loss_pct > 0.0 else None),
                take_profit_pct=(float(take_profit_pct) if take_profit_pct > 0.0 else None),
                max_hold_hours=(int(max_hold_hours) if max_hold_hours > 0 else None),
                periods_per_year=8760.0,
                action_allocation_bins=loaded.action_allocation_bins,
                action_level_bins=loaded.action_level_bins,
                action_max_offset_bps=loaded.action_max_offset_bps,
                trade_hour_mode=trade_hour_mode,
            )
            rec = {
                "total_return": float(result.total_return),
                "sortino": float(result.sortino),
                "max_drawdown": float(result.max_drawdown),
            }
            cell_rows.append(rec)
            if fail_fast:
                if rec["max_drawdown"] > float(fail_fast_max_dd):
                    failed_fast_reason = (
                        f"window {idx} max_drawdown={rec['max_drawdown']:.3f} > "
                        f"{float(fail_fast_max_dd):.3f}"
                    )
                    break
                if rec["total_return"] < 0.0 and len(cell_rows) >= int(fail_fast_min_completed):
                    failed_fast_reason = (
                        f"window {idx} total_return={rec['total_return']:.4f} < 0 "
                        f"after {len(cell_rows)} completed"
                    )
                    break
                if len(cell_rows) >= int(fail_fast_min_completed):
                    impossible, below_target, target_total = _median_target_impossible(
                        cell_rows,
                        n_windows=int(n_windows),
                        window_days=int(window_days),
                        target_monthly=float(target_monthly),
                    )
                    if impossible:
                        failed_fast_reason = (
                            "median_target_impossible: "
                            f"{below_target}/{int(n_windows)} windows below "
                            f"target_total={target_total:.4f}"
                        )
                        break

        cell = _summarise_window_metrics(cell_rows)
        if failed_fast_reason is not None:
            cell["failed_fast"] = True
            cell["failed_reason"] = failed_fast_reason
            cell["n_completed_before_bail"] = int(len(cell_rows))
            by_slippage[str(int(bps))] = cell
            return {
                "status": "failed_fast",
                "backend": "pufferlib_market_intrabar_hourly",
                "failed_reason": failed_fast_reason,
                "by_slippage": by_slippage,
            }
        by_slippage[str(int(bps))] = cell

    return {"status": "ok", "backend": "pufferlib_market_intrabar_hourly", "by_slippage": by_slippage}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val-data", required=True,
                    help="Path to pufferlib_market .bin. For fp4 checkpoints that "
                         "don't match this shape the script falls back to "
                         "same-backend eval and keeps going.")
    ap.add_argument("--n-windows", type=int, default=30)
    ap.add_argument("--window-days", type=int, default=100)
    ap.add_argument("--min-window-days", type=int, default=100,
                    help="Minimum window length allowed to pass the final "
                         "promotion gate (default 100). Shorter smoke evals "
                         "still write artifacts but exit 3 unless this is "
                         "lowered explicitly.")
    ap.add_argument("--decision-lag", type=int, default=2,
                    help="Bars/days between observation and executable "
                         "decision. Production eval defaults to 2.")
    ap.add_argument("--min-decision-lag", type=int, default=2,
                    help="Minimum decision lag allowed to pass the final "
                         "promotion gate (default 2). Lower-lag smoke evals "
                         "still write artifacts but exit 3 unless this is "
                         "lowered explicitly.")
    ap.add_argument("--slippage-bps", default="0,5,10,20",
                    help="Comma-separated slippage levels in bps")
    ap.add_argument("--min-max-slippage-bps", type=int, default=20,
                    help="Minimum maximum slippage that must be covered by "
                         "the final promotion gate (default 20 bps). Lower "
                         "values are useful for smoke tests but are not a "
                         "production promotion check.")
    ap.add_argument("--required-slippage-bps", default="0,5,10,20",
                    help="Comma-separated slippage levels that must be present "
                         "in completed eval results for the final promotion "
                         "gate. Default matches the production realism grid. "
                         "Set to empty only for explicit smoke/legacy checks.")
    ap.add_argument("--fee-rate", type=float, default=0.001)
    ap.add_argument("--min-fee-rate", type=float, default=0.001,
                    help="Minimum base fee rate allowed to pass the promotion "
                         "gate. Default 0.001 = 10 bps production realism; "
                         "set 0 only for explicit smoke/legacy checks.")
    ap.add_argument("--short-borrow-apr", type=float, default=0.0625,
                    help="Annualized borrow / margin rate charged by the "
                         "marketsim. Default 0.0625 = 6.25%% production realism.")
    ap.add_argument("--min-short-borrow-apr", type=float, default=0.0625,
                    help="Minimum borrow / margin APR allowed to pass the "
                         "promotion gate. Set 0 only for explicit smoke/legacy checks.")
    ap.add_argument("--max-leverage", type=float, default=1.5)
    ap.add_argument("--max-leverage-target", type=float, default=2.0,
                    help="Maximum leverage allowed to pass the promotion gate. "
                         "Default 2x matches the intended aggressive account "
                         "packing ceiling; set 0 only for explicit smoke/legacy checks.")
    ap.add_argument(
        "--execution-granularity",
        choices=["daily", "hourly_intrabar"],
        default="daily",
        help="Use the default C daily marketsim or replay daily decisions through hourly OHLC execution.",
    )
    ap.add_argument(
        "--allow-daily-promotion",
        action="store_true",
        help="Allow daily execution-granularity artifacts to pass the promotion "
             "gate. Use only for explicit smoke/legacy checks; production "
             "realism requires hourly_intrabar so fill_buffer and max_hold "
             "are exercised.",
    )
    ap.add_argument("--hourly-data-root", default=None,
                    help="Required for --execution-granularity hourly_intrabar. "
                         "Root with stocks/ and/or crypto/ hourly CSVs.")
    ap.add_argument("--daily-start-date", default=None,
                    help="Required for --execution-granularity hourly_intrabar. "
                         "UTC start date of row 0 in the MKTD file.")
    ap.add_argument("--hourly-fill-buffer-bps", type=float, default=5.0,
                    help="Hourly limit fill-through buffer for hourly_intrabar mode.")
    ap.add_argument("--min-hourly-fill-buffer-bps", type=float, default=5.0,
                    help="Minimum hourly fill-through buffer allowed to pass "
                         "the promotion gate. Default 5 bps matches production "
                         "realism; set 0 only for explicit smoke/legacy checks.")
    ap.add_argument("--hourly-max-hold-hours", type=int, default=6,
                    help="Hourly max-hold guard in hourly_intrabar mode. "
                         "Default 6h matches production realism; set 0 only "
                         "with --max-hourly-hold-hours-target 0 for smoke checks.")
    ap.add_argument("--max-hourly-hold-hours-target", type=int, default=6,
                    help="Maximum hourly max-hold setting allowed to pass the "
                         "promotion gate. Set 0 to disable only for explicit "
                         "smoke/legacy checks.")
    ap.add_argument("--hourly-stop-loss-pct", type=float, default=0.0,
                    help="Optional intrabar stop-loss fraction in hourly_intrabar mode.")
    ap.add_argument("--hourly-take-profit-pct", type=float, default=0.0,
                    help="Optional intrabar take-profit fraction in hourly_intrabar mode.")
    ap.add_argument("--hourly-trade-hour-mode", choices=["first_tradable", "last_tradable"], default="first_tradable",
                    help="When to query the daily policy inside each day for hourly_intrabar mode.")
    ap.add_argument("--monthly-target", type=float, default=0.27,
                    help="Minimum acceptable median monthly return (worst slip).")
    ap.add_argument("--max-dd-target", type=float, default=0.25,
                    help="Maximum acceptable worst-window max drawdown across "
                         "slippage cells (default 0.25 = 25%%). The final "
                         "promotion gate requires both --monthly-target and "
                         "this drawdown cap.")
    ap.add_argument("--max-negative-windows", type=int, default=0,
                    help="Maximum losing OOS windows allowed by the final "
                         "promotion gate (default 0). Set negative to disable "
                         "for smoke/legacy inspection only.")
    ap.add_argument("--min-completed-windows", type=int, default=None,
                    help="Minimum completed windows required in every returned "
                         "slippage cell. Defaults to --n-windows for production "
                         "gates; lower explicitly for smoke/legacy inspection.")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--video", action="store_true",
                    help="Render an MP4+HTML of the rollout when running the "
                         "same-backend fallback.")
    ap.add_argument("--no-fail-fast", dest="fail_fast", action="store_false",
                    help="Disable the early-exit on >max-dd or negative-window "
                         "checks (default: enabled).")
    ap.set_defaults(fail_fast=True)
    ap.add_argument("--fail-fast-max-dd", type=float, default=0.20,
                    help="Bail any sweep cell whose worst window drawdown "
                         "exceeds this fraction (default 0.20 = 20%%).")
    ap.add_argument("--fail-fast-min-completed", type=int, default=3,
                    help="Min completed windows before the negative-window "
                         "check fires (default 3).")
    ap.add_argument("--out-dir", default=None,
                    help="Where to write the JSON + MD. Defaults to the ckpt dir.")
    args = ap.parse_args(argv)

    ckpt = Path(args.checkpoint).resolve()
    val = Path(args.val_data).resolve()
    if not ckpt.exists():
        print(f"eval_100d: checkpoint not found: {ckpt}", file=sys.stderr)
        return 2
    if not val.exists():
        print(f"eval_100d: val data not found: {val}", file=sys.stderr)
        return 2
    checkpoint_sha256 = _file_sha256(ckpt)
    try:
        slippages = _parse_int_csv(
            str(args.slippage_bps),
            label="slippage_bps",
            min_value=0,
        )
        required_slippages = _parse_int_csv(
            str(args.required_slippage_bps),
            label="required_slippage_bps",
            allow_empty=True,
            min_value=0,
        )
    except ValueError as exc:
        print(f"eval_100d: {exc}", file=sys.stderr)
        return 2
    min_completed_windows = (
        int(args.n_windows)
        if args.min_completed_windows is None
        else int(args.min_completed_windows)
    )
    input_errors = _validate_main_numeric_args(
        args,
        min_completed_windows=min_completed_windows,
    )
    if input_errors:
        print(f"eval_100d: {input_errors[0]}", file=sys.stderr)
        return 2
    preflight_errors = _static_promotion_preflight(
        args,
        slippages=slippages,
        required_slippages=required_slippages,
        min_completed_windows=min_completed_windows,
    )
    if preflight_errors:
        print(f"eval_100d: {preflight_errors[0]}", file=sys.stderr)
        out_dir = Path(args.out_dir) if args.out_dir else ckpt.parent
        md = _write_static_preflight_artifacts(
            out_dir=out_dir,
            ckpt=ckpt,
            val=val,
            checkpoint_sha256=checkpoint_sha256,
            args=args,
            slippages=slippages,
            required_slippages=required_slippages,
            min_completed_windows=min_completed_windows,
            failures=preflight_errors,
        )
        print(md)
        return 3
    hourly_max_hold_for_gate = (
        int(args.hourly_max_hold_hours)
        if args.execution_granularity == "hourly_intrabar"
        else None
    )
    max_hourly_hold_hours_target = int(args.max_hourly_hold_hours_target)
    fee_rate_for_gate = float(args.fee_rate)
    min_fee_rate = float(args.min_fee_rate)
    short_borrow_apr_for_gate = float(args.short_borrow_apr)
    min_short_borrow_apr = float(args.min_short_borrow_apr)
    max_leverage_for_gate = float(args.max_leverage)
    max_leverage_target = float(args.max_leverage_target)
    hourly_fill_buffer_for_gate = (
        float(args.hourly_fill_buffer_bps)
        if args.execution_granularity == "hourly_intrabar"
        else None
    )
    min_hourly_fill_buffer_bps = float(args.min_hourly_fill_buffer_bps)

    # Build a cfg shaped like fp4_ppo_stocks12.yaml so evaluate_policy_file
    # reads the knobs it already knows.
    cfg = {
        "env": {
            "val_data": str(val.relative_to(REPO)) if val.is_relative_to(REPO) else str(val),
            "fee_rate": float(args.fee_rate),
            "short_borrow_apr": short_borrow_apr_for_gate,
            "max_leverage_scalar_fallback": float(args.max_leverage),
        },
        "eval": {
            "slippage_bps": slippages,
            "n_windows": int(args.n_windows),
            "eval_hours": int(args.window_days),
            "seed": int(args.seed),
            "video": bool(args.video),
            "decision_lag": int(args.decision_lag),
            "fail_fast": bool(args.fail_fast),
            "fail_fast_max_dd": float(args.fail_fast_max_dd),
            "fail_fast_min_completed": int(args.fail_fast_min_completed),
        },
    }

    if args.execution_granularity == "hourly_intrabar":
        if not args.hourly_data_root:
            print("eval_100d: --hourly-data-root is required for hourly_intrabar", file=sys.stderr)
            return 2
        if not args.daily_start_date:
            print("eval_100d: --daily-start-date is required for hourly_intrabar", file=sys.stderr)
            return 2
        result = _evaluate_intrabar_hourly(
            checkpoint=ckpt,
            val_data=val,
            hourly_data_root=Path(args.hourly_data_root).resolve(),
            daily_start_date=str(args.daily_start_date),
            n_windows=int(args.n_windows),
            window_days=int(args.window_days),
            slippages=slippages,
            fee_rate=float(args.fee_rate),
            short_borrow_apr=short_borrow_apr_for_gate,
            max_leverage=float(args.max_leverage),
            seed=int(args.seed),
            fill_buffer_bps=float(args.hourly_fill_buffer_bps),
            max_hold_hours=int(args.hourly_max_hold_hours),
            stop_loss_pct=float(args.hourly_stop_loss_pct),
            take_profit_pct=float(args.hourly_take_profit_pct),
            trade_hour_mode=str(args.hourly_trade_hour_mode),
            decision_lag=int(args.decision_lag),
            target_monthly=float(args.monthly_target),
            fail_fast=bool(args.fail_fast),
            fail_fast_max_dd=float(args.fail_fast_max_dd),
            fail_fast_min_completed=int(args.fail_fast_min_completed),
        )
    else:
        from fp4.bench.eval_generic import evaluate_policy_file
        result = evaluate_policy_file(ckpt, cfg, REPO)

    if result.get("status") not in ("ok", "failed_fast"):
        print(f"eval_100d: evaluate_policy_file returned status={result.get('status')}: "
              f"{result.get('reason', '<no reason>')}", file=sys.stderr)
        out_dir = Path(args.out_dir) if args.out_dir else ckpt.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        aggregate = _aggregate({}, window_days=int(args.window_days))
        gate = _promotion_status(
            aggregate,
            target_monthly=float(args.monthly_target),
            max_dd_target=float(args.max_dd_target),
            window_days=int(args.window_days),
            min_window_days=int(args.min_window_days),
            decision_lag=int(args.decision_lag),
            min_decision_lag=int(args.min_decision_lag),
            fee_rate=fee_rate_for_gate,
            min_fee_rate=min_fee_rate,
            short_borrow_apr=short_borrow_apr_for_gate,
            min_short_borrow_apr=min_short_borrow_apr,
            max_leverage=max_leverage_for_gate,
            max_leverage_target=max_leverage_target,
            execution_granularity=str(args.execution_granularity),
            require_hourly_intrabar=not bool(args.allow_daily_promotion),
            slippage_bps=[],
            min_max_slippage_bps=int(args.min_max_slippage_bps),
            required_slippage_bps=required_slippages,
            hourly_fill_buffer_bps=hourly_fill_buffer_for_gate,
            min_hourly_fill_buffer_bps=min_hourly_fill_buffer_bps,
            hourly_max_hold_hours=hourly_max_hold_for_gate,
            max_hourly_hold_hours=max_hourly_hold_hours_target,
            max_negative_windows=int(args.max_negative_windows),
            min_completed_windows=int(min_completed_windows),
        )
        gate["passed"] = False
        status = str(result.get("status"))
        reason = str(result.get("reason", "<no reason>"))
        gate["failures"] = [
            *list(gate.get("failures", [])),
            f"evaluator_status {status}: {reason}",
        ]
        md = (
            f"# 100d unseen-data eval — `{ckpt.name}`\n\n"
            f"- checkpoint_sha256: `{checkpoint_sha256}`\n"
            f"- **status**: EVALUATOR_NON_OK  ({status}: {reason})\n"
            "- promotion_gate: FAIL\n"
            f"- execution_granularity: {args.execution_granularity}\n"
            f"- backend: {result.get('backend', 'unknown')}\n"
        )
        (out_dir / f"{ckpt.stem}_eval100d.md").write_text(md)
        (out_dir / f"{ckpt.stem}_eval100d.json").write_text(json.dumps({
            "checkpoint": str(ckpt),
            "checkpoint_sha256": checkpoint_sha256,
            "val_data": str(val),
            "n_windows": int(args.n_windows),
            "window_days": int(args.window_days),
            "slippage_bps": slippages,
            "monthly_target": float(args.monthly_target),
            "max_dd_target": float(args.max_dd_target),
            "min_window_days": int(args.min_window_days),
            "decision_lag": int(args.decision_lag),
            "min_decision_lag": int(args.min_decision_lag),
            "fee_rate": fee_rate_for_gate,
            "min_fee_rate": min_fee_rate,
            "short_borrow_apr": short_borrow_apr_for_gate,
            "min_short_borrow_apr": min_short_borrow_apr,
            "max_leverage": max_leverage_for_gate,
            "max_leverage_target": max_leverage_target,
            "execution_granularity": str(args.execution_granularity),
            "require_hourly_intrabar": not bool(args.allow_daily_promotion),
            "min_max_slippage_bps": int(args.min_max_slippage_bps),
            "required_slippage_bps": required_slippages,
            "hourly_fill_buffer_bps": hourly_fill_buffer_for_gate,
            "min_hourly_fill_buffer_bps": min_hourly_fill_buffer_bps,
            "hourly_max_hold_hours": hourly_max_hold_for_gate,
            "max_hourly_hold_hours": max_hourly_hold_hours_target,
            "max_negative_windows": int(args.max_negative_windows),
            "min_completed_windows": int(min_completed_windows),
            "raw": result,
            "aggregate": aggregate,
            "promotion_gate": gate,
        }, indent=2, default=str))
        return 1
    if result.get("status") == "failed_fast":
        # Still emit JSON + a short MD so the leaderboard can record the dud
        # without spending more time. No videos rendered downstream either —
        # the same-backend path already skips them on failed_fast.
        out_dir = Path(args.out_dir) if args.out_dir else ckpt.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        aggregate = _aggregate(result.get("by_slippage", {}), window_days=int(args.window_days))
        gate = _promotion_status(
            aggregate,
            target_monthly=float(args.monthly_target),
            max_dd_target=float(args.max_dd_target),
            window_days=int(args.window_days),
            min_window_days=int(args.min_window_days),
            decision_lag=int(args.decision_lag),
            min_decision_lag=int(args.min_decision_lag),
            fee_rate=fee_rate_for_gate,
            min_fee_rate=min_fee_rate,
            short_borrow_apr=short_borrow_apr_for_gate,
            min_short_borrow_apr=min_short_borrow_apr,
            max_leverage=max_leverage_for_gate,
            max_leverage_target=max_leverage_target,
            execution_granularity=str(args.execution_granularity),
            require_hourly_intrabar=not bool(args.allow_daily_promotion),
            slippage_bps=_evaluated_slippage_bps(aggregate),
            min_max_slippage_bps=int(args.min_max_slippage_bps),
            required_slippage_bps=required_slippages,
            hourly_fill_buffer_bps=hourly_fill_buffer_for_gate,
            min_hourly_fill_buffer_bps=min_hourly_fill_buffer_bps,
            hourly_max_hold_hours=hourly_max_hold_for_gate,
            max_hourly_hold_hours=max_hourly_hold_hours_target,
            max_negative_windows=int(args.max_negative_windows),
            min_completed_windows=int(min_completed_windows),
        )
        gate["passed"] = False
        gate["failures"] = [
            *list(gate.get("failures", [])),
            f"failed_fast: {result.get('failed_reason', '<no reason>')}",
        ]
        bail_md = (
            f"# 100d unseen-data eval — `{ckpt.name}`\n\n"
            f"- checkpoint_sha256: `{checkpoint_sha256}`\n"
            f"- **status**: FAILED_FAST  ({result.get('failed_reason', '<no reason>')})\n"
            f"- promotion_gate: FAIL\n"
            f"- backend: {result.get('backend', 'pufferlib_market')}\n"
        )
        (out_dir / f"{ckpt.stem}_eval100d.md").write_text(bail_md)
        (out_dir / f"{ckpt.stem}_eval100d.json").write_text(json.dumps({
            "checkpoint": str(ckpt),
            "checkpoint_sha256": checkpoint_sha256,
            "val_data": str(val), "raw": result,
            "aggregate": aggregate,
            "promotion_gate": gate,
            "monthly_target": float(args.monthly_target),
            "max_dd_target": float(args.max_dd_target),
            "min_window_days": int(args.min_window_days),
            "decision_lag": int(args.decision_lag),
            "min_decision_lag": int(args.min_decision_lag),
            "fee_rate": fee_rate_for_gate,
            "min_fee_rate": min_fee_rate,
            "short_borrow_apr": short_borrow_apr_for_gate,
            "min_short_borrow_apr": min_short_borrow_apr,
            "max_leverage": max_leverage_for_gate,
            "max_leverage_target": max_leverage_target,
            "execution_granularity": str(args.execution_granularity),
            "require_hourly_intrabar": not bool(args.allow_daily_promotion),
            "min_max_slippage_bps": int(args.min_max_slippage_bps),
            "required_slippage_bps": required_slippages,
            "hourly_fill_buffer_bps": hourly_fill_buffer_for_gate,
            "min_hourly_fill_buffer_bps": min_hourly_fill_buffer_bps,
            "hourly_max_hold_hours": hourly_max_hold_for_gate,
            "max_hourly_hold_hours": max_hourly_hold_hours_target,
            "max_negative_windows": int(args.max_negative_windows),
            "min_completed_windows": int(min_completed_windows),
            "n_windows": int(args.n_windows), "window_days": int(args.window_days),
            "slippage_bps": slippages,
        }, indent=2, default=str))
        print(bail_md)
        return 3

    by_slip = result.get("by_slippage", {})
    aggregate = _aggregate(by_slip, window_days=int(args.window_days))
    out_dir = Path(args.out_dir) if args.out_dir else ckpt.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    gate = _promotion_status(
        aggregate,
        target_monthly=float(args.monthly_target),
        max_dd_target=float(args.max_dd_target),
        window_days=int(args.window_days),
        min_window_days=int(args.min_window_days),
        decision_lag=int(args.decision_lag),
        min_decision_lag=int(args.min_decision_lag),
        fee_rate=fee_rate_for_gate,
        min_fee_rate=min_fee_rate,
        short_borrow_apr=short_borrow_apr_for_gate,
        min_short_borrow_apr=min_short_borrow_apr,
        max_leverage=max_leverage_for_gate,
        max_leverage_target=max_leverage_target,
        execution_granularity=str(args.execution_granularity),
        require_hourly_intrabar=not bool(args.allow_daily_promotion),
        slippage_bps=_evaluated_slippage_bps(aggregate),
        min_max_slippage_bps=int(args.min_max_slippage_bps),
        required_slippage_bps=required_slippages,
        hourly_fill_buffer_bps=hourly_fill_buffer_for_gate,
        min_hourly_fill_buffer_bps=min_hourly_fill_buffer_bps,
        hourly_max_hold_hours=hourly_max_hold_for_gate,
        max_hourly_hold_hours=max_hourly_hold_hours_target,
        max_negative_windows=int(args.max_negative_windows),
        min_completed_windows=int(min_completed_windows),
    )
    full = {
        "checkpoint": str(ckpt),
        "checkpoint_sha256": checkpoint_sha256,
        "val_data": str(val),
        "n_windows": int(args.n_windows),
        "window_days": int(args.window_days),
        "slippage_bps": slippages,
        "monthly_target": float(args.monthly_target),
        "max_dd_target": float(args.max_dd_target),
        "min_window_days": int(args.min_window_days),
        "decision_lag": int(args.decision_lag),
        "min_decision_lag": int(args.min_decision_lag),
        "fee_rate": fee_rate_for_gate,
        "min_fee_rate": min_fee_rate,
        "short_borrow_apr": short_borrow_apr_for_gate,
        "min_short_borrow_apr": min_short_borrow_apr,
        "max_leverage": max_leverage_for_gate,
        "max_leverage_target": max_leverage_target,
        "execution_granularity": str(args.execution_granularity),
        "require_hourly_intrabar": not bool(args.allow_daily_promotion),
        "min_max_slippage_bps": int(args.min_max_slippage_bps),
        "required_slippage_bps": required_slippages,
        "hourly_fill_buffer_bps": hourly_fill_buffer_for_gate,
        "min_hourly_fill_buffer_bps": min_hourly_fill_buffer_bps,
        "hourly_max_hold_hours": hourly_max_hold_for_gate,
        "max_hourly_hold_hours": max_hourly_hold_hours_target,
        "max_negative_windows": int(args.max_negative_windows),
        "min_completed_windows": int(min_completed_windows),
        "raw": result,
        "aggregate": aggregate,
        "promotion_gate": gate,
    }
    (out_dir / f"{ckpt.stem}_eval100d.json").write_text(json.dumps(full, indent=2, default=str))
    md = _render_md(
        ckpt=ckpt, checkpoint_sha256=checkpoint_sha256, aggregate=aggregate,
        window_days=int(args.window_days), n_windows=int(args.n_windows),
        target_monthly=float(args.monthly_target),
        max_dd_target=float(args.max_dd_target),
        eval_result=result,
        min_window_days=int(args.min_window_days),
        decision_lag=int(args.decision_lag),
        min_decision_lag=int(args.min_decision_lag),
        fee_rate=fee_rate_for_gate,
        min_fee_rate=min_fee_rate,
        short_borrow_apr=short_borrow_apr_for_gate,
        min_short_borrow_apr=min_short_borrow_apr,
        max_leverage=max_leverage_for_gate,
        max_leverage_target=max_leverage_target,
        execution_granularity=str(args.execution_granularity),
        require_hourly_intrabar=not bool(args.allow_daily_promotion),
        slippage_bps=slippages,
        min_max_slippage_bps=int(args.min_max_slippage_bps),
        required_slippage_bps=required_slippages,
        hourly_fill_buffer_bps=hourly_fill_buffer_for_gate,
        min_hourly_fill_buffer_bps=min_hourly_fill_buffer_bps,
        hourly_max_hold_hours=hourly_max_hold_for_gate,
        max_hourly_hold_hours=max_hourly_hold_hours_target,
        max_negative_windows=int(args.max_negative_windows),
        min_completed_windows=int(min_completed_windows),
    )
    (out_dir / f"{ckpt.stem}_eval100d.md").write_text(md)
    print(md)

    return 0 if bool(gate["passed"]) else 3


if __name__ == "__main__":
    raise SystemExit(main())
