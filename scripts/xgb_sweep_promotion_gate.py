#!/usr/bin/env python3
"""Fail-fast promotion gate for xgbnew sweep artifacts.

This is intentionally narrower than a generic leaderboard sorter: production
promotion should fail closed unless a cell clears the current realism targets
on enough unseen calendar time and the requested stress fee regime.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import re
import shlex
import sys
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class GateConfig:
    fee_regime: str
    min_median_monthly_pct: float
    max_worst_dd_pct: float
    max_neg_windows: int
    min_oos_days: int
    min_windows: int
    min_p10_monthly_pct: float | None
    min_fill_buffer_bps: float | None = None
    reject_fail_fast: bool = True
    reject_skip_stress: bool = True
    require_complete: bool = True
    require_expected_windows: bool = True
    live_config: dict[str, Any] | None = None
    live_model_paths: tuple[str, ...] | None = None
    live_symbols_file: str | None = None
    live_data_root: str | None = None
    live_spy_csv: str | None = None
    live_blend_mode: str | None = None


@dataclass(frozen=True)
class GateResult:
    path: str
    passed: bool
    reason: str
    oos_days: int
    n_cells_considered: int
    best_cell: dict[str, Any] | None


LIVE_CONFIG_DEFAULTS: dict[str, Any] = {
    "top_n": 2,
    "min_picks": 0,
    "opportunistic_watch_n": 0,
    "opportunistic_entry_discount_bps": 0.0,
    "leverage": 0.25,
    "min_score": 0.0,
    "hold_through": False,
    "inference_min_dolvol": 5e6,
    "inference_max_spread_bps": 30.0,
    "inference_min_vol_20d": 0.0,
    "inference_max_vol_20d": 0.0,
    "max_ret_20d_rank_pct": 1.0,
    "min_ret_5d_rank_pct": 0.0,
    "regime_cs_iqr_max": 0.0,
    "regime_cs_skew_min": -1e9,
    "skip_prob": 0.0,
    "skip_seed": 0,
    "regime_gate_window": 0,
    "vol_target_ann": 0.0,
    "inv_vol_target_ann": 0.0,
    "inv_vol_floor": 0.05,
    "inv_vol_cap": 3.0,
    "no_picks_fallback_symbol": "",
    "no_picks_fallback_alloc_scale": 0.0,
    "conviction_scaled_alloc": False,
    "conviction_alloc_low": 0.55,
    "conviction_alloc_high": 0.85,
    "allocation_mode": "equal",
    "allocation_temp": 1.0,
    "score_uncertainty_penalty": 0.0,
}


_MISSING = object()
_ALLOCATION_MODES = {"equal", "score_norm", "softmax"}
_UNMODELED_LIVE_SIDECAR_FLAGS = ("--crypto-weekend", "--eod-deleverage")
_SHELL_ASSIGN_RE = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
_SHELL_UNSET_RE = re.compile(r"^unset\s+(.+)$")
_SHELL_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _parse_date(value: Any) -> date | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def infer_oos_days(payload: dict[str, Any]) -> int:
    start = _parse_date(payload.get("oos_start"))
    end = _parse_date(payload.get("oos_end"))
    if start is None or end is None or end < start:
        return 0
    return (end - start).days + 1


def _metric(cell: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(cell.get(key, default))
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _required_metric(cell: dict[str, Any], key: str) -> tuple[float | None, str | None]:
    if key not in cell:
        return None, f"{key} missing"
    try:
        value = float(cell[key])
    except (TypeError, ValueError):
        return None, f"{key} non-finite"
    if not math.isfinite(value):
        return None, f"{key} non-finite"
    return value, None


def _required_int_metric(cell: dict[str, Any], key: str) -> tuple[int | None, str | None]:
    if key not in cell:
        return None, f"{key} missing"
    value = _parse_int_like(cell[key])
    if value is None or value < 0:
        return None, f"{key} invalid"
    return value, None


def _flag_raw_value(tokens: list[str], flag: str) -> Any:
    prefix = flag + "="
    value: Any = _MISSING
    for idx, token in enumerate(tokens):
        if token == flag and idx + 1 < len(tokens):
            value = tokens[idx + 1]
            continue
        if token == flag:
            raise ValueError(f"{flag} requires a value")
        if token.startswith(prefix):
            value = token.split("=", 1)[1]
    return value


def _flag_value(tokens: list[str], flag: str, default: Any = None) -> Any:
    value = _flag_raw_value(tokens, flag)
    if value is _MISSING:
        return default
    return value


def _flag_float(tokens: list[str], flag: str, default: float) -> float:
    value = _flag_raw_value(tokens, flag)
    if value is _MISSING:
        return float(default)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{flag} must be a finite float, got {value!r}") from None
    if not math.isfinite(parsed):
        raise ValueError(f"{flag} must be a finite float, got {value!r}")
    return parsed


def _flag_nonnegative_float(tokens: list[str], flag: str, default: float) -> float:
    parsed = _flag_float(tokens, flag, default)
    if parsed < 0.0:
        raise ValueError(f"{flag} must be >= 0, got {parsed!r}")
    return parsed


def _flag_positive_float(tokens: list[str], flag: str, default: float) -> float:
    parsed = _flag_float(tokens, flag, default)
    if parsed <= 0.0:
        raise ValueError(f"{flag} must be > 0, got {parsed!r}")
    return parsed


def _flag_at_least_float(tokens: list[str], flag: str, default: float, minimum: float) -> float:
    parsed = _flag_float(tokens, flag, default)
    if parsed < float(minimum):
        raise ValueError(f"{flag} must be >= {float(minimum):g}, got {parsed!r}")
    return parsed


def _flag_bounded_float(
    tokens: list[str],
    flag: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    parsed = _flag_float(tokens, flag, default)
    if not float(minimum) <= parsed <= float(maximum):
        raise ValueError(
            f"{flag} must be between {float(minimum):g} and {float(maximum):g}, "
            f"got {parsed!r}"
        )
    return parsed


def _flag_int(tokens: list[str], flag: str, default: int) -> int:
    value = _flag_raw_value(tokens, flag)
    if value is _MISSING:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{flag} must be an integer, got {value!r}")


def _flag_positive_int(tokens: list[str], flag: str, default: int) -> int:
    parsed = _flag_int(tokens, flag, default)
    if parsed < 1:
        raise ValueError(f"{flag} must be >= 1, got {parsed!r}")
    return parsed


def _flag_nonnegative_int(tokens: list[str], flag: str, default: int) -> int:
    parsed = _flag_int(tokens, flag, default)
    if parsed < 0:
        raise ValueError(f"{flag} must be >= 0, got {parsed!r}")
    return parsed


def _flag_choice(tokens: list[str], flag: str, default: str, choices: set[str]) -> str:
    value = str(_flag_value(tokens, flag, default) or default).lower()
    if value not in choices:
        allowed = "|".join(sorted(choices))
        raise ValueError(f"{flag} must be one of {allowed}, got {value!r}")
    return value


def _read_launch_exec_command(path: Path) -> str:
    lines = path.read_text().splitlines()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("exec "):
            continue
        parts = [stripped[len("exec "):].rstrip("\\").strip()]
        while stripped.endswith("\\") and idx + 1 < len(lines):
            idx += 1
            stripped = lines[idx].strip()
            parts.append(stripped.rstrip("\\").strip())
        return " ".join(part for part in parts if part)
    raise ValueError(f"{path}: no exec command found")


def _strip_inline_shell_comment(line: str) -> str:
    in_single = False
    in_double = False
    out: list[str] = []
    for char in line:
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            break
        out.append(char)
    return "".join(out).strip()


def _expand_shell_vars(value: str, variables: dict[str, str]) -> str:
    expanded = value
    for _ in range(8):
        next_value = _SHELL_VAR_RE.sub(
            lambda match: variables.get(match.group(1) or match.group(2), ""),
            expanded,
        )
        if next_value == expanded:
            break
        expanded = next_value
    return expanded


def _read_launch_assignments(path: Path) -> dict[str, str]:
    variables: dict[str, str] = {}
    for line in path.read_text().splitlines():
        stripped = _strip_inline_shell_comment(line)
        if not stripped or stripped.startswith(("if ", "fi", "then", "else", "exec ")):
            continue
        unset_match = _SHELL_UNSET_RE.match(stripped)
        if unset_match:
            try:
                names = shlex.split(unset_match.group(1), posix=True)
            except ValueError:
                continue
            for name in names:
                variables.pop(name, None)
            continue
        match = _SHELL_ASSIGN_RE.match(stripped)
        if not match:
            continue
        name, raw_value = match.groups()
        try:
            parts = shlex.split(raw_value, posix=True)
        except ValueError:
            continue
        if len(parts) != 1:
            continue
        variables[name] = _expand_shell_vars(parts[0], variables)
    return variables


def _read_launch_exec_command_expanded(path: Path) -> str:
    return _expand_shell_vars(_read_launch_exec_command(path), _read_launch_assignments(path))


def _read_launch_tokens(path: Path) -> list[str]:
    return shlex.split(_read_launch_exec_command_expanded(path))


def _normal_repo_path(value: Any) -> str:
    path = Path(str(value))
    if not path.is_absolute():
        path = REPO_ROOT / path
    return str(path.resolve(strict=False))


def _normal_model_path(value: Any) -> str:
    return _normal_repo_path(value)


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _live_manifest_path(model_paths: tuple[str, ...]) -> str | None:
    parents = {Path(path).parent for path in model_paths}
    if len(parents) != 1:
        return None
    manifest = next(iter(parents)) / "alltrain_ensemble.json"
    return str(manifest.resolve(strict=False))


def extract_model_paths_from_launch(path: Path) -> tuple[str, ...]:
    tokens = _read_launch_tokens(path)
    raw_values: list[str] = []
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token == "--model-paths":
            if idx + 1 >= len(tokens):
                raise ValueError("--model-paths requires a value")
            raw_values.extend(tokens[idx + 1].split(","))
            idx += 2
            continue
        if token.startswith("--model-paths="):
            raw_values.extend(token.split("=", 1)[1].split(","))
        idx += 1
    return tuple(_normal_model_path(value.strip()) for value in raw_values if value.strip())


def extract_symbols_file_from_launch(path: Path) -> str:
    tokens = _read_launch_tokens(path)
    value = _flag_value(tokens, "--symbols-file", "symbol_lists/stocks_wide_1000_v1.txt")
    if value is None or not str(value).strip():
        raise ValueError("--symbols-file must not be empty")
    return _normal_repo_path(str(value).strip())


def extract_data_root_from_launch(path: Path) -> str:
    tokens = _read_launch_tokens(path)
    value = _flag_value(tokens, "--data-root", "trainingdata")
    if value is None or not str(value).strip():
        raise ValueError("--data-root must not be empty")
    return _normal_repo_path(str(value).strip())


def extract_spy_csv_from_launch(path: Path) -> str:
    tokens = _read_launch_tokens(path)
    value = _flag_value(tokens, "--spy-csv", None)
    if value is not None:
        if not str(value).strip():
            raise ValueError("--spy-csv must not be empty")
        return _normal_repo_path(str(value).strip())
    data_root = extract_data_root_from_launch(path)
    return str((Path(data_root) / "SPY.csv").resolve(strict=False))


def unmodeled_live_sidecars_from_launch(path: Path) -> list[str]:
    tokens = _read_launch_tokens(path)
    return [flag for flag in _UNMODELED_LIVE_SIDECAR_FLAGS if flag in tokens]


def _tokens_invoke_xgb_live_trader(tokens: list[str]) -> bool:
    for idx, token in enumerate(tokens):
        if token == "-m" and idx + 1 < len(tokens) and tokens[idx + 1] == "xgbnew.live_trader":
            return True
        token_path = token.replace("\\", "/")
        if token_path.endswith("/xgbnew/live_trader.py") or token_path == "xgbnew/live_trader.py":
            return True
    return False


def validate_live_launch_mode(path: Path) -> str | None:
    tokens = _read_launch_tokens(path)
    if not _tokens_invoke_xgb_live_trader(tokens):
        return (
            "launch-script does not invoke xgbnew.live_trader; "
            "cannot audit live trading behavior"
        )
    if "--dry-run" in tokens:
        return "launch-script enables --dry-run; production promotion requires live orders"
    if "--live" not in tokens:
        return "launch-script omits --live; production promotion requires live mode"
    return None


def extract_live_config_from_launch(path: Path) -> dict[str, Any]:
    tokens = _read_launch_tokens(path)
    top_n = _flag_positive_int(tokens, "--top-n", int(LIVE_CONFIG_DEFAULTS["top_n"]))
    min_picks = _flag_nonnegative_int(
        tokens,
        "--min-picks",
        int(LIVE_CONFIG_DEFAULTS["min_picks"]),
    )
    if min_picks > top_n:
        raise ValueError(f"--min-picks must be <= --top-n, got {min_picks!r} > {top_n!r}")
    conviction_alloc_low = _flag_float(
        tokens, "--conviction-alloc-low", float(LIVE_CONFIG_DEFAULTS["conviction_alloc_low"])
    )
    conviction_alloc_high = _flag_float(
        tokens, "--conviction-alloc-high", float(LIVE_CONFIG_DEFAULTS["conviction_alloc_high"])
    )
    if conviction_alloc_high <= conviction_alloc_low:
        raise ValueError("--conviction-alloc-high must be > --conviction-alloc-low")
    return {
        "top_n": top_n,
        "min_picks": min_picks,
        "opportunistic_watch_n": int(LIVE_CONFIG_DEFAULTS["opportunistic_watch_n"]),
        "opportunistic_entry_discount_bps": float(
            LIVE_CONFIG_DEFAULTS["opportunistic_entry_discount_bps"]
        ),
        "leverage": _flag_positive_float(
            tokens, "--allocation", float(LIVE_CONFIG_DEFAULTS["leverage"])
        ),
        "min_score": _flag_bounded_float(
            tokens,
            "--min-score",
            float(LIVE_CONFIG_DEFAULTS["min_score"]),
            minimum=0.0,
            maximum=1.0,
        ),
        "hold_through": "--hold-through" in tokens,
        "inference_min_dolvol": _flag_nonnegative_float(
            tokens, "--min-dollar-vol", float(LIVE_CONFIG_DEFAULTS["inference_min_dolvol"])
        ),
        "inference_max_spread_bps": _flag_nonnegative_float(
            tokens, "--max-spread-bps", float(LIVE_CONFIG_DEFAULTS["inference_max_spread_bps"])
        ),
        "inference_min_vol_20d": _flag_nonnegative_float(
            tokens, "--min-vol-20d", float(LIVE_CONFIG_DEFAULTS["inference_min_vol_20d"])
        ),
        "inference_max_vol_20d": _flag_nonnegative_float(
            tokens, "--max-vol-20d", float(LIVE_CONFIG_DEFAULTS["inference_max_vol_20d"])
        ),
        "max_ret_20d_rank_pct": _flag_bounded_float(
            tokens,
            "--max-ret-20d-rank-pct",
            float(LIVE_CONFIG_DEFAULTS["max_ret_20d_rank_pct"]),
            minimum=0.0,
            maximum=1.0,
        ),
        "min_ret_5d_rank_pct": _flag_bounded_float(
            tokens,
            "--min-ret-5d-rank-pct",
            float(LIVE_CONFIG_DEFAULTS["min_ret_5d_rank_pct"]),
            minimum=0.0,
            maximum=1.0,
        ),
        "regime_cs_iqr_max": _flag_nonnegative_float(
            tokens, "--regime-cs-iqr-max", float(LIVE_CONFIG_DEFAULTS["regime_cs_iqr_max"])
        ),
        "regime_cs_skew_min": _flag_float(
            tokens, "--regime-cs-skew-min", float(LIVE_CONFIG_DEFAULTS["regime_cs_skew_min"])
        ),
        "skip_prob": float(LIVE_CONFIG_DEFAULTS["skip_prob"]),
        "skip_seed": int(LIVE_CONFIG_DEFAULTS["skip_seed"]),
        "regime_gate_window": _flag_nonnegative_int(
            tokens,
            "--regime-gate-window",
            int(LIVE_CONFIG_DEFAULTS["regime_gate_window"]),
        ),
        "vol_target_ann": _flag_nonnegative_float(
            tokens, "--vol-target-ann", float(LIVE_CONFIG_DEFAULTS["vol_target_ann"])
        ),
        "inv_vol_target_ann": _flag_nonnegative_float(
            tokens,
            "--inv-vol-target-ann",
            float(LIVE_CONFIG_DEFAULTS["inv_vol_target_ann"]),
        ),
        "inv_vol_floor": _flag_positive_float(
            tokens, "--inv-vol-floor", float(LIVE_CONFIG_DEFAULTS["inv_vol_floor"])
        ),
        "inv_vol_cap": _flag_at_least_float(
            tokens,
            "--inv-vol-cap",
            float(LIVE_CONFIG_DEFAULTS["inv_vol_cap"]),
            1.0,
        ),
        "no_picks_fallback_symbol": str(_flag_value(tokens, "--no-picks-fallback", "") or "").upper(),
        "no_picks_fallback_alloc_scale": (
            _flag_nonnegative_float(tokens, "--no-picks-fallback-alloc", 0.5)
            if _flag_value(tokens, "--no-picks-fallback", "") else 0.0
        ),
        "conviction_scaled_alloc": "--conviction-scaled-alloc" in tokens,
        "conviction_alloc_low": conviction_alloc_low,
        "conviction_alloc_high": conviction_alloc_high,
        "allocation_mode": _flag_choice(
            tokens,
            "--allocation-mode",
            str(LIVE_CONFIG_DEFAULTS["allocation_mode"]),
            _ALLOCATION_MODES,
        ),
        "allocation_temp": _flag_positive_float(
            tokens, "--allocation-temp", float(LIVE_CONFIG_DEFAULTS["allocation_temp"])
        ),
        "score_uncertainty_penalty": _flag_nonnegative_float(
            tokens,
            "--score-uncertainty-penalty",
            float(LIVE_CONFIG_DEFAULTS["score_uncertainty_penalty"]),
        ),
    }


def _same_float(a: Any, b: Any, *, tol: float = 1e-9) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def _parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        if value in (0, 1):
            return bool(value)
        return None
    if isinstance(value, float):
        if value in (0.0, 1.0) and math.isfinite(value):
            return bool(value)
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    return None


def _parse_int_like(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not number.is_integer():
        return None
    return int(number)


def cell_matches_live_config(cell: dict[str, Any], live_config: dict[str, Any]) -> bool:
    for key, expected in live_config.items():
        actual = cell.get(key, LIVE_CONFIG_DEFAULTS.get(key, ""))
        if isinstance(expected, bool):
            actual_bool = _parse_bool(actual)
            if actual_bool is None or actual_bool != expected:
                return False
            continue
        if isinstance(expected, int) and not isinstance(expected, bool):
            actual_int = _parse_int_like(actual)
            if actual_int is None or actual_int != expected:
                return False
            continue
        if isinstance(expected, float):
            if not _same_float(actual, expected):
                return False
            continue
        if str(actual) != str(expected):
            return False
    return True


def _passes_cell(cell: dict[str, Any], config: GateConfig) -> tuple[bool, str]:
    fail_fast_triggered = _parse_bool(cell.get("fail_fast_triggered", False))
    if fail_fast_triggered is None:
        return False, "fail_fast_triggered invalid"
    if config.reject_fail_fast and fail_fast_triggered:
        reason = str(cell.get("fail_fast_reason", "") or "cell was pruned")
        return False, f"fail_fast_triggered: {reason}"

    if config.reject_skip_stress and "skip_prob" in cell:
        skip_prob, reason = _required_metric(cell, "skip_prob")
        if reason is not None or skip_prob is None:
            return False, reason or "skip_prob non-finite"
        if skip_prob != 0.0:
            return False, f"skip_prob {skip_prob:.4f} != 0.0"

    if config.min_fill_buffer_bps is not None:
        fill_buffer, reason = _required_metric(cell, "fill_buffer_bps")
        if reason is not None or fill_buffer is None:
            return False, reason or "fill_buffer_bps non-finite"
        if fill_buffer < float(config.min_fill_buffer_bps):
            return False, (
                f"fill_buffer_bps {fill_buffer:.2f} < "
                f"{float(config.min_fill_buffer_bps):.2f}"
            )

    median, reason = _required_metric(cell, "median_monthly_pct")
    if reason is not None or median is None:
        return False, reason or "median_monthly_pct non-finite"
    if median < config.min_median_monthly_pct:
        return False, f"median_monthly_pct {median:.2f} < {config.min_median_monthly_pct:.2f}"

    worst_dd, reason = _required_metric(cell, "worst_dd_pct")
    if reason is not None or worst_dd is None:
        return False, reason or "worst_dd_pct non-finite"
    if worst_dd > config.max_worst_dd_pct:
        return False, f"worst_dd_pct {worst_dd:.2f} > {config.max_worst_dd_pct:.2f}"

    n_neg, reason = _required_int_metric(cell, "n_neg")
    if reason is not None or n_neg is None:
        return False, reason or "n_neg invalid"
    if n_neg > config.max_neg_windows:
        return False, f"n_neg {n_neg} > {config.max_neg_windows}"

    n_windows, reason = _required_int_metric(cell, "n_windows")
    if reason is not None or n_windows is None:
        return False, reason or "n_windows invalid"
    required_windows = int(config.min_windows)
    if config.require_expected_windows and "expected_n_windows" in cell:
        expected_windows, reason = _required_int_metric(cell, "expected_n_windows")
        if reason is not None or expected_windows is None:
            return False, reason or "expected_n_windows invalid"
        required_windows = max(required_windows, expected_windows)
    if n_windows < required_windows:
        return False, f"n_windows {n_windows} < {required_windows}"

    if config.min_p10_monthly_pct is not None:
        p10, reason = _required_metric(cell, "p10_monthly_pct")
        if reason is not None or p10 is None:
            return False, reason or "p10_monthly_pct non-finite"
        if p10 < config.min_p10_monthly_pct:
            return False, f"p10_monthly_pct {p10:.2f} < {config.min_p10_monthly_pct:.2f}"

    return True, "passed"


def _default_min_fill_buffer_bps(fee_regime: str) -> float | None:
    if str(fee_regime) == "stress36x":
        return 15.0
    return None


def evaluate_sweep(path: Path, config: GateConfig) -> GateResult:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected top-level JSON object")
    cells = payload.get("cells")
    if not isinstance(cells, list):
        raise ValueError(f"{path}: expected top-level 'cells' list")

    oos_days = infer_oos_days(payload)
    if config.require_complete and payload.get("complete") is not True:
        state = payload.get("complete", "<missing>")
        return GateResult(
            path=str(path),
            passed=False,
            reason=f"sweep complete flag is {state!r}",
            oos_days=oos_days,
            n_cells_considered=0,
            best_cell=None,
        )
    if oos_days < config.min_oos_days:
        return GateResult(
            path=str(path),
            passed=False,
            reason=f"oos_days {oos_days} < {config.min_oos_days}",
            oos_days=oos_days,
            n_cells_considered=0,
            best_cell=None,
        )
    if config.live_model_paths is not None:
        raw_model_paths = payload.get("model_paths")
        if not isinstance(raw_model_paths, list) or not raw_model_paths:
            return GateResult(
                path=str(path),
                passed=False,
                reason="sweep model_paths missing",
                oos_days=oos_days,
                n_cells_considered=0,
                best_cell=None,
            )
        sweep_model_paths = tuple(_normal_model_path(value) for value in raw_model_paths)
        if sweep_model_paths != config.live_model_paths:
            return GateResult(
                path=str(path),
                passed=False,
                reason="launch model_paths do not match sweep model_paths",
                oos_days=oos_days,
                n_cells_considered=0,
                best_cell=None,
            )
        raw_model_sha256 = payload.get("model_sha256")
        if raw_model_sha256 is not None:
            if (
                not isinstance(raw_model_sha256, list)
                or len(raw_model_sha256) != len(config.live_model_paths)
                or not all(
                    isinstance(value, str)
                    and _SHA256_RE.fullmatch(value.strip().lower())
                    for value in raw_model_sha256
                )
            ):
                return GateResult(
                    path=str(path),
                    passed=False,
                    reason="sweep model_sha256 invalid",
                    oos_days=oos_days,
                    n_cells_considered=0,
                    best_cell=None,
                )
            try:
                live_model_sha256 = tuple(_file_sha256(path) for path in config.live_model_paths)
            except OSError as exc:
                return GateResult(
                    path=str(path),
                    passed=False,
                    reason=f"launch model hash read failed: {exc}",
                    oos_days=oos_days,
                    n_cells_considered=0,
                    best_cell=None,
                )
            sweep_model_sha256 = tuple(value.strip().lower() for value in raw_model_sha256)
            if sweep_model_sha256 != live_model_sha256:
                return GateResult(
                    path=str(path),
                    passed=False,
                    reason="launch model_sha256 do not match sweep model_sha256",
                    oos_days=oos_days,
                    n_cells_considered=0,
                    best_cell=None,
                )
        raw_manifest = payload.get("ensemble_manifest")
        if raw_manifest is not None:
            if not isinstance(raw_manifest, dict):
                return GateResult(
                    path=str(path),
                    passed=False,
                    reason="sweep ensemble_manifest invalid",
                    oos_days=oos_days,
                    n_cells_considered=0,
                    best_cell=None,
                )
            raw_manifest_path = raw_manifest.get("path")
            raw_manifest_sha256 = raw_manifest.get("sha256")
            if not isinstance(raw_manifest_path, str) or not isinstance(raw_manifest_sha256, str):
                return GateResult(
                    path=str(path),
                    passed=False,
                    reason="sweep ensemble_manifest invalid",
                    oos_days=oos_days,
                    n_cells_considered=0,
                    best_cell=None,
                )
            live_manifest_path = _live_manifest_path(config.live_model_paths)
            if live_manifest_path is None:
                return GateResult(
                    path=str(path),
                    passed=False,
                    reason="launch model paths do not share an ensemble manifest",
                    oos_days=oos_days,
                    n_cells_considered=0,
                    best_cell=None,
                )
            sweep_manifest_path = _normal_model_path(raw_manifest_path)
            if sweep_manifest_path != live_manifest_path:
                return GateResult(
                    path=str(path),
                    passed=False,
                    reason="launch ensemble_manifest path does not match sweep ensemble_manifest path",
                    oos_days=oos_days,
                    n_cells_considered=0,
                    best_cell=None,
                )
            if not _SHA256_RE.fullmatch(raw_manifest_sha256.strip().lower()):
                return GateResult(
                    path=str(path),
                    passed=False,
                    reason="sweep ensemble_manifest sha256 invalid",
                    oos_days=oos_days,
                    n_cells_considered=0,
                    best_cell=None,
                )
            try:
                live_manifest_sha256 = _file_sha256(live_manifest_path)
            except OSError as exc:
                return GateResult(
                    path=str(path),
                    passed=False,
                    reason=f"launch ensemble manifest hash read failed: {exc}",
                    oos_days=oos_days,
                    n_cells_considered=0,
                    best_cell=None,
                )
            if raw_manifest_sha256.strip().lower() != live_manifest_sha256:
                return GateResult(
                    path=str(path),
                    passed=False,
                    reason="launch ensemble_manifest sha256 does not match sweep ensemble_manifest sha256",
                    oos_days=oos_days,
                    n_cells_considered=0,
                    best_cell=None,
                )
    if config.live_symbols_file is not None:
        raw_symbols_file = payload.get("symbols_file")
        if not isinstance(raw_symbols_file, str) or not raw_symbols_file.strip():
            return GateResult(
                path=str(path),
                passed=False,
                reason="sweep symbols_file missing",
                oos_days=oos_days,
                n_cells_considered=0,
                best_cell=None,
            )
        sweep_symbols_file = _normal_repo_path(raw_symbols_file.strip())
        if sweep_symbols_file != config.live_symbols_file:
            return GateResult(
                path=str(path),
                passed=False,
                reason="launch symbols_file does not match sweep symbols_file",
                oos_days=oos_days,
                n_cells_considered=0,
                best_cell=None,
            )
    if config.live_data_root is not None:
        raw_data_root = payload.get("data_root")
        if not isinstance(raw_data_root, str) or not raw_data_root.strip():
            return GateResult(
                path=str(path),
                passed=False,
                reason="sweep data_root missing",
                oos_days=oos_days,
                n_cells_considered=0,
                best_cell=None,
            )
        sweep_data_root = _normal_repo_path(raw_data_root.strip())
        if sweep_data_root != config.live_data_root:
            return GateResult(
                path=str(path),
                passed=False,
                reason="launch data_root does not match sweep data_root",
                oos_days=oos_days,
                n_cells_considered=0,
                best_cell=None,
            )
    if (
        config.live_spy_csv is not None
        and config.live_config is not None
        and (
            float(config.live_config.get("vol_target_ann", 0.0) or 0.0) > 0.0
            or int(config.live_config.get("regime_gate_window", 0) or 0) > 0
        )
    ):
        raw_spy_csv = payload.get("spy_csv")
        if not isinstance(raw_spy_csv, str) or not raw_spy_csv.strip():
            return GateResult(
                path=str(path),
                passed=False,
                reason="sweep spy_csv missing",
                oos_days=oos_days,
                n_cells_considered=0,
                best_cell=None,
            )
        sweep_spy_csv = _normal_repo_path(raw_spy_csv.strip())
        if sweep_spy_csv != config.live_spy_csv:
            return GateResult(
                path=str(path),
                passed=False,
                reason="launch spy_csv does not match sweep spy_csv",
                oos_days=oos_days,
                n_cells_considered=0,
                best_cell=None,
            )
        raw_spy_sha256 = payload.get("spy_csv_sha256")
        if raw_spy_sha256 is not None:
            if (
                not isinstance(raw_spy_sha256, str)
                or not _SHA256_RE.fullmatch(raw_spy_sha256.strip().lower())
            ):
                return GateResult(
                    path=str(path),
                    passed=False,
                    reason="sweep spy_csv_sha256 invalid",
                    oos_days=oos_days,
                    n_cells_considered=0,
                    best_cell=None,
                )
            try:
                live_spy_sha256 = _file_sha256(config.live_spy_csv)
            except OSError as exc:
                return GateResult(
                    path=str(path),
                    passed=False,
                    reason=f"launch spy_csv hash read failed: {exc}",
                    oos_days=oos_days,
                    n_cells_considered=0,
                    best_cell=None,
                )
            if raw_spy_sha256.strip().lower() != live_spy_sha256:
                return GateResult(
                    path=str(path),
                    passed=False,
                    reason="launch spy_csv_sha256 does not match sweep spy_csv_sha256",
                    oos_days=oos_days,
                    n_cells_considered=0,
                    best_cell=None,
                )
    if config.live_blend_mode is not None:
        raw_blend_mode = payload.get("blend_mode")
        if not isinstance(raw_blend_mode, str) or not raw_blend_mode.strip():
            return GateResult(
                path=str(path),
                passed=False,
                reason="sweep blend_mode missing",
                oos_days=oos_days,
                n_cells_considered=0,
                best_cell=None,
            )
        sweep_blend_mode = raw_blend_mode.strip().lower()
        if sweep_blend_mode != config.live_blend_mode:
            return GateResult(
                path=str(path),
                passed=False,
                reason="launch blend_mode does not match sweep blend_mode",
                oos_days=oos_days,
                n_cells_considered=0,
                best_cell=None,
            )

    candidates = [
        cell for cell in cells
        if isinstance(cell, dict) and str(cell.get("fee_regime", "")) == config.fee_regime
    ]
    if config.live_config is not None:
        candidates = [
            cell for cell in candidates
            if cell_matches_live_config(cell, config.live_config)
        ]
    if not candidates:
        reason = f"no cells for fee_regime={config.fee_regime!r}"
        if config.live_config is not None:
            reason += " matching live_config"
        return GateResult(
            path=str(path),
            passed=False,
            reason=reason,
            oos_days=oos_days,
            n_cells_considered=0,
            best_cell=None,
        )

    candidates.sort(
        key=lambda c: (
            _metric(c, "median_monthly_pct"),
            _metric(c, "p10_monthly_pct"),
            -_metric(c, "worst_dd_pct"),
            -_metric(c, "n_neg"),
        ),
        reverse=True,
    )
    best = candidates[0]
    first_failure_reason = "no passing cells"
    for cell in candidates:
        ok, reason = _passes_cell(cell, config)
        if ok:
            return GateResult(
                path=str(path),
                passed=True,
                reason=reason,
                oos_days=oos_days,
                n_cells_considered=len(candidates),
                best_cell=cell,
            )
        if first_failure_reason == "no passing cells":
            first_failure_reason = reason
    return GateResult(
        path=str(path),
        passed=False,
        reason=first_failure_reason,
        oos_days=oos_days,
        n_cells_considered=len(candidates),
        best_cell=best,
    )


def _expand_paths(values: list[str]) -> list[Path]:
    out: list[Path] = []
    for value in values:
        matches = [Path(path) for path in sorted(glob.glob(value))] if any(ch in value for ch in "*?[") else []
        if matches:
            out.extend(path for path in matches if path.is_file())
        else:
            out.append(Path(value))
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_json", nargs="+", help="Sweep JSON path(s) or shell-style globs.")
    parser.add_argument("--fee-regime", default="stress36x")
    parser.add_argument("--min-median-monthly-pct", type=float, default=27.0)
    parser.add_argument("--max-worst-dd-pct", type=float, default=25.0)
    parser.add_argument("--max-neg-windows", type=int, default=0)
    parser.add_argument("--min-oos-days", type=int, default=100)
    parser.add_argument("--min-windows", type=int, default=1)
    parser.add_argument("--min-p10-monthly-pct", type=float, default=None)
    parser.add_argument(
        "--min-fill-buffer-bps",
        type=float,
        default=None,
        help="Minimum evaluated fill buffer for eligible cells. Default is "
             "15 for stress36x and disabled for other fee regimes; set "
             "negative to disable for legacy artifact inspection.",
    )
    parser.add_argument(
        "--allow-fail-fast-cells",
        action="store_true",
        help="Allow cells marked fail_fast_triggered to pass. Default rejects "
             "them because their metrics may be partial-window artifacts.",
    )
    parser.add_argument(
        "--allow-skip-stress-cells",
        action="store_true",
        help="Allow cells with skip_prob > 0 to pass. Default rejects them "
             "because missed-order stress is not intentional live behavior "
             "and can accidentally skip bad trades.",
    )
    parser.add_argument(
        "--allow-partial-sweep",
        action="store_true",
        help="Allow sweep JSONs with complete=false or missing complete flag. "
             "Default rejects partial checkpoints for production promotion.",
    )
    parser.add_argument(
        "--ignore-expected-windows",
        action="store_true",
        help="Ignore per-cell expected_n_windows from current sweep artifacts "
             "and use only --min-windows. Intended for legacy/research "
             "inspection, not production promotion.",
    )
    parser.add_argument(
        "--launch-script",
        type=Path,
        default=None,
        help="Optional xgb-daily live launch.sh. When set, only cells matching "
             "the actual live top_n/allocation/min-score/liquidity/vol/regime "
             "knobs are eligible.",
    )
    parser.add_argument(
        "--allow-unmodeled-live-sidecars",
        action="store_true",
        help="Allow launch scripts that enable embedded live sidecars such as "
             "--crypto-weekend or --eod-deleverage. Default rejects them "
             "because xgb sweep artifacts do not evaluate those side effects.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON summary.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.launch_script is not None:
            mode_reason = validate_live_launch_mode(args.launch_script)
            if mode_reason is not None:
                print(mode_reason, file=sys.stderr)
                return 3
            sidecars = unmodeled_live_sidecars_from_launch(args.launch_script)
            if sidecars and not bool(args.allow_unmodeled_live_sidecars):
                joined = ", ".join(sidecars)
                print(
                    "launch-script enables unmodeled live sidecars "
                    f"({joined}); rerun with --allow-unmodeled-live-sidecars "
                    "only for explicit research/ops inspection",
                    file=sys.stderr,
                )
                return 3
        live_config = (
            extract_live_config_from_launch(args.launch_script)
            if args.launch_script is not None else None
        )
        live_model_paths = None
        live_symbols_file = None
        live_data_root = None
        live_spy_csv = None
        live_blend_mode = None
        if args.launch_script is not None:
            live_symbols_file = extract_symbols_file_from_launch(args.launch_script)
            live_data_root = extract_data_root_from_launch(args.launch_script)
            live_spy_csv = extract_spy_csv_from_launch(args.launch_script)
            live_blend_mode = "mean"
            parsed_model_paths = extract_model_paths_from_launch(args.launch_script)
            if not parsed_model_paths:
                print(
                    "launch-script omits --model-paths; cannot prove live model "
                    "artifacts match the sweep artifact",
                    file=sys.stderr,
                )
                return 3
            live_model_paths = parsed_model_paths
    except ValueError as exc:
        print(f"launch-script parse error: {exc}", file=sys.stderr)
        return 2
    config = GateConfig(
        fee_regime=str(args.fee_regime),
        min_median_monthly_pct=float(args.min_median_monthly_pct),
        max_worst_dd_pct=float(args.max_worst_dd_pct),
        max_neg_windows=int(args.max_neg_windows),
        min_oos_days=int(args.min_oos_days),
        min_windows=int(args.min_windows),
        min_p10_monthly_pct=args.min_p10_monthly_pct,
        min_fill_buffer_bps=(
            _default_min_fill_buffer_bps(str(args.fee_regime))
            if args.min_fill_buffer_bps is None
            else None if float(args.min_fill_buffer_bps) < 0.0
            else float(args.min_fill_buffer_bps)
        ),
        reject_fail_fast=not bool(args.allow_fail_fast_cells),
        reject_skip_stress=not bool(args.allow_skip_stress_cells),
        require_complete=not bool(args.allow_partial_sweep),
        require_expected_windows=not bool(args.ignore_expected_windows),
        live_config=live_config,
        live_model_paths=live_model_paths,
        live_symbols_file=live_symbols_file,
        live_data_root=live_data_root,
        live_spy_csv=live_spy_csv,
        live_blend_mode=live_blend_mode,
    )
    paths = _expand_paths(list(args.sweep_json))
    results = [evaluate_sweep(path, config) for path in paths]
    passed = [result for result in results if result.passed]

    if args.json:
        print(json.dumps({
            "config": asdict(config),
            "passed": bool(passed),
            "results": [asdict(result) for result in results],
        }, indent=2, sort_keys=True))
    else:
        for result in results:
            best = result.best_cell or {}
            print(
                f"{'PASS' if result.passed else 'FAIL'} {result.path}: {result.reason}; "
                f"oos_days={result.oos_days} cells={result.n_cells_considered} "
                f"best_med={_metric(best, 'median_monthly_pct'):.2f}% "
                f"best_p10={_metric(best, 'p10_monthly_pct'):.2f}% "
                f"best_dd={_metric(best, 'worst_dd_pct'):.2f}% "
                f"neg={int(_metric(best, 'n_neg'))}/{int(_metric(best, 'n_windows'))}"
            )
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
