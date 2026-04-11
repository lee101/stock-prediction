from __future__ import annotations

from pathlib import Path

from src.binan.hybrid_cycle_trace import DEFAULT_TRACE_DIR


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROD_EVAL_DATA_PATH = REPO_ROOT / "pufferlib_market" / "data" / "mixed23_latest_val_20250922_20260320.bin"
DEFAULT_PROD_EVAL_HOURS = 30
DEFAULT_PROD_EVAL_WINDOWS = 50
DEFAULT_PROD_EVAL_SEED = 42
DEFAULT_PROD_EVAL_FEE_RATE = 0.001
DEFAULT_PROD_EVAL_SLIPPAGE_BPS = 5.0
DEFAULT_PROD_EVAL_FILL_BUFFER_BPS = 5.0
DEFAULT_PROD_EVAL_DECISION_LAG = 2
DEFAULT_REPLAY_EVAL_HOURLY_ROOT = "trainingdatahourly"
DEFAULT_REPLAY_EVAL_START_DATE = "2025-06-01"
DEFAULT_REPLAY_EVAL_END_DATE = "2026-02-05"
DEFAULT_REPLAY_EVAL_FILL_BUFFER_BPS = 5.0
DEFAULT_REPLAY_EVAL_SLIPPAGE_BPS_VALUES = (0.0, 5.0, 10.0, 20.0)
DEFAULT_DAILY_PERIODS_PER_YEAR = 365.0
DEFAULT_HOURLY_PERIODS_PER_YEAR = 8760.0
DEFAULT_REPLAY_ROBUST_START_STATES = ""
DEFAULT_PROD_EVAL_ALLOW_SHORTS = False
DEFAULT_PROD_EVAL_SKIP_REPLAY_EVAL = False
DEFAULT_PROD_EVAL_RUNTIME_TRACE_DIR = DEFAULT_TRACE_DIR
DEFAULT_PROD_EVAL_RUNTIME_AUDIT_HOURS = 24.0
DEFAULT_PROD_EVAL_SKIP_RUNTIME_AUDIT = False
DEFAULT_PROD_EVAL_REQUIRE_RUNTIME_MATCH = True
DEFAULT_PROD_EVAL_REQUIRE_RUNTIME_HEALTH = True
DEFAULT_PROD_EVAL_RUNTIME_MIN_HEALTHY_COMPLETED = 1
DEFAULT_PROD_EVAL_RUNTIME_MAX_DEGRADED_STATUS_COUNT = 0
DEFAULT_PROD_EVAL_RUNTIME_MAX_DEGRADED_FALLBACK_COUNT = 0
DEFAULT_PROD_EVAL_RUNTIME_MAX_GEMINI_SKIPPED_COUNT = 0


def normalize_replay_eval_slippage_bps_values(values: object) -> tuple[float, ...]:
    if isinstance(values, str):
        raw_tokens = [token.strip() for token in values.replace(",", " ").split()]
    elif isinstance(values, (list, tuple, set)):
        raw_tokens = [str(token).strip() for token in values]
    else:
        raw_tokens = [str(values).strip()] if values not in (None, "") else []

    normalized: list[float] = []
    seen: set[float] = set()
    for token in raw_tokens:
        if not token:
            continue
        value = float(token)
        if value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    if not normalized:
        raise ValueError("replay_eval_slippage_bps_values must contain at least one numeric value")
    return tuple(sorted(normalized))


def build_expected_prod_eval_config() -> dict[str, object]:
    return {
        "data_path": str(DEFAULT_PROD_EVAL_DATA_PATH.resolve(strict=False)),
        "eval_hours": DEFAULT_PROD_EVAL_HOURS,
        "n_windows": DEFAULT_PROD_EVAL_WINDOWS,
        "seed": DEFAULT_PROD_EVAL_SEED,
        "fee_rate": DEFAULT_PROD_EVAL_FEE_RATE,
        "slippage_bps": DEFAULT_PROD_EVAL_SLIPPAGE_BPS,
        "fill_buffer_bps": DEFAULT_PROD_EVAL_FILL_BUFFER_BPS,
        "decision_lag": DEFAULT_PROD_EVAL_DECISION_LAG,
        "periods_per_year": DEFAULT_DAILY_PERIODS_PER_YEAR,
        "replay_eval_hourly_root": DEFAULT_REPLAY_EVAL_HOURLY_ROOT,
        "replay_eval_start_date": DEFAULT_REPLAY_EVAL_START_DATE,
        "replay_eval_end_date": DEFAULT_REPLAY_EVAL_END_DATE,
        "replay_eval_fill_buffer_bps": DEFAULT_REPLAY_EVAL_FILL_BUFFER_BPS,
        "replay_eval_slippage_bps_values": list(DEFAULT_REPLAY_EVAL_SLIPPAGE_BPS_VALUES),
        "replay_eval_hourly_periods_per_year": DEFAULT_HOURLY_PERIODS_PER_YEAR,
        "replay_robust_start_states": DEFAULT_REPLAY_ROBUST_START_STATES,
        "runtime_trace_dir": str(DEFAULT_PROD_EVAL_RUNTIME_TRACE_DIR.resolve(strict=False)),
        "runtime_audit_hours": DEFAULT_PROD_EVAL_RUNTIME_AUDIT_HOURS,
        "skip_runtime_audit": DEFAULT_PROD_EVAL_SKIP_RUNTIME_AUDIT,
        "require_runtime_match": DEFAULT_PROD_EVAL_REQUIRE_RUNTIME_MATCH,
        "require_runtime_health": DEFAULT_PROD_EVAL_REQUIRE_RUNTIME_HEALTH,
        "runtime_min_healthy_completed": DEFAULT_PROD_EVAL_RUNTIME_MIN_HEALTHY_COMPLETED,
        "runtime_max_degraded_status_count": DEFAULT_PROD_EVAL_RUNTIME_MAX_DEGRADED_STATUS_COUNT,
        "runtime_max_degraded_fallback_count": DEFAULT_PROD_EVAL_RUNTIME_MAX_DEGRADED_FALLBACK_COUNT,
        "runtime_max_gemini_skipped_count": DEFAULT_PROD_EVAL_RUNTIME_MAX_GEMINI_SKIPPED_COUNT,
        "allow_shorts": DEFAULT_PROD_EVAL_ALLOW_SHORTS,
        "skip_replay_eval": DEFAULT_PROD_EVAL_SKIP_REPLAY_EVAL,
    }
