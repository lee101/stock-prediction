from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

from loguru import logger

from marketsimulator.state import get_state

EntryKey = Tuple[Optional[str], Optional[str]]

TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}

_DRAW_CAPS_CACHE: Optional[Tuple[str, Dict[EntryKey, float]]] = None
_DRAW_RESUME_CACHE: Optional[Tuple[str, Dict[EntryKey, float]]] = None
_THRESHOLD_MAP_CACHE: Dict[str, Tuple[str, Dict[EntryKey, float]]] = {}
_SYMBOL_SIDE_CACHE: Optional[Tuple[str, Dict[str, str]]] = None
_SYMBOL_KELLY_SCALE_CACHE: Optional[Tuple[str, Dict[str, float]]] = None
_SYMBOL_MAX_HOLD_CACHE: Optional[Tuple[str, Dict[str, float]]] = None
_SYMBOL_MIN_COOLDOWN_CACHE: Optional[Tuple[str, Dict[str, float]]] = None
_SYMBOL_MAX_ENTRIES_CACHE: Optional[Tuple[str, Dict[EntryKey, int]]] = None
_SYMBOL_FORCE_PROBE_CACHE: Optional[Tuple[str, Dict[str, bool]]] = None
_SYMBOL_MIN_MOVE_CACHE: Optional[Tuple[str, Dict[str, float]]] = None
_SYMBOL_MIN_PREDICTED_MOVE_CACHE: Optional[Tuple[str, Dict[str, float]]] = None
_SYMBOL_MIN_STRATEGY_RETURN_CACHE: Optional[Tuple[str, Dict[str, float]]] = None
_TREND_SUMMARY_CACHE: Optional[Tuple[Tuple[str, float], Dict[str, Dict[str, float]]]] = None
_TREND_RESUME_CACHE: Optional[Tuple[str, Dict[str, float]]] = None

_SYMBOL_RUN_ENTRY_COUNTS: Dict[EntryKey, int] = {}
_SYMBOL_RUN_ENTRY_ID: Optional[str] = None


def _get_env_float(name: str) -> Optional[float]:
    raw = os.getenv(name)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; expected float.", name, raw)
        return None


def _parse_threshold_map(env_name: str) -> Dict[EntryKey, float]:
    cache_key_raw = os.getenv(env_name)
    cache_key = cache_key_raw or ""
    cached = _THRESHOLD_MAP_CACHE.get(env_name)
    if cached is None or cached[0] != cache_key:
        parsed: Dict[EntryKey, float] = {}
        if cache_key_raw:
            for item in cache_key_raw.split(","):
                entry = item.strip()
                if not entry:
                    continue
                try:
                    key_part, value_part = entry.split(":", 1)
                    value = float(value_part)
                except ValueError:
                    logger.warning("Ignoring invalid %s entry: %s", env_name, entry)
                    continue
                key = key_part.strip()
                if not key:
                    logger.warning("Ignoring invalid %s entry with empty key.", env_name)
                    continue
                symbol_key: Optional[str] = None
                strategy_key: Optional[str] = None
                if "@" in key:
                    sym_raw, strat_raw = key.split("@", 1)
                    symbol_key = sym_raw.strip().lower() or None
                    strategy_key = strat_raw.strip().lower() or None
                elif key.isupper():
                    symbol_key = key.lower()
                else:
                    strategy_key = key.lower()
                parsed[(symbol_key, strategy_key)] = value
        _THRESHOLD_MAP_CACHE[env_name] = (cache_key, parsed)
    return _THRESHOLD_MAP_CACHE[env_name][1]


def _lookup_threshold(env_name: str, symbol: Optional[str], strategy: Optional[str]) -> Optional[float]:
    parsed = _parse_threshold_map(env_name)
    symbol_key = symbol.lower() if symbol else None
    strategy_key = strategy.lower() if strategy else None
    for candidate in (
        (symbol_key, strategy_key),
        (symbol_key, None),
        (None, strategy_key),
        (None, None),
    ):
        if candidate in parsed:
            return parsed[candidate]
    return None


def _drawdown_cap_for(strategy: Optional[str], symbol: Optional[str] = None) -> Optional[float]:
    global _DRAW_CAPS_CACHE
    env_raw = os.getenv("MARKETSIM_KELLY_DRAWDOWN_CAP_MAP")
    cache_key = env_raw or ""
    if _DRAW_CAPS_CACHE is None or _DRAW_CAPS_CACHE[0] != cache_key:
        _DRAW_CAPS_CACHE = (cache_key, _parse_threshold_map("MARKETSIM_KELLY_DRAWDOWN_CAP_MAP"))
    caps = _DRAW_CAPS_CACHE[1] if _DRAW_CAPS_CACHE else {}
    symbol_key = symbol.lower() if symbol else None
    strategy_key = strategy.lower() if strategy else None
    for candidate in (
        (symbol_key, strategy_key),
        (symbol_key, None),
        (None, strategy_key),
        (None, None),
    ):
        if candidate in caps:
            return caps[candidate]
    return _get_env_float("MARKETSIM_KELLY_DRAWDOWN_CAP")


def _drawdown_resume_for(
    strategy: Optional[str], cap: Optional[float], symbol: Optional[str] = None
) -> Optional[float]:
    global _DRAW_RESUME_CACHE
    env_raw = os.getenv("MARKETSIM_DRAWDOWN_RESUME_MAP")
    cache_key = env_raw or ""
    if _DRAW_RESUME_CACHE is None or _DRAW_RESUME_CACHE[0] != cache_key:
        _DRAW_RESUME_CACHE = (cache_key, _parse_threshold_map("MARKETSIM_DRAWDOWN_RESUME_MAP"))
    overrides = _DRAW_RESUME_CACHE[1] if _DRAW_RESUME_CACHE else {}
    symbol_key = symbol.lower() if symbol else None
    strategy_key = strategy.lower() if strategy else None
    for candidate in (
        (symbol_key, strategy_key),
        (symbol_key, None),
        (None, strategy_key),
        (None, None),
    ):
        if candidate in overrides:
            return overrides[candidate]
    resume_abs = _get_env_float("MARKETSIM_DRAWDOWN_RESUME")
    if resume_abs is not None:
        return resume_abs
    factor = _get_env_float("MARKETSIM_DRAWDOWN_RESUME_FACTOR") or 0.8
    if factor <= 0 or cap is None:
        return None
    return cap * factor


def _symbol_kelly_scale(symbol: Optional[str]) -> Optional[float]:
    global _SYMBOL_KELLY_SCALE_CACHE
    if symbol is None:
        return None
    env_raw = os.getenv("MARKETSIM_SYMBOL_KELLY_SCALE_MAP")
    cache_key = env_raw or ""
    if _SYMBOL_KELLY_SCALE_CACHE is None or _SYMBOL_KELLY_SCALE_CACHE[0] != cache_key:
        parsed: Dict[str, float] = {}
        if env_raw:
            for item in env_raw.split(","):
                entry = item.strip()
                if not entry or ":" not in entry:
                    logger.warning("Ignoring malformed MARKETSIM_SYMBOL_KELLY_SCALE_MAP entry: %s", entry)
                    continue
                symbol_key, value = entry.split(":", 1)
                try:
                    parsed[symbol_key.strip().lower()] = float(value)
                except ValueError:
                    logger.warning("Ignoring invalid MARKETSIM_SYMBOL_KELLY_SCALE_MAP value: %s", entry)
        _SYMBOL_KELLY_SCALE_CACHE = (cache_key, parsed)
    overrides = _SYMBOL_KELLY_SCALE_CACHE[1] if _SYMBOL_KELLY_SCALE_CACHE else {}
    return overrides.get(symbol.lower())


def _kelly_drawdown_scale(strategy: Optional[str], symbol: Optional[str] = None) -> float:
    cap = _drawdown_cap_for(strategy, symbol)
    if not cap or cap <= 0:
        scale = 1.0
    else:
        min_scale = _get_env_float("MARKETSIM_KELLY_DRAWDOWN_MIN_SCALE") or 0.0
        try:
            state = get_state()
            drawdown_pct = getattr(state, "drawdown_pct", None)
        except RuntimeError:
            drawdown_pct = None
        if drawdown_pct is None:
            scale = 1.0
        else:
            scale = max(0.0, 1.0 - (drawdown_pct / cap))
            if min_scale > 0:
                scale = max(min_scale, scale)
            scale = min(1.0, scale)

    symbol_scale = _symbol_kelly_scale(symbol)
    if symbol_scale is not None:
        scale *= max(0.0, min(symbol_scale, 1.0))
    min_scale = _get_env_float("MARKETSIM_KELLY_DRAWDOWN_MIN_SCALE") or 0.0
    if min_scale > 0:
        scale = max(min_scale, scale)
    return min(1.0, scale)


def _allowed_side_for(symbol: Optional[str]) -> Optional[str]:
    global _SYMBOL_SIDE_CACHE
    if symbol is None:
        return None
    env_raw = os.getenv("MARKETSIM_SYMBOL_SIDE_MAP")
    cache_key = env_raw or ""
    if _SYMBOL_SIDE_CACHE is None or _SYMBOL_SIDE_CACHE[0] != cache_key:
        parsed: Dict[str, str] = {}
        if env_raw:
            for item in env_raw.split(","):
                entry = item.strip()
                if not entry:
                    continue
                if ":" not in entry:
                    logger.warning("Ignoring malformed MARKETSIM_SYMBOL_SIDE_MAP entry: %s", entry)
                    continue
                symbol_key, side = entry.split(":", 1)
                norm_symbol = symbol_key.strip().lower()
                norm_side = side.strip().lower()
                if norm_symbol and norm_side in {"buy", "sell", "both"}:
                    parsed[norm_symbol] = norm_side
                else:
                    logger.warning("Ignoring invalid MARKETSIM_SYMBOL_SIDE_MAP entry: %s", entry)
        _SYMBOL_SIDE_CACHE = (cache_key, parsed)
    overrides = _SYMBOL_SIDE_CACHE[1] if _SYMBOL_SIDE_CACHE else {}
    return overrides.get(symbol.lower())


def _symbol_max_hold_seconds(symbol: Optional[str]) -> Optional[float]:
    global _SYMBOL_MAX_HOLD_CACHE
    if symbol is None:
        return None
    env_raw = os.getenv("MARKETSIM_SYMBOL_MAX_HOLD_SECONDS_MAP")
    cache_key = env_raw or ""
    if _SYMBOL_MAX_HOLD_CACHE is None or _SYMBOL_MAX_HOLD_CACHE[0] != cache_key:
        parsed: Dict[str, float] = {}
        if env_raw:
            for item in env_raw.split(","):
                entry = item.strip()
                if not entry or ":" not in entry:
                    logger.warning("Ignoring malformed MARKETSIM_SYMBOL_MAX_HOLD_SECONDS_MAP entry: %s", entry)
                    continue
                symbol_key, seconds_raw = entry.split(":", 1)
                try:
                    parsed[symbol_key.strip().lower()] = float(seconds_raw)
                except ValueError:
                    logger.warning("Ignoring invalid MARKETSIM_SYMBOL_MAX_HOLD_SECONDS_MAP value: %s", entry)
        _SYMBOL_MAX_HOLD_CACHE = (cache_key, parsed)
    overrides = _SYMBOL_MAX_HOLD_CACHE[1] if _SYMBOL_MAX_HOLD_CACHE else {}
    return overrides.get(symbol.lower())


def _symbol_min_cooldown_minutes(symbol: Optional[str]) -> Optional[float]:
    global _SYMBOL_MIN_COOLDOWN_CACHE
    if symbol is None:
        return None
    env_raw = os.getenv("MARKETSIM_SYMBOL_MIN_COOLDOWN_MAP")
    cache_key = env_raw or ""
    if _SYMBOL_MIN_COOLDOWN_CACHE is None or _SYMBOL_MIN_COOLDOWN_CACHE[0] != cache_key:
        parsed: Dict[str, float] = {}
        if env_raw:
            for item in env_raw.split(","):
                entry = item.strip()
                if not entry or ":" not in entry:
                    logger.warning("Ignoring malformed MARKETSIM_SYMBOL_MIN_COOLDOWN_MAP entry: %s", entry)
                    continue
                symbol_key, value_raw = entry.split(":", 1)
                try:
                    parsed[symbol_key.strip().lower()] = float(value_raw)
                except ValueError:
                    logger.warning("Ignoring invalid MARKETSIM_SYMBOL_MIN_COOLDOWN_MAP value: %s", entry)
        _SYMBOL_MIN_COOLDOWN_CACHE = (cache_key, parsed)
    overrides = _SYMBOL_MIN_COOLDOWN_CACHE[1] if _SYMBOL_MIN_COOLDOWN_CACHE else {}
    return overrides.get(symbol.lower())


def _symbol_max_entries_per_run(
    symbol: Optional[str], strategy: Optional[str] = None
) -> Tuple[Optional[int], Optional[EntryKey]]:
    global _SYMBOL_MAX_ENTRIES_CACHE
    env_raw = os.getenv("MARKETSIM_SYMBOL_MAX_ENTRIES_MAP")
    cache_key = env_raw or ""
    if _SYMBOL_MAX_ENTRIES_CACHE is None or _SYMBOL_MAX_ENTRIES_CACHE[0] != cache_key:
        parsed: Dict[EntryKey, int] = {}
        if env_raw:
            for item in env_raw.split(","):
                entry = item.strip()
                if not entry or ":" not in entry:
                    logger.warning("Ignoring malformed MARKETSIM_SYMBOL_MAX_ENTRIES_MAP entry: %s", entry)
                    continue
                key_raw, value_raw = entry.split(":", 1)
                symbol_key: Optional[str] = None
                strategy_key: Optional[str] = None
                if "@" in key_raw:
                    sym_raw, strat_raw = key_raw.split("@", 1)
                    symbol_key = sym_raw.strip().lower() or None
                    strategy_key = strat_raw.strip().lower() or None
                else:
                    key_clean = key_raw.strip().lower()
                    symbol_key = key_clean or None
                try:
                    parsed[(symbol_key, strategy_key)] = int(float(value_raw))
                except ValueError:
                    logger.warning("Ignoring invalid MARKETSIM_SYMBOL_MAX_ENTRIES_MAP value: %s", entry)
        _SYMBOL_MAX_ENTRIES_CACHE = (cache_key, parsed)
    overrides = _SYMBOL_MAX_ENTRIES_CACHE[1] if _SYMBOL_MAX_ENTRIES_CACHE else {}
    symbol_key = symbol.lower() if symbol else None
    strategy_key = strategy.lower() if strategy else None
    for candidate in (
        (symbol_key, strategy_key),
        (symbol_key, None),
        (None, strategy_key),
        (None, None),
    ):
        if candidate in overrides:
            return overrides[candidate], candidate
    return None, None


def _symbol_min_move(symbol: Optional[str]) -> Optional[float]:
    global _SYMBOL_MIN_MOVE_CACHE
    if symbol is None:
        return None
    env_raw = os.getenv("MARKETSIM_SYMBOL_MIN_MOVE_MAP")
    cache_key = env_raw or ""
    if _SYMBOL_MIN_MOVE_CACHE is None or _SYMBOL_MIN_MOVE_CACHE[0] != cache_key:
        parsed: Dict[str, float] = {}
        if env_raw:
            for item in env_raw.split(","):
                entry = item.strip()
                if not entry or ":" not in entry:
                    logger.warning("Ignoring malformed MARKETSIM_SYMBOL_MIN_MOVE_MAP entry: %s", entry)
                    continue
                key_raw, value_raw = entry.split(":", 1)
                try:
                    parsed[key_raw.strip().lower()] = float(value_raw)
                except ValueError:
                    logger.warning("Ignoring invalid MARKETSIM_SYMBOL_MIN_MOVE_MAP value: %s", entry)
        _SYMBOL_MIN_MOVE_CACHE = (cache_key, parsed)
    overrides = _SYMBOL_MIN_MOVE_CACHE[1] if _SYMBOL_MIN_MOVE_CACHE else {}
    return overrides.get(symbol.lower())


def _symbol_min_predicted_move(symbol: Optional[str]) -> Optional[float]:
    global _SYMBOL_MIN_PREDICTED_MOVE_CACHE
    if symbol is None:
        return None
    env_raw = os.getenv("MARKETSIM_SYMBOL_MIN_PREDICTED_MOVE_MAP")
    cache_key = env_raw or ""
    if (
        _SYMBOL_MIN_PREDICTED_MOVE_CACHE is None
        or _SYMBOL_MIN_PREDICTED_MOVE_CACHE[0] != cache_key
    ):
        parsed: Dict[str, float] = {}
        if env_raw:
            for item in env_raw.split(","):
                entry = item.strip()
                if not entry or ":" not in entry:
                    logger.warning(
                        "Ignoring malformed MARKETSIM_SYMBOL_MIN_PREDICTED_MOVE_MAP entry: %s",
                        entry,
                    )
                    continue
                key_raw, value_raw = entry.split(":", 1)
                try:
                    parsed[key_raw.strip().lower()] = abs(float(value_raw))
                except ValueError:
                    logger.warning(
                        "Ignoring invalid MARKETSIM_SYMBOL_MIN_PREDICTED_MOVE_MAP value: %s",
                        entry,
                    )
        _SYMBOL_MIN_PREDICTED_MOVE_CACHE = (cache_key, parsed)
    overrides = (
        _SYMBOL_MIN_PREDICTED_MOVE_CACHE[1] if _SYMBOL_MIN_PREDICTED_MOVE_CACHE else {}
    )
    return overrides.get(symbol.lower())


def _symbol_min_strategy_return(symbol: Optional[str]) -> Optional[float]:
    global _SYMBOL_MIN_STRATEGY_RETURN_CACHE
    if symbol is None:
        return None
    env_raw = os.getenv("MARKETSIM_SYMBOL_MIN_STRATEGY_RETURN_MAP")
    cache_key = env_raw or ""
    if (
        _SYMBOL_MIN_STRATEGY_RETURN_CACHE is None
        or _SYMBOL_MIN_STRATEGY_RETURN_CACHE[0] != cache_key
    ):
        parsed: Dict[str, float] = {}
        if env_raw:
            for item in env_raw.split(","):
                entry = item.strip()
                if not entry or ":" not in entry:
                    logger.warning(
                        "Ignoring malformed MARKETSIM_SYMBOL_MIN_STRATEGY_RETURN_MAP entry: %s",
                        entry,
                    )
                    continue
                key_raw, value_raw = entry.split(":", 1)
                try:
                    parsed[key_raw.strip().lower()] = float(value_raw)
                except ValueError:
                    logger.warning(
                        "Ignoring invalid MARKETSIM_SYMBOL_MIN_STRATEGY_RETURN_MAP value: %s",
                        entry,
                    )
        _SYMBOL_MIN_STRATEGY_RETURN_CACHE = (cache_key, parsed)
    overrides = _SYMBOL_MIN_STRATEGY_RETURN_CACHE[1] if _SYMBOL_MIN_STRATEGY_RETURN_CACHE else {}
    return overrides.get(symbol.lower())


def _symbol_force_probe(symbol: Optional[str]) -> bool:
    global _SYMBOL_FORCE_PROBE_CACHE
    if symbol is None:
        return False
    env_raw = os.getenv("MARKETSIM_SYMBOL_FORCE_PROBE_MAP")
    cache_key = env_raw or ""
    if _SYMBOL_FORCE_PROBE_CACHE is None or _SYMBOL_FORCE_PROBE_CACHE[0] != cache_key:
        parsed: Dict[str, bool] = {}
        if env_raw:
            for item in env_raw.split(","):
                entry = item.strip()
                if not entry:
                    continue
                if ":" in entry:
                    key_raw, value_raw = entry.split(":", 1)
                    value_norm = value_raw.strip().lower()
                    parsed[key_raw.strip().lower()] = value_norm in TRUTHY_ENV_VALUES
                else:
                    parsed[entry.strip().lower()] = True
        _SYMBOL_FORCE_PROBE_CACHE = (cache_key, parsed)
    overrides = _SYMBOL_FORCE_PROBE_CACHE[1] if _SYMBOL_FORCE_PROBE_CACHE else {}
    return bool(overrides.get(symbol.lower()))


def _symbol_trend_pnl_threshold(symbol: Optional[str]) -> Optional[float]:
    if symbol is None:
        return None
    env_raw = os.getenv("MARKETSIM_TREND_PNL_SUSPEND_MAP")
    if not env_raw:
        return None
    for item in env_raw.split(","):
        entry = item.strip()
        if not entry or ":" not in entry:
            continue
        key_raw, value_raw = entry.split(":", 1)
        if key_raw.strip().lower() == symbol.lower():
            try:
                return float(value_raw)
            except ValueError:
                logger.warning("Invalid MARKETSIM_TREND_PNL_SUSPEND_MAP value: %s", entry)
                return None
    return None


def _symbol_trend_resume_threshold(symbol: Optional[str]) -> Optional[float]:
    global _TREND_RESUME_CACHE
    if symbol is None:
        return None
    env_raw = os.getenv("MARKETSIM_TREND_PNL_RESUME_MAP")
    cache_key = env_raw or ""
    if _TREND_RESUME_CACHE is None or _TREND_RESUME_CACHE[0] != cache_key:
        parsed: Dict[str, float] = {}
        if env_raw:
            for item in env_raw.split(","):
                entry = item.strip()
                if not entry or ":" not in entry:
                    logger.warning("Ignoring malformed MARKETSIM_TREND_PNL_RESUME_MAP entry: %s", entry)
                    continue
                key_raw, value_raw = entry.split(":", 1)
                try:
                    parsed[key_raw.strip().lower()] = float(value_raw)
                except ValueError:
                    logger.warning("Ignoring invalid MARKETSIM_TREND_PNL_RESUME_MAP value: %s", entry)
        _TREND_RESUME_CACHE = (cache_key, parsed)
    overrides = _TREND_RESUME_CACHE[1] if _TREND_RESUME_CACHE else {}
    return overrides.get(symbol.lower())


def _load_trend_summary() -> Dict[str, Dict[str, float]]:
    global _TREND_SUMMARY_CACHE
    path_raw = os.getenv("MARKETSIM_TREND_SUMMARY_PATH")
    if not path_raw:
        return {}
    path = Path(path_raw)
    if not path.exists():
        logger.debug("Trend summary path %s not found; skipping suspend checks.", path)
        return {}
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}
    cache_key = (path_raw, mtime)
    if _TREND_SUMMARY_CACHE and _TREND_SUMMARY_CACHE[0] == cache_key:
        return _TREND_SUMMARY_CACHE[1]
    try:
        with path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load trend summary %s: %s", path, exc)
        return {}
    _TREND_SUMMARY_CACHE = (cache_key, summary)
    return summary


def _get_trend_stat(symbol: str, key: str) -> Optional[float]:
    summary = _load_trend_summary()
    if not summary:
        return None
    symbol_info = summary.get(symbol.upper())
    if not symbol_info:
        return None
    value = symbol_info.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def reset_symbol_entry_counters(run_id: Optional[str] = None) -> None:
    """Clear per-run entry counters to allow fresh simulations or trading sessions."""
    global _SYMBOL_RUN_ENTRY_COUNTS, _SYMBOL_RUN_ENTRY_ID
    _SYMBOL_RUN_ENTRY_COUNTS = {}
    _SYMBOL_RUN_ENTRY_ID = run_id


def _normalize_entry_key(symbol: Optional[str], strategy: Optional[str]) -> Optional[EntryKey]:
    if symbol is None:
        return None
    return (symbol.lower(), strategy.lower() if strategy else None)


def _current_symbol_entry_count(symbol: str, strategy: Optional[str], *, key: Optional[EntryKey] = None) -> int:
    use_key = key if key is not None else _normalize_entry_key(symbol, strategy)
    if use_key is None:
        return 0
    return _SYMBOL_RUN_ENTRY_COUNTS.get(use_key, 0)


def _increment_symbol_entry(symbol: str, strategy: Optional[str], *, key: Optional[EntryKey] = None) -> int:
    use_key = key if key is not None else _normalize_entry_key(symbol, strategy)
    if use_key is None:
        return 0
    new_count = _SYMBOL_RUN_ENTRY_COUNTS.get(use_key, 0) + 1
    _SYMBOL_RUN_ENTRY_COUNTS[use_key] = new_count
    return new_count


def _format_entry_limit_key(key: Optional[EntryKey]) -> Optional[str]:
    if key is None:
        return None
    symbol_key, strategy_key = key
    if symbol_key and strategy_key:
        return f"{symbol_key}@{strategy_key}"
    if symbol_key:
        return symbol_key
    if strategy_key:
        return f"@{strategy_key}"
    return "__default__"


def get_entry_counter_snapshot() -> Dict[str, Dict[str, Dict[str, Optional[float]]]]:
    """Return per-key and per-symbol entry counter statistics for the current run."""
    snapshot_per_key: Dict[str, Dict[str, Optional[float]]] = {}
    aggregated: Dict[str, Dict[str, Optional[float]]] = {}

    for (symbol_key, strategy_key), count in _SYMBOL_RUN_ENTRY_COUNTS.items():
        label_symbol = (symbol_key or "__global__").upper()
        label_key = label_symbol if strategy_key is None else f"{label_symbol}@{strategy_key}"
        resolved_limit, matched_key = _symbol_max_entries_per_run(
            label_symbol if symbol_key is not None else None,
            strategy_key,
        )
        approx_trade_limit = float(max(resolved_limit, 0) * 2) if resolved_limit is not None else None
        snapshot_per_key[label_key] = {
            "entries": int(count),
            "entry_limit": float(resolved_limit) if resolved_limit is not None else None,
            "approx_trade_limit": approx_trade_limit,
            "resolved_limit_key": _format_entry_limit_key(matched_key),
        }

        aggregate = aggregated.setdefault(
            label_symbol,
            {
                "entries": 0,
                "entry_limits": [],
            },
        )
        aggregate["entries"] += int(count)
        if resolved_limit is not None:
            aggregate["entry_limits"].append(float(resolved_limit))

    per_symbol: Dict[str, Dict[str, Optional[float]]] = {}
    for symbol_label, info in aggregated.items():
        candidates = info["entry_limits"]
        entry_limit = min(candidates) if candidates else None
        approx_trade_limit = float(max(entry_limit, 0) * 2) if entry_limit is not None else None
        per_symbol[symbol_label] = {
            "entries": info["entries"],
            "entry_limit": entry_limit,
            "approx_trade_limit": approx_trade_limit,
        }

    return {
        "per_key": snapshot_per_key,
        "per_symbol": per_symbol,
    }


__all__ = [
    "EntryKey",
    "TRUTHY_ENV_VALUES",
    "_allowed_side_for",
    "_current_symbol_entry_count",
    "_drawdown_cap_for",
    "_drawdown_resume_for",
    "_format_entry_limit_key",
    "_get_env_float",
    "_get_trend_stat",
    "_increment_symbol_entry",
    "_kelly_drawdown_scale",
    "_load_trend_summary",
    "_lookup_threshold",
    "_normalize_entry_key",
    "_parse_threshold_map",
    "_symbol_force_probe",
    "_symbol_kelly_scale",
    "_symbol_max_entries_per_run",
    "_symbol_max_hold_seconds",
    "_symbol_min_cooldown_minutes",
    "_symbol_min_move",
    "_symbol_min_predicted_move",
    "_symbol_min_strategy_return",
    "_symbol_trend_pnl_threshold",
    "_symbol_trend_resume_threshold",
    "get_entry_counter_snapshot",
    "reset_symbol_entry_counters",
]
