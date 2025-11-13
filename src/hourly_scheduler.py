from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

_SYMBOL_TOKEN = re.compile(r"['\"]([A-Za-z0-9\.\-_]+)['\"]")


def _dedupe_preserve_order(symbols: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for symbol in symbols:
        upper = symbol.upper()
        if upper in seen:
            continue
        seen.add(upper)
        ordered.append(upper)
    return ordered


def extract_symbols_from_text(text: str) -> List[str]:
    """
    Return all ticker-like tokens embedded in a blob of text.

    Tokens are detected via single or double quoted substrings containing
    alphanumeric characters, hyphen, period, or underscore.
    """
    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        cleaned_lines.append(line)
    cleaned = "\n".join(cleaned_lines)
    return _dedupe_preserve_order(match.group(1) for match in _SYMBOL_TOKEN.finditer(cleaned))


def load_symbols_from_file(path: Path) -> List[str]:
    """Parse symbols from the provided file, returning an empty list if unavailable."""
    try:
        text = path.read_text()
    except FileNotFoundError:
        return []
    except OSError:
        return []
    return extract_symbols_from_text(text)


def _split_env_symbols(raw: str) -> List[str]:
    return _dedupe_preserve_order(part.strip().upper() for part in raw.split(",") if part.strip())


def resolve_hourly_symbols(
    env_value: Optional[str],
    candidate_files: Sequence[Path],
    defaults: Sequence[str],
) -> List[str]:
    """
    Resolve the ordered list of symbols for the hourly trading loop.

    Priority:
        1. Explicit environment override ``env_value`` (comma separated list)
        2. First candidate file containing at least one detectable symbol
        3. Provided ``defaults`` sequence
    """
    if env_value:
        resolved = _split_env_symbols(env_value)
        if resolved:
            return resolved

    for path in candidate_files:
        parsed = load_symbols_from_file(path)
        if parsed:
            return parsed

    return _dedupe_preserve_order(defaults)


def _ensure_aware(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment


def _hour_floor(moment: datetime) -> datetime:
    aware = _ensure_aware(moment)
    return aware.replace(minute=0, second=0, microsecond=0)


@dataclass
class HourlyRunCoordinator:
    """
    Helper that ensures the hourly loop runs at most once per hour.

    Args:
        analysis_window_minutes: Window (from the start of the hour) in which we permit execution.
        allow_immediate_start: When True, the first call to ``should_run`` always returns True.
        allow_catch_up: When True, missed windows still execute once per hour as soon as detected.
    """

    analysis_window_minutes: int = 12
    allow_immediate_start: bool = True
    last_run_hour: Optional[datetime] = None
    allow_catch_up: bool = False

    def should_run(self, moment: datetime) -> bool:
        """Return True when the current time warrants an hourly analysis cycle."""
        now = _ensure_aware(moment)
        if self.last_run_hour is None:
            return self.allow_immediate_start or self._within_window(now)

        current_hour = _hour_floor(now)
        previous_hour = _hour_floor(self.last_run_hour)
        if current_hour <= previous_hour:
            return False
        if self._within_window(now):
            return True
        return self.allow_catch_up

    def mark_executed(self, moment: datetime) -> None:
        """Record that the loop executed for the provided moment."""
        self.last_run_hour = _hour_floor(moment)

    def _within_window(self, moment: datetime) -> bool:
        window = max(0, int(self.analysis_window_minutes))
        return (moment.minute + moment.second / 60.0) <= window


__all__ = [
    "extract_symbols_from_text",
    "load_symbols_from_file",
    "resolve_hourly_symbols",
    "HourlyRunCoordinator",
]
