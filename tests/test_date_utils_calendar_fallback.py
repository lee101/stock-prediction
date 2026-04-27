from __future__ import annotations

import builtins
from datetime import datetime
from types import SimpleNamespace

from src import date_utils


def test_missing_exchange_calendar_fallback_is_cached(monkeypatch):
    original_import = builtins.__import__
    warnings: list[str] = []
    import_attempts = {"count": 0}

    def fake_import(name, *args, **kwargs):
        if name == "exchange_calendars":
            import_attempts["count"] += 1
            raise ImportError("missing in test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(date_utils, "_nyse_calendar", None)
    monkeypatch.setattr(date_utils, "logger", SimpleNamespace(warning=warnings.append))
    date_utils._is_nyse_session_cached.cache_clear()

    try:
        assert date_utils.is_nyse_open_on_date(datetime(2026, 4, 27))
        assert not date_utils.is_nyse_open_on_date(datetime(2026, 4, 26))
    finally:
        date_utils._is_nyse_session_cached.cache_clear()

    assert import_attempts["count"] == 1
    assert warnings == ["exchange_calendars not installed, falling back to weekday check"]
