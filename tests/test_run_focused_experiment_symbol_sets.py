from __future__ import annotations

from unified_hourly_experiment import run_focused_experiment


def test_dbx_is_not_in_longable_bucket() -> None:
    longable = set(run_focused_experiment.LONGABLE.split(","))
    shortable = set(run_focused_experiment.SHORTABLE.split(","))

    assert "DBX" not in longable
    assert "DBX" in shortable
