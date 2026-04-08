"""Tests for fp4.curriculum.LeverageCurriculum (P6-2)."""
from __future__ import annotations

import pytest

from fp4.curriculum import LeverageCurriculum


def test_holds_before_ramp_steps():
    cur = LeverageCurriculum(start=1.0, target=2.0, ramp_steps=1000, cap=5.0)
    for step in (0, 100, 999):
        assert cur.current_cap(step=step, last_sortino=10.0) == 1.0


def test_holds_when_sortino_nonpositive():
    cur = LeverageCurriculum(start=1.0, target=2.0, ramp_steps=0, cap=5.0)
    # Zero or negative Sortino → no advance.
    assert cur.current_cap(step=0, last_sortino=0.0) == 1.0
    assert cur.current_cap(step=10, last_sortino=-1.5) == 1.0


def test_advances_toward_target_then_cap():
    cur = LeverageCurriculum(
        start=1.0, target=2.0, ramp_steps=0, cap=5.0, step_size=0.5
    )
    # First call: 1 + 0.5*(2-1) = 1.5
    c1 = cur.current_cap(step=0, last_sortino=1.0)
    assert c1 == pytest.approx(1.5)
    # Second: 1.5 + 0.5*(2-1.5) = 1.75
    c2 = cur.current_cap(step=1, last_sortino=1.0)
    assert c2 == pytest.approx(1.75)
    # Many iterations — should reach target, then climb toward cap.
    for _ in range(200):
        cur.current_cap(step=2, last_sortino=1.0)
    assert cur.current_cap(step=3, last_sortino=1.0) == pytest.approx(5.0, rel=1e-3)


def test_never_exceeds_cap():
    cur = LeverageCurriculum(
        start=1.0, target=2.0, ramp_steps=0, cap=3.0, step_size=0.9
    )
    for i in range(10000):
        v = cur.current_cap(step=i, last_sortino=5.0)
        assert v <= 3.0 + 1e-9


def test_validation_errors():
    with pytest.raises(ValueError):
        LeverageCurriculum(start=0.0)
    with pytest.raises(ValueError):
        LeverageCurriculum(start=2.0, target=1.0)
    with pytest.raises(ValueError):
        LeverageCurriculum(target=2.0, cap=1.0)
    with pytest.raises(ValueError):
        LeverageCurriculum(ramp_steps=-5)
    with pytest.raises(ValueError):
        LeverageCurriculum(step_size=0.0)
    with pytest.raises(ValueError):
        LeverageCurriculum(step_size=1.5)


def test_state_dict_roundtrip():
    cur = LeverageCurriculum(start=1.0, target=2.0, ramp_steps=0, cap=5.0)
    cur.current_cap(step=10, last_sortino=2.0)
    state = cur.state_dict()
    cur2 = LeverageCurriculum()
    cur2.load_state_dict(state)
    assert cur2.state_dict() == state
