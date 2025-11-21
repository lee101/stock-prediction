from datetime import datetime, timedelta, timezone
import sys
import types

import pytest

# Stub alpaca_trade_api before importing modules that depend on it (defensive for indirect imports).
if "alpaca_trade_api" not in sys.modules:
    fake_tradeapi = types.SimpleNamespace(
        REST=lambda *args, **kwargs: None,
        TimeFrame=None,
        TimeFrameUnit=None,
    )
    sys.modules["alpaca_trade_api"] = fake_tradeapi
    sys.modules["alpaca_trade_api.rest"] = types.SimpleNamespace(
        TimeFrame=None,
        TimeFrameUnit=None,
        APIError=Exception,
    )

from src.maxdiff_entry_guard import _effective_entry_quantities, _parse_iso_timestamp


def test_pending_reserves_cap_prevents_reorders():
    now = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    status = {
        "pending_qty": 2.0,
        "pending_expires_at": (now + timedelta(seconds=90)).isoformat(),
        "position_qty": 0.0,
    }

    effective, remaining, pending, reached = _effective_entry_quantities(
        status=status,
        current_qty=0.0,
        open_order_qty=0.0,
        target_qty=2.0,
        now=now,
        pending_ttl_seconds=120,
    )

    assert reached is True
    assert effective == pytest.approx(2.0)
    assert remaining == pytest.approx(0.0)
    # Pending expiry should be extended when reservations remain
    assert _parse_iso_timestamp(status["pending_expires_at"]) > now
    assert pending == pytest.approx(2.0)


def test_pending_expiry_releases_capacity():
    now = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    status = {
        "pending_qty": 2.0,
        "pending_expires_at": (now - timedelta(seconds=5)).isoformat(),
        "position_qty": 0.0,
    }

    effective, remaining, pending, reached = _effective_entry_quantities(
        status=status,
        current_qty=0.0,
        open_order_qty=0.0,
        target_qty=2.0,
        now=now,
        pending_ttl_seconds=120,
    )

    assert reached is False
    assert effective == pytest.approx(0.0)
    assert remaining == pytest.approx(2.0)
    assert pending == pytest.approx(0.0)


def test_fills_reduce_pending_reservation():
    now = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    status = {
        "pending_qty": 1.5,
        "pending_expires_at": (now + timedelta(seconds=300)).isoformat(),
        "position_qty": 0.0,
    }

    effective, remaining, pending, reached = _effective_entry_quantities(
        status=status,
        current_qty=1.0,  # Newly observed fill
        open_order_qty=0.2,
        target_qty=2.0,
        now=now,
        pending_ttl_seconds=180,
    )

    # Pending should drop by the filled amount (1.0), leaving 0.5
    assert pending == pytest.approx(0.5)
    # Effective exposure counts position + open orders + pending remainder
    assert effective == pytest.approx(1.0 + 0.2 + 0.5)
    assert remaining == pytest.approx(2.0 - effective)
    assert reached is False
