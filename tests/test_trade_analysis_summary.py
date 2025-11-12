from src.trade_analysis_summary import (
    build_analysis_summary_messages,
    format_metric_parts,
)


def test_format_metric_parts_skips_none_and_invalid_values():
    parts = [
        ("valid", 1.2345, 2),
        ("none_value", None, 3),
        ("string_value", "oops", 1),
    ]
    result = format_metric_parts(parts)
    assert result == "valid=1.23"


def test_build_analysis_summary_messages_includes_core_sections():
    data = {
        "strategy": "s1",
        "side": "buy",
        "trade_mode": "normal",
        "trade_blocked": True,
        "block_reason": "cooldown",
        "avg_return": 0.1234,
        "annual_return": 0.5,
        "simple_return": 0.2,
        "strategy_returns": {"highlow": 0.33},
        "predicted_movement": 1.2,
        "expected_move_pct": 0.01,
        "price_skill": 0.2,
        "edge_strength": 0.4,
        "directional_edge": 0.5,
        "predicted_close": 100.5,
        "predicted_high": 101.0,
        "predicted_low": 99.5,
        "last_close": 98.0,
        "walk_forward_oos_sharpe": 1.5,
        "walk_forward_notes": ["note-a", "note-b"],
    }
    compact, detailed = build_analysis_summary_messages("AAPL", data)

    assert "AAPL analysis" in compact
    assert "returns[" in compact and "edges[" in compact and "prices[" in compact
    assert "block_reason=cooldown" in compact
    assert "walk_forward_notes=note-a; note-b" in compact

    assert "returns: avg=0.123" in detailed
    assert "edges: move=1.200" in detailed
    assert "prices: pred_close=100.500" in detailed
    assert "block_reason: cooldown" in detailed
    assert "walk_forward_notes: note-a; note-b" in detailed


def test_build_analysis_summary_messages_adds_probe_notes():
    data = {
        "strategy": "s2",
        "side": "sell",
        "trade_mode": "probe",
        "pending_probe": True,
        "probe_active": False,
        "probe_transition_ready": True,
        "probe_expired": False,
        "probe_age_seconds": 321.9,
        "probe_started_at": "2025-11-11T10:00:00Z",
        "probe_expires_at": "2025-11-12T10:00:00Z",
        "strategy_returns": {},
    }
    compact, detailed = build_analysis_summary_messages("MSFT", data)

    assert "probe=pending,transition-ready,age=321s,start=2025-11-11T10:00:00Z,expires=2025-11-12T10:00:00Z" in compact
    assert "probe: pending,transition-ready,age=321s,start=2025-11-11T10:00:00Z,expires=2025-11-12T10:00:00Z" in detailed
